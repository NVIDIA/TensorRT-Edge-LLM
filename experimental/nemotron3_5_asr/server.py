# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal model-specific transcription server for Nemotron-3.5-ASR."""

import argparse
import asyncio
import importlib
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from tensorrt_edgellm import __version__
from tensorrt_edgellm import runtime as runtime_facade
from tensorrt_edgellm._native import NativeManifestNotFoundError

logger = logging.getLogger("edgellm.nemotron_asr.server")

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
_LANGUAGE_TAG = re.compile(
    r"(?:^|\s)<([A-Za-z]{2,3}(?:-[A-Za-z]{2})?)>(?=\s|$)")


@dataclass(frozen=True)
class Transcription:
    text: str
    language: Optional[str]


class ApiError(Exception):

    def __init__(self,
                 message: str,
                 status_code: int = 400,
                 error_type: str = "invalid_request_error",
                 param: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.param = param

    def payload(self) -> dict:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": None,
            }
        }


async def _run_to_completion(operation, *args):
    """Do not release batch-1 admission while native inference is running."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.gather(worker, return_exceptions=True))
        raise


def _load_runtime_module():
    try:
        return runtime_facade.load()
    except NativeManifestNotFoundError:
        try:
            return importlib.import_module("_edgellm_runtime")
        except ImportError as exc:
            raise RuntimeError(
                "build the experimental native runtime and add its pybind "
                "directory to PYTHONPATH") from exc


def _split_language_tag(text: str):
    matches = list(_LANGUAGE_TAG.finditer(text))
    if not matches:
        return text.strip(), None
    return (" ".join(_LANGUAGE_TAG.sub(" ",
                                       text).split()), matches[-1].group(1))


def _load_bundle(model: str):
    root = Path(model).expanduser().resolve()
    layouts = (
        (root / "audio_encoder.engine", root / "rnnt_step.engine"),
        (root / "audio" / "audio_encoder.engine",
         root / "rnnt" / "rnnt_step.engine"),
    )
    if sum(all(path.is_file() for path in layout) for layout in layouts) != 1:
        raise ValueError(
            f"{str(root)!r} must contain one complete Nemotron-3.5-ASR "
            "engine layout")
    try:
        with open(root / "config.json", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{str(root)!r} does not contain a valid config.json") from exc
    if config.get("model_type") != "nemotron3_5_asr":
        raise ValueError("ASR bundle config.json has the wrong model_type")
    for filename in ("tokenizer.json", "processor_config.json"):
        if not (root / filename).is_file():
            raise ValueError(f"ASR bundle is missing {filename}")
    return root, config


def _prompt_dictionary(root: Path, num_prompts: int) -> Dict[str, int]:
    try:
        with open(root / "processor_config.json", encoding="utf-8") as file:
            values = json.load(file).get("prompt_dictionary") or {}
    except (OSError, ValueError) as exc:
        raise ValueError("invalid ASR processor_config.json") from exc
    if not isinstance(values, dict) or not values:
        raise ValueError("ASR prompt_dictionary must be a non-empty object")
    prompts = {}
    for name, prompt_id in values.items():
        key = str(name).strip().lower()
        if (not key or isinstance(prompt_id, bool)
                or not isinstance(prompt_id, int)
                or not 0 <= prompt_id < num_prompts):
            raise ValueError(f"invalid ASR prompt entry {name!r}")
        if key in prompts and prompts[key] != prompt_id:
            raise ValueError(f"conflicting ASR prompt entry {name!r}")
        prompts[key] = prompt_id
    if "auto" not in prompts:
        raise ValueError("ASR prompt_dictionary must define 'auto'")
    return prompts


class NemotronAsr:
    """Batch-1 greedy RNN-T runtime for one complete model bundle."""

    def __init__(self, model: str, model_name: str = "") -> None:
        root, config = _load_bundle(model)
        num_prompts = config.get("num_prompts", 128)
        if (isinstance(num_prompts, bool) or not isinstance(num_prompts, int)
                or num_prompts <= 0):
            raise ValueError("ASR bundle config.json has invalid num_prompts")
        runtime_class = getattr(_load_runtime_module(), "NemotronAsrRuntime",
                                None)
        if runtime_class is None:
            raise RuntimeError(
                "the native runtime was built without experimental models")
        self.model_name = model_name or root.name
        self.model = str(root)
        self._prompts = _prompt_dictionary(root, num_prompts)
        self._runtime = runtime_class(engine_dir=str(root),
                                      tokenizer_dir=str(root))

    @property
    def max_audio_seconds(self) -> float:
        return float(self._runtime.max_mel_frames) * 0.01

    def transcribe(self, audio: bytes, language: str = "") -> Transcription:
        selected = (language or "auto").strip().lower()
        if selected not in self._prompts:
            raise ValueError(f"unsupported ASR language {language!r}")
        output = self._runtime.transcribe(audio, self._prompts[selected])
        text, detected = _split_language_tag(output.text)
        return Transcription(
            text, detected or (language.strip() if language else None))

    def close(self) -> None:
        self._runtime = None


def create_app(runtime: NemotronAsr) -> FastAPI:
    """Create the model-specific health, discovery, and transcription API."""
    admission = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="TensorRT Edge-LLM Nemotron-3.5-ASR",
                  version=__version__,
                  lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(item) for item in location[1:]) or None
        error = ApiError(str(first.get("msg", "invalid request")), param=param)
        return JSONResponse(status_code=400, content=error.payload())

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, exc: Exception):
        logger.error("Unhandled ASR server error", exc_info=exc)
        error = ApiError("internal server error", 500, "internal_server_error")
        return JSONResponse(status_code=500, content=error.payload())

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": runtime.model_name,
            "max_audio_seconds": runtime.max_audio_seconds,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object":
            "list",
            "data": [{
                "id": runtime.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tensorrt-edgellm",
            }],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
            file: UploadFile = File(...),
            model: str = Form(""),
            prompt: str = Form(""),
            language: str = Form(""),
            response_format: str = Form("json"),
            temperature: float = Form(0.0),
    ):
        if model and model != runtime.model_name:
            raise ApiError(f"model {model!r} is not served by this process",
                           404, "not_found_error", "model")
        if prompt:
            raise ApiError("Nemotron-3.5-ASR does not support a text prompt",
                           param="prompt")
        if temperature != 0.0:
            raise ApiError("Nemotron-3.5-ASR uses deterministic greedy decode",
                           param="temperature")
        if response_format not in {"json", "text"}:
            raise ApiError("response_format must be json or text",
                           param="response_format")

        audio = await file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if len(audio) > MAX_AUDIO_UPLOAD_BYTES:
            raise ApiError("audio upload exceeds the 25 MiB limit",
                           413,
                           param="file")
        if not audio:
            raise ApiError("audio file is empty", param="file")
        try:
            async with admission:
                output = await _run_to_completion(runtime.transcribe, audio,
                                                  language)
        except ValueError as exc:
            raise ApiError(str(exc), param="language") from exc
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("Audio too long:"):
                raise ApiError(message, 413, param="file") from exc
            if message.startswith(("Audio decode failed", "Audio too short:",
                                   "Mel extraction failed")):
                raise ApiError(message, param="file") from exc
            raise ApiError(message, 500, "engine_error") from exc

        if response_format == "text":
            return PlainTextResponse(output.text)
        response = {"text": output.text}
        if output.language:
            response["language"] = output.language
        return response

    return app


def _bounded_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return port


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a prebuilt Nemotron-3.5-ASR model bundle")
    parser.add_argument("model",
                        help="Complete checkpoint-direct or ONNX-built bundle")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_bounded_port, default=8000)
    parser.add_argument("--log-level",
                        choices=("debug", "info", "warning", "error"),
                        default="info")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = create_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    try:
        runtime = NemotronAsr(args.model, args.served_model_name)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"failed to load Nemotron-3.5-ASR: {exc}") from exc
    try:
        import uvicorn
    except ImportError as exc:
        runtime.close()
        raise RuntimeError(
            "uvicorn is required; install tensorrt-edgellm[server]") from exc
    logger.info("Loaded model=%s bundle=%s max_audio_seconds=%.2f",
                runtime.model_name, runtime.model, runtime.max_audio_seconds)
    uvicorn.run(create_app(runtime),
                host=args.host,
                port=args.port,
                log_level=args.log_level)


if __name__ == "__main__":
    main()
