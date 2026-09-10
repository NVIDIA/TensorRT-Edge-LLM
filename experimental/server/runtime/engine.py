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
"""Checkpoint-native Python API for TensorRT Edge-LLM.

Checkpoints are lowered directly by :mod:`experimental.builder`. Supported
weights remain checkpoint-backed TensorRT inputs and are validated and loaded
once when the runtime starts; this module has no ONNX export/build path.

Example::

    from experimental.server import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3.5-0.8B")
    outputs = llm.generate(
        ["What is the capital of France?"],
        SamplingParams(temperature=0.7, max_tokens=256),
    )
    for output in outputs:
        print(output.text)

    # Or start an OpenAI-compatible server:
    llm.serve(port=8000)
"""

import importlib.util
import json
import logging
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Mapping,
                    Optional, Sequence, Union)

from ..config import ContextCacheConfig
from ..parsing.tool_calling import (ToolConfig, parse_assistant_output,
                                    validate_tool_request)
from ..parsing.tool_chat_template import (ToolChatTemplateFormatter,
                                          needs_tool_chat_template)
from .engine_layout import BundleLayout, EngineType, inspect_bundle

logger = logging.getLogger("edgellm.server")

if TYPE_CHECKING:
    from .engine_build import BuildOptions

_PLUGIN_LIB_NAME = "libNvInfer_edgellm_plugin.so"
_MAX_LOGIT_BIAS_TOKENS = 1024
_MAX_LOGIT_BIAS_TOKEN_ID = (1 << 31) - 1
_MIN_LOGIT_BIAS = -100.0
_MAX_LOGIT_BIAS = 100.0
_DEFAULT_MAX_INPUT_LEN = 4096
_DEFAULT_MAX_BATCH_SIZE = 1
_DEFAULT_MAX_KV_CACHE_CAPACITY = 8192
_DEFAULT_DRAFT_TOP_K = 10
_DEFAULT_DRAFT_STEP = 6
_DEFAULT_VERIFY_TREE_SIZE = 60

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Sampling parameters for one generation request."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    enable_thinking: bool = False
    disable_spec_decode: bool = False
    num_logprobs: int = 0
    stop: List[str] = field(default_factory=list)
    logit_bias: Dict[int, float] = field(default_factory=dict)
    skip_special_tokens: bool = True
    reuse_context: bool = True
    cache_generated_tokens: bool = True


@dataclass
class LogprobEntry:
    """One top-K log-probability entry for a single generated token.

    ``token`` is the piece decoded as UTF-8 with ``errors="replace"`` (a
    byte-level BPE token may be only part of a multi-byte character, so it can
    contain U+FFFD); ``bytes`` carries the raw token bytes losslessly.
    """

    token_id: int
    logprob: float
    token: str
    bytes: List[int]


def _convert_logprobs(raw) -> List[List[LogprobEntry]]:
    """Convert the C++/pybind logprobs (list of list of native LogprobEntry with
    a raw-bytes ``piece``) into engine LogprobEntry dataclasses."""
    return [[
        LogprobEntry(token_id=e.token_id,
                     logprob=e.logprob,
                     token=e.piece.decode("utf-8", "replace"),
                     bytes=list(e.piece)) for e in step
    ] for step in raw]


@dataclass
class CompletionOutput:
    """Output of a single generation request."""

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class StreamDelta:
    """Single delta from a streaming generation.

    Text deltas carry ``text``/``token_ids``; audio deltas (Omni streaming)
    carry ``audio_bytes`` (int16 LE mono PCM) instead. ``finished`` marks the
    end of the text stream; generator exhaustion ends the audio stream.
    """

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    finished: bool = False
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)
    audio_bytes: Optional[bytes] = None


@dataclass
class AudioParams:
    """Talker / vocoder knobs for one Omni audio-output request."""

    voice: str = ""
    talker_temperature: float = 0.9
    talker_top_k: int = 50
    talker_top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_audio_length: int = 4096
    codec_chunk_frames: int = 10
    talker_prefill_threshold: int = 4


#: Sample rate of Omni Code2Wav PCM output.
OMNI_AUDIO_SAMPLE_RATE = 24000


def _native_audio_params(rt, audio: "AudioParams"):
    """Convert the AudioParams dataclass to the pybind OmniAudioParams."""
    omni_params = rt.OmniAudioParams()
    omni_params.speaker_name = audio.voice
    for name, value in asdict(audio).items():
        if name != "voice":
            setattr(omni_params, name, value)
    return omni_params


class _CancellableIterator:
    """Iterator whose thread-safe close signal can interrupt native waits."""

    def __init__(self, iterator, cancel) -> None:
        self._iterator = iterator
        self._cancel = cancel
        self._cancel_lock = threading.Lock()
        self._cancelled = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self) -> None:
        with self._cancel_lock:
            if not self._cancelled:
                self._cancelled = True
                self._cancel()
        try:
            self._iterator.close()
        except (RuntimeError, ValueError):
            # Another thread is inside next(). The cancel signal wakes it; the
            # async adapter waits for that call before releasing admission.
            pass


def _pump_channels(rt, run, text_channel, audio_channel, sem=None):
    """Drive one generation in a worker thread, yielding StreamDeltas.

    Shared by the Omni dual-stream path (both channels) and the standalone
    TTS path (``text_channel=None``). The drain-once retry after
    is_finished()/is_cancelled() closes the race where the producer finishes
    between an empty pop and the check. The worker releases ``sem`` when the
    C++ call returns.
    """

    def _cancel():
        if text_channel is not None:
            text_channel.cancel()
        audio_channel.cancel()

    def _iterate():
        error_holder = [None]

        def _run():
            try:
                run()
            except Exception as error:  # noqa: BLE001 - re-raised below
                error_holder[0] = error
                _cancel()
            finally:
                if sem is not None:
                    sem.release()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        text_done = text_channel is None
        audio_done = False
        try:
            while not (text_done and audio_done):
                if not text_done:
                    chunk = text_channel.wait_pop(timeout_ms=20)
                    if chunk is None and (text_channel.is_finished()
                                          or text_channel.is_cancelled()):
                        chunk = text_channel.try_pop()
                        if chunk is None:
                            text_done = True
                    if chunk is not None:
                        reason = finish_reason_name(
                            rt, chunk.reason) if chunk.finished else None
                        yield StreamDelta(
                            text=chunk.text,
                            token_ids=list(chunk.token_ids),
                            prompt_tokens=(chunk.prompt_token_count
                                           if chunk.prompt_token_count >= 0
                                           else None),
                            finished=chunk.finished,
                            finish_reason=reason,
                            logprobs=_convert_logprobs(chunk.logprobs),
                        )
                        text_done = chunk.finished

                if not audio_done:
                    # Text drives pacing while it flows (non-blocking audio
                    # poll); once text ends, block on audio instead.
                    audio_chunk = audio_channel.wait_pop(
                        timeout_ms=100 if text_done else 0)
                    if audio_chunk is None and (audio_channel.is_finished() or
                                                audio_channel.is_cancelled()):
                        audio_chunk = audio_channel.try_pop()
                        if audio_chunk is None:
                            audio_done = True
                    if audio_chunk is not None:
                        if audio_chunk.pcm16:
                            yield StreamDelta(audio_bytes=audio_chunk.pcm16)
                        audio_done = audio_chunk.is_final
        finally:
            if not (text_done and audio_done):
                _cancel()
            # Channels own buffers consumed by the native worker. Runtime
            # teardown and admission release must wait for worker exit.
            worker.join()
        if error_holder[0] is not None:
            raise error_holder[0]

    return _CancellableIterator(_iterate(), _cancel)


def _stream_tts(rt,
                runtime,
                text: str,
                audio: "AudioParams",
                sem,
                infer_guard=None,
                ensure_open=None) -> Iterator["StreamDelta"]:
    """Run one standalone TTS request; yields audio-only StreamDeltas.

    ``runtime`` is any pybind object exposing ``handle_request_tts``
    (LLMRuntime with the Omni stack loaded, or the TTS-only TTSRuntime).
    ``infer_guard`` serializes against text inference sharing the same CUDA
    stream (unused by the TTS-only runtime, which serves no text).
    """
    state = {}

    def _cancel():
        stream = state.get("stream")
        if stream is not None:
            stream.close()

    def _iterate():
        omni_params = _native_audio_params(rt, audio)
        audio_channel = rt.AudioStreamChannel()
        if sem is not None:
            sem.acquire()
            try:
                if ensure_open is not None:
                    ensure_open()
            except BaseException:
                sem.release()
                raise

        def _run():
            if infer_guard is None:
                runtime.handle_request_tts(text, omni_params, audio_channel)
                return
            with infer_guard:
                runtime.handle_request_tts(text, omni_params, audio_channel)

        stream = _pump_channels(rt, _run, None, audio_channel, sem=sem)
        state["stream"] = stream
        yield from stream

    return _CancellableIterator(_iterate(), _cancel)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_model_id(model: str) -> str:
    """Return a clean id to advertise via /v1/models and echo in responses.

    A local checkpoint path would otherwise leak the full filesystem path as
    the model id, so use its directory name. A Hugging Face ID is kept as-is.
    """
    src = model
    if src and (os.path.isabs(src) or os.path.isdir(src)):
        return os.path.basename(os.path.normpath(src))
    return src


def _read_bundle_builder_config(bundle_dir: str) -> dict:
    """Read the text runtime profile from a complete model bundle."""
    for filename in ("config.json", "base_config.json"):
        cfg_path = os.path.join(bundle_dir, filename)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                return json.load(f).get("builder_config", {})
    return {}


@dataclass(frozen=True)
class _SpecDecodeRuntimeOptions:
    """Drafting shape resolved from one speculative engine bundle."""

    top_k: int
    step: int
    verify_size: int
    dflash_block_size: int = 0


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _resolve_spec_decode_runtime_options(
    bundle_dir: str,
    method: str,
    num_speculative_tokens: Optional[int],
    draft_top_k: Optional[int],
    draft_step: Optional[int],
    verify_tree_size: Optional[int],
) -> _SpecDecodeRuntimeOptions:
    """Resolve method-specific defaults against the compiled engine profile."""
    if method == "none":
        return _SpecDecodeRuntimeOptions(
            draft_top_k or _DEFAULT_DRAFT_TOP_K,
            draft_step or _DEFAULT_DRAFT_STEP,
            verify_tree_size or _DEFAULT_VERIFY_TREE_SIZE,
        )

    base = _read_json(os.path.join(bundle_dir, "base_config.json"))
    draft = _read_json(os.path.join(bundle_dir, "draft_config.json"))
    engine_method = str(base.get("spec_decode_type", method))
    compatible_methods = {method}
    if method == "mtp":
        compatible_methods.add("gemma4_mtp")
    if engine_method not in compatible_methods:
        raise ValueError(
            f"requested speculative method {method!r}, but the compiled "
            f"bundle uses {engine_method!r}")
    max_verify_size = int(
        base.get("builder_config", {}).get("max_verify_tree_size",
                                           _DEFAULT_VERIFY_TREE_SIZE))

    if engine_method == "eagle3":
        return _SpecDecodeRuntimeOptions(
            draft_top_k or _DEFAULT_DRAFT_TOP_K,
            num_speculative_tokens or draft_step or _DEFAULT_DRAFT_STEP,
            verify_tree_size or max_verify_size,
        )

    if engine_method in {"mtp", "gemma4_mtp"}:
        top_k = draft_top_k or 1
        if engine_method == "gemma4_mtp" and top_k != 1:
            raise ValueError("Gemma4 MTP supports linear drafting only; "
                             "set draft_top_k=1")
        step = num_speculative_tokens or draft_step or _DEFAULT_DRAFT_STEP
        verify_size = (verify_tree_size
                       or (step + 1 if top_k == 1 else max_verify_size))
        return _SpecDecodeRuntimeOptions(top_k, step, verify_size)

    if engine_method in {"dflash", "jetspec"}:
        if draft_step not in (None, 1):
            raise ValueError(
                f"{engine_method} emits a complete block in one draft step; "
                "set draft_step=1")
        mode_config = (draft.get(f"{engine_method}_config")
                       or draft.get("dflash_config") or {})
        checkpoint_block_size = int(
            mode_config.get("block_size", draft.get("block_size", 0)))
        block_size = num_speculative_tokens or checkpoint_block_size
        if checkpoint_block_size < 2:
            raise ValueError(
                f"compiled {engine_method} draft has an invalid proposal "
                f"block size {checkpoint_block_size}")
        if not 2 <= block_size <= checkpoint_block_size:
            raise ValueError(
                f"{engine_method} num_speculative_tokens must be within the "
                f"compiled proposal block size [2, {checkpoint_block_size}]")
        top_k = draft_top_k or 1
        verify_size = (verify_tree_size
                       or (block_size if top_k == 1 else max_verify_size))
        return _SpecDecodeRuntimeOptions(top_k, 1, verify_size, block_size)

    if engine_method == "dspark":
        if draft_step not in (None, 1):
            raise ValueError(
                "dspark emits a complete block in one draft step; set "
                "draft_step=1")
        mode_config = draft.get("dspark_config") or {}
        block_size = int(
            mode_config.get("block_size", draft.get("block_size", 0)))
        proposal_size = num_speculative_tokens or block_size
        if not 1 <= proposal_size <= block_size:
            raise ValueError(
                "dspark num_speculative_tokens must be within the compiled "
                f"proposal block size [1, {block_size}]")
        top_k = draft_top_k or 1
        if top_k > 1 and proposal_size != block_size:
            raise ValueError(
                "dspark tree drafting always uses the complete checkpoint "
                "proposal block")
        verify_size = verify_tree_size or proposal_size + 1
        return _SpecDecodeRuntimeOptions(top_k, 1, verify_size)

    raise ValueError(f"unsupported speculative engine mode {engine_method!r}")


def _ensure_plugin_path() -> None:
    """Set EDGELLM_PLUGIN_PATH if not already set.

    Searches relative to this package and common build locations.
    """
    if os.environ.get("EDGELLM_PLUGIN_PATH"):
        return
    project_root = Path(__file__).resolve().parents[3]
    search_dirs = [
        project_root / "build" / "core",
        project_root / "build" / "lib",
    ]
    for d in search_dirs:
        candidate = d / _PLUGIN_LIB_NAME
        if candidate.is_file():
            os.environ["EDGELLM_PLUGIN_PATH"] = str(candidate)
            return


def _import_runtime():
    """Import the C++ pybind module."""
    try:
        runtime_facade = importlib.import_module("tensorrt_edgellm.runtime")
        native_package = importlib.import_module("tensorrt_edgellm._native")
        try:
            return runtime_facade.load()
        except native_package.NativeManifestNotFoundError:
            # Source checkouts have no generated variants.json.
            pass
    except ImportError:
        pass
    _ensure_plugin_path()
    try:
        return importlib.import_module("tensorrt_edgellm._edgellm_runtime")
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parents[3]
    search_dirs = []
    if os.environ.get("EDGELLM_PYBIND_DIR"):
        search_dirs.append(Path(os.environ["EDGELLM_PYBIND_DIR"]))
    if os.environ.get("BUILD_DIR"):
        search_dirs.append(Path(os.environ["BUILD_DIR"]) / "pybind")
    search_dirs.extend([
        project_root / "experimental" / "pybind" / "build",
        project_root / "build" / "pybind",
    ])
    search_dirs.extend(project_root.glob("build/lib.*"))
    for cand_dir in search_dirs:
        if not cand_dir.is_dir():
            continue
        so_files = list(cand_dir.glob("*_edgellm_runtime*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location(
                "_edgellm_runtime", so_files[0])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["tensorrt_edgellm._edgellm_runtime"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Could not import _edgellm_runtime. Build the C++ extension first:\n"
        "  TRT_PACKAGE_DIR=/path/to/tensorrt python experimental/server/setup_pybind.py build_ext --inplace"
    )


def _normalize_logit_bias(
        logit_bias: Optional[Dict[Any, Any]]) -> Dict[int, float]:
    """Validate and normalize an OpenAI-compatible logit_bias map."""
    if logit_bias is None:
        return {}
    if not isinstance(logit_bias, dict):
        raise ValueError(
            "'logit_bias' must be an object mapping token IDs to bias values")
    if len(logit_bias) > _MAX_LOGIT_BIAS_TOKENS:
        raise ValueError(f"'logit_bias' has {len(logit_bias)} entries; max is "
                         f"{_MAX_LOGIT_BIAS_TOKENS}")

    normalized: Dict[int, float] = {}
    for token, bias in logit_bias.items():
        if isinstance(token, bool):
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if isinstance(token, int):
            token_id = token
        elif isinstance(token, str):
            try:
                token_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"'logit_bias' token ID {token!r} is not an integer"
                ) from exc
        else:
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if token_id < 0 or token_id > _MAX_LOGIT_BIAS_TOKEN_ID:
            raise ValueError(
                f"'logit_bias' token ID must be in "
                f"[0, {_MAX_LOGIT_BIAS_TOKEN_ID}], got {token_id}")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be a number")
        try:
            bias_value = float(bias)
        except OverflowError as exc:
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias}") from exc
        if (not math.isfinite(bias_value) or bias_value < _MIN_LOGIT_BIAS
                or bias_value > _MAX_LOGIT_BIAS):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias_value}")
        normalized[token_id] = bias_value
    return normalized


def _native_context_cache_config(rt, config: ContextCacheConfig):
    """Translate validated public configuration to the pybind value type."""
    native = rt.ContextCacheConfig()
    native.enabled = config.enabled
    native.max_records = config.max_records
    native.recurrent_snapshot_pool_bytes = config.recurrent_snapshot_pool_bytes
    native.partial_kv_snapshot_pool_bytes = config.partial_kv_snapshot_pool_bytes
    return native


def _set_context_cache_request_policies(rt, request,
                                        params: SamplingParams) -> None:
    """Map the two public request controls onto native cache policies."""
    if not isinstance(params.reuse_context, bool):
        raise ValueError("reuse_context must be boolean")
    if not isinstance(params.cache_generated_tokens, bool):
        raise ValueError("cache_generated_tokens must be boolean")
    request.context_cache_lookup_policy = (
        rt.ContextCacheLookupPolicy.USE_CACHE
        if params.reuse_context else rt.ContextCacheLookupPolicy.BYPASS)
    request.context_cache_commit_policy = (
        rt.ContextCacheCommitPolicy.INCLUDING_GENERATED_TOKENS
        if params.cache_generated_tokens else
        rt.ContextCacheCommitPolicy.PREFILL_STATE_ONLY)


def _engine_config_value(builder_config: dict, field_name: str,
                         requested_value: int) -> int:
    if field_name not in builder_config:
        return requested_value

    engine_value = int(builder_config[field_name])
    if engine_value != requested_value:
        logger.warning(
            "Using %s=%d from engine builder_config instead of requested %d",
            field_name,
            engine_value,
            requested_value,
        )
    return engine_value


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------


class LLM:
    """Checkpoint-direct entry point for offline and HTTP inference.

    ``model`` accepts a local checkpoint or Hugging Face ID. The experimental
    builder compiles every model component into a profile-specific cache bundle
    and externalizes every supported weight kind.
    """

    #: Selects the HTTP contract without a hierarchy of capability flags.
    runtime_kind = "chat"

    def __init__(
        self,
        model: str,
        *,
        cache_dir: str = "",
        engine_cache_max_size_gb: float = 50.0,
        clear_engine_cache: bool = False,
        max_input_len: int = _DEFAULT_MAX_INPUT_LEN,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_kv_cache_capacity: int = _DEFAULT_MAX_KV_CACHE_CAPACITY,
        draft_top_k: Optional[int] = None,
        draft_step: Optional[int] = None,
        verify_tree_size: Optional[int] = None,
        build_options: Optional["BuildOptions"] = None,
        speculative_config: Optional[Any] = None,
        context_cache_config: Optional[Union[ContextCacheConfig,
                                             Mapping[str, Any]]] = None,
    ):
        if not model:
            raise ValueError("'model' must be provided")
        if (isinstance(engine_cache_max_size_gb, bool)
                or not math.isfinite(engine_cache_max_size_gb)
                or engine_cache_max_size_gb <= 0):
            raise ValueError("engine_cache_max_size_gb must be positive")
        for name, value in (("draft_top_k", draft_top_k),
                            ("draft_step", draft_step), ("verify_tree_size",
                                                         verify_tree_size)):
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        self._model_id = _derive_model_id(model)
        self._draft_top_k = draft_top_k or _DEFAULT_DRAFT_TOP_K
        self._draft_step = draft_step or _DEFAULT_DRAFT_STEP
        self._verify_tree_size = (verify_tree_size
                                  or _DEFAULT_VERIFY_TREE_SIZE)
        self._dflash_block_size = 0
        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity
        self._context_cache_config = ContextCacheConfig.parse(
            context_cache_config)
        self._tool_template_formatter: Optional[
            ToolChatTemplateFormatter] = None
        self._admission_sem = threading.Semaphore(1)
        self._infer_lock = threading.Lock()
        self._close_lock = threading.Lock()
        # Context-reuse observability: the counters as of the last logged
        # request, so each request can report its own delta. Guarded because the
        # streaming path calls the C++ runtime from a background thread.
        self._ctx_reuse_metric_lock = threading.Lock()
        self._prev_ctx_reused_tokens = 0
        self._prev_ctx_matched_tokens = 0
        self._prev_ctx_hit_sequences = 0
        self._prev_ctx_admitted_sequences = 0
        self._closed = False
        self._runtime = None

        from .engine_build import BuildOptions, cache_root, prepare_model

        options = build_options or BuildOptions(
            max_input_len=max_input_len,
            max_batch_size=max_batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
        )
        spec_method = options.spec_type
        num_speculative_tokens = None
        if speculative_config:
            from ..config import SpeculativeConfig

            spec = SpeculativeConfig.parse(speculative_config)
            options = replace(options,
                              spec_type=spec.method,
                              draft_model_dir=spec.draft_model)
            spec_method = spec.method
            num_speculative_tokens = spec.num_speculative_tokens

        resolved_top_k = draft_top_k or (10 if spec_method == "eagle3" else 1)
        if (options.builder_spec_type == "gemma4_mtp" and resolved_top_k != 1):
            raise ValueError("Gemma4 MTP supports linear drafting only; "
                             "set draft_top_k=1")
        tree_base = (resolved_top_k > 1 and options.builder_spec_type
                     in {"mtp", "dflash", "jetspec"})
        if options.spec_type != "none":
            options = replace(options, tree_base=tree_base)

        prepared = prepare_model(
            model,
            cache_dir,
            options,
            max_cache_size_bytes=int(engine_cache_max_size_gb * (1 << 30)),
            clear_cache=clear_engine_cache,
        )
        self._cache_dir = cache_root(cache_dir)
        self._model_dir = prepared.model_dir
        self._draft_model_dir = prepared.draft_model_dir
        runtime_options = _resolve_spec_decode_runtime_options(
            prepared.bundle_dir,
            spec_method,
            num_speculative_tokens,
            draft_top_k,
            draft_step,
            verify_tree_size,
        )
        self._draft_top_k = runtime_options.top_k
        self._draft_step = runtime_options.step
        self._verify_tree_size = runtime_options.verify_size
        self._dflash_block_size = runtime_options.dflash_block_size
        self._init_from_bundle(prepared.bundle_dir)

        self._load_runtime()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_from_bundle(self, bundle_dir: str) -> None:
        """Validate one complete checkpoint-native runtime bundle."""
        self._layout = inspect_bundle(bundle_dir)
        if self._layout.engine_type not in (EngineType.LLM,
                                            EngineType.SPEC_DECODE):
            raise ValueError("no checkpoint-direct LLM engine found in "
                             f"{self._layout.root!r}")

        self._bundle_dir = self._layout.root
        builder_config = _read_bundle_builder_config(self._bundle_dir)
        self._max_input_len = _engine_config_value(builder_config,
                                                   "max_input_len",
                                                   self._max_input_len)
        self._max_batch_size = _engine_config_value(builder_config,
                                                    "max_batch_size",
                                                    self._max_batch_size)
        self._max_kv_cache_capacity = _engine_config_value(
            builder_config, "max_kv_cache_capacity",
            self._max_kv_cache_capacity)

        self._media_dir = self._layout.media_dir
        logger.info("Using cached engine bundle: %s", self._bundle_dir)

    def _load_runtime(self) -> None:
        """Load checkpoint-backed weights once, then initialize the runtime."""
        self._rt = _import_runtime()
        context_cache_config = _native_context_cache_config(
            self._rt, self._context_cache_config)
        logger.info("Loading runtime bundle from %s", self._bundle_dir)
        if self._layout.engine_type == EngineType.SPEC_DECODE:
            logger.info(
                "Speculative decoding enabled (top_k=%d, step=%d, "
                "verify_size=%d, block_size=%d)",
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
                self._dflash_block_size,
            )
            self._runtime = self._rt.LLMRuntime(
                self._bundle_dir,
                self._media_dir,
                {},
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
                self._model_dir,
                self._draft_model_dir,
                context_cache_config,
                self._dflash_block_size,
            )
        else:
            self._runtime = self._rt.LLMRuntime(
                self._bundle_dir,
                self._media_dir,
                {},
                self._model_dir,
                context_cache_config,
            )
        self._runtime.capture_decoding_cuda_graph()
        self._load_omni_runtime()
        logger.info("Engine loaded and ready.")

    def _load_omni_runtime(self) -> None:
        """Load the Qwen3-Omni audio-output stack when its engines exist."""
        if not self._layout.has_speech:
            return
        logger.info("Auto-detected Omni engines: talker=%s",
                    self._layout.talker_dir)

        self._runtime.load_omni(self._layout.talker_dir,
                                self._layout.code_predictor_dir,
                                self._layout.code2wav_dir, self._bundle_dir,
                                self._model_dir)
        logger.info("Omni audio output ready.")

    def _tool_template_dirs(self) -> List[str]:
        return [self._model_dir]

    def _get_tool_template_formatter(self) -> ToolChatTemplateFormatter:
        if self._tool_template_formatter is None:
            self._tool_template_formatter = ToolChatTemplateFormatter(
                self._tool_template_dirs())
        return self._tool_template_formatter

    def _tool_choice_for_template(
            self, tool_config: ToolConfig) -> Union[str, Dict[str, Any]]:
        if tool_config.forced_name:
            return {
                "type": "function",
                "function": {
                    "name": tool_config.forced_name
                },
            }
        return tool_config.tool_choice

    def _visual_config(self) -> dict:
        """Read the model-specific visual component configuration once."""
        cached = getattr(self, "_visual_config_cache", None)
        if cached is not None:
            return cached
        cfg: dict = {}
        root = self._media_dir
        cfg_path = os.path.join(root, "visual", "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except (OSError, ValueError):
                cfg = {}
        self._visual_config_cache = cfg
        return cfg

    def _video_model_family(self) -> str:
        """Frame-sampling family ("qwen" / "internvl" / "nemotron") from the
        visual engine's model_type. Types without a video path (phi4mm, gemma,
        ...) are rejected: their runners read only the first frame."""
        cached = getattr(self, "_video_family_cache", None)
        if cached is not None:
            return cached
        config = self._visual_config()
        model_type = config.get("model_type", "")
        qwen_video_types = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5",
                            "qwen3_omni")
        # Audio-side model types have no video path (qwen3_omni_audio_encoder,
        # qwen3_omni_code2wav, qwen3_asr*); the omni ones share the qwen3_omni
        # prefix, so exclude before the prefix match.
        is_audio_type = any(tag in model_type
                            for tag in ("audio", "code2wav", "asr"))
        root = self._media_dir
        has_visual = os.path.isfile(
            os.path.join(root, "visual", "visual.engine"))
        if "internvl" in model_type and has_visual:
            family = "internvl"
        elif ("nemotron" in model_type and not is_audio_type and has_visual
              and config.get("supports_video", True)):
            family = "nemotron"
        elif (model_type.startswith(qwen_video_types) and not is_audio_type
              and has_visual):
            family = "qwen"
        else:
            # Covers audio-only engines (audio/ but no visual/) and model
            # types whose runners have no video path (phi4mm, gemma, ...).
            raise ValueError(
                f"video input is not supported for model_type={model_type!r}"
                " in this runtime bundle; supported families: Qwen-VL "
                "(qwen2_vl/qwen2_5_vl/qwen3_vl/qwen3_5/qwen3_omni), InternVL, "
                "and Nemotron-Omni")
        self._video_family_cache = family
        return family

    def _video_frame_limits(self) -> dict:
        """Engine-profile inputs for frame-count clamping (see video_sampling):
        builder token bounds from the visual config.json + patch geometry from
        preprocessor_config.json. Empty dict when unavailable (no clamping)."""
        cached = getattr(self, "_video_limits_cache", None)
        if cached is not None:
            return cached
        limits: dict = {}
        cfg = self._visual_config()
        builder = cfg.get("builder_config") or {}
        root = self._media_dir
        pre: dict = {}
        pre_path = os.path.join(root, "visual", "preprocessor_config.json")
        if os.path.isfile(pre_path):
            try:
                with open(pre_path) as f:
                    pre = json.load(f)
            except (OSError, ValueError):
                pre = {}
        pre = pre.get("image_processor", pre)
        if builder.get("max_image_tokens"):
            limits = {
                "model_type":
                cfg.get("model_type", ""),
                "min_image_tokens":
                int(builder.get("min_image_tokens", 1)),
                "max_image_tokens":
                int(builder["max_image_tokens"]),
                "max_image_tokens_per_image":
                int(builder.get("max_image_tokens_per_image", 0)),
                "max_cu_seqlen_groups":
                int(builder.get("max_cu_seqlen_groups", 0)),
                "patch_size":
                int(pre.get("patch_size", 0)),
                "merge_size":
                int(pre.get("merge_size", 0)),
                "temporal_patch_size":
                int(pre.get("temporal_patch_size", 2)),
                # Nemotron-Omni video geometry (top-level visual config.json).
                "video_pruning_rate":
                float(cfg.get("video_pruning_rate", 0.0)),
                "video_temporal_patch_size":
                int(cfg.get("video_temporal_patch_size", 2)),
                "video_target_num_patches":
                int(cfg.get("video_target_num_patches", 1024)),
                "downsample_ratio":
                float(cfg.get("downsample_ratio", 0.5)),
            }
        self._video_limits_cache = limits
        return limits

    def _prepare_messages_for_runtime(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        enable_thinking: bool = False,
        derive_replay_tail: bool = False,
    ):
        """Prepare messages for the C++ runtime.

        Returns the replay-tail length alongside the prepared messages. It is
        non-zero only when the caller asked for it, which is what lets a
        Hybrid+MTP checkpoint be reused across turns.
        """
        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        template_tools = (tool_config.tools
                          if tool_config.tool_choice != "none" else [])
        image_buffers = _load_image_buffers(self._rt, messages,
                                            self._video_model_family,
                                            self._video_frame_limits)

        if needs_tool_chat_template(messages, template_tools,
                                    tool_config.tool_choice):
            template_tool_choice = None
            if tool_config.tool_choice != "none":
                template_tool_choice = self._tool_choice_for_template(
                    tool_config)
            formatter = self._get_tool_template_formatter()
            replay_tail_length = 0
            if derive_replay_tail:
                # Derive the multi-turn replay tail from the tokenized template
                # so a Hybrid+MTP checkpoint can be reused across turns.
                prompt, replay_tail_length = formatter.format_with_replay_tail(
                    messages,
                    tools=template_tools,
                    tool_choice=template_tool_choice,
                    parallel_tool_calls=tool_config.parallel_tool_calls,
                    enable_thinking=enable_thinking,
                )
            else:
                prompt = formatter.format(
                    messages,
                    tools=template_tools,
                    tool_choice=template_tool_choice,
                    parallel_tool_calls=tool_config.parallel_tool_calls,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            cpp_messages = _convert_messages_to_cpp(
                self._rt,
                [{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return (cpp_messages, image_buffers, False, False,
                    replay_tail_length)

        cpp_messages = _convert_messages_to_cpp(self._rt, messages)
        return cpp_messages, image_buffers, True, True, 0

    def _make_generation_request(
        self,
        messages: List[Dict[str, Any]],
        params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        stream_channel: Optional[Any] = None,
    ):
        normalized_logit_bias = _normalize_logit_bias(params.logit_bias)
        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        # The replay tail only matters for a prefill-state-only commit against a
        # draft model with reuse enabled: that is the deployment whose
        # checkpoint must land on a turn boundary the next render reproduces.
        # The server tests build bare objects that skip __init__, so read the
        # config defensively and let the `and` chain short-circuit before it
        # reaches the attributes only a constructed LLM has.
        cache_config = getattr(self, "_context_cache_config", None)
        derive_replay_tail = (cache_config is not None and cache_config.enabled
                              and not params.cache_generated_tokens
                              and self.has_draft_model)
        (cpp_messages, image_buffers, apply_template, add_prompt,
         replay_tail_length) = (self._prepare_messages_for_runtime(
             messages,
             tools=tool_config.tools,
             tool_choice=tool_config.tool_choice,
             tool_config=tool_config,
             enable_thinking=params.enable_thinking,
             derive_replay_tail=derive_replay_tail,
         ))

        audio_buffers = _load_audio_buffers(self._rt, messages)

        request = self._rt.LLMGenerationRequest()
        req = self._rt.Request(messages=cpp_messages)
        req.image_buffers = image_buffers
        req.audio_buffers = audio_buffers
        req.stop_strings = params.stop
        req.logit_bias = normalized_logit_bias
        request.requests = [req]
        if stream_channel is not None:
            request.stream_channels = [stream_channel]
        request.temperature = params.temperature
        request.top_p = params.top_p
        request.top_k = params.top_k
        request.max_generate_length = params.max_tokens
        request.apply_chat_template = apply_template
        request.add_generation_prompt = add_prompt
        request.enable_thinking = params.enable_thinking
        request.disable_spec_decode = params.disable_spec_decode
        request.num_logprobs = params.num_logprobs
        _set_context_cache_request_policies(self._rt, request, params)
        request.context_cache_replay_tail_length = replay_tail_length
        return request

    def _count_prepared_prompt_tokens(self, request) -> Optional[int]:
        """Count tokens only for an explicit token-count API request."""
        if not hasattr(self._runtime, "count_prompt_tokens"):
            return None
        rows = getattr(request, "requests", ())
        if any(
                getattr(row, "image_buffers",
                        ()) or getattr(row, "audio_buffers", ())
                or getattr(row, "past_trajectory", None) is not None
                for row in rows):
            return None
        counts = self._runtime.count_prompt_tokens(request)
        return counts[0] if counts else None

    def _parse_generation_output(
        self,
        text: str,
        token_ids: List[int],
        prompt_tokens: Optional[int],
        finish_reason: Optional[str],
        tool_config: ToolConfig,
        *,
        tool_parser: str = "auto",
        reasoning_parser: str = "none",
    ) -> CompletionOutput:
        parsed = parse_assistant_output(
            text,
            tool_config,
            self._model_dir,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        tool_calls = [call.to_openai() for call in parsed.tool_calls]
        return CompletionOutput(
            text=parsed.content,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            tool_calls=tool_calls,
            reasoning=parsed.reasoning or None,
        )

    def _complete_prepared_request(
        self,
        request,
        params: SamplingParams,
        tool_config: ToolConfig,
        *,
        tool_parser: str = "auto",
        reasoning_parser: str = "none",
    ) -> CompletionOutput:
        response = self._handle_request(request)
        text = response.output_texts[0] if response.output_texts else ""
        token_ids = response.output_ids[0] if response.output_ids else []
        prompt_tokens = (response.prompt_token_counts[0]
                         if response.prompt_token_counts else None)
        finish_reason = (finish_reason_name(self._rt,
                                            response.finish_reasons[0])
                         if response.finish_reasons else "stop")
        output = self._parse_generation_output(
            text,
            token_ids,
            prompt_tokens,
            finish_reason,
            tool_config,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        if params.num_logprobs > 0 and response.logprobs:
            output.logprobs = _convert_logprobs(response.logprobs[0])
        return output

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def _admission(self):
        """Per-instance gate from media decode through inference completion:
        queued requests must not each pin decoded frames. Semaphore, not Lock --
        streaming releases from the worker/SSE side."""
        return self._admission_sem

    def _infer_guard(self):
        """Lock serializing every entry into the stateful C++ runtime."""
        return self._infer_lock

    def _ensure_open(self) -> None:
        if self._closed or self._runtime is None:
            raise RuntimeError("Edge-LLM runtime is closed")

    @staticmethod
    def _ratio(num: int, denom: int) -> float:
        return num / denom if denom > 0 else 0.0

    def _log_context_reuse_metrics(self) -> None:
        """Emit an INFO line making context-cache reuse visible.

        ``get_context_cache_metrics()`` returns cumulative coordinator counters
        for the whole server run, so the per-request figures are the delta since
        the previous logged request. ContextCacheMetrics carries no total-prompt
        -token field, so the headline hit rate is sequence-level
        (hit_sequences / admitted_sequences); reused/matched token counts are
        reported alongside to show token-level reuse is actually happening.
        """
        cc = self._runtime.get_context_cache_metrics()
        if cc is None:
            return
        with self._ctx_reuse_metric_lock:
            cum_reused = int(cc.reused_tokens)
            cum_matched = int(cc.matched_tokens)
            cum_hit_seqs = int(cc.hit_sequences)
            cum_admitted = int(cc.admitted_sequences)
            req_reused = cum_reused - self._prev_ctx_reused_tokens
            req_matched = cum_matched - self._prev_ctx_matched_tokens
            req_hit_seqs = cum_hit_seqs - self._prev_ctx_hit_sequences
            req_admitted = cum_admitted - self._prev_ctx_admitted_sequences
            self._prev_ctx_reused_tokens = cum_reused
            self._prev_ctx_matched_tokens = cum_matched
            self._prev_ctx_hit_sequences = cum_hit_seqs
            self._prev_ctx_admitted_sequences = cum_admitted
        logger.info(
            "[context-reuse] request: reused=%d matched=%d hitSeqs=%d/%d "
            "hitRate=%.4f | cumulative: reused=%d matched=%d hitSeqs=%d/%d "
            "hitRate=%.4f records=%d hybridRestores=%d",
            req_reused,
            req_matched,
            req_hit_seqs,
            req_admitted,
            self._ratio(req_hit_seqs, req_admitted),
            cum_reused,
            cum_matched,
            cum_hit_seqs,
            cum_admitted,
            self._ratio(cum_hit_seqs, cum_admitted),
            int(cc.current_records),
            int(cc.hybrid_restores),
        )

    def _handle_request(self, request):
        """Serialized entry to the C++ runtime."""
        with self._infer_guard():
            self._ensure_open()
            response = self._runtime.handle_request(request)
        # Callable on duck-typed non-LLM objects in the server tests, which have
        # no config to inherit a class default from.
        cache_config = getattr(self, "_context_cache_config", None)
        if cache_config is not None and cache_config.enabled:
            self._log_context_reuse_metrics()
        return response

    def close(self) -> None:
        """Drain active work and release native engines and device memory."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            with self._admission_sem:
                with self._infer_lock:
                    self._runtime = None

    def __enter__(self) -> "LLM":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def generate(
        self,
        prompts: Union[str, List[str], List[List[Dict[str, Any]]]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_parser: str = "auto",
        reasoning_parser: str = "none",
    ) -> List[CompletionOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string, a list of prompt strings, or
                a list of OpenAI-style message lists.
            sampling_params: Sampling configuration. Defaults to
                ``SamplingParams()``.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            List of ``CompletionOutput`` objects, one per prompt.
        """
        params = sampling_params or SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]
        message_batches = []
        for p in prompts:
            if isinstance(p, str):
                message_batches.append([{"role": "user", "content": p}])
            elif isinstance(p, list):
                message_batches.append(p)
            else:
                raise TypeError(f"Unsupported prompt type: {type(p)}")

        outputs = []
        for messages in message_batches:
            tool_config = validate_tool_request(messages, tools, tool_choice)
            with self._admission():
                self._ensure_open()
                request = self._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice,
                    tool_config=tool_config,
                )

                output = self._complete_prepared_request(
                    request,
                    params,
                    tool_config,
                    tool_parser=tool_parser,
                    reasoning_parser=reasoning_parser,
                )
            outputs.append(output)

        return outputs

    def chat(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_parser: str = "auto",
        reasoning_parser: str = "none",
    ) -> CompletionOutput:
        """Single-turn chat completion (convenience wrapper).

        Args:
            messages: OpenAI-style message list.
            sampling_params: Sampling configuration.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            A single ``CompletionOutput``.
        """
        return self.generate([messages],
                             sampling_params,
                             tools=tools,
                             tool_choice=tool_choice,
                             tool_parser=tool_parser,
                             reasoning_parser=reasoning_parser)[0]

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        prebuilt_request: Optional[Any] = None,
    ) -> Iterator[StreamDelta]:
        """Stream generation deltas for a single message list.

        Runs ``handleRequest`` in a background thread with a
        ``StreamChannel`` attached, yielding ``StreamDelta`` objects as
        tokens are produced.
        """
        params = sampling_params or SamplingParams()
        state = {}

        def _cancel():
            channel = state.get("channel")
            if channel is not None:
                channel.cancel()

        def _iterate():
            self._ensure_open()
            channel = self._rt.StreamChannel.create()
            state["channel"] = channel
            channel.set_skip_special_tokens(params.skip_special_tokens)

            # The HTTP layer owns admission for a prebuilt request. Direct
            # callers acquire the per-LLM gate here.
            sem = None if prebuilt_request is not None else self._admission()
            if sem is not None:
                sem.acquire()
            try:
                self._ensure_open()
                if prebuilt_request is not None:
                    request = prebuilt_request
                    request.stream_channels = [channel]
                else:
                    request = self._make_generation_request(
                        messages,
                        params,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream_channel=channel,
                    )
            except BaseException:
                if sem is not None:
                    sem.release()
                raise

            error_holder = [None]

            def _run():
                try:
                    self._handle_request(request)
                except Exception as exc:
                    error_holder[0] = exc
                    channel.cancel()
                finally:
                    if sem is not None:
                        sem.release()

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()

            try:
                while True:
                    chunk = channel.wait_pop(timeout_ms=200)
                    if chunk is None:
                        if channel.is_finished() or channel.is_cancelled():
                            break
                        continue
                    reason = finish_reason_name(
                        self._rt, chunk.reason) if chunk.finished else None
                    yield StreamDelta(
                        text=chunk.text,
                        token_ids=list(chunk.token_ids),
                        prompt_tokens=(chunk.prompt_token_count
                                       if chunk.prompt_token_count >= 0 else
                                       None),
                        finished=chunk.finished,
                        finish_reason=reason,
                        logprobs=_convert_logprobs(chunk.logprobs),
                    )
                    if chunk.finished:
                        break
            finally:
                if not (channel.is_finished() or channel.is_cancelled()):
                    channel.cancel()
                worker.join()

            if error_holder[0] is not None:
                raise error_holder[0]

        return _CancellableIterator(_iterate(), _cancel)

    def generate_stream_with_audio(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        audio_params: Optional[AudioParams] = None,
        prebuilt_request: Optional[Any] = None,
    ) -> Iterator[StreamDelta]:
        """Stream text and audio deltas for a single Omni request.

        Runs the Thinker-Talker streaming pipeline in a background thread.
        Text deltas arrive through a ``StreamChannel`` and PCM chunks through
        an ``AudioStreamChannel``; the two are interleaved into one generator.
        Admission follows generate_stream: the HTTP layer owns the gate when
        it passes ``prebuilt_request``; otherwise it is acquired here.
        """
        if not self._layout.has_speech:
            raise ValueError("Omni audio output not available: talker / "
                             "code_predictor / code2wav engines not loaded.")
        params = sampling_params or SamplingParams()
        state = {}

        def _cancel():
            stream = state.get("stream")
            if stream is not None:
                stream.close()
            for name in ("text_channel", "audio_channel"):
                channel = state.get(name)
                if channel is not None:
                    channel.cancel()

        def _iterate():
            self._ensure_open()
            channel = self._rt.StreamChannel.create()
            audio_channel = self._rt.AudioStreamChannel()
            state["text_channel"] = channel
            state["audio_channel"] = audio_channel
            channel.set_skip_special_tokens(True)
            omni_params = _native_audio_params(self._rt, audio_params
                                               or AudioParams())
            sem = None if prebuilt_request is not None else self._admission()
            if sem is not None:
                sem.acquire()
            try:
                self._ensure_open()
                if prebuilt_request is not None:
                    request = prebuilt_request
                    request.stream_channels = [channel]
                else:
                    request = self._make_generation_request(
                        messages,
                        params,
                        stream_channel=channel,
                    )
            except BaseException:
                if sem is not None:
                    sem.release()
                raise

            def _run():
                # Audio and text share one runtime and CUDA stream.
                with self._infer_guard():
                    self._ensure_open()
                    self._runtime.handle_request_streaming_audio(
                        request, audio_channel, omni_params)

            stream = _pump_channels(self._rt,
                                    _run,
                                    channel,
                                    audio_channel,
                                    sem=sem)
            state["stream"] = stream
            yield from stream

        return _CancellableIterator(_iterate(), _cancel)

    # ------------------------------------------------------------------
    # Server API
    # ------------------------------------------------------------------

    def generate_speech_stream(
        self,
        text: str,
        audio_params: Optional[AudioParams] = None,
    ) -> Iterator[StreamDelta]:
        """Standalone TTS on the Omni stack: synthesize ``text`` directly.

        No Thinker generation pass — the input text goes straight to the
        Talker. Yields audio-only StreamDeltas.
        """
        if not self._layout.has_speech:
            raise ValueError("TTS not available: Omni audio components "
                             "(talker/code_predictor/code2wav) not loaded")
        sem = self._admission()
        return _stream_tts(self._rt,
                           self._runtime,
                           text,
                           audio_params or AudioParams(),
                           sem,
                           infer_guard=self._infer_guard(),
                           ensure_open=self._ensure_open)

    def list_voices(self) -> List[str]:
        """Speaker names accepted as ``voice``; empty when not Omni-capable."""
        if not self._layout.has_speech:
            return []
        self._ensure_open()
        return sorted(self._runtime.get_speaker_names())

    def serve(self,
              host: str = "0.0.0.0",
              port: int = 8000,
              *,
              served_model_name: str = "",
              api_key: str = "",
              reasoning_parser: str = "auto",
              tool_call_parser: str = "auto",
              enable_auto_tool_choice: bool = False,
              max_queued_requests: int = 16,
              queue_timeout: float = 600.0,
              allowed_local_media_path: Optional[str] = None) -> None:
        """Start the HTTP frontend for this runtime."""
        from ..api.app import run_http_server
        from ..config import ApiConfig
        from .engine_client import EngineClient

        config = ApiConfig(
            host=host,
            port=port,
            served_model_name=served_model_name,
            api_key=api_key,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            enable_auto_tool_choice=enable_auto_tool_choice,
            max_queued_requests=max_queued_requests,
            queue_timeout=queue_timeout,
            allowed_local_media_path=allowed_local_media_path or "",
        )
        run_http_server(EngineClient(self, config), config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> str:
        """Path to the resolved model checkpoint."""
        return self._model_dir

    @property
    def model_id(self) -> str:
        """User-facing model identifier supplied at initialization."""
        return self._model_id

    @property
    def bundle_dir(self) -> str:
        """Profile-specific engine bundle selected from the cache."""
        return self._bundle_dir

    @property
    def cache_dir(self) -> str:
        """Root containing downloaded checkpoints and built bundles."""
        return self._cache_dir

    @property
    def max_batch_size(self) -> int:
        """Maximum batch size supported by the loaded engine."""
        return self._max_batch_size

    @property
    def video_capable(self) -> bool:
        """Whether this model bundle supports video input."""
        try:
            self._video_model_family()
            return True
        except ValueError:
            return False

    @property
    def has_draft_model(self) -> bool:
        """Whether speculative decoding is active."""
        return self._runtime.has_draft_model()

    @property
    def context_cache_enabled(self) -> bool:
        """Whether this runtime reuses matching text prefixes."""
        return self._context_cache_config.enabled

    def get_context_cache_metrics(self):
        """Return native reuse counters, or ``None`` when reuse is disabled."""
        with self._infer_guard():
            self._ensure_open()
            return self._runtime.get_context_cache_metrics()

    @property
    def bundle_layout(self) -> BundleLayout:
        """Immutable component contract for the selected model bundle."""
        return self._layout


class TTS:
    """Checkpoint-direct serving for a model-owned TTS component stack."""

    runtime_kind = "tts"
    has_draft_model = False

    def __init__(
        self,
        model: str,
        *,
        cache_dir: str = "",
        engine_cache_max_size_gb: float = 50.0,
        clear_engine_cache: bool = False,
        max_input_len: int = _DEFAULT_MAX_INPUT_LEN,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_kv_cache_capacity: int = _DEFAULT_MAX_KV_CACHE_CAPACITY,
        build_options: Optional["BuildOptions"] = None,
    ) -> None:
        if not model:
            raise ValueError("'model' must be provided")
        if (isinstance(engine_cache_max_size_gb, bool)
                or not math.isfinite(engine_cache_max_size_gb)
                or engine_cache_max_size_gb <= 0):
            raise ValueError("engine_cache_max_size_gb must be positive")
        from .engine_build import BuildOptions, cache_root, prepare_model
        options = build_options or BuildOptions(
            max_input_len=max_input_len,
            max_batch_size=max_batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
        )
        if options.spec_type != "none":
            raise ValueError(
                "standalone TTS models do not support speculative decoding")
        prepared = prepare_model(
            model,
            cache_dir,
            options,
            max_cache_size_bytes=int(engine_cache_max_size_gb * (1 << 30)),
            clear_cache=clear_engine_cache,
        )
        self._layout = inspect_bundle(prepared.bundle_dir)
        if not self._layout.has_speech:
            raise ValueError(
                f"model {model!r} does not provide a complete TTS runtime")

        self._model_dir = prepared.model_dir
        self._model_id = _derive_model_id(model)
        self._cache_dir = cache_root(cache_dir)
        self._bundle_dir = prepared.bundle_dir
        self._rt = _import_runtime()
        logger.info("Loading TTS runtime from %s", prepared.bundle_dir)
        self._runtime = self._rt.TTSRuntime(
            talker_engine_dir=self._layout.talker_dir,
            code_predictor_engine_dir=self._layout.code_predictor_dir,
            code2wav_engine_dir=self._layout.code2wav_dir,
            tokenizer_dir=self._layout.talker_dir,
            checkpoint_dir=prepared.model_dir,
        )
        logger.info("TTS runtime ready")
        self._admission_sem = threading.Semaphore(1)
        self._close_lock = threading.Lock()
        self._closed = False

    def _admission(self):
        """Per-instance admission gate (mirrors LLM._admission)."""
        return self._admission_sem

    def generate_speech_stream(
        self,
        text: str,
        audio_params: Optional[AudioParams] = None,
    ) -> Iterator[StreamDelta]:
        """Synthesize ``text``; yields audio-only StreamDeltas."""
        sem = self._admission()
        return _stream_tts(self._rt,
                           self._runtime,
                           text,
                           audio_params or AudioParams(),
                           sem,
                           ensure_open=self._ensure_open)

    def _ensure_open(self) -> None:
        if self._closed or self._runtime is None:
            raise RuntimeError("Edge-LLM runtime is closed")

    def list_voices(self) -> List[str]:
        """Speaker names accepted as ``voice``."""
        self._ensure_open()
        return sorted(self._runtime.get_speaker_names())

    def close(self) -> None:
        """Drain active speech generation and release native resources."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            with self._admission_sem:
                self._runtime = None

    def __enter__(self) -> "TTS":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def bundle_dir(self) -> str:
        return self._bundle_dir

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def bundle_layout(self) -> BundleLayout:
        """Immutable component contract for the selected model bundle."""
        return self._layout

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP server (speech endpoint only)."""
        from ..api.app import run_http_server
        from ..config import ApiConfig
        from .engine_client import EngineClient

        config = ApiConfig(host=host, port=port)
        run_http_server(EngineClient(self, config), config)


def load_model(**kwargs):
    """Select the model-specific runtime from provider checkpoint metadata."""
    from .engine_build import resolve_model_dir

    original_model = kwargs["model"]
    resolved = resolve_model_dir(original_model, kwargs.get("cache_dir", ""))
    with open(os.path.join(resolved, "config.json"), encoding="utf-8") as file:
        model_type = json.load(file).get("model_type")
    if model_type == "qwen3_tts":
        runtime_class = TTS
        for name in ("draft_top_k", "draft_step", "verify_tree_size"):
            if kwargs.pop(name, None) is not None:
                raise ValueError(
                    f"standalone TTS models do not support {name}")
        if kwargs.pop("speculative_config", None):
            raise ValueError(
                "standalone TTS models do not support speculative decoding")
        context_cache = ContextCacheConfig.parse(
            kwargs.pop("context_cache_config", None))
        if context_cache.enabled:
            raise ValueError(
                "standalone TTS models do not use a KV context cache")
    else:
        runtime_class = LLM
    runtime = runtime_class(**{**kwargs, "model": resolved})
    runtime._model_id = _derive_model_id(original_model)
    return runtime


# ---------------------------------------------------------------------------
# Message conversion & image loading
# ---------------------------------------------------------------------------


def finish_reason_name(rt_module, reason) -> Optional[str]:
    """Map a C++ FinishReason enum value to its OpenAI-compatible string.

    NOT_FINISHED maps to None — reaching this function with a non-terminal
    reason indicates a bug; surfacing None instead of silently returning "stop"
    makes it visible. The fallback "stop" catches truly-unknown enum values
    (e.g. future C++ enum additions). STOP_WORDS and END_ID both map to "stop"
    since OpenAI does not distinguish them.
    """
    return {
        rt_module.FinishReason.NOT_FINISHED: None,
        rt_module.FinishReason.END_ID: "stop",
        rt_module.FinishReason.LENGTH: "length",
        rt_module.FinishReason.CANCELLED: "cancelled",
        rt_module.FinishReason.ERROR: "error",
        rt_module.FinishReason.STOP_WORDS: "stop",
    }.get(reason, "stop")


def _convert_messages_to_cpp(rt_module, messages: List[Dict[str, Any]]):
    """Convert Python message dicts to C++ Message objects."""
    cpp_messages = []
    for msg in messages:
        cpp_msg = rt_module.Message()
        cpp_msg.role = msg["role"]
        content = msg["content"]
        contents_list = []
        if isinstance(content, str):
            contents_list.append(rt_module.MessageContent("text", content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    contents_list.append(rt_module.MessageContent(
                        "text", item))
                elif isinstance(item, dict):
                    ct = item.get("type", "text")
                    if ct == "text":
                        contents_list.append(
                            rt_module.MessageContent(
                                "text",
                                item.get("text", ""),
                            ))
                    elif ct in ("image", "image_url"):
                        contents_list.append(
                            rt_module.MessageContent("image", ""))
                    elif ct in ("video", "video_url"):
                        # Frames are decoded out-of-band by _load_image_buffers; the chat
                        # template expands this placeholder into the video triplet and the
                        # ViT runner keys off ImageData.isVideo.
                        contents_list.append(
                            rt_module.MessageContent("video", ""))
                    elif ct in ("audio", "input_audio", "audio_url"):
                        # Audio bytes are decoded out-of-band by
                        # `_load_audio_buffers`; the chat template just emits
                        # an opaque audio placeholder here. The per-model
                        # audio runner expands that into model-specific
                        # special tokens (Qwen3: <|audio_start|> +
                        # N×<|audio_pad|> + <|audio_end|>; Nemotron-Omni:
                        # N×<so_embedding>).
                        contents_list.append(
                            rt_module.MessageContent("audio", ""))
                    else:
                        raise ValueError(f"Unsupported content type: {ct}")
        cpp_msg.contents = contents_list
        cpp_messages.append(cpp_msg)
    return cpp_messages


def _load_image_buffers(rt_module,
                        messages: List[Dict[str, Any]],
                        video_family_fn=lambda: "qwen",
                        video_frame_limits_fn=lambda: {}):
    """Build the ordered ImageData list for the messages: images and videos
    share one list the C++ runner matches positionally against the
    <|image_pad|> / <|video_pad|> placeholders, so append in message order."""
    images = []
    items = [
        item for msg in messages if isinstance(msg.get("content"), list)
        for item in msg["content"] if isinstance(item, dict)
    ]
    from ..media.media_source import resolve_image_message

    image_sources = {
        id(item): resolve_image_message(item)
        for item in items if item.get("type") in ("image", "image_url")
    }
    # Videos and images share one engine token profile: track the remaining
    # budget so multiple media cannot each claim full capacity. Lazy so
    # non-video requests never touch the video family whitelist.
    has_video = any(
        item.get("type") in ("video", "video_url") for item in items)
    family = video_family_fn() if has_video else "qwen"
    if has_video and family == "nemotron":
        # The C++ Nemotron video path handles exactly one video and no mixed-in
        # images per request (batch of one); reject other layouts here rather
        # than letting them fail inside the runner.
        n_videos = sum(1 for it in items
                       if it.get("type") in ("video", "video_url"))
        n_images = sum(1 for it in items
                       if it.get("type") in ("image", "image_url"))
        if n_videos > 1 or n_images > 0:
            raise ValueError(
                "Nemotron-Omni video requests support exactly one video and no "
                f"images (got {n_videos} videos, {n_images} images)")
    limits = video_frame_limits_fn() if has_video else {}
    budget = limits.get("max_image_tokens") if limits else None
    video_tokens = 0
    # Pre-pruning token count for the engine-minimum check: the ViT processes
    # every tubelet, so Nemotron's EVS-pruned estimate would understate what the
    # min-profile actually receives. Non-EVS families track the same value.
    video_raw_tokens = 0
    # Request-wide decoded-pixel budget: several videos each under the
    # per-video ceiling must not jointly exhaust host memory.
    pixel_budget = None
    # Request-wide cu_seqlens group budget (Qwen only; InternVL has no
    # cu_seqlens binding): use builder-recorded capacity or derive it from the
    # current component profile.
    cu_budget = None
    if limits and family != "internvl":
        cu_budget = (limits.get("max_cu_seqlen_groups")
                     or limits["max_image_tokens"] //
                     max(1, limits.get("min_image_tokens", 1)))
    image_upper = 0
    # Phase 1: reserve every image up front so the video sampler's budget is
    # order-independent ([image, video] and [video, image] behave identically).
    if budget is not None:
        from ..media.video_sampling import estimate_image_tokens
        for item in items:
            if item.get("type") not in ("image", "image_url"):
                continue
            source = image_sources[id(item)]
            if isinstance(source, str) and os.path.isfile(source):
                est = estimate_image_tokens(source,
                                            family,
                                            limits,
                                            do_resize=bool(
                                                item.get("do_resize", True)))
            else:
                est = int(limits.get("max_image_tokens_per_image", 0))
            image_upper += est
            budget -= est
            if cu_budget is not None:
                # One cu_seqlens entry per image (Qwen families only;
                # InternVL has no cu_seqlens binding).
                cu_budget -= 1
    # Phase 2: build the buffers in original message order (the C++ runner
    # matches them positionally against the placeholders).
    for item in items:
        itype = item.get("type")
        if itype in ("image", "image_url"):
            source = image_sources[id(item)]
            image = (rt_module.load_image_from_bytes(source) if isinstance(
                source, bytes) else rt_module.load_image_from_path(source))
            image.do_resize = bool(item.get("do_resize", True))
            images.append(image)
        elif itype in ("video", "video_url"):
            from ..media.video_sampling import (MAX_DECODE_PIXELS,
                                                load_video_buffer)
            if pixel_budget is None:
                pixel_budget = MAX_DECODE_PIXELS
            buffer, est_tokens, used_px, used_groups = load_video_buffer(
                rt_module,
                item,
                family,
                frame_limits=limits,
                budget=budget,
                pixel_budget=pixel_budget,
                cu_budget=cu_budget)
            images.append(buffer)
            video_tokens += est_tokens
            raw_tokens = est_tokens
            if family == "nemotron":
                from ..media.video_sampling import _nemotron_tubelet_geometry
                geom = _nemotron_tubelet_geometry(limits)
                if geom:
                    t_frames, tokens_per_tubelet, _q = geom
                    raw_tokens = (-(-buffer.frames // t_frames)) \
                        * tokens_per_tubelet
            video_raw_tokens += raw_tokens
            pixel_budget -= used_px
            if cu_budget is not None:
                cu_budget -= used_groups
            if budget is not None:
                budget -= est_tokens
    # Engine bounds are request-wide (all media accumulate in one ViT batch),
    # so validate after the loop: two videos jointly reaching the minimum are
    # fine, one alone may not be.
    if cu_budget is not None and cu_budget < 0:
        raise ValueError(
            "request media exceed the visual engine's cu_seqlens capacity; "
            "reduce the media count")
    if budget is not None and budget < 0:
        raise ValueError(
            "request media need more visual tokens than the engine's "
            f"budget of {limits['max_image_tokens']}; reduce the media in "
            "the request")
    if (video_raw_tokens or image_upper) and limits and \
            limits.get("min_image_tokens"):
        # The engine minimum is request-wide; the upper estimate is pre-EVS
        # (raw tubelets for Nemotron). It can fall short for a too-short clip or
        # do_resize=false media; resized per-item images are floored above this.
        upper_tokens = video_raw_tokens + image_upper
        if upper_tokens < limits["min_image_tokens"]:
            raise ValueError(
                f"request media yield ~{upper_tokens} visual tokens but the "
                f"engine needs at least {limits['min_image_tokens']}; use "
                "longer videos or raise nframes/fps")
    return images


def _load_audio_buffers(rt_module, messages: List[Dict[str, Any]]):
    """Load audio content from messages into AudioData buffers.

    Returns an empty list when no audio is present, keeping the byte-identical
    fast path for text-only and image-only requests.
    """
    from ..media.audio_preprocess import load_audio_buffers
    return load_audio_buffers(rt_module, messages)
