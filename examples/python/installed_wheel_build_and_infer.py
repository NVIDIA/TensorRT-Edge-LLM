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
"""Build an engine and run one prompt using an installed EdgeLLM wheel."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import experimental.server as server_api
import tensorrt_edgellm
from tensorrt_edgellm._native.load import resolve_payload


def _arguments(values: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("engine_dir", type=Path)
    parser.add_argument("--prompt", default="Please introduce NVIDIA.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--expected-variant")
    parser.add_argument("--result", type=Path)
    return parser.parse_args(values)


def _require_installed_package() -> Path:
    environment = Path(sys.prefix).resolve(strict=True)
    package = Path(tensorrt_edgellm.__file__).resolve(strict=True)
    server = Path(server_api.__file__).resolve(strict=True)
    for name, path in (("tensorrt_edgellm", package), ("experimental.server",
                                                       server)):
        if not path.is_relative_to(environment):
            raise RuntimeError(
                f"Imported {name} from {path}, outside {environment}.")
    return package


def _build_and_infer(model_dir: Path, engine_dir: Path, prompt: str,
                     max_tokens: int) -> Tuple[str, int]:
    shutil.rmtree(engine_dir, ignore_errors=True)
    llm = server_api.LLM(
        model=str(model_dir),
        cache_dir=str(engine_dir),
        clear_engine_cache=True,
        max_input_len=128,
        max_kv_cache_capacity=256,
        max_batch_size=1,
    )
    try:
        outputs = llm.generate(
            [prompt],
            server_api.SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=max_tokens,
            ),
        )
    finally:
        llm.close()

    engines = list(engine_dir.rglob("*.engine"))
    if not engines or any(path.stat().st_size == 0 for path in engines):
        raise RuntimeError(
            "The installed high-level API produced no usable engine.")
    if len(outputs) != 1 or not outputs[0].text or not outputs[0].token_ids:
        raise RuntimeError("EdgeLLM returned no generated output.")
    return outputs[0].text, len(outputs[0].token_ids)


def main(values: Optional[Sequence[str]] = None) -> int:
    args = _arguments(values)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive.")
    package = _require_installed_package()
    model_dir = args.model_dir.resolve(strict=True)
    engine_dir = args.engine_dir.resolve()
    payload = resolve_payload()
    if args.expected_variant and payload.variant_id != args.expected_variant:
        raise RuntimeError(
            f"Selected {payload.variant_id}, expected {args.expected_variant}."
        )

    output_text, output_token_count = _build_and_infer(
        model_dir,
        engine_dir,
        args.prompt,
        args.max_tokens,
    )
    result = {
        "package": str(package),
        "variant_id": payload.variant_id,
        "extension": str(payload.extension),
        "plugin": str(payload.plugin),
        "engine_dir": str(engine_dir),
        "output_text": output_text,
        "output_token_count": output_token_count,
    }
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    print("INSTALLED_WHEEL_BUILD_AND_INFERENCE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
