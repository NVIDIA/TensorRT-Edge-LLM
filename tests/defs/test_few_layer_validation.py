# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Few-layer numeric validation smoke test.

Early-fail smoke: runs the first N decoder layers of a checkpoint through both
a PyTorch golden and the EdgeLLM engine, comparing per-round logits + KV /
recurrent / conv state (see ``scripts/few-layer-validation.sh``). The script's
exit code is the PASS/FAIL gate; this test is a thin wrapper that resolves the
checkpoint, points the script at the CI build, and runs it.

Host (server) only: the golden runs PyTorch + transformers locally, so the test
skips under remote (edge-device) execution. The parametrization is driven by the
``test_param`` ids listed in ``tests/test_lists/*.yml`` (one per model)."""

import logging
import os
import sys
from typing import Optional

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import _HF_CHECKPOINT_FILES, _find_directory
from .utils.device import DeviceConfig

# test_param id -> validation spec.
#   dir_name      : checkpoint dir/relpath, searched recursively under the model
#                   roots (llm_models_dir, then edgellm_data_dir).
#   num_layers    : leading decoder layers to validate.
#   cos_threshold : min per-tensor cosine. Quantized recipes need a looser one than
#                   fp16, and MoE looser still; docs/.../few-layer-validation.md
#                   ("Why two gates") derives where each number comes from.
#   needs_cutedsl : True when the truncated graph uses a CuTe DSL kernel.
#   min_compute_capability: first architecture supported by the model precision.
#   timeout       : seconds for the whole script. Export and build scale with the
#                   checkpoint, not with num_layers, so a large MoE needs far more.
#   extra_args    : the decoding / cache mode a case exercises. The golden is plain
#                   autoregressive in every mode, since both MTP and context reuse
#                   are defined by producing what a vanilla full prefill would.
_FEW_LAYER_MODELS = {
    "Qwen3-0.6B": {
        "dir_name": "Qwen3/Qwen3-0.6B",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": False,
    },
    # ModelOpt export keys the body under ``backbone.``; the golden realigns it to
    # the native ``model.`` prefix (see _load_quantized_state). The threshold is
    # loose for headroom across arches, not because this model needs it.
    "NVIDIA-Nemotron-3-Nano-4B-NVFP4": {
        "dir_name": "NVIDIA-Nemotron-3-Nano-4B-NVFP4",
        "num_layers": 4,
        "cos_threshold": 0.96,
        "needs_cutedsl": True,
    },
    # GDN hybrid, and the broadest case here: three linear_attention layers plus
    # one full_attention, so one run exercises the recurrent + conv state and the
    # KV cache side by side, including the dense renumbering that keeps the two
    # sides' layer indices aligned.
    "Qwen3.5-0.8B": {
        "dir_name": "Qwen3.5-0.8B",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": True,
        "min_compute_capability": 80,
    },
    # MTP speculative decoding against the same vanilla golden: the base model's
    # committed state has to match plain decoding whatever the draft proposed.
    # Acceptance collapses to one token per round on a truncated base (the draft
    # head is fed a hidden state from the wrong depth), so this covers the
    # verify / accept / commit path rather than deep multi-token rewinds.
    "Qwen3.5-0.8B-MTP": {
        "dir_name": "Qwen3.5-0.8B",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": True,
        "min_compute_capability": 80,
        "extra_args": ["--mtp"],
    },
    # Three requests sharing a long prefix at batch_size 1: the first populates
    # the cache and the rest hit it, while the golden prefills each in full.
    "Qwen3.5-0.8B-ContextReuse": {
        "dir_name":
        "Qwen3.5-0.8B",
        "num_layers":
        4,
        "cos_threshold":
        0.99,
        "needs_cutedsl":
        True,
        "min_compute_capability":
        80,
        "extra_args": [
            "--context-reuse",
            "--input-file",
            "tests/test_cases/llm_context_reuse.json",
        ],
    },
    # Same layer mix as Qwen3.5-0.8B but with a 256-expert MoE FFN per layer.
    # Discrete routing and the wide LM head amplify small hidden-state drift in
    # logits; recurrent and KV states remain above 0.99. See "Why two gates".
    "Qwen3.5-35B-A3B-GPTQ-Int4": {
        "dir_name": "Qwen3.5-35B-A3B-GPTQ-Int4",
        "num_layers": 4,
        "cos_threshold": 0.96,
        "needs_cutedsl": True,
        "min_compute_capability": 80,
        "timeout": 3600,
    },
    # Gemma4 dense. First 4 layers are sliding-window attention, run through the
    # CuTe DSL FMHA-v2 vision-block kernel (validated at cos 0.99972).
    "gemma-4-12B-it": {
        "dir_name": "gemma/gemma-4-12B-it",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": True,
    },
    # Gemma4 26B-A4B MoE, NVFP4 (the only supported precision for this model --
    # see docs/source/user_guide/getting_started/supported-models.md). The
    # golden swaps HF's stacked expert parameters for per-expert placeholders
    # so each separately quantized NVFP4 projection can be patched. Worst
    # cosine drifts to ~0.929 by the last decode round, an NVFP4 floor like
    # Nemotron NVFP4 above (validated worst 0.92888).
    "gemma-4-26B-A4B-NVFP4": {
        "dir_name": "gemma/nvidia-Gemma-4-26B-A4B-NVFP4",
        "num_layers": 4,
        "cos_threshold": 0.92,
        "needs_cutedsl": True,
    },
}


def _resolve_model_dir(env_config: EnvironmentConfig,
                       dir_name: str) -> Optional[str]:
    """Locate a checkpoint dir under the known model roots (recursive search)."""
    roots = tuple(
        filter(None, (env_config.llm_models_dir, env_config.edgellm_data_dir)))
    for root in roots:
        for parent in (root, os.path.join(root, "models")):
            found = _find_directory(parent,
                                    dir_name,
                                    max_depth=0,
                                    require_files=_HF_CHECKPOINT_FILES)
            if found:
                return found

    for root in roots:
        found = _find_directory(root,
                                dir_name,
                                require_files=_HF_CHECKPOINT_FILES)
        if found:
            return found
    return None


def test_few_layer_validation(test_param: str, env_config: EnvironmentConfig,
                              remote_config, test_logger: logging.Logger):
    """Functional few-layer golden-vs-engine numeric smoke."""
    spec = _FEW_LAYER_MODELS.get(test_param)
    if spec is None:
        pytest.fail(f"unknown few-layer model id '{test_param}'; "
                    f"known ids: {sorted(_FEW_LAYER_MODELS)}")

    # Host (server) only: the golden runs PyTorch + transformers locally, which
    # the edge boards (remote execution) do not have.
    if remote_config is not None:
        pytest.skip("few-layer validation runs on the host (server) only; "
                    "the PyTorch golden needs torch + transformers")

    # These cases require target-specific CuTe DSL kernels and/or precision.
    device_config = DeviceConfig.auto_detect(remote_config, test_logger)
    cc = device_config.compute_capability
    if cc is None:
        pytest.fail(f"cannot determine target SM for {test_param}")
    minimum_cc = spec.get("min_compute_capability",
                          100 if spec["needs_cutedsl"] else 0)
    if cc < minimum_cc:
        pytest.skip(f"{test_param} requires SM{minimum_cc}+; this runner is "
                    f"compute capability {cc}")

    model_dir = _resolve_model_dir(env_config, spec["dir_name"])
    if model_dir is None:
        pytest.skip(f"checkpoint '{spec['dir_name']}' not found under "
                    f"{env_config.llm_models_dir} / "
                    f"{env_config.edgellm_data_dir}")

    # build_dir may be relative ("build"); the script needs an absolute path
    # since it resolves binaries against it independent of its own CWD.
    build_dir = env_config.build_dir
    if not os.path.isabs(build_dir):
        build_dir = os.path.join(env_config.llm_sdk_dir, build_dir)

    script = os.path.join(env_config.llm_sdk_dir, "scripts",
                          "few-layer-validation.sh")
    cmd = [
        "bash",
        script,
        "--model",
        model_dir,
        "--num-layers",
        str(spec["num_layers"]),
        "--build-dir",
        build_dir,
        # The pytest interpreter has torch + transformers + the TRT wheel.
        "--python",
        sys.executable,
        "--cos",
        str(spec["cos_threshold"]),
        "--target-sm",
        str(cc),
    ]
    # --input-file is resolved against the repo, which is where the script runs from.
    for arg in spec.get("extra_args", []):
        cmd.append(
            os.path.join(env_config.llm_sdk_dir, arg) if arg.
            startswith("tests/test_cases/") else arg)

    with timer_context(f"few-layer-validation [{test_param}]", test_logger):
        result = run_command(cmd=cmd,
                             remote_config=remote_config,
                             timeout=spec.get("timeout", 1200),
                             logger=test_logger)

    if not result["success"]:
        pytest.fail(f"few-layer validation FAILED for {test_param} "
                    f"(exit {result.get('returncode')}); see log above")
