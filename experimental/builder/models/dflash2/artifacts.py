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
"""DFlash2 runtime configuration and selector sidecar."""

import os

from ...core import contracts
from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from ...core.artifacts.tensors import save_safetensors
from ...core.safetensors_np import SafetensorsStore
from . import weights

_SELECTOR_KEYS = {
    "predecessor_codebook": "candidate_selector.predecessor_codebook",
    "successor_codebook": "candidate_selector.successor_codebook",
}


def _write_selector_sidecar(config, args, engine_dir: str) -> None:
    output_dir = contracts.component_spec(
        args.resolved_component).output_dir(engine_dir)
    expected_shape = (config.vocab_size, config.dflash2_selector_rank)
    tensors = {}
    with SafetensorsStore(args.model_dir) as store:
        for output_name, checkpoint_name in _SELECTOR_KEYS.items():
            if not store.has(checkpoint_name):
                raise KeyError(
                    "DFlash2 draft checkpoint is missing required tensor "
                    f"{checkpoint_name!r}")
            tensor = store.get_f16(checkpoint_name)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"DFlash2 selector tensor {checkpoint_name!r} must have "
                    f"shape {expected_shape}, got {tuple(tensor.shape)}")
            tensors[output_name] = tensor

    save_safetensors(os.path.join(output_dir, "dflash2_selector.safetensors"),
                     tensors)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    del bundle
    if config is None:
        raise ValueError("DFlash2 artifacts require an LLM configuration")
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights)
    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        _write_selector_sidecar(config, args, engine_dir)
