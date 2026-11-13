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
"""DSpark runtime configuration and sequential-head sidecars."""

import json
import os

import numpy as np

from ...core import contracts
from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from ...core.artifacts.tensors import save_safetensors
from ...core.safetensors_np import SafetensorsStore
from . import weights

_HEAD_KEYS = {
    "markov_w1": "markov_head.markov_w1.weight",
    "markov_w2": "markov_head.markov_w2.weight",
    "confidence_weight": "confidence_head.proj.weight",
    "confidence_bias": "confidence_head.proj.bias",
}

_NVFP4_HEAD_SCALES = {
    "markov_w2": ("markov_head.markov_w2.weight_scale",
                  "markov_head.markov_w2.weight_scale_2"),
}


def _load_head_tensor(store: SafetensorsStore, output_name: str,
                      checkpoint_name: str):
    scale_keys = _NVFP4_HEAD_SCALES.get(output_name)
    checkpoint_dtype = store.dtype(checkpoint_name)
    if checkpoint_dtype in ("F4", "F4_E2M1", "I8", "U8"):
        expected = " / ".join(scale_keys or ())
        raise KeyError(
            f"DSpark head tensor {checkpoint_name!r} is packed but its "
            f"NVFP4 scale tensors are missing: {expected}")
    return store.get_f16(checkpoint_name), {
        "checkpoint_dtype": checkpoint_dtype,
    }


def _write_head_sidecars(config, args, engine_dir: str) -> None:
    output_dir = contracts.component_spec(
        args.resolved_component).output_dir(engine_dir)
    required = {"markov_w1", "markov_w2"}
    if config.dspark_enable_confidence_head:
        required.update(("confidence_weight", "confidence_bias"))

    tensors = {}
    packed_w2 = {}
    source = {}
    with SafetensorsStore(args.model_dir) as store:
        for output_name, checkpoint_name in _HEAD_KEYS.items():
            if not store.has(checkpoint_name):
                if output_name in required:
                    raise KeyError(
                        "DSpark draft checkpoint is missing required tensor "
                        f"{checkpoint_name!r}")
                continue
            group_size = config.quant.module_group_size(checkpoint_name)
            if (output_name == "markov_w2" and store.dtype(checkpoint_name)
                    in ("F4", "F4_E2M1", "I8", "U8")):
                scale_keys = _NVFP4_HEAD_SCALES[output_name]
                if group_size != 16 or not all(
                        store.has(key) for key in scale_keys):
                    raise ValueError(
                        "DSpark packed W2 requires NVFP4 scales with group_size=16"
                    )
                packed = store.get_packed_fp4(checkpoint_name).copy()
                scales = store.get_fp8_bytes(scale_keys[0]).copy()
                global_scale = store.get_scalar_f32(scale_keys[1])
                if (packed.ndim != 2 or any(dim <= 0 for dim in packed.shape)
                        or packed.shape[1] % 8 != 0 or scales.shape
                        != (packed.shape[0], packed.shape[1] // 8)
                        or not np.isfinite(global_scale) or global_scale <= 0):
                    raise ValueError("Invalid DSpark packed W2 scales")
                packed_w2 = {
                    "packed_weight": packed,
                    "block_scales": scales,
                    "global_scale": np.asarray([global_scale],
                                               dtype=np.float32),
                }
                source[output_name] = {
                    "checkpoint_key": checkpoint_name,
                    "checkpoint_dtype": store.dtype(checkpoint_name),
                    "packed_shape": list(packed.shape),
                    "group_size": group_size,
                    "nvfp4_dequantized": False,
                }
                continue
            tensor, tensor_metadata = _load_head_tensor(
                store, output_name, checkpoint_name)
            if output_name == "confidence_weight" and tensor.ndim == 2:
                if tensor.shape[0] != 1:
                    raise ValueError(
                        "DSpark confidence weight must have one output row")
                tensor = tensor[0]
            tensors[output_name] = tensor
            source[output_name] = {
                "checkpoint_key": checkpoint_name,
                "shape": [int(dimension) for dimension in tensor.shape],
                "dtype": "float16",
                **tensor_metadata,
            }

    save_safetensors(os.path.join(output_dir, "dspark_heads.safetensors"),
                     tensors)
    packed_path = os.path.join(output_dir, "dspark_w2_nvfp4.safetensors")
    if packed_w2:
        save_safetensors(packed_path, packed_w2)
    elif os.path.exists(packed_path):
        os.remove(packed_path)
    info = {
        "format": "tensorrt-edgellm-dspark-heads-v1",
        "source": args.model_dir,
        "markov_head_type": config.dspark_markov_head_type,
        "markov_rank": config.dspark_markov_rank,
        "enable_confidence_head": config.dspark_enable_confidence_head,
        "confidence_head_with_markov":
        config.dspark_confidence_head_with_markov,
        "w2_format": "nvfp4" if packed_w2 else "fp16",
        "tensor_keys": sorted(tensors),
        "source_tensors": source,
    }
    with open(os.path.join(output_dir, "dspark_heads_info.json"),
              "w") as info_file:
        json.dump(info, info_file, indent=2)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    del bundle
    if config is None:
        raise ValueError("DSpark artifacts require an LLM configuration")
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights)
    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        _write_head_sidecars(config, args, engine_dir)
