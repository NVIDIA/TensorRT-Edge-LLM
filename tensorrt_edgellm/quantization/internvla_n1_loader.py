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
"""Load InternVLA-N1's System 2 for quantization.

An InternVLA-N1 checkpoint declares ``model_type: "internvla_n1"`` and ships no
modeling code and no ``auto_map``, so every ``AutoModel*`` factory fails on it --
the same situation ``qwen3_asr_loader`` handles for Qwen3-ASR.

Underneath the declaration, System 2 *is* a Qwen2.5-VL: the weights are
``visual.blocks.*``, ``visual.merger.*``, ``model.layers.*`` and ``lm_head.*`` in
the stock layout, and ``vision_config.model_type`` already says ``qwen2_5_vl``.
So the checkpoint loads by presenting it as one, which is the same move
``_prepare_alpamayo_visual_params`` makes when it forces Alpamayo's tower to
``qwen3_vl``.

The remaining tensors -- ``model.traj_dit.*``, ``model.rgb_model.*``,
``model.rgb_resampler.*``, ``model.memory_encoder.*``, ``model.cond_projector.*``
and ``model.action_*`` -- are System 1 and the bridge. They are not part of the
graph being quantized, so the loader ignores them and
:func:`restore_system1_tensors` copies them into the export afterwards.

That copy is not optional. Without it the export is missing the bridge, and the
failure is silent: ``tensorrt-edgellm-export`` matches no keys for those modules,
leaves them default-initialised and still exits 0, so the first sign of trouble
is a garbled trajectory much later.
"""
import json
import os
import shutil
import tempfile

#: Tensors that belong to System 1 or the System-2 -> System-1 bridge. They are
#: absent from the quantized graph and must be copied into the export verbatim.
SYSTEM1_PREFIXES = (
    "model.traj_dit.",
    "model.rgb_model.",
    "model.rgb_resampler.",
    "model.memory_encoder.",
    "model.cond_projector.",
    "model.action_encoder.",
    "model.action_decoder.",
    "model.latent_queries",
)


def is_internvla_n1_model(model_dir: str) -> bool:
    """True if ``<model_dir>/config.json`` declares an InternVLA-N1 checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return json.load(f).get("model_type") == "internvla_n1"
    except (OSError, ValueError):
        return False


def load_internvla_n1_system2(model_dir: str, torch_dtype, device):
    """Load System 2 as a Qwen2.5-VL and return ``(model, tokenizer, processor)``.

    The checkpoint is presented to ``transformers`` through a temporary directory
    of symlinks whose ``config.json`` says ``qwen2_5_vl``. Symlinks keep this
    cheap on a 15 GB checkpoint, and rewriting only the copy means the source is
    never modified.
    """
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              AutoTokenizer)

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory(prefix="internvla_n1_s2_") as staging:
        for name in os.listdir(model_dir):
            if name == "config.json":
                continue
            os.symlink(os.path.join(model_dir, name),
                       os.path.join(staging, name))

        # System 1 lives under keys transformers will not recognise. Dropping the
        # declarations as well as the weights keeps the config a valid Qwen2.5-VL
        # one rather than a Qwen2.5-VL config with unexplained extras.
        config["model_type"] = "qwen2_5_vl"
        config["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        for key in ("system1", "model_cfg", "n_query"):
            config.pop(key, None)
        with open(os.path.join(staging, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model = AutoModelForImageTextToText.from_pretrained(
            staging,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staging)
        try:
            processor = AutoProcessor.from_pretrained(staging)
        except Exception:
            processor = None

    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor


def restore_system1_tensors(model_dir: str, output_dir: str) -> int:
    """Copy System 1 and the bridge from the source checkpoint into the export.

    Kept at the source dtype rather than quantized: the bridge is four rows
    through a Linear/GELU/Linear, so quantizing it saves nothing measurable and
    would put error directly on the tensor System 1 steers by.

    Returns the number of tensors copied.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return 0
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    wanted = [k for k in weight_map if k.startswith(SYSTEM1_PREFIXES)]
    if not wanted:
        return 0

    tensors = {}
    for key in wanted:
        with safe_open(os.path.join(model_dir, weight_map[key]),
                       framework="pt") as f:
            tensors[key] = f.get_tensor(key)
    save_file(tensors, os.path.join(output_dir, "model-system1.safetensors"))

    # The index is rebuilt over every shard present rather than edited in place:
    # some quantization formats write a single unsharded model.safetensors with
    # no index at all, so an edit would work for the sharded formats and
    # silently do nothing for the others.
    weight_map_out, total = {}, 0
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(output_dir, name)
        total += os.path.getsize(path)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                weight_map_out[key] = name
    with open(os.path.join(output_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": total
                },
                "weight_map": weight_map_out
            },
            f,
            indent=2)

    # The exported config must declare InternVLA-N1 again: the loader presented
    # the checkpoint as a Qwen2.5-VL, and an export left that way would be
    # dispatched to the wrong exporter and lose System 1 a second time.
    _restore_config(model_dir, output_dir)
    return len(tensors)


def _restore_config(model_dir: str, output_dir: str) -> None:
    src_path = os.path.join(model_dir, "config.json")
    dst_path = os.path.join(output_dir, "config.json")
    if not (os.path.isfile(src_path) and os.path.isfile(dst_path)):
        return
    with open(src_path) as f:
        src = json.load(f)
    with open(dst_path) as f:
        dst = json.load(f)
    # Keep everything quantization wrote (quantization_config and friends) and
    # restore only the identity and the System-1 declarations it could not know.
    dst["model_type"] = src["model_type"]
    dst["architectures"] = src["architectures"]
    for key in ("system1", "model_cfg", "n_query"):
        if key in src:
            dst[key] = src[key]
    with open(dst_path, "w") as f:
        json.dump(dst, f, indent=2)


def copy_auxiliary_files(model_dir: str, output_dir: str) -> None:
    """Copy tokenizer/processor files the quantizer does not regenerate."""
    for name in ("chat_template.json", "preprocessor_config.json",
                 "added_tokens.json", "special_tokens_map.json"):
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))


__all__ = [
    "is_internvla_n1_model",
    "load_internvla_n1_system2",
    "restore_system1_tensors",
    "copy_auxiliary_files",
    "SYSTEM1_PREFIXES",
]
