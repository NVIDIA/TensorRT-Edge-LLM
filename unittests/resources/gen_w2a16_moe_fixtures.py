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
"""Generate checkpoint fixtures for unitTestKernelsMoe W2A16 tests.

The golden outputs use an independent GGML decoder and FP32 PyTorch GEMMs.
No native runner or test bridge is loaded. Weights are packed by the builder's
``tensorrt_edgellm.checkpoint.iq2_xxs`` module; ``raw.bin`` keeps the checkpoint
blocks so the C++ tests can also check the runtime checkpoint repack. Set
EDGE_LLM_W2A16_TEST_DATA to the output directory when running the C++ tests.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "kernelSrcs"))
sys.path.insert(0, str(_ROOT))
from w2a16_reference import dequantize_blocks

from tensorrt_edgellm.checkpoint.iq2_xxs import lookup_table, pack_weights

LAYERS = [1, 3, 5, 10, 14, 17, 21, 23, 28, 34, 37, 72, 74, 76]


def weights(path, layer):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    text = cfg.get("llm_config", cfg)
    block = text["block_configs"][layer]
    e, i = block["n_routed_experts"], block["moe_intermediate_size"]
    assert block["num_experts_per_tok"] == 9
    assert text["mlp_hidden_act"] == "relu2"
    index = json.loads(
        (path / "model.safetensors.index.json").read_text())["weight_map"]
    suffix = f".layers.{layer}.mixer.experts.0.up_proj.weight"
    matches = [
        key[:-len(".0.up_proj.weight")] for key in index
        if key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one expert prefix for layer {layer}, found {matches}")
    prefix = matches[0]
    raw = [
        torch.empty((e, i, 4, 66), dtype=torch.uint8),
        torch.empty((e, 1024, i // 256, 66), dtype=torch.uint8)
    ]
    by_shard = {}
    for expert in range(e):
        for projection, name in enumerate(("up", "down")):
            key = f"{prefix}.{expert}.{name}_proj.weight"
            by_shard.setdefault(index[key], []).append(
                (expert, projection, key))
    for shard, items in by_shard.items():
        with safe_open(path / shard, framework="pt", device="cpu") as f:
            for expert, projection, key in items:
                raw[projection][expert].copy_(f.get_tensor(key))
    packed = [
        tuple(torch.from_numpy(t) for t in pack_weights(w.numpy()))
        for w in raw
    ]
    return dict(e=e,
                i=i,
                raw=raw,
                packed=packed,
                lut=torch.from_numpy(lookup_table()),
                unit=torch.ones(e),
                dense={},
                layer=layer)


def reference(w, x, ids, routing):
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        if expert not in w["dense"]:
            w["dense"][expert] = tuple(
                dequantize_blocks(raw[expert]).bfloat16().cuda().float()
                for raw in w["raw"])
        first, second = w["dense"][expert]
        row, slot = (ids == expert).nonzero(as_tuple=True)
        projected = (x[row].float() @ first.T).bfloat16().float()
        activated = projected.relu().square().bfloat16().float()
        contribution = ((activated @ second.T) *
                        routing[row, slot, None]).bfloat16().float()
        result.index_add_(0, row, contribution)
    return result.bfloat16()


def inputs(w, m, pattern, seed=2026):
    torch.manual_seed(seed)
    x = torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.arange(m * 9, device="cuda", dtype=torch.int32).reshape(
        m, 9) % w["e"]
    if pattern == "hot":
        ids[:] = torch.arange(9, device="cuda", dtype=torch.int32)
    elif pattern == "imbalanced":
        ids[:, :7] = torch.arange(7, device="cuda", dtype=torch.int32)
        ids[:, 7:] = 7 + (ids[:, 7:] % (w["e"] - 7))
    routing = torch.softmax(torch.randn((m, 9), device="cuda"), dim=-1)
    return x, ids.contiguous(), routing


def write_fixture(path, w, tokens, tensors):
    # Versioned little-endian header followed by contiguous tensor bytes.
    with path.open("wb") as f:
        f.write(struct.pack("<4I", 0x57324131, w["e"], w["i"], tokens))
        for tensor in tensors:
            tensor.detach().cpu().contiguous().view(
                torch.uint8).numpy().tofile(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--smoke",
                        action="store_true",
                        help="Generate only decode and prefill balanced cases")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    for layer in args.layers:
        w = weights(args.checkpoint, layer)
        directory = args.output / f"layer{layer}"
        directory.mkdir(parents=True, exist_ok=True)
        q1, s1 = w["packed"][0]
        q2, s2 = w["packed"][1]
        write_fixture(directory / "weights.bin", w, 0,
                      (q1, s1, q2, s2, w["lut"], w["unit"]))
        write_fixture(directory / "raw.bin", w, 0, w["raw"])
        if args.smoke:
            cases = [(1, "balanced"), (16, "balanced")]
        else:
            cases = [(m, p) for m in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 128)
                     for p in ("balanced", "hot", "imbalanced")]
            cases += [(33, "hot"), (8, "changed"), (33, "changed")]
            tile_tokens = (w["e"] // 2, w["e"], 2 * w["e"], 4 * w["e"],
                           6 * w["e"], 12 * w["e"])
            cases += [(m, p) for m in tile_tokens
                      for p in ("balanced", "hot", "imbalanced", "changed")]
            if layer == 1:
                cases += [(m, "hot")
                          for m in (2, 3, 4, 5, 6, 7, 9, 31, 32, 204, 205, 409,
                                    410, 819, 820, 1228, 1229, 8192)]
        cases = list(dict.fromkeys(cases))
        for m, pattern in cases:
            if pattern == "changed":
                x, ids, routing = inputs(w, m, "imbalanced", seed=7)
                ids = (ids + w["e"] - 17) % w["e"]
            else:
                x, ids, routing = inputs(w, m, pattern)
            expected = reference(w, x, ids, routing)
            write_fixture(directory / f"m{m}_{pattern}.bin", w, m,
                          (x, ids, routing, expected))
            print(f"layer{layer} E{w['e']} I{w['i']} M{m} {pattern}",
                  flush=True)
        del w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
