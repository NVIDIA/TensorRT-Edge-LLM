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
"""CPU pin of the ``BLACKWELL_MOE_N128_K64_V1`` routed-MoE W4A16 weight layout.

The layout is the single weight buffer shared by the tcgen05 grouped prefill
GEMM and the decode kernels of ``Nvfp4A16BlackwellMoePlugin``.
These tests pin the executable specification
(:func:`nvfp4_a16_blackwell_moe_offsets`) against the repacker, check that the
repack is a pure byte permutation with an fp32 verbatim global scale, and pin
the Nemotron 3.5 Lightning shapes.  No GPU is required.
"""

import json
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

torch = pytest.importorskip("torch")

from tensorrt_edgellm.checkpoint.repacking import (  # noqa: E402
    NVFP4_A16_BLACKWELL_MOE_TILE_K, NVFP4_A16_BLACKWELL_MOE_TILE_N,
    decode_modelopt_nvfp4, nvfp4_a16_blackwell_moe_offsets,
    repack_nvfp4_a16_blackwell_linear, repack_nvfp4_a16_blackwell_moe_experts,
    swizzle_nvfp4_a16_blackwell_moe_row_tiles, unpack_nvfp4_codes)

_TILE_N = NVFP4_A16_BLACKWELL_MOE_TILE_N
_TILE_K = NVFP4_A16_BLACKWELL_MOE_TILE_K
_GROUP = 16

# Nemotron 3.5 Lightning routed experts: FC1 (up_proj) [I=1856, H=2688],
# FC2 (down_proj) [H=2688, I=1856].
_NEMOTRON_H = 2688
_NEMOTRON_I = 1856
_SHAPES = [(128, 64), (256, 128), (_NEMOTRON_I, _NEMOTRON_H),
           (_NEMOTRON_H, _NEMOTRON_I)]


def _random_expert(n: int, k: int, seed: int):
    """Synthetic ModelOpt-style expert: packed codes, E4M3 scales, fp32 ws2."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), generator=g,
                           dtype=torch.int64).to(torch.uint8)
    # 0x00..0x77 are positive finite E4M3 values (no NaN codes).
    scales = torch.randint(0,
                           0x78, (n, k // _GROUP),
                           generator=g,
                           dtype=torch.int64).to(torch.int8)
    ws2 = torch.tensor([0.0172380004], dtype=torch.float32) * (1.0 + seed)
    return packed, scales, ws2


def _sample_coords(n: int, k: int):
    if n * k <= 128 * 128:
        return [(i, j) for i in range(n) for j in range(k)]
    coords = [(i, j) for i in range(0, n, 37) for j in range(0, k, 29)]
    coords += [(n - 1, k - 1), (n - 1, 0), (0, k - 1), (n // 2, k // 2)]
    return coords


def _unswizzle_rows(qweights: torch.Tensor) -> torch.Tensor:
    """Undo the TMA SWIZZLE_32B image (rows 4-7 of every 8 swap 16-byte halves)."""
    t = qweights.reshape(*qweights.shape[:-2], 16, 8, 2, 16)
    out = t.clone()
    out[..., 4:8, 0, :] = t[..., 4:8, 1, :]
    out[..., 4:8, 1, :] = t[..., 4:8, 0, :]
    return out.reshape(qweights.shape)


def _unrepack(qweights: torch.Tensor, block_scales: torch.Tensor):
    """Inverse tile permutation: layout -> ``[N_pad, K/2]`` codes, ``[N_pad, K/16]`` scales."""
    n_tiles, k_tiles = qweights.shape[0], qweights.shape[1]
    packed = (_unswizzle_rows(qweights.view(torch.uint8)).permute(
        0, 2, 1, 3).reshape(n_tiles * _TILE_N,
                            k_tiles * (_TILE_K // 2)).contiguous())
    scales = (block_scales.permute(0, 2, 1, 3).reshape(
        n_tiles * _TILE_N, k_tiles * (_TILE_K // _GROUP)).contiguous())
    return packed, scales


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_offsets_match_repack(shape):
    """The closed-form spec addresses exactly the bytes the repacker wrote."""
    n, k = shape
    packed, scales, ws2 = _random_expert(n, k, seed=1)
    q, s, g, n_logical, n_padded = repack_nvfp4_a16_blackwell_linear(
        packed, scales, ws2, pad_n_to=_TILE_N)
    q = swizzle_nvfp4_a16_blackwell_moe_row_tiles(
        q)  # MoE layout = dense tiles + SW32 image
    assert n_logical == n
    assert n_padded == ((n + _TILE_N - 1) // _TILE_N) * _TILE_N
    assert tuple(q.shape) == (n_padded // _TILE_N, k // _TILE_K, _TILE_N,
                              _TILE_K // 2)
    assert tuple(s.shape) == (n_padded // _TILE_N, k // _TILE_K, _TILE_N,
                              _TILE_K // _GROUP)
    q_flat = q.view(torch.uint8).reshape(-1).numpy()
    s_flat = s.reshape(-1).numpy()
    codes = unpack_nvfp4_codes(packed).numpy()
    scales_np = scales.numpy()
    num_k_tiles = k // _TILE_K
    for (i, j) in _sample_coords(n, k):
        qb, hi, sb = nvfp4_a16_blackwell_moe_offsets(i, j, num_k_tiles)
        byte = int(q_flat[qb])
        nibble = (byte >> 4) & 0xF if hi else byte & 0xF
        assert nibble == int(codes[i, j]), (i, j)
        assert int(s_flat[sb]) == int(scales_np[i, j // _GROUP]), (i, j)
    assert g.dtype == torch.float32 and tuple(g.shape) == (1, )
    assert g.item() == ws2.item()


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_roundtrip_dequant_is_bit_exact(shape):
    """repack -> un-repack reproduces the checkpoint bytes and its dequant."""
    n, k = shape
    packed, scales, ws2 = _random_expert(n, k, seed=2)
    q, s, g, _, n_padded = repack_nvfp4_a16_blackwell_linear(packed,
                                                             scales,
                                                             ws2,
                                                             pad_n_to=_TILE_N)
    q = swizzle_nvfp4_a16_blackwell_moe_row_tiles(q)
    # The swizzle is an involution and a pure permutation of each 4 KB tile.
    assert torch.equal(
        swizzle_nvfp4_a16_blackwell_moe_row_tiles(
            swizzle_nvfp4_a16_blackwell_moe_row_tiles(q)), q)
    packed_rt, scales_rt = _unrepack(q, s)
    assert torch.equal(packed_rt[:n], packed)
    assert torch.equal(scales_rt[:n], scales)
    assert int(packed_rt[n:].to(torch.int64).sum()) == 0
    assert int(scales_rt[n:].to(torch.int64).abs().sum()) == 0
    ref = decode_modelopt_nvfp4(packed, scales, ws2)
    got = decode_modelopt_nvfp4(packed_rt, scales_rt, g)
    assert got.shape == (n_padded, k)
    assert np.array_equal(got[:n], ref)
    assert not np.any(got[n:])


def test_repack_is_pure_byte_permutation():
    n, k = _NEMOTRON_I, _NEMOTRON_H
    packed, scales, ws2 = _random_expert(n, k, seed=3)
    q, s, g, _, n_padded = repack_nvfp4_a16_blackwell_linear(packed,
                                                             scales,
                                                             ws2,
                                                             pad_n_to=_TILE_N)
    pad = n_padded - n
    src_codes = torch.cat(
        [packed.reshape(-1),
         torch.zeros(pad * k // 2, dtype=torch.uint8)])
    src_scales = torch.cat(
        [scales.reshape(-1),
         torch.zeros(pad * k // _GROUP, dtype=torch.int8)])
    assert torch.equal(
        torch.sort(q.view(torch.uint8).reshape(-1)).values,
        torch.sort(src_codes).values)
    assert torch.equal(
        torch.sort(s.reshape(-1)).values,
        torch.sort(src_scales).values)
    # Verbatim fp32 multiplier: no 2**7 skip-flop factor, no fp16 narrowing.
    assert g.dtype == torch.float32
    assert g.item() == ws2.item()


def test_experts_stack_nemotron_shapes():
    num_experts = 3
    fc1 = [
        _random_expert(_NEMOTRON_I, _NEMOTRON_H, 10 + e)
        for e in range(num_experts)
    ]
    fc2 = [
        _random_expert(_NEMOTRON_H, _NEMOTRON_I, 20 + e)
        for e in range(num_experts)
    ]
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = (
        repack_nvfp4_a16_blackwell_moe_experts(
            [t[0] for t in fc1], [t[1] for t in fc1], [t[2] for t in fc1],
            [t[0] for t in fc2], [t[1] for t in fc2], [t[2] for t in fc2]))
    # FC1: N=1856 -> 1920 (15 tiles), K=2688 (42 tiles, unpadded).
    assert tuple(fc1_q.shape) == (num_experts, 15, 42, 128, 32)
    assert tuple(fc1_s.shape) == (num_experts, 15, 42, 128, 4)
    # FC2: N=2688 (21 tiles), K=1856 (29 tiles, unpadded).
    assert tuple(fc2_q.shape) == (num_experts, 21, 29, 128, 32)
    assert tuple(fc2_s.shape) == (num_experts, 21, 29, 128, 4)
    assert fc1_q.dtype == torch.int8 and fc2_q.dtype == torch.int8
    assert fc1_s.dtype == torch.int8 and fc2_s.dtype == torch.int8
    assert tuple(
        fc1_g.shape) == (num_experts, ) and fc1_g.dtype == torch.float32
    assert tuple(
        fc2_g.shape) == (num_experts, ) and fc2_g.dtype == torch.float32
    for t in (fc1_q, fc1_s, fc2_q, fc2_s, fc1_g, fc2_g):
        assert t.is_contiguous()
    # Each expert plane equals the standalone dense repack of that expert plus
    # the SW32 row-byte image (the MoE layout's only difference).
    for e in range(num_experts):
        q, s, g, _, _ = repack_nvfp4_a16_blackwell_linear(*fc1[e],
                                                          pad_n_to=_TILE_N)
        assert torch.equal(fc1_q[e],
                           swizzle_nvfp4_a16_blackwell_moe_row_tiles(q))
        assert torch.equal(fc1_s[e], s)
        assert fc1_g[e].item() == g.item() == fc1[e][2].item()
        q, s, g, _, _ = repack_nvfp4_a16_blackwell_linear(*fc2[e],
                                                          pad_n_to=_TILE_N)
        assert torch.equal(fc2_q[e],
                           swizzle_nvfp4_a16_blackwell_moe_row_tiles(q))
        assert torch.equal(fc2_s[e], s)
        assert fc2_g[e].item() == g.item() == fc2[e][2].item()
    # Expert stride in bytes is one contiguous plane: N_pad*K/2 and N_pad*K/16.
    assert fc1_q[1].data_ptr() - fc1_q[0].data_ptr() == 1920 * 2688 // 2
    assert fc1_s[1].data_ptr() - fc1_s[0].data_ptr() == 1920 * 2688 // 16
    assert fc2_q[1].data_ptr() - fc2_q[0].data_ptr() == 2688 * 1856 // 2


def test_fc1_pad_rows_are_zero_codes_and_zero_scales():
    packed, scales, ws2 = _random_expert(_NEMOTRON_I, _NEMOTRON_H, seed=4)
    q, s, _, _, n_padded = repack_nvfp4_a16_blackwell_linear(packed,
                                                             scales,
                                                             ws2,
                                                             pad_n_to=_TILE_N)
    assert n_padded == 1920
    q_flat = q.view(torch.uint8).reshape(-1).numpy()
    s_flat = s.reshape(-1).numpy()
    num_k_tiles = _NEMOTRON_H // _TILE_K
    for i in range(_NEMOTRON_I, n_padded):
        for j in range(0, _NEMOTRON_H, 61):
            qb, _, sb = nvfp4_a16_blackwell_moe_offsets(i, j, num_k_tiles)
            assert int(q_flat[qb]) == 0
            assert int(s_flat[sb]) == 0


def test_rejects_invalid_shapes_and_ragged_lists():
    good1 = _random_expert(256, 128, seed=5)
    good2 = _random_expert(128, 256, seed=6)
    # K must be a multiple of 64.
    bad_k = _random_expert(128, 96, seed=7)
    with pytest.raises(ValueError):
        repack_nvfp4_a16_blackwell_linear(*bad_k, pad_n_to=_TILE_N)
    # Ragged per-expert lists.
    with pytest.raises(ValueError):
        repack_nvfp4_a16_blackwell_moe_experts([good1[0]], [good1[1]],
                                               [good1[2], good1[2]],
                                               [good2[0]], [good2[1]],
                                               [good2[2]])
    # FC2 K must equal the logical FC1 N (the layout never pads K).
    fc2_wrong_k = _random_expert(128, 192, seed=8)
    with pytest.raises(ValueError):
        repack_nvfp4_a16_blackwell_moe_experts([good1[0]], [good1[1]],
                                               [good1[2]], [fc2_wrong_k[0]],
                                               [fc2_wrong_k[1]],
                                               [fc2_wrong_k[2]])
    # FC2 N (hidden size) must already be a multiple of 128.
    fc1_h200 = _random_expert(256, 192, seed=9)
    fc2_n200 = _random_expert(200, 256, seed=10)
    with pytest.raises(ValueError):
        repack_nvfp4_a16_blackwell_moe_experts([fc1_h200[0]], [fc1_h200[1]],
                                               [fc1_h200[2]], [fc2_n200[0]],
                                               [fc2_n200[1]], [fc2_n200[2]])
    # Expert planes must agree in shape.
    other = _random_expert(384, 128, seed=11)
    with pytest.raises(ValueError):
        repack_nvfp4_a16_blackwell_moe_experts(
            [good1[0], other[0]], [good1[1], other[1]], [good1[2], other[2]],
            [good2[0], good2[0]], [good2[1], good2[1]], [good2[2], good2[2]])


_DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-"
    "30B-A3B-NVFP4/snapshots/6dbbd757ea75a8ece6e0702872e3ae53f9987728")


def test_real_checkpoint_layer1_expert0_spot_check():
    """Optional: pin the layout against a real ModelOpt expert when available."""
    snapshot = os.environ.get("NEMOTRON35_NVFP4_SNAPSHOT", _DEFAULT_SNAPSHOT)
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        pytest.skip("Nemotron 3.5 Lightning NVFP4 snapshot not available")
    pytest.importorskip("safetensors")
    from safetensors import safe_open  # noqa: F401
    weight_map = json.load(open(index_path))["weight_map"]
    prefix = "backbone.layers.1.mixer.experts.0."
    tensors = {}
    for proj in ("up_proj", "down_proj"):
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            key = f"{prefix}{proj}.{suffix}"
            shard = os.path.join(snapshot, weight_map[key])
            with safe_open(shard, framework="pt", device="cpu") as f:
                tensors[f"{proj}.{suffix}"] = f.get_tensor(key)
    up = (tensors["up_proj.weight"], tensors["up_proj.weight_scale"],
          tensors["up_proj.weight_scale_2"])
    down = (tensors["down_proj.weight"], tensors["down_proj.weight_scale"],
            tensors["down_proj.weight_scale_2"])
    assert tuple(up[0].shape) == (_NEMOTRON_I, _NEMOTRON_H // 2)
    assert tuple(down[0].shape) == (_NEMOTRON_H, _NEMOTRON_I // 2)
    fc1_q, fc1_s, fc1_g, fc2_q, fc2_s, fc2_g = (
        repack_nvfp4_a16_blackwell_moe_experts([up[0]], [up[1]], [up[2]],
                                               [down[0]], [down[1]],
                                               [down[2]]))
    assert tuple(fc1_q.shape) == (1, 15, 42, 128, 32)
    assert tuple(fc2_q.shape) == (1, 21, 29, 128, 32)
    assert fc1_g.item() == up[2].float().item()
    assert fc2_g.item() == down[2].float().item()
    for (q, s, g, (packed, scales, ws2)) in ((fc1_q[0], fc1_s[0], fc1_g, up),
                                             (fc2_q[0], fc2_s[0], fc2_g,
                                              down)):
        packed_rt, scales_rt = _unrepack(q, s)
        n = packed.shape[0]
        ref = decode_modelopt_nvfp4(packed, scales, ws2)
        got = decode_modelopt_nvfp4(packed_rt[:n],
                                    scales_rt[:n].view(torch.int8), g)
        assert np.array_equal(got, ref)
