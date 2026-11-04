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
"""Tests for ``tensorrt_edgellm/checkpoint/repacking.py`` GPTQ repacking —
specifically the symmetric path where ``qzeros`` is omitted.

Some symmetric GPTQ checkpoints (e.g. Qwen3.5 int4) store ``qzeros`` as an
empty ``[num_groups, 0]`` tensor. ``repack_gptq_to_plugin`` must treat these
as symmetric quantization with the implicit 4-bit midpoint zero (8), which
makes the zero-point offset adjustment a no-op.
"""

import os
import sys

import pytest

# Load the package from the repository root without installing it.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch

    from tensorrt_edgellm.checkpoint.repacking import repack_gptq_to_plugin
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tensorrt_edgellm/torch not importable: {exc}",
                allow_module_level=True)


def _random_qweight(in_features: int,
                    out_features: int,
                    seed: int = 0) -> torch.Tensor:
    """Build a GPTQ-packed ``qweight`` ``[in//8, out]`` int32 from random nibbles."""
    g = torch.Generator().manual_seed(seed)
    nibbles = torch.randint(0,
                            16, (in_features, out_features),
                            generator=g,
                            dtype=torch.int32)
    qweight = torch.zeros(in_features // 8, out_features, dtype=torch.int32)
    for k in range(8):
        qweight |= (nibbles[k::8, :] & 0xF) << (4 * k)
    return qweight


def _packed_midpoint_qzeros(num_groups: int, out_features: int,
                            zero_point_offset: int) -> torch.Tensor:
    """Explicit AWQ-packed ``qzeros`` encoding ``stored_zero = 8 - offset`` everywhere."""
    v = (8 - zero_point_offset) & 0xF
    packed_val = 0
    for k in range(8):
        packed_val |= v << (4 * k)
    return torch.full((num_groups, out_features // 8),
                      packed_val,
                      dtype=torch.int32)


def test_symmetric_empty_qzeros_matches_explicit_midpoint():
    """Symmetric (empty ``[num_groups, 0]`` qzeros) must produce the same result
    as an explicit asymmetric call whose stored zeros are the midpoint (8)."""
    # _pack_intweights requires in_features % 64 == 0 and out_features % 4 == 0.
    in_features, out_features, num_groups = 128, 16, 4
    qweight = _random_qweight(in_features, out_features, seed=1)

    sym_qzeros = torch.empty(num_groups, 0, dtype=torch.int32)
    qw_sym, perm_sym = repack_gptq_to_plugin(qweight, sym_qzeros)

    explicit_qzeros = _packed_midpoint_qzeros(num_groups,
                                              out_features,
                                              zero_point_offset=1)
    qw_ref, perm_ref = repack_gptq_to_plugin(qweight, explicit_qzeros)

    assert torch.equal(qw_sym, qw_ref)
    assert torch.equal(perm_sym, perm_ref)


def test_symmetric_output_shape_and_dtype():
    """Empty qzeros must not crash and must yield correctly shaped int8 output."""
    in_features, out_features, num_groups = 128, 16, 4
    qweight = _random_qweight(in_features, out_features, seed=2)
    sym_qzeros = torch.empty(num_groups, 0, dtype=torch.int32)

    qw_out, perm = repack_gptq_to_plugin(qweight, sym_qzeros)

    # _pack_intweights packs 4 N-rows per int16 -> int8 view doubles the rows.
    assert qw_out.shape == (out_features // 2, in_features)
    assert qw_out.dtype == torch.int8
    assert perm.shape == (in_features, )
    # Sequential g_idx -> identity activation permutation.
    assert torch.equal(perm, torch.arange(in_features, dtype=torch.int64))


def test_symmetric_num_groups_falls_back_to_gidx():
    """Fully empty qzeros (shape[0]==0) must infer num_groups from ``g_idx``."""
    in_features, out_features, num_groups = 128, 16, 4
    group_size = in_features // num_groups
    qweight = _random_qweight(in_features, out_features, seed=3)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    empty_qzeros = torch.empty(0, dtype=torch.int32)
    qw_gidx, _ = repack_gptq_to_plugin(qweight, empty_qzeros, g_idx=g_idx)

    # Equivalent explicit-midpoint asymmetric call with matching num_groups.
    explicit_qzeros = _packed_midpoint_qzeros(num_groups,
                                              out_features,
                                              zero_point_offset=1)
    qw_ref, _ = repack_gptq_to_plugin(qweight, explicit_qzeros)

    assert qw_gidx.shape == (out_features // 2, in_features)
    assert torch.equal(qw_gidx, qw_ref)
