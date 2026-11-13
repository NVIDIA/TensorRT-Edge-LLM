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
"""``tensorrt_edgellm/checkpoint/iq2_xxs.py`` against the kernel packers.

The builder packs IQ2_XXS blocks with the NumPy module when the weights stay
in the engine. Its output must be byte-identical to the packers that define
the W2A16 MoE and GEMM kernel contracts under ``kernelSrcs``.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_prepare(kernel: str):
    """Load a kernel packer by path; its package imports the CuTe DSL SDK."""
    path = os.path.join(_REPO_ROOT, "kernelSrcs", kernel, "prepare.py")
    spec = importlib.util.spec_from_file_location(f"{kernel}_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    import torch

    from tensorrt_edgellm.checkpoint import iq2_xxs
    gemm_prepare = _load_prepare("w2a16_gemm_cutedsl")
    moe_prepare = _load_prepare("w2a16_moe_cutedsl")
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"IQ2_XXS packers not importable: {exc}",
                allow_module_level=True)


def _blocks(shape, seed):
    """Random IQ2_XXS blocks with finite FP16 bases."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(*shape, 66), dtype=np.uint8)
    bases = rng.uniform(-1, 1, size=shape).astype(np.float16)
    blocks[..., :2] = bases[..., None].view(np.uint8).reshape(*shape, 2)
    return blocks


@pytest.mark.parametrize("experts,rows,k_blocks", [(3, 128, 4), (2, 384, 8),
                                                   (1, 1024, 8)])
def test_moe_packing_matches_kernel_packer(experts, rows, k_blocks):
    blocks = _blocks((experts, rows, k_blocks), seed=rows + k_blocks)
    words, bases = iq2_xxs.pack_weights(blocks)
    ref_words, ref_bases = moe_prepare.pack_weights(torch.from_numpy(blocks))
    np.testing.assert_array_equal(words, ref_words.numpy())
    np.testing.assert_array_equal(bases, ref_bases.numpy())
    np.testing.assert_array_equal(
        iq2_xxs.lookup_table().view(np.int16),
        moe_prepare.make_lookup_table("cpu").reshape(-1).numpy())


@pytest.mark.parametrize("rows,k_blocks", [(256, 4), (5376, 16), (4096, 21),
                                           (24, 3)])
def test_dense_packing_matches_kernel_packer(rows, k_blocks):
    blocks = _blocks((rows, k_blocks), seed=rows * 7 + k_blocks)
    values, bases = iq2_xxs.pack_dense_weights(blocks)
    ref_values, ref_bases = gemm_prepare.pack_weights(torch.from_numpy(blocks))
    np.testing.assert_array_equal(values, ref_values.numpy())
    np.testing.assert_array_equal(
        bases.view(np.uint8).reshape(-1),
        ref_bases.numpy().reshape(-1))
    np.testing.assert_array_equal(
        iq2_xxs.dense_lookup_table(),
        gemm_prepare.make_lookup_table("cpu").view(torch.int16).numpy().view(
            np.uint16))


def test_non_finite_bases_are_rejected():
    blocks = _blocks((128, 4), seed=0)
    blocks[0, 0, :2] = np.array([np.inf], dtype=np.float16).view(np.uint8)
    with pytest.raises(ValueError, match="finite"):
        iq2_xxs.pack_dense_weights(blocks)
    with pytest.raises(ValueError, match="finite"):
        iq2_xxs.pack_weights(blocks[None])
