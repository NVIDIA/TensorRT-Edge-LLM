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

# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
#
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# See README.md for the supported IQ2_XXS/BF16 contract.

# Magnitude table: GGML 64302f42eb959556633c89ae651915d8157dcd63 (MIT).
"""Lossless IQ2_XXS repacking for the W2A16 MoE and GEMM plugins."""

import numpy as np

_IQ2_XXS_GRID_2BIT = bytes.fromhex(
    "00000200050008000a00110014002000220028002a0041004400500058006100"
    "6400800082008a00a20001010401100115014001840198010002020222028202"
    "010404041004210424044004420448046004810484049004a404000502050805"
    "200546056905800591050906100640068406a406000805080808140828084108"
    "440850085208880804094009020a140a01100410101021104010601084109010"
    "951000110811201150115a118011241245120014081420142514491480141815"
    "6215001616160118041810184018811800190519a019511a002002200a204420"
    "6120802082202921482100220222012404241024402456240025412564259026"
    "082820289428442a014004401040184021402440404048405640604081408440"
    "9040004120416141804185410142104248425642684200440844204480449944"
    "124524450046014804481048404845480049584961498249454a904a00500850"
    "1150195020508050885004514251a4519152905492540a550156545600581158"
    "195864584059085a046010604060686000615561186260620064056410651265"
    "84654268008002800a8041808280048118814081118201840484108415844084"
    "608400854685948509864086608602880489118a0490109024904090a1901691"
    "8091459200942294449451958198209902a050a085a009a100a218a450a804a9")


def lookup_table() -> np.ndarray:
    """Return the 2048-byte W2A16 MoE pair-selector table as int8."""
    packed = np.frombuffer(_IQ2_XXS_GRID_2BIT, dtype=np.uint8)
    codes = np.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)],
                     axis=1).reshape(256, 8).astype(np.int16)
    lo, hi = 2 * codes[:, ::2], 2 * codes[:, 1::2]
    return np.ascontiguousarray((lo | ((lo + 1) << 4) | (hi << 8) |
                                 ((hi + 1) << 12)).reshape(-1)).view(np.int8)


def pack_weights(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retile uint8[E,N,K/256,66] expert blocks for W2A16MoePlugin.

    Returns the int32 payload [E,N/64,K/128,8,4,8,2] and the FP16 bases
    [E,K/256,N/16,8,2] as int32 words, both flattened. Bytes are only
    reordered; nothing is dequantized.
    """
    if blocks.dtype != np.uint8 or blocks.ndim != 4 or blocks.shape[-1] != 66:
        raise ValueError("expected IQ2_XXS uint8 [E,N,K/256,66]")
    experts, rows, kb, _ = blocks.shape
    if min(experts, rows, kb) <= 0 or rows % 128:
        raise ValueError("IQ2_XXS rows must be a positive multiple of 128")
    words = np.empty((experts, rows // 64, kb * 2, 8, 4, 8, 2), dtype=np.int32)
    bases = np.empty((experts, kb, rows // 16, 8, 2), dtype=np.float16)
    for expert in range(experts):
        for row in range(0, rows, 256):
            end = min(row + 256, rows)
            chunk = blocks[expert, row:end]
            scales = np.ascontiguousarray(chunk[..., :2]).view(
                np.float16).reshape(end - row, kb)
            if not np.isfinite(scales).all():
                raise ValueError("IQ2_XXS bases must be finite")
            q = np.ascontiguousarray(chunk[..., 2:]).view(np.int32)
            descriptors = np.ascontiguousarray(
                q.reshape(end - row,
                          kb * 16).T).reshape(kb * 16, (end - row) // 16, 2,
                                              8).transpose(0, 1, 3, 2)
            words[expert, row // 64:end // 64] = descriptors.reshape(
                kb * 2, 8, (end - row) // 64, 4, 8,
                2).transpose(2, 0, 1, 3, 4, 5)
            bases[expert, :, row // 16:end // 16] = scales.T.reshape(
                kb, (end - row) // 16, 2, 8).transpose(0, 1, 3, 2)
    return (np.ascontiguousarray(words.reshape(-1)),
            np.ascontiguousarray(bases.reshape(-1)).view(np.int32))


def dense_lookup_table():
    """Return the W2A16 GEMM BF16 magnitude grid [256,8] as raw uint16 bits."""
    packed = np.frombuffer(_IQ2_XXS_GRID_2BIT, dtype=np.uint8)
    codes = np.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)],
                     axis=1).reshape(256, 8)
    # 8, 25 and 43 are exact in BF16, so truncating the FP32 bits is lossless.
    magnitudes = np.array((8, 25, 43, 0), dtype=np.float32)[codes]
    return np.ascontiguousarray(magnitudes.view(np.uint32) >> 16).astype(
        np.uint16)


def pack_dense_weights(blocks):
    """Retile dense [N,K/256,66] bytes for W2A16GemmPlugin without dequantizing.

    Returns the payload [N/P,K/256,P,64] (P=128 when N%128=0, else 8) and FP16
    bases [K/256,N], matching kernelSrcs/w2a16_gemm_cutedsl/prepare.py.
    """
    if blocks.dtype != np.uint8 or blocks.ndim != 3 or blocks.shape[-1] != 66:
        raise ValueError("expected IQ2_XXS uint8 [N,K/256,66]")
    rows, kb, _ = blocks.shape
    if min(rows, kb) <= 0 or rows % 8:
        raise ValueError("IQ2_XXS rows must be a positive multiple of 8")
    bases = np.ascontiguousarray(blocks[..., :2]).view(np.float16).reshape(
        rows, kb)
    if not np.isfinite(bases).all():
        raise ValueError("IQ2_XXS bases must be finite")
    packing = 128 if rows % 128 == 0 else 8
    values = blocks[..., 2:].reshape(rows // packing, packing, kb,
                                     64).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(values), np.ascontiguousarray(bases.T)
