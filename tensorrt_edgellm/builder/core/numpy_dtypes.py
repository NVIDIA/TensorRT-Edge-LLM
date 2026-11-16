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
"""NumPy conversions for the BF16, FP8 E4M3, and FP4 E2M1 data used by
NVFP4 checkpoints and MoE weight packing."""

import numpy as np

__all__ = [
    "bf16_bytes_to_f32",
    "round_f32_to_bf16",
    "fp8_e4m3_bytes_to_f32",
    "f32_to_fp8_e4m3_bytes",
    "FP4_E2M1_LEVELS",
    "FP8_E4M3_MAX",
]

FP8_E4M3_MAX = 448.0

# Positive E2M1 (FP4) levels, low 3 bits = magnitude index.
FP4_E2M1_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=np.float32)
# Midpoints between consecutive E2M1 levels (searchsorted-based quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)

# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------


def bf16_bytes_to_f32(u16: np.ndarray) -> np.ndarray:
    """Reinterpret raw little-endian BF16 (uint16) into float32."""
    u16 = np.ascontiguousarray(u16, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def round_f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 precision using round-to-nearest-even.

    The returned array retains float32 storage.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # Round-to-nearest-even: add 0x7FFF + LSB-of-retained-mantissa.
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u + bias) & np.uint32(0xFFFF0000)
    # Preserve NaN (don't let rounding turn a NaN into inf).
    is_nan = np.isnan(x)
    out = rounded.view(np.float32).copy()
    out[is_nan] = x[is_nan]
    return out


# ---------------------------------------------------------------------------
# FP8 E4M3 (float8_e4m3fn: bias 7, no inf, NaN == 0x7F / 0xFF, max == 448)
# ---------------------------------------------------------------------------


def _build_e4m3_decode_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 0x1
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            val = float(m) * (2.0**-9)  # subnormal: 2^(-6) * m/8
        elif e == 0xF and m == 0x7:
            val = float("nan")
        else:
            val = (1.0 + m / 8.0) * (2.0**(e - 7))
        table[b] = -val if s else val
    return table


_E4M3_DECODE = _build_e4m3_decode_table()

# Ascending non-negative representable magnitudes (codes 0x00..0x7E; 0x7F = NaN).
_E4M3_POS_MAGS = _E4M3_DECODE[0:127].astype(np.float32)
# Bucket midpoints for nearest-value encoding via searchsorted.
_E4M3_MIDPOINTS = ((_E4M3_POS_MAGS[:-1] + _E4M3_POS_MAGS[1:]) / 2.0).astype(
    np.float32)


def fp8_e4m3_bytes_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode raw FP8 E4M3 (uint8) bytes to float32."""
    u8 = np.ascontiguousarray(u8, dtype=np.uint8)
    return _E4M3_DECODE[u8]


def f32_to_fp8_e4m3_bytes(x: np.ndarray) -> np.ndarray:
    """Encode float32 to FP8 E4M3 (float8_e4m3fn) raw bytes (uint8).

    Uses saturating round-to-nearest conversion. NaN and infinity inputs are
    clamped to the maximum representable magnitude.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    sign = (np.signbit(x)).astype(np.uint8)
    ax = np.abs(x)
    ax = np.where(np.isfinite(ax), ax, FP8_E4M3_MAX)
    ax = np.minimum(ax, FP8_E4M3_MAX)
    # Nearest representable magnitude code in [0, 126].
    codes = np.searchsorted(_E4M3_MIDPOINTS, ax, side="left").astype(np.uint8)
    return (codes | (sign << np.uint8(7))).astype(np.uint8)
