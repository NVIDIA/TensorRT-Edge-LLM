# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# See README.md for the supported IQ2_XXS/BF16 contract.

# Magnitude table: GGML 64302f42eb959556633c89ae651915d8157dcd63 (MIT).
"""Lossless IQ2_XXS packing for the W2A16 runner (outside inference/capture)."""
import torch

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


def make_lookup_table(device):
    """Return the 2048-byte GGML IQ2_XXS pair-selector table on device."""
    packed = torch.tensor(list(_IQ2_XXS_GRID_2BIT), dtype=torch.uint8)
    codes = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)),
                        1).reshape(256, 8).int()
    lo, hi = 2 * codes[:, 0::2], 2 * codes[:, 1::2]
    return (lo | ((lo + 1) << 4) | (hi << 8) | ((hi + 1) << 12)).to(
        torch.int16).to(device).contiguous()


def pack_weights(blocks):
    """Reorder uint8[E,N,K/256,66] blocks into tiled int32 payload and FP16 bases.

    No quantization, magnitude lookup, or floating-point arithmetic is applied.
    Payload: int32[E,N/64,K/128,8,4,8,2].
    Bases: FP16[E,K/256,N/16,8,2] (passed as int32 words to the kernel).
    The two planes have exactly the same total byte count as the raw blocks.
    """
    if blocks.dtype != torch.uint8 or blocks.ndim != 4 or blocks.shape[
            -1] != 66:
        raise ValueError("Expected uint8 IQ2_XXS blocks [E,N,K/256,66]")
    e, n, kb, _ = blocks.shape
    if min(e, n, kb) <= 0 or n % 128:
        raise ValueError("Expected positive dimensions and N divisible by 128")
    words = torch.empty((e, n // 64, kb * 2, 8, 4, 8, 2),
                        dtype=torch.int32,
                        device=blocks.device)
    bases = torch.empty((e, kb, n // 16, 8, 2),
                        dtype=torch.float16,
                        device=blocks.device)
    for expert in range(e):
        for row in range(0, n, 256):
            end = min(row + 256, n)
            chunk = blocks[expert, row:end]
            d = chunk[..., :2].contiguous().view(torch.float16).reshape(
                end - row, kb)
            if not bool(torch.isfinite(d).all()):
                raise ValueError("IQ2_XXS bases must be finite")
            q = chunk[..., 2:].contiguous().view(torch.int32)
            descriptors = q.reshape(end - row, kb * 16).T.contiguous().reshape(
                kb * 16, (end - row) // 16, 2, 8).transpose(2, 3)
            words[expert, row // 64:end // 64].copy_(
                descriptors.reshape(kb * 2, 8, (end - row) // 64, 4, 8,
                                    2).permute(2, 0, 1, 3, 4, 5))
            bases[expert, :, row // 16:end // 16].copy_(
                d.T.reshape(kb, (end - row) // 16, 2, 8).transpose(-2, -1))
    return words.reshape(-1), bases.reshape(-1).view(torch.int32)
