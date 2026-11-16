# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# GGML magnitude table is MIT licensed; see README.md.
"""Prepare owned dense IQ2_XXS storage before inference or graph capture."""
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
    """Return the 4 KiB BF16 magnitude table on the requested device."""
    packed = torch.tensor(list(_IQ2_XXS_GRID_2BIT), dtype=torch.uint8)
    codes = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)),
                        1)
    magnitudes = torch.tensor((8, 25, 43, 0), dtype=torch.bfloat16)
    return magnitudes[codes.long()].reshape(256, 8).to(device).contiguous()


def pack_weights(blocks):
    """Retile uint8 [N,K/256,66] blocks without modifying or expanding them.

    Returns owned payload bytes [N/B,K/256,B,64] and base bytes [K/256,N,2].
    B is 128 when N is divisible by 128, otherwise 8. CPU preparation is
    supported for fixtures; CUDA preparation must complete before capture.
    """
    if (blocks.dtype != torch.uint8 or blocks.ndim != 3
            or blocks.shape[-1] != 66 or min(blocks.shape) <= 0
            or blocks.shape[0] % 8):
        raise ValueError(
            'Expected uint8 [N,K/256,66], positive dimensions and N divisible by 8'
        )
    if not blocks.is_contiguous():
        raise ValueError('IQ2_XXS blocks must be contiguous')
    if blocks.is_cuda:
        with torch.cuda.device(blocks.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    'IQ2_XXS packing must finish before capture')
    n, kb, _ = blocks.shape
    bases = blocks[..., :2].contiguous()
    if not torch.isfinite(bases.view(torch.float16)).all().item():
        raise ValueError('IQ2_XXS bases must be finite')
    packing = 128 if n % 128 == 0 else 8
    values = blocks[..., 2:].reshape(n // packing, packing, kb,
                                     64).permute(0, 2, 1, 3).contiguous()
    return values, bases.permute(1, 0, 2).contiguous()
