# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# See README.md for the supported IQ2_XXS/BF16 contract.

from dataclasses import dataclass


@dataclass(frozen=True)
class Iq2Codec:
    block_weights: int = 256
    pack_factor: int = 4
    metadata_bytes: int = 2
    subscale_bytes: int = 0

    def lut_bytes(self, *, selectors: bool) -> int:
        return 2048 if selectors else 4096


_CODEC = Iq2Codec()


def block_codec(name: str) -> Iq2Codec:
    if name != "iq2_xxs":
        raise ValueError("Only IQ2_XXS weights are supported")
    return _CODEC
