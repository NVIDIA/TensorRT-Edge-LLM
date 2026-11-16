# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GeForce Blackwell SM12x sparse-attention kernel recipe."""

from .mainloop import SparseAttentionForwardSm12x


def make_kernel(
    *,
    prefetch_first_exact_k: bool = True,
    prefetch_next_route_k: bool = True,
):
    return SparseAttentionForwardSm12x(
        prefetch_first_exact_k=prefetch_first_exact_k,
        prefetch_next_route_k=prefetch_next_route_k,
    )


__all__ = ["make_kernel"]
