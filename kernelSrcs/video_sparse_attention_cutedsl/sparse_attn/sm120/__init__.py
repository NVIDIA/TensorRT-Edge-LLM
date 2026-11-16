# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GeForce Blackwell (SM12x) sparse-attention backend."""

from .kernel import make_kernel

__all__ = ["make_kernel"]
