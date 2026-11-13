# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cutlass.cute as cute


def sub_packed_f32x2(a, b):
    return cute.arch.add_packed_f32x2(a, (-b[0], -b[1]))
