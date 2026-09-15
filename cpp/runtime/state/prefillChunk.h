/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
//! Fixed prompt-only partition. Input is the number of unconsumed, already-tokenized prompt tokens.
//! Keep intermediate chunks on 64-token boundaries and keep resumed final chunks between 64 and 128 tokens.
inline int32_t nextPrefillChunkSize(int32_t remaining)
{
    if (remaining <= 0)
    {
        throw std::logic_error("No remaining prompt tokens");
    }
    constexpr int32_t kCHUNK_CAP = 128;
    constexpr int32_t kALIGNMENT = 64;
    if (remaining <= kCHUNK_CAP)
    {
        return remaining;
    }
    return remaining < kCHUNK_CAP + kALIGNMENT ? kALIGNMENT : kCHUNK_CAP;
}
} // namespace rt
} // namespace trt_edgellm
