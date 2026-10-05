/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/cudaMacros.h"

#include <cstdint>

namespace trt_edgellm
{

//! Resolve the effective SM12x GDN PDL mode. The same value selects the dual-role
//! fused Q/K normalization and fused GDN prefill launch modes.
constexpr bool useGdnPdl(bool requested, int32_t seqLen, int32_t smVersion,
    bool toolchainSupportsPdl = SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH != 0) noexcept
{
    return requested && toolchainSupportsPdl && seqLen > 1 && (smVersion == 120 || smVersion == 121);
}

} // namespace trt_edgellm
