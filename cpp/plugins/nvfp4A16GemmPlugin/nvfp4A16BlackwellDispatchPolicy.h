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

#include "kernels/nvfp4A16BlackwellSupport.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace trt_edgellm::plugins
{

enum class Nvfp4A16BlackwellDispatchDtype : int32_t
{
    kFp16 = 0,
    kBf16 = 1,
};

enum class Nvfp4A16BlackwellDispatchBackend : int32_t
{
    kDefault = 0,
    kGemv = 1,
    kTcgen05 = 2,
};

enum class Nvfp4A16BlackwellKernel : int32_t
{
    kUnsupported = 0,
    kGemv = 1,
    kTcgen05 = 2,
};

struct Nvfp4A16BlackwellDispatch
{
    Nvfp4A16BlackwellKernel kernel;
    int32_t splitK;
};

namespace detail
{

struct Nvfp4A16BlackwellGemvPolicy
{
    // Entries correspond to M={1,2,4,8,16}. Exact-shape tables are tuning
    // hints only; they do not define the supported problem shapes.
    std::array<int32_t, 5> splitK;
};

// Sealed on NVIDIA SM110 with three warmups, ten warm samples, and ten cold-L2
// samples. Split-K minimizes warm+cold median for the shared projections;
// lm_head intentionally stays at split-K=1 because its 1024 N tiles already
// expose enough CTAs and the measured deltas were below repeatability.
// Evidence SHA256:
//   vector GEMV tuning: f76407d6105ee2f21eabd9f4962fc3745c9802ec5dc8102d7997bca591be986e
//   plugin TCGen:       d7e94e0c867fc28776cbfa048db2018b9a627de3f47ba2240067f4f26c981625
inline constexpr Nvfp4A16BlackwellGemvPolicy kSharedUpFp16Policy{{2, 2, 2, 2, 4}};
inline constexpr Nvfp4A16BlackwellGemvPolicy kSharedUpBf16Policy{{2, 2, 2, 2, 4}};
inline constexpr Nvfp4A16BlackwellGemvPolicy kSharedDownPolicy{{4, 4, 4, 4, 4}};
inline constexpr Nvfp4A16BlackwellGemvPolicy kLmHeadPolicy{{1, 1, 1, 1, 1}};
inline constexpr Nvfp4A16BlackwellGemvPolicy kGenericPolicy{{1, 1, 1, 1, 1}};

constexpr bool isSupportedDtype(Nvfp4A16BlackwellDispatchDtype dtype) noexcept
{
    return dtype == Nvfp4A16BlackwellDispatchDtype::kFp16 || dtype == Nvfp4A16BlackwellDispatchDtype::kBf16;
}

constexpr Nvfp4A16BlackwellGemvPolicy const& getGemvPolicy(
    int32_t n, int32_t k, Nvfp4A16BlackwellDispatchDtype dtype) noexcept
{
    if (n == 3712 && k == 2688)
    {
        return dtype == Nvfp4A16BlackwellDispatchDtype::kBf16 ? kSharedUpBf16Policy : kSharedUpFp16Policy;
    }
    if (n == 2688 && k == 3712)
    {
        return kSharedDownPolicy;
    }
    if (n == 131072 && k == 2688)
    {
        return kLmHeadPolicy;
    }
    return kGenericPolicy;
}

constexpr int32_t gemvMIndex(int32_t m) noexcept
{
    switch (m)
    {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 8: return 3;
    case 16: return 4;
    default: return -1;
    }
}

} // namespace detail

//! Select the fixed SM110 dense NVFP4-A16 backend without CUDA or TensorRT state.
constexpr Nvfp4A16BlackwellDispatch getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend backend, int32_t m,
    int32_t n, int32_t k, Nvfp4A16BlackwellDispatchDtype dtype) noexcept
{
    if (!nvfp4_a16_blackwell::isTmaRepresentableProblem(m, n, k) || !detail::isSupportedDtype(dtype))
    {
        return {Nvfp4A16BlackwellKernel::kUnsupported, 1};
    }

    if (backend == Nvfp4A16BlackwellDispatchBackend::kTcgen05)
    {
        return {Nvfp4A16BlackwellKernel::kTcgen05, 1};
    }

    int32_t const gemvIndex = detail::gemvMIndex(m);
    auto const& policy = detail::getGemvPolicy(n, k, dtype);
    if (backend == Nvfp4A16BlackwellDispatchBackend::kGemv)
    {
        return gemvIndex < 0 ? Nvfp4A16BlackwellDispatch{Nvfp4A16BlackwellKernel::kUnsupported, 1}
                             : Nvfp4A16BlackwellDispatch{
                                   Nvfp4A16BlackwellKernel::kGemv, policy.splitK[static_cast<std::size_t>(gemvIndex)]};
    }
    if (backend != Nvfp4A16BlackwellDispatchBackend::kDefault)
    {
        return {Nvfp4A16BlackwellKernel::kUnsupported, 1};
    }
    if (m == 1)
    {
        return {Nvfp4A16BlackwellKernel::kGemv, policy.splitK[0]};
    }
    return {Nvfp4A16BlackwellKernel::kTcgen05, 1};
}

} // namespace trt_edgellm::plugins
