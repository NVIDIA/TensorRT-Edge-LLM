/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
#include "cuteDslW2A16GemmRunner.h"
#include "kernels/contextAttentionKernels/cuteDslTensorDescriptors.h"
#include "kernels/cuteDslModuleLoader.h"
#include "w2a16SmallMGemm.h"
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_w2a16_gemm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#include <array>
#include <limits>

namespace trt_edgellm
{
namespace
{
constexpr int32_t kMAX_M = 32768;
constexpr int32_t kSM_THOR = 110;
constexpr int32_t kSM_GEFORCE = 120;
constexpr int32_t kSM_GB10 = 121;
constexpr int32_t kNUM_SMS = 48;
constexpr int32_t kPACKING_SMALL = 8;
constexpr int32_t kPACKING_LARGE = 128;
constexpr int32_t kQUANT_BLOCK_SIZE = 256;
constexpr int32_t kMAX_GRID_Z = 65535;
constexpr int32_t kMIN_TENSOR_CORE_N = 64;
constexpr int32_t kTENSOR_CORE_M = 16;
// Prefill rows reuse each decoded weight tile across a taller M tile.
constexpr int32_t kWIDE_M = 64;
constexpr int32_t kWIDEST_M = 128;
constexpr int32_t kWIDEST_M_MIN_ROWS = 1024;
constexpr size_t kWORKSPACE_ALIGNMENT = 256;
constexpr size_t kPOINTER_ALIGNMENT = 16;
constexpr size_t kLOOKUP_BYTES = 4096;
constexpr size_t kELEMENT_BYTES = 2;
constexpr int32_t kVALUES_PER_BYTE = 4;

struct Config
{
    int32_t m, n, k, slices, packing;
};

bool validStream(cudaStream_t stream) noexcept
{
    return stream != nullptr && stream != cudaStreamLegacy && stream != cudaStreamPerThread;
}

bool validHardware(int32_t smVersion, int32_t numSms) noexcept
{
    return ((smVersion == kSM_THOR || smVersion == kSM_GEFORCE) && numSms > 0)
        || (smVersion == kSM_GB10 && numSms == kNUM_SMS);
}

//! Up to kW2A16_SMALL_M_MAX_TOKENS rows run the CUDA small-M kernel instead of an AOT variant.
bool useSmallM(int32_t m) noexcept
{
    return m <= kernel::kW2A16_SMALL_M_MAX_TOKENS;
}

bool validShape(int32_t m, int32_t n, int32_t k) noexcept
{
    // The GEMM N tile occupies grid.z, which is limited to 65535.
    return m >= 1 && m <= kMAX_M && n > 0 && n % kPACKING_SMALL == 0 && n <= kMAX_GRID_Z * kMIN_TENSOR_CORE_N && k > 0
        && k % kQUANT_BLOCK_SIZE == 0;
}

Config selectConfig(int32_t m, int32_t n, int32_t k) noexcept
{
    int32_t const packing = n % kPACKING_LARGE == 0 ? kPACKING_LARGE : kPACKING_SMALL;
    if (m <= 8)
    {
        if (packing == kPACKING_SMALL || (m <= 2 && (n >= k || k <= 1024)))
        {
            return {packing == kPACKING_SMALL ? (m < 8 ? 1 : 2) : m, 4, 256, 1, packing};
        }
        if (n >= kPACKING_LARGE * kNUM_SMS)
        {
            return {8, 64, 256, 1, packing};
        }
    }
    if (m >= kWIDEST_M_MIN_ROWS)
    {
        return {kWIDEST_M, 128, 64, 1, packing};
    }
    if (m >= kWIDE_M)
    {
        return {kWIDE_M, 128, 128, 1, packing};
    }
    Config result{kTENSOR_CORE_M, m > kTENSOR_CORE_M ? 128 : kMIN_TENSOR_CORE_N,
        m > kTENSOR_CORE_M ? kQUANT_BLOCK_SIZE : 128, 1, packing};
    int64_t const grid = ((m + kTENSOR_CORE_M - 1) / kTENSOR_CORE_M) * ((int64_t{n} + result.n - 1) / result.n);
    double best = (grid + kNUM_SMS - 1) / kNUM_SMS;
    if (grid < kNUM_SMS)
    {
        for (int32_t const split : {2, 4, 8})
        {
            if (k / split >= 512)
            {
                double const cost
                    = static_cast<double>((grid * split + kNUM_SMS - 1) / kNUM_SMS) / split + 0.005 * (split - 1);
                if (cost < best)
                {
                    best = cost;
                    result.slices = split;
                }
            }
        }
    }
    return result;
}

template <typename Module>
struct ModuleState
{
    static inline detail::LazyKernelModule<Module> state{};
};

template <typename Module, auto Load, auto Unload>
bool preload(cudaStream_t stream) noexcept
{
    return detail::ensureModuleLoaded<Load, Unload>(ModuleState<Module>::state, "w2a16_gemm", stream);
}

template <typename Module, auto Launch>
int32_t launch(CuteDslW2A16GemmParams const& p, void* scratch, cudaStream_t stream) noexcept
{
    auto& state = ModuleState<Module>::state;
    if (state.status.load(std::memory_order_acquire) != detail::LazyKernelModuleStatus::kLoaded)
    {
        return static_cast<int32_t>(cudaErrorInitializationError);
    }
    int32_t const packing = p.n % kPACKING_LARGE == 0 ? kPACKING_LARGE : kPACKING_SMALL;
    try
    {
        return Launch(&state.module, const_cast<void*>(p.input), const_cast<void*>(p.values),
            const_cast<void*>(p.bases), const_cast<void*>(p.lookup), p.output, scratch, p.m, p.n / packing,
            p.k / kQUANT_BLOCK_SIZE, stream);
    }
    catch (...)
    {
        return static_cast<int32_t>(cudaErrorUnknown);
    }
}

struct Variant
{
    Config config;
    bool (*load)(cudaStream_t) noexcept;
    int32_t (*run)(CuteDslW2A16GemmParams const&, void*, cudaStream_t) noexcept;
};

template <auto Wrapper, auto Load, auto Unload>
constexpr Variant makeVariant(Config config) noexcept
{
    using Module = cutedsl::WrapperArgT<0, decltype(Wrapper)>;
    return {config, preload<Module, Load, Unload>, launch<Module, Wrapper>};
}

Variant const* selectVariant(Config const& c) noexcept
{
    static constexpr std::array variants{
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p8_Kernel_Module_Unload>({1, 4, 256, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p8_Kernel_Module_Unload>({2, 4, 256, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p8_Kernel_Module_Unload>({16, 64, 128, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p8_Kernel_Module_Unload>({16, 64, 128, 2, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p8_Kernel_Module_Unload>({16, 64, 128, 4, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p8_Kernel_Module_Unload>({16, 64, 128, 8, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p8_Kernel_Module_Unload>({16, 128, 256, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p8_Kernel_Module_Unload>({16, 128, 256, 2, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p8_Kernel_Module_Unload>({16, 128, 256, 4, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p8_Kernel_Module_Unload>({16, 128, 256, 8, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m1_n4_k256_s1_p128_Kernel_Module_Unload>({1, 4, 256, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m2_n4_k256_s1_p128_Kernel_Module_Unload>({2, 4, 256, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s1_p128_Kernel_Module_Unload>({16, 64, 128, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s2_p128_Kernel_Module_Unload>({16, 64, 128, 2, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s4_p128_Kernel_Module_Unload>({16, 64, 128, 4, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n64_k128_s8_p128_Kernel_Module_Unload>({16, 64, 128, 8, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s1_p128_Kernel_Module_Unload>({16, 128, 256, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s2_p128_Kernel_Module_Unload>({16, 128, 256, 2, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s4_p128_Kernel_Module_Unload>({16, 128, 256, 4, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m16_n128_k256_s8_p128_Kernel_Module_Unload>({16, 128, 256, 8, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m8_n64_k256_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m8_n64_k256_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m8_n64_k256_s1_p128_Kernel_Module_Unload>({8, 64, 256, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p8_Kernel_Module_Unload>({64, 128, 128, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m64_n128_k128_s1_p128_Kernel_Module_Unload>({64, 128, 128, 1, 128}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p8_wrapper,
            w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p8_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p8_Kernel_Module_Unload>({128, 128, 64, 1, 8}),
        makeVariant<cute_dsl_w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p128_wrapper,
            w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p128_Kernel_Module_Load,
            w2a16_gemm_blackwell_geforce_m128_n128_k64_s1_p128_Kernel_Module_Unload>({128, 128, 64, 1, 128}),
    };
    for (auto const& v : variants)
    {
        auto const& a = v.config;
        if (a.m == c.m && a.n == c.n && a.k == c.k && a.slices == c.slices && a.packing == c.packing)
        {
            return &v;
        }
    }
    return nullptr;
}

bool hardwareSupported() noexcept
{
    int32_t device{}, major{}, minor{}, sms{};
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess
        && cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess
        && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess
        && validHardware(major * 10 + minor, sms);
}

struct Region
{
    uintptr_t address;
    size_t bytes;
};

bool overlaps(Region a, Region b) noexcept
{
    return a.bytes != 0 && b.bytes != 0
        && (a.address <= b.address ? b.address - a.address < a.bytes : a.address - b.address < b.bytes);
}
} // namespace

bool CuteDslW2A16GemmRunner::canImplement(int32_t m, int32_t n, int32_t k, int32_t smVersion, int32_t numSms) noexcept
{
    return validHardware(smVersion, numSms) && validShape(m, n, k);
}

size_t CuteDslW2A16GemmRunner::getWorkspaceSize(int32_t m, int32_t n, int32_t k) noexcept
{
    if (!validShape(m, n, k))
    {
        return 0;
    }
    auto const c = selectConfig(m, n, k);
    return c.slices == 1 ? 0 : static_cast<size_t>(c.slices) * m * n * sizeof(float);
}

bool CuteDslW2A16GemmRunner::ensureKernelModules(CuteDslW2A16GemmParams const& p, cudaStream_t stream) noexcept
{
    if (!validStream(stream) || !validShape(p.m, p.n, p.k) || !hardwareSupported())
    {
        return false;
    }
    if (useSmallM(p.m))
    {
        return true;
    }
    auto const* v = selectVariant(selectConfig(p.m, p.n, p.k));
    return v != nullptr && v->load(stream);
}

int32_t CuteDslW2A16GemmRunner::run(
    CuteDslW2A16GemmParams const& p, void* workspace, size_t workspaceBytes, cudaStream_t stream) const noexcept
{
    if (!validStream(stream) || !validShape(p.m, p.n, p.k))
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    size_t const required = getWorkspaceSize(p.m, p.n, p.k);
    if (workspaceBytes < required
        || (required != 0
            && (workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % kWORKSPACE_ALIGNMENT != 0)))
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    std::array const reads{Region{reinterpret_cast<uintptr_t>(p.input), kELEMENT_BYTES * p.m * p.k},
        Region{reinterpret_cast<uintptr_t>(p.values), static_cast<size_t>(p.n) * p.k / kVALUES_PER_BYTE},
        Region{reinterpret_cast<uintptr_t>(p.bases),
            static_cast<size_t>(p.n) * (p.k / kQUANT_BLOCK_SIZE) * kELEMENT_BYTES},
        Region{reinterpret_cast<uintptr_t>(p.lookup), kLOOKUP_BYTES}};
    Region const output{reinterpret_cast<uintptr_t>(p.output), kELEMENT_BYTES * p.m * p.n};
    Region const scratch{reinterpret_cast<uintptr_t>(workspace), required};
    for (auto const& r : {reads[0], reads[1], reads[2], reads[3], output, scratch})
    {
        if (r.bytes != 0
            && (r.address == 0 || r.address % kPOINTER_ALIGNMENT != 0
                || r.bytes > std::numeric_limits<uintptr_t>::max() - r.address))
        {
            return static_cast<int32_t>(cudaErrorInvalidValue);
        }
    }
    for (auto const& r : reads)
    {
        if (overlaps(output, r) || overlaps(scratch, r))
        {
            return static_cast<int32_t>(cudaErrorInvalidValue);
        }
    }
    if (overlaps(output, scratch))
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    if (!hardwareSupported())
    {
        return static_cast<int32_t>(cudaErrorNotSupported);
    }
    if (useSmallM(p.m))
    {
        return static_cast<int32_t>(
            kernel::launchW2A16SmallMGemm(p.input, p.values, p.bases, p.lookup, p.output, p.m, p.n, p.k, stream));
    }
    auto const* v = selectVariant(selectConfig(p.m, p.n, p.k));
    return v == nullptr ? static_cast<int32_t>(cudaErrorNotSupported) : v->run(p, workspace, stream);
}
} // namespace trt_edgellm
#endif
