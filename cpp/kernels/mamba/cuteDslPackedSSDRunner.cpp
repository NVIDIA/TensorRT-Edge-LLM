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

#if defined(CUTE_DSL_SSD_ENABLED)

#include "cuteDslSSDRunner.h"

#include <array>
#include <limits>
#include <type_traits>

namespace trt_edgellm
{
namespace
{
constexpr size_t kALIGNMENT{256};

size_t packedScratchBytes(int32_t totalTokens, int32_t nheads)
{
    size_t const elements = static_cast<size_t>(totalTokens) * nheads;
    if (elements > (std::numeric_limits<size_t>::max() / 2 - kALIGNMENT) / sizeof(float))
    {
        return 0;
    }
    size_t const bytes = elements * sizeof(float);
    return (bytes + kALIGNMENT - 1) / kALIGNMENT * kALIGNMENT;
}

bool alignedPointer(void const* pointer)
{
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}
} // namespace

bool CuteDslSSDRunner::canImplementPackedStatePool(int32_t dim, int32_t dstate, int32_t smVersion)
{
#if defined(CUTE_DSL_SSD_SM12X_ENABLED)
    return (smVersion == 120 || smVersion == 121) && (dim == 64 || dim == 80 || dim == 128)
        && (dstate == 64 || dstate == 128);
#else
    return false;
#endif
}

size_t CuteDslSSDRunner::getPackedWorkspaceSize(int32_t totalTokens, int32_t nheads)
{
    if (totalTokens <= 0 || nheads <= 0)
    {
        return 0;
    }
    return 2 * packedScratchBytes(totalTokens, nheads);
}

#if defined(CUTE_DSL_SSD_SM12X_ENABLED)
namespace
{
// All descriptor dimensions are dynamic; only the innermost stride is implicit.
template <typename Tensor, size_t Rank>
Tensor makePackedTensor(void* pointer, std::array<int32_t, Rank> const& shape)
{
    static_assert(Rank == std::extent_v<decltype(Tensor::dynamic_shapes)>);
    Tensor tensor{};
    tensor.data = pointer;
    for (size_t i = 0; i < Rank; ++i)
    {
        tensor.dynamic_shapes[i] = shape[i];
    }
    if constexpr (Rank > 1)
    {
        static_assert(Rank - 1 == std::extent_v<decltype(Tensor::dynamic_strides)>);
        int64_t stride{1};
        for (size_t i = Rank - 1; i > 0; --i)
        {
            stride *= shape[i];
            tensor.dynamic_strides[i - 1] = stride;
        }
    }
    return tensor;
}

template <auto Load, auto Unload, int32_t ScanGroupSize, typename Module, typename X, typename Dt, typename A,
    typename B, typename C, typename D, typename Bias, typename CuSeqlens, typename StateIndices, typename State,
    typename Output, typename Delta, typename Prefix>
int32_t runPackedVariant(PackedSSDParams const& params, cudaStream_t stream, size_t scratchBytes,
    char const* moduleName,
    int32_t (*launch)(Module*, X*, Dt*, A*, B*, C*, D*, Bias*, CuSeqlens*, StateIndices*, State*, Output*, Delta*,
        Prefix*, cudaStream_t))
{
    static_assert(ScanGroupSize > 0);
    if (params.totalTokens / ScanGroupSize + params.batch > 65535)
    {
        return -1;
    }
    static detail::LazyKernelModule<Module> sModule;
    if (!detail::ensureModuleLoaded<Load, Unload>(sModule, moduleName, stream))
    {
        return -1;
    }
    auto x = makePackedTensor<X>(params.x, std::array{params.totalTokens, params.nheads, params.dim});
    auto dt = makePackedTensor<Dt>(params.dt, std::array{params.totalTokens, params.nheads});
    auto a = makePackedTensor<A>(params.A, std::array{params.nheads});
    auto b = makePackedTensor<B>(params.B, std::array{params.totalTokens, params.ngroups, params.dstate});
    auto c = makePackedTensor<C>(params.C, std::array{params.totalTokens, params.ngroups, params.dstate});
    auto d = makePackedTensor<D>(params.D, std::array{params.nheads});
    auto bias = makePackedTensor<Bias>(params.dtBias, std::array{params.nheads});
    auto cu = makePackedTensor<CuSeqlens>(params.cuSeqlens, std::array{params.batch + 1});
    auto indices = makePackedTensor<StateIndices>(params.stateIndices, std::array{params.batch});
    auto state = makePackedTensor<State>(
        params.state, std::array{params.stateSlots, params.nheads, params.dim, params.dstate});
    auto output = makePackedTensor<Output>(params.output, std::array{params.totalTokens, params.nheads, params.dim});
    auto delta = makePackedTensor<Delta>(params.workspace, std::array{params.totalTokens, params.nheads});
    auto prefix = makePackedTensor<Prefix>(
        static_cast<char*>(params.workspace) + scratchBytes, std::array{params.totalTokens, params.nheads});
    return launch(
        &sModule.module, &x, &dt, &a, &b, &c, &d, &bias, &cu, &indices, &state, &output, &delta, &prefix, stream);
}
} // namespace
#endif

int32_t CuteDslSSDRunner::runPacked(PackedSSDParams const& params, cudaStream_t stream)
{
    if (!canImplementPackedStatePool(params.dim, params.dstate, params.smVersion) || stream == nullptr
        || params.totalTokens < 0 || params.batch < 0 || params.batch > 65535 || params.stateSlots < params.batch
        || params.nheads <= 0 || params.nheads > 65535 || params.ngroups <= 0 || params.nheads % params.ngroups != 0)
    {
        return -1;
    }
    if (params.totalTokens == 0)
    {
        return 0;
    }
    size_t const scratchBytes = packedScratchBytes(params.totalTokens, params.nheads);
    if (params.batch == 0 || scratchBytes == 0 || params.workspaceSize < 2 * scratchBytes || !alignedPointer(params.x)
        || !alignedPointer(params.dt) || !alignedPointer(params.A) || !alignedPointer(params.B)
        || !alignedPointer(params.C) || !alignedPointer(params.D) || !alignedPointer(params.dtBias)
        || !alignedPointer(params.cuSeqlens) || !alignedPointer(params.stateIndices) || !alignedPointer(params.state)
        || !alignedPointer(params.output) || !alignedPointer(params.workspace))
    {
        return -1;
    }
#if defined(CUTE_DSL_SSD_SM12X_ENABLED)
    if (params.dim == 64 && params.dstate == 64)
    {
        return runPackedVariant<ssd_prefill_sm12x_d64_n64_Kernel_Module_Load,
            ssd_prefill_sm12x_d64_n64_Kernel_Module_Unload, ssd_prefill_sm12x_d64_n64_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d64_n64", cute_dsl_ssd_prefill_sm12x_d64_n64_wrapper);
    }
    if (params.dim == 64 && params.dstate == 128)
    {
        return runPackedVariant<ssd_prefill_sm12x_d64_n128_Kernel_Module_Load,
            ssd_prefill_sm12x_d64_n128_Kernel_Module_Unload, ssd_prefill_sm12x_d64_n128_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d64_n128", cute_dsl_ssd_prefill_sm12x_d64_n128_wrapper);
    }
    if (params.dim == 80 && params.dstate == 64)
    {
        return runPackedVariant<ssd_prefill_sm12x_d80_n64_Kernel_Module_Load,
            ssd_prefill_sm12x_d80_n64_Kernel_Module_Unload, ssd_prefill_sm12x_d80_n64_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d80_n64", cute_dsl_ssd_prefill_sm12x_d80_n64_wrapper);
    }
    if (params.dim == 80 && params.dstate == 128)
    {
        return runPackedVariant<ssd_prefill_sm12x_d80_n128_Kernel_Module_Load,
            ssd_prefill_sm12x_d80_n128_Kernel_Module_Unload, ssd_prefill_sm12x_d80_n128_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d80_n128", cute_dsl_ssd_prefill_sm12x_d80_n128_wrapper);
    }
    if (params.dim == 128 && params.dstate == 64)
    {
        return runPackedVariant<ssd_prefill_sm12x_d128_n64_Kernel_Module_Load,
            ssd_prefill_sm12x_d128_n64_Kernel_Module_Unload, ssd_prefill_sm12x_d128_n64_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d128_n64", cute_dsl_ssd_prefill_sm12x_d128_n64_wrapper);
    }
    if (params.dim == 128 && params.dstate == 128)
    {
        return runPackedVariant<ssd_prefill_sm12x_d128_n128_Kernel_Module_Load,
            ssd_prefill_sm12x_d128_n128_Kernel_Module_Unload, ssd_prefill_sm12x_d128_n128_kSCAN_GROUP_SIZE>(
            params, stream, scratchBytes, "ssd_prefill_sm12x_d128_n128", cute_dsl_ssd_prefill_sm12x_d128_n128_wrapper);
    }
#endif
    return -1;
}

} // namespace trt_edgellm
#endif // CUTE_DSL_SSD_ENABLED
