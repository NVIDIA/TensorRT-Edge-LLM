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

#include "nvfp4A16BlackwellGemmRunner.h"

#include "kernels/nvfp4A16BlackwellSupport.h"

#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
#include "common/cudaUtils.h"
#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_nvfp4_a16_blackwell_gemm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

namespace trt_edgellm
{
namespace kernels
{
namespace
{

#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_Kernel_Module_t> gFp16Tm8Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_Kernel_Module_t> gFp16Tm16Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_Kernel_Module_t> gFp16Tm32Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_Kernel_Module_t> gFp16Tm64Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_Kernel_Module_t> gFp16Tm128Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_Kernel_Module_t> gFp16Tm256Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_Kernel_Module_t> gBf16Tm8Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_Kernel_Module_t> gBf16Tm16Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_Kernel_Module_t> gBf16Tm32Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_Kernel_Module_t> gBf16Tm64Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_Kernel_Module_t> gBf16Tm128Module{};
detail::LazyKernelModule<nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_Kernel_Module_t> gBf16Tm256Module{};

cudaError_t currentDeviceInfo(int32_t& smVersion, int32_t& maxActiveClusters) noexcept
{
    try
    {
        smVersion = trt_edgellm::getSMVersion();
        maxActiveClusters = trt_edgellm::getDeviceMultiProcessorCount();
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
    if (smVersion <= 0 || maxActiveClusters <= 0)
    {
        return cudaErrorInvalidDevice;
    }
    return cudaSuccess;
}

template <auto Loader, auto Unloader, auto Wrapper, typename Module>
cudaError_t launchVariant(detail::LazyKernelModule<Module>& module, char const* name,
    Nvfp4A16BlackwellGemmParams const& params, int32_t maxActiveClusters, cudaStream_t stream) noexcept
{
    if (!detail::ensureModuleLoaded<Loader, Unloader>(module, name, stream))
    {
        return cudaErrorInitializationError;
    }
    if (params.activation == nullptr)
    {
        return cudaSuccess;
    }

    int32_t const launchResult
        = Wrapper(&module.module, const_cast<void*>(params.activation), const_cast<void*>(params.qweight),
            const_cast<void*>(params.blockScales), const_cast<float*>(params.globalScale), params.output,
            params.numTokens, params.outFeatures, params.inFeatures, maxActiveClusters, stream);
    return launchResult == 0 ? cudaSuccess : cudaErrorUnknown;
}

template <auto Loader, auto Unloader, typename Module>
bool loadVariant(detail::LazyKernelModule<Module>& module, char const* name, cudaStream_t stream) noexcept
{
    return detail::ensureModuleLoaded<Loader, Unloader>(module, name, stream);
}

bool dtypeCompiled(Nvfp4A16BlackwellDtype dtype) noexcept
{
    return dtype == Nvfp4A16BlackwellDtype::kFp16 || dtype == Nvfp4A16BlackwellDtype::kBf16;
}

cudaError_t launchSelected(Nvfp4A16BlackwellTokenTile tile, Nvfp4A16BlackwellGemmParams const& params,
    int32_t maxActiveClusters, cudaStream_t stream) noexcept
{
    if (params.dtype == Nvfp4A16BlackwellDtype::kFp16)
    {
        switch (tile)
        {
        case Nvfp4A16BlackwellTokenTile::kM8:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_wrapper>(
                gFp16Tm8Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM16:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_wrapper>(
                gFp16Tm16Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM32:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_wrapper>(
                gFp16Tm32Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM64:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_wrapper>(
                gFp16Tm64Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM128:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_wrapper>(
                gFp16Tm128Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM256:
            return launchVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_wrapper>(
                gFp16Tm256Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64", params, maxActiveClusters, stream);
        }
    }
    if (params.dtype == Nvfp4A16BlackwellDtype::kBf16)
    {
        switch (tile)
        {
        case Nvfp4A16BlackwellTokenTile::kM8:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_wrapper>(
                gBf16Tm8Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM16:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_wrapper>(
                gBf16Tm16Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM32:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_wrapper>(
                gBf16Tm32Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM64:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_wrapper>(
                gBf16Tm64Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM128:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_wrapper>(
                gBf16Tm128Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64", params, maxActiveClusters, stream);
        case Nvfp4A16BlackwellTokenTile::kM256:
            return launchVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_Kernel_Module_Load,
                nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_Kernel_Module_Unload,
                cute_dsl_nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_wrapper>(
                gBf16Tm256Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64", params, maxActiveClusters, stream);
        }
    }
    return cudaErrorNotSupported;
}

cudaError_t loadAllVariants(cudaStream_t stream) noexcept
{
    bool loaded{true};
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64_Kernel_Module_Unload>(
                 gFp16Tm8Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn8_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64_Kernel_Module_Unload>(
                 gFp16Tm16Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn16_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64_Kernel_Module_Unload>(
                 gFp16Tm32Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn32_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64_Kernel_Module_Unload>(
                 gFp16Tm64Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn64_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64_Kernel_Module_Unload>(
                 gFp16Tm128Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn128_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64_Kernel_Module_Unload>(
                 gFp16Tm256Module, "nvfp4_a16_blackwell_gemm_fp16_tm128_tn256_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64_Kernel_Module_Unload>(
                 gBf16Tm8Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn8_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64_Kernel_Module_Unload>(
                 gBf16Tm16Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn16_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64_Kernel_Module_Unload>(
                 gBf16Tm32Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn32_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64_Kernel_Module_Unload>(
                 gBf16Tm64Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn64_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64_Kernel_Module_Unload>(
                 gBf16Tm128Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn128_tk64", stream)
        && loaded;
    loaded = loadVariant<nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_Kernel_Module_Load,
                 nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64_Kernel_Module_Unload>(
                 gBf16Tm256Module, "nvfp4_a16_blackwell_gemm_bf16_tm128_tn256_tk64", stream)
        && loaded;
    return loaded ? cudaSuccess : cudaErrorInitializationError;
}
#endif // CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED

} // namespace

cudaError_t Nvfp4A16BlackwellGemmRunner::prepare(Nvfp4A16BlackwellDtype dtype, int32_t numTokens, int32_t outFeatures,
    int32_t inFeatures, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = currentDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    if (!isSupported(smVersion, dtype, numTokens, outFeatures, inFeatures))
    {
        return cudaErrorNotSupported;
    }

    Nvfp4A16BlackwellGemmParams const params{
        nullptr, nullptr, nullptr, nullptr, nullptr, numTokens, outFeatures, inFeatures, dtype};
    return launchSelected(selectTokenTile(numTokens), params, maxActiveClusters, stream);
#else
    (void) dtype;
    (void) numTokens;
    (void) outFeatures;
    (void) inFeatures;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t Nvfp4A16BlackwellGemmRunner::loadKernelModules(cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = currentDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    (void) maxActiveClusters;
    if (smVersion != nvfp4_a16_blackwell::kTargetSm)
    {
        return cudaErrorNotSupported;
    }
    return loadAllVariants(stream);
#else
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

bool Nvfp4A16BlackwellGemmRunner::isSupported(int32_t smVersion, Nvfp4A16BlackwellDtype dtype, int32_t numTokens,
    int32_t outFeatures, int32_t inFeatures) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    if (smVersion != nvfp4_a16_blackwell::kTargetSm
        || !nvfp4_a16_blackwell::isTmaRepresentableProblem(numTokens, outFeatures, inFeatures))
    {
        return false;
    }
    if (dtype != Nvfp4A16BlackwellDtype::kFp16 && dtype != Nvfp4A16BlackwellDtype::kBf16)
    {
        return false;
    }
    return dtypeCompiled(dtype);
#else
    (void) smVersion;
    (void) dtype;
    (void) numTokens;
    (void) outFeatures;
    (void) inFeatures;
    return false;
#endif
}

Nvfp4A16BlackwellTokenTile Nvfp4A16BlackwellGemmRunner::selectTokenTile(int32_t numTokens) noexcept
{
    if (numTokens <= 8)
    {
        return Nvfp4A16BlackwellTokenTile::kM8;
    }
    if (numTokens <= 16)
    {
        return Nvfp4A16BlackwellTokenTile::kM16;
    }
    if (numTokens <= 32)
    {
        return Nvfp4A16BlackwellTokenTile::kM32;
    }
    if (numTokens <= 64)
    {
        return Nvfp4A16BlackwellTokenTile::kM64;
    }
    if (numTokens <= 128)
    {
        return Nvfp4A16BlackwellTokenTile::kM128;
    }
    return Nvfp4A16BlackwellTokenTile::kM256;
}

size_t Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(Nvfp4A16BlackwellGemmParams const& params) noexcept
{
    (void) params;
    return 0;
}

cudaError_t Nvfp4A16BlackwellGemmRunner::run(
    Nvfp4A16BlackwellGemmParams const& params, void* workspace, size_t workspaceSize, cudaStream_t stream) noexcept
{
    (void) workspace;
    (void) workspaceSize;
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    if (params.activation == nullptr || params.qweight == nullptr || params.blockScales == nullptr
        || params.globalScale == nullptr || params.output == nullptr || params.numTokens <= 0)
    {
        return cudaErrorInvalidValue;
    }
    if (params.dtype != Nvfp4A16BlackwellDtype::kFp16 && params.dtype != Nvfp4A16BlackwellDtype::kBf16)
    {
        return cudaErrorInvalidValue;
    }

    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = currentDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    if (!isSupported(smVersion, params.dtype, params.numTokens, params.outFeatures, params.inFeatures))
    {
        return cudaErrorNotSupported;
    }
    return launchSelected(selectTokenTile(params.numTokens), params, maxActiveClusters, stream);
#else
    (void) params;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace kernels
} // namespace trt_edgellm
