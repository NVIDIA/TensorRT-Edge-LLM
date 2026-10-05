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

#include "nvfp4A16BlackwellMoeRunner.h"

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/fp4SupportKernels/buildLayout.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "kernels/nvfp4A16BlackwellSupport.h"
#include "nvfp4A16BlackwellMoeJitCompiler.h"

#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_nvfp4_a16_blackwell_moe_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

namespace moe = nvfp4_a16_blackwell_moe;

constexpr size_t kAlignment{256};

size_t alignUp(size_t const bytes) noexcept
{
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

size_t elementSize(moe::DecodeDtype const dtype) noexcept
{
    (void) dtype;
    return 2; // FP16 and BF16
}

//! Decode FC1 split-K for a shape: the sealed policy value, or the benchmark-only
//! override EDGELLM_MOE_DECODE_FC1_SPLITK=1|2|4|8 (read once), clamped to the
//! K-tile count.  Only the launch and numGpuOps depend on it: the workspace is
//! always sized for kDecodeFc1MaxSplitK so the size TensorRT records at engine
//! build cannot disagree with a run-time override.
int32_t decodeFc1SplitK(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    static int32_t const requested = []() {
        char const* const env = std::getenv("EDGELLM_MOE_DECODE_FC1_SPLITK");
        if (env != nullptr)
        {
            int32_t const v = std::atoi(env);
            if ((v == 1 || v == 2 || v == 4 || v == 8) && v <= moe::kDecodeFc1MaxSplitK)
            {
                return v;
            }
        }
        return moe::kDecodeFc1SplitK;
    }();
    return std::max<int32_t>(1, std::min<int32_t>(requested, p.hiddenSize / nvfp4_a16_blackwell::kKTile));
}

#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
//! Effective PDL mode: the request from the plugin (EDGELLM_ENABLE_PDL) gated by
//! toolchain support.  The kernels are SM110-only, so no SM gate is needed.
constexpr bool usePdl(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    return p.enablePdl && SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH != 0;
}

//! Sigmoid top-k routing into topkIndices / topkWeights: the warp-per-token fast
//! path for the ungrouped contract (Nemotron), the shared grouped kernel otherwise.
//! The shared kernel has no griddepcontrol wait, so it is launched without the
//! PDL attribute (the following kernel's wait still orders it correctly).
cudaError_t launchRouting(Nvfp4A16BlackwellMoeParams const& p, int32_t* const topkIndices, float* const topkWeights,
    cudaStream_t const stream) noexcept
{
    try
    {
        if (p.nGroup == 1 && p.numExperts <= 512)
        {
            p.jit->launchRoute(p.routerLogits, p.correctionBias, p.numTokens, p.normTopkProb, p.routedScalingFactor,
                topkIndices, topkWeights, usePdl(p), stream);
            return cudaSuccess;
        }
        rt::Tensor const logits(const_cast<float*>(p.routerLogits), {p.numTokens, p.numExperts}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT);
        rt::Tensor weights(topkWeights, {p.numTokens, p.topK}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor indices(topkIndices, {p.numTokens, p.topK}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor const bias(
            const_cast<float*>(p.correctionBias), {p.numExperts}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::OptionalInputTensor optionalBias = std::nullopt;
        if (p.correctionBias != nullptr)
        {
            optionalBias = std::cref(bias);
        }
        moeSigmoidGroupTopk(logits, weights, indices, p.topK, p.nGroup, p.topkGroup, p.normTopkProb,
            p.routedScalingFactor, stream, optionalBias);
        return cudaGetLastError();
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
}
#endif // CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED

//! Single source of truth for the workspace carve-out (sizing and run use it).
struct WorkspaceLayout
{
    size_t topkWeights{0};
    size_t topkIndices{0};
    size_t tileGroupIdx{0};
    size_t tileMnLimit{0};
    size_t permutedIdx{0};
    size_t numValidTiles{0};
    size_t permutedActivations{0};
    size_t fc1Output{0};
    size_t fc1Partials{0};
    size_t fc2Partials{0};
    size_t total{0};
    int64_t maxRowsPadded{0};
    int64_t maxTiles{0};
    int32_t tokenTile{0};
};

WorkspaceLayout buildWorkspaceLayout(Nvfp4A16BlackwellMoeParams const& p, int32_t const tokenTile) noexcept
{
    WorkspaceLayout l{};
    l.tokenTile = tokenTile;
    int64_t const slots = static_cast<int64_t>(p.numTokens) * p.topK;
    l.maxRowsPadded = moe::maxRowsPadded(p.numTokens, p.topK, p.numExperts, tokenTile);
    l.maxTiles = l.maxRowsPadded / tokenTile;
    size_t const act = elementSize(p.dtype);
    size_t cursor = 0;
    auto place = [&cursor](size_t& offset, size_t const bytes) {
        offset = cursor;
        cursor += alignUp(bytes);
    };
    place(l.topkWeights, static_cast<size_t>(slots) * sizeof(float));
    place(l.topkIndices, static_cast<size_t>(slots) * sizeof(int32_t));
    place(l.tileGroupIdx, static_cast<size_t>(l.maxTiles) * sizeof(int32_t));
    place(l.tileMnLimit, static_cast<size_t>(l.maxTiles) * sizeof(int32_t));
    place(l.permutedIdx, static_cast<size_t>(l.maxRowsPadded) * sizeof(int32_t));
    place(l.numValidTiles, sizeof(int32_t));
    place(l.permutedActivations, static_cast<size_t>(l.maxRowsPadded) * p.hiddenSize * act);
    // fc1Output serves the prefill permuted intermediate [maxRowsPadded, I_pad]
    // and the decode slot intermediate [slots, I_pad]; the former dominates.
    place(l.fc1Output, static_cast<size_t>(std::max<int64_t>(l.maxRowsPadded, slots)) * p.interSizePadded * act);
    // Decode split-K partials.  The decode kernels only run for numTokens <=
    // kDecodeMaxTokens unless the backend is forced, so size the partials for
    // that many tokens (a prefill profile up to 4096 tokens would otherwise
    // reserve hundreds of MB for buffers it never touches).
    int64_t const decodeTokens = p.backend == moe::Backend::kDecode
        ? static_cast<int64_t>(p.numTokens)
        : std::min<int64_t>(p.numTokens, moe::kDecodeMaxTokens);
    // Sized for the largest FC1 split-K the launch may pick (never the env override).
    place(l.fc1Partials,
        static_cast<size_t>(moe::kDecodeFc1MaxSplitK) * decodeTokens * p.topK * p.interSizePadded * sizeof(float));
    size_t const fc2PartialBytes = moe::kDecodeFc2SplitK > 1
        ? static_cast<size_t>(moe::kDecodeFc2SplitK) * decodeTokens * p.hiddenSize * sizeof(float)
        : 0;
    place(l.fc2Partials, fc2PartialBytes);
    l.total = cursor;
    return l;
}

bool validShape(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    if (p.dtype != moe::DecodeDtype::kFP16)
    {
        return false; // BF16 variants are not baked into the AOT group.
    }
    if (p.numTokens <= 0 || p.numExperts <= 0 || p.numExperts > 512 || p.topK <= 0 || p.topK > 32
        || p.topK > p.numExperts)
    {
        return false;
    }
    if (p.nGroup <= 0 || p.numExperts % p.nGroup != 0 || p.topkGroup <= 0 || p.topkGroup > p.nGroup)
    {
        return false;
    }
    if (p.hiddenSize <= 0 || p.hiddenSize % nvfp4_a16_blackwell::kNTile != 0 || p.interSize <= 0
        || p.interSize % nvfp4_a16_blackwell::kKTile != 0 || p.interSizePadded < p.interSize
        || p.interSizePadded % nvfp4_a16_blackwell::kNTile != 0
        || p.interSizePadded - p.interSize >= nvfp4_a16_blackwell::kNTile)
    {
        return false;
    }
    // The kernel stages one expert id per token tile in shared memory; the tile
    // count is largest at the profile's max token count with its selected tile
    // (the smaller-tile bands end at 512 and 2048 tokens and stay far below).
    int32_t const tileAtMax = static_cast<int32_t>(moe::selectTokenTile(p.numTokens));
    if (moe::maxRowsPadded(p.numTokens, p.topK, p.numExperts, tileAtMax) / tileAtMax > moe::kMaxTokenTiles)
    {
        return false;
    }
    // Both grouped GEMMs must stay TMA-representable at the largest tile.
    int64_t const rows = moe::maxRowsPadded(p.numTokens, p.topK, p.numExperts, moe::kLargestTokenTile);
    return nvfp4_a16_blackwell::isTmaRepresentableProblem(rows, p.interSizePadded, p.hiddenSize)
        && nvfp4_a16_blackwell::isTmaRepresentableProblem(rows, p.hiddenSize, p.interSize)
        && static_cast<int64_t>(p.numExperts) * p.interSizePadded * p.hiddenSize / 2 < (int64_t{1} << 40)
        && static_cast<int64_t>(p.numExperts) * p.hiddenSize * p.interSize / 2 < (int64_t{1} << 40);
}

#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn8_tk64_Kernel_Module_t> gFc1Tn8{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn16_tk64_Kernel_Module_t> gFc1Tn16{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn32_tk64_Kernel_Module_t> gFc1Tn32{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn64_tk64_Kernel_Module_t> gFc1Tn64{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn128_tk64_Kernel_Module_t> gFc1Tn128{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn8_tk64_Kernel_Module_t> gFc2Tn8{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn16_tk64_Kernel_Module_t> gFc2Tn16{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn32_tk64_Kernel_Module_t> gFc2Tn32{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn64_tk64_Kernel_Module_t> gFc2Tn64{};
detail::LazyKernelModule<nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn128_tk64_Kernel_Module_t> gFc2Tn128{};

//! Arguments of one grouped GEMM wrapper call (activation == nullptr means
//! "load the module only").
struct GroupedArgs
{
    void const* activation{nullptr};
    void const* qweight{nullptr};
    void const* blockScales{nullptr};
    float const* globalScales{nullptr};
    void* output{nullptr};
    int32_t const* tileGroupIdx{nullptr};
    int32_t const* numValidTiles{nullptr};
    int32_t const* permutedIdx{nullptr};
    float const* topkWeights{nullptr};
    int32_t numRowsPadded{0};
    int32_t activationLd{0};
    int32_t outFeatures{0};
    int32_t inFeatures{0};
    int32_t numExperts{0};
    int32_t numTokens{0};
    int32_t topK{0};
    int32_t maxActiveClusters{0};
    int32_t enablePdl{0};
};

template <auto Loader, auto Unloader, auto Wrapper, typename Module>
cudaError_t launchGrouped(
    detail::LazyKernelModule<Module>& module, char const* name, GroupedArgs const& a, cudaStream_t stream) noexcept
{
    if (!detail::ensureModuleLoaded<Loader, Unloader>(module, name, stream))
    {
        return cudaErrorInitializationError;
    }
    if (a.activation == nullptr)
    {
        return cudaSuccess;
    }
    int32_t const result = Wrapper(&module.module, const_cast<void*>(a.activation), const_cast<void*>(a.qweight),
        const_cast<void*>(a.blockScales), const_cast<float*>(a.globalScales), a.output,
        const_cast<int32_t*>(a.tileGroupIdx), const_cast<int32_t*>(a.numValidTiles),
        const_cast<int32_t*>(a.permutedIdx), const_cast<float*>(a.topkWeights), a.numRowsPadded, a.activationLd,
        a.outFeatures, a.inFeatures, a.numExperts, a.numTokens, a.topK, a.maxActiveClusters, a.enablePdl, stream);
    return result == 0 ? cudaSuccess : cudaErrorUnknown;
}

#define NVFP4_A16_MOE_LAUNCH(MODULE, NAME)                                                                             \
    launchGrouped<NAME##_Kernel_Module_Load, NAME##_Kernel_Module_Unload, cute_dsl_##NAME##_wrapper>(                  \
        MODULE, #NAME, args, stream)

cudaError_t launchFc1(moe::TokenTile const tile, GroupedArgs const& args, cudaStream_t stream) noexcept
{
    switch (tile)
    {
    case moe::TokenTile::kTn8:
        return NVFP4_A16_MOE_LAUNCH(gFc1Tn8, nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn8_tk64);
    case moe::TokenTile::kTn16:
        return NVFP4_A16_MOE_LAUNCH(gFc1Tn16, nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn16_tk64);
    case moe::TokenTile::kTn32:
        return NVFP4_A16_MOE_LAUNCH(gFc1Tn32, nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn32_tk64);
    case moe::TokenTile::kTn64:
        return NVFP4_A16_MOE_LAUNCH(gFc1Tn64, nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn64_tk64);
    case moe::TokenTile::kTn128:
        return NVFP4_A16_MOE_LAUNCH(gFc1Tn128, nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn128_tk64);
    }
    return cudaErrorInvalidValue;
}

cudaError_t launchFc2(moe::TokenTile const tile, GroupedArgs const& args, cudaStream_t stream) noexcept
{
    switch (tile)
    {
    case moe::TokenTile::kTn8:
        return NVFP4_A16_MOE_LAUNCH(gFc2Tn8, nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn8_tk64);
    case moe::TokenTile::kTn16:
        return NVFP4_A16_MOE_LAUNCH(gFc2Tn16, nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn16_tk64);
    case moe::TokenTile::kTn32:
        return NVFP4_A16_MOE_LAUNCH(gFc2Tn32, nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn32_tk64);
    case moe::TokenTile::kTn64:
        return NVFP4_A16_MOE_LAUNCH(gFc2Tn64, nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn64_tk64);
    case moe::TokenTile::kTn128:
        return NVFP4_A16_MOE_LAUNCH(gFc2Tn128, nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn128_tk64);
    }
    return cudaErrorInvalidValue;
}

#undef NVFP4_A16_MOE_LAUNCH

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
    return (smVersion > 0 && maxActiveClusters > 0) ? cudaSuccess : cudaErrorInvalidDevice;
}

// The wrapper receives the SM count as max_active_clusters; cache it at
// prepare() so enqueue never queries the device (CUDA-graph safe). Atomic
// because TensorRT may configure several layers from different threads; the
// value is a property of the single device, identical for every instance.
std::atomic<int32_t> gMaxActiveClusters{0};

//! Benchmark-only override of the token tile (EDGELLM_MOE_FORCE_TILE=8|16|32|64|128),
//! read once; used by the committed Marlin-vs-Blackwell benchmark to sweep tiles
//! per token count when re-sealing selectTokenTile. 0 = follow the policy.
int32_t forcedTokenTile() noexcept
{
    static int32_t const value = []() noexcept {
        char const* const env = std::getenv("EDGELLM_MOE_FORCE_TILE");
        int32_t const v = env == nullptr ? 0 : std::atoi(env);
        return (v == 8 || v == 16 || v == 32 || v == 64 || v == 128) ? v : 0;
    }();
    return value;
}

//! Tile actually used for a grouped-GEMM launch at numTokens: the forced tile when
//! set and representable, otherwise the sealed policy.
int32_t prefillTokenTile(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    int32_t const forced = forcedTokenTile();
    if (forced > 0 && moe::maxRowsPadded(p.numTokens, p.topK, p.numExperts, forced) / forced <= moe::kMaxTokenTiles)
    {
        return forced;
    }
    return static_cast<int32_t>(moe::selectTokenTile(p.numTokens));
}

//! Tiles reachable by the auto policy for token counts in (kDecodeMaxTokens, maxTokens].
bool tileReachable(moe::TokenTile const tile, int32_t const maxTokens, moe::Backend const backend) noexcept
{
    if (backend == moe::Backend::kDecode)
    {
        return false;
    }
    int32_t const firstPrefillTokens = backend == moe::Backend::kPrefill ? 1 : moe::kDecodeMaxTokens + 1;
    if (maxTokens < firstPrefillTokens)
    {
        return false;
    }
    // selectTokenTile is monotone in numTokens; check its range endpoints.
    moe::TokenTile const lo = moe::selectTokenTile(firstPrefillTokens);
    moe::TokenTile const hi = moe::selectTokenTile(maxTokens);
    return static_cast<int32_t>(lo) <= static_cast<int32_t>(tile)
        && static_cast<int32_t>(tile) <= static_cast<int32_t>(hi);
}

cudaError_t runPrefill(
    Nvfp4A16BlackwellMoeParams const& p, unsigned char* ws, WorkspaceLayout const& l, cudaStream_t stream) noexcept
{
    try
    {
        int32_t const slots = p.numTokens * p.topK;
        cudaError_t err = launchRouting(
            p, reinterpret_cast<int32_t*>(ws + l.topkIndices), reinterpret_cast<float*>(ws + l.topkWeights), stream);
        if (err != cudaSuccess)
        {
            return err;
        }
        if (p.nGroup == 1 && p.numExperts <= 512)
        {
            // Ungrouped contract (Nemotron): the parallel single-CTA layout
            // builder (~10 us with the warp-per-token routing at T=256 versus
            // ~75 us for the generic pair).
            p.jit->launchLayout(reinterpret_cast<int32_t const*>(ws + l.topkIndices), slots, l.tokenTile,
                reinterpret_cast<int32_t*>(ws + l.permutedIdx), reinterpret_cast<int32_t*>(ws + l.tileGroupIdx),
                reinterpret_cast<int32_t*>(ws + l.numValidTiles), usePdl(p), stream);
        }
        else
        {
            MoELayoutBuffers layout{};
            layout.tileIdxToGroupIdx = rt::Tensor(ws + l.tileGroupIdx, {static_cast<int32_t>(l.maxTiles)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
            layout.tileIdxToMnLimit = rt::Tensor(ws + l.tileMnLimit, {static_cast<int32_t>(l.maxTiles)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
            layout.permutedIdxToExpandedIdx = rt::Tensor(ws + l.permutedIdx, {static_cast<int32_t>(l.maxRowsPadded)},
                rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
            layout.numNonExitingTiles
                = rt::Tensor(ws + l.numValidTiles, {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
            buildLayoutGpu(layout, reinterpret_cast<int32_t const*>(ws + l.topkIndices), p.numTokens, p.topK,
                p.numExperts, l.tokenTile, stream);
        }

        p.jit->launchGather(p.hiddenStates, reinterpret_cast<int32_t const*>(ws + l.permutedIdx),
            reinterpret_cast<int32_t const*>(ws + l.numValidTiles), l.tokenTile, p.numTokens, l.maxRowsPadded,
            ws + l.permutedActivations, p.output, usePdl(p), stream);

        moe::TokenTile const tile = static_cast<moe::TokenTile>(l.tokenTile);
        GroupedArgs fc1{};
        fc1.activation = ws + l.permutedActivations;
        fc1.qweight = p.fc1QWeights;
        fc1.blockScales = p.fc1BlockScales;
        fc1.globalScales = p.fc1GlobalScales;
        fc1.output = ws + l.fc1Output;
        fc1.tileGroupIdx = reinterpret_cast<int32_t const*>(ws + l.tileGroupIdx);
        fc1.numValidTiles = reinterpret_cast<int32_t const*>(ws + l.numValidTiles);
        fc1.permutedIdx = reinterpret_cast<int32_t const*>(ws + l.permutedIdx);
        fc1.topkWeights = reinterpret_cast<float const*>(ws + l.topkWeights);
        fc1.numRowsPadded = static_cast<int32_t>(l.maxRowsPadded);
        fc1.activationLd = p.hiddenSize;
        fc1.outFeatures = p.interSizePadded;
        fc1.inFeatures = p.hiddenSize;
        fc1.numExperts = p.numExperts;
        fc1.numTokens = p.numTokens;
        fc1.topK = p.topK;
        fc1.maxActiveClusters = gMaxActiveClusters.load(std::memory_order_relaxed);
        fc1.enablePdl = usePdl(p) ? 1 : 0;
        err = launchFc1(tile, fc1, stream);
        if (err != cudaSuccess)
        {
            return err;
        }

        GroupedArgs fc2 = fc1;
        fc2.activation = ws + l.fc1Output;
        fc2.qweight = p.fc2QWeights;
        fc2.blockScales = p.fc2BlockScales;
        fc2.globalScales = p.fc2GlobalScales;
        fc2.output = p.output;
        fc2.activationLd = p.interSizePadded;
        fc2.outFeatures = p.hiddenSize;
        fc2.inFeatures = p.interSize;
        return launchFc2(tile, fc2, stream);
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
}

cudaError_t runDecode(
    Nvfp4A16BlackwellMoeParams const& p, unsigned char* ws, WorkspaceLayout const& l, cudaStream_t stream) noexcept
{
    try
    {
        int32_t* const topkIndices = reinterpret_cast<int32_t*>(ws + l.topkIndices);
        float* const topkWeights = reinterpret_cast<float*>(ws + l.topkWeights);
        // Routing runs as its own (tiny) kernel: fusing it into every FC1 CTA cost
        // 131 registers, one CTA per SM and ~40% of FC1's streaming bandwidth.
        cudaError_t const err = launchRouting(p, topkIndices, topkWeights, stream);
        if (err != cudaSuccess)
        {
            return err;
        }
        Nvfp4A16BlackwellMoeJitKey const& key = p.jit->getKey();
        bool const pdl = usePdl(p);
        p.jit->launchFc1(p.hiddenStates, topkIndices, p.fc1QWeights, p.fc1BlockScales, p.fc1GlobalScales,
            ws + l.fc1Output, key.fc1SplitK > 1 ? reinterpret_cast<float*>(ws + l.fc1Partials) : nullptr, p.numTokens,
            pdl, stream);
        p.jit->launchFc2(ws + l.fc1Output, topkIndices, topkWeights, p.fc2QWeights, p.fc2BlockScales, p.fc2GlobalScales,
            p.output, key.fc2SplitK > 1 ? reinterpret_cast<float*>(ws + l.fc2Partials) : nullptr, p.numTokens, pdl,
            stream);
        return cudaSuccess;
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
}

//! The JIT runner attached to the params must hold the bundle compiled for this
//! layer's shape and dtype (split-K / prefetch come from the bundle itself).
bool jitMatchesShape(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    if (p.jit == nullptr || !p.jit->isLoaded())
    {
        return false;
    }
    Nvfp4A16BlackwellMoeJitKey const& key = p.jit->getKey();
    return key.sm == nvfp4_a16_blackwell::kTargetSm && key.layout == kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI
        && key.numExperts == p.numExperts && key.topK == p.topK && key.hiddenSize == p.hiddenSize
        && key.interSize == p.interSize && key.interSizePadded == p.interSizePadded
        && key.dataType
        == (p.dtype == moe::DecodeDtype::kBF16 ? Nvfp4A16BlackwellMoeDataType::kBF16
                                               : Nvfp4A16BlackwellMoeDataType::kHALF);
}
#endif // CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED

} // namespace

Nvfp4A16BlackwellMoeJitKey makeNvfp4A16BlackwellMoeJitKey(Nvfp4A16BlackwellMoeParams const& p) noexcept
{
    Nvfp4A16BlackwellMoeJitKey key{};
    key.numExperts = p.numExperts;
    key.topK = p.topK;
    key.hiddenSize = p.hiddenSize;
    key.interSize = p.interSize;
    key.interSizePadded = p.interSizePadded;
    key.dataType = p.dtype == moe::DecodeDtype::kBF16 ? Nvfp4A16BlackwellMoeDataType::kBF16
                                                      : Nvfp4A16BlackwellMoeDataType::kHALF;
    key.fc1SplitK = decodeFc1SplitK(p);
    key.fc2SplitK
        = std::max<int32_t>(1, std::min<int32_t>(moe::kDecodeFc2SplitK, p.interSize / nvfp4_a16_blackwell::kKTile));
    // Decode FC2 pre-wait prefetch slots: the explicit request (tests), else the
    // benchmark-only override EDGELLM_MOE_DECODE_FC2_PREFETCH (read once), else the
    // sealed policy value; clamped to topK, the policy maximum and the 48 KB static
    // shared-memory limit of the baked FC2 kernel.
    static int32_t const requested = []() {
        char const* const env = std::getenv("EDGELLM_MOE_DECODE_FC2_PREFETCH");
        if (env != nullptr)
        {
            int32_t const v = std::atoi(env);
            if (v >= 0 && v <= moe::kDecodeFc2MaxPrefetchSlots)
            {
                return v;
            }
        }
        return moe::kDecodeFc2PrefetchSlots;
    }();
    int32_t slots = std::min<int32_t>(p.fc2PrefetchSlots >= 0 ? p.fc2PrefetchSlots : requested, p.topK);
    slots = std::max<int32_t>(0, std::min<int32_t>(slots, moe::kDecodeFc2MaxPrefetchSlots));
    key.fc2PrefetchSlots = slots;
    while (key.fc2PrefetchSlots > 0 && getNvfp4A16BlackwellMoeFc2SharedBytes(key) > 48 * 1024)
    {
        --key.fc2PrefetchSlots;
    }
    return key;
}

bool Nvfp4A16BlackwellMoeRunner::isSupported(int32_t const smVersion, Nvfp4A16BlackwellMoeParams const& shape) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
    return smVersion == nvfp4_a16_blackwell::kTargetSm && validShape(shape)
        && canCompileNvfp4A16BlackwellMoeJitKernel(makeNvfp4A16BlackwellMoeJitKey(shape));
#else
    (void) smVersion;
    (void) shape;
    return false;
#endif
}

cudaError_t Nvfp4A16BlackwellMoeRunner::prepare(
    Nvfp4A16BlackwellMoeParams const& shape, cudaStream_t const stream) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
    int32_t smVersion{0};
    int32_t maxActiveClusters{0};
    cudaError_t const deviceError = currentDeviceInfo(smVersion, maxActiveClusters);
    if (deviceError != cudaSuccess)
    {
        return deviceError;
    }
    if (smVersion != nvfp4_a16_blackwell::kTargetSm || !validShape(shape))
    {
        return cudaErrorNotSupported;
    }
    gMaxActiveClusters.store(maxActiveClusters, std::memory_order_relaxed);
    GroupedArgs const loadOnly{};
    for (moe::TokenTile const tile : {moe::TokenTile::kTn8, moe::TokenTile::kTn16, moe::TokenTile::kTn32,
             moe::TokenTile::kTn64, moe::TokenTile::kTn128})
    {
        if (!tileReachable(tile, shape.numTokens, shape.backend) && static_cast<int32_t>(tile) != forcedTokenTile())
        {
            continue;
        }
        cudaError_t err = launchFc1(tile, loadOnly, stream);
        if (err != cudaSuccess)
        {
            return err;
        }
        err = launchFc2(tile, loadOnly, stream);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    return cudaSuccess;
#else
    (void) shape;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

size_t Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(Nvfp4A16BlackwellMoeParams const& shape) noexcept
{
    if (!validShape(shape))
    {
        return 0;
    }
    // Size for the largest tile so any runtime tile fits.
    return buildWorkspaceLayout(shape, moe::kLargestTokenTile).total;
}

int32_t Nvfp4A16BlackwellMoeRunner::numGpuOps(Nvfp4A16BlackwellMoeParams const& params) noexcept
{
    moe::Backend const backend = moe::resolveBackend(params.backend, params.numTokens);
    if (backend == moe::Backend::kDecode)
    {
        Nvfp4A16BlackwellMoeJitKey const key = params.jit != nullptr && params.jit->isLoaded()
            ? params.jit->getKey()
            : makeNvfp4A16BlackwellMoeJitKey(params);
        return 3 + (key.fc1SplitK > 1 ? 1 : 0) + (key.fc2SplitK > 1 ? 1 : 0); // routing, FC1, FC2 (+ reduces)
    }
    return 5; // routing, tile layout, gather, FC1, FC2
}

cudaError_t Nvfp4A16BlackwellMoeRunner::run(Nvfp4A16BlackwellMoeParams const& p, void* workspace,
    size_t const workspaceSize, cudaStream_t const stream) noexcept
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED)
    if (!validShape(p) || workspace == nullptr || p.routerLogits == nullptr || p.hiddenStates == nullptr
        || p.output == nullptr || p.fc1QWeights == nullptr || p.fc1BlockScales == nullptr
        || p.fc1GlobalScales == nullptr || p.fc2QWeights == nullptr || p.fc2BlockScales == nullptr
        || p.fc2GlobalScales == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (!jitMatchesShape(p))
    {
        return cudaErrorInvalidValue; // the JIT bundle of this layer must be loaded (plugin clone / prepare)
    }
    moe::Backend const backend = moe::resolveBackend(p.backend, p.numTokens);
    int32_t const tokenTile = backend == moe::Backend::kDecode ? moe::kLargestTokenTile : prefillTokenTile(p);
    WorkspaceLayout const layout = buildWorkspaceLayout(p, tokenTile);
    if (layout.total > workspaceSize)
    {
        return cudaErrorInvalidValue;
    }
    unsigned char* const ws = static_cast<unsigned char*>(workspace);
    if (backend == moe::Backend::kDecode)
    {
        return runDecode(p, ws, layout, stream);
    }
    if (gMaxActiveClusters.load(std::memory_order_relaxed) <= 0)
    {
        return cudaErrorNotReady; // prepare() must run before the first enqueue
    }
    return runPrefill(p, ws, layout, stream);
#else
    (void) p;
    (void) workspace;
    (void) workspaceSize;
    (void) stream;
    return cudaErrorNotSupported;
#endif
}

} // namespace kernel
} // namespace trt_edgellm
