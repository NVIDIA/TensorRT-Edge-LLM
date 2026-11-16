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

#include "q8_0Kernels.cuh"
#include "q8_0Linear.h"

#include <array>
#include <limits>
#include <optional>

namespace trt_edgellm::kernel
{
namespace
{
//! On discrete GPUs a DP4A GEMV beats one 8-row tensor-core tile up to this many rows.
constexpr int32_t kGEMV_MAX_ROWS_DISCRETE = 2;
//! On integrated GPUs the GEMV stays bandwidth-bound up to one 16-row token tile; prefill-size launches are
//! DP4A-bound there and take the tensor cores.
constexpr int32_t kGEMV_MAX_ROWS_INTEGRATED = 16;
//! Thor has 20 SMs, so its DP4A GEMV is already compute-bound above 4 rows.
constexpr int32_t kGEMV_MAX_ROWS_THOR = 4;
constexpr int32_t kSM_THOR = 110;
//! The activation fragments are read with 16-byte vector loads.
constexpr uintptr_t kWORKSPACE_ALIGNMENT = 16;
//! Largest GEMV token tile; more rows take extra passes over the weights.
constexpr int32_t kGEMV_MAX_TOKENS = 16;
constexpr int32_t kGEMV_MAX_CHANNELS = 4;
constexpr int32_t kGEMV_MAX_WARPS_PER_CHANNEL = q8_0::kWarpsPerCta;
//! A single-row GEMV narrows its channel group until this many CTAs per SM exist.
constexpr int32_t kGEMV_CTAS_PER_SM = 8;
//! ... but keeps at least this many output channels per CTA.
constexpr int32_t kGEMV_MIN_CHANNELS_PER_CTA = 2;
constexpr int32_t kMMA_MIN_COMPUTE_MAJOR = 8;
//! The tensor-core grid splits K until this many warps per SM exist ...
constexpr int32_t kMMA_WARPS_PER_SM = 16;
//! ... and no warp walks more than this many block pairs.
constexpr int32_t kMMA_MAX_PAIRS_PER_WARP = 16;
constexpr int32_t kMMA_MAX_WARPS_PER_CHANNEL_TILE = 32;

struct DeviceTraits
{
    int32_t computeMajor;
    int32_t computeMinor;
    int32_t smCount;
    int32_t integrated;
};

//! Queries the current device's constants. Successful queries are cached per thread and device: one process can
//! drive several GPUs, and a failed query must not be remembered.
cudaError_t deviceTraits(DeviceTraits& traits)
{
    constexpr int32_t kCACHED_DEVICES = 64;
    thread_local std::array<std::optional<DeviceTraits>, kCACHED_DEVICES> cache;
    int32_t device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        return status;
    }
    bool const cacheable = device >= 0 && device < kCACHED_DEVICES;
    if (cacheable && cache[device])
    {
        traits = *cache[device];
        return cudaSuccess;
    }
    DeviceTraits result{};
    status = cudaDeviceGetAttribute(&result.computeMajor, cudaDevAttrComputeCapabilityMajor, device);
    if (status == cudaSuccess)
    {
        status = cudaDeviceGetAttribute(&result.computeMinor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (status == cudaSuccess)
    {
        status = cudaDeviceGetAttribute(&result.smCount, cudaDevAttrMultiProcessorCount, device);
    }
    if (status == cudaSuccess)
    {
        status = cudaDeviceGetAttribute(&result.integrated, cudaDevAttrIntegrated, device);
    }
    if (status != cudaSuccess)
    {
        return status;
    }
    if (cacheable)
    {
        cache[device] = result;
    }
    traits = result;
    return cudaSuccess;
}

bool isThor(DeviceTraits const& device)
{
    int32_t const smVersion = device.computeMajor * 10 + device.computeMinor;
    return smVersion == kSM_THOR;
}

//! Integrated (LPDDR) GPUs stream the GEMV's contiguous per-lane blocks faster than the tensor-core tiles' 16-row
//! strided pairs, so they keep the GEMV until DP4A compute becomes the limit. So do heads with too many channel tiles
//! for the tensor-core grid to split K (lm_head), up to one token tile.
int32_t gemvMaxRows(DeviceTraits const& device, int32_t outFeatures)
{
    int32_t const channelTiles = (outFeatures + q8_0::kChannelsPerTile - 1) / q8_0::kChannelsPerTile;
    bool const wideHead
        = static_cast<int64_t>(channelTiles) >= static_cast<int64_t>(device.smCount) * kMMA_WARPS_PER_SM;
    if (device.integrated == 0)
    {
        return wideHead ? q8_0::kTokensPerTile : kGEMV_MAX_ROWS_DISCRETE;
    }
    return isThor(device) ? kGEMV_MAX_ROWS_THOR : kGEMV_MAX_ROWS_INTEGRATED;
}

int32_t gemvWarpsPerChannel(int32_t blocksPerRow)
{
    int32_t warps = 1;
    while (2 * warps <= kGEMV_MAX_WARPS_PER_CHANNEL && 2 * warps * q8_0::kWarpSize <= blocksPerRow)
    {
        warps *= 2;
    }
    return warps;
}

int32_t gemvTokens(int32_t rows)
{
    int32_t tokens = 1;
    while (tokens < rows && tokens < kGEMV_MAX_TOKENS)
    {
        tokens *= 2;
    }
    return tokens;
}

int32_t gemvChannels(int32_t rows, int32_t outFeatures, int32_t warpsPerChannel, int32_t smCount)
{
    int32_t channels = kGEMV_MAX_CHANNELS;
    auto const channelsPerCta = [&](int32_t c) { return q8_0::kWarpsPerCta / warpsPerChannel * c; };
    auto const ctas
        = [&](int32_t c) { return (static_cast<int64_t>(outFeatures) + channelsPerCta(c) - 1) / channelsPerCta(c); };
    while (rows == 1 && channels > 1 && channelsPerCta(channels / 2) >= kGEMV_MIN_CHANNELS_PER_CTA
        && ctas(channels) < static_cast<int64_t>(smCount) * kGEMV_CTAS_PER_SM)
    {
        channels /= 2;
    }
    return channels;
}

int32_t mmaWarpsPerChannelTile(int32_t channelTiles, int32_t pairs, int32_t smCount)
{
    int32_t warps = 1;
    while (warps < kMMA_MAX_WARPS_PER_CHANNEL_TILE && 2 * warps <= pairs
        && (static_cast<int64_t>(channelTiles) * warps < static_cast<int64_t>(smCount) * kMMA_WARPS_PER_SM
            || pairs > warps * kMMA_MAX_PAIRS_PER_WARP))
    {
        warps *= 2;
    }
    return warps;
}

template <typename TOut>
struct GemmArgs
{
    int8_t const* codes;
    float const* scales;
    q8_0::WeightSet weights;
    TOut* output;
    int32_t rows;
    int32_t outFeatures;
    int32_t blocksPerRow;
    cudaStream_t stream;
};

template <int32_t kTokens, int32_t kChannels, int32_t kWarpsPerChannel, typename TOut>
void launchGemv(GemmArgs<TOut> const& args)
{
    constexpr int32_t kChannelsPerCta = q8_0::kWarpsPerCta / kWarpsPerChannel * kChannels;
    dim3 const grid((args.outFeatures + kChannelsPerCta - 1) / kChannelsPerCta, (args.rows + kTokens - 1) / kTokens);
    q8_0::packedGemv<kTokens, kChannels, kWarpsPerChannel, TOut>
        <<<grid, dim3(q8_0::kWarpSize, q8_0::kWarpsPerCta), 0, args.stream>>>(
            args.codes, args.scales, args.weights, args.output, args.rows, args.outFeatures, args.blocksPerRow);
}

template <int32_t kTokens, int32_t kChannels, typename TOut>
void launchGemvByWarps(int32_t warpsPerChannel, GemmArgs<TOut> const& args)
{
    switch (warpsPerChannel)
    {
    case 1: launchGemv<kTokens, kChannels, 1>(args); break;
    case 2: launchGemv<kTokens, kChannels, 2>(args); break;
    default: launchGemv<kTokens, kChannels, 4>(args); break;
    }
}

template <int32_t kTokens, typename TOut>
void launchGemvByChannels(int32_t channels, int32_t warpsPerChannel, GemmArgs<TOut> const& args)
{
    switch (channels)
    {
    case 1: launchGemvByWarps<kTokens, 1>(warpsPerChannel, args); break;
    case 2: launchGemvByWarps<kTokens, 2>(warpsPerChannel, args); break;
    default: launchGemvByWarps<kTokens, 4>(warpsPerChannel, args); break;
    }
}

template <int32_t kTokenTiles, int32_t kWarpsPerChannelTile, typename TOut>
void launchMma(GemmArgs<TOut> const& args)
{
    constexpr int32_t kCtaWarps = kWarpsPerChannelTile > q8_0::kWarpsPerCta ? kWarpsPerChannelTile : q8_0::kWarpsPerCta;
    constexpr int32_t kChannelsPerCta = kCtaWarps / kWarpsPerChannelTile * q8_0::kChannelsPerTile;
    dim3 const grid((args.outFeatures + kChannelsPerCta - 1) / kChannelsPerCta,
        (q8_0::tokenTiles(args.rows) + kTokenTiles - 1) / kTokenTiles);
    q8_0::packedGemmMma<kTokenTiles, kWarpsPerChannelTile, TOut>
        <<<grid, dim3(q8_0::kWarpSize, kCtaWarps), 0, args.stream>>>(
            args.codes, args.scales, args.weights, args.output, args.rows, args.outFeatures, args.blocksPerRow);
}

template <int32_t kTokenTiles, typename TOut>
void launchMmaByWarps(int32_t warpsPerChannelTile, GemmArgs<TOut> const& args)
{
    switch (warpsPerChannelTile)
    {
    case 1: launchMma<kTokenTiles, 1>(args); break;
    case 2: launchMma<kTokenTiles, 2>(args); break;
    case 4: launchMma<kTokenTiles, 4>(args); break;
    case 8: launchMma<kTokenTiles, 8>(args); break;
    case 16: launchMma<kTokenTiles, 16>(args); break;
    default: launchMma<kTokenTiles, 32>(args); break;
    }
}

bool tiledEligible(int32_t rows, int32_t blocksPerRow, DeviceTraits const& device)
{
    return rows >= q8_0::kTiledMinRows && blocksPerRow % q8_0::kTiledBlocks == 0
        && device.computeMajor >= kMMA_MIN_COMPUTE_MAJOR;
}

enum class Kernel
{
    kGemv,
    kMma,
    kTiled,
};

template <typename TOut>
cudaError_t gemm(Kernel kernel, GemmArgs<TOut> const& args, DeviceTraits const& device)
{
    if (kernel == Kernel::kTiled)
    {
        cudaError_t const status = cudaFuncSetAttribute(
            q8_0::packedGemmTiled<TOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, q8_0::kTiledSmemBytes);
        if (status != cudaSuccess)
        {
            return status;
        }
        dim3 const grid((args.outFeatures + q8_0::kTiledChannels - 1) / q8_0::kTiledChannels,
            (args.rows + q8_0::kTiledTokens - 1) / q8_0::kTiledTokens);
        q8_0::packedGemmTiled<TOut><<<grid, q8_0::kTiledThreads, q8_0::kTiledSmemBytes, args.stream>>>(
            args.codes, args.scales, args.weights, args.output, args.rows, args.outFeatures, args.blocksPerRow);
        return cudaGetLastError();
    }
    if (kernel == Kernel::kMma)
    {
        int32_t const channelTiles = (args.outFeatures + q8_0::kChannelsPerTile - 1) / q8_0::kChannelsPerTile;
        int32_t const warps = mmaWarpsPerChannelTile(channelTiles, args.blocksPerRow / 2, device.smCount);
        if (q8_0::tokenTiles(args.rows) == 1)
        {
            launchMmaByWarps<1>(warps, args);
        }
        else
        {
            launchMmaByWarps<2>(warps, args);
        }
        return cudaGetLastError();
    }
    int32_t const warpsPerChannel = gemvWarpsPerChannel(args.blocksPerRow);
    int32_t const channels = gemvChannels(args.rows, args.outFeatures, warpsPerChannel, device.smCount);
    switch (gemvTokens(args.rows))
    {
    case 1: launchGemvByChannels<1>(channels, warpsPerChannel, args); break;
    case 2: launchGemvByChannels<2>(channels, warpsPerChannel, args); break;
    case 4: launchGemvByChannels<4>(channels, warpsPerChannel, args); break;
    case 8: launchGemvByChannels<8>(channels, warpsPerChannel, args); break;
    default: launchGemvByChannels<kGEMV_MAX_TOKENS>(channels, warpsPerChannel, args); break;
    }
    return cudaGetLastError();
}

template <typename TOut>
cudaError_t gemm(Kernel kernel, int8_t const* codes, float const* scales, q8_0::WeightSet const& weights, void* output,
    int32_t rows, int32_t blocksPerRow, DeviceTraits const& device, cudaStream_t stream)
{
    GemmArgs<TOut> const args{codes, scales, weights, static_cast<TOut*>(output), rows,
        weights.columnEnd[weights.count - 1], blocksPerRow, stream};
    return gemm(kernel, args, device);
}

template <typename TIn>
cudaError_t quantize(
    void const* input, int8_t* codes, float* scales, int32_t rows, int32_t blocksPerRow, cudaStream_t stream)
{
    // Rows past `rows` in the last token tile are never quantized: they only feed mma columns that are not stored.
    dim3 const grid((blocksPerRow + q8_0::kWarpsPerCta - 1) / q8_0::kWarpsPerCta, rows);
    q8_0::quantizeActivationFragments<TIn><<<grid, dim3(q8_0::kWarpSize, q8_0::kWarpsPerCta), 0, stream>>>(
        static_cast<TIn const*>(input), codes, scales, rows, blocksPerRow);
    return cudaGetLastError();
}
} // namespace

cudaError_t q8_0GemvMaxRows(int32_t outFeatures, int32_t& maxRows)
{
    DeviceTraits device{};
    cudaError_t const status = deviceTraits(device);
    if (status == cudaSuccess)
    {
        maxRows = gemvMaxRows(device, outFeatures);
    }
    return status;
}

size_t q8_0WorkspaceSize(int32_t rows, int32_t inFeatures)
{
    return static_cast<size_t>(q8_0::tokenTiles(rows)) * q8_0::kTokensPerTile * (inFeatures / q8_0::kBlockSize)
        * q8_0::kActivationBytesPerBlock;
}

cudaError_t q8_0Linear(void const* input, DataType inputType, void const* const* packedWeights,
    int32_t const* outFeatures, int32_t numWeights, void* output, DataType outputType, void* workspace,
    size_t workspaceSize, int32_t rows, int32_t inFeatures, cudaStream_t stream)
{
    return q8_0LinearForPath(input, inputType, packedWeights, outFeatures, numWeights, output, outputType, workspace,
        workspaceSize, rows, inFeatures, Q8_0Path::kAuto, stream);
}

cudaError_t q8_0LinearForPath(void const* input, DataType inputType, void const* const* packedWeights,
    int32_t const* outFeatures, int32_t numWeights, void* output, DataType outputType, void* workspace,
    size_t workspaceSize, int32_t rows, int32_t inFeatures, Q8_0Path path, cudaStream_t stream)
{
    if (input == nullptr || packedWeights == nullptr || outFeatures == nullptr || numWeights <= 0
        || numWeights > kQ8_0_MAX_WEIGHTS || output == nullptr || workspace == nullptr || rows <= 0 || inFeatures <= 0
        || inFeatures % q8_0::kBlockSize != 0 || workspaceSize < q8_0WorkspaceSize(rows, inFeatures)
        || reinterpret_cast<uintptr_t>(workspace) % kWORKSPACE_ALIGNMENT != 0
        || (inputType != DataType::kHALF && inputType != DataType::kBF16)
        || (outputType != DataType::kHALF && outputType != DataType::kBF16 && outputType != DataType::kFLOAT))
    {
        return cudaErrorInvalidValue;
    }
    static_assert(kQ8_0_MAX_WEIGHTS == q8_0::kMaxWeights);
    q8_0::WeightSet weights{};
    weights.count = numWeights;
    // A pair of 34-byte blocks is 4-byte aligned only for a 4-byte aligned base.
    bool weightsWordAligned = true;
    int64_t columns = 0;
    for (int32_t w = 0; w < numWeights; ++w)
    {
        if (packedWeights[w] == nullptr || outFeatures[w] <= 0)
        {
            return cudaErrorInvalidValue;
        }
        columns += outFeatures[w];
        if (columns > std::numeric_limits<int32_t>::max())
        {
            return cudaErrorInvalidValue;
        }
        weights.weights[w] = static_cast<q8_0::PackedWeightBlock const*>(packedWeights[w]);
        weights.columnEnd[w] = static_cast<int32_t>(columns);
        weightsWordAligned
            = weightsWordAligned && reinterpret_cast<uintptr_t>(packedWeights[w]) % sizeof(uint32_t) == 0;
    }
    DeviceTraits device{};
    cudaError_t status = deviceTraits(device);
    if (status != cudaSuccess)
    {
        return status;
    }

    int32_t const totalColumns = static_cast<int32_t>(columns);
    int32_t const blocksPerRow = inFeatures / q8_0::kBlockSize;
    bool const mmaEligible
        = device.computeMajor >= kMMA_MIN_COMPUTE_MAJOR && blocksPerRow % 2 == 0 && weightsWordAligned;
    bool const tiled = tiledEligible(rows, blocksPerRow, device);
    Kernel kernel = Kernel::kGemv;
    switch (path)
    {
    case Q8_0Path::kGemv: break;
    case Q8_0Path::kMma:
        if (!mmaEligible)
        {
            return cudaErrorNotSupported;
        }
        kernel = Kernel::kMma;
        break;
    case Q8_0Path::kTiled:
        if (!tiled)
        {
            return cudaErrorNotSupported;
        }
        kernel = Kernel::kTiled;
        break;
    case Q8_0Path::kAuto:
        if (tiled)
        {
            kernel = Kernel::kTiled;
        }
        else if (mmaEligible && rows > gemvMaxRows(device, totalColumns))
        {
            kernel = Kernel::kMma;
        }
        break;
    }

    auto* codes = static_cast<int8_t*>(workspace);
    auto* scales = reinterpret_cast<float*>(codes + q8_0::activationCodeBytes(rows, blocksPerRow));
    status = inputType == DataType::kHALF ? quantize<half>(input, codes, scales, rows, blocksPerRow, stream)
                                          : quantize<__nv_bfloat16>(input, codes, scales, rows, blocksPerRow, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    switch (outputType)
    {
    case DataType::kHALF: return gemm<half>(kernel, codes, scales, weights, output, rows, blocksPerRow, device, stream);
    case DataType::kBF16:
        return gemm<__nv_bfloat16>(kernel, codes, scales, weights, output, rows, blocksPerRow, device, stream);
    default: return gemm<float>(kernel, codes, scales, weights, output, rows, blocksPerRow, device, stream);
    }
}
} // namespace trt_edgellm::kernel
