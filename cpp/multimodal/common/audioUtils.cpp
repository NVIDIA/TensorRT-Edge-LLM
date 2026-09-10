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

#include "multimodal/common/audioUtils.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/preprocessKernels/audioFbankKernels.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

bool uploadHostMelFp32ToFp16Gpu(
    rt::Tensor const& hostMel, rt::Tensor& devOut, cudaStream_t stream, std::string const& debugName)
{
    Coords const& hsh = hostMel.getShape();
    if (hsh.getNumDims() != 2)
    {
        LOG_ERROR("uploadHostMelFp32ToFp16Gpu: expected 2-D host mel, got %d-D", hsh.getNumDims());
        return false;
    }
    int64_t const numel = hsh.volume();
    devOut = rt::Tensor({1, hsh[0], hsh[1]}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, debugName);
    std::vector<__half> halfBuf(static_cast<size_t>(numel));
    float const* srcF32 = hostMel.dataPointer<float>();
    for (int64_t i = 0; i < numel; ++i)
    {
        halfBuf[i] = __float2half(srcF32[i]);
    }
    CUDA_CHECK(cudaMemcpyAsync(devOut.rawPointer(), halfBuf.data(), static_cast<size_t>(numel) * sizeof(__half),
        cudaMemcpyHostToDevice, stream));
    return true;
}

bool uploadHostPcmF32ToGpu(std::vector<float> const& hostPcm, rt::Tensor& devOut, cudaStream_t stream)
{
    if (hostPcm.empty())
    {
        LOG_ERROR("uploadHostPcmF32ToGpu: empty PCM.");
        return false;
    }
    int64_t const numSamples = static_cast<int64_t>(hostPcm.size());
    if (!devOut.reshape({numSamples}))
    {
        LOG_ERROR("uploadHostPcmF32ToGpu: PCM (%ld samples) exceeds the pre-allocated staging capacity.",
            static_cast<long>(numSamples));
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(devOut.rawPointer(), hostPcm.data(), static_cast<size_t>(numSamples) * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    return true;
}

bool fillMelFilterFp16Kmajor(
    float const* melFilterF32, int32_t nMel, int32_t nFreq, int32_t kPad, rt::Tensor& out, cudaStream_t stream)
{
    if (melFilterF32 == nullptr || nMel <= 0 || nFreq <= 0 || nFreq > kPad)
    {
        LOG_ERROR("fillMelFilterFp16Kmajor: bad mel filter (nMel=%d, nFreq=%d, kPad=%d).", nMel, nFreq, kPad);
        return false;
    }
    Coords const& osh = out.getShape();
    if (out.getDeviceType() != rt::DeviceType::kGPU || out.getDataType() != nvinfer1::DataType::kHALF
        || osh.getNumDims() != 2 || osh[0] != nMel || osh[1] != kPad)
    {
        LOG_ERROR("fillMelFilterFp16Kmajor: out must be a pre-allocated [%d, %d] Half GPU tensor.", nMel, kPad);
        return false;
    }
    // Cast + K-pad [nMel, nFreq] F32 → [nMel, kPad] F16 (row-major, K
    // contiguous), matching the AOT CuTe DSL GEMM A-matrix ABI. Trailing
    // kPad − nFreq cols stay zero; the matching cols in mag (GEMM B) are
    // zeroed by fbankWhisper's memset, so the dot product is invariant on the
    // active [nMel, T_out] window.
    std::vector<__half> melFp16Host(static_cast<size_t>(nMel) * kPad, __half(0.0f));
    for (int32_t m = 0; m < nMel; ++m)
    {
        for (int32_t k = 0; k < nFreq; ++k)
        {
            melFp16Host[static_cast<size_t>(m) * kPad + k]
                = __float2half_rn(melFilterF32[static_cast<size_t>(m) * nFreq + k]);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        out.rawPointer(), melFp16Host.data(), melFp16Host.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
    return true;
}

void makeFftTwiddleHost(int32_t nFft, std::vector<float>& twoChan)
{
    constexpr double kTwoPiNeg = -2.0 * 3.14159265358979323846;
    twoChan.resize(static_cast<size_t>(nFft) * 2);
    for (int32_t k = 0; k < nFft; ++k)
    {
        double const ang = kTwoPiNeg * static_cast<double>(k) / static_cast<double>(nFft);
        twoChan[2 * k + 0] = static_cast<float>(std::cos(ang));
        twoChan[2 * k + 1] = static_cast<float>(std::sin(ang));
    }
}

void makeCentredWindowHost(std::vector<float> const& window, int32_t nFft, std::vector<float>& out)
{
    int32_t const winLength = static_cast<int32_t>(window.size());
    out.assign(static_cast<size_t>(nFft), 0.0f);
    int32_t const offset = (nFft - winLength) / 2;
    for (int32_t i = 0; i < winLength; ++i)
    {
        out[static_cast<size_t>(offset + i)] = window[static_cast<size_t>(i)];
    }
}

int64_t computeFeatExtractOutputLength(int64_t inputLength, int32_t nWindow, int32_t numConvStages)
{
    check::check(numConvStages >= 1, "numConvStages must be >= 1, got: " + std::to_string(numConvStages));

    // Floor division that always rounds toward negative infinity, matching Python's "//" operator.
    auto floorDiv = [](int64_t a, int64_t b) -> int64_t {
        int64_t q = a / b;
        // Adjust if the remainder is nonzero and the signs of a and b differ
        if ((a % b != 0) && ((a ^ b) < 0))
        {
            --q;
        }
        return q;
    };

    // Each stride-2 conv applies ``out = (in - 1) // 2 + 1`` along the time dimension.
    // Compose ``numConvStages`` of them. Lambda kept local so the recurrence stays adjacent
    // to its only call sites below.
    auto downsampleN = [&floorDiv, numConvStages](int64_t length) -> int64_t {
        for (int32_t i = 0; i < numConvStages; ++i)
        {
            length = floorDiv(length - 1, 2) + 1;
        }
        return length;
    };

    // Chunk size = nWindow * 2, matching the chunk size used in computeChunkInfo.
    int64_t const chunkSize = nWindow * 2;

    // The encoder runs ``numConvStages`` stride-2 Conv2D layers along time.
    // Qwen3-Next Omni's 4 stages give 7 for the same chunkSize=100.
    int64_t const tailLen = downsampleN(inputLength % chunkSize);
    int64_t const perChunkLen = downsampleN(chunkSize);
    return tailLen + floorDiv(inputLength, chunkSize) * perChunkLen;
}

ChunkInfo computeChunkInfo(int64_t featureLength, int32_t nWindow)
{
    ChunkInfo info;
    // BUGFIX: Original Qwen3-Omni uses n_window * 2 for chunking (see modeling_qwen3_omni.py line 1145)
    // chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
    // chunk_lengths = torch.tensor([self.n_window * 2] * chunk_num.sum(), ...)
    int64_t chunkSize = nWindow * 2; // Must be n_window * 2, not n_window
    info.numChunks = (featureLength + chunkSize - 1) / chunkSize;
    info.chunkLengths.resize(info.numChunks, chunkSize);
    info.chunkOffsets.resize(info.numChunks);

    for (int64_t i = 0; i < info.numChunks; ++i)
    {
        info.chunkOffsets[i] = i * chunkSize;
    }

    int64_t remainder = featureLength % chunkSize;
    if (remainder != 0)
    {
        info.chunkLengths[info.numChunks - 1] = remainder;
    }

    info.maxChunkLength = std::max(*std::max_element(info.chunkLengths.begin(), info.chunkLengths.end()), chunkSize);
    return info;
}

bool chunkAndPadFeatures(
    rt::Tensor const& melSpectrogram, ChunkInfo const& chunkInfo, rt::Tensor& paddedFeature, cudaStream_t stream)
{
    auto const& inputShape = melSpectrogram.getShape();
    if (inputShape.getNumDims() != 3 || inputShape[0] != 1)
    {
        LOG_ERROR("Invalid input shape, expected [1, mel_bins, time_steps]");
        return false;
    }

    // Input: [1, mel_bins, time_steps]
    int64_t melBins = inputShape[1];
    int64_t timeSteps = inputShape[2];

    // Output: [num_chunks, mel_bins, max_chunk_length]
    int64_t outputSize = chunkInfo.numChunks * melBins * chunkInfo.maxChunkLength;

    if (paddedFeature.getShape().volume() < outputSize)
    {
        // Allocate as FP16 (Half precision)
        paddedFeature = rt::Tensor(
            {chunkInfo.numChunks, melBins, chunkInfo.maxChunkLength}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    }
    else
    {
        check::check(paddedFeature.reshape({chunkInfo.numChunks, melBins, chunkInfo.maxChunkLength}),
            "Failed to reshape paddedFeature");
    }

    // Zero out padding
    CUDA_CHECK(
        cudaMemsetAsync(paddedFeature.rawPointer(), 0, paddedFeature.getShape().volume() * sizeof(__half), stream));

    __half const* srcPtr = melSpectrogram.dataPointer<__half>();

    // Copy chunks (both input and output are FP16)
    for (int64_t chunkIdx = 0; chunkIdx < chunkInfo.numChunks; ++chunkIdx)
    {
        int64_t chunkOffset = chunkInfo.chunkOffsets[chunkIdx];
        int64_t chunkLength = chunkInfo.chunkLengths[chunkIdx];

        if (chunkOffset + chunkLength > timeSteps)
        {
            chunkLength = timeSteps - chunkOffset;
        }

        // Copy chunk data (FP16 device-to-device)
        __half* dstPtr = paddedFeature.dataPointer<__half>();
        for (int64_t melIdx = 0; melIdx < melBins; ++melIdx)
        {
            size_t srcOffset = melIdx * timeSteps + chunkOffset;
            size_t dstOffset = chunkIdx * melBins * chunkInfo.maxChunkLength + melIdx * chunkInfo.maxChunkLength;

            CUDA_CHECK(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, chunkLength * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    return true;
}

bool createPaddedMask(ChunkInfo const& chunkInfo, [[maybe_unused]] int32_t nWindow, rt::Tensor& paddedMask,
    std::vector<int64_t>& afterCNNLens, cudaStream_t stream, int32_t numConvStages)
{
    // Compute aftercnn_lens for each chunk
    afterCNNLens.clear();
    afterCNNLens.reserve(chunkInfo.numChunks);

    int64_t maxLenAfterCNN = 0;
    for (int64_t i = 0; i < chunkInfo.numChunks; ++i)
    {
        int64_t lenAfterCNN = computeFeatExtractOutputLength(chunkInfo.chunkLengths[i], nWindow, numConvStages);
        afterCNNLens.push_back(lenAfterCNN);
        maxLenAfterCNN = std::max(maxLenAfterCNN, lenAfterCNN);
    }

    // Output: [num_chunks, max_len_after_cnn]
    int64_t outputSize = chunkInfo.numChunks * maxLenAfterCNN;
    if (paddedMask.getShape().volume() < outputSize)
    {
        // Reallocate if needed
        paddedMask
            = rt::Tensor({chunkInfo.numChunks, maxLenAfterCNN}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    }
    else
    {
        check::check(paddedMask.reshape({chunkInfo.numChunks, maxLenAfterCNN}), "Failed to reshape paddedMask");
    }

    std::vector<int64_t> maskHost(chunkInfo.numChunks * maxLenAfterCNN, 0);

    // For each chunk, mark valid positions after CNN
    for (int64_t chunkIdx = 0; chunkIdx < chunkInfo.numChunks; ++chunkIdx)
    {
        int64_t validLength = afterCNNLens[chunkIdx];

        for (int64_t i = 0; i < validLength; ++i)
        {
            maskHost[chunkIdx * maxLenAfterCNN + i] = 1;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        paddedMask.rawPointer(), maskHost.data(), maskHost.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    return true;
}

bool preprocessAudioForEncoder(rt::Tensor const& melSpectrogram, int32_t nWindow, rt::Tensor& paddedFeature,
    rt::Tensor& paddedMaskAfterCNN, std::vector<int64_t>& afterCNNLens, cudaStream_t stream, int32_t numConvStages)
{
    auto const& inputShape = melSpectrogram.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("Invalid input shape, expected [1, mel_bins, time_steps]");
        return false;
    }

    // Input shape: [1, mel_bins, time_steps]
    int64_t timeSteps = inputShape[2];

    ChunkInfo chunkInfo = computeChunkInfo(timeSteps, nWindow);

    if (!chunkAndPadFeatures(melSpectrogram, chunkInfo, paddedFeature, stream))
    {
        LOG_ERROR("Failed to chunk and pad features");
        return false;
    }

    if (!createPaddedMask(chunkInfo, nWindow, paddedMaskAfterCNN, afterCNNLens, stream, numConvStages))
    {
        LOG_ERROR("Failed to create padded mask");
        return false;
    }

    return true;
}

bool convertMaskToIndices(rt::Tensor const& paddedMask, rt::Tensor& paddedMaskIndices, cudaStream_t stream)
{
    // Validate input shape
    auto const& maskShape = paddedMask.getShape();
    if (maskShape.getNumDims() != 2)
    {
        LOG_ERROR("Expected 2D mask, got %d dimensions", maskShape.getNumDims());
        return false;
    }

    int64_t numChunks = maskShape[0];
    int64_t maxLen = maskShape[1];
    int64_t totalElements = numChunks * maxLen;

    // Step 1: Copy mask from device to host
    std::vector<int64_t> maskHost(totalElements);
    CUDA_CHECK(cudaMemcpyAsync(
        maskHost.data(), paddedMask.rawPointer(), totalElements * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 2: Find all nonzero positions (equivalent to torch.nonzero)
    std::vector<int64_t> indicesFlat;
    indicesFlat.reserve(totalElements * 2); // Pre-allocate maximum possible size

    for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx)
    {
        for (int64_t posIdx = 0; posIdx < maxLen; ++posIdx)
        {
            int64_t linearIdx = chunkIdx * maxLen + posIdx;
            if (maskHost[linearIdx] != 0)
            {
                indicesFlat.push_back(chunkIdx); // First dimension index
                indicesFlat.push_back(posIdx);   // Second dimension index
            }
        }
    }

    // Step 3: Create output tensor [num_valid, 2]
    int64_t numValid = indicesFlat.size() / 2;

    if (numValid == 0)
    {
        LOG_WARNING("No valid elements in mask! Creating empty indices tensor");
        check::check(paddedMaskIndices.reshape({0, 2}), "Failed to reshape paddedMaskIndices");
        return true;
    }

    // Allocate or reshape output tensor
    if (paddedMaskIndices.getShape().volume() < numValid * 2)
    {
        paddedMaskIndices = rt::Tensor({numValid, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    }
    else
    {
        check::check(paddedMaskIndices.reshape({numValid, 2}), "Failed to reshape paddedMaskIndices");
    }

    // Step 4: Copy indices from host to device
    CUDA_CHECK(cudaMemcpyAsync(paddedMaskIndices.rawPointer(), indicesFlat.data(), indicesFlat.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    return true;
}

bool createChunkwiseAttentionMask(std::vector<int64_t> const& afterCNNLens, int32_t nWindow, int32_t nWindowInfer,
    rt::Tensor& attentionMask, cudaStream_t stream)
{
    // Matches modeling_qwen3_omni.py _prepare_attention_mask + cu_seqlens logic:
    //   window_aftercnn = padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window * 2))
    //   cu_seqlens built from aftercnn_lens split by window_aftercnn
    //   Block-diagonal mask: tokens within the same window attend to each other.
    int64_t const chunkSize = static_cast<int64_t>(nWindow) * 2;
    int64_t maxLenAfterCNN = 0;
    for (auto len : afterCNNLens)
    {
        maxLenAfterCNN = std::max(maxLenAfterCNN, len);
    }
    int64_t const windowAfterCNN = maxLenAfterCNN * (nWindowInfer / chunkSize);

    // Global after-CNN length = sum of per-chunk after-CNN lengths
    int64_t globalAfterCNNLen = 0;
    for (auto len : afterCNNLens)
    {
        globalAfterCNNLen += len;
    }

    // Build window lengths from global aftercnn_lens
    std::vector<int64_t> windowLens;
    int64_t numFull = globalAfterCNNLen / windowAfterCNN;
    for (int64_t i = 0; i < numFull; ++i)
    {
        windowLens.push_back(windowAfterCNN);
    }
    int64_t remainder = globalAfterCNNLen % windowAfterCNN;
    if (remainder != 0)
    {
        windowLens.push_back(remainder);
    }

    int64_t totalLen = 0;
    for (auto len : windowLens)
    {
        totalLen += len;
    }

    if (totalLen == 0)
    {
        LOG_ERROR("Total attention length is 0");
        return false;
    }

    // Allocate or reshape mask tensor [total_len, total_len]
    if (attentionMask.getShape().volume() < totalLen * totalLen)
    {
        attentionMask = rt::Tensor({totalLen, totalLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    }
    else
    {
        check::check(attentionMask.reshape({totalLen, totalLen}), "Failed to reshape attentionMask");
    }

    constexpr float NEG_INF = -65504.0f;
    std::vector<__half> maskHost(totalLen * totalLen);
    for (int64_t i = 0; i < totalLen * totalLen; ++i)
    {
        maskHost[i] = __float2half(NEG_INF);
    }

    // Set block-diagonal regions to 0 (allow attention within each window)
    int64_t offset = 0;
    for (size_t winIdx = 0; winIdx < windowLens.size(); ++winIdx)
    {
        int64_t winLen = windowLens[winIdx];
        for (int64_t i = 0; i < winLen; ++i)
        {
            for (int64_t j = 0; j < winLen; ++j)
            {
                maskHost[(offset + i) * totalLen + (offset + j)] = __float2half(0.0f);
            }
        }
        offset += winLen;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        attentionMask.rawPointer(), maskHost.data(), maskHost.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));

    return true;
}

int32_t computeNumMelFrames(int64_t numPcmSamples, int32_t nFft, int32_t hopLength, int32_t padLength)
{
    int64_t const nPadded = numPcmSamples + 2 * padLength;
    int32_t const T_full = static_cast<int32_t>((nPadded - nFft) / hopLength + 1);
    return T_full - 1;
}

bool fbankWhisper(rt::Tensor const& pcmF32, FbankResources& resources, rt::Tensor& melOutF16, cudaStream_t stream)
{
    try
    {
        if (pcmF32.getShape().getNumDims() != 1)
        {
            LOG_ERROR("fbankWhisper: pcmF32 must be [N] mono FP32; got %d dims.", pcmF32.getShape().getNumDims());
            return false;
        }
        // STFT/mel params from the runner-populated resources (derived from
        // MelExtractorConfig — single source of truth, no constants here).
        int32_t const nFft = resources.nFft;
        int32_t const hopLength = resources.hopLength;
        int32_t const padLength = resources.padLength;
        int32_t const nMel = resources.nMel;
        float const melFloor = resources.melFloor;

        int64_t const N = pcmF32.getShape()[0];
        int64_t const nPadded = N + 2 * padLength;
        if (nPadded < nFft)
        {
            LOG_ERROR(
                "fbankWhisper: PCM too short for fbank (N=%ld, need >= %d after pad).", static_cast<long>(N), nFft);
            return false;
        }
        // Reuse computeNumMelFrames — the same call the runner uses to size
        // melSpec — so the kernel output width cannot drift from the
        // pre-allocated melSpec. T_full = T_out + 1 restores the trailing frame
        // that the HF `stft[..., :-1]` trim drops.
        int32_t const T_out = computeNumMelFrames(N, nFft, hopLength, padLength);
        if (T_out <= 0)
        {
            LOG_ERROR("fbankWhisper: T_out non-positive (N=%ld).", static_cast<long>(N));
            return false;
        }
        int32_t const T_full = T_out + 1;
        int32_t const N_pad = fbankNPad(T_out);

        // Metadata-only reshape of the pre-allocated workspace to this clip's
        // size; maxLogScalar stays [1] and is not reshaped.
        check::check(resources.framedF32.reshape({static_cast<int64_t>(T_full), nFft}),
            "fbankWhisper: framedF32 exceeds pre-allocated capacity.");
        check::check(resources.magFp16.reshape({static_cast<int64_t>(N_pad), kFbankKPad}),
            "fbankWhisper: magFp16 exceeds pre-allocated capacity.");
        // melPowerFp16 (GEMM C) needs no pre-zeroing: the downstream
        // log10MaxReduce / logMelNormalize kernels bound-check t < T_out and
        // never read the padding columns [T_out, N_pad).
        check::check(resources.melPowerFp16.reshape({nMel, static_cast<int64_t>(N_pad)}),
            "fbankWhisper: melPowerFp16 exceeds pre-allocated capacity.");

        // Zero the whole padded mag buffer. Stage 4 of stftR2C400FusedMagsq
        // writes only [0, T_out) × [0, nFreq); rows [T_out, N_pad) and cols
        // [nFreq, K_pad) must stay zero so the AOT GEMM (no residue handling)
        // produces correct values on the [nMel, T_out] active subset (the
        // zero rows/cols sum to 0 in the dot product).
        CUDA_CHECK(cudaMemsetAsync(
            resources.magFp16.rawPointer(), 0, static_cast<size_t>(N_pad) * kFbankKPad * sizeof(__half), stream));

        // Pipeline. Each wrapper validates its tensor contracts via check::check
        // and throws on mismatch — the surrounding try/catch turns that into a
        // logged false return.
        kernel::pcmToFramesAndWindow(
            pcmF32, resources.hannWindow, resources.framedF32, nFft, hopLength, padLength, stream);
        kernel::stftR2C400FusedMagsq(
            resources.framedF32, resources.fftTwiddle, resources.magFp16, T_out, kFbankKPad, stream);
        kernel::melLinearGemmFp16TC(resources.melFilterFp16Kmajor, resources.magFp16, resources.melPowerFp16, stream);
        kernel::log10MaxReduce(resources.melPowerFp16, T_out, resources.maxLogScalar, melFloor, stream);
        kernel::logMelNormalizeAndCastF16(
            resources.melPowerFp16, T_out, resources.maxLogScalar, melOutF16, melFloor, stream);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("fbankWhisper failed: %s", e.what());
        return false;
    }
}

int32_t computeNumMelFramesParakeet(int64_t numPcmSamples, int32_t hopLength)
{
    if (numPcmSamples <= 0 || hopLength <= 0)
    {
        return 0;
    }
    // Plain floor — HF ParakeetFeatureExtractor features length (center=True).
    return static_cast<int32_t>(numPcmSamples / hopLength);
}

bool fbankParakeet(
    rt::Tensor const& pcmF32, FbankResourcesParakeet& resources, rt::Tensor& melOutF16, cudaStream_t stream)
{
    try
    {
        if (pcmF32.getShape().getNumDims() != 1)
        {
            LOG_ERROR("fbankParakeet: pcmF32 must be [N] mono FP32; got %d dims.", pcmF32.getShape().getNumDims());
            return false;
        }
        // STFT/mel params from the runner-populated resources (derived from
        // MelExtractorConfig, except normEps == kParakeetZScoreEps).
        int32_t const nFft = resources.nFft;
        int32_t const hopLength = resources.hopLength;
        int32_t const centerPad = resources.centerPad;
        int32_t const nMel = resources.nMel;
        float const preemph = resources.preemph;
        float const logGuard = resources.logGuard;
        float const normEps = resources.normEps;

        int64_t const N = pcmF32.getShape()[0];
        // T_out = floor(N / hop) (no Whisper drop-last). Virtual center zero-pad,
        // so the only length requirement is N >= hop (T_out >= 1).
        int32_t const T_out = computeNumMelFramesParakeet(N, hopLength);
        if (T_out <= 0)
        {
            LOG_ERROR("fbankParakeet: T_out non-positive (N=%ld, hop=%d).", static_cast<long>(N), hopLength);
            return false;
        }
        int32_t const N_pad = fbankNPad(T_out);
        int32_t const kPad = fbankKPadParakeet(resources.nFreq);

        // Reshape the init-time pre-allocated workspace to this clip (metadata
        // only — capacity was sized at the maxFrames bound in initFbankResources;
        // the runner gates clip length against maxFrames before calling). framedF32
        // is [T_out, nFft] — parakeet frames exactly T_out (no trailing drop-last
        // frame, unlike Whisper). melPowerF32 (GEMM C) is FP32 — the parakeet
        // deviation from Whisper's FP16 C — and needs no pre-zeroing: the stats /
        // normalize kernels bound-check t < T_out and never read the padding cols
        // [T_out, N_pad). mean / invDenom stay [nMel] for every clip.
        check::check(resources.framedF32.reshape({static_cast<int64_t>(T_out), nFft}),
            "fbankParakeet: framedF32 exceeds pre-allocated capacity.");
        check::check(resources.magFp16.reshape({static_cast<int64_t>(N_pad), kPad}),
            "fbankParakeet: magFp16 exceeds pre-allocated capacity.");
        check::check(resources.melPowerF32.reshape({nMel, static_cast<int64_t>(N_pad)}),
            "fbankParakeet: melPowerF32 exceeds pre-allocated capacity.");

        // No per-clip mag memset. magFp16's K-pad columns [nFreq, K_pad) and unused
        // rows [T_out, N_pad) were zeroed ONCE over the full [nPadMax, K_pad] buffer
        // at init (initFbankResources). stftR2C512FusedMagsq writes only
        // [0, T_out) × [0, nFreq) and never touches the pad region, and the reshape
        // above only shrinks the view (no re-allocation), so those zeros persist
        // across clips — the AOT GEMM (no K-residue handling) stays exact on the
        // [nMel, T_out] active subset without a per-clip memset.

        // Pipeline. Each wrapper validates its tensor contracts via check::check
        // and throws on mismatch — the surrounding try/catch turns that into a
        // logged false return.
        kernel::pcmPreemphFramesAndWindow(
            pcmF32, resources.windowF32, resources.framedF32, nFft, hopLength, centerPad, preemph, T_out, stream);
        kernel::stftR2C512FusedMagsq(resources.framedF32, resources.fftTwiddle, resources.magFp16, T_out, kPad, stream);
        kernel::melLinearGemmFp16inFp32out(
            resources.melFilterFp16Kmajor, resources.magFp16, resources.melPowerF32, stream);
        kernel::melStatsLnPerFeature(
            resources.melPowerF32, T_out, logGuard, normEps, resources.mean, resources.invDenom, stream);
        kernel::melNormalizeZScoreTimeFirst(
            resources.melPowerF32, T_out, logGuard, resources.mean, resources.invDenom, melOutF16, stream);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("fbankParakeet failed: %s", e.what());
        return false;
    }
}

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
