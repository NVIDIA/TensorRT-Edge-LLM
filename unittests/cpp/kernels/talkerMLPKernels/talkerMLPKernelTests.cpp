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

#include "common/checkMacros.h"
#include "common/tensor.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "testUtils.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{

float siluF32(float x)
{
    return x / (1.0f + std::exp(-x));
}

// CPU reference: output = FC2(SiLU(FC1(input) + bias1)) + bias2
// Weight layout: row-major [outDim, inDim] (same as PyTorch Linear.weight)
void referenceTalkerMLP(std::vector<half> const& input, std::vector<half> const& fc1Weight,
    std::vector<half> const& fc1Bias, std::vector<half> const& fc2Weight, std::vector<half> const& fc2Bias,
    std::vector<half>& output, int64_t numTokens, int64_t inputDim, int64_t hiddenDim, int64_t outputDim)
{
    std::vector<float> workspace(numTokens * hiddenDim, 0.0f);
    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t h = 0; h < hiddenDim; ++h)
        {
            float acc = 0.0f;
            for (int64_t k = 0; k < inputDim; ++k)
            {
                acc += __half2float(input[n * inputDim + k]) * __half2float(fc1Weight[h * inputDim + k]);
            }
            acc += __half2float(fc1Bias[h]);
            workspace[n * hiddenDim + h] = siluF32(acc);
        }
    }

    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t o = 0; o < outputDim; ++o)
        {
            float acc = 0.0f;
            for (int64_t h = 0; h < hiddenDim; ++h)
            {
                acc += workspace[n * hiddenDim + h] * __half2float(fc2Weight[o * hiddenDim + h]);
            }
            acc += __half2float(fc2Bias[o]);
            output[n * outputDim + o] = __float2half(acc);
        }
    }
}

// CPU reference for the assistant preamble. halfAdd matches __hadd2 bit-exactly:
// the exact sum of two half values is representable in float, and both paths round
// to half with round-to-nearest-even.
half halfAdd(half a, half b)
{
    return __float2half(__half2float(a) + __half2float(b));
}

//! Builds the expected preamble rows on CPU. languageId < 0 reproduces the historical
//! 8-row layout; languageId >= 0 the 9-row CustomVoice layout.
void referenceAssistantPreamble(std::vector<half> const& projected, std::vector<half> const& ttsPad,
    std::vector<half> const& ttsBos, std::vector<half> const& ttsEos, std::vector<half> const& embTable,
    int32_t codecNothinkId, int32_t codecThinkBosId, int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId,
    int32_t codecBosId, int32_t codecThinkId, int32_t languageId, int32_t textLen, int32_t hiddenDim,
    std::vector<half>& output)
{
    bool const hasLanguage = (languageId >= 0);
    int32_t const prefixLen = 8 + (hasLanguage ? 1 : 0);
    int32_t const totalRows = prefixLen + textLen + 2;
    output.resize(static_cast<size_t>(totalRows) * hiddenDim);

    auto emitCopy = [&](int32_t row, half const* src) {
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            output[static_cast<size_t>(row) * hiddenDim + d] = src[d];
        }
    };
    auto emitAdd = [&](int32_t row, half const* a, int32_t tokenId) {
        half const* b = embTable.data() + static_cast<size_t>(tokenId) * hiddenDim;
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            output[static_cast<size_t>(row) * hiddenDim + d] = halfAdd(a[d], b[d]);
        }
    };

    int32_t row = 0;
    for (; row < 3; ++row)
    {
        emitCopy(row, projected.data() + static_cast<size_t>(row) * hiddenDim);
    }
    emitAdd(row++, ttsPad.data(), hasLanguage ? codecThinkId : codecNothinkId);
    emitAdd(row++, ttsPad.data(), codecThinkBosId);
    if (hasLanguage)
    {
        emitAdd(row++, ttsPad.data(), languageId);
    }
    emitAdd(row++, ttsPad.data(), codecThinkEosId);
    emitAdd(row++, ttsPad.data(), speakerId);
    emitAdd(row++, ttsBos.data(), codecPadId);

    for (int32_t i = 0; i < textLen; ++i)
    {
        half const* textRow = projected.data() + static_cast<size_t>(3 + i) * hiddenDim;
        int32_t const codecId = (i == textLen - 1) ? codecBosId : codecPadId;
        half const* b = embTable.data() + static_cast<size_t>(codecId) * hiddenDim;
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            output[static_cast<size_t>(row) * hiddenDim + d] = halfAdd(textRow[d], b[d]);
        }
        ++row;
    }
    emitAdd(row++, ttsEos.data(), codecPadId);
    emitAdd(row++, ttsPad.data(), codecBosId);
}

} // namespace

// ============================================================================
// Fixture for tests that require CuTe DSL GEMM (invokeTalkerMLP / invokeLinearLayer)
// ============================================================================
class TalkerMLPTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));
#ifndef CUTE_DSL_GEMM_ENABLED
        GTEST_SKIP() << "CuTe DSL GEMM not enabled in this build";
#else
        if (!trt_edgellm::CuteDslGemmRunner::loadKernelModule())
        {
            GTEST_SKIP() << "Failed to load CuTe DSL GEMM kernel module";
        }
#endif
    }

    void TearDown() override
    {
#ifdef CUTE_DSL_GEMM_ENABLED
        trt_edgellm::CuteDslGemmRunner::unloadKernelModule();
#endif
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ============================================================================
// Fixture for tests that only need CUDA (no cuBLAS dependency)
// ============================================================================
class TalkerKernelTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ===== invokeTalkerMLP =====

TEST_F(TalkerMLPTest, MLPAccuracy)
{
    // (numTokens, inputDim, hiddenDim, outputDim)
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> testCases = {
        {1, 64, 64, 32},
        {2, 64, 64, 32},
        {4, 2048, 2048, 1024},
    };

    for (auto const& [numTokens, inputDim, hiddenDim, outputDim] : testCases)
    {
        SCOPED_TRACE("numTokens=" + std::to_string(numTokens) + ", inputDim=" + std::to_string(inputDim)
            + ", hiddenDim=" + std::to_string(hiddenDim) + ", outputDim=" + std::to_string(outputDim));

        // Scale init range down for large dimensions to avoid FP16 overflow in accumulation
        float const initScale = (inputDim > 256) ? 0.1f : 0.5f;

        std::vector<half> hostInput(numTokens * inputDim);
        std::vector<half> hostFc1W(hiddenDim * inputDim);
        std::vector<half> hostFc1B(hiddenDim);
        std::vector<half> hostFc2W(outputDim * hiddenDim);
        std::vector<half> hostFc2B(outputDim);

        uniformFloatInitialization(hostInput, -initScale * 2, initScale * 2);
        uniformFloatInitialization(hostFc1W, -initScale, initScale);
        uniformFloatInitialization(hostFc1B, -0.1f, 0.1f);
        uniformFloatInitialization(hostFc2W, -initScale, initScale);
        uniformFloatInitialization(hostFc2B, -0.1f, 0.1f);

        std::vector<half> refOutput(numTokens * outputDim);
        referenceTalkerMLP(
            hostInput, hostFc1W, hostFc1B, hostFc2W, hostFc2B, refOutput, numTokens, inputDim, hiddenDim, outputDim);

        rt::Coords inputShape{numTokens, inputDim};
        rt::Coords fc1WShape{hiddenDim, inputDim};
        rt::Coords fc1BShape{hiddenDim};
        rt::Coords fc2WShape{outputDim, hiddenDim};
        rt::Coords fc2BShape{outputDim};
        rt::Coords outputShape{numTokens, outputDim};
        rt::Coords workspaceShape{numTokens, hiddenDim};

        rt::Tensor gpuInput(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1W(fc1WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1B(fc1BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2W(fc2WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2B(fc2BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuOutput(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuWorkspace(workspaceShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        copyHostToDevice(gpuInput, hostInput);
        copyHostToDevice(gpuFc1W, hostFc1W);
        copyHostToDevice(gpuFc1B, hostFc1B);
        copyHostToDevice(gpuFc2W, hostFc2W);
        copyHostToDevice(gpuFc2B, hostFc2B);

        kernel::invokeTalkerMLP(gpuInput, gpuFc1W, gpuFc1B, gpuFc2W, gpuFc2B, gpuOutput, gpuWorkspace, stream);

        auto const gpuResult = copyDeviceToHost<half>(gpuOutput);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // For large-dim GEMM, allow a small fraction of outliers due to FP16 accumulation differences
        auto [rtol, atol] = getTolerance<half>();
        int32_t mismatches = 0;
        for (size_t i = 0; i < gpuResult.size(); ++i)
        {
            if (!isclose(gpuResult[i], refOutput[i], rtol, atol))
            {
                ++mismatches;
            }
        }
        EXPECT_LT(mismatches, std::max(1, static_cast<int32_t>(gpuResult.size() / 100)))
            << "Too many mismatches: " << mismatches << " / " << gpuResult.size();
    }
}

// ===== Gather / Scatter =====

TEST_F(TalkerKernelTest, GatherScatterRoundTrip)
{
    int64_t const srcTokens = 8;
    int64_t const hiddenDim = 64;
    int64_t const numIndices = 4;

    std::vector<half> hostSource(srcTokens * hiddenDim);
    uniformFloatInitialization(hostSource, -1.0f, 1.0f);
    std::vector<int32_t> hostIndices = {2, 5, 0, 7};

    rt::Coords srcShape{srcTokens, hiddenDim};
    rt::Coords idxShape{numIndices};
    rt::Coords gatherShape{numIndices, hiddenDim};
    rt::Coords scatterShape{srcTokens, hiddenDim};

    rt::Tensor gpuSource(srcShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuIndices(idxShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor gpuGatherOut(gatherShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuScatterOut(scatterShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(gpuSource, hostSource);
    copyHostToDevice(gpuIndices, hostIndices);
    CUDA_CHECK(cudaMemset(gpuScatterOut.rawPointer(), 0, srcTokens * hiddenDim * sizeof(half)));

    kernel::invokeGather(gpuSource, gpuIndices, gpuGatherOut, stream);
    kernel::invokeScatter(gpuGatherOut, gpuIndices, gpuScatterOut, stream);

    auto const scatterResult = copyDeviceToHost<half>(gpuScatterOut);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t idx = 0; idx < numIndices; ++idx)
    {
        int32_t const srcRow = hostIndices[idx];
        for (int64_t d = 0; d < hiddenDim; ++d)
        {
            EXPECT_TRUE(isclose(scatterResult[srcRow * hiddenDim + d], hostSource[srcRow * hiddenDim + d], 0.f, 0.f))
                << "Gather-scatter round trip mismatch at row " << srcRow << " dim " << d;
        }
    }
}

// ===== AssistantPreamble =====

namespace
{

struct PreambleTokenIds
{
    int32_t skipThink{10}; // codec_nothink_id equivalent
    int32_t think{11};
    int32_t thinkBos{12};
    int32_t thinkEos{13};
    int32_t speaker{14};
    int32_t pad{15};
    int32_t bos{16};
    int32_t language{17};
};

//! Host-side inputs shared by GPU run and CPU reference. Generated once so multiple
//! kernel invocations (e.g. 8-row vs 9-row) can reuse identical data.
struct PreambleInputs
{
    std::vector<half> projected;
    std::vector<half> ttsPad;
    std::vector<half> ttsBos;
    std::vector<half> ttsEos;
    std::vector<half> embTable;
    int32_t textLen{0};
    int32_t hiddenDim{0};
    int64_t vocabSize{32};

    static PreambleInputs makeRandom(int32_t textLen, int32_t hiddenDim)
    {
        PreambleInputs in;
        in.textLen = textLen;
        in.hiddenDim = hiddenDim;
        in.projected.resize(static_cast<size_t>(3 + textLen) * hiddenDim);
        in.ttsPad.resize(hiddenDim);
        in.ttsBos.resize(hiddenDim);
        in.ttsEos.resize(hiddenDim);
        in.embTable.resize(static_cast<size_t>(in.vocabSize) * hiddenDim);
        uniformFloatInitialization(in.projected, -1.0f, 1.0f);
        uniformFloatInitialization(in.ttsPad, -1.0f, 1.0f);
        uniformFloatInitialization(in.ttsBos, -1.0f, 1.0f);
        uniformFloatInitialization(in.ttsEos, -1.0f, 1.0f);
        uniformFloatInitialization(in.embTable, -1.0f, 1.0f);
        return in;
    }
};

//! Runs invokeAssistantPreamble on GPU for the given inputs; returns the output rows.
std::vector<half> runPreambleKernel(
    PreambleInputs const& in, int32_t languageId, int32_t codecThinkId, cudaStream_t stream)
{
    PreambleTokenIds const ids;
    int32_t const prefixLen = 8 + ((languageId >= 0) ? 1 : 0);
    int64_t const totalRows = prefixLen + in.textLen + 2;
    int64_t const projRows = 3 + in.textLen;
    int64_t const hd = in.hiddenDim;

    rt::Tensor gpuProjected({projRows, hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuTtsPad({hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuTtsBos({hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuTtsEos({hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuEmbTable({in.vocabSize, hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuOutput({totalRows, hd}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(gpuProjected, in.projected);
    copyHostToDevice(gpuTtsPad, in.ttsPad);
    copyHostToDevice(gpuTtsBos, in.ttsBos);
    copyHostToDevice(gpuTtsEos, in.ttsEos);
    copyHostToDevice(gpuEmbTable, in.embTable);

    kernel::invokeAssistantPreamble(gpuProjected, gpuTtsPad, gpuTtsBos, gpuTtsEos, gpuEmbTable, ids.skipThink,
        ids.thinkBos, ids.thinkEos, ids.speaker, ids.pad, ids.bos, codecThinkId, languageId, in.textLen, gpuOutput,
        stream);

    auto gpuResult = copyDeviceToHost<half>(gpuOutput);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return gpuResult;
}

//! Runs the CPU reference for the same inputs.
std::vector<half> runPreambleReference(PreambleInputs const& in, int32_t languageId, int32_t codecThinkId)
{
    PreambleTokenIds const ids;
    std::vector<half> refOutput;
    referenceAssistantPreamble(in.projected, in.ttsPad, in.ttsBos, in.ttsEos, in.embTable, ids.skipThink, ids.thinkBos,
        ids.thinkEos, ids.speaker, ids.pad, ids.bos, codecThinkId, languageId, in.textLen, in.hiddenDim, refOutput);
    return refOutput;
}

void expectRowsBitExact(std::vector<half> const& gpu, std::vector<half> const& ref, int32_t hiddenDim)
{
    ASSERT_EQ(gpu.size(), ref.size());
    for (size_t i = 0; i < gpu.size(); ++i)
    {
        EXPECT_TRUE(isclose(gpu[i], ref[i], 0.f, 0.f))
            << "Mismatch at row " << (i / hiddenDim) << " dim " << (i % hiddenDim) << ": gpu=" << __half2float(gpu[i])
            << " ref=" << __half2float(ref[i]);
    }
}

} // namespace

// Locks the historical 8-row no-language layout (languageId = -1). Guards the
// byte-identical requirement for all existing TTS / Omni prefill inputs.
TEST_F(TalkerKernelTest, AssistantPreamble8RowLegacy)
{
    for (int32_t textLen : {1, 4, 17})
    {
        SCOPED_TRACE("textLen=" + std::to_string(textLen));
        auto const in = PreambleInputs::makeRandom(textLen, /*hiddenDim=*/64);
        auto const gpu = runPreambleKernel(in, /*languageId=*/-1, /*codecThinkId=*/-1, stream);
        auto const ref = runPreambleReference(in, /*languageId=*/-1, /*codecThinkId=*/-1);
        expectRowsBitExact(gpu, ref, 64);
    }
}

// Verifies the 9-row CustomVoice layout: row3 uses codecThinkId (not the no-think token) and the
// language row is injected between thinkBos and thinkEos; text/suffix rows shift by one.
TEST_F(TalkerKernelTest, AssistantPreamble9RowLanguage)
{
    PreambleTokenIds const ids;
    for (int32_t textLen : {1, 4, 17})
    {
        SCOPED_TRACE("textLen=" + std::to_string(textLen));
        auto const in = PreambleInputs::makeRandom(textLen, /*hiddenDim=*/64);
        auto const gpu = runPreambleKernel(in, /*languageId=*/ids.language, ids.think, stream);
        auto const ref = runPreambleReference(in, /*languageId=*/ids.language, ids.think);
        expectRowsBitExact(gpu, ref, 64);
    }
}

// The 8-row output must be the 9-row output with the language row removed: rows [0..4]
// identical (except row 3: no-think vs think token), 8-row rows [5..] == 9-row rows [6..].
// Same PreambleInputs instance feeds both runs, so this property is checked on identical
// data and catches slot-mapping regressions independent of the CPU reference.
TEST_F(TalkerKernelTest, AssistantPreambleLanguageRowInsertionProperty)
{
    int32_t const hiddenDim = 64;
    int32_t const textLen = 4;
    PreambleTokenIds const ids;

    auto const in = PreambleInputs::makeRandom(textLen, hiddenDim);
    auto const gpu8 = runPreambleKernel(in, /*languageId=*/-1, /*codecThinkId=*/-1, stream);
    auto const gpu9 = runPreambleKernel(in, /*languageId=*/ids.language, ids.think, stream);

    int32_t const rows8 = 8 + textLen + 2;
    int32_t const rows9 = 9 + textLen + 2;
    ASSERT_EQ(gpu8.size(), static_cast<size_t>(rows8) * hiddenDim);
    ASSERT_EQ(gpu9.size(), static_cast<size_t>(rows9) * hiddenDim);

    auto rowsEqual = [&](int32_t row8, int32_t row9) {
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            if (!isclose(gpu8[static_cast<size_t>(row8) * hiddenDim + d],
                    gpu9[static_cast<size_t>(row9) * hiddenDim + d], 0.f, 0.f))
            {
                return false;
            }
        }
        return true;
    };

    // Rows 0-2 (projected) and row 4 (thinkBos) identical; row 3 differs (no-think vs think).
    EXPECT_TRUE(rowsEqual(0, 0));
    EXPECT_TRUE(rowsEqual(1, 1));
    EXPECT_TRUE(rowsEqual(2, 2));
    EXPECT_FALSE(rowsEqual(3, 3)) << "Row 3 must differ: no-think (8-row) vs think (9-row)";
    EXPECT_TRUE(rowsEqual(4, 4));
    // 8-row rows [5..end] correspond to 9-row rows [6..end] (language row skipped).
    for (int32_t r = 5; r < rows8; ++r)
    {
        EXPECT_TRUE(rowsEqual(r, r + 1)) << "8-row row " << r << " should equal 9-row row " << (r + 1);
    }
}

// ===== invokePrefillRowAssemble (descriptor-driven prefill assembly) =====

namespace
{

//! GPU tensors + a row-descriptor list driving invokePrefillRowAssemble directly
//! (mirrors the runtime's inline prefill assembly in prepareTalkerInput).
struct BuilderHarness
{
    rt::Tensor gpuProjected;
    rt::Tensor gpuTtsPad;
    rt::Tensor gpuTtsBos;
    rt::Tensor gpuTtsEos;
    rt::Tensor gpuEmbTable;
    int64_t hiddenDim;
    PreambleTokenIds ids;
    std::vector<trt_edgellm::kernel::PrefillRowDesc> rows;

    BuilderHarness(PreambleInputs const& in, int64_t projRows)
        : gpuProjected({projRows, in.hiddenDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF)
        , gpuTtsPad({static_cast<int64_t>(in.hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF)
        , gpuTtsBos({static_cast<int64_t>(in.hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF)
        , gpuTtsEos({static_cast<int64_t>(in.hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF)
        , gpuEmbTable({in.vocabSize, in.hiddenDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF)
        , hiddenDim(in.hiddenDim)
    {
        copyHostToDevice(gpuProjected, in.projected);
        copyHostToDevice(gpuTtsPad, in.ttsPad);
        copyHostToDevice(gpuTtsBos, in.ttsBos);
        copyHostToDevice(gpuTtsEos, in.ttsEos);
        copyHostToDevice(gpuEmbTable, in.embTable);
    }

    half const* projRow(int64_t r) const
    {
        return static_cast<half const*>(gpuProjected.rawPointer()) + r * hiddenDim;
    }
    half const* embRow(int32_t tokenId) const
    {
        return static_cast<half const*>(gpuEmbTable.rawPointer()) + static_cast<int64_t>(tokenId) * hiddenDim;
    }
    half const* pad() const
    {
        return static_cast<half const*>(gpuTtsPad.rawPointer());
    }
    void push(half const* a, half const* b)
    {
        rows.push_back({a, b});
    }

    void addInstructRows(int64_t startRow, int64_t numRows)
    {
        for (int64_t i = 0; i < numRows; ++i)
        {
            push(projRow(startRow + i), nullptr);
        }
    }
    void addRoleRows(int64_t startRow)
    {
        for (int64_t i = 0; i < 3; ++i)
        {
            push(projRow(startRow + i), nullptr);
        }
    }
    void addThinkBlock(int32_t languageId)
    {
        bool const hasLanguage = (languageId >= 0);
        push(pad(), embRow(hasLanguage ? ids.think : ids.skipThink));
        push(pad(), embRow(ids.thinkBos));
        if (hasLanguage)
        {
            push(pad(), embRow(languageId));
        }
        push(pad(), embRow(ids.thinkEos));
    }
    void addSpeakerToken(int32_t speakerId)
    {
        push(pad(), embRow(speakerId));
    }
    void addBosPadRow()
    {
        push(static_cast<half const*>(gpuTtsBos.rawPointer()), embRow(ids.pad));
    }
    void addTextRows(int64_t startRow, int64_t numTextRows)
    {
        for (int64_t i = 0; i < numTextRows; ++i)
        {
            push(projRow(startRow + i), embRow((i == numTextRows - 1) ? ids.bos : ids.pad));
        }
    }
    void addSuffixRows()
    {
        push(static_cast<half const*>(gpuTtsEos.rawPointer()), embRow(ids.pad));
        push(pad(), embRow(ids.bos));
    }

    int64_t numRows() const
    {
        return static_cast<int64_t>(rows.size());
    }
    void build(rt::Tensor& output, cudaStream_t stream)
    {
        int64_t const bytes = numRows() * static_cast<int64_t>(sizeof(trt_edgellm::kernel::PrefillRowDesc));
        rt::Tensor descs({bytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
        CUDA_CHECK(cudaMemcpyAsync(descs.rawPointer(), rows.data(), bytes, cudaMemcpyHostToDevice, stream));
        trt_edgellm::kernel::invokePrefillRowAssemble(
            reinterpret_cast<trt_edgellm::kernel::PrefillRowDesc const*>(descs.rawPointer()),
            static_cast<int32_t>(numRows()), static_cast<int32_t>(hiddenDim), output, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

} // namespace

// Builder CustomVoice recipe must reproduce invokeAssistantPreamble bit-exactly for both the
// 8-row (no language) and 9-row (language) layouts.
TEST_F(TalkerKernelTest, PrefillBuilderMatchesFusedKernel)
{
    PreambleTokenIds const ids;
    int32_t const hiddenDim = 64;
    for (int32_t languageId : {-1, ids.language})
    {
        for (int32_t textLen : {1, 4, 17})
        {
            SCOPED_TRACE("languageId=" + std::to_string(languageId) + " textLen=" + std::to_string(textLen));
            auto const in = PreambleInputs::makeRandom(textLen, hiddenDim);

            // Reference: fused kernel path.
            auto const fused = runPreambleKernel(in, languageId, (languageId >= 0) ? ids.think : -1, stream);

            // Builder path with the equivalent CustomVoice recipe.
            BuilderHarness h(in, /*projRows=*/3 + textLen);
            h.rows.clear();
            h.addRoleRows(0);
            h.addThinkBlock(languageId);
            h.addSpeakerToken(ids.speaker);
            h.addBosPadRow();
            h.addTextRows(3, textLen);
            h.addSuffixRows();

            int32_t const prefixLen = 8 + ((languageId >= 0) ? 1 : 0);
            int64_t const totalRows = prefixLen + textLen + 2;
            ASSERT_EQ(h.numRows(), totalRows);

            rt::Tensor gpuOut(
                {totalRows, static_cast<int64_t>(hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            h.build(gpuOut, stream);
            auto const built = copyDeviceToHost<half>(gpuOut);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            expectRowsBitExact(built, fused, hiddenDim);
        }
    }
}

// Instruct rows prepend as pure text-projected copies; the remaining layout shifts by K and
// matches the CPU reference row-for-row.
TEST_F(TalkerKernelTest, PrefillBuilderInstructSegment)
{
    PreambleTokenIds const ids;
    int32_t const hiddenDim = 64;
    int32_t const textLen = 4;
    int64_t const instructRows = 5;
    int64_t const projRows = instructRows + 3 + textLen; // [instruct K][role 3][text N]

    auto in = PreambleInputs::makeRandom(textLen, hiddenDim);
    // Extend projected to cover the instruct segment.
    in.projected.resize(static_cast<size_t>(projRows) * hiddenDim);
    uniformFloatInitialization(in.projected, -1.0f, 1.0f);

    BuilderHarness h(in, projRows);
    h.rows.clear();
    h.addInstructRows(0, instructRows);
    h.addRoleRows(instructRows);
    h.addThinkBlock(/*languageId=*/-1);
    h.addSpeakerToken(ids.speaker);
    h.addBosPadRow();
    h.addTextRows(instructRows + 3, textLen);
    h.addSuffixRows();

    int64_t const totalRows = instructRows + 8 + textLen + 2;
    ASSERT_EQ(h.numRows(), totalRows);
    rt::Tensor gpuOut({totalRows, static_cast<int64_t>(hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    h.build(gpuOut, stream);
    auto const built = copyDeviceToHost<half>(gpuOut);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Rows [0..K): exact copies of projected instruct rows.
    for (int64_t r = 0; r < instructRows; ++r)
    {
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            EXPECT_TRUE(isclose(built[r * hiddenDim + d], in.projected[r * hiddenDim + d], 0.f, 0.f))
                << "instruct row " << r << " dim " << d;
        }
    }
    // Remaining rows: CPU reference over the shifted projected view.
    std::vector<half> shiftedProjected(in.projected.begin() + instructRows * hiddenDim, in.projected.end());
    std::vector<half> refTail;
    referenceAssistantPreamble(shiftedProjected, in.ttsPad, in.ttsBos, in.ttsEos, in.embTable, ids.skipThink,
        ids.thinkBos, ids.thinkEos, ids.speaker, ids.pad, ids.bos, /*codecThinkId=*/-1, /*languageId=*/-1, textLen,
        hiddenDim, refTail);
    for (size_t i = 0; i < refTail.size(); ++i)
    {
        EXPECT_TRUE(isclose(built[instructRows * hiddenDim + i], refTail[i], 0.f, 0.f))
            << "shifted row mismatch at flat index " << i;
    }
}

// VoiceDesign recipe: no speaker row — prefix is one row shorter and rows after the think
// block shift up by one.
TEST_F(TalkerKernelTest, PrefillBuilderNoSpeakerRecipe)
{
    PreambleTokenIds const ids;
    int32_t const hiddenDim = 64;
    int32_t const textLen = 4;

    auto const in = PreambleInputs::makeRandom(textLen, hiddenDim);
    BuilderHarness h(in, 3 + textLen);
    h.rows.clear();
    h.addRoleRows(0);
    h.addThinkBlock(/*languageId=*/-1);
    // No speaker row (VoiceDesign / speaker-less request).
    h.addBosPadRow();
    h.addTextRows(3, textLen);
    h.addSuffixRows();

    int64_t const totalRows = 7 + textLen + 2;
    ASSERT_EQ(h.numRows(), totalRows);
    rt::Tensor gpuOut({totalRows, static_cast<int64_t>(hiddenDim)}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    h.build(gpuOut, stream);
    auto const built = copyDeviceToHost<half>(gpuOut);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare against the 8-row fused output with the speaker row (index 6) removed.
    auto const fused = runPreambleKernel(in, /*languageId=*/-1, /*codecThinkId=*/-1, stream);
    int64_t builtRow = 0;
    for (int64_t fusedRow = 0; fusedRow < static_cast<int64_t>(fused.size()) / hiddenDim; ++fusedRow)
    {
        if (fusedRow == 6) // speaker row in the fused layout
        {
            continue;
        }
        for (int32_t d = 0; d < hiddenDim; ++d)
        {
            EXPECT_TRUE(isclose(built[builtRow * hiddenDim + d], fused[fusedRow * hiddenDim + d], 0.f, 0.f))
                << "built row " << builtRow << " vs fused row " << fusedRow << " dim " << d;
        }
        ++builtRow;
    }
    EXPECT_EQ(builtRow, totalRows);
}

// ===== TalkerLogitAdjust =====

// Helper: upload logits and seenTokens to GPU, run kernel, download result.
static std::vector<float> runTalkerLogitAdjust(std::vector<float> const& hostLogits, int32_t suppressStart,
    int32_t suppressEnd, int32_t codecEosId, std::vector<int32_t> const& hostSeenTokens, float repetitionPenalty,
    cudaStream_t stream)
{
    int32_t const vocabSize = static_cast<int32_t>(hostLogits.size());

    rt::Tensor gpuLogits({1, static_cast<int64_t>(vocabSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(
        cudaMemcpy(gpuLogits.rawPointer(), hostLogits.data(), vocabSize * sizeof(float), cudaMemcpyHostToDevice));

    int32_t const maxSeen = std::max(static_cast<int32_t>(hostSeenTokens.size()), 1);
    rt::Tensor gpuSeen({static_cast<int64_t>(maxSeen)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    if (!hostSeenTokens.empty())
    {
        CUDA_CHECK(cudaMemcpy(gpuSeen.rawPointer(), hostSeenTokens.data(), hostSeenTokens.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice));
    }

    kernel::invokeTalkerLogitAdjust(gpuSeen, gpuLogits, suppressStart, suppressEnd, codecEosId,
        static_cast<int32_t>(hostSeenTokens.size()), repetitionPenalty, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return copyDeviceToHost<float>(gpuLogits);
}

// Suppression range [suppressStart, suppressEnd) is set to -inf, except codecEosId.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_SuppressionWithEosExempt)
{
    int32_t const vocabSize = 256;
    int32_t const suppressStart = 50;
    int32_t const suppressEnd = 150;
    int32_t const codecEosId = 100;

    auto result = runTalkerLogitAdjust(
        std::vector<float>(vocabSize, 1.0f), suppressStart, suppressEnd, codecEosId, {}, 1.0f, stream);

    for (int32_t i = 0; i < vocabSize; ++i)
    {
        if (i >= suppressStart && i < suppressEnd && i != codecEosId)
        {
            EXPECT_TRUE(std::isinf(result[i]) && result[i] < 0) << "Token " << i << " should be -inf";
        }
        else
        {
            EXPECT_FLOAT_EQ(result[i], 1.0f) << "Token " << i << " should be unchanged";
        }
    }
}

// No exception token (-1): all tokens in suppress range become -inf.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_SuppressionNoExempt)
{
    int32_t const vocabSize = 128;
    int32_t const suppressStart = 0;
    int32_t const suppressEnd = 64;

    auto result = runTalkerLogitAdjust(
        std::vector<float>(vocabSize, 2.0f), suppressStart, suppressEnd, /*codecEosId=*/-1, {}, 1.0f, stream);

    for (int32_t i = 0; i < suppressEnd; ++i)
    {
        EXPECT_TRUE(std::isinf(result[i]) && result[i] < 0) << "Token " << i << " should be -inf";
    }
    for (int32_t i = suppressEnd; i < vocabSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 2.0f) << "Token " << i << " should be unchanged";
    }
}

// Repetition penalty applied to seenTokens: positive logit divided, negative multiplied.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_RepetitionPenalty)
{
    int32_t const vocabSize = 64;
    int32_t const suppressStart = 50;
    int32_t const suppressEnd = 60;
    float const penalty = 2.0f;

    // token 5: positive logit (4.0 → 4.0/2 = 2.0)
    // token 10: negative logit (-4.0 → -4.0*2 = -8.0)
    std::vector<float> hostLogits(vocabSize, 1.0f);
    hostLogits[5] = 4.0f;
    hostLogits[10] = -4.0f;

    auto result = runTalkerLogitAdjust(hostLogits, suppressStart, suppressEnd,
        /*codecEosId=*/-1, {5, 10}, penalty, stream);

    EXPECT_FLOAT_EQ(result[5], 2.0f) << "Positive logit should be divided by penalty";
    EXPECT_FLOAT_EQ(result[10], -8.0f) << "Negative logit should be multiplied by penalty";
    // Unseen tokens in normal range unchanged
    EXPECT_FLOAT_EQ(result[0], 1.0f) << "Unseen token should be unchanged";
    EXPECT_FLOAT_EQ(result[20], 1.0f) << "Unseen token should be unchanged";
}
