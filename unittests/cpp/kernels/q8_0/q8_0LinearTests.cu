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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/q8_0/q8_0Kernels.cuh"
#include "kernels/q8_0/q8_0Linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace trt_edgellm;

namespace
{
using kernel::q8_0::kBlockSize;
using kernel::q8_0::PackedWeightBlock;

struct QuantizedActivationBlock
{
    float scale;
    int8_t codes[kBlockSize];
};

std::vector<PackedWeightBlock> makeWeights(int32_t outFeatures, int32_t blocksPerRow, std::mt19937& rng)
{
    std::vector<PackedWeightBlock> weights(static_cast<size_t>(outFeatures) * blocksPerRow);
    std::uniform_int_distribution<int32_t> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.001F, 0.02F);
    for (auto& block : weights)
    {
        block.scale = __float2half(scale(rng));
        for (auto& c : block.codes)
        {
            c = static_cast<int8_t>(code(rng));
        }
    }
    return weights;
}

// Q8_1 activation quantization on the CPU, independent of the kernel: one FP32 scale per 32 values, codes rounded
// half away from zero.
std::vector<QuantizedActivationBlock> quantize(std::vector<float> const& input, int32_t rows, int32_t blocksPerRow)
{
    std::vector<QuantizedActivationBlock> blocks(static_cast<size_t>(rows) * blocksPerRow);
    for (size_t b = 0; b < blocks.size(); ++b)
    {
        float const* values = input.data() + b * kBlockSize;
        float peak = 0.0F;
        for (int32_t i = 0; i < kBlockSize; ++i)
        {
            peak = std::max(peak, std::fabs(values[i]));
        }
        float const inverse = peak > 0.0F ? 127.0F / peak : 0.0F;
        blocks[b].scale = peak > 0.0F ? 1.0F / inverse : 0.0F;
        for (int32_t i = 0; i < kBlockSize; ++i)
        {
            blocks[b].codes[i] = static_cast<int8_t>(std::round(values[i] * inverse));
        }
    }
    return blocks;
}

// FP64 oracle over CPU-quantized activations.
std::vector<double> reference(std::vector<QuantizedActivationBlock> const& activations,
    std::vector<PackedWeightBlock> const& weights, int32_t rows, int32_t outFeatures, int32_t blocksPerRow)
{
    std::vector<double> out(static_cast<size_t>(rows) * outFeatures);
    for (int32_t r = 0; r < rows; ++r)
    {
        for (int32_t n = 0; n < outFeatures; ++n)
        {
            double sum = 0.0;
            for (int32_t b = 0; b < blocksPerRow; ++b)
            {
                auto const& a = activations[static_cast<size_t>(r) * blocksPerRow + b];
                auto const& w = weights[static_cast<size_t>(n) * blocksPerRow + b];
                int32_t dot = 0;
                for (int32_t i = 0; i < kBlockSize; ++i)
                {
                    dot += static_cast<int32_t>(a.codes[i]) * w.codes[i];
                }
                sum += static_cast<double>(dot) * a.scale * __half2float(w.scale);
            }
            out[static_cast<size_t>(r) * outFeatures + n] = sum;
        }
    }
    return out;
}

size_t typeSize(DataType type)
{
    return type == DataType::kFLOAT ? 4 : 2;
}

// Round to the activation type, so the CPU quantizer sees exactly what the kernel reads.
std::vector<uint16_t> toBits(std::vector<float>& values, DataType type)
{
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (type == DataType::kHALF)
        {
            half const h = __float2half_rn(values[i]);
            values[i] = __half2float(h);
            std::memcpy(&bits[i], &h, sizeof(uint16_t));
        }
        else
        {
            __nv_bfloat16 const h = __float2bfloat16_rn(values[i]);
            values[i] = __bfloat162float(h);
            std::memcpy(&bits[i], &h, sizeof(uint16_t));
        }
    }
    return bits;
}

float fromBits(uint8_t const* data, size_t index, DataType type)
{
    if (type == DataType::kFLOAT)
    {
        float value;
        std::memcpy(&value, data + index * 4, 4);
        return value;
    }
    if (type == DataType::kHALF)
    {
        half value;
        std::memcpy(&value, data + index * 2, 2);
        return __half2float(value);
    }
    __nv_bfloat16 value;
    std::memcpy(&value, data + index * 2, 2);
    return __bfloat162float(value);
}

using Shape = std::tuple<int32_t, int32_t, int32_t>;
using Types = std::tuple<DataType, DataType>;

class Q8_0LinearTest : public ::testing::TestWithParam<std::tuple<Shape, Types>>
{
};
} // namespace

// Given packed Q8_0 weights, when rows select either the DP4A path or the tiled MMA path (including partial token and
// channel tiles and K that is not a whole stage), then the output matches an FP64 oracle over CPU-quantized
// activations, for each activation and output type the plugin accepts.
TEST_P(Q8_0LinearTest, MatchesReference)
{
    auto const [shape, types] = GetParam();
    auto const [rows, outFeatures, inFeatures] = shape;
    auto const [inputType, outputType] = types;
    int32_t const blocksPerRow = inFeatures / kBlockSize;
    std::mt19937 rng(rows * 131 + outFeatures * 7 + inFeatures);
    std::uniform_real_distribution<float> value(-2.0F, 2.0F);
    std::vector<float> input(static_cast<size_t>(rows) * inFeatures);
    for (auto& v : input)
    {
        v = value(rng);
    }
    auto const inputBits = toBits(input, inputType);
    auto const weights = makeWeights(outFeatures, blocksPerRow, rng);

    void* dInput{};
    void* dWeights{};
    void* dOutput{};
    void* dWorkspace{};
    size_t const outputCount = static_cast<size_t>(rows) * outFeatures;
    size_t const outputBytes = outputCount * typeSize(outputType);
    size_t const workspaceSize = kernel::q8_0WorkspaceSize(rows, inFeatures);
    size_t const weightBytes = weights.size() * sizeof(PackedWeightBlock);
    CUDA_CHECK(cudaMalloc(&dInput, inputBits.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&dWeights, weightBytes));
    CUDA_CHECK(cudaMalloc(&dOutput, outputBytes));
    CUDA_CHECK(cudaMalloc(&dWorkspace, workspaceSize));
    CUDA_CHECK(cudaMemcpy(dInput, inputBits.data(), inputBits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWeights, weights.data(), weightBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0xff, outputBytes));
    void const* weightList[] = {dWeights};
    ASSERT_EQ(kernel::q8_0Linear(dInput, inputType, weightList, &outFeatures, 1, dOutput, outputType, dWorkspace,
                  workspaceSize, rows, inFeatures, nullptr),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> actual(outputBytes);
    CUDA_CHECK(cudaMemcpy(actual.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost));

    // The same rows split as three weights in separate allocations (as Q, K and V) share one quantization and fill
    // consecutive output columns, byte-identical to the single weight.
    int32_t const first = outFeatures / 2;
    int32_t const second = (outFeatures - first) / 2;
    int32_t const parts[] = {first, second, outFeatures - first - second};
    if (parts[0] > 0 && parts[1] > 0 && parts[2] > 0)
    {
        void* split[3]{};
        size_t row = 0;
        for (int32_t p = 0; p < 3; ++p)
        {
            size_t const bytes = static_cast<size_t>(parts[p]) * blocksPerRow * sizeof(PackedWeightBlock);
            CUDA_CHECK(cudaMalloc(&split[p], bytes));
            CUDA_CHECK(cudaMemcpy(split[p], weights.data() + row * blocksPerRow, bytes, cudaMemcpyHostToDevice));
            row += parts[p];
        }
        CUDA_CHECK(cudaMemset(dOutput, 0xff, outputBytes));
        ASSERT_EQ(kernel::q8_0Linear(dInput, inputType, split, parts, 3, dOutput, outputType, dWorkspace, workspaceSize,
                      rows, inFeatures, nullptr),
            cudaSuccess);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> packed(outputBytes);
        CUDA_CHECK(cudaMemcpy(packed.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost));
        EXPECT_TRUE(packed == actual) << "split weights differ from one weight";
        for (void* part : split)
        {
            CUDA_CHECK(cudaFree(part));
        }
    }
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dWeights));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dWorkspace));

    auto const expected = reference(quantize(input, rows, blocksPerRow), weights, rows, outFeatures, blocksPerRow);
    double errorSquares = 0.0;
    double referenceSquares = 0.0;
    for (size_t i = 0; i < outputCount; ++i)
    {
        float const got = fromBits(actual.data(), i, outputType);
        ASSERT_TRUE(std::isfinite(got)) << "index " << i;
        errorSquares += (got - expected[i]) * (got - expected[i]);
        referenceSquares += expected[i] * expected[i];
    }
    // The kernel divides with fast math, so a value within a few ULP of a rounding tie can quantize one code away
    // from the CPU oracle; over K = 4096 that is about 1e-4 relative. FP16 and BF16 add one output rounding. A wrong
    // quantizer scale or code is off by orders of magnitude.
    double const tolerance = outputType == DataType::kBF16 ? 8e-3 : 1e-3;
    EXPECT_LT(std::sqrt(errorSquares / referenceSquares), tolerance);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Q8_0LinearTest,
    ::testing::Combine(
        ::testing::Values(std::make_tuple(1, 256, 4096), std::make_tuple(31, 1024, 4096),
            std::make_tuple(32, 1024, 4096), std::make_tuple(100, 4096, 1024), std::make_tuple(257, 256, 4096),
            std::make_tuple(65, 130, 128), std::make_tuple(40, 100, 96), std::make_tuple(512, 1024, 4096)),
        ::testing::Values(std::make_tuple(DataType::kHALF, DataType::kHALF),
            std::make_tuple(DataType::kHALF, DataType::kBF16), std::make_tuple(DataType::kBF16, DataType::kHALF),
            std::make_tuple(DataType::kHALF, DataType::kFLOAT))));

namespace
{
constexpr int32_t kBlock = 32;
constexpr size_t kPackedBlockBytes = 34;

#define Q8_TEST_CUDA(call) ASSERT_EQ((call), cudaSuccess)

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        mStatus = cudaMalloc(&mData, bytes == 0 ? 1 : bytes);
        if (mStatus != cudaSuccess)
        {
            mData = nullptr;
        }
    }
    ~DeviceBuffer()
    {
        cudaFree(mData);
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    void* get() const
    {
        return mData;
    }
    cudaError_t status() const
    {
        return mStatus;
    }

private:
    void* mData{nullptr};
    cudaError_t mStatus{cudaSuccess};
};

//! Packed Q8_0 checkpoint rows: per 32-value block, an FP16 scale followed by 32 int8 codes.
struct PackedWeights
{
    int32_t outFeatures;
    int32_t inFeatures;
    std::vector<uint8_t> bytes;

    uint8_t const* block(int32_t channel, int32_t b) const
    {
        return bytes.data() + (static_cast<size_t>(channel) * (inFeatures / kBlock) + b) * kPackedBlockBytes;
    }
};

PackedWeights makePackedWeights(int32_t outFeatures, int32_t inFeatures, uint32_t seed)
{
    PackedWeights weights{outFeatures, inFeatures, {}};
    size_t const blocks = static_cast<size_t>(outFeatures) * (inFeatures / kBlock);
    weights.bytes.resize(blocks * kPackedBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale(1.0e-3F, 2.0e-2F);
    uint32_t state = seed * 2654435761U + 1U;
    for (size_t b = 0; b < blocks; ++b)
    {
        uint8_t* dst = weights.bytes.data() + b * kPackedBlockBytes;
        half const s = __float2half(scale(rng));
        std::memcpy(dst, &s, sizeof(half));
        for (int32_t k = 0; k < kBlock; ++k)
        {
            state = state * 1664525U + 1013904223U;
            dst[sizeof(half) + k]
                = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int32_t>((state >> 24) % 255U) - 127));
        }
    }
    return weights;
}

//! Row 0 holds an all-zero block (zero scale) and a large outlier that sets its block's peak/127 scale.
std::vector<float> makeActivations(int32_t rows, int32_t inFeatures, uint32_t seed)
{
    std::vector<float> values(static_cast<size_t>(rows) * inFeatures);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : values)
    {
        v = normal(rng);
    }
    for (int32_t k = 0; k < kBlock && k < inFeatures; ++k)
    {
        values[k] = 0.0F;
    }
    values[inFeatures - 1] = 40.0F;
    return values;
}

//! Q8_1 activation row: per 32-value block, int8 codes and one FP32 scale.
struct QuantizedRow
{
    std::vector<int8_t> codes;
    std::vector<float> scales;
};

constexpr int32_t kTokensPerTile = 8;

//! Reads row `r` back from the workspace, stated independently of the kernel: per (8-row token tile, block), 8 x 32
//! codes where token t owns bytes [32 t, 32 t + 32) and its word 2 q + h holds k = 16 h + 4 q .. 16 h + 4 q + 3; then
//! all FP32 scales as [tile][block][token]. Checks each block against llama.cpp's Q8_1 quantizer: scale = peak / 127
//! and code = round(value * 127 / peak), where fast-math division may round a tie either way.
QuantizedRow storedRow(std::vector<uint8_t> const& workspace, float const* input, int32_t rows, int32_t inFeatures,
    int32_t r, std::string const& label)
{
    int32_t const blocks = inFeatures / kBlock;
    size_t const tileCodeBytes = static_cast<size_t>(kTokensPerTile) * kBlock;
    size_t const codeBytes = static_cast<size_t>((rows + kTokensPerTile - 1) / kTokensPerTile) * blocks * tileCodeBytes;
    QuantizedRow out{std::vector<int8_t>(inFeatures), std::vector<float>(blocks)};
    int32_t const t = r % kTokensPerTile;
    for (int32_t b = 0; b < blocks; ++b)
    {
        size_t const tileBlock = static_cast<size_t>(r / kTokensPerTile) * blocks + b;
        std::memcpy(&out.scales[b], workspace.data() + codeBytes + (tileBlock * kTokensPerTile + t) * sizeof(float),
            sizeof(float));
        float const* values = input + static_cast<size_t>(r) * inFeatures + b * kBlock;
        float peak = 0.0F;
        for (int32_t k = 0; k < kBlock; ++k)
        {
            peak = std::fmax(peak, std::fabs(values[k]));
        }
        EXPECT_NEAR(out.scales[b], peak / 127.0F, 1.0e-6F * peak) << label << " row " << r << " block " << b;
        for (int32_t k = 0; k < kBlock; ++k)
        {
            int32_t const word = 2 * ((k % 16) / 4) + k / 16;
            int8_t code{};
            std::memcpy(&code, workspace.data() + tileBlock * tileCodeBytes + t * kBlock + word * 4 + k % 4, 1);
            out.codes[b * kBlock + k] = code;
            double const exact = peak > 0.0F ? values[k] * 127.0 / peak : 0.0;
            EXPECT_LE(std::fabs(code - exact), 0.5 + 1.0e-4) << label << " row " << r << " block " << b << " k " << k;
        }
    }
    return out;
}

struct ReferenceValue
{
    double value;
    double magnitude; //!< Sum of absolute per-block terms; bounds FP32 reassociation error.
};

//! FP64 oracle of the Q8_0 x Q8_1 contract: exact integer dot per block, scaled by both block scales.
ReferenceValue referenceDot(PackedWeights const& weights, QuantizedRow const& row, int32_t channel)
{
    ReferenceValue result{0.0, 0.0};
    for (int32_t b = 0; b < weights.inFeatures / kBlock; ++b)
    {
        uint8_t const* block = weights.block(channel, b);
        half scaleBits;
        std::memcpy(&scaleBits, block, sizeof(half));
        int64_t dot = 0;
        for (int32_t k = 0; k < kBlock; ++k)
        {
            dot += static_cast<int64_t>(static_cast<int8_t>(block[sizeof(half) + k])) * row.codes[b * kBlock + k];
        }
        double const term = static_cast<double>(dot) * __half2float(scaleBits) * row.scales[b];
        result.value += term;
        result.magnitude += std::fabs(term);
    }
    return result;
}

using trt_edgellm::kernel::Q8_0Path;

//! FP16/BF16 activations as the kernel reads them, and the same values widened to FP32 for the host oracle.
struct Activations
{
    DataType type;
    std::vector<float> values;
    std::vector<uint16_t> bits;
};

Activations roundActivations(std::vector<float> values, DataType type)
{
    auto bits = toBits(values, type);
    return Activations{type, std::move(values), std::move(bits)};
}

//! Runs the Q8_0 linear on `path` with weights placed `weightOffset` bytes past a 256-byte aligned allocation and
//! stores the call status in `status`. Outputs are widened to FP32, which is exact for every output type.
void runLinearStatus(PackedWeights const& weights, Activations const& input, int32_t rows, DataType outputType,
    std::vector<float>& output, size_t weightOffset, Q8_0Path path, cudaError_t& status,
    std::vector<uint8_t>* workspaceOut = nullptr)
{
    int32_t const n = weights.outFeatures;
    int32_t const k = weights.inFeatures;
    size_t const outputBytes = static_cast<size_t>(rows) * n * typeSize(outputType);
    DeviceBuffer dWeights(weights.bytes.size() + weightOffset);
    DeviceBuffer dInput(static_cast<size_t>(rows) * k * sizeof(uint16_t));
    DeviceBuffer dOutput(outputBytes);
    size_t const workspaceSize = trt_edgellm::kernel::q8_0WorkspaceSize(rows, k);
    DeviceBuffer dWorkspace(workspaceSize);
    for (DeviceBuffer const* buffer : {&dWeights, &dInput, &dOutput, &dWorkspace})
    {
        ASSERT_EQ(buffer->status(), cudaSuccess) << cudaGetErrorString(buffer->status());
    }
    auto* weightPtr = static_cast<uint8_t*>(dWeights.get()) + weightOffset;
    Q8_TEST_CUDA(cudaMemcpy(weightPtr, weights.bytes.data(), weights.bytes.size(), cudaMemcpyHostToDevice));
    Q8_TEST_CUDA(cudaMemcpy(
        dInput.get(), input.bits.data(), static_cast<size_t>(rows) * k * sizeof(uint16_t), cudaMemcpyHostToDevice));
    // A NaN sentinel in the workspace and output catches reads of unwritten workspace and unwritten outputs.
    Q8_TEST_CUDA(cudaMemset(dWorkspace.get(), 0xFF, workspaceSize));
    Q8_TEST_CUDA(cudaMemset(dOutput.get(), 0xFF, outputBytes));
    cudaStream_t stream;
    Q8_TEST_CUDA(cudaStreamCreate(&stream));
    void const* weightList[] = {weightPtr};
    status = trt_edgellm::kernel::q8_0LinearForPath(dInput.get(), input.type, weightList, &n, 1, dOutput.get(),
        outputType, dWorkspace.get(), workspaceSize, rows, k, path, stream);
    Q8_TEST_CUDA(cudaStreamSynchronize(stream));
    Q8_TEST_CUDA(cudaStreamDestroy(stream));
    std::vector<uint8_t> raw(outputBytes);
    Q8_TEST_CUDA(cudaMemcpy(raw.data(), dOutput.get(), outputBytes, cudaMemcpyDeviceToHost));
    output.resize(static_cast<size_t>(rows) * n);
    for (size_t i = 0; i < output.size(); ++i)
    {
        output[i] = fromBits(raw.data(), i, outputType);
    }
    if (workspaceOut != nullptr)
    {
        workspaceOut->resize(workspaceSize);
        Q8_TEST_CUDA(cudaMemcpy(workspaceOut->data(), dWorkspace.get(), workspaceSize, cudaMemcpyDeviceToHost));
    }
}

void runLinear(PackedWeights const& weights, Activations const& input, int32_t rows, DataType outputType,
    std::vector<float>& output, size_t weightOffset = 0, Q8_0Path path = Q8_0Path::kAuto,
    std::vector<uint8_t>* workspaceOut = nullptr)
{
    cudaError_t status = cudaErrorUnknown;
    runLinearStatus(weights, input, rows, outputType, output, weightOffset, path, status, workspaceOut);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

std::vector<int32_t> sampledChannels(int32_t outFeatures)
{
    std::vector<int32_t> channels;
    // Large heads are checked on a stride that visits every 16-channel tile position and the tail.
    int32_t const stride = outFeatures > 8192 ? 61 : 1;
    for (int32_t c = 0; c < outFeatures; c += stride)
    {
        channels.push_back(c);
    }
    for (int32_t c = std::max(0, outFeatures - 3); c < outFeatures; ++c)
    {
        channels.push_back(c);
    }
    return channels;
}

//! FP32 accumulation of up to K/32 block terms in any order stays within this fraction of their absolute sum.
constexpr double kRelativeAccumulationTolerance = 1.0e-5;

//! Relative rounding error of the final store, plus half of FP16's smallest subnormal.
double outputRounding(DataType type, double value)
{
    constexpr double kFp16HalfSubnormal = 3.0e-8;
    switch (type)
    {
    case DataType::kHALF: return std::ldexp(std::fabs(value), -11) + kFp16HalfSubnormal;
    case DataType::kBF16: return std::ldexp(std::fabs(value), -8);
    default: return 0.0;
    }
}

void expectMatchesOracle(PackedWeights const& weights, Activations const& input, int32_t rows, DataType outputType,
    std::vector<float> const& output, std::vector<uint8_t> const& workspace, std::string const& label)
{
    int32_t const n = weights.outFeatures;
    int32_t const k = weights.inFeatures;
    int32_t failures = 0;
    for (int32_t r = 0; r < rows; ++r)
    {
        QuantizedRow const quantized = storedRow(workspace, input.values.data(), rows, k, r, label);
        for (int32_t c : sampledChannels(n))
        {
            ReferenceValue const ref = referenceDot(weights, quantized, c);
            double const got = output[static_cast<size_t>(r) * n + c];
            double const tolerance
                = kRelativeAccumulationTolerance * ref.magnitude + outputRounding(outputType, ref.value) + 1.0e-30;
            if (!(std::fabs(got - ref.value) <= tolerance) && failures++ < 5)
            {
                ADD_FAILURE() << label << " row " << r << " channel " << c << ": got " << got << ", expected "
                              << ref.value << " +/- " << tolerance;
            }
        }
    }
    EXPECT_EQ(failures, 0) << label;
}

struct Geometry
{
    char const* name;
    int32_t outFeatures;
    int32_t inFeatures;
};

// Nemotron-3.5-Bolt Q8_0 projections: base and draft lm_head, latent MoE fc1/fc2, attention q/o and k/v.
constexpr Geometry kBoltGeometries[] = {
    {"lm_head", 131072, 4096},
    {"latent_fc1", 1024, 4096},
    {"latent_fc2", 4096, 1024},
    {"attention_qo", 4096, 4096},
    {"attention_kv", 256, 4096},
};
// 1-16 rows cover every GEMV token tile; 17, 24 and 64 cross the integrated GEMV-to-tensor-core switch and use
// three or more 8-row tiles.
constexpr int32_t kRowCounts[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 24, 64};
constexpr int32_t kMaxRows = 64;

//! The activation and output type pairs of Q8_0LinearTest.MatchesReference.
constexpr std::pair<DataType, DataType> kTypePairs[] = {{DataType::kHALF, DataType::kHALF},
    {DataType::kHALF, DataType::kBF16}, {DataType::kBF16, DataType::kHALF}, {DataType::kHALF, DataType::kFLOAT}};

std::string typeLabel(DataType inputType, DataType outputType)
{
    auto const name = [](DataType t) { return t == DataType::kHALF ? "f16" : t == DataType::kBF16 ? "bf16" : "f32"; };
    return std::string(name(inputType)) + "->" + name(outputType);
}

bool deviceSupportsMma()
{
    int32_t device = 0;
    int32_t major = 0;
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && major >= 8;
}
} // namespace

// Given packed Q8_0 weights in every Bolt projection geometry and FP16 or BF16 activations whose first row has a
// zero block and an outlier,
// When the Q8_0 linear runs on 1 to 8, 16, 17, 24 and 64 rows through the automatic dispatch and through each of the
// DP4A GEMV and (on SM80+) the tensor-core kernels, for every activation and output type pair of
// Q8_0LinearTest.MatchesReference,
// Then every stored activation block is llama.cpp's Q8_1 quantization of its row (scale peak / 127, codes within
// rounding of value * 127 / peak), and every output of every path equals the FP64 product of those stored activations
// and the Q8_0 weights, within FP32 accumulation error plus one rounding to the output type.
TEST(Q8_0LinearPathTest, BoltGeometriesMatchQuantizedOracleOnEveryPath)
{
    for (auto const& geometry : kBoltGeometries)
    {
        PackedWeights const weights = makePackedWeights(geometry.outFeatures, geometry.inFeatures, 17U);
        std::vector<float> const values = makeActivations(kMaxRows, geometry.inFeatures, 29U);
        for (auto const& [inputType, outputType] : kTypePairs)
        {
            Activations const input = roundActivations(values, inputType);
            for (int32_t rows : kRowCounts)
            {
                for (auto const& [path, pathName] : {std::pair{Q8_0Path::kAuto, "auto"},
                         std::pair{Q8_0Path::kGemv, "gemv"}, std::pair{Q8_0Path::kMma, "mma"}})
                {
                    if (path == Q8_0Path::kMma && !deviceSupportsMma())
                    {
                        continue;
                    }
                    std::vector<float> output;
                    std::vector<uint8_t> workspace;
                    runLinear(weights, input, rows, outputType, output, 0, path, &workspace);
                    ASSERT_FALSE(HasFatalFailure());
                    expectMatchesOracle(weights, input, rows, outputType, output, workspace,
                        std::string(geometry.name) + " " + typeLabel(inputType, outputType)
                            + " M=" + std::to_string(rows) + " path=" + pathName);
                }
            }
        }
    }
}

// Given packed Q8_0 weights and a batch of activation rows,
// When the same rows are computed in launches of 1, 2, 3, 8 and 16 rows, grouped by the side of the device's GEMV
// row limit each launch falls on,
// Then each row's output is bitwise identical across the launches of each group, so a row does not depend on
// how many other rows share the launch.
TEST(Q8_0LinearPathTest, RowOutputDoesNotDependOnOtherRowsInTheLaunch)
{
    for (auto const& geometry : {kBoltGeometries[1], kBoltGeometries[2], kBoltGeometries[4]})
    {
        PackedWeights const weights = makePackedWeights(geometry.outFeatures, geometry.inFeatures, 5U);
        Activations const input = roundActivations(makeActivations(16, geometry.inFeatures, 11U), DataType::kHALF);
        int32_t gemvMaxRows = 0;
        ASSERT_EQ(kernel::q8_0GemvMaxRows(geometry.outFeatures, gemvMaxRows), cudaSuccess);
        std::vector<int32_t> gemvGroup, mmaGroup;
        for (int32_t const rows : {1, 2, 3, 8, 16})
        {
            (rows <= gemvMaxRows ? gemvGroup : mmaGroup).push_back(rows);
        }
        for (auto const& group : {gemvGroup, mmaGroup})
        {
            if (group.empty())
            {
                continue;
            }
            std::vector<float> first;
            runLinear(weights, input, group.front(), DataType::kFLOAT, first);
            ASSERT_FALSE(HasFatalFailure());
            for (size_t i = 1; i < group.size(); ++i)
            {
                std::vector<float> other;
                runLinear(weights, input, group[i], DataType::kFLOAT, other);
                ASSERT_FALSE(HasFatalFailure());
                size_t const shared = static_cast<size_t>(group.front()) * geometry.outFeatures;
                EXPECT_EQ(std::memcmp(first.data(), other.data(), shared * sizeof(float)), 0)
                    << geometry.name << ": rows of a " << group.front() << "-row launch differ in a " << group[i]
                    << "-row launch";
            }
        }
    }
}

// Given Q8_0 weights whose rows hold an odd number of blocks, or whose base is only 2-byte aligned, and
// output-channel and row counts that leave partial tiles,
// When the Q8_0 linear runs on 1, 3, 8, 9 and 20 rows, and the tensor-core kernel is requested explicitly,
// Then every output still equals the FP64 quantized oracle, and the tensor-core request is refused with
// cudaErrorNotSupported exactly for the odd and misaligned weights (and on pre-SM80 devices).
TEST(Q8_0LinearPathTest, UnalignedAndPartialTileShapesMatchQuantizedOracle)
{
    struct Case
    {
        int32_t outFeatures;
        int32_t inFeatures;
        size_t weightOffset;
        bool mmaSupported;
    };
    for (Case const& c : {Case{37, 96, 0, false}, Case{37, 128, 2, false}, Case{45, 256, 0, true}})
    {
        PackedWeights const weights = makePackedWeights(c.outFeatures, c.inFeatures, 3U);
        Activations const input = roundActivations(makeActivations(20, c.inFeatures, 7U), DataType::kHALF);
        std::string const shape = "N=" + std::to_string(c.outFeatures) + " K=" + std::to_string(c.inFeatures)
            + " offset=" + std::to_string(c.weightOffset);
        for (int32_t rows : {1, 3, 8, 9, 20})
        {
            std::vector<float> output;
            std::vector<uint8_t> workspace;
            runLinear(weights, input, rows, DataType::kFLOAT, output, c.weightOffset, Q8_0Path::kAuto, &workspace);
            ASSERT_FALSE(HasFatalFailure());
            expectMatchesOracle(
                weights, input, rows, DataType::kFLOAT, output, workspace, shape + " M=" + std::to_string(rows));
        }
        std::vector<float> mmaOutput;
        std::vector<uint8_t> mmaWorkspace;
        cudaError_t mmaStatus = cudaErrorUnknown;
        runLinearStatus(
            weights, input, 9, DataType::kFLOAT, mmaOutput, c.weightOffset, Q8_0Path::kMma, mmaStatus, &mmaWorkspace);
        ASSERT_FALSE(HasFatalFailure());
        bool const mmaRuns = c.mmaSupported && deviceSupportsMma();
        EXPECT_EQ(mmaStatus, mmaRuns ? cudaSuccess : cudaErrorNotSupported) << shape;
        if (mmaRuns)
        {
            expectMatchesOracle(weights, input, 9, DataType::kFLOAT, mmaOutput, mmaWorkspace, shape + " M=9 path=mma");
        }
    }
}
