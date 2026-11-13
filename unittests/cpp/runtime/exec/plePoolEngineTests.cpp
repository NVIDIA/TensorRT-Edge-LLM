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

#include "builder/builderUtils.h"
#include "common/checkMacros.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "runtime/exec/engineExecutor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr char kPOOL_NAME[]{"ple_cache_pool"};
constexpr char kMAP_NAME[]{"ple_row_map"};
constexpr char kOUTPUT_NAME[]{"ple_embeddings"};
constexpr int64_t kROW_WIDTH{160};
constexpr int64_t kHEADS{16};
constexpr int64_t kMAX_BATCH{2};
constexpr int64_t kMAX_SEQ{8};
constexpr float kSCALE{1.1875F};

float rowValue(int32_t slot, int64_t column, int32_t epoch)
{
    return slot == 0 ? 0.0F : static_cast<float>((slot % 31 + column + epoch * 7) % 33 - 16) * 0.5F;
}

class PlePoolEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        char const* const fixture = std::getenv("EDGELLM_PLE_PROBE_ONNX");
        if (fixture == nullptr)
        {
            GTEST_SKIP() << "Set EDGELLM_PLE_PROBE_ONNX to a scale=1.1875 exported PLE pool fixture.";
        }
        mOnnxPath = fixture;
        ASSERT_TRUE(std::filesystem::is_regular_file(mOnnxPath)) << mOnnxPath;
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mPluginLibrary = loadEdgellmPluginLib();
        ASSERT_NE(mPluginLibrary, nullptr);
        recordDeviceMemory("init");
    }

    void recordDeviceMemory(std::string const& phase)
    {
        size_t freeBytes{};
        size_t totalBytes{};
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        RecordProperty(phase + "_free_bytes", std::to_string(freeBytes));
        std::cout << "PLE_PROBE_DEVICE_MEMORY phase=" << phase << " free_bytes=" << freeBytes
                  << " total_bytes=" << totalBytes << '\n';
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            mExecutor.reset();
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
            mStream = nullptr;
        }
    }

    void buildFixture()
    {
        builder::applyCompileWorkarounds();
        auto [trtBuilder, network] = builder::createBuilderAndNetwork();
        ASSERT_NE(trtBuilder, nullptr);
        ASSERT_NE(network, nullptr);
        auto parser = builder::parseOnnxModel(network.get(), mOnnxPath.string());
        ASSERT_NE(parser, nullptr) << "Production ONNX parser rejected " << mOnnxPath;
        ASSERT_EQ(network->getNbInputs(), 2);
        ASSERT_EQ(network->getNbOutputs(), 1);
        nvinfer1::ITensor* pool = nullptr;
        nvinfer1::ITensor* rowMap = nullptr;
        for (int32_t i = 0; i < network->getNbInputs(); ++i)
        {
            auto* input = network->getInput(i);
            if (std::string(input->getName()) == kPOOL_NAME)
            {
                pool = input;
            }
            if (std::string(input->getName()) == kMAP_NAME)
            {
                rowMap = input;
            }
            input->setAllowedFormats(1U << static_cast<uint32_t>(nvinfer1::TensorFormat::kLINEAR));
        }
        ASSERT_NE(pool, nullptr);
        ASSERT_NE(rowMap, nullptr);
        mPoolType = fromTrt(pool->getType());
        ASSERT_TRUE(mPoolType == trt_edgellm::DataType::kFP8 || mPoolType == trt_edgellm::DataType::kHALF
            || mPoolType == trt_edgellm::DataType::kBF16);
        ASSERT_EQ(pool->getDimensions().nbDims, 2);
        mRowWidth = pool->getDimensions().d[1];
        ASSERT_GT(mRowWidth, 0);
        mRowBytes = mRowWidth * (mPoolType == trt_edgellm::DataType::kFP8 ? 1 : 2);
        auto metadataPath = mOnnxPath;
        metadataPath.replace_extension(".json");
        std::ifstream metadataFile(metadataPath);
        mMetadata = nlohmann::json::parse(metadataFile);
        mScale = mMetadata.at("scale").get<float>();
        mDynamicCapacity = pool->getDimensions().d[0] == -1;
        mMaximumCapacity = mMetadata.at("capacity").get<int64_t>();
        mCapacity = mDynamicCapacity ? std::min<int64_t>(64, mMaximumCapacity) : pool->getDimensions().d[0];
        ASSERT_GE(mCapacity, 4);
        ASSERT_LE(mCapacity, std::numeric_limits<int32_t>::max());
        ASSERT_EQ(rowMap->getType(), nvinfer1::DataType::kINT32);
        int32_t const mapRank = rowMap->getDimensions().nbDims;
        ASSERT_TRUE(mapRank == 2 || mapRank == 3);
        mRagged = mapRank == 2;
        mHeads = rowMap->getDimensions().d[mapRank - 1];
        ASSERT_GT(mHeads, 0);
        auto* output = network->getOutput(0);
        ASSERT_EQ(std::string(output->getName()), kOUTPUT_NAME);
        ASSERT_EQ(output->getDimensions().nbDims, mapRank);
        ASSERT_EQ(output->getDimensions().d[mapRank - 1], mHeads * mRowWidth);
        mOutputType = fromTrt(output->getType());
        ASSERT_TRUE(mOutputType == trt_edgellm::DataType::kHALF || mOutputType == trt_edgellm::DataType::kBF16);

        auto config = builder::createBuilderConfig(trtBuilder.get());
        ASSERT_NE(config, nullptr);
        config->setProfileStream(mStream);
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        auto* profile = trtBuilder->createOptimizationProfile();
        ASSERT_NE(profile, nullptr);
        auto const poolShape = pool->getDimensions();
        ASSERT_TRUE(builder::setOptimizationProfile(profile, kPOOL_NAME,
            mDynamicCapacity ? builder::createDims({1, mRowWidth}) : poolShape,
            mDynamicCapacity ? builder::createDims({mCapacity, mRowWidth}) : poolShape,
            mDynamicCapacity ? builder::createDims({mMaximumCapacity, mRowWidth}) : poolShape));
        ASSERT_TRUE(builder::setOptimizationProfile(profile, kMAP_NAME,
            mRagged ? builder::createDims({1, mHeads}) : builder::createDims({1, 1, mHeads}),
            mRagged ? builder::createDims({3, mHeads}) : builder::createDims({1, 3, mHeads}),
            mRagged ? builder::createDims({kMAX_BATCH * kMAX_SEQ, mHeads})
                    : builder::createDims({kMAX_BATCH, kMAX_SEQ, mHeads})));
        ASSERT_GE(config->addOptimizationProfile(profile), 0);
        mEnginePath = mOnnxPath;
        mEnginePath.replace_extension(".engine");
        ASSERT_TRUE(
            builder::buildAndSerializeEngine(trtBuilder.get(), network.get(), config.get(), mEnginePath.string()));

        TensorRegistry registry;
        // The executor binds the pool's actual shape from TensorMap; C is not a request sequence dimension.
        if (!mDynamicCapacity)
        {
            registry.addTensor({kPOOL_NAME, TensorIO::kInput, mPoolType, {fixed(mCapacity), fixed(mRowWidth)}});
        }
        if (mRagged)
        {
            registry.addTensor({kMAP_NAME, TensorIO::kInput, trt_edgellm::DataType::kINT32,
                {sym(&InferenceDims::seqLen), fixed(mHeads)}});
            registry.addTensor({kOUTPUT_NAME, TensorIO::kOutput, mOutputType,
                {sym(&InferenceDims::seqLen), fixed(mHeads * mRowWidth)}});
        }
        else
        {
            registry.addTensor({kMAP_NAME, TensorIO::kInput, trt_edgellm::DataType::kINT32,
                {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(mHeads)}});
            registry.addTensor({kOUTPUT_NAME, TensorIO::kOutput, mOutputType,
                {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(mHeads * mRowWidth)}});
        }
        mExecutor = EngineExecutor::createWithRegistry(mEnginePath, std::move(registry));
        ASSERT_NE(mExecutor, nullptr);
        auto const& engine = mExecutor->getEngine();
        ASSERT_EQ(fromTrt(engine.getTensorDataType(kPOOL_NAME)), mPoolType);
        ASSERT_EQ(engine.getTensorDataType(kMAP_NAME), nvinfer1::DataType::kINT32);
        ASSERT_EQ(fromTrt(engine.getTensorDataType(kOUTPUT_NAME)), mOutputType);
        for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
        {
            char const* name = engine.getIOTensorName(i);
            EXPECT_EQ(engine.getTensorFormat(name), nvinfer1::TensorFormat::kLINEAR) << name;
            std::cout << "PLE_PROBE_IO name=" << name
                      << " dtype=" << static_cast<int32_t>(engine.getTensorDataType(name))
                      << " format=" << engine.getTensorFormatDesc(name)
                      << " shape=" << trt_edgellm::rt::toCoords(engine.getTensorShape(name)).formatString() << '\n';
        }
        auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine.createEngineInspector());
        ASSERT_NE(inspector, nullptr);
        std::ofstream inspection(mEnginePath.string() + ".json");
        ASSERT_TRUE(inspection.good());
        char const* const engineInformation = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        ASSERT_NE(engineInformation, nullptr);
        inspection << engineInformation;
        ASSERT_TRUE(inspection.good());
        int64_t const contextBytes = mExecutor->getRequiredContextMemorySize();
        RecordProperty("pool_bytes", std::to_string(mCapacity * mRowBytes));
        RecordProperty("context_bytes", std::to_string(contextBytes));
        std::cout << "PLE_PROBE_MEMORY pool_bytes=" << mCapacity * mRowBytes << " context_bytes=" << contextBytes
                  << " engine_bytes=" << std::filesystem::file_size(mEnginePath)
                  << " workspace_budget=" << config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE) << '\n';
        recordDeviceMemory("engine");

        mContextMemory = Tensor({std::max<int64_t>(contextBytes, 1)}, DeviceType::kGPU, trt_edgellm::DataType::kINT8);
        ASSERT_TRUE(mExecutor->setContextMemory(mContextMemory));
        mPool = Tensor({mCapacity, mRowWidth}, DeviceType::kGPU, mPoolType);
        Coords const mapShape = mRagged ? Coords{kMAX_BATCH * kMAX_SEQ, mHeads} : Coords{kMAX_BATCH, kMAX_SEQ, mHeads};
        Coords const outputShape = mRagged ? Coords{kMAX_BATCH * kMAX_SEQ, mHeads * mRowWidth}
                                           : Coords{kMAX_BATCH, kMAX_SEQ, mHeads * mRowWidth};
        mRowMap = Tensor(mapShape, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
        mOutput = Tensor(outputShape, DeviceType::kGPU, mOutputType);
        mHostMap = Tensor(mapShape, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mHostOutput = Tensor(outputShape, DeviceType::kCPU, mOutputType);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(mPool.rawPointer()) % 256, 0);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(mRowMap.rawPointer()) % 256, 0);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(mOutput.rawPointer()) % 256, 0);
        mBindings.set(kPOOL_NAME, mPool);
        mBindings.set(kMAP_NAME, mRowMap);
        mBindings.set(kOUTPUT_NAME, mOutput);
        recordDeviceMemory("allocated");
    }

    float storedValue(int32_t slot, int64_t column, int32_t epoch) const
    {
        float value = rowValue(slot, column, epoch);
        if (mPoolType == trt_edgellm::DataType::kFP8)
        {
            return static_cast<float>(__nv_fp8_e4m3(value));
        }
        if (slot != 0)
        {
            value += static_cast<float>(column % 7) / 256.0F;
        }
        return mPoolType == trt_edgellm::DataType::kHALF ? __half2float(__float2half_rn(value))
                                                         : __bfloat162float(__float2bfloat16_rn(value));
    }

    void runCase(int64_t batch, int64_t seq, int32_t epoch, bool capture)
    {
        SCOPED_TRACE("B=" + std::to_string(batch) + " S=" + std::to_string(seq) + " epoch=" + std::to_string(epoch));
        int64_t const rows = batch * seq * mHeads;
        std::vector<int32_t> slots{static_cast<int32_t>(mCapacity - 1), 0, 1, static_cast<int32_t>(mCapacity / 2), 2};
        for (int64_t const byteBoundary : {1LL << 31, 1LL << 32})
        {
            int64_t const crossingRow = byteBoundary / mRowBytes;
            for (int64_t const slot : {crossingRow - 1, crossingRow, crossingRow + 1})
            {
                if (slot >= 0 && slot < mCapacity)
                {
                    slots.push_back(static_cast<int32_t>(slot));
                }
            }
        }
        std::sort(slots.begin(), slots.end());
        slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
        auto* ids = mHostMap.dataPointer<int32_t>();
        for (int64_t i = 0; i < rows; ++i)
        {
            ids[i] = slots[(i + epoch) % slots.size()];
        }
        Tensor hostRows({static_cast<int64_t>(slots.size()), mRowWidth}, DeviceType::kCPU, mPoolType);
        auto* weights = static_cast<char*>(hostRows.rawPointer());
        for (size_t row = 0; row < slots.size(); ++row)
        {
            for (int64_t col = 0; col < mRowWidth; ++col)
            {
                int64_t const index = row * mRowWidth + col;
                float const value = storedValue(slots[row], col, epoch);
                if (mPoolType == trt_edgellm::DataType::kFP8)
                {
                    static_cast<__nv_fp8_e4m3*>(hostRows.rawPointer())[index] = __nv_fp8_e4m3(value);
                }
                else if (mPoolType == trt_edgellm::DataType::kHALF)
                {
                    static_cast<half*>(hostRows.rawPointer())[index] = __float2half_rn(value);
                }
                else
                {
                    static_cast<__nv_bfloat16*>(hostRows.rawPointer())[index] = __float2bfloat16_rn(value);
                }
            }
            auto* target = static_cast<char*>(mPool.rawPointer()) + static_cast<int64_t>(slots[row]) * mRowBytes;
            CUDA_CHECK(cudaMemcpyAsync(target, weights + row * mRowBytes, mRowBytes, cudaMemcpyHostToDevice, mStream));
        }
        CUDA_CHECK(cudaMemcpyAsync(mRowMap.rawPointer(), ids, rows * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        InferenceDims const dims{batch, mRagged ? batch * seq : seq, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, batch + 1, 0};
        ASSERT_TRUE(mExecutor->prepare(0, dims, mBindings, mStream));
        if (capture)
        {
            ASSERT_TRUE(mExecutor->captureGraph(mStream));
        }
        ASSERT_TRUE(mExecutor->execute(mStream));
        int64_t const outputElements = rows * mRowWidth;
        CUDA_CHECK(cudaMemcpyAsync(mHostOutput.rawPointer(), mOutput.rawPointer(), outputElements * sizeof(half),
            cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        recordDeviceMemory("run_" + std::to_string(epoch));
        bool exercisedRounding = false;
        for (int64_t i = 0; i < outputElements; ++i)
        {
            float const scaled = storedValue(ids[i / mRowWidth], i % mRowWidth, epoch) * mScale;
            float const expected = mOutputType == trt_edgellm::DataType::kHALF
                ? __half2float(__float2half_rn(scaled))
                : __bfloat162float(__float2bfloat16_rn(scaled));
            exercisedRounding |= expected != scaled;
            float const actual = mOutputType == trt_edgellm::DataType::kHALF
                ? __half2float(static_cast<half const*>(mHostOutput.rawPointer())[i])
                : __bfloat162float(static_cast<__nv_bfloat16 const*>(mHostOutput.rawPointer())[i]);
            ASSERT_EQ(actual, expected) << "element=" << i << " slot=" << ids[i / mRowWidth];
        }
        if (mPoolType == trt_edgellm::DataType::kFP8 && mOutputType == trt_edgellm::DataType::kBF16
            && mScale == 1.1875F)
        {
            EXPECT_TRUE(exercisedRounding);
        }
    }

    std::filesystem::path mOnnxPath;
    std::filesystem::path mEnginePath;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<void, DlDeleter> mPluginLibrary;
    std::unique_ptr<EngineExecutor> mExecutor;
    int64_t mCapacity{};
    int64_t mMaximumCapacity{};
    int64_t mRowWidth{kROW_WIDTH};
    int64_t mRowBytes{kROW_WIDTH};
    int64_t mHeads{kHEADS};
    float mScale{kSCALE};
    trt_edgellm::DataType mPoolType{trt_edgellm::DataType::kFP8};
    bool mDynamicCapacity{false};
    bool mRagged{false};
    nlohmann::json mMetadata;
    trt_edgellm::DataType mOutputType{trt_edgellm::DataType::kHALF};
    Tensor mContextMemory;
    Tensor mPool;
    Tensor mRowMap;
    Tensor mOutput;
    Tensor mHostMap;
    Tensor mHostOutput;
    TensorMap mBindings;
};

TEST_F(PlePoolEngineTest, ExportedGatherDequantizeAndGraphReplay)
{
    ASSERT_NO_FATAL_FAILURE(buildFixture());
    ASSERT_NO_FATAL_FAILURE(runCase(1, 1, 0, false));
    ASSERT_NO_FATAL_FAILURE(runCase(2, 3, 1, true));
    ASSERT_NO_FATAL_FAILURE(runCase(2, 3, 2, false));
    ASSERT_NO_FATAL_FAILURE(runCase(1, 5, 3, true));
    ASSERT_NO_FATAL_FAILURE(runCase(1, 5, 4, false));
    ASSERT_NO_FATAL_FAILURE(runCase(2, 3, 5, false));
    if (mDynamicCapacity)
    {
        std::vector<int64_t> capacities{8, 64, ((1LL << 31) - 1) / mRowBytes, (1LL << 31) / mRowBytes + 1,
            (1LL << 32) / mRowBytes + 1, mMaximumCapacity};
        for (int64_t capacity : capacities)
        {
            if (capacity > mMaximumCapacity)
            {
                continue;
            }
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            mCapacity = capacity;
            mPool = Tensor({mCapacity, mRowWidth}, DeviceType::kGPU, mPoolType);
            mBindings.set(kPOOL_NAME, mPool);
            std::cout << "PLE_DYNAMIC_CAPACITY slots=" << capacity << " engine=" << mEnginePath.string() << '\n';
            ASSERT_NO_FATAL_FAILURE(runCase(2, 8, 6, true));
            ASSERT_NO_FATAL_FAILURE(runCase(2, 8, 7, false));
        }
    }
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
