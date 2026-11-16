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

#if defined(ENABLE_EXPERIMENTAL_WEIGHTLESS) && !IS_TRT_RTX

#include "common/logger.h"
#include "common/trtInterop.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/weight/refitWeights.h"
#include "scratchDir.h"
#include "testUtils.h"

#include <NvInfer.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace
{

class RefitEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mDir = makeScratchDir("refitEngineTests");
        mRuntime.reset(nvinfer1::createInferRuntime(TrtLogger::instance()));
        ASSERT_NE(mRuntime, nullptr);
    }

    void TearDown() override
    {
        mRuntime.reset();
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mDir);
    }

    //! Build a matrix multiply with either a true null placeholder or ordinary populated refittable weights.
    std::unique_ptr<nvinfer1::IHostMemory> buildPlan(bool stripped)
    {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLogger::instance()));
        ELLM_CHECK(builder != nullptr, "Test builder creation failed");
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        ELLM_CHECK(network != nullptr && config != nullptr, "Test network creation failed");
        std::vector<__half> values = matrixValues();
        nvinfer1::Weights const weights{
            nvinfer1::DataType::kHALF, stripped ? nullptr : values.data(), static_cast<int64_t>(values.size())};
        auto* constant = network->addConstant(nvinfer1::Dims2{2, 2}, weights);
        ELLM_CHECK(constant != nullptr, "Internal TRT rejected null constant");
        ELLM_CHECK(network->setWeightsName(constant->getWeights(), kWeightName), "Test weight naming failed");
        ELLM_CHECK(network->markWeightsRefittable(kWeightName), "Test refit marking failed");
        auto* input = network->addInput("input", nvinfer1::DataType::kHALF, nvinfer1::Dims2{1, 2});
        ELLM_CHECK(input != nullptr, "Test input creation failed");
        auto* multiply = network->addMatrixMultiply(
            *input, nvinfer1::MatrixOperation::kNONE, *constant->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
        ELLM_CHECK(multiply != nullptr, "Test matrix multiply creation failed");
        multiply->getOutput(0)->setName("output");
        network->markOutput(*multiply->getOutput(0));
        config->setFlag(stripped ? nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL : nvinfer1::BuilderFlag::kREFIT);
        if (stripped)
        {
            config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
        }
        config->setBuilderOptimizationLevel(3);
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        ELLM_CHECK(plan != nullptr, "Test engine build failed");
        return plan;
    }

    //! Materialize the source matrix in independently owned device memory.
    std::unique_ptr<RefitWeightSet> makeWeights()
    {
        auto result = std::make_unique<RefitWeightSet>();
        result->storage = Tensor(Coords{2, 2}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        copyHostToDevice(result->storage, matrixValues());
        result->weights.emplace_back(
            result->storage.rawPointer(), Coords{2, 2}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, kWeightName);
        return result;
    }

    //! Persist the serialized artifact for the production executor factory.
    std::filesystem::path writePlan(nvinfer1::IHostMemory const& plan)
    {
        auto const path = mDir / "test.engine";
        std::ofstream file(path, std::ios::binary);
        file.write(static_cast<char const*>(plan.data()), plan.size());
        return path;
    }

    //! Execute the test matrix multiply after the caller has released refit source storage.
    std::vector<float> infer(nvinfer1::ICudaEngine& engine)
    {
        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine.createExecutionContext());
        ELLM_CHECK(context != nullptr, "Test context creation failed");
        Tensor input(Coords{1, 2}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        Tensor output(Coords{1, 2}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        copyHostToDevice(input, std::vector<__half>{__float2half(1.0F), __float2half(2.0F)});
        ELLM_CHECK(context->setTensorAddress("input", input.rawPointer()), "Test input binding failed");
        ELLM_CHECK(context->setTensorAddress("output", output.rawPointer()), "Test output binding failed");
        ELLM_CHECK(context->enqueueV3(mStream), "Test inference failed");
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> result;
        for (__half value : copyDeviceToHost<__half>(output))
        {
            result.push_back(__half2float(value));
        }
        return result;
    }

    static std::vector<__half> matrixValues()
    {
        return {__float2half(1.0F), __float2half(2.0F), __float2half(3.0F), __float2half(4.0F)};
    }

    static constexpr char const* kWeightName = "linear.weight";
    cudaStream_t mStream{};
    std::filesystem::path mDir;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
};

TEST_F(RefitEngineTest, RefitsBeforeContextAndSurvivesSourceRelease)
{
    // Given a freshly deserialized stripped engine, when refit storage is released, then inference uses the supplied
    // values.
    auto plan = buildPlan(true);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    ASSERT_NE(engine, nullptr);
    auto weights = makeWeights();
    refitEngine(*engine, *weights, mStream);
    weights.reset();
    std::vector<float> const expected{5.0F, 11.0F};
    EXPECT_EQ(infer(*engine), expected);
}

TEST_F(RefitEngineTest, DraftFactoriesRefitBeforeContextCreation)
{
    // Given a stripped draft engine and checkpoint weights, when loaded through its runtime factory,
    // then refit completes before context creation and inference uses the weights after temporary storage is released.
    auto plan = buildPlan(true);
    auto const path = writePlan(*plan);
    for (auto const mode : {SpecDecodeMode::kEAGLE, SpecDecodeMode::kMTP, SpecDecodeMode::kDFlash,
             SpecDecodeMode::kJetSpec, SpecDecodeMode::kDSpark, SpecDecodeMode::kGemma4MTP})
    {
        SCOPED_TRACE(static_cast<int32_t>(mode));
        DeploymentConfig deployment;
        deployment.base.specDecodeType = mode;
        deployment.draft.emplace();
        deployment.specConfig.emplace();
        auto executor = EngineExecutor::createForDraft(path, deployment, {}, {makeWeights(), mStream});
        EXPECT_EQ(infer(const_cast<nvinfer1::ICudaEngine&>(executor->getEngine())), (std::vector<float>{5.0F, 11.0F}));
    }
}

TEST_F(RefitEngineTest, ComponentRefitReleasesTemporaryStorageBeforeInference)
{
    // Given a stripped component engine and checkpoint weights, when loaded through the shared refit path,
    // then refit completes before context creation and inference uses the weights after temporary storage is released.
    auto plan = buildPlan(true);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    refitEngineIfNeeded(*engine, {makeWeights(), mStream});
    EXPECT_EQ(infer(*engine), (std::vector<float>{5.0F, 11.0F}));
}

TEST_F(RefitEngineTest, RejectsIncompleteAndIncompatibleWeightSets)
{
    // Given a stripped engine, when names or prototypes disagree, then refit rejects the manifest before inference.
    auto plan = buildPlan(true);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    ASSERT_NE(engine, nullptr);
    RefitWeightSet empty;
    EXPECT_THAT([&] { refitEngine(*engine, empty, mStream); },
        ThrowsMessage<std::runtime_error>(HasSubstr("no refit weights")));
    auto weights = makeWeights();
    EXPECT_THAT([&] { refitEngine(*engine, *weights, nullptr); },
        ThrowsMessage<std::runtime_error>(HasSubstr("non-default CUDA stream")));
    for (auto const& shape : std::vector<Coords>{Coords{2}, Coords{2, 2}})
    {
        weights->weights.clear();
        weights->weights.emplace_back(weights->storage.rawPointer(), shape, DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, shape.volume() == 2 ? kWeightName : "unknown.weight");
        EXPECT_THAT([&] { refitEngine(*engine, *weights, mStream); },
            ThrowsMessage<std::runtime_error>(
                HasSubstr(shape.volume() == 2 ? "incompatible storage, dtype or count" : "missing refit weight")));
    }
    weights->weights.clear();
    weights->weights.emplace_back(
        weights->storage.rawPointer(), Coords{2, 2}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, kWeightName);
    EXPECT_THAT([&] { refitEngine(*engine, *weights, mStream); },
        ThrowsMessage<std::runtime_error>(HasSubstr("incompatible storage, dtype or count")));
    weights->weights.emplace_back(
        weights->storage.rawPointer(), Coords{2, 2}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "extra.weight");
    EXPECT_THAT([&] { refitEngine(*engine, *weights, mStream); },
        ThrowsMessage<std::runtime_error>(HasSubstr("count differs")));
}

TEST_F(RefitEngineTest, PackedFp4RefitValidatesByteCountAndExecutes)
{
    // Packed checkpoint storage must supply two FP4 elements per byte.
    cudaDeviceProp properties{};
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 10)
    {
        GTEST_SKIP() << "Native FP4 requires Blackwell or later";
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLogger::instance()));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* input = network->addInput("input", nvinfer1::DataType::kHALF, nvinfer1::Dims2{1, 32});
    auto* weight = network->addConstant({2, {32, 32}}, {nvinfer1::DataType::kFP4, nullptr, 1024});
    ASSERT_NE(weight, nullptr);
    ASSERT_TRUE(network->setWeightsName(weight->getWeights(), kWeightName));
    std::vector<__half> scales(64, __float2half(1.0F));
    auto* scale = network->addConstant({2, {32, 2}}, {nvinfer1::DataType::kHALF, scales.data(), 64});
    auto* dequantize = network->addDequantize(*weight->getOutput(0), *scale->getOutput(0), nvinfer1::DataType::kHALF);
    ASSERT_NE(dequantize, nullptr);
    dequantize->setAxis(1);
    auto* multiply = network->addMatrixMultiply(
        *input, nvinfer1::MatrixOperation::kNONE, *dequantize->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    multiply->getOutput(0)->setName("output");
    network->markOutput(*multiply->getOutput(0));
    config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    config->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(plan, nullptr);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    ASSERT_NE(engine, nullptr);
    RefitWeightSet weights;
    weights.storage = Tensor(Coords{512}, DeviceType::kGPU, trt_edgellm::DataType::kUINT8);
    copyHostToDevice(weights.storage, std::vector<uint8_t>(512, 0x22));
    weights.weights.emplace_back(
        weights.storage.rawPointer(), Coords{511}, DeviceType::kGPU, trt_edgellm::DataType::kUINT8, kWeightName);
    EXPECT_THAT([&] { refitEngine(*engine, weights, mStream); },
        ThrowsMessage<std::runtime_error>(HasSubstr("incompatible storage, dtype or count")));
    weights.weights.clear();
    weights.weights.emplace_back(
        weights.storage.rawPointer(), Coords{32, 16}, DeviceType::kGPU, trt_edgellm::DataType::kUINT8, kWeightName);
    refitEngine(*engine, weights, mStream);
    weights = RefitWeightSet{};
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    Tensor x(Coords{1, 32}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    Tensor y(Coords{1, 32}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    copyHostToDevice(x, std::vector<__half>(32, __float2half(1.0F)));
    ASSERT_TRUE(context->setTensorAddress("input", x.rawPointer()));
    ASSERT_TRUE(context->setTensorAddress("output", y.rawPointer()));
    ASSERT_TRUE(context->enqueueV3(mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    for (__half value : copyDeviceToHost<__half>(y))
    {
        EXPECT_FLOAT_EQ(__half2float(value), 32.0F);
    }
}

TEST_F(RefitEngineTest, FactoryRejectsStrippedButAcceptsPopulatedRefittableEngine)
{
    // Given stripped and populated refittable artifacts, when loading without refit, then only stripped artifacts fail.
    auto stripped = buildPlan(true);
    auto strippedEngine
        = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(stripped->data(), stripped->size()));
    ASSERT_NE(strippedEngine, nullptr);
    EXPECT_FALSE(strippedEngine->isRefittable());
    EXPECT_GE(strippedEngine->getEngineStat(nvinfer1::EngineStat::kSTRIPPED_WEIGHTS_SIZE), 0);
    auto const strippedPath = writePlan(*stripped);
    EXPECT_THAT([&] { EngineExecutor::createWithRegistry(strippedPath, {}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Stripped engine requires")));
    EngineRefitOptions refit{makeWeights(), mStream};
    auto loaded = EngineExecutor::createWithRegistry(strippedPath, {}, std::move(refit));
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->getNumIOTensors(), 2);
    std::vector<float> const expected{5.0F, 11.0F};
    EXPECT_EQ(infer(const_cast<nvinfer1::ICudaEngine&>(loaded->getEngine())), expected);
    auto populated = buildPlan(false);
    auto const populatedPath = writePlan(*populated);
    auto ordinary = EngineExecutor::createWithRegistry(populatedPath, {});
    ASSERT_NE(ordinary, nullptr);
    EXPECT_TRUE(ordinary->getEngine().isRefittable());
}

} // namespace

#endif
