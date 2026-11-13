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

#include "action/gr00tPolicyRunner.h"

#include "action/gr00tKernels.h"

#include "common/cudaUtils.h"
#include "common/gr00tBindings.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

using trt_edgellm::DataType;
using nvinfer1::Dims;

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

constexpr int32_t kModulationRank = 3; //!< [steps, sites, modulationDim]
constexpr int32_t kTimeBiasRank = 2;   //!< [steps, innerDim]

int32_t requireInt(nlohmann::json const& cfg, char const* key, char const* component)
{
    if (!cfg.contains(key))
    {
        throw std::runtime_error(std::string("GR00T ") + component + " config.json is missing required key: " + key);
    }
    return cfg.at(key).get<int32_t>();
}

std::vector<int32_t> requireIntArray(nlohmann::json const& cfg, char const* key, char const* component)
{
    if (!cfg.contains(key))
    {
        throw std::runtime_error(std::string("GR00T ") + component + " config.json is missing required key: " + key);
    }
    return cfg.at(key).get<std::vector<int32_t>>();
}

rt::Tensor& findTensor(std::vector<rt::Tensor>& tensors, std::string const& name)
{
    auto const match = std::find_if(
        tensors.begin(), tensors.end(), [&name](rt::Tensor const& tensor) { return tensor.getName() == name; });
    if (match == tensors.end())
    {
        throw std::runtime_error("GR00T constant table is missing tensor: " + name);
    }
    return *match;
}

} // namespace

Gr00tPolicyRunner::Gr00tPolicyRunner(std::string const& engineDir, cudaStream_t stream)
{
    parseConfigs(engineDir);
    loadComponent(engineDir, "vl_backbone", mBackbone);
    loadComponent(engineDir, "action", mAction);
    loadConstantTables(engineDir, stream);
    allocateTensors();
}

Gr00tPolicyRunner::~Gr00tPolicyRunner() noexcept
{
    invalidateGraph();
}

void Gr00tPolicyRunner::invalidateGraph() noexcept
{
    if (mGraphExec != nullptr)
    {
        cudaGraphExecDestroy(mGraphExec);
        mGraphExec = nullptr;
    }
    if (mGraph != nullptr)
    {
        cudaGraphDestroy(mGraph);
        mGraph = nullptr;
    }
    mGraphReady = false;
}

void Gr00tPolicyRunner::loadComponent(std::string const& engineDir, std::string const& component, Engine& out)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / component / (component + ".engine");
    out.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(TrtLogger::instance()));
    if (!out.runtime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime for GR00T " + component);
    }
    out.engine = deserializeCudaEngineFromFile(*out.runtime, enginePath.string());
    if (!out.engine)
    {
        throw std::runtime_error("Failed to load GR00T engine: " + enginePath.string());
    }
    out.context = std::unique_ptr<nvinfer1::IExecutionContext>(out.engine->createExecutionContext());
    if (!out.context)
    {
        throw std::runtime_error("Failed to create execution context for GR00T " + component);
    }
    setNonBlockingAuxStreams(out.context.get(), out.engine.get(), out.auxStreams);
    LOG_INFO("Loaded GR00T %s engine: %s", component.c_str(), enginePath.string().c_str());
}

void Gr00tPolicyRunner::parseConfigs(std::string const& engineDir)
{
    std::filesystem::path const root(engineDir);
    auto const backbone = loadJsonFile(root / "vl_backbone" / "config.json", "component config");
    auto const action = loadJsonFile(root / "action" / "config.json", "component config");
    // The structural checks below cannot tell two builds of the same architecture apart.
    BundleIdentity const bundle = requireIdentity(backbone, "vl_backbone config.json");
    requireSameIdentity(bundle, requireIdentity(action, "action config.json"), "action", "vl_backbone");
    requireSameIdentity(bundle,
        requireIdentity(loadJsonFile(root / "visual" / "config.json", "component config"), "visual config.json"),
        "visual", "vl_backbone");
    requireSameIdentity(bundle,
        requireIdentity(loadJsonFile(root / "action" / "policy.json", "policy contract"), "action/policy.json"),
        "action/policy.json", "vl_backbone");

    mConfig.numHiddenLayers = requireInt(backbone, "num_hidden_layers", "vl_backbone");
    mConfig.hiddenSize = requireInt(backbone, "hidden_size", "vl_backbone");
    mConfig.headDim = requireInt(backbone, "head_dim", "vl_backbone");
    mConfig.numDeepstackFeatures = requireInt(backbone, "num_deepstack_features", "vl_backbone");
    mConfig.ditNumHeads = requireInt(backbone, "dit_num_heads", "vl_backbone");
    mConfig.ditHeadDim = requireInt(backbone, "dit_head_dim", "vl_backbone");
    mConfig.textCrossBlocks = requireIntArray(backbone, "text_cross_blocks", "vl_backbone");
    mConfig.imageCrossBlocks = requireIntArray(backbone, "image_cross_blocks", "vl_backbone");
    mConfig.maxVlLen = backbone.at("builder_config").at("max_vl_len").get<int32_t>();
    mConfig.crossBlocks = mConfig.textCrossBlocks;
    mConfig.crossBlocks.insert(
        mConfig.crossBlocks.end(), mConfig.imageCrossBlocks.begin(), mConfig.imageCrossBlocks.end());
    std::sort(mConfig.crossBlocks.begin(), mConfig.crossBlocks.end());

    mConfig.actionHorizon = requireInt(action, "action_horizon", "action");
    mConfig.actionDim = requireInt(action, "action_dim", "action");
    mConfig.stateDim = requireInt(action, "state_dim", "action");
    mConfig.stateHistoryLength = requireInt(action, "state_history_length", "action");
    mConfig.numDenoiseSteps = requireInt(action, "num_denoise_steps", "action");
    mConfig.numModulationSites = requireInt(action, "num_modulation_sites", "action");
    mConfig.modulationDim = requireInt(action, "modulation_dim", "action");
    mConfig.innerDim = requireInt(action, "inner_dim", "action");
}

void Gr00tPolicyRunner::loadConstantTables(std::string const& engineDir, cudaStream_t stream)
{
    std::filesystem::path const actionDir = std::filesystem::path(engineDir) / "action";
    std::vector<rt::Tensor> modulation;
    if (!rt::safetensors::loadSafetensors(actionDir / "adanorm_modulation.safetensors", modulation, stream))
    {
        throw std::runtime_error("Failed to load GR00T adanorm_modulation.safetensors");
    }
    mModulation = std::move(findTensor(modulation, "adanorm_modulation"));

    std::vector<rt::Tensor> timeBias;
    if (!rt::safetensors::loadSafetensors(actionDir / "action_time_bias.safetensors", timeBias, stream))
    {
        throw std::runtime_error("Failed to load GR00T action_time_bias.safetensors");
    }
    mTimeBias = std::move(findTensor(timeBias, "action_time_bias"));

    rt::Coords const shape = mModulation.getShape();
    if (shape.getNumDims() != kModulationRank || shape[0] != mConfig.numDenoiseSteps
        || shape[1] != mConfig.numModulationSites || shape[2] != mConfig.modulationDim)
    {
        throw std::runtime_error("GR00T modulation table does not match the action contract");
    }
    rt::Coords const biasShape = mTimeBias.getShape();
    if (biasShape.getNumDims() != kTimeBiasRank || biasShape[0] != mConfig.numDenoiseSteps
        || biasShape[1] != mConfig.innerDim)
    {
        throw std::runtime_error("GR00T action time bias does not match the action contract");
    }
}

void Gr00tPolicyRunner::allocateTensors()
{
    // The profiles, not the contract, decide what the loaded engines accept: a
    // rebuild at a different --max-batch-size leaves the exported config alone.
    Dims const backboneMax
        = mBackbone.engine->getProfileShape(binding_names::kInputsEmbeds, 0, nvinfer1::OptProfileSelector::kMAX);
    Dims const actionMax
        = mAction.engine->getProfileShape(binding_names::kNoiseTrajectory, 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxBatch = static_cast<int32_t>(std::min(backboneMax.d[0], actionMax.d[0]));
    if (mMaxBatch < 1)
    {
        throw std::runtime_error("GR00T engines admit no request batch; rebuild them");
    }
    int32_t const maxVl = static_cast<int32_t>(backboneMax.d[1]);
    LOG_INFO("GR00T engines admit batch up to %d and a VL context up to %d", mMaxBatch, maxVl);

    int32_t const b = mMaxBatch;
    int32_t const h = mConfig.actionHorizon;
    int32_t const a = mConfig.actionDim;

    rt::Coords const crossShape{b, maxVl, mConfig.ditNumHeads, mConfig.ditHeadDim};
    mCrossKey.resize(mConfig.crossBlocks.size());
    mCrossValue.resize(mConfig.crossBlocks.size());
    for (size_t i = 0; i < mConfig.crossBlocks.size(); ++i)
    {
        mCrossKey[i] = rt::Tensor(crossShape, rt::DeviceType::kGPU, DataType::kHALF, "gr00t::crossKey");
        mCrossValue[i] = rt::Tensor(crossShape, rt::DeviceType::kGPU, DataType::kHALF, "gr00t::crossValue");
    }

    mTextMask = rt::Tensor({b, 1, 1, maxVl}, rt::DeviceType::kGPU, DataType::kBOOL, "gr00t::textMask");
    mImageMask = rt::Tensor({b, 1, 1, maxVl}, rt::DeviceType::kGPU, DataType::kBOOL, "gr00t::imageMask");
    mMaskHost = rt::Tensor({2, b, 1, 1, maxVl}, rt::DeviceType::kCPU, DataType::kBOOL, "gr00t::maskHost");

    mState = rt::Tensor(
        {b, mConfig.stateHistoryLength, mConfig.stateDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "gr00t::state");
    mStateHost = rt::Tensor(
        {b, mConfig.stateHistoryLength, mConfig.stateDim}, rt::DeviceType::kCPU, DataType::kFLOAT, "gr00t::stateHost");

    mTrajectory = rt::Tensor({b, h, a}, rt::DeviceType::kGPU, DataType::kFLOAT, "gr00t::trajectory");
    mTrajectoryHost = rt::Tensor({b, h, a}, rt::DeviceType::kCPU, DataType::kFLOAT, "gr00t::trajectoryHost");
    mVelocity = rt::Tensor({b, h, a}, rt::DeviceType::kGPU, DataType::kFLOAT, "gr00t::velocity");
}

void Gr00tPolicyRunner::setInitialNoise(std::vector<float> noise)
{
    mExternalNoise = std::move(noise);
}

void Gr00tPolicyRunner::setState(std::vector<float> state, cudaStream_t stream)
{
    size_t const expected = static_cast<size_t>(mActiveBatch) * mConfig.stateHistoryLength * mConfig.stateDim;
    if (state.size() != expected)
    {
        throw std::runtime_error("GR00T state size does not match [B, history, state_dim]");
    }
    std::copy(state.begin(), state.end(), mStateHost.dataPointer<float>());
    CUDA_CHECK(cudaMemcpyAsync(
        mState.rawPointer(), mStateHost.rawPointer(), expected * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Gr00tPolicyRunner::setCrossMasks(
    std::vector<uint8_t> const& textMask, std::vector<uint8_t> const& imageMask, cudaStream_t stream)
{
    size_t const expected = static_cast<size_t>(mActiveBatch) * mVlLen;
    if (textMask.size() != expected || imageMask.size() != expected)
    {
        throw std::runtime_error("GR00T cross masks must be [B, vlLen]");
    }
    auto* staged = mMaskHost.dataPointer<uint8_t>();
    std::copy(textMask.begin(), textMask.end(), staged);
    std::copy(imageMask.begin(), imageMask.end(), staged + expected);
    CUDA_CHECK(cudaMemcpyAsync(mTextMask.rawPointer(), staged, expected, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mImageMask.rawPointer(), staged + expected, expected, cudaMemcpyHostToDevice, stream));
}

void Gr00tPolicyRunner::prefill(rt::Tensor const& inputsEmbeds, std::vector<rt::Tensor> const& deepstackEmbeds,
    rt::Tensor const& ropeCosSin, rt::Tensor const& posIds, int32_t batch, int32_t vlLen, cudaStream_t stream)
{
    if (batch < 1 || batch > mMaxBatch)
    {
        throw std::runtime_error(
            "GR00T prefill batch " + std::to_string(batch) + " is outside [1, " + std::to_string(mMaxBatch) + "]");
    }
    if (static_cast<int32_t>(deepstackEmbeds.size()) != mConfig.numDeepstackFeatures)
    {
        throw std::runtime_error("GR00T prefill needs " + std::to_string(mConfig.numDeepstackFeatures)
            + " DeepStack tensors, got " + std::to_string(deepstackEmbeds.size()));
    }
    if (batch != mActiveBatch || vlLen != mVlLen)
    {
        invalidateGraph();
    }
    mActiveBatch = batch;
    mVlLen = vlLen;

    bool ok
        = mBackbone.context->setInputShape(binding_names::kInputsEmbeds, Dims{3, {batch, vlLen, mConfig.hiddenSize}});
    ok &= mBackbone.context->setInputShape(binding_names::kRopeCosSin, Dims{3, {batch, vlLen, mConfig.headDim}});
    ok &= mBackbone.context->setInputShape(binding_names::kAttentionPosId, Dims{2, {batch, vlLen}});
    ok &= mBackbone.context->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(inputsEmbeds.rawPointer()));
    ok &= mBackbone.context->setTensorAddress(binding_names::kRopeCosSin, const_cast<void*>(ropeCosSin.rawPointer()));
    ok &= mBackbone.context->setTensorAddress(binding_names::kAttentionPosId, const_cast<void*>(posIds.rawPointer()));
    for (int32_t i = 0; i < mConfig.numDeepstackFeatures; ++i)
    {
        std::string const name = binding_names::deepstackEmbeds(i);
        ok &= mBackbone.context->setInputShape(name.c_str(), Dims{3, {batch, vlLen, mConfig.hiddenSize}});
        ok &= mBackbone.context->setTensorAddress(name.c_str(), const_cast<void*>(deepstackEmbeds[i].rawPointer()));
    }
    for (size_t i = 0; i < mConfig.crossBlocks.size(); ++i)
    {
        int32_t const block = mConfig.crossBlocks[i];
        ok &= mBackbone.context->setTensorAddress(binding_names::crossKey(block).c_str(), mCrossKey[i].rawPointer());
        ok &= mBackbone.context->setTensorAddress(
            binding_names::crossValue(block).c_str(), mCrossValue[i].rawPointer());
    }
    if (!ok)
    {
        throw std::runtime_error("Failed to bind GR00T vl_backbone inputs");
    }
    if (!mBackbone.context->enqueueV3(stream))
    {
        throw std::runtime_error("GR00T vl_backbone execution failed");
    }
}

void Gr00tPolicyRunner::initializeNoise(int32_t batch)
{
    size_t const elems = static_cast<size_t>(batch) * mConfig.actionHorizon * mConfig.actionDim;
    float* host = mTrajectoryHost.dataPointer<float>();
    if (!mExternalNoise.empty())
    {
        if (mExternalNoise.size() != elems)
        {
            throw std::runtime_error("GR00T external noise size does not match [B, horizon, action_dim]");
        }
        std::copy(mExternalNoise.begin(), mExternalNoise.end(), host);
        return;
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (size_t i = 0; i < elems; ++i)
    {
        host[i] = normal(mNoiseGenerator);
    }
}

std::vector<float> Gr00tPolicyRunner::generate(cudaStream_t stream)
{
    if (mVlLen == 0)
    {
        throw std::runtime_error("GR00T generate() called before prefill()");
    }

    int32_t const b = mActiveBatch;
    int32_t const h = mConfig.actionHorizon;
    int32_t const a = mConfig.actionDim;
    size_t const elems = static_cast<size_t>(b) * static_cast<size_t>(h) * static_cast<size_t>(a);

    initializeNoise(b);
    CUDA_CHECK(cudaMemcpyAsync(
        mTrajectory.rawPointer(), mTrajectoryHost.rawPointer(), elems * sizeof(float), cudaMemcpyHostToDevice, stream));

    bool ok = mAction.context->setInputShape(binding_names::kNoiseTrajectory, Dims{3, {b, h, a}});
    ok &= mAction.context->setInputShape(
        binding_names::kState, Dims{3, {b, mConfig.stateHistoryLength, mConfig.stateDim}});
    ok &= mAction.context->setInputShape(binding_names::kTextMask, Dims{4, {b, 1, 1, mVlLen}});
    ok &= mAction.context->setInputShape(binding_names::kImageMask, Dims{4, {b, 1, 1, mVlLen}});
    ok &= mAction.context->setTensorAddress(binding_names::kNoiseTrajectory, mTrajectory.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kState, mState.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kTextMask, mTextMask.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kImageMask, mImageMask.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kVelocity, mVelocity.rawPointer());
    for (size_t i = 0; i < mConfig.crossBlocks.size(); ++i)
    {
        int32_t const block = mConfig.crossBlocks[i];
        Dims const shape{4, {b, mVlLen, mConfig.ditNumHeads, mConfig.ditHeadDim}};
        ok &= mAction.context->setInputShape(binding_names::crossKey(block).c_str(), shape);
        ok &= mAction.context->setInputShape(binding_names::crossValue(block).c_str(), shape);
        ok &= mAction.context->setTensorAddress(binding_names::crossKey(block).c_str(), mCrossKey[i].rawPointer());
        ok &= mAction.context->setTensorAddress(binding_names::crossValue(block).c_str(), mCrossValue[i].rawPointer());
    }
    if (!ok)
    {
        throw std::runtime_error("Failed to bind GR00T action engine inputs");
    }

    int64_t const modStepElems
        = static_cast<int64_t>(mConfig.numModulationSites) * static_cast<int64_t>(mConfig.modulationDim);
    float const dt = 1.0F / static_cast<float>(mConfig.numDenoiseSteps);

    auto enqueueLoop = [&](cudaStream_t s) {
        for (int32_t step = 0; step < mConfig.numDenoiseSteps; ++step)
        {
            // Rebinding before each enqueue is what bakes a distinct slice
            // address into each captured node; the base allocation never moves.
            if (!mAction.context->setTensorAddress(binding_names::kAdanormModulation,
                    mModulation.dataPointer<half>() + static_cast<int64_t>(step) * modStepElems))
            {
                return false;
            }
            if (!mAction.context->setTensorAddress(binding_names::kActionTimeBias,
                    mTimeBias.dataPointer<half>() + static_cast<int64_t>(step) * mConfig.innerDim))
            {
                return false;
            }
            if (!mAction.context->enqueueV3(s))
            {
                return false;
            }
            launchEulerUpdate(
                mTrajectory.dataPointer<float>(), mVelocity.dataPointer<float>(), dt, static_cast<int32_t>(elems), s);
        }
        return true;
    };

    if (mUseCudaGraph)
    {
        if (!mGraphReady)
        {
            // TensorRT initializes internal state lazily on the first enqueue and
            // that initialization synchronizes, which is illegal once capture has
            // begun. Warm up on the same stream first.
            if (!enqueueLoop(stream))
            {
                throw std::runtime_error("GR00T action warmup enqueue failed");
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
            bool const captured = enqueueLoop(stream);
            cudaGraph_t graph{nullptr};
            cudaError_t const endErr = cudaStreamEndCapture(stream, &graph);
            if (!captured || endErr != cudaSuccess || graph == nullptr)
            {
                LOG_WARNING("GR00T denoise-loop CUDA graph capture failed (%s); using plain enqueues",
                    cudaGetErrorString(endErr));
                mUseCudaGraph = false;
            }
            else
            {
                mGraph = graph;
                CUDA_CHECK(instantiateCudaGraph(&mGraphExec, mGraph));
                mGraphReady = true;
                LOG_INFO("Captured the GR00T denoise loop (%d steps) into one CUDA graph", mConfig.numDenoiseSteps);
            }
            // The warmup and the capture attempt both advanced x_t; restore x_0 before the real run.
            CUDA_CHECK(cudaMemcpyAsync(mTrajectory.rawPointer(), mTrajectoryHost.rawPointer(), elems * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }
        if (mGraphReady)
        {
            CUDA_CHECK(cudaGraphLaunch(mGraphExec, stream));
        }
    }
    if (!mUseCudaGraph && !enqueueLoop(stream))
    {
        throw std::runtime_error("GR00T action engine execution failed");
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mTrajectoryHost.rawPointer(), mTrajectory.rawPointer(), elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return std::vector<float>(mTrajectoryHost.dataPointer<float>(), mTrajectoryHost.dataPointer<float>() + elems);
}

} // namespace gr00t
} // namespace trt_edgellm
