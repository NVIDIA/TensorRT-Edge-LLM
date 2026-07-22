/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "action/actionModelTypes.h"
#include "common/tensor.h"
#include "runtime/linearKVCache.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! How frozen VLM/LM context is passed into the action engine for a request.
enum class ActionContextHandoff
{
    PREFIX_KV,      //!< Per-layer k_cache_i / v_cache_i from LinearKVCache (e.g. Alpamayo)
    CONTEXT_TENSOR, //!< context_embs and related tensors wired from language prefill (e.g. GR00T)
    UNKNOWN
};

//! How x_t is updated after each denoise engine forward.
enum class ActionRolloutMode
{
    FLOW_MATCHING, //!< Engine outputs next state; copy denoised -> noise
    VELOCITY,      //!< Engine outputs velocity; integrate x_t += dt * velocity on host
    UNKNOWN
};

//! Fields loaded from action/config.json. Not every field applies to every model.
struct ActionRunnerConfig
{
    float ropeTheta{0.0F};
    int32_t mropeSectionH{0};
    int32_t mropeSectionW{0};
    int32_t numDecoderLayers{0};
    int32_t numTrajTokens{0};
    int32_t trajTokenStart{0};
    int32_t maxKVCacheCapacity{0};
    int32_t numInferenceTimesteps{0};
    int32_t actionHorizon{0};
    int32_t actionDim{0};
};

//! Model-agnostic orchestration for a compiled action / diffusion step engine.
class ActionRunner
{
public:
    ActionRunner(std::string const& engineDir, cudaStream_t stream, LinearKVCache::CacheConfig const& kvCacheConfig);

    ~ActionRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);
    bool resetExecutionContext(cudaStream_t stream);

    action::ActionModelType getModelType() const noexcept
    {
        return mModelType;
    }

    ActionContextHandoff getContextHandoff() const noexcept
    {
        return mContextHandoff;
    }

    ActionRolloutMode getRolloutMode() const noexcept
    {
        return mRolloutMode;
    }

    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseSeed = seed;
        mRng.seed(static_cast<uint32_t>(seed));
    }

    int32_t getMaxKVCacheCapacity() const noexcept
    {
        return mConfig.maxKVCacheCapacity;
    }

    bool preprocess(LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer);

    //! Prefix-KV / flow-matching path (Alpamayo-style). Requires populated kvcache and optional MRoPE deltas.
    std::vector<std::vector<FutureTrajectoryPoint>> sampleTrajectory(cudaStream_t stream, int32_t activeBatchSize,
        LinearKVCache& kvcache, std::vector<int64_t> const& vlmOutputsRopeDeltas);

    //! Context-tensor / velocity path (GR00T-style). Wire context via copyInputFrom / wireLanguageOutputs first.
    bool sampleActions(cudaStream_t stream);

    bool hasInputTensor(std::string const& name) const noexcept;
    bool copyInputFrom(std::string const& name, rt::Tensor const& src, cudaStream_t stream);
    bool wireLanguageOutputs(std::vector<rt::Tensor const*> const& languageOutputs, cudaStream_t stream);
    //! Bind non-rollout action inputs (state, embodiment_id, etc.) from the request.
    bool wireStaticInputs(LLMGenerationRequest const& request, cudaStream_t stream);
    //! Build PI0.5 suffix position_ids / attention_mask from the packed prefix length.
    bool preparePi05SuffixInputs(int32_t activeBatchSize, int32_t prefixValidLen, cudaStream_t stream);
    bool isRolloutManagedInput(std::string const& name) const noexcept;
    rt::Tensor& getActions();
    rt::Tensor const& getActions() const;
    bool copyActionsToHost(std::vector<std::vector<float>>& actionsPerBatch, cudaStream_t stream) const;

private:
    bool parseModelConfig(std::string const& configPath);
    void allocateTensors(LinearKVCache::CacheConfig const& kvCacheConfig);
    bool reshapeActionTensorsForActiveBatch(int32_t activeBatchSize);
    void initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize);
    void setDynamicInputShapes(int32_t activeBatchSize);
    int32_t const* getActualKVLengths(cudaStream_t stream, int32_t activeBatchSize);
    std::pair<rt::Tensor&, rt::Tensor&> getSeparateKVCacheForDecoderLayer(
        cudaStream_t stream, LinearKVCache& kvcache, int32_t decoderLayerIdx, int32_t activeBatchSize);

    bool engineHasTensor(std::string const& name) const noexcept;
    void allocatePrefixKVTensors(LinearKVCache::CacheConfig const& kvCacheConfig);
    void allocateVelocityTensors();
    bool bindVelocityTensors() noexcept;
    bool setTimestepForStep(int32_t step, cudaStream_t stream);
    bool updateActionsOnHost(cudaStream_t stream, float stepSize);
    void buildEngineBindingNames();
    std::size_t velocityInputIndex(std::string const& name) const;
    std::string const& engineBindingName(std::size_t inputIndex) const;

    static constexpr int32_t kDefaultDenoiseSteps = 10;

    cudaStream_t mStream{nullptr};
    action::ActionModelType mModelType{action::ActionModelType::UNKNOWN};
    ActionContextHandoff mContextHandoff{ActionContextHandoff::UNKNOWN};
    ActionRolloutMode mRolloutMode{ActionRolloutMode::UNKNOWN};
    int32_t mNoiseSeed{5};
    ActionRunnerConfig mConfig{};
    nlohmann::json mConfigJson;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    void* mExecContextMemory{nullptr};
    int64_t mExecContextMemoryCapacity{0};

    int32_t mMaxActionBatchSize{0};
    int32_t mActiveActionBatchSize{0};
    bool mUsesMRope{false};

    // Shared rollout noise [B, horizon, dim]
    rt::Tensor mNoiseHost;
    rt::Tensor mNoiseDevice;

    // Prefix-KV / flow-matching tensors
    rt::Tensor mDenoisedDevice;
    rt::Tensor mDenoisedHost;
    rt::Tensor mTimeStepsT0Device;
    rt::Tensor mTimeStepsT1Device;
    rt::Tensor mTimeStepsT0Host;
    rt::Tensor mTimeStepsT1Host;
    rt::Tensor mKvcacheActualLengthsHost;
    int32_t* mKvcacheActualLengthsDevice{nullptr};
    rt::Tensor mKvcacheActualLengthsBroadcastDevice;
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsHost;
    rt::Tensor mPositionIdsDevice;
    rt::Tensor mRopePositionIdsHost;
    rt::Tensor mRopePositionIdsDevice;
    int32_t mNumKVHeads{0};
    int32_t mMaxSequenceLength{0};
    int32_t mKvHeadDim{0};
    int64_t mRopeHeadDim{0};
    std::vector<rt::Tensor> mKCacheLayers;
    std::vector<rt::Tensor> mVCacheLayers;

    // Context-tensor / velocity tensors
    std::vector<std::string> mInputNames;
    std::vector<std::string> mEngineBindingNames;
    std::vector<rt::Tensor> mInputTensors;
    rt::Tensor mPredVelocity;
    rt::Tensor mPredVelocityHost;
    rt::Tensor mTimestepHost;
    std::string mNoiseInputName{"actions"};
    std::string mTimestepName{"timestep"};
    std::string mContextEmbedsName{"context_embs"};
    std::string mPredVelocityName{"velocity"};
    std::string mTimestepSchedule{"discrete_buckets"};
    std::vector<std::pair<int32_t, int32_t>> mLmToActionSlots;
    std::vector<std::string> mLmWiredInputNames;
    int32_t mNumTimestepBuckets{1};
    int32_t mRolloutDtSign{1};
    std::mt19937 mRng{0};
};

} // namespace rt
} // namespace trt_edgellm
