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

#pragma once

#include "action/actionKvBatch.h"
#include "action/actionModelTypes.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/externalWeightManager.h"
#include "tokenizer/tokenizer.h"

#include <NvInfer.h>
#include <array>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration parsed from the action engine's config.json
struct ActionConfig
{
    std::vector<std::array<float, 2>> kvCacheScales; //!< FP8 codec used by the exported action plugins.
    float ropeTheta{0.0F};                           //!< RoPE base frequency (from rope_theta)
    int32_t mropeSectionH{0};    //!< MRoPE frequency pairs for height dimension (from rope_parameters.mrope_section[1])
    int32_t mropeSectionW{0};    //!< MRoPE frequency pairs for width dimension (from rope_parameters.mrope_section[2])
    int32_t numDecoderLayers{0}; //!< Number of transformer decoder layers (from num_hidden_layers)
    int32_t numTrajTokens{0};    //!< Number of trajectory history tokens (from num_traj_tokens)
    int32_t trajTokenStart{0};   //!< Vocabulary ID of the first trajectory token (from traj_token_start)
    int32_t maxKVCacheCapacity{0}; //!< Maximum KV cache sequence capacity (from builder_config.max_kv_cache_capacity)
    int32_t maxActionBatchSize{0}; //!< Action engine batch capacity (from builder_config.max_batch_size)
    int32_t numKVHeads{0};         //!< Number of key-value heads (from num_key_value_heads)
    int32_t numActionTokens{0};    //!< Number of action tokens from n_diffusion_tokens
    int32_t headDim{0};            //!< Head dimension (from head_dim)
};

//! \brief Standalone action / diffusion head for Alpamayo 1 trajectory prediction.
//!
//! Consumes the VLM KV cache after generation and produces future trajectory waypoints via a
//! flow-matching denoising loop.
class Alpamayo1ActionRunner
{
public:
    //! Validate a KV binding's declared dtype before sharing the LLM pool address.
    static void validateKVPoolDataType(
        nvinfer1::DataType poolType, nvinfer1::DataType bindingType, std::string const& bindingName)
    {
        bool const compatible = poolType == bindingType
            || (poolType == nvinfer1::DataType::kFP8 && bindingType == nvinfer1::DataType::kINT8);
        ELLM_CHECK(compatible,
            "Cannot share KV pool: LLM uses " + std::string(getDataTypeString(fromTrt(poolType)))
                + ", but action binding " + bindingName + " expects " + getDataTypeString(fromTrt(bindingType)) + ".");
    }

    //! Validate the fixed action input profile against the actual LLM physical pool size.
    static void validateKVPoolPageCount(int64_t actualPages, int64_t minPages, int64_t optPages, int64_t maxPages)
    {
        ELLM_CHECK(minPages == actualPages && optPages == actualPages && maxPages == actualPages,
            "Action pool profile does not match the LLM pool: actual pages=" + std::to_string(actualPages)
                + ", action MIN/OPT/MAX=" + std::to_string(minPages) + "/" + std::to_string(optPages) + "/"
                + std::to_string(maxPages) + ". Rebuild action with matching max_kv_pool_pages.");
    }

    using KvAllGatherFn
        = std::function<bool(void const* sendBuffer, void* recvBuffer, int32_t sendCount, cudaStream_t stream)>;

    //! \brief Load action engine, config, and allocate tensors
    //! \param engineDir Path to directory containing action.engine and config.json
    //! \param checkpointDir Original model checkpoint used for runtime weights
    //! \param stream CUDA stream for operations
    //! \param kvCacheConfig KV cache layout from the LLM (from KVCacheManager::Config())
    //! \throws std::runtime_error If engine loading, configuration parsing, or allocation fails.
    //!
    //! config.json must include rope_theta and num_hidden_layers (decoder layer count)
    Alpamayo1ActionRunner(std::string const& engineDir, std::string const& checkpointDir, cudaStream_t stream,
        KVCacheManager::Config const& kvCacheConfig, int32_t tensorParallelSize, int32_t tensorParallelRank,
        std::vector<std::array<float, 2>> kvCacheScales);

    ~Alpamayo1ActionRunner() noexcept = default;

    //! \brief Get the required context memory size for this engine
    //! \return Required context memory size in bytes
    int64_t getRequiredContextMemorySize() const;

    //! \brief Set shared context memory for the execution context
    //! \param sharedContextMemory Tensor containing the shared device memory (must be on GPU)
    //! \return True on success, false if the tensor is too small
    //! \note The tensor size must be >= getRequiredContextMemorySize(). Must be called before infer().
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! \brief Install the FP16 TP all-gather used to reconstruct full action KV tensors.
    void setKvAllGather(KvAllGatherFn kvAllGather)
    {
        mKvAllGather = std::move(kvAllGather);
    }

    bool isRoot() const noexcept
    {
        return mTensorParallelRank == 0;
    }

    //! \brief Get action head model type
    //! \return Action model type
    action::ActionModelType getModelType() const noexcept
    {
        return mModelType;
    }

    //! \brief Set the random seed used when initializing the diffusion noise trajectory
    //! \param seed Random seed value
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseSeed = seed;
    }

    //! \brief Get the max KV cache capacity the action engine was built with
    //! \return Maximum KV cache capacity (from builder_config in engine config.json)
    int32_t getMaxKVCacheCapacity() const noexcept
    {
        return mConfig.maxKVCacheCapacity;
    }

    //! \brief Run one batched diffusion/flow-matching loop and return future trajectory waypoints for all batch items.
    //! Call preprocess() once per request before this (prefill path).
    //! \param stream CUDA stream for operations
    //! \param kvcache KV cache page pool containing the VLM outputs
    //! \param batch Dense action-local page table, lengths, and positional metadata
    std::vector<std::vector<FutureTrajectoryPoint>> sampleTrajectory(
        cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch);

    /*!
     * \brief Preprocess batched token IDs for Alpamayo (e.g. replace <|traj_history|> pads with trajectory
     * tokens), validate batch size against allocated action buffers, and initialize diffusion noise for the request.
     * \return True on success, false if trajectory placeholder fill or noise setup fails
     */
    bool preprocess(LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer);

private:
    // Number of denoising steps run by the diffusion head
    static constexpr int32_t kNumDenoiseSteps = 10;

    //! \brief Allocate tensors for the action runner
    void allocateTensors(KVCacheManager::Config const& kvCacheConfig);

    //! \brief Initialize the noise trajectory
    //! \param randomSeed Random seed for the noise trajectory
    //! \param activeBatchSize Number of batch rows to fill
    void initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize);

    //! \brief Set TensorRT input shapes for the current active batch (dynamic batch / seq_length).
    //! \param activeBatchSize Number of active sequences
    void setPagedInputShapes(int32_t activeBatchSize, HybridCacheManager const& kvcache);

    //! Shape the dense action inputs used for TP>1.
    void setDenseInputShapes(int32_t activeBatchSize);

    //! Gather and dequantize TP KV shards into rank-zero dense action caches.
    bool gatherDenseKV(cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch);

    //! Reshape active-batch buffers on the dense TP path.
    bool reshapeActionTensorsForActiveBatch(int32_t activeBatchSize);

    //! \brief Load model fields from config.json (e.g. rope_theta, num_hidden_layers).
    bool parseModelConfig(std::string const& configPath);

    cudaStream_t mStream{nullptr};
    action::ActionModelType mModelType{action::ActionModelType::ALPAMAYO1};
    int32_t mNoiseSeed{5};  //!< Random seed for diffusion noise trajectory initialization
    ActionConfig mConfig{}; //!< Model configuration parsed from config.json

    AuxStreamSet mAuxStreams{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::unique_ptr<ExternalWeightManager> mExternalWeights{nullptr};

    rt::Tensor mNoiseTrajectoryDevice;
    rt::Tensor mNoiseTrajectoryHost;
    rt::Tensor mDenoisedTrajectoryDevice;
    rt::Tensor mDenoisedTrajectoryHost;
    rt::Tensor mTimeStepsT0Device;
    rt::Tensor mTimeStepsT1Device;
    rt::Tensor mTimeStepsT0Host;
    rt::Tensor mTimeStepsT1Host;
    rt::Tensor mQueryLengthsHost;
    rt::Tensor mQueryLengthsDevice;
    rt::Tensor mAttentionSequenceLengthsHost;
    rt::Tensor mAttentionSequenceLengthsDevice;
    rt::Tensor mQueryStartOffsetsHost;
    rt::Tensor mQueryStartOffsetsDevice;
    rt::Tensor mContextMaskSelectorDevice;
    rt::Tensor mExecutionPhaseMarkerDevice;
    rt::Tensor mContextSequenceCountCarrierDevice;
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsHost;
    rt::Tensor mPositionIdsDevice;
    rt::Tensor mRopePositionIdsHost;
    rt::Tensor mRopePositionIdsDevice;

    int32_t mNumWaypoints{64};
    int64_t mRopeHeadDim{128};
    int32_t mMaxSequenceLength{0};
    std::unique_ptr<KVPageTable> mActionPageTable;
    std::vector<Tensor> mActionKVPools;
    //! Configured LLM max batch size, validated against the action engine profile.
    int32_t mMaxActionBatchSize{0};
    int32_t mTensorParallelSize{1};
    int32_t mTensorParallelRank{0};
    int32_t mLocalKVHeads{0};
    bool mUseFp8KV{false};
    KvAllGatherFn mKvAllGather{};
    std::vector<std::array<float, 2>> mKVCacheScales;
    rt::Tensor mLocalKVScratch;
    rt::Tensor mGatheredKVScratch;
    std::vector<rt::Tensor> mKCacheLayers;
    std::vector<rt::Tensor> mVCacheLayers;
};

} // namespace rt
} // namespace trt_edgellm
