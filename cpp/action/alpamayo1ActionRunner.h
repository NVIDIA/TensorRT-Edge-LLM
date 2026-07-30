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

#include "action/actionModelTypes.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration parsed from the action engine's config.json
struct ActionConfig
{
    float ropeTheta{0.0F};       //!< RoPE base frequency (from rope_theta)
    int32_t mropeSectionH{0};    //!< MRoPE frequency pairs for height dimension (from rope_parameters.mrope_section[1])
    int32_t mropeSectionW{0};    //!< MRoPE frequency pairs for width dimension (from rope_parameters.mrope_section[2])
    int32_t numDecoderLayers{0}; //!< Number of transformer decoder layers (from num_hidden_layers)
    int32_t numTrajTokens{0};    //!< Number of trajectory history tokens (from num_traj_tokens)
    int32_t trajTokenStart{0};   //!< Vocabulary ID of the first trajectory token (from traj_token_start)
    int32_t maxKVCacheCapacity{0}; //!< Maximum KV cache sequence capacity (from builder_config.max_kv_cache_capacity)
    int32_t numKVHeads{0};         //!< Number of key-value heads (from num_key_value_heads)
    int32_t headDim{0};            //!< Head dimension (from head_dim)
};

//! \brief Standalone action / diffusion head for Alpamayo 1 trajectory prediction.
//!
//! Consumes the VLM KV cache after generation and produces future trajectory waypoints via a
//! flow-matching denoising loop.
class Alpamayo1ActionRunner
{
public:
    //! \brief Load action engine, config, and allocate tensors
    //! \param engineDir Path to directory containing action.engine and config.json
    //! \param stream CUDA stream for operations
    //! \param kvCacheConfig KV cache layout from the LLM (from KVCacheManager::Config())
    //! \param basePageTableIsIdentity Whether the base cache manager's KV page table is (and is
    //!        guaranteed to remain) an identity mapping. See the identity-only contract note on
    //!        getSeparateKVCacheForDecoderLayer() -- this consumer reads physical slot rows directly
    //!        and is not page-table-aware, so it must refuse to load rather than
    //!        silently consume unrelated physical pages once non-identity reuse exists.
    //! \throws std::runtime_error If engine loading, configuration parsing, or allocation fails, or
    //!         if `basePageTableIsIdentity` is false.
    //!
    //! config.json must include rope_theta and num_hidden_layers (decoder layer count)
    Alpamayo1ActionRunner(std::string const& engineDir, cudaStream_t stream,
        KVCacheManager::Config const& kvCacheConfig, bool basePageTableIsIdentity);

    ~Alpamayo1ActionRunner() noexcept = default;

    //! \brief Get the required context memory size for this engine
    //! \return Required context memory size in bytes
    int64_t getRequiredContextMemorySize() const;

    //! \brief Set shared context memory for the execution context
    //! \param sharedContextMemory Tensor containing the shared device memory (must be on GPU)
    //! \return True on success, false if the tensor is too small
    //! \note The tensor size must be >= getRequiredContextMemorySize(). Must be called before infer().
    bool setContextMemory(rt::Tensor& sharedContextMemory);

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
    //! \param activeBatchSize Number of active sequences
    //! \param kvcache KV cache containing the VLM outputs; used for KV cache lengths and layer tensors
    //! \param vlmOutputsRopeDeltas Per-batch VLM RoPE deltas (e.g. from vision runner getMropeRopeDeltasPerBatch); size
    //! must match batch.
    std::vector<std::vector<FutureTrajectoryPoint>> sampleTrajectory(cudaStream_t stream, int32_t activeBatchSize,
        HybridCacheManager& kvcache, std::vector<int64_t> const& vlmOutputsRopeDeltas);

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

    //! \brief Reshape batch-sized tensors to activeBatchSize
    //! \return false if any reshape fails
    bool reshapeActionTensorsForActiveBatch(int32_t activeBatchSize);

    //! \brief Initialize the noise trajectory
    //! \param randomSeed Random seed for the noise trajectory
    //! \param activeBatchSize Number of batch rows to fill
    void initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize);

    //! \brief Set TensorRT input shapes for the current active batch (dynamic batch / seq_length).
    //! \param activeBatchSize Number of active sequences
    void setDynamicInputShapes(int32_t activeBatchSize);

    //! \brief Copy all KV cache lengths from device to host and return pointer to host buffer.
    //! \param stream CUDA stream for execution
    //! \param activeBatchSize Active batch size
    //! \return Pointer to the host buffer containing the KV cache lengths
    int32_t const* getActualKVLengths(cudaStream_t stream, int32_t activeBatchSize);

    //! \brief Deinterleave combined [maxBatchSize, 2, H, S, D] (KV Cache layout from attention plugin) for one layer
    //! into owned buffers and return refs to them. The Alpamayo action expert's exported graph consumes separate K/V
    //! caches of shape [2, maxBatchSize, H, S, D] (the TRT native attention op layout), so this runner repacks the
    //! plugin-path combined buffer into that layout.
    //!
    //! CONTRACT: this reads KV from physical slot row `b` of the combined pool
    //! directly -- it does not receive or consult the base cache manager's KVPageTable. It is
    //! therefore only correct while the base cache is identity-mapped (slot == physical row), which
    //! is enforced once, at construction time, by the `basePageTableIsIdentity` check in the
    //! constructor. If non-identity base-cache reuse is ever enabled for a model with an Alpamayo
    //! action head, this function must gain page-aware gather logic before that check is relaxed.
    //! \param stream CUDA stream for execution
    //! \param kvcache KV cache containing the VLM outputs; used to read active batch size and KV cache lengths
    //! \param decoderLayerIdx Index of the decoder layer
    //! \param activeBatchSize Number of active batch rows to copy from the combined KV buffer
    //! \return Pair of references to the K and V cache tensors for the decoder layer
    std::pair<rt::Tensor&, rt::Tensor&> getSeparateKVCacheForDecoderLayer(
        cudaStream_t stream, HybridCacheManager& kvcache, int32_t decoderLayerIdx, int32_t activeBatchSize);

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

    rt::Tensor mNoiseTrajectoryDevice;
    rt::Tensor mNoiseTrajectoryHost;
    rt::Tensor mDenoisedTrajectoryDevice;
    rt::Tensor mDenoisedTrajectoryHost;
    rt::Tensor mTimeStepsT0Device;
    rt::Tensor mTimeStepsT1Device;
    rt::Tensor mTimeStepsT0Host;
    rt::Tensor mTimeStepsT1Host;
    rt::Tensor mKvcacheActualLengthsHost;
    int32_t* mKvcacheActualLengthsDevice{nullptr};
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsHost;
    rt::Tensor mPositionIdsDevice;
    rt::Tensor mRopePositionIdsHost;
    rt::Tensor mRopePositionIdsDevice;

    int32_t mNumWaypoints{64};
    int64_t mRopeHeadDim{128};
    int32_t mMaxSequenceLength{0};
    //! Allocated max batch for action staging (min of engine profile and KV cache maxBatchSize).
    int32_t mMaxActionBatchSize{0};
    std::vector<rt::Tensor> mKCacheLayers;
    std::vector<rt::Tensor> mVCacheLayers;
};

} // namespace rt
} // namespace trt_edgellm
