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

#include "action/cosmos3PolicyRunner.h"
#include "common/tensor.h"
#include "multimodal/cosmos3VaeEncoderRunner.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! \brief Orchestrates the Cosmos3 policy path by composing core-style TensorRT engines:
//!   1. UND prefill engine (prefill-only tower) over the prompt embeddings,
//!   2. the Wan VAE encoder over the conditioning frame,
//!   3. the GEN diffusion policy runner (UniPC loop) cross-attending to the frozen UND K/V.
//!
//! Engines share a single device-memory pool (like the core runtime), and every device buffer is
//! allocated once at construction for the engine-profile maximum shapes; per-request work is limited
//! to shape updates and enqueues. The UND prefill engine emits the per-layer K/V directly in the
//! seq-major [B,S,numKVHeads,headDim] layout the GEN graph consumes (no KV-cache AttentionPlugin,
//! no transpose), keeping all Cosmos3 wiring out of the core runtime.
class Cosmos3Runtime
{
public:
    //! \param engineDir Directory with und_prefill/, vae_encoder/, and gen/ component subdirectories.
    //! \param stream CUDA stream.
    explicit Cosmos3Runtime(std::string const& engineDir, cudaStream_t stream);
    ~Cosmos3Runtime() noexcept = default;

    void setNoiseSeed(int32_t seed)
    {
        mPolicyRunner->setNoiseSeed(seed);
    }

    void setNumInferenceSteps(int32_t steps)
    {
        mPolicyRunner->setNumInferenceSteps(steps);
    }

    void setUseCudaGraph(bool enable) noexcept
    {
        mPolicyRunner->setUseCudaGraph(enable);
    }

    //! \brief Select the video-subsample path for subsequent generatePolicy() calls. factor == 1
    //! (default) is the regular path; factor > 1 runs the optimized path over fewer GEN video frames
    //! (see Cosmos3PolicyRunner::setVideoSubsampleFactor). Only the GEN denoise extent changes; the VAE
    //! conditioning encode and UND prefill are unaffected.
    void setVideoSubsampleFactor(int32_t factor)
    {
        mPolicyRunner->setVideoSubsampleFactor(factor);
    }

    //! \brief Select the action-chunk length for subsequent generatePolicy() calls (0 = engine canonical
    //! max, unchanged default; positive = clamped into the engine's built action range). See
    //! Cosmos3PolicyRunner::setActionChunkSize.
    void setActionChunkSize(int32_t chunk)
    {
        mPolicyRunner->setActionChunkSize(chunk);
    }

    //! \brief Enable guidance-interval CFG on the GEN loop (see Cosmos3PolicyRunner::setGuidance).
    //! guidance == 1 (default) disables it. When enabled, generatePolicy must be given uncondEmbeds.
    //! Enabling CFG (guidance != 1) lazily allocates the second (unconditional) UND K/V set; a no-op
    //! re-enable does not re-allocate.
    void setGuidance(float guidance, float intervalLo, float intervalHi);

    //! \brief Policy component contract (action chunk size, raw action dim, domain, ...).
    Cosmos3PolicyConfig const& policyConfig() const noexcept
    {
        return mPolicyRunner->getConfig();
    }

    //! \brief Run the full policy pipeline and return the action chunk [B, actionChunkSize, rawActionDim].
    //! \param pixelValues  Device FLOAT32 conditioning-pixel tensor [B,3,t_in,H,W] (preprocessed).
    //! \param inputsEmbeds Device FLOAT16 conditional prompt embeddings [B,S,hidden] (embed_tokens upstream).
    //! \param uncondEmbeds Optional FLOAT16 unconditional (empty-prompt) embeddings [B,S',hidden] for CFG;
    //!                     nullptr (or guidance == 1) runs the single-forward conditional path.
    std::vector<float> generatePolicy(rt::Tensor const& pixelValues, rt::Tensor const& inputsEmbeds,
        rt::Tensor const* uncondEmbeds, std::vector<float> const& currentState, cudaStream_t stream);

private:
    //! \brief Run the UND prefill engine over the prompt embeddings, writing per-layer K/V into the
    //! given target buffers (mUndK/mUndV for the conditional pass, mUndKUncond/mUndVUncond for CFG).
    void prefillUnd(rt::Tensor const& inputsEmbeds, std::vector<rt::Tensor>& undK, std::vector<rt::Tensor>& undV,
        cudaStream_t stream);
    //! \brief Allocate the shared context-memory pool and bind it to all three engines.
    void allocateSharedContextMemory();
    //! \brief Allocate all UND prefill I/O buffers at the engine-profile maximum sequence length,
    //! precompute the prefix-safe rope cos/sin + position ids, and bind every address once.
    void allocateUndBuffers(cudaStream_t stream);
    //! \brief Allocate the second (unconditional, empty-prompt) UND K/V set used only by guidance-interval
    //! CFG. Called from setGuidance only when CFG is enabled (guidance != 1); guarded against double-alloc.
    void allocateUncondBuffers();

    // UND prefill engine (owned here; prefill-only tower emitting per-layer K/V as graph outputs).
    std::unique_ptr<nvinfer1::IRuntime> mTextRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mTextEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mTextContext{nullptr};

    // UND prefill contract values; parsed strictly from und_prefill/config.json (no fallback defaults).
    int32_t mNumLayers{0};
    int32_t mNumKVHeads{0};
    int32_t mHeadDim{0};
    int32_t mHiddenSize{0};
    float mRopeTheta{0.0F};
    int32_t mMaxUndLen{0};       //!< engine-profile maximum prompt length (allocation bound).
    int32_t mMaxUndBatch{1};     //!< engine-profile maximum batch (allocation bound).
    int32_t mRopeFilledBatch{0}; //!< request shape the rope/pos-id caches are currently packed for.
    int32_t mRopeFilledLen{0};

    std::unique_ptr<Cosmos3VaeEncoderRunner> mVaeRunner;
    std::unique_ptr<Cosmos3PolicyRunner> mPolicyRunner;

    rt::Tensor mSharedContextMemory;

    float mGuidance{1.0F}; //!< CFG scale; 1 disables the unconditional pass.

    // Per-layer seq-major UND K/V (FLOAT16), allocated once at [maxB,maxS,...] and reshaped per request.
    std::vector<rt::Tensor> mUndK; //!< conditional [B,S,numKVHeads,headDim] per layer
    std::vector<rt::Tensor> mUndV; //!< conditional [B,S,numKVHeads,headDim] per layer
    // Unconditional (empty-prompt) UND K/V per layer, allocated ONLY when guidance-interval CFG is
    // enabled (via setGuidance -> allocateUncondBuffers); empty on the default guidance == 1 path.
    std::vector<rt::Tensor> mUndKUncond;
    std::vector<rt::Tensor> mUndVUncond;
    rt::Tensor mTextRopeCosSin;     //!< text-only mRoPE cache, packed [B,S,headDim] (refilled on shape change).
    rt::Tensor mAttentionPosId;     //!< iota positions, packed [B,S] INT32 (refilled on shape change).
    rt::Tensor mAttentionPosIdHost; //!< pinned host staging for mAttentionPosId (allocated once at max shape).
    rt::Tensor mUndHidden;          //!< [B,S,hidden] scratch for the engine's hidden_states output (unused).
};

} // namespace cosmos3
} // namespace trt_edgellm
