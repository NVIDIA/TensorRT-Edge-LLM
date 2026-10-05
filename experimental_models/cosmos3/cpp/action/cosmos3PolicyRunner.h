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

#include "action/cosmos3Scheduler.h"
#include "common/tensor.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! \brief Configuration parsed from the Cosmos3 GEN engine's config.json.
//! Every field is required and parsed strictly from the component contract; there are no
//! fallback defaults (a missing key fails engine load instead of silently mis-running).
struct Cosmos3PolicyConfig
{
    int32_t numHiddenLayers{0};
    int32_t numKVHeads{0};
    int32_t headDim{0};
    float ropeTheta{0.0F};
    int32_t mropeSectionH{0};
    int32_t mropeSectionW{0};
    int32_t latentChannel{0};
    int32_t latentPatchSize{0};
    int32_t videoLatentFrames{0};    //!< latent temporal length t for the policy video latent (profile max).
    int32_t minVideoLatentFrames{1}; //!< smallest latent temporal length the engine's dynamic profile admits.
    int32_t actionChunkSize{0};
    int32_t stateRows{0};     //!< Clean leading current-state rows (Policy-DROID: 1).
    int32_t historyLength{0}; //!< Internal leading rows omitted from served output.
    bool useState{false};
    int32_t rawActionDim{0};
    int32_t maxActionDim{0};
    int32_t numInferenceSteps{0};
    float flowShift{0.0F};
    float timestepScale{0.0F};
    int32_t domainId{0};
    // unified_3d_mrope construction parameters (must match the reference transformer).
    float fps{0.0F};     //!< media/control fps used for temporal-position modulation.
    float baseFps{0.0F}; //!< model base fps.
    int32_t temporalCompressionFactor{0};
    int32_t temporalModalityMargin{0}; //!< gap separating text vs media position spaces.
    int32_t actionStartFrameOffset{0}; //!< action grid start-frame offset.
};

//! \brief Cosmos3 diffusion policy head. Runs the joint video+action flow-matching denoising loop in
//! process on the GPU (one GEN engine `enqueueV3` per step), cross-attending into the frozen UND KV.
//!
//! Mirrors the in-process / shared-context-memory / KV-reuse design of the core runtime,
//! but is a separate class with Cosmos3's joint-latent I/O and the host UniPC scheduler. The Wan VAE
//! decode is intentionally never run (video pixels are out of scope); only the action chunk is returned.
class Cosmos3PolicyRunner
{
public:
    //! \param engineDir Directory containing gen.engine and config.json.
    //! \param stream CUDA stream.
    Cosmos3PolicyRunner(std::string const& engineDir, cudaStream_t stream);
    ~Cosmos3PolicyRunner() noexcept;

    //! \brief Required TensorRT context-memory size (for the shared device-memory pool).
    int64_t getRequiredContextMemorySize() const;

    //! \brief Bind shared context memory (must be >= getRequiredContextMemorySize()).
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! \brief Set the random seed for the initial noise latents.
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseSeed = seed;
    }

    //! \brief Enable CUDA-graph capture/replay of the per-step GEN engine forward. Each (batch, temporal
    //! extent) signature is captured once on its first generate() and cached; later requests of that
    //! signature replay the cached graph, so alternating video-subsample factors never re-capture.
    void setUseCudaGraph(bool enable) noexcept
    {
        mUseCudaGraph = enable;
    }

    //! \brief Override the number of diffusion denoise steps for this process.
    void setNumInferenceSteps(int32_t steps);

    //! \brief Select the video-subsample path for subsequent generate() calls. factor == 1 (default) is
    //! the regular path (full videoLatentFrames temporal extent, unchanged behavior); factor > 1 runs the
    //! optimized path over fewer video frames: t_frames = actionChunkSize / factor + 1 generated frames ->
    //! (t_frames - 1) / temporalCompressionFactor + 1 latent planes, shrinking the GEN video-token count.
    //! The reduced extent must lie within the engine's dynamic video-token profile (built for the largest
    //! supported factor); buffers stay allocated at the regular maximum, so a smaller factor is a
    //! metadata-only reshape with no hot-path allocation.
    void setVideoSubsampleFactor(int32_t factor);

    //! \brief Select the action-chunk length (number of action timesteps the GEN expert emits) for
    //! subsequent generate() calls. 0 (default) requests the engine's canonical/profile-max chunk
    //! (unchanged behavior); a positive value is clamped into the engine's built action range. The
    //! returned action tensor's chunk dimension follows the (clamped) request. Requires an engine built
    //! with a dynamic action-token profile (--min-action-chunk / --max-action-chunk) to vary at runtime.
    void setActionChunkSize(int32_t chunk);

    Cosmos3PolicyConfig const& getConfig() const noexcept
    {
        return mConfig;
    }

    //! \brief Enable guidance-interval classifier-free guidance (CFG). When guidance != 1, the denoise
    //! steps whose (integer) timestep falls inside (lo, hi) run a second unconditional GEN forward and
    //! blend velocities  v = v_uncond + guidance * (v_cond - v_uncond). Matches the imaginaire4 reference
    //! (guidance 3.0, interval [960, 1001] -> CFG on the first step only with the [999,937,833,624]
    //! schedule). guidance == 1 (default) preserves the single-forward behavior exactly. Enabling CFG
    //! (guidance != 1) lazily allocates the CFG blend scratch; a no-op re-enable does not re-allocate.
    void setGuidance(float guidance, float intervalLo, float intervalHi);

    //! \brief Run the policy denoising loop and return the action chunk. The request batch B is
    //! derived from condLatent and may be any value in [1, engine-profile max batch]; buffers are
    //! preallocated at the profile maximum, so batch changes are metadata-only reshapes.
    //! \param condLatent  Device FLOAT32 conditioning latent, [B, latentChannel, t, h, w] (frame 0 is used).
    //! \param undKeys     Per-layer frozen conditional UND keys, FLOAT16 [B, undLen, numKVHeads, headDim] each.
    //! \param undValues   Per-layer frozen conditional UND values, matching layout.
    //! \param undKeysUncond   Optional unconditional (empty-prompt) UND keys for CFG; empty to disable CFG.
    //! \param undValuesUncond Optional unconditional UND values; must pair with undKeysUncond.
    //! \param currentState Current model-space state [B, rawActionDim] when stateRows == 1; empty otherwise.
    //! \return Flattened future-action chunk [B, actionChunkSize, rawActionDim]; clean state rows are omitted.
    std::vector<float> generate(rt::Tensor const& condLatent, std::vector<rt::Tensor> const& undKeys,
        std::vector<rt::Tensor> const& undValues, std::vector<rt::Tensor> const& undKeysUncond,
        std::vector<rt::Tensor> const& undValuesUncond, std::vector<float> const& currentState, cudaStream_t stream);

private:
    void parseModelConfig(std::string const& configPath);
    void allocateTensors(cudaStream_t stream);
    //! \brief Allocate the conditional-velocity CFG blend scratch (mPredCondScratch). Called from
    //! setGuidance only when CFG is enabled (guidance != 1); the default single-forward path never
    //! allocates it. Guarded against double-allocation.
    void allocateCfgScratch();
    //! \brief Requested (pre-clamp) video temporal planes for the current mVideoSubsampleFactor (== the
    //! regular max for factor 1, fewer for factor > 1). generate() clamps this into the built profile.
    int32_t requestedTemporal() const;
    //! \brief Packed video element count (latentChannel * mActiveT * h * w) for the active temporal extent;
    //! the action block sits at base + batch * this within the max-sized packed allocation.
    int32_t activeVideoElems() const;
    //! \brief Packed action element count (mActiveActionLen * maxActionDim) for the active action chunk.
    int32_t activeActionElems() const;
    //! \brief Cached CUDA-graph exec for a (batch, activeT, actionLen, undLen) workload signature, or
    //! nullptr if that signature has not been captured yet.
    cudaGraphExec_t findGenGraph(int32_t batch, int32_t activeT, int32_t actionLen, int32_t undLen) const;
    //! \brief Build the packed (cos|sin) unified_3d_mRoPE cache and attention position ids for the GEN
    //! stream. Per-token float (fps-modulated) positions are computed on host (tiny), and the
    //! [genLen, headDim] transcendental cache is computed on device by the interleaved-mRoPE kernel.
    void buildRopeAndPositions(int32_t actionLen, int32_t undLen, cudaStream_t stream);
    //! \brief Initialize the packed device state [video ⧺ action]: Philox device noise, frame-0
    //! conditioning injection (2D D2D copy), and padded-action-dim zeroing (2D memset).
    void initializeLatents(rt::Tensor const& condLatent, cudaStream_t stream);
    //! \brief Set dynamic input shapes for one denoising step.
    void setDynamicInputShapes(int32_t batch, int32_t actionLen, int32_t undLen);
    //! \brief One-time per-request setup of the constant-across-steps engine inputs: shapes, mRoPE cos/sin
    //! + position ids, noisy masks, and all tensor-address bindings. Hoisted out of the per-step loop.
    void prepareStatic(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
        std::vector<rt::Tensor> const& undValues, cudaStream_t stream);
    //! \brief (Re)bind the UND cross-attention context: set the und_k/und_v input shapes to `undLen`,
    //! rebuild the undLen-dependent mRoPE cos/sin + position ids, and point the und_k/und_v bindings at
    //! the given K/V set. Used to swap between the conditional and unconditional contexts on a CFG step.
    void bindUndContext(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
        std::vector<rt::Tensor> const& undValues, cudaStream_t stream);
    //! \brief Run one GEN engine forward + the device-resident UniPC update. On a guidance step (CFG
    //! enabled and this timestep inside the guidance interval) it runs a second unconditional forward and
    //! blends velocities before the scheduler update.
    bool runDenoiseStep(int32_t stepIdx, rt::Tensor const& condLatent, cudaStream_t stream);
    //! \brief Re-inject the clean frame-0 conditioning latent and re-zero the padded action dims
    //! (device-side: one 2D D2D copy + one 2D memset).
    void reinjectConditioning(rt::Tensor const& condLatent, cudaStream_t stream);
    //! \brief Zero model velocity on clean video/state tokens before the UniPC update.
    void maskCleanPredictions(cudaStream_t stream);

    int32_t mNoiseSeed{0};
    Cosmos3PolicyConfig mConfig{};
    std::unique_ptr<Cosmos3Scheduler> mScheduler;

    // Guidance-interval CFG. guidance == 1 disables it (single conditional forward, original behavior).
    float mGuidance{1.0F};
    float mGuidanceLo{960.0F};
    float mGuidanceHi{1001.0F};
    // Per-request CFG context, set in generate() before the denoise loop; consumed by runDenoiseStep.
    bool mCfgActive{false};
    int32_t mCfgActionLen{0};
    int32_t mCfgCondUndLen{0};
    int32_t mCfgUncondUndLen{0};
    std::vector<rt::Tensor> const* mCfgUndKeysCond{nullptr};
    std::vector<rt::Tensor> const* mCfgUndValuesCond{nullptr};
    std::vector<rt::Tensor> const* mCfgUndKeysUncond{nullptr};
    std::vector<rt::Tensor> const* mCfgUndValuesUncond{nullptr};

    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};

    //! Optional CUDA-graph capture of the per-step GEN engine enqueue. The denoise steps enqueue the
    //! same engine with fixed shapes + fixed I/O addresses (after prepareStatic), so one capture replays
    //! for every step; only the timestep + latent buffer *contents* change per step (updated in place).
    //! One graph is cached per (batch, activeT, actionLen, undLen) workload signature: changing any axis
    //! replays a cached graph instead of re-capturing. Reasonable policy workloads visit only a handful
    //! of signatures, so mGenGraphs stays tiny — one cudaGraph_t + cudaGraphExec_t per distinct signature.
    bool mUseCudaGraph{false};
    struct GenGraph
    {
        int32_t batch{0};
        int32_t activeT{0};
        int32_t actionLen{0};
        int32_t undLen{0};
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
    };
    //! Upper bound on distinct workload signatures cached as CUDA graphs. Reasonable policy workloads
    //! visit only a handful; the cap prevents an adversarial request stream (varying batch/T/action/und
    //! every call) from leaking one cudaGraphExec_t per signature. Signatures past the cap fall back to
    //! enqueueV3.
    static constexpr size_t kMaxCachedGenGraphs = 16U;
    std::vector<GenGraph> mGenGraphs;           //!< captured-once graphs keyed by (batch, activeT, actionLen, undLen).
    cudaGraphExec_t mCurrentGraphExec{nullptr}; //!< cached exec for the in-flight request (null => enqueueV3).
    bool mGraphCacheFull{false};                //!< warn-once latch for the graph-cache cap.

    int32_t mVideoElems{0};  //!< latentChannel * maxT * h * w (regular/profile-max allocation extent).
    int32_t mActionElems{0}; //!< actionChunkSize * maxActionDim
    int32_t mRopeHeadDim{0};
    int32_t mMaxBatch{1};             //!< engine-profile maximum batch (allocation bound).
    int32_t mActiveBatch{1};          //!< per-request batch, derived from the conditioning latent.
    int32_t mVideoSubsampleFactor{1}; //!< 1 = regular path; >1 = optimized (fewer video frames).
    int32_t mActiveT{0};              //!< per-request active video temporal planes (clamped to [mMinT, maxT]).
    int32_t mMinT{1};                 //!< engine-profile minimum video temporal planes.
    int32_t mRequestedActionChunk{0}; //!< requested action chunk (0 => profile-max); clamped per request.
    int32_t mActiveActionLen{0};      //!< per-request active action chunk (clamped to [min, max]).
    int32_t mMinActionChunk{0};       //!< engine-profile minimum action chunk (allocation/clamp bound).
    int32_t mMaxActionChunk{0};       //!< engine-profile maximum action chunk (allocation/clamp bound).
    int32_t mActiveUndLen{0};         //!< per-request und context length (clamped to [1, mMaxUndLen]).
    int32_t mMaxUndLen{0};            //!< engine-profile maximum und context length.
    int32_t mLastWarnedT{-1};         //!< last out-of-range video-plane request warned (warn-once).
    int32_t mLastWarnedAction{-1};    //!< last out-of-range action-chunk request warned (warn-once).
    int32_t mLastWarnedUnd{-1};       //!< last out-of-range und-len request warned (warn-once).
    std::vector<int64_t> mVideoShape; //!< {maxBatch, C, t, h, w}

    // Device buffers (allocated once at construction for the engine profile shape). The video/action
    // latents live as ONE packed [video ⧺ action] device allocation (the engine binds video at the base
    // pointer and action at base + videoBytes), matching the joint layout the device-resident UniPC
    // scheduler updates in place; the predictions are packed the same way. The denoising state never
    // visits the host; only the final action slice is copied back.
    rt::Tensor mStateDevice;     //!< packed FLOAT32 [B*videoElems | B*actionElems] current latents.
    rt::Tensor mPredDevice;      //!< packed FLOAT32 model predictions.
    rt::Tensor mPredCondScratch; //!< packed FLOAT32 scratch for the conditional velocity during a CFG blend; allocated
                                 //!< only when CFG is enabled.
    rt::Tensor mTimestepDevice;
    rt::Tensor mTokenNoisyMaskDevice;
    rt::Tensor mActionNoisyMaskDevice;
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsDevice; //!< attention_pos_id [B, genLen] (INT32)
    rt::Tensor mPositionsDevice;   //!< [maxB, 3, genLen] float T/H/W planes for the core mRoPE kernel.

    // Pinned host staging (kCPU tensors are cudaMallocHost-backed) for the small per-request /
    // per-step engine inputs, filled in place and uploaded async with no per-call allocation.
    rt::Tensor mTimestepHost;        //!< [maxB] float, refilled every denoise step.
    rt::Tensor mTokenNoisyMaskHost;  //!< [maxB, numVideoTokens, 1] float.
    rt::Tensor mActionNoisyMaskHost; //!< [maxB, actionChunkSize, 1] float.
    rt::Tensor mCurrentStateHost;    //!< [maxB, rawActionDim] float, refilled once per request.
};

} // namespace cosmos3
} // namespace trt_edgellm
