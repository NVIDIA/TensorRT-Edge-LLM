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

#include "action/cosmos3PolicyRunner.h"

#include "action/cosmos3Kernels.h"
#include "common/cosmos3Bindings.h"
#include "profiling/nvtx_wrapper.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>

using namespace trt_edgellm;
using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace cosmos3
{

Cosmos3PolicyRunner::Cosmos3PolicyRunner(std::string const& engineDir, cudaStream_t stream)
{
    LOG_DEBUG("Loading Cosmos3 policy runner from %s", engineDir.c_str());

    std::string const enginePath = engineDir + "/gen.engine";
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "Failed to create execution context");
    ELLM_CHECK(mContext->setOptimizationProfileAsync(0, stream), "Failed to set optimization profile");

    parseModelConfig(engineDir + "/config.json"); // strict: throws on any missing contract key

    Cosmos3Scheduler::Config schedCfg;
    schedCfg.shift = mConfig.flowShift;
    mScheduler = std::make_unique<Cosmos3Scheduler>(schedCfg);

    allocateTensors(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Cosmos3PolicyRunner::~Cosmos3PolicyRunner() noexcept
{
    for (auto& g : mGenGraphs)
    {
        if (g.exec != nullptr)
        {
            static_cast<void>(cudaGraphExecDestroy(g.exec));
        }
        if (g.graph != nullptr)
        {
            static_cast<void>(cudaGraphDestroy(g.graph));
        }
    }
}

cudaGraphExec_t Cosmos3PolicyRunner::findGenGraph(
    int32_t batch, int32_t activeT, int32_t actionLen, int32_t undLen) const
{
    for (auto const& g : mGenGraphs)
    {
        if (g.batch == batch && g.activeT == activeT && g.actionLen == actionLen && g.undLen == undLen)
        {
            return g.exec;
        }
    }
    return nullptr;
}

void Cosmos3PolicyRunner::setNumInferenceSteps(int32_t steps)
{
    if (steps <= 0)
    {
        throw std::invalid_argument("Cosmos3PolicyRunner requires a positive denoise step count");
    }
    mConfig.numInferenceSteps = steps;
}

void Cosmos3PolicyRunner::setVideoSubsampleFactor(int32_t factor)
{
    if (factor < 1)
    {
        throw std::invalid_argument("Cosmos3PolicyRunner video_subsample_factor must be >= 1");
    }
    mVideoSubsampleFactor = factor;
}

int32_t Cosmos3PolicyRunner::requestedTemporal() const
{
    int32_t const maxT = static_cast<int32_t>(mVideoShape[2]);
    if (mVideoSubsampleFactor <= 1)
    {
        return maxT;
    }
    int32_t const tFrames = mConfig.actionChunkSize / mVideoSubsampleFactor + 1;
    return (tFrames - 1) / mConfig.temporalCompressionFactor + 1;
}

int32_t Cosmos3PolicyRunner::activeVideoElems() const
{
    return static_cast<int32_t>(mVideoShape[1]) * mActiveT * static_cast<int32_t>(mVideoShape[3])
        * static_cast<int32_t>(mVideoShape[4]);
}

int32_t Cosmos3PolicyRunner::activeActionElems() const
{
    return mActiveActionLen * mConfig.maxActionDim;
}

void Cosmos3PolicyRunner::setActionChunkSize(int32_t chunk)
{
    // 0 (or negative) requests the profile-max (canonical) chunk; a positive value is clamped into the
    // engine's built action range at generate() time.
    mRequestedActionChunk = chunk;
}

void Cosmos3PolicyRunner::parseModelConfig(std::string const& configPath)
{
    std::ifstream f(configPath);
    ELLM_CHECK(f.is_open(), "Failed to open Cosmos3 GEN config: " + configPath);
    Json const j = Json::parse(f);

    // Every field below is part of the exported component contract; parse strictly (no fallback
    // defaults) so a stale or truncated config fails load instead of silently mis-running.
    auto requireInt = [&](char const* key) {
        ELLM_CHECK(j.contains(key), "Cosmos3 GEN config missing required key: " + std::string(key));
        return j.at(key).get<int32_t>();
    };
    auto requireFloat = [&](char const* key) {
        ELLM_CHECK(j.contains(key), "Cosmos3 GEN config missing required key: " + std::string(key));
        return j.at(key).get<float>();
    };

    mConfig.numHiddenLayers = requireInt("num_hidden_layers");
    mConfig.numKVHeads = requireInt("num_key_value_heads");
    mConfig.headDim = requireInt("head_dim");
    mConfig.ropeTheta = requireFloat("rope_theta");
    mConfig.latentChannel = requireInt("latent_channel");
    mConfig.latentPatchSize = requireInt("latent_patch_size");
    mConfig.actionChunkSize = requireInt("action_chunk_size");
    mConfig.stateRows = j.value("state_rows", 0);
    mConfig.historyLength = j.value("history_length", mConfig.stateRows);
    mConfig.useState = j.value("use_state", false);
    ELLM_CHECK(mConfig.stateRows == mConfig.historyLength, "Cosmos3 state_rows and history_length must match");
    ELLM_CHECK(mConfig.useState == (mConfig.stateRows > 0), "Cosmos3 use_state must agree with state_rows");
    mConfig.rawActionDim = requireInt("raw_action_dim");
    mConfig.maxActionDim = requireInt("max_action_dim");
    mConfig.numInferenceSteps = requireInt("num_inference_steps");
    mConfig.flowShift = requireFloat("flow_shift");
    mConfig.timestepScale = requireFloat("timestep_scale");
    mConfig.domainId = requireInt("domain_id");
    mConfig.videoLatentFrames = requireInt("video_latent_frames");
    // Lower profile bound is optional for backward compatibility with engines exported before the
    // dynamic video-token profile: absent => 1 (clamp only guards the [1, max] range).
    mConfig.minVideoLatentFrames = j.value("min_video_latent_frames", 1);
    mConfig.fps = requireFloat("fps");
    mConfig.baseFps = requireFloat("base_fps");
    mConfig.temporalCompressionFactor = requireInt("temporal_compression_factor");
    mConfig.temporalModalityMargin = requireInt("temporal_modality_margin");
    mConfig.actionStartFrameOffset = requireInt("action_start_frame_offset");

    ELLM_CHECK(j.contains("rope_scaling") && j.at("rope_scaling").contains("mrope_section"),
        "Cosmos3 GEN config missing required key: rope_scaling.mrope_section");
    auto const section = j.at("rope_scaling").at("mrope_section").get<std::vector<int32_t>>();
    ELLM_CHECK(section.size() == 3, "rope_scaling.mrope_section must have 3 entries");
    mConfig.mropeSectionH = section[1];
    mConfig.mropeSectionW = section[2];
}

void Cosmos3PolicyRunner::allocateTensors(cudaStream_t stream)
{
    // Allocate everything ONCE at the engine-profile MAXIMUM shapes: the latent grid is fixed by the
    // contract (min == opt == max), and the batch axis max is widened by the builder's --max-batch-size.
    // Per-request batches within the profile are metadata-only reshapes over these buffers.
    Dims const maxVideo = mEngine->getProfileShape(binding_names::kVideoLatent, 0, OptProfileSelector::kMAX);
    int32_t const batch = static_cast<int32_t>(maxVideo.d[0]);
    int32_t const channel = static_cast<int32_t>(maxVideo.d[1]);
    int32_t const tDim = static_cast<int32_t>(maxVideo.d[2]);
    int32_t const hDim = static_cast<int32_t>(maxVideo.d[3]);
    int32_t const wDim = static_cast<int32_t>(maxVideo.d[4]);
    // Per-axis bounds from the engine's built dynamic profile. Every device buffer is sized to the MAX
    // of each axis; per-request shapes (batch, video planes, action chunk, und len) are clamped into
    // these bounds and rebound as metadata reshapes (no hot-path allocation).
    mMinT
        = static_cast<int32_t>(mEngine->getProfileShape(binding_names::kVideoLatent, 0, OptProfileSelector::kMIN).d[2]);
    mMaxActionChunk = static_cast<int32_t>(
        mEngine->getProfileShape(binding_names::kActionLatent, 0, OptProfileSelector::kMAX).d[1]);
    mMinActionChunk = static_cast<int32_t>(
        mEngine->getProfileShape(binding_names::kActionLatent, 0, OptProfileSelector::kMIN).d[1]);
    mMaxUndLen = static_cast<int32_t>(
        mEngine->getProfileShape(binding_names::formatUndKName(0).c_str(), 0, OptProfileSelector::kMAX).d[1]);
    mMaxBatch = batch;
    ELLM_CHECK(mMaxActionChunk == mConfig.actionChunkSize + mConfig.stateRows,
        "Cosmos3 action profile must equal action_chunk_size + state_rows");
    mActiveBatch = batch;
    mVideoShape = {batch, channel, tDim, hDim, wDim};
    mActiveT = tDim;                    // default (vsf == 1) is the full/regular temporal extent.
    mActiveActionLen = mMaxActionChunk; // default request is the profile-max (canonical) action chunk.
    mVideoElems = channel * tDim * hDim * wDim;
    mActionElems = mMaxActionChunk * mConfig.maxActionDim;
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const genLen = numVideoTokens + mMaxActionChunk;

    Dims const ropeDims
        = mEngine->getProfileShape(trt_edgellm::binding_names::kRopeCosSin, 0, OptProfileSelector::kOPT);
    mRopeHeadDim = static_cast<int32_t>(ropeDims.d[ropeDims.nbDims - 1]);

    // Packed [video ⧺ action] state and prediction buffers; the engine binds the video tensor at the
    // base pointer and the action tensor at base + videoBytes (see prepareStatic).
    int64_t const totalElems = static_cast<int64_t>(batch) * (mVideoElems + mActionElems);
    mStateDevice = rt::Tensor(rt::Coords{totalElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::state");
    mPredDevice = rt::Tensor(rt::Coords{totalElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::pred");
    // mPredCondScratch (the conditional-velocity CFG blend scratch) is intentionally NOT allocated here:
    // it is only needed when guidance-interval CFG is enabled, so it is allocated lazily in setGuidance.
    mTimestepDevice
        = rt::Tensor(std::vector<int64_t>{batch}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::timestep");
    mTokenNoisyMaskDevice = rt::Tensor(
        rt::Coords{batch, numVideoTokens, 1}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::tokenNoisyMask");
    mActionNoisyMaskDevice = rt::Tensor(
        rt::Coords{batch, mMaxActionChunk, 1}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::actionNoisyMask");
    mRopeCosSinDevice = rt::Tensor(
        rt::Coords{batch, genLen, mRopeHeadDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::ropeCosSin");
    mPositionsDevice
        = rt::Tensor(rt::Coords{batch, 3, genLen}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::ropePositions");

    mTimestepHost
        = rt::Tensor(std::vector<int64_t>{batch}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::timestepHost");
    mTokenNoisyMaskHost = rt::Tensor(
        rt::Coords{batch, numVideoTokens, 1}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::tokenNoisyMaskHost");
    mActionNoisyMaskHost = rt::Tensor(
        rt::Coords{batch, mMaxActionChunk, 1}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::actionNoisyMaskHost");
    mCurrentStateHost = rt::Tensor(
        rt::Coords{batch, mConfig.rawActionDim}, rt::DeviceType::kCPU, DataType::kFLOAT, "cosmos3::currentStateHost");

    // attention_pos_id: identity gather indices 0..genLen-1; genLen is fixed for the policy grid, so the
    // buffer is filled once here.
    mPositionIdsDevice
        = rt::Tensor(rt::Coords{batch, genLen}, rt::DeviceType::kGPU, DataType::kINT32, "cosmos3::attnPosId");
    std::vector<int32_t> posIds(static_cast<size_t>(batch) * genLen);
    for (int32_t b = 0; b < batch; ++b)
    {
        std::iota(posIds.begin() + static_cast<size_t>(b) * genLen,
            posIds.begin() + (static_cast<size_t>(b) + 1) * genLen, 0);
    }
    CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), posIds.data(), posIds.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    // Device history buffers for the device-resident UniPC scheduler.
    mScheduler->prepare(totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Cosmos3PolicyRunner::setGuidance(float guidance, float intervalLo, float intervalHi)
{
    mGuidance = guidance;
    mGuidanceLo = intervalLo;
    mGuidanceHi = intervalHi;
    // Allocate the CFG blend scratch only when CFG is actually enabled; the default guidance == 1
    // single-forward path must not pay for an always-on redundant buffer on an edge target.
    if (guidance != 1.0F)
    {
        allocateCfgScratch();
    }
}

void Cosmos3PolicyRunner::allocateCfgScratch()
{
    // Guard against re-allocation if setGuidance is called more than once with CFG enabled.
    if (mPredCondScratch.getMemoryCapacity() > 0)
    {
        return;
    }
    // Same packed [video ⧺ action] layout and profile-maximum extent as mPredDevice; reshaped to the
    // active batch in generate() when CFG is active.
    int64_t const totalElems = static_cast<int64_t>(mMaxBatch) * (mVideoElems + mActionElems);
    mPredCondScratch
        = rt::Tensor(rt::Coords{totalElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::predCondScratch");
}

int64_t Cosmos3PolicyRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool Cosmos3PolicyRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (sharedContextMemory.getMemoryCapacity() < getRequiredContextMemorySize())
    {
        return false;
    }
    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

void Cosmos3PolicyRunner::buildRopeAndPositions(int32_t actionLen, int32_t undLen, cudaStream_t stream)
{
    // Reproduce the reference unified_3d_mRoPE exactly (transformer_cosmos3.py:196-353,1224-1322).
    // The GEN sequence is [video tokens (t-major) ; action tokens]. Video and action positions both start
    // at media_temporal_offset = real_text_len + temporal_modality_margin and are fps-modulated, producing
    // *float* temporal positions. The per-token (T,H,W) positions and the [genLen, headDim] transcendental
    // cos/sin cache are both computed on device (the integer-position core kernel would truncate the float
    // positions).
    int32_t const tDim = mActiveT;
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const hp = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize));
    int32_t const wp = static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize));
    int32_t const numVideoTokens = tDim * hp * wp;
    int32_t const genLen = numVideoTokens + actionLen;

    // The optimized path subsamples the generated video by mVideoSubsampleFactor, so the conditioning fps
    // that modulates the video temporal positions scales by 1/factor (matches the imaginaire4 reference's
    // conditioning_fps). factor == 1 leaves the regular positions unchanged. The action temporal stream is
    // unchanged (its chunk count and control rate do not subsample).
    double const condFps = static_cast<double>(mConfig.fps) / static_cast<double>(mVideoSubsampleFactor);
    double const mediaOffset = static_cast<double>(undLen) + static_cast<double>(mConfig.temporalModalityMargin);
    bool const videoFpsMod = (condFps > 0.0) && (tDim > 1);
    // Per the reference, the vision call passes base_temporal_compression_factor=None -> uses tcf, so the
    // video temporal stride reduces to base_fps/fps; the action call uses tcf as the *base* tcf with its own
    // tcf=1, giving stride (base_fps/base_tcf)/fps.
    double const videoBaseTps
        = static_cast<double>(mConfig.baseFps) / static_cast<double>(mConfig.temporalCompressionFactor);
    double const videoTps = condFps / static_cast<double>(mConfig.temporalCompressionFactor);
    double const actionBaseTps
        = static_cast<double>(mConfig.baseFps) / static_cast<double>(mConfig.temporalCompressionFactor);
    double const actionTps = static_cast<double>(mConfig.fps); // action temporal_compression_factor = 1.

    // Fill the [B, 3, genLen] T/H/W position planes directly on device (one thread per token); the
    // planes depend only on the fixed latent grid and are identical across batch.
    int32_t const batch = mActiveBatch;
    kernel::launchBuildMRopePositions(mPositionsDevice, batch, genLen, numVideoTokens, hp, wp,
        mConfig.actionStartFrameOffset, videoFpsMod, mediaOffset, videoTps, videoBaseTps, actionTps, actionBaseTps,
        stream);
    // The core interleaved-mRoPE kernel (float-position overload: the fps-modulated temporal
    // positions are fractional) fills the packed [B, genLen, headDim] (cos | sin) cache on device.
    trt_edgellm::kernel::initializeMRopeCosSin(static_cast<float*>(mRopeCosSinDevice.rawPointer()),
        static_cast<float*>(mPositionsDevice.rawPointer()), mConfig.ropeTheta, mRopeHeadDim, genLen, batch,
        /*interleaved=*/true, mConfig.mropeSectionH, mConfig.mropeSectionW, stream);
}

void Cosmos3PolicyRunner::initializeLatents(rt::Tensor const& condLatent, cudaStream_t stream)
{
    // Standard-normal noise over the whole packed [video ; action] state (Philox, seeded, on
    // device), then the conditioning constraints: frame-0 latent injection and padded-action-dim
    // zeroing.
    kernel::launchNormalNoiseFill(mStateDevice, static_cast<uint64_t>(mNoiseSeed), /*offset=*/0, stream);
    reinjectConditioning(condLatent, stream);
}

void Cosmos3PolicyRunner::reinjectConditioning(rt::Tensor const& condLatent, cudaStream_t stream)
{
    int32_t const batch = mActiveBatch;
    int32_t const channel = static_cast<int32_t>(mVideoShape[1]);
    int32_t const tDim = mActiveT;
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);

    rt::Coords const& condShape = condLatent.getShape();
    ELLM_CHECK(condShape.getNumDims() == 5 && condShape[0] == batch && condShape[1] == channel && condShape[2] >= 1
            && condShape[3] == hDim && condShape[4] == wDim,
        "Cosmos3 cond_latent must be [B,C,T,h,w] matching the policy latent grid");

    // video[b, c, 0, :, :] = cond[b, c, 0, :, :]: one strided device-to-device 2D copy
    // (rows = batch*channel frame-0 planes; dst row pitch spans t frames, src row pitch spans T frames).
    size_t const planeBytes = static_cast<size_t>(hDim) * wDim * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(mStateDevice.rawPointer(), static_cast<size_t>(tDim) * planeBytes,
        condLatent.rawPointer(), static_cast<size_t>(condShape[2]) * planeBytes, planeBytes,
        static_cast<size_t>(batch) * channel, cudaMemcpyDeviceToDevice, stream));

    // action[b, a, rawActionDim:maxActionDim] = 0: one strided 2D memset over the action rows.
    size_t const videoBytes = static_cast<size_t>(batch) * activeVideoElems() * sizeof(float);
    char* actionBase = static_cast<char*>(mStateDevice.rawPointer()) + videoBytes;
    size_t const rowPitch = static_cast<size_t>(mConfig.maxActionDim) * sizeof(float);
    size_t const tailBytes = static_cast<size_t>(mConfig.maxActionDim - mConfig.rawActionDim) * sizeof(float);
    CUDA_CHECK(cudaMemset2DAsync(actionBase + static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), rowPitch, 0,
        tailBytes, static_cast<size_t>(batch) * mActiveActionLen, stream));
    if (mConfig.stateRows > 0)
    {
        ELLM_CHECK(mConfig.stateRows == 1, "Cosmos3 runtime currently supports exactly one current-state row");
        size_t const stateBytes = static_cast<size_t>(mConfig.rawActionDim) * sizeof(float);
        size_t const batchActionPitch = static_cast<size_t>(mActiveActionLen) * mConfig.maxActionDim * sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(actionBase, batchActionPitch, mCurrentStateHost.rawPointer(), stateBytes,
            stateBytes, batch, cudaMemcpyHostToDevice, stream));
    }
}

void Cosmos3PolicyRunner::maskCleanPredictions(cudaStream_t stream)
{
    int32_t const batch = mActiveBatch;
    int32_t const channel = static_cast<int32_t>(mVideoShape[1]);
    int32_t const tDim = mActiveT;
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    size_t const planeBytes = static_cast<size_t>(hDim) * wDim * sizeof(float);
    char* predBase = static_cast<char*>(mPredDevice.rawPointer());

    CUDA_CHECK(cudaMemset2DAsync(
        predBase, static_cast<size_t>(tDim) * planeBytes, 0, planeBytes, static_cast<size_t>(batch) * channel, stream));

    size_t const videoBytes = static_cast<size_t>(batch) * activeVideoElems() * sizeof(float);
    char* actionPred = predBase + videoBytes;
    size_t const rowPitch = static_cast<size_t>(mConfig.maxActionDim) * sizeof(float);
    size_t const tailBytes = static_cast<size_t>(mConfig.maxActionDim - mConfig.rawActionDim) * sizeof(float);
    CUDA_CHECK(cudaMemset2DAsync(actionPred + static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), rowPitch, 0,
        tailBytes, static_cast<size_t>(batch) * mActiveActionLen, stream));
    for (int32_t b = 0; b < batch && mConfig.stateRows > 0; ++b)
    {
        char* dst = actionPred + static_cast<size_t>(b) * mActiveActionLen * mConfig.maxActionDim * sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(
            dst, 0, static_cast<size_t>(mConfig.stateRows) * mConfig.maxActionDim * sizeof(float), stream));
    }
}

void Cosmos3PolicyRunner::setDynamicInputShapes(int32_t batch, int32_t actionLen, int32_t undLen)
{
    int32_t const channel = static_cast<int32_t>(mVideoShape[1]);
    int32_t const tDim = mActiveT;
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const genLen = numVideoTokens + actionLen;

    bool ok = true;
    ok &= mContext->setInputShape(binding_names::kVideoLatent, Dims{5, {batch, channel, tDim, hDim, wDim}});
    ok &= mContext->setInputShape(binding_names::kActionLatent, Dims{3, {batch, actionLen, mConfig.maxActionDim}});
    ok &= mContext->setInputShape(binding_names::kTimestep, Dims{1, {batch}});
    ok &= mContext->setInputShape(binding_names::kTokenNoisyMask, Dims{3, {batch, numVideoTokens, 1}});
    ok &= mContext->setInputShape(binding_names::kActionNoisyMask, Dims{3, {batch, actionLen, 1}});
    ok &= mContext->setInputShape(trt_edgellm::binding_names::kRopeCosSin, Dims{3, {batch, genLen, mRopeHeadDim}});
    ok &= mContext->setInputShape(trt_edgellm::binding_names::kAttentionPosId, Dims{2, {batch, genLen}});
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        Dims const kvShape{4, {batch, undLen, mConfig.numKVHeads, mConfig.headDim}};
        ok &= mContext->setInputShape(binding_names::formatUndKName(i).c_str(), kvShape);
        ok &= mContext->setInputShape(binding_names::formatUndVName(i).c_str(), kvShape);
    }
    if (!ok)
    {
        throw std::runtime_error("Cosmos3PolicyRunner::setDynamicInputShapes failed");
    }
}

void Cosmos3PolicyRunner::prepareStatic(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
    std::vector<rt::Tensor> const& undValues, cudaStream_t stream)
{
    // One-time per-request setup. Everything here is CONSTANT across the denoising steps: the input
    // shapes, the unified_3d_mRoPE cos/sin + position ids, the noisy masks, and all tensor-address
    // bindings (latent/pred buffers are reused in place; UND K/V are frozen). Hoisting this out of the
    // per-step loop avoids redundant TRT shape re-propagation, a 3551x64 transcendental rope rebuild, and
    // 56 UND-K/V setTensorAddress calls on every step.
    int32_t const batch = mActiveBatch;
    int32_t const tDim = mActiveT;
    int32_t const hDim = static_cast<int32_t>(mVideoShape[3]);
    int32_t const wDim = static_cast<int32_t>(mVideoShape[4]);
    int32_t const numVideoTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize)) * tDim;
    int32_t const framePatchTokens = static_cast<int32_t>(divUp(hDim, mConfig.latentPatchSize))
        * static_cast<int32_t>(divUp(wDim, mConfig.latentPatchSize));

    setDynamicInputShapes(batch, actionLen, undLen);
    buildRopeAndPositions(actionLen, undLen, stream);

    // token_noisy_mask: frame-0 patch tokens are clean (0); all others noisy (1). Constant across steps.
    auto* tmask = static_cast<float*>(mTokenNoisyMaskHost.rawPointer());
    std::fill(tmask, tmask + static_cast<size_t>(batch) * numVideoTokens, 1.0F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t k = 0; k < framePatchTokens; ++k)
        {
            tmask[static_cast<size_t>(b) * numVideoTokens + k] = 0.0F;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mTokenNoisyMaskDevice.rawPointer(), tmask,
        static_cast<size_t>(batch) * numVideoTokens * sizeof(float), cudaMemcpyHostToDevice, stream));
    auto* amask = static_cast<float*>(mActionNoisyMaskHost.rawPointer());
    std::fill(amask, amask + static_cast<size_t>(batch) * actionLen, 1.0F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t row = 0; row < mConfig.stateRows; ++row)
        {
            amask[static_cast<size_t>(b) * actionLen + row] = 0.0F;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mActionNoisyMaskDevice.rawPointer(), amask,
        static_cast<size_t>(batch) * actionLen * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Bind I/O addresses once (buffers are reused in place across steps). The video/action latents and
    // predictions are views into the packed [video ⧺ action] state/pred allocations at base/base+offset.
    // The action block offset follows the ACTIVE video extent (fewer planes on the optimized path).
    size_t const videoBytes = static_cast<size_t>(batch) * activeVideoElems() * sizeof(float);
    char* stateBase = static_cast<char*>(mStateDevice.rawPointer());
    char* predBase = static_cast<char*>(mPredDevice.rawPointer());
    bool ok = true;
    ok &= mContext->setTensorAddress(binding_names::kVideoLatent, stateBase);
    ok &= mContext->setTensorAddress(binding_names::kActionLatent, stateBase + videoBytes);
    ok &= mContext->setTensorAddress(binding_names::kTimestep, mTimestepDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kTokenNoisyMask, mTokenNoisyMaskDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kActionNoisyMask, mActionNoisyMaskDevice.rawPointer());
    ok &= mContext->setTensorAddress(trt_edgellm::binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());
    ok &= mContext->setTensorAddress(trt_edgellm::binding_names::kAttentionPosId, mPositionIdsDevice.rawPointer());
    ok &= mContext->setTensorAddress(binding_names::kVideoPred, predBase);
    ok &= mContext->setTensorAddress(binding_names::kActionPred, predBase + videoBytes);
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndKName(i).c_str(), undKeys[static_cast<size_t>(i)].rawPointer());
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndVName(i).c_str(), undValues[static_cast<size_t>(i)].rawPointer());
    }
    if (!ok)
    {
        throw std::runtime_error("Cosmos3PolicyRunner::prepareStatic failed to bind GEN engine tensors");
    }
}

void Cosmos3PolicyRunner::bindUndContext(int32_t actionLen, int32_t undLen, std::vector<rt::Tensor> const& undKeys,
    std::vector<rt::Tensor> const& undValues, cudaStream_t stream)
{
    // Point the cross-attention context at a different UND K/V set: the und_k/und_v input shapes carry
    // undLen, and the GEN mRoPE media positions are offset by undLen (buildRopeAndPositions), so both the
    // shapes and the rope cache must be rebuilt when the context length changes.
    setDynamicInputShapes(mActiveBatch, actionLen, undLen);
    buildRopeAndPositions(actionLen, undLen, stream);
    bool ok = true;
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndKName(i).c_str(), undKeys[static_cast<size_t>(i)].rawPointer());
        ok &= mContext->setInputTensorAddress(
            binding_names::formatUndVName(i).c_str(), undValues[static_cast<size_t>(i)].rawPointer());
    }
    ELLM_CHECK(ok, "Cosmos3PolicyRunner::bindUndContext failed to bind und K/V");
}

bool Cosmos3PolicyRunner::runDenoiseStep(int32_t stepIdx, rt::Tensor const& condLatent, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(stepRange, "cosmos3::denoise_step");
    int32_t const batch = mActiveBatch;

    // timestep (raw; the graph applies timestep_scale internally). The only per-step engine input besides
    // the latents (a scalar per batch element).
    float const timestep = mScheduler->timestepAt(stepIdx);
    auto* tsHost = static_cast<float*>(mTimestepHost.rawPointer());
    std::fill(tsHost, tsHost + batch, timestep);
    CUDA_CHECK(cudaMemcpyAsync(mTimestepDevice.rawPointer(), tsHost, static_cast<size_t>(batch) * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    // Guidance-interval CFG: only the step(s) whose timestep falls inside (lo, hi) pay the extra
    // unconditional forward (with the reference [999,937,833,624]/shift-5 schedule that is the first
    // step only). Everything else runs the original single conditional forward.
    bool const cfgThisStep = mCfgActive && timestep > mGuidanceLo && timestep < mGuidanceHi;

    if (cfgThisStep)
    {
        // Conditional forward (the conditional context is bound coming into the step).
        if (!mContext->enqueueV3(stream))
        {
            LOG_ERROR("Cosmos3PolicyRunner: conditional GEN forward failed at step %d", stepIdx);
            return false;
        }
        int64_t const predBytes = mPredDevice.getShape().volume() * static_cast<int64_t>(sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(mPredCondScratch.rawPointer(), mPredDevice.rawPointer(),
            static_cast<size_t>(predBytes), cudaMemcpyDeviceToDevice, stream));

        // Unconditional forward: swap to the empty-prompt context, run, then restore the conditional one.
        bindUndContext(mCfgActionLen, mCfgUncondUndLen, *mCfgUndKeysUncond, *mCfgUndValuesUncond, stream);
        if (!mContext->enqueueV3(stream))
        {
            LOG_ERROR("Cosmos3PolicyRunner: unconditional GEN forward failed at step %d", stepIdx);
            return false;
        }
        // v = v_uncond + guidance * (v_cond - v_uncond) = guidance * v_cond + (1 - guidance) * v_uncond.
        // mPredDevice currently holds v_uncond; mPredCondScratch holds v_cond (aliasing v1==out is safe:
        // fusedCombineKernel reads then writes per element).
        kernel::launchFusedCombine(mPredDevice, mGuidance, mPredCondScratch, 1.0F - mGuidance, mPredDevice, 0.0F,
            nullptr, 0.0F, nullptr, stream);
        bindUndContext(mCfgActionLen, mCfgCondUndLen, *mCfgUndKeysCond, *mCfgUndValuesCond, stream);
    }
    else
    {
        bool const engineOk = mCurrentGraphExec != nullptr ? (cudaGraphLaunch(mCurrentGraphExec, stream) == cudaSuccess)
                                                           : mContext->enqueueV3(stream);
        if (!engineOk)
        {
            LOG_ERROR("Cosmos3PolicyRunner: GEN forward failed at step %d (cudagraph=%d)", stepIdx,
                static_cast<int32_t>(mCurrentGraphExec != nullptr));
            return false;
        }
    }

    // The GEN engine uses a TensorRT auxiliary stream, while the sampler immediately reads and then
    // overwrites its shared prediction/state buffers.  On TensorRT 11.0 / GB300, relying on enqueueV3's
    // main-stream join here changed the deterministic action output; complete the GEN work before the
    // first non-TensorRT consumer touches those buffers.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Device-resident UniPC update over the packed [video ⧺ action] state, then re-impose the
    // conditioning constraints (frame-0 latent, zero padded action dims) — all on device; the
    // denoising state never visits the host.
    maskCleanPredictions(stream);
    mScheduler->step(mPredDevice, mStateDevice, stepIdx, stream);
    reinjectConditioning(condLatent, stream);
    return true;
}

std::vector<float> Cosmos3PolicyRunner::generate(rt::Tensor const& condLatent, std::vector<rt::Tensor> const& undKeys,
    std::vector<rt::Tensor> const& undValues, std::vector<rt::Tensor> const& undKeysUncond,
    std::vector<rt::Tensor> const& undValuesUncond, std::vector<float> const& currentState, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(genRange, "cosmos3::gen_denoise");

    std::vector<float> result;
    if (static_cast<int32_t>(undKeys.size()) != mConfig.numHiddenLayers
        || static_cast<int32_t>(undValues.size()) != mConfig.numHiddenLayers)
    {
        LOG_ERROR("Cosmos3PolicyRunner: expected %d UND K/V layers, got %zu/%zu", mConfig.numHiddenLayers,
            undKeys.size(), undValues.size());
        return result;
    }
    int32_t const undLen = static_cast<int32_t>(undKeys.front().getShape()[1]);

    // Per-request batch, derived from the conditioning latent (any value within the engine profile).
    ELLM_CHECK(condLatent.getShape().getNumDims() == 5, "cond_latent must be [B,C,T,h,w]");
    int32_t const batch = static_cast<int32_t>(condLatent.getShape()[0]);
    ELLM_CHECK(batch >= 1 && batch <= mMaxBatch,
        "Cosmos3 policy request batch " + std::to_string(batch) + " exceeds the engine profile maximum "
            + std::to_string(mMaxBatch) + " (build with a larger --max-batch-size)");
    ELLM_CHECK(static_cast<int32_t>(undKeys.front().getShape()[0]) == batch,
        "UND K/V batch does not match the conditioning latent batch");
    mActiveBatch = batch;
    if (mConfig.stateRows > 0)
    {
        ELLM_CHECK(currentState.size() == static_cast<size_t>(batch) * mConfig.rawActionDim,
            "State-conditioned Cosmos3 policy requires one raw_action_dim state row per batch item");
        std::copy(currentState.begin(), currentState.end(), static_cast<float*>(mCurrentStateHost.rawPointer()));
    }
    // Resolve the per-request shape on every dynamic GEN axis and clamp each into the engine's built
    // profile [min, max]; warn once per axis (only when the out-of-range request value changes) so a
    // serving/benchmark loop does not spam. Buffers are preallocated at each axis max, so the smaller
    // request is a metadata-only reshape below.
    auto warnClamp = [&](char const* axis, int32_t requested, int32_t clamped, int32_t lo, int32_t hi,
                         int32_t& lastWarned) {
        if (clamped != requested && requested != lastWarned)
        {
            LOG_WARNING("Cosmos3PolicyRunner: %s %d is outside the engine's built profile [%d, %d]; clamped to %d.",
                axis, requested, lo, hi, clamped);
            lastWarned = requested;
        }
    };
    int32_t const maxT = static_cast<int32_t>(mVideoShape[2]);
    int32_t const reqT = requestedTemporal();
    mActiveT = std::max(mMinT, std::min(reqT, maxT));
    warnClamp("video latent planes", reqT, mActiveT, mMinT, maxT, mLastWarnedT);

    int32_t const reqAction = mRequestedActionChunk > 0 ? mRequestedActionChunk + mConfig.stateRows : mMaxActionChunk;
    mActiveActionLen = std::max(mMinActionChunk, std::min(reqAction, mMaxActionChunk));
    warnClamp("action_chunk_size", reqAction, mActiveActionLen, mMinActionChunk, mMaxActionChunk, mLastWarnedAction);

    mActiveUndLen = std::min(undLen, mMaxUndLen); // GEN und profile min is 1; only the upper bound needs a guard.
    warnClamp("und_len", undLen, mActiveUndLen, 1, mMaxUndLen, mLastWarnedUnd);

    // Metadata-only reshape of the packed state/pred buffers and the scheduler history to the active
    // batch, video extent, and action chunk (allocated at the profile maximum in the constructor).
    int64_t const activeElems = static_cast<int64_t>(batch) * (activeVideoElems() + activeActionElems());
    ELLM_CHECK(mStateDevice.reshape(rt::Coords{activeElems}), "Cosmos3 state reshape failed");
    ELLM_CHECK(mPredDevice.reshape(rt::Coords{activeElems}), "Cosmos3 pred reshape failed");
    mScheduler->prepare(activeElems);

    // Guidance-interval CFG setup. Active only when guidance != 1 and an unconditional (empty-prompt)
    // UND K/V set is supplied. It rebinds engine shapes/addresses on the guided step, so it is
    // incompatible with CUDA-graph replay; the graph path is skipped for this request below (without
    // clearing mUseCudaGraph, so a later non-CFG request still replays cached graphs).
    mCfgActive = (mGuidance != 1.0F) && !undKeysUncond.empty() && !undValuesUncond.empty();
    if (mCfgActive)
    {
        ELLM_CHECK(static_cast<int32_t>(undKeysUncond.size()) == mConfig.numHiddenLayers
                && static_cast<int32_t>(undValuesUncond.size()) == mConfig.numHiddenLayers,
            "CFG unconditional UND K/V must have numHiddenLayers entries");
        ELLM_CHECK(static_cast<int32_t>(undKeysUncond.front().getShape()[0]) == batch,
            "CFG unconditional UND K/V batch does not match the request batch");
        // The blend scratch is allocated by setGuidance when CFG is enabled; reshape it to the active
        // batch here (only on the CFG path, so the non-CFG path never touches this buffer).
        ELLM_CHECK(mPredCondScratch.getMemoryCapacity() > 0, "CFG enabled but pred-scratch not allocated");
        ELLM_CHECK(mPredCondScratch.reshape(rt::Coords{activeElems}), "Cosmos3 pred-scratch reshape failed");
        mCfgActionLen = mActiveActionLen;
        mCfgCondUndLen = mActiveUndLen;
        mCfgUncondUndLen = static_cast<int32_t>(undKeysUncond.front().getShape()[1]);
        mCfgUndKeysCond = &undKeys;
        mCfgUndValuesCond = &undValues;
        mCfgUndKeysUncond = &undKeysUncond;
        mCfgUndValuesUncond = &undValuesUncond;
    }

    mScheduler->initialize(mConfig.numInferenceSteps);

    initializeLatents(condLatent, stream);

    // One-time setup: shapes, mRoPE, masks, and tensor bindings are constant across the denoising steps.
    prepareStatic(mActiveActionLen, mActiveUndLen, undKeys, undValues, stream);

    // CUDA-graph capture/replay of the per-step GEN engine forward. Shapes and I/O addresses are now
    // fixed for this (batch, activeT, action, und) signature; only the timestep + latent buffer contents
    // change per step (updated in place). Each distinct workload signature is captured once and cached;
    // a later request of the same signature replays the cached graph without re-capturing.
    // CFG rebinds engine shapes/addresses on the guided step, so it cannot replay a captured graph;
    // skip the graph path for this request only. mUseCudaGraph is preserved so later non-CFG requests
    // still use graphs.
    bool const useGraph = mUseCudaGraph && !mCfgActive;
    mCurrentGraphExec = nullptr;
    if (useGraph)
    {
        mCurrentGraphExec = findGenGraph(mActiveBatch, mActiveT, mActiveActionLen, mActiveUndLen);
        if (mCurrentGraphExec == nullptr && mGenGraphs.size() >= kMaxCachedGenGraphs)
        {
            // Cache is full: run this (uncached) signature with enqueueV3 rather than leaking another
            // cudaGraphExec_t. Warn once so a pathological signature stream is visible without spamming.
            if (!mGraphCacheFull)
            {
                LOG_WARNING(
                    "Cosmos3PolicyRunner: CUDA-graph cache reached its cap of %zu signatures; running "
                    "further signatures with enqueueV3.",
                    kMaxCachedGenGraphs);
                mGraphCacheFull = true;
            }
        }
        else if (mCurrentGraphExec == nullptr)
        {
            if (!mContext->enqueueV3(stream)) // warmup so TRT internal state is initialized before capture
            {
                LOG_ERROR("Cosmos3PolicyRunner: warmup enqueueV3 failed before CUDA-graph capture");
                return result;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto captured = captureTRTCudaGraph(mContext.get(), stream);
            if (captured.has_value())
            {
                mGenGraphs.push_back(GenGraph{
                    mActiveBatch, mActiveT, mActiveActionLen, mActiveUndLen, captured->first, captured->second});
                mCurrentGraphExec = captured->second;
                LOG_INFO(
                    "Cosmos3PolicyRunner: captured GEN denoise step into a CUDA graph "
                    "(batch %d, t %d, action %d, und %d)",
                    mActiveBatch, mActiveT, mActiveActionLen, mActiveUndLen);
            }
            else
            {
                LOG_WARNING("Cosmos3PolicyRunner: CUDA-graph capture failed; using enqueueV3");
                mUseCudaGraph = false;
            }
        }
    }

    int32_t const numSteps = mScheduler->numInferenceSteps();
    for (int32_t step = 0; step < numSteps; ++step)
    {
        if (!runDenoiseStep(step, condLatent, stream))
        {
            return result;
        }
    }

    // Slice the action chunk action_latent[:, :, :rawActionDim] straight out of the packed device state:
    // one strided 2D D2H copy (row = one action step; rawActionDim of maxActionDim columns).
    size_t const videoBytes = static_cast<size_t>(batch) * activeVideoElems() * sizeof(float);
    int32_t const outputRows = mActiveActionLen - mConfig.stateRows;
    ELLM_CHECK(outputRows > 0, "Cosmos3 policy has no predicted rows after dropping state context");
    result.resize(static_cast<size_t>(batch) * outputRows * mConfig.rawActionDim);
    char const* actionBase = static_cast<char const*>(mStateDevice.rawPointer()) + videoBytes;
    for (int32_t b = 0; b < batch; ++b)
    {
        char const* src = actionBase
            + (static_cast<size_t>(b) * mActiveActionLen + mConfig.stateRows) * mConfig.maxActionDim * sizeof(float);
        float* dst = result.data() + static_cast<size_t>(b) * outputRows * mConfig.rawActionDim;
        CUDA_CHECK(cudaMemcpy2DAsync(dst, static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), src,
            static_cast<size_t>(mConfig.maxActionDim) * sizeof(float),
            static_cast<size_t>(mConfig.rawActionDim) * sizeof(float), outputRows, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace cosmos3
} // namespace trt_edgellm
