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

#include "action/pi05ActionRunner.h"

#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/pi05Bindings.h"
#include "kernels/pi05Kernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>

using nvinfer1::DataType;
using nvinfer1::Dims;

namespace trt_edgellm
{
namespace pi05
{
namespace
{
//! The extent that puts the AttentionPlugin in tree decoding: a denoise step attends a
//! fixed query chunk over the prefix cache plus its same-step appended K/V.
constexpr int64_t kTreeDecodingPhaseExtent = static_cast<int64_t>(rt::ExecutionPhase::kDiffusionDenoise);
} // namespace

Pi05ActionRunner::Pi05ActionRunner(std::string const& engineDir, Pi05PolicyConfig const& config, cudaStream_t stream)
    : mConfig(config)
    , mStream(stream)
{
    mAction.load(engineDir, "action");
    if (mConfig.hoistedAdarmsCond)
    {
        mCond.load(engineDir, "cond");
    }
    mNumDenoiseSteps = mConfig.numDenoiseSteps;
    allocateTensors();

    // Row S of the table is the expert's first position, so it covers the prefix span too.
    kernel::initializeNormalRopeCosSin(mRopeCache.dataPointer<float>(), mConfig.ropeTheta, 1.0F, 1.0F, mConfig.headDim,
        mConfig.maxPrefixLen + mConfig.actionHorizon, mStream);

    CUDA_CHECK(cudaEventCreate(&mBegin));
    CUDA_CHECK(cudaEventCreate(&mEnd));
}

Pi05ActionRunner::~Pi05ActionRunner() noexcept
{
    invalidateGraph();
    for (cudaEvent_t event : {mBegin, mEnd})
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

int64_t Pi05ActionRunner::getRequiredContextMemorySize() const
{
    return std::max(mAction.getRequiredContextMemorySize(), mCond.getRequiredContextMemorySize());
}

bool Pi05ActionRunner::setContextMemory(rt::Tensor& pool)
{
    return mAction.setContextMemory(pool) && mCond.setContextMemory(pool);
}

void Pi05ActionRunner::allocateTensors()
{
    // The profile, not the contract's max_batch_size, decides what the loaded engine
    // accepts: a rebuild at a different --maxBatchSize leaves the exported config alone.
    Dims const actionMax
        = mAction.engine->getProfileShape(binding_names::kNoiseTrajectory, 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxBatch = static_cast<int32_t>(actionMax.d[0]);
    if (mMaxBatch < 1)
    {
        throw std::runtime_error("pi0.5 action engine admits no request batch; rebuild it");
    }

    int32_t const maxBatch = mMaxBatch;
    int32_t const horizon = mConfig.actionHorizon;
    int32_t const actionDim = mConfig.actionDim;
    int32_t const headDim = mConfig.headDim;

    mKVSeqLens = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::kvSeqLens");
    mKVSeqLensHost
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::kvSeqLensHost");
    allocatePagedInputs(maxBatch);

    mRopeCache = rt::Tensor(
        {mConfig.maxPrefixLen + horizon, headDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::ropeCache");
    mActionRopeCosSin
        = rt::Tensor({maxBatch, horizon, headDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::actionRope");
    mActionPosIds = rt::Tensor({maxBatch, horizon}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::actionPos");
    mActionPosIdsHost = rt::Tensor({maxBatch, horizon}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::actionPosHost");

    mNoiseDevice = rt::Tensor({maxBatch, horizon, actionDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::noise");
    mNoiseHost = rt::Tensor({maxBatch, horizon, actionDim}, rt::DeviceType::kCPU, DataType::kFLOAT, "pi05::noiseHost");
    mPredDevice = rt::Tensor({maxBatch, horizon, actionDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::pred");
    mTimestepDevice
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::timestep");
    mTimestepHost
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kCPU, DataType::kFLOAT, "pi05::timestepHost");

    if (!mConfig.hoistedAdarmsCond)
    {
        return;
    }
    // One allocation for the whole schedule: the captured graph bakes a distinct
    // slice address per step, so the base must not move once a graph exists.
    int64_t const maxRows = static_cast<int64_t>(mConfig.maxDenoiseSteps) * maxBatch;
    mAdarmsMod = rt::Tensor({maxRows, mConfig.numAdarmsSites, mConfig.modulationDim}, rt::DeviceType::kGPU,
        DataType::kHALF, "pi05::adarmsMod");
    mCondTimestepDevice
        = rt::Tensor(std::vector<int64_t>{maxRows}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::condTimestep");
    mCondTimestepHost
        = rt::Tensor(std::vector<int64_t>{maxRows}, rt::DeviceType::kCPU, DataType::kFLOAT, "pi05::condTimestepHost");
}

void Pi05ActionRunner::allocatePagedInputs(int32_t maxBatch)
{
    int32_t const pages = pagesPerSeq(mConfig);
    int32_t const horizon = mConfig.actionHorizon;
    int32_t const words = (horizon + 31) / 32;

    mKVPageTable = rt::Tensor({maxBatch, 2, pages}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::kvPageTable");
    mKVPageTableHost
        = rt::Tensor({maxBatch, 2, pages}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::kvPageTableHost");
    mAttentionMask
        = rt::Tensor({maxBatch * horizon, words}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::attentionMask");
    mKVCacheStartIdx
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::kvCacheStartIdx");

    mQueryLengths
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::queryLengths");
    mQueryLengthsHost
        = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::queryLengthsHost");
    mQueryStartOffsets = rt::Tensor(
        std::vector<int64_t>{maxBatch + 1}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::queryStartOffsets");
    mQueryStartOffsetsHost = rt::Tensor(
        std::vector<int64_t>{maxBatch + 1}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::queryStartOffsetsHost");
    std::fill_n(mQueryLengthsHost.dataPointer<int32_t>(), maxBatch, horizon);
    int32_t* const offsets = mQueryStartOffsetsHost.dataPointer<int32_t>();
    for (int32_t slot = 0; slot <= maxBatch; ++slot)
    {
        offsets[slot] = slot * horizon;
    }
    // Pinned members, written here and never again, so the copies below stay valid
    // without a host-side sync.
    CUDA_CHECK(cudaMemcpyAsync(mQueryLengths.rawPointer(), mQueryLengthsHost.rawPointer(),
        mQueryLengthsHost.getMemoryCapacity(), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mQueryStartOffsets.rawPointer(), mQueryStartOffsetsHost.rawPointer(),
        mQueryStartOffsetsHost.getMemoryCapacity(), cudaMemcpyHostToDevice, mStream));

    // Both carriers select through their extent and neither is read, so one allocation
    // serves both: the context-count carrier binds this address at extent 0.
    mPhaseMarker = rt::Tensor(
        std::vector<int64_t>{kTreeDecodingPhaseExtent}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::phaseMarker");

    // Every query token attends to every other; the bits past the horizon are unread.
    CUDA_CHECK(cudaMemsetAsync(mAttentionMask.rawPointer(), 0xFF, mAttentionMask.getMemoryCapacity(), mStream));
    CUDA_CHECK(cudaMemsetAsync(mKVCacheStartIdx.rawPointer(), 0, mKVCacheStartIdx.getMemoryCapacity(), mStream));
}

bool Pi05ActionRunner::refreshPageTable(int64_t numPages)
{
    if (numPages == mPageTableNumPages)
    {
        return true;
    }
    int32_t const pages = pagesPerSeq(mConfig);
    int32_t const slots = static_cast<int32_t>(mKVPageTable.getShape()[0]);
    // The engine's profile can admit more slots than the pool the runtime allocated;
    // rows past what the pool covers stay unmapped and are never bound.
    auto const mapped = static_cast<int32_t>(std::min<int64_t>(slots, numPages / pages));
    if (mapped < 1)
    {
        LOG_ERROR("pi0.5 paged K/V pool holds %ld pages, short of the %d one request needs",
            static_cast<long>(numPages), pages);
        return false;
    }
    // Each batch slot owns pages [slot*pages, (slot+1)*pages) of each plane, and the V plane's ids
    // continue past the pool's page count -- the layout the fused RoPE/append kernel
    // and XQA both address.
    int32_t* host = mKVPageTableHost.dataPointer<int32_t>();
    std::fill_n(host, static_cast<size_t>(slots) * 2 * pages, rt::kUNUSED_PAGE_ENTRY);
    for (int32_t slot = 0; slot < mapped; ++slot)
    {
        for (int32_t page = 0; page < pages; ++page)
        {
            int32_t const kPage = slot * pages + page;
            host[(slot * 2 + 0) * pages + page] = kPage;
            host[(slot * 2 + 1) * pages + page] = kPage + static_cast<int32_t>(numPages);
        }
    }
    // The staging buffer is a pinned member rather than a scoped temporary, so no sync is
    // needed to keep it alive. Safe to rewrite because generate() drains the stream before
    // returning, so no earlier copy still reads it.
    CUDA_CHECK(cudaMemcpyAsync(mKVPageTable.rawPointer(), mKVPageTableHost.rawPointer(),
        mKVPageTableHost.getMemoryCapacity(), cudaMemcpyHostToDevice, mStream));
    mPageTableNumPages = numPages;
    return true;
}

bool Pi05ActionRunner::computeModulation()
{
    // The schedule is known before the request starts, so repeating each timestep per
    // request makes the cond output already the [step][request] layout the expert slices.
    float const dt = -1.0F / static_cast<float>(mNumDenoiseSteps);
    int32_t const rows = mNumDenoiseSteps * mActiveBatch;
    float* host = mCondTimestepHost.dataPointer<float>();
    for (int32_t step = 0; step < mNumDenoiseSteps; ++step)
    {
        std::fill_n(
            host + static_cast<size_t>(step) * mActiveBatch, mActiveBatch, 1.0F + dt * static_cast<float>(step));
    }
    CUDA_CHECK(cudaMemcpyAsync(mCondTimestepDevice.rawPointer(), host, static_cast<size_t>(rows) * sizeof(float),
        cudaMemcpyHostToDevice, mStream));

    int64_t const rowElems = static_cast<int64_t>(mConfig.numAdarmsSites) * mConfig.modulationDim;
    // The cond profile bounds the number of timesteps per run, not the batch, so a
    // long schedule at batch runs as successive slices of the same allocation.
    for (int32_t begin = 0; begin < rows; begin += mConfig.maxDenoiseSteps)
    {
        int32_t const len = std::min(mConfig.maxDenoiseSteps, rows - begin);
        bool ok = mCond.context->setInputShape(binding_names::kTimestep, Dims{1, {len}});
        ok &= mCond.context->setTensorAddress(
            binding_names::kTimestep, mCondTimestepDevice.dataPointer<float>() + begin);
        ok &= mCond.context->setTensorAddress(
            binding_names::kAdarmsModulation, mAdarmsMod.dataPointer<half>() + begin * rowElems);
        if (!ok || !mCond.context->enqueueV3(mStream))
        {
            LOG_ERROR("pi0.5 cond engine execution failed");
            return false;
        }
    }
    mModulationSteps = mNumDenoiseSteps;
    mModulationBatch = mActiveBatch;
    LOG_INFO(
        "Precomputed the pi0.5 AdaRMS modulation for %d denoise steps at batch %d", mNumDenoiseSteps, mActiveBatch);
    return true;
}

void Pi05ActionRunner::setUseCudaGraph(bool enable) noexcept
{
    if (mUseCudaGraph != enable)
    {
        invalidateGraph();
    }
    mUseCudaGraph = enable;
}

void Pi05ActionRunner::setNumDenoiseSteps(int32_t steps)
{
    if (steps < 1)
    {
        throw std::invalid_argument("pi0.5 denoise step count must be >= 1");
    }
    if (mConfig.hoistedAdarmsCond && steps > mConfig.maxDenoiseSteps)
    {
        throw std::invalid_argument("pi0.5 denoise step count exceeds the cond engine's profile ("
            + std::to_string(mConfig.maxDenoiseSteps) + "); re-export with a larger MAX_DENOISE_STEPS");
    }
    if (steps != mNumDenoiseSteps)
    {
        invalidateGraph();
    }
    mNumDenoiseSteps = steps;
}

void Pi05ActionRunner::invalidateGraph() noexcept
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
    mGraphKVCache.clear();
    mGraphReady = false;
}

bool Pi05ActionRunner::capturedKVCacheMoved(std::vector<rt::Tensor> const& kvCache) const noexcept
{
    for (size_t i = 0; i < mGraphKVCache.size(); ++i)
    {
        if (mGraphKVCache[i] != kvCache[i].rawPointer())
        {
            return true;
        }
    }
    return false;
}

void Pi05ActionRunner::setInitialNoise(std::vector<float> noise)
{
    mExternalNoise = std::move(noise);
}

void Pi05ActionRunner::initializeNoise(int32_t batch)
{
    size_t const expected = static_cast<size_t>(batch) * static_cast<size_t>(mConfig.actionHorizon)
        * static_cast<size_t>(mConfig.actionDim);
    if (!mExternalNoise.empty())
    {
        if (mExternalNoise.size() != expected)
        {
            throw std::runtime_error("pi0.5 external noise size does not match [B, horizon, action_dim]");
        }
        std::copy(mExternalNoise.begin(), mExternalNoise.end(), mNoiseHost.dataPointer<float>());
        return;
    }
    size_t const chunk = static_cast<size_t>(mConfig.actionHorizon) * static_cast<size_t>(mConfig.actionDim);
    float* data = mNoiseHost.dataPointer<float>();
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (size_t i = 0; i < chunk; ++i)
    {
        data[i] = dist(mNoiseGen);
    }
    // A seeded batch is one request replicated, so every entry starts from the same x_0.
    // Drawing per entry instead makes the spread the caller measures track the noise.
    for (int32_t b = 1; b < batch; ++b)
    {
        std::copy_n(data, chunk, data + static_cast<size_t>(b) * chunk);
    }
}

std::vector<float> Pi05ActionRunner::generate(std::vector<rt::Tensor>& kvCache, int32_t batch, int32_t prefixLen)
{
    // The buffers below are sized from the export's profile, and the first of them is
    // written before the engine sees a shape, so the request is screened here instead.
    std::vector<float> result;
    if (batch < 1 || batch > mMaxBatch)
    {
        LOG_ERROR("pi0.5 action request batch %d is outside the range the action engine was built for (1 to %d)", batch,
            mMaxBatch);
        return result;
    }
    if (prefixLen < 1)
    {
        LOG_ERROR("pi0.5 action runner was given an empty prefix; the K/V cache was never filled");
        return result;
    }
    if (prefixLen > mConfig.maxPrefixLen)
    {
        LOG_ERROR("pi0.5 prefix length %d exceeds the RoPE table the export was built for (%d)", prefixLen,
            mConfig.maxPrefixLen);
        return result;
    }
    if (kvCache.size() < static_cast<size_t>(mConfig.numHiddenLayers))
    {
        LOG_ERROR("pi0.5 action runner was given %zu K/V cache layers but the expert binds %d", kvCache.size(),
            mConfig.numHiddenLayers);
        return result;
    }

    cudaEventRecord(mBegin, mStream);
    if (batch != mActiveBatch || capturedKVCacheMoved(kvCache))
    {
        // A capture bakes in the batch-dependent shapes, the cache addresses and the
        // modulation stride, but not prefixLen: it reaches only buffers that keep their
        // addresses and are restaged on this stream before every launch.
        invalidateGraph();
    }
    mActiveBatch = batch;
    mPrefixLen = prefixLen;

    int32_t const activeBatch = mActiveBatch;
    int32_t const horizon = mConfig.actionHorizon;
    int32_t const actionDim = mConfig.actionDim;
    size_t const elems
        = static_cast<size_t>(activeBatch) * static_cast<size_t>(horizon) * static_cast<size_t>(actionDim);

    initializeNoise(activeBatch);

    // Both x_t and the timestep are integrated in place on device, so anything that runs the
    // loop -- including the capture warmup -- consumes them. Every entry into the loop restores
    // both; restoring only x_t leaves the first graph replay starting from t = 0.
    auto resetLoopState = [&] {
        CUDA_CHECK(cudaMemcpyAsync(mNoiseDevice.rawPointer(), mNoiseHost.rawPointer(), elems * sizeof(float),
            cudaMemcpyHostToDevice, mStream));
        if (!mConfig.hoistedAdarmsCond)
        {
            for (int32_t i = 0; i < activeBatch; ++i)
            {
                mTimestepHost.dataPointer<float>()[i] = 1.0F;
            }
            CUDA_CHECK(cudaMemcpyAsync(mTimestepDevice.rawPointer(), mTimestepHost.rawPointer(),
                static_cast<size_t>(activeBatch) * sizeof(float), cudaMemcpyHostToDevice, mStream));
        }
    };
    resetLoopState();

    // The expert's tokens follow the prefix, so its rows start at prefixLen.
    stageRopeInputs(mRopeCache, mConfig.headDim, activeBatch, horizon, mPrefixLen, mActionRopeCosSin, mActionPosIds,
        mActionPosIdsHost, mStream);

    // Token-major bindings: one row per query token, requests concatenated.
    int64_t const execTokens = static_cast<int64_t>(activeBatch) * horizon;
    bool ok
        = mAction.context->setInputShape(binding_names::kNoiseTrajectory, Dims{3, {activeBatch, horizon, actionDim}});
    ok &= mAction.context->setInputShape(binding_names::kRopeCosSin, Dims{2, {execTokens, mConfig.headDim}});
    ok &= mAction.context->setInputShape(binding_names::kAttentionPosId, Dims{1, {execTokens}});
    ok &= mAction.context->setTensorAddress(binding_names::kNoiseTrajectory, mNoiseDevice.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kAttentionPosId, mActionPosIds.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kActionPred, mPredDevice.rawPointer());

    // Hoisted: one modulation row per (step, request), sliced from mAdarmsMod below.
    int64_t const modStepElems
        = static_cast<int64_t>(activeBatch) * mConfig.numAdarmsSites * static_cast<int64_t>(mConfig.modulationDim);
    if (mConfig.hoistedAdarmsCond)
    {
        if ((mModulationSteps != mNumDenoiseSteps || mModulationBatch != activeBatch) && !computeModulation())
        {
            return result;
        }
        ok &= mAction.context->setInputShape(
            binding_names::kAdarmsModulation, Dims{3, {activeBatch, mConfig.numAdarmsSites, mConfig.modulationDim}});
    }
    else
    {
        ok &= mAction.context->setInputShape(binding_names::kTimestep, Dims{1, {activeBatch}});
        ok &= mAction.context->setTensorAddress(binding_names::kTimestep, mTimestepDevice.rawPointer());
    }

    ok &= mAction.context->setTensorAddress(binding_names::kRopeCosSin, mActionRopeCosSin.rawPointer());

    // The plugin writes the action tokens into the slots just below this
    // length, so it is the post-append cache length, not the prefix length.
    int64_t const kvLen = static_cast<int64_t>(mPrefixLen) + horizon;
    if (kvLen > mConfig.kvCacheCapacity)
    {
        LOG_ERROR("pi0.5 prefix %d plus horizon %d exceeds the K/V cache capacity %d", mPrefixLen, horizon,
            mConfig.kvCacheCapacity);
        return result;
    }
    std::fill_n(mKVSeqLensHost.dataPointer<int32_t>(), activeBatch, static_cast<int32_t>(kvLen));
    CUDA_CHECK(cudaMemcpyAsync(mKVSeqLens.rawPointer(), mKVSeqLensHost.rawPointer(),
        static_cast<size_t>(activeBatch) * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
    // The pool spans every slot whatever the request batch, so the page-id space
    // the table indexes does not move with the batch.
    int32_t const pages = pagesPerSeq(mConfig);
    int64_t const numPages = kvCache[0].getShape()[1];
    if (!refreshPageTable(numPages))
    {
        return result;
    }
    Dims const cacheShape{5, {2, numPages, rt::kTOKENS_PER_PAGE, mConfig.numKVHeads, mConfig.headDim}};
    ok &= mAction.context->setInputShape(binding_names::kKVCacheStartIndex, Dims{1, {activeBatch}});
    ok &= mAction.context->setInputShape(binding_names::kKVPageTable, Dims{3, {activeBatch, 2, pages}});
    ok &= mAction.context->setInputShape(binding_names::kAttentionMask, Dims{2, {execTokens, (horizon + 31) / 32}});
    ok &= mAction.context->setTensorAddress(binding_names::kKVCacheStartIndex, mKVCacheStartIdx.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kKVPageTable, mKVPageTable.rawPointer());
    ok &= mAction.context->setTensorAddress(binding_names::kAttentionMask, mAttentionMask.rawPointer());
    // The two length inputs are not interchangeable: query_lengths counts the rows
    // this request contributes, attention_sequence_lengths the cache behind them.
    ok &= mAction.context->setInputShape(binding_names::kQueryLengths, Dims{1, {activeBatch}});
    ok &= mAction.context->setTensorAddress(binding_names::kQueryLengths, mQueryLengths.rawPointer());
    ok &= mAction.context->setInputShape(binding_names::kQueryStartOffsets, Dims{1, {activeBatch + 1}});
    ok &= mAction.context->setTensorAddress(binding_names::kQueryStartOffsets, mQueryStartOffsets.rawPointer());
    ok &= mAction.context->setInputShape(binding_names::kAttentionSequenceLengths, Dims{1, {activeBatch}});
    ok &= mAction.context->setTensorAddress(binding_names::kAttentionSequenceLengths, mKVSeqLens.rawPointer());
    ok &= mAction.context->setInputShape(binding_names::kExecutionPhaseMarker, Dims{1, {kTreeDecodingPhaseExtent}});
    ok &= mAction.context->setTensorAddress(binding_names::kExecutionPhaseMarker, mPhaseMarker.rawPointer());
    ok &= mAction.context->setInputShape(binding_names::kContextSequenceCountCarrier, Dims{1, {0}});
    ok &= mAction.context->setTensorAddress(binding_names::kContextSequenceCountCarrier, mPhaseMarker.rawPointer());
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        ok &= mAction.context->setInputShape(binding_names::formatActionKVCacheName(i).c_str(), cacheShape);
        ok &= mAction.context->setTensorAddress(
            binding_names::formatActionKVCacheName(i).c_str(), kvCache[i].rawPointer());
        ok &= mAction.context->setTensorAddress(
            binding_names::formatActionPresentKVCacheName(i).c_str(), kvCache[i].rawPointer());
    }

    if (!ok)
    {
        LOG_ERROR("Failed to bind pi0.5 action engine inputs");
        return result;
    }

    // Flow matching integrates t: 1 -> 0 with a fixed step. The graph emits velocity
    // only and the Euler update runs on the device, so the loop needs no per-step
    // synchronization and stays capturable as one CUDA graph.
    float const dt = -1.0F / static_cast<float>(mNumDenoiseSteps);

    auto enqueueLoop = [&] {
        for (int32_t step = 0; step < mNumDenoiseSteps; ++step)
        {
            // Rebinding before each enqueue is what bakes a distinct slice
            // address into each captured node; the base allocation never moves.
            if (mConfig.hoistedAdarmsCond
                && !mAction.context->setTensorAddress(binding_names::kAdarmsModulation,
                    mAdarmsMod.dataPointer<half>() + static_cast<int64_t>(step) * modStepElems))
            {
                return false;
            }
            if (!mAction.context->enqueueV3(mStream))
            {
                return false;
            }
            float const nextT = 1.0F + dt * static_cast<float>(step + 1);
            launchEulerStep(mNoiseDevice.dataPointer<float>(), mPredDevice.dataPointer<float>(), dt,
                static_cast<int64_t>(elems), mConfig.hoistedAdarmsCond ? nullptr : mTimestepDevice.dataPointer<float>(),
                nextT, activeBatch, mStream);
        }
        return true;
    };

    if (mUseCudaGraph)
    {
        if (!mGraphReady)
        {
            // TensorRT's first enqueue initializes lazily and synchronizes, which is
            // illegal once capture has begun; EngineExecutor::captureGraph warms up the
            // same way.
            if (!enqueueLoop())
            {
                LOG_ERROR("pi0.5 action warmup enqueue failed");
                return result;
            }
            CUDA_CHECK(cudaStreamSynchronize(mStream));

            // Relaxed (not ThreadLocal): ThreadLocal forbids synchronization
            // anywhere in this thread for the duration of the capture.
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
            bool const captureEnqueueSucceeded = enqueueLoop();
            cudaGraph_t capturedGraph{nullptr};
            cudaError_t const captureEndStatus = cudaStreamEndCapture(mStream, &capturedGraph);
            if (!captureEnqueueSucceeded || captureEndStatus != cudaSuccess || capturedGraph == nullptr)
            {
                // Each arm names its own cause: cudaGetErrorString would say "no error"
                // for the other two.
                if (!captureEnqueueSucceeded)
                {
                    LOG_WARNING("pi0.5 denoise-loop capture could not enqueue; using plain enqueues");
                }
                else if (captureEndStatus != cudaSuccess)
                {
                    LOG_WARNING("pi0.5 denoise-loop capture failed (%s); using plain enqueues",
                        cudaGetErrorString(captureEndStatus));
                }
                else
                {
                    LOG_WARNING("pi0.5 denoise-loop capture returned no graph; using plain enqueues");
                }
                // endCapture can hand back a graph and still report failure upstream of it.
                if (capturedGraph != nullptr)
                {
                    cudaGraphDestroy(capturedGraph);
                }
                mUseCudaGraph = false;
            }
            else
            {
                // Instantiate into locals first: handing the graph to the members before
                // this succeeds would leak the previous one when a retry re-captures.
                cudaGraphExec_t exec{nullptr};
                cudaError_t const status = cudaGraphInstantiate(&exec, capturedGraph, nullptr, nullptr, 0);
                if (status != cudaSuccess)
                {
                    LOG_WARNING("pi0.5 denoise-loop graph instantiate failed (%s); using plain enqueues",
                        cudaGetErrorString(status));
                    cudaGraphDestroy(capturedGraph);
                    mUseCudaGraph = false;
                }
                else
                {
                    mGraph = capturedGraph;
                    mGraphExec = exec;
                    mGraphKVCache.resize(static_cast<size_t>(mConfig.numHiddenLayers));
                    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
                    {
                        mGraphKVCache[i] = kvCache[i].rawPointer();
                    }
                    mGraphReady = true;
                    LOG_INFO("Captured the pi0.5 denoise loop (%d steps) into one CUDA graph", mNumDenoiseSteps);
                }
            }
            // The warmup ran the loop for real; whichever path follows starts from x_0 and t = 1.
            resetLoopState();
        }
        if (mGraphReady)
        {
            CUDA_CHECK(cudaGraphLaunch(mGraphExec, mStream));
        }
    }
    if (!mUseCudaGraph && !enqueueLoop())
    {
        LOG_ERROR("pi0.5 action engine execution failed");
        return result;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mNoiseHost.rawPointer(), mNoiseDevice.rawPointer(), elems * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaEventRecord(mEnd, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    result.assign(mNoiseHost.dataPointer<float>(), mNoiseHost.dataPointer<float>() + elems);
    return result;
}

float Pi05ActionRunner::getElapsedMs() const noexcept
{
    float elapsed = 0.0F;
    if (cudaEventElapsedTime(&elapsed, mBegin, mEnd) != cudaSuccess)
    {
        return 0.0F;
    }
    return elapsed;
}

} // namespace pi05
} // namespace trt_edgellm
