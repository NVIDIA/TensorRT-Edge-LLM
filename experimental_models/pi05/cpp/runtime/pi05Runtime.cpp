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

#include "runtime/pi05Runtime.h"

#include "common/logger.h"
#include "common/pi05Bindings.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/pi05Kernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/weightsTransform/fp16/linear/fp16LayoutConvert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using nvinfer1::DataType;
using nvinfer1::Dims;

namespace trt_edgellm
{
namespace pi05
{

namespace
{

//! Read a required integer field; a missing key is a contract violation.
int32_t requireInt(nlohmann::json const& cfg, char const* key, char const* component)
{
    if (!cfg.contains(key))
    {
        throw std::runtime_error(std::string("pi0.5 ") + component + " config.json is missing required key: " + key);
    }
    return cfg.at(key).get<int32_t>();
}

//! Read a component config. The runtime deliberately does not link
//! edgellmBuilder, so this does not reuse builder::loadJsonConfig.
nlohmann::json loadComponentConfig(std::filesystem::path const& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open pi0.5 component config: " + path.string());
    }
    nlohmann::json config;
    file >> config;
    return config;
}

float requireFloat(nlohmann::json const& cfg, char const* key, char const* component)
{
    if (!cfg.contains(key))
    {
        throw std::runtime_error(std::string("pi0.5 ") + component + " config.json is missing required key: " + key);
    }
    return cfg.at(key).get<float>();
}

//! One export stamps every component it writes. The structural checks in parseConfigs
//! cannot separate two exports of the same architecture, so a hand-assembled directory
//! whose towers disagree on weights passes all of them.
void requireExportId(nlohmann::json const& cfg, std::string const& expected, char const* component)
{
    std::string const stamped = cfg.value(kExportIdKey, std::string{});
    if (stamped != expected)
    {
        throw std::runtime_error(std::string("pi0.5 ") + component + " is from export '" + stamped
            + "' but prefix is from '" + expected + "'; rebuild the engine directory from one export");
    }
}

} // namespace

Pi05Runtime::Pi05Runtime(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    parseConfigs(engineDir);
    mVisualRunner = std::make_unique<Pi05VisualRunner>(engineDir, mConfig, mStream);
    mPrefix.load(engineDir, "prefix");
    mActionRunner = std::make_unique<Pi05ActionRunner>(engineDir, mConfig, mStream);
    allocateSharedContextMemory();
    allocateTensors();
    loadEmbedTable(engineDir);

    // Positions are contiguous, so one table covers the prefix: it takes rows [0, S)
    // and the action expert reads rows [S, S + H) of its own.
    kernel::initializeNormalRopeCosSin(mRopeCache.dataPointer<float>(), mConfig.ropeTheta, 1.0F, 1.0F, mConfig.headDim,
        mConfig.maxPrefixLen + mConfig.actionHorizon, mStream);

    for (cudaEvent_t& event : mStageEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

Pi05Runtime::~Pi05Runtime() noexcept
{
    for (cudaEvent_t& event : mStageEvents)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

void Pi05Runtime::markStage(int32_t index) noexcept
{
    cudaEventRecord(mStageEvents[index], mStream);
}

void Pi05Runtime::collectStageTimes() noexcept
{
    mStageTimes.visualMs = mVisualRunner->getElapsedMs();
    mStageTimes.actionMs = mActionRunner->getElapsedMs();
    float* const out[] = {&mStageTimes.assembleMs, &mStageTimes.prefixMs};
    for (int32_t i = 0; i + 1 < kNumStageEvents; ++i)
    {
        if (cudaEventElapsedTime(out[i], mStageEvents[i], mStageEvents[i + 1]) != cudaSuccess)
        {
            *out[i] = 0.0F;
        }
    }
}

void Pi05Runtime::logBatchSpread(std::vector<float> const& actions, int32_t batch) const
{
    size_t const chunk = static_cast<size_t>(mConfig.actionHorizon) * mConfig.actionDim;
    if (batch < 2 || actions.size() < chunk * static_cast<size_t>(batch))
    {
        return;
    }
    for (int32_t b = 1; b < batch; ++b)
    {
        double maxDiff = 0.0;
        for (size_t i = 0; i < chunk; ++i)
        {
            maxDiff = std::max(maxDiff, std::abs(static_cast<double>(actions[b * chunk + i] - actions[i])));
        }
        LOG_INFO("pi0.5 batch entry %d vs entry 0: max abs diff %.3e", b, maxDiff);
    }
}

void Pi05Runtime::loadEmbedTable(std::string const& engineDir)
{
    std::filesystem::path const path = std::filesystem::path(engineDir) / "embed_tokens.safetensors";
    std::vector<rt::Tensor> tensors;
    if (!rt::safetensors::loadSafetensors(path, tensors, mStream) || tensors.empty())
    {
        throw std::runtime_error("pi0.5 engine directory is missing the token-embedding table: " + path.string());
    }
    mEmbedTable = std::move(tensors.front());
    LOG_INFO("Loaded pi0.5 token embeddings: %s", path.string().c_str());
}

rt::Tensor const& Pi05Runtime::assemblePrefix(
    rt::Tensor const& imageFeatures, std::vector<int32_t> const& tokenIds, int32_t batch)
{
    rt::Coords const featShape = imageFeatures.getShape();
    auto const imageTokens = static_cast<int32_t>(featShape[0] * featShape[1]);
    auto const numTokens = static_cast<int32_t>(tokenIds.size());
    auto const prefixLen = imageTokens + numTokens;
    int64_t const vocabSize = mEmbedTable.getShape()[0];
    size_t const rowBytes = static_cast<size_t>(mConfig.hiddenSize) * sizeof(half);

    for (int32_t i = 0; i < numTokens; ++i)
    {
        // kernel::embeddingLookup zero-fills an id outside the table instead of failing,
        // which would reach the tower as a silently blank prompt token.
        if (tokenIds[i] < 0 || tokenIds[i] >= vocabSize)
        {
            throw std::runtime_error("pi0.5 token id " + std::to_string(tokenIds[i]) + " at position "
                + std::to_string(i) + " is outside the embedding table's " + std::to_string(vocabSize) + " rows");
        }
    }
    // The engine bound is on the assembled prefix, not on the language tokens alone, and
    // against the allocation rather than the current shape: mInputsEmbeds is reshaped per
    // request. Checked here so the cause is named instead of surfacing as a failed prefill.
    if (prefixLen > mConfig.maxPrefixLen)
    {
        throw std::runtime_error("pi0.5 prefix is " + std::to_string(imageTokens) + " image + "
            + std::to_string(numTokens) + " language = " + std::to_string(prefixLen)
            + " tokens, more than the capacity " + std::to_string(mConfig.maxPrefixLen)
            + " the engines were built for");
    }
    if (!mInputsEmbeds.reshape({batch, prefixLen, mConfig.hiddenSize}))
    {
        throw std::runtime_error("pi0.5 prefix does not fit the assembled buffer");
    }

    CUDA_CHECK(cudaMemcpyAsync(mInputsEmbeds.rawPointer(), imageFeatures.rawPointer(),
        static_cast<size_t>(imageTokens) * rowBytes, cudaMemcpyDeviceToDevice, mStream));

    if (numTokens > 0)
    {
        std::copy(tokenIds.begin(), tokenIds.end(), mTokenIdsHost.dataPointer<int32_t>());
        if (!mTokenIds.reshape({1, numTokens}))
        {
            throw std::runtime_error("pi0.5 token-id staging does not fit the prompt");
        }
        CUDA_CHECK(cudaMemcpyAsync(mTokenIds.rawPointer(), mTokenIdsHost.rawPointer(),
            static_cast<size_t>(numTokens) * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));

        // A view of the language rows of the prefix, so the gather writes its final home.
        rt::Tensor langEmbeds(static_cast<std::byte*>(mInputsEmbeds.rawPointer()) + imageTokens * rowBytes,
            {1, numTokens, mConfig.hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF, "pi05::langEmbeds");
        kernel::embeddingLookup(mTokenIds, mEmbedTable, std::nullopt, langEmbeds, mStream);
        // Gemma scales token embeddings by sqrt(hidden_size) exactly once, in openpi's
        // embed_prefix; its transformers_replace comments out the tower's own normalizer.
        CUDA_CHECK(
            kernel::launchScaleFp16(langEmbeds.rawPointer(), static_cast<int64_t>(numTokens) * mConfig.hiddenSize,
                std::sqrt(static_cast<float>(mConfig.hiddenSize)), mStream));
    }

    size_t const prefixBytes = static_cast<size_t>(prefixLen) * rowBytes;
    for (int32_t b = 1; b < batch; ++b)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mInputsEmbeds.rawPointer()) + b * prefixBytes,
            mInputsEmbeds.rawPointer(), prefixBytes, cudaMemcpyDeviceToDevice, mStream));
    }
    return mInputsEmbeds;
}

void Pi05Runtime::parseConfigs(std::string const& engineDir)
{
    std::filesystem::path const root(engineDir);
    nlohmann::json const prefixCfg = loadComponentConfig(root / "prefix" / "config.json");
    nlohmann::json const actionCfg = loadComponentConfig(root / "action" / "config.json");
    nlohmann::json const visualCfg = loadComponentConfig(root / "visual" / "config.json");

    std::string const exportId = prefixCfg.value(kExportIdKey, std::string{});
    if (exportId.empty())
    {
        throw std::runtime_error("pi0.5 prefix config.json carries no " + std::string(kExportIdKey)
            + "; re-export the checkpoint so the components can be verified to come from one export");
    }
    requireExportId(visualCfg, exportId, "visual");
    requireExportId(actionCfg, exportId, "action");
    mConfig.exportId = exportId;

    mConfig.numHiddenLayers = requireInt(prefixCfg, "num_hidden_layers", "prefix");
    mConfig.numKVHeads = requireInt(prefixCfg, "num_key_value_heads", "prefix");
    mConfig.headDim = requireInt(prefixCfg, "head_dim", "prefix");
    mConfig.hiddenSize = requireInt(prefixCfg, "hidden_size", "prefix");
    mConfig.ropeTheta = requireFloat(prefixCfg, "rope_theta", "prefix");
    mConfig.maxPrefixLen = requireInt(prefixCfg.at("builder_config"), "max_prefix_len", "prefix");

    mConfig.actionDim = requireInt(actionCfg, "action_dim", "action");
    mConfig.actionHorizon = requireInt(actionCfg, "action_horizon", "action");
    mConfig.numDenoiseSteps = requireInt(actionCfg, "num_denoise_steps", "action");

    // Absent in a pre-hoist export, which computes the modulation in-graph.
    mConfig.hoistedAdarmsCond = actionCfg.value("hoisted_adarms_cond", false);
    mConfig.kvCacheCapacity = requireInt(actionCfg.at("builder_config"), "kv_cache_capacity", "action");
    // The expert binds one cache layout; an export without the key was built for
    // another, and would read pages this runtime never allocates.
    if (!actionCfg.at("builder_config").value("paged_kv_cache", false))
    {
        throw std::runtime_error("pi0.5 action export predates the paged K/V cache; re-export and rebuild the bundle");
    }
    if (mConfig.kvCacheCapacity % rt::kTOKENS_PER_PAGE != 0)
    {
        throw std::runtime_error("pi0.5 paged K/V capacity " + std::to_string(mConfig.kvCacheCapacity)
            + " is not a whole number of " + std::to_string(rt::kTOKENS_PER_PAGE) + "-token pages");
    }
    if (mConfig.numKVHeads != 1)
    {
        // The prefix graph emits [B, S, H, D], which only matches the cache's
        // head-major halves when there is a single K/V head.
        throw std::runtime_error("pi0.5 XQA exports require one K/V head; got " + std::to_string(mConfig.numKVHeads));
    }

    if (mConfig.hoistedAdarmsCond)
    {
        nlohmann::json const condCfg = loadComponentConfig(root / "cond" / "config.json");
        requireExportId(condCfg, exportId, "cond");
        mConfig.numAdarmsSites = requireInt(condCfg, "num_adarms_sites", "cond");
        mConfig.modulationDim = requireInt(condCfg, "modulation_dim", "cond");
        mConfig.maxDenoiseSteps = requireInt(condCfg.at("builder_config"), "max_denoise_steps", "cond");
        if (mConfig.numAdarmsSites != requireInt(actionCfg, "num_adarms_sites", "action"))
        {
            throw std::runtime_error(
                "pi0.5 action and cond components disagree on the AdaRMS site count; the expert "
                "slices the modulation by site index and cannot run with a mismatch");
        }
    }

    mConfig.numImageTokens = requireInt(visualCfg, "num_image_tokens", "visual");
    mConfig.imageSize = requireInt(visualCfg, "image_size", "visual");

    // projection_dim, not hidden_size: the latter is SigLIP's own encoder width (1152). The
    // prefix consumes the visual rows directly, so a mismatched pair writes past the allocation
    // instead of failing a shape check.
    int32_t const visualOut = requireInt(visualCfg, "projection_dim", "visual");
    if (visualOut != mConfig.hiddenSize)
    {
        throw std::runtime_error("pi0.5 visual projects to " + std::to_string(visualOut) + " but the prefix expects "
            + std::to_string(mConfig.hiddenSize) + "; the components are from different exports");
    }
    for (auto const& [key, label] : {std::pair{"head_dim", "head dim"}, std::pair{"num_key_value_heads", "K/V heads"}})
    {
        int32_t const actionValue = requireInt(actionCfg, key, "action");
        int32_t const prefixValue = requireInt(prefixCfg, key, "prefix");
        if (actionValue != prefixValue)
        {
            throw std::runtime_error(std::string("pi0.5 prefix and action towers disagree on ") + label + "; the "
                + "expert attends over the prefix K/V and cannot run with a mismatch");
        }
    }
    if (mConfig.kvCacheCapacity < mConfig.maxPrefixLen + mConfig.actionHorizon)
    {
        throw std::runtime_error("pi0.5 K/V cache holds " + std::to_string(mConfig.kvCacheCapacity)
            + " slots but the prefix and the action chunk need "
            + std::to_string(mConfig.maxPrefixLen + mConfig.actionHorizon));
    }

    int32_t const actionLayers = requireInt(actionCfg, "num_hidden_layers", "action");
    if (actionLayers != mConfig.numHiddenLayers)
    {
        throw std::runtime_error(
            "pi0.5 prefix and action towers disagree on layer count; the expert consumes the "
            "prefix K/V one-to-one and cannot run with a mismatch");
    }
}

void Pi05Runtime::allocateSharedContextMemory()
{
    // One pool for every component: they enqueue strictly in sequence on the instance's
    // stream, and TensorRT joins each context's auxiliary streams back into that stream
    // before its enqueue completes, so no two contexts are ever live at once.
    int64_t poolBytes = std::max({mVisualRunner->getRequiredContextMemorySize(), mPrefix.getRequiredContextMemorySize(),
        mActionRunner->getRequiredContextMemorySize()});
    mSharedContextMemory = rt::Tensor(
        std::vector<int64_t>{poolBytes}, rt::DeviceType::kGPU, DataType::kINT8, "pi05::sharedContextMemory");
    if (!mVisualRunner->setContextMemory(mSharedContextMemory) || !mPrefix.setContextMemory(mSharedContextMemory)
        || !mActionRunner->setContextMemory(mSharedContextMemory))
    {
        throw std::runtime_error("pi0.5 shared context-memory pool is smaller than a component requires");
    }
    LOG_INFO(
        "pi0.5 components share one %.1f MiB context-memory pool", static_cast<double>(poolBytes) / (1024.0 * 1024.0));
}

void Pi05Runtime::allocateTensors()
{
    // The profiles, not the contract's max_batch_size, decide what the loaded engines
    // accept: a rebuild at a different --maxBatchSize leaves the exported config alone.
    Dims const prefixMax
        = mPrefix.engine->getProfileShape(binding_names::kInputsEmbeds, 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxBatch = static_cast<int32_t>(std::min<int64_t>(prefixMax.d[0], mActionRunner->getMaxBatch()));
    if (mMaxBatch < 1)
    {
        throw std::runtime_error("pi0.5 engines admit no request batch; rebuild them");
    }
    LOG_INFO("pi0.5 engines admit batch up to %d (prefix %ld, action %d)", mMaxBatch, static_cast<long>(prefixMax.d[0]),
        mActionRunner->getMaxBatch());

    int32_t const maxBatch = mMaxBatch;
    int32_t const numLayers = mConfig.numHiddenLayers;
    int32_t const headDim = mConfig.headDim;

    // Sized for the widest prefix the profile admits; assemblePrefix reshapes into it.
    mInputsEmbeds = rt::Tensor({maxBatch, mConfig.maxPrefixLen, mConfig.hiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "pi05::inputsEmbeds");

    mKVCache.resize(static_cast<size_t>(numLayers));
    // Plane-major: every request's K, then every request's V.
    rt::Coords const cacheShape{
        2, static_cast<int64_t>(maxBatch) * pagesPerSeq(mConfig), rt::kTOKENS_PER_PAGE, mConfig.numKVHeads, headDim};
    for (int32_t i = 0; i < numLayers; ++i)
    {
        mKVCache[i] = rt::Tensor(cacheShape, rt::DeviceType::kGPU, DataType::kHALF, "pi05::kvCache");
    }
    if (maxBatch > 1)
    {
        mPrefixKVStaging.resize(static_cast<size_t>(2 * numLayers));
        rt::Coords const stagingShape{maxBatch, mConfig.maxPrefixLen, mConfig.numKVHeads, headDim};
        for (auto& staging : mPrefixKVStaging)
        {
            staging = rt::Tensor(stagingShape, rt::DeviceType::kGPU, DataType::kHALF, "pi05::prefixKVStaging");
        }
    }
    // Language tokens never outnumber the prefix the profile admits, so one allocation
    // at that bound serves every request without a per-request host tensor.
    mTokenIds = rt::Tensor({1, mConfig.maxPrefixLen}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::tokenIds");
    mTokenIdsHost = rt::Tensor({1, mConfig.maxPrefixLen}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::tokenIdsHost");

    mRopeCache = rt::Tensor({mConfig.maxPrefixLen + mConfig.actionHorizon, headDim}, rt::DeviceType::kGPU,
        DataType::kFLOAT, "pi05::ropeCache");
    mPrefixRopeCosSin = rt::Tensor(
        {maxBatch, mConfig.maxPrefixLen, headDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "pi05::prefixRope");
    mPrefixPosIds
        = rt::Tensor({maxBatch, mConfig.maxPrefixLen}, rt::DeviceType::kGPU, DataType::kINT32, "pi05::prefixPos");
    mPrefixPosIdsHost
        = rt::Tensor({maxBatch, mConfig.maxPrefixLen}, rt::DeviceType::kCPU, DataType::kINT32, "pi05::prefixPosHost");
}

//! Byte offset from a layer's cache base to slot 0's V. The pool is plane-major, so the V
//! plane starts past every slot's K, not past slot 0's. Getting this wrong writes the
//! prefix V over another slot's K and reads V from pages never written -- in-range
//! numbers, wrong robot commands.
size_t Pi05Runtime::vPlaneOffsetBytes() const noexcept
{
    size_t const slotBytes
        = static_cast<size_t>(mConfig.numKVHeads) * mConfig.kvCacheCapacity * mConfig.headDim * sizeof(half);
    return slotBytes * static_cast<size_t>(mMaxBatch);
}

void Pi05Runtime::scatterPrefixKV()
{
    size_t const entryBytes = static_cast<size_t>(mPrefixLen) * mConfig.numKVHeads * mConfig.headDim * sizeof(half);
    size_t const slotStride
        = static_cast<size_t>(mConfig.numKVHeads) * mConfig.kvCacheCapacity * mConfig.headDim * sizeof(half);
    size_t const vOffset = vPlaneOffsetBytes();
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        auto* dst = static_cast<std::byte*>(mKVCache[i].rawPointer());
        CUDA_CHECK(cudaMemcpy2DAsync(dst, slotStride, mPrefixKVStaging[2 * i].rawPointer(), entryBytes, entryBytes,
            static_cast<size_t>(mActiveBatch), cudaMemcpyDeviceToDevice, mStream));
        CUDA_CHECK(cudaMemcpy2DAsync(dst + vOffset, slotStride, mPrefixKVStaging[2 * i + 1].rawPointer(), entryBytes,
            entryBytes, static_cast<size_t>(mActiveBatch), cudaMemcpyDeviceToDevice, mStream));
    }
}

bool Pi05Runtime::prefill(rt::Tensor const& inputsEmbeds)
{
    rt::Coords const shape = inputsEmbeds.getShape();
    int32_t const batch = static_cast<int32_t>(shape[0]);
    int32_t const prefixLen = static_cast<int32_t>(shape[1]);
    if (batch < 1 || batch > mMaxBatch)
    {
        LOG_ERROR(
            "pi0.5 request batch %d is outside the range the engines were built for (1 to %d); "
            "rebuild them with pi05_policy_build --maxBatchSize %d",
            batch, mMaxBatch, batch);
        return false;
    }
    if (prefixLen > mConfig.maxPrefixLen)
    {
        LOG_ERROR("pi0.5 prefix length %d exceeds the engine capacity %d", prefixLen, mConfig.maxPrefixLen);
        return false;
    }
    mActiveBatch = batch;
    mPrefixLen = prefixLen;

    stageRopeInputs(mRopeCache, mConfig.headDim, mActiveBatch, mPrefixLen, 0, mPrefixRopeCosSin, mPrefixPosIds,
        mPrefixPosIdsHost, mStream);

    bool ok = mPrefix.context->setInputShape(
        binding_names::kInputsEmbeds, Dims{3, {mActiveBatch, mPrefixLen, mConfig.hiddenSize}});
    ok &= mPrefix.context->setInputShape(
        binding_names::kRopeCosSin, Dims{3, {mActiveBatch, mPrefixLen, mConfig.headDim}});
    ok &= mPrefix.context->setInputShape(binding_names::kAttentionPosId, Dims{2, {mActiveBatch, mPrefixLen}});
    ok &= mPrefix.context->setTensorAddress(binding_names::kInputsEmbeds, const_cast<void*>(inputsEmbeds.rawPointer()));
    ok &= mPrefix.context->setTensorAddress(binding_names::kRopeCosSin, mPrefixRopeCosSin.rawPointer());
    ok &= mPrefix.context->setTensorAddress(binding_names::kAttentionPosId, mPrefixPosIds.rawPointer());

    // The prefix graph packs requests at the prefix length while the cache halves stride by
    // capacity, so only batch 1 lines up: bind its outputs into the halves directly, and
    // scatter through staging above that.
    bool const direct = mActiveBatch == 1;
    auto const vOffsetElems = static_cast<int64_t>(vPlaneOffsetBytes() / sizeof(half));
    for (int32_t i = 0; i < mConfig.numHiddenLayers; ++i)
    {
        void* kPtr = direct ? mKVCache[i].rawPointer() : mPrefixKVStaging[2 * i].rawPointer();
        void* vPtr = direct ? mKVCache[i].dataPointer<half>() + vOffsetElems : mPrefixKVStaging[2 * i + 1].rawPointer();

        ok &= mPrefix.context->setTensorAddress(binding_names::formatPrefixKName(i).c_str(), kPtr);
        ok &= mPrefix.context->setTensorAddress(binding_names::formatPrefixVName(i).c_str(), vPtr);
    }

    if (!ok || !mPrefix.context->enqueueV3(mStream))
    {
        LOG_ERROR("pi0.5 prefix engine execution failed");
        return false;
    }
    if (!direct)
    {
        scatterPrefixKV();
    }
    return true;
}

//! Drains the stream on the failure and exception paths only. The success path already
//! ends in a synchronizing D2H, so a retry after an early return is the one case where
//! the caller could otherwise rewrite a pinned staging buffer still being read.
class StreamDrainOnFailure
{
public:
    StreamDrainOnFailure(cudaStream_t stream) noexcept
        : mStream(stream)
    {
    }

    ~StreamDrainOnFailure()
    {
        if (!mSucceeded)
        {
            cudaStreamSynchronize(mStream);
        }
    }

    void succeeded() noexcept
    {
        mSucceeded = true;
    }

private:
    cudaStream_t mStream;
    bool mSucceeded{false};
};

std::vector<float> Pi05Runtime::generate(
    rt::Tensor const& pixelValues, std::vector<int32_t> const& tokenIds, int32_t batch)
{
    StreamDrainOnFailure drain(mStream);
    rt::Tensor const& imageFeatures = mVisualRunner->encode(pixelValues);
    markStage(0);
    rt::Tensor const& inputsEmbeds = assemblePrefix(imageFeatures, tokenIds, batch);
    markStage(1);
    if (!prefill(inputsEmbeds))
    {
        return {};
    }
    markStage(2);
    std::vector<float> actions = mActionRunner->generate(mKVCache, mActiveBatch, mPrefixLen);
    if (actions.empty())
    {
        return actions;
    }
    collectStageTimes();
    logBatchSpread(actions, batch);
    drain.succeeded();
    return actions;
}

} // namespace pi05
} // namespace trt_edgellm
