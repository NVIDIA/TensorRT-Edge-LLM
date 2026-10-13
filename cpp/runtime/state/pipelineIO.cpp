/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/state/pipelineIO.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/stringUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace trt_edgellm
{
namespace rt
{

AsyncHostStagingFence::AsyncHostStagingFence(AsyncHostStagingFence&& other) noexcept
    : mEvent(std::exchange(other.mEvent, nullptr))
    , mPending(std::exchange(other.mPending, false))
{
}

AsyncHostStagingFence& AsyncHostStagingFence::operator=(AsyncHostStagingFence&& other) noexcept
{
    if (this != &other)
    {
        release();
        mEvent = std::exchange(other.mEvent, nullptr);
        mPending = std::exchange(other.mPending, false);
    }
    return *this;
}

AsyncHostStagingFence::~AsyncHostStagingFence()
{
    release();
}

void AsyncHostStagingFence::wait()
{
    if (mPending)
    {
        CUDA_CHECK(cudaEventSynchronize(mEvent));
        mPending = false;
    }
}

void AsyncHostStagingFence::record(cudaStream_t stream)
{
    if (mEvent == nullptr)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(mEvent, stream));
    mPending = true;
}

void AsyncHostStagingFence::release() noexcept
{
    if (mPending && mEvent != nullptr)
    {
        static_cast<void>(cudaEventSynchronize(mEvent));
    }
    if (mEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(mEvent));
    }
    mEvent = nullptr;
    mPending = false;
}

void allocateBasicIO(PipelineIO& io, int32_t maxBatch, int32_t vocabSize)
{
    // Standard LLM logits are FLOAT for the sampler. DiffusionGemma keeps the
    // same dtype for canvas logits because final logit softcapping exports an
    // F32 logits binding.
    io.outputLogits
        = Tensor({maxBatch, vocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "PipelineIO::outputLogits");
    io.selectTokenIndices
        = Tensor({maxBatch, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "PipelineIO::selectTokenIndices");
    io.phaseIsEncoder = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::phaseIsEncoder");
    io.contextMaskSelector
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::contextMaskSelector");
    io.contextLengths = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::contextLengths");
    io.hostContextLengths
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostContextLengths");
    io.hostSelectTokenIndices
        = Tensor({maxBatch, 1}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "PipelineIO::hostSelectTokenIndices");
    io.hostPhaseIsEncoder = Tensor({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostPhaseIsEncoder");
}

void allocateRaggedMetadata(PipelineIO& io, int32_t maxTokens, int32_t maxSequences, int32_t maxLogitsRows,
    int32_t maxKVCacheCapacity, int32_t hiddenSize)
{
    io.inputsEmbeds
        = Tensor({maxTokens, hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "PipelineIO::inputsEmbeds");
    io.positions = Tensor({maxTokens}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::positions");
    io.queryStartOffsets
        = Tensor({maxSequences + 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::queryStartOffsets");
    io.queryLengths = Tensor({maxSequences}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::queryLengths");
    io.pastLengths = Tensor({maxSequences}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::pastLengths");
    io.attentionSequenceLengths
        = Tensor({maxSequences}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::attentionSequenceLengths");
    io.stateIndices = Tensor({maxSequences}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::stateIndices");
    io.logitsIndices
        = Tensor({maxLogitsRows}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "PipelineIO::logitsIndices");
    io.raggedKVPageTable = Tensor({maxSequences, 2, computeMaxPagesPerSeq(maxKVCacheCapacity)}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::raggedKVPageTable");
    io.raggedSwaKVPageTable = Tensor({maxSequences, 2, computeMaxPagesPerSeq(maxKVCacheCapacity)}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::raggedSwaKVPageTable");
    io.executionPhaseMarker = Tensor({static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode)}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::executionPhaseMarker");
    io.contextSequenceCountCarrier = Tensor(
        {maxSequences}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::contextSequenceCountCarrier");

    io.hostPositions = Tensor({maxTokens}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostPositions");
    io.hostQueryStartOffsets
        = Tensor({maxSequences + 1}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostQueryStartOffsets");
    io.hostQueryLengths
        = Tensor({maxSequences}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostQueryLengths");
    io.hostPastLengths
        = Tensor({maxSequences}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostPastLengths");
    io.hostAttentionSequenceLengths = Tensor(
        {maxSequences}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostAttentionSequenceLengths");
    io.hostStateIndices
        = Tensor({maxSequences}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostStateIndices");
    io.hostLogitsIndices
        = Tensor({maxLogitsRows}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "PipelineIO::hostLogitsIndices");
}

void PipelineIO::uploadRaggedMetadata(RaggedExecutionBatch const& batch, cudaStream_t stream)
{
    mRaggedMetadataUploadFence.wait();
    int32_t const tokens = batch.shape.physicalTokens;
    int32_t const sequences = batch.shape.numSequences;
    int32_t const logits = batch.shape.numLogits;
    check::check(positions.reshape({tokens}) && queryStartOffsets.reshape({sequences + 1})
            && queryLengths.reshape({sequences}) && pastLengths.reshape({sequences})
            && attentionSequenceLengths.reshape({sequences}) && stateIndices.reshape({sequences})
            && logitsIndices.reshape({logits})
            && contextSequenceCountCarrier.reshape({batch.shape.numContextSequences}),
        "Ragged device metadata reshape failed");
    check::check(hostPositions.reshape({tokens}) && hostQueryStartOffsets.reshape({sequences + 1})
            && hostQueryLengths.reshape({sequences}) && hostPastLengths.reshape({sequences})
            && hostAttentionSequenceLengths.reshape({sequences}) && hostStateIndices.reshape({sequences})
            && hostLogitsIndices.reshape({logits}),
        "Ragged host metadata reshape failed");

    auto stage = [](Tensor& destination, auto const& source) {
        using Value = typename std::decay_t<decltype(source)>::value_type;
        std::copy(source.begin(), source.end(), destination.dataPointer<Value>());
    };
    stage(hostPositions, batch.positions);
    stage(hostQueryStartOffsets, batch.queryStartOffsets);
    stage(hostQueryLengths, batch.queryLengths);
    stage(hostPastLengths, batch.pastLengths);
    stage(hostAttentionSequenceLengths, batch.attentionSequenceLengths);
    stage(hostStateIndices, batch.stateIndices);
    stage(hostLogitsIndices, batch.logitsIndices);

    auto upload = [stream](Tensor& destination, Tensor const& source) {
        size_t const bytes = static_cast<size_t>(source.getShape().volume()) * utils::getTypeSize(source.getDataType());
        CUDA_CHECK(
            cudaMemcpyAsync(destination.rawPointer(), source.rawPointer(), bytes, cudaMemcpyHostToDevice, stream));
    };
    upload(positions, hostPositions);
    upload(queryStartOffsets, hostQueryStartOffsets);
    upload(queryLengths, hostQueryLengths);
    upload(pastLengths, hostPastLengths);
    upload(attentionSequenceLengths, hostAttentionSequenceLengths);
    upload(stateIndices, hostStateIndices);
    upload(logitsIndices, hostLogitsIndices);
    mRaggedMetadataUploadFence.record(stream);
}

void PipelineIO::uploadStateIndices(
    std::vector<ResidentRef> const* residentRefs, int32_t numSequences, cudaStream_t stream)
{
    check::check(numSequences > 0, "State-index sequence count must be positive");
    check::check(residentRefs == nullptr || residentRefs->size() >= static_cast<size_t>(numSequences),
        "Missing resident-slot references for state-index upload");
    mRaggedMetadataUploadFence.wait();
    check::check(stateIndices.reshape({numSequences}) && hostStateIndices.reshape({numSequences}),
        "State-index metadata reshape failed");
    int32_t* hostIndices = hostStateIndices.dataPointer<int32_t>();
    for (int32_t sequence = 0; sequence < numSequences; ++sequence)
    {
        hostIndices[sequence]
            = residentRefs == nullptr ? sequence : (*residentRefs)[static_cast<size_t>(sequence)].slot;
    }
    CUDA_CHECK(cudaMemcpyAsync(stateIndices.rawPointer(), hostStateIndices.rawPointer(),
        static_cast<size_t>(numSequences) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mRaggedMetadataUploadFence.record(stream);
}

void PipelineIO::waitForStepHostStaging()
{
    mStepHostUploadFence.wait();
}

void PipelineIO::recordStepHostUploads(cudaStream_t stream)
{
    mStepHostUploadFence.record(stream);
}

void prepareRaggedKVPageTable(PipelineIO& io, KVPageTable const& pageTable, int32_t numSequences, cudaStream_t stream)
{
    pageTable.gatherRows(io.raggedKVPageTable, io.stateIndices, numSequences, stream);
}

void prepareRaggedSwaKVPageTable(
    PipelineIO& io, KVPageTable const& pageTable, int32_t numSequences, cudaStream_t stream)
{
    pageTable.gatherRows(io.raggedSwaKVPageTable, io.stateIndices, numSequences, stream);
}

void prepareRaggedExecutionBindings(PipelineIO& io, SharedResources& resources, LLMEngineConfig const& cfg,
    RaggedExecutionBatch const& batch, int32_t kvCacheIndex, cudaStream_t stream)
{
    check::check(kvCacheIndex >= 0 && static_cast<size_t>(kvCacheIndex) < resources.kvPageTables.size(),
        "Ragged execution KV-cache index is out of range");
    check::check(static_cast<size_t>(kvCacheIndex) < resources.cacheManagers.size(),
        "Ragged execution cache-manager index is out of range");
    io.uploadRaggedMetadata(batch, stream);
    resources.cacheManagers[static_cast<size_t>(kvCacheIndex)]->materializeExecutionLengths(io.pastLengths, stream);
    prepareRaggedKVPageTable(
        io, *resources.kvPageTables[static_cast<size_t>(kvCacheIndex)], batch.shape.numSequences, stream);
    if (cfg.usesBoundedSwaKVCache())
    {
        KVPageTable* const swaPageTable = resources.getSwaKVPageTable(kvCacheIndex);
        check::check(swaPageTable != nullptr, "Bounded SWA page table is missing");
        prepareRaggedSwaKVPageTable(io, *swaPageTable, batch.shape.numSequences, stream);
    }
    prepareRaggedRope(io, resources, cfg, batch.shape.physicalTokens, batch.shape.numSequences, stream);
}

void allocateDeepstackEmbeds(
    PipelineIO& io, int32_t numFeatures, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize, nvinfer1::DataType dtype)
{
    io.deepstackEmbeds.clear();
    io.deepstackEmbeds.reserve(numFeatures);
    for (int32_t i = 0; i < numFeatures; ++i)
    {
        io.deepstackEmbeds.emplace_back(
            Coords{maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::deepstackEmbeds");
    }
}

void allocateSpecDecodeHiddenStates(PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t baseHiddenDim,
    int32_t draftHiddenDim, nvinfer1::DataType dtype, bool allocateDraftHiddenStates)
{
    io.baseHiddenStates
        = Tensor({maxBatch, maxSeq, baseHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::baseHiddenStates");
    if (!allocateDraftHiddenStates)
    {
        return;
    }
    io.draftHiddenStatesIn
        = Tensor({maxBatch, maxSeq, draftHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::draftHiddenStatesIn");
    io.draftHiddenStatesOut
        = Tensor({maxBatch, maxSeq, draftHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::draftHiddenStatesOut");
}

void allocateMRope(
    PipelineIO& io, int32_t residentRows, int32_t activeRows, int32_t maxKVCacheCapacity, int32_t rotaryDim)
{
    io.mropeCosSin = Tensor({residentRows, maxKVCacheCapacity, rotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "PipelineIO::mropeCosSin");
    io.mropeActiveCosSin = Tensor({activeRows, maxKVCacheCapacity, rotaryDim}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "PipelineIO::mropeActiveCosSin");
}

void prepareTextOnlyMRope(PipelineIO& io, LLMEngineConfig const& cfg, int32_t activeRows, cudaStream_t stream)
{
    check::check(activeRows > 0 && activeRows <= cfg.maxSupportedBatchSize,
        "Text-only MRoPE preparation received an invalid active-row count");
    check::check(io.mropeActiveCosSin.reshape({activeRows, cfg.maxKVCacheCapacity, cfg.rotaryDim}),
        "Text-only MRoPE scratch reshape failed");
    kernel::initializeTextOnlyMRopeCosSin(io.mropeActiveCosSin.dataPointer<float>(), cfg.ropeConfig.rotaryTheta,
        cfg.rotaryDim, cfg.maxKVCacheCapacity, activeRows, stream);
}

void allocateRaggedRopeBuffers(PipelineIO& io, int32_t genericRows, int32_t dualRows, int32_t rotaryDim,
    int32_t slidingRotaryDim, int32_t fullRotaryDim)
{
    if (rotaryDim > 0)
    {
        check::check(genericRows > 0, "Generic RoPE allocation requires a positive row capacity");
        io.raggedRopeCosSin = Tensor(
            {genericRows, rotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "PipelineIO::raggedRopeCosSin");
    }
    if (slidingRotaryDim > 0 || fullRotaryDim > 0)
    {
        check::check(dualRows > 0 && slidingRotaryDim > 0 && fullRotaryDim > 0,
            "Dual RoPE allocation requires positive row capacity and rotary dimensions");
        io.raggedRopeCosSinSliding = Tensor({dualRows, slidingRotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "PipelineIO::raggedRopeCosSinSliding");
        io.raggedRopeCosSinFull = Tensor({dualRows, fullRotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "PipelineIO::raggedRopeCosSinFull");
    }
}

void allocateRaggedRope(PipelineIO& io, LLMEngineConfig const& cfg)
{
    allocateRaggedRopeBuffers(io, cfg.useDualRope ? 0 : cfg.maxPhysicalTokens,
        cfg.useDualRope ? cfg.maxPhysicalTokens : 0, cfg.useDualRope ? 0 : cfg.rotaryDim,
        cfg.useDualRope ? cfg.slidingRotaryDim : 0, cfg.useDualRope ? cfg.fullRotaryDim : 0);
}

void prepareRaggedRope(PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t physicalTokens,
    int32_t numSequences, cudaStream_t stream)
{
    auto gather = [&](Tensor const& source, Tensor& output, int32_t rotaryDim, int32_t sourceRows) {
        check::check(output.reshape({physicalTokens, rotaryDim}), "Token-aligned RoPE reshape failed");
        kernel::launchGatherTokenAlignedRope(source.dataPointer<float>(), output.dataPointer<float>(),
            io.positions.dataPointer<int32_t>(), io.queryStartOffsets.dataPointer<int32_t>(),
            io.queryLengths.dataPointer<int32_t>(), sourceRows == 1 ? nullptr : io.stateIndices.dataPointer<int32_t>(),
            physicalTokens, numSequences, sourceRows, cfg.maxKVCacheCapacity, rotaryDim, stream);
    };

    if (cfg.useDualRope)
    {
        Tensor const& sliding
            = res.ropePool.getOrCreate(cfg.slidingRopeConfig, cfg.slidingRotaryDim, cfg.maxKVCacheCapacity, stream);
        Tensor const& full
            = res.ropePool.getOrCreate(cfg.fullRopeConfig, cfg.fullRotaryDim, cfg.maxKVCacheCapacity, stream);
        gather(sliding, io.raggedRopeCosSinSliding, cfg.slidingRotaryDim, 1);
        gather(full, io.raggedRopeCosSinFull, cfg.fullRotaryDim, 1);
        return;
    }

    if (cfg.ropeConfig.type == RopeType::kMRope)
    {
        gather(io.mropeCosSin, io.raggedRopeCosSin, cfg.rotaryDim, cfg.recurrentPoolRows);
    }
    else
    {
        Tensor const& source = res.ropePool.getOrCreate(cfg.ropeConfig, cfg.rotaryDim, cfg.maxKVCacheCapacity, stream);
        gather(source, io.raggedRopeCosSin, cfg.rotaryDim, 1);
    }
}

void scatterActiveMRopeToResident(
    PipelineIO& io, RaggedExecutionBatch const& batch, LLMEngineConfig const& cfg, cudaStream_t stream)
{
    int32_t const activeRows = batch.shape.numSequences;
    check::check(activeRows > 0 && batch.stateIndices.size() == static_cast<size_t>(activeRows),
        "MRoPE resident scatter requires one state index per active row");
    Coords const activeShape = io.mropeActiveCosSin.getShape();
    check::check(activeShape.getNumDims() == 3 && activeShape[0] == activeRows
            && activeShape[1] == cfg.maxKVCacheCapacity && activeShape[2] == cfg.rotaryDim,
        format::fmtstr("Active MRoPE preprocessing scratch shape %s does not match [%d, %d, %d]",
            activeShape.formatString().c_str(), activeRows, cfg.maxKVCacheCapacity, cfg.rotaryDim));
    check::check(io.mropeCosSin.reshape({cfg.recurrentPoolRows, cfg.maxKVCacheCapacity, cfg.rotaryDim}),
        "Resident MRoPE cache reshape failed");

    size_t const rowBytes
        = static_cast<size_t>(cfg.maxKVCacheCapacity) * static_cast<size_t>(cfg.rotaryDim) * sizeof(float);
    for (int32_t activeRow = 0; activeRow < activeRows; ++activeRow)
    {
        int32_t const residentRow = batch.stateIndices[static_cast<size_t>(activeRow)];
        check::check(residentRow >= 0 && residentRow < cfg.recurrentPoolRows,
            "MRoPE resident scatter state index is out of range");
    }
    kernel::launchScatterActiveRows(io.mropeActiveCosSin.rawPointer(), io.mropeCosSin.rawPointer(),
        io.stateIndices.dataPointer<int32_t>(), activeRows, cfg.recurrentPoolRows, rowBytes, stream);
}

namespace
{
bool hasDeepstackFeatures(LLMEngineConfig const& cfg) noexcept
{
    return !cfg.isDiffusionBackbone && cfg.numDeepstackFeatures > 0;
}

void bindDiffusionGemmaBackboneTensorMap(TensorMap& map, PipelineIO& io, LLMEngineConfig const& cfg)
{
    map.set(binding_names::kPhaseIsEncoder, io.phaseIsEncoder);
    // DiffusionGemma gathers logits for a canvas of positions, so its engine
    // input is the model-owned select_token_indices binding rather than the
    // autoregressive logits selection binding.
    map.set(binding_names::kSelectTokenIndices, io.selectTokenIndices);
    if (cfg.contextMaskSelectorEnabled)
    {
        map.set(binding_names::kContextMaskSelector, io.contextMaskSelector);
    }
}

void bindUnifiedDecoderMetadata(TensorMap& map, PipelineIO& io)
{
    map.set(binding_names::kPositions, io.positions);
    map.set(binding_names::kQueryStartOffsets, io.queryStartOffsets);
    map.set(binding_names::kQueryLengths, io.queryLengths);
    map.set(binding_names::kPastLengths, io.pastLengths);
    map.set(binding_names::kAttentionSequenceLengths, io.attentionSequenceLengths);
    map.set(binding_names::kStateIndices, io.stateIndices);
    map.set(binding_names::kExecutionPhaseMarker, io.executionPhaseMarker);
    map.set(binding_names::kContextSequenceCountCarrier, io.contextSequenceCountCarrier);
}
} // namespace

void StreamingPrefillBuffers::populateFromPrefill(Tensor const& liveInputEmbeds, Tensor const& liveEngineHiddenStates,
    int32_t batch, int32_t prefillLen, int32_t hiddenSize, int32_t maxBatch, int32_t maxSeq, cudaStream_t stream)
{
    auto const dtype = nvinfer1::DataType::kHALF;
    if (inputEmbeds.isEmpty())
    {
        inputEmbeds = Tensor(
            {maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::streamingPrefill.inputEmbeds");
        engineHiddenStates = Tensor(
            {maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::streamingPrefill.engineHiddenStates");
    }
    check::check(inputEmbeds.reshape({batch, prefillLen, hiddenSize}), "Tensor reshape failed");
    check::check(engineHiddenStates.reshape({batch, prefillLen, hiddenSize}), "Tensor reshape failed");

    size_t const bytes = static_cast<size_t>(batch) * prefillLen * hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemcpyAsync(
        inputEmbeds.rawPointer(), liveInputEmbeds.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        engineHiddenStates.rawPointer(), liveEngineHiddenStates.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
}

void bindRopeTensors(TensorMap& map, PipelineIO& io, [[maybe_unused]] SharedResources& res, LLMEngineConfig const& cfg)
{
    if (cfg.useDualRope)
    {
        map.set(binding_names::kRopeCosSinSliding, io.raggedRopeCosSinSliding);
        map.set(binding_names::kRopeCosSinFull, io.raggedRopeCosSinFull);
        return;
    }
    map.set(binding_names::kRopeCosSin, io.raggedRopeCosSin);
}

static void buildTensorMapImpl(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex)
{
    // Core I/O
    map.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    map.set(binding_names::kLogits, io.outputLogits);
    bindUnifiedDecoderMetadata(map, io);
    map.set(binding_names::kLogitsIndices, io.logitsIndices);
    if (cfg.isDiffusionBackbone)
    {
        bindDiffusionGemmaBackboneTensorMap(map, io, cfg);
    }
    if (cfg.useVisionBidirectionalAttention)
    {
        map.set(binding_names::kVisionBlockIds, io.visionBlockIds);
    }

    bindRopeTensors(map, io, res, cfg);

    // Hybrid cache routing: walk absolute decoder-layer indices, route by
    // `cfg.layerTypes[absIdx]`, and bind per-layer tensors using LOCAL indices.
    auto& cacheMgr = *res.cacheManagers[kvCacheIndex];
    auto& kvMgr = cacheMgr.getKVCacheManager();
    auto& mambaMgr = cacheMgr.getMambaCacheManager();

    int32_t localAttnIdx = 0;
    int32_t localMambaIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
    {
        if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kAttention)
        {
            // Check if this attention layer shares KV cache from a donor layer.
            int32_t const donorIdx
                = (!cfg.kvSharingDonors.empty() && localAttnIdx < static_cast<int32_t>(cfg.kvSharingDonors.size()))
                ? cfg.kvSharingDonors[localAttnIdx]
                : -1;
            if (donorIdx >= 0)
            {
                check::check(donorIdx < kvMgr.numLayers(),
                    "buildTensorMap: KV sharing donor index is outside the cache manager.");
                KVLayerConfig const& consumerConfig = kvMgr.getLayerConfig(localAttnIdx);
                KVLayerConfig const& donorConfig = kvMgr.getLayerConfig(donorIdx);
                check::check(consumerConfig.numKVHeads == donorConfig.numKVHeads
                        && consumerConfig.headDim == donorConfig.headDim,
                    "buildTensorMap: KV sharing consumer and donor pool dimensions must match.");
                check::check(cfg.getKVPoolPagesForLayer(consumerConfig) == cfg.getKVPoolPagesForLayer(donorConfig),
                    "buildTensorMap: KV sharing consumer and donor must use the same active cache policy.");
            }

            // Plugin (combined KV): bind to donor's pool if shared, else own pool.
            auto& combinedKV
                = (donorIdx >= 0) ? kvMgr.getCombinedKVCache(donorIdx) : kvMgr.getCombinedKVCache(localAttnIdx);
            map.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
            map.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV); // alias: in-place
            ++localAttnIdx;
        }
        else if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kMamba)
        {
            auto& rec = mambaMgr.getRecurrentState(localMambaIdx);
            auto& conv = mambaMgr.getConvState(localMambaIdx);
            map.set(binding_names::formatRecurrentStateName(localMambaIdx, /*isPast=*/true), rec);
            map.set(binding_names::formatRecurrentStateName(localMambaIdx, /*isPast=*/false), rec);
            map.set(binding_names::formatConvStateName(localMambaIdx, /*isPast=*/true), conv);
            map.set(binding_names::formatConvStateName(localMambaIdx, /*isPast=*/false), conv);
            // Spec-decode hybrid base: bind the per-layer intermediate state outputs.
            // `hasIntermediateRecurrentStates()` is true iff the MambaCacheManager
            // was built with `maxIntermediateSeqLen > 0` (set by createForSpecDecode
            // for hybrid MTP bases). EAGLE3 base lacks recurrent layers entirely,
            // so this branch wouldn't fire for it regardless.
            if (mambaMgr.hasIntermediateRecurrentStates())
            {
                if (mambaMgr.recurrentUsesReplay())
                {
                    // Mamba: bind the four replay-stash outputs (dA/x/B/dt). The accepted recurrent
                    // state is reconstructed from these after verification.
                    map.set(binding_names::formatReplayDaStateName(localMambaIdx),
                        mambaMgr.getReplayDaState(localMambaIdx));
                    map.set(
                        binding_names::formatReplayUStateName(localMambaIdx), mambaMgr.getReplayUState(localMambaIdx));
                    map.set(
                        binding_names::formatReplayBStateName(localMambaIdx), mambaMgr.getReplayBState(localMambaIdx));
                    map.set(binding_names::formatReplayDtStateName(localMambaIdx),
                        mambaMgr.getReplayDtState(localMambaIdx));
                }
                else
                {
                    // GDN/DDTree: bind the per-token full-state snapshot output.
                    map.set(binding_names::formatIntermediateRecurrentStateName(localMambaIdx),
                        mambaMgr.getIntermediateRecurrentState(localMambaIdx));
                }
            }
            if (mambaMgr.hasIntermediateConvStates())
            {
                map.set(binding_names::formatIntermediateConvStateName(localMambaIdx),
                    mambaMgr.getIntermediateConvState(localMambaIdx));
            }
            ++localMambaIdx;
        }
        else
        {
            check::check(false, format::fmtstr("buildTensorMap: unknown LayerType at absolute layer index %d", absIdx));
        }
    }

    // The full table is always present. Bounded mode uses the independent sparse SWA namespace;
    // full mode aliases the SWA binding to the ordinary table so context reuse follows the existing
    // full-cache lifecycle. The shape-only mode input selects the matching plugin path.
    map.set(binding_names::kKVPageTable, io.raggedKVPageTable);
    if (cfg.supportsBoundedSwaKVCache())
    {
        KVPageTable* const swaPageTable = res.getSwaKVPageTable(kvCacheIndex);
        if (cfg.usesBoundedSwaKVCache())
        {
            check::check(
                swaPageTable != nullptr, "buildTensorMap: bounded SWA mode requires an independent sparse page table.");
            map.set(binding_names::kSwaKVPageTable, io.raggedSwaKVPageTable);
        }
        else
        {
            check::check(swaPageTable == nullptr,
                "buildTensorMap: full SWA mode must not allocate an independent sparse page table.");
            map.set(binding_names::kSwaKVPageTable, io.raggedKVPageTable);
        }
        check::check(!res.swaKVCacheMode.isEmpty(), "buildTensorMap: SWA mode backing storage is missing.");
        map.set(binding_names::kSwaKVCacheMode, res.swaKVCacheMode);
    }

    // Deepstack: initial bind is the shared zero buffer (sized large enough
    // to cover the worst-case non-prefill shape). DeepstackBinding (owned by
    // the runtime) swaps to `io.deepstackEmbeds[i]` just before base prefill
    // and back on non-prefill phases.
    if (hasDeepstackFeatures(cfg))
    {
        for (size_t i = 0; i < io.deepstackEmbeds.size(); ++i)
        {
            map.set(binding_names::formatDeepstackEmbedsName(static_cast<int32_t>(i)), res.zeroBuffer);
        }
    }

    // Hidden states output. SpecDecode base engines write their target features
    // into baseHiddenStates. The vanilla LLM path uses
    // outputHiddenStates instead (shape uses cfg.hiddenSize). Any subset may be
    // bound here; the engine introspection in EngineExecutor::prepare() will set
    // the address only if the engine actually exposes the binding.
    if (cfg.isSpecDecodeBase && !io.baseHiddenStates.isEmpty())
    {
        map.set(binding_names::kOutputHiddenStates, io.baseHiddenStates);
    }
    else if (!io.outputHiddenStates.isEmpty())
    {
        map.set(binding_names::kOutputHiddenStates, io.outputHiddenStates);
    }

    // Accept-layer output (Omni-Next Thinker), orthogonal to the above: on a
    // SpecDecode base `hidden_states` is the draft's post-norm feed, so the
    // Talker's mid-stack tensor arrives under its own name. Engines without the
    // binding ignore this entry, so they stay loadable.
    if (!io.outputHiddenStates.isEmpty())
    {
        map.set(binding_names::kAcceptHiddenStates, io.outputHiddenStates);
    }

    // SpecDecode base-engine token-aligned attention metadata. For vanilla LLMs
    // these tensors are empty and the bindings are not set.
    if (cfg.isSpecDecodeBase && !io.packedAttentionMask.isEmpty())
    {
        map.set(binding_names::kAttentionMask, io.packedAttentionMask);
        map.set(binding_names::kAttentionPosId, io.specDecodePositionIds);
    }
    if (!io.skipSoftmaxScale.isEmpty())
    {
        map.set(binding_names::kSkipSoftmaxScale, io.skipSoftmaxScale);
    }
    if (!io.specTreeParentIds.isEmpty())
    {
        map.set(binding_names::kTreeParentIds, io.specTreeParentIds);
    }
    if (!io.specTreeDepths.isEmpty())
    {
        map.set(binding_names::kTreeDepths, io.specTreeDepths);
    }
    map.set(binding_names::kValidTreeCounts, io.queryLengths);

    // LoRA bindings are NOT set here because adapter tensor names may differ
    // from engine binding names (e.g. fused QKV).  LoRAManager::refreshTensorMap()
    // populates them after buildTensorMap().
}

void buildTensorMap(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex)
{
    buildTensorMapImpl(map, io, res, cfg, kvCacheIndex);
}

void buildTensorMapForDiffusionBackbone(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex)
{
    check::check(cfg.isDiffusionBackbone, "buildTensorMapForDiffusionBackbone requires a DiffusionGemma backbone.");
    buildTensorMapImpl(map, io, res, cfg, kvCacheIndex);
}

void bindDiffusionUnifiedBackboneTensors(TensorMap& map, PipelineIO& io, Tensor& logits, Tensor& canvasIds,
    Tensor& prevSelfConditioningEmbeds, Tensor& nextSelfConditioningEmbeds, Tensor& selfConditioningTemperature)
{
    map.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    map.set(binding_names::kLogits, logits);
    map.set(binding_names::kCanvasIds, canvasIds);
    bindDiffusionUnifiedBackboneSelfConditioningTensors(map, prevSelfConditioningEmbeds, nextSelfConditioningEmbeds);
    map.set(binding_names::kSelfConditioningTemperature, selfConditioningTemperature);
}

void bindDiffusionUnifiedBackboneSelfConditioningTensors(
    TensorMap& map, Tensor& prevSelfConditioningEmbeds, Tensor& nextSelfConditioningEmbeds)
{
    map.set(binding_names::kPrevSelfConditioningEmbeds, prevSelfConditioningEmbeds);
    map.set(binding_names::kNextSelfConditioningEmbeds, nextSelfConditioningEmbeds);
}

void buildTensorMapForSpecDecodeDraft(TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg)
{
    // Reuse the shared buildTensorMap for common bindings (core I/O, RoPE,
    // KV cache, kvcache_start_index). Draft engine uses kvCacheIndex=1.
    buildTensorMap(map, io, res, cfg, /*kvCacheIndex=*/1);

    // Draft-specific hidden-state bindings: the base model's hidden states feed
    // the draft engine as input; the draft engine produces its own hidden states
    // on output (the kOutputHiddenStates entry added by buildTensorMap —
    // gated on cfg.isSpecDecodeBase which is false for the draft config —
    // is overridden here regardless).
    map.set(binding_names::kBaseModelHiddenStates, io.baseHiddenStates);
    map.set(binding_names::kDraftModelHiddenStates, io.draftHiddenStatesIn);
    map.set(binding_names::kOutputHiddenStates, io.draftHiddenStatesOut);

    // Attention mask and position IDs for proposal decoding. The TRT engine expects
    // the INT32 packed mask (not the INT8 unpacked one). Position IDs are written
    // by proposal/verify input preparation kernels before each execute.
    map.set(binding_names::kAttentionMask, io.packedAttentionMask);
    map.set(binding_names::kAttentionPosId, io.specDecodePositionIds);
}

void buildTensorMapForGemma4MTPDraft(
    TensorMap& map, PipelineIO& io, SharedResources& res, DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildTensorMapForGemma4MTPDraft requires bundle.draft");
    check::check(bundle.specConfig.has_value(), "buildTensorMapForGemma4MTPDraft requires bundle.specConfig");
    check::check(bundle.specDecodeMode() == SpecDecodeMode::kGemma4MTP,
        "buildTensorMapForGemma4MTPDraft requires spec_decode_type=gemma4_mtp");
    check::check(!res.cacheManagers.empty(), "buildTensorMapForGemma4MTPDraft requires base cache manager");

    LLMEngineConfig const& draftCfg = *bundle.draft;

    map.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    map.set(binding_names::kLogits, io.outputLogits);
    map.set(binding_names::kBaseModelHiddenStates, io.draftHiddenStatesIn);
    map.set(binding_names::kOutputHiddenStates, io.draftHiddenStatesOut);
    bindUnifiedDecoderMetadata(map, io);

    bindRopeTensors(map, io, res, draftCfg);

    auto& baseCacheManager = *res.cacheManagers[0];
    // kv_page_table: the assistant reads the TARGET model's paged pool, so it binds the
    // target's page table (identity while reuse is off) — same object the base engine binds.
    map.set(binding_names::kKVPageTable, io.raggedKVPageTable);
    for (auto const& entry : draftCfg.gemma4MTPKVSharingMap)
    {
        rt::Tensor& targetKV = baseCacheManager.getCombinedKVCache(entry.targetAbsoluteLayerIdx);
        map.set(binding_names::formatKVCacheName(entry.assistantLayerIdx, /*isPast=*/true), targetKV);
    }
}

PipelineIO PipelineIO::createForLLM(LLMEngineConfig const& cfg, cudaStream_t stream)
{
    PipelineIO io;

    allocateBasicIO(io, cfg.maxSupportedBatchSize, cfg.outputVocabSize);
    int32_t const maxLogitsRows = cfg.isDiffusionBackbone ? cfg.maxPhysicalTokens : cfg.maxNumSequences;
    allocateRaggedMetadata(
        io, cfg.maxPhysicalTokens, cfg.maxNumSequences, maxLogitsRows, cfg.maxKVCacheCapacity, cfg.hiddenSize);
    allocateRaggedRope(io, cfg);

    if (cfg.isDiffusionBackbone)
    {
        int32_t const maxCanvasLen = cfg.diffusionCanvasLength;
        io.outputLogits = Tensor({cfg.maxSupportedBatchSize * maxCanvasLen, cfg.outputVocabSize}, DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "PipelineIO::outputLogits");
        io.selectTokenIndices = Tensor({cfg.maxSupportedBatchSize * maxCanvasLen}, DeviceType::kGPU,
            nvinfer1::DataType::kINT64, "PipelineIO::selectTokenIndices");
        io.hostSelectTokenIndices = Tensor({cfg.maxSupportedBatchSize * maxCanvasLen}, DeviceType::kCPU,
            nvinfer1::DataType::kINT64, "PipelineIO::hostSelectTokenIndices");
    }

    if (cfg.useVisionBidirectionalAttention)
    {
        io.visionBlockIds = Tensor(
            {cfg.maxPhysicalTokens}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::visionBlockIds");
    }

    if (hasDeepstackFeatures(cfg))
    {
        allocateDeepstackEmbeds(
            io, cfg.numDeepstackFeatures, 1, cfg.maxPhysicalTokens, cfg.hiddenSize, nvinfer1::DataType::kHALF);
        LOG_INFO("Allocated %d token-major deepstack tensors with shape [%d, %d]", cfg.numDeepstackFeatures,
            cfg.maxPhysicalTokens, cfg.hiddenSize);
    }

    // Engine-output hidden states for the vanilla LLM path. Always allocated:
    // streaming consumers (Qwen3-Omni Talker) read it; if the engine emits
    // hidden_states but no consumer is set, the buffer is harmless write-target;
    // if the engine has no hidden_states output the binding is silently skipped.
    io.outputHiddenStates = Tensor({cfg.maxPhysicalTokens, cfg.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "PipelineIO::outputHiddenStates");

    if (cfg.ropeConfig.type == RopeType::kMRope)
    {
        int32_t const ropeRows = cfg.recurrentPoolRows;
        allocateMRope(io, ropeRows, cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity, cfg.rotaryDim);
        // Give every resident slot valid sequential positions before any multimodal request publishes into it.
        kernel::initializeTextOnlyMRopeCosSin(io.mropeCosSin.dataPointer<float>(), cfg.ropeConfig.rotaryTheta,
            cfg.rotaryDim, cfg.maxKVCacheCapacity, ropeRows, stream);
    }

    // Runtime skip-softmax override carrier (shape-only).
    io.skipSoftmaxScale = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT8, "PipelineIO::skipSoftmaxScale");
    CUDA_CHECK(cudaMemsetAsync(io.skipSoftmaxScale.rawPointer(), 0, io.skipSoftmaxScale.getMemoryCapacity(), stream));

    return io;
}

PipelineIO PipelineIO::createForSpecDecode(DeploymentConfig const& bundle, int32_t maxRuntimeBatchSize,
    cudaStream_t stream, bool hasAcceptHiddenOutput, bool hasTreeMetadataInputs)
{
    check::check(bundle.draft.has_value(), "PipelineIO::createForSpecDecode requires DeploymentConfig.draft to be set");
    check::check(bundle.specConfig.has_value(),
        "PipelineIO::createForSpecDecode requires DeploymentConfig.specConfig to be set");

    PipelineIO io;

    int32_t const maxDraftProposalSize = bundle.specConfig->maxDraftProposalSize;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftRuntimeHiddenSize
        = bundle.specDecodeMode() == SpecDecodeMode::kGemma4MTP ? baseOutputHiddenDim : draftHiddenSize;
    int32_t const draftVocabSize = bundle.draft->vocabSize;

    // Use max of base and draft dimensions for shared tensors
    int32_t const maxInputLength = std::max(bundle.base.maxSupportedInputLength, bundle.draft->maxSupportedInputLength);
    int32_t const effectiveMaxDraftProposalSize
        = std::max({maxDraftProposalSize, bundle.specConfig->verifySize, bundle.specConfig->dflashBlockSize});
    int32_t const maxLogitsSize = maxRuntimeBatchSize * effectiveMaxDraftProposalSize;
    int32_t const maxVocabSize = std::max(bundle.base.outputVocabSize, draftVocabSize);
    int32_t const maxTensorSeqLen = std::max(maxInputLength, effectiveMaxDraftProposalSize);

    allocateBasicIO(io, maxRuntimeBatchSize, maxVocabSize);

    int32_t const maxPhysicalTokens = std::max(bundle.base.maxPhysicalTokens, bundle.draft->maxPhysicalTokens);
    int32_t const maxSequences = std::max(bundle.base.maxNumSequences, bundle.draft->maxNumSequences);
    int32_t const maxKVCacheCapacity = std::max(bundle.base.maxKVCacheCapacity, bundle.draft->maxKVCacheCapacity);
    int32_t const maxHiddenSize = std::max(bundle.base.hiddenSize, bundle.draft->hiddenSize);
    allocateRaggedMetadata(io, maxPhysicalTokens, maxSequences, maxLogitsSize, maxKVCacheCapacity, maxHiddenSize);

    int32_t const genericRows = std::max(bundle.base.useDualRope ? 0 : bundle.base.maxPhysicalTokens,
        bundle.draft->useDualRope ? 0 : bundle.draft->maxPhysicalTokens);
    int32_t const dualRows = std::max(bundle.base.useDualRope ? bundle.base.maxPhysicalTokens : 0,
        bundle.draft->useDualRope ? bundle.draft->maxPhysicalTokens : 0);
    int32_t const rotaryDim = std::max(
        bundle.base.useDualRope ? 0 : bundle.base.rotaryDim, bundle.draft->useDualRope ? 0 : bundle.draft->rotaryDim);
    int32_t const slidingRotaryDim = std::max(bundle.base.useDualRope ? bundle.base.slidingRotaryDim : 0,
        bundle.draft->useDualRope ? bundle.draft->slidingRotaryDim : 0);
    int32_t const fullRotaryDim = std::max(bundle.base.useDualRope ? bundle.base.fullRotaryDim : 0,
        bundle.draft->useDualRope ? bundle.draft->fullRotaryDim : 0);
    allocateRaggedRopeBuffers(io, genericRows, dualRows, rotaryDim, slidingRotaryDim, fullRotaryDim);

    // Override outputLogits to support proposal-sized outputs: [maxLogitsSize, maxVocabSize].
    // dtype is kFLOAT (matching allocateBasicIO); only the shape changes for SpecDecode.
    io.outputLogits = rt::Tensor(
        {maxLogitsSize, maxVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "PipelineIO::outputLogits");

    // Allocate hidden states for SpecDecode. Cached block-draft modes bind the
    // draft target-hidden input to compact base hidden states, so they do not
    // need the generic EAGLE/MTP draft hidden-state ping-pong buffers.
    allocateSpecDecodeHiddenStates(io, maxRuntimeBatchSize, maxTensorSeqLen, baseOutputHiddenDim,
        draftRuntimeHiddenSize, nvinfer1::DataType::kHALF, !isCachedBlockDraftMode(bundle.specDecodeMode()));

    // Accept-layer hidden states for the Qwen3-Omni Talker, only when the base
    // engine can actually fill them. Sized on the base hidden size, not
    // baseOutputHiddenDim — the latter is the draft's input width and is
    // 3x hidden for EAGLE3.
    if (hasAcceptHiddenOutput)
    {
        io.outputHiddenStates = Tensor({maxRuntimeBatchSize, maxTensorSeqLen, bundle.base.hiddenSize}, DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "PipelineIO::outputHiddenStates");
    }

    if (hasDeepstackFeatures(bundle.base))
    {
        allocateDeepstackEmbeds(io, bundle.base.numDeepstackFeatures, maxRuntimeBatchSize, maxInputLength,
            bundle.base.hiddenSize, nvinfer1::DataType::kHALF);
        LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]", bundle.base.numDeepstackFeatures,
            maxRuntimeBatchSize, maxInputLength, bundle.base.hiddenSize);
    }

    if (bundle.base.ropeConfig.type == RopeType::kMRope)
    {
        int32_t const residentRows = bundle.base.recurrentPoolRows;
        allocateMRope(io, residentRows, maxRuntimeBatchSize, bundle.base.maxKVCacheCapacity, bundle.base.rotaryDim);
        kernel::initializeTextOnlyMRopeCosSin(io.mropeCosSin.dataPointer<float>(), bundle.base.ropeConfig.rotaryTheta,
            bundle.base.rotaryDim, bundle.base.maxKVCacheCapacity, residentRows, stream);
    }

    // SpecDecode-specific engine I/O: token-aligned attention metadata and a
    // proposal-sized selectTokenIndices override.
    int64_t const packedMaskLen = static_cast<int64_t>(divUp(effectiveMaxDraftProposalSize, 32));
    io.packedAttentionMask = Tensor({maxRuntimeBatchSize, maxTensorSeqLen, packedMaskLen}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::packedAttentionMask");
    CUDA_CHECK(
        cudaMemsetAsync(io.packedAttentionMask.rawPointer(), 0, io.packedAttentionMask.getMemoryCapacity(), stream));

    io.specDecodePositionIds = Tensor({maxRuntimeBatchSize, maxTensorSeqLen}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::specDecodePositionIds");
    CUDA_CHECK(cudaMemsetAsync(
        io.specDecodePositionIds.rawPointer(), 0, io.specDecodePositionIds.getMemoryCapacity(), stream));

    io.selectTokenIndices = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize}, DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "PipelineIO::selectTokenIndices");
    CUDA_CHECK(
        cudaMemsetAsync(io.selectTokenIndices.rawPointer(), 0, io.selectTokenIndices.getMemoryCapacity(), stream));

    io.skipSoftmaxScale = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT8, "PipelineIO::skipSoftmaxScale");
    CUDA_CHECK(cudaMemsetAsync(io.skipSoftmaxScale.rawPointer(), 0, io.skipSoftmaxScale.getMemoryCapacity(), stream));

    SpecDecodeMode const mode = bundle.specDecodeMode();
    bool const useSpecTree = mode == SpecDecodeMode::kDFlash
        || ((isCachedBlockDraftMode(mode) || mode == SpecDecodeMode::kMTP || mode == SpecDecodeMode::kGemma4MTP
                || mode == SpecDecodeMode::kDSpark)
            && bundle.specConfig->draftingTopK > 1);
    if (useSpecTree || hasTreeMetadataInputs)
    {
        io.specTreeParentIds = Tensor({maxRuntimeBatchSize, maxTensorSeqLen}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "PipelineIO::specTreeParentIds");
        CUDA_CHECK(
            cudaMemsetAsync(io.specTreeParentIds.rawPointer(), 0, io.specTreeParentIds.getMemoryCapacity(), stream));

        io.specTreeDepths = Tensor({maxRuntimeBatchSize, maxTensorSeqLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
            "PipelineIO::specTreeDepths");
        CUDA_CHECK(cudaMemsetAsync(io.specTreeDepths.rawPointer(), 0, io.specTreeDepths.getMemoryCapacity(), stream));
    }

    return io;
}

} // namespace rt
} // namespace trt_edgellm
