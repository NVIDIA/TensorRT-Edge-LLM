/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/sequenceStepRuntime.h"
#include "common/bindingNames.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/state/prefillChunk.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{
void require(bool condition, char const* message)
{
    if (!condition)
    {
        throw std::logic_error(message);
    }
}
} // namespace

struct SequenceStepRuntime::Lease
{
    explicit Lease(std::atomic<bool>& gate)
        : gate(gate)
    {
        bool expected = false;
        require(gate.compare_exchange_strong(expected, true, std::memory_order_acquire), "Runtime already leased");
    }
    ~Lease()
    {
        if (!poisoned)
        {
            gate.store(false, std::memory_order_release);
        }
    }
    std::atomic<bool>& gate;
    bool poisoned{};
};

struct SequenceStepRuntime::View
{
    TensorMap bindings;
    std::map<std::string, Tensor> rows;
};

SequenceStepRuntime::SequenceStepRuntime(LLMRankRuntime& runtime, cudaStream_t stream, bool captureGraphs)
    : mRuntime(runtime)
    , mStream(stream)
    , mLease(std::make_unique<Lease>(runtime.mHandleRequestInProgress))
    , mSlots(2, runtime.mDeployment.base.maxSupportedInputLength, runtime.mDeployment.base.maxKVCacheCapacity)
    , mHostStarts({2}, DeviceType::kCPU, nvinfer1::DataType::kINT32)
    , mDeviceStarts({2}, DeviceType::kGPU, nvinfer1::DataType::kINT32)
    , mHostSelect({2, 1}, DeviceType::kCPU, nvinfer1::DataType::kINT64)
{
    auto const& config = runtime.mDeployment.base;
    require(stream != nullptr, "Step runtime requires an explicit stream");
    require(runtime.mMaxRuntimeBatchSize == 2 && !runtime.mMapping.isParallel() && config.modelType == "qwen3_5_text"
            && !runtime.hasDraftModel() && !runtime.mContextCache && config.maxSupportedLoraRank == 0
            && !config.isDiffusionBackbone && !config.useVisionBidirectionalAttention && !config.useContextDependentRope
            && !config.useDualRope && !runtime.mDeepstack && !runtime.mGemma4Ple && config.reducedVocabSize == 0,
        "Step runtime supports two-slot vanilla text-only Qwen3.5 without context reuse or LoRA");
    require(runtime.mSharedResources->kvPageTables[0]->isIdentity(), "Step session requires identity page ownership");
    for (int32_t index = 0; index < 3; ++index)
    {
        auto view = std::make_unique<View>();
        view->bindings = runtime.mBaseTensorMap;
        int32_t const first = index == 1 ? 1 : 0;
        int32_t const count = index == 2 ? 2 : 1;
        auto bindRow = [&](std::string const& name) {
            Tensor* original = runtime.mBaseTensorMap.get(name);
            require(original != nullptr, "Missing selected-row binding");
            auto shape = original->getShape();
            require(shape[0] == 2, "Step session must be constructed before legacy execution reshapes state");
            size_t const bytes = original->getMemoryCapacity() / 2;
            shape[0] = count;
            auto* pointer = static_cast<std::byte*>(original->rawPointer()) + first * bytes;
            auto inserted = view->rows.emplace(name, Tensor(pointer, shape, DeviceType::kGPU, original->getDataType()));
            view->bindings.set(name, inserted.first->second);
        };
        for (int32_t layer = 0; layer < config.numLinearAttnLayers; ++layer)
        {
            bindRow(binding_names::formatRecurrentStateName(layer, true));
            bindRow(binding_names::formatRecurrentStateName(layer, false));
            bindRow(binding_names::formatConvStateName(layer, true));
            bindRow(binding_names::formatConvStateName(layer, false));
        }
        bindRow(binding_names::kKVPageTable);
        if (config.ropeConfig.type == RopeType::kMRope)
        {
            bindRow(binding_names::kRopeCosSin);
        }
        view->bindings.set(binding_names::kKVCacheStartIndex, mDeviceStarts);
        mViews[index] = std::move(view);
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&mComplete, cudaEventDisableTiming));
    if (captureGraphs)
    {
        try
        {
            captureDecodeViews();
        }
        catch (...)
        {
            mLease->poisoned = true;
            cudaStreamSynchronize(mStream);
            cudaEventDestroy(mComplete);
            mComplete = nullptr;
            throw;
        }
    }
}

void SequenceStepRuntime::captureDecodeViews()
{
    // Capture warmups mutate hybrid state; only disposable startup sequences may own it here.
    SequenceOptions options;
    options.maxOutputTokens = 2;
    auto first = acquire(1, {0}, options);
    auto second = acquire(2, {0}, options);
    for (auto handle : {first, second})
    {
        beginPrefillChunk(handle);
        completeStep();
        acceptToken(handle, 0);
        beginDecode({handle, {}}, 1);
        completeStep();
        require(mRuntime.mBaseExecutor->captureGraph(mStream), "Selected-slot graph capture failed");
        acceptToken(handle, 0);
    }
    // Recreate logical inputs while retaining the same physical addresses for the paired view.
    finish(first);
    finish(second);
    release(first);
    release(second);
    first = acquire(3, {0}, options);
    second = acquire(4, {0}, options);
    for (auto handle : {first, second})
    {
        beginPrefillChunk(handle);
        completeStep();
        acceptToken(handle, 0);
    }
    beginDecode({first, second}, 2);
    completeStep();
    require(mRuntime.mBaseExecutor->captureGraph(mStream), "Paired graph capture failed");
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    finish(first);
    finish(second);
    release(first);
    release(second);
    mRuntime.zeroRecurrentStates(0, mStream);
    mRuntime.zeroRecurrentStates(1, mStream);
}

SequenceStepRuntime::~SequenceStepRuntime()
{
    // Drain borrowed buffers before releasing the lease, including partially enqueued failed steps.
    if (cudaStreamSynchronize(mStream) != cudaSuccess)
    {
        mLease->poisoned = true;
    }
    if (mComplete != nullptr)
    {
        cudaEventDestroy(mComplete);
    }
}

void SequenceStepRuntime::poison() noexcept
{
    mLease->poisoned = true;
}

bool SequenceStepRuntime::healthy() const noexcept
{
    return !mLease->poisoned;
}

void SequenceStepRuntime::requireIdle() const
{
    require(healthy() && !mPending, "Step runtime is failed or has an unfinished forward");
}

SequenceState const& SequenceStepRuntime::state(SequenceHandle handle) const
{
    return mSlots.get(handle);
}

SequenceHandle SequenceStepRuntime::acquire(uint64_t requestId, std::vector<int32_t> prompt, SequenceOptions options)
{
    requireIdle();
    int64_t const vocabulary = mRuntime.mEmbedding.table.getShape()[0];
    for (int32_t token : prompt)
    {
        require(token >= 0 && token < vocabulary, "Prompt token outside embedding vocabulary");
    }
    auto handle = mSlots.acquire(requestId, std::move(prompt), std::move(options));
    try
    {
        mRuntime.zeroRecurrentStates(handle.slot, mStream);
    }
    catch (...)
    {
        mLease->poisoned = true;
        throw;
    }
    return handle;
}

Tensor const& SequenceStepRuntime::beginPrefill(SequenceHandle handle, int32_t count)
{
    requireIdle();
    auto const& sequence = state(handle);
    require(sequence.phase() == SequencePhase::kPrefill && count > 0
            && count <= static_cast<int32_t>(sequence.prompt().size()) - sequence.promptCursor(),
        "Invalid prefill span");
    return enqueue({handle, {}}, 1, count, false);
}

Tensor const& SequenceStepRuntime::beginPrefillChunk(SequenceHandle handle)
{
    requireIdle();
    auto const& sequence = state(handle);
    int32_t const remaining = static_cast<int32_t>(sequence.prompt().size()) - sequence.promptCursor();
    require(sequence.promptCursor() % 64 == 0 && (sequence.promptCursor() == 0 || remaining >= 64),
        "Chunk policy cannot continue an incompatible manually partitioned prompt");
    return beginPrefill(handle, nextPrefillChunkSize(remaining));
}

Tensor const& SequenceStepRuntime::beginDecode(std::array<SequenceHandle, 2> const& handles, int32_t count)
{
    requireIdle();
    require(count == 1 || count == 2, "Invalid decode batch size");
    for (int32_t i = 0; i < count; ++i)
    {
        auto const& sequence = state(handles[i]);
        require(sequence.phase() == SequencePhase::kDecode, "Decode requires a pending sampled token");
        require(i == 0 || handles[i].slot == handles[i - 1].slot + 1, "Slots must be distinct and physically ordered");
    }
    return enqueue(handles, count, 1, true);
}

Tensor const& SequenceStepRuntime::enqueue(
    std::array<SequenceHandle, 2> const& handles, int32_t count, int32_t span, bool decode)
{
    auto& io = *mRuntime.mPipelineIO;
    auto const& config = mRuntime.mDeployment.base;
    // A nonempty-cache S=1 binding selects attention decode and requires absolute context lengths.
    bool const executeDecode = decode || (span == 1 && state(handles[0]).committedTokens() > 0);
    int32_t const physicalSpan = span;
    require(mRuntime.mIdsInput.reshape({count, physicalSpan})
            && mRuntime.mHostPackedTokenIds.reshape({count, physicalSpan})
            && io.inputsEmbeds.reshape({count, physicalSpan, config.hiddenSize})
            && io.outputLogits.reshape({count, config.outputVocabSize}) && io.contextLengths.reshape({count})
            && io.hostContextLengths.reshape({count}) && io.selectTokenIndices.reshape({count, 1}),
        "Step tensor shape exceeds allocated capacity");
    auto* packed = mRuntime.mHostPackedTokenIds.dataPointer<int32_t>();
    std::fill_n(packed, count * physicalSpan, 0);
    for (int32_t row = 0; row < count; ++row)
    {
        auto const& sequence = state(handles[row]);
        mHostStarts.dataPointer<int32_t>()[row] = sequence.committedTokens();
        io.hostContextLengths.dataPointer<int32_t>()[row] = executeDecode ? sequence.committedTokens() + 1 : span;
        mHostSelect.dataPointer<int64_t>()[row] = executeDecode ? 0 : span - 1;
        if (decode)
        {
            packed[row] = sequence.output().back();
        }
        else
        {
            std::copy_n(sequence.prompt().data() + sequence.promptCursor(), span, packed + row * physicalSpan);
        }
    }
    mPending = true;
    mHandles = handles;
    mCount = count;
    mSpan = span;
    mDecode = decode;
    try
    {
        // Text positions are request-independent; initialize once before the first forward in this session.
        if (!mRopeInitialized && config.ropeConfig.type == RopeType::kMRope)
        {
            kernel::initializeTextOnlyMRopeCosSin(io.mropeCosSin.dataPointer<float>(), config.ropeConfig.rotaryTheta,
                config.rotaryDim, config.maxKVCacheCapacity, 2, mStream);
            mRopeInitialized = true;
        }
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.mIdsInput.rawPointer(), packed,
            static_cast<size_t>(count) * physicalSpan * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mDeviceStarts.rawPointer(), mHostStarts.rawPointer(), count * sizeof(int32_t),
            cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(io.contextLengths.rawPointer(), io.hostContextLengths.rawPointer(),
            count * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(io.selectTokenIndices.rawPointer(), mHostSelect.rawPointer(),
            count * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        kernel::embeddingLookup(mRuntime.mIdsInput, mRuntime.mEmbedding.table, mRuntime.mEmbedding.scalesAsOptional(),
            io.inputsEmbeds, mStream);
        auto const dims = executeDecode
            ? config.decodeDims(count)
            : config.prefillDims(count, physicalSpan, state(handles[0]).committedTokens() == 0);
        auto const& view = *mViews[count == 2 ? 2 : handles[0].slot];
        require(mRuntime.mBaseExecutor->prepare(executeDecode ? 1 : 0, dims, view.bindings, mStream),
            "Step engine preparation failed");
        require(mRuntime.mBaseExecutor->execute(mStream), "Step engine execution failed");
        CUDA_CHECK(cudaEventRecord(mComplete, mStream));
    }
    catch (...)
    {
        mLease->poisoned = true;
        throw;
    }
    return io.outputLogits;
}

void SequenceStepRuntime::completeStep()
{
    require(healthy() && mPending, "No healthy pending step");
    try
    {
        CUDA_CHECK(cudaEventSynchronize(mComplete));
        for (int32_t row = 0; row < mCount; ++row)
        {
            if (mDecode)
            {
                mSlots.commitDecode(mHandles[row]);
            }
            else
            {
                mSlots.commitPrompt(mHandles[row], mSpan);
            }
        }
        mPending = false;
    }
    catch (...)
    {
        mLease->poisoned = true;
        throw;
    }
}

void SequenceStepRuntime::acceptToken(SequenceHandle handle, int32_t token, uint64_t randomDraws)
{
    requireIdle();
    require(token >= 0 && token < mRuntime.mDeployment.base.outputVocabSize, "Sample outside vocabulary");
    mSlots.acceptToken(handle, token, randomDraws);
}

void SequenceStepRuntime::finish(SequenceHandle handle)
{
    requireIdle();
    mSlots.finish(handle);
}

void SequenceStepRuntime::release(SequenceHandle handle)
{
    requireIdle();
    mSlots.release(handle);
}
} // namespace rt
} // namespace trt_edgellm
