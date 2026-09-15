/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/bindingNames.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/greedySchedulerBackend.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/sequenceStepRuntime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{
void require(bool condition, std::string const& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

int32_t greedy(std::vector<float> const& logits)
{
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

bool compare(std::string const& label, std::vector<float> const& reference, std::vector<float> const& candidate,
    bool exact = false)
{
    require(reference.size() == candidate.size(), "Logit count mismatch");
    double squaredError = 0.0;
    double squaredReference = 0.0;
    float maxError = 0.0F;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        finite = finite && std::isfinite(reference[i]) && std::isfinite(candidate[i]);
        double const error = static_cast<double>(candidate[i]) - reference[i];
        squaredError += error * error;
        squaredReference += static_cast<double>(reference[i]) * reference[i];
        maxError = std::max(maxError, static_cast<float>(std::abs(error)));
    }
    double const relativeL2 = std::sqrt(squaredError / std::max(squaredReference, 1.0e-30));
    bool const sameGreedy = greedy(reference) == greedy(candidate);
    // Preliminary FP16 correctness screen, not a model-quality acceptance criterion.
    bool const passed = finite && sameGreedy && (exact ? maxError == 0.0F : maxError <= 0.1F && relativeL2 <= 0.005);
    std::cout << "COMPARE " << label << " max_abs=" << maxError << " relative_l2=" << relativeL2
              << " reference_greedy=" << greedy(reference) << " candidate_greedy=" << greedy(candidate)
              << " passed=" << passed << std::endl;
    return passed;
}
} // namespace

//! Model-dependent, eager-only feasibility probe; deliberately not a serving API.
class ContinuousBatchingProbe
{
public:
    ContinuousBatchingProbe(LLMRankRuntime& runtime, cudaStream_t stream)
        : mRuntime(runtime)
        , mStream(stream)
        , mOriginalMap(runtime.mBaseTensorMap)
        , mHostLengths({2}, DeviceType::kCPU, nvinfer1::DataType::kINT32)
        , mHostLogits({2, runtime.mDeployment.base.outputVocabSize}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
        , mScratch({kCOPY_BYTES}, DeviceType::kCPU, nvinfer1::DataType::kUINT8)
    {
        auto const& config = runtime.mDeployment.base;
        require(runtime.mMaxRuntimeBatchSize == 2 && !runtime.hasDraftModel(), "Requires two-slot vanilla engine");
        require(config.maxSupportedLoraRank == 0 && !config.useVisionBidirectionalAttention
                && config.reducedVocabSize == 0 && !config.isDiffusionBackbone,
            "Unsupported probe configuration");
        require(runtime.mPipelineIO->outputLogits.getDataType() == nvinfer1::DataType::kFLOAT,
            "Probe requires FP32 logits");
        require(runtime.mSharedResources->kvPageTables[0]->isIdentity(), "Probe requires identity KV pages");
        if (config.ropeConfig.type == RopeType::kMRope)
        {
            auto& rope = runtime.mPipelineIO->mropeCosSin;
            require(rope.reshape({2, config.maxKVCacheCapacity, config.rotaryDim}), "RoPE shape");
            kernel::initializeTextOnlyMRopeCosSin(rope.dataPointer<float>(), config.ropeConfig.rotaryTheta,
                config.rotaryDim, config.maxKVCacheCapacity, 2, stream);
        }
        auto& mamba = cache().getMambaCacheManager();
        require(mamba.numLayers() > 0, "Probe requires hybrid recurrent state");
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            addBinding(binding_names::formatRecurrentStateName(layer, true));
            addBinding(binding_names::formatRecurrentStateName(layer, false));
            addBinding(binding_names::formatConvStateName(layer, true));
            addBinding(binding_names::formatConvStateName(layer, false));
        }
        addBinding(binding_names::kKVPageTable);
        reset(0);
        reset(1);
        memory("initialized");
    }

    ~ContinuousBatchingProbe()
    {
        mRuntime.mBaseTensorMap = mOriginalMap;
    }

    //! Reset only the selected sequence's recurrent state and logical endpoint.
    void reset(int32_t slot)
    {
        require(slot >= 0 && slot < 2, "Invalid slot");
        mRuntime.zeroRecurrentStates(slot, mStream);
        mCommitted[slot] = 0;
    }

    //! Execute one contiguous physical-slot view using existing native primitives.
    std::vector<std::vector<float>> step(
        int32_t firstSlot, std::vector<std::vector<int32_t>> const& tokens, bool decode = false)
    {
        int32_t const count = static_cast<int32_t>(tokens.size());
        require(count > 0 && count <= 2 && firstSlot >= 0 && firstSlot + count <= 2, "Invalid selected slots");
        require(decode || count == 1, "Probe prefill selects one slot at a time");
        for (auto const& row : tokens)
        {
            require(!row.empty() && (!decode || row.size() == 1), "Invalid step token count");
        }
        select(firstSlot, count);
        DecodingInferenceContext context;
        context.initialize(count, 8, std::nullopt, {}, "", mStream);
        context.temperature = 0.0F;
        context.topK = 1;
        context.tokenIds = tokens;
        for (int32_t i = 0; i < count; ++i)
        {
            require(mCommitted[firstSlot + i] + static_cast<int32_t>(tokens[i].size())
                    <= mRuntime.mDeployment.base.maxKVCacheCapacity,
                "KV endpoint exceeds engine capacity");
            context.effectivePrefillLengths[i] = static_cast<int32_t>(tokens[i].size());
        }
        // Nonempty-cache S=1 is decoded by the attention plugin and needs absolute context lengths.
        bool const promptTail = !decode && tokens[0].size() == 1 && mCommitted[firstSlot] > 0;
        bool const executeDecode = decode || promptTail;
        std::cout << "STEP slot=" << firstSlot << " batch=" << count << " profile=" << (executeDecode ? 1 : 0)
                  << " logical_decode=" << decode << " prompt_tail=" << promptTail << " start=" << mCommitted[firstSlot]
                  << " tokens=" << tokens[0].size() << std::endl;
        bool const success = executeDecode ? mRuntime.mDecoderRegistry->cachePrimingStrategy().decodeStep(context)
                                           : mRuntime.runBaseModelPrefill(context, nullptr, false);
        require(success, "Native execution failed");
        int32_t const vocab = mRuntime.mDeployment.base.outputVocabSize;
        CUDA_CHECK(cudaMemcpyAsync(mHostLogits.rawPointer(), mRuntime.mPipelineIO->outputLogits.rawPointer(),
            static_cast<size_t>(count) * vocab * sizeof(float), cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mHostLengths.rawPointer(), cache().getKVCacheLengths().rawPointer(),
            count * sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<std::vector<float>> results;
        for (int32_t i = 0; i < count; ++i)
        {
            mCommitted[firstSlot + i] += static_cast<int32_t>(tokens[i].size());
            require(mHostLengths.dataPointer<int32_t>()[i] == mCommitted[firstSlot + i], "Incorrect committed length");
            auto const* begin = mHostLogits.dataPointer<float>() + i * vocab;
            results.emplace_back(begin, begin + vocab);
        }
        return results;
    }

    //! Snapshot all recurrent/conv bytes and the materialized attention-KV prefix.
    void observeEndpoint(int32_t slot, int32_t length)
    {
        mCommitted.at(slot) = length;
    }

    //! Capture sampler output scratch to detect accidental sampling by forward-only APIs.
    std::vector<std::byte> samplingBytes()
    {
        auto& indices = mRuntime.mSamplingIndices;
        std::vector<std::byte> result(indices.getMemoryCapacity());
        copyToHost(indices.rawPointer(), result.data(), result.size());
        return result;
    }

    //! Snapshot all recurrent/conv bytes and the materialized attention-KV prefix.
    void snapshot(int32_t slot)
    {
        mSnapshots.clear();
        mSnapshotSlot = slot;
        mSnapshotLength = mCommitted[slot];
        auto capture = [&](void* pointer, size_t bytes) {
            Snapshot entry{pointer, std::vector<std::byte>(bytes)};
            copyToHost(pointer, entry.bytes.data(), bytes);
            mSnapshots.push_back(std::move(entry));
        };
        auto& mamba = cache().getMambaCacheManager();
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            for (Tensor* tensor : {&mamba.getRecurrentState(layer), &mamba.getConvState(layer)})
            {
                size_t const bytes = tensor->getMemoryCapacity() / 2;
                capture(static_cast<std::byte*>(tensor->rawPointer()) + slot * bytes, bytes);
            }
        }
        auto& kv = cache().getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            auto& tensor = kv.getCombinedKVCache(layer);
            auto const shape = tensor.getShape();
            size_t const tokenBytes = shape[3] * shape[4] * sizeof(half);
            require(tensor.getDataType() == nvinfer1::DataType::kHALF, "Probe requires FP16 KV");
            int64_t const pagesPerSlot = (mRuntime.mDeployment.base.maxKVCacheCapacity + shape[2] - 1) / shape[2];
            require(shape[1] == 2 * pagesPerSlot, "Probe requires a two-slot pool with no surplus pages");
            size_t const halfBytes = tensor.getMemoryCapacity() / 2;
            size_t const slotBytes = halfBytes / 2;
            require(slotBytes / tokenBytes >= static_cast<size_t>(mCommitted[slot]), "KV snapshot range");
            for (int32_t plane = 0; plane < 2; ++plane)
            {
                capture(static_cast<std::byte*>(tensor.rawPointer()) + plane * halfBytes + slot * slotBytes,
                    mCommitted[slot] * tokenBytes);
            }
        }
        auto& table = mRuntime.mSharedResources->kvPageTables[0]->kernelView();
        size_t const tableBytes = table.getMemoryCapacity() / 2;
        capture(static_cast<std::byte*>(table.rawPointer()) + slot * tableBytes, tableBytes);
    }

    //! Require byte-for-byte preservation of the inactive request's live state.
    void verifySnapshot()
    {
        require(mSnapshotSlot >= 0 && mCommitted[mSnapshotSlot] == mSnapshotLength, "Inactive endpoint changed");
        size_t totalBytes = 0;
        for (auto const& entry : mSnapshots)
        {
            for (size_t offset = 0; offset < entry.bytes.size(); offset += kCOPY_BYTES)
            {
                size_t const bytes = std::min(static_cast<size_t>(kCOPY_BYTES), entry.bytes.size() - offset);
                CUDA_CHECK(cudaMemcpyAsync(mScratch.rawPointer(), static_cast<std::byte*>(entry.pointer) + offset,
                    bytes, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                require(std::memcmp(mScratch.rawPointer(), entry.bytes.data() + offset, bytes) == 0,
                    "Inactive state modified");
                totalBytes += bytes;
            }
        }
        std::cout << "ISOLATION slot=" << mSnapshotSlot << " exact_bytes=" << totalBytes << " passed=1" << std::endl;
    }

    int32_t vocabularySize() const
    {
        return mRuntime.mDeployment.base.outputVocabSize;
    }

    //! Compare live physical rows without allocating another state-sized snapshot.
    bool compareActive(std::string const& label, int32_t length)
    {
        Tensor candidate({kCOPY_BYTES}, DeviceType::kCPU, nvinfer1::DataType::kUINT8);
        bool passed = true;
        auto check = [&](Tensor& tensor, size_t rowBytes, size_t planeOffset, size_t bytes, std::string const& name) {
            double squaredError = 0.0, squaredReference = 0.0;
            double maxError = 0.0;
            bool finite = true;
            size_t const elementBytes
                = tensor.getDataType() == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half);
            for (size_t offset = 0; offset < bytes; offset += kCOPY_BYTES)
            {
                size_t const count = std::min(static_cast<size_t>(kCOPY_BYTES), bytes - offset);
                auto* source = static_cast<std::byte*>(tensor.rawPointer()) + planeOffset + offset;
                CUDA_CHECK(cudaMemcpyAsync(mScratch.rawPointer(), source, count, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(
                    cudaMemcpyAsync(candidate.rawPointer(), source + rowBytes, count, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                for (size_t i = 0; i < count / elementBytes; ++i)
                {
                    double const a = elementBytes == sizeof(float) ? mScratch.dataPointer<float>()[i]
                                                                   : __half2float(mScratch.dataPointer<half>()[i]);
                    double const b = elementBytes == sizeof(float) ? candidate.dataPointer<float>()[i]
                                                                   : __half2float(candidate.dataPointer<half>()[i]);
                    finite = finite && std::isfinite(a) && std::isfinite(b);
                    maxError = std::max(maxError, std::abs(a - b));
                    squaredError += (a - b) * (a - b);
                    squaredReference += a * a;
                }
            }
            double const relative = std::sqrt(squaredError / std::max(squaredReference, 1.0e-30));
            bool const ok = finite && maxError <= 0.1 && relative <= 0.005;
            passed = ok && passed;
            std::cout << "ACTIVE_STATE " << label << " " << name << " max_abs=" << maxError
                      << " relative_l2=" << relative << " finite=" << finite << " passed=" << ok << std::endl;
        };
        auto& mamba = cache().getMambaCacheManager();
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            auto& recurrent = mamba.getRecurrentState(layer);
            auto& conv = mamba.getConvState(layer);
            check(recurrent, recurrent.getMemoryCapacity() / 2, 0, recurrent.getMemoryCapacity() / 2,
                "recurrent_" + std::to_string(layer));
            check(conv, conv.getMemoryCapacity() / 2, 0, conv.getMemoryCapacity() / 2, "conv_" + std::to_string(layer));
        }
        auto& kv = cache().getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            auto& tensor = kv.getCombinedKVCache(layer);
            auto const shape = tensor.getShape();
            size_t const tokenBytes = shape[3] * shape[4] * sizeof(half);
            size_t const planeBytes = tensor.getMemoryCapacity() / 2;
            for (int32_t plane = 0; plane < 2; ++plane)
            {
                check(tensor, planeBytes / 2, plane * planeBytes, length * tokenBytes,
                    "kv_" + std::to_string(layer) + "_" + std::to_string(plane));
            }
        }
        return passed;
    }

    //! Record device allocator availability, not process RSS or exclusive GPU use.
    void memory(std::string const& label)
    {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        std::cout << "MEMORY " << label << " cuda_free=" << freeBytes << " cuda_total=" << totalBytes << std::endl;
    }

private:
    struct Binding
    {
        Tensor* original;
        Coords shape;
    };
    struct Snapshot
    {
        void* pointer;
        std::vector<std::byte> bytes;
    };
    static constexpr int64_t kCOPY_BYTES = 1024 * 1024;

    HybridCacheManager& cache()
    {
        return *mRuntime.mSharedResources->cacheManagers[0];
    }

    void addBinding(std::string const& name)
    {
        auto* tensor = mOriginalMap.get(name);
        require(tensor != nullptr && tensor->getShape()[0] == 2, "Invalid physical binding: " + name);
        mBindings.emplace(name, Binding{tensor, tensor->getShape()});
    }

    void select(int32_t firstSlot, int32_t count)
    {
        for (auto const& [name, binding] : mBindings)
        {
            auto shape = binding.shape;
            size_t const rowBytes = binding.original->getMemoryCapacity() / 2;
            shape[0] = count;
            auto* pointer = static_cast<std::byte*>(binding.original->rawPointer()) + firstSlot * rowBytes;
            mViews[name] = Tensor(pointer, shape, DeviceType::kGPU, binding.original->getDataType());
            mRuntime.mBaseTensorMap.set(name, mViews.at(name));
        }
        require(mHostLengths.reshape({count}), "Selected lengths shape");
        for (int32_t i = 0; i < count; ++i)
        {
            mHostLengths.dataPointer<int32_t>()[i] = mCommitted[firstSlot + i];
        }
        // Cache-manager lengths are execution scratch here; canonical endpoints stay slot-owned.
        Tensor lengths(mHostLengths.rawPointer(), {count}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cache().resetForNewSequences(lengths, mStream);
    }

    void copyToHost(void* source, std::byte* destination, size_t total)
    {
        for (size_t offset = 0; offset < total; offset += kCOPY_BYTES)
        {
            size_t const bytes = std::min(static_cast<size_t>(kCOPY_BYTES), total - offset);
            CUDA_CHECK(cudaMemcpyAsync(mScratch.rawPointer(), static_cast<std::byte*>(source) + offset, bytes,
                cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            std::memcpy(destination + offset, mScratch.rawPointer(), bytes);
        }
    }

    LLMRankRuntime& mRuntime;
    cudaStream_t mStream;
    TensorMap mOriginalMap;
    Tensor mHostLengths;
    Tensor mHostLogits;
    Tensor mScratch;
    std::array<int32_t, 2> mCommitted{};
    std::map<std::string, Binding> mBindings;
    std::map<std::string, Tensor> mViews;
    std::vector<Snapshot> mSnapshots;
    int32_t mSnapshotSlot{-1};
    int32_t mSnapshotLength{};
};

//! Run bounded synthetic continuation and staggered slot-isolation fixtures.
bool runProbe(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream)
{
    ContinuousBatchingProbe probe(runtime, stream);
    auto const vocabulary = tokenizer.encode("The red fox walks beside the blue river. Count one two three four. ");
    require(!vocabulary.empty(), "Empty synthetic fixture");
    std::vector<int32_t> prompt;
    for (int32_t i = 0; i < 129; ++i)
    {
        prompt.push_back(static_cast<int32_t>(vocabulary[i % vocabulary.size()]));
    }
    bool passed = true;
    for (int32_t length : {1, 3, 4, 63, 64, 65, 127, 128, 129})
    {
        std::vector<int32_t> const input(prompt.begin(), prompt.begin() + length);
        probe.reset(0);
        auto const reference = probe.step(0, {input})[0];
        int32_t const forcedToken = greedy(reference);
        auto const referenceDecode = probe.step(0, {{forcedToken}}, true)[0];
        probe.reset(1);
        std::vector<float> candidate;
        for (int32_t offset = 0; offset < length;)
        {
            int32_t const end = std::min(offset + 64, length);
            candidate = probe.step(1, {{input.begin() + offset, input.begin() + end}})[0];
            offset = end;
        }
        passed = compare("chunk_prefill_" + std::to_string(length), reference, candidate, length <= 64) && passed;
        auto const candidateDecode = probe.step(1, {{forcedToken}}, true)[0];
        passed = compare("chunk_decode_" + std::to_string(length), referenceDecode, candidateDecode, length <= 64)
            && passed;
    }

    probe.reset(0);
    auto const firstA = probe.step(0, {prompt})[0];
    auto const tokenA = greedy(firstA);
    auto const nextA = probe.step(0, {{tokenA}}, true)[0];
    probe.reset(1);
    std::vector<int32_t> promptB(prompt.rbegin(), prompt.rend());
    auto const firstB = probe.step(1, {promptB})[0];
    auto const tokenB = greedy(firstB);
    auto const nextB = probe.step(1, {{tokenB}}, true)[0];

    probe.reset(0);
    passed = compare("A_reuse", firstA, probe.step(0, {prompt})[0], true) && passed;
    probe.snapshot(0);
    probe.reset(1);
    probe.step(1, {{promptB.begin(), promptB.begin() + 64}});
    probe.verifySnapshot();
    probe.snapshot(1);
    passed = compare("A_decode_during_B_prefill", nextA, probe.step(0, {{tokenA}}, true)[0], true) && passed;
    probe.verifySnapshot();
    probe.snapshot(0);
    probe.step(1, {{promptB.begin() + 64, promptB.begin() + 128}});
    probe.verifySnapshot();
    auto const chunkB = probe.step(1, {{promptB.back()}})[0];
    probe.verifySnapshot();
    passed = compare("B_one_token_tail", firstB, chunkB) && passed;
    probe.snapshot(0);
    passed = compare("B_decode_after_A", nextB, probe.step(1, {{tokenB}}, true)[0]) && passed;
    probe.verifySnapshot();

    auto const secondTokenA = greedy(nextA);
    auto const secondTokenB = greedy(nextB);
    probe.reset(0);
    probe.reset(1);
    probe.step(0, {prompt});
    probe.step(1, {promptB});
    probe.step(0, {{tokenA}}, true);
    probe.step(1, {{tokenB}}, true);
    auto const secondA = probe.step(0, {{secondTokenA}}, true)[0];
    auto const secondB = probe.step(1, {{secondTokenB}}, true)[0];
    probe.reset(0);
    probe.reset(1);
    probe.step(0, {prompt});
    probe.step(1, {promptB});
    probe.step(0, {{tokenA}}, true);
    probe.step(1, {{tokenB}}, true);
    auto const batched = probe.step(0, {{secondTokenA}, {secondTokenB}}, true);
    passed = compare("two_slot_decode_A", secondA, batched[0]) && passed;
    passed = compare("two_slot_decode_B", secondB, batched[1]) && passed;
    probe.memory("completed");
    return passed;
}
} // namespace rt
} // namespace trt_edgellm

namespace trt_edgellm
{
namespace rt
{
//! Validate the production step interfaces against the independent P1/legacy path.
bool runStepTests(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream, bool graphs = false)
{
    auto const words = tokenizer.encode("A red fox crosses a blue river. One two three four. ");
    require(!words.empty(), "Missing fixture tokens");
    std::vector<int32_t> promptA;
    std::vector<int32_t> promptB;
    for (int32_t i = 0; i < 129; ++i)
    {
        promptB.push_back(words[i % words.size()]);
        if (i < 65)
        {
            promptA.push_back(words[i % words.size()]);
        }
    }
    std::vector<float> firstA, firstB, nextA, nextB, chunkFirstB, chunkNextB;
    {
        ContinuousBatchingProbe reference(runtime, stream);
        firstA = reference.step(0, {promptA})[0];
        firstB = reference.step(1, {promptB})[0];
        nextA = reference.step(0, {{greedy(firstA)}}, true)[0];
        nextB = reference.step(1, {{greedy(firstB)}}, true)[0];
        reference.reset(1);
        reference.step(1, {{promptB.begin(), promptB.begin() + 64}});
        reference.step(1, {{promptB.begin() + 64, promptB.begin() + 128}});
        chunkFirstB = reference.step(1, {{promptB.back()}})[0];
        chunkNextB = reference.step(1, {{greedy(firstB)}}, true)[0];
        bool const baselineQuality = compare("P1_decode_tail_quality_diagnostic", nextB, chunkNextB);
        std::cout << "P1_TAIL_BASELINE quality_passed=" << baselineQuality << std::endl;
    }
    bool passed = true;
    Tensor hostLogits({2, static_cast<int64_t>(firstA.size())}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    {
        ContinuousBatchingProbe observer(runtime, stream);
        auto const samplerBefore = observer.samplingBytes();
        SequenceStepRuntime steps(runtime, stream, graphs);
        auto rejects = [&](auto operation, char const* label) {
            bool rejected = false;
            try
            {
                operation();
            }
            catch (std::logic_error const&)
            {
                rejected = true;
            }
            require(rejected && steps.healthy(), label);
            std::cout << "REJECT " << label << " passed=1" << std::endl;
        };
        rejects([&] { SequenceStepRuntime duplicate(runtime, stream); }, "second_session");
        LLMGenerationRequest request{};
        LLMGenerationResponse response;
        require(!runtime.handleRequest(request, response, stream), "Legacy call entered a leased runtime");
        auto finish = [&](Tensor const& logits) {
            steps.completeStep();
            size_t const elements = logits.getShape().volume();
            CUDA_CHECK(cudaMemcpyAsync(hostLogits.rawPointer(), logits.rawPointer(), elements * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::vector<float>> values;
            for (int64_t row = 0; row < logits.getShape()[0]; ++row)
            {
                float const* begin = hostLogits.dataPointer<float>() + row * firstA.size();
                values.emplace_back(begin, begin + firstA.size());
            }
            return values;
        };
        SequenceOptions optionsA;
        optionsA.maxOutputTokens = 3;
        optionsA.temperature = 0.0F;
        SequenceOptions optionsB;
        optionsB.maxOutputTokens = 7;
        optionsB.seed = 123;
        auto a = steps.acquire(1, promptA, optionsA);
        auto b = steps.acquire(2, promptB, optionsB);
        auto const& logitsA = steps.beginPrefill(a, 65);
        require(steps.state(a).committedTokens() == 0, "Forward published state before completion");
        rejects([&] { steps.release(a); }, "release_in_flight");
        rejects([&] { steps.beginPrefill(b, 64); }, "overlapping_forward");
        passed = compare("native_prefill_A", firstA, finish(logitsA)[0], true) && passed;
        require(steps.state(a).output().empty(), "Prefill sampled unexpectedly");
        steps.acceptToken(a, greedy(firstA));
        require(steps.state(a).committedTokens() == 65, "Sample committed to KV early");
        observer.observeEndpoint(a.slot, 65);
        observer.snapshot(a.slot);
        finish(steps.beginPrefill(b, 64));
        observer.verifySnapshot();
        observer.observeEndpoint(b.slot, 64);
        observer.snapshot(b.slot);
        passed = compare("native_decode_A_while_B_prefills", nextA, finish(steps.beginDecode({a, {}}, 1))[0], true)
            && passed;
        observer.verifySnapshot();
        observer.observeEndpoint(a.slot, 66);
        observer.snapshot(a.slot);
        finish(steps.beginPrefill(b, 64));
        auto const tail = finish(steps.beginPrefill(b, 1))[0];
        passed = compare("native_sampling_free_tail_matches_P1", chunkFirstB, tail, true) && passed;
        require(steps.state(b).output().empty() && steps.state(b).committedTokens() == 129,
            "Prompt tail generated a token or committed incorrectly");
        observer.verifySnapshot();
        steps.acceptToken(b, greedy(firstB), 3);
        passed = compare("native_decode_B_after_tail_matches_P1", chunkNextB, finish(steps.beginDecode({b, {}}, 1))[0],
                     true)
            && passed;
        require(steps.state(b).randomCounter() == 3 && steps.state(a).randomCounter() == 0,
            "Random counters crossed requests");

        observer.observeEndpoint(b.slot, 130);
        observer.snapshot(b.slot);
        steps.finish(a);
        steps.release(a);
        auto c = steps.acquire(3, promptA, optionsA);
        require(c.slot == a.slot && c.generation != a.generation, "Slot was not safely reused");
        rejects([&] { steps.release(a); }, "stale_release");
        passed = compare("native_reused_slot_C", firstA, finish(steps.beginPrefill(c, 65))[0], true) && passed;
        observer.verifySnapshot();
        require(steps.state(b).options().maxOutputTokens == 7 && steps.state(b).output().size() == 1,
            "Partner request lost independent state");
        steps.release(c);
        steps.release(b);

        a = steps.acquire(4, promptA, optionsA);
        b = steps.acquire(5, promptB, optionsB);
        finish(steps.beginPrefill(a, 65));
        finish(steps.beginPrefill(b, 129));
        steps.acceptToken(a, greedy(firstA));
        steps.acceptToken(b, greedy(firstB));
        rejects([&] { steps.beginDecode({a, a}, 2); }, "duplicate_decode_slot");
        rejects([&] { steps.beginDecode({b, a}, 2); }, "unordered_decode_slots");
        auto const pair = finish(steps.beginDecode({a, b}, 2));
        passed = compare("native_unequal_batch_A", nextA, pair[0], true) && passed;
        passed = compare("native_unequal_batch_B", nextB, pair[1], true) && passed;
        require(steps.state(a).committedTokens() == 66 && steps.state(b).committedTokens() == 130,
            "Per-row endpoints did not commit independently");
        steps.release(a);
        steps.release(b);
        require(observer.samplingBytes() == samplerBefore, "Forward-only steps touched sampler output scratch");
        std::cout << "FORWARD_ONLY sampler_unchanged=1 passed=1" << std::endl;
    }

    LLMGenerationRequest legacy{};
    legacy.requests.resize(1);
    legacy.formattedRequests.resize(1);
    legacy.requests[0].messages.push_back({"user", {{"text", "Synthetic pretokenized regression fixture"}}});
    legacy.preTokenizedInputIds = {promptA};
    legacy.temperature = 0.0F;
    legacy.topK = 1;
    legacy.topP = 1.0F;
    legacy.maxGenerateLength = 2;
    legacy.applyChatTemplate = false;
    LLMGenerationResponse response;
    require(runtime.handleRequest(legacy, response, stream), "Legacy API failed after step session destruction");
    require(
        response.outputIds.size() == 1 && response.outputIds[0] == std::vector<int32_t>{greedy(firstA), greedy(nextA)},
        "Legacy output changed after releasing the step lease");
    std::cout << "LEGACY_RESTORED passed=1" << std::endl;
    if (graphs)
    {
        auto const stats = runtime.executionStats();
        passed = passed && stats.captures == 3 && stats.replays >= 4;
        std::cout << "P7_GRAPH_GATE passed=" << passed << " captures=" << stats.captures << " replays=" << stats.replays
                  << " eager=" << stats.eager << " profile_switches=" << stats.profileSwitches << std::endl;
    }
    std::cout << "P2_STATE_GATE passed=" << passed << " full_chunk_quality=separate_P1_TAIL_BASELINE" << std::endl;
    return passed;
}

//! Full-prompt numerical screen, separate from the matching-recipe P2 mechanism gate.
bool runChunkTests(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream, bool extended)
{
    auto const words = tokenizer.encode("A red fox crosses a blue river. One two three four. ");
    require(!words.empty(), "Missing fixture tokens");
    ContinuousBatchingProbe observer(runtime, stream);
    SequenceStepRuntime steps(runtime, stream);
    Tensor host({2, observer.vocabularySize()}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    auto finish = [&](Tensor const& logits) {
        steps.completeStep();
        size_t const count = logits.getShape().volume();
        require(static_cast<size_t>(host.getMemoryCapacity()) >= count * sizeof(float), "Host logit capacity");
        CUDA_CHECK(cudaMemcpyAsync(
            host.rawPointer(), logits.rawPointer(), count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return std::vector<float>(host.dataPointer<float>(), host.dataPointer<float>() + logits.getShape()[1]);
    };
    SequenceOptions options;
    options.maxOutputTokens = 12;
    bool passed = true;
    uint64_t id = 0;
    std::vector<std::pair<std::string, std::vector<int32_t>>> fixtures;
    auto lengths = extended
        ? std::vector<int32_t>{130, 131, 132, 191, 192, 193, 258, 259, 260, 319, 320, 321}
        : std::vector<int32_t>{1, 3, 4, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513, 1025, 2049, 6144};
    if (extended)
    {
        for (int32_t length = 129; length <= 193; ++length)
        {
            if (std::find(lengths.begin(), lengths.end(), length) == lengths.end())
            {
                lengths.push_back(length);
            }
        }
    }
    for (int32_t length : lengths)
    {
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < length; ++i)
        {
            prompt.push_back(words[i % words.size()]);
        }
        fixtures.emplace_back("policy_" + std::to_string(length), std::move(prompt));
    }
    if (extended)
    {
        std::string records;
        for (int32_t i = 0; i < 100; ++i)
        {
            records += "Record " + std::to_string(i) + ": warehouse " + std::to_string(i % 7) + " has "
                + std::to_string(17 * i + 3) + " blue items and " + std::to_string(11 * i + 5) + " red items.\n";
        }
        std::vector<std::string> texts{
            "Explain why the Moon changes shape during a month. Use plain language and distinguish phases from "
            "eclipses.",
            "Write a Python function that merges two sorted lists. Explain empty inputs and duplicate values.\n"
            "def merge(a, b):\n    # Preserve ordering and duplicates.\n    pass\n",
            "Return JSON with fields city and greeting for Chennai, 東京, and Zürich. Preserve Unicode text. "
            "The greeting in Tamil is வணக்கம். Do not invent population values.",
            records + "Which warehouse appears in record 73, and how many blue items does that record contain?"};
        for (size_t i = 0; i < texts.size(); ++i)
        {
            if (i < 3)
            {
                std::string const topic = texts[i];
                for (int32_t repeat = 0; repeat < 4; ++repeat)
                {
                    texts[i] += "\nAdditional requested detail " + std::to_string(repeat) + ": " + topic;
                }
            }
            LLMGenerationRequest::Request request;
            request.messages.push_back({"system", {{"text", "You are a helpful assistant. Answer directly."}}});
            request.messages.push_back({"user", {{"text", texts[i]}}});
            LLMGenerationRequest::FormattedRequest formatted;
            require(tokenizer.applyChatTemplate(request, formatted, true, true, false), "Chat formatting failed");
            auto tokens = tokenizer.encode(formatted.formattedCompleteRequest);
            require(!tokens.empty() && tokens.size() <= 6144, "Formatted fixture length");
            fixtures.emplace_back("chat_" + std::to_string(i) + "_" + std::to_string(tokens.size()), std::move(tokens));
        }
    }
    for (auto const& [label, prompt] : fixtures)
    {
        int32_t const length = static_cast<int32_t>(prompt.size());
        auto a = steps.acquire(++id, prompt, options);
        auto b = steps.acquire(++id, prompt, options);
        auto reference = finish(steps.beginPrefill(a, length));
        std::vector<float> actual;
        int32_t chunks = 0;
        while (steps.state(b).phase() == SequencePhase::kPrefill)
        {
            int32_t const before = steps.state(b).promptCursor();
            actual = finish(steps.beginPrefillChunk(b));
            int32_t const span = steps.state(b).promptCursor() - before;
            require(span > 0 && span <= 128 && (before == 0 || span >= 64), "Invalid bounded partition");
            require(steps.state(b).output().empty(), "Prefill emitted a completion");
            ++chunks;
        }
        std::cout << "PARTITION " << label << " chunks=" << chunks << std::endl;
        passed = compare(label, reference, actual, length <= 128) && passed;
        passed = observer.compareActive(label, length) && passed;
        int32_t const continuation = label.find("chat_") == 0 ? 8 : 4;
        for (int32_t token = 0; token < continuation; ++token)
        {
            int32_t const teacher = greedy(reference);
            steps.acceptToken(a, teacher);
            steps.acceptToken(b, teacher);
            reference = finish(steps.beginDecode({a, {}}, 1));
            actual = finish(steps.beginDecode({b, {}}, 1));
            passed = compare(label + "_teacher_" + std::to_string(token), reference, actual, length <= 128) && passed;
        }
        passed = observer.compareActive(label + "_after_teacher", length + continuation) && passed;
        steps.release(a);
        steps.release(b);
    }
    if (extended)
    {
        std::vector<int32_t> prompt(129);
        for (size_t i = 0; i < prompt.size(); ++i)
        {
            prompt[i] = words[i % words.size()];
        }
        auto a = steps.acquire(++id, prompt, options);
        auto b = steps.acquire(++id, prompt, options);
        auto reference = finish(steps.beginPrefill(a, 129));
        finish(steps.beginPrefill(b, 64));
        finish(steps.beginPrefill(b, 64));
        auto raw = finish(steps.beginPrefill(b, 1));
        bool rawQuality = compare("raw_singleton_prefill", reference, raw);
        rawQuality = observer.compareActive("raw_singleton_prefill", 129) && rawQuality;
        steps.acceptToken(a, greedy(reference));
        steps.acceptToken(b, greedy(reference));
        reference = finish(steps.beginDecode({a, {}}, 1));
        raw = finish(steps.beginDecode({b, {}}, 1));
        rawQuality = compare("raw_singleton_teacher", reference, raw) && rawQuality;
        rawQuality = observer.compareActive("raw_singleton_teacher", 130) && rawQuality;
        std::cout << "RAW_TAIL_DIAGNOSTIC quality_passed=" << rawQuality << std::endl;
        steps.release(a);
        steps.release(b);

        std::vector<int32_t> shortPrompt(prompt.begin(), prompt.begin() + 65);
        std::vector<int32_t> longPrompt(513);
        for (size_t i = 0; i < longPrompt.size(); ++i)
        {
            longPrompt[i] = words[(i + 3) % words.size()];
        }
        auto baseline = steps.acquire(++id, shortPrompt, options);
        std::vector<std::vector<float>> referenceA{finish(steps.beginPrefill(baseline, 65))};
        for (int32_t i = 0; i < 5; ++i)
        {
            steps.acceptToken(baseline, greedy(referenceA.back()));
            referenceA.push_back(finish(steps.beginDecode({baseline, {}}, 1)));
        }
        steps.release(baseline);
        baseline = steps.acquire(++id, longPrompt, options);
        auto const referenceB = finish(steps.beginPrefill(baseline, 513));
        steps.acceptToken(baseline, greedy(referenceB));
        auto const nextB = finish(steps.beginDecode({baseline, {}}, 1));
        steps.release(baseline);

        auto incompatible = steps.acquire(++id, shortPrompt, options);
        finish(steps.beginPrefill(incompatible, 64));
        bool rejectedPolicy = false;
        try
        {
            steps.beginPrefillChunk(incompatible);
        }
        catch (std::logic_error const&)
        {
            rejectedPolicy = true;
        }
        require(rejectedPolicy && steps.healthy(), "Policy accepted manually created singleton tail");
        steps.release(incompatible);
        std::cout << "REJECT mixed_manual_partition passed=1" << std::endl;
        for (bool reversed : {false, true})
        {
            if (reversed)
            {
                b = steps.acquire(++id, longPrompt, options);
                a = steps.acquire(++id, shortPrompt, options);
            }
            else
            {
                a = steps.acquire(++id, shortPrompt, options);
                b = steps.acquire(++id, longPrompt, options);
            }
            std::cout << "INTERLEAVE decoding_slot=" << a.slot << " prefill_slot=" << b.slot << std::endl;
            auto const scratch = observer.samplingBytes();
            passed
                = compare("interleaved_A_prefill", referenceA[0], finish(steps.beginPrefillChunk(a)), true) && passed;
            steps.acceptToken(a, greedy(referenceA[0]));
            int32_t index = 0;
            while (steps.state(b).phase() == SequencePhase::kPrefill)
            {
                observer.observeEndpoint(a.slot, steps.state(a).committedTokens());
                observer.snapshot(a.slot);
                auto const chunk = finish(steps.beginPrefillChunk(b));
                observer.verifySnapshot();
                observer.observeEndpoint(b.slot, steps.state(b).committedTokens());
                observer.snapshot(b.slot);
                passed = compare("interleaved_A_decode_" + std::to_string(index), referenceA[index + 1],
                             finish(steps.beginDecode({a, {}}, 1)), true)
                    && passed;
                observer.verifySnapshot();
                ++index;
                if (steps.state(b).phase() == SequencePhase::kPrefill)
                {
                    steps.acceptToken(a, greedy(referenceA[index]));
                    bool rejected = false;
                    try
                    {
                        steps.acceptToken(b, 0);
                    }
                    catch (std::logic_error const&)
                    {
                        rejected = true;
                    }
                    require(rejected, "Sample accepted during partial prefill");
                }
                else
                {
                    passed = compare("interleaved_B_final", referenceB, chunk, true) && passed;
                }
            }
            require(index == 5 && steps.state(b).output().empty(), "Interleaved prompt accounting");
            steps.acceptToken(b, greedy(referenceB));
            passed = compare("interleaved_B_teacher", nextB, finish(steps.beginDecode({b, {}}, 1)), true) && passed;
            require(observer.samplingBytes() == scratch, "Chunk forward touched sampler scratch");
            require(steps.state(b).committedTokens() == 514 && steps.state(b).output().size() == 1,
                "Final sample/cache accounting");
            steps.release(a);
            steps.release(b);
        }
        std::cout << "P3_INTERLEAVING_GATE passed=" << passed << std::endl;
    }
    observer.memory("p3_complete");
    std::cout << "P3_QUALITY_GATE passed=" << passed << std::endl;
    return passed;
}
bool runSchedulerTests(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream)
{
    ContinuousBatchingProbe observer(runtime, stream);
    auto const words = tokenizer.encode("A red fox crosses a blue river. One two three four. ");
    auto prompt = [&](int32_t length) {
        std::vector<int32_t> tokens;
        for (int32_t i = 0; i < length; ++i)
        {
            tokens.push_back(words.at(i % words.size()));
        }
        return tokens;
    };
    auto const pa = prompt(65), pb = prompt(1025), pc = prompt(129);
    std::vector<SchedulerEvent> events;
    events.reserve(1024);
    bool armed = false, submitted = false;
    ContinuousScheduler* owner = nullptr;
    SchedulerTicket b, c;
    auto backend = std::make_unique<GreedySchedulerBackend>(runtime, stream, observer.vocabularySize());
    ContinuousScheduler scheduler(std::move(backend), 8, 65536, [&](SchedulerEvent const& event) {
        if (event.kind == SchedulerEvent::Kind::kIdle)
        {
            return;
        }
        events.push_back(event);
        if (armed && !submitted && event.kind == SchedulerEvent::Kind::kDecode)
        {
            submitted = true;
            b = owner->submit(pb, 12);
            c = owner->submit(pc, 12);
        }
    });
    owner = &scheduler;
    auto result = [&](SchedulerTicket const& ticket) {
        require(ticket.result().wait_for(std::chrono::seconds(60)) == std::future_status::ready, "Stranded ticket");
        auto value = ticket.result().get();
        require(value.status == SchedulerStatus::kCompleted, "Scheduler request failed");
        return value.tokens;
    };
    auto const ra = result(scheduler.submit(pa, 6));
    auto const rb = result(scheduler.submit(pb, 12));
    auto const rc = result(scheduler.submit(pc, 12));
    // Configure the diagnostic through submission's mutex happens-before boundary.
    armed = true;
    auto a = scheduler.submit(pa, 6);
    auto const actualA = result(a);
    auto const actualB = result(b);
    auto const actualC = result(c);
    scheduler.close();
    bool const exact = actualA == ra && actualB == rb && actualC == rc;
    SequenceHandle ah{}, ch{};
    bool aReleased = false, reused = false, bContinued = false, pair = false, staggered = false;
    for (auto const& e : events)
    {
        std::cout << "P4_EVENT kind=" << static_cast<int>(e.kind) << " request=" << e.request
                  << " slot=" << e.handle.slot << " generation=" << e.handle.generation << " partner=" << e.partner
                  << " cursor=" << e.cursor << " us=" << e.microseconds << std::endl;
        if (e.request == a.id() && e.kind == SchedulerEvent::Kind::kAdmit)
        {
            ah = e.handle;
        }
        if (e.request == a.id() && e.kind == SchedulerEvent::Kind::kDecode)
        {
            staggered = true;
        }
        if (e.request == b.id() && e.kind == SchedulerEvent::Kind::kAdmit)
        {
            require(staggered, "B admitted before A decode");
        }
        if (e.request == a.id() && e.kind == SchedulerEvent::Kind::kRelease)
        {
            aReleased = true;
        }
        if (e.request == c.id() && e.kind == SchedulerEvent::Kind::kAdmit)
        {
            ch = e.handle;
            reused = aReleased && ah.slot == ch.slot && ch.generation > ah.generation;
        }
        if (reused && e.request == b.id() && e.kind == SchedulerEvent::Kind::kPrefill)
        {
            bContinued = true;
        }
        if (e.partner && (e.request == b.id() || e.request == c.id()))
        {
            pair = true;
        }
    }
    bool const passed = exact && reused && bContinued && pair && scheduler.healthy();
    std::cout << "P4_SCHEDULER_GATE passed=" << passed << " exact_outputs=" << exact << " reused=" << reused
              << " b_continued=" << bContinued << " paired_decode=" << pair << std::endl;
    return passed;
}
bool runPolicyTests(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream)
{
    ContinuousBatchingProbe observer(runtime, stream);
    auto const words = tokenizer.encode("A fox crosses a river. Explain the colors and count to five. ");
    auto prompt = [&](int32_t length) {
        std::vector<int32_t> tokens;
        tokens.reserve(length);
        for (int32_t i = 0; i < length; ++i)
        {
            tokens.push_back(words.at(i % words.size()));
        }
        return tokens;
    };
    auto const pa = prompt(65), pb = prompt(513), pc = prompt(129);
    SchedulerRequestOptions oa, ob, oc;
    oa.generation.maxOutputTokens = 6;
    oa.generation.temperature = 0.7F;
    oa.generation.topK = 20;
    oa.generation.topP = 0.9F;
    oa.generation.seed = 123;
    oa.generation.numLogprobs = 5;
    oa.generation.ignoreEos = true;
    ob = oa;
    ob.generation.maxOutputTokens = 12;
    ob.generation.temperature = 1.2F;
    ob.generation.topK = 40;
    ob.generation.seed = 456;
    ob.generation.numLogprobs = 0;
    oc = oa;
    oc.generation.maxOutputTokens = 12;
    oc.generation.temperature = 0;
    oc.generation.topK = 1;
    oc.generation.numLogprobs = 2;
    std::atomic<bool> armed{false};
    bool submitted = false;
    ContinuousScheduler* owner = nullptr;
    SchedulerTicket b, c;
    bool paired = false;
    std::atomic<bool> cancelMode{false};
    bool cancelSubmitted = false;
    SchedulerTicket nativeB, nativeC;
    auto get = [&](SchedulerTicket const& ticket) {
        require(
            ticket.result().wait_for(std::chrono::seconds(60)) == std::future_status::ready, "Policy ticket stranded");
        return ticket.result().get();
    };
    {
        ContinuousScheduler scheduler(
            std::make_unique<SamplingSchedulerBackend>(runtime, stream, observer.vocabularySize(), &tokenizer), 8,
            262144, [&](SchedulerEvent const& e) {
                if (cancelMode.load())
                {
                    if (!cancelSubmitted && e.kind == SchedulerEvent::Kind::kDecode)
                    {
                        cancelSubmitted = true;
                        nativeB = owner->submit(pb, ob);
                        nativeC = owner->submit(pc, oc);
                        nativeC.cancel();
                    }
                    if (cancelSubmitted && e.request == nativeB.id() && e.kind == SchedulerEvent::Kind::kPrefill)
                    {
                        nativeB.cancel();
                    }
                }
                if (armed.load() && e.kind == SchedulerEvent::Kind::kDecode)
                {
                    if (!submitted)
                    {
                        submitted = true;
                        b = owner->submit(pb, ob);
                        c = owner->submit(pc, oc);
                    }
                    if (e.partner)
                    {
                        paired = true;
                    }
                }
            });
        owner = &scheduler;
        auto const ra = get(scheduler.submit(pa, oa)), rb = get(scheduler.submit(pb, ob)),
                   rc = get(scheduler.submit(pc, oc));
        armed.store(true);
        auto const a = scheduler.submit(pa, oa);
        auto const aa = get(a), ab = get(b), ac = get(c);
        armed.store(false);
        require(aa.status == SchedulerStatus::kCompleted && ab.status == SchedulerStatus::kCompleted
                && ac.status == SchedulerStatus::kCompleted,
            "Mixed requests failed");
        require(aa.tokens == ra.tokens && ab.tokens == rb.tokens && ac.tokens == rc.tokens,
            "Seeded serial/concurrent mismatch");
        require(aa.text == ra.text && ab.text == rb.text && ac.text == rc.text, "Independent text mismatch");
        require(aa.tokens.size() == 6 && ab.tokens.size() == 12 && ac.tokens.size() == 12 && paired,
            "Mixed batch/limits missing");
        require(
            aa.randomCounter == 6 && ab.randomCounter == 12 && ac.randomCounter == 0, "RNG counters crossed requests");
        require(aa.logprobs.size() == 6 && ab.logprobs.empty() && ac.logprobs.size() == 12
                && aa.logprobs.front().topCount == 5 && ac.logprobs.front().topCount == 2,
            "Logprob history missing");
        for (size_t i = 0; i < aa.logprobs.size(); ++i)
        {
            require(aa.logprobs[i].logprob == ra.logprobs[i].logprob, "Logprob baseline mismatch");
        }
        std::cout << "P5_MIXED_GATE passed=1 outputs=6,12,12 rng=6,12,0 paired=1 exact_seeded=1" << std::endl;
        auto stopOptions = oc;
        stopOptions.generation.ignoreEos = false;
        stopOptions.generation.numLogprobs = 50;
        int32_t const forced = tokenizer.encode("Z").at(0);
        auto const piece = tokenizer.idToPiece(forced, true);
        require(!piece.empty(), "Empty forced token piece");
        stopOptions.generation.logitBias[forced] = 100;
        stopOptions.generation.stopStrings = {piece + piece};
        stopOptions.streamRecords = 8;
        auto stopTicket = scheduler.submit(pa, stopOptions);
        auto stopped = get(stopTicket);
        require(stopped.status == SchedulerStatus::kCompleted && stopped.finish == SequenceFinish::kStop
                && stopped.tokens.size() == 2 && stopped.text.empty(),
            "Cross-token stop failed");
        std::string streamed;
        while (true)
        {
            auto next = stopTicket.read(std::chrono::milliseconds(0));
            if (next.update)
            {
                streamed += next.update->text;
            }
            if (next.closed)
            {
                break;
            }
        }
        require(streamed.empty(), "Stop prefix leaked to stream");
        auto eosOptions = oc;
        eosOptions.generation.ignoreEos = false;
        eosOptions.generation.logitBias[tokenizer.getEosId()] = 100;
        auto eos = get(scheduler.submit(pa, eosOptions));
        require(eos.finish == SequenceFinish::kEos && eos.tokens.size() == 1 && eos.text.empty(), "Primary EOS failed");
        auto slowOptions = oa;
        slowOptions.streamRecords = 1;
        slowOptions.streamBytes = 32;
        auto slow = scheduler.submit(pa, slowOptions);
        auto peer = scheduler.submit(pb, ob);
        require(get(slow).status == SchedulerStatus::kSlowConsumer, "Slow consumer not terminated");
        require(get(peer).tokens == rb.tokens, "Slow consumer corrupted partner");
        auto deadlineOptions = oa;
        deadlineOptions.queueDeadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
        require(
            get(scheduler.submit(pa, deadlineOptions)).status == SchedulerStatus::kDeadline, "Queue deadline failed");
        auto streamOptions = oa;
        streamOptions.streamRecords = 32;
        auto streamTicket = scheduler.submit(pa, streamOptions);
        auto streamResult = get(streamTicket);
        std::string streamText;
        size_t recordCount = 0;
        while (true)
        {
            auto next = streamTicket.read(std::chrono::milliseconds(0));
            if (next.update)
            {
                streamText += next.update->text;
                if (next.update->sample.token >= 0)
                {
                    ++recordCount;
                }
            }
            if (next.closed)
            {
                break;
            }
        }
        require(streamText == streamResult.text && recordCount == 6, "Streaming result differs from final result");
        cancelMode.store(true);
        auto cancellationLeader = scheduler.submit(pa, oa);
        auto leaderResult = get(cancellationLeader);
        require(leaderResult.tokens == ra.tokens, "Prefill cancellation changed partner output");
        require(
            get(nativeB).status == SchedulerStatus::kCancelled && get(nativeC).status == SchedulerStatus::kCancelled,
            "Native active/queued cancellation failed");
        cancelMode.store(false);
        nativeB.cancel();
        require(get(scheduler.submit(pc, oc)).tokens == rc.tokens, "Stale ticket affected reused slot");
        auto activeDeadline = oa;
        activeDeadline.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        require(get(scheduler.submit(prompt(2049), activeDeadline)).status == SchedulerStatus::kDeadline,
            "Native active deadline failed");
        require(get(scheduler.submit(pa, oa)).tokens == ra.tokens, "Deadline cleanup changed later output");
        std::cout
            << "P5_CANCEL_GATE passed=1 prefill_cancel=1 queued_cancel=1 stale_ticket=1 active_deadline=1 peer_exact=1"
            << std::endl;
        scheduler.close();
        require(scheduler.healthy(), "Policy tests poisoned healthy runtime");
        std::cout << "P5_STOP_STREAM_GATE passed=1 stop_tokens=2 eos_tokens=1 logprob_top=50 slow_peer_exact=1"
                  << std::endl;
    }
    // Inject a reported corrupting failure after real GPU work without deliberately damaging the device.
    class InjectedFailure : public SamplingSchedulerBackend
    {
    public:
        using SamplingSchedulerBackend::SamplingSchedulerBackend;
        void prefill(SequenceHandle handle) override
        {
            SamplingSchedulerBackend::prefill(handle);
            throw std::runtime_error("injected post-forward CUDA failure");
        }
    };
    {
        ContinuousScheduler failed(
            std::make_unique<InjectedFailure>(runtime, stream, observer.vocabularySize(), &tokenizer));
        auto ticket = failed.submit(pa, oa);
        require(get(ticket).status == SchedulerStatus::kFailed, "Fault left ticket live");
        failed.close();
        require(!failed.healthy(), "Fault failed to mark scheduler unready");
        bool rejected = false;
        try
        {
            failed.submit(pa, oa);
        }
        catch (std::exception const&)
        {
            rejected = true;
        }
        require(rejected, "Failed scheduler admitted request");
    }
    bool poisoned = false;
    try
    {
        SequenceStepRuntime forbidden(runtime, stream);
    }
    catch (std::exception const&)
    {
        poisoned = true;
    }
    require(poisoned, "Failed sampling released parent lease");
    std::cout << "P5_FAULT_GATE passed=1 injected_post_forward=1 parent_lease_poisoned=1" << std::endl;
    return true;
}
} // namespace rt
} // namespace trt_edgellm

int main(int argc, char** argv)
{
    if (argc != 3
        && (argc != 4
            || (std::string(argv[3]) != "--policies" && std::string(argv[3]) != "--scheduler"
                && std::string(argv[3]) != "--steps" && std::string(argv[3]) != "--graphs"
                && (std::string(argv[3]) != "--chunks" && std::string(argv[3]) != "--chunks-extra"))))
    {
        std::cerr << "Usage: continuous_batching_probe ENGINE_DIR CHECKPOINT_DIR "
                     "[--steps|--graphs|--chunks|--chunks-extra|--scheduler|--policies]\n";
        return 2;
    }
    cudaStream_t stream{};
    try
    {
        auto plugin = trt_edgellm::loadEdgellmPluginLib();
        trt_edgellm::rt::require(plugin != nullptr, "Plugin loading failed");
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bool passed = false;
        {
            trt_edgellm::tokenizer::Tokenizer tokenizer;
            trt_edgellm::rt::require(tokenizer.loadFromHF(argv[1], false), "Tokenizer loading failed");
            trt_edgellm::rt::LLMRankRuntime runtime(argv[1], "", {}, std::nullopt, stream,
                trt_edgellm::rt::ParallelMapping{}, tokenizer, trt_edgellm::rt::ContextCacheConfig{}, argv[2], "");
            passed = argc == 4 && std::string(argv[3]) == "--policies"
                ? trt_edgellm::rt::runPolicyTests(runtime, tokenizer, stream)
                : argc == 4 && std::string(argv[3]) == "--scheduler"
                ? trt_edgellm::rt::runSchedulerTests(runtime, tokenizer, stream)
                : argc == 4
                ? (std::string(argv[3]).find("--chunks") == 0 ? trt_edgellm::rt::runChunkTests(runtime, tokenizer,
                                                                    stream, std::string(argv[3]) == "--chunks-extra")
                                                              : trt_edgellm::rt::runStepTests(runtime, tokenizer,
                                                                    stream, std::string(argv[3]) == "--graphs"))
                : trt_edgellm::rt::runProbe(runtime, tokenizer, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "PROBE_RESULT passed=" << passed << std::endl;
        return passed ? 0 : 1;
    }
    catch (std::exception const& error)
    {
        std::cerr << "PROBE_ERROR " << error.what() << std::endl;
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
        return 1;
    }
}
