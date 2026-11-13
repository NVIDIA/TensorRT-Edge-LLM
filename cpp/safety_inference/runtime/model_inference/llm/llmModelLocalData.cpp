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

#include "llmModelLocalData.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace trt_edgellm::safety::llm
{
namespace
{

constexpr auto kMEMORY_COUNT = static_cast<RuntimeResourceKey>(ModelMemory::kCOUNT);
constexpr RuntimeResourceKey kPREFILL_STREAM{kMEMORY_COUNT};
constexpr RuntimeResourceKey kDECODE_STREAM{kMEMORY_COUNT + 1};
constexpr RuntimeResourceKey kFIRST_BINDING{kMEMORY_COUNT + 2};

SafetyStatus invalid() noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "LLM resource metadata is incompatible");
}

RuntimeTensorProfile const* profile(RuntimeTensorDesc const& desc, int32_t id) noexcept
{
    auto const it = std::find_if(
        desc.profiles.begin(), desc.profiles.end(), [id](auto const& value) { return value.profileId == id; });
    return it == desc.profiles.end() ? nullptr : &*it;
}

bool sameShape(nvinfer1::Dims const& a, nvinfer1::Dims const& b) noexcept
{
    return a.nbDims == b.nbDims && std::equal(a.d, a.d + a.nbDims, b.d);
}

bool matches(RuntimeTensorView const& view, RuntimeTensorDesc const& expected) noexcept
{
    if (view.desc == nullptr || !validateTensorDesc(*view.desc).okStatus() || view.desc->name != expected.name
        || view.desc->dataType != expected.dataType || view.desc->ioMode != expected.ioMode
        || view.desc->profiles.size() != expected.profiles.size() || view.memory.data == nullptr
        || view.memory.kind != RuntimeMemoryKind::kDEVICE)
    {
        return false;
    }
    for (auto const& entry : expected.profiles)
    {
        auto const* actual = profile(*view.desc, entry.profileId);
        if (actual == nullptr || !sameShape(actual->shape, entry.shape) || actual->sizeInBytes != entry.sizeInBytes
            || view.memory.sizeInBytes < entry.sizeInBytes)
        {
            return false;
        }
    }
    return true;
}

} // namespace

ModelLocalDataFactory::ModelLocalDataFactory(
    ModelResourceSpec spec, StageResourceSpec prefill, StageResourceSpec decode)
    : mSpec(spec)
    , mPrefill(std::move(prefill))
    , mDecode(std::move(decode))
{
}

SafetyStatus ModelLocalDataFactory::validate(Capacities& output) const
{
    output = {};
    constexpr std::size_t kFIXED_BINDINGS{6};
    if (mSpec.owner == 0 || mSpec.vocabSize <= 0 || mSpec.hiddenSize <= 0 || mSpec.layerCount <= 0
        || mSpec.kvHeadCount <= 0 || mSpec.headDim <= 0 || mSpec.padTokenId < 0 || mSpec.padTokenId >= mSpec.vocabSize
        || mPrefill.stage.empty() || mDecode.stage.empty() || mPrefill.stage == mDecode.stage || mPrefill.profileId < 0
        || mDecode.profileId < 0 || mPrefill.profileId == mDecode.profileId)
    {
        return invalid();
    }
    auto const bindingCount = kFIXED_BINDINGS + 2 * static_cast<std::size_t>(mSpec.layerCount);
    if (bindingCount > (std::numeric_limits<RuntimeResourceKey>::max() - kFIRST_BINDING) / 2
        || mPrefill.bindings.size() != bindingCount || mDecode.bindings.size() != bindingCount)
    {
        return invalid();
    }
    int64_t kvCapacity{-1};
    int64_t ropeWidth{-1};
    int64_t ropeLength{std::numeric_limits<int32_t>::max()};
    for (auto const* stage : {&mPrefill, &mDecode})
    {
        for (auto const& desc : stage->bindings)
        {
            if (!validateTensorDesc(desc).okStatus()
                || std::count_if(stage->bindings.begin(), stage->bindings.end(),
                       [&](auto const& other) { return other.name == desc.name; })
                    != 1)
            {
                return invalid();
            }
        }
        auto binding = [&](std::string const& name, nvinfer1::DataType type, nvinfer1::TensorIOMode mode,
                           int32_t rank) -> RuntimeTensorProfile const* {
            auto const it = std::find_if(
                stage->bindings.begin(), stage->bindings.end(), [&](auto const& desc) { return desc.name == name; });
            if (it == stage->bindings.end() || it->dataType != type || it->ioMode != mode)
            {
                return nullptr;
            }
            auto const* selected = profile(*it, stage->profileId);
            return selected != nullptr && selected->shape.nbDims == rank && selected->shape.d[0] == 1 ? selected
                                                                                                      : nullptr;
        };
        using Type = nvinfer1::DataType;
        using Mode = nvinfer1::TensorIOMode;
        auto const* embeds = binding("inputs_embeds", Type::kHALF, Mode::kINPUT, 3);
        auto const* rope = binding("rope_rotary_cos_sin", Type::kFLOAT, Mode::kINPUT, 3);
        auto const* logits = binding("logits", Type::kFLOAT, Mode::kOUTPUT, 3);
        auto const* last = binding("last_token_ids", Type::kINT64, Mode::kINPUT, 2);
        if (embeds == nullptr || rope == nullptr || logits == nullptr || last == nullptr
            || binding("context_lengths", Type::kINT32, Mode::kINPUT, 1) == nullptr
            || binding("kvcache_start_index", Type::kINT32, Mode::kINPUT, 1) == nullptr
            || embeds->shape.d[2] != mSpec.hiddenSize || logits->shape.d[1] != 1
            || logits->shape.d[2] != mSpec.vocabSize || last->shape.d[1] != 1
            || embeds->shape.d[1] > std::numeric_limits<int32_t>::max()
            || (stage == &mDecode && embeds->shape.d[1] != 1) || rope->shape.d[2] > mSpec.headDim
            || rope->shape.d[2] % 2 != 0 || (ropeWidth != -1 && ropeWidth != rope->shape.d[2]))
        {
            return invalid();
        }
        ropeWidth = rope->shape.d[2];
        ropeLength = std::min(ropeLength, rope->shape.d[1]);
        output.tokens = std::max(output.tokens, static_cast<std::size_t>(embeds->shape.d[1]));
        output.logitsBytes = std::max(output.logitsBytes, logits->sizeInBytes);
        output.ropeBytes = std::max(output.ropeBytes, rope->sizeInBytes);
        for (int32_t layer = 0; layer < mSpec.layerCount; ++layer)
        {
            auto const* past = binding("past_key_values_" + std::to_string(layer), Type::kHALF, Mode::kINPUT, 5);
            auto const* present = binding("present_key_values_" + std::to_string(layer), Type::kHALF, Mode::kOUTPUT, 5);
            if (past == nullptr || present == nullptr || !sameShape(past->shape, present->shape)
                || past->shape.d[1] != 2 || past->shape.d[2] != mSpec.kvHeadCount || past->shape.d[4] != mSpec.headDim
                || past->shape.d[3] > std::numeric_limits<int32_t>::max()
                || (kvCapacity != -1 && kvCapacity != past->shape.d[3]))
            {
                return invalid();
            }
            kvCapacity = past->shape.d[3];
        }
    }
    output.contextLength = static_cast<int32_t>(std::min(kvCapacity, ropeLength));
    if (output.tokens > static_cast<std::size_t>(output.contextLength)
        || static_cast<std::size_t>(mSpec.vocabSize)
            > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 2 / mSpec.hiddenSize)
    {
        return invalid();
    }
    return SafetyStatus::ok();
}

SafetyStatus ModelLocalDataFactory::declareResources(ResourceDeclaration& output) const noexcept
{
    output = {};
    try
    {
        Capacities capacities{};
        auto status = validate(capacities);
        if (!status.okStatus())
        {
            return status;
        }
        ResourceDeclaration declaration{};
        declaration.owner = mSpec.owner;
        using Kind = RuntimeMemoryKind;
        auto add = [&](ModelMemory key, std::size_t bytes, Kind kind) {
            declaration.memory.push_back({static_cast<RuntimeResourceKey>(key), bytes, kind});
        };
        add(ModelMemory::kTOKEN_IDS, capacities.tokens * sizeof(int32_t), Kind::kPINNED);
        add(ModelMemory::kEMBEDDING_TOKEN_IDS, capacities.tokens * sizeof(int32_t), Kind::kPINNED);
        add(ModelMemory::kCONTEXT_LENGTH, sizeof(int32_t), Kind::kPINNED);
        add(ModelMemory::kKV_START, sizeof(int32_t), Kind::kPINNED);
        add(ModelMemory::kLAST_TOKEN, sizeof(int64_t), Kind::kPINNED);
        add(ModelMemory::kLOGITS, capacities.logitsBytes, Kind::kPINNED);
        add(ModelMemory::kEMBEDDING_TABLE, static_cast<std::size_t>(mSpec.vocabSize) * mSpec.hiddenSize * 2,
            Kind::kDEVICE);
        add(ModelMemory::kROPE_CACHE, capacities.ropeBytes, Kind::kDEVICE);
        add(ModelMemory::kDEVICE_TOKEN_IDS, capacities.tokens * sizeof(int32_t), Kind::kDEVICE);
        declaration.borrowed.push_back(
            {kPREFILL_STREAM, RuntimeBorrowedResourceKind::kCUDA_STREAM, mPrefill.stage, 0, {}});
        declaration.borrowed.push_back(
            {kDECODE_STREAM, RuntimeBorrowedResourceKind::kCUDA_STREAM, mDecode.stage, 0, {}});
        RuntimeResourceKey key{kFIRST_BINDING};
        for (auto const* stage : {&mPrefill, &mDecode})
        {
            for (auto const& desc : stage->bindings)
            {
                declaration.borrowed.push_back(
                    {key++, RuntimeBorrowedResourceKind::kIO_BINDING, stage->stage, 0, desc.name});
            }
        }
        output = std::move(declaration);
        return SafetyStatus::ok();
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "LLM resource declaration allocation failed");
    }
    catch (std::length_error const&)
    {
        return invalid();
    }
    catch (...)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kFailedExecution, "Unexpected LLM resource declaration failure");
    }
}

SafetyStatus ModelLocalDataFactory::create(
    PreparedResources const& resources, std::unique_ptr<SafetyModelLocalData>& output) const noexcept
{
    if (output != nullptr)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Model-local output already exists");
    }
    auto const initialization = resources.validateInitialization(mSpec.owner);
    if (!initialization.okStatus())
    {
        return initialization;
    }
    try
    {
        ResourceDeclaration declaration{};
        auto status = declareResources(declaration);
        if (!status.okStatus())
        {
            return status;
        }
        Capacities capacities{};
        status = validate(capacities);
        if (!status.okStatus())
        {
            return status;
        }
        auto data = std::make_unique<ModelLocalData>();
        data->spec = mSpec;
        data->maxPrefillTokens = static_cast<int32_t>(capacities.tokens);
        data->maxContextLength = capacities.contextLength;
        for (auto const& request : declaration.memory)
        {
            auto& view = data->storage[request.key];
            status = resources.memory(request.key, view);
            if (!status.okStatus())
            {
                return status;
            }
            if (view.kind != request.kind || view.sizeInBytes < request.sizeInBytes)
            {
                return invalid();
            }
        }
        RuntimeResourceKey key{kFIRST_BINDING};
        for (auto const* stage : {&mPrefill, &mDecode})
        {
            auto& prepared = stage == &mPrefill ? data->prefill : data->decode;
            prepared.stage = stage->stage;
            prepared.profileId = stage->profileId;
            status = resources.stream(stage == &mPrefill ? kPREFILL_STREAM : kDECODE_STREAM, prepared.stream);
            if (!status.okStatus())
            {
                return status;
            }
            for (auto const& desc : stage->bindings)
            {
                RuntimeTensorView const* view = nullptr;
                status = resources.tensor(key++, view);
                if (!status.okStatus())
                {
                    return status;
                }
                if (!matches(*view, desc))
                {
                    return invalid();
                }
                prepared.bindings.push_back(view);
            }
        }
        // Metadata validation completes before touching any prepared payload.
        for (auto kind : {ModelMemory::kTOKEN_IDS, ModelMemory::kEMBEDDING_TOKEN_IDS})
        {
            auto const& buffer = data->storage[static_cast<std::size_t>(kind)];
            std::fill_n(static_cast<int32_t*>(buffer.data), capacities.tokens, mSpec.padTokenId);
        }
        *static_cast<int32_t*>(data->storage[static_cast<std::size_t>(ModelMemory::kCONTEXT_LENGTH)].data) = 0;
        *static_cast<int32_t*>(data->storage[static_cast<std::size_t>(ModelMemory::kKV_START)].data) = 0;
        *static_cast<int64_t*>(data->storage[static_cast<std::size_t>(ModelMemory::kLAST_TOKEN)].data) = 0;
        auto const& logits = data->storage[static_cast<std::size_t>(ModelMemory::kLOGITS)];
        std::fill_n(static_cast<float*>(logits.data), capacities.logitsBytes / sizeof(float), 0.0F);
        output = std::move(data);
        return SafetyStatus::ok();
    }
    catch (std::bad_alloc const&)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "LLM model-local metadata allocation failed");
    }
    catch (std::length_error const&)
    {
        return invalid();
    }
    catch (...)
    {
        return SafetyStatus::fatal(
            SafetyStatusCode::kFailedExecution, "Unexpected LLM model-local preparation failure");
    }
}

} // namespace trt_edgellm::safety::llm
