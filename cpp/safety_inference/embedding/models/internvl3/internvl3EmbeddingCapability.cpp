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

#include "internvl3EmbeddingCapability.h"

#include "safety_inference/embedding/fp16Embedding.h"
#include "safety_inference/embedding/ropeCache.h"
#include "safety_inference/runtime/safetyStatusUtils.h"
#include "safety_inference/trt_runner/lmStageContext.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace trt_edgellm::safety
{
namespace
{
constexpr RuntimeResourceKey kUPLOAD{300};
constexpr RuntimeResourceKey kCOMMON_STREAM{301};
constexpr RuntimeResourceKey kVISUAL_OUTPUT{302};
constexpr RuntimeResourceKey kMODEL_DATA{303};
constexpr RuntimeResourceKey kINITIALIZATION_EVENT{304};
constexpr std::size_t kUPLOAD_BYTES{1024U * 1024U};

RuntimeTensorView const* findBinding(llm::StageResources const& stage, std::string_view name) noexcept
{
    auto const found = std::find_if(stage.bindings.begin(), stage.bindings.end(), [name](auto const* binding) {
        return binding != nullptr && binding->desc != nullptr && binding->desc->name == name;
    });
    return found == stage.bindings.end() ? nullptr : *found;
}

bool validHalfBinding(RuntimeTensorView const* view, nvinfer1::TensorIOMode mode, int32_t hiddenSize) noexcept
{
    if (view == nullptr || view->desc == nullptr || view->memory.data == nullptr
        || view->memory.kind != RuntimeMemoryKind::kDEVICE || view->desc->dataType != nvinfer1::DataType::kHALF
        || view->desc->ioMode != mode || view->desc->profiles.size() != 1)
    {
        return false;
    }
    auto const& shape = view->desc->profiles.front().shape;
    return shape.nbDims >= 2 && shape.d[shape.nbDims - 1] == hiddenSize;
}

struct SafetensorView
{
    std::uint64_t dataOffset{0};
    std::uint64_t tensorOffset{0};
    std::uint64_t tensorBytes{0};
};

SafetyStatus parseEmbeddingHeader(std::ifstream& input, ProcessingLimits const& limits, SafetensorView& output) noexcept
{
    std::uint64_t headerBytes{0};
    input.read(reinterpret_cast<char*>(&headerBytes), sizeof(headerBytes));
    if (!input || headerBytes == 0 || headerBytes > 16U * 1024U * 1024U)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid embedding safetensors header");
    }
    try
    {
        std::string header(static_cast<std::size_t>(headerBytes), '\0');
        input.read(header.data(), static_cast<std::streamsize>(header.size()));
        auto const document = nlohmann::json::parse(header);
        auto const& tensor = document.at("embedding");
        auto const shape = tensor.at("shape").get<std::vector<std::uint64_t>>();
        auto const offsets = tensor.at("data_offsets").get<std::vector<std::uint64_t>>();
        std::uint64_t const expected = static_cast<std::uint64_t>(limits.vocabularySize) * limits.hiddenSize * 2U;
        if (tensor.at("dtype") != "F16"
            || shape
                != std::vector<std::uint64_t>{static_cast<std::uint64_t>(limits.vocabularySize),
                    static_cast<std::uint64_t>(limits.hiddenSize)}
            || offsets.size() != 2 || offsets[1] < offsets[0] || offsets[1] - offsets[0] != expected)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Embedding tensor contract mismatch");
        }
        output = {sizeof(headerBytes) + headerBytes, offsets[0], expected};
        return SafetyStatus::ok();
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Cannot parse embedding safetensors metadata");
    }
}

SafetyStatus uploadEmbedding(std::filesystem::path const& path, ProcessingLimits const& limits,
    RuntimeMemoryView upload, RuntimeMemoryView destination, cudaStream_t stream, cudaEvent_t completion) noexcept
{
    std::ifstream input(path, std::ios::binary);
    SafetensorView tensor{};
    auto status = input ? parseEmbeddingHeader(input, limits, tensor)
                        : SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Cannot open embedding artifact");
    if (!status.okStatus() || tensor.tensorBytes > destination.sizeInBytes)
    {
        return status.okStatus()
            ? SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Embedding table storage is too small")
            : status;
    }
    input.seekg(static_cast<std::streamoff>(tensor.dataOffset + tensor.tensorOffset));
    for (std::size_t copied = 0; copied < tensor.tensorBytes;)
    {
        std::size_t const bytes = std::min(upload.sizeInBytes, static_cast<std::size_t>(tensor.tensorBytes) - copied);
        input.read(static_cast<char*>(upload.data), static_cast<std::streamsize>(bytes));
        auto cudaStatus = input ? cudaMemcpyAsync(static_cast<char*>(destination.data) + copied, upload.data, bytes,
                                      cudaMemcpyHostToDevice, stream)
                                : cudaErrorInvalidValue;
        if (cudaStatus == cudaSuccess)
        {
            cudaStatus = cudaEventRecord(completion, stream);
        }
        if (cudaStatus == cudaSuccess)
        {
            cudaStatus = cudaEventSynchronize(completion);
        }
        if (cudaStatus != cudaSuccess)
        {
            return SafetyStatus::fatal(SafetyStatusCode::kCudaError, "Embedding upload failed");
        }
        copied += bytes;
    }
    return SafetyStatus::ok();
}

SafetyStatus copyDevice(RuntimeTensorView const* destination, RuntimeMemoryView source, cudaStream_t stream) noexcept
{
    if (destination == nullptr || destination->memory.data == nullptr || destination->desc == nullptr
        || destination->desc->profiles.size() != 1
        || destination->desc->profiles.front().sizeInBytes > source.sizeInBytes)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid model-local device copy");
    }
    auto const bytes = destination->desc->profiles.front().sizeInBytes;
    return fromCudaError(
        cudaMemcpyAsync(destination->memory.data, source.data, bytes, cudaMemcpyDeviceToDevice, stream),
        "Model-local device copy");
}
} // namespace

SafetyStatus InternVL3EmbeddingCapability::declareResources(
    EmbeddingArtifactSpec const& artifact, ProcessingLimits const& limits, ResourceDeclaration& declaration) noexcept
{
    if (mState != CapabilityPreparationState::kEMPTY || artifact.path.empty() || limits.vocabularySize <= 0
        || limits.hiddenSize <= 0 || limits.maxPromptTokens == 0)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid embedding declaration");
    }
    try
    {
        declaration.memory.push_back({kUPLOAD, kUPLOAD_BYTES, RuntimeMemoryKind::kPINNED});
        declaration.borrowed.push_back({kCOMMON_STREAM, RuntimeBorrowedResourceKind::kCUDA_STREAM, "prefill", 0, {}});
        declaration.borrowed.push_back(
            {kVISUAL_OUTPUT, RuntimeBorrowedResourceKind::kIO_BINDING, "visual_encode", 0, "output"});
        declaration.borrowed.push_back(
            {kMODEL_DATA, RuntimeBorrowedResourceKind::kMODEL_LOCAL_DATA, "prefill", 0, "llm"});
        declaration.events.push_back(kINITIALIZATION_EVENT);
        mArtifact = artifact;
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot declare embedding resources");
    }
    mLimits = limits;
    mState = CapabilityPreparationState::kDECLARED;
    return SafetyStatus::ok();
}

SafetyStatus InternVL3EmbeddingCapability::initialize(EmbeddingArtifactSpec const&, PreparedResources const& prepared,
    LMStageContext const& visual, LMStageContext const& prefill, LMStageContext const& decode) noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED || visual.mainStream == nullptr
        || visual.mainStream != prefill.mainStream || prefill.mainStream != decode.mainStream)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Embedding contexts do not share a stream");
    }
    SafetyModelLocalData* model{nullptr};
    auto status = prepared.memory(kUPLOAD, mUpload);
    if (status.okStatus())
    {
        status = prepared.stream(kCOMMON_STREAM, mStream);
    }
    if (status.okStatus())
    {
        status = prepared.tensor(kVISUAL_OUTPUT, mVisualOutput);
    }
    if (status.okStatus())
    {
        status = prepared.modelLocalData(kMODEL_DATA, model);
    }
    if (status.okStatus())
    {
        status = prepared.event(kINITIALIZATION_EVENT, mInitializationCompletion);
    }
    if (!status.okStatus())
    {
        return status;
    }
    mData = static_cast<llm::ModelLocalData*>(model);
    if (mStream != prefill.mainStream || mData == nullptr || mData->spec.vocabSize != mLimits.vocabularySize
        || mData->spec.hiddenSize != mLimits.hiddenSize
        || !validHalfBinding(mVisualOutput, nvinfer1::TensorIOMode::kOUTPUT, mLimits.hiddenSize)
        || !validHalfBinding(
            findBinding(mData->prefill, "inputs_embeds"), nvinfer1::TensorIOMode::kINPUT, mLimits.hiddenSize)
        || !validHalfBinding(
            findBinding(mData->decode, "inputs_embeds"), nvinfer1::TensorIOMode::kINPUT, mLimits.hiddenSize))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Embedding bindings are incompatible");
    }
    mVisualContext = &visual;
    mPrefillContext = &prefill;
    mDecodeContext = &decode;
    mState = CapabilityPreparationState::kREADY;
    return SafetyStatus::ok();
}

SafetyStatus InternVL3EmbeddingCapability::initializeModelData(llm::ModelLocalData& data) noexcept
{
    if (mState != CapabilityPreparationState::kREADY || &data != mData)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidState, "Embedding model data is not bound");
    }
    auto status = uploadEmbedding(mArtifact.path, mLimits, mUpload,
        data.storage[static_cast<std::size_t>(llm::ModelMemory::kEMBEDDING_TABLE)], mStream, mInitializationCompletion);
    auto const* rope = findBinding(data.prefill, "rope_rotary_cos_sin");
    if (status.okStatus() && (rope == nullptr || rope->desc->profiles.size() != 1))
    {
        status = SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Missing RoPE binding");
    }
    if (status.okStatus())
    {
        auto const& shape = rope->desc->profiles.front().shape;
        if (shape.nbDims <= 2 || shape.d[2] > std::numeric_limits<int32_t>::max())
        {
            status = SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid RoPE binding shape");
        }
        else
        {
            RopeCacheSpec const spec{
                data.maxContextLength, static_cast<int32_t>(shape.d[2]), mLimits.ropeTheta, mLimits.ropeScale};
            status = initializeRopeCache(spec, mUpload,
                data.storage[static_cast<std::size_t>(llm::ModelMemory::kROPE_CACHE)], mStream,
                mInitializationCompletion);
        }
    }
    if (status.okStatus())
    {
        status = copyDevice(findBinding(data.prefill, "rope_rotary_cos_sin"),
            data.storage[static_cast<std::size_t>(llm::ModelMemory::kROPE_CACHE)], mStream);
    }
    if (status.okStatus())
    {
        status = copyDevice(findBinding(data.decode, "rope_rotary_cos_sin"),
            data.storage[static_cast<std::size_t>(llm::ModelMemory::kROPE_CACHE)], mStream);
    }
    if (!status.okStatus())
    {
        return status;
    }
    auto cudaStatus = cudaEventRecord(mInitializationCompletion, mStream);
    if (cudaStatus == cudaSuccess)
    {
        cudaStatus = cudaEventSynchronize(mInitializationCompletion);
    }
    return cudaStatus == cudaSuccess ? SafetyStatus::ok() : fromCudaError(cudaStatus, "Embedding initialization");
}

SafetyStatus InternVL3EmbeddingCapability::writeControls(
    llm::StageResources const& stage, std::size_t tokenCount, int32_t kvLength) noexcept
{
    auto const* context = findBinding(stage, "context_lengths");
    auto const* start = findBinding(stage, "kvcache_start_index");
    auto const* last = findBinding(stage, "last_token_ids");
    if (context == nullptr || start == nullptr || last == nullptr || tokenCount == 0
        || tokenCount > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid embedding control bindings");
    }
    auto& contextHost = mData->storage[static_cast<std::size_t>(llm::ModelMemory::kCONTEXT_LENGTH)];
    auto& startHost = mData->storage[static_cast<std::size_t>(llm::ModelMemory::kKV_START)];
    auto& lastHost = mData->storage[static_cast<std::size_t>(llm::ModelMemory::kLAST_TOKEN)];
    *static_cast<int32_t*>(contextHost.data) = static_cast<int32_t>(tokenCount);
    *static_cast<int32_t*>(startHost.data) = kvLength;
    *static_cast<int64_t*>(lastHost.data) = static_cast<int64_t>(tokenCount - 1);
    auto status
        = cudaMemcpyAsync(context->memory.data, contextHost.data, sizeof(int32_t), cudaMemcpyHostToDevice, mStream);
    if (status == cudaSuccess)
    {
        status = cudaMemcpyAsync(start->memory.data, startHost.data, sizeof(int32_t), cudaMemcpyHostToDevice, mStream);
    }
    if (status == cudaSuccess)
    {
        status = cudaMemcpyAsync(last->memory.data, lastHost.data, sizeof(int64_t), cudaMemcpyHostToDevice, mStream);
    }
    return status == cudaSuccess ? SafetyStatus::ok() : fromCudaError(status, "Embedding control upload");
}

SafetyStatus InternVL3EmbeddingCapability::embed(
    LMStageContext const& context, TokenIdView tokens, int32_t kvLength) noexcept
{
    llm::StageResources const* stage = &context == mPrefillContext ? &mData->prefill
        : &context == mDecodeContext                               ? &mData->decode
                                                                   : nullptr;
    if (mState != CapabilityPreparationState::kREADY || stage == nullptr || tokens.data == nullptr || tokens.size == 0
        || tokens.size > mLimits.maxPromptTokens)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid embedding request");
    }
    for (std::size_t index = 0; index < tokens.size; ++index)
    {
        if (tokens.data[index] < 0 || tokens.data[index] >= mLimits.vocabularySize)
        {
            return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Embedding token is outside vocabulary");
        }
    }
    auto const& host = mData->storage[static_cast<std::size_t>(llm::ModelMemory::kEMBEDDING_TOKEN_IDS)];
    auto const& device = mData->storage[static_cast<std::size_t>(llm::ModelMemory::kDEVICE_TOKEN_IDS)];
    std::size_t const bytes = tokens.size * sizeof(int32_t);
    std::memcpy(host.data, tokens.data, bytes);
    auto cudaStatus = cudaMemcpyAsync(device.data, host.data, bytes, cudaMemcpyHostToDevice, mStream);
    auto const* output = findBinding(*stage, "inputs_embeds");
    if (cudaStatus == cudaSuccess)
    {
        cudaStatus = launchFp16EmbeddingLookup(static_cast<int32_t const*>(device.data), tokens.size,
            static_cast<half const*>(mData->storage[static_cast<std::size_t>(llm::ModelMemory::kEMBEDDING_TABLE)].data),
            mLimits.vocabularySize, mLimits.hiddenSize, static_cast<half*>(output->memory.data), mStream);
    }
    if (cudaStatus != cudaSuccess)
    {
        return fromCudaError(cudaStatus, "Embedding lookup");
    }
    return writeControls(*stage, tokens.size, kvLength);
}

SafetyStatus InternVL3EmbeddingCapability::embedPrompt(LMStageContext const& context, TokenIdView prompt) noexcept
{
    return embed(context, prompt, 0);
}

SafetyStatus InternVL3EmbeddingCapability::embedDecodeToken(
    LMStageContext const& context, int32_t token, int32_t kvLength) noexcept
{
    return embed(context, {&token, 1}, kvLength);
}

SafetyStatus InternVL3EmbeddingCapability::overlayVisualEmbeddings(LMStageContext const& context,
    LMStageOutputs const& outputs, std::size_t imageOffset, std::size_t imageTokenCount) noexcept
{
    if (mState != CapabilityPreparationState::kREADY || &context != mPrefillContext || outputs.context != mVisualContext
        || mData == nullptr || mVisualOutput == nullptr || imageTokenCount == 0 || imageOffset > mLimits.maxPromptTokens
        || imageTokenCount > mLimits.maxPromptTokens - imageOffset)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Invalid InternVL visual embedding span");
    }
    auto const* destination = findBinding(mData->prefill, "inputs_embeds");
    std::size_t const rowBytes = static_cast<std::size_t>(mLimits.hiddenSize) * sizeof(half);
    std::size_t const destinationRows = destination == nullptr ? 0 : destination->memory.sizeInBytes / rowBytes;
    if (destination == nullptr || imageTokenCount > mVisualOutput->memory.sizeInBytes / rowBytes
        || imageOffset > destinationRows || imageTokenCount > destinationRows - imageOffset)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "Invalid InternVL visual embedding span");
    }
    auto* target = static_cast<char*>(destination->memory.data) + imageOffset * rowBytes;
    return fromCudaError(cudaMemcpyAsync(target, mVisualOutput->memory.data, imageTokenCount * rowBytes,
                             cudaMemcpyDeviceToDevice, mStream),
        "InternVL visual embedding handoff");
}

SafetyStatus InternVL3EmbeddingCapability::prepareWarmup(LMStageContext const& context, int64_t profile) noexcept
{
    if (mState != CapabilityPreparationState::kREADY || context.mainStream != mStream
        || context.plan.profiles.size() != 1 || context.plan.profiles.front() != profile)
    {
        return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Invalid warmup context");
    }
    SafetyTensorView const* input{nullptr};
    auto status = context.getTensorView(context.plan.id == "visual_encode" ? "input" : "inputs_embeds", input);
    if (!status.okStatus() || input == nullptr)
    {
        return status;
    }
    auto cudaStatus = cudaMemsetAsync(input->data, 0, input->profileBytes, mStream);
    if (cudaStatus != cudaSuccess || context.plan.id == "visual_encode")
    {
        return cudaStatus == cudaSuccess ? SafetyStatus::ok() : fromCudaError(cudaStatus, "Warmup input");
    }
    auto const& stage = context.plan.id == "prefill" ? mData->prefill : mData->decode;
    status = writeControls(stage, 1, 0);
    return status;
}

SafetyStatus InternVL3EmbeddingCapability::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    mUpload = {};
    mVisualOutput = nullptr;
    mData = nullptr;
    mVisualContext = nullptr;
    mPrefillContext = nullptr;
    mDecodeContext = nullptr;
    mStream = nullptr;
    mInitializationCompletion = nullptr;
    mState = CapabilityPreparationState::kCLOSED;
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
