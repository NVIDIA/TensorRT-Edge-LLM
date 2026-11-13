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

#include "multimodal/minimax_h3/minimaxH3Stages.h"

#include "common/checkMacros.h"
#include "profiling/timer.h"

#include <algorithm>
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

namespace
{

constexpr char const* kRunnerName = "minimax_h3.text_encoder";
constexpr char const* kPromptInput = "prompt";
constexpr char const* kTextHiddenPort = "text_hidden";
std::string const kTimerStage = "minimax_h3_text_encoder";

} // namespace

TextEncoderStage::TextEncoderStage(multiStage::StageInitContext const& init)
    : mArtifactDir(init.artifactDir)
    , mStream(init.stream)
{
}

void TextEncoderStage::preallocate()
{
    ELLM_CHECK(mTokenizer.loadFromHF(mArtifactDir), "Failed to load the MiniMax-H3 tokenizer from " + mArtifactDir);
    mEngine = std::make_unique<MiniMaxH3Engine>(mArtifactDir, "text_encoder", mStream);
    mMaxTokens = mEngine->config().at("max_text_tokens");
    int32_t const hiddenSize = mEngine->config().at("hidden_size");
    mInputIds = rt::Tensor({1, mMaxTokens}, rt::DeviceType::kGPU, DataType::kINT32, "minimax_h3::input_ids");
    mInputIdsHost = rt::Tensor({1, mMaxTokens}, rt::DeviceType::kCPU, DataType::kINT32, "minimax_h3::input_ids_host");
    mTextHidden
        = rt::Tensor({1, mMaxTokens, hiddenSize}, rt::DeviceType::kGPU, DataType::kHALF, "minimax_h3::text_hidden");
}

bool TextEncoderStage::admit(multiStage::StageExecContext const& ctx, std::string& reason)
{
    auto const* prompt = ctx.in != nullptr ? ctx.in->find(kPromptInput) : nullptr;
    if (prompt == nullptr || prompt->shape.size() != 1
        || (prompt->elementBytes != 1 && prompt->elementBytes != static_cast<int32_t>(sizeof(int32_t))))
    {
        reason = "MiniMax-H3 takes a prompt as UTF-8 text or INT32 token ids";
        return false;
    }
    return true;
}

void TextEncoderStage::prepare(multiStage::StageExecContext& ctx)
{
    auto const& prompt = ctx.in->at(kPromptInput);
    std::vector<int32_t> ids;
    if (prompt.elementBytes == 1)
    {
        std::string const text(static_cast<char const*>(prompt.data), static_cast<size_t>(prompt.shape[0]));
        auto const parsed = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
        ELLM_CHECK(
            !parsed.is_object() || !parsed.contains("messages"), "MiniMax-H3 takes a plain prompt, not a conversation");
        ids = mTokenizer.encode(text);
    }
    else
    {
        auto const* values = static_cast<int32_t const*>(prompt.data);
        ids.assign(values, values + prompt.shape[0]);
    }
    mTokens = static_cast<int32_t>(ids.size());
    ELLM_CHECK(mTokens > 0 && mTokens <= mMaxTokens,
        "MiniMax-H3 prompt has " + std::to_string(mTokens) + " tokens; the text encoder takes 1 to "
            + std::to_string(mMaxTokens));
    std::copy(ids.begin(), ids.end(), mInputIdsHost.dataPointer<int32_t>());
}

multiStage::StageStatus TextEncoderStage::step(multiStage::StageExecContext& ctx)
{
    if (ctx.slot->abandon.requested())
    {
        return multiStage::StageStatus::kDone;
    }
    int64_t const hiddenSize = mTextHidden.getShape()[2];
    ELLM_CHECK(mInputIds.reshape({1, mTokens}) && mTextHidden.reshape({1, mTokens, hiddenSize}),
        "Prompt does not fit the text buffers");
    {
        TIME_STAGE(kTimerStage, ctx.stream);
        CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mInputIdsHost.rawPointer(), mTokens * sizeof(int32_t),
            cudaMemcpyHostToDevice, ctx.stream));
        mEngine->bind("input_ids", mInputIds);
        mEngine->bind("text_hidden", mTextHidden);
        mEngine->enqueue(ctx.stream);
    }
    ctx.out->bind(kTextHiddenPort,
        multiStage::PortValue{mTextHidden.rawPointer(), {1, mTokens, hiddenSize}, multiStage::PortKind::kEmbeds,
            static_cast<int32_t>(sizeof(half)), /*complete=*/true});
    return multiStage::StageStatus::kDone;
}

void TextEncoderStage::finish(multiStage::StageExecContext&, multiStage::FinishReason) {}

void TextEncoderStage::reset()
{
    mTokens = 0;
}

nlohmann::json TextEncoderStage::diagnostics(multiStage::RequestSlot const&) const
{
    return {{"prompt_tokens", mTokens}};
}

std::unique_ptr<multiStage::IStage> createTextEncoderStage(multiStage::StageInitContext const& init)
{
    return std::make_unique<TextEncoderStage>(init);
}

multiStage::RunnerDecl textEncoderDeclaration()
{
    multiStage::RunnerDecl decl;
    decl.name = kRunnerName;
    decl.kind = multiStage::StageKind::kEngine;
    decl.ports = {{kTextHiddenPort, multiStage::PortKind::kEmbeds}};
    decl.inputs = {kPromptInput};
    return decl;
}

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
