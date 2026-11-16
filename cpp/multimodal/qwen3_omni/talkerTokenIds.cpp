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

#include "multimodal/qwen3_omni/talkerTokenIds.h"

#include "common/checkMacros.h"

#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

namespace
{

std::string escaped(std::string const& text)
{
    std::string shown;
    for (char const c : text)
    {
        shown += c == '\n' ? std::string("\\n") : std::string(1, c);
    }
    return shown;
}

//! encode() drops the rest of a pre-tokenized piece at the first sub-word the vocabulary lacks, so a lone id is the
//! answer only when that token spells the whole string.
int32_t singleToken(tokenizer::Tokenizer const& tokenizer, std::string const& text)
{
    auto const ids = tokenizer.encode(text);
    ELLM_CHECK(ids.size() == 1 && tokenizer.idToPiece(ids.front(), /*skipSpecialTokens=*/false) == text,
        "the talker's tokenizer does not encode '" + escaped(text) + "' as one token");
    return ids.front();
}

} // namespace

TalkerTokenIds resolveTalkerTokenIds(tokenizer::Tokenizer const& tokenizer, bool withChatTemplate, bool withMediaPads)
{
    TalkerTokenIds ids;
    ids.ttsPad = singleToken(tokenizer, "<tts_pad>");
    ids.ttsBos = singleToken(tokenizer, "<tts_text_bos>");
    ids.ttsEos = singleToken(tokenizer, "<tts_text_eod>");
    if (withChatTemplate)
    {
        ids.imStart = singleToken(tokenizer, "<|im_start|>");
        ids.imEnd = singleToken(tokenizer, "<|im_end|>");
        ids.systemRole = singleToken(tokenizer, "system");
        ids.userRole = singleToken(tokenizer, "user");
        ids.assistantRole = singleToken(tokenizer, "assistant");
        ids.thinkOpen = singleToken(tokenizer, "<think>");
        ids.thinkClose = singleToken(tokenizer, "</think>");
        ids.newline = singleToken(tokenizer, "\n");
        ids.doubleNewline = singleToken(tokenizer, "\n\n");
    }
    if (withMediaPads)
    {
        ids.audioPad = singleToken(tokenizer, "<|audio_pad|>");
        ids.imagePad = singleToken(tokenizer, "<|image_pad|>");
        ids.videoPad = singleToken(tokenizer, "<|video_pad|>");
    }
    return ids;
}

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
