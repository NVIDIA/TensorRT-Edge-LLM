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

#pragma once

#include "tokenizer/tokenizer.h"

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

//! Tokens a prefill layout writes as prefill rows or finds in the token stream.
struct TalkerTokenIds
{
    //! Every layout's text-side rows. tts_eos is spelled <tts_text_eod>; <tts_text_bos_single> is another token.
    int32_t ttsPad{-1};
    int32_t ttsBos{-1};
    int32_t ttsEos{-1};
    //! Chat-template tokens, -1 unless asked for: the tts layout strips a fixed wrapper instead.
    int32_t imStart{-1};
    int32_t imEnd{-1};
    int32_t systemRole{-1};
    int32_t userRole{-1};
    int32_t assistantRole{-1};
    int32_t thinkOpen{-1};
    int32_t thinkClose{-1};
    int32_t newline{-1};
    int32_t doubleNewline{-1};
    //! Media placeholders, -1 unless asked for: only the segments layout looks for them.
    int32_t audioPad{-1};
    int32_t imagePad{-1};
    int32_t videoPad{-1};
};

//! Look the tts tokens up in @p tokenizer, the chat-template tokens only if @p withChatTemplate and the media
//! placeholders only if @p withMediaPads. Throws naming the first string that does not encode to exactly one token
//! spelling all of it.
TalkerTokenIds resolveTalkerTokenIds(tokenizer::Tokenizer const& tokenizer, bool withChatTemplate, bool withMediaPads);

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
