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

#include "multimodal/qwen3/qwen3vlViTRunner.h"
#include "scratchDir.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kX = 0;
constexpr int32_t kCOLON_LT = 3; //!< `:<`, present only when `:` and `<` share one BPE piece
constexpr int32_t kZERO = 4;
constexpr int32_t kDOT = 5;
constexpr int32_t kFIVE = 6;
constexpr int32_t kSPACE = 7;
constexpr int32_t kS = 8;
constexpr int32_t kE = 9;
constexpr int32_t kC = 10;
constexpr int32_t kO = 11;
constexpr int32_t kN = 12;
constexpr int32_t kD = 13;
constexpr int32_t kGT = 14;
constexpr int32_t kVISION_START = 20;
constexpr int32_t kVISION_END = 21;
constexpr int32_t kIMAGE_PAD = 22;
constexpr int32_t kVIDEO_PAD = 23;

//! A whole-string pre-tokenizer, so adjacent pieces merge whenever the vocabulary holds their concatenation.
std::unique_ptr<tokenizer::Tokenizer> loadTokenizer()
{
    auto const dir = makeScratchDir("qwen3vlPromptTokenization");
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {
    "x": 0, ":": 1, "<": 2, ":<": 3, "0": 4, ".": 5, "5": 6, " ": 7, "s": 8, "e": 9, "c": 10, "o": 11, "n": 12,
    "d": 13, ">": 14
  }, "merges": []},
  "added_tokens": [{"id": 20, "content": "<|vision_start|>", "special": true},
    {"id": 21, "content": "<|vision_end|>", "special": true},
    {"id": 22, "content": "<|image_pad|>", "special": true},
    {"id": 23, "content": "<|video_pad|>", "special": true}],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
    auto tokenizer = std::make_unique<tokenizer::Tokenizer>();
    EXPECT_TRUE(tokenizer->loadFromHF(dir));
    std::filesystem::remove_all(dir);
    return tokenizer;
}

} // namespace

TEST(Qwen3VLPromptTokenizationTest, VideoTimestampSharesBpeWithPrecedingText)
{
    // Given a prompt whose text ends in `:` right before a two-frame video placeholder, and a vocabulary that merges
    // `:<`, when the prompt is tokenized, then the timestamp's `<` merges with that `:` as the Hugging Face processor
    // does, and the video pad expands to its span's token count.
    auto const tokenizer = loadTokenizer();
    rt::QwenViTConfig config;
    config.visionStartTokenId = kVISION_START;
    config.visionEndTokenId = kVISION_END;
    config.imageTokenId = kIMAGE_PAD;
    config.videoTokenId = kVIDEO_PAD;
    config.temporalPatchSize = 2;
    std::vector<rt::imageUtils::ImageData> media(1);
    media[0].isVideo = true;
    media[0].frames = 2;
    media[0].fps = 1.0;
    std::vector<rt::VisionSpan> spans(1);
    spans[0].llm.numTokens = 3;

    auto const ids = rt::qwen3vl::tokenizeExpandedPrompt(
        "x:<|vision_start|><|video_pad|><|vision_end|>", media, spans, 0, 1, config, *tokenizer);

    std::vector<int32_t> const expected{kX, kCOLON_LT, kZERO, kDOT, kFIVE, kSPACE, kS, kE, kC, kO, kN, kD, kS, kGT,
        kVISION_START, kIMAGE_PAD, kIMAGE_PAD, kIMAGE_PAD, kVISION_END};
    EXPECT_EQ(ids, expected);
}
