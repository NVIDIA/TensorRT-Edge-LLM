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

#include <cstdint>
#include <filesystem>
#include <fstream>

//! Tiny on-disk tokenizers for the guided-decoding and jump-forward tests: enough vocabulary to
//! spell `{"a":1,"b":2}`, plus the merges those tests need, without loading a model.
namespace guidedFixture
{

// Byte-level BPE. Merge order follows vocabulary rank, so `"` + `,` re-encodes as `",`, and the
// two-byte prefix at 15 lets the bytes of the CJK character merge up to token 16.
constexpr int32_t kOPEN_BRACE = 0;
constexpr int32_t kCLOSE_BRACE = 1;
constexpr int32_t kQUOTE = 2;
constexpr int32_t kCOLON = 3;
constexpr int32_t kA = 4;
constexpr int32_t kONE = 5;
constexpr int32_t kTWO = 6;
constexpr int32_t kCOMMA = 7;
constexpr int32_t kB = 8;
constexpr int32_t kSPACE = 9;
constexpr int32_t kEOS = 10;
constexpr int32_t kTHINK_END = 11;
constexpr int32_t kQUOTE_COMMA = 12; //!< `",`
constexpr int32_t kSPACE_QUOTE = 13; //!< ` "`
constexpr int32_t kQUOTE_COLON = 14; //!< `":`
constexpr int32_t kCJK_PREFIX = 15;  //!< First two bytes of U+4E2D
constexpr int32_t kCJK = 16;         //!< U+4E2D
constexpr int32_t kBPE_VOCAB_SIZE = 17;

inline void writeBpeTokenizer(std::filesystem::path const& dir)
{
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {
    "{": 0, "}": 1, "\"": 2, ":": 3, "a": 4, "1": 5, "2": 6, ",": 7, "b": 8, " ": 9, "<eos>": 10,
    "</think>": 11, "\",": 12, " \"": 13, "\":": 14, "ä¸": 15, "中": 16
  }, "merges": []},
  "added_tokens": [{"id": 10, "content": "<eos>"}, {"id": 11, "content": "</think>"}],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << R"JSON({"eos_token": {"content": "<eos>"}})JSON";
}

// SentencePiece-style Metaspace tokenizer: encode() prepends the word marker to text that does not
// start with a space, which is the case the jump-forward planner must refuse.
constexpr int32_t kMS_MARKER = 0;   //!< `▁`
constexpr int32_t kMS_A = 1;        //!< `a`
constexpr int32_t kMS_MARKER_A = 2; //!< `▁a`
constexpr int32_t kMS_COMMA = 3;
constexpr int32_t kMS_EOS = 4;
constexpr int32_t kMETASPACE_VOCAB_SIZE = 5;

inline void writeMetaspaceTokenizer(std::filesystem::path const& dir)
{
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "byte_fallback": true,
    "vocab": {"▁": 0, "a": 1, "▁a": 2, ",": 3, "<eos>": 4},
    "merges": [["▁", "a"]]},
  "added_tokens": [{"id": 4, "content": "<eos>", "special": true}],
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"}
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << R"JSON({"eos_token": {"content": "<eos>"}})JSON";
}

} // namespace guidedFixture
