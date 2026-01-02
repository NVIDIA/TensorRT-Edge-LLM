/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * reference: https://github.com/ggerganov/llama.cpp/blob/master/src/unicode-data.h
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{

/*!
 * @brief NFD (Normalization Form Decomposition) range structure
 *
 * Represents a range of codepoints and their NFD mappings.
 */
struct rangeNfd
{
    uint32_t first; //!< First codepoint in range
    uint32_t last;  //!< Last codepoint in range
    uint32_t nfd;   //!< NFD mapping value
};

//! Maximum number of Unicode codepoints
static uint32_t const MAX_CODEPOINTS = 0x110000;

//! @brief Unicode ranges with flags for codepoint properties
extern std::vector<std::pair<uint32_t, uint16_t>> const unicodeRangesFlags;

//! @brief Set of whitespace codepoints
extern std::unordered_set<uint32_t> const unicodeSetWhitespace;

//! @brief Mapping from codepoints to their lowercase equivalents
extern std::unordered_map<uint32_t, uint32_t> const unicodeMapLowercase;

//! @brief Mapping from codepoints to their uppercase equivalents
extern std::unordered_map<uint32_t, uint32_t> const unicodeMapUppercase;

//! @brief NFD normalization ranges
extern std::vector<rangeNfd> const unicodeRangesNfd;

} // namespace tokenizer
} // namespace trt_edgellm