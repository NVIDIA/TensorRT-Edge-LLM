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
 * reference: https://github.com/ggerganov/llama.cpp/blob/master/src/unicode.h
 */

#pragma once

#include "common/logger.h"
#include "tokenizer.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{

/*!
 * @defgroup TokenizerHelpers Helper functions for tokenization
 * @{
 */

/// \cond INTERNAL
//! Regex pattern for special characters that need escaping
static std::regex const specialChars{R"([[\^$.|?*+(){}])"};
/// \endcond

/*!
 * @brief Reverse token-to-rank mapping
 * @param encoder Token to rank mapping
 * @return Rank to token mapping
 */
RanksToToken reverseEncoder(TokenToRanks const& encoder);

/*!
 * @brief Decode HuggingFace format token to normal UTF-8
 * @param hfToken HuggingFace format token string
 * @return Decoded UTF-8 string
 */
std::string decodeHFTokenToNormal(std::string const& hfToken);

/*!
 * @brief Normalize regex expressions for C++ regex compatibility
 * @param expr Regular expression string
 * @return Normalized regex expression
 */
std::string normalizeRegex(std::string const& expr);

//! @}

/**
 * @brief Validate file size before reading
 * @param filePath Path to the file to check
 * @param maxSizeBytes Maximum allowed file size in bytes
 * @return true if file exists, size can be determined, and is within limit;
 *         false if file doesn't exist, size cannot be determined, or exceeds limit
 */
bool validateFileSize(std::filesystem::path const& filePath, size_t maxSizeBytes);

/*!
 * @defgroup UnicodeUtils Unicode utility functions and structures
 * @{
 */

/*!
 * @brief Unicode codepoint flags
 *
 * Bitfield structure for Unicode codepoint properties.
 */
struct codepointFlags
{
    //! @brief Category flag constants
    enum CategoryFlags
    {
        UNDEFINED = 0x0001,       //!< Undefined category
        NUMBER = 0x0002,          //!< Number category (\\p{N})
        LETTER = 0x0004,          //!< Letter category (\\p{L})
        SEPARATOR = 0x0008,       //!< Separator category (\\p{Z})
        ACCENT_MARK = 0x0010,     //!< Accent mark category (\\p{M})
        PUNCTUATION = 0x0020,     //!< Punctuation category (\\p{P})
        SYMBOL = 0x0040,          //!< Symbol category (\\p{S})
        CONTROL = 0x0080,         //!< Control character category (\\p{C})
        MASK_CATEGORIES = 0x00FF, //!< Mask for category flags
    };

    uint16_t isUndefined : 1;   //!< Is undefined
    uint16_t isNumber : 1;      //!< Is number (\\p{N})
    uint16_t isLetter : 1;      //!< Is letter (\\p{L})
    uint16_t isSeparator : 1;   //!< Is separator (\\p{Z})
    uint16_t isAccentMark : 1;  //!< Is accent mark (\\p{M})
    uint16_t isPunctuation : 1; //!< Is punctuation (\\p{P})
    uint16_t isSymbol : 1;      //!< Is symbol (\\p{S})
    uint16_t isControl : 1;     //!< Is control character (\\p{C})
    uint16_t isWhitespace : 1;  //!< Is whitespace (\\s)
    uint16_t isLowercase : 1;   //!< Is lowercase
    uint16_t isUppercase : 1;   //!< Is uppercase
    uint16_t isNfd : 1;         //!< Has NFD form

    /*!
     * @brief Construct from uint16 flags
     * @param flags Flag value
     */
    inline codepointFlags(uint16_t const flags = 0)
    {
        *reinterpret_cast<uint16_t*>(this) = flags;
    }

    //! @brief Convert to uint16
    //! @return Flags as uint16
    inline uint16_t asUint() const
    {
        return *reinterpret_cast<uint16_t const*>(this);
    }

    //! @brief Get category flag
    //! @return Category flag value
    inline uint16_t categoryFlag() const
    {
        return this->asUint() & MASK_CATEGORIES;
    }
};

/// \cond INTERNAL
// Unicode category mappings
//! @brief Map from regex patterns to category flags
static std::map<std::string, int> const kUatEnum = {
    {"\\p{N}", codepointFlags::NUMBER},
    {"\\p{L}", codepointFlags::LETTER},
    {"\\p{P}", codepointFlags::PUNCTUATION},
};

//! @brief Map from category flags to codepoints
static std::map<int, int> const kUcatCpt = {
    {codepointFlags::NUMBER, 0xD1},
    {codepointFlags::LETTER, 0xD2},
    {codepointFlags::PUNCTUATION, 0xD3},
};

//! @brief Map from category flags to character ranges
static std::map<int, std::string> const kUcatMap = {
    {codepointFlags::NUMBER, "\x30-\x39"},          // 0-9
    {codepointFlags::LETTER, "\x41-\x5A\x61-\x7A"}, // A-Za-z
    {codepointFlags::PUNCTUATION,
        "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"}, // !-#%-*,-/:-;?-@\[-\]_\{\}
};
/// \endcond

/*!
 * @brief Collapse Unicode categories in regex
 * @param expr Regex expression with Unicode categories
 * @param regex Output compiled regex
 * @return True on success, false on failure
 */
bool unicodeCollapseRegex(std::string const& expr, std::regex& regex);

/*!
 * @brief Collapse codepoints to text
 * @param cpts Vector of codepoints
 * @return Collapsed text string
 */
std::string unicodeCollapseText(std::vector<uint32_t> const& cpts);

/*!
 * @brief Split text using regex and return split positions
 * @param text Input text
 * @param regex Regex pattern for splitting
 * @return Vector of split positions
 */
std::vector<size_t> unicodeRegexSplit(std::string const& text, std::regex const& regex);

/*!
 * @brief Convert UTF-8 string to codepoints
 * @param utf8 UTF-8 encoded string
 * @return Vector of codepoints
 */
std::vector<uint32_t> unicodeCptsFromUtf8(std::string const& utf8);

/*!
 * @brief Extract single codepoint from UTF-8 string
 * @param utf8 UTF-8 encoded string
 * @param offset Offset in string (updated after extraction)
 * @return Extracted codepoint
 */
uint32_t unicodeCptFromUtf8(std::string const& utf8, size_t& offset);

/*!
 * @brief Convert codepoint to UTF-8 string
 * @param cp Codepoint value
 * @return UTF-8 encoded string
 */
std::string unicodeCptToUtf8(uint32_t cp);

/*!
 * @brief Get flags for a codepoint
 * @param cp Codepoint value
 * @return Codepoint flags
 */
codepointFlags unicodeCptFlags(uint32_t const cp);

//! @}

} // namespace tokenizer
} // namespace trt_edgellm