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
 * reference: https://github.com/ggerganov/llama.cpp/blob/master/src/unicode.cpp
 */

#include <cassert>
#include <stdexcept>

#include "tokenizerUtils.h"
#include "unicodeData.h"

namespace trt_edgellm
{
namespace tokenizer
{

RanksToToken reverseEncoder(TokenToRanks const& encoder)
{
    RanksToToken decoder;
    for (auto const& [key, value] : encoder)
    {
        decoder[value] = key;
    }

    assert(encoder.size() == decoder.size());
    return decoder;
}

[[maybe_unused]] static std::unordered_map<uint32_t, uint8_t> unicodeCptToByteMap()
{
    std::unordered_map<uint32_t, uint8_t> map;
    for (uint32_t ch = 0x21; ch <= 0x7E; ++ch)
    { // u'!' to u'~'
        assert(ch < 256);
        map[ch] = ch;
    }
    for (uint32_t ch = 0xA1; ch <= 0xAC; ++ch)
    { // u'¡' to u'¬'
        assert(ch < 256);
        map[ch] = ch;
    }
    for (uint32_t ch = 0xAE; ch <= 0xFF; ++ch)
    { // u'®' to u'ÿ'
        assert(ch < 256);
        map[ch] = ch;
    }

    auto n = 0;
    for (uint32_t ch = 0; ch < 256; ++ch)
    {
        if (map.find(ch) == map.end())
        {
            map[256 + n] = ch;
            ++n;
        }
    }

    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map()
{
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch)
    { // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicodeCptToUtf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch)
    { // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicodeCptToUtf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch)
    { // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicodeCptToUtf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch)
    {
        if (map.find(unicodeCptToUtf8(ch)) == map.end())
        {
            map[unicodeCptToUtf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

std::string decodeHFTokenToNormal(std::string const& hfToken)
{
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    std::string decoded;

    auto const cpts = unicodeCptsFromUtf8(hfToken);
    for (auto const cpt : cpts)
    {
        auto const utf8 = unicodeCptToUtf8(cpt);
        auto it = map.find(utf8);
        assert(it != map.end());
        decoded += it->second;
    }

    return decoded;
}

std::string normalizeRegex(std::string const& expr)
{
    std::string normalizedExpr;

    for (size_t i = 0; i < expr.size();)
    {
        // case 1: (?i) case-insensitive modifier
        // e.g. (?i:'s|'t|'re|'ve|'m|'ll|'d) => (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        if (expr.find("(?i:", i) != std::string::npos)
        {
            normalizedExpr += "(?:";

            size_t next = expr.find("(?i:", i) + 4;
            size_t end = expr.find(")", next);
            auto part = expr.substr(next, end - next);

            for (size_t j = next; j < end; ++j)
            {
                char c = expr[j];
                if (isalpha(c))
                {
                    normalizedExpr += '[';
                    normalizedExpr += tolower(c);
                    normalizedExpr += toupper(c);
                    normalizedExpr += ']';
                }
                else
                {
                    normalizedExpr += c;
                }
            }

            normalizedExpr += ")";
            i = end + 1;
        }
        else
        {
            normalizedExpr += expr[i];
            ++i;
        }
    }

    return normalizedExpr;
}

bool validateFileSize(std::filesystem::path const& filePath, size_t maxSizeBytes)
{
    std::error_code ec;
    auto fileSize = std::filesystem::file_size(filePath, ec);
    if (ec)
    {
        LOG_ERROR("Failed to get file size for %s: %s", filePath.c_str(), ec.message().c_str());
        return false;
    }

    if (fileSize > maxSizeBytes)
    {
        LOG_ERROR("File too large: %s (%zu bytes, max: %zu)", filePath.c_str(), fileSize, maxSizeBytes);
        return false;
    }

    return true;
}

/**
 * Unicode Utils
 */
std::string unicodeCptToUtf8(uint32_t cp)
{
    std::string result;

    if (/* 0x00 <= cp && */ cp <= 0x7f)
    {
        result.push_back(cp);
        return result;
    }
    if (0x80 <= cp && cp <= 0x7ff)
    {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x800 <= cp && cp <= 0xffff)
    {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x10000 <= cp && cp <= 0x10ffff)
    {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicodeCptsFromUtf8(std::string const& utf8)
{
    std::vector<uint32_t> result;
    result.reserve(utf8.size());

    size_t offset = 0;
    while (offset < utf8.size())
    {
        result.push_back(unicodeCptFromUtf8(utf8, offset));
    }
    return result;
}

uint32_t unicodeCptFromUtf8(std::string const& utf8, size_t& offset)
{
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80))
    {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40))
    {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20))
    {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80))
        {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10))
    {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80))
        {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08))
    {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)
            || !((utf8[offset + 3] & 0xc0) == 0x80))
        {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12)
            | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

bool unicodeCollapseRegex(std::string const& expr, std::regex& regex)
{
    // search for unicode categories in expr
    bool needRegexCollapse = false;
    for (auto const& ucat : kUatEnum)
    {
        if (expr.find(ucat.first) != std::string::npos)
        {
            needRegexCollapse = true;
            break;
        }
    }

    if (needRegexCollapse)
    {
        try
        {
            // sanity-check that the original regex does not contain any non-ASCII characters
            auto const cpts_regex = unicodeCptsFromUtf8(expr);
            for (size_t i = 0; i < cpts_regex.size(); ++i)
            {
                if (cpts_regex[i] >= 128)
                {
                    throw std::runtime_error(
                        "Regex includes both unicode categories and non-ASCII characters - not supported");
                }
            }

            // generate a collapsed representation of the regex
            std::string regexExprCollapsed;

            // track if we are inside [], because nested [] are not allowed
            bool inside = false;
            for (size_t i = 0; i < expr.size(); ++i)
            {
                if (expr[i] == '[' && (i == 0 || expr[i - 1] != '\\'))
                {
                    regexExprCollapsed += '[';
                    inside = true;
                    continue;
                }

                if (inside && expr[i] == ']' && expr[i - 1] != '\\')
                {
                    regexExprCollapsed += ']';
                    inside = false;
                    continue;
                }

                if (expr[i + 0] == '\\' && i + 4 < expr.size() && expr[i + 1] == 'p' && expr[i + 2] == '{'
                    && expr[i + 4] == '}')
                {
                    std::string const pat = expr.substr(i, 5);
                    if (kUatEnum.find(pat) != kUatEnum.end())
                    {
                        if (!inside)
                        {
                            regexExprCollapsed += '[';
                        }
                        regexExprCollapsed += kUcatCpt.at(kUatEnum.at(pat));
                        regexExprCollapsed += kUcatMap.at(kUatEnum.at(pat));
                        if (!inside)
                        {
                            regexExprCollapsed += ']';
                        }
                        i += 4;
                        continue;
                    }
                }

                regexExprCollapsed += expr[i];
            }

            regex = std::regex(regexExprCollapsed);
        }
        catch (std::regex_error& e)
        {
            LOG_ERROR("Failed to process regex: %s", expr.c_str());
            throw std::runtime_error("Failed to process regex");
        }
    }
    else
    {
        regex = std::regex(expr);
    }

    return needRegexCollapse;
}

std::string unicodeCollapseText(std::vector<uint32_t> const& cpts)
{
    std::string textCollapsed;

    // collapse all unicode categories
    textCollapsed.resize(cpts.size());

    for (size_t i = 0; i < cpts.size(); ++i)
    {
        // keep single-byte codepoints as is
        if (cpts[i] < 128)
        {
            textCollapsed[i] = cpts[i];
            continue;
        }

        auto const flags = unicodeCptFlags(cpts[i]);

        if (flags.isAccentMark)
        {
            // NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
            // textCollapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
            textCollapsed[i] = (char) 0x0B; // <vertical tab> as whitespace fallback
        }
        else if (kUcatCpt.find(flags.categoryFlag()) != kUcatCpt.end())
        {
            textCollapsed[i] = kUcatCpt.at(flags.categoryFlag());
        }
        else
        {
            textCollapsed[i] = (char) 0xD0; // fallback
        }
    }

    return textCollapsed;
}

std::vector<size_t> unicodeRegexSplit(std::string const& text, std::regex const& regex)
{
    std::vector<size_t> bpeOffsets; // store the offset of each word

    std::cregex_iterator itBegin(text.data(), text.data() + text.size(), regex);
    std::cregex_iterator itEnd;

    int64_t startIdx = 0;
    for (std::cregex_iterator it = itBegin; it != itEnd; ++it)
    {
        std::cmatch match = *it;

        // Add part before match
        if (match.position() > startIdx)
        {
            bpeOffsets.emplace_back(match.position() - startIdx);
        }
        bpeOffsets.emplace_back(match.length());
        startIdx = match.position() + match.length();
    }

    // Add remaining part
    if (startIdx < (int64_t) text.size())
    {
        bpeOffsets.emplace_back(text.size() - startIdx);
    }

    return bpeOffsets;
}

static std::vector<codepointFlags> unicodeCptFlagsArray()
{
    std::vector<codepointFlags> cpt_flags(MAX_CODEPOINTS, codepointFlags::UNDEFINED);

    assert(unicodeRangesFlags.front().first == 0);
    assert(unicodeRangesFlags.back().first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicodeRangesFlags.size(); ++i)
    {
        auto const range_ini = unicodeRangesFlags[i - 1]; // codepoint_ini, flags
        auto const range_end = unicodeRangesFlags[i];     // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt)
        {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicodeSetWhitespace)
    {
        cpt_flags[cpt].isWhitespace = true;
    }

    for (auto p : unicodeMapLowercase)
    {
        cpt_flags[p.second].isLowercase = true;
    }

    for (auto p : unicodeMapUppercase)
    {
        cpt_flags[p.second].isUppercase = true;
    }

    for (auto& range : unicodeRangesNfd)
    { // start, last, nfd
        cpt_flags[range.nfd].isNfd = true;
    }

    return cpt_flags;
}

codepointFlags unicodeCptFlags(uint32_t const cp)
{
    static codepointFlags const undef(codepointFlags::UNDEFINED);
    static auto const cptFlags = unicodeCptFlagsArray();
    return cp < cptFlags.size() ? cptFlags[cp] : undef;
}

} // namespace tokenizer
} // namespace trt_edgellm