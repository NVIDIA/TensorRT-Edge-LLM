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

#include "preTokenizer.h"
#include "tokenizerUtils.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <stdexcept>

namespace trt_edgellm
{
namespace tokenizer
{

// RegexSplit implementation
RegexSplit::RegexSplit(std::string const& pattern)
    : mPattern(pattern)
    , mNeedRegexCollapse(false)
{
    if (pattern.empty())
    {
        throw std::invalid_argument("RegexSplit: Empty regex pattern provided");
    }

    try
    {
        mNeedRegexCollapse = unicodeCollapseRegex(pattern, mRegex);
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("RegexSplit: Failed to process regex pattern '" + pattern + "': " + e.what());
    }
}

std::vector<std::string> RegexSplit::process(std::string const& text) const
{
    if (text.empty())
    {
        return {};
    }

    if (text.size() > MAX_TEXT_SIZE_BYTES) // 1MB limit
    {
        throw std::runtime_error(
            "RegexSplit: Input text too large for regex processing: " + std::to_string(text.size()) + " bytes");
    }

    auto const cpts = unicodeCptsFromUtf8(text);

    // collapse for unicode regex match
    std::string textCollapsed;
    if (mNeedRegexCollapse)
    {
        try
        {
            textCollapsed = unicodeCollapseText(cpts);
        }
        catch (std::exception const& e)
        {
            throw std::runtime_error("RegexSplit: Failed to collapse unicode text: " + std::string(e.what()));
        }
    }
    else
    {
        textCollapsed = text;
    }

    std::vector<size_t> bpeOffsets;
    try
    {
        bpeOffsets = unicodeRegexSplit(textCollapsed, mRegex);
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("RegexSplit: Regex execution failed: " + std::string(e.what()));
    }

    std::vector<std::string> bpeWords;
    bpeWords.reserve(bpeOffsets.size());

    size_t wordStart = 0;
    for (auto const& offset : bpeOffsets)
    {
        bpeWords.emplace_back();
        for (size_t i = wordStart; i < wordStart + offset; ++i)
        {
            bpeWords.back() += unicodeCptToUtf8(cpts[i]);
        }
        wordStart += offset;
    }

    return bpeWords;
}

Metaspace::Metaspace(std::string replacement, std::string prependScheme, bool split)
    : mReplacement(std::move(replacement))
    , mPrepend(prependScheme != "never")
    , mSplit(split)
{
    if (mReplacement.empty())
    {
        throw std::invalid_argument("Metaspace: replacement must not be empty");
    }
    if (prependScheme != "first" && prependScheme != "always" && prependScheme != "never")
    {
        throw std::invalid_argument("Metaspace: unsupported prepend_scheme '" + prependScheme + "'");
    }
}

std::vector<std::string> Metaspace::process(std::string const& text) const
{
    if (text.empty())
    {
        return {};
    }
    if (text.size() > MAX_TEXT_SIZE_BYTES)
    {
        throw std::runtime_error(
            "Metaspace: Input text too large for processing: " + std::to_string(text.size()) + " bytes");
    }

    std::string transformed;
    transformed.reserve(text.size() + mReplacement.size());
    if (mPrepend && text.front() != ' ')
    {
        transformed += mReplacement;
    }
    for (char const character : text)
    {
        if (character == ' ')
        {
            transformed += mReplacement;
        }
        else
        {
            transformed += character;
        }
    }

    if (!mSplit)
    {
        return {std::move(transformed)};
    }

    std::vector<std::string> pieces;
    size_t begin = 0;
    while (begin < transformed.size())
    {
        size_t const next = transformed.find(mReplacement, begin + mReplacement.size());
        pieces.emplace_back(transformed.substr(begin, next - begin));
        if (next == std::string::npos)
        {
            break;
        }
        begin = next;
    }
    return pieces;
}

// Sequence implementation
Sequence::Sequence(std::vector<std::unique_ptr<PreTokenizer>> steps) noexcept
    : mSteps(std::move(steps))
{
    // Empty sequences are allowed - they act as pass-through
}

bool Sequence::prependsSpaceMarker() const noexcept
{
    return std::any_of(mSteps.begin(), mSteps.end(), [](auto const& step) { return step->prependsSpaceMarker(); });
}

std::vector<std::string> Sequence::process(std::string const& text) const
{
    if (text.empty())
    {
        return {};
    }

    if (text.size() > MAX_TEXT_SIZE_BYTES) // 1MB limit for consistency
    {
        throw std::runtime_error(
            "Sequence: Input text too large for processing: " + std::to_string(text.size()) + " bytes");
    }

    if (mSteps.empty())
    {
        return {text}; // Return text as-is if no steps
    }

    // Process text through the sequence of steps
    std::vector<std::string> currentPieces = {text};

    for (auto const& step : mSteps)
    {
        std::vector<std::string> nextPieces;
        nextPieces.reserve(currentPieces.size() * 2); // Reserve space to reduce allocations

        for (auto const& piece : currentPieces)
        {
            auto stepResult = step->process(piece);
            nextPieces.insert(nextPieces.end(), std::make_move_iterator(stepResult.begin()),
                std::make_move_iterator(stepResult.end()));
        }

        currentPieces = std::move(nextPieces);
    }

    return currentPieces;
}

void Sequence::addStep(std::unique_ptr<PreTokenizer> step)
{
    if (!step)
    {
        throw std::invalid_argument("Sequence::addStep: Cannot add null step");
    }

    mSteps.push_back(std::move(step));
}

PreTokenizer const* Sequence::getStep(size_t index) const noexcept
{
    if (index >= mSteps.size())
    {
        return nullptr;
    }
    return mSteps[index].get();
}

} // namespace tokenizer
} // namespace trt_edgellm
