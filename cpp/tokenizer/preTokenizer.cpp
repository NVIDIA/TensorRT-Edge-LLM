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

#include "preTokenizer.h"
#include "tokenizerUtils.h"
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

// Sequence implementation
Sequence::Sequence(std::vector<std::unique_ptr<PreTokenizer>> steps)
    : mSteps(std::move(steps))
{
    // Empty sequences are allowed - they act as pass-through
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
