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

#pragma once

#include <cstdint>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{

using Rank = std::int32_t;

// Size limit for tokenizer text processing (1MB)
constexpr size_t MAX_TEXT_SIZE_BYTES = 1024 * 1024;

/**
 * @brief Base class for pretokenizer steps
 */
class PreTokenizer
{
public:
    virtual ~PreTokenizer() noexcept = default;

    /**
     * @brief Process text and return split pieces
     * @param text Input text to process
     * @return Vector of text pieces after processing
     */
    virtual std::vector<std::string> process(std::string const& text) const = 0;

    /**
     * @brief Get the type name of this step
     * @return String identifying the step type
     */
    virtual std::string getTypeName() const = 0;

    //! Whether process() prepends a word marker to text that does not start with a space.
    virtual bool prependsSpaceMarker() const noexcept
    {
        return false;
    }
};

/**
 * @brief RegexSplit step that splits text using a regex pattern
 */
class RegexSplit : public PreTokenizer
{
public:
    /*!
     * @brief Constructor with regex pattern
     * @param pattern Regex pattern for splitting text
     * @throws std::invalid_argument if the pattern is empty
     * @throws std::runtime_error if the pattern is invalid
     */
    explicit RegexSplit(std::string const& pattern);
    ~RegexSplit() override = default;

    /*!
     * @brief Process text and return split pieces
     * @param text Input text to process
     * @return Vector of text pieces after processing
     * @throws std::runtime_error if the text is too large for regex processing
     * @throws std::runtime_error if Unicode text collapse fails
     * @throws std::runtime_error if Unicode regex split fails
     */
    std::vector<std::string> process(std::string const& text) const override;
    /*!
     * @brief Get the type name of this step
     * @return String identifying the step type
     */
    std::string getTypeName() const override
    {
        return "RegexSplit";
    }

    /*!
     * @brief Get the regex pattern
     * @return Reference to the pattern string
     */
    std::string const& getPattern() const noexcept
    {
        return mPattern;
    }

private:
    std::string mPattern;
    std::regex mRegex;
    bool mNeedRegexCollapse;
};

/**
 * @brief SentencePiece-style whitespace replacement used by Metaspace tokenizers.
 */
class Metaspace : public PreTokenizer
{
public:
    Metaspace(std::string replacement, std::string prependScheme, bool split);

    std::vector<std::string> process(std::string const& text) const override;

    std::string getTypeName() const override
    {
        return "Metaspace";
    }

    bool prependsSpaceMarker() const noexcept override
    {
        return mPrepend;
    }

private:
    std::string mReplacement;
    bool mPrepend{false};
    bool mSplit{true};
};

/**
 * @brief PreTokenizer class for splitting text before main tokenization
 * Now supports a sequence of processing steps
 */
class Sequence : public PreTokenizer
{
public:
    //! Default constructor - creates empty sequence (acts as pass-through)
    Sequence() noexcept = default;

    /*!
     * @brief Constructor with sequence of pretokenizer steps
     * @param steps Vector of pretokenizer steps to apply in order
     */
    explicit Sequence(std::vector<std::unique_ptr<PreTokenizer>> steps) noexcept;

    ~Sequence() noexcept = default;

    /*!
     * @brief Process text and return split pieces
     * @param text Input text to process
     * @return Vector of text pieces after processing
     * @throws std::runtime_error if the text is too large for processing
     */
    std::vector<std::string> process(std::string const& text) const override;

    /*!
     * @brief Get the type name of this step
     * @return String identifying the step type
     */
    std::string getTypeName() const override
    {
        return "Sequence";
    }

    bool prependsSpaceMarker() const noexcept override;

    /**
     * @brief Add a processing step to the sequence
     * @param step Unique pointer to the step to add
     * @throws std::invalid_argument if the step is null
     */
    void addStep(std::unique_ptr<PreTokenizer> step);

    /**
     * @brief Get the number of processing steps
     * @return Number of steps in the sequence
     */
    size_t getStepCount() const noexcept
    {
        return mSteps.size();
    }

    /**
     * @brief Get step at specified index
     * @param index Index of the step
     * @return Pointer to the step, or nullptr if index is invalid
     */
    PreTokenizer const* getStep(size_t index) const noexcept;

private:
    std::vector<std::unique_ptr<PreTokenizer>> mSteps;
};

} // namespace tokenizer
} // namespace trt_edgellm
