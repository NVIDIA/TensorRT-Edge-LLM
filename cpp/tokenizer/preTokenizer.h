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
    virtual ~PreTokenizer() = default;

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
     */
    explicit RegexSplit(std::string const& pattern);
    ~RegexSplit() override = default;

    std::vector<std::string> process(std::string const& text) const override;
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
 * @brief PreTokenizer class for splitting text before main tokenization
 * Now supports a sequence of processing steps
 */
class Sequence : public PreTokenizer
{
public:
    //! Default constructor - creates empty sequence (acts as pass-through)
    Sequence() = default;

    /*!
     * @brief Constructor with sequence of pretokenizer steps
     * @param steps Vector of pretokenizer steps to apply in order
     */
    explicit Sequence(std::vector<std::unique_ptr<PreTokenizer>> steps);

    ~Sequence() = default;

    std::vector<std::string> process(std::string const& text) const override;
    std::string getTypeName() const override
    {
        return "Sequence";
    }

    /**
     * @brief Add a processing step to the sequence
     * @param step Unique pointer to the step to add
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
