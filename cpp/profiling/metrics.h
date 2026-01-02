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
#include <string>

namespace trt_edgellm
{

//! \cond INTERNAL
//! Global profiling control flag accessors (defined in timer.cpp)
//! When false, no profiling data (metrics or timing) will be recorded
bool getProfilingEnabled();
void setProfilingEnabled(bool enabled);
//! \endcond

namespace metrics
{

/*!
 * @brief Stage name constants
 *
 * Centralized stage names to avoid hardcoding strings.
 */
namespace StageNames
{
inline std::string const kLLM_PREFILL = "llm_prefill";                               //!< LLM prefill stage
inline std::string const kLLM_GENERATION = "llm_generation";                         //!< LLM generation stage
inline std::string const kMULTIMODAL_PROCESSING = "multimodal_processing";           //!< Multimodal processing stage
inline std::string const kEAGLE_DRAFT_PREFILL = "eagle_draft_prefill";               //!< Eagle draft prefill stage
inline std::string const kEAGLE_CONSTRUCT_DRAFT_TREE = "eagle_construct_draft_tree"; //!< Eagle draft tree construction
inline std::string const kEAGLE_BASE_VERIFICATION = "eagle_base_verification";       //!< Eagle base verification stage
} // namespace StageNames

/*!
 * @brief Base class for performance metrics
 *
 * Provides common interface and total runs tracking.
 */
class BaseMetrics
{
public:
    //! @brief Virtual destructor
    virtual ~BaseMetrics() = default;

    //! @brief Get total number of runs
    //! @return Total runs count
    int64_t getTotalRuns() const
    {
        return totalRuns;
    }

protected:
    int64_t totalRuns{0}; //!< Total number of recorded runs
};

/*!
 * @brief LLM prefill stage metrics
 *
 * Tracks reused and computed tokens during prefill.
 */
class LLMPrefillMetrics : public BaseMetrics
{
public:
    int64_t reusedTokens{0};   //!< Number of reused tokens from cache
    int64_t computedTokens{0}; //!< Number of newly computed tokens

    //! @brief Record a prefill run
    //! @param reused Number of reused tokens
    //! @param computed Number of computed tokens
    void recordRun(int64_t reused, int64_t computed)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        reusedTokens += reused;
        computedTokens += computed;
    }
};

/*!
 * @brief LLM generation stage metrics
 *
 * Tracks generated tokens during decoding.
 */
class LLMGenerationMetrics : public BaseMetrics
{
public:
    int64_t generatedTokens{0}; //!< Total number of generated tokens

    //! @brief Record a generation run
    //! @param generated Number of generated tokens
    void recordRun(int64_t generated)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        generatedTokens += generated;
    }
};

/*!
 * @brief Multimodal processing stage metrics
 *
 * Tracks image processing statistics.
 */
class MultimodalMetrics : public BaseMetrics
{
public:
    int64_t totalImages{0};      //!< Total number of processed images
    int64_t totalImageTokens{0}; //!< Total number of image tokens generated

    //! @brief Record a multimodal processing run
    //! @param imageCount Number of images processed
    //! @param imageTokens Number of image tokens generated
    void recordRun(int64_t imageCount, int64_t imageTokens)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalImages += imageCount;
        totalImageTokens += imageTokens;
    }
};

/*!
 * @brief Eagle speculative decoding generation metrics
 *
 * Tracks iterations and tokens generated during Eagle spec-decode.
 */
class EagleGenerationMetrics : public BaseMetrics
{
public:
    int64_t totalIterations{0};      //!< Total number of Eagle iterations
    int64_t totalGeneratedTokens{0}; //!< Total number of generated tokens

    //! @brief Record an Eagle generation run
    //! @param iterations Number of iterations
    //! @param generatedTokens Number of generated tokens
    void recordRun(int64_t iterations, int64_t generatedTokens)
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalIterations += iterations;
        totalGeneratedTokens += generatedTokens;
    }
};

} // namespace metrics
} // namespace trt_edgellm
