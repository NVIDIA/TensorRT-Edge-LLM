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
#include <string>

namespace trt_edgellm
{

//! \cond INTERNAL
//! Global profiling control flag accessors (defined in timer.cpp)
//! When false, no profiling data (metrics or timing) will be recorded
bool getProfilingEnabled() noexcept;
void setProfilingEnabled(bool enabled) noexcept;
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
inline std::string const kLLM_PREFILL = "llm_prefill";                     //!< LLM prefill stage
inline std::string const kLLM_GENERATION = "llm_generation";               //!< LLM generation stage
inline std::string const kLLM_LAYER = "llm_layer";                         //!< LLM layer profiling
inline std::string const kMULTIMODAL_PROCESSING = "multimodal_processing"; //!< Multimodal processing stage (legacy)
inline std::string const kAUDIO_ENCODER = "audio_encoder";                 //!< Audio encoder stage
inline std::string const kVISION_ENCODER = "vision_encoder";               //!< Vision encoder stage
inline std::string const kSPEC_DECODE_DRAFT_PREFILL = "spec_decode_draft_prefill"; //!< Speculative decode draft prefill
inline std::string const kSPEC_DECODE_DRAFT_PROPOSAL
    = "spec_decode_draft_proposal";                                              //!< Speculative decode draft proposal
inline std::string const kSPEC_DECODE_DRAFT_ACCEPT = "spec_decode_draft_accept"; //!< Speculative decode draft accept
inline std::string const kSPEC_DECODE_BASE_VERIFICATION
    = "spec_decode_base_verification";                             //!< Speculative decode base verification
inline std::string const kCODE2WAV = "code2wav";                   //!< Code2Wav vocoder stage
inline std::string const kTALKER_GENERATION = "talker_generation"; //!< Talker audio frame generation
inline std::string const kCODE_PREDICTOR
    = "code_predictor"; //!< CodePredictor RVQ code generation (legacy aggregate; superseded by
                        //!< kCODEPREDICTOR_PREFILL/kCODEPREDICTOR_GENERATION)
inline std::string const kCODEPREDICTOR_PREFILL = "codepredictor_prefill"; //!< CodePredictor prefill (per-frame)
inline std::string const kCODEPREDICTOR_GENERATION
    = "codepredictor_generation";                                //!< CodePredictor generation loop (per-frame)
inline std::string const kTALKER_PREFILL = "talker_prefill";     //!< Talker prefill stage (single-shot)
inline std::string const kACTION_INFERENCE = "action_inference"; //!< Action head trajectory sampling stage
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
    virtual ~BaseMetrics() noexcept = default;

    //! @brief Get total number of runs
    //! @return Total runs count
    int64_t getTotalRuns() const noexcept
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
    void recordRun(int64_t reused, int64_t computed) noexcept
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
    void recordRun(int64_t generated) noexcept
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
 * Tracks image and audio processing statistics.
 */
class MultimodalMetrics : public BaseMetrics
{
public:
    int64_t totalImages{0};      //!< Total number of processed images
    int64_t totalImageTokens{0}; //!< Total number of image tokens generated
    int64_t totalAudios{0};      //!< Total number of processed audio clips (Qwen3-Omni)
    int64_t totalAudioTokens{0}; //!< Total number of audio tokens generated (Qwen3-Omni)

    //! @brief Record a multimodal processing run
    //! @param imageCount Number of images processed
    //! @param imageTokens Number of image tokens generated
    //! @param audioCount Number of audio clips processed (optional, for Qwen3-Omni)
    //! @param audioTokens Number of audio tokens generated (optional, for Qwen3-Omni)
    void recordRun(int64_t imageCount, int64_t imageTokens, int64_t audioCount = 0, int64_t audioTokens = 0) noexcept
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalImages += imageCount;
        totalImageTokens += imageTokens;
        totalAudios += audioCount;
        totalAudioTokens += audioTokens;
    }
};

/*!
 * @brief Speculative decoding generation metrics
 *
 * Tracks iterations and tokens generated during speculative decoding.
 */
class SpecDecodeGenerationMetrics : public BaseMetrics
{
public:
    int64_t totalIterations{0};      //!< Total number of speculative decoding iterations
    int64_t totalGeneratedTokens{0}; //!< Total number of generated tokens, including base-prefill tokens
    int64_t totalAcceptedTokens{0};  //!< Total tokens produced by speculative decoding iterations

    //! @brief Record a speculative decoding generation run
    //! @param iterations Number of iterations
    //! @param generatedTokens Number of generated tokens
    //! @param acceptedTokens Number of tokens produced by speculative decoding iterations
    void recordRun(int64_t iterations, int64_t generatedTokens, int64_t acceptedTokens) noexcept
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalIterations += iterations;
        totalGeneratedTokens += generatedTokens;
        totalAcceptedTokens += acceptedTokens;
    }
};

/*!
 * @brief Omni Talker pipeline metrics
 *
 * Tracks audio frame generation, RVQ codes, prefill time, and exit reason.
 */
class OmniTalkerMetrics : public BaseMetrics
{
public:
    int64_t totalFrames{0};      //!< Total audio frames generated (each frame = numCodesPerFrame RVQ codes)
    int64_t totalRvqCodes{0};    //!< Total RVQ codes generated (frames * codesPerFrame)
    float prefillGpuTimeMs{0};   //!< Talker prefill GPU time in milliseconds
    int32_t prefillSeqLength{0}; //!< Talker prefill input sequence length
    std::string exitReason;      //!< "eos" or "max_length"
    bool isStreaming{false};     //!< Whether streaming mode was used

    void recordRun(int64_t frames, int64_t rvqCodes, float prefillMs, int32_t prefillSeqLen, std::string const& exit,
        bool streaming) noexcept
    {
        if (!getProfilingEnabled())
        {
            return;
        }
        totalRuns++;
        totalFrames += frames;
        totalRvqCodes += rvqCodes;
        prefillGpuTimeMs = prefillMs;
        prefillSeqLength = prefillSeqLen;
        exitReason = exit;
        isStreaming = streaming;
    }
};

/*!
 * @brief Omni audio latency metrics
 *
 * Tracks time to first audio code (TTFA), real-time factor (RTF), and audio output info.
 * Time to first playable audio (TTFPA) is derived at JSON output time from
 * talker_generation + code2wav stage times.
 */
struct OmniLatencyMetrics
{
    float timeToFirstAudioCodeMs{0};     //!< Request start to first codec token sampled (includes Thinker)
    float timeToFirstPlayableAudioMs{0}; //!< Request start to first playable audio chunk complete
    float endToEndMs{0};                 //!< Request start to all audio output complete
    float realTimeFactor{0};             //!< audio_duration / talker_generation_time (< 1.0 = faster than real-time)
    float audioDurationSeconds{0};       //!< Total audio output duration in seconds
    int64_t audioSamples{0};             //!< Total audio output samples
    int32_t sampleRate{24000};           //!< Audio sample rate
};

} // namespace metrics
} // namespace trt_edgellm
