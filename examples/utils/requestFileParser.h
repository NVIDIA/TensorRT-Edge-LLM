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

#include "runtime/llmRuntimeUtils.h"
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace exampleUtils
{

/*!
 * @brief Parse the llm_inference-style JSON input file.
 *
 * Reads a JSON file with the following top-level schema:
 *   {
 *     "batch_size": int,                      // optional, default 1
 *     "temperature": float,                   // optional, default 1.0
 *     "top_p": float,                         // optional, default 0.8
 *     "top_k": int,                           // optional, default 50
 *     "sampling_seed": uint64,                 // optional stable default seed
 *     "spec_proposal_sampling": str,           // auto (default), greedy, or probabilistic
 *     "logit_bias": {token_id: bias},         // optional default for all requests
 *     "max_generate_length": int,             // optional, default 256
 *     "diffusion_config": {                    // optional DiffusionGemma runtime overrides
 *       "max_denoising_steps": int             // optional, 0/default uses runtime default capped by engine config
 *     },
 *     "num_logprobs": int,                    // optional default for all requests, 0..kMaxLogprobsK
 *     "apply_chat_template": bool,            // optional, default true
 *     "add_generation_prompt": bool,          // optional, default true
 *     "enable_thinking": bool,                // optional, default false
 *     "context_cache_lookup_policy": str,     // optional: "use_cache" (default) or "bypass"
 *     "context_cache_commit_policy": str,     // optional: "including_generated_tokens" (default) or
 *                                              //           "prefill_state_only"
 *     "available_lora_weights": {name: path}, // optional
 *     "requests": [
 *       {
 *         "lora_name": str,                   // optional
 *         "logit_bias": {token_id: bias},     // optional, overrides top-level default
 *         "num_logprobs": int,                // optional, overrides top-level default (batch runs at max)
 *         "save_system_prompt_kv_cache": bool,
 *         "disable_spec_decode": bool,
 *         "sampling_seed": uint64,             // optional per-request override,
 *         "context_cache_lookup_policy": str,  // optional per-request override,
 *         "messages": [
 *           {"role": "...", "content": "..."},
 *           {"role": "...", "content": [{"type":"text", "text":"..."},
 *                                       {"type":"image", "image":"/path/to/img"},
 *                                       {"type":"audio", "audio":"/path/to/audio.wav"}]}
 *         ]
 *       }
 *     ]
 *   }
 *
 * Requests are grouped into batches of size `batch_size` (or `batchSizeOverride` if >= 0).
 * `maxGenerateLengthOverride` (>= 0) takes precedence over the value in the file.
 * `numLogprobsOverride` (>= 0) takes precedence over both the top-level and per-request
 * `num_logprobs` values in the file (same override convention as the other two).
 *
 * @return {loraWeightsMap, batchedRequests}
 * @throws std::runtime_error on parse errors, missing required fields, or limits violations.
 */
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseRequestFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1,
    int32_t numLogprobsOverride = -1);

} // namespace exampleUtils
} // namespace trt_edgellm
