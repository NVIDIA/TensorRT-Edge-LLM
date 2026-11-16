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

#include <cstddef>

namespace trt_edgellm::limits
{

/*!
 * @brief Centralized input limits for validation and resource protection.
 */

// Location string for error messages to reference this file.
constexpr char const* kInputLimitsLocation = "cpp/common/inputLimits.h";

namespace security
{

/*!
 * @brief DoS-prevention and unsafe input rejection limits.
 */

// Reasonable upper bound to avoid excessive memory allocation before runtime init.
constexpr int kReasonableMaxBatchSize = 16;

// Validation limits for message parsing.
constexpr size_t kMaxMessageContentSizeBytes = 128 * 1024; // 128KB per content item
constexpr size_t kMaxMessagesPerRequest = 64;
constexpr size_t kMaxContentItemsPerMessage = 18;
// Match vLLM's MAX_NUM_LOGIT_BIAS_TOKENS sparse logit-bias guardrail.
constexpr size_t kMaxLogitBiasTokens = 1024;
constexpr float kMinLogitBias = -100.0F;
constexpr float kMaxLogitBias = 100.0F;
// Guided-decoding guide string (JSON schema / regex / EBNF / structural tag).
// Grammar compilation is superlinear in guide size and runs synchronously on the
// host before any GPU work, so an unbounded guide is a denial-of-service vector.
constexpr size_t kMaxGuidedDecodingGuideBytes = 128 * 1024;

} // namespace security

namespace multimodal
{

/*!
 * @brief Image input limits.
 */

// Row scratch bound for the image content digest (rt::imageUtils::launchImageContentDigest). The row count
// depends on frame count, height and plane count only, so this bounds arbitrarily wide frames; an
// image past it has no digest, which takes its request out of both media caches.
constexpr int64_t kMaxImageDigestRows = 2 * 1000 * 1000; // 32MB of pinned host scratch at 16B/row

} // namespace multimodal

namespace tokenizer
{

/*!
 * @brief Tokenizer file size and text processing limits.
 */

// File size limits for tokenizer-related configuration.
constexpr size_t kMaxConfigFileSizeBytes = 100 * 1024 * 1024; // 100MB limit for config files
constexpr size_t kChatTemplateFileSizeBytes = 1024 * 1024;    // 1MB limit for chat template file

// Input validation limits for tokenization.
// Raw text before tokenization. Each token is approx 4 bytes on average, so 1MB text -> ~256K tokens.
constexpr size_t kMaxInputTextSizeBytes = 1024 * 1024;  // 1MB limit for input text before tokenization
constexpr size_t kMaxTokenPieceSizeBytes = 1024 * 1024; // 1MB limit for token encoder piece size

} // namespace tokenizer

} // namespace trt_edgellm::limits
