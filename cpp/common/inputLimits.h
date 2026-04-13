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
constexpr size_t kMaxContentItemsPerMessage = 64; // Alpamayo uses 4 cameras x 4 frames = 16 images + text items

} // namespace security

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
