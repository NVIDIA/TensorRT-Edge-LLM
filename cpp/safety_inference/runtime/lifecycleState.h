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

namespace trt_edgellm::safety
{

//! Shared per-runtime lifecycle that gates permitted operations; calls are externally serialized.
//! SafetyStatus carries each operation's result and fatal flag; fatal errors make this lifecycle terminal.
enum class LifecycleState
{
    kINITIALIZATION,
    kOPERATIONAL,
    kCLOSED, //!< Execution is closed; outstanding owned resources may finish healthy teardown.
    kFAILED, //!< Terminal lifecycle; SafetyStatus retains the error category and diagnostic.
};

} // namespace trt_edgellm::safety
