/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace trt_edgellm
{
namespace rt
{

//! How a runtime session is opened (LLMInferenceRuntime::openSession).
struct SessionOptions
{
    //! Whether the session may execute compact steps. Without it the session neither resolves compact execution
    //! nor reserves anything for it, and reports compact execution as unavailable.
    bool allowCompactExecution{false};
};

} // namespace rt
} // namespace trt_edgellm
