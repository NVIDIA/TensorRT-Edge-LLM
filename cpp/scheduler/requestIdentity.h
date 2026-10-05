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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{
//! Request scheduling: the engine that owns a runtime, the handles callers hold, and the channels
//! between them. These types live in their own namespace because `rt` already has a
//! `ContextCacheCoordinator::RequestHandle`, which is unrelated to the handle callers hold.
namespace scheduler
{

//! Assigned by RequestEngine::submit() and never reused.
//!
//! Reuse would be a correctness bug, not just untidy: cancel() is asynchronous, so a request may
//! already have retired by the time the actor drains the command. A recycled id would let a late
//! cancel terminate a different request. A monotonic 64-bit counter makes a stale cancel a lookup
//! miss instead.
using RequestId = uint64_t;

//! Reserved value for "no request"; submit() never returns it.
inline constexpr RequestId kInvalidRequestId = 0;

//! How a request stopped, from the actor's point of view.
//!
//! Coarser than `FinishReason` in streaming.h, which describes the generation outcome carried on
//! each chunk: a request whose status is kCompleted may still carry kEndId, kLength or kStopWords.
enum class TerminalStatus : uint8_t
{
    //! Generation ended on its own terms.
    kCompleted,
    //! A cancel command was drained before generation ended.
    kCancelled,
    //! The step that carried this request failed.
    kExecutionError,
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
