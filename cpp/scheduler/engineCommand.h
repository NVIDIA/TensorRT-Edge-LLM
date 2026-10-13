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

#include "runtime/llmRuntimeUtils.h"
#include "scheduler/requestIdentity.h"
#include "scheduler/requestRecord.h"

#include <cstdint>
#include <memory>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! How far shutdown lets in-flight work run.
enum class ShutdownMode : uint8_t
{
    //! Let every admitted request run to its own terminal state, then stop.
    kDrain,
    //! Terminate queued and resident requests as kCancelled, then stop.
    kCancel,
};

enum class CommandType : uint8_t
{
    kSubmit,
    kShutdown,
};

//! One instruction posted to the actor.
//!
//! Commands are heap-allocated by the producer and normally destroyed by the actor. That hand-off
//! needs no protection because there is exactly one consumer, and it keeps the ring's cells small:
//! the request payload never moves through the ring itself. See CommandQueue for the two paths that
//! destroy a command somewhere other than the actor.
struct EngineCommand
{
    CommandType type{CommandType::kSubmit};

    //! kSubmit only. Unset for kShutdown. (Cancellation does not travel as a command: it is a flag
    //! planted on the record, see EngineChannel::cancelRecord.)
    RequestId requestId{kInvalidRequestId};

    //! kSubmit only. Owned; the actor takes it when the request is admitted.
    std::unique_ptr<LLMGenerationRequest> request;

    //! kSubmit only. Shared with the RequestHandle the caller already holds, so the actor can
    //! publish an outcome even if the caller drops the handle mid-flight.
    std::shared_ptr<RequestRecord> record;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
