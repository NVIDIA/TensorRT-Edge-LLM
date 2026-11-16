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

#include "runtime/embedding/pleFileUtils.h"
#include "runtime/embedding/pleReadPlan.h"

#include <memory>

namespace trt_edgellm
{
namespace rt
{

//! Fixed PLE readers with pinned scratch buffers. Borrowed file handles outlive this executor.
class PleReadExecutor
{
public:
    PleReadExecutor(std::vector<PleFileHandle> const& files, size_t readerCount, size_t maxReadBytes);
    ~PleReadExecutor() noexcept;
    PleReadExecutor(PleReadExecutor const&) = delete;
    PleReadExecutor& operator=(PleReadExecutor const&) = delete;

    //! Execute a validated plan with disjoint destinations and ranges fitting maxReadBytes.
    //! Calls are serialized by the owner.
    //! All readers finish before return or throw; counters include completed partial reads on failure.
    void read(std::vector<PleReadRange> const& ranges, char* packed, size_t& readCalls, size_t& readBytes);

private:
    struct State;
    std::unique_ptr<State> mState;
};

} // namespace rt
} // namespace trt_edgellm
