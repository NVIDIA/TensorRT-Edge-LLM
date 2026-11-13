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

/*!
 * @brief Register an NCCL communicator with the plugin communication registry.
 * @return True if the plugin symbol was found and registration completed.
 */
bool registerNcclCommForAllReducePlugin(int32_t deviceId, void* ncclComm) noexcept;

/*!
 * @brief Unregister an NCCL communicator from the plugin communication registry.
 * @return True if the matching registration was found and removed.
 */
bool unregisterNcclCommForAllReducePlugin(int32_t deviceId, void* ncclComm) noexcept;


} // namespace rt
} // namespace trt_edgellm
