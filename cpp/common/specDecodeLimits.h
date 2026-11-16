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

//! Verification tokens (tree nodes, root included) one speculative-decoding step may send through the base engine.
//! Every tree component is sized to it: the DDTree builder's node budget, the MTP / DSpark tree checks in
//! DeploymentConfig, the speculative tree kernels (one thread per node, a packed N / 32-word mask per node), and the
//! decode attention kernels' tree-mask width (ceil(N / 32) int32 words per query row).
//! Raising it means re-validating all of them; the kernels do not check it on their own.
constexpr int32_t kMAX_SPEC_VERIFY_TREE_TOKENS{128};

} // namespace trt_edgellm
