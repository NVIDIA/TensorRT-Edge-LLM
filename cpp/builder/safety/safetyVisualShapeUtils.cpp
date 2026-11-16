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

#include "builder/safety/safetyVisualShapeUtils.h"

#include <stdexcept>

namespace trt_edgellm::safety::builder
{
int64_t getSafetyVisualTileCount(int64_t imageTokens, int64_t tokensPerTile)
{
    if (imageTokens <= 0 || tokensPerTile <= 0 || imageTokens % tokensPerTile != 0)
    {
        throw std::invalid_argument("Safety visual MSS imageTokens must be a positive multiple of tokens per tile.");
    }
    return imageTokens / tokensPerTile;
}
} // namespace trt_edgellm::safety::builder
