/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

namespace trt_edgellm
{
namespace multimodal
{

//! Enum for supported multimodal model types
enum class ModelType
{
    QWEN2_VL,   //!< Qwen2-VL model
    QWEN2_5_VL, //!< Qwen2.5-VL model
    QWEN3_VL,   //!< Qwen3-VL model
    INTERNVL,   //!< InternVL model
    PHI4MM,     //!< Phi-4MM model
    UNKNOWN     //!< Unknown or unsupported model type
};

//! Convert string to ModelType enum
//! @param modelTypeStr String representation of model type
//! @return Corresponding ModelType enum value
inline ModelType stringToModelType(std::string const& modelTypeStr)
{
    if (modelTypeStr == "qwen2_vl")
        return ModelType::QWEN2_VL;
    if (modelTypeStr == "qwen2_5_vl")
        return ModelType::QWEN2_5_VL;
    if (modelTypeStr == "qwen3_vl")
        return ModelType::QWEN3_VL;
    if (modelTypeStr == "internvl" || modelTypeStr == "internvl_vision")
        return ModelType::INTERNVL;
    if (modelTypeStr == "phi4mm")
        return ModelType::PHI4MM;
    return ModelType::UNKNOWN;
}

} // namespace multimodal
} // namespace trt_edgellm
