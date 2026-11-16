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

//! Tensor element type. The values equal nvinfer1::DataType's, so the TensorRT boundary converts with a cast
//! (common/trtInterop.h checks every value).
enum class DataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
    kINT4 = 9,
    kFP4 = 10,
    kE8M0 = 11,
};

//! Short, human-readable name for a data type (e.g. "FLOAT16"). Used in logs and error messages; not intended for
//! serialization.
constexpr char const* getDataTypeString(DataType const dataType) noexcept
{
    switch (dataType)
    {
    case DataType::kINT64: return "INT64";
    case DataType::kINT32: return "INT32";
    case DataType::kFLOAT: return "FLOAT32";
    case DataType::kHALF: return "FLOAT16";
    case DataType::kBF16: return "BFLOAT16";
    case DataType::kFP8: return "FLOAT8_E4M3";
    case DataType::kINT8: return "INT8";
    case DataType::kUINT8: return "UINT8";
    case DataType::kBOOL: return "BOOL";
    case DataType::kE8M0: return "FLOAT8_E8M0";
    case DataType::kFP4: return "FLOAT4";
    case DataType::kINT4: return "INT4";
    }
    return "UNKNOWN";
}

} // namespace trt_edgellm
