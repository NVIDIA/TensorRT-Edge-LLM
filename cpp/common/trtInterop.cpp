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

#include "common/trtInterop.h"

#define EDGELLM_CHECK_TRT_DATATYPE(name)                                                                               \
    static_assert(static_cast<int32_t>(trt_edgellm::DataType::name) == static_cast<int32_t>(nvinfer1::DataType::name))
EDGELLM_CHECK_TRT_DATATYPE(kFLOAT);
EDGELLM_CHECK_TRT_DATATYPE(kHALF);
EDGELLM_CHECK_TRT_DATATYPE(kINT8);
EDGELLM_CHECK_TRT_DATATYPE(kINT32);
EDGELLM_CHECK_TRT_DATATYPE(kBOOL);
EDGELLM_CHECK_TRT_DATATYPE(kUINT8);
EDGELLM_CHECK_TRT_DATATYPE(kFP8);
EDGELLM_CHECK_TRT_DATATYPE(kBF16);
EDGELLM_CHECK_TRT_DATATYPE(kINT64);
EDGELLM_CHECK_TRT_DATATYPE(kINT4);
#if IS_TRT_RTX || NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 8)
EDGELLM_CHECK_TRT_DATATYPE(kFP4);
#endif
#if IS_TRT_RTX || NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 12)
EDGELLM_CHECK_TRT_DATATYPE(kE8M0);
static_assert(nvinfer1::EnumMax<nvinfer1::DataType>() == 12, "Add the new nvinfer1::DataType to trt_edgellm::DataType");
#endif
#undef EDGELLM_CHECK_TRT_DATATYPE

namespace trt_edgellm
{

static_assert(static_cast<int32_t>(logger::Severity::kINTERNAL_ERROR)
    == static_cast<int32_t>(nvinfer1::ILogger::Severity::kINTERNAL_ERROR));
static_assert(
    static_cast<int32_t>(logger::Severity::kERROR) == static_cast<int32_t>(nvinfer1::ILogger::Severity::kERROR));
static_assert(
    static_cast<int32_t>(logger::Severity::kWARNING) == static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
static_assert(
    static_cast<int32_t>(logger::Severity::kINFO) == static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO));
static_assert(
    static_cast<int32_t>(logger::Severity::kVERBOSE) == static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE));

TrtLogger& TrtLogger::instance() noexcept
{
    static TrtLogger* sInstance = new TrtLogger();
    return *sInstance;
}

} // namespace trt_edgellm
