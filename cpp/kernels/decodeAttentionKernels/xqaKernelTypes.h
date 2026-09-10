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

// Shared CACHE_ELEM_ENUM protocol for the host JIT compiler and NVRTC-compiled XQA source.
// Keep these as macros so kernel source can use them in preprocessor conditions.
#define TRT_EDGELLM_XQA_CACHE_ELEM_INPUT 0
#define TRT_EDGELLM_XQA_CACHE_ELEM_INT8 1
#define TRT_EDGELLM_XQA_CACHE_ELEM_FP8_E4M3 2

namespace xqa
{
namespace kernels
{

enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

struct XQAKernelMetaInfo
{
    enum XQAKernelVariant
    {
        KERNEL_VARIANT_STANDARD,
        KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512,
        KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4,
        KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512,
        KERNEL_VARIANT_2CTA_HEAD_DIM512,
    };
};

constexpr int kKV_CACHE_ELEM_INPUT{TRT_EDGELLM_XQA_CACHE_ELEM_INPUT};
constexpr int kKV_CACHE_ELEM_INT8{TRT_EDGELLM_XQA_CACHE_ELEM_INT8};
constexpr int kKV_CACHE_ELEM_FP8_E4M3{TRT_EDGELLM_XQA_CACHE_ELEM_FP8_E4M3};

} // namespace kernels
} // namespace xqa
