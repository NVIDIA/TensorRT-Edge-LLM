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

#pragma once

// Avoid pulling <cuda.h> when nvcc already exposes version macros. XQA cubin generation uses a
// lightweight stdint shim that conflicts with CUDA's transitive stdint includes.
#if !defined(CUDA_VERSION) && !defined(__CUDACC_VER_MAJOR__)
#include <cuda.h>
#endif

// Normalize host CUDA_VERSION and nvcc's compiler-version macros into one value for feature gates.
#if defined(CUDA_VERSION)
#define EDGELLM_CUDA_VERSION CUDA_VERSION
#elif defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#define EDGELLM_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)
#endif

/*!
 * @brief CUDA feature detection helpers
 *
 * Centralizes CUDA-version-based feature macros to avoid scattering `#if CUDA_VERSION ...`
 * checks throughout the codebase.
 *
 * Usage example:
 * @code
 * #include "common/cudaMacros.h"
 *
 * #if SUPPORTS_FP8
 * // Safe to reference __nv_fp8_e4m3 and related helpers.
 * #endif
 * @endcode
 *
 * @note When FP8 is supported (CUDA >= 11.8), this header also includes `<cuda_fp8.h>`,
 * so callers do NOT need to conditionally include it themselves.
 */

#if defined(EDGELLM_CUDA_VERSION) && (EDGELLM_CUDA_VERSION >= 11080)
#include <cuda_fp8.h>
#define SUPPORTS_FP8 1
#else
#define SUPPORTS_FP8 0
#endif

#if defined(EDGELLM_CUDA_VERSION) && (EDGELLM_CUDA_VERSION >= 11080)
#define SUPPORTS_CLUSTER_LAUNCH 1
#else
#define SUPPORTS_CLUSTER_LAUNCH 0
#endif

#if defined(EDGELLM_CUDA_VERSION) && (EDGELLM_CUDA_VERSION >= 11080)
#define SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH 1
#else
#define SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH 0
#endif

#if defined(EDGELLM_CUDA_VERSION) && (EDGELLM_CUDA_VERSION >= 12080)
#include <cuda_fp4.h>
#define SUPPORTS_FP4 1
#else
#define SUPPORTS_FP4 0
#endif
