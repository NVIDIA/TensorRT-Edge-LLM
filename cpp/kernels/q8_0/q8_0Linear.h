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

#include "common/dataType.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace trt_edgellm::kernel
{

//! Version of the Q8_1 activation workspace layout and size. Engines record it, because
//! IPluginV3 enqueue cannot see the workspace size TensorRT reserved at build time; bump it
//! whenever q8_0WorkspaceSize() or the layout changes.
constexpr int32_t kQ8_0_ACTIVATION_LAYOUT_VERSION = 2;
//! Most packed weights one q8_0Linear call accepts.
constexpr int32_t kQ8_0_MAX_WEIGHTS = 4;

//! Packed Q8_0 linear. FP16 or BF16 activations are quantized to Q8_1 once in
//! the workspace; checkpoint weights remain packed on device. Weight w writes
//! columns [sum(outFeatures[0..w)), + outFeatures[w]) of the [rows,
//! sum(outFeatures)] output, which is FP16, BF16 or FP32. The workspace must be
//! 16-byte aligned.
size_t q8_0WorkspaceSize(int32_t rows, int32_t inFeatures);
cudaError_t q8_0Linear(void const* input, DataType inputType, void const* const* packedWeights,
    int32_t const* outFeatures, int32_t numWeights, void* output, DataType outputType, void* workspace,
    size_t workspaceSize, int32_t rows, int32_t inFeatures, cudaStream_t stream);

//! Kernel path selection. kAuto is what q8_0Linear uses; the forced paths exist so tests can
//! check every kernel on one device. kMma and kTiled return cudaErrorNotSupported when the
//! device, row count or weight geometry cannot run them.
enum class Q8_0Path
{
    kAuto,
    kGemv,
    kMma,
    kTiled,
};
//! Largest row count for which q8_0Linear runs the GEMV rather than the tensor-core GEMM on the current device,
//! for weights with outFeatures output columns in total (below the 32-row tiled path).
cudaError_t q8_0GemvMaxRows(int32_t outFeatures, int32_t& maxRows);
cudaError_t q8_0LinearForPath(void const* input, DataType inputType, void const* const* packedWeights,
    int32_t const* outFeatures, int32_t numWeights, void* output, DataType outputType, void* workspace,
    size_t workspaceSize, int32_t rows, int32_t inFeatures, Q8_0Path path, cudaStream_t stream);

} // namespace trt_edgellm::kernel
