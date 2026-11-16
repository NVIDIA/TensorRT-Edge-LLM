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

#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{

#ifdef SUPPORTS_DEBUG_TENSOR

/*!
 * @brief TensorRT IDebugListener that dumps intermediate tensors to safetensors files.
 *
 * Attach to an IExecutionContext whose engine was built with
 * markUnfusedTensorsAsDebugTensors() and call setUnfusedTensorsDebugState(true).
 * Output: <outputDir>/step_NNN_<label>.safetensors
 *
 * @note Incompatible with CUDA graph launch.
 */
class DebugTensorDumper : public nvinfer1::IDebugListener
{
public:
    //! @param outputDir Directory in which safetensors files are written.
    //!                  Created automatically if it does not exist.
    explicit DebugTensorDumper(std::filesystem::path outputDir);

    DebugTensorDumper(DebugTensorDumper const&) = delete;
    DebugTensorDumper& operator=(DebugTensorDumper const&) = delete;

    /*!
     * @brief Called by TensorRT after each unfused debug tensor is computed.
     *
     * Copies the tensor data from device to a CPU-owned Tensor and records it for
     * the current step.  @p addr is valid only for the duration of this call.
     * Duplicate tensor names within a step are silently skipped.
     *
     * @param addr     Device or host pointer to tensor data.
     * @param location kDEVICE or kHOST.
     * @param type     TensorRT data type.
     * @param shape    Shape of the tensor.
     * @param name     Internal TRT tensor name (matches IEngineInspector output).
     * @param stream   CUDA stream of the execution context.
     * @return true on success; false on error (TRT logs the failure).
     */
    bool processDebugTensor(void const* addr, nvinfer1::TensorLocation location, nvinfer1::DataType type,
        nvinfer1::Dims const& shape, char const* name, cudaStream_t stream) override;

    //! Label the upcoming enqueue and reset per-step state.
    //! @param label Appended to the output filename (e.g. "prefill", "decode").
    void beginStep(std::string const& label);

    //! Write collected tensors to a safetensors file and advance the step counter.
    //! Must be called after enqueueV3() + cudaStreamSynchronize().
    void flush(cudaStream_t stream);

private:
    std::filesystem::path mOutputDir;
    int32_t mStepIndex{0};
    std::string mStepLabel{"step"};
    std::vector<rt::Tensor> mPendingTensors;    //!< CPU-owned tensors collected this step
    std::unordered_set<std::string> mSeenNames; //!< Deduplication guard: names seen this step
};

#endif // SUPPORTS_DEBUG_TENSOR

} // namespace trt_edgellm
