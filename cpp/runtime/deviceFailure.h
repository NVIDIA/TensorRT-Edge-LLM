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

#include <cuda_runtime_api.h>

#include <atomic>
#include <exception>
#include <string>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{

//! A failed request may be reused only after its stream drains successfully. The first failed drain is permanent.
class DeviceFailureState
{
public:
    void drainFailedRequest(cudaStream_t stream) noexcept
    {
        if (mError.load(std::memory_order_acquire) != cudaSuccess)
        {
            return;
        }
        cudaError_t const error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
        {
            cudaError_t expected = cudaSuccess;
            mError.compare_exchange_strong(expected, error, std::memory_order_release, std::memory_order_relaxed);
        }
    }

    std::string unavailableReason() const
    {
        cudaError_t const error = mError.load(std::memory_order_acquire);
        return error == cudaSuccess ? std::string{}
                                    : std::string(cudaGetErrorName(error)) + " (" + std::to_string(error)
                + "): CUDA request cleanup failed; restart the process";
    }

private:
    std::atomic<cudaError_t> mError{cudaSuccess};
};

//! Whether @p error is the exception a failed CUDA_CHECK or CUDA_DRIVER_CHECK throws (common/checkMacros.h). Those
//! throw plain std::runtime_error, so the message prefix is what identifies them; deviceFailureTests pins it.
inline bool isCudaCallFailure(std::exception const& error) noexcept
{
    std::string_view const message{error.what()};
    return message.rfind("CUDA runtime error in ", 0) == 0 || message.rfind("CUDA driver API error in ", 0) == 0;
}

//! Whether @p stream holds an error, such as a sticky error from a faulted kernel. Work still in flight is not one.
inline bool streamInError(cudaStream_t stream) noexcept
{
    cudaError_t const status = cudaStreamQuery(stream);
    return status != cudaSuccess && status != cudaErrorNotReady;
}

//! Whether @p error left the device state behind @p stream unknown: a CUDA call failed, or the stream holds an
//! error.
inline bool failedOnDevice(std::exception const& error, cudaStream_t stream) noexcept
{
    return isCudaCallFailure(error) || streamInError(stream);
}

//! Whether an enqueue on @p stream that reported failure failed on the device: the stream holds an error, or a CUDA
//! error is pending (a launch that failed). An enqueue refused before it launched any work leaves neither.
inline bool enqueueFailedOnDevice(cudaStream_t stream) noexcept
{
    return streamInError(stream) || cudaPeekAtLastError() != cudaSuccess;
}

} // namespace rt
} // namespace trt_edgellm
