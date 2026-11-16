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

#include "common/pi05Common.h"
#include "common/tensor.h"

#include <cuda_runtime.h>
#include <string>

namespace trt_edgellm
{
namespace pi05
{

//! \brief The pi0.5 SigLIP tower: preprocessed camera views in, prefix-width image
//! features out. Runs once per request, ahead of prefix assembly.
//!
//! Single-stream by construction, like the runtime that owns it: the constructor takes
//! the only stream this instance enqueues, copies or records on. NOT thread-safe.
class Pi05VisualRunner
{
public:
    //! \param engineDir Engine root; the tower is loaded from its visual/ subdirectory.
    //! \param config The parsed contract of the export \p engineDir holds.
    //! \param stream The instance's only stream, for allocation-time work and every request.
    Pi05VisualRunner(std::string const& engineDir, Pi05PolicyConfig const& config, cudaStream_t stream);
    ~Pi05VisualRunner() noexcept;

    int64_t getRequiredContextMemorySize() const
    {
        return mEngine.getRequiredContextMemorySize();
    }

    bool setContextMemory(rt::Tensor& pool)
    {
        return mEngine.setContextMemory(pool);
    }

    //! \brief Run the tower over the preprocessed views.
    //! \param pixelValues Device FLOAT16 [numViews, 3, imageSize, imageSize] in [-1, 1].
    //! \return Device FLOAT16 [numViews, numImageTokens, hiddenSize], owned by this runner.
    rt::Tensor const& encode(rt::Tensor const& pixelValues);

    //! \brief Stream time of the last encode(), in ms; 0 before the stream has drained it.
    float getElapsedMs() const noexcept;

private:
    Pi05PolicyConfig mConfig;
    cudaStream_t mStream{nullptr};
    Pi05Component mEngine;
    cudaEvent_t mBegin{nullptr};
    cudaEvent_t mEnd{nullptr};
    rt::Tensor mImageFeatures;
};

} // namespace pi05
} // namespace trt_edgellm
