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

#include "common/dataType.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"

#include <NvInferRuntime.h>

namespace trt_edgellm
{

inline nvinfer1::DataType toTrt(DataType type) noexcept
{
    return static_cast<nvinfer1::DataType>(type);
}

inline DataType fromTrt(nvinfer1::DataType type) noexcept
{
    return static_cast<DataType>(type);
}

namespace rt
{

static_assert(kMAX_DIMS <= nvinfer1::Dims::MAX_DIMS);

inline nvinfer1::Dims toTrtDims(Coords const& coords) noexcept
{
    nvinfer1::Dims dims{};
    dims.nbDims = coords.getNumDims();
    for (int32_t i = 0; i < coords.getNumDims(); ++i)
    {
        dims.d[i] = coords[i];
    }
    return dims;
}

inline nvinfer1::Dims toTrtDims(Tensor const& tensor) noexcept
{
    return toTrtDims(tensor.getShape());
}

inline Coords toCoords(nvinfer1::Dims const& dims)
{
    return Coords(dims.d, dims.d + dims.nbDims);
}

} // namespace rt

//! The nvinfer1::ILogger handed to TensorRT; forwards to gLogger. Immortal for the same reason as EdgeLLMLogger:
//! TensorRT's process-lifetime plugin registry keeps a pointer to it.
class TrtLogger : public nvinfer1::ILogger
{
public:
    static TrtLogger& instance() noexcept;

    void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
    {
        logger::EdgeLLMLogger::instance().log(static_cast<logger::Severity>(severity), msg);
    }

    //! Virtual so that a plugin library, which has its own gLogger, reads the level of the module that created this
    //! logger.
    virtual logger::Severity getLevel() const noexcept
    {
        return logger::EdgeLLMLogger::instance().getLevel();
    }

private:
    TrtLogger() = default;
};

} // namespace trt_edgellm
