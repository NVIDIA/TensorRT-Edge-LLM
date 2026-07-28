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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tensor.h"
#include "runtime/wfmRuntimeUtils.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{

//! Cosmos MoT backbone engine: und_seq + gen_seq + rotary -> last_hidden_state.
class MotBackboneRunner
{
public:
    //! \p engineDir is the ``mot_backbone/`` component directory (contains config.json + engine).
    explicit MotBackboneRunner(std::string const& engineDir, cudaStream_t stream);

    ~MotBackboneRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    std::string const& getOutputName() const noexcept
    {
        return mOutputName;
    }

    rt::Coords const& getUndSeqShape() const noexcept
    {
        return mUndSeqShape;
    }

    rt::Coords const& getGenSeqShape() const noexcept
    {
        return mGenSeqShape;
    }

    rt::Coords const& getLastHiddenShape() const noexcept
    {
        return mOutputShape;
    }

    rt::Tensor& getUndSeq() noexcept
    {
        return mUndSeqTensor;
    }

    rt::Tensor const& getUndSeq() const noexcept
    {
        return mUndSeqTensor;
    }

    rt::Tensor& getGenSeq() noexcept
    {
        return mGenSeqTensor;
    }

    rt::Tensor const& getGenSeq() const noexcept
    {
        return mGenSeqTensor;
    }

    rt::Tensor& getCosUnd() noexcept
    {
        return mCosUndTensor;
    }

    rt::Tensor const& getCosUnd() const noexcept
    {
        return mCosUndTensor;
    }

    rt::Tensor& getSinUnd() noexcept
    {
        return mSinUndTensor;
    }

    rt::Tensor const& getSinUnd() const noexcept
    {
        return mSinUndTensor;
    }

    rt::Tensor& getCosGen() noexcept
    {
        return mCosGenTensor;
    }

    rt::Tensor const& getCosGen() const noexcept
    {
        return mCosGenTensor;
    }

    rt::Tensor& getSinGen() noexcept
    {
        return mSinGenTensor;
    }

    rt::Tensor const& getSinGen() const noexcept
    {
        return mSinGenTensor;
    }

    rt::Tensor& getLastHiddenState() noexcept
    {
        return mLastHiddenTensor;
    }

    rt::Tensor const& getLastHiddenState() const noexcept
    {
        return mLastHiddenTensor;
    }

    bool copyUndSeqFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyGenSeqFrom(rt::Tensor const& src, cudaStream_t stream);
    bool copyRotaryFrom(CosmosTextPhase0 const& phase0, cudaStream_t stream);
    bool runBackbone(cudaStream_t stream) noexcept;

private:
    bool loadConfig();
    bool loadEngine(cudaStream_t stream);
    bool validateAndFillConfig();
    bool allocateBuffers();
    bool bindTensors() noexcept;
    bool copyTensorInput(rt::Tensor& dst, rt::Tensor const& src, char const* tensorLabel, cudaStream_t stream);

    std::string mEngineDir;
    nlohmann::json mConfigJson;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    std::string mUndSeqInputName{"und_seq"};
    std::string mGenSeqInputName{"gen_seq"};
    std::string mCosUndInputName{"cos_und"};
    std::string mSinUndInputName{"sin_und"};
    std::string mCosGenInputName{"cos_gen"};
    std::string mSinGenInputName{"sin_gen"};
    std::string mOutputName{"last_hidden_state"};

    rt::Coords mUndSeqShape;
    rt::Coords mGenSeqShape;
    rt::Coords mRotaryUndShape;
    rt::Coords mRotaryGenShape;
    rt::Coords mOutputShape;

    nvinfer1::DataType mSeqType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mRotaryType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kHALF};

    rt::Tensor mUndSeqTensor;
    rt::Tensor mGenSeqTensor;
    rt::Tensor mCosUndTensor;
    rt::Tensor mSinUndTensor;
    rt::Tensor mCosGenTensor;
    rt::Tensor mSinGenTensor;
    rt::Tensor mLastHiddenTensor;
};

} // namespace rt
} // namespace trt_edgellm
