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

#include "common/checkMacros.h"
#include "common/executionPhase.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace trt_edgellm::plugins
{

struct RaggedPluginMetadata
{
    int32_t numSequences;
    int32_t numContextSequences;
    int32_t numDecodeSequences;
    int32_t physicalTokens;
    int32_t contextExecutionRows; //!< T_exec - N_decode; split geometry for ordinary context/decode work
    int32_t decodeExecutionRows;  //!< N_decode; phase 8 permanently limits decode suffixes to one row each
    rt::ExecutionPhase phase;
};

inline RaggedPluginMetadata decodeRaggedPluginMetadata(char const* pluginName,
    nvinfer1::PluginTensorDesc const& activation, nvinfer1::PluginTensorDesc const& queryLengths,
    nvinfer1::PluginTensorDesc const& queryStartOffsets, nvinfer1::PluginTensorDesc const& phaseMarker,
    nvinfer1::PluginTensorDesc const& contextCountCarrier)
{
    auto const extent
        = [](nvinfer1::PluginTensorDesc const& desc) { return desc.dims.nbDims > 0 ? desc.dims.d[0] : -1; };
    int32_t const n = extent(queryLengths);
    int32_t const nctx = extent(contextCountCarrier);
    int32_t const texec = extent(activation);
    int32_t const phaseExtent = extent(phaseMarker);
    auto fail = [&](char const* reason) {
        check::check(false,
            std::string(pluginName) + " ragged metadata: " + reason + " (phase=" + std::to_string(phaseExtent) + ", N="
                + std::to_string(n) + ", Nctx=" + std::to_string(nctx) + ", T_exec=" + std::to_string(texec) + ")");
    };
    if (activation.dims.nbDims < 1 || queryLengths.dims.nbDims != 1 || queryStartOffsets.dims.nbDims != 1
        || phaseMarker.dims.nbDims != 1 || contextCountCarrier.dims.nbDims != 1)
    {
        fail("activation must have a token dimension and all metadata carriers must be rank 1");
    }
    if (n <= 0 || texec < n || queryStartOffsets.dims.d[0] != n + 1)
    {
        fail("invalid N/offset/T_exec relationship");
    }
    if (!rt::isExecutionPhaseExtent(phaseExtent))
    {
        fail("phase extent must be in [1, 8]");
    }
    if (nctx < 0 || nctx > n)
    {
        fail("Nctx must be in [0, N]");
    }
    auto const phase = static_cast<rt::ExecutionPhase>(phaseExtent);
    bool const contextPhase
        = phase == rt::ExecutionPhase::kContextPrefill || phase == rt::ExecutionPhase::kContextChunk;
    if (phase == rt::ExecutionPhase::kMixedPrefillDecode)
    {
        if (nctx <= 0 || nctx >= n)
        {
            fail("phase 8 requires 0 < Nctx < N");
        }
        fail("phase 8 mixed prefill/decode is reserved but unsupported");
    }
    if (texec % n != 0)
    {
        fail("entry-padded homogeneous execution requires T_exec divisible by N");
    }
    if (nctx != (contextPhase ? n : 0))
    {
        fail("homogeneous phase disagrees with Nctx");
    }

    int32_t const ndecode = n - nctx;
    int32_t const contextExecutionRows = texec - ndecode;
    if (contextExecutionRows < 0)
    {
        fail("derived T_context_exec is negative");
    }
    if (phase == rt::ExecutionPhase::kAutoregressiveDecode)
    {
        if (contextExecutionRows != 0)
        {
            fail("ordinary decode requires one execution row per sequence");
        }
    }
    return {n, nctx, ndecode, texec, contextExecutionRows, ndecode, phase};
}

inline int32_t validateIndexedResidentStateDescriptors(char const* pluginName, int32_t numSequences,
    nvinfer1::PluginTensorDesc const& stateIndices, nvinfer1::PluginTensorDesc const& stateInput,
    nvinfer1::PluginTensorDesc const& stateOutput, nvinfer1::DataType stateType,
    std::initializer_list<int32_t> trailingDimensions)
{
    auto fail = [&](char const* reason) {
        check::check(false, std::string(pluginName) + " indexed resident state: " + reason);
    };
    if (numSequences <= 0 || stateIndices.dims.nbDims != 1 || stateIndices.type != nvinfer1::DataType::kINT32
        || stateIndices.format != nvinfer1::TensorFormat::kLINEAR || stateIndices.dims.d[0] != numSequences)
    {
        fail("state_indices must be INT32 [N]");
    }

    int32_t const expectedRank = static_cast<int32_t>(trailingDimensions.size()) + 1;
    if (stateInput.dims.nbDims != expectedRank || stateOutput.dims.nbDims != expectedRank
        || stateInput.type != stateType || stateOutput.type != stateType || stateInput.dims.d[0] <= 0
        || stateOutput.dims.d[0] != stateInput.dims.d[0] || stateInput.format != nvinfer1::TensorFormat::kLINEAR
        || stateOutput.format != nvinfer1::TensorFormat::kLINEAR)
    {
        fail("resident state input/output descriptor is invalid");
    }
    int32_t dimension = 1;
    for (int32_t const expected : trailingDimensions)
    {
        if (expected <= 0 || stateInput.dims.d[dimension] != expected || stateOutput.dims.d[dimension] != expected)
        {
            fail("resident state input/output shape does not match the kernel contract");
        }
        ++dimension;
    }
    return static_cast<int32_t>(stateInput.dims.d[0]);
}

} // namespace trt_edgellm::plugins
