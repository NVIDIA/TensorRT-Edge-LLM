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

#include <common/tensor.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Per-layer pointer bundle for batched MTP state scatter (analogous to KVLayerInfo).
struct MtpLayerInfo
{
    void* recurrentDst; //!< Main recurrent state pool [batchSize, recElements] FP32
    void* recurrentSrc; //!< Producer-specific intermediate recurrent buffer (GDN uses compact replay storage)
    void* convDst;      //!< Main conv state pool [batchSize, convElements] FP16
    void* convSrc;      //!< Intermediate conv state buffer [batchSize, verifyTreeSize, convElements] FP16
};

/// Batched scatter of accepted recurrent states (FP32) across all GDN layers.
/// For each (layer, batch): dst[b, :] = src[b, acceptLengths[b] - 1, :].
/// Skip if acceptLengths[b] <= 0; no-op if acceptLengths[b] >= verifyTreeSize.
/// stateElements must be divisible by 8 (DVec<float> vec_size).
void mtpScatterRecurrentStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptLengths, cudaStream_t stream);

/// Batched scatter of accepted conv states (FP16). Same shape as recurrent variant.
void mtpScatterConvStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptLengths, cudaStream_t stream);

/// Batched scatter of accepted recurrent states (FP32) for DDTree verification.
/// For each (layer, batch), finds the last stateNodeId >= 0 within
/// acceptedStateNodeIds[b, 0:min(acceptLengths[b], maxAcceptLen)] and copies:
/// dst[b, :] = src[b, nodeId, :].
/// Skip if no non-negative state node id is present.
/// stateElements must be divisible by 8 (DVec<float> vec_size).
void mtpScatterAcceptedTreeRecurrentStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers,
    int32_t activeBatchSize, int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptedStateNodeIds,
    int32_t maxAcceptLen, int32_t const* acceptLengths, cudaStream_t stream);

/// Batched scatter of accepted conv states (FP16) for DDTree verification. Same
/// accepted-state-node semantics as the recurrent variant.
void mtpScatterAcceptedTreeConvStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptedStateNodeIds, int32_t maxAcceptLen,
    int32_t const* acceptLengths, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
