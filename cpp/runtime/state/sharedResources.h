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

#include "common/tensor.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/externalWeightManager.h"
#include "runtime/state/kvPageTable.h"
#include "runtime/state/loraManager.h"
#include "runtime/state/ropeCache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Process-lifetime resources shared across runners.
struct SharedResources
{
    //! One HybridCacheManager per engine (index 0 = base, 1 = draft for SpecDecode).
    //! unique_ptr because HybridCacheManager is move-only.
    std::vector<std::unique_ptr<HybridCacheManager>> cacheManagers;

    //! One KVPageTable per cache manager, index-aligned with `cacheManagers`. Each table starts as an uploaded identity
    //! mapping. Batch eviction compacts logical rows while leaving physical pages in place.
    std::vector<std::unique_ptr<KVPageTable>> kvPageTables;

    RopeCache ropePool;
    std::unique_ptr<LoRAManager> loraManager;
    std::unique_ptr<ExternalWeightManager> externalWeightManager;
    Tensor zeroBuffer;

    //! Build SharedResources for the vanilla single-engine LLM runtime
    //! (KV cache, RoPE pool, LoRA manager, external weight manager, zero buffer).
    //!
    //! Recurrent / conv state dtypes for hybrid models are read from
    //! `cfg.recurrentStateDtype` / `cfg.convStateDtype` — they are parsed
    //! strictly from `config.json` by `parseEngineConfig`.
    //!
    //! The returned `externalWeightManager` is constructed by this factory; the
    //! runtime is responsible for loading files, validating against the base
    //! engine, and publishing it to a TensorMap. This keeps `SharedResources`
    //! decoupled from `EngineExecutor` (no engine I/O or validation happens
    //! inside this factory).
    static std::unique_ptr<SharedResources> createForLLM(LLMEngineConfig const& cfg,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    //! Build SharedResources for a two-engine speculative-decoding runtime
    //! (base + draft KV caches, shared RoPE pool, LoRA manager, external weight
    //! manager, zero buffer).
    //!
    //! As with `createForLLM`, the returned `externalWeightManager` is
    //! constructed by this factory, and the runtime must load files, validate
    //! against the base engine, and publish it to a TensorMap. External weights
    //! currently apply to the base engine only.
    static std::unique_ptr<SharedResources> createForSpecDecode(DeploymentConfig const& bundle,
        int32_t maxRuntimeBatchSize, std::unordered_map<std::string, std::string> const& loraWeightsMap,
        cudaStream_t stream);
};

void allocateZeroBuffer(SharedResources& res, int64_t bytes);

} // namespace rt
} // namespace trt_edgellm
