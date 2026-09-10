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

#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

struct DecoderRegistryInit
{
    //! Still needed by the decoders that read their own strategy-specific sidecars (EAGLE's d2t table, DSpark's
    //! head weights, DFlash's draft vocab map). Draft engine weights arrive via `draftWeights` instead.
    std::filesystem::path engineDir;
    std::optional<SpecDecodeDraftingConfig> draftingConfig;
    //! Validated draft executor whose ownership is consumed by the selected speculative decoder.
    std::unique_ptr<EngineExecutor> draftExecutor;
    //! Loaded and engine-validated draft weights, consumed by the selected speculative decoder.
    ExternalWeightManager draftWeights;
    cudaStream_t stream{};
};

//! Return whether an available speculative decoder should yield to the default decoder for this request.
bool shouldSelectDefaultDecoder(
    DecodingStrategyKind speculativeDecoderKind, LLMGenerationRequest const& request) noexcept;

class DecoderRegistry final
{
public:
    DecoderRegistry(DecodingRuntimeContext& runtime, DecoderRegistryInit init);

    DecodingStrategy& select(LLMGenerationRequest const& request) const noexcept;
    DecodingStrategy& cachePrimingStrategy() const noexcept;
    bool captureCudaGraphs(cudaStream_t stream) const;
    int64_t getRequiredContextMemorySize() const noexcept;
    void setContextMemory(Tensor& memory) const;

    bool hasSpeculativeDecoder() const noexcept
    {
        return mSpeculativeDecoder != nullptr;
    }

    char const* speculativeDecoderName() const noexcept
    {
        return mSpeculativeDecoder ? mSpeculativeDecoder->name() : "vanilla";
    }

private:
    std::unique_ptr<DecodingStrategy> mDefaultDecoder;
    std::unique_ptr<DecodingStrategy> mSpeculativeDecoder;
};

} // namespace rt
} // namespace trt_edgellm
