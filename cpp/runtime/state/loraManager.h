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
#include "runtime/exec/tensorMap.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nvinfer1
{
class ICudaEngine;
} // namespace nvinfer1

namespace trt_edgellm
{
namespace rt
{

class EngineExecutor;

//! Manages LoRA adapter weights and switching.
//!
//! Maintains a map of named adapters, each containing a set of weight tensors
//! identified by binding name.  At any time at most one adapter is "active".
//! Switching adapters is O(1) — no GPU copies are required because the
//! TensorRT engine binds directly to the stored tensor pointers.
//!
//! Usage:
//!   1. `loadWeights()` or `addWeights()` to register adapter(s).
//!   2. `switchWeights(name)` to activate an adapter.
//!   3. `getActiveWeight(bindingName)` to retrieve the tensor for engine binding.
//!   4. `resetWeights()` to deactivate (bind dummy zero-tensors).
class LoRAManager
{
public:
    //! @brief Default constructor
    LoRAManager() = default;

    //! Deleted copy to prevent accidental duplication of GPU resources.
    LoRAManager(LoRAManager const&) = delete;
    LoRAManager& operator=(LoRAManager const&) = delete;

    //! Allow move.
    LoRAManager(LoRAManager&&) noexcept = default;
    LoRAManager& operator=(LoRAManager&&) noexcept = default;

    //! Load LoRA adapter weights from a safetensors file.
    //!
    //! Each tensor in the safetensors file is stored under its tensor name
    //! as the binding name.
    //!
    //! @param name     Adapter name (user-facing identifier).
    //! @param path     Path to the `.safetensors` file.
    //! @param stream   CUDA stream for async loading.
    //! @throws std::runtime_error if file cannot be read or format is invalid.
    void loadWeights(std::string const& name, std::filesystem::path const& path, cudaStream_t stream);

    //! Register adapter weights directly (useful for unit testing without I/O).
    //!
    //! @param name     Adapter name.
    //! @param weights  Map of binding-name to tensor (tensors are moved from).
    void addWeights(std::string const& name, std::map<std::string, rt::Tensor> weights);

    //! Activate an adapter by name.
    //!
    //! @param name  Adapter name (must have been loaded/added previously).
    //! @throws std::runtime_error if the adapter name is not found.
    void switchWeights(std::string const& name);

    //! Deactivate any adapter.  After this call `getActiveWeight()` returns
    //! a reference to a zero-filled dummy tensor (of shape [1]).
    void resetWeights();

    //! Retrieve the currently active tensor for a given binding name.  O(1).
    //!
    //! @param bindingName  The engine binding name (e.g. "lora_A_layer_0").
    //! @return Reference to the weight tensor (dummy tensor if no adapter is active).
    //! @throws std::runtime_error if `bindingName` is not found in the active adapter.
    rt::Tensor& getActiveWeight(std::string const& bindingName);

    //! Return the name of the currently active adapter, or an empty string
    //! if no adapter is active.
    std::string const& getActiveAdapterName() const noexcept;

    //! Monotonic generation local to one adapter name. Re-registering only that name advances it.
    uint64_t getAdapterGeneration(std::string const& name) const;

    //! Generation of the active adapter, or zero when no adapter is active.
    uint64_t getActiveAdapterGeneration() const;

    //! Return all binding names across all loaded adapters.
    //! Useful for initialising a `TensorMap` with the correct keys.
    std::vector<std::string> getBindingNames() const;

    //! Return all loaded adapter names.
    std::vector<std::string> getAdapterNames() const;

    //! Check whether any adapter is currently active.
    bool hasActiveAdapter() const noexcept;

    //! Check whether the active adapter contains a weight under @p bindingName.
    //!
    //! Fused engines and non-fused adapters sometimes use different naming
    //! conventions (e.g. `qkv_proj.*` vs separate `q_proj.*` / `k_proj.*` /
    //! `v_proj.*`). `refreshTensorMap` uses this predicate to decide whether
    //! to bind the adapter's weight or fall back to a dummy tensor, without
    //! paying the cost of `try` / `catch` around `getActiveWeight`.
    //!
    //! Returns false when no adapter is active.
    bool hasWeightFor(std::string const& bindingName) const noexcept;

    //! Register the engine's LoRA I/O bindings and create rank=1 dummy tensors
    //! with the correct engine shapes.  Must be called once after the EngineExecutor
    //! is constructed so that `refreshTensorMap()` knows which names to
    //! populate.
    //!
    //! Encapsulates the LoRA binding-shape convention:
    //!   - `lora_A_*` weights have shape [k, rank]; dummy sets last dim to 1.
    //!   - `lora_B_*` weights have shape [rank, n]; dummy sets first dim to 1.
    //!
    //! @param runner Source of the engine I/O list and per-binding max shapes.
    void initializeEngineBindings(EngineExecutor const& runner);

    //! Refresh all LoRA entries in the given TensorMap.
    //!
    //! For each registered engine binding name, either the active adapter's
    //! weight tensor or the per-binding dummy tensor is written into @p map.
    //! Must be called after every `switchWeights()` / `resetWeights()`.
    //!
    //! @param map  TensorMap to update with the current LoRA bindings.
    void refreshTensorMap(TensorMap& map);

private:
    //! Per-adapter storage: bindingName -> Tensor.
    using WeightMap = std::map<std::string, rt::Tensor>;

    //! All loaded adapters: adapterName -> WeightMap.
    std::map<std::string, WeightMap> mAdapters{};

    //! Per-name generation used by context-cache identity.
    std::map<std::string, uint64_t> mAdapterGenerations{};

    //! Name of the active adapter (empty when none active).
    std::string mActiveAdapterName{};

    //! Dummy zero-filled tensor returned when no adapter is active.
    rt::Tensor mDummyTensor{};

    //! Engine I/O names that correspond to LoRA bindings.
    std::vector<std::string> mEngineBindingNames{};

    //! Per-binding dummy tensors (correct engine shape, zero-filled).
    std::map<std::string, rt::Tensor> mEngineDummyTensors{};
};

} // namespace rt
} // namespace trt_edgellm
