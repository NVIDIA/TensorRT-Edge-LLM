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

// What a deployment's layer topology is, and where one forward's rows land in each sequence's cache.
//
// Both are read off what the runtime states -- the config for the topology, the bindings for the layout --
// rather than restated by a test.

#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "runtime/exec/tensorMap.h"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace substitute_engine
{

using namespace trt_edgellm;

struct FakeLayerComposition
{
    int32_t attentionLayers{};
    int32_t recurrentLayers{};
    trt_edgellm::DataType kvType{trt_edgellm::DataType::kHALF};

    bool isHybrid() const noexcept
    {
        return attentionLayers > 0 && recurrentLayers > 0;
    }

    //! Derive the topology from the same config.json object makeTinyVanillaConfig()/writeModelDir() use, so a test
    //! never has to restate the layer counts or `kv_cache_dtype` a second time in a different shape.
    //!
    //! A hybrid config states `num_attention_layers` explicitly; a plain one leaves every decoder layer to
    //! attention.
    static FakeLayerComposition fromConfigJson(Json const& config)
    {
        FakeLayerComposition composition;
        composition.recurrentLayers = config.value("num_linear_attn_layers", 0);
        composition.attentionLayers
            = config.value("num_attention_layers", config.at("num_hidden_layers").get<int32_t>());

        std::string const dtype = config.at("kv_cache_dtype").get<std::string>();
        if (dtype == "fp16")
        {
            composition.kvType = trt_edgellm::DataType::kHALF;
        }
        else if (dtype == "bf16")
        {
            composition.kvType = trt_edgellm::DataType::kBF16;
        }
        else
        {
            // Fail fast rather than silently mis-writing bytes for a KV dtype whose write path nothing has
            // validated. fp8 and int8 are the ones this will reject.
            throw std::runtime_error("FakeLayerComposition: unsupported kv_cache_dtype '" + dtype + "'");
        }
        return composition;
    }
};

//! Which of a decoder's per-sequence caches an entry belongs to. Attention layers carry the two halves of the KV
//! cache; recurrent layers carry a running state and a convolution window instead. A hybrid model has both, and a
//! single forward writes both, so this is a property of the binding rather than of the model.

struct FakeForwardLayout
{
    std::vector<int32_t> queryStarts; //!< `batch + 1` physical row offsets
    std::vector<int32_t> queryLengths;
    std::vector<int32_t> pastLengths;
    //! Each row's position within its sequence, which is not its cache slot. Rows of an ordinary prefill or decode
    //! carry strictly increasing positions; the candidate rows of a speculative verification forward are siblings
    //! of one tree node and share one position while occupying consecutive slots.
    std::vector<int32_t> rowPositions;
    size_t physicalRows{};

    static FakeForwardLayout from(rt::TensorMap const& map, cudaStream_t stream)
    {
        FakeForwardLayout layout;
        auto const read = [&](char const* name) {
            rt::Tensor* const tensor = map.get(name);
            if (tensor == nullptr)
            {
                return std::vector<int32_t>{};
            }
            std::vector<int32_t> host(static_cast<size_t>(tensor->getShape().volume()));
            CUDA_CHECK(cudaMemcpyAsync(
                host.data(), tensor->rawPointer(), host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return host;
        };
        layout.queryStarts = read(binding_names::kQueryStartOffsets);
        layout.queryLengths = read(binding_names::kQueryLengths);
        layout.pastLengths = read(binding_names::kPastLengths);
        layout.rowPositions = read(binding_names::kPositions);
        layout.physicalRows = layout.rowPositions.size();
        return layout;
    }

    bool valid() const noexcept
    {
        return physicalRows > 0 && queryStarts.size() >= 2U && queryLengths.size() + 1U >= queryStarts.size()
            && pastLengths.size() + 1U >= queryStarts.size();
    }

    int64_t batch() const noexcept
    {
        return static_cast<int64_t>(queryStarts.size()) - 1;
    }

    int64_t pastOf(int64_t slot) const noexcept
    {
        return pastLengths[static_cast<size_t>(slot)];
    }

    //! Whether row `earlier` is one of row `row`'s ancestors within this forward.
    //!
    //! A row attends to the committed prefix plus whatever preceded it *on its own branch*. In a prefill or a
    //! decode there is only one branch, so every earlier row qualifies. In a speculative verification forward the
    //! rows are competing candidates, and a candidate that folded its siblings in would be answering a question
    //! nobody asked -- which is exactly what makes its answer differ from the vanilla path's.
    bool precedes(int32_t earlier, int32_t row) const noexcept
    {
        return rowPositions[static_cast<size_t>(earlier)] < rowPositions[static_cast<size_t>(row)];
    }

    //! The half-open range of real rows belonging to `slot`, clamped to the rows this forward actually has.
    std::pair<int32_t, int32_t> rowsOf(int64_t slot) const noexcept
    {
        auto const limit = static_cast<int32_t>(physicalRows);
        int32_t const begin = std::clamp(queryStarts[static_cast<size_t>(slot)], 0, limit);
        int32_t const end = std::clamp(begin + queryLengths[static_cast<size_t>(slot)], begin,
            std::min(limit, std::clamp(queryStarts[static_cast<size_t>(slot) + 1], begin, limit)));
        return {begin, end};
    }
};

//! Writes a recurrent layer's per-sequence state, dtype-aware.
//!
//! Much simpler than the KV case, and deliberately so: a recurrent pool is `[maxBatch, ...]` with no paging, so
//! the only thing that moves between forwards is which row `state_indices` points a sequence at. That row index is
//! a scheduling decision, which is exactly why the value stored in it is keyed by the sequence's position instead
//! -- a row that ends up holding another sequence's progress is then visible as a wrong number rather than as an
//! index a test has to interpret.

} // namespace substitute_engine
