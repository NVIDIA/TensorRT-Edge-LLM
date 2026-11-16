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

#include "runtime/weight/refitWeights.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtInterop.h"

#include <algorithm>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

bool validateWeightlessConfig(nlohmann::json const& config)
{
    if (config.contains("weightless_engine"))
    {
        ELLM_CHECK(config.at("weightless_engine").is_boolean(), "weightless_engine must be a boolean");
    }
    bool const enabled = config.value("weightless_engine", false);
    auto const bindings = config.value("checkpoint_weight_bindings", nlohmann::json::array());
    ELLM_CHECK(bindings.is_array(), "checkpoint_weight_bindings must be an array");
    bool hasRefit = false;
    for (auto const& binding : bindings)
    {
        ELLM_CHECK(binding.is_object(), "Checkpoint weight bindings must be objects");
        std::string const destination = binding.value("destination", "input");
        ELLM_CHECK(destination == "input" || destination == "refit", "Unknown checkpoint weight destination");
        if (destination == "refit")
        {
            hasRefit = true;
            ELLM_CHECK(binding.value("role", "") != "embedding" && binding.value("engine_name", "") != "__embedding__"
                    && !binding.contains("plugin_resource_id") && !binding.contains("plugin_resource_kind"),
                "Embedding tables and plugin resources cannot be refit weights");
        }
    }
    ELLM_CHECK(enabled == hasRefit, "weightless_engine metadata and refit bindings must be present together");
#if !defined(ENABLE_EXPERIMENTAL_WEIGHTLESS)
    ELLM_CHECK(!enabled,
        "Weightless engine requires a runtime built with ENABLE_EXPERIMENTAL_WEIGHTLESS=ON and qualified internal TRT");
#endif
    return enabled;
}

void refitEngineIfNeeded(nvinfer1::ICudaEngine& engine, EngineRefitOptions refit)
{
    if (refit.weights != nullptr)
    {
        refitEngine(engine, *refit.weights, refit.stream);
    }
#if defined(ENABLE_EXPERIMENTAL_WEIGHTLESS) && !IS_TRT_RTX
    else
    {
        // Null-placeholder plans can have zero serialized weight bytes; only -1 denotes a non-stripped engine.
        ELLM_CHECK(engine.getEngineStat(nvinfer1::EngineStat::kSTRIPPED_WEIGHTS_SIZE) < 0,
            "Stripped engine requires weightless metadata and checkpoint refit weights");
    }
#endif
}

void refitEngine(nvinfer1::ICudaEngine& engine, RefitWeightSet const& weights, cudaStream_t stream)
{
#if defined(ENABLE_EXPERIMENTAL_WEIGHTLESS) && !IS_TRT_RTX
    ELLM_CHECK(stream != nullptr, "Weightless refit requires an explicit non-default CUDA stream");
    ELLM_CHECK(!weights.weights.empty(), "Weightless engine has no refit weights");
    auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(engine, TrtLogger::instance()));
    ELLM_CHECK(refitter != nullptr, "Failed to create weightless engine refitter");
    int32_t const count = refitter->getAllWeights(0, nullptr);
    ELLM_CHECK(count > 0 && static_cast<size_t>(count) == weights.weights.size(),
        "Engine refittable-weight count differs from checkpoint manifest");
    std::vector<char const*> names(count);
    ELLM_CHECK(refitter->getAllWeights(count, names.data()) == count, "Failed to enumerate refit weights");
    std::unordered_map<std::string, Tensor const*> supplied;
    for (auto const& tensor : weights.weights)
    {
        ELLM_CHECK(supplied.emplace(tensor.getName(), &tensor).second, "Duplicate refit weight: " + tensor.getName());
    }
    for (char const* name : names)
    {
        ELLM_CHECK(name != nullptr, "Engine returned a null refit weight name");
        auto const entry = supplied.find(name);
        ELLM_CHECK(entry != supplied.end(), "Checkpoint manifest is missing refit weight: " + std::string(name));
        Tensor const& tensor = *entry->second;
        auto const prototype = refitter->getWeightsPrototype(name);
        // Tensor storage is byte-addressable; native FP4 weights retain the
        // checkpoint's packed U8 shape until they reach TensorRT's refitter.
        bool const packedFp4 = fromTrt(prototype.type) == trt_edgellm::DataType::kFP4
            && tensor.getDataType() == trt_edgellm::DataType::kUINT8 && prototype.count > 0 && prototype.count % 2 == 0
            && tensor.getShape().volume() == prototype.count / 2;
        ELLM_CHECK(tensor.getDeviceType() == DeviceType::kGPU && tensor.rawPointer() != nullptr
                && (packedFp4
                    || (tensor.getDataType() == fromTrt(prototype.type)
                        && tensor.getShape().volume() == prototype.count)),
            "Refit weight has incompatible storage, dtype or count: " + std::string(name));
        nvinfer1::Weights const values{prototype.type, tensor.rawPointer(), prototype.count};
        ELLM_CHECK(refitter->setNamedWeights(name, values, nvinfer1::TensorLocation::kDEVICE),
            "Failed to supply refit weight: " + std::string(name));
    }
    ELLM_CHECK(refitter->getMissingWeights(0, nullptr) == 0, "Engine still has missing refit weights");
    detail::completeRefit(
        [&] { return refitter->refitCudaEngineAsync(stream); }, [&] { return cudaStreamSynchronize(stream); });
#else
    (void) engine;
    (void) weights;
    (void) stream;
    ELLM_CHECK(false, "Weightless refit is unavailable in this runtime build");
#endif
}

} // namespace rt
} // namespace trt_edgellm
