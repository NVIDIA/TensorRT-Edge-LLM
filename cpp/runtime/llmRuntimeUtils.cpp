/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/llmRuntimeUtils.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace rt
{

std::ostream& operator<<(std::ostream& os, RopeType const& type)
{
    switch (type)
    {
    case RopeType::kDefault: os << "Default"; break;
    case RopeType::kDynamic: os << "Dynamic"; break;
    case RopeType::kLongRope: os << "LongRope"; break;
    case RopeType::kMRope: os << "MRope"; break;
    }
    return os;
}

std::string formatRopeConfig(RopeConfig const& config)
{
    std::stringstream ss;
    ss << "RopeConfig:"
       << "  type: " << config.type << "  rotaryScale: " << config.rotaryScale
       << "  rotaryTheta: " << config.rotaryTheta << "  maxPositionEmbeddings: " << config.maxPositionEmbeddings;
    if (config.type == RopeType::kLongRope)
    {
        ss << "LongRopeConfig:"
           << "  originalMaxPositionEmbeddings: " << config.longRope.value().originalMaxPositionEmbeddings;
    }
    return ss.str();
}

RopeConfig collectRopeConfig(nlohmann::json const& config)
{
    RopeConfig ropeConfig{};
    auto ropeScalingIt = config.find("rope_scaling");
    if (ropeScalingIt != config.end())
    {
        auto ropeTypeIt = ropeScalingIt->find("type");
        if (ropeTypeIt == ropeScalingIt->end())
        {
            ropeTypeIt = ropeScalingIt->find("rope_type");
        }
        auto mropeSectionIt = ropeScalingIt->find("mrope_section");
        if (ropeTypeIt != ropeScalingIt->end())
        {
            std::string const ropeTypeStr = ropeTypeIt->get<std::string>();
            if (ropeTypeStr == "default" && mropeSectionIt != ropeScalingIt->end())
            {
                // transformers `Qwen2_5_VLVisionConfig` change type from 'mrope' to 'default'
                ropeConfig.type = RopeType::kMRope;
            }
            else if (ropeTypeStr == "default" || ropeTypeStr == "llama3")
            {
                // Route the llama3 config to default type.
                ropeConfig.type = RopeType::kDefault;
            }
            else if (ropeTypeStr == "dynamic")
            {
                ropeConfig.type = RopeType::kDynamic;
            }
            else if (ropeTypeStr == "longrope")
            {
                ropeConfig.type = RopeType::kLongRope;
            }
        }

        // Parse long rope scaling parameters when requested
        if (ropeConfig.type == RopeType::kLongRope)
        {
            LongRopeParams params{};
            auto longFactorIt = ropeScalingIt->find("long_factor");
            check::check((longFactorIt != ropeScalingIt->end() && longFactorIt->is_array()),
                "rope_scaling.long_factor must be a non-empty array for longrope");
            params.longFactor = longFactorIt->get<std::vector<float>>();

            auto shortFactorIt = ropeScalingIt->find("short_factor");
            check::check((shortFactorIt != ropeScalingIt->end() && shortFactorIt->is_array()),
                "rope_scaling.short_factor must be a non-empty array for longrope");
            params.shortFactor = shortFactorIt->get<std::vector<float>>();

            check::check(params.longFactor.size() == params.shortFactor.size(),
                "rope_scaling.long_factor size differs from short_factor size");

            check::check(config.contains("original_max_position_embeddings"),
                "original_max_position_embeddings is not specified in the model config");
            params.originalMaxPositionEmbeddings = config["original_max_position_embeddings"].get<int32_t>();

            ropeConfig.longRope = std::move(params);
        }
    }
    else
    {
        LOG_WARNING(
            "rope_scaling is not specified in the model config, using default rope type. This could misalign with the "
            "model configuration, please check the config file to ensure the correctness");
        ropeConfig.type = RopeType::kDefault;
    }

    // Detect RopeTheta
    if (config.contains("rope_theta"))
    {
        ropeConfig.rotaryTheta = config["rope_theta"].get<float>();
    }
    else
    {
        LOG_WARNING("rope_theta is not specified in the model config, using default value: %f", ropeConfig.rotaryTheta);
    }

    // Detect MaxPositionEmbeddings
    if (config.contains("max_position_embeddings"))
    {
        ropeConfig.maxPositionEmbeddings = config["max_position_embeddings"].get<int32_t>();
    }
    else
    {
        LOG_WARNING("max_position_embeddings is not specified in the model config, using default value: %d",
            ropeConfig.maxPositionEmbeddings);
    }

    LOG_INFO("Collected rope config: %s", formatRopeConfig(ropeConfig).c_str());
    return ropeConfig;
}

bool initializeRopeCosSinCache(
    rt::Tensor& cosSinCache, RopeConfig const& config, nlohmann::json const& modelConfig, cudaStream_t stream)
{
    if (config.type == RopeType::kMRope)
    {
        LOG_ERROR("MRope is context dependent rope type, which cannot be initialized with basic parameters.");
        return false;
    }
    else if (config.type == RopeType::kLongRope)
    {
        LOG_ERROR("Please use initializeLongRopeCosSinCache instead.");
        return false;
    }

    // Tensor shape: [1, maxLength, rotaryDim]
    if (cosSinCache.getShape().getNumDims() != 3 || cosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("Persistent RopeCosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }
    int64_t ropeMaxLength = cosSinCache.getShape()[1];
    int64_t rotaryDim = cosSinCache.getShape()[2];
    if (config.type == RopeType::kDefault || config.type == RopeType::kDynamic)
    {
        if (config.type == RopeType::kDynamic && ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_ERROR("Dynamic rope type with sequence length larger than maxPositionEmbeddings is not supported.");
            return false;
        }
        if (ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_WARNING(
                "maxLength %d is greater than maxPositionEmbeddings %d indicated by model config, this could cause "
                "inaccurate generation results",
                ropeMaxLength, config.maxPositionEmbeddings);
        }

        try
        {
            kernel::initializeNormalRopeCosSin(cosSinCache.dataPointer<float>(), config.rotaryTheta, config.rotaryScale,
                rotaryDim, ropeMaxLength, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("CUDA kernel launch for initializeNormalRopeCosSin failed: %s", e.what());
            return false;
        }
    }
    return true;
}

bool initializeLongRopeCosSinCache(rt::Tensor& shortCosSinCache, rt::Tensor& longCosSinCache, RopeConfig const& config,
    nlohmann::json const& modelConfig, cudaStream_t stream)
{

    if (config.type != RopeType::kLongRope)
    {
        LOG_ERROR("This function is only used for initializing LongRope cos/sin cache.");
        return false;
    }

    // Tensor shape: [1, maxLength, rotaryDim]
    if (shortCosSinCache.getShape().getNumDims() != 3 || shortCosSinCache.getDataType() != DataType::kFLOAT
        || longCosSinCache.getShape().getNumDims() != 3 || longCosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("Persistent RopeCosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }

    int64_t ropeMaxLength = shortCosSinCache.getShape()[1];
    int64_t rotaryDim = shortCosSinCache.getShape()[2];

    // Validate long/short factors before creating tensors and launching the kernel
    if (!config.longRope.has_value() || config.longRope->longFactor.empty() || config.longRope->shortFactor.empty())
    {
        LOG_ERROR("LongRope requires non-empty long_factor and short_factor arrays in rope_scaling.");
        return false;
    }

    Coords shape{static_cast<int64_t>(config.longRope->longFactor.size())};

    Tensor longFactorTensor(shape, DeviceType::kGPU, DataType::kFLOAT, "long_factor");
    Tensor shortFactorTensor(shape, DeviceType::kGPU, DataType::kFLOAT, "short_factor");
    CUDA_CHECK(cudaMemcpyAsync(longFactorTensor.rawPointer(), config.longRope->longFactor.data(),
        longFactorTensor.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(shortFactorTensor.rawPointer(), config.longRope->shortFactor.data(),
        shortFactorTensor.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    try
    {
        kernel::initializeLongRopeCosSin(shortCosSinCache.dataPointer<float>(), longCosSinCache.dataPointer<float>(),
            shortFactorTensor.dataPointer<float>(), longFactorTensor.dataPointer<float>(), config.rotaryTheta,
            rotaryDim, ropeMaxLength, config.maxPositionEmbeddings, config.longRope->originalMaxPositionEmbeddings,
            stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("CUDA kernel launch for initializeLongRopeCosSin failed: %s", e.what());
        return false;
    }
    return true;
}

//=============================================================================
// CPU Vector Compaction Utility for Batch Eviction
//=============================================================================

template <typename T>
void compactVector(std::vector<int32_t> const& batchMapping, std::vector<T>& vec)
{
    // Validate that vector size matches batchMapping size
    // In batch eviction, batchMapping[i] indicates where batch i should move to
    if (vec.size() != batchMapping.size())
    {
        throw std::invalid_argument(
            format::fmtstr("compactVector: vector size (%zu) does not match batchMapping size (%zu)", vec.size(),
                batchMapping.size()));
    }

    std::vector<T> compacted;
    compacted.reserve(vec.size());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (batchMapping[i] >= 0)
        {
            compacted.push_back(std::move(vec[i]));
        }
    }

    vec = std::move(compacted);
}

template void compactVector<int8_t>(std::vector<int32_t> const&, std::vector<int8_t>&);
template void compactVector<int32_t>(std::vector<int32_t> const&, std::vector<int32_t>&);
template void compactVector<std::vector<int32_t>>(std::vector<int32_t> const&, std::vector<std::vector<int32_t>>&);
template void compactVector<std::string>(std::vector<int32_t> const&, std::vector<std::string>&);

// Build batch mapping from finished states
// Returns a vector mapping old batch indices to new indices (-1 for evicted batches)
std::vector<int32_t> buildBatchMapping(std::vector<int8_t> const& finishedStates)
{
    int32_t const oldActiveBatch = static_cast<int32_t>(finishedStates.size());
    std::vector<int32_t> mapping(oldActiveBatch);

    int32_t newIdx = 0;
    for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
    {
        if (!finishedStates[oldIdx])
        {
            mapping[oldIdx] = newIdx;
            newIdx++;
        }
        else
        {
            mapping[oldIdx] = -1; // Mark for eviction
        }
    }

    return mapping;
}

} // namespace rt
} // namespace trt_edgellm