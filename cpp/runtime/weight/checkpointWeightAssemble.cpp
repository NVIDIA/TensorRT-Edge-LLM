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

#include "runtime/weight/checkpointWeightAssemble.h"

#include "common/checkMacros.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"
#include "kernels/weightsTransform/fp16/linear/fp16LayoutConvert.h"
#include "kernels/weightsTransform/fp16/moe/fp16MoeAssemble.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/common/gptqActivationPermutation.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/common/int4QkvScaleConcat.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/v1/int4PluginV1Repack.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/v2/int4CuteDslRepack.h"
#include "kernels/weightsTransform/int4/moe/gptqMarlin/gptqMarlinRepack.h"
#include "kernels/weightsTransform/int4/moe/gptqMarlin/int4MoeScaleRepack.h"
#include "kernels/weightsTransform/nvfp4/moe/common/nvfp4MoeScaleTransform.h"
#include "kernels/weightsTransform/nvfp4/moe/common/nvfp4MoeWeightTransform.h"
#include "kernels/weightsTransform/nvfp4/moe/sm110/nvfp4MoePluginTransform.h"
#include "kernels/weightsTransform/nvfp4/moe/sm120/nvfp4MoePluginGeforceTransform.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

using Json = nlohmann::json;
using View = CheckpointReader::View;

Json const& checkpointKeys(Json const& binding)
{
    ELLM_CHECK(binding.contains("checkpoint_keys") && binding["checkpoint_keys"].is_array(),
        "checkpoint binding missing checkpoint_keys");
    return binding["checkpoint_keys"];
}

std::string const& keyName(Json const& keys, size_t index)
{
    ELLM_CHECK(index < keys.size() && keys[index].is_string(), "invalid checkpoint key entry");
    return keys[index].get_ref<std::string const&>();
}

View requireView(CheckpointReader const& checkpoint, std::string const& key, std::string const& label)
{
    View view;
    ELLM_CHECK(checkpoint.find(key, view), "Missing " + label + ": " + key);
    ELLM_CHECK(view.deviceData != nullptr, "Checkpoint tensor is not CUDA-mapped: " + key);
    return view;
}

bool findView(CheckpointReader const& checkpoint, std::string const& key, View& view)
{
    if (!checkpoint.find(key, view))
    {
        return false;
    }
    ELLM_CHECK(view.bytes == 0 || view.deviceData != nullptr, "Checkpoint tensor is not CUDA-mapped: " + key);
    return true;
}

int32_t int4PluginVersion(Json const& binding)
{
    int32_t const version = binding.value("int4_gemm_plugin_version", 2);
    ELLM_CHECK(version == 1 || version == 2, "int4_gemm_plugin_version must be 1 or 2");
    return version;
}

void validateOutput(Tensor const& output, Coords const& shape, nvinfer1::DataType dtype, std::string const& label)
{
    ELLM_CHECK(output.getShape() == shape, label + " shape does not match engine binding");
    ELLM_CHECK(output.getDataType() == dtype, label + " dtype does not match engine binding");
}

bool validateGptqGroups(View const& groups, int32_t K, int32_t numGroups, int32_t groupSize, std::string const& key)
{
    ELLM_CHECK(groups.dtype == nvinfer1::DataType::kINT32 && groups.shape.volume() == K,
        "GPTQ g_idx must be INT32 [K]: " + key);
    ELLM_CHECK(numGroups > 0 && groupSize > 0 && numGroups * groupSize == K, "Invalid GPTQ group geometry: " + key);
    auto const* values = reinterpret_cast<int32_t const*>(groups.data);
    bool sequential = true;
    for (int32_t k = 0; k < K; ++k)
    {
        ELLM_CHECK(values[k] >= 0 && values[k] < numGroups, "GPTQ g_idx contains an out-of-range group: " + key);
        sequential = sequential && values[k] == k / groupSize;
    }
    for (int32_t group = 0; group < numGroups; ++group)
    {
        int32_t count = 0;
        for (int32_t k = 0; k < K; ++k)
        {
            count += values[k] == group ? 1 : 0;
        }
        ELLM_CHECK(count == groupSize, "GPTQ g_idx group size does not match the checkpoint contract: " + key);
    }
    return sequential;
}

Tensor const* findPreparedWeight(std::vector<Tensor> const& preparedWeights, std::string const& name)
{
    if (name.empty())
    {
        return nullptr;
    }
    auto const found = std::find_if(preparedWeights.begin(), preparedWeights.end(),
        [&name](Tensor const& tensor) { return tensor.getName() == name; });
    ELLM_CHECK(found != preparedWeights.end(), "Missing prepared checkpoint binding: " + name);
    return &*found;
}

void assembleGptqActivationPermutation(
    CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() == 1, "GPTQ activation permutation expects one g_idx checkpoint key");
    std::string const& key = keyName(keys, 0);
    View const groups = requireView(checkpoint, key, "GPTQ g_idx");
    int32_t const K = static_cast<int32_t>(groups.shape.volume());
    int32_t const groupSize = binding.value("group_size", 0);
    ELLM_CHECK(groupSize > 0 && K % groupSize == 0, "Invalid GPTQ activation permutation group size: " + key);
    int32_t const numGroups = K / groupSize;
    validateGptqGroups(groups, K, numGroups, groupSize, key);
    validateOutput(output, Coords{K}, nvinfer1::DataType::kINT32, "GPTQ activation permutation output");
    CUDA_CHECK(kernel::launchGptqActivationPermutation(reinterpret_cast<int32_t const*>(groups.deviceData),
        static_cast<int32_t*>(output.rawPointer()), K, numGroups, groupSize, stream));
}

void loadIdentityWeight(CheckpointReader& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    constexpr size_t kSourceWindowBytes = 32ULL << 20;
    Json const& keys = checkpointKeys(binding);
    View source;
    std::string sourceKey;
    for (size_t index = 0; index < keys.size(); ++index)
    {
        std::string const& candidate = keyName(keys, index);
        if (checkpoint.findHost(candidate, source))
        {
            sourceKey = candidate;
            break;
        }
    }
    ELLM_CHECK(!sourceKey.empty(), "Failed to load checkpoint tensor for " + output.getName());

    nvinfer1::DataType const sourceType = source.dtype;
    nvinfer1::DataType const outputType = output.getDataType();
    bool const needCast = outputType == nvinfer1::DataType::kHALF && sourceType != nvinfer1::DataType::kHALF
        && (sourceType == nvinfer1::DataType::kBF16 || sourceType == nvinfer1::DataType::kFLOAT);
    Coords const want = output.getShape();
    Coords const have = source.shape;

    auto runRange = [&](size_t sourceOffset, size_t sourceBytes, auto&& launch) {
        View const mapped = checkpoint.registerTensorRange(sourceKey, sourceOffset, sourceBytes);
        try
        {
            launch(mapped);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        catch (...)
        {
            static_cast<void>(cudaStreamSynchronize(stream));
            checkpoint.unregisterTensors();
            throw;
        }
        checkpoint.unregisterTensors();
        checkpoint.discardTensorRange(sourceKey, sourceOffset, sourceBytes);
    };

    int32_t const runtimeRank = binding.value("runtime_rank", 0);
    if (binding.value("tp_rank0_only", false) && runtimeRank != 0)
    {
        CUDA_CHECK(cudaMemsetAsync(output.rawPointer(), 0, static_cast<size_t>(output.getMemoryCapacity()), stream));
        return;
    }

    bool const hasShard = binding.contains("tp_shard");
    int32_t shardAxis = -1;
    int32_t shardSize = 1;
    int32_t shardRank = 0;
    if (hasShard)
    {
        Json const& shard = binding["tp_shard"];
        ELLM_CHECK(shard.is_object(), "tp_shard must be an object for " + output.getName());
        shardAxis = shard.value("axis", -1);
        shardSize = shard.value("size", 0);
        shardRank = shard.value("rank", -1);
        ELLM_CHECK(shardAxis >= 0 && shardAxis < have.getNumDims(), "invalid tp_shard axis for " + output.getName());
        ELLM_CHECK(shardSize > 1 && shardRank >= 0 && shardRank < shardSize,
            "invalid tp_shard size or rank for " + output.getName());
        ELLM_CHECK(have[shardAxis] % shardSize == 0, "checkpoint weight dimension is not divisible by tp_shard size");
        Coords expected = have;
        expected[shardAxis] /= shardSize;
        ELLM_CHECK(expected == want,
            "checkpoint TP shard shape " + expected.formatString() + " does not match binding " + want.formatString()
                + " for " + output.getName());
    }

    bool const needTranspose = !hasShard && have != want;
    if (needTranspose)
    {
        ELLM_CHECK(want.getNumDims() == 2 && have.getNumDims() == 2 && have[0] == want[1] && have[1] == want[0],
            "checkpoint weight shape " + have.formatString() + " does not match binding " + want.formatString()
                + " for " + output.getName());
    }

    auto copyContiguous = [&](size_t sourceOffset, int64_t elements) {
        ELLM_CHECK(elements > 0, "checkpoint shard is empty for " + output.getName());
        size_t const sourceElementBytes = utils::getTypeSize(sourceType);
        size_t const outputElementBytes = utils::getTypeSize(outputType);
        size_t const windowElements = std::max<size_t>(1, kSourceWindowBytes / sourceElementBytes);
        for (size_t first = 0; first < static_cast<size_t>(elements); first += windowElements)
        {
            size_t const count = std::min(windowElements, static_cast<size_t>(elements) - first);
            size_t const rangeOffset = sourceOffset + first * sourceElementBytes;
            size_t const sourceBytes = count * sourceElementBytes;
            auto* destination = static_cast<uint8_t*>(output.rawPointer()) + first * outputElementBytes;
            runRange(rangeOffset, sourceBytes, [&](View const& mapped) {
                if (!needCast)
                {
                    CUDA_CHECK(kernel::launchCopyBytes(mapped.deviceData, destination, sourceBytes, stream));
                }
                else if (sourceType == nvinfer1::DataType::kFLOAT)
                {
                    CUDA_CHECK(kernel::launchFp32ToFp16(mapped.deviceData, destination, count, stream));
                }
                else
                {
                    CUDA_CHECK(kernel::launchBf16ToFp16(mapped.deviceData, destination, count, stream));
                }
            });
        }
    };

    if (hasShard)
    {
        ELLM_CHECK(!needCast || outputType == nvinfer1::DataType::kHALF,
            "checkpoint TP cast currently requires an FP16 engine binding");
        ELLM_CHECK(needCast || sourceType == outputType,
            "checkpoint TP shard dtype does not match binding for " + output.getName());
        ELLM_CHECK(source.bytes == static_cast<size_t>(source.shape.volume()) * utils::getTypeSize(sourceType),
            "checkpoint weight byte size does not match metadata for " + output.getName());
        ELLM_CHECK(static_cast<size_t>(output.getMemoryCapacity())
                == static_cast<size_t>(want.volume()) * utils::getTypeSize(outputType),
            "checkpoint shard byte size does not match binding for " + output.getName());

        if (shardAxis == 0)
        {
            int64_t const localElements = want.volume();
            int64_t const sourceElementsPerShard = have.volume() / shardSize;
            ELLM_CHECK(sourceElementsPerShard == localElements,
                "contiguous checkpoint shard element count does not match binding for " + output.getName());
            size_t const sourceOffset = needCast
                ? static_cast<size_t>(shardRank * sourceElementsPerShard) * utils::getTypeSize(sourceType)
                : static_cast<size_t>(shardRank) * static_cast<size_t>(output.getMemoryCapacity());
            if (needCast)
            {
                copyContiguous(sourceOffset, localElements);
            }
            else
            {
                size_t const bytes = static_cast<size_t>(output.getMemoryCapacity());
                size_t const windowBytes = std::max<size_t>(1, kSourceWindowBytes);
                for (size_t first = 0; first < bytes; first += windowBytes)
                {
                    size_t const count = std::min(windowBytes, bytes - first);
                    auto* destination = static_cast<uint8_t*>(output.rawPointer()) + first;
                    runRange(sourceOffset + first, count, [&](View const& mapped) {
                        CUDA_CHECK(kernel::launchCopyBytes(mapped.deviceData, destination, count, stream));
                    });
                }
            }
            return;
        }

        ELLM_CHECK(have.getNumDims() == 2 && shardAxis == 1,
            "non-contiguous TP checkpoint sharding currently supports only matrix axis 1");
        int64_t const rows = have[0];
        int64_t const sourceColumns = have[1];
        int64_t const destinationColumns = want[1];
        int64_t const sourceColumnOffset = static_cast<int64_t>(shardRank) * destinationColumns;
        size_t const sourceRowBytes = static_cast<size_t>(sourceColumns) * utils::getTypeSize(sourceType);
        size_t const destinationRowBytes = static_cast<size_t>(destinationColumns) * utils::getTypeSize(outputType);
        size_t const rowsPerWindow = std::max<size_t>(1, kSourceWindowBytes / sourceRowBytes);
        for (size_t firstRow = 0; firstRow < static_cast<size_t>(rows); firstRow += rowsPerWindow)
        {
            size_t const rowCount = std::min(rowsPerWindow, static_cast<size_t>(rows) - firstRow);
            size_t const rangeOffset = firstRow * sourceRowBytes;
            size_t const rangeBytes = rowCount * sourceRowBytes;
            auto* destination = static_cast<uint8_t*>(output.rawPointer()) + firstRow * destinationRowBytes;
            runRange(rangeOffset, rangeBytes, [&](View const& mapped) {
                if (needCast && sourceType == nvinfer1::DataType::kFLOAT)
                {
                    CUDA_CHECK(kernel::launchFp32SliceToFp16(mapped.deviceData, destination, rowCount, sourceColumns,
                        sourceColumnOffset, destinationColumns, stream));
                }
                else if (needCast)
                {
                    CUDA_CHECK(kernel::launchBf16SliceToFp16(mapped.deviceData, destination, rowCount, sourceColumns,
                        sourceColumnOffset, destinationColumns, stream));
                }
                else
                {
                    size_t const sourceColumnOffsetBytes
                        = static_cast<size_t>(sourceColumnOffset) * utils::getTypeSize(sourceType);
                    CUDA_CHECK(kernel::launchCopy2DBytes(mapped.deviceData, destination, rowCount, sourceRowBytes,
                        sourceColumnOffsetBytes, destinationRowBytes, stream));
                }
            });
        }
        return;
    }

    if (!needTranspose)
    {
        ELLM_CHECK(!needCast || outputType == nvinfer1::DataType::kHALF,
            "checkpoint cast currently requires an FP16 engine binding");
        ELLM_CHECK(needCast || sourceType == outputType,
            "checkpoint weight dtype does not match binding for " + output.getName());
        ELLM_CHECK(source.bytes == static_cast<size_t>(source.shape.volume()) * utils::getTypeSize(sourceType),
            "checkpoint weight byte size does not match metadata for " + output.getName());
        ELLM_CHECK(static_cast<size_t>(output.getMemoryCapacity())
                == static_cast<size_t>(want.volume()) * utils::getTypeSize(outputType),
            "checkpoint weight byte size does not match binding for " + output.getName());
        if (!needCast)
        {
            size_t const bytes = static_cast<size_t>(output.getMemoryCapacity());
            for (size_t first = 0; first < bytes; first += kSourceWindowBytes)
            {
                size_t const count = std::min(kSourceWindowBytes, bytes - first);
                auto* destination = static_cast<uint8_t*>(output.rawPointer()) + first;
                runRange(first, count, [&](View const& mapped) {
                    CUDA_CHECK(kernel::launchCopyBytes(mapped.deviceData, destination, count, stream));
                });
            }
        }
        else
        {
            copyContiguous(0, have.volume());
        }
        return;
    }

    ELLM_CHECK(
        outputType == nvinfer1::DataType::kHALF, "checkpoint transpose currently requires an FP16 engine binding");
    runRange(0, source.bytes, [&](View const& mapped) {
        int32_t const rows = static_cast<int32_t>(have[0]);
        int32_t const columns = static_cast<int32_t>(have[1]);
        if (sourceType == nvinfer1::DataType::kBF16)
        {
            CUDA_CHECK(
                kernel::launchBf16TransposeToFp16(mapped.deviceData, output.rawPointer(), rows, columns, stream));
        }
        else if (sourceType == nvinfer1::DataType::kFLOAT)
        {
            CUDA_CHECK(
                kernel::launchFp32TransposeToFp16(mapped.deviceData, output.rawPointer(), rows, columns, stream));
        }
        else
        {
            ELLM_CHECK(sourceType == nvinfer1::DataType::kHALF,
                "checkpoint transpose only supports FP16, BF16, or FP32 sources");
            CUDA_CHECK(kernel::launchTransposeFp16(mapped.deviceData, output.rawPointer(), rows, columns, stream));
        }
    });
}

void castWeightToFp32(CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() == 1, "FP32 cast expects exactly one checkpoint tensor");
    std::string const& key = keyName(keys, 0);
    View const source = requireView(checkpoint, key, "FP32 cast source");
    validateOutput(output, source.shape, nvinfer1::DataType::kFLOAT, "FP32 cast output");
    if (source.dtype == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(kernel::launchCopyBytes(source.deviceData, output.rawPointer(), source.bytes, stream));
    }
    else if (source.dtype == nvinfer1::DataType::kBF16)
    {
        CUDA_CHECK(kernel::launchBf16ToFp32(source.deviceData, output.rawPointer(), source.shape.volume(), stream));
    }
    else
    {
        ELLM_CHECK(source.dtype == nvinfer1::DataType::kHALF,
            "FP32 cast supports only FP16, BF16, and FP32 checkpoint tensors: " + key);
        CUDA_CHECK(kernel::launchFp16ToFp32(source.deviceData, output.rawPointer(), source.shape.volume(), stream));
    }
}

void assembleGptqFfnQweight(CheckpointReader const& checkpoint, Json const& binding,
    Tensor const* activationPermutation, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(!keys.empty(), "gptq_ffn_qweight missing checkpoint_keys");
    View const qweight = requireView(checkpoint, keyName(keys, 0), "GPTQ qweight");
    ELLM_CHECK(qweight.dtype == nvinfer1::DataType::kINT32 && qweight.shape.getNumDims() == 2,
        "GPTQ qweight must be INT32 [K/8,N]");
    int32_t const K = static_cast<int32_t>(qweight.shape[0]) * 8;
    int32_t const N = static_cast<int32_t>(qweight.shape[1]);

    View qzeros;
    bool hasZeros = false;
    int32_t numGroups = 0;
    int32_t groupSize = K;
    if (keys.size() > 1 && findView(checkpoint, keyName(keys, 1), qzeros) && qzeros.bytes > 0)
    {
        ELLM_CHECK(
            qzeros.dtype == nvinfer1::DataType::kINT32 && qzeros.shape.getNumDims() == 2 && qzeros.shape[1] * 8 == N,
            "GPTQ qzeros must be INT32 [groups,N/8]");
        numGroups = static_cast<int32_t>(qzeros.shape[0]);
        ELLM_CHECK(numGroups > 0 && K % numGroups == 0, "invalid GPTQ group count");
        groupSize = K / numGroups;
        hasZeros = true;
    }

    if (keys.size() > 2)
    {
        View groups;
        std::string const& groupKey = keyName(keys, 2);
        if (findView(checkpoint, groupKey, groups) && groups.bytes > 0)
        {
            if (!hasZeros)
            {
                ELLM_CHECK(groups.dtype == nvinfer1::DataType::kINT32 && groups.shape.volume() == K,
                    "GPTQ g_idx must be INT32 [K]: " + groupKey);
                auto const* values = reinterpret_cast<int32_t const*>(groups.data);
                int32_t maximum = 0;
                for (int32_t k = 0; k < K; ++k)
                {
                    maximum = std::max(maximum, values[k]);
                }
                numGroups = maximum + 1;
                ELLM_CHECK(numGroups > 0 && K % numGroups == 0, "invalid GPTQ g_idx group count");
                groupSize = K / numGroups;
            }
            bool const sequential = validateGptqGroups(groups, K, numGroups, groupSize, groupKey);
            ELLM_CHECK(sequential || activationPermutation != nullptr,
                "Non-sequential GPTQ groups require an activation permutation engine input: " + groupKey);
        }
    }

    auto const* zeros = hasZeros ? reinterpret_cast<int32_t const*>(qzeros.deviceData) : nullptr;
    int32_t const* permutation = nullptr;
    if (activationPermutation != nullptr)
    {
        validateOutput(*activationPermutation, Coords{K}, nvinfer1::DataType::kINT32, "GPTQ activation permutation");
        permutation = static_cast<int32_t const*>(activationPermutation->rawPointer());
    }
    int32_t const zeroPointOffset = binding.value("zero_point_offset", 1);
    if (int4PluginVersion(binding) == 2)
    {
        validateOutput(output, Coords{kernel::int4CuteDslFragmentRows(N, K), 512}, nvinfer1::DataType::kINT8,
            "GPTQ INT4 V2 output");
        CUDA_CHECK(kernel::launchGptqInt4CuteDslRepack(reinterpret_cast<int32_t const*>(qweight.deviceData), zeros,
            permutation, static_cast<int8_t*>(output.rawPointer()), N, K, numGroups, groupSize, zeroPointOffset,
            stream));
    }
    else
    {
        validateOutput(output, Coords{N / 2, K}, nvinfer1::DataType::kINT8, "GPTQ INT4 V1 output");
        CUDA_CHECK(kernel::launchGptqInt4PluginV1Repack(reinterpret_cast<int32_t const*>(qweight.deviceData), zeros,
            permutation, static_cast<int8_t*>(output.rawPointer()), N, K, numGroups, groupSize, zeroPointOffset,
            stream));
    }
}

std::array<int32_t, 3> qkvProjectionWidths(Json const& binding)
{
    Json const widths = binding.value("projection_out_features", Json::array());
    ELLM_CHECK(widths.is_array() && widths.size() == 3, "fused GPTQ Q/K/V binding requires three output widths");
    std::array<int32_t, 3> result{};
    for (size_t index = 0; index < result.size(); ++index)
    {
        result[index] = widths[index].get<int32_t>();
        ELLM_CHECK(result[index] > 0, "fused GPTQ Q/K/V output widths must be positive");
    }
    return result;
}

void assembleGptqQkvQweight(CheckpointReader const& checkpoint, Json const& binding,
    Tensor const* activationPermutation, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() == 9, "fused GPTQ Q/K/V qweight requires three keys per projection");
    std::array<int32_t, 3> const widths = qkvProjectionWidths(binding);

    View const firstQweight = requireView(checkpoint, keyName(keys, 0), "GPTQ Q qweight");
    ELLM_CHECK(firstQweight.dtype == nvinfer1::DataType::kINT32 && firstQweight.shape.getNumDims() == 2,
        "fused GPTQ Q/K/V qweight must be INT32 [K/8,N]");
    int32_t const K = static_cast<int32_t>(firstQweight.shape[0]) * 8;
    int32_t const totalWidth = widths[0] + widths[1] + widths[2];
    View firstGroups;
    bool const hasGroups = findView(checkpoint, keyName(keys, 2), firstGroups) && firstGroups.bytes > 0;
    for (size_t projection = 1; projection < widths.size(); ++projection)
    {
        std::string const& groupKey = keyName(keys, projection * 3 + 2);
        View groups;
        bool const projectionHasGroups = findView(checkpoint, groupKey, groups) && groups.bytes > 0;
        ELLM_CHECK(projectionHasGroups == hasGroups, "Qwen3 GPTQ Q/K/V must use a uniform activation-order layout");
        if (hasGroups)
        {
            ELLM_CHECK(groups.dtype == firstGroups.dtype && groups.shape == firstGroups.shape
                    && groups.bytes == firstGroups.bytes
                    && std::memcmp(groups.data, firstGroups.data, groups.bytes) == 0,
                "Qwen3 GPTQ Q/K/V activation permutations must match: " + groupKey);
        }
    }
    int32_t const version = int4PluginVersion(binding);
    if (version == 1)
    {
        validateOutput(output, Coords{totalWidth / 2, K}, nvinfer1::DataType::kINT8, "fused GPTQ Q/K/V V1 output");
    }
    else
    {
        validateOutput(output, Coords{kernel::int4CuteDslFragmentRows(totalWidth, K), 512}, nvinfer1::DataType::kINT8,
            "fused GPTQ Q/K/V V2 output");
    }

    size_t outputOffset = 0;
    for (size_t projection = 0; projection < widths.size(); ++projection)
    {
        Json projectionKeys = Json::array();
        for (size_t key = 0; key < 3; ++key)
        {
            projectionKeys.push_back(keys[projection * 3 + key]);
        }
        Json projectionBinding{
            {"checkpoint_keys", std::move(projectionKeys)},
            {"int4_gemm_plugin_version", version},
            {"zero_point_offset", binding.value("zero_point_offset", 1)},
        };
        int32_t const width = widths[projection];
        Coords const shape
            = version == 1 ? Coords{width / 2, K} : Coords{kernel::int4CuteDslFragmentRows(width, K), 512};
        Tensor projectionOutput(static_cast<int8_t*>(output.rawPointer()) + outputOffset, shape, DeviceType::kGPU,
            nvinfer1::DataType::kINT8, output.getName());
        assembleGptqFfnQweight(checkpoint, projectionBinding, activationPermutation, projectionOutput, stream);
        outputOffset += static_cast<size_t>(shape.volume());
    }
}

void assembleGptqQkvScales(CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() == 3, "fused GPTQ Q/K/V scales require one key per projection");
    std::array<int32_t, 3> const widths = qkvProjectionWidths(binding);
    std::array<View, 3> sources{
        requireView(checkpoint, keyName(keys, 0), "GPTQ Q scales"),
        requireView(checkpoint, keyName(keys, 1), "GPTQ K scales"),
        requireView(checkpoint, keyName(keys, 2), "GPTQ V scales"),
    };
    int32_t const numGroups = static_cast<int32_t>(sources[0].shape[0]);
    for (size_t projection = 0; projection < sources.size(); ++projection)
    {
        Coords const expectedShape{numGroups, widths[projection]};
        ELLM_CHECK(sources[projection].dtype == nvinfer1::DataType::kHALF && sources[projection].shape == expectedShape,
            "GPTQ Q/K/V scales must be FP16 [groups,output_width]");
    }
    int32_t const totalWidth = widths[0] + widths[1] + widths[2];
    validateOutput(output, Coords{numGroups, totalWidth}, nvinfer1::DataType::kHALF, "fused GPTQ Q/K/V scale output");
    CUDA_CHECK(kernel::launchGptqInt4QkvScaleConcat(sources[0].deviceData, widths[0], sources[1].deviceData, widths[1],
        sources[2].deviceData, widths[2], numGroups, output.rawPointer(), stream));
}

void assembleAwqFfnQweight(CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() >= 2, "awq_ffn_qweight requires qweight and qzeros keys");
    View const qweight = requireView(checkpoint, keyName(keys, 0), "AWQ qweight");
    View const qzeros = requireView(checkpoint, keyName(keys, 1), "AWQ qzeros");
    ELLM_CHECK(qweight.dtype == nvinfer1::DataType::kINT32 && qweight.shape.getNumDims() == 2,
        "AWQ qweight must be INT32 [K,N/8]");
    ELLM_CHECK(qzeros.dtype == nvinfer1::DataType::kINT32 && qzeros.shape.getNumDims() == 2
            && qzeros.shape[1] == qweight.shape[1],
        "AWQ qzeros must be INT32 [groups,N/8]");
    int32_t const K = static_cast<int32_t>(qweight.shape[0]);
    int32_t const N = static_cast<int32_t>(qweight.shape[1]) * 8;
    int32_t const numGroups = static_cast<int32_t>(qzeros.shape[0]);
    ELLM_CHECK(numGroups > 0 && K % numGroups == 0, "invalid AWQ group count");
    int32_t const groupSize = K / numGroups;

    auto const* weight = reinterpret_cast<int32_t const*>(qweight.deviceData);
    auto const* zeros = reinterpret_cast<int32_t const*>(qzeros.deviceData);
    if (int4PluginVersion(binding) == 2)
    {
        validateOutput(output, Coords{kernel::int4CuteDslFragmentRows(N, K), 512}, nvinfer1::DataType::kINT8,
            "AWQ INT4 V2 output");
        CUDA_CHECK(kernel::launchAwqInt4CuteDslRepack(
            weight, zeros, static_cast<int8_t*>(output.rawPointer()), N, K, numGroups, groupSize, stream));
    }
    else
    {
        validateOutput(output, Coords{N / 2, K}, nvinfer1::DataType::kINT8, "AWQ INT4 V1 output");
        CUDA_CHECK(kernel::launchAwqInt4PluginV1Repack(
            weight, zeros, static_cast<int8_t*>(output.rawPointer()), N, K, numGroups, groupSize, stream));
    }
}

void assembleModelOptAwqFfnQweight(
    CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    ELLM_CHECK(keys.size() == 1, "modelopt_awq_ffn_qweight expects one checkpoint key");
    View const source = requireView(checkpoint, keyName(keys, 0), "ModelOpt AWQ weight");
    ELLM_CHECK(source.shape.getNumDims() == 2, "ModelOpt AWQ weight must be [N/2,K]");
    int32_t const nHalf = static_cast<int32_t>(source.shape[0]);
    int32_t const K = static_cast<int32_t>(source.shape[1]);
    int32_t const N = 2 * nHalf;
    ELLM_CHECK(source.bytes == static_cast<size_t>(nHalf) * K, "ModelOpt AWQ weight must contain uint8 bytes");

    if (int4PluginVersion(binding) == 2)
    {
        validateOutput(output, Coords{kernel::int4CuteDslFragmentRows(N, K), 512}, nvinfer1::DataType::kINT8,
            "ModelOpt INT4 V2 output");
        CUDA_CHECK(kernel::launchModelOptInt4CuteDslRepack(
            source.deviceData, static_cast<int8_t*>(output.rawPointer()), N, K, stream));
    }
    else
    {
        validateOutput(output, Coords{N / 2, K}, nvinfer1::DataType::kINT8, "ModelOpt INT4 V1 output");
        CUDA_CHECK(kernel::launchModelOptInt4PluginV1Repack(
            source.deviceData, static_cast<int8_t*>(output.rawPointer()), N, K, stream));
    }
}

template <typename Function>
void forCheckpointSourceBatches(int32_t count, Function&& function)
{
    for (int32_t begin = 0; begin < count; begin += kernel::kCheckpointSourcesPerLaunch)
    {
        function(begin, std::min(kernel::kCheckpointSourcesPerLaunch, count - begin));
    }
}

void stackExpertMatrices(CheckpointReader const& checkpoint, Json const& keys, int32_t numExperts, int32_t keyStride,
    int32_t keyOffset, Tensor& output, int32_t& rows, int32_t& columns, cudaStream_t stream)
{
    ELLM_CHECK(static_cast<int32_t>(keys.size()) >= numExperts * keyStride, "NVFP4 expert key list too short");
    View const first = requireView(checkpoint, keyName(keys, static_cast<size_t>(keyOffset)), "NVFP4 expert tensor");
    ELLM_CHECK(first.shape.getNumDims() == 2, "NVFP4 expert matrix must be rank-2");
    rows = static_cast<int32_t>(first.shape[0]);
    columns = static_cast<int32_t>(first.shape[1]);
    size_t const perExpertBytes = first.bytes;
    ELLM_CHECK(perExpertBytes == static_cast<size_t>(rows) * columns, "NVFP4 expert matrix must contain bytes");
    Coords const outputShape = output.getShape();
    ELLM_CHECK(outputShape.getNumDims() == 3 && outputShape[0] == numExperts && outputShape[1] >= rows
            && outputShape[2] >= columns && output.getDataType() == nvinfer1::DataType::kINT8,
        "NVFP4 stacked matrix shape does not match engine binding");

    auto* destination = static_cast<uint8_t*>(output.rawPointer());
    int32_t const outputRows = static_cast<int32_t>(outputShape[1]);
    int32_t const outputColumns = static_cast<int32_t>(outputShape[2]);
    size_t const outputBytesPerExpert = static_cast<size_t>(outputRows) * outputColumns;
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> sources{};
        for (int32_t local = 0; local < count; ++local)
        {
            int32_t const expert = begin + local;
            size_t const key = static_cast<size_t>(expert * keyStride + keyOffset);
            View const view = requireView(checkpoint, keyName(keys, key), "NVFP4 expert tensor");
            ELLM_CHECK(view.dtype == first.dtype && view.shape == first.shape && view.bytes == perExpertBytes,
                "NVFP4 expert shape mismatch: " + keyName(keys, key));
            sources[static_cast<size_t>(local)] = view.deviceData;
        }
        auto* batchOutput = destination + static_cast<size_t>(begin) * outputBytesPerExpert;
        if (outputRows == rows && outputColumns == columns)
        {
            CUDA_CHECK(kernel::launchCheckpointSourceBatchCopy(
                sources.data(), count, batchOutput, static_cast<int64_t>(perExpertBytes), stream));
        }
        else
        {
            CUDA_CHECK(kernel::launchCheckpointSourceBatchPaddedCopy(
                sources.data(), count, batchOutput, rows, columns, outputRows, outputColumns, stream));
        }
    });
}

void assembleNvfp4Scales(CheckpointReader const& checkpoint, Json const& keys, int32_t numExperts, int32_t keyStride,
    int32_t firstOffset, int32_t secondOffset, bool concatLayout, Tensor& output, cudaStream_t stream)
{
    bool const paired = secondOffset >= 0;
    View const first = requireView(checkpoint, keyName(keys, static_cast<size_t>(firstOffset)), "NVFP4 scale");
    ELLM_CHECK(first.shape.getNumDims() == 2, "NVFP4 scale source must be rank-2");
    int32_t const rows = static_cast<int32_t>(first.shape[0]);
    int32_t const columns = static_cast<int32_t>(first.shape[1]);
    ELLM_CHECK((first.dtype == nvinfer1::DataType::kFP8 || first.dtype == nvinfer1::DataType::kINT8)
            && first.bytes == static_cast<size_t>(rows) * columns,
        "NVFP4 scale source must contain FP8 bytes");
    int32_t const sourceOutputRows = paired ? 2 * rows : rows;
    Coords const outputShape = output.getShape();
    ELLM_CHECK(outputShape.getNumDims() == 6 && outputShape[0] == numExperts
            && outputShape[1] >= (sourceOutputRows + 127) / 128 && outputShape[2] >= (columns + 3) / 4
            && outputShape[3] == 32 && outputShape[4] == 4 && outputShape[5] == 4
            && output.getDataType() == nvinfer1::DataType::kINT8,
        "NVFP4 scale output shape does not match engine binding");

    int32_t const outputRows = static_cast<int32_t>(outputShape[1]) * 128;
    int32_t const outputKsf = static_cast<int32_t>(outputShape[2]) * 4;
    size_t const bytesPerExpert = static_cast<size_t>(outputShape.volume() / numExperts) * sizeof(int8_t);
    auto* destination = static_cast<int8_t*>(output.rawPointer());
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstSources{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondSources{};
        for (int32_t local = 0; local < count; ++local)
        {
            int32_t const expert = begin + local;
            size_t const base = static_cast<size_t>(expert * keyStride);
            View const firstView = requireView(checkpoint, keyName(keys, base + firstOffset), "NVFP4 scale");
            ELLM_CHECK(
                firstView.shape == first.shape && firstView.dtype == first.dtype, "NVFP4 scale expert shape mismatch");
            firstSources[static_cast<size_t>(local)] = firstView.deviceData;
            if (paired)
            {
                View const secondView = requireView(checkpoint, keyName(keys, base + secondOffset), "NVFP4 scale");
                ELLM_CHECK(secondView.shape == first.shape && secondView.dtype == first.dtype,
                    "NVFP4 scale expert shape mismatch");
                secondSources[static_cast<size_t>(local)] = secondView.deviceData;
            }
        }
        auto* batchDestination = reinterpret_cast<int8_t*>(
            reinterpret_cast<uint8_t*>(destination) + static_cast<size_t>(begin) * bytesPerExpert);
        CUDA_CHECK(kernel::launchNvfp4MoeScaleTransformSourceBatchPadded(firstSources.data(),
            paired ? secondSources.data() : nullptr, count, batchDestination, rows, columns, outputRows, outputKsf,
            concatLayout ? kernel::Nvfp4MoeFc1Layout::kConcatenated : kernel::Nvfp4MoeFc1Layout::kInterleaved64,
            stream));
    });
}

float readScaleScalar(View const& view, std::string const& key)
{
    ELLM_CHECK(view.bytes > 0, "Empty scale tensor: " + key);
    if (view.dtype == nvinfer1::DataType::kFLOAT)
    {
        float value;
        std::memcpy(&value, view.data, sizeof(value));
        return value;
    }
    if (view.dtype == nvinfer1::DataType::kHALF)
    {
        __half value;
        std::memcpy(&value, view.data, sizeof(value));
        return __half2float(value);
    }
    if (view.dtype == nvinfer1::DataType::kBF16)
    {
        uint16_t bits;
        std::memcpy(&bits, view.data, sizeof(bits));
        uint32_t const fp32Bits = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &fp32Bits, sizeof(value));
        return value;
    }
    ELLM_CHECK(view.dtype == nvinfer1::DataType::kFP8 || view.dtype == nvinfer1::DataType::kINT8,
        "Unsupported NVFP4 scale dtype: " + key);
    uint8_t const bits = view.data[0];
    float const sign = (bits & 0x80) != 0 ? -1.0f : 1.0f;
    int32_t const exponent = (bits >> 3) & 0xF;
    int32_t const mantissa = bits & 0x7;
    if (exponent == 0)
    {
        return sign * std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

float effectiveNvfp4Alpha(CheckpointReader const& checkpoint, std::string const& key, bool reciprocal)
{
    float value = readScaleScalar(requireView(checkpoint, key, "NVFP4 alpha"), key);
    if (reciprocal)
    {
        ELLM_CHECK(value != 0.0f, "NVFP4 reciprocal alpha is zero: " + key);
        value = 1.0f / value;
    }
    ELLM_CHECK(std::isfinite(value) && value > 0.0f, "NVFP4 alpha must be finite and positive: " + key);
    return value;
}

struct Nvfp4ProjectionSource
{
    View weight;
    View scale;
    int32_t rows;
    int32_t rowBytes;
    int32_t scaleColumns;
};

Nvfp4ProjectionSource requireNvfp4Projection(
    CheckpointReader const& checkpoint, std::string const& weightKey, std::string const& scaleKey)
{
    View const weight = requireView(checkpoint, weightKey, "NVFP4 packed weight");
    View const scale = requireView(checkpoint, scaleKey, "NVFP4 block scale");
    ELLM_CHECK(weight.shape.getNumDims() == 2 && scale.shape.getNumDims() == 2,
        "NVFP4 packed weight and scale must be rank-2");
    int32_t const rows = static_cast<int32_t>(weight.shape[0]);
    int32_t const rowBytes = static_cast<int32_t>(weight.shape[1]);
    int32_t const scaleColumns = static_cast<int32_t>(scale.shape[1]);
    ELLM_CHECK(rows > 0 && rowBytes > 0 && scale.shape[0] == rows && rowBytes == scaleColumns * 8
            && weight.bytes == static_cast<size_t>(rows) * rowBytes
            && scale.bytes == static_cast<size_t>(rows) * scaleColumns
            && (scale.dtype == nvinfer1::DataType::kFP8 || scale.dtype == nvinfer1::DataType::kINT8),
        "NVFP4 provider weight/scale layout is inconsistent");
    return {weight, scale, rows, rowBytes, scaleColumns};
}

void validateNvfp4Projection(
    Nvfp4ProjectionSource const& source, Nvfp4ProjectionSource const& reference, std::string const& key)
{
    ELLM_CHECK(source.weight.shape == reference.weight.shape && source.weight.dtype == reference.weight.dtype
            && source.weight.bytes == reference.weight.bytes && source.scale.shape == reference.scale.shape
            && source.scale.dtype == reference.scale.dtype && source.scale.bytes == reference.scale.bytes,
        "NVFP4 expert projection shape mismatch: " + key);
}

void assembleNormalizedNvfp4Weights(CheckpointReader const& checkpoint, Json const& keys, int32_t numExperts,
    int32_t keyStride, int32_t firstWeightOffset, int32_t secondWeightOffset, int32_t firstScaleOffset,
    int32_t secondScaleOffset, int32_t firstAlphaOffset, int32_t secondAlphaOffset, bool reciprocalAlpha,
    bool concatLayout, Tensor& output, cudaStream_t stream)
{
    bool const paired = secondWeightOffset >= 0;
    Nvfp4ProjectionSource const first = requireNvfp4Projection(checkpoint,
        keyName(keys, static_cast<size_t>(firstWeightOffset)), keyName(keys, static_cast<size_t>(firstScaleOffset)));
    if (paired)
    {
        Nvfp4ProjectionSource const second
            = requireNvfp4Projection(checkpoint, keyName(keys, static_cast<size_t>(secondWeightOffset)),
                keyName(keys, static_cast<size_t>(secondScaleOffset)));
        validateNvfp4Projection(second, first, keyName(keys, static_cast<size_t>(secondWeightOffset)));
    }

    Coords const outputShape = output.getShape();
    int32_t const minimumRows = paired ? 2 * first.rows : first.rows;
    ELLM_CHECK(outputShape.getNumDims() == 3 && outputShape[0] == numExperts && outputShape[1] >= minimumRows
            && outputShape[2] >= first.rowBytes && outputShape[2] % 8 == 0 && (!paired || outputShape[1] % 2 == 0)
            && output.getDataType() == nvinfer1::DataType::kINT8,
        "normalized NVFP4 weight output shape does not match engine binding");

    int32_t const outputRows = static_cast<int32_t>(outputShape[1]);
    int32_t const outputRowBytes = static_cast<int32_t>(outputShape[2]);
    size_t const bytesPerExpert = static_cast<size_t>(outputRows) * outputRowBytes;
    auto* destination = static_cast<uint8_t*>(output.rawPointer());
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstWeights{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondWeights{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstScales{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondScales{};
        std::array<float, kernel::kCheckpointSourcesPerLaunch> firstAlphas{};
        std::array<float, kernel::kCheckpointSourcesPerLaunch> secondAlphas{};
        for (int32_t local = 0; local < count; ++local)
        {
            size_t const base = static_cast<size_t>((begin + local) * keyStride);
            Nvfp4ProjectionSource const firstSource = requireNvfp4Projection(
                checkpoint, keyName(keys, base + firstWeightOffset), keyName(keys, base + firstScaleOffset));
            validateNvfp4Projection(firstSource, first, keyName(keys, base + firstWeightOffset));
            firstWeights[static_cast<size_t>(local)] = firstSource.weight.deviceData;
            firstScales[static_cast<size_t>(local)] = firstSource.scale.deviceData;
            firstAlphas[static_cast<size_t>(local)]
                = effectiveNvfp4Alpha(checkpoint, keyName(keys, base + firstAlphaOffset), reciprocalAlpha);
            if (paired)
            {
                Nvfp4ProjectionSource const secondSource = requireNvfp4Projection(
                    checkpoint, keyName(keys, base + secondWeightOffset), keyName(keys, base + secondScaleOffset));
                validateNvfp4Projection(secondSource, first, keyName(keys, base + secondWeightOffset));
                secondWeights[static_cast<size_t>(local)] = secondSource.weight.deviceData;
                secondScales[static_cast<size_t>(local)] = secondSource.scale.deviceData;
                secondAlphas[static_cast<size_t>(local)]
                    = effectiveNvfp4Alpha(checkpoint, keyName(keys, base + secondAlphaOffset), reciprocalAlpha);
            }
        }
        CUDA_CHECK(kernel::launchNvfp4MoeWeightNormalizeSourceBatchPadded(firstWeights.data(),
            paired ? secondWeights.data() : nullptr, firstScales.data(), paired ? secondScales.data() : nullptr,
            firstAlphas.data(), paired ? secondAlphas.data() : nullptr, count,
            destination + static_cast<size_t>(begin) * bytesPerExpert, first.rows, first.rowBytes, first.scaleColumns,
            outputRows, outputRowBytes,
            concatLayout ? kernel::Nvfp4MoeFc1Layout::kConcatenated : kernel::Nvfp4MoeFc1Layout::kInterleaved64,
            stream));
    });
}

void assembleNormalizedNvfp4Scales(CheckpointReader const& checkpoint, Json const& keys, int32_t numExperts,
    int32_t keyStride, int32_t firstWeightOffset, int32_t secondWeightOffset, int32_t firstScaleOffset,
    int32_t secondScaleOffset, int32_t firstAlphaOffset, int32_t secondAlphaOffset, bool reciprocalAlpha,
    bool concatLayout, Tensor& output, cudaStream_t stream)
{
    bool const paired = secondWeightOffset >= 0;
    Nvfp4ProjectionSource const first = requireNvfp4Projection(checkpoint,
        keyName(keys, static_cast<size_t>(firstWeightOffset)), keyName(keys, static_cast<size_t>(firstScaleOffset)));
    Coords const outputShape = output.getShape();
    int32_t const minimumRows = paired ? 2 * first.rows : first.rows;
    ELLM_CHECK(outputShape.getNumDims() == 6 && outputShape[0] == numExperts
            && outputShape[1] >= (minimumRows + 127) / 128 && outputShape[2] >= (first.scaleColumns + 3) / 4
            && outputShape[3] == 32 && outputShape[4] == 4 && outputShape[5] == 4
            && output.getDataType() == nvinfer1::DataType::kINT8,
        "normalized NVFP4 scale output shape does not match engine binding");

    int32_t const outputRows = static_cast<int32_t>(outputShape[1]) * 128;
    int32_t const outputScaleColumns = static_cast<int32_t>(outputShape[2]) * 4;
    size_t const bytesPerExpert = static_cast<size_t>(outputShape.volume() / numExperts);
    auto* destination = static_cast<int8_t*>(output.rawPointer());
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstWeights{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondWeights{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstScales{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondScales{};
        std::array<float, kernel::kCheckpointSourcesPerLaunch> firstAlphas{};
        std::array<float, kernel::kCheckpointSourcesPerLaunch> secondAlphas{};
        for (int32_t local = 0; local < count; ++local)
        {
            size_t const base = static_cast<size_t>((begin + local) * keyStride);
            Nvfp4ProjectionSource const firstSource = requireNvfp4Projection(
                checkpoint, keyName(keys, base + firstWeightOffset), keyName(keys, base + firstScaleOffset));
            validateNvfp4Projection(firstSource, first, keyName(keys, base + firstWeightOffset));
            firstWeights[static_cast<size_t>(local)] = firstSource.weight.deviceData;
            firstScales[static_cast<size_t>(local)] = firstSource.scale.deviceData;
            firstAlphas[static_cast<size_t>(local)]
                = effectiveNvfp4Alpha(checkpoint, keyName(keys, base + firstAlphaOffset), reciprocalAlpha);
            if (paired)
            {
                Nvfp4ProjectionSource const secondSource = requireNvfp4Projection(
                    checkpoint, keyName(keys, base + secondWeightOffset), keyName(keys, base + secondScaleOffset));
                validateNvfp4Projection(secondSource, first, keyName(keys, base + secondWeightOffset));
                secondWeights[static_cast<size_t>(local)] = secondSource.weight.deviceData;
                secondScales[static_cast<size_t>(local)] = secondSource.scale.deviceData;
                secondAlphas[static_cast<size_t>(local)]
                    = effectiveNvfp4Alpha(checkpoint, keyName(keys, base + secondAlphaOffset), reciprocalAlpha);
            }
        }
        auto* batchDestination = reinterpret_cast<int8_t*>(
            reinterpret_cast<uint8_t*>(destination) + static_cast<size_t>(begin) * bytesPerExpert);
        CUDA_CHECK(kernel::launchNvfp4MoeScaleNormalizeSourceBatchPadded(firstWeights.data(),
            paired ? secondWeights.data() : nullptr, firstScales.data(), paired ? secondScales.data() : nullptr,
            firstAlphas.data(), paired ? secondAlphas.data() : nullptr, count, batchDestination, first.rows,
            first.rowBytes, first.scaleColumns, outputRows, outputScaleColumns,
            concatLayout ? kernel::Nvfp4MoeFc1Layout::kConcatenated : kernel::Nvfp4MoeFc1Layout::kInterleaved64,
            stream));
    });
}

void assembleNvfp4Alpha(CheckpointReader const& checkpoint, Json const& binding, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    int32_t const numExperts = binding.value("num_experts", 0);
    int32_t const keysPerExpert = binding.value("keys_per_expert", 1);
    ELLM_CHECK(numExperts > 0 && numExperts <= 512 && keysPerExpert > 0
            && static_cast<int32_t>(keys.size()) == numExperts * keysPerExpert,
        "NVFP4 alpha key count does not match the expert layout");
    validateOutput(output, Coords{numExperts}, nvinfer1::DataType::kFLOAT, "NVFP4 alpha output");

    bool const reciprocal = binding.value("reciprocal_alpha", false);
    std::array<float, 512> values{};
    for (int32_t expert = 0; expert < numExperts; ++expert)
    {
        size_t const base = static_cast<size_t>(expert * keysPerExpert);
        float value = 0.0F;
        for (int32_t field = 1; field < keysPerExpert; ++field)
        {
            value = std::max(
                value, effectiveNvfp4Alpha(checkpoint, keyName(keys, base + static_cast<size_t>(field)), reciprocal));
        }
        value = std::max(value, effectiveNvfp4Alpha(checkpoint, keyName(keys, base), reciprocal));
        values[static_cast<size_t>(expert)] = value;
    }
    CUDA_CHECK(kernel::launchWriteFp32(values.data(), numExperts, output.rawPointer(), stream));
}

void assembleInt4MoeQweights(CheckpointReader const& checkpoint, Json const& binding, int32_t projectionsPerExpert,
    Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    int32_t const numExperts = binding.value("num_experts", 0);
    int32_t constexpr keysPerProjection = 3;
    ELLM_CHECK(numExperts > 0 && (projectionsPerExpert == 1 || projectionsPerExpert == 2)
            && static_cast<int32_t>(keys.size()) == numExperts * projectionsPerExpert * keysPerProjection,
        "INT4 MoE qweight binding key count mismatch");

    View const first = requireView(checkpoint, keyName(keys, 0), "INT4 MoE qweight");
    ELLM_CHECK(first.dtype == nvinfer1::DataType::kINT32 && first.shape.getNumDims() == 2,
        "INT4 MoE qweight must be provider GPTQ INT32 [K/8,N]");
    int32_t const K = static_cast<int32_t>(first.shape[0]) * 8;
    int32_t const projectionN = static_cast<int32_t>(first.shape[1]);
    int32_t const N = projectionsPerExpert * projectionN;

    View firstZeros;
    bool const hasZeros = findView(checkpoint, keyName(keys, 1), firstZeros) && firstZeros.bytes > 0;
    int32_t numGroups = 0;
    int32_t groupSize = binding.value("group_size", 64);
    if (hasZeros)
    {
        ELLM_CHECK(firstZeros.dtype == nvinfer1::DataType::kINT32 && firstZeros.shape.getNumDims() == 2
                && firstZeros.shape[1] * 8 == projectionN,
            "INT4 MoE qzeros must be INT32 [groups,N/8]");
        numGroups = static_cast<int32_t>(firstZeros.shape[0]);
        ELLM_CHECK(numGroups > 0 && K % numGroups == 0, "invalid INT4 MoE group count");
        groupSize = K / numGroups;
    }

    validateOutput(output, Coords{numExperts, K / 16, 2 * N * 4}, nvinfer1::DataType::kINT8, "INT4 MoE qweight output");
    size_t const bytesPerExpert = static_cast<size_t>(K / 16) * 2 * N * sizeof(int32_t);
    auto* destination = static_cast<uint8_t*>(output.rawPointer());
    int32_t const zeroPointOffset = binding.value("zero_point_offset", 1);

    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<int32_t const*, kernel::kCheckpointSourcesPerLaunch> firstWeights{};
        std::array<int32_t const*, kernel::kCheckpointSourcesPerLaunch> secondWeights{};
        std::array<int32_t const*, kernel::kCheckpointSourcesPerLaunch> firstZeroPointers{};
        std::array<int32_t const*, kernel::kCheckpointSourcesPerLaunch> secondZeroPointers{};
        for (int32_t local = 0; local < count; ++local)
        {
            int32_t const expert = begin + local;
            for (int32_t projection = 0; projection < projectionsPerExpert; ++projection)
            {
                size_t const base
                    = static_cast<size_t>((expert * projectionsPerExpert + projection) * keysPerProjection);
                View const qweight = requireView(checkpoint, keyName(keys, base), "INT4 MoE qweight");
                ELLM_CHECK(qweight.dtype == first.dtype && qweight.shape == first.shape,
                    "INT4 MoE qweight shape mismatch: " + keyName(keys, base));
                auto const* qweightPointer = reinterpret_cast<int32_t const*>(qweight.deviceData);
                if (projection == 0)
                {
                    firstWeights[static_cast<size_t>(local)] = qweightPointer;
                }
                else
                {
                    secondWeights[static_cast<size_t>(local)] = qweightPointer;
                }

                View qzeros;
                bool const projectionHasZeros
                    = findView(checkpoint, keyName(keys, base + 1), qzeros) && qzeros.bytes > 0;
                ELLM_CHECK(projectionHasZeros == hasZeros, "INT4 MoE projections disagree on qzeros");
                if (hasZeros)
                {
                    ELLM_CHECK(qzeros.dtype == firstZeros.dtype && qzeros.shape == firstZeros.shape,
                        "INT4 MoE qzeros shape mismatch: " + keyName(keys, base + 1));
                    auto const* zeroPointer = reinterpret_cast<int32_t const*>(qzeros.deviceData);
                    if (projection == 0)
                    {
                        firstZeroPointers[static_cast<size_t>(local)] = zeroPointer;
                    }
                    else
                    {
                        secondZeroPointers[static_cast<size_t>(local)] = zeroPointer;
                    }
                }

                View groups;
                std::string const& groupKey = keyName(keys, base + 2);
                if (findView(checkpoint, groupKey, groups) && groups.bytes > 0)
                {
                    bool const sequential = validateGptqGroups(groups, K, numGroups, groupSize, groupKey);
                    ELLM_CHECK(
                        sequential, "INT4 MoE checkpoint loading does not support activation-order GPTQ: " + groupKey);
                }
            }
        }

        CUDA_CHECK(kernel::launchGptqMarlinRepackSourceBatch(firstWeights.data(),
            projectionsPerExpert == 2 ? secondWeights.data() : nullptr, hasZeros ? firstZeroPointers.data() : nullptr,
            projectionsPerExpert == 2 && hasZeros ? secondZeroPointers.data() : nullptr,
            reinterpret_cast<int32_t*>(destination + static_cast<size_t>(begin) * bytesPerExpert), count, projectionN,
            K, numGroups, groupSize, zeroPointOffset, stream));
    });
}

void assembleInt4MoeScales(CheckpointReader const& checkpoint, Json const& binding, int32_t projectionsPerExpert,
    Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    int32_t const numExperts = binding.value("num_experts", 0);
    ELLM_CHECK(numExperts > 0 && (projectionsPerExpert == 1 || projectionsPerExpert == 2)
            && static_cast<int32_t>(keys.size()) == numExperts * projectionsPerExpert,
        "INT4 MoE scale binding key count mismatch");
    View const first = requireView(checkpoint, keyName(keys, 0), "INT4 MoE scale");
    ELLM_CHECK(first.dtype == nvinfer1::DataType::kHALF && first.shape.getNumDims() == 2,
        "INT4 MoE scale must be FP16 [groups,N]");
    int32_t const groups = static_cast<int32_t>(first.shape[0]);
    int32_t const projectionN = static_cast<int32_t>(first.shape[1]);
    int32_t const N = projectionsPerExpert * projectionN;
    validateOutput(output, Coords{numExperts, groups, N}, nvinfer1::DataType::kHALF, "INT4 MoE scale output");

    size_t const bytesPerExpert = static_cast<size_t>(groups) * N * sizeof(uint16_t);
    auto* destination = static_cast<uint8_t*>(output.rawPointer());
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint16_t const*, kernel::kCheckpointSourcesPerLaunch> firstSources{};
        std::array<uint16_t const*, kernel::kCheckpointSourcesPerLaunch> secondSources{};
        for (int32_t local = 0; local < count; ++local)
        {
            int32_t const expert = begin + local;
            for (int32_t projection = 0; projection < projectionsPerExpert; ++projection)
            {
                size_t const key = static_cast<size_t>(expert * projectionsPerExpert + projection);
                View const view = requireView(checkpoint, keyName(keys, key), "INT4 MoE scale");
                ELLM_CHECK(view.dtype == first.dtype && view.shape == first.shape,
                    "INT4 MoE scale shape mismatch: " + keyName(keys, key));
                auto const* pointer = reinterpret_cast<uint16_t const*>(view.deviceData);
                if (projection == 0)
                {
                    firstSources[static_cast<size_t>(local)] = pointer;
                }
                else
                {
                    secondSources[static_cast<size_t>(local)] = pointer;
                }
            }
        }
        CUDA_CHECK(kernel::launchInt4MoeScaleRepackSourceBatch(firstSources.data(),
            projectionsPerExpert == 2 ? secondSources.data() : nullptr,
            reinterpret_cast<uint16_t*>(destination + static_cast<size_t>(begin) * bytesPerExpert), count, groups,
            projectionN, stream));
    });
}

kernel::Fp16MoeSourceType fp16MoeSourceType(nvinfer1::DataType dtype)
{
    if (dtype == nvinfer1::DataType::kHALF)
    {
        return kernel::Fp16MoeSourceType::kFp16;
    }
    if (dtype == nvinfer1::DataType::kBF16)
    {
        return kernel::Fp16MoeSourceType::kBf16;
    }
    ELLM_CHECK(dtype == nvinfer1::DataType::kFLOAT, "FP16 MoE checkpoint weights must be FP16, BF16, or FP32");
    return kernel::Fp16MoeSourceType::kFp32;
}

void assembleFp16Moe(
    CheckpointReader const& checkpoint, Json const& binding, bool paired, Tensor& output, cudaStream_t stream)
{
    Json const& keys = checkpointKeys(binding);
    int32_t const numExperts = binding.value("num_experts", 0);
    int32_t const keysPerExpert = paired ? 2 : 1;
    ELLM_CHECK(numExperts > 0 && static_cast<int32_t>(keys.size()) == numExperts * keysPerExpert,
        "FP16 MoE checkpoint key count mismatch");

    View const first = requireView(checkpoint, keyName(keys, 0), "FP16 MoE weight");
    ELLM_CHECK(first.shape.getNumDims() == 2, "FP16 MoE checkpoint weights must be rank-2");
    int32_t const rows = static_cast<int32_t>(first.shape[0]);
    int32_t const columns = static_cast<int32_t>(first.shape[1]);
    int32_t const outputRows = paired ? 2 * rows : rows;
    validateOutput(output, Coords{numExperts, outputRows, columns}, nvinfer1::DataType::kHALF, "FP16 MoE output");
    kernel::Fp16MoeSourceType const sourceType = fp16MoeSourceType(first.dtype);

    size_t const bytesPerExpert = static_cast<size_t>(outputRows) * columns * sizeof(uint16_t);
    auto* destination = static_cast<uint8_t*>(output.rawPointer());
    forCheckpointSourceBatches(numExperts, [&](int32_t begin, int32_t count) {
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> firstSources{};
        std::array<uint8_t const*, kernel::kCheckpointSourcesPerLaunch> secondSources{};
        for (int32_t local = 0; local < count; ++local)
        {
            int32_t const expert = begin + local;
            size_t const key = static_cast<size_t>(expert * keysPerExpert);
            View const firstView = requireView(checkpoint, keyName(keys, key), "FP16 MoE weight");
            ELLM_CHECK(firstView.dtype == first.dtype && firstView.shape == first.shape,
                "FP16 MoE expert shape or dtype mismatch: " + keyName(keys, key));
            firstSources[static_cast<size_t>(local)] = firstView.deviceData;
            if (paired)
            {
                View const secondView = requireView(checkpoint, keyName(keys, key + 1), "FP16 MoE weight");
                ELLM_CHECK(secondView.dtype == first.dtype && secondView.shape == first.shape,
                    "FP16 MoE expert shape or dtype mismatch: " + keyName(keys, key + 1));
                secondSources[static_cast<size_t>(local)] = secondView.deviceData;
            }
        }
        CUDA_CHECK(kernel::launchFp16MoeSourceBatch(firstSources.data(), paired ? secondSources.data() : nullptr, count,
            destination + static_cast<size_t>(begin) * bytesPerExpert, rows, columns, sourceType, stream));
    });
}

void fillGeneratedWeight(Json const& binding, Tensor& output, cudaStream_t stream)
{
    ELLM_CHECK(output.getDataType() == nvinfer1::DataType::kFLOAT,
        "generated weight binding currently requires F32: " + output.getName());
    CUDA_CHECK(kernel::launchFillFp32(
        output.rawPointer(), output.getShape().volume(), binding.value("fill_value", 0.0f), stream));
}

} // namespace

CheckpointWeightPhase checkpointWeightPhase(Json const& binding)
{
    // GPTQ activation-order qweights consume this prepared permutation. Keep
    // recipe knowledge here rather than teaching ExternalWeightManager about
    // individual transform names.
    return binding.value("assemble", "") == "gptq_activation_permutation" ? CheckpointWeightPhase::kPrerequisite
                                                                          : CheckpointWeightPhase::kWeight;
}

bool checkpointWeightManagesSourceRegistration(Json const& binding)
{
    std::string const assemble = binding.value("assemble", "");
    return assemble.empty() || assemble == "identity";
}

void loadCheckpointWeight(CheckpointReader& checkpoint, Json const& binding, std::vector<Tensor> const& preparedWeights,
    Tensor& output, cudaStream_t stream)
{
    ELLM_CHECK(binding.is_object() && binding.contains("engine_name"), "invalid checkpoint weight binding");
    ELLM_CHECK(output.getName() == binding.at("engine_name").get<std::string>(),
        "checkpoint output tensor name does not match binding");
    std::string const assemble = binding.value("assemble", "");
    Tensor const* activationPermutation
        = findPreparedWeight(preparedWeights, binding.value("activation_permutation_engine_name", std::string{}));
    if (assemble == "fill")
    {
        fillGeneratedWeight(binding, output, stream);
    }
    else if (assemble == "cast_to_fp32")
    {
        castWeightToFp32(checkpoint, binding, output, stream);
    }
    else if (assemble.empty() || assemble == "identity")
    {
        loadIdentityWeight(checkpoint, binding, output, stream);
        float const embeddingScale = binding.value("embedding_scale", 1.0f);
        if (embeddingScale != 1.0f)
        {
            ELLM_CHECK(output.getDataType() == nvinfer1::DataType::kHALF, "Embedding scale requires an FP16 tensor");
            CUDA_CHECK(
                kernel::launchScaleFp16(output.rawPointer(), output.getShape().volume(), embeddingScale, stream));
        }
    }
    else if (assemble == "gptq_activation_permutation")
    {
        assembleGptqActivationPermutation(checkpoint, binding, output, stream);
    }
    else if (assemble == "gptq_ffn_qweight")
    {
        assembleGptqFfnQweight(checkpoint, binding, activationPermutation, output, stream);
    }
    else if (assemble == "gptq_qkv_qweight")
    {
        assembleGptqQkvQweight(checkpoint, binding, activationPermutation, output, stream);
    }
    else if (assemble == "gptq_qkv_scales")
    {
        assembleGptqQkvScales(checkpoint, binding, output, stream);
    }
    else if (assemble == "awq_ffn_qweight")
    {
        assembleAwqFfnQweight(checkpoint, binding, output, stream);
    }
    else if (assemble == "modelopt_awq_ffn_qweight")
    {
        assembleModelOptAwqFfnQweight(checkpoint, binding, output, stream);
    }
    else if (assemble == "nvfp4_gated_fc1_qweight")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts * 6, "NVFP4 FC1 qweight key count mismatch");
        assembleNormalizedNvfp4Weights(checkpoint, keys, experts, 6, 0, 1, 2, 3, 4, 5,
            binding.value("reciprocal_alpha", false), binding.value("fc1_layout", "interleave") == "concat", output,
            stream);
    }
    else if (assemble == "nvfp4_gated_fc1_scale")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts * 6, "NVFP4 FC1 scale key count mismatch");
        assembleNormalizedNvfp4Scales(checkpoint, keys, experts, 6, 0, 1, 2, 3, 4, 5,
            binding.value("reciprocal_alpha", false), binding.value("fc1_layout", "interleave") == "concat", output,
            stream);
    }
    else if (assemble == "nvfp4_gated_fc2_qweight")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts * 3, "NVFP4 FC2 qweight key count mismatch");
        assembleNormalizedNvfp4Weights(checkpoint, keys, experts, 3, 0, -1, 1, -1, 2, -1,
            binding.value("reciprocal_alpha", false), true, output, stream);
    }
    else if (assemble == "nvfp4_gated_fc2_scale")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts * 3, "NVFP4 FC2 scale key count mismatch");
        assembleNormalizedNvfp4Scales(checkpoint, keys, experts, 3, 0, -1, 1, -1, 2, -1,
            binding.value("reciprocal_alpha", false), true, output, stream);
    }
    else if (assemble == "nvfp4_expert_qweight")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts, "NVFP4 expert qweight key count mismatch");
        int32_t rows = 0;
        int32_t columns = 0;
        stackExpertMatrices(checkpoint, keys, experts, 1, 0, output, rows, columns, stream);
    }
    else if (assemble == "nvfp4_expert_scale")
    {
        Json const& keys = checkpointKeys(binding);
        int32_t const experts = binding.value("num_experts", 0);
        ELLM_CHECK(
            experts > 0 && static_cast<int32_t>(keys.size()) == experts, "NVFP4 expert scale key count mismatch");
        assembleNvfp4Scales(checkpoint, keys, experts, 1, 0, -1, true, output, stream);
    }
    else if (assemble == "nvfp4_fc1_alpha" || assemble == "nvfp4_fc2_alpha")
    {
        assembleNvfp4Alpha(checkpoint, binding, output, stream);
    }
    else if (assemble == "int4_moe_gate_up")
    {
        assembleInt4MoeQweights(checkpoint, binding, 2, output, stream);
    }
    else if (assemble == "int4_moe_down")
    {
        assembleInt4MoeQweights(checkpoint, binding, 1, output, stream);
    }
    else if (assemble == "int4_moe_gate_up_scales")
    {
        assembleInt4MoeScales(checkpoint, binding, 2, output, stream);
    }
    else if (assemble == "int4_moe_down_scales")
    {
        assembleInt4MoeScales(checkpoint, binding, 1, output, stream);
    }
    else if (assemble == "fp16_moe_fc1")
    {
        assembleFp16Moe(checkpoint, binding, true, output, stream);
    }
    else if (assemble == "fp16_moe_fc2")
    {
        assembleFp16Moe(checkpoint, binding, false, output, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported checkpoint weight assemble op: " + assemble);
    }
}

} // namespace rt
} // namespace trt_edgellm
