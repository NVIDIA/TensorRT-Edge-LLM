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

#include "benchLogger.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>

using namespace trt_edgellm;
using Json = nlohmann::json;

// ==================== String Conversions ====================

std::string modeToString(BenchMode mode)
{
    switch (mode)
    {
    case BenchMode::kPREFILL: return "prefill";
    case BenchMode::kDECODE: return "decode";
    case BenchMode::kEAGLE_VERIFY: return "eagle_verify";
    case BenchMode::kEAGLE_DRAFT_PROPOSAL: return "eagle_draft_proposal";
    case BenchMode::kEAGLE_DRAFT_PREFILL: return "eagle_draft_prefill";
    case BenchMode::kEAGLE_DRAFT_ACCEPT: return "eagle_draft_accept";
    case BenchMode::kDFLASH_DRAFT_PROPOSAL: return "dflash_draft_proposal";
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND: return "dflash_draft_first_round";
    case BenchMode::kDFLASH_VERIFY: return "dflash_verify";
    case BenchMode::kDFLASH_DDTREE_BUILD: return "dflash_ddtree_build";
    case BenchMode::kVISUAL: return "visual";
    default: return "unknown";
    }
}

// ==================== Kernel Utilities ====================

namespace
{

// Check if any of the given strings contain a substring (case-sensitive).
bool containsAny(std::string const& haystack, std::initializer_list<char const*> needles)
{
    for (auto const* needle : needles)
    {
        if (haystack.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

} // anonymous namespace

KernelCategory classifyKernel(std::string const& layerName, std::string const& onnxOp, std::string const& tacticName)
{
    // --- MHA: attention plugins and fused MHA kernels ---
    // Tactic: flash attention, fmha kernels
    if (containsAny(tacticName, {"fmha", "flash", "mha_v2", "mhav2"}))
        return KernelCategory::MHA;
    // ONNX op or layer name: attention plugin layers
    if (containsAny(onnxOp, {"Attention", "attention", "MHA", "mha"})
        || containsAny(layerName, {"Attention", "attention", "MHA", "mha", "fmha", "FMHA"}))
        return KernelCategory::MHA;

    // --- GEMM: MatMul, Conv, projections, int4/fp8 quantized GEMM ---
    // Tactic: CUDA core or tensor core matmul kernels
    if (containsAny(tacticName, {"mma", "gemm", "xmma", "hmma", "imma"}))
        return KernelCategory::GEMM;
    // ONNX op: standard matmul/conv ops, projections
    if (containsAny(onnxOp, {"MatMul", "matmul", "Gemm", "gemm", "Linear", "linear"}))
        return KernelCategory::GEMM;
    // Layer name: plugin names, quantized GEMM
    if (containsAny(layerName, {"Gemm", "gemm", "GEMM", "MatMul", "matmul", "proj"}))
        return KernelCategory::GEMM;

    // --- Everything else: normalization, elementwise, data movement, etc. ---
    return KernelCategory::KGEN_OTHER;
}

LayerTimingData& OrderedLayerTimings::getOrInsert(std::string const& name)
{
    auto it = indexMap.find(name);
    if (it != indexMap.end())
    {
        return layers[it->second];
    }
    size_t idx = layers.size();
    layers.push_back(LayerTimingData{name, classifyKernel(name), {}});
    indexMap[name] = idx;
    return layers[idx];
}

KernelTimes extractKernelTimes(
    layerProfiler::LayerProfilerMetrics const& metrics, std::map<std::string, LayerMetadata> const& layerMetadata)
{
    KernelTimes times;

    for (auto const& layer : metrics.layers)
    {
        double layerTotal = std::accumulate(layer.timeMs.begin(), layer.timeMs.end(), 0.0);
        double avgTime = layer.timeMs.empty() ? 0.0 : layerTotal / layer.timeMs.size();

        std::string onnxOp;
        std::string tacticName;
        auto it = layerMetadata.find(layer.name);
        if (it != layerMetadata.end())
        {
            onnxOp = it->second.onnxOp;
            tacticName = it->second.tacticName;
        }

        switch (classifyKernel(layer.name, onnxOp, tacticName))
        {
        case KernelCategory::MHA: times.mhaTimeMs += avgTime; break;
        case KernelCategory::GEMM: times.gemmTimeMs += avgTime; break;
        case KernelCategory::KGEN_OTHER: times.kgenOtherTimeMs += avgTime; break;
        }
        times.totalTimeMs += avgTime;
    }

    return times;
}

void accumulateLayerTimings(layerProfiler::LayerProfilerMetrics const& metrics, OrderedLayerTimings& layerTimings)
{
    for (auto const& layer : metrics.layers)
    {
        auto& data = layerTimings.getOrInsert(layer.name);
        // Get the current iteration's time (last sample)
        if (!layer.timeMs.empty())
        {
            data.timesMs.push_back(layer.timeMs.back());
        }
    }
}

std::pair<double, double> computeStats(std::vector<double> const& values)
{
    if (values.empty())
        return {0.0, 0.0};
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();
    double sqSum = 0.0;
    for (auto v : values)
        sqSum += (v - mean) * (v - mean);
    double stddev = values.size() > 1 ? std::sqrt(sqSum / (values.size() - 1)) : 0.0;
    return {mean, stddev};
}

// ==================== Engine Metadata ====================

std::unique_ptr<nvinfer1::ICudaEngine> loadStandaloneEngine(std::filesystem::path const& enginePath)
{
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    return deserializeCudaEngineFromFile(*runtime, enginePath);
}

namespace
{

// Helper: extract ONNX op string from "Metadata" field (can be string, array, or object)
std::string extractOnnxOp(Json const& layer)
{
    // Priority: Metadata > Origin > ParameterType
    if (layer.contains("Metadata"))
    {
        auto const& meta = layer["Metadata"];
        if (meta.is_string())
        {
            return meta.get<std::string>();
        }
        if (meta.is_array())
        {
            // ["op1", "op2"] -> "op1 + op2"
            std::string result;
            for (auto const& elem : meta)
            {
                if (elem.is_string())
                {
                    if (!result.empty())
                        result += " + ";
                    result += elem.get<std::string>();
                }
            }
            return result;
        }
        if (meta.is_object())
        {
            // {"onnx_node": "op"} or {"onnx_node": ["op1", "op2"]} or {"origin": "op"}
            for (auto const& key : {"onnx_node", "origin"})
            {
                if (!meta.contains(key))
                    continue;
                auto const& val = meta[key];
                if (val.is_string())
                    return val.get<std::string>();
                if (val.is_array())
                {
                    std::string result;
                    for (auto const& elem : val)
                    {
                        if (elem.is_string())
                        {
                            if (!result.empty())
                                result += " + ";
                            result += elem.get<std::string>();
                        }
                    }
                    return result;
                }
            }
        }
    }
    if (layer.contains("Origin") && layer["Origin"].is_string())
        return layer["Origin"].get<std::string>();
    if (layer.contains("ParameterType") && layer["ParameterType"].is_string())
        return layer["ParameterType"].get<std::string>();
    return "";
}

// Helper: extract tactic string from layer JSON
std::string extractTactic(Json const& layer)
{
    // Priority: TacticName > TacticValue > Tactic (int)
    if (layer.contains("TacticName") && layer["TacticName"].is_string())
        return layer["TacticName"].get<std::string>();
    if (layer.contains("TacticValue") && layer["TacticValue"].is_string())
        return layer["TacticValue"].get<std::string>();
    if (layer.contains("Tactic") && layer["Tactic"].is_number())
        return std::to_string(layer["Tactic"].get<int64_t>());
    return "";
}

// Helper: format Dimensions array as "1x2048x3584"
std::string formatDims(Json const& dims)
{
    std::string result;
    for (auto const& d : dims)
    {
        if (!result.empty())
            result += 'x';
        result += std::to_string(d.get<int64_t>());
    }
    return result;
}

// Helper: extract shapes and data types from Inputs/Outputs array
void extractIOInfo(
    Json const& ioArray, std::string& shapes, std::string& dataTypes, bool wantShapes, bool wantDataTypes)
{
    for (auto const& elem : ioArray)
    {
        if (wantShapes && elem.contains("Dimensions") && elem["Dimensions"].is_array())
        {
            shapes += '[' + formatDims(elem["Dimensions"]) + ']';
        }
        if (wantDataTypes && elem.contains("Format/Datatype") && elem["Format/Datatype"].is_string())
        {
            std::string const& fmt = elem["Format/Datatype"].get_ref<std::string const&>();
            auto pos = fmt.rfind(' ');
            std::string precision = (pos != std::string::npos) ? fmt.substr(pos + 1) : fmt;
            // Filter out plugin placeholder values
            if (precision != "format" && precision != "Format" && !precision.empty())
            {
                if (!dataTypes.empty())
                    dataTypes += ',';
                dataTypes += precision;
            }
        }
    }
}

} // anonymous namespace

std::map<std::string, LayerMetadata> extractLayerMetadata(nvinfer1::ICudaEngine* engine, ExtractLayerInfo const& info)
{
    std::map<std::string, LayerMetadata> layerMetadata;

    if (!engine)
    {
        return layerMetadata;
    }

    auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine->createEngineInspector());
    if (!inspector)
    {
        LOG_WARNING("Failed to create engine inspector");
        return layerMetadata;
    }

    int32_t numLayers = engine->getNbLayers();
    LOG_INFO("Extracting layer metadata for %d layers (this may take a moment)...", numLayers);

    // Temporarily suppress TRT INFO logs during inspector calls (noisy L2 cache management messages).
    auto prevLevel = gLogger.getLevel();
    gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);

    for (int32_t i = 0; i < numLayers; ++i)
    {
        char const* layerInfoStr = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON);
        if (!layerInfoStr)
        {
            continue;
        }

        Json layer;
        try
        {
            layer = Json::parse(layerInfoStr);
        }
        catch (Json::parse_error const&)
        {
            continue;
        }

        if (!layer.contains("Name") || !layer["Name"].is_string())
        {
            continue;
        }

        std::string layerName = layer["Name"].get<std::string>();
        LayerMetadata meta;

        if (info.onnxOps)
        {
            meta.onnxOp = extractOnnxOp(layer);
        }
        if (info.tactics)
        {
            meta.tacticName = extractTactic(layer);
        }
        if (info.shapes || info.dataTypes)
        {
            if (layer.contains("Inputs") && layer["Inputs"].is_array())
            {
                extractIOInfo(layer["Inputs"], meta.inputShapes, meta.inputDataTypes, info.shapes, info.dataTypes);
            }
            if (layer.contains("Outputs") && layer["Outputs"].is_array())
            {
                extractIOInfo(layer["Outputs"], meta.outputShapes, meta.outputDataTypes, info.shapes, info.dataTypes);
            }
        }

        // Plugin layers: TRT inspector reports placeholder types. Clear them.
        if (info.dataTypes && meta.tacticName == "0x0000000000000000")
        {
            meta.inputDataTypes.clear();
            meta.outputDataTypes.clear();
        }

        layerMetadata[std::move(layerName)] = std::move(meta);
    }

    gLogger.setLevel(prevLevel);

    LOG_INFO("Extracted metadata for %zu layers (onnx_ops: %s, shapes: %s, tactics: %s, data_types: %s)",
        layerMetadata.size(), info.onnxOps ? "yes" : "no", info.shapes ? "yes" : "no", info.tactics ? "yes" : "no",
        info.dataTypes ? "yes" : "no");
    return layerMetadata;
}

// ==================== CSV Writers ====================

namespace
{

// Helper to write mode-specific CSV columns (header or data)
void writeModeSpecificCsvHeader(std::ofstream& csvFile, BenchOutputParams const& params)
{
    switch (params.mode)
    {
    case BenchMode::kVISUAL: csvFile << ",image_height,image_width,image_tokens"; break;
    case BenchMode::kPREFILL:
    case BenchMode::kEAGLE_DRAFT_PREFILL: csvFile << ",input_len"; break;
    case BenchMode::kDECODE: csvFile << ",past_kv_len"; break;
    case BenchMode::kEAGLE_VERIFY: csvFile << ",verify_tree_size,past_kv_len"; break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL: csvFile << ",draft_tree_size,past_kv_len"; break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT: csvFile << ",draft_step,accept_len,past_kv_len"; break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL: csvFile << ",seed,block_size,draft_delta_len,past_kv_len"; break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND: csvFile << ",seed,block_size,input_len"; break;
    case BenchMode::kDFLASH_VERIFY: csvFile << ",seed,verify_tree_size,past_kv_len"; break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        csvFile << ",seed,block_size,verify_tree_size,candidate_topk,past_kv_len";
        break;
    default: break;
    }
}

void writeModeSpecificCsvData(std::ofstream& csvFile, BenchOutputParams const& params, int64_t imageTokens)
{
    switch (params.mode)
    {
    case BenchMode::kVISUAL:
        csvFile << "," << params.imageHeight << "," << params.imageWidth << "," << imageTokens;
        break;
    case BenchMode::kPREFILL:
    case BenchMode::kEAGLE_DRAFT_PREFILL: csvFile << "," << params.inputLen; break;
    case BenchMode::kDECODE: csvFile << "," << params.pastKVLen; break;
    case BenchMode::kEAGLE_VERIFY: csvFile << "," << params.verifyTreeSize << "," << params.pastKVLen; break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL: csvFile << "," << params.draftTreeSize << "," << params.pastKVLen; break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT:
        csvFile << "," << params.draftStep << "," << params.acceptLen << "," << params.pastKVLen;
        break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL:
        csvFile << "," << params.seed << "," << params.blockSize << "," << params.draftDeltaLen << ","
                << params.pastKVLen;
        break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND:
        csvFile << "," << params.seed << "," << params.blockSize << "," << params.inputLen;
        break;
    case BenchMode::kDFLASH_VERIFY:
        csvFile << "," << params.seed << "," << params.verifyTreeSize << "," << params.pastKVLen;
        break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        csvFile << "," << params.seed << "," << params.blockSize << "," << params.verifyTreeSize << ","
                << params.candidateTopK << "," << params.pastKVLen;
        break;
    default: break;
    }
}

} // anonymous namespace

void writeLayerInfoCsv(OrderedLayerTimings const& layerTimings, std::string const& outputPath,
    BenchOutputParams const& params, int64_t imageTokens, std::map<std::string, LayerMetadata> const& layerMetadata)
{
    std::ofstream csvFile(outputPath);
    if (!csvFile.is_open())
    {
        LOG_ERROR("Failed to open layer info CSV file: %s", outputPath.c_str());
        return;
    }

    // Write header
    csvFile << "layer_name,onnx_op,input_shapes,output_shapes,tactic_name,input_data_types,output_data_types,"
               "category,time_ms_mean,time_ms_std,time_ms_min,time_ms_max,batch_size";
    writeModeSpecificCsvHeader(csvFile, params);
    csvFile << std::endl;

    // Write each layer's data (preserves original model layer order)
    for (auto const& data : layerTimings.layers)
    {
        if (data.timesMs.empty())
            continue;

        auto [mean, std] = computeStats(data.timesMs);
        double minVal = *std::min_element(data.timesMs.begin(), data.timesMs.end());
        double maxVal = *std::max_element(data.timesMs.begin(), data.timesMs.end());

        // Look up layer metadata (ONNX op name, shapes, data types, and tactic name)
        std::string onnxOp;
        std::string inputShapes;
        std::string outputShapes;
        std::string tacticName;
        std::string inputDataTypes;
        std::string outputDataTypes;
        auto it = layerMetadata.find(data.name);
        if (it != layerMetadata.end())
        {
            onnxOp = it->second.onnxOp;
            inputShapes = it->second.inputShapes;
            outputShapes = it->second.outputShapes;
            tacticName = it->second.tacticName;
            inputDataTypes = it->second.inputDataTypes;
            outputDataTypes = it->second.outputDataTypes;
        }

        // Classify using all available metadata (onnx op + tactic are more accurate than layer name alone)
        KernelCategory category = classifyKernel(data.name, onnxOp, tacticName);
        std::string categoryStr;
        switch (category)
        {
        case KernelCategory::MHA: categoryStr = "mha"; break;
        case KernelCategory::GEMM: categoryStr = "gemm"; break;
        case KernelCategory::KGEN_OTHER: categoryStr = "kgen_other"; break;
        }

        csvFile << std::fixed << std::setprecision(4);
        csvFile << "\"" << data.name << "\",\"" << onnxOp << "\",\"" << inputShapes << "\",\"" << outputShapes
                << "\",\"" << tacticName << "\",\"" << inputDataTypes << "\",\"" << outputDataTypes << "\","
                << categoryStr << "," << mean << "," << std << "," << minVal << "," << maxVal << ","
                << params.batchSize;
        writeModeSpecificCsvData(csvFile, params, imageTokens);
        csvFile << std::endl;
    }

    csvFile.close();
    LOG_INFO("Layer info CSV saved to: %s", outputPath.c_str());
}

void writeE2ECsv(std::string const& outputPath, BenchOutputParams const& params, float e2eTimeMs, int32_t numTokens,
    int64_t imageTokens)
{
    std::ofstream csvFile(outputPath);
    if (!csvFile.is_open())
    {
        LOG_ERROR("Failed to open E2E CSV file: %s", outputPath.c_str());
        return;
    }

    if (params.mode == BenchMode::kDFLASH_DDTREE_BUILD)
    {
        csvFile << "mode,batch_size,e2e_time_ms,trees_per_second";
        writeModeSpecificCsvHeader(csvFile, params);
        csvFile << std::endl;

        csvFile << std::fixed << std::setprecision(4);
        csvFile << modeToString(params.mode) << "," << params.batchSize << "," << e2eTimeMs << ","
                << (1000.0f * params.batchSize / e2eTimeMs);
    }
    else
    {
        // Preserve the established token-based schema for engine modes.
        csvFile << "mode,batch_size,osl,e2e_time_ms,per_token_ms,throughput_tps";
        writeModeSpecificCsvHeader(csvFile, params);
        csvFile << std::endl;

        csvFile << std::fixed << std::setprecision(4);
        csvFile << modeToString(params.mode) << "," << params.batchSize << "," << params.osl << "," << e2eTimeMs << ","
                << (e2eTimeMs / numTokens) << "," << (1000.0f * numTokens / e2eTimeMs);
    }
    writeModeSpecificCsvData(csvFile, params, imageTokens);
    csvFile << std::endl;

    csvFile.close();
    LOG_INFO("E2E timing CSV saved to: %s", outputPath.c_str());
}

std::string buildLayerCsvPath(std::string const& outputDir, BenchOutputParams const& params)
{
    std::string path = outputDir + "/layer_";
    switch (params.mode)
    {
    case BenchMode::kPREFILL: path += "prefill_inputlen" + std::to_string(params.inputLen); break;
    case BenchMode::kEAGLE_DRAFT_PREFILL:
        path += "eagle_draft_prefill_inputlen" + std::to_string(params.inputLen);
        break;
    case BenchMode::kDECODE: path += "decode_pastkvlen" + std::to_string(params.pastKVLen); break;
    case BenchMode::kEAGLE_VERIFY:
        path += "eagle_verify_treesize" + std::to_string(params.verifyTreeSize) + "_pastkvlen"
            + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        path += "eagle_draft_proposal_treesize" + std::to_string(params.draftTreeSize) + "_pastkvlen"
            + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT:
        path += "eagle_draft_accept_draftstep" + std::to_string(params.draftStep) + "_acceptlen"
            + std::to_string(params.acceptLen) + "_pastkvlen" + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL:
        path += "dflash_draft_proposal_bs" + std::to_string(params.blockSize) + "_delta"
            + std::to_string(params.draftDeltaLen) + "_kv" + std::to_string(params.pastKVLen) + "_batch"
            + std::to_string(params.batchSize) + "_seed" + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND:
        path += "dflash_draft_first_round_bs" + std::to_string(params.blockSize) + "_inputlen"
            + std::to_string(params.inputLen) + "_batch" + std::to_string(params.batchSize) + "_seed"
            + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_VERIFY:
        path += "dflash_verify_ts" + std::to_string(params.verifyTreeSize) + "_kv" + std::to_string(params.pastKVLen)
            + "_batch" + std::to_string(params.batchSize) + "_seed" + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        path += "dflash_ddtree_build_bs" + std::to_string(params.blockSize) + "_ts"
            + std::to_string(params.verifyTreeSize) + "_tk" + std::to_string(params.candidateTopK) + "_kv"
            + std::to_string(params.pastKVLen) + "_batch" + std::to_string(params.batchSize) + "_seed"
            + std::to_string(params.seed);
        break;
    case BenchMode::kVISUAL:
        path += "visual_" + std::to_string(params.imageHeight) + "x" + std::to_string(params.imageWidth);
        break;
    default: path += modeToString(params.mode); break;
    }
    path += ".csv";
    return path;
}

std::string buildE2ECsvPath(std::string const& outputDir, BenchOutputParams const& params)
{
    std::string path = outputDir + "/e2e_";
    switch (params.mode)
    {
    case BenchMode::kVISUAL:
        path += "visual_" + std::to_string(params.imageHeight) + "x" + std::to_string(params.imageWidth);
        break;
    case BenchMode::kPREFILL: path += "prefill_inputlen" + std::to_string(params.inputLen); break;
    case BenchMode::kEAGLE_DRAFT_PREFILL:
        path += "eagle_draft_prefill_inputlen" + std::to_string(params.inputLen);
        break;
    case BenchMode::kEAGLE_VERIFY:
        path += "eagle_verify_treesize" + std::to_string(params.verifyTreeSize) + "_pastkvlen"
            + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        path += "eagle_draft_proposal_treesize" + std::to_string(params.draftTreeSize) + "_pastkvlen"
            + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT:
        path += "eagle_draft_accept_draftstep" + std::to_string(params.draftStep) + "_acceptlen"
            + std::to_string(params.acceptLen) + "_pastkvlen" + std::to_string(params.pastKVLen);
        break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL:
        path += "dflash_draft_proposal_bs" + std::to_string(params.blockSize) + "_delta"
            + std::to_string(params.draftDeltaLen) + "_kv" + std::to_string(params.pastKVLen) + "_batch"
            + std::to_string(params.batchSize) + "_seed" + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND:
        path += "dflash_draft_first_round_bs" + std::to_string(params.blockSize) + "_inputlen"
            + std::to_string(params.inputLen) + "_batch" + std::to_string(params.batchSize) + "_seed"
            + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_VERIFY:
        path += "dflash_verify_ts" + std::to_string(params.verifyTreeSize) + "_kv" + std::to_string(params.pastKVLen)
            + "_batch" + std::to_string(params.batchSize) + "_seed" + std::to_string(params.seed);
        break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        path += "dflash_ddtree_build_bs" + std::to_string(params.blockSize) + "_ts"
            + std::to_string(params.verifyTreeSize) + "_tk" + std::to_string(params.candidateTopK) + "_kv"
            + std::to_string(params.pastKVLen) + "_batch" + std::to_string(params.batchSize) + "_seed"
            + std::to_string(params.seed);
        break;
    case BenchMode::kDECODE: path += "decode_pastkvlen" + std::to_string(params.pastKVLen); break;
    default: path += modeToString(params.mode); break;
    }
    path += ".csv";
    return path;
}

// ==================== Log Helpers ====================

void logBenchConfig(BenchOutputParams const& params, int64_t imageTokens)
{
    LOG_INFO("Bench Config:");
    LOG_INFO("  Mode: %s", modeToString(params.mode).c_str());
    LOG_INFO("  Batch Size: %d", params.batchSize);
    LOG_INFO("  Iterations: %d", params.iterations);
    LOG_INFO("  Seed: %" PRIu64, params.seed);
    switch (params.mode)
    {
    case BenchMode::kPREFILL:
        LOG_INFO("  Input Len: %d", params.inputLen);
        LOG_INFO("  Reuse KV Len: %d", params.reuseKVLen);
        break;
    case BenchMode::kDECODE:
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        LOG_INFO("  OSL: %d", params.osl);
        break;
    case BenchMode::kEAGLE_VERIFY:
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        LOG_INFO("  Verify Tree Size: %d", params.verifyTreeSize);
        break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        LOG_INFO("  Draft Tree Size: %d", params.draftTreeSize);
        break;
    case BenchMode::kEAGLE_DRAFT_ACCEPT:
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        LOG_INFO("  Draft Step: %d", params.draftStep);
        LOG_INFO("  Accept Len: %d", params.acceptLen);
        break;
    case BenchMode::kEAGLE_DRAFT_PREFILL:
        LOG_INFO("  Input Len: %d", params.inputLen);
        LOG_INFO("  Reuse KV Len: %d", params.reuseKVLen);
        break;
    case BenchMode::kDFLASH_DRAFT_PROPOSAL:
        LOG_INFO("  Block Size: %d", params.blockSize);
        LOG_INFO("  Draft Delta Len: %d", params.draftDeltaLen);
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        break;
    case BenchMode::kDFLASH_DRAFT_FIRST_ROUND:
        LOG_INFO("  Block Size: %d", params.blockSize);
        LOG_INFO("  Input Len: %d", params.inputLen);
        break;
    case BenchMode::kDFLASH_VERIFY:
        LOG_INFO("  Block Size: %d", params.blockSize);
        LOG_INFO("  Verify Tree Size: %d", params.verifyTreeSize);
        LOG_INFO("  Candidate Top K: %d", params.candidateTopK);
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        break;
    case BenchMode::kDFLASH_DDTREE_BUILD:
        LOG_INFO("  Block Size: %d", params.blockSize);
        LOG_INFO("  Verify Tree Size: %d", params.verifyTreeSize);
        LOG_INFO("  Candidate Top K: %d", params.candidateTopK);
        LOG_INFO("  Past KV Len: %d", params.pastKVLen);
        break;
    case BenchMode::kVISUAL:
        LOG_INFO("  Image Size: %dx%d, Image Tokens: %ld", params.imageHeight, params.imageWidth, imageTokens);
        break;
    default: break;
    }
}

void logResultsSummary(BenchOutputParams const& params, std::vector<KernelTimes> const& timesPerIter,
    float e2eTimeMsResult, int64_t imageTokens)
{
    LOG_INFO("=== Results Summary ===");
    LOG_INFO("Mode: %s", modeToString(params.mode).c_str());
    LOG_INFO("Batch Size: %d", params.batchSize);
    LOG_INFO("Iterations: %d", params.iterations);

    // Layer profiling summary
    if (!timesPerIter.empty())
    {
        std::vector<double> totals, mhas, gemms, kgenOthers;
        for (auto const& t : timesPerIter)
        {
            totals.push_back(t.totalTimeMs);
            mhas.push_back(t.mhaTimeMs);
            gemms.push_back(t.gemmTimeMs);
            kgenOthers.push_back(t.kgenOtherTimeMs);
        }

        auto [totalMean, totalStd] = computeStats(totals);
        auto [mhaMean, mhaStd] = computeStats(mhas);
        auto [gemmMean, gemmStd] = computeStats(gemms);
        auto [kgenOtherMean, kgenOtherStd] = computeStats(kgenOthers);

        LOG_INFO("");
        LOG_INFO("Layer Profiling Time (for breakdown analysis):");
        LOG_INFO("  Total:      %.4f +/- %.4f ms", totalMean, totalStd);
        LOG_INFO("  MHA:        %.4f +/- %.4f ms (%.1f%%)", mhaMean, mhaStd,
            totalMean > 0 ? 100.0 * mhaMean / totalMean : 0.0);
        LOG_INFO("  GEMM:       %.4f +/- %.4f ms (%.1f%%)", gemmMean, gemmStd,
            totalMean > 0 ? 100.0 * gemmMean / totalMean : 0.0);
        LOG_INFO("  Kgen+Other: %.4f +/- %.4f ms (%.1f%%)", kgenOtherMean, kgenOtherStd,
            totalMean > 0 ? 100.0 * kgenOtherMean / totalMean : 0.0);
    }

    // E2E summary
    if (e2eTimeMsResult > 0)
    {
        LOG_INFO("");
        LOG_INFO("E2E Time (actual performance): %.4f ms", e2eTimeMsResult);

        switch (params.mode)
        {
        case BenchMode::kPREFILL:
            LOG_INFO("InputLen: %d, ReuseKVLen: %d, ContextLen: %d", params.inputLen, params.reuseKVLen,
                params.inputLen + params.reuseKVLen);
            LOG_INFO("Tokens/sec (E2E): %.1f", (params.inputLen * 1000.0) / e2eTimeMsResult);
            break;
        case BenchMode::kDECODE:
        {
            int32_t decTokens = (params.osl > 1) ? (params.osl - 1) : 1;
            LOG_INFO("PastKVLen: %d, OSL: %d", params.pastKVLen, params.osl);
            LOG_INFO("Tokens/sec (E2E): %.1f", (decTokens * 1000.0) / e2eTimeMsResult);
            break;
        }
        case BenchMode::kEAGLE_VERIFY:
        {
            LOG_INFO("VerifyTreeSize: %d, PastKVLen: %d, OSL: %d", params.verifyTreeSize, params.pastKVLen, params.osl);
            if (params.osl > 1)
            {
                LOG_INFO("Tokens/sec (E2E): %.1f", ((params.osl - 1) * 1000.0) / e2eTimeMsResult);
            }
            break;
        }
        case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        {
            LOG_INFO("DraftTreeSize: %d, PastKVLen: %d, OSL: %d", params.draftTreeSize, params.pastKVLen, params.osl);
            if (params.osl > 1)
            {
                LOG_INFO("Tokens/sec (E2E): %.1f", ((params.osl - 1) * 1000.0) / e2eTimeMsResult);
            }
            break;
        }
        case BenchMode::kEAGLE_DRAFT_ACCEPT:
        {
            LOG_INFO("DraftStep: %d, AcceptLen: %d, PastKVLen: %d, OSL: %d", params.draftStep, params.acceptLen,
                params.pastKVLen, params.osl);
            if (params.osl > 1)
            {
                LOG_INFO("Tokens/sec (E2E): %.1f", ((params.osl - 1) * 1000.0) / e2eTimeMsResult);
            }
            break;
        }
        case BenchMode::kEAGLE_DRAFT_PREFILL:
            LOG_INFO("InputLen: %d, ReuseKVLen: %d, ContextLen: %d", params.inputLen, params.reuseKVLen,
                params.inputLen + params.reuseKVLen);
            break;
        case BenchMode::kVISUAL:
            LOG_INFO("Image Size: %dx%d, Image Tokens: %ld", params.imageHeight, params.imageWidth, imageTokens);
            break;
        default: break;
        }
    }
}
