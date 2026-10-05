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

#include "common/logger.h"
#include "common/trtUtils.h"
#include "profiling/layerProfiler.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ==================== Enums ====================

//! Benchmarking mode
enum class BenchMode
{
    kNONE, //!< No mode specified (error state)
    kPREFILL,
    kDECODE,
    kEAGLE_VERIFY,
    kEAGLE_DRAFT_PROPOSAL,
    kEAGLE_DRAFT_PREFILL,
    kEAGLE_DRAFT_ACCEPT,
    kDFLASH_DRAFT_PROPOSAL,
    kDFLASH_DRAFT_FIRST_ROUND,
    kDFLASH_VERIFY,
    kDFLASH_DDTREE_BUILD,
    kVISUAL
};

//! Kernel category classification
enum class KernelCategory
{
    MHA,       //!< Attention kernels
    GEMM,      //!< Matrix multiply kernels (linear layers, projections)
    KGEN_OTHER //!< All other kernels (kernel generation, data movement, misc operations)
};

// ==================== Data Structures ====================

//! Layer metadata including ONNX op name, shapes, and tactic name
struct LayerMetadata
{
    std::string onnxOp;
    std::string inputShapes;     //!< e.g., "[1x2048x3584][1x2048x3584]"
    std::string outputShapes;    //!< e.g., "[1x2048x3584]"
    std::string tacticName;      //!< Selected kernel/tactic name
    std::string inputDataTypes;  //!< e.g., "FP16,FP16"
    std::string outputDataTypes; //!< e.g., "FP16"
};

//! Aggregated kernel time breakdown by category
struct KernelTimes
{
    double mhaTimeMs{0.0};
    double gemmTimeMs{0.0};
    double kgenOtherTimeMs{0.0};
    double totalTimeMs{0.0};
};

//! Per-layer timing data across iterations
struct LayerTimingData
{
    std::string name;
    KernelCategory category;
    std::vector<double> timesMs;
};

//! Ordered layer timing collection — preserves original layer order from the model
struct OrderedLayerTimings
{
    std::vector<LayerTimingData> layers;
    std::unordered_map<std::string, size_t> indexMap; //!< name -> index in layers vector

    bool empty() const
    {
        return layers.empty();
    }

    LayerTimingData& getOrInsert(std::string const& name);
};

//! Flags for which layer metadata to extract from the engine inspector
struct ExtractLayerInfo
{
    bool shapes{false};
    bool onnxOps{false};
    bool tactics{false};
    bool dataTypes{false};

    bool any() const
    {
        return shapes || onnxOps || tactics || dataTypes;
    }
};

//! Parameters passed to CSV writers and log helpers describing the current bench configuration
struct BenchOutputParams
{
    BenchMode mode{BenchMode::kNONE};
    int32_t batchSize{1};
    int32_t inputLen{0};
    int32_t pastKVLen{0};
    int32_t verifyTreeSize{0};
    int32_t draftTreeSize{0};
    int32_t acceptLen{0};
    int32_t osl{1};
    int32_t imageHeight{0};
    int32_t imageWidth{0};
    int32_t reuseKVLen{0};
    int32_t iterations{0};
    int32_t acceptRate{5};
    int32_t draftStep{6};
    int32_t blockSize{0};     // DFlash block size
    int32_t draftDeltaLen{0}; // per-round delta length for DFlash draft
    int32_t candidateTopK{0}; // DDTree branching factor
    uint64_t seed{0};         // Random seed used to initialize benchmark inputs
};

// ==================== String Conversions ====================

//! Convert BenchMode enum to human-readable string
std::string modeToString(BenchMode mode);

// ==================== Kernel Utilities ====================

//! Classify a layer into MHA / GEMM / KGEN_OTHER using all available metadata.
//! Best-effort kernel classification using string matching on layer/op/tactic names.
//! Limitations: tactic names (e.g., "cublas", "flash") are TRT-internal and may change across versions.
//! ONNX op "MatMul" may misclassify QKV matmul inside fused attention as GEMM.
//! Priority: tactic name (most specific) > ONNX op > layer name (least specific).
KernelCategory classifyKernel(
    std::string const& layerName, std::string const& onnxOp = "", std::string const& tacticName = "");

//! Extract kernel time breakdown from layer profiler metrics
KernelTimes extractKernelTimes(trt_edgellm::layerProfiler::LayerProfilerMetrics const& metrics,
    std::map<std::string, LayerMetadata> const& layerMetadata = std::map<std::string, LayerMetadata>{});

//! Accumulate layer timings from metrics (preserves original layer order)
void accumulateLayerTimings(
    trt_edgellm::layerProfiler::LayerProfilerMetrics const& metrics, OrderedLayerTimings& layerTimings);

//! Compute mean and standard deviation
std::pair<double, double> computeStats(std::vector<double> const& values);

// ==================== Engine Metadata ====================

//! Load a standalone engine from file for dtype queries and layer metadata extraction
std::unique_ptr<nvinfer1::ICudaEngine> loadStandaloneEngine(std::filesystem::path const& enginePath);

//! Extract layer metadata from a standalone TRT engine using IEngineInspector.
//! Parses each layer's JSON individually (typically a few KB each).
//! Shapes reflect optimization profile bounds (not runtime shapes) since no execution context is set.
std::map<std::string, LayerMetadata> extractLayerMetadata(nvinfer1::ICudaEngine* engine, ExtractLayerInfo const& info);

// ==================== CSV Writers ====================

//! Write per-step layer profiling data to CSV.
//! Layer times are always per single step; total sequence time is captured separately in the E2E CSV.
void writeLayerInfoCsv(OrderedLayerTimings const& layerTimings, std::string const& outputPath,
    BenchOutputParams const& params, int64_t imageTokens = 0,
    std::map<std::string, LayerMetadata> const& layerMetadata = {});

//! Write E2E timing to CSV file
void writeE2ECsv(std::string const& outputPath, BenchOutputParams const& params, float e2eTimeMs, int32_t numTokens,
    int64_t imageTokens = 0);

//! Build the layer CSV file path based on mode and parameters
std::string buildLayerCsvPath(std::string const& outputDir, BenchOutputParams const& params);

//! Build the E2E CSV file path based on mode and parameters
std::string buildE2ECsvPath(std::string const& outputDir, BenchOutputParams const& params);

// ==================== Log Helpers ====================

//! Log bench configuration and mode-specific parameters
void logBenchConfig(BenchOutputParams const& params, int64_t imageTokens = 0);

//! Log results summary (layer profiling breakdown + E2E timing)
void logResultsSummary(BenchOutputParams const& params, std::vector<KernelTimes> const& timesPerIter,
    float e2eTimeMsResult, int64_t imageTokens = 0);
