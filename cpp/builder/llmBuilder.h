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

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <NvInfer.h>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for LLM model building.
//! Contains all parameters needed to configure the TensorRT engine building process
//! for Large Language Models, including standard and speculative-decoding engines.
struct LLMBuilderConfig
{
    int64_t maxInputLen{1024};        //!< Maximum input sequence length for the model
    bool specDraft{false};            //!< Whether this is a speculative-decoding draft model
    bool specBase{false};             //!< Whether this is a speculative-decoding base model
    int64_t maxBatchSize{4};          //!< Maximum batch size for inference
    int64_t maxLoraRank{0};           //!< Maximum LoRA rank (0 = no LoRA support)
    int64_t maxKVCacheCapacity{4096}; //!< Maximum KV cache capacity (sequence length)
    int64_t maxVerifyTreeSize{60};    //!< Maximum length of input_ids passed into spec base model for verification
    int64_t maxDraftTreeSize{60};     //!< Maximum length of input_ids passed into spec draft model for draft generation
    bool profilingDetailed{false};    //!< Enable detailed profiling verbosity for layer info extraction
    //! Exact physical K-page count for the engine's KV pool. Zero selects the minimum active pages.
    //! Kept after the original aggregate fields so existing positional initializers remain source-compatible.
    int64_t maxKVPoolPages{0};

    //! Resolve the exact physical K-page count serialized into the engine binding shape.
    //! @return `maxKVPoolPages`, or the minimum active pages when it is zero
    //! @throws std::runtime_error for invalid dimensions, overflow, or a below-minimum override
    int64_t resolvedKVPoolPages() const
    {
        ELLM_CHECK(maxBatchSize > 0 && maxBatchSize <= std::numeric_limits<int32_t>::max(),
            "LLMBuilderConfig: maxBatchSize must fit a positive int32.");
        ELLM_CHECK(maxKVCacheCapacity > 0 && maxKVCacheCapacity <= rt::kMAX_KV_CACHE_CAPACITY,
            "LLMBuilderConfig: maxKVCacheCapacity must remain int32 after page alignment (maximum "
                + std::to_string(rt::kMAX_KV_CACHE_CAPACITY) + ").");
        int64_t const minimumActivePages = rt::computeMinimumKvPoolPages(maxBatchSize, maxKVCacheCapacity);
        ELLM_CHECK(minimumActivePages <= rt::kMAX_KV_POOL_PAGES,
            "LLMBuilderConfig: minimum active pages (" + std::to_string(minimumActivePages) + ")"
                + " exceeds the largest int32-addressable paged-KV pool " + std::to_string(rt::kMAX_KV_POOL_PAGES)
                + ".");
        ELLM_CHECK(maxKVPoolPages == 0 || maxKVPoolPages >= minimumActivePages,
            "LLMBuilderConfig: maxKVPoolPages (" + std::to_string(maxKVPoolPages)
                + ") must be zero or at least the minimum active pages (" + std::to_string(minimumActivePages) + ").");
        ELLM_CHECK(maxKVPoolPages <= rt::kMAX_KV_POOL_PAGES,
            "LLMBuilderConfig: maxKVPoolPages (" + std::to_string(maxKVPoolPages)
                + ") exceeds the largest int32-addressable paged-KV pool (" + std::to_string(rt::kMAX_KV_POOL_PAGES)
                + ").");
        return maxKVPoolPages == 0 ? minimumActivePages : maxKVPoolPages;
    }

    int64_t tpSize{1}; //!< Tensor parallel size
    int64_t tpRank{0}; //!< Tensor parallel rank

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["max_input_len"] = maxInputLen;
        json["spec_draft"] = specDraft;
        json["spec_base"] = specBase;
        json["max_batch_size"] = maxBatchSize;
        json["max_lora_rank"] = maxLoraRank;
        json["max_kv_cache_capacity"] = maxKVCacheCapacity;
        json["max_kv_pool_pages"] = resolvedKVPoolPages();
        json["tp_size"] = tpSize;
        // Only include speculative-decoding limits for the engine role that owns them.
        if (specBase)
        {
            json["max_verify_tree_size"] = maxVerifyTreeSize;
        }
        if (specDraft)
        {
            json["max_draft_tree_size"] = maxDraftTreeSize;
        }
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return LLMBuilderConfig object with parsed parameters
    //! @throws nlohmann::json::type_error If JSON value types don't match expected types
    static LLMBuilderConfig fromJson(Json const& json)
    {
        LLMBuilderConfig config;
        if (json.contains("max_input_len"))
        {
            config.maxInputLen = json["max_input_len"];
        }
        if (json.contains("spec_draft"))
        {
            config.specDraft = json["spec_draft"];
        }
        else if (json.contains("eagle_draft"))
        {
            config.specDraft = json["eagle_draft"];
        }
        if (json.contains("spec_base"))
        {
            config.specBase = json["spec_base"];
        }
        else if (json.contains("eagle_base"))
        {
            config.specBase = json["eagle_base"];
        }
        if (json.contains("max_batch_size"))
        {
            config.maxBatchSize = json["max_batch_size"];
        }
        if (json.contains("max_lora_rank"))
        {
            config.maxLoraRank = json["max_lora_rank"];
        }
        if (json.contains("max_kv_cache_capacity"))
        {
            config.maxKVCacheCapacity = json["max_kv_cache_capacity"];
        }
        if (json.contains("max_kv_pool_pages"))
        {
            config.maxKVPoolPages = json["max_kv_pool_pages"];
        }
        if (json.contains("max_verify_tree_size"))
        {
            config.maxVerifyTreeSize = json["max_verify_tree_size"];
        }
        if (json.contains("max_draft_tree_size"))
        {
            config.maxDraftTreeSize = json["max_draft_tree_size"];
        }
        if (json.contains("tp_size"))
        {
            config.tpSize = json["tp_size"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "LLMBuilderConfig:\n";
        oss << "  maxInputLen: " << maxInputLen << "\n";
        oss << "  specDraft: " << (specDraft ? "true" : "false") << "\n";
        oss << "  specBase: " << (specBase ? "true" : "false") << "\n";
        oss << "  maxBatchSize: " << maxBatchSize << "\n";
        oss << "  maxLoraRank: " << maxLoraRank << "\n";
        oss << "  maxKVCacheCapacity: " << maxKVCacheCapacity << "\n";
        oss << "  maxKVPoolPages: " << resolvedKVPoolPages() << "\n";
        oss << "  tpSize: " << tpSize << "\n";
        oss << "  tpRank: " << tpRank << "\n";
        // Only show speculative-decoding limits for the engine role that owns them.
        if (specBase)
        {
            oss << "  maxVerifyTreeSize: " << maxVerifyTreeSize << "\n";
        }
        if (specDraft)
        {
            oss << "  maxDraftTreeSize: " << maxDraftTreeSize << "\n";
        }
        return oss.str();
    }
};

//! Builder class for Large Language Model TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for various types of LLMs including standard, speculative-decoding, and VLM models.
class LLMBuilder
{
public:
    //! Constructor for LLMBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    LLMBuilder(
        std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config);

    //! Destructor.
    ~LLMBuilder() noexcept = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    //! @throws std::filesystem::filesystem_error If filesystem operations fail
    bool build();

private:
    std::filesystem::path mOnnxDir;   //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir; //!< Directory for saving built engine
    LLMBuilderConfig mBuilderConfig;  //!< Build configuration

    //! Parse the model configuration from config.json.
    //! Extracts model dimensions and parameters needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    //! @throws json::type_error if JSON value types don't match expected types
    //! @throws json::parse_error if JSON parsing fails
    bool parseConfig();

    //! Set up optimization profiles for LLM models.
    //! Creates context and generation profiles with appropriate dynamic shapes.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    //! @throws json::type_error if JSON value types don't match expected types
    bool setupLLMOptimizationProfiles(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Set up common optimization profiles shared by all LLM types.
    //! Configures context lengths, rotary embeddings, and KV cache profiles.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupCommonProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up RoPE optimization profiles for single-RoPE or dual-RoPE model inputs.
    //! Configures only the RoPE cache bindings that are present in the network.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupRopeProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for vanilla LLM models.
    //! Configures input IDs and last token IDs for standard transformer models.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupVanillaProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Effective opt-shape token count for a generation-time optimization profile.
    //!
    //! Set opt = max for spec-decode base engines. Empirical measurements on Qwen3-8B
    //! (Jetson Thor) show that anchoring the TRT autotuner target at max picks better
    //! kernels for the actual runtime workload than the historical max/2 fallback,
    //! regardless of spec_decode_type or how max relates to the runtime verifyTreeSize:
    //!
    //!   - DFlash `max=64` at runtime ts=64: opt=max is ~9 ms faster than opt=max/2
    //!   - DFlash `max=128` at runtime ts=64: opt=max is ~0.7 ms faster
    //!   - Eagle3 `max=150` at runtime ts=32/64: opt=max is ~1.5 ms faster
    //!   - Eagle3 `max=150` at runtime ts=72 (close to max/2): tied within noise
    //!
    //! opt=max is never worse than max/2 in any tested configuration. See
    //! `dflash-edgellm-vs-zlab/docs/2026-07-13-opt-shape-experiment-findings.md`
    //! for the full dataset.
    //!
    //! @param maxToken The `max` shape count for this profile
    //!                 (e.g., `maxVerifyTreeSize` for verify, `maxDraftTreeSize` for draft).
    int64_t effectiveOptTokens(int64_t maxToken) const;

    //! Set up optimization profiles for speculative tree-decoding models.
    //! Configures hidden-state and attention-mask inputs used by spec decode.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupSpecDecodeProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up optimization profiles for the DiffusionGemma backbone engine.
    //! Configures full-canvas denoise shapes and finalized-token commit shapes.
    //! @param contextProfile Optimization profile for prompt/commit processing
    //! @param generationProfile Optimization profile for denoise processing
    //! @return true if setup was successful, false otherwise
    bool setupDiffusionBackboneProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for DFlash draft models.
    //! DFlash draft consumes proposal embeddings, dflash_target_hidden_concat,
    //! dflash_delta_lengths, and per-layer KV cache bindings.
    bool setupDFlashDraftProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up optimization profiles for Gemma4 assistant draft models.
    //! Gemma4 assistants read base embeddings, target hidden states, target KV,
    //! and dual RoPE caches, but do not own KV cache or tree-mask inputs.
    bool setupGemma4MTPDraftProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for DSpark draft models.
    //! DSpark draft consumes proposal embeddings, target-hidden delta, delta-lengths,
    //! a proposal self-attention page table, and per-layer paged KV cache bindings.
    bool setupDSparkDraftProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up optimization profiles for Gemma4 PLE tensor inputs.
    //! Gemma4 E-model engines receive one ple_token_embeds_* tensor per layer.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupPleProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for Deepstack embeddings (Qwen3VL).
    //! Configures deepstack embedding inputs with the same profile as inputs_embeds.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    //! @throws json::type_error if JSON value types don't match expected types
    bool setupDeepstackProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for lm_head_weight input (CodePredictor models).
    //! CodePredictor has 15 different lm_heads for RVQ layers, and the weight is dynamically bound at runtime.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupLmHeadWeightProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for LoRA-enabled models.
    //! Configures LoRA weight matrices with dynamic rank support.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for LoRA input analysis
    //! @return true if setup was successful, false otherwise
    bool setupLoraProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profiles for KV cache tensors.
    //! Configures dynamic shapes for key-value cache inputs across all layers.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupKVCacheProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up optimization profiles for recurrent state tensors (Mamba/GDN/linear-attention layers).
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupRecurrentStateProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for conv state tensors (causal conv1d layers).
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupConvStateProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for MTP intermediate recurrent state output tensors.
    //! These are per-step checkpoints of GDN recurrent states during tree verification.
    //! Only needed for hybrid MTP/DFlash/JetSpec/DSpark base verification engines.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupIntermediateRecurrentStateProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up optimization profiles for MTP intermediate conv state output tensors.
    //! These are per-step checkpoints of conv1d states during tree verification.
    //! Only needed for hybrid MTP/DFlash/JetSpec/DSpark base verification engines.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupIntermediateConvStateProfiles(
        nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile);

    //! Set up hybrid linear-attention profiles needed only by speculative verification engines.
    //! Configures the shape-only phase marker and optional DDTree parent/depth metadata inputs.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for optional input analysis
    //! @return true if setup was successful, false otherwise
    bool setupLinearAttentionSpecVerifyProfiles(nvinfer1::IOptimizationProfile& contextProfile,
        nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    //! Copy tokenizer files to the engine directory.
    //! Copies tokenizer_config.json and tokenizer.json files needed for inference.
    //! @return true if copying was successful, false otherwise
    bool copyTokenizerFiles();

    //! Copy Eagle-specific files to the engine directory.
    //! Copies d2t.safetensors file for Eagle3 draft models.
    //! @return true if copying was successful, false otherwise
    bool copyEagleFiles();

    //! Copy DSpark Markov/confidence head sidecars to the engine directory.
    //! @return true if copying was successful or this is not a DSpark draft.
    bool copyDSparkFiles();

    //! Copy vocabulary mapping files to the engine directory.
    //! Copies vocab_map.safetensors file if reduced vocabulary is used.
    //! @return true if copying was successful, false otherwise
    //! @throws nlohmann::json::type_error If JSON value types don't match expected types
    bool copyVocabMappingFiles();

    //! Copy embedding table file to the engine directory.
    //! Copies embedding.safetensors file for spec base and vanilla LLM models.
    //! @return true if copying was successful, false otherwise
    bool copyEmbeddingFile();

    //! Copy externalized model weight files to the engine directory.
    //! @return true if copying was successful, false otherwise
    bool copyExternalWeightFiles();

    // Model dimensions extracted from config.json
    int64_t mHiddenSize{0};                   //!< Hidden size of the model
    int64_t mNumKVHeads{0};                   //!< Number of key-value heads
    int64_t mHeadSize{0};                     //!< Size of each attention head
    int64_t mRotaryDim{0};                    //!< Dimension for rotary position embeddings
    int64_t mSlidingRotaryDim{0};             //!< Dimension for sliding-attention rotary embeddings
    int64_t mFullRotaryDim{0};                //!< Dimension for full-attention rotary embeddings
    int32_t mNbKVCacheInputs{0};              //!< Number of KV cache inputs (layers)
    std::vector<int64_t> mPerLayerHeadSize;   //!< Per-layer head size (for heterogeneous models like Gemma4)
    std::vector<int64_t> mPerLayerNumKVHeads; //!< Per-layer KV head count (for heterogeneous models like Gemma4)
    int32_t mTargetModelOutputHiddenDim{0};   //!< Target output hidden dimension
    int32_t mNumDeepstackFeatures{0};         //!< Number of deepstack features (for Qwen3VL)
    // TODO: Use better mechanism to organize model configuration.
    int32_t mNumLinearAttnLayers{0};    //!< Number of recurrent layers (Mamba/GDN/linear-attention)
    int32_t mRecurrentStateNumHeads{0}; //!< Number of recurrent state heads
    int32_t mRecurrentStateHeadDim{0};  //!< Recurrent state head dimension
    int32_t mRecurrentStateSize{0};     //!< Recurrent state size
    int32_t mConvDim{0};                //!< Conv state dimension
    int32_t mConvKernel{0};             //!< Conv kernel size (d_conv)
    Json mModelConfig;                  //!< Parsed model configuration
    Json mSharedModelConfig;            //!< Shared runtime config before rank-local overrides
    bool mIsDiffusionBackbone{false};   //!< Whether this builder builds the DiffusionGemma DLLM engine
    int64_t mDiffusionCanvasLength{0};  //!< DiffusionGemma fixed canvas/block length
};

} // namespace builder
} // namespace trt_edgellm
