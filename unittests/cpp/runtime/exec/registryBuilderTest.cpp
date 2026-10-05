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

#include "runtime/exec/registryBuilder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/kvCacheManager.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

//! Helper: check that a name exists in a name list.
bool hasName(std::vector<std::string> const& names, std::string const& name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

//! Populate `layerTypes` + `kvLayerConfigs` from the scalar fields.
//! Mirrors the fallback path in `parseEngineConfig` (Task B): attention layers
//! come first, then Mamba layers. Per-layer KV config is uniform at
//! `(numKVHeads, headDim)`. Tests that update the scalars must call this after
//! the change so the per-layer registry emission reflects the new counts.
void populateHybridFieldsFromScalars(LLMEngineConfig& cfg)
{
    int32_t const attn = cfg.numAttentionLayers;
    int32_t const mamba = cfg.numLinearAttnLayers;
    cfg.layerTypes.clear();
    cfg.layerTypes.reserve(static_cast<size_t>(attn + mamba));
    for (int32_t i = 0; i < attn; ++i)
    {
        cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
    }
    for (int32_t i = 0; i < mamba; ++i)
    {
        cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
    }
    cfg.kvLayerConfigs.assign(static_cast<size_t>(attn), KVLayerConfig{cfg.numKVHeads, cfg.headDim});
}

//! Helper: create a minimal LLM config for testing.
LLMEngineConfig makeBasicLLMConfig()
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 4096;
    cfg.outputVocabSize = 32000;
    cfg.numAttentionLayers = 32;
    cfg.numDecoderLayers = 32;
    cfg.numKVHeads = 8;
    cfg.headDim = 128;
    cfg.rotaryDim = 128;
    cfg.maxSupportedBatchSize = 4;
    cfg.maxSupportedInputLength = 2048;
    cfg.maxKVCacheCapacity = 4096;
    int64_t const minimumActivePages = computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    cfg.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    cfg.raggedBackend = RaggedBackendKind::kEntryPaddedCompatibility;
    cfg.maxNumSequences = cfg.maxSupportedBatchSize;
    cfg.maxQueryLength = cfg.maxSupportedInputLength;
    cfg.maxPhysicalTokens = cfg.maxNumSequences * cfg.maxQueryLength;
    cfg.recurrentPoolRows = cfg.maxNumSequences;
    populateHybridFieldsFromScalars(cfg);
    return cfg;
}

LLMEngineConfig makeRaggedLLMConfig()
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    return cfg;
}

} // namespace

// =====================================================================
// buildRegistryForLLM — standard (plugin KV cache) mode
// =====================================================================

TEST(RegistryBuilderTest, StandardLLMHasExpectedTensors)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    // Core I/O
    EXPECT_TRUE(hasName(names, "inputs_embeds"));
    EXPECT_TRUE(hasName(names, "logits"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kPositions));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kQueryStartOffsets));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kExecutionPhaseMarker));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kContextSequenceCountCarrier));
    EXPECT_FALSE(hasName(names, "context_lengths"));
    EXPECT_FALSE(hasName(names, "last_token_ids"));
    EXPECT_FALSE(hasName(names, "kvcache_start_index"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kKVPageTable));
    EXPECT_TRUE(hasName(names, "rope_rotary_cos_sin"));

    // KV cache: 32 layers x 2 (past + present) = 64 entries
    EXPECT_TRUE(hasName(names, "past_key_values_0"));
    EXPECT_TRUE(hasName(names, "past_key_values_31"));
    EXPECT_TRUE(hasName(names, "present_key_values_0"));
    EXPECT_TRUE(hasName(names, "present_key_values_31"));

    EXPECT_EQ(names.size(), 77u);
}

TEST(RegistryBuilderTest, RaggedLLMUsesTokenMajorAbiBindings)
{
    namespace bn = trt_edgellm::binding_names;
    LLMEngineConfig const cfg = makeRaggedLLMConfig();
    auto const specs = buildRegistryForLLM(cfg).allExpandedSpecs();
    auto const find = [&](char const* name) {
        return std::find_if(specs.begin(), specs.end(), [&](TensorSpec const& spec) { return spec.name == name; });
    };

    auto const inputs = find(bn::kInputsEmbeds);
    ASSERT_NE(inputs, specs.end());
    ASSERT_EQ(inputs->shape.size(), 2U);
    EXPECT_EQ(inputs->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(inputs->shape[1].value, cfg.hiddenSize);

    struct ExpectedInput
    {
        char const* name;
        nvinfer1::DataType dtype;
        int64_t InferenceDims::* extent;
    };
    std::vector<ExpectedInput> const expected{{bn::kPositions, nvinfer1::DataType::kINT32, &InferenceDims::seqLen},
        {bn::kQueryStartOffsets, nvinfer1::DataType::kINT32, &InferenceDims::queryOffsetLen},
        {bn::kQueryLengths, nvinfer1::DataType::kINT32, &InferenceDims::batch},
        {bn::kPastLengths, nvinfer1::DataType::kINT32, &InferenceDims::batch},
        {bn::kAttentionSequenceLengths, nvinfer1::DataType::kINT32, &InferenceDims::batch},
        {bn::kStateIndices, nvinfer1::DataType::kINT32, &InferenceDims::batch},
        {bn::kLogitsIndices, nvinfer1::DataType::kINT64, &InferenceDims::selectLen},
        {bn::kExecutionPhaseMarker, nvinfer1::DataType::kINT32, &InferenceDims::executionPhaseLen},
        {bn::kContextSequenceCountCarrier, nvinfer1::DataType::kINT32, &InferenceDims::contextSequenceCount}};
    for (auto const& item : expected)
    {
        auto const it = find(item.name);
        ASSERT_NE(it, specs.end()) << item.name;
        EXPECT_EQ(it->dtype, item.dtype) << item.name;
        ASSERT_EQ(it->shape.size(), 1U) << item.name;
        EXPECT_EQ(it->shape[0].symbol, item.extent) << item.name;
    }

    EXPECT_EQ(find(bn::kContextLengths), specs.end());
    EXPECT_EQ(find(bn::kLastTokenIds), specs.end());
    auto const pageTable = find(bn::kKVPageTable);
    ASSERT_NE(pageTable, specs.end());
    ASSERT_EQ(pageTable->shape.size(), 3U);
    EXPECT_EQ(pageTable->shape[0].symbol, &InferenceDims::batch);
}

TEST(RegistryBuilderTest, DiffusionBackboneHasSelectorAndPhaseTensors)
{
    namespace bn = trt_edgellm::binding_names;
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.isDiffusionBackbone = true;
    cfg.contextMaskSelectorEnabled = true;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_TRUE(hasName(names, bn::kInputsEmbeds));
    EXPECT_TRUE(hasName(names, bn::kLogits));
    EXPECT_TRUE(hasName(names, bn::kPhaseIsEncoder));
    EXPECT_TRUE(hasName(names, bn::kSelectTokenIndices));
    EXPECT_FALSE(hasName(names, bn::kLogitsIndices));
    EXPECT_TRUE(hasName(names, bn::kContextMaskSelector));
    EXPECT_TRUE(hasName(names, bn::kKVPageTable));
    EXPECT_FALSE(hasName(names, bn::kLastTokenIds));

    auto specs = reg.allExpandedSpecs();
    auto const logits
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kLogits; });
    ASSERT_NE(logits, specs.end());
    EXPECT_EQ(logits->io, TensorIO::kOutput);
    EXPECT_EQ(logits->dtype, nvinfer1::DataType::kFLOAT);
    ASSERT_EQ(logits->shape.size(), 2u);
    EXPECT_TRUE(logits->shape[0].isSymbolic());
    EXPECT_EQ(logits->shape[0].symbol, &InferenceDims::selectLen);

    auto const selector = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kContextMaskSelector; });
    ASSERT_NE(selector, specs.end());
    ASSERT_EQ(selector->shape.size(), 1u);
    EXPECT_TRUE(selector->shape[0].isSymbolic());
    EXPECT_EQ(selector->shape[0].symbol, &InferenceDims::contextMaskSelectorLen);
}

TEST(RegistryBuilderTest, DiffusionBackboneUnifiedConditioningAddsInputs)
{
    namespace bn = trt_edgellm::binding_names;
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.isDiffusionBackbone = true;
    cfg.contextMaskSelectorEnabled = true;
    cfg.diffusionUnifiedConditioning = true;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_TRUE(hasName(names, bn::kCanvasIds));
    EXPECT_TRUE(hasName(names, bn::kPrevSelfConditioningEmbeds));
    EXPECT_TRUE(hasName(names, bn::kSelfConditioningTemperature));
    EXPECT_TRUE(hasName(names, bn::kNextSelfConditioningEmbeds));
    EXPECT_FALSE(hasName(names, "prev_logits"));
    EXPECT_FALSE(hasName(names, "temperature"));
    EXPECT_FALSE(hasName(names, "prev_logits_valid"));
    EXPECT_FALSE(hasName(names, "conditioned_inputs_embeds"));

    auto specs = reg.allExpandedSpecs();
    auto const prevFeedback = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kPrevSelfConditioningEmbeds; });
    ASSERT_NE(prevFeedback, specs.end());
    EXPECT_EQ(prevFeedback->io, TensorIO::kInput);
    EXPECT_EQ(prevFeedback->dtype, nvinfer1::DataType::kHALF);
    ASSERT_EQ(prevFeedback->shape.size(), 2u);
    EXPECT_EQ(prevFeedback->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(prevFeedback->shape[1].value, cfg.hiddenSize);

    auto const nextFeedback = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kNextSelfConditioningEmbeds; });
    ASSERT_NE(nextFeedback, specs.end());
    EXPECT_EQ(nextFeedback->io, TensorIO::kOutput);
    EXPECT_EQ(nextFeedback->dtype, nvinfer1::DataType::kHALF);
    ASSERT_EQ(nextFeedback->shape.size(), 2u);
    EXPECT_EQ(nextFeedback->shape[0].symbol, &InferenceDims::selectLen);
    EXPECT_EQ(nextFeedback->shape[1].value, cfg.hiddenSize);
}

TEST(RegistryBuilderTest, KVCacheBindingUsesEnginePoolPages)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.kvPoolPages += 7;

    auto const specs = buildRegistryForLLM(cfg).allExpandedSpecs();
    auto const it = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& spec) { return spec.name == "past_key_values_0"; });

    ASSERT_NE(it, specs.end());
    ASSERT_EQ(it->shape.size(), 5U);
    EXPECT_EQ(it->shape[1].value, cfg.kvPoolPages);
}

TEST(RegistryBuilderTest, StandardLLMHasCorrectSpecAttributes)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    // Find inputs_embeds and check its properties
    auto it = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "inputs_embeds"; });
    ASSERT_NE(it, specs.end());
    EXPECT_EQ(it->io, TensorIO::kInput);
    EXPECT_EQ(it->dtype, nvinfer1::DataType::kHALF);
    EXPECT_EQ(it->shape.size(), 2u);
    EXPECT_TRUE(it->shape[0].isSymbolic());
    EXPECT_EQ(it->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(it->shape[1].value, 4096);

    // Find logits
    auto logIt = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "logits"; });
    ASSERT_NE(logIt, specs.end());
    EXPECT_EQ(logIt->io, TensorIO::kOutput);
    EXPECT_EQ(logIt->dtype, nvinfer1::DataType::kFLOAT);
}

// =====================================================================
// Deepstack
// =====================================================================

TEST(RegistryBuilderTest, DeepstackAddsExtraTensors)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.numDeepstackFeatures = 3;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_TRUE(hasName(names, "deepstack_embeds_0"));
    EXPECT_TRUE(hasName(names, "deepstack_embeds_1"));
    EXPECT_TRUE(hasName(names, "deepstack_embeds_2"));

    EXPECT_EQ(names.size(), 20u);
}

TEST(RegistryBuilderTest, DeepstackShapeMatchesConfig)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 1;
    cfg.numDeepstackFeatures = 1;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    auto it
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "deepstack_embeds_0"; });
    ASSERT_NE(it, specs.end());
    EXPECT_EQ(it->io, TensorIO::kInput);
    EXPECT_EQ(it->dtype, nvinfer1::DataType::kHALF);
    EXPECT_EQ(it->shape.size(), 2u);
    EXPECT_TRUE(it->shape[0].isSymbolic());
    EXPECT_EQ(it->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(it->shape[1].value, 4096);
}

TEST(RegistryBuilderTest, NoDeepstackWhenFeatureCountIsZero)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 1;
    cfg.numDeepstackFeatures = 0;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_FALSE(hasName(names, "deepstack_embeds_0"));
}

// =====================================================================
// SpecDecode speculative decoding (base engine side)
// =====================================================================

TEST(RegistryBuilderTest, SpecDecodeBaseAddsProposalTensors)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.isSpecDecodeBase = true;
    cfg.maxVerifyTreeSize = 16;
    cfg.maxDraftTreeSize = 16;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_TRUE(hasName(names, "hidden_states"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kAttentionMask));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kAttentionPosId));

    EXPECT_EQ(names.size(), 20u);
}

TEST(RegistryBuilderTest, SpecDecodeBaseUsesConfiguredOutputHiddenDim)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.isSpecDecodeBase = true;

    populateHybridFieldsFromScalars(cfg);
    int32_t constexpr kBaseOutputHiddenDim = 4096;
    auto reg = buildRegistryForLLM(cfg, kBaseOutputHiddenDim);
    auto specs = reg.allExpandedSpecs();

    auto hiddenIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "hidden_states"; });
    ASSERT_NE(hiddenIt, specs.end());
    ASSERT_EQ(hiddenIt->shape.size(), 2u);
    EXPECT_EQ(hiddenIt->shape[1].value, kBaseOutputHiddenDim);
}

TEST(RegistryBuilderTest, NoSpecDecodeTensorsWhenDisabled)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.isSpecDecodeBase = false;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_FALSE(hasName(names, "hidden_states"));
    EXPECT_FALSE(hasName(names, trt_edgellm::binding_names::kAttentionMask));
    EXPECT_FALSE(hasName(names, trt_edgellm::binding_names::kAttentionPosId));
}

// =====================================================================
// Mamba / recurrent state
// =====================================================================

TEST(RegistryBuilderTest, MambaStateAddsRecurrentAndConvTensors)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 4;
    cfg.numLinearAttnLayers = 2;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    // Recurrent state: 2 layers x 2 (past + present) = 4
    EXPECT_TRUE(hasName(names, "recurrent_state_0"));
    EXPECT_TRUE(hasName(names, "recurrent_state_1"));
    EXPECT_TRUE(hasName(names, "present_recurrent_state_0"));
    EXPECT_TRUE(hasName(names, "present_recurrent_state_1"));

    // Conv state: 2 layers x 2 (past + present) = 4
    EXPECT_TRUE(hasName(names, "conv_state_0"));
    EXPECT_TRUE(hasName(names, "conv_state_1"));
    EXPECT_TRUE(hasName(names, "present_conv_state_0"));
    EXPECT_TRUE(hasName(names, "present_conv_state_1"));

    EXPECT_EQ(names.size(), 25u);
}

TEST(RegistryBuilderTest, RecurrentStateShapeMatchesConfig)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 2;
    cfg.numLinearAttnLayers = 1;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    auto recIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "recurrent_state_0"; });
    ASSERT_NE(recIt, specs.end());
    EXPECT_EQ(recIt->shape.size(), 4u);
    EXPECT_FALSE(recIt->shape[0].isSymbolic());
    EXPECT_EQ(recIt->shape[0].value, cfg.recurrentPoolRows);
    EXPECT_EQ(recIt->shape[1].value, 16);  // numHeads
    EXPECT_EQ(recIt->shape[2].value, 64);  // headDim
    EXPECT_EQ(recIt->shape[3].value, 128); // stateSize

    auto convIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "conv_state_0"; });
    ASSERT_NE(convIt, specs.end());
    EXPECT_EQ(convIt->shape.size(), 3u);
    EXPECT_FALSE(convIt->shape[0].isSymbolic());
    EXPECT_EQ(convIt->shape[0].value, cfg.recurrentPoolRows);
    EXPECT_EQ(convIt->shape[1].value, 256); // convDim
    EXPECT_EQ(convIt->shape[2].value, 4);   // convKernel
}

TEST(RegistryBuilderTest, NoRecurrentStateWhenZeroLinearLayers)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.numLinearAttnLayers = 0;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_FALSE(hasName(names, "recurrent_state_0"));
    EXPECT_FALSE(hasName(names, "conv_state_0"));
}

// =====================================================================
// Combined features
// =====================================================================

TEST(RegistryBuilderTest, AllFeaturesEnabled)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 4;
    cfg.numDeepstackFeatures = 2;
    cfg.isSpecDecodeBase = true;
    cfg.specDecodeType = SpecDecodeMode::kMTP;
    cfg.numLinearAttnLayers = 2;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_EQ(names.size(), 34u);
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kExecutionPhaseMarker));
    EXPECT_TRUE(hasName(names, "intermediate_recurrent_state_0"));
    EXPECT_TRUE(hasName(names, "intermediate_recurrent_state_1"));
    EXPECT_TRUE(hasName(names, "intermediate_conv_state_0"));
    EXPECT_TRUE(hasName(names, "intermediate_conv_state_1"));
}

TEST(RegistryBuilderTest, MtpBaseAddsIntermediateStateOutputs)
{
    // Hybrid MTP base → engine emits intermediate state outputs per mamba layer.
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 4;
    cfg.isSpecDecodeBase = true;
    cfg.specDecodeType = SpecDecodeMode::kMTP;
    cfg.maxVerifyTreeSize = 4;
    cfg.numLinearAttnLayers = 2;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();
    auto names = reg.allTensorNames();

    EXPECT_TRUE(hasName(names, "intermediate_recurrent_state_0"));
    EXPECT_TRUE(hasName(names, "intermediate_recurrent_state_1"));
    EXPECT_TRUE(hasName(names, "intermediate_conv_state_0"));
    EXPECT_TRUE(hasName(names, "intermediate_conv_state_1"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kExecutionPhaseMarker));

    auto markerIt = std::find_if(specs.begin(), specs.end(),
        [](TensorSpec const& s) { return s.name == trt_edgellm::binding_names::kExecutionPhaseMarker; });
    ASSERT_NE(markerIt, specs.end());
    EXPECT_EQ(markerIt->io, TensorIO::kInput);
    ASSERT_EQ(markerIt->shape.size(), 1u);
    EXPECT_TRUE(markerIt->shape[0].isSymbolic());
    EXPECT_EQ(markerIt->shape[0].symbol, &InferenceDims::executionPhaseLen);

    // Shape: [T_exec, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
    auto irecIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "intermediate_recurrent_state_0"; });
    ASSERT_NE(irecIt, specs.end());
    EXPECT_EQ(irecIt->io, TensorIO::kOutput);
    ASSERT_EQ(irecIt->shape.size(), 4u);
    EXPECT_EQ(irecIt->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(irecIt->shape[1].value, 16);
    EXPECT_EQ(irecIt->shape[2].value, 64);
    EXPECT_EQ(irecIt->shape[3].value, 128);

    // Shape: [T_exec, convDim, convKernel]
    auto iconvIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "intermediate_conv_state_0"; });
    ASSERT_NE(iconvIt, specs.end());
    EXPECT_EQ(iconvIt->io, TensorIO::kOutput);
    ASSERT_EQ(iconvIt->shape.size(), 3u);
    EXPECT_EQ(iconvIt->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(iconvIt->shape[1].value, 256);
    EXPECT_EQ(iconvIt->shape[2].value, 4);
}

TEST(RegistryBuilderTest, DSparkBaseAddsSpecVerifyPhaseMarker)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 4;
    cfg.isSpecDecodeBase = true;
    cfg.specDecodeType = SpecDecodeMode::kDSpark;
    cfg.numLinearAttnLayers = 2;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto const specs = buildRegistryForLLM(cfg).allExpandedSpecs();
    auto const markerIt = std::find_if(specs.begin(), specs.end(),
        [](TensorSpec const& spec) { return spec.name == trt_edgellm::binding_names::kExecutionPhaseMarker; });

    ASSERT_NE(markerIt, specs.end());
    EXPECT_EQ(markerIt->io, TensorIO::kInput);
    ASSERT_EQ(markerIt->shape.size(), 1U);
    EXPECT_TRUE(markerIt->shape[0].isSymbolic());
    EXPECT_EQ(markerIt->shape[0].symbol, &InferenceDims::executionPhaseLen);
}

TEST(RegistryBuilderTest, NoIntermediateStatesWhenSpecDecodeDisabled)
{
    // Hybrid base WITHOUT SpecDecode → no intermediate state outputs.
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 4;
    cfg.numLinearAttnLayers = 2;
    cfg.recurrentStateNumHeads = 16;
    cfg.recurrentStateHeadDim = 64;
    cfg.recurrentStateSize = 128;
    cfg.convDim = 256;
    cfg.convKernel = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    EXPECT_FALSE(hasName(names, "intermediate_recurrent_state_0"));
    EXPECT_FALSE(hasName(names, "intermediate_conv_state_0"));
}

// =====================================================================
// buildRegistryForSpecDecodeDraft
// =====================================================================

TEST(RegistryBuilderTest, DraftEngineHasExpectedTensors)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 4;
    cfg.numDecoderLayers = 4;
    cfg.isSpecDecodeBase = true;
    cfg.maxVerifyTreeSize = 16;
    cfg.maxDraftTreeSize = 16;

    populateHybridFieldsFromScalars(cfg);
    DeploymentConfig bundle;
    bundle.draft = cfg;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 12288;
    specConfig.draftHiddenSize = 2048;
    bundle.specConfig = specConfig;
    auto reg = buildRegistryForSpecDecodeDraft(bundle);
    auto names = reg.allTensorNames();

    // Core I/O
    EXPECT_TRUE(hasName(names, "inputs_embeds"));
    EXPECT_TRUE(hasName(names, "hidden_states_input"));
    EXPECT_TRUE(hasName(names, "hidden_states_from_draft"));
    EXPECT_FALSE(hasName(names, "last_token_ids"));
    EXPECT_FALSE(hasName(names, "context_lengths"));
    EXPECT_FALSE(hasName(names, "kvcache_start_index"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kExecutionPhaseMarker));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kKVPageTable));
    EXPECT_TRUE(hasName(names, "rope_rotary_cos_sin"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kAttentionMask));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kAttentionPosId));

    // Outputs
    EXPECT_TRUE(hasName(names, "logits"));
    EXPECT_TRUE(hasName(names, "hidden_states"));

    // KV cache (4 layers, plugin mode)
    EXPECT_TRUE(hasName(names, "past_key_values_0"));
    EXPECT_TRUE(hasName(names, "past_key_values_3"));
    EXPECT_TRUE(hasName(names, "present_key_values_0"));
    EXPECT_TRUE(hasName(names, "present_key_values_3"));

    EXPECT_EQ(names.size(), 26u);
}

TEST(RegistryBuilderTest, DraftEngineSpecShapesAreCorrect)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.isSpecDecodeBase = true;

    populateHybridFieldsFromScalars(cfg);
    DeploymentConfig bundle;
    bundle.draft = cfg;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 12288;
    specConfig.draftHiddenSize = 2048;
    bundle.specConfig = specConfig;
    auto reg = buildRegistryForSpecDecodeDraft(bundle);
    auto specs = reg.allExpandedSpecs();

    // inputs_embeds should use draftHiddenSize
    auto ieIt = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "inputs_embeds"; });
    ASSERT_NE(ieIt, specs.end());
    ASSERT_EQ(ieIt->shape.size(), 2u);
    EXPECT_EQ(ieIt->shape[1].value, 2048);

    // hidden_states_input should use baseOutputHiddenDim
    auto hsIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "hidden_states_input"; });
    ASSERT_NE(hsIt, specs.end());
    ASSERT_EQ(hsIt->shape.size(), 2u);
    EXPECT_EQ(hsIt->shape[1].value, 12288);

    // hidden_states_from_draft should use draftHiddenSize
    auto dsIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "hidden_states_from_draft"; });
    ASSERT_NE(dsIt, specs.end());
    ASSERT_EQ(dsIt->shape.size(), 2u);
    EXPECT_EQ(dsIt->shape[1].value, 2048);
}

TEST(RegistryBuilderTest, DraftEngineKVCacheUsesPluginPath)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.isSpecDecodeBase = true;

    populateHybridFieldsFromScalars(cfg);
    DeploymentConfig bundle;
    bundle.draft = cfg;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 12288;
    specConfig.draftHiddenSize = 2048;
    bundle.specConfig = specConfig;
    auto reg = buildRegistryForSpecDecodeDraft(bundle);
    auto specs = reg.allExpandedSpecs();

    // KV cache should be 5D paged-pool shape [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]
    auto kvIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "past_key_values_0"; });
    ASSERT_NE(kvIt, specs.end());
    EXPECT_EQ(kvIt->shape.size(), 5u);
    EXPECT_EQ(kvIt->shape[0].value, 2); // combined K+V dimension (leading, pool contract)
}

TEST(RegistryBuilderTest, SpecDraftRegistriesKVPageTableRowsTrackActiveBatch)
{
    LLMEngineConfig draft = makeBasicLLMConfig();
    draft.numAttentionLayers = 2;
    draft.numDecoderLayers = 2;
    draft.maxDraftTreeSize = 16;

    populateHybridFieldsFromScalars(draft);
    DeploymentConfig bundle;
    bundle.base = makeBasicLLMConfig();
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 4096;
    specConfig.draftHiddenSize = 4096;
    specConfig.dflashBlockSize = 8;
    bundle.specConfig = specConfig;

    auto checkPageTable = [&](TensorRegistry const& reg, char const* registryName) {
        SCOPED_TRACE(registryName);
        auto specs = reg.allExpandedSpecs();
        auto pageTableIt = std::find_if(
            specs.begin(), specs.end(), [](TensorSpec const& spec) { return spec.name == "kv_page_table"; });
        ASSERT_NE(pageTableIt, specs.end());
        EXPECT_EQ(pageTableIt->io, TensorIO::kInput);
        EXPECT_EQ(pageTableIt->dtype, nvinfer1::DataType::kINT32);
        ASSERT_EQ(pageTableIt->shape.size(), 3u);
        EXPECT_EQ(pageTableIt->shape[0].symbol, &InferenceDims::batch);
        EXPECT_EQ(pageTableIt->shape[1].value, 2);
        EXPECT_EQ(pageTableIt->shape[2].value, computeMaxPagesPerSeq(draft.maxKVCacheCapacity));

        for (int64_t const activeBatch : {1, 2})
        {
            InferenceDims const dims = draft.proposalDims(activeBatch, /*proposalSize=*/6, /*draftTopK=*/12);
            auto const resolved = reg.resolveShape(pageTableIt->shape, dims);
            ASSERT_EQ(resolved.nbDims, 3);
            EXPECT_EQ(resolved.d[0], activeBatch);
            EXPECT_NE(resolved.d[0], draft.maxSupportedBatchSize);
            EXPECT_EQ(resolved.d[1], 2);
            EXPECT_EQ(resolved.d[2], computeMaxPagesPerSeq(draft.maxKVCacheCapacity));
        }
    };

    draft.specDecodeType = SpecDecodeMode::kEAGLE;
    bundle.base.specDecodeType = draft.specDecodeType;
    bundle.draft = draft;
    checkPageTable(buildRegistryForSpecDecodeDraft(bundle), "EAGLE/MTP");

    draft.specDecodeType = SpecDecodeMode::kDFlash;
    draft.dflashVersion = DFlashVersion::kV2;
    draft.specDraftBlockSize = 8;
    draft.specSelectorTopK = 16;
    draft.specSelectorRank = 256;
    bundle.base.specDecodeType = draft.specDecodeType;
    bundle.draft = draft;
    checkPageTable(buildRegistryForDFlashDraft(bundle), "DFlash/JetSpec");

    draft.specDecodeType = SpecDecodeMode::kGemma4MTP;
    bundle.base.specDecodeType = draft.specDecodeType;
    bundle.draft = draft;
    checkPageTable(buildRegistryForGemma4MTPDraft(bundle), "Gemma4 MTP");

    draft.specDecodeType = SpecDecodeMode::kDSpark;
    bundle.base.specDecodeType = draft.specDecodeType;
    bundle.draft = draft;
    checkPageTable(buildRegistryForDSparkDraft(bundle), "DSpark");
}

TEST(RegistryBuilderTest, DFlash2RegistryExposesRuntimeSelectorIntermediates)
{
    LLMEngineConfig draft = makeBasicLLMConfig();
    draft.specDecodeType = SpecDecodeMode::kDFlash;
    draft.dflashVersion = DFlashVersion::kV2;
    draft.specDraftBlockSize = 8;
    draft.specSelectorTopK = 16;
    draft.specSelectorRank = 256;

    DeploymentConfig bundle;
    bundle.base = makeBasicLLMConfig();
    bundle.base.specDecodeType = SpecDecodeMode::kDFlash;
    bundle.base.dflashVersion = DFlashVersion::kV2;
    bundle.draft = draft;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = draft.hiddenSize;
    specConfig.draftHiddenSize = draft.hiddenSize;
    specConfig.dflashBlockSize = 16;
    bundle.specConfig = specConfig;

    auto const specs = buildRegistryForDFlashDraft(bundle).allExpandedSpecs();
    auto findSpec = [&](char const* name) {
        return std::find_if(specs.begin(), specs.end(), [name](TensorSpec const& spec) { return spec.name == name; });
    };
    EXPECT_EQ(findSpec("spec_proposal_uniforms"), specs.end());
    EXPECT_EQ(findSpec("spec_proposal_token_ids"), specs.end());

    auto const supportIds = findSpec(trt_edgellm::binding_names::kSpecProposalSupportIds);
    ASSERT_NE(supportIds, specs.end());
    ASSERT_EQ(supportIds->shape.size(), 3U);
    EXPECT_EQ(supportIds->shape[1].value, 15);
    EXPECT_EQ(supportIds->shape[2].value, 16);

    auto const unaryValues = findSpec(trt_edgellm::binding_names::kSpecProposalUnaryValues);
    ASSERT_NE(unaryValues, specs.end());
    ASSERT_EQ(unaryValues->shape.size(), supportIds->shape.size());
    for (size_t i = 0; i < supportIds->shape.size(); ++i)
    {
        EXPECT_EQ(unaryValues->shape[i].symbol, supportIds->shape[i].symbol);
        EXPECT_EQ(unaryValues->shape[i].value, supportIds->shape[i].value);
    }

    auto const projectedHidden = findSpec(trt_edgellm::binding_names::kSpecProposalProjectedHidden);
    ASSERT_NE(projectedHidden, specs.end());
    ASSERT_EQ(projectedHidden->shape.size(), 3U);
    EXPECT_EQ(projectedHidden->shape[1].value, 15);
    EXPECT_EQ(projectedHidden->shape[2].value, 256);
}

TEST(RegistryBuilderTest, BlockDraftRegistriesDoNotRequireLogitsIndices)
{
    namespace bn = trt_edgellm::binding_names;
    LLMEngineConfig draft = makeBasicLLMConfig();
    draft.numAttentionLayers = 2;
    draft.numDecoderLayers = 2;
    populateHybridFieldsFromScalars(draft);

    DeploymentConfig bundle;
    bundle.base = makeBasicLLMConfig();
    bundle.draft = draft;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 4096;
    specConfig.draftHiddenSize = 4096;
    bundle.specConfig = specConfig;

    for (auto const& reg : {buildRegistryForDFlashDraft(bundle), buildRegistryForDSparkDraft(bundle)})
    {
        EXPECT_FALSE(hasName(reg.allTensorNames(), bn::kLogitsIndices));
    }
}

TEST(RegistryBuilderTest, Gemma4MTPDraftDoesNotRequireLogitsIndices)
{
    namespace bn = trt_edgellm::binding_names;
    LLMEngineConfig draft = makeBasicLLMConfig();
    draft.specDecodeType = SpecDecodeMode::kGemma4MTP;
    draft.sharesTargetKV = true;
    draft.hasOwnKVCache = false;

    DeploymentConfig bundle;
    bundle.base = makeBasicLLMConfig();
    bundle.base.specDecodeType = SpecDecodeMode::kGemma4MTP;
    bundle.draft = draft;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 4096;
    bundle.specConfig = specConfig;

    EXPECT_FALSE(hasName(buildRegistryForGemma4MTPDraft(bundle).allTensorNames(), bn::kLogitsIndices));
}

// =====================================================================
// Hybrid model — numAttentionLayers < numDecoderLayers
// =====================================================================

TEST(RegistryBuilderTest, HybridModelKVCacheCountMatchesAttentionLayers)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 10;
    cfg.numDecoderLayers = 20;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto names = reg.allTensorNames();

    // Should have 10 past + 10 present KV cache entries, not 20
    EXPECT_TRUE(hasName(names, "past_key_values_9"));
    EXPECT_FALSE(hasName(names, "past_key_values_10"));

    EXPECT_EQ(names.size(), 33u);
}

// Heterogeneous-KV models (Gemma-4, Qwen3-Next, etc.) give each attention
// layer its own (numKVHeads, headDim). The registry must emit per-layer
// specs whose fixed dims come from `cfg.kvLayerConfigs[i]` rather than the
// scalar `numKVHeads` / `headDim` fallback. This guards against accidental
// regression to the uniform-broadcast path.
TEST(RegistryBuilderTest, HeterogeneousKVLayerEmitsPerLayerSpecs)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    // Two attention-only layers with different KV shapes.
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.numLinearAttnLayers = 0;

    // Set layerTypes and kvLayerConfigs explicitly — do NOT call
    // populateHybridFieldsFromScalars: that would broadcast uniform KV config
    // from the scalar fields and defeat the point of this test.
    cfg.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kAttention};
    cfg.kvLayerConfigs
        = {KVLayerConfig{/*numKVHeads=*/8, /*headDim=*/64}, KVLayerConfig{/*numKVHeads=*/4, /*headDim=*/128}};

    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    // past_key_values_0: plugin paged-pool shape [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]
    auto layer0
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "past_key_values_0"; });
    ASSERT_NE(layer0, specs.end());
    ASSERT_EQ(layer0->shape.size(), 5u);
    EXPECT_EQ(layer0->shape[3].value, 8);  // numKVHeads for layer 0
    EXPECT_EQ(layer0->shape[4].value, 64); // headDim for layer 0

    // past_key_values_1: different KV config
    auto layer1
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "past_key_values_1"; });
    ASSERT_NE(layer1, specs.end());
    ASSERT_EQ(layer1->shape.size(), 5u);
    EXPECT_EQ(layer1->shape[3].value, 4);   // numKVHeads for layer 1
    EXPECT_EQ(layer1->shape[4].value, 128); // headDim for layer 1

    // Sanity: the two specs must differ on the fixed dims.
    EXPECT_NE(layer0->shape[3].value, layer1->shape[3].value);
    EXPECT_NE(layer0->shape[4].value, layer1->shape[4].value);
}

TEST(RegistryBuilderTest, SwaCapableRegistryUsesActivePageCountAndAlwaysCarriesModeInputs)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.numSwaPages = 64;
    cfg.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kAttention};
    cfg.kvLayerConfigs = {
        KVLayerConfig{/*numKVHeads=*/8, /*headDim=*/128},
        KVLayerConfig{/*numKVHeads=*/8, /*headDim=*/128, /*kvCacheCapacity=*/129},
    };

    auto const boundedSpecs = buildRegistryForLLM(cfg).allExpandedSpecs();
    auto const full = std::find_if(boundedSpecs.begin(), boundedSpecs.end(),
        [](TensorSpec const& spec) { return spec.name == "past_key_values_0"; });
    auto const swa = std::find_if(boundedSpecs.begin(), boundedSpecs.end(),
        [](TensorSpec const& spec) { return spec.name == "past_key_values_1"; });
    auto const mode = std::find_if(boundedSpecs.begin(), boundedSpecs.end(),
        [](TensorSpec const& spec) { return spec.name == trt_edgellm::binding_names::kSwaKVCacheMode; });
    ASSERT_NE(full, boundedSpecs.end());
    ASSERT_NE(swa, boundedSpecs.end());
    ASSERT_NE(mode, boundedSpecs.end());

    EXPECT_EQ(full->shape[1].value, computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity));
    EXPECT_EQ(swa->shape[1].value, cfg.numSwaPages);
    EXPECT_NE(full->shape[1].value, swa->shape[1].value);
    ASSERT_EQ(mode->shape.size(), 1U);
    EXPECT_EQ(mode->shape[0].symbol, &InferenceDims::swaKVCacheModeLen);
    EXPECT_TRUE(hasName(buildRegistryForLLM(cfg).allTensorNames(), trt_edgellm::binding_names::kSwaKVPageTable));

    int32_t const capabilityMarker = cfg.kvLayerConfigs[1].kvCacheCapacity;
    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    auto const fullModeSpecs = buildRegistryForLLM(cfg).allExpandedSpecs();
    auto const fullModeSwa = std::find_if(fullModeSpecs.begin(), fullModeSpecs.end(),
        [](TensorSpec const& spec) { return spec.name == "past_key_values_1"; });
    ASSERT_NE(fullModeSwa, fullModeSpecs.end());
    EXPECT_EQ(fullModeSwa->shape[1].value, cfg.kvPoolPages);
    EXPECT_EQ(cfg.kvLayerConfigs[1].kvCacheCapacity, capabilityMarker);
    EXPECT_TRUE(hasName(buildRegistryForLLM(cfg).allTensorNames(), trt_edgellm::binding_names::kSwaKVPageTable));
    EXPECT_TRUE(hasName(buildRegistryForLLM(cfg).allTensorNames(), trt_edgellm::binding_names::kSwaKVCacheMode));
}

// =====================================================================
// Symbolic dimension resolution integration
// =====================================================================

TEST(RegistryBuilderTest, SymbolicDimsCanBeResolved)
{
    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 1;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    // Find inputs_embeds and resolve its shape
    auto it = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "inputs_embeds"; });
    ASSERT_NE(it, specs.end());

    // Route through the production recipe path rather than a raw aggregate init.
    // prefillDims populates all InferenceDims fields; a raw positional init here
    // would need re-labelling every time InferenceDims grows a new member.
    InferenceDims const dims = cfg.prefillDims(/*batch=*/4, /*seqLen=*/128, ExecutionPhase::kContextPrefill);
    auto resolved = reg.resolveShape(it->shape, dims);
    EXPECT_EQ(resolved.nbDims, 2);
    EXPECT_EQ(resolved.d[0], 512);
    EXPECT_EQ(resolved.d[1], 4096);
}

class RegistryBuilderKVDtypeTest : public ::testing::TestWithParam<nvinfer1::DataType>
{
};

TEST_P(RegistryBuilderKVDtypeTest, KVCacheBindingDtypeMatchesConfigPluginPath)
{
    nvinfer1::DataType const kvDtype = GetParam();

    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.kvCacheDtype = kvDtype;
    cfg.numAttentionLayers = 4;
    cfg.numDecoderLayers = 4;

    populateHybridFieldsFromScalars(cfg);
    auto reg = buildRegistryForLLM(cfg);
    auto specs = reg.allExpandedSpecs();

    int pastCount = 0;
    int presentCount = 0;
    for (auto const& spec : specs)
    {
        if (spec.name.rfind("past_key_values_", 0) == 0)
        {
            EXPECT_EQ(spec.dtype, kvDtype) << "past binding " << spec.name << " has wrong dtype";
            ++pastCount;
        }
        else if (spec.name.rfind("present_key_values_", 0) == 0)
        {
            EXPECT_EQ(spec.dtype, kvDtype) << "present binding " << spec.name << " has wrong dtype";
            ++presentCount;
        }
    }
    EXPECT_EQ(pastCount, 4);
    EXPECT_EQ(presentCount, 4);
}

TEST_P(RegistryBuilderKVDtypeTest, DraftEngineKVCacheBindingDtypeMatchesConfig)
{
    nvinfer1::DataType const kvDtype = GetParam();

    LLMEngineConfig cfg = makeBasicLLMConfig();
    cfg.kvCacheDtype = kvDtype;
    cfg.numAttentionLayers = 3;
    cfg.numDecoderLayers = 3;
    cfg.isSpecDecodeBase = true;

    populateHybridFieldsFromScalars(cfg);
    DeploymentConfig bundle;
    bundle.draft = cfg;
    SpecDecodeConfig specConfig{};
    specConfig.baseOutputHiddenDim = 12288;
    specConfig.draftHiddenSize = 2048;
    bundle.specConfig = specConfig;
    auto reg = buildRegistryForSpecDecodeDraft(bundle);
    auto specs = reg.allExpandedSpecs();

    int kvBindingCount = 0;
    for (auto const& spec : specs)
    {
        if (spec.name.rfind("past_key_values_", 0) == 0 || spec.name.rfind("present_key_values_", 0) == 0)
        {
            EXPECT_EQ(spec.dtype, kvDtype) << "draft KV binding " << spec.name << " has wrong dtype";
            ++kvBindingCount;
        }
    }
    // 3 layers * 2 (past + present) = 6
    EXPECT_EQ(kvBindingCount, 6);
}

INSTANTIATE_TEST_SUITE_P(AllKVDtypes, RegistryBuilderKVDtypeTest,
    ::testing::Values(nvinfer1::DataType::kHALF, nvinfer1::DataType::kFP8, nvinfer1::DataType::kBF16));
