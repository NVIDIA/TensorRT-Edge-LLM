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
    populateHybridFieldsFromScalars(cfg);
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
    EXPECT_TRUE(hasName(names, "context_lengths"));
    EXPECT_TRUE(hasName(names, "last_token_ids"));
    // kvcache_start_index is registered with a symbolic `start_index_len` dim
    // (0 for initial-prefill sentinel, batch otherwise). See registryBuilder.
    EXPECT_TRUE(hasName(names, "kvcache_start_index"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kKVPageTable));
    EXPECT_TRUE(hasName(names, "rope_rotary_cos_sin"));

    // KV cache: 32 layers x 2 (past + present) = 64 entries
    EXPECT_TRUE(hasName(names, "past_key_values_0"));
    EXPECT_TRUE(hasName(names, "past_key_values_31"));
    EXPECT_TRUE(hasName(names, "present_key_values_0"));
    EXPECT_TRUE(hasName(names, "present_key_values_31"));

    // Total: 7 core (incl. kvcache_start_index and kv_page_table) + 64 KV = 71
    EXPECT_EQ(names.size(), 71u);
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
    EXPECT_TRUE(hasName(names, bn::kContextMaskSelector));
    EXPECT_TRUE(hasName(names, bn::kKVPageTable));
    EXPECT_FALSE(hasName(names, bn::kLastTokenIds));

    auto specs = reg.allExpandedSpecs();
    auto const logits
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kLogits; });
    ASSERT_NE(logits, specs.end());
    EXPECT_EQ(logits->io, TensorIO::kOutput);
    EXPECT_EQ(logits->dtype, nvinfer1::DataType::kFLOAT);
    ASSERT_EQ(logits->shape.size(), 3u);
    EXPECT_TRUE(logits->shape[1].isSymbolic());
    EXPECT_EQ(logits->shape[1].symbol, &InferenceDims::selectLen);

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
    ASSERT_EQ(prevFeedback->shape.size(), 3u);
    EXPECT_EQ(prevFeedback->shape[0].symbol, &InferenceDims::batch);
    EXPECT_EQ(prevFeedback->shape[1].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(prevFeedback->shape[2].value, cfg.hiddenSize);

    auto const nextFeedback = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == bn::kNextSelfConditioningEmbeds; });
    ASSERT_NE(nextFeedback, specs.end());
    EXPECT_EQ(nextFeedback->io, TensorIO::kOutput);
    EXPECT_EQ(nextFeedback->dtype, nvinfer1::DataType::kHALF);
    ASSERT_EQ(nextFeedback->shape.size(), 3u);
    EXPECT_EQ(nextFeedback->shape[0].symbol, &InferenceDims::batch);
    EXPECT_EQ(nextFeedback->shape[1].symbol, &InferenceDims::selectLen);
    EXPECT_EQ(nextFeedback->shape[2].value, cfg.hiddenSize);
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
    EXPECT_EQ(it->shape.size(), 3u);
    EXPECT_TRUE(it->shape[0].isSymbolic()); // batch
    EXPECT_TRUE(it->shape[1].isSymbolic()); // seq_len
    EXPECT_EQ(it->shape[2].value, 4096);    // hiddenSize

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

    // 7 core (incl. kvcache_start_index and kv_page_table) + 4 KV (2 layers * 2) + 3 deepstack = 14
    EXPECT_EQ(names.size(), 14u);
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
    EXPECT_EQ(it->shape.size(), 3u);
    EXPECT_TRUE(it->shape[0].isSymbolic()); // batch
    EXPECT_TRUE(it->shape[1].isSymbolic()); // seq_len
    EXPECT_EQ(it->shape[2].value, 4096);    // hiddenSize
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
    EXPECT_TRUE(hasName(names, "attention_mask"));
    EXPECT_TRUE(hasName(names, "attention_pos_id"));

    // 7 core (incl. kvcache_start_index and kv_page_table) + 4 KV + 3 SpecDecode = 14
    EXPECT_EQ(names.size(), 14u);
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
    EXPECT_FALSE(hasName(names, "attention_mask"));
    EXPECT_FALSE(hasName(names, "attention_pos_id"));
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

    // 7 core (incl. kvcache_start_index and kv_page_table) + 4 KV (2 attn layers) + 4 recurrent + 4 conv = 19
    EXPECT_EQ(names.size(), 19u);
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
    EXPECT_TRUE(recIt->shape[0].isSymbolic()); // batch
    EXPECT_EQ(recIt->shape[1].value, 16);      // numHeads
    EXPECT_EQ(recIt->shape[2].value, 64);      // headDim
    EXPECT_EQ(recIt->shape[3].value, 128);     // stateSize

    auto convIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "conv_state_0"; });
    ASSERT_NE(convIt, specs.end());
    EXPECT_EQ(convIt->shape.size(), 3u);
    EXPECT_TRUE(convIt->shape[0].isSymbolic()); // batch
    EXPECT_EQ(convIt->shape[1].value, 256);     // convDim
    EXPECT_EQ(convIt->shape[2].value, 4);       // convKernel
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

    // 7 core (incl. kvcache_start_index and kv_page_table) + 4 KV (2 layers) + 2 deepstack + 4 SpecDecode
    //   + 4 recurrent + 4 conv + 4 intermediate (2 layers × {recurrent, conv})
    // = 29. The extra 4 intermediate-state outputs and phase marker come from the MTP-base path.
    EXPECT_EQ(names.size(), 29u);
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kSpecVerifyPhaseMarker));
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
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kSpecVerifyPhaseMarker));

    auto markerIt = std::find_if(specs.begin(), specs.end(),
        [](TensorSpec const& s) { return s.name == trt_edgellm::binding_names::kSpecVerifyPhaseMarker; });
    ASSERT_NE(markerIt, specs.end());
    EXPECT_EQ(markerIt->io, TensorIO::kInput);
    ASSERT_EQ(markerIt->shape.size(), 1u);
    EXPECT_TRUE(markerIt->shape[0].isSymbolic());
    EXPECT_EQ(markerIt->shape[0].symbol, &InferenceDims::specVerifyPhaseLen);

    // Shape: [batch, seqLen, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
    auto irecIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "intermediate_recurrent_state_0"; });
    ASSERT_NE(irecIt, specs.end());
    EXPECT_EQ(irecIt->io, TensorIO::kOutput);
    ASSERT_EQ(irecIt->shape.size(), 5u);
    EXPECT_TRUE(irecIt->shape[0].isSymbolic()); // batch
    EXPECT_TRUE(irecIt->shape[1].isSymbolic()); // seqLen
    EXPECT_EQ(irecIt->shape[2].value, 16);      // numHeads
    EXPECT_EQ(irecIt->shape[3].value, 64);      // headDim
    EXPECT_EQ(irecIt->shape[4].value, 128);     // stateSize

    // Shape: [batch, seqLen, convDim, convKernel]
    auto iconvIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "intermediate_conv_state_0"; });
    ASSERT_NE(iconvIt, specs.end());
    EXPECT_EQ(iconvIt->io, TensorIO::kOutput);
    ASSERT_EQ(iconvIt->shape.size(), 4u);
    EXPECT_TRUE(iconvIt->shape[0].isSymbolic()); // batch
    EXPECT_TRUE(iconvIt->shape[1].isSymbolic()); // seqLen
    EXPECT_EQ(iconvIt->shape[2].value, 256);     // convDim
    EXPECT_EQ(iconvIt->shape[3].value, 4);       // convKernel
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
        [](TensorSpec const& spec) { return spec.name == trt_edgellm::binding_names::kSpecVerifyPhaseMarker; });

    ASSERT_NE(markerIt, specs.end());
    EXPECT_EQ(markerIt->io, TensorIO::kInput);
    ASSERT_EQ(markerIt->shape.size(), 1U);
    EXPECT_TRUE(markerIt->shape[0].isSymbolic());
    EXPECT_EQ(markerIt->shape[0].symbol, &InferenceDims::specVerifyPhaseLen);
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
    EXPECT_TRUE(hasName(names, "last_token_ids"));
    EXPECT_TRUE(hasName(names, "context_lengths"));
    // kvcache_start_index is registered with a symbolic start_index_len dim
    // (0 for initial-prefill sentinel, batch otherwise). Draft engine always
    // uses plugin attention so this applies unconditionally.
    EXPECT_TRUE(hasName(names, "kvcache_start_index"));
    EXPECT_TRUE(hasName(names, trt_edgellm::binding_names::kKVPageTable));
    EXPECT_TRUE(hasName(names, "rope_rotary_cos_sin"));
    EXPECT_TRUE(hasName(names, "attention_mask"));
    EXPECT_TRUE(hasName(names, "attention_pos_id"));

    // Outputs
    EXPECT_TRUE(hasName(names, "logits"));
    EXPECT_TRUE(hasName(names, "hidden_states"));

    // KV cache (4 layers, plugin mode)
    EXPECT_TRUE(hasName(names, "past_key_values_0"));
    EXPECT_TRUE(hasName(names, "past_key_values_3"));
    EXPECT_TRUE(hasName(names, "present_key_values_0"));
    EXPECT_TRUE(hasName(names, "present_key_values_3"));

    // 12 core/output (incl. kvcache_start_index and kv_page_table) + 8 KV = 20
    EXPECT_EQ(names.size(), 20u);
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
    EXPECT_EQ(ieIt->shape[2].value, 2048);

    // hidden_states_input should use baseOutputHiddenDim
    auto hsIt
        = std::find_if(specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "hidden_states_input"; });
    ASSERT_NE(hsIt, specs.end());
    EXPECT_EQ(hsIt->shape[2].value, 12288);

    // hidden_states_from_draft should use draftHiddenSize
    auto dsIt = std::find_if(
        specs.begin(), specs.end(), [](TensorSpec const& s) { return s.name == "hidden_states_from_draft"; });
    ASSERT_NE(dsIt, specs.end());
    EXPECT_EQ(dsIt->shape[2].value, 2048);
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

    // 7 core (incl. kvcache_start_index and kv_page_table) + 20 KV (10 layers * 2) = 27
    EXPECT_EQ(names.size(), 27u);
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
    InferenceDims const dims = cfg.prefillDims(/*batch=*/4, /*seqLen=*/128, /*kvCacheAllEmpty=*/true);
    auto resolved = reg.resolveShape(it->shape, dims);
    EXPECT_EQ(resolved.nbDims, 3);
    EXPECT_EQ(resolved.d[0], 4);
    EXPECT_EQ(resolved.d[1], 128);
    EXPECT_EQ(resolved.d[2], 4096);
}

// =====================================================================
// Tier-1 #9: FP8 KV cache regression test — cycle {kHALF, kFP8, kBF16}
// and assert the registry emits past_key_values_* bindings whose dtype
// matches `cfg.kvCacheDtype`.
// =====================================================================

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
