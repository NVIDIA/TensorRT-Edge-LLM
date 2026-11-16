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

#include "nativeCudaTestUtils.h"
#include "safety_inference/kernels/safetyAttentionModules.h"
#include "safety_inference/runtime/safetyErrorState.h"
#include "safety_inference/trt_runner/safetyGraphInitialization.h"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace
{
using namespace trt_edgellm::safety;
using namespace trt_edgellm::safety::native_test;
using namespace nvinfer1;
using namespace nvinfer2::safe;

class Recorder : public ISafeRecorder
{
public:
    int32_t getNbErrors() const noexcept override
    {
        return mErrorCount;
    }
    ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mLastCode : ErrorCode::kUNSPECIFIED_ERROR;
    }
    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
    {
        return validIndex(errorIdx) ? mLastError.data() : "";
    }
    bool hasOverflowed() const noexcept override
    {
        return false;
    }
    void clear() noexcept override
    {
        mErrorCount = 0;
        mLastCode = ErrorCode::kSUCCESS;
        mLastError.fill('\0');
    }
    bool reportError(ErrorCode code, ErrorDesc description) noexcept override
    {
        mLastCode = code;
        if (description == nullptr)
        {
            mLastError[0] = '\0';
        }
        else
        {
            std::strncpy(mLastError.data(), description, mLastError.size() - 1U);
            mLastError.back() = '\0';
        }
        ++mErrorCount;
        return true;
    }
    RefCount incRefCount() noexcept override
    {
        return ++mReferences;
    }
    RefCount decRefCount() noexcept override
    {
        return --mReferences;
    }

private:
    bool validIndex(int32_t errorIdx) const noexcept
    {
        return errorIdx >= 0 && errorIdx < mErrorCount.load();
    }

    ErrorCode mLastCode{ErrorCode::kSUCCESS};
    std::array<char, ISafeRecorder::kMAX_SAFE_DESC_LENGTH + 1U> mLastError{};
    std::atomic<int32_t> mErrorCount{0};
    std::atomic<int32_t> mReferences{0};
};
Recorder sRecorder;
Recorder sOtherRecorder;

class Registry : public ISafePluginRegistry
{
public:
    std::vector<IPluginCreatorInterface*> creators;
    std::vector<ISafeRecorder*> registrationRecorders;
    int32_t registrationCalls{};
    bool failEnumeration{};
    int32_t failRegistration{-1};

    ErrorCode registerCreator(
        IPluginCreatorInterface& creator, char const* pluginNamespace, ISafeRecorder& recorder) noexcept override
    {
        if (registrationCalls++ == failRegistration)
        {
            return ErrorCode::kUNSPECIFIED_ERROR;
        }
        auto* safeCreator = static_cast<ISafePluginCreatorV3One*>(&creator);
        EXPECT_STREQ(pluginNamespace, safeCreator->getPluginNamespace());
        creators.push_back(&creator);
        registrationRecorders.push_back(&recorder);
        return ErrorCode::kSUCCESS;
    }
    ErrorCode getAllCreators(IPluginCreatorInterface* const*& result, int32_t& count) const noexcept override
    {
        result = creators.data();
        count = creators.size();
        return failEnumeration ? ErrorCode::kUNSPECIFIED_ERROR : ErrorCode::kSUCCESS;
    }
    ErrorCode getCreator(IPluginCreatorInterface*&, char const*, char const*, char const*) noexcept override
    {
        ADD_FAILURE() << "Registration must not probe absence through a failing DLI lookup";
        return ErrorCode::kINVALID_CONFIG;
    }
    ErrorCode setSafeRecorder(ISafeRecorder&) noexcept override
    {
        return ErrorCode::kSUCCESS;
    }
    ErrorCode getSafeRecorder(ISafeRecorder*& recorder) const noexcept override
    {
        recorder = &sRecorder;
        return ErrorCode::kSUCCESS;
    }
    ErrorCode deregisterCreator(IPluginCreatorInterface const&) noexcept override
    {
        ADD_FAILURE() << "Process-lifetime creators must not be deregistered by graph teardown";
        return ErrorCode::kUNSPECIFIED_ERROR;
    }
};

struct Fields
{
    int32_t qHeads{2};
    int32_t kvHeads{1};
    int32_t headSize{64};
    int32_t disabled{0};
    int32_t noSlidingWindow{-1};
    float scale{0.125F};
    float scales[3]{1.F, 1.F, 1.F};
    float rmsNormEps{1e-6F};
    std::vector<PluginField> values;
    explicit Fields(AttentionRole role)
    {
        values = {{role == AttentionRole::kLLM ? "num_q_heads" : "num_heads", &qHeads, PluginFieldType::kINT32, 1},
            {"head_size", &headSize, PluginFieldType::kINT32, 1},
            {"attention_scale", &scale, PluginFieldType::kFLOAT32, 1},
            {"qkv_scales", scales, PluginFieldType::kFLOAT32, 3}};
        if (role == AttentionRole::kLLM)
        {
            values.push_back({"num_kv_heads", &kvHeads, PluginFieldType::kINT32, 1});
            // Full vanilla field set emitted by Standard AttentionPlugin::getFieldsToSerialize.
            for (auto const* name : {"enable_tree_attention", "enable_qk_norm", "qk_norm_post_rope", "enable_kv_shared",
                     "enable_contiguous_query_swa", "enable_attention_sink", "enable_fp8_kv_cache",
                     "enable_vision_block_attention", "enable_context_mask_selector", "supports_bounded_kv_cache"})
            {
                values.push_back({name, &disabled, PluginFieldType::kINT32, 1});
            }
            values.push_back({"sliding_window_size", &noSlidingWindow, PluginFieldType::kINT32, 1});
            values.push_back({"rms_norm_eps", &rmsNormEps, PluginFieldType::kFLOAT32, 1});
            values.push_back({"xqa_jit_kernels", nullptr, PluginFieldType::kCHAR, 0});
        }
    }
    PluginFieldCollection collection() const
    {
        return {static_cast<int32_t>(values.size()), values.data()};
    }
};

TensorDescriptor descriptor(std::initializer_list<int64_t> shape, DataType type)
{
    TensorDescriptor result{};
    result.dataType = type;
    result.shape.nbDims = shape.size();
    result.stride.nbDims = shape.size();
    std::copy(shape.begin(), shape.end(), result.shape.d);
    int64_t stride = 1;
    for (int32_t i = result.shape.nbDims - 1; i >= 0; --i)
    {
        result.stride.d[i] = stride;
        stride *= result.shape.d[i];
    }
    result.bytesPerComponent = type == DataType::kHALF ? 2 : 4;
    result.sizeInBytes = stride * result.bytesPerComponent;
    result.memPlacement = MemoryPlacement::kGPU;
    return result;
}

std::unique_ptr<IPluginV3> makePlugin(AttentionRole role, TensorRTPhase phase = TensorRTPhase::kRUNTIME,
    int32_t headSize = 64, int32_t qHeads = 2, int32_t kvHeads = 1)
{
    Fields fields(role);
    fields.headSize = headSize;
    fields.qHeads = qHeads;
    fields.kvHeads = kvHeads;
    auto collection = fields.collection();
    auto& creator = safetyAttentionCreator(role);
    creator.setSafeRecorder(sRecorder);
    return std::unique_ptr<IPluginV3>(creator.createPlugin("test", &collection, phase));
}

TEST(SafetyRegistry, RegistersBothCreatorsAndVerifiesRepeatedRegistration)
{
    // Given an empty registry, when registration is repeated, then both creators retain the process recorder once.
    // Mutation mapping: W2, W3, W4.
    Registry registry;
    auto& processRecorder = processRegistrationRecorder();
    processRecorder.clear();
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kSUCCESS);
    EXPECT_EQ(registry.registrationCalls, 2);
    ASSERT_EQ(registry.registrationRecorders.size(), 2U);
    EXPECT_EQ(registry.registrationRecorders[0], &processRecorder);
    EXPECT_EQ(registry.registrationRecorders[1], &processRecorder);
    EXPECT_EQ(safetyAttentionCreator(AttentionRole::kLLM).getSafeRecorder(), &processRecorder);
    EXPECT_EQ(safetyAttentionCreator(AttentionRole::kVISUAL).getSafeRecorder(), &processRecorder);
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kSUCCESS);
    EXPECT_EQ(registry.registrationCalls, 2);
}

TEST(SafetyRegistry, ReportsMalformedCreatorList)
{
    Registry registry;
    auto& processRecorder = processRegistrationRecorder();
    processRecorder.clear();
    registry.creators.resize(101);
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kINVALID_CONFIG);
    EXPECT_STREQ(processRecorder.getErrorDesc(0),
        "Invalid Safety plugin creator list: expected count 0..100 and non-null storage when nonempty");
    processRecorder.clear();
    registry.creators.resize(1);
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kINVALID_CONFIG);
    EXPECT_STREQ(processRecorder.getErrorDesc(0), "Null creator entry in Safety plugin registry");
    EXPECT_EQ(registry.registrationCalls, 0);
}

TEST(SafetyRegistry, PropagatesEnumerationAndRegistrationFailures)
{
    Registry registry;
    processRegistrationRecorder().clear();
    registry.failEnumeration = true;
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_EQ(registry.registrationCalls, 0);
    registry.failEnumeration = false;
    registry.failRegistration = 0;
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_EQ(registry.registrationCalls, 1);
}

TEST(SafetyFields, RejectsMalformedAndUnsupportedFeatures)
{
    for (auto role : {AttentionRole::kLLM, AttentionRole::kVISUAL})
    {
        Fields fields(role);
        AttentionParameters parameters{};
        auto collection = fields.collection();
        EXPECT_TRUE(parseAttentionFields(role, &collection, parameters));
        fields.scales[1] = 2.F;
        EXPECT_FALSE(parseAttentionFields(role, &collection, parameters));
        fields.scales[1] = 1.F;
        fields.headSize = 63;
        EXPECT_FALSE(parseAttentionFields(role, &collection, parameters));
        fields.headSize = 64;
        fields.values.push_back(fields.values.front());
        collection = fields.collection();
        EXPECT_FALSE(parseAttentionFields(role, &collection, parameters));
        fields.values.back() = {"unrecognized", &fields.disabled, PluginFieldType::kINT32, 1};
        EXPECT_FALSE(parseAttentionFields(role, &collection, parameters));
        EXPECT_FALSE(parseAttentionFields(role, nullptr, parameters));
    }
    Fields fields(AttentionRole::kLLM);
    fields.disabled = 1;
    auto collection = fields.collection();
    AttentionParameters parameters{};
    EXPECT_FALSE(parseAttentionFields(AttentionRole::kLLM, &collection, parameters));
}

TEST(SafetyFields, EveryAdvertisedDisabledIntegerFlagRejectsEnabledValues)
{
    auto const* advertised = safetyAttentionCreator(AttentionRole::kLLM).getFieldNames();
    for (auto const* name : kDISABLED_INTEGER_FIELDS)
    {
        SCOPED_TRACE(name);
        Fields fields(AttentionRole::kLLM);
        fields.values.resize(5);
        fields.values.push_back({name, &fields.disabled, PluginFieldType::kINT32, 1});
        auto collection = fields.collection();
        AttentionParameters parameters{};
        EXPECT_TRUE(parseAttentionFields(AttentionRole::kLLM, &collection, parameters));
        auto const* found = std::find_if(advertised->fields, advertised->fields + advertised->nbFields,
            [&](PluginField const& field) { return std::strcmp(field.name, name) == 0; });
        ASSERT_NE(found, advertised->fields + advertised->nbFields);
        EXPECT_EQ(found->type, PluginFieldType::kINT32);
        EXPECT_EQ(found->length, 1);
        for (int32_t enabled : {-1, 1})
        {
            fields.disabled = enabled;
            EXPECT_FALSE(parseAttentionFields(AttentionRole::kLLM, &collection, parameters));
        }
        fields.disabled = 0;
        fields.values.back().length = 2;
        EXPECT_FALSE(parseAttentionFields(AttentionRole::kLLM, &collection, parameters));
        fields.values.back().length = 1;
        fields.values.back().type = PluginFieldType::kFLOAT32;
        EXPECT_FALSE(parseAttentionFields(AttentionRole::kLLM, &collection, parameters));
    }
}

TEST(SafetyFields, CreatesFromStandardVanillaMetadataWithoutAdmittingJitKernels)
{
    for (auto phase : {TensorRTPhase::kBUILD, TensorRTPhase::kRUNTIME})
    {
        Fields fields(AttentionRole::kLLM);
        auto collection = fields.collection();
        auto& creator = safetyAttentionCreator(AttentionRole::kLLM);
        creator.setSafeRecorder(sRecorder);
        auto create = [&]() { return std::unique_ptr<IPluginV3>(creator.createPlugin("test", &collection, phase)); };
        ASSERT_NE(create(), nullptr);
        auto& blob = fields.values.back();
        char byte{};
        blob.data = &byte;
        EXPECT_NE(create(), nullptr);
        for (auto const* data : {static_cast<void const*>(nullptr), static_cast<void const*>(&byte)})
        {
            blob.data = data;
            for (int32_t length : {-1, 1})
            {
                blob.length = length;
                EXPECT_EQ(create(), nullptr);
            }
            blob.length = 0;
            blob.type = PluginFieldType::kINT32;
            EXPECT_EQ(create(), nullptr);
            blob.type = PluginFieldType::kCHAR;
        }
        fields.values.front().data = nullptr;
        EXPECT_EQ(create(), nullptr);
    }
    Fields visual(AttentionRole::kVISUAL);
    visual.values.push_back({"xqa_jit_kernels", nullptr, PluginFieldType::kCHAR, 0});
    auto collection = visual.collection();
    AttentionParameters parameters{};
    EXPECT_FALSE(parseAttentionFields(AttentionRole::kVISUAL, &collection, parameters));
}

TEST(SafetyDescriptors, MaximumHeadsDoNotConsumeTheTokenAxisLimit)
{
    AttentionParameters parameters{AttentionRole::kLLM, kMAX_HEADS, kMAX_HEADS, 128, 0.125F};
    int32_t const packedWidth = (parameters.qHeads + 2 * parameters.kvHeads) * parameters.headSize;
    TensorDescriptor inputs[]{descriptor({1, 1, packedWidth}, DataType::kHALF),
        descriptor({1, 2, parameters.kvHeads, 129, parameters.headSize}, DataType::kHALF),
        descriptor({1}, DataType::kINT32), descriptor({1, kMAX_TOKENS, 128}, DataType::kFLOAT),
        descriptor({0}, DataType::kINT32)};
    TensorDescriptor outputs[]{descriptor({1, 1, parameters.qHeads, parameters.headSize}, DataType::kHALF), inputs[1]};
    for (auto validate : {validateAttentionDescriptors, validateAttentionBuildDescriptors})
    {
        AttentionLaunch launch{};
        ASSERT_TRUE(validate(parameters, inputs, 5, outputs, 2, launch));
        EXPECT_EQ(launch.tokens, 1);
        EXPECT_EQ(launch.headSize, 128);
        inputs[0] = descriptor({1, kMAX_TOKENS, packedWidth}, DataType::kHALF);
        outputs[0] = descriptor({1, kMAX_TOKENS, parameters.qHeads, parameters.headSize}, DataType::kHALF);
        EXPECT_TRUE(validate(parameters, inputs, 5, outputs, 2, launch));
        inputs[0] = descriptor({1, kMAX_TOKENS + 1, packedWidth}, DataType::kHALF);
        outputs[0] = descriptor({1, kMAX_TOKENS + 1, parameters.qHeads, parameters.headSize}, DataType::kHALF);
        EXPECT_FALSE(validate(parameters, inputs, 5, outputs, 2, launch));
        inputs[0] = descriptor({1, 1, packedWidth}, DataType::kHALF);
        outputs[0] = descriptor({1, 1, parameters.qHeads, parameters.headSize}, DataType::kHALF);
        inputs[3] = descriptor({1, kMAX_TOKENS + 1, 128}, DataType::kFLOAT);
        EXPECT_FALSE(validate(parameters, inputs, 5, outputs, 2, launch));
        inputs[3] = descriptor({1, kMAX_TOKENS, 128}, DataType::kFLOAT);
        inputs[0] = descriptor({1, std::numeric_limits<int32_t>::max(), packedWidth}, DataType::kHALF);
        EXPECT_FALSE(validate(parameters, inputs, 5, outputs, 2, launch));
        inputs[0] = descriptor({1, 1, packedWidth}, DataType::kHALF);
    }
}

TEST(SafetyDescriptors, VisualTokenAndSegmentLimitsAreIndependentOfMetadataExtent)
{
    AttentionParameters parameters{AttentionRole::kVISUAL, kMAX_HEADS, kMAX_HEADS, 128, 0.125F};
    auto visual = [&](int32_t tokens) { return descriptor({tokens, kMAX_HEADS, 128}, DataType::kHALF); };
    TensorDescriptor inputs[]{visual(kMAX_TOKENS), visual(kMAX_TOKENS), visual(kMAX_TOKENS),
        descriptor({kMAX_TOKENS + 1}, DataType::kINT32), descriptor({1}, DataType::kINT32)};
    TensorDescriptor output = inputs[0];
    for (auto validate : {validateAttentionDescriptors, validateAttentionBuildDescriptors})
    {
        AttentionLaunch launch{};
        EXPECT_TRUE(validate(parameters, inputs, 5, &output, 1, launch));
        EXPECT_EQ(launch.sequences, kMAX_TOKENS);
        inputs[3] = descriptor({kMAX_TOKENS + 2}, DataType::kINT32);
        EXPECT_FALSE(validate(parameters, inputs, 5, &output, 1, launch));
        inputs[3] = descriptor({kMAX_TOKENS + 1}, DataType::kINT32);
        for (int32_t index = 0; index < 3; ++index)
        {
            inputs[index] = visual(kMAX_TOKENS + 1);
        }
        output = inputs[0];
        EXPECT_FALSE(validate(parameters, inputs, 5, &output, 1, launch));
        for (int32_t index = 0; index < 3; ++index)
        {
            inputs[index] = visual(kMAX_TOKENS);
        }
        output = inputs[0];
    }
}

TEST(SafetyPlugin, IdentitySerializationCloneAndLifecycle)
{
    for (auto role : {AttentionRole::kLLM, AttentionRole::kVISUAL})
    {
        auto plugin = makePlugin(role);
        ASSERT_NE(plugin, nullptr);
        auto* core = static_cast<IPluginV3OneSafeCore*>(plugin->getCapabilityInterface(PluginCapabilityType::kCORE));
        auto* runtime
            = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        ASSERT_NE(runtime, nullptr);
        EXPECT_STREQ(core->getPluginVersion(), "1");
        EXPECT_STREQ(core->getPluginNamespace(), role == AttentionRole::kLLM ? "trt" : "trt_edgellm");
        EXPECT_STREQ(core->getPluginName(), role == AttentionRole::kLLM ? "AttentionPlugin" : "ViTAttentionPlugin");
        EXPECT_EQ(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD), nullptr);
        AttentionParameters parameters{};
        EXPECT_TRUE(parseAttentionFields(role, runtime->getFieldsToSerialize(), parameters));
        std::unique_ptr<IPluginV3> copy(runtime->clone());
        ASSERT_NE(copy, nullptr);
        plugin.reset();
        auto* copyRuntime
            = static_cast<IPluginV3OneSafeRuntime*>(copy->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        EXPECT_TRUE(parseAttentionFields(role, copyRuntime->getFieldsToSerialize(), parameters));
        EXPECT_NE(copyRuntime->enqueue(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), 0);
        EXPECT_NE(copyRuntime->setTactic(1), 0);
        EXPECT_NE(copyRuntime->initResource(nullptr), 0);
    }
    EXPECT_EQ(getSafetyPluginCreator(nullptr, "AttentionPlugin"), nullptr);
    EXPECT_EQ(getSafetyPluginCreator("trt", "UnadmittedPlugin"), nullptr);
}

TEST(SafetyDescriptors, EnforcesContiguousCacheAndLinearStorage)
{
    // Given a five-input batched decoder, when validating storage, then only linear contiguous KV is admitted.
    AttentionParameters parameters{AttentionRole::kLLM, 2, 1, 64, 0.125F};
    TensorDescriptor inputs[]{descriptor({1, 3, 256}, DataType::kHALF), descriptor({1, 2, 1, 513, 64}, DataType::kHALF),
        descriptor({1}, DataType::kINT32), descriptor({1, 512, 32}, DataType::kFLOAT),
        descriptor({1}, DataType::kINT32)};
    TensorDescriptor outputs[]{descriptor({1, 3, 2, 64}, DataType::kHALF), inputs[1]};
    AttentionLaunch launch{};
    ASSERT_TRUE(validateAttentionDescriptors(parameters, inputs, 5, outputs, 2, launch));
    EXPECT_EQ(launch.rotaryDim, 32);
    EXPECT_EQ(launch.capacity, 513);
    inputs[0].stride.d[1] += 1;
    EXPECT_FALSE(validateAttentionDescriptors(parameters, inputs, 5, outputs, 2, launch));
    inputs[0].stride.d[1] -= 1;
    inputs[1] = descriptor({1, 2, 2, 513, 64}, DataType::kHALF);
    outputs[1] = inputs[1];
    EXPECT_FALSE(validateAttentionDescriptors(parameters, inputs, 5, outputs, 2, launch));
    inputs[1] = descriptor({1, 2, 1, 513, 64}, DataType::kHALF);
    outputs[1] = inputs[1];
    outputs[0].sizeInBytes = 2;
    EXPECT_FALSE(validateAttentionDescriptors(parameters, inputs, 5, outputs, 2, launch));
    outputs[0] = descriptor({1, 3, 2, 64}, DataType::kHALF);
    EXPECT_FALSE(validateAttentionDescriptors(parameters, inputs, 6, outputs, 2, launch));
}

class Resources : public ISafePluginResourceContext
{
public:
    RuntimeErrorInformation* error{};
    ISafeMemAllocator* getSafeMemAllocator() const noexcept override
    {
        return nullptr;
    }
    ISafeRecorder* getSafeRecorder() const noexcept override
    {
        return &sRecorder;
    }
    RuntimeErrorInformation* getRuntimeErrorInformation() const noexcept override
    {
        return error;
    }
};

void checkVisual(int32_t dim, int32_t tokens, int32_t split)
{
    Stream stream;
    auto plugin = makePlugin(AttentionRole::kVISUAL, TensorRTPhase::kRUNTIME, dim);
    ASSERT_NE(plugin, nullptr);
    auto* runtime
        = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    constexpr int32_t heads = 2;
    Buffer<half> q(tokens * heads * dim), k(q.count), v(q.count), output(q.count);
    Buffer<int32_t> ends(3);
    Buffer<uint64_t> error(1);
    Buffer<char> workspace(256 + q.count * sizeof(half));
    ends.host[0] = 0;
    ends.host[1] = split;
    ends.host[2] = tokens;
    for (size_t i = 0; i < q.count; ++i)
    {
        q.host[i] = __float2half(std::sin(i * 0.17F));
        k.host[i] = __float2half(std::cos(i * 0.13F));
        v.host[i] = __float2half(std::sin(i * 0.07F));
    }
    q.upload(stream.value);
    k.upload(stream.value);
    v.upload(stream.value);
    ends.upload(stream.value);
    error.upload(stream.value);
    Resources resources;
    resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
    ASSERT_EQ(runtime->initResource(&resources), 0);
    TensorDescriptor in[]{descriptor({tokens, heads, dim}, DataType::kHALF),
        descriptor({tokens, heads, dim}, DataType::kHALF), descriptor({tokens, heads, dim}, DataType::kHALF),
        descriptor({3}, DataType::kINT32), descriptor({std::max(split, tokens - split)}, DataType::kINT32)};
    TensorDescriptor out[]{in[0]};
    in[4].sizeInBytes = 0;
    in[4].memPlacement = MemoryPlacement::kNONE;
    void const* inputs[]{q.device, k.device, v.device, ends.device, nullptr};
    void* outputs[]{output.device};
    ASSERT_EQ(runtime->enqueue(in, out, inputs, outputs, workspace.device, stream.value), 0);
    output.download(stream.value);
    error.download(stream.value);
    checked(cudaStreamSynchronize(stream.value));
    EXPECT_EQ(error.host[0], 0U);
    for (int32_t token = 0; token < tokens; ++token)
    {
        int32_t const begin = token < split ? 0 : split;
        int32_t const end = token < split ? split : tokens;
        for (int32_t head = 0; head < heads; ++head)
        {
            std::vector<double> scores(end - begin);
            double denominator = 0;
            for (int32_t key = begin; key < end; ++key)
            {
                double score = 0;
                for (int32_t d = 0; d < dim; ++d)
                {
                    score += __half2float(q.host[(token * heads + head) * dim + d])
                        * __half2float(k.host[(key * heads + head) * dim + d]);
                }
                scores[key - begin] = std::exp(score * 0.125);
                denominator += scores[key - begin];
            }
            for (int32_t d = 0; d < dim; ++d)
            {
                double expected = 0;
                for (int32_t key = begin; key < end; ++key)
                {
                    expected += scores[key - begin] * __half2float(v.host[(key * heads + head) * dim + d]);
                }
                EXPECT_NEAR(__half2float(output.host[(token * heads + head) * dim + d]), expected / denominator, 0.002);
            }
        }
    }
    ends.host[1] = tokens + 1;
    ends.upload(stream.value);
    ASSERT_EQ(runtime->enqueue(in, out, inputs, outputs, workspace.device, stream.value), 0);
    error.download(stream.value);
    checked(cudaStreamSynchronize(stream.value));
    EXPECT_NE(error.host[0], 0U);
}

enum class MetadataError
{
    kNONE,
    kCACHE_OVERFLOW,
    kEMPTY_QUERY,
    kOVERSIZED_QUERY,
    kNEGATIVE_START,
    kROPE_OVERFLOW
};

//! Select fixed-profile geometry, cache ownership and one metadata corruption.
struct LlmConfig
{
    int32_t start{0};
    int32_t tokens{3};
    int32_t headSize{64};
    bool inPlaceCache{false};
    MetadataError error{MetadataError::kNONE};
};

//! Native LLM test data and independent CPU reference; the stream outlives every buffer.
struct LlmCase
{
    static constexpr int32_t kHEADS{4}, kKV_HEADS{2}, kCAPACITY{259}, kROTARY{32};
    LlmConfig const config;
    Stream stream;
    Buffer<half> packed, past, present, output;
    Buffer<float> rope{512 * kROTARY};
    Buffer<int32_t> lengths{1}, starts{1};
    Buffer<uint64_t> error{1};
    Buffer<char> workspace;
    std::vector<half> expectedPool, queries;

    //! Allocate and populate data for the selected case; native CUDA allocation failures throw.
    explicit LlmCase(LlmConfig const& selected)
        : config(selected)
        , packed(config.tokens * (kHEADS + 2 * kKV_HEADS) * config.headSize)
        , past(2 * kCAPACITY * kKV_HEADS * config.headSize)
        , present(past.count)
        , output(config.tokens * kHEADS * config.headSize)
        , workspace(256 + output.count * sizeof(half))
        , queries(output.count)
    {
        lengths.host[0] = std::min(2, config.tokens);
        starts.host[0] = config.start;
        for (size_t i = 0; i < packed.count; ++i)
        {
            packed.host[i] = __float2half(std::sin(i * 0.02F));
        }
        for (size_t i = 0; i < past.count; ++i)
        {
            past.host[i] = __float2half(std::cos(i * 0.03F) * 0.2F);
        }
        for (int32_t pos = 0; pos < 512; ++pos)
        {
            for (int32_t d = 0; d < kROTARY / 2; ++d)
            {
                rope.host[pos * kROTARY + d] = std::cos(pos * 0.1F + d * 0.01F);
                rope.host[pos * kROTARY + kROTARY / 2 + d] = std::sin(pos * 0.1F + d * 0.01F);
            }
        }
        expectedPool.assign(past.host, past.host + past.count);
        prepareReference();
        std::fill(packed.host + lengths.host[0] * (kHEADS + 2 * kKV_HEADS) * config.headSize,
            packed.host + packed.count, __float2half(std::numeric_limits<float>::quiet_NaN()));
        corruptMetadata();
    }

    //! Locate a token/component in the head-major contiguous cache.
    int32_t offset(int32_t position, bool value, int32_t kvHead, int32_t component) const
    {
        return (((value ? kKV_HEADS : 0) + kvHead) * kCAPACITY + position) * config.headSize + component;
    }

    //! Compute rotated queries and cache contents before any metadata is corrupted.
    void prepareReference()
    {
        for (int32_t t = 0; t < lengths.host[0]; ++t)
        {
            for (int32_t h = 0; h < kHEADS + kKV_HEADS; ++h)
            {
                for (int32_t d = 0; d < config.headSize; ++d)
                {
                    auto* vector = packed.host + (t * (kHEADS + 2 * kKV_HEADS) + h) * config.headSize;
                    float x = __half2float(vector[d]);
                    if (d < kROTARY)
                    {
                        int32_t const pair = d < kROTARY / 2 ? d + kROTARY / 2 : d - kROTARY / 2;
                        float const sign = d < kROTARY / 2 ? -1.F : 1.F;
                        x = x * rope.host[(config.start + t) * kROTARY + d % (kROTARY / 2)]
                            + sign * __half2float(vector[pair])
                                * rope.host[(config.start + t) * kROTARY + kROTARY / 2 + d % (kROTARY / 2)];
                    }
                    if (h < kHEADS)
                    {
                        queries[(t * kHEADS + h) * config.headSize + d] = __float2half(x);
                    }
                    else
                    {
                        expectedPool[offset(config.start + t, false, h - kHEADS, d)] = __float2half(x);
                        expectedPool[offset(config.start + t, true, h - kHEADS, d)]
                            = vector[kKV_HEADS * config.headSize + d];
                    }
                }
            }
        }
    }

    //! Apply exactly the selected malformed-metadata case after creating the valid oracle.
    void corruptMetadata()
    {
        switch (config.error)
        {
        case MetadataError::kNONE: break;
        case MetadataError::kCACHE_OVERFLOW: starts.host[0] = kCAPACITY; break;
        case MetadataError::kEMPTY_QUERY: lengths.host[0] = 0; break;
        case MetadataError::kOVERSIZED_QUERY: lengths.host[0] = config.tokens + 1; break;
        case MetadataError::kNEGATIVE_START: starts.host[0] = -1; break;
        case MetadataError::kROPE_OVERFLOW: starts.host[0] = 193; break;
        }
    }

    //! Initialize one plugin, execute on the owned stream and retrieve results before native cleanup.
    void run()
    {
        auto plugin = makePlugin(AttentionRole::kLLM, TensorRTPhase::kRUNTIME, config.headSize, kHEADS, kKV_HEADS);
        ASSERT_NE(plugin, nullptr);
        auto* runtime
            = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        packed.upload(stream.value);
        past.upload(stream.value);
        rope.upload(stream.value);
        lengths.upload(stream.value);
        starts.upload(stream.value);
        error.upload(stream.value);
        Resources resources;
        resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
        ASSERT_EQ(runtime->initResource(&resources), 0);
        TensorDescriptor in[]{
            descriptor({1, config.tokens, (kHEADS + 2 * kKV_HEADS) * config.headSize}, DataType::kHALF),
            descriptor({1, 2, kKV_HEADS, kCAPACITY, config.headSize}, DataType::kHALF),
            descriptor({1}, DataType::kINT32),
            descriptor({1, config.error == MetadataError::kROPE_OVERFLOW ? 193 : 512, kROTARY}, DataType::kFLOAT),
            descriptor({1}, DataType::kINT32)};
        TensorDescriptor out[]{descriptor({1, config.tokens, kHEADS, config.headSize}, DataType::kHALF), in[1]};
        void const* inputs[]{packed.device, past.device, lengths.device, rope.device, starts.device};
        void* outputs[]{output.device, config.inPlaceCache ? past.device : present.device};
        ASSERT_EQ(runtime->enqueue(in, out, inputs, outputs, workspace.device, stream.value), 0);
        output.download(stream.value);
        error.download(stream.value);
        if (config.inPlaceCache)
        {
            past.download(stream.value);
        }
        else
        {
            present.download(stream.value);
        }
        checked(cudaStreamSynchronize(stream.value));
    }

    //! Verify error/cache preservation or compare valid output with independent causal softmax.
    void checkResults() const
    {
        if (config.error != MetadataError::kNONE)
        {
            EXPECT_EQ(error.host[0], 12U);
            if (config.inPlaceCache)
            {
                for (size_t i = 0; i < past.count; ++i)
                {
                    EXPECT_EQ(__half2float(past.host[i]), __half2float(__float2half(std::cos(i * 0.03F) * 0.2F)));
                }
            }
            return;
        }
        EXPECT_EQ(error.host[0], 0U);
        auto* poolResult = config.inPlaceCache ? past.host : present.host;
        for (size_t i = 0; i < expectedPool.size(); ++i)
        {
            EXPECT_NEAR(__half2float(poolResult[i]), __half2float(expectedPool[i]), 0.001);
        }
        for (int32_t t = 0; t < config.tokens; ++t)
        {
            for (int32_t h = 0; h < kHEADS; ++h)
            {
                std::vector<double> scores(config.start + t + 1);
                double denominator = 0;
                for (int32_t key = 0; key <= config.start + t; ++key)
                {
                    double dot = 0;
                    for (int32_t d = 0; d < config.headSize; ++d)
                    {
                        dot += __half2float(queries[(t * kHEADS + h) * config.headSize + d])
                            * __half2float(expectedPool[offset(key, false, h / (kHEADS / kKV_HEADS), d)]);
                    }
                    scores[key] = std::exp(dot * 0.125);
                    denominator += scores[key];
                }
                for (int32_t d = 0; d < config.headSize; ++d)
                {
                    double expected = 0;
                    for (int32_t key = 0; key <= config.start + t; ++key)
                    {
                        expected
                            += scores[key] * __half2float(expectedPool[offset(key, true, h / (kHEADS / kKV_HEADS), d)]);
                    }
                    EXPECT_NEAR(__half2float(output.host[(t * kHEADS + h) * config.headSize + d]),
                        t < lengths.host[0] ? expected / denominator : 0, 0.003);
                }
            }
        }
    }
};

//! Run a named LLM case and avoid result assertions after failed plugin initialization/execution.
void checkLlm(LlmConfig const& config)
{
    LlmCase data(config);
    ASSERT_NO_FATAL_FAILURE(data.run());
    data.checkResults();
}

TEST(SafetyGpu, VisualPluginMatchesReferenceAndReportsInvalidSegments)
{
    checkVisual(64, 5, 2);
}

TEST(SafetyGpu, VisualRaggedSequencesCrossTileBoundaryD64AndD128)
{
    checkVisual(64, 261, 130);
    checkVisual(128, 261, 130);
}

TEST(SafetyGpu, InvalidLengthsAndStartDoNotModifyCache)
{
    // Given invalid length/start metadata, when executing, then report the error and preserve cached tokens.
    for (auto failure : {MetadataError::kEMPTY_QUERY, MetadataError::kOVERSIZED_QUERY, MetadataError::kNEGATIVE_START,
             MetadataError::kROPE_OVERFLOW})
    {
        SCOPED_TRACE(static_cast<int32_t>(failure));
        LlmConfig config{};
        config.start = 127;
        config.inPlaceCache = true;
        config.error = failure;
        checkLlm(config);
    }
}

TEST(SafetyGpu, LlmPrefill)
{
    checkLlm({});
}
TEST(SafetyGpu, LlmContiguousChunkAcrossTileBoundary)
{
    // Given cached prefix tokens, when appending a chunk, then attention and KV match the CPU reference.
    LlmConfig config{};
    config.start = 127;
    checkLlm(config);
}
TEST(SafetyGpu, LlmCacheCapacityBoundary)
{
    // Given two remaining cache slots, when appending two tokens, then the final slot is usable without rounding.
    LlmConfig config{};
    config.start = LlmCase::kCAPACITY - 2;
    checkLlm(config);
}

TEST(SafetyGpu, LlmInPlaceCache)
{
    LlmConfig config{};
    config.start = 129;
    config.inPlaceCache = true;
    checkLlm(config);
}
TEST(SafetyGpu, CacheOverflowReportsAsynchronousError)
{
    // Given a start at cache capacity, when executing, then report an error without changing the cache.
    LlmConfig config{};
    config.inPlaceCache = true;
    config.error = MetadataError::kCACHE_OVERFLOW;
    checkLlm(config);
}
} // namespace

namespace
{
class ConflictingCreator : public ISafePluginCreatorV3One
{
public:
    IPluginV3* createPlugin(char const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return nullptr;
    }
    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return nullptr;
    }
    char const* getPluginName() const noexcept override
    {
        return "AttentionPlugin";
    }
    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }
    char const* getPluginNamespace() const noexcept override
    {
        return "trt";
    }
    ISafeRecorder* getSafeRecorder() const noexcept override
    {
        return &sRecorder;
    }
    void setSafeRecorder(ISafeRecorder&) noexcept override {}
};

TEST(SafetyRegistry, RejectsDifferentCreatorWithSameIdentityBeforeMutation)
{
    // Given an admitted key with another identity, when registration inspects it, then conflict is terminal.
    // Mutation mapping: W5.
    Registry registry;
    ConflictingCreator conflicting;
    auto& processRecorder = processRegistrationRecorder();
    processRecorder.clear();
    registry.creators.push_back(&conflicting);
    EXPECT_EQ(registerSafetyPlugins(registry), ErrorCode::kINVALID_CONFIG);
    EXPECT_STREQ(processRecorder.getErrorDesc(0),
        "Safety plugin identity is already registered by a different creator instance");
    EXPECT_EQ(registry.registrationCalls, 0);
}

TEST(SafetyPlugin, LegacyDecoderBuildProfile)
{
    // Given batched FP16 decoder bindings, when configuring the builder, then accept five inputs and two FP16 outputs.
    auto plugin = makePlugin(AttentionRole::kLLM, TensorRTPhase::kBUILD);
    ASSERT_NE(plugin, nullptr);
    auto* build = static_cast<IPluginV3OneSafeBuildMSS*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    ASSERT_NE(build, nullptr);
    TensorDescriptor tensors[]{descriptor({1, 3, 256}, DataType::kHALF),
        descriptor({1, 2, 1, 513, 64}, DataType::kHALF), descriptor({1}, DataType::kINT32),
        descriptor({1, 1024, 32}, DataType::kFLOAT), descriptor({1}, DataType::kINT32),
        descriptor({1, 3, 2, 64}, DataType::kHALF), descriptor({1, 2, 1, 513, 64}, DataType::kHALF)};
    EXPECT_EQ(build->configurePlugin(tensors, 5, tensors + 5, 2), 0);
    for (int32_t i = 0; i < 7; ++i)
    {
        EXPECT_TRUE(build->supportsFormatCombination(i, tensors, 5, 2)) << i;
    }
    DataType const inputTypes[]{DataType::kHALF, DataType::kHALF, DataType::kINT32, DataType::kFLOAT, DataType::kINT32};
    DataType outputTypes[]{DataType::kFLOAT, DataType::kFLOAT};
    ASSERT_EQ(build->getOutputDataTypes(outputTypes, 2, inputTypes, 5), 0);
    EXPECT_EQ(outputTypes[0], DataType::kHALF);
    EXPECT_EQ(outputTypes[1], DataType::kHALF);
}

TEST(SafetyPlugin, BuilderProfileWorkspaceAndFieldRoundTrip)
{
    auto plugin = makePlugin(AttentionRole::kVISUAL, TensorRTPhase::kBUILD);
    ASSERT_NE(plugin, nullptr);
    auto* build = static_cast<IPluginV3OneSafeBuildMSS*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    ASSERT_NE(build, nullptr);
    TensorDescriptor inputs[]{descriptor({5, 2, 64}, DataType::kHALF), descriptor({5, 2, 64}, DataType::kHALF),
        descriptor({5, 2, 64}, DataType::kHALF), descriptor({3}, DataType::kINT32), descriptor({3}, DataType::kINT32)};
    TensorDescriptor outputs[]{inputs[0]};
    EXPECT_EQ(build->configurePlugin(inputs, 5, outputs, 1), 0);
    EXPECT_EQ(build->getWorkspaceSize(inputs, 5, outputs, 1), 256U);
    EXPECT_EQ(build->getNbTactics(), 0);
    inputs[0].dataType = DataType::kFLOAT;
    EXPECT_NE(build->configurePlugin(inputs, 5, outputs, 1), 0);
    auto* runtime
        = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    auto& creator = safetyAttentionCreator(AttentionRole::kVISUAL);
    std::unique_ptr<IPluginV3> restored(
        creator.createPlugin("restored", runtime->getFieldsToSerialize(), TensorRTPhase::kRUNTIME));
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->getCapabilityInterface(PluginCapabilityType::kBUILD), nullptr);
}

Registry* sRegistryUnderTest{};
int32_t sGraphCreationCalls{};
ISafeRecorder* sRegistryLookupRecorder{};
ISafeRecorder* sGraphRecorderPassed{};
ErrorCode sGraphCreationStatus{ErrorCode::kUNSPECIFIED_ERROR};
std::byte sGraphStorage{};
ITRTGraph* const sGraphToken{reinterpret_cast<ITRTGraph*>(&sGraphStorage)};
ISafeRecorder* sGraphOwnedRecorder{};
RefCount sGraphAcquireCount{-1};
RefCount sGraphReleaseCount{-1};

void resetGraphDoubles(ErrorCode creationStatus = ErrorCode::kUNSPECIFIED_ERROR) noexcept
{
    sGraphCreationCalls = 0;
    sRegistryLookupRecorder = nullptr;
    sGraphRecorderPassed = nullptr;
    sGraphCreationStatus = creationStatus;
    sGraphOwnedRecorder = nullptr;
    sGraphAcquireCount = -1;
    sGraphReleaseCount = -1;
}

// These DLI entry points are test doubles, not evidence of engine deserialization.
} // namespace

namespace nvinfer2
{
namespace safe
{
extern "C" ISafePluginRegistry* getSafePluginRegistry(ISafeRecorder& recorder) noexcept
{
    sRegistryLookupRecorder = &recorder;
    return sRegistryUnderTest;
}

extern "C" ErrorCode createTRTGraph(ITRTGraph*& graph, void const*, int64_t, AsciiChar const* companionSoPath,
    ISafeRecorder& recorder, bool, ISafeMemAllocator*) noexcept
{
    EXPECT_STREQ(companionSoPath, "/tmp/safety-test-companion.so");
    ++sGraphCreationCalls;
    sGraphRecorderPassed = &recorder;
    EXPECT_NE(sRegistryUnderTest, nullptr);
    EXPECT_EQ(sRegistryUnderTest->creators.size(), 2U);
    if (sGraphCreationStatus == ErrorCode::kSUCCESS)
    {
        sGraphOwnedRecorder = &recorder;
        sGraphAcquireCount = recorder.incRefCount();
        graph = sGraphToken;
    }
    return sGraphCreationStatus;
}

extern "C" ErrorCode destroyTRTGraph(ITRTGraph*& graph) noexcept
{
    if (graph != sGraphToken || sGraphOwnedRecorder == nullptr)
    {
        return ErrorCode::kINVALID_ARGUMENT;
    }
    sGraphReleaseCount = sGraphOwnedRecorder->decRefCount();
    sGraphOwnedRecorder = nullptr;
    graph = nullptr;
    return ErrorCode::kSUCCESS;
}
} // namespace safe
} // namespace nvinfer2

namespace
{
TEST(SafetyGraphInitialization, RegistrationPrecedesGraphCreationAndFailuresStopInitialization)
{
    // Given distinct process and graph recorders, when graph creation runs, then each DLI seam receives its owner.
    // Mutation mapping: W1, W6.
    SafetyAttentionModules modules;
    Registry registry;
    processRegistrationRecorder().clear();
    sOtherRecorder.clear();
    sRegistryUnderTest = &registry;
    resetGraphDoubles();
    ITRTGraph* graph = nullptr;
    char engine{};
    char const* companionSoPath = "/tmp/safety-test-companion.so";
    EXPECT_EQ(
        createSafetyGraph(modules, graph, nullptr, 1, sOtherRecorder, companionSoPath), ErrorCode::kINVALID_ARGUMENT);
    EXPECT_EQ(sGraphCreationCalls, 0);
    sOtherRecorder.clear();
    registry.failEnumeration = true;
    EXPECT_EQ(
        createSafetyGraph(modules, graph, &engine, 1, sOtherRecorder, companionSoPath), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_EQ(sGraphCreationCalls, 0);
    EXPECT_EQ(sRegistryLookupRecorder, &processRegistrationRecorder());
    ASSERT_EQ(sOtherRecorder.getNbErrors(), 1);
    EXPECT_EQ(sOtherRecorder.getErrorCode(0), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_STREQ(sOtherRecorder.getErrorDesc(0), "Safety plugin registration failed");
    sOtherRecorder.clear();
    registry.failEnumeration = false;
    EXPECT_EQ(
        createSafetyGraph(modules, graph, &engine, 1, sOtherRecorder, companionSoPath), ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_EQ(sGraphCreationCalls, 1);
    EXPECT_EQ(sGraphRecorderPassed, &sOtherRecorder);
    EXPECT_NE(sGraphRecorderPassed, sRegistryLookupRecorder);
    EXPECT_EQ(graph, nullptr);
    sRegistryUnderTest = nullptr;
}

TEST(SafetyGraphInitialization, MissingRegistryLatchesRuntimeFailure)
{
    // Given no Safety registry, when graph initialization starts, then the runtime receives an unspecified fatal.
    // Mutation mapping: W7.
    SafetyAttentionModules modules;
    processRegistrationRecorder().clear();
    sRegistryUnderTest = nullptr;
    resetGraphDoubles();
    SafetyErrorState state;
    ITRTGraph* graph{};
    char engine{};
    EXPECT_EQ(createSafetyGraph(modules, graph, &engine, 1, state.recorder(), "/tmp/safety-test-companion.so"),
        ErrorCode::kUNSPECIFIED_ERROR);
    EXPECT_EQ(sGraphCreationCalls, 0);
    auto const fatal = state.status();
    EXPECT_TRUE(fatal.fatalStatus());
    EXPECT_EQ(fatal.code(), SafetyStatusCode::kFailedExecution);
    EXPECT_STREQ(fatal.message(), "TensorRT Safety plugin registry is unavailable");
}

TEST(SafetyGraphInitialization, RegistrationDiagnosticLatchesAffectedRuntime)
{
    // Given malformed process registry state, when registration fails, then its exact diagnostic reaches the runtime.
    // Mutation mapping: W8.
    SafetyAttentionModules modules;
    Registry registry;
    registry.creators.resize(101);
    processRegistrationRecorder().clear();
    sRegistryUnderTest = &registry;
    resetGraphDoubles();
    SafetyErrorState state;
    ITRTGraph* graph{};
    char engine{};
    EXPECT_EQ(createSafetyGraph(modules, graph, &engine, 1, state.recorder(), "/tmp/safety-test-companion.so"),
        ErrorCode::kINVALID_CONFIG);
    EXPECT_EQ(sGraphCreationCalls, 0);
    auto const fatal = state.status();
    EXPECT_TRUE(fatal.fatalStatus());
    EXPECT_EQ(fatal.code(), SafetyStatusCode::kInvalidArgument);
    EXPECT_STREQ(fatal.message(),
        "Invalid Safety plugin creator list: expected count 0..100 and non-null storage when nonempty");
    sRegistryUnderTest = nullptr;
}

TEST(SafetyGraphInitialization, RuntimeRecorderPropagatesFatalAndOutlivesGraph)
{
    // Given a runtime recorder, when a graph reports a fatal error and is destroyed, then policy latches and refs
    // clear. Mutation mapping: P12-P14.
    SafetyAttentionModules modules;
    Registry registry;
    sRegistryUnderTest = &registry;
    resetGraphDoubles(ErrorCode::kSUCCESS);
    SafetyErrorState state;
    auto& runtimeRecorder = state.recorder();
    ITRTGraph* graph{};
    char engine{};
    ASSERT_EQ(createSafetyGraph(modules, graph, &engine, 1, runtimeRecorder, "/tmp/safety-test-companion.so"),
        ErrorCode::kSUCCESS);
    ASSERT_EQ(graph, sGraphToken);
    EXPECT_EQ(sGraphRecorderPassed, &runtimeRecorder);
    EXPECT_EQ(sGraphAcquireCount, 1);
    ASSERT_TRUE(sGraphRecorderPassed->reportError(ErrorCode::kINTERNAL_ERROR, "graph execution failed"));
    auto const fatal = state.status();
    EXPECT_TRUE(fatal.fatalStatus());
    EXPECT_EQ(fatal.code(), SafetyStatusCode::kFailedExecution);
    EXPECT_STREQ(fatal.message(), "graph execution failed");
    EXPECT_EQ(destroyTRTGraph(graph), ErrorCode::kSUCCESS);
    EXPECT_EQ(graph, nullptr);
    EXPECT_EQ(sGraphReleaseCount, 0);
    sRegistryUnderTest = nullptr;
    resetGraphDoubles();
}

// Missing or relative companions must fail before registry mutation or graph creation.
TEST(SafetyGraphInitialization, RejectsMissingOrRelativeCompanion)
{
    SafetyAttentionModules modules;
    Registry registry;
    sRegistryUnderTest = &registry;
    resetGraphDoubles();
    ITRTGraph* graph = nullptr;
    char engine{};
    char const* invalidPaths[]{nullptr, "", "engine.plan.so"};
    for (auto const* path : invalidPaths)
    {
        EXPECT_EQ(createSafetyGraph(modules, graph, &engine, 1, sOtherRecorder, path), ErrorCode::kINVALID_ARGUMENT);
        EXPECT_EQ(sGraphCreationCalls, 0);
        EXPECT_TRUE(registry.creators.empty());
        EXPECT_EQ(graph, nullptr);
    }
    sRegistryUnderTest = nullptr;
}
} // namespace

namespace
{
TEST(SafetyGpu, LlmSingleTokenD128Decode)
{
    LlmConfig config{};
    config.start = 128;
    config.inPlaceCache = true;
    config.tokens = 1;
    config.headSize = 128;
    checkLlm(config);
}
} // namespace

namespace
{
TEST(SafetyGpu, UnalignedBuffersAreRejectedWithoutRevivingPlugin)
{
    Stream stream;
    Buffer<half> data(2 * 64), output(data.count);
    Buffer<int32_t> ends(2);
    Buffer<uint64_t> error(1);
    Buffer<char> workspace(512);
    Resources resources;
    resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
    TensorDescriptor in[]{descriptor({1, 2, 64}, DataType::kHALF), descriptor({1, 2, 64}, DataType::kHALF),
        descriptor({1, 2, 64}, DataType::kHALF), descriptor({2}, DataType::kINT32), descriptor({1}, DataType::kINT32)};
    TensorDescriptor out[]{in[0]};
    for (int32_t failure = 0; failure < 10; ++failure)
    {
        SCOPED_TRACE(failure);
        auto plugin = makePlugin(AttentionRole::kVISUAL);
        ASSERT_NE(plugin, nullptr);
        auto* runtime
            = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        resources.error = reinterpret_cast<RuntimeErrorInformation*>(failure == 8 ? static_cast<void*>(output.device)
                : failure == 9 ? static_cast<void*>(workspace.device + 8)
                               : static_cast<void*>(error.device));
        ASSERT_EQ(runtime->initResource(&resources), 0);
        void const* inputs[]{data.device, data.device, data.device, ends.device, nullptr};
        void* outputs[]{output.device};
        void* scratch = workspace.device;
        if (failure < 3)
        {
            inputs[failure] = data.device + 1;
        }
        else if (failure == 3)
        {
            inputs[3] = reinterpret_cast<char const*>(ends.device) + 1;
        }
        else if (failure == 4)
        {
            outputs[0] = output.device + 1;
        }
        else if (failure == 5)
        {
            scratch = workspace.device + 16;
        }
        else if (failure == 6)
        {
            outputs[0] = data.device;
        }
        else if (failure == 7)
        {
            scratch = data.device;
        }
        EXPECT_NE(runtime->enqueue(in, out, inputs, outputs, scratch, stream.value), 0);
        inputs[0] = inputs[1] = inputs[2] = data.device;
        inputs[3] = ends.device;
        outputs[0] = output.device;
        EXPECT_NE(runtime->enqueue(in, out, inputs, outputs, workspace.device, stream.value), 0);
        EXPECT_NE(runtime->initResource(&resources), 0);
    }
}

TEST(SafetyGpu, InitializationCannotBeRetriedAfterFailureOrSuccess)
{
    Buffer<uint64_t> error(1);
    Resources resources;
    resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
    for (bool invalidFirst : {true, false})
    {
        SCOPED_TRACE(invalidFirst);
        auto plugin = makePlugin(AttentionRole::kVISUAL);
        ASSERT_NE(plugin, nullptr);
        auto* runtime
            = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        if (invalidFirst)
        {
            EXPECT_NE(runtime->initResource(nullptr), 0);
        }
        else
        {
            EXPECT_EQ(runtime->initResource(&resources), 0);
        }
        EXPECT_NE(runtime->initResource(&resources), 0);
        EXPECT_NE(runtime->initResource(&resources), 0);
    }
}
} // namespace

namespace
{
TEST(SafetyGpu, LlmRejectsPartialCacheAliasAndUnalignedRope)
{
    // Given overlapping or unaligned bindings, when enqueuing, then reject before launching and stay terminal.
    Stream stream;
    Buffer<half> packed(4 * 64), cache(2 * 128 * 64 + 16), output(2 * 64);
    Buffer<float> rope(32 + 4);
    Buffer<int32_t> lengths(1), starts(1);
    Buffer<uint64_t> error(1);
    Buffer<char> workspace(512);
    Resources resources;
    resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
    TensorDescriptor in[]{descriptor({1, 1, 4 * 64}, DataType::kHALF), descriptor({1, 2, 1, 128, 64}, DataType::kHALF),
        descriptor({1}, DataType::kINT32), descriptor({1, 1, 32}, DataType::kFLOAT), descriptor({0}, DataType::kINT32)};
    TensorDescriptor out[]{descriptor({1, 1, 2, 64}, DataType::kHALF), in[1]};
    for (int32_t failure = 0; failure < 3; ++failure)
    {
        SCOPED_TRACE(failure);
        auto plugin = makePlugin(AttentionRole::kLLM);
        ASSERT_NE(plugin, nullptr);
        auto* runtime
            = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        ASSERT_EQ(runtime->initResource(&resources), 0);
        void const* inputs[]{packed.device, cache.device, lengths.device, rope.device, nullptr};
        void* outputs[]{output.device, cache.device};
        if (failure == 0)
        {
            outputs[1] = cache.device + 8;
        }
        else if (failure == 1)
        {
            outputs[0] = cache.device;
        }
        else
        {
            inputs[3] = rope.device + 1;
        }
        EXPECT_NE(runtime->enqueue(in, out, inputs, outputs, workspace.device, stream.value), 0);
        EXPECT_NE(runtime->initResource(&resources), 0);
    }
}
} // namespace

namespace
{
TEST(SafetyPlugin, BuilderAcceptsUnresolvedStorageButRuntimeRequiresConcreteDescriptors)
{
    auto plugin = makePlugin(AttentionRole::kVISUAL, TensorRTPhase::kBUILD);
    ASSERT_NE(plugin, nullptr);
    auto* build = static_cast<IPluginV3OneSafeBuildMSS*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    ASSERT_NE(build, nullptr);
    TensorDescriptor tensors[]{descriptor({5, 2, 64}, DataType::kHALF), descriptor({5, 2, 64}, DataType::kHALF),
        descriptor({5, 2, 64}, DataType::kHALF), descriptor({3}, DataType::kINT32), descriptor({3}, DataType::kINT32),
        descriptor({5, 2, 64}, DataType::kHALF)};
    for (auto& tensor : tensors)
    {
        tensor.sizeInBytes = 0;
        tensor.stride.nbDims = -1;
        tensor.memPlacement = MemoryPlacement::kNONE;
    }
    for (int32_t position = 0; position < 6; ++position)
    {
        EXPECT_TRUE(build->supportsFormatCombination(position, tensors, 5, 1));
    }
    EXPECT_EQ(build->configurePlugin(tensors, 5, tensors + 5, 1), 0);
    EXPECT_EQ(build->getWorkspaceSize(tensors, 5, tensors + 5, 1), 256U);
    AttentionParameters parameters{AttentionRole::kVISUAL, 2, 2, 64, 0.125F};
    AttentionLaunch launch{};
    EXPECT_FALSE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
    tensors[1].dataType = DataType::kINT32;
    EXPECT_TRUE(build->supportsFormatCombination(0, tensors, 5, 1));
    EXPECT_FALSE(build->supportsFormatCombination(1, tensors, 5, 1));
    EXPECT_NE(build->configurePlugin(tensors, 5, tensors + 5, 1), 0);
    tensors[1].dataType = DataType::kHALF;
    tensors[0].componentsPerVector = 8;
    tensors[0].vectorizedDim = 2;
    EXPECT_FALSE(build->supportsFormatCombination(0, tensors, 5, 1));
    EXPECT_NE(build->configurePlugin(tensors, 5, tensors + 5, 1), 0);
    tensors[0].componentsPerVector = 1;
    tensors[0].vectorizedDim = -1;
    tensors[0].shape.d[1] = 3;
    EXPECT_NE(build->configurePlugin(tensors, 5, tensors + 5, 1), 0);
    tensors[0].shape.d[1] = 2;
    tensors[0].shape.d[0] = -1;
    EXPECT_NE(build->configurePlugin(tensors, 5, tensors + 5, 1), 0);
}
} // namespace

namespace
{
TEST(SafetyContract, UnspecifiedPlacementStillRequiresConcreteRuntimeStorage)
{
    AttentionParameters parameters{AttentionRole::kVISUAL, 2, 2, 64, 0.125F};
    AttentionLaunch launch{};
    TensorDescriptor tensors[]{descriptor({5, 2, 64}, DataType::kHALF), descriptor({5, 2, 64}, DataType::kHALF),
        descriptor({5, 2, 64}, DataType::kHALF), descriptor({3}, DataType::kINT32), descriptor({3}, DataType::kINT32),
        descriptor({5, 2, 64}, DataType::kHALF)};
    for (auto& tensor : tensors)
    {
        tensor.memPlacement = MemoryPlacement::kNONE;
    }
    EXPECT_TRUE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
    for (auto placement : {MemoryPlacement::kCPU, MemoryPlacement::kCPU_PINNED, MemoryPlacement::kMANAGED})
    {
        tensors[0].memPlacement = placement;
        EXPECT_FALSE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
    }
    tensors[0].memPlacement = MemoryPlacement::kNONE;
    tensors[0].stride.nbDims = -1;
    EXPECT_FALSE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
    tensors[0].stride.nbDims = 3;
    tensors[0].sizeInBytes = 0;
    EXPECT_FALSE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
    tensors[0].sizeInBytes = 5 * 2 * 64 * sizeof(half);
    tensors[0].stride.d[0] = 1;
    EXPECT_FALSE(validateAttentionDescriptors(parameters, tensors, 5, tensors + 5, 1, launch));
}
} // namespace

namespace
{
TEST(SafetyPlugin, FactoryUsesExporterNamespaces)
{
    // Given exporter identities, when resolving creators, then only each role's exact namespace is accepted.
    EXPECT_EQ(getSafetyPluginCreator("trt", "AttentionPlugin"), &safetyAttentionCreator(AttentionRole::kLLM));
    EXPECT_EQ(
        getSafetyPluginCreator("trt_edgellm", "ViTAttentionPlugin"), &safetyAttentionCreator(AttentionRole::kVISUAL));
    EXPECT_EQ(getSafetyPluginCreator("trt", "ViTAttentionPlugin"), nullptr);
    EXPECT_EQ(getSafetyPluginCreator("trt_edgellm", "AttentionPlugin"), nullptr);
    Registry registry;
    ASSERT_EQ(registerSafetyPlugins(registry), ErrorCode::kSUCCESS);
    ASSERT_EQ(registry.creators.size(), 2U);
    EXPECT_STREQ(static_cast<ISafePluginCreatorV3One*>(registry.creators[0])->getPluginNamespace(), "trt");
    EXPECT_STREQ(static_cast<ISafePluginCreatorV3One*>(registry.creators[1])->getPluginNamespace(), "trt_edgellm");
}

TEST(SafetyGpu, CloneRetainsOwnerAfterCreationScopeEnds)
{
    // Given an unprepared owner, when its plugin is cloned outside the scope, then it cannot borrow another owner.
    Buffer<uint64_t> error(1);
    Resources resources;
    resources.error = reinterpret_cast<RuntimeErrorInformation*>(error.device);
    SafetyAttentionModules unprepared;
    std::unique_ptr<IPluginV3> plugin;
    {
        SafetyAttentionModuleScope scope(unprepared);
        plugin = makePlugin(AttentionRole::kVISUAL);
    }
    ASSERT_NE(plugin, nullptr);
    auto* runtime
        = static_cast<IPluginV3OneSafeRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    std::unique_ptr<IPluginV3> clone(runtime->clone());
    ASSERT_NE(clone, nullptr);
    auto* clonedRuntime
        = static_cast<IPluginV3OneSafeRuntime*>(clone->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    EXPECT_NE(clonedRuntime->initResource(&resources), 0);
    EXPECT_NE(runtime->initResource(&resources), 0);
    // Given restored creator routing, when creating another plugin, then the healthy owner's initialization succeeds.
    auto independent = makePlugin(AttentionRole::kVISUAL);
    auto* independentRuntime
        = static_cast<IPluginV3OneSafeRuntime*>(independent->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    EXPECT_EQ(independentRuntime->initResource(&resources), 0);
}
} // namespace
