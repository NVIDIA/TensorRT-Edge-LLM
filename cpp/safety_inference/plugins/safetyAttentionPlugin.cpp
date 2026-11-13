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

#include "safetyAttentionPlugin.h"
#include "../kernels/safetyAttentionModules.h"

#if defined(EDGELLM_SAFETY_PLUGIN_BUILDER)
#include <NvInfer.h>
#endif

#include <cstring>
#include <limits>
#include <new>

namespace trt_edgellm
{
namespace safety
{
namespace
{
using namespace nvinfer1;
using namespace nvinfer2::safe;

// Only creator routing is shared, under external Initialization serialization. No execution reads this pointer.
SafetyAttentionModules* sModuleOwner{};

char const* pluginNamespace(AttentionRole role) noexcept
{
    return role == AttentionRole::kLLM ? "trt" : "trt_edgellm";
}

char const* pluginName(AttentionRole role) noexcept
{
    return role == AttentionRole::kLLM ? "AttentionPlugin" : "ViTAttentionPlugin";
}

int32_t report(ISafeRecorder* recorder, char const* message) noexcept
{
    if (recorder != nullptr)
    {
        recorder->reportError(ErrorCode::kUNSPECIFIED_ERROR, message);
    }
    return 1;
}

bool inputHasData(bool visual, int32_t index, TensorDescriptor const& descriptor) noexcept
{
    return index != 4 || (!visual && descriptor.shape.d[0] != 0);
}

struct BindingRange
{
    uintptr_t begin{};
    uint64_t bytes{};
};

bool overlap(BindingRange a, BindingRange b) noexcept
{
    return a.bytes != 0 && b.bytes != 0 && a.begin < b.begin + b.bytes && b.begin < a.begin + a.bytes;
}

bool disjointBindings(bool visual, AttentionLaunch const& shape, TensorDescriptor const* inputDesc,
    TensorDescriptor const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    uint64_t* errorMask) noexcept
{
    int32_t const nInputs = 5;
    int32_t const nOutputs = visual ? 1 : 2;
    std::array<BindingRange, 8> io{};
    for (int32_t i = 0; i < nInputs + nOutputs; ++i)
    {
        bool const input = i < nInputs;
        auto const& desc = input ? inputDesc[i] : outputDesc[i - nInputs];
        bool const noData = input && !inputHasData(visual, i, desc);
        uint64_t bytes = noData ? 0 : desc.bytesPerComponent;
        for (int32_t dim = 0; dim < desc.shape.nbDims; ++dim)
        {
            bytes *= desc.shape.d[dim];
        }
        auto const pointer = input ? inputs[i] : outputs[i - nInputs];
        io[i] = {reinterpret_cast<uintptr_t>(pointer), bytes};
        if (io[i].begin > std::numeric_limits<uintptr_t>::max() - bytes)
        {
            return false;
        }
    }
    BindingRange const scratch{reinterpret_cast<uintptr_t>(workspace), visual ? 256 : attentionWorkspaceBytes(shape)};
    BindingRange const error{reinterpret_cast<uintptr_t>(errorMask), sizeof(*errorMask)};
    if (scratch.begin > std::numeric_limits<uintptr_t>::max() - scratch.bytes
        || error.begin > std::numeric_limits<uintptr_t>::max() - error.bytes || overlap(scratch, error))
    {
        return false;
    }
    for (int32_t i = 0; i < nInputs + nOutputs; ++i)
    {
        if (overlap(io[i], scratch) || overlap(io[i], error))
        {
            return false;
        }
        for (int32_t out = nInputs; out < nInputs + nOutputs; ++out)
        {
            bool const inPlaceCache = !visual && i == 1 && out == nInputs + 1 && inputs[1] == outputs[1];
            if (i != out && !inPlaceCache && overlap(io[i], io[out]))
            {
                return false;
            }
        }
    }
    return true;
}

class AttentionRuntime : public IPluginV3, public IPluginV3OneSafeCore, public IPluginV3OneSafeRuntime
{
public:
    explicit AttentionRuntime(
        AttentionParameters parameters, ISafeRecorder* recorder, SafetyAttentionModules* modules) noexcept
        : mParameters(parameters)
        , mRecorder(recorder)
        , mModules(modules)
    {
        mFields[0] = {parameters.role == AttentionRole::kLLM ? "num_q_heads" : "num_heads", &mParameters.qHeads,
            PluginFieldType::kINT32, 1};
        mFields[1] = {"head_size", &mParameters.headSize, PluginFieldType::kINT32, 1};
        mFields[2] = {"attention_scale", &mParameters.scale, PluginFieldType::kFLOAT32, 1};
        mFields[3] = {"num_kv_heads", &mParameters.kvHeads, PluginFieldType::kINT32, 1};
        mCollection = {parameters.role == AttentionRole::kLLM ? 4 : 3, mFields.data()};
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE)
        {
            return static_cast<IPluginV3OneSafeCore*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneSafeRuntime*>(this);
        }
        return nullptr;
    }

    char const* getPluginName() const noexcept override
    {
        return pluginName(mParameters.role);
    }
    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }
    char const* getPluginNamespace() const noexcept override
    {
        return pluginNamespace(mParameters.role);
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        return tactic == 0 ? 0 : report(mRecorder, "Safety attention supports only tactic 0");
    }

    int32_t initResource(ISafePluginResourceContext const* context) noexcept override
    {
        if (mInitializationAttempted)
        {
            mReady = false;
            return report(mRecorder, "Safety attention initialization cannot be retried");
        }
        mInitializationAttempted = true;
        mReady = false;
        if (context == nullptr)
        {
            return report(mRecorder, "Safety attention requires a graph resource context");
        }
        mRecorder = context->getSafeRecorder();
        mError = context->getRuntimeErrorInformation();
        if (mRecorder == nullptr || mError == nullptr)
        {
            return report(mRecorder, "Safety attention requires a graph recorder and device error buffer");
        }
        if (mModules == nullptr || mModules->prepare() != CUDA_SUCCESS || prepareAttentionKernels() != cudaSuccess)
        {
            return report(mRecorder, "Safety attention kernel initialization failed");
        }
        mReady = true;
        return 0;
    }

    int32_t enqueue(TensorDescriptor const* inputsDesc, TensorDescriptor const* outputsDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        auto reject = [&](char const* message) {
            mReady = false;
            return report(mRecorder, message);
        };
        bool const visual = mParameters.role == AttentionRole::kVISUAL;
        int32_t const inputCount = 5;
        AttentionLaunch shape{};
        if (!mReady || inputs == nullptr || outputs == nullptr || workspace == nullptr || stream == nullptr
            || !validateAttentionDescriptors(mParameters, inputsDesc, inputCount, outputsDesc, visual ? 1 : 2, shape))
        {
            return reject("Invalid Safety attention execution descriptors or lifecycle");
        }
        if (outputs[0] == nullptr || (!visual && outputs[1] == nullptr))
        {
            return reject("Null Safety attention output");
        }
        auto aligned = [](void const* pointer, uintptr_t alignment) {
            return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
        };
        if (!aligned(workspace, 256) || !aligned(outputs[0], 16) || (!visual && !aligned(outputs[1], 16)))
        {
            return reject("Unaligned Safety attention workspace or output");
        }
        for (int32_t i = 0; i < inputCount; ++i)
        {
            bool const hasData = inputHasData(visual, i, inputsDesc[i]);
            uintptr_t const alignment = visual ? (i < 3 ? 16 : 4) : (i < 2 || i == 3 ? 16 : 4);
            if (hasData && (inputs[i] == nullptr || !aligned(inputs[i], alignment)))
            {
                return reject("Null or unaligned Safety attention input");
            }
        }
        if (!disjointBindings(visual, shape, inputsDesc, outputsDesc, inputs, outputs, workspace, &mError->bitMask))
        {
            return reject("Overlapping Safety attention bindings");
        }
        cudaError_t status;
        if (visual)
        {
            status = launchSafetyVisualAttention(*mModules, mBindings, shape, static_cast<half const*>(inputs[0]),
                static_cast<half const*>(inputs[1]), static_cast<half const*>(inputs[2]),
                static_cast<int32_t const*>(inputs[3]), static_cast<half*>(outputs[0]), workspace, &mError->bitMask,
                stream);
        }
        else
        {
            auto const* starts = inputsDesc[4].shape.d[0] == 0 ? nullptr : static_cast<int32_t const*>(inputs[4]);
            status = launchSafetyLlmAttention(*mModules, mBindings, shape, static_cast<half const*>(inputs[0]),
                static_cast<half const*>(inputs[1]), static_cast<int32_t const*>(inputs[2]),
                static_cast<float const*>(inputs[3]), starts, static_cast<half*>(outputs[0]),
                static_cast<half*>(outputs[1]), workspace, &mError->bitMask, stream);
        }
        if (status != cudaSuccess)
        {
            return reject("Safety attention CUDA launch failed");
        }
        return 0;
    }

    IPluginV3* clone() noexcept override
    {
        auto* result = new (std::nothrow) AttentionRuntime(mParameters, mRecorder, mModules);
        if (result == nullptr)
        {
            report(mRecorder, "Safety attention clone allocation failed");
        }
        return result;
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mCollection;
    }

protected:
    AttentionParameters mParameters;
    ISafeRecorder* mRecorder{nullptr};
    SafetyAttentionModules* mModules{nullptr};

private:
    // Serialization fields point into this object; cloning reconstructs these pointers.
    std::array<PluginField, 4> mFields{};
    PluginFieldCollection mCollection{};
    RuntimeErrorInformation* mError{nullptr};
    bool mReady{false};
    fmha::PreparedBindings mBindings{};
    bool mInitializationAttempted{false};
};

#if defined(EDGELLM_SAFETY_PLUGIN_BUILDER)
class AttentionBuild final : public AttentionRuntime, public IPluginV3OneSafeBuildMSS
{
public:
    explicit AttentionBuild(
        AttentionParameters parameters, ISafeRecorder* recorder, SafetyAttentionModules* modules) noexcept
        : AttentionRuntime(parameters, recorder, modules)
    {
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        return type == PluginCapabilityType::kBUILD ? static_cast<IPluginV3OneSafeBuildMSS*>(this)
                                                    : AttentionRuntime::getCapabilityInterface(type);
    }

    IPluginV3* clone() noexcept override
    {
        auto* result = new (std::nothrow) AttentionBuild(mParameters, mRecorder, mModules);
        if (result == nullptr)
        {
            report(mRecorder, "Safety attention build clone allocation failed");
        }
        return result;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return mParameters.role == AttentionRole::kLLM ? 2 : 1;
    }

    int32_t configurePlugin(
        TensorDescriptor const* in, int32_t nbInputs, TensorDescriptor const* out, int32_t nbOutputs) noexcept override
    {
        AttentionLaunch shape{};
        return validateAttentionBuildDescriptors(mParameters, in, nbInputs, out, nbOutputs, shape)
            ? 0
            : report(mRecorder, "Unsupported Safety attention fixed profile");
    }

    bool supportsFormatCombination(
        int32_t pos, TensorDescriptor const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        return inOut != nullptr && nbInputs == 5 && nbOutputs == getNbOutputs() && pos >= 0
            && pos < nbInputs + nbOutputs && supportsAttentionFormat(mParameters, pos, inOut[pos]);
    }

    int32_t getOutputDataTypes(
        DataType* outputs, int32_t nbOutputs, DataType const* inputs, int32_t nbInputs) const noexcept override
    {
        if (outputs == nullptr || inputs == nullptr || nbOutputs != getNbOutputs() || nbInputs != 5)
        {
            return report(mRecorder, "Invalid Safety attention output type query");
        }
        for (int32_t i = 0; i < nbOutputs; ++i)
        {
            outputs[i] = DataType::kHALF;
        }
        return 0;
    }

    int32_t getSymbolicOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& builder) const noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbOutputs != getNbOutputs() || nbInputs != 5
            || inputs[0].nbDims != 3)
        {
            return report(mRecorder, "Invalid Safety attention output shape query");
        }
        outputs[0] = inputs[0];
        if (nbOutputs == 2)
        {
            if (inputs[1].nbDims != 5)
            {
                return report(mRecorder, "Safety attention requires a contiguous KV cache");
            }
            outputs[0].nbDims = 4;
            outputs[0].d[2] = builder.constant(mParameters.qHeads);
            outputs[0].d[3] = builder.constant(mParameters.headSize);
            outputs[1] = inputs[1];
        }
        return 0;
    }

    size_t getWorkspaceSize(TensorDescriptor const* inputs, int32_t nbInputs, TensorDescriptor const* outputs,
        int32_t nbOutputs) const noexcept override
    {
        AttentionLaunch shape{};
        if (!validateAttentionBuildDescriptors(mParameters, inputs, nbInputs, outputs, nbOutputs, shape))
        {
            report(mRecorder, "Invalid Safety attention workspace profile");
            return 0;
        }
        return mParameters.role == AttentionRole::kVISUAL ? 256 : attentionWorkspaceBytes(shape);
    }
};
#endif

class AttentionCreator final : public ISafePluginCreatorV3One
{
public:
    explicit AttentionCreator(AttentionRole role) noexcept
        : mRole(role)
    {
        mFields[0] = {role == AttentionRole::kLLM ? "num_q_heads" : "num_heads", nullptr, PluginFieldType::kINT32, 1};
        mFields[1] = {"head_size", nullptr, PluginFieldType::kINT32, 1};
        mFields[2] = {"attention_scale", nullptr, PluginFieldType::kFLOAT32, 1};
        mFields[3] = {"qkv_scales", nullptr, PluginFieldType::kFLOAT32, 3};
        int32_t count = 4;
        if (role == AttentionRole::kLLM)
        {
            mFields[count++] = {"num_kv_heads", nullptr, PluginFieldType::kINT32, 1};
            mFields[count++] = {"sliding_window_size", nullptr, PluginFieldType::kINT32, 1};
            for (auto const* field : kDISABLED_INTEGER_FIELDS)
            {
                mFields[count++] = {field, nullptr, PluginFieldType::kINT32, 1};
            }
            mFields[count++] = {"rms_norm_eps", nullptr, PluginFieldType::kFLOAT32, 1};
            mFields[count++] = {"xqa_jit_kernels", nullptr, PluginFieldType::kCHAR, 0};
        }
        mCollection = {count, mFields.data()};
    }

    char const* getPluginName() const noexcept override
    {
        return pluginName(mRole);
    }
    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }
    char const* getPluginNamespace() const noexcept override
    {
        return pluginNamespace(mRole);
    }
    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    ISafeRecorder* getSafeRecorder() const noexcept override
    {
        return mRecorder;
    }
    void setSafeRecorder(ISafeRecorder& recorder) noexcept override
    {
        mRecorder = &recorder;
    }

    IPluginV3* createPlugin(
        char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept override
    {
        AttentionParameters parameters{};
        if (name == nullptr || !parseAttentionFields(mRole, fields, parameters))
        {
            report(mRecorder, "Unsupported Safety attention fields");
            return nullptr;
        }
        IPluginV3* result = nullptr;
        if (phase == TensorRTPhase::kRUNTIME)
        {
            result = new (std::nothrow) AttentionRuntime(parameters, mRecorder, sModuleOwner);
        }
#if defined(EDGELLM_SAFETY_PLUGIN_BUILDER)
        else if (phase == TensorRTPhase::kBUILD)
        {
            result = new (std::nothrow) AttentionBuild(parameters, mRecorder, sModuleOwner);
        }
#endif
        if (result == nullptr)
        {
            report(mRecorder, "Safety attention creation failed or build capability unavailable");
        }
        return result;
    }

private:
    AttentionRole mRole;
    std::array<PluginField, kMAX_FIELDS> mFields{};
    PluginFieldCollection mCollection{};
    ISafeRecorder* mRecorder{nullptr};
};
} // namespace

SafetyAttentionModuleScope::SafetyAttentionModuleScope(SafetyAttentionModules& modules) noexcept
    : mPrevious(sModuleOwner)
{
    sModuleOwner = &modules;
}

SafetyAttentionModuleScope::~SafetyAttentionModuleScope() noexcept
{
    sModuleOwner = mPrevious;
}

ISafePluginCreatorV3One& safetyAttentionCreator(AttentionRole role) noexcept
{
    static AttentionCreator llm{AttentionRole::kLLM};
    static AttentionCreator visual{AttentionRole::kVISUAL};
    return role == AttentionRole::kLLM ? llm : visual;
}
} // namespace safety
} // namespace trt_edgellm

extern "C" nvinfer1::IPluginCreatorInterface* getSafetyPluginCreator(
    char const* pluginNamespace, char const* pluginName) noexcept
{
    if (pluginNamespace == nullptr || pluginName == nullptr)
    {
        return nullptr;
    }
    using namespace trt_edgellm::safety;
    if (std::strcmp(pluginNamespace, "trt") == 0 && std::strcmp(pluginName, "AttentionPlugin") == 0)
    {
        return &safetyAttentionCreator(AttentionRole::kLLM);
    }
    if (std::strcmp(pluginNamespace, "trt_edgellm") == 0 && std::strcmp(pluginName, "ViTAttentionPlugin") == 0)
    {
        return &safetyAttentionCreator(AttentionRole::kVISUAL);
    }
    return nullptr;
}

#if defined(EDGELLM_SAFETY_PLUGIN_BUILDER)
namespace
{
struct SafetyAttentionBuildSession
{
    trt_edgellm::safety::SafetyAttentionModules modules;
    trt_edgellm::safety::SafetyAttentionModuleScope scope{modules};
    bool loaded{false};
};
} // namespace

extern "C" int32_t createSafetyAttentionBuildSession(void** session, bool loadModules) noexcept
{
    if (session == nullptr)
    {
        return static_cast<int32_t>(CUDA_ERROR_INVALID_VALUE);
    }
    *session = nullptr;
    auto* owner = new (std::nothrow) SafetyAttentionBuildSession();
    if (owner == nullptr)
    {
        return static_cast<int32_t>(CUDA_ERROR_OUT_OF_MEMORY);
    }
    if (loadModules)
    {
        auto const status = owner->modules.load();
        if (status != CUDA_SUCCESS)
        {
            delete owner;
            return static_cast<int32_t>(status);
        }
        owner->loaded = true;
    }
    *session = owner;
    return static_cast<int32_t>(CUDA_SUCCESS);
}

extern "C" int32_t destroySafetyAttentionBuildSession(void* session, bool releaseModules) noexcept
{
    auto* owner = static_cast<SafetyAttentionBuildSession*>(session);
    if (owner == nullptr)
    {
        return static_cast<int32_t>(CUDA_ERROR_INVALID_VALUE);
    }
    auto const status = releaseModules && owner->loaded ? owner->modules.release() : CUDA_SUCCESS;
    delete owner;
    return static_cast<int32_t>(status);
}
#endif
