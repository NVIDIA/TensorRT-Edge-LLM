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

#include "lowRankCandidateHeadPlugin.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "kernels/speculative/lowRankCandidateHeadKernels.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kPLUGIN_NAME{"LowRankCandidateHeadPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kFIELD_NUM_CANDIDATES{"num_candidates"};
constexpr char const* kFIELD_HIDDEN_SIZE{"hidden_size"};
constexpr char const* kFIELD_RANK{"rank"};
constexpr char const* kFIELD_VOCAB_SIZE{"vocab_size"};
constexpr char const* kFIELD_MAX_ROWS{"max_rows"};
constexpr char const* kFIELD_WORKSPACE_BYTES{"workspace_bytes"};
constexpr int32_t kNB_INPUTS{4};
constexpr int32_t kNB_OUTPUTS{2};
constexpr int32_t kIN_HIDDEN{0};
constexpr int32_t kIN_DOWN_PROJ{1};
constexpr int32_t kIN_UP_PROJ{2};
constexpr int32_t kIN_LM_HEAD{3};
constexpr int32_t kOUT_IDS{0};
constexpr int32_t kOUT_LOG_PROBS{1};
constexpr int32_t kMATRIX_DIMS{2};
constexpr char const* kINPUT_NAMES[kNB_INPUTS]{"hidden", "down_proj", "up_proj", "lm_head"};

//! Looks a field up by name; absent fields return nullopt so unknown and optional fields need no special casing.
std::optional<int64_t> findIntegerField(PluginFieldCollection const* fields, char const* name)
{
    for (int32_t i = 0; fields != nullptr && i < fields->nbFields; ++i)
    {
        PluginField const& field = fields->fields[i];
        if (field.name == nullptr || std::strcmp(field.name, name) != 0)
        {
            continue;
        }
        if (field.length != 1 || field.data == nullptr)
        {
            throw std::invalid_argument(std::string("LowRankCandidateHeadPlugin: invalid field ") + name);
        }
        if (field.type == PluginFieldType::kINT32)
        {
            return *static_cast<int32_t const*>(field.data);
        }
        if (field.type == PluginFieldType::kINT64)
        {
            return *static_cast<int64_t const*>(field.data);
        }
        throw std::invalid_argument(std::string("LowRankCandidateHeadPlugin: field ") + name + " must be an integer");
    }
    return std::nullopt;
}

int64_t requireIntegerField(PluginFieldCollection const* fields, char const* name)
{
    std::optional<int64_t> const value = findIntegerField(fields, name);
    if (!value.has_value())
    {
        throw std::invalid_argument(std::string("LowRankCandidateHeadPlugin: missing field ") + name);
    }
    return *value;
}

int32_t requireInt32Field(PluginFieldCollection const* fields, char const* name)
{
    int64_t const value = requireIntegerField(fields, name);
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
    {
        throw std::invalid_argument(std::string("LowRankCandidateHeadPlugin: field ") + name + " is out of range");
    }
    return static_cast<int32_t>(value);
}

bool isMatrix(Dims const& dims, int64_t rows, int64_t cols) noexcept
{
    return dims.nbDims == kMATRIX_DIMS && dims.d[0] == rows && dims.d[1] == cols;
}

struct ResolvedHead
{
    kernel::LowRankCandidateHeadShape shape;
    int32_t maxRows;
};

//! Weights are engine constants or, with an externalized lm_head, inputs whose profile pins one shape.
std::optional<ResolvedHead> resolveHead(
    DynamicPluginTensorDesc const* inputs, int32_t numCandidates, std::string const& layerName) noexcept
{
    Dims const& hidden = inputs[kIN_HIDDEN].max;
    Dims const& down = inputs[kIN_DOWN_PROJ].max;
    Dims const& up = inputs[kIN_UP_PROJ].max;
    Dims const& lmHead = inputs[kIN_LM_HEAD].max;
    bool const fixedWeights = dimsEqual(inputs[kIN_DOWN_PROJ].min, down) && dimsEqual(inputs[kIN_UP_PROJ].min, up)
        && dimsEqual(inputs[kIN_LM_HEAD].min, lmHead);
    bool const consistent = fixedWeights && down.nbDims == kMATRIX_DIMS && down.d[0] > 0 && down.d[1] > 0
        && up.nbDims == kMATRIX_DIMS && up.d[0] > 0 && isMatrix(up, up.d[0], down.d[0])
        && isMatrix(lmHead, up.d[0], down.d[1]) && hidden.nbDims == kMATRIX_DIMS && hidden.d[1] == down.d[1]
        && inputs[kIN_HIDDEN].min.nbDims == kMATRIX_DIMS && inputs[kIN_HIDDEN].min.d[1] == down.d[1] && hidden.d[0] >= 1
        && hidden.d[0] <= std::numeric_limits<int32_t>::max() && up.d[0] <= std::numeric_limits<int32_t>::max();
    if (!consistent)
    {
        LOG_ERROR(
            "LowRankCandidateHeadPlugin '%s': inconsistent input shapes hidden=%s down_proj=%s up_proj=%s lm_head=%s; "
            "expected hidden [R, H] with R >= 1, down_proj [r, H], up_proj [V, r] and lm_head [V, H] with fixed "
            "weight shapes",
            layerName.c_str(), dimsToString(hidden).c_str(), dimsToString(down).c_str(), dimsToString(up).c_str(),
            dimsToString(lmHead).c_str());
        return std::nullopt;
    }
    kernel::LowRankCandidateHeadShape shape;
    shape.hiddenSize = static_cast<int32_t>(down.d[1]);
    shape.rank = static_cast<int32_t>(down.d[0]);
    shape.vocabSize = static_cast<int32_t>(up.d[0]);
    shape.numCandidates = numCandidates;
    return ResolvedHead{shape, static_cast<int32_t>(hidden.d[0])};
}

cudaDeviceProp currentDeviceProperties()
{
    int32_t device{};
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

//! TensorRT may query a profile's workspace before it configures a larger profile of the same instance, and the
//! runner is laid out for the largest; reserving the largest pass makes every query return the same size.
size_t reservedWorkspaceBytes(kernel::LowRankCandidateHeadShape const& shape)
{
    return kernel::LowRankCandidateHeadRunner::workspaceBytes(
        shape, kernel::LowRankCandidateHeadRunner::kMaxRowsPerPass, currentDeviceProperties());
}

//! Building a runner sets kernel attributes and queries occupancy, so each device builds one per head and reservation.
//! Runners hold no CUDA resources and are never erased, so the returned reference stays valid.
kernel::LowRankCandidateHeadRunner const& sharedRunner(kernel::LowRankCandidateHeadShape const& shape, int32_t maxRows,
    int64_t workspaceBytes, std::string const& layerName)
{
    using Key = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t>;
    static std::mutex sMutex;
    static std::map<Key, kernel::LowRankCandidateHeadRunner> sRunners;
    int32_t device{};
    CUDA_CHECK(cudaGetDevice(&device));
    Key const key{device, shape.hiddenSize, shape.rank, shape.vocabSize, shape.numCandidates, maxRows, workspaceBytes};
    std::lock_guard<std::mutex> lock(sMutex);
    auto found = sRunners.find(key);
    if (found == sRunners.end())
    {
        cudaDeviceProp const prop = currentDeviceProperties();
        size_t const reserved = workspaceBytes > 0 ? static_cast<size_t>(workspaceBytes) : 0;
        // Results do not depend on the configuration, so a smaller one that fits the reservation only changes the
        // speed; withinWorkspace throws, naming the smallest workspace this device needs, when none fits.
        kernel::LowRankCandidateHeadRunner const runner
            = kernel::LowRankCandidateHeadRunner::withinWorkspace(shape, maxRows, prop, reserved);
        size_t const preferred = kernel::LowRankCandidateHeadRunner::workspaceBytes(shape, maxRows, prop);
        if (reserved < preferred)
        {
            LOG_INFO(
                "LowRankCandidateHeadPlugin '%s': the engine reserved %zu workspace bytes, fewer than the %zu this "
                "device prefers; using a selection configuration that needs %zu",
                layerName.c_str(), reserved, preferred, runner.workspaceBytes());
        }
        found = sRunners.emplace(key, runner).first;
    }
    return found->second;
}

} // namespace

PluginFieldCollection LowRankCandidateHeadPluginCreator::mFieldCollection{};
std::vector<PluginField> LowRankCandidateHeadPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(LowRankCandidateHeadPluginCreator);

LowRankCandidateHeadPlugin::LowRankCandidateHeadPlugin(std::string const& name, int32_t numCandidates,
    int32_t hiddenSize, int32_t rank, int32_t vocabSize, int32_t maxRows, int64_t workspaceBytes)
    : mLayerName(name)
    , mNumCandidates(numCandidates)
    , mHiddenSize(hiddenSize)
    , mRank(rank)
    , mVocabSize(vocabSize)
    , mMaxRows(maxRows)
    , mWorkspaceBytes(workspaceBytes)
{
    // The upper bound is the runner's (configurePlugin), so the plugin and the kernels cannot disagree on it.
    if (numCandidates < 1)
    {
        throw std::invalid_argument(
            "LowRankCandidateHeadPlugin: num_candidates must be positive, got " + std::to_string(numCandidates));
    }
}

LowRankCandidateHeadPlugin::LowRankCandidateHeadPlugin(LowRankCandidateHeadPlugin const& other)
    : mLayerName(other.mLayerName)
    , mNamespace(other.mNamespace)
    , mNumCandidates(other.mNumCandidates)
    , mHiddenSize(other.mHiddenSize)
    , mRank(other.mRank)
    , mVocabSize(other.mVocabSize)
    , mMaxRows(other.mMaxRows)
    , mWorkspaceBytes(other.mWorkspaceBytes)
{
}

LowRankCandidateHeadPlugin::~LowRankCandidateHeadPlugin() noexcept = default;

IPluginCapability* LowRankCandidateHeadPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* LowRankCandidateHeadPlugin::clone() noexcept
{
    try
    {
        return new LowRankCandidateHeadPlugin(*this);
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin clone failed: %s", error.what());
        return nullptr;
    }
}

char const* LowRankCandidateHeadPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* LowRankCandidateHeadPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* LowRankCandidateHeadPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LowRankCandidateHeadPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (...)
    {
    }
}

int32_t LowRankCandidateHeadPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t LowRankCandidateHeadPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    for (int32_t i = 0; i < kNB_INPUTS; ++i)
    {
        if (inputTypes[i] != DataType::kHALF)
        {
            LOG_ERROR("LowRankCandidateHeadPlugin '%s': %s must be fp16", mLayerName.c_str(), kINPUT_NAMES[i]);
            return -1;
        }
    }
    outputTypes[kOUT_IDS] = DataType::kINT32;
    outputTypes[kOUT_LOG_PROBS] = DataType::kFLOAT;
    return 0;
}

int32_t LowRankCandidateHeadPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS
        || inputs[kIN_HIDDEN].nbDims != kMATRIX_DIMS)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': expected %d inputs with a 2-D hidden and %d outputs, got %d and %d",
            mLayerName.c_str(), kNB_INPUTS, kNB_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }
    for (int32_t i = 0; i < kNB_OUTPUTS; ++i)
    {
        outputs[i].nbDims = kMATRIX_DIMS;
        outputs[i].d[0] = inputs[kIN_HIDDEN].d[0];
        outputs[i].d[1] = exprBuilder.constant(mNumCandidates);
    }
    return 0;
}

bool LowRankCandidateHeadPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || pos < 0
        || pos >= kNB_INPUTS + kNB_OUTPUTS)
    {
        return false;
    }
    PluginTensorDesc const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos < kNB_INPUTS)
    {
        return desc.type == DataType::kHALF;
    }
    return desc.type == (pos == kNB_INPUTS + kOUT_IDS ? DataType::kINT32 : DataType::kFLOAT);
}

int32_t LowRankCandidateHeadPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': expected %d inputs and %d outputs, got %d and %d",
            mLayerName.c_str(), kNB_INPUTS, kNB_OUTPUTS, nbInputs, nbOutputs);
        return -1;
    }
    try
    {
        std::optional<ResolvedHead> const head = resolveHead(inputs, mNumCandidates, mLayerName);
        if (!head.has_value())
        {
            return -1;
        }
        bool const configured = mMaxRows > 0;
        if (configured
            && (head->shape.hiddenSize != mHiddenSize || head->shape.rank != mRank
                || head->shape.vocabSize != mVocabSize))
        {
            LOG_ERROR("LowRankCandidateHeadPlugin '%s': configured again for a different head (H %d, rank %d, V %d)",
                mLayerName.c_str(), head->shape.hiddenSize, head->shape.rank, head->shape.vocabSize);
            return -1;
        }
        // An instance configured for several optimization profiles serves the largest of them.
        mHiddenSize = head->shape.hiddenSize;
        mRank = head->shape.rank;
        mVocabSize = head->shape.vocabSize;
        mMaxRows = std::max(mMaxRows, head->maxRows);
        mWorkspaceBytes = static_cast<int64_t>(reservedWorkspaceBytes(head->shape));
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': unsupported head: %s", mLayerName.c_str(), error.what());
        return -1;
    }
}

size_t LowRankCandidateHeadPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // configurePlugin rejects the same inputs first, so none of the zero returns below reaches an engine.
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return 0;
    }
    try
    {
        std::optional<ResolvedHead> const head = resolveHead(inputs, mNumCandidates, mLayerName);
        if (!head.has_value())
        {
            return 0;
        }
        return reservedWorkspaceBytes(head->shape);
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': workspace query failed: %s", mLayerName.c_str(), error.what());
        return 0;
    }
}

int32_t LowRankCandidateHeadPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    Dims const& hidden = inputs[kIN_HIDDEN].dims;
    int64_t const rows = hidden.nbDims == kMATRIX_DIMS ? hidden.d[0] : -1;
    bool const valid = rows >= 0 && rows <= mMaxRows && hidden.d[1] == mHiddenSize
        && isMatrix(inputs[kIN_DOWN_PROJ].dims, mRank, mHiddenSize)
        && isMatrix(inputs[kIN_UP_PROJ].dims, mVocabSize, mRank)
        && isMatrix(inputs[kIN_LM_HEAD].dims, mVocabSize, mHiddenSize)
        && isMatrix(outputs[kOUT_IDS].dims, rows, mNumCandidates)
        && isMatrix(outputs[kOUT_LOG_PROBS].dims, rows, mNumCandidates);
    if (!valid)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': runtime hidden %s does not fit the engine (H %d, max rows %d)",
            mLayerName.c_str(), dimsToString(hidden).c_str(), mHiddenSize, mMaxRows);
        return -1;
    }
    return 0;
}

IPluginV3* LowRankCandidateHeadPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

int32_t LowRankCandidateHeadPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr)
    {
        return -1;
    }
    try
    {
        kernel::LowRankCandidateHeadShape shape;
        shape.hiddenSize = mHiddenSize;
        shape.rank = mRank;
        shape.vocabSize = mVocabSize;
        shape.numCandidates = mNumCandidates;
        kernel::LowRankCandidateHeadRunner const& runner = sharedRunner(shape, mMaxRows, mWorkspaceBytes, mLayerName);
        auto const input = [&](int32_t index) {
            return rt::Tensor(const_cast<void*>(inputs[index]), rt::toCoords(inputDesc[index].dims),
                rt::DeviceType::kGPU, fromTrt(inputDesc[index].type));
        };
        auto const output = [&](int32_t index) {
            return rt::Tensor(outputs[index], rt::toCoords(outputDesc[index].dims), rt::DeviceType::kGPU,
                fromTrt(outputDesc[index].type));
        };
        rt::Tensor const hidden = input(kIN_HIDDEN);
        rt::Tensor const downProj = input(kIN_DOWN_PROJ);
        rt::Tensor const upProj = input(kIN_UP_PROJ);
        rt::Tensor const lmHead = input(kIN_LM_HEAD);
        rt::Tensor candidateIds = output(kOUT_IDS);
        rt::Tensor candidateLogProbs = output(kOUT_LOG_PROBS);
        runner.run(hidden, downProj, upProj, lmHead, candidateIds, candidateLogProbs, workspace, stream);
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin '%s': enqueue failed: %s", mLayerName.c_str(), error.what());
        return -1;
    }
}

PluginFieldCollection const* LowRankCandidateHeadPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(kFIELD_NUM_CANDIDATES, &mNumCandidates, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back(kFIELD_HIDDEN_SIZE, &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back(kFIELD_RANK, &mRank, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back(kFIELD_VOCAB_SIZE, &mVocabSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back(kFIELD_MAX_ROWS, &mMaxRows, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back(kFIELD_WORKSPACE_BYTES, &mWorkspaceBytes, PluginFieldType::kINT64, 1);
        mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFieldsToSerialize.fields = mDataToSerialize.data();
        return &mFieldsToSerialize;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin serialization failed: %s", error.what());
        return nullptr;
    }
}

LowRankCandidateHeadPluginCreator::LowRankCandidateHeadPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    // Build-phase attributes only; the resolved fields reach createPlugin from the serialized engine.
    mPluginAttributes = {{kFIELD_NUM_CANDIDATES, nullptr, PluginFieldType::kINT32, 1}};
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* LowRankCandidateHeadPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* LowRankCandidateHeadPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* LowRankCandidateHeadPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* LowRankCandidateHeadPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LowRankCandidateHeadPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (...)
    {
    }
}

IPluginV3* LowRankCandidateHeadPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept
{
    try
    {
        int32_t const numCandidates = requireInt32Field(fields, kFIELD_NUM_CANDIDATES);
        std::unique_ptr<LowRankCandidateHeadPlugin> plugin;
        if (phase == TensorRTPhase::kRUNTIME)
        {
            plugin = std::make_unique<LowRankCandidateHeadPlugin>(name == nullptr ? "" : name, numCandidates,
                requireInt32Field(fields, kFIELD_HIDDEN_SIZE), requireInt32Field(fields, kFIELD_RANK),
                requireInt32Field(fields, kFIELD_VOCAB_SIZE), requireInt32Field(fields, kFIELD_MAX_ROWS),
                requireIntegerField(fields, kFIELD_WORKSPACE_BYTES));
        }
        else
        {
            plugin = std::make_unique<LowRankCandidateHeadPlugin>(
                name == nullptr ? "" : name, numCandidates, 0, 0, 0, 0, 0);
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LowRankCandidateHeadPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
