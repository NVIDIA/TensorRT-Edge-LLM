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

#include "runtime/pi05Policy.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pi05Bindings.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

namespace
{

// The quantile arrays Pi05Policy normalizes the state and unnormalizes the action
// chunk with. Absent is not an error here; the coverage check below is what rejects it.
std::vector<float> readVector(nlohmann::json const& node, char const* key)
{
    if (!node.contains(key) || node.at(key).is_null())
    {
        return {};
    }
    return node.at(key).get<std::vector<float>>();
}

NormStats readStats(nlohmann::json const& root, char const* key)
{
    if (!root.contains(key))
    {
        return {};
    }
    nlohmann::json const& node = root.at(key);
    NormStats stats;
    stats.q01 = readVector(node, "q01");
    stats.q99 = readVector(node, "q99");
    return stats;
}

//! Locate norm_stats.json anywhere under assets/; openpi nests it two levels deep.
//! Every depth is collected before choosing, so a top-level file cannot be returned
//! without seeing the nested one it would be ambiguous with.
std::filesystem::path findNormStats(std::filesystem::path const& assetsDir)
{
    if (!std::filesystem::is_directory(assetsDir))
    {
        return {};
    }
    std::vector<std::filesystem::path> found;
    for (auto const& entry : std::filesystem::recursive_directory_iterator(assetsDir))
    {
        if (entry.is_regular_file() && entry.path().filename() == "norm_stats.json")
        {
            found.push_back(entry.path());
        }
    }
    if (found.size() > 1)
    {
        // Several embodiments staged together: the caller must say which one, since
        // picking the wrong statistics silently mis-scales every command.
        std::sort(found.begin(), found.end());
        std::string listed;
        for (auto const& path : found)
        {
            listed += "\n  " + std::filesystem::relative(path, assetsDir).string();
        }
        throw std::runtime_error("Multiple norm_stats.json under " + assetsDir.string()
            + "; stage exactly one embodiment's assets for this engine directory:" + listed);
    }
    return found.empty() ? std::filesystem::path{} : found.front();
}

nlohmann::json readJson(std::filesystem::path const& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open " + path.string());
    }
    nlohmann::json doc;
    file >> doc;
    return doc;
}

//! Enough statistics of the right kind to cover \p dims components.
bool statsUsable(NormStats const& stats, int32_t dims)
{
    return std::min(stats.q01.size(), stats.q99.size()) >= static_cast<size_t>(dims);
}

//! Python ``str.strip()`` plus the two substitutions openpi applies to a task
//! string; the underscore rule is why dataset task ids tokenize as words.
std::string cleanTask(std::string const& task)
{
    static constexpr char kWhitespace[] = " \t\n\r\f\v";
    size_t const begin = task.find_first_not_of(kWhitespace);
    if (begin == std::string::npos)
    {
        return {};
    }
    std::string cleaned = task.substr(begin, task.find_last_not_of(kWhitespace) - begin + 1);
    std::replace(cleaned.begin(), cleaned.end(), '_', ' ');
    std::replace(cleaned.begin(), cleaned.end(), '\n', ' ');
    return cleaned;
}

//! ``np.digitize(x, np.linspace(-1, 1, bins + 1)[:-1]) - 1``: the count of bin
//! edges at or below \p x, minus one. Values under the first edge yield -1, and
//! the reference writes that -1 into the prompt, so it is reproduced here.
int32_t digitize(float value, int32_t bins)
{
    double const step = 2.0 / static_cast<double>(bins);
    int32_t count = 0;
    for (int32_t i = 0; i < bins; ++i)
    {
        if (-1.0 + static_cast<double>(i) * step <= static_cast<double>(value))
        {
            ++count;
        }
    }
    return count - 1;
}

void requireStateDim(size_t size, int32_t dim)
{
    if (static_cast<int32_t>(size) != dim)
    {
        throw std::invalid_argument(
            "pi0.5 state has " + std::to_string(size) + " dims, contract declares " + std::to_string(dim));
    }
}

//! openpi's quantile map. The epsilon is part of the reference formula, not a guard
//! against a degenerate spread, so it is added whatever q99 - q01 comes to.
float normalizeQuantile(float value, float q01, float q99)
{
    return 2.0F * (value - q01) / (q99 - q01 + kQuantileEpsilon) - 1.0F;
}

float unnormalizeQuantile(float normalized, float q01, float q99)
{
    return (normalized + 1.0F) * 0.5F * (q99 - q01 + kQuantileEpsilon) + q01;
}

//! One row per openpi configuration this runtime implements. A manifest pairing a
//! configuration with another embodiment's adapter emits plausible but wrong commands,
//! so the pairing is fixed here rather than taken from the bundle.
struct PolicyConfigShape
{
    char const* policyConfig;
    char const* adapterName;
    Pi05Adapter adapter;
    bool discreteStateInput;
};

constexpr PolicyConfigShape kPolicyConfigs[]{
    {"pi05_libero", "libero", Pi05Adapter::kLibero, false},
    {"pi05_droid", "droid", Pi05Adapter::kDroid, true},
    {"pi05_aloha", "aloha", Pi05Adapter::kAloha, true},
};

std::string knownPolicyConfigs()
{
    std::string names;
    for (auto const& row : kPolicyConfigs)
    {
        names += (names.empty() ? "" : ", ");
        names += row.policyConfig;
    }
    return names;
}

Pi05Adapter readAdapter(nlohmann::json const& contract, std::string const& path)
{
    int32_t const version = contract.value("contract_version", 0);
    if (version != kContractVersion)
    {
        throw std::runtime_error("pi0.5 " + path + " was exported against contract version " + std::to_string(version)
            + " but this runtime implements " + std::to_string(kContractVersion)
            + "; re-export it with tensorrt-edgellm-export");
    }
    std::string const family = contract.value("model_family", std::string{});
    if (family != "pi05")
    {
        throw std::runtime_error("pi0.5 " + path + " declares model family '" + family + "', not pi05");
    }
    std::string const name = contract.value("adapter", std::string{});
    auto const found = std::find_if(std::begin(kPolicyConfigs), std::end(kPolicyConfigs),
        [&name](PolicyConfigShape const& row) { return name == row.adapterName; });
    if (found == std::end(kPolicyConfigs))
    {
        std::string known;
        for (auto const& row : kPolicyConfigs)
        {
            known += (known.empty() ? "" : ", ") + std::string(row.adapterName);
        }
        throw std::runtime_error("pi0.5 " + path + " declares adapter '" + name + "'; this runtime implements " + known
            + ". Its observation and action processing is embodiment-specific");
    }
    return found->adapter;
}

std::vector<Pi05CameraSlot> readCameraSlots(nlohmann::json const& cameras)
{
    // An export from before the named-slot contract carries "present"/"empty" under the same
    // version, so say what to do rather than letting the key lookup throw.
    if (!cameras.contains("slots"))
    {
        throw std::runtime_error(
            "pi0.5 policy.json declares no camera slots; re-export the "
            "checkpoint with tensorrt-edgellm-export");
    }
    std::vector<Pi05CameraSlot> slots;
    for (auto const& slot : cameras.at("slots"))
    {
        slots.push_back(Pi05CameraSlot{slot.at("name").get<std::string>(), slot.at("required").get<bool>()});
    }
    return slots;
}

//! openpi's Aloha conversions, transcribed from ``policies/aloha_policy.py``. The
//! constants are the Aloha runtime's own gripper limits and the Interbotix linkage's.
constexpr float kAlohaJointFlip[]{
    1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
constexpr size_t kAlohaGripperDims[]{6U, 13U};

float gripperToAngular(float value)
{
    float const linear = value * (0.05800F - 0.01844F) + 0.01844F;
    constexpr float kArmLength = 0.036F;
    constexpr float kHornRadius = 0.022F;
    float const cosine
        = (kHornRadius * kHornRadius + linear * linear - kArmLength * kArmLength) / (2.0F * kHornRadius * linear);
    float const radians = std::asin(std::clamp(cosine, -1.0F, 1.0F));
    return (radians - 0.5476F) / (1.6296F - 0.5476F);
}

float gripperFromAngular(float value)
{
    return (value + 0.5476F + 0.6213F) / (1.4910F + 0.6213F);
}

} // namespace

bool Pi05Policy::available(std::string const& engineDir)
{
    return std::filesystem::exists(std::filesystem::path(engineDir) / "policy.json");
}

void Pi05Policy::loadContract(std::string const& engineDir)
{
    std::filesystem::path const root(engineDir);
    std::filesystem::path const contractPath = root / "policy.json";
    if (!std::filesystem::exists(contractPath))
    {
        throw std::runtime_error("pi0.5 policy.json not found under " + engineDir
            + "; re-export the checkpoint so the observation contract is staged with the engines");
    }
    nlohmann::json const contract = readJson(contractPath);
    mContract.adapter = readAdapter(contract, contractPath.string());
    nlohmann::json const& state = contract.at("state");
    nlohmann::json const& action = contract.at("action");
    mContract.policyConfig = contract.value("policy_config", std::string{});
    mContract.checkpointFingerprint = contract.value("checkpoint_fingerprint", std::string{});
    mContract.exportId = contract.value(kExportIdKey, std::string{});
    if (mContract.exportId.empty())
    {
        throw std::runtime_error("pi0.5 " + contractPath.string() + " carries no " + kExportIdKey
            + "; re-export the checkpoint so the engines staged beside it can be checked against it");
    }
    mContract.stateDim = state.at("dim").get<int32_t>();
    mContract.robotActionDim = action.at("dim").get<int32_t>();
    mContract.modelActionDim = action.value("max_dim", mContract.robotActionDim);
    mContract.actionHorizon = action.at("horizon").get<int32_t>();
    mContract.numBins = state.value("num_bins", kStateBins);
    if (mContract.numBins != kStateBins)
    {
        throw std::runtime_error("pi0.5 policy.json declares num_bins " + std::to_string(mContract.numBins)
            + " but openpi discretizes the state into " + std::to_string(kStateBins) + "; re-export the checkpoint");
    }
    mContract.discreteStateInput = contract.at("discrete_state_input").get<bool>();
    auto const* shape = std::find_if(std::begin(kPolicyConfigs), std::end(kPolicyConfigs),
        [this](PolicyConfigShape const& row) { return mContract.policyConfig == row.policyConfig; });
    if (shape == std::end(kPolicyConfigs))
    {
        throw std::runtime_error("pi0.5 policy.json declares policy_config '" + mContract.policyConfig
            + "', which this runtime does not implement; it knows " + knownPolicyConfigs());
    }
    std::string const declaredAdapter = contract.value("adapter", std::string{});
    if (declaredAdapter != shape->adapterName || mContract.discreteStateInput != shape->discreteStateInput)
    {
        throw std::runtime_error("pi0.5 policy.json pairs policy_config '" + mContract.policyConfig + "' with adapter '"
            + declaredAdapter + "' and discrete_state_input " + (mContract.discreteStateInput ? "true" : "false")
            + "; openpi defines it as adapter '" + shape->adapterName + "' and "
            + (shape->discreteStateInput ? "true" : "false") + "; re-export the checkpoint");
    }
    mContract.cameras = readCameraSlots(contract.at("cameras"));
    mContract.ignoredCameras = contract.at("cameras").value("ignored", std::vector<std::string>{});
    if (mContract.cameras.empty() || !mContract.cameras.front().required)
    {
        throw std::runtime_error("pi0.5 policy.json declares no required camera; the prefix cannot be assembled");
    }
    if (static_cast<int32_t>(mContract.cameras.size()) > kMaxCameraSlots)
    {
        throw std::runtime_error("pi0.5 policy.json declares more cameras than the model's "
            + std::to_string(kMaxCameraSlots) + " image slots");
    }
    if (mContract.robotActionDim > mContract.modelActionDim)
    {
        throw std::runtime_error("pi0.5 policy.json declares a robot action dim wider than the model's");
    }
    auto const alohaDims = static_cast<int32_t>(std::size(kAlohaJointFlip));
    if (mContract.adapter == Pi05Adapter::kAloha
        && (mContract.stateDim != alohaDims || mContract.robotActionDim != alohaDims))
    {
        throw std::runtime_error("pi0.5 the aloha adapter converts " + std::to_string(alohaDims)
            + " joints; this policy.json declares a different state or action width");
    }
    auto const resolution = contract.at("image_resolution").get<std::vector<int32_t>>();
    if (resolution.size() != 2)
    {
        throw std::runtime_error("pi0.5 policy.json image_resolution must be [height, width]");
    }
    mContract.imageHeight = resolution[0];
    mContract.imageWidth = resolution[1];
    // Required rather than defaulted: the exporter always writes these, so a manifest
    // missing one is not an older export but a damaged or hand-edited bundle.
    nlohmann::json const& tokenizerContract = contract.at("tokenizer");
    mContract.maxTokenLen = tokenizerContract.at("max_length").get<int32_t>();
    mContract.tokenizerVocabSize = tokenizerContract.at("vocab_size").get<int32_t>();
    if (mContract.maxTokenLen <= 0 || mContract.tokenizerVocabSize <= 0)
    {
        throw std::runtime_error("pi0.5 policy.json declares tokenizer max_length "
            + std::to_string(mContract.maxTokenLen) + " and vocab_size " + std::to_string(mContract.tokenizerVocabSize)
            + "; both must be positive");
    }
    // tokenize() hardcodes this pair, so a manifest asking for anything else would be
    // silently ignored and produce a prompt the checkpoint was not trained on.
    if (!tokenizerContract.at("add_bos").get<bool>() || tokenizerContract.at("add_eos").get<bool>())
    {
        throw std::runtime_error("pi0.5 policy.json asks for add_bos="
            + std::string(tokenizerContract.at("add_bos").get<bool>() ? "true" : "false")
            + " add_eos=" + (tokenizerContract.at("add_eos").get<bool>() ? "true" : "false")
            + "; this runtime encodes prompts with add_bos=true add_eos=false");
    }

    std::filesystem::path const statsPath = findNormStats(root / "assets");
    if (statsPath.empty())
    {
        throw std::runtime_error("pi0.5 norm_stats.json not found under " + (root / "assets").string()
            + "; the state cannot be normalized and actions cannot be converted to robot units without it");
    }
    nlohmann::json const doc = readJson(statsPath);
    // openpi serializes as {"norm_stats": {"<key>": {...}}}.
    nlohmann::json const& stats = doc.contains("norm_stats") ? doc.at("norm_stats") : doc;
    mStateStats = readStats(stats, "state");
    mActionStats = readStats(stats, "actions");
    if (!statsUsable(mStateStats, mContract.stateDim) || !statsUsable(mActionStats, mContract.robotActionDim))
    {
        throw std::runtime_error(
            "pi0.5 norm_stats.json has no q01/q99 covering the declared state/action dims: " + statsPath.string());
    }
    LOG_INFO("pi0.5 policy: %s, checkpoint %s", mContract.policyConfig.c_str(),
        mContract.checkpointFingerprint.empty() ? "unrecorded" : mContract.checkpointFingerprint.c_str());
    LOG_INFO("pi0.5 policy: state %dd, action %dd of %d, horizon %d, %dx%d, %s", mContract.stateDim,
        mContract.robotActionDim, mContract.modelActionDim, mContract.actionHorizon, mContract.imageHeight,
        mContract.imageWidth, statsPath.string().c_str());
    LOG_INFO("pi0.5 cameras: %s", cameraOrderSummary(mContract.cameras).c_str());
}

Pi05Policy::Pi05Policy(std::string const& engineDir)
    : mEngineDir(engineDir)
{
    loadContract(engineDir);
}

Pi05Policy::Pi05Policy(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
    , mStream(stream)
{
    loadContract(engineDir);
    mRuntime = std::make_unique<Pi05Runtime>(engineDir, stream);
    Pi05PolicyConfig const& cfg = mRuntime->getConfig();
    // The runtime only ties the components to each other, so engines rebuilt over an older
    // bundle keep that bundle's camera order, prompt contract and normalization statistics.
    if (mContract.exportId != cfg.exportId)
    {
        throw std::runtime_error("pi0.5 engines under " + engineDir + " are from export " + cfg.exportId
            + " but the policy.json staged beside them is from export " + mContract.exportId
            + "; rebuild the bundle from one export");
    }
    if (mContract.imageHeight != cfg.imageSize || mContract.imageWidth != cfg.imageSize)
    {
        throw std::runtime_error("pi0.5 policy.json image_resolution disagrees with the visual engine contract");
    }
    // max_dim is the padded width the engine emits; dim is the embodiment's own prefix of it.
    if (mContract.modelActionDim != cfg.actionDim || mContract.actionHorizon != cfg.actionHorizon)
    {
        throw std::runtime_error("pi0.5 policy.json declares a [" + std::to_string(mContract.actionHorizon) + ", "
            + std::to_string(mContract.modelActionDim) + "] action chunk but the action engine emits ["
            + std::to_string(cfg.actionHorizon) + ", " + std::to_string(cfg.actionDim) + "]");
    }
    // A bundle whose declared slots and prompt cannot fit the prefix the engines were built
    // for is unrunnable, so it is refused here rather than at the first request.
    int64_t const declaredPrefix = numCameras() * static_cast<int64_t>(cfg.numImageTokens) + mContract.maxTokenLen;
    if (declaredPrefix > cfg.maxPrefixLen)
    {
        throw std::runtime_error("pi0.5 policy.json declares " + std::to_string(numCameras()) + " camera slots and "
            + std::to_string(mContract.maxTokenLen) + " prompt tokens, needing " + std::to_string(declaredPrefix)
            + " prefix slots, but the engines were built for " + std::to_string(cfg.maxPrefixLen)
            + "; re-export the checkpoint");
    }
    // Sized once for every slot the contract declares, not per call; a request that leaves an
    // optional one out reshapes it down. The host half is pinned because the upload is async.
    std::vector<int64_t> const shape{numCameras(), 3, cfg.imageSize, cfg.imageSize};
    mPixelValues = rt::Tensor(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "pi05::pixelValues");
    mPixelValuesHost = rt::Tensor(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF, "pi05::pixelValuesHost");
}

Pi05Policy::~Pi05Policy() = default;

Pi05Runtime& Pi05Policy::runtime()
{
    if (mRuntime == nullptr)
    {
        throw std::runtime_error("pi0.5 policy was built from " + mEngineDir
            + " without engines; construct it with a stream to run inference");
    }
    return *mRuntime;
}

std::vector<Pi05CameraView const*> Pi05Policy::resolveActiveViews(std::vector<Pi05CameraView> const& views) const
{
    for (size_t v = 0; v < views.size(); ++v)
    {
        Pi05CameraView const& view = views[v];
        bool const hasRgb = view.rgb != nullptr;
        if (hasRgb == !view.imagePath.empty())
        {
            throw std::invalid_argument("pi0.5 camera view " + std::to_string(v)
                + (hasRgb ? " sets both a decoded frame and an image path"
                          : " sets neither a decoded frame nor an image path")
                + "; a view carries exactly one source");
        }
        if (hasRgb && (view.height <= 0 || view.width <= 0))
        {
            throw std::invalid_argument("pi0.5 camera view " + std::to_string(v)
                + " supplies a decoded frame with a non-positive height or width; both are needed to resize it");
        }
    }

    auto const kUnfilled = std::numeric_limits<size_t>::max();
    std::vector<size_t> slots(mContract.cameras.size(), kUnfilled);
    bool const anyNamed
        = std::any_of(views.begin(), views.end(), [](Pi05CameraView const& v) { return !v.name.empty(); });
    if (!anyNamed)
    {
        if (views.size() > slots.size())
        {
            throw std::invalid_argument("pi0.5 request supplies " + std::to_string(views.size())
                + " unnamed view(s) for " + std::to_string(slots.size())
                + " slot(s), in order: " + cameraOrderSummary(mContract.cameras));
        }
        std::iota(slots.begin(), slots.begin() + static_cast<ptrdiff_t>(views.size()), size_t{0});
    }
    for (size_t v = 0; anyNamed && v < views.size(); ++v)
    {
        std::string const& name = views[v].name;
        if (name.empty())
        {
            throw std::invalid_argument(
                "pi0.5 request names some camera views but not all; name every view or none, in contract order: "
                + cameraOrderSummary(mContract.cameras));
        }
        if (std::find(mContract.ignoredCameras.begin(), mContract.ignoredCameras.end(), name)
            != mContract.ignoredCameras.end())
        {
            // openpi's input transform accepts the name and drops the frame, so the prefix
            // never sees it and the request is not an error.
            LOG_INFO("pi0.5 ignoring camera %s, as %s does", name.c_str(), mContract.policyConfig.c_str());
            continue;
        }
        auto const slot = std::find_if(mContract.cameras.begin(), mContract.cameras.end(),
            [&name](Pi05CameraSlot const& s) { return s.name == name; });
        if (slot == mContract.cameras.end())
        {
            throw std::invalid_argument("pi0.5 contract has no camera named " + name + "; it declares "
                + cameraOrderSummary(mContract.cameras));
        }
        auto const index = static_cast<size_t>(std::distance(mContract.cameras.begin(), slot));
        if (slots[index] != kUnfilled)
        {
            throw std::invalid_argument("pi0.5 request supplies camera " + name + " twice");
        }
        slots[index] = v;
    }

    std::vector<Pi05CameraView const*> ordered;
    for (size_t i = 0; i < slots.size(); ++i)
    {
        if (slots[i] == kUnfilled)
        {
            if (mContract.cameras[i].required)
            {
                throw std::invalid_argument("pi0.5 request supplies no image for required camera "
                    + mContract.cameras[i].name + "; the contract declares " + cameraOrderSummary(mContract.cameras));
            }
            continue;
        }
        ordered.push_back(&views[slots[i]]);
    }
    return ordered;
}

double Pi05Policy::stageOneView(unsigned char const* rgb, int32_t srcH, int32_t srcW, size_t viewIdx)
{
    using Clock = std::chrono::steady_clock;
    auto const viewElems = static_cast<size_t>(3) * mContract.imageHeight * mContract.imageWidth;
    mPlanarView.resize(viewElems);
    auto const start = Clock::now();
    resizeWithPad(rgb, srcH, srcW, mContract.imageHeight, mContract.imageWidth, mPlanarView.data());
    auto* rows = mPixelValuesHost.dataPointer<__half>();
    for (size_t i = 0; i < viewElems; ++i)
    {
        rows[viewIdx * viewElems + i] = __float2half(mPlanarView[i]);
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void Pi05Policy::stagePixelValues(std::vector<Pi05CameraView const*> const& ordered, Pi05ObservationTimes& times)
{
    using Clock = std::chrono::steady_clock;
    auto const viewElems = static_cast<size_t>(3) * mContract.imageHeight * mContract.imageWidth;
    for (size_t v = 0; v < ordered.size(); ++v)
    {
        Pi05CameraView const& view = *ordered[v];
        if (view.rgb != nullptr)
        {
            times.resizeMs += stageOneView(view.rgb, view.height, view.width, v);
            continue;
        }
        auto const decodeStart = Clock::now();
        double preprocessMs{0.0};
        int64_t srcH{0};
        int64_t srcW{0};
        {
            // Scoped so the decoded frame's pinned host buffer is freed, and charged, here:
            // allocating and pinning it is part of the cost of being handed a file path.
            rt::imageUtils::ImageData const image = rt::imageUtils::loadRgbImageFromFile(view.imagePath);
            srcH = image.height;
            srcW = image.width;
            preprocessMs = stageOneView(image.data(), static_cast<int32_t>(srcH), static_cast<int32_t>(srcW), v);
        }
        times.decodeMs += std::chrono::duration<double, std::milli>(Clock::now() - decodeStart).count() - preprocessMs;
        times.resizeMs += preprocessMs;
        LOG_INFO("Loaded %s (%ldx%ld) -> %dx%d", view.imagePath.c_str(), srcW, srcH, mContract.imageWidth,
            mContract.imageHeight);
    }
    // No sync: the upload and the engines share mStream, and the previous request
    // drained it before returning, so the pinned staging is free again.
    CUDA_CHECK(cudaMemcpyAsync(mPixelValues.rawPointer(), mPixelValuesHost.dataPointer<__half>(),
        ordered.size() * viewElems * sizeof(__half), cudaMemcpyHostToDevice, mStream));
}

std::vector<int32_t> Pi05Policy::tokenize(std::string const& prompt)
{
    if (mTokenizer == nullptr)
    {
        std::filesystem::path const dir = std::filesystem::path(mEngineDir) / "text_tokenizer";
        auto loaded = std::make_unique<tokenizer::Tokenizer>();
        if (!loaded->loadFromHF(dir))
        {
            throw std::runtime_error("Failed to load the PaliGemma tokenizer from " + dir.string());
        }
        // Not compared against the manifest's vocab_size: that records the SentencePiece
        // piece count, while the converted HF tokenizer carries four more for the special
        // tokens. An id the embedding table cannot hold is caught in assemblePrefix.
        if (loaded->getNumVocab() < mContract.tokenizerVocabSize)
        {
            throw std::runtime_error("pi0.5 text_tokenizer/ holds " + std::to_string(loaded->getNumVocab())
                + " tokens, fewer than the " + std::to_string(mContract.tokenizerVocabSize)
                + " pieces policy.json was exported with; the staged tokenizer is not that one");
        }
        mTokenizer = std::move(loaded);
    }
    std::vector<int32_t> ids = mTokenizer->encode(prompt, /*addBos=*/true, /*addEos=*/false);
    if (!mContract.discreteStateInput)
    {
        // openpi encodes the "\n" separately from the task text, as the start-of-answer token.
        std::vector<int32_t> const newline = mTokenizer->encode("\n", /*addBos=*/false, /*addEos=*/false);
        ids.insert(ids.end(), newline.begin(), newline.end());
    }
    if (static_cast<int32_t>(ids.size()) > mContract.maxTokenLen)
    {
        LOG_WARNING(
            "Prompt tokenizes to %zu tokens, truncating to the contract's %d", ids.size(), mContract.maxTokenLen);
        ids.resize(static_cast<size_t>(mContract.maxTokenLen));
    }
    return ids;
}

Pi05ActionChunk Pi05Policy::inferTensors(
    rt::Tensor const& pixelValues, std::vector<int32_t> const& tokenIds, int32_t batch)
{
    validatePixelValues(pixelValues);
    if (batch < 1)
    {
        throw std::invalid_argument("pi0.5 request batch must be at least 1");
    }
    Pi05Runtime& engines = runtime();
    using Clock = std::chrono::steady_clock;

    Pi05ActionChunk chunk;
    chunk.tokenIds = tokenIds;
    chunk.batch = batch;
    auto const engineStart = Clock::now();
    chunk.normalizedActions = engines.generate(pixelValues, tokenIds, batch);
    chunk.timings.engineMs = std::chrono::duration<double, std::milli>(Clock::now() - engineStart).count();
    if (chunk.normalizedActions.empty())
    {
        throw std::runtime_error("pi0.5 policy produced no actions");
    }

    Pi05PolicyConfig const& cfg = engines.getConfig();
    chunk.horizon = cfg.actionHorizon;
    chunk.modelActionDim = cfg.actionDim;
    chunk.timings.stages = engines.getStageTimes();
    chunk.timings.policyMs = std::chrono::duration<double, std::milli>(Clock::now() - engineStart).count();
    return chunk;
}

Pi05ActionChunk Pi05Policy::infer(Pi05Observation const& observation)
{
    (void) runtime(); // fail before any of the observation work when there are no engines
    using Clock = std::chrono::steady_clock;
    Pi05ObservationTimes times;
    auto const observationStart = Clock::now();
    std::vector<Pi05CameraView const*> const ordered = resolveActiveViews(observation.cameras);
    if (observation.batch < 1)
    {
        throw std::runtime_error(
            "pi0.5 request asks for batch " + std::to_string(observation.batch) + "; it must be at least 1");
    }
    auto const restStart = Clock::now();
    std::vector<float> const adapted = adaptInputState(observation.state);
    std::string prompt = buildPrompt(observation.task, adapted);
    std::vector<int32_t> const tokenIds = tokenize(prompt);
    // An omitted optional slot leaves the staging longer than the request: bind the
    // active views alone, since the prefix carries no mask to drop the rest with.
    if (!mPixelValues.reshape(
            rt::Coords{static_cast<int64_t>(ordered.size()), 3, mContract.imageHeight, mContract.imageWidth}))
    {
        throw std::runtime_error("pi0.5 could not bind the request's active views");
    }
    times.restMs = std::chrono::duration<double, std::milli>(Clock::now() - restStart).count();
    // Last, because it starts an async copy out of a pinned member: a throw after this
    // point would leave that copy in flight for the next request to overwrite. The
    // engine-side batch bound is still inferTensors', but the sign check is cheap here.
    stagePixelValues(ordered, times);
    auto const observationEnd = Clock::now();
    times.totalMs = std::chrono::duration<double, std::milli>(observationEnd - observationStart).count();

    Pi05ActionChunk chunk = inferTensors(mPixelValues, tokenIds, observation.batch);
    auto const chunkElems = static_cast<size_t>(chunk.horizon) * chunk.modelActionDim;
    chunk.robotActions = postprocessActions(std::vector<float>(chunk.normalizedActions.begin(),
                                                chunk.normalizedActions.begin() + static_cast<ptrdiff_t>(chunkElems)),
        chunk.horizon, chunk.modelActionDim, adapted);
    chunk.prompt = std::move(prompt);
    chunk.timings.observation = times;
    // Widen inferTensors' window to the front end this call also paid for.
    chunk.timings.policyMs = std::chrono::duration<double, std::milli>(Clock::now() - observationStart).count();
    return chunk;
}

void Pi05Policy::validateViewCount(int32_t views) const
{
    auto const required = static_cast<int32_t>(std::count_if(
        mContract.cameras.begin(), mContract.cameras.end(), [](Pi05CameraSlot const& slot) { return slot.required; }));
    if (views < required || views > kMaxCameraSlots)
    {
        throw std::invalid_argument("pi0.5 request has " + std::to_string(views) + " view(s); the contract needs "
            + std::to_string(required) + " and the model carries " + std::to_string(kMaxCameraSlots)
            + " image slot(s), in order: " + cameraOrderSummary(mContract.cameras));
    }
}

void Pi05Policy::validatePixelValues(rt::Tensor const& pixelValues) const
{
    rt::Coords const shape = pixelValues.getShape();
    if (shape.getNumDims() != 4 || shape[1] != 3 || shape[2] != mContract.imageHeight
        || shape[3] != mContract.imageWidth)
    {
        throw std::invalid_argument("pi0.5 pixel values are " + shape.formatString()
            + "; the contract declares [views, 3, " + std::to_string(mContract.imageHeight) + ", "
            + std::to_string(mContract.imageWidth) + "]");
    }
    validateViewCount(static_cast<int32_t>(shape[0]));
    if (pixelValues.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::invalid_argument(
            "pi0.5 pixel values must be FLOAT16; the visual engine reads them as fp16 "
            "whatever the tensor declares");
    }
    if (pixelValues.getDeviceType() != rt::DeviceType::kGPU)
    {
        throw std::invalid_argument(
            "pi0.5 pixel values must be a device tensor; the visual engine is handed "
            "their address and never copies them");
    }
}

std::vector<float> Pi05Policy::adaptInputState(std::vector<float> const& state) const
{
    requireStateDim(state.size(), mContract.stateDim);
    if (mContract.adapter != Pi05Adapter::kAloha)
    {
        return state;
    }
    std::vector<float> adapted = state;
    for (size_t d = 0; d < adapted.size(); ++d)
    {
        adapted[d] *= kAlohaJointFlip[d];
    }
    for (size_t d : kAlohaGripperDims)
    {
        adapted[d] = gripperToAngular(adapted[d]);
    }
    return adapted;
}

std::string Pi05Policy::buildPrompt(std::string const& task, std::vector<float> const& adapted) const
{
    requireStateDim(adapted.size(), mContract.stateDim);
    if (!mContract.discreteStateInput)
    {
        // openpi's TokenizePrompt with discrete_state_input False: the task text alone,
        // tokenize() appending the newline. The state is validated above but never used.
        return cleanTask(task);
    }
    std::ostringstream bins;
    for (int32_t d = 0; d < mContract.stateDim; ++d)
    {
        float const normalized = normalizeQuantile(adapted[static_cast<size_t>(d)],
            mStateStats.q01[static_cast<size_t>(d)], mStateStats.q99[static_cast<size_t>(d)]);
        int32_t const bin = digitize(normalized, mContract.numBins);
        if (bin < 0)
        {
            LOG_WARNING(
                "pi0.5 state dim %d normalizes to %.4f, below the discretization range; the prompt "
                "carries the reference's out-of-range bin -1",
                d, normalized);
        }
        bins << (d ? " " : "") << bin;
    }
    return "Task: " + cleanTask(task) + ", State: " + bins.str() + ";\nAction: ";
}

std::vector<float> Pi05Policy::unnormalizeActions(
    std::vector<float> const& normalized, int32_t horizon, int32_t actionDim) const
{
    if (mContract.robotActionDim > actionDim)
    {
        throw std::invalid_argument("pi0.5 robot action dim exceeds the model action dim");
    }
    size_t const expected = static_cast<size_t>(horizon) * static_cast<size_t>(actionDim);
    if (normalized.size() != expected)
    {
        throw std::invalid_argument("pi0.5 normalized chunk size does not match horizon * actionDim");
    }
    int32_t const robotDim = mContract.robotActionDim;
    std::vector<float> out(static_cast<size_t>(horizon) * static_cast<size_t>(robotDim));
    for (int32_t t = 0; t < horizon; ++t)
    {
        for (int32_t d = 0; d < robotDim; ++d)
        {
            out[static_cast<size_t>(t) * robotDim + d]
                = unnormalizeQuantile(normalized[static_cast<size_t>(t) * actionDim + d],
                    mActionStats.q01[static_cast<size_t>(d)], mActionStats.q99[static_cast<size_t>(d)]);
        }
    }
    return out;
}

std::vector<float> Pi05Policy::postprocessActions(
    std::vector<float> const& normalized, int32_t horizon, int32_t actionDim, std::vector<float> const& adapted) const
{
    std::vector<float> out = unnormalizeActions(normalized, horizon, actionDim);
    if (mContract.adapter != Pi05Adapter::kAloha)
    {
        return out;
    }
    requireStateDim(adapted.size(), mContract.stateDim);
    int32_t const robotDim = mContract.robotActionDim;
    for (int32_t t = 0; t < horizon; ++t)
    {
        float* const row = out.data() + static_cast<size_t>(t) * robotDim;
        for (int32_t d = 0; d < robotDim; ++d)
        {
            // openpi's delta mask covers the arm joints only; both grippers are absolute.
            bool const gripper
                = std::find(std::begin(kAlohaGripperDims), std::end(kAlohaGripperDims), static_cast<size_t>(d))
                != std::end(kAlohaGripperDims);
            row[d] = (gripper ? row[d] : row[d] + adapted[static_cast<size_t>(d)]) * kAlohaJointFlip[d];
        }
        for (size_t d : kAlohaGripperDims)
        {
            row[d] = gripperFromAngular(row[d]);
        }
    }
    return out;
}

std::string cameraOrderSummary(std::vector<Pi05CameraSlot> const& slots)
{
    std::ostringstream summary;
    for (size_t i = 0; i < slots.size(); ++i)
    {
        summary << (i ? ", " : "") << i << '=' << slots[i].name << (slots[i].required ? "" : " (optional)");
    }
    return summary.str();
}

namespace
{
//! Pillow's triangle resampler, one axis. The support widens with the downscale factor,
//! and the 8-bit passes accumulate in fixed point; a float accumulation lands a level off.
constexpr int32_t kPrecisionBits = 32 - 8 - 2;

struct ResampleAxis
{
    std::vector<int32_t> weights; //!< \p taps entries per output pixel, scaled by 2^kPrecisionBits
    std::vector<int32_t> starts;
    int32_t taps{0};
};

ResampleAxis makeResampleAxis(int32_t srcSize, int32_t dstSize)
{
    double const scale = static_cast<double>(srcSize) / dstSize;
    double const support = std::max(scale, 1.0);
    ResampleAxis axis;
    axis.taps = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    axis.weights.assign(static_cast<size_t>(dstSize) * axis.taps, 0);
    axis.starts.assign(static_cast<size_t>(dstSize), 0);
    for (int32_t i = 0; i < dstSize; ++i)
    {
        double const center = (static_cast<double>(i) + 0.5) * scale;
        int32_t const begin = std::max(0, static_cast<int32_t>(center - support + 0.5));
        int32_t const end = std::min(srcSize, static_cast<int32_t>(center + support + 0.5));
        axis.starts[static_cast<size_t>(i)] = begin;
        auto const tapAt = [&](int32_t j) {
            return std::max(0.0, 1.0 - std::abs((static_cast<double>(j) + 0.5 - center) / support));
        };
        double total = 0.0;
        for (int32_t j = begin; j < end; ++j)
        {
            total += tapAt(j);
        }
        for (int32_t j = begin; total > 0.0 && j < end; ++j)
        {
            axis.weights[static_cast<size_t>(i) * axis.taps + (j - begin)]
                = static_cast<int32_t>(std::lround(tapAt(j) / total * (1 << kPrecisionBits)));
        }
    }
    return axis;
}

//! Pillow rounds each pass back to 8 bits before the next one reads it.
unsigned char round8(int32_t accumulated)
{
    return static_cast<unsigned char>(std::clamp(accumulated >> kPrecisionBits, 0, 255));
}
} // namespace

void resizeWithPad(unsigned char const* pixels, int32_t srcH, int32_t srcW, int32_t height, int32_t width, float* out)
{
    // Aspect-preserving target box, centred in the padded frame.
    double const ratio = std::max(static_cast<double>(srcW) / width, static_cast<double>(srcH) / height);
    auto const boxH = static_cast<int32_t>(static_cast<double>(srcH) / ratio);
    auto const boxW = static_cast<int32_t>(static_cast<double>(srcW) / ratio);
    int32_t const padTop = (height - boxH) / 2;
    int32_t const padLeft = (width - boxW) / 2;

    size_t const plane = static_cast<size_t>(height) * width;
    std::fill_n(out, 3 * plane, -1.0F); // black in [-1, 1]: the padding value after the [0, 1] shift
    if (boxH <= 0 || boxW <= 0)
    {
        return;
    }

    ResampleAxis const horizontal = makeResampleAxis(srcW, boxW);
    std::vector<unsigned char> rows(static_cast<size_t>(srcH) * boxW * 3);
    for (int32_t y = 0; y < srcH; ++y)
    {
        for (int32_t x = 0; x < boxW; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < horizontal.taps; ++k)
                {
                    int32_t const j = horizontal.starts[static_cast<size_t>(x)] + k;
                    if (j >= srcW)
                    {
                        break;
                    }
                    acc += horizontal.weights[static_cast<size_t>(x) * horizontal.taps + k]
                        * pixels[(static_cast<size_t>(y) * srcW + j) * 3 + c];
                }
                rows[(static_cast<size_t>(y) * boxW + x) * 3 + c] = round8(acc);
            }
        }
    }

    ResampleAxis const vertical = makeResampleAxis(srcH, boxH);
    for (int32_t y = 0; y < boxH; ++y)
    {
        for (int32_t x = 0; x < boxW; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < vertical.taps; ++k)
                {
                    int32_t const j = vertical.starts[static_cast<size_t>(y)] + k;
                    if (j >= srcH)
                    {
                        break;
                    }
                    acc += vertical.weights[static_cast<size_t>(y) * vertical.taps + k]
                        * rows[(static_cast<size_t>(j) * boxW + x) * 3 + c];
                }
                out[static_cast<size_t>(c) * plane + static_cast<size_t>(padTop + y) * width + (padLeft + x)]
                    = static_cast<float>(round8(acc)) / 255.0F * 2.0F - 1.0F;
            }
        }
    }
}

} // namespace pi05
} // namespace trt_edgellm
