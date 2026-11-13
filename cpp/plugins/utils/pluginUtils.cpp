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

#include "pluginUtils.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtInterop.h"
#include "kernels/compactPhase/compactPhaseSupport.h"
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace plugins
{
namespace
{

struct CompactBuildReport
{
    std::mutex mutex;
    uint64_t lastId{0};
    //! Reports begun and not yet ended.
    std::set<uint64_t> open;
    //! A report began while another was open. The layers of the overlapping builds are mixed, so no report open at
    //! the time vouches for its build.
    bool overlapped{false};
    //! A layer could not be recorded, so the report cannot vouch for the build.
    bool incomplete{false};
    //! (plugin kind, layer name) -> whether every workspace query of the layer reserved the compact workspace.
    std::map<std::pair<std::string, std::string>, bool> layers;
};

CompactBuildReport& compactBuildReport()
{
    static CompactBuildReport sReport;
    return sReport;
}

//! Whether the reported build may run compact steps: every compact-capable layer reserved the compact workspace. Logs
//! why not otherwise.
bool builtForCompact(CompactBuildReport const& report)
{
    char const* const without = "; the engine will run without compact steps.";
    if (report.overlapped || report.incomplete)
    {
        LOG_WARNING("Compact build report: %s%s",
            report.overlapped ? "another engine build in this process overlapped this one"
                              : "a plugin layer could not be recorded",
            without);
        return false;
    }
    if (report.layers.empty())
    {
        LOG_WARNING(
            "Compact build report: the build queried no attention or recurrent plugin layer of this plugin library "
            "(was another copy of the library loaded?)%s",
            without);
        return false;
    }
    constexpr size_t kMAX_NAMED_LAYERS{8};
    std::string unreserved;
    size_t numUnreserved{0};
    for (auto const& [layer, layerReserved] : report.layers)
    {
        if (!layerReserved && ++numUnreserved <= kMAX_NAMED_LAYERS)
        {
            unreserved += (unreserved.empty() ? "" : ", ") + layer.first + " '" + layer.second + "'";
        }
    }
    if (numUnreserved > 0)
    {
        if (numUnreserved > kMAX_NAMED_LAYERS)
        {
            unreserved += " and " + std::to_string(numUnreserved - kMAX_NAMED_LAYERS) + " more";
        }
        LOG_WARNING(
            "Compact build report: %zu of %zu attention and recurrent plugin layers have no compact execution path "
            "on this GPU in this build and did not reserve the compact workspace: %s%s",
            numUnreserved, report.layers.size(), unreserved.c_str(), without);
        return false;
    }
    LOG_INFO("Compact build report: all %zu attention and recurrent plugin layers reserved the compact workspace.",
        report.layers.size());
    return true;
}

bool endCompactBuildReport(uint64_t reportId) noexcept
{
    auto& report = compactBuildReport();
    std::lock_guard<std::mutex> const lock(report.mutex);
    if (report.open.erase(reportId) == 0)
    {
        return false;
    }
    try
    {
        return builtForCompact(report);
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Compact build report: %s; the engine will run without compact steps.", error.what());
        return false;
    }
}

} // namespace
} // namespace plugins
} // namespace trt_edgellm

extern "C" EDGELLM_PLUGIN_EXPORT uint64_t edgellmBeginCompactBuildReport() noexcept
{
    auto& report = trt_edgellm::plugins::compactBuildReport();
    std::lock_guard<std::mutex> const lock(report.mutex);
    if (report.open.empty())
    {
        report.layers.clear();
        report.overlapped = false;
        report.incomplete = false;
    }
    else
    {
        report.overlapped = true;
    }
    try
    {
        uint64_t const id = ++report.lastId;
        report.open.insert(id);
        return id;
    }
    catch (std::exception const&)
    {
        report.overlapped = true;
        return 0;
    }
}

extern "C" EDGELLM_PLUGIN_EXPORT bool edgellmEndCompactBuildReport(uint64_t reportId) noexcept
{
    return trt_edgellm::plugins::endCompactBuildReport(reportId);
}

// Follow the TRT initLibNvInferPlugins(void*, char const*) convention:
// the host application calls this after dlopen to pass its logger,
// so plugin LOG_DEBUG messages respect the application's log level.
extern "C" EDGELLM_PLUGIN_EXPORT bool initEdgellmPlugins(void* logger, char const* /*libNamespace*/)
{
    if (logger)
    {
        auto* appLogger = dynamic_cast<trt_edgellm::TrtLogger*>(static_cast<nvinfer1::ILogger*>(logger));
        if (appLogger)
        {
            trt_edgellm::gLogger.setLevel(appLogger->getLevel());
        }
    }
    trt_edgellm::gLogger.debug(
        "Edge-LLM plugin library initialized", trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__));
    return true;
}

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

void applyThorSMRenumberWAR(int32_t& smVersion)
{
    smVersion = thorRenumberedSm(smVersion);
}

std::string resolvePluginLayerName(char const* name, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name != nullptr && std::strcmp(field.name, kLAYER_NAME_FIELD) == 0
            && field.type == PluginFieldType::kCHAR && field.data != nullptr && field.length > 0)
        {
            return std::string(static_cast<char const*>(field.data), static_cast<size_t>(field.length));
        }
    }
    return name != nullptr ? std::string(name) : std::string{};
}

PluginField layerNameField(std::string const& layerName)
{
    return PluginField(kLAYER_NAME_FIELD, layerName.empty() ? nullptr : layerName.data(), PluginFieldType::kCHAR,
        static_cast<int32_t>(layerName.size()));
}

std::string describePluginLayer(char const* kind, std::string const& layerName, std::string const& shape)
{
    std::string description(kind);
    if (!layerName.empty())
    {
        description += " '" + layerName + "'";
    }
    return description + " (" + shape + ")";
}

void reportCompactWorkspace(char const* kind, std::string const& layerName, bool reserved) noexcept
{
    auto& report = compactBuildReport();
    std::lock_guard<std::mutex> const lock(report.mutex);
    if (report.open.empty())
    {
        return;
    }
    try
    {
        bool& layerReserved = report.layers.try_emplace({kind, layerName}, reserved).first->second;
        layerReserved = layerReserved && reserved;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Compact build report: cannot record %s '%s': %s", kind, layerName.c_str(), error.what());
        report.incomplete = true;
    }
}

size_t alignTensorSize(size_t size)
{
    return ((size + kDEVICE_ALIGNMENT - 1) / kDEVICE_ALIGNMENT) * kDEVICE_ALIGNMENT;
}

size_t accumulateWorkspaceSize(size_t currentSize, rt::Coords const& shape, DataType dataType)
{
    size_t alignedSize = alignTensorSize(currentSize);
    size_t tensorSizeBytes = rt::utils::getTypeSize(fromTrt(dataType)) * static_cast<size_t>(shape.volume());

    return alignedSize + alignTensorSize(tensorSizeBytes);
}

rt::Tensor assignTensorFromWorkspace(std::byte*& workspace, rt::Coords const& shape, DataType dataType)
{
    size_t space = kDEVICE_ALIGNMENT;
    void* workspace_void = workspace;
    check::check(
        workspace != nullptr && std::align(kDEVICE_ALIGNMENT, kDEVICE_ALIGNMENT, workspace_void, space) == workspace,
        "Workspace pointer shall be valid and aligned to device alignment granularity.");

    // Create non-owned tensor instance from the workspace pointer.
    rt::Tensor tensor(workspace, shape, rt::DeviceType::kGPU, fromTrt(dataType));

    // Move the workspace pointer to the next aligned position after this tensor.
    size_t alignedSize = alignTensorSize(tensor.getMemoryCapacity());
    workspace += alignedSize;
    return tensor;
}

} // namespace plugins
} // namespace trt_edgellm
