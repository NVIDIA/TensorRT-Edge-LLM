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

#include "builder/safety/safetyEngineSerialization.h"

#include "common/logger.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <vector>

namespace trt_edgellm::safety::builder
{
namespace
{
struct ArtifactFile
{
    std::filesystem::path destination;
    std::filesystem::path staged;
    std::filesystem::path backup;
    bool backedUp{false};
    bool published{false};
};

class ArtifactTransaction
{
public:
    explicit ArtifactTransaction(std::filesystem::path const& enginePath)
        : mDirectory((enginePath.parent_path() / ".edgellm-safety-XXXXXX").string())
    {
        // mkdtemp replaces the six trailing Xs with a unique directory suffix.
        if (::mkdtemp(mDirectory.data()) == nullptr)
        {
            throw std::system_error(errno, std::generic_category(), "Cannot create Safety artifact staging directory");
        }
    }

    ~ArtifactTransaction() noexcept
    {
        bool recovered = true;
        if (!mCommitted)
        {
            for (auto& file : mFiles)
            {
                std::error_code error;
                if (file.published)
                {
                    std::filesystem::remove(file.destination, error);
                    recovered = !error && recovered;
                }
                if (file.backedUp)
                {
                    error.clear();
                    std::filesystem::rename(file.backup, file.destination, error);
                    recovered = !error && recovered;
                }
            }
        }
        std::error_code error;
        if (recovered)
        {
            std::filesystem::remove_all(mDirectory, error);
        }
        if (!recovered || error)
        {
            LOG_ERROR("Safety artifact cleanup/recovery incomplete; inspect staging directory %s.", mDirectory.c_str());
        }
    }

    ArtifactTransaction(ArtifactTransaction const&) = delete;
    ArtifactTransaction& operator=(ArtifactTransaction const&) = delete;

    void save(std::filesystem::path const& enginePath, std::string_view engine, std::string_view companion,
        SafetyMetadataWriter const& writeMetadata, std::initializer_list<std::string_view> optionalMetadata)
    {
        std::filesystem::path const directory(mDirectory);
        // Stage the engine and companion .so together; retain backups for rollback if saving fails.
        mFiles = {{enginePath, directory / "engine", directory / "old-engine"},
            {enginePath.string() + ".so", directory / "companion", directory / "old-companion"}};
        write(mFiles[0].staged, engine);
        write(mFiles[1].staged, companion);

        if (writeMetadata)
        {
            auto const metadataDirectory = directory / "metadata";
            auto const backupDirectory = directory / "old-metadata";
            std::filesystem::create_directory(metadataDirectory);
            std::filesystem::create_directory(backupDirectory);
            if (!writeMetadata(metadataDirectory))
            {
                throw std::runtime_error("Failed to stage Safety metadata or sidecars");
            }
            for (auto const& entry : std::filesystem::directory_iterator(metadataDirectory))
            {
                auto const name = entry.path().filename();
                if (!std::filesystem::is_regular_file(entry.symlink_status()) || name == enginePath.filename()
                    || name == mFiles[1].destination.filename())
                {
                    throw std::runtime_error("Invalid staged Safety metadata file: " + entry.path().string());
                }
                mFiles.push_back({enginePath.parent_path() / name, entry.path(), backupDirectory / name});
            }
            for (auto const filename : optionalMetadata)
            {
                std::filesystem::path const name(filename);
                if (name.empty() || name != name.filename() || name == "." || name == ".."
                    || name == enginePath.filename() || name == mFiles[1].destination.filename())
                {
                    throw std::runtime_error("Invalid optional Safety metadata filename: " + name.string());
                }
                auto const destination = enginePath.parent_path() / name;
                if (std::none_of(mFiles.begin(), mFiles.end(),
                        [&destination](ArtifactFile const& file) { return file.destination == destination; }))
                {
                    // No staged replacement: back up the old sidecar and delete it only on commit.
                    mFiles.push_back({destination, {}, backupDirectory / name});
                }
            }
        }
        else if (optionalMetadata.size() != 0)
        {
            throw std::runtime_error("Optional Safety metadata requires a metadata writer");
        }

        // Validate the complete set before moving any original artifact.
        for (auto const& file : mFiles)
        {
            auto const status = std::filesystem::symlink_status(file.destination);
            if (std::filesystem::exists(status) && !std::filesystem::is_regular_file(status))
            {
                throw std::runtime_error(
                    "Safety artifact destination is not a regular file: " + file.destination.string());
            }
        }
        for (auto& file : mFiles)
        {
            if (std::filesystem::exists(file.destination))
            {
                std::filesystem::rename(file.destination, file.backup);
                file.backedUp = true;
            }
        }
        // Publish the engine after its companion and metadata.
        for (size_t index = 1; index < mFiles.size(); ++index)
        {
            if (mFiles[index].staged.empty())
            {
                continue;
            }
            auto& file = mFiles[index];
            std::filesystem::rename(file.staged, file.destination);
            file.published = true;
        }
        std::filesystem::rename(mFiles[0].staged, mFiles[0].destination);
        mFiles[0].published = true;
        mCommitted = true;
    }

private:
    static void write(std::filesystem::path const& path, std::string_view bytes)
    {
        if (bytes.empty() || bytes.size() > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::runtime_error("Invalid serialized Safety artifact size");
        }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.close();
    }

    std::string mDirectory;
    std::vector<ArtifactFile> mFiles;
    bool mCommitted{false};
};

} // namespace

bool saveSafetyEngineArtifacts(std::filesystem::path const& enginePath, std::string_view engine,
    std::string_view companion, SafetyMetadataWriter const& writeMetadata,
    std::initializer_list<std::string_view> optionalMetadata) noexcept
{
    try
    {
        ArtifactTransaction transaction(enginePath);
        transaction.save(enginePath, engine, companion, writeMetadata, optionalMetadata);
        return true;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to save Safety engine artifacts: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("Failed to save Safety engine artifacts with an unknown error.");
    }
    return false;
}

bool buildAndSerializeSafetyEngine(nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition& network,
    nvinfer1::IBuilderConfig& config, std::filesystem::path const& enginePath,
    SafetyMetadataWriter const& writeMetadata, std::initializer_list<std::string_view> optionalMetadata) noexcept
{
    try
    {
        if (config.getEngineCapability() != nvinfer1::EngineCapability::kSAFETY)
        {
            LOG_ERROR("Safety serialization requires EngineCapability::kSAFETY.");
            return false;
        }
        auto artifacts = std::unique_ptr<nvinfer1::ISafeSerializedNetworkArtifacts>(
            builder.buildSerializedSafeNetwork(network, config, /*emitCheckerBlob=*/false));
        if (!artifacts)
        {
            LOG_ERROR("TensorRT failed to produce the Safety engine and companion library.");
            return false;
        }
        auto const* engine = artifacts->getSerializedNetwork();
        auto const* companion = artifacts->getCompanionSo();
        if (!engine || !companion || !engine->data() || engine->size() == 0 || !companion->data()
            || companion->size() == 0)
        {
            LOG_ERROR("TensorRT returned an empty Safety engine or companion library.");
            return false;
        }
        // Borrowed buffers remain owned by artifacts throughout both file writes.
        bool const saved
            = saveSafetyEngineArtifacts(enginePath, {static_cast<char const*>(engine->data()), engine->size()},
                {static_cast<char const*>(companion->data()), companion->size()}, writeMetadata, optionalMetadata);
        if (saved)
        {
            LOG_INFO("Safety engine artifacts saved to %s.", enginePath.c_str());
        }
        return saved;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Safety engine serialization failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("Safety engine serialization failed with an unknown error.");
    }
    return false;
}
} // namespace trt_edgellm::safety::builder
