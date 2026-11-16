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

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace trt_edgellm::safety::builder
{
namespace
{
class SafetyEngineSerializationTest : public testing::Test
{
protected:
    void SetUp() override
    {
        auto pattern = (std::filesystem::temp_directory_path() / "edgellm-artifact-test-XXXXXX").string();
        ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
        mDirectory = pattern;
        mEngine = mDirectory / "visual_safety.engine";
        mCompanion = mEngine.string() + ".so";
    }

    void TearDown() override
    {
        std::error_code error;
        if (!mDirectory.empty())
        {
            std::filesystem::remove_all(mDirectory, error);
        }
    }

    std::string read(std::filesystem::path const& file)
    {
        std::ifstream input(file, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    void write(std::filesystem::path const& file, std::string const& bytes)
    {
        std::ofstream output(file, std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << bytes;
        output.close();
    }

    void expectNoStagingDirectories()
    {
        for (auto const& entry : std::filesystem::directory_iterator(mDirectory))
        {
            EXPECT_NE(entry.path().filename().string().find(".edgellm-safety-"), 0U);
        }
    }

    std::filesystem::path mDirectory;
    std::filesystem::path mEngine;
    std::filesystem::path mCompanion;
};

TEST_F(SafetyEngineSerializationTest, SavesBinaryPair)
{
    // Given binary buffers, when saved, then both files retain embedded NUL bytes with no staging residue.
    std::string const engine{"engine\0data", 11};
    std::string const companion{"so\0data", 7};
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, engine, companion));
    EXPECT_EQ(read(mEngine), engine);
    EXPECT_EQ(read(mCompanion), companion);
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, ReplacesBothArtifacts)
{
    // Given an existing pair, when replaced, then both files belong to the new build.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so"));
    EXPECT_EQ(read(mEngine), "new-engine");
    EXPECT_EQ(read(mCompanion), "new-so");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, EmptyCompanionPreservesExistingPair)
{
    // Given an existing pair, when the new companion is empty, then saving fails without changing either file.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", {}));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_EQ(read(mCompanion), "old-so");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, EmptyEnginePreservesExistingPair)
{
    // Given an existing pair, when the new engine is empty, then saving fails without changing either file.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, {}, "new-so"));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_EQ(read(mCompanion), "old-so");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsCompanionDirectoryWithoutChangingEngine)
{
    // Given a companion destination directory, when saving, then the directory and old engine remain unchanged.
    write(mEngine, "old-engine");
    std::filesystem::create_directory(mCompanion);
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so"));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_TRUE(std::filesystem::is_directory(mCompanion));
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsEngineDirectoryWithoutPublishingCompanion)
{
    // Given an engine destination directory, when saving, then no companion is published.
    std::filesystem::create_directory(mEngine);
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "engine", "so"));
    EXPECT_FALSE(std::filesystem::exists(mCompanion));
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsSymlinkWithoutFollowingIt)
{
    // Given a symlink destination, when saving, then neither the link nor its target is modified.
    write(mEngine, "old-engine");
    std::filesystem::create_symlink(mEngine, mCompanion);
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so"));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_TRUE(std::filesystem::is_symlink(mCompanion));
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, MissingParentFailsWithoutCreatingArtifacts)
{
    // Given a missing parent, when saving, then failure creates no output or staging files.
    EXPECT_FALSE(saveSafetyEngineArtifacts(mDirectory / "missing" / "engine", "engine", "so"));
    EXPECT_TRUE(std::filesystem::is_empty(mDirectory));
}

TEST_F(SafetyEngineSerializationTest, PublishesMetadataAndSidecarsWithEngine)
{
    // Given an old set and an unrelated file, when saving a new set, then all owned files update together.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    write(mDirectory / "config.json", "old-config");
    write(mDirectory / "tokenizer.json", "old-tokenizer");
    write(mDirectory / "unrelated.txt", "keep");
    auto const metadata = [this](std::filesystem::path const& directory) {
        EXPECT_EQ(read(mEngine), "old-engine");
        EXPECT_EQ(read(mDirectory / "config.json"), "old-config");
        write(directory / "config.json", "new-config");
        write(directory / "tokenizer.json", "new-tokenizer");
        return true;
    };
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata));
    EXPECT_EQ(read(mEngine), "new-engine");
    EXPECT_EQ(read(mCompanion), "new-so");
    EXPECT_EQ(read(mDirectory / "config.json"), "new-config");
    EXPECT_EQ(read(mDirectory / "tokenizer.json"), "new-tokenizer");
    EXPECT_EQ(read(mDirectory / "unrelated.txt"), "keep");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, MetadataFailurePreservesPreviousSet)
{
    // Given a valid set, when metadata staging fails or throws after a partial write, then the old set survives.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    write(mDirectory / "config.json", "old-config");
    write(mDirectory / "preprocessor_config.json", "old-sidecar");
    for (bool const throwError : {false, true})
    {
        auto const metadata = [this, throwError](std::filesystem::path const& directory) {
            write(directory / "config.json", "partial-config");
            write(directory / "preprocessor_config.json", "partial-sidecar");
            if (throwError)
            {
                throw std::runtime_error("Injected metadata write failure");
            }
            return false;
        };
        EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata));
        EXPECT_EQ(read(mEngine), "old-engine");
        EXPECT_EQ(read(mCompanion), "old-so");
        EXPECT_EQ(read(mDirectory / "config.json"), "old-config");
        EXPECT_EQ(read(mDirectory / "preprocessor_config.json"), "old-sidecar");
        expectNoStagingDirectories();
    }
}

TEST_F(SafetyEngineSerializationTest, RemovesOnlyDeclaredMissingSidecars)
{
    // Given stale owned sidecars and unrelated files, when omitted from a new build, then only owned sidecars
    // disappear.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    write(mDirectory / "config.json", "old-config");
    write(mDirectory / "tokenizer.json", "old-tokenizer");
    write(mDirectory / "generation_config.json", "old-generation");
    write(mDirectory / "chat_template.jinja", "old-template");
    write(mDirectory / "chat_template.processor", "old-processor-template");
    write(mDirectory / "preprocessor_config.json", "old-preprocessor");
    write(mDirectory / "unrelated.txt", "keep");
    auto const metadata = [this](std::filesystem::path const& directory) {
        write(directory / "config.json", "new-config");
        return true;
    };
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata,
        {"tokenizer.json", "generation_config.json", "chat_template.jinja", "chat_template.processor", "tokenizer.json",
            "absent.json"}));
    EXPECT_EQ(read(mEngine), "new-engine");
    EXPECT_EQ(read(mCompanion), "new-so");
    EXPECT_EQ(read(mDirectory / "config.json"), "new-config");
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "tokenizer.json"));
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "generation_config.json"));
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "chat_template.jinja"));
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "chat_template.processor"));
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "absent.json"));
    EXPECT_EQ(read(mDirectory / "preprocessor_config.json"), "old-preprocessor");
    EXPECT_EQ(read(mDirectory / "unrelated.txt"), "keep");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, DeclaredPresentSidecarsAreReplaced)
{
    // Given a declared optional sidecar staged by this build, when publishing, then it is replaced rather than deleted.
    write(mDirectory / "preprocessor_config.json", "old-preprocessor");
    auto const metadata = [this](std::filesystem::path const& directory) {
        write(directory / "preprocessor_config.json", "new-preprocessor");
        return true;
    };
    ASSERT_TRUE(saveSafetyEngineArtifacts(
        mEngine, "engine", "so", metadata, {"preprocessor_config.json", "preprocessor_config.json"}));
    EXPECT_EQ(read(mDirectory / "preprocessor_config.json"), "new-preprocessor");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, MissingSidecarsSurviveStagingFailure)
{
    // Given a stale sidecar queued for removal, when staging fails, then the old artifact set remains intact.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    write(mDirectory / "preprocessor_config.json", "old-preprocessor");
    auto const metadata = [](std::filesystem::path const&) { return false; };
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata, {"preprocessor_config.json"}));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_EQ(read(mCompanion), "old-so");
    EXPECT_EQ(read(mDirectory / "preprocessor_config.json"), "old-preprocessor");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsUnsafeOptionalMetadataNames)
{
    // Given an unsafe deletion name, when staging metadata, then no engine, sibling file, or companion is changed.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    auto const metadata = [](std::filesystem::path const&) { return true; };
    for (auto const& name : {std::string{}, std::string{"."}, std::string{".."}, std::string{"../outside.json"},
             (mDirectory / "outside.json").string(), std::string{"nested/file.json"}, mEngine.filename().string(),
             mCompanion.filename().string()})
    {
        SCOPED_TRACE(name);
        EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata, {name}));
        EXPECT_EQ(read(mEngine), "old-engine");
        EXPECT_EQ(read(mCompanion), "old-so");
        expectNoStagingDirectories();
    }
}

TEST_F(SafetyEngineSerializationTest, RejectsRemovalOfNonRegularSidecar)
{
    // Given a sidecar destination that is a directory or symlink, when omitted, then deletion is rejected without
    // touching it.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    auto const metadata = [](std::filesystem::path const&) { return true; };
    auto const sidecar = mDirectory / "preprocessor_config.json";
    std::filesystem::create_directory(sidecar);
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata, {"preprocessor_config.json"}));
    EXPECT_TRUE(std::filesystem::is_directory(sidecar));
    std::filesystem::remove(sidecar);
    std::filesystem::create_symlink(mEngine, sidecar);
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata, {"preprocessor_config.json"}));
    EXPECT_TRUE(std::filesystem::is_symlink(sidecar));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_EQ(read(mCompanion), "old-so");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, OptionalMetadataRequiresWriter)
{
    // Given deletion names without a metadata writer, when saving, then stale files cannot be removed implicitly.
    write(mDirectory / "tokenizer.json", "old-tokenizer");
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "engine", "so", {}, {"tokenizer.json"}));
    EXPECT_FALSE(std::filesystem::exists(mEngine));
    EXPECT_FALSE(std::filesystem::exists(mCompanion));
    EXPECT_EQ(read(mDirectory / "tokenizer.json"), "old-tokenizer");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, MetadataFailureLeavesFreshOutputEmpty)
{
    // Given a fresh output directory, when a sidecar cannot be staged, then no partial build becomes visible.
    auto const metadata = [this](std::filesystem::path const& directory) {
        write(directory / "config.json", "config");
        std::filesystem::copy_file(mDirectory / "missing-sidecar", directory / "tokenizer.json");
        return true;
    };
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "engine", "so", metadata));
    EXPECT_TRUE(std::filesystem::is_empty(mDirectory));
}

TEST_F(SafetyEngineSerializationTest, InvalidMetadataDestinationPreservesExistingSet)
{
    // Given a non-file metadata destination, when publishing staged metadata, then no original file is moved.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    write(mDirectory / "config.json", "old-config");
    std::filesystem::create_directory(mDirectory / "tokenizer.json");
    auto const metadata = [this](std::filesystem::path const& directory) {
        write(directory / "config.json", "new-config");
        write(directory / "tokenizer.json", "new-tokenizer");
        return true;
    };
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata));
    EXPECT_EQ(read(mEngine), "old-engine");
    EXPECT_EQ(read(mCompanion), "old-so");
    EXPECT_EQ(read(mDirectory / "config.json"), "old-config");
    EXPECT_TRUE(std::filesystem::is_directory(mDirectory / "tokenizer.json"));
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsStagedMetadataSymlink)
{
    // Given a writer that stages a symlink, when saving, then the link is not published or followed.
    write(mDirectory / "target", "untouched");
    auto const metadata = [this](std::filesystem::path const& directory) {
        std::filesystem::create_symlink(mDirectory / "target", directory / "config.json");
        return true;
    };
    EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "engine", "so", metadata));
    EXPECT_FALSE(std::filesystem::exists(mEngine));
    EXPECT_FALSE(std::filesystem::exists(mCompanion));
    EXPECT_FALSE(std::filesystem::exists(mDirectory / "config.json"));
    EXPECT_EQ(read(mDirectory / "target"), "untouched");
    expectNoStagingDirectories();
}

TEST_F(SafetyEngineSerializationTest, RejectsMetadataEngineNameCollision)
{
    // Given metadata using an engine artifact name, when saving, then it cannot overwrite either engine buffer.
    ASSERT_TRUE(saveSafetyEngineArtifacts(mEngine, "old-engine", "old-so"));
    for (auto const& name : {mEngine.filename(), mCompanion.filename()})
    {
        auto const metadata = [this, &name](std::filesystem::path const& directory) {
            write(directory / name, "not-an-engine");
            return true;
        };
        EXPECT_FALSE(saveSafetyEngineArtifacts(mEngine, "new-engine", "new-so", metadata));
        EXPECT_EQ(read(mEngine), "old-engine");
        EXPECT_EQ(read(mCompanion), "old-so");
        expectNoStagingDirectories();
    }
}
} // namespace
} // namespace trt_edgellm::safety::builder
