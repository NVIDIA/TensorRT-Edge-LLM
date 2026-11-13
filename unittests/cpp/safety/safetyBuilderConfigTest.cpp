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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder/safety/safetyBuilderConfig.h"
#include "builder/safety/remoteAutoTuningUtils.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

using namespace trt_edgellm;

TEST(RemoteAutoTuningConfigTest, AcceptsTensorRTSshForm)
{
    // Given a complete TensorRT SSH URI, when validated, then it is accepted without an error.
    EXPECT_TRUE(safety::builder::validateRemoteAutoTuningConfig(
        "ssh://user@10.0.0.1:22?remote_exec_path=/opt/trt/bin&remote_lib_path=/opt/trt/lib"));
}

TEST(RemoteAutoTuningConfigTest, OmittedUriSelectsLocalTuningForSafetyProxy)
{
    // Given omitted, invalid and valid URIs, when validated, then only the invalid nonempty URI fails.
    EXPECT_TRUE(safety::builder::validateSafetyAutoTuningConfig(""));
    EXPECT_FALSE(safety::builder::validateSafetyAutoTuningConfig("not-a-remote-uri"));
    EXPECT_TRUE(safety::builder::validateSafetyAutoTuningConfig("ssh://user@host"));
}

TEST(SafetyBuilderConfigTest, MetadataRecordsTuningMethodWithoutClassifyingTarget)
{
    // Given local and remote builds, when serialized, then metadata records tuning mode without a proxy label.
    safety::builder::SafetyVisualBuilderConfig visualConfig;
    safety::builder::SafetyLLMBuilderConfig llmConfig;
    for (std::string const uri : {"", "ssh://user@host"})
    {
        visualConfig.remoteAutoTuningConfig = uri;
        llmConfig.remoteAutoTuningConfig = uri;
        for (auto const& json : {visualConfig.toJson(), llmConfig.toJson()})
        {
            EXPECT_EQ(json.at("remote_auto_tuning"), !uri.empty());
            EXPECT_EQ(json.at("engine_capability"), "safety");
            EXPECT_FALSE(json.contains("safety_proxy"));
        }
    }
}

TEST(RemoteAutoTuningConfigTest, RejectsMissingProtocolOrAuthorityParts)
{
    // Given empty or incomplete SSH URIs, when validated for remote tuning, then they are rejected.
    EXPECT_FALSE(safety::builder::validateRemoteAutoTuningConfig(""));
    EXPECT_FALSE(safety::builder::validateRemoteAutoTuningConfig("user@host"));
    EXPECT_FALSE(safety::builder::validateRemoteAutoTuningConfig("ssh://host"));
}

TEST(RemoteAutoTuningConfigTest, CpuOnlyDetectionMatchesWholeOption)
{
    // Given exact and similarly named options, when scanned, then only the CPU-only option matches.
    using State = safety::builder::TensorRTCpuOnlyOptionState;
    EXPECT_EQ(safety::builder::getTensorRTCpuOnlyOptionState("--foo=1 --cpu_only=1 --bar=2"), State::kENABLED);
    EXPECT_EQ(safety::builder::getTensorRTCpuOnlyOptionState("--cpu_only"), State::kENABLED);
    EXPECT_EQ(safety::builder::getTensorRTCpuOnlyOptionState("--not_cpu_only=1"), State::kABSENT);
}

TEST(RemoteAutoTuningConfigTest, CpuOnlyDetectionRejectsDisabledMalformedAndConflictingValues)
{
    // Given disabled or malformed CPU-only options, when scanned, then even enabled duplicates cannot mask them.
    for (auto const* options : {"--cpu_only=0", "--cpu_only=", "--cpu_only=10", "--cpu_only=false",
             "--cpu_only=1 --cpu_only=0", "--cpu_only=0 --cpu_only=1", "\t--cpu_only=0\n--other=1"})
    {
        SCOPED_TRACE(options);
        EXPECT_EQ(safety::builder::getTensorRTCpuOnlyOptionState(options),
            safety::builder::TensorRTCpuOnlyOptionState::kCONFLICTING);
    }
}

TEST(RemoteAutoTuningConfigTest, CpuOnlyScopeRestoresEnvironment)
{
    // Given existing TRT options, when a CPU-only scope ends, then the original environment is restored.
    class EnvironmentRestore
    {
    public:
        EnvironmentRestore()
        {
            char const* const value = std::getenv("TRT_INTERNAL_OPTIONS");
            mHadValue = value != nullptr;
            mValue = value != nullptr ? value : "";
        }

        ~EnvironmentRestore()
        {
            if (mHadValue)
            {
                setenv("TRT_INTERNAL_OPTIONS", mValue.c_str(), 1);
            }
            else
            {
                unsetenv("TRT_INTERNAL_OPTIONS");
            }
        }

    private:
        std::string mValue;
        bool mHadValue{false};
    } restore;

    ASSERT_EQ(setenv("TRT_INTERNAL_OPTIONS", "--existing=1", 1), 0);
    {
        safety::builder::ScopedTensorRTCpuOnlyMode scope;
        ASSERT_TRUE(scope.enable());
        ASSERT_NE(std::getenv("TRT_INTERNAL_OPTIONS"), nullptr);
        EXPECT_EQ(safety::builder::getTensorRTCpuOnlyOptionState(std::getenv("TRT_INTERNAL_OPTIONS")),
            safety::builder::TensorRTCpuOnlyOptionState::kENABLED);
    }
    ASSERT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), "--existing=1");

    // Given an explicitly enabled option, when the scope is entered twice, then its original spelling is preserved.
    for (auto const* options : {"--existing=1 --cpu_only=1", "--cpu_only", "--cpu_only=1 --cpu_only"})
    {
        SCOPED_TRACE(options);
        ASSERT_EQ(setenv("TRT_INTERNAL_OPTIONS", options, 1), 0);
        {
            safety::builder::ScopedTensorRTCpuOnlyMode scope;
            ASSERT_TRUE(scope.enable());
            ASSERT_TRUE(scope.enable());
            EXPECT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), options);
        }
        EXPECT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), options);
    }

    // Given a conflicting option, when remote CPU-only mode is requested, then it fails without changing the
    // environment.
    for (auto const* options :
        {"--cpu_only=0", "--cpu_only=1 --cpu_only=0", "--cpu_only=0 --cpu_only=1", "--cpu_only=", "--cpu_only=false"})
    {
        SCOPED_TRACE(options);
        ASSERT_EQ(setenv("TRT_INTERNAL_OPTIONS", options, 1), 0);
        {
            safety::builder::ScopedTensorRTCpuOnlyMode scope;
            EXPECT_FALSE(scope.enable());
            EXPECT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), options);
        }
        EXPECT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), options);
    }

    // Given an unset variable, when the scope ends, then it is unset again rather than replaced by an empty value.
    ASSERT_EQ(unsetenv("TRT_INTERNAL_OPTIONS"), 0);
    {
        safety::builder::ScopedTensorRTCpuOnlyMode scope;
        ASSERT_TRUE(scope.enable());
        ASSERT_NE(std::getenv("TRT_INTERNAL_OPTIONS"), nullptr);
        EXPECT_STREQ(std::getenv("TRT_INTERNAL_OPTIONS"), "--cpu_only=1");
    }
    EXPECT_EQ(std::getenv("TRT_INTERNAL_OPTIONS"), nullptr);
}

TEST(SafetyBuilderConfigTest, SerializedMetadataDoesNotContainRemoteCredentialsOrPluginPath)
{
    // Given sensitive builder inputs, when serialized, then neither credentials nor the plugin path are exposed.
    safety::builder::SafetyLLMBuilderConfig config;
    config.remoteAutoTuningConfig = "ssh://user:password@host/path";
    config.safetyPluginLibrary = "/secret/location/libplugin.so";
    config.profiles.push_back({1, 128, 4096, 256});

    safety::builder::Json const json = config.toJson();
    EXPECT_EQ(json.at("engine_capability"), "safety");
    EXPECT_TRUE(json.at("remote_auto_tuning"));
    EXPECT_FALSE(json.contains("remote_auto_tuning_config"));
    EXPECT_FALSE(json.contains("safety_plugin_library"));
    EXPECT_EQ(json.dump().find("password"), std::string::npos);
    EXPECT_EQ(json.dump().find("secret/location"), std::string::npos);
}
