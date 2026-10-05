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

#pragma once

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <string>

//! An empty scratch directory for the test that is running, to be removed in TearDown.
//!
//! The name carries the process id and the test name because neither alone is enough. The temp directory is shared
//! between users and between the ctest groups, which run concurrently, so without the pid one run's `remove_all`
//! deletes the directory another run is reading, or fails outright on a directory owned by another uid. Without the
//! test name, a suite whose fixture stages files in SetUp has the same collision between its own tests when the
//! group is sharded.
inline std::filesystem::path makeScratchDir(std::string const& suite)
{
    auto const* const info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto const dir = std::filesystem::temp_directory_path()
        / ("edgellm." + suite + "." + std::to_string(getpid()) + "." + (info != nullptr ? info->name() : "static"));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}
