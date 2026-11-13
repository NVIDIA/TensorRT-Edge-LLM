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

// Covers the substitute engines' shared prepare() action directly. Every deployment the suites assemble binds
// kv_page_table, so the action's unbound-binding path is unreachable from an assembled runtime and nothing else
// holds it in place: driving the action with a hand-built map is the only way to pin the behavior down.

#include "substituteEngine.h"

#include <gtest/gtest-spi.h>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

// Given a tensor map that binds no kv_page_table, as a deployment that dropped the binding would hand to prepare()
// When the substitute engines' shared prepare() action runs against it
// Then it names the missing binding in a test failure and reports the prepare as failed, rather than dereferencing
//      the null tensor or recording a page that was never read
TEST_F(SubstituteEngineTest, PrepareActionReportsAnUnboundKVPageTableInsteadOfDereferencingIt)
{
    rt::TensorMap const emptyMap;
    rt::InferenceDims const dims{};

    ::testing::TestPartResultArray failures;
    bool prepared{true};
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        prepared = prepareAction(FakeOutputScale::kLogits)(/*profile=*/0, dims, emptyMap, mStream);
    }

    EXPECT_FALSE(prepared);
    ASSERT_EQ(failures.size(), 1);
    EXPECT_THAT(failures.GetTestPartResult(0).message(), ::testing::HasSubstr(binding_names::kKVPageTable));
    // A prepare() that could not read the table must contribute no entry: the assembly suites index this vector by
    // prepare() order, so a placeholder would shift every later assertion onto the wrong round.
    EXPECT_TRUE(mFirstPagePerPrepare.empty());
}

} // namespace
