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

#include "profiling/metrics.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(MetricsTest, SpecDecodeGenerationMetricsAccumulateAcceptedTokens)
{
    setProfilingEnabled(true);

    metrics::SpecDecodeGenerationMetrics generationMetrics;
    generationMetrics.recordRun(/* iterations = */ 4, /* generatedTokens = */ 11, /* acceptedTokens = */ 9);
    generationMetrics.recordRun(/* iterations = */ 3, /* generatedTokens = */ 8, /* acceptedTokens = */ 6);

    EXPECT_EQ(generationMetrics.getTotalRuns(), 2);
    EXPECT_EQ(generationMetrics.totalIterations, 7);
    EXPECT_EQ(generationMetrics.totalGeneratedTokens, 19);
    EXPECT_EQ(generationMetrics.totalAcceptedTokens, 15);

    setProfilingEnabled(false);
}
