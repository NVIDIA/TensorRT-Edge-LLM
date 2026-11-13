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

#include "builder/builderUtils.h"
#include "common/trtUtils.h"

#include <gtest/gtest.h>

#if IS_TRT_RTX

TEST(RtxBuilderConfigTest, BuilderConfigTargetsTheCurrentGpu)
{
    // Given a TensorRT-RTX builder
    // When Edge-LLM creates the builder config that every engine builder uses
    // Then the config targets only the current GPU, so TensorRT-RTX can choose kernels that only some GPUs support,
    // such as MXFP8 block-scaled GEMMs
    auto const [builder, network] = trt_edgellm::builder::createBuilderAndNetwork();
    ASSERT_NE(builder, nullptr);

    auto const config = trt_edgellm::builder::createBuilderConfig(builder.get());

    ASSERT_NE(config, nullptr);
    ASSERT_EQ(config->getNbComputeCapabilities(), 1);
    EXPECT_EQ(config->getComputeCapability(0), nvinfer1::ComputeCapability::kCURRENT);
}

#endif // IS_TRT_RTX
