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

//! NVRTC bundle of the Thor W4A16 MoE CUDA-core kernels: key validation, the
//! shared-memory budget, the runner-side key policy, compile / serialize round
//! trips (any host with NVRTC 13, no GPU needed) and module loading (SM110).
#include "common/cudaUtils.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeDispatchPolicy.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeJitCompiler.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeJitRunner.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeRunner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace
{

namespace moe = kernel::nvfp4_a16_blackwell_moe;

//! Nemotron 3.5 Lightning routed MoE with the sealed dispatch policy.
Nvfp4A16BlackwellMoeJitKey nemotronKey()
{
    Nvfp4A16BlackwellMoeJitKey key{};
    key.numExperts = 128;
    key.topK = 6;
    key.hiddenSize = 2688;
    key.interSize = 1856;
    key.interSizePadded = 1920;
    key.fc1SplitK = moe::kDecodeFc1SplitK;
    key.fc2SplitK = moe::kDecodeFc2SplitK;
    key.fc2PrefetchSlots = moe::kDecodeFc2PrefetchSlots;
    return key;
}

kernel::Nvfp4A16BlackwellMoeParams nemotronParams(int32_t const fc2PrefetchSlots = -1)
{
    kernel::Nvfp4A16BlackwellMoeParams p{};
    p.numTokens = 1;
    p.numExperts = 128;
    p.topK = 6;
    p.hiddenSize = 2688;
    p.interSize = 1856;
    p.interSizePadded = 1920;
    p.fc2PrefetchSlots = fc2PrefetchSlots;
    return p;
}

bool nvrtcTargetsThor()
{
    return getPluginJitNvrtcMajorVersion() >= 13; // sm_110a needs the CUDA 13 NVRTC
}

TEST(Nvfp4A16BlackwellMoeJitTest, ValidatesKeys)
{
    Nvfp4A16BlackwellMoeJitKey const base = nemotronKey();
    EXPECT_TRUE(canCompileNvfp4A16BlackwellMoeJitKernel(base));
    EXPECT_EQ(describeNvfp4A16BlackwellMoeJitKeyProblem(base), nullptr);

    auto reject = [&base](auto&& mutate, char const* const what) {
        Nvfp4A16BlackwellMoeJitKey key = base;
        mutate(key);
        EXPECT_FALSE(canCompileNvfp4A16BlackwellMoeJitKernel(key)) << what;
        EXPECT_NE(describeNvfp4A16BlackwellMoeJitKeyProblem(key), nullptr) << what;
        EXPECT_THROW(compileNvfp4A16BlackwellMoeJitKernel(key), std::invalid_argument) << what;
    };
    reject([](auto& k) { k.sm = 120; }, "sm");
    reject([](auto& k) { k.layout = 2U; }, "layout ABI");
    reject([](auto& k) { k.sourceAbi = 99U; }, "source ABI");
    reject([](auto& k) { k.numExperts = 0; }, "experts 0");
    reject([](auto& k) { k.numExperts = 513; }, "experts 513");
    reject([](auto& k) { k.topK = 33; }, "topK 33");
    reject([](auto& k) { k.topK = k.numExperts + 1; }, "topK > experts");
    reject([](auto& k) { k.hiddenSize = 2700; }, "hidden % 128");
    reject([](auto& k) { k.interSize = 1800; }, "inter % 64");
    reject([](auto& k) { k.interSizePadded = 1856; }, "pad < 128 multiple");
    reject([](auto& k) { k.interSizePadded = 2048; }, "pad too large");
    reject([](auto& k) { k.fc1SplitK = 0; }, "fc1 split 0");
    reject([](auto& k) { k.fc1SplitK = 64; }, "fc1 split > K tiles");
    reject([](auto& k) { k.fc2SplitK = 30; }, "fc2 split > K tiles");
    reject([](auto& k) { k.fc2PrefetchSlots = 7; }, "prefetch > topK");
    reject([](auto& k) { k.dataType = static_cast<Nvfp4A16BlackwellMoeDataType>(7); }, "dtype");

    Nvfp4A16BlackwellMoeJitKey bf16 = base;
    bf16.dataType = Nvfp4A16BlackwellMoeDataType::kBF16;
    EXPECT_TRUE(canCompileNvfp4A16BlackwellMoeJitKernel(bf16));
}

TEST(Nvfp4A16BlackwellMoeJitTest, SharedMemoryBudgetGatesTheKey)
{
    Nvfp4A16BlackwellMoeJitKey const base = nemotronKey();
    // FC1: ceil(42 K tiles / split-K 2) staged rows of 272 B; FC2: topK x ceil(29 / 8)
    // rows plus the pre-wait prefetch image per staged slot.
    EXPECT_EQ(getNvfp4A16BlackwellMoeFc1SharedBytes(base), 21 * 272);
    Nvfp4A16BlackwellMoeJitKey prefetch2 = base;
    prefetch2.fc2PrefetchSlots = 2;
    EXPECT_GT(getNvfp4A16BlackwellMoeFc2SharedBytes(prefetch2), getNvfp4A16BlackwellMoeFc2SharedBytes(base));
    EXPECT_LE(getNvfp4A16BlackwellMoeFc2SharedBytes(prefetch2), 48 * 1024);
    EXPECT_TRUE(canCompileNvfp4A16BlackwellMoeJitKernel(prefetch2));

    // A wide intermediate with split-K 1 stages too many FC2 rows for the 48 KB
    // static limit; the key is rejected rather than compiled into a launch failure.
    Nvfp4A16BlackwellMoeJitKey wide = base;
    wide.hiddenSize = 256;
    wide.interSize = 4096;
    wide.interSizePadded = 4096;
    wide.fc1SplitK = 1;
    wide.fc2SplitK = 1;
    wide.fc2PrefetchSlots = 2;
    EXPECT_GT(getNvfp4A16BlackwellMoeFc2SharedBytes(wide), 48 * 1024);
    EXPECT_FALSE(canCompileNvfp4A16BlackwellMoeJitKernel(wide));
}

TEST(Nvfp4A16BlackwellMoeJitTest, RunnerKeyFollowsTheSealedPolicy)
{
    Nvfp4A16BlackwellMoeJitKey const key = kernel::makeNvfp4A16BlackwellMoeJitKey(nemotronParams());
    EXPECT_EQ(key.sm, 110);
    EXPECT_EQ(key.layout, kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI);
    EXPECT_EQ(key.sourceAbi, kNVFP4_A16_BLACKWELL_MOE_SOURCE_ABI);
    EXPECT_EQ(key.dataType, Nvfp4A16BlackwellMoeDataType::kHALF);
    EXPECT_EQ(key.fc2SplitK, moe::kDecodeFc2SplitK); // 29 K tiles > 8
    if (std::getenv("EDGELLM_MOE_DECODE_FC1_SPLITK") == nullptr)
    {
        EXPECT_EQ(key.fc1SplitK, moe::kDecodeFc1SplitK);
    }
    if (std::getenv("EDGELLM_MOE_DECODE_FC2_PREFETCH") == nullptr)
    {
        EXPECT_EQ(key.fc2PrefetchSlots, moe::kDecodeFc2PrefetchSlots);
    }
    EXPECT_TRUE(canCompileNvfp4A16BlackwellMoeJitKernel(key));

    // The explicit request wins over policy and environment, clamped to topK and
    // the policy maximum.
    EXPECT_EQ(kernel::makeNvfp4A16BlackwellMoeJitKey(nemotronParams(2)).fc2PrefetchSlots, 2);
    EXPECT_EQ(kernel::makeNvfp4A16BlackwellMoeJitKey(nemotronParams(0)).fc2PrefetchSlots, 0);
    kernel::Nvfp4A16BlackwellMoeParams topK1 = nemotronParams(2);
    topK1.topK = 1;
    EXPECT_EQ(kernel::makeNvfp4A16BlackwellMoeJitKey(topK1).fc2PrefetchSlots, 1);
    // Shapes whose staging would exceed 48 KB fall back to fewer (here zero) slots.
    kernel::Nvfp4A16BlackwellMoeParams wide = nemotronParams(2);
    wide.hiddenSize = 256;
    wide.interSize = 4096;
    wide.interSizePadded = 4096;
    Nvfp4A16BlackwellMoeJitKey const wideKey = kernel::makeNvfp4A16BlackwellMoeJitKey(wide);
    EXPECT_EQ(wideKey.fc2PrefetchSlots, 0);
    EXPECT_TRUE(canCompileNvfp4A16BlackwellMoeJitKernel(wideKey));
    // Small K: split-K is clamped to the K-tile count.
    kernel::Nvfp4A16BlackwellMoeParams small = nemotronParams();
    small.hiddenSize = 256;
    small.interSize = 192;
    small.interSizePadded = 256;
    EXPECT_EQ(kernel::makeNvfp4A16BlackwellMoeJitKey(small).fc2SplitK, 3);
}

TEST(Nvfp4A16BlackwellMoeJitTest, CompilesAndRoundTripsTheBundle)
{
    if (!nvrtcTargetsThor())
    {
        GTEST_SKIP() << "sm_110a NVRTC compilation requires CUDA 13";
    }
    Nvfp4A16BlackwellMoeJitKey const key = nemotronKey();
    Nvfp4A16BlackwellMoeJitKernel const kernel = compileNvfp4A16BlackwellMoeJitKernel(key);
    EXPECT_TRUE(kernel.key == key);
    ASSERT_FALSE(kernel.cubin.empty());
    EXPECT_TRUE(computeNvfp4A16BlackwellMoeJitDigest(key, kernel.cubin.data(), kernel.cubin.size()) == kernel.digest);
    // Second request of the same key is served by the compile cache.
    Nvfp4A16BlackwellMoeJitKernel const again = compileNvfp4A16BlackwellMoeJitKernel(key);
    EXPECT_EQ(again.cubin, kernel.cubin);

    std::vector<uint8_t> blob = serializeNvfp4A16BlackwellMoeJitKernel(kernel);
    Nvfp4A16BlackwellMoeJitKernel const decoded = deserializeNvfp4A16BlackwellMoeJitKernel(blob.data(), blob.size());
    EXPECT_TRUE(decoded.key == key);
    EXPECT_TRUE(decoded.digest == kernel.digest);
    EXPECT_EQ(decoded.cubin, kernel.cubin);

    EXPECT_THROW(deserializeNvfp4A16BlackwellMoeJitKernel(blob.data(), blob.size() / 2), std::runtime_error);
    blob.back() ^= 0x5AU;
    EXPECT_THROW(deserializeNvfp4A16BlackwellMoeJitKernel(blob.data(), blob.size()), std::runtime_error);
    EXPECT_THROW(deserializeNvfp4A16BlackwellMoeJitKernel(nullptr, 0), std::invalid_argument);
}

TEST(Nvfp4A16BlackwellMoeJitTest, BothDataTypesAndPrefetchVariantsCompile)
{
    if (!nvrtcTargetsThor())
    {
        GTEST_SKIP() << "sm_110a NVRTC compilation requires CUDA 13";
    }
    Nvfp4A16BlackwellMoeJitKey key = nemotronKey();
    key.dataType = Nvfp4A16BlackwellMoeDataType::kBF16;
    key.fc1SplitK = 1;
    EXPECT_FALSE(compileNvfp4A16BlackwellMoeJitKernel(key).cubin.empty());
    key = nemotronKey();
    key.fc2PrefetchSlots = 2;
    EXPECT_FALSE(compileNvfp4A16BlackwellMoeJitKernel(key).cubin.empty());
}

TEST(Nvfp4A16BlackwellMoeJitTest, UnloadedRunnerRejectsLaunches)
{
    Nvfp4A16BlackwellMoeJitRunner const runner;
    EXPECT_FALSE(runner.isLoaded());
    int32_t indices{};
    float weights{};
    float logits{};
    EXPECT_THROW(
        runner.launchRoute(&logits, nullptr, 1, true, 1.0F, &indices, &weights, false, nullptr), std::runtime_error);
    EXPECT_THROW(
        runner.launchFc2(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, false, nullptr),
        std::runtime_error);
}

TEST(Nvfp4A16BlackwellMoeJitTest, LoadsTheModuleOnThor)
{
    if (!nvrtcTargetsThor() || getSMVersion() != 110)
    {
        GTEST_SKIP() << "module loading needs the Thor cubin and an SM110 device";
    }
    Nvfp4A16BlackwellMoeJitKernel kernel = compileNvfp4A16BlackwellMoeJitKernel(nemotronKey());
    Nvfp4A16BlackwellMoeJitRunner runner;
    runner.load(kernel);
    EXPECT_TRUE(runner.isLoaded());
    EXPECT_TRUE(runner.getKey() == kernel.key);
    // A second runner for the same bundle shares the registry entry.
    Nvfp4A16BlackwellMoeJitRunner other;
    other.load(kernel);
    EXPECT_TRUE(other.isLoaded());
    // Tampered payloads never reach cuModuleLoadData.
    kernel.cubin.front() ^= 0x01U;
    Nvfp4A16BlackwellMoeJitRunner tampered;
    EXPECT_THROW(tampered.load(kernel), std::invalid_argument);
    EXPECT_FALSE(tampered.isLoaded());
}

} // namespace
} // namespace trt_edgellm
