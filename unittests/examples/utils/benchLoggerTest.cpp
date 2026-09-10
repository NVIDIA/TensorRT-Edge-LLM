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

#include "benchLogger.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace
{

std::string readFile(std::filesystem::path const& path)
{
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

TEST(BenchLoggerTest, DFlashPathsIncludeBatchSizeAndSeed)
{
    BenchOutputParams params;
    params.mode = BenchMode::kDFLASH_DRAFT_PROPOSAL;
    params.batchSize = 4;
    params.seed = 1234567890123ULL;
    params.blockSize = 16;
    params.draftDeltaLen = 8;
    params.pastKVLen = 32;

    EXPECT_EQ(buildLayerCsvPath("out", params),
        "out/layer_dflash_draft_proposal_bs16_delta8_kv32_batch4_seed1234567890123.csv");
    EXPECT_EQ(
        buildE2ECsvPath("out", params), "out/e2e_dflash_draft_proposal_bs16_delta8_kv32_batch4_seed1234567890123.csv");

    params.mode = BenchMode::kDFLASH_DRAFT_FIRST_ROUND;
    params.inputLen = 128;
    EXPECT_EQ(buildLayerCsvPath("out", params),
        "out/layer_dflash_draft_first_round_bs16_inputlen128_batch4_seed1234567890123.csv");
    EXPECT_EQ(buildE2ECsvPath("out", params),
        "out/e2e_dflash_draft_first_round_bs16_inputlen128_batch4_seed1234567890123.csv");

    params.mode = BenchMode::kDFLASH_VERIFY;
    params.verifyTreeSize = 64;
    EXPECT_EQ(buildLayerCsvPath("out", params), "out/layer_dflash_verify_ts64_kv32_batch4_seed1234567890123.csv");
    EXPECT_EQ(buildE2ECsvPath("out", params), "out/e2e_dflash_verify_ts64_kv32_batch4_seed1234567890123.csv");

    params.mode = BenchMode::kDFLASH_DDTREE_BUILD;
    params.candidateTopK = 4;
    EXPECT_EQ(buildLayerCsvPath("out", params),
        "out/layer_dflash_ddtree_build_bs16_ts64_tk4_kv32_batch4_seed1234567890123.csv");
    EXPECT_EQ(
        buildE2ECsvPath("out", params), "out/e2e_dflash_ddtree_build_bs16_ts64_tk4_kv32_batch4_seed1234567890123.csv");
}

TEST(BenchLoggerTest, DDTreeCsvIncludesAlignedIdentityColumns)
{
    BenchOutputParams params;
    params.mode = BenchMode::kDFLASH_DDTREE_BUILD;
    params.batchSize = 2;
    params.seed = 99;
    params.blockSize = 16;
    params.verifyTreeSize = 64;
    params.candidateTopK = 4;
    params.pastKVLen = 32;

    std::filesystem::path const output = std::filesystem::temp_directory_path() / "benchLoggerTest_ddtree.csv";
    writeE2ECsv(output.string(), params, 2.0F, 1);

    EXPECT_EQ(readFile(output),
        "mode,batch_size,e2e_time_ms,trees_per_second,seed,block_size,verify_tree_size,candidate_topk,past_kv_len\n"
        "dflash_ddtree_build,2,2.0000,1000.0000,99,16,64,4,32\n");

    std::filesystem::remove(output);
}

} // namespace
