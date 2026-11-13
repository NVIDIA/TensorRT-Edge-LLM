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

//! Row IDs into the Qwen3.8-Flash-Next hashed n-gram table, checked against HuggingFace.
//!
//! The goldens are integer-exact, not approximate: a wrong hash reads a different row of a 320M-row
//! table, which no tolerance would catch. Regenerate them with
//! unittests/resources/gen_qwen4exp_ngram_golden.py (see its docstring).

#include "runtime/preprocess/ngramRowIds.h"

#include "common/checkMacros.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

Json loadGolden()
{
    std::string const path = std::string(PROJECT_ROOT_DIR) + "/unittests/resources/qwen4exp_ngram_row_ids_golden.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << "missing golden fixture: " << path;
    Json document;
    file >> document;
    return document;
}

//! Owns the constant arrays alongside the params that point at them.
struct Constants
{
    std::vector<uint64_t> multipliers;
    std::vector<int64_t> vocabSizes;
    std::vector<int64_t> offsets;
    NGramRowIdParams params{};

    explicit Constants(Json const& json)
        : multipliers(json["layer_multipliers"].get<std::vector<uint64_t>>())
        , vocabSizes(json["ngram_heads_vocab_sizes"].get<std::vector<int64_t>>())
        , offsets(json["ngram_heads_offsets"].get<std::vector<int64_t>>())
    {
        params.ngramSize = json["ngram_size"].get<int32_t>();
        params.headsPerNgram = json["heads_per_ngram"].get<int32_t>();
        params.ngramHeads = json["ngram_heads"].get<int32_t>();
        params.eosTokenId = json["eos_token_id"].get<int32_t>();
        params.layerMultipliers = multipliers.data();
        params.headVocabSizes = vocabSizes.data();
        params.headOffsets = offsets.data();
    }
};

//! Flatten a [batch][seq] JSON array of token IDs.
std::vector<int32_t> flattenTokens(Json const& inputIds, int32_t& batchSize, int32_t& seqLen)
{
    batchSize = static_cast<int32_t>(inputIds.size());
    seqLen = static_cast<int32_t>(inputIds[0].size());
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(batchSize) * seqLen);
    for (auto const& sequence : inputIds)
    {
        EXPECT_EQ(static_cast<int32_t>(sequence.size()), seqLen);
        for (auto const& token : sequence)
        {
            tokens.push_back(token.get<int32_t>());
        }
    }
    return tokens;
}

//! Flatten a [batch][seq][heads] JSON array of row IDs.
std::vector<int64_t> flattenRows(Json const& rowIds)
{
    std::vector<int64_t> rows;
    for (auto const& sequence : rowIds)
    {
        for (auto const& position : sequence)
        {
            for (auto const& row : position)
            {
                rows.push_back(row.get<int64_t>());
            }
        }
    }
    return rows;
}

//! Run every step of one case through `run`, carrying the token history the way the runtime does.
//!
//! `run` receives (tokens, history, rows, nextHistory, batchSize, seqLen) and is expected to fill
//! rows and nextHistory. History starts at EOS, which is what makes the first two positions of a
//! sequence see a segment start.
template <typename RunStep>
void checkCase(Json const& testCase, Constants const& constants, RunStep run)
{
    int32_t const contextLen = constants.params.ngramSize - 1;
    int32_t batchSize = 0;
    int32_t seqLen = 0;
    flattenTokens(testCase["steps"][0]["input_ids"], batchSize, seqLen);

    std::vector<int32_t> history(static_cast<size_t>(batchSize) * contextLen, constants.params.eosTokenId);

    for (auto const& step : testCase["steps"])
    {
        std::vector<int32_t> const tokens = flattenTokens(step["input_ids"], batchSize, seqLen);
        std::vector<int64_t> const expected = flattenRows(step["row_ids"]);
        std::vector<int64_t> rows(expected.size(), -1);

        // nextHistory aliases history on purpose: the runtime updates the state in place.
        run(tokens, history, rows, history, batchSize, seqLen);

        ASSERT_EQ(rows.size(), expected.size());
        for (size_t index = 0; index < expected.size(); ++index)
        {
            ASSERT_EQ(rows[index], expected[index])
                << "case " << testCase["name"].get<std::string>() << ", flat row element " << index;
        }
    }

    std::vector<int32_t> expectedHistory;
    for (auto const& sequence : testCase["final_token_history"])
    {
        for (auto const& token : sequence)
        {
            expectedHistory.push_back(token.get<int32_t>());
        }
    }
    EXPECT_EQ(history, expectedHistory) << "case " << testCase["name"].get<std::string>();
}

class NGramRowIdTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        sGolden = new Json(loadGolden());
    }

    static void TearDownTestSuite()
    {
        delete sGolden;
        sGolden = nullptr;
    }

    static Json& golden()
    {
        return *sGolden;
    }

    static Json* sGolden;
};

Json* NGramRowIdTest::sGolden = nullptr;

} // namespace

//! The constants the goldens were generated with are the ones the checkpoint stores; a change here
//! invalidates every row ID below.
TEST_F(NGramRowIdTest, ConstantsMatchCheckpoint)
{
    Constants const constants(golden()["constants"]);
    EXPECT_EQ(constants.params.ngramSize, 3);
    EXPECT_EQ(constants.params.headsPerNgram, 8);
    EXPECT_EQ(constants.params.ngramHeads, 16);
    EXPECT_EQ(constants.params.eosTokenId, 248044);
    EXPECT_EQ(constants.multipliers, (std::vector<uint64_t>{23703573157769ULL, 20109073645365ULL, 8052911324071ULL}));
    EXPECT_EQ(constants.vocabSizes.front(), 20000003);
    EXPECT_EQ(constants.vocabSizes.back(), 20000171);
    EXPECT_EQ(constants.offsets.front(), 0);

    // Heads tile the table without gaps, and the last head ends one alignment step below the
    // 320,001,536 rows the checkpoint's 128 shards hold.
    int64_t total = 0;
    for (size_t head = 0; head < constants.vocabSizes.size(); ++head)
    {
        EXPECT_EQ(constants.offsets[head], total);
        total += constants.vocabSizes[head];
    }
    EXPECT_EQ(total, 320001446);
    EXPECT_LE(total, 128 * 2500012);
}

TEST_F(NGramRowIdTest, RemainderFollowsTorchOnNegativePatterns)
{
    for (auto const& testCase : golden()["remainder_cases"])
    {
        uint64_t const bits = testCase["bits"].get<uint64_t>();
        int64_t const divisor = testCase["divisor"].get<int64_t>();
        int64_t const expected = testCase["expected"].get<int64_t>();
        EXPECT_EQ(ngramRemainder(bits, divisor), expected) << "bits " << bits << " % " << divisor;
        EXPECT_GE(ngramRemainder(bits, divisor), 0);
    }
}

TEST_F(NGramRowIdTest, MatchesHuggingFace)
{
    Constants const constants(golden()["constants"]);
    for (auto const& testCase : golden()["cases"])
    {
        checkCase(testCase, constants,
            [&](std::vector<int32_t> const& tokens, std::vector<int32_t>& history, std::vector<int64_t>& rows,
                std::vector<int32_t>& nextHistory, int32_t batchSize, int32_t seqLen) {
                computeNgramRowIds(tokens.data(), history.data(), rows.data(), nextHistory.data(), batchSize, seqLen,
                    constants.params);
            });
    }
}

//! Splitting a sequence into chunks must give the same rows as a single prefill, which is what makes
//! the two-token history a sufficient carry between chunks.
TEST_F(NGramRowIdTest, ChunkedPrefillMatchesSingleShot)
{
    Constants const constants(golden()["constants"]);
    int32_t const contextLen = constants.params.ngramSize - 1;

    auto rowsForCase = [&](std::string const& name) {
        std::vector<int64_t> collected;
        int32_t batchSize = 0;
        int32_t seqLen = 0;
        for (auto const& testCase : golden()["cases"])
        {
            if (testCase["name"].get<std::string>() != name)
            {
                continue;
            }
            flattenTokens(testCase["steps"][0]["input_ids"], batchSize, seqLen);
            std::vector<int32_t> history(static_cast<size_t>(batchSize) * contextLen, constants.params.eosTokenId);
            std::vector<std::vector<int64_t>> perSequence(batchSize);
            for (auto const& step : testCase["steps"])
            {
                std::vector<int32_t> const tokens = flattenTokens(step["input_ids"], batchSize, seqLen);
                std::vector<int64_t> rows(static_cast<size_t>(batchSize) * seqLen * constants.params.ngramHeads);
                computeNgramRowIds(
                    tokens.data(), history.data(), rows.data(), history.data(), batchSize, seqLen, constants.params);
                int64_t const perSequenceCount = static_cast<int64_t>(seqLen) * constants.params.ngramHeads;
                for (int32_t sequence = 0; sequence < batchSize; ++sequence)
                {
                    auto const begin = rows.begin() + sequence * perSequenceCount;
                    perSequence[sequence].insert(perSequence[sequence].end(), begin, begin + perSequenceCount);
                }
            }
            for (auto const& sequence : perSequence)
            {
                collected.insert(collected.end(), sequence.begin(), sequence.end());
            }
        }
        EXPECT_FALSE(collected.empty()) << "case not found: " << name;
        return collected;
    };

    EXPECT_EQ(rowsForCase("chunked_1_then_rest"), rowsForCase("prose_single_shot"));
    EXPECT_EQ(rowsForCase("chunked_1_2_3_rest"), rowsForCase("prose_single_shot"));
    EXPECT_EQ(rowsForCase("chunked_across_eos"), rowsForCase("eos_boundaries"));
    EXPECT_EQ(rowsForCase("batch_chunked"), rowsForCase("batch_single_shot"));
}

//! Every row must land inside its own head's slice; a hash that escaped its modulus would otherwise
//! read a neighbouring head's rows and still look plausible.
TEST_F(NGramRowIdTest, RowsStayInsideTheirHeadSlice)
{
    Constants const constants(golden()["constants"]);
    for (auto const& testCase : golden()["cases"])
    {
        for (auto const& step : testCase["steps"])
        {
            for (auto const& sequence : step["row_ids"])
            {
                for (auto const& position : sequence)
                {
                    for (int32_t head = 0; head < constants.params.ngramHeads; ++head)
                    {
                        int64_t const row = position[head].get<int64_t>();
                        EXPECT_GE(row, constants.offsets[head]);
                        EXPECT_LT(row, constants.offsets[head] + constants.vocabSizes[head]);
                    }
                }
            }
        }
    }
}

//! The header lets nextTokenHistory alias tokenHistory, and the runtime relies on it: the in-place
//! form is what every other test here exercises, so the separate-buffer form and the equality
//! between the two are otherwise unchecked. What this catches is breakage that shows up only when
//! the buffers overlap, such as a reversed slot loop or a change to historyAt's index arithmetic.
TEST_F(NGramRowIdTest, InPlaceHistoryMatchesSeparateBuffers)
{
    Constants const constants(golden()["constants"]);
    int32_t const contextLen = constants.params.ngramSize - 1;

    for (auto const& testCase : golden()["cases"])
    {
        int32_t batchSize = 0;
        int32_t seqLen = 0;
        flattenTokens(testCase["steps"][0]["input_ids"], batchSize, seqLen);
        size_t const historySize = static_cast<size_t>(batchSize) * contextLen;

        std::vector<int32_t> hostInPlace(historySize, constants.params.eosTokenId);
        std::vector<int32_t> hostSeparateIn(historySize, constants.params.eosTokenId);

        for (auto const& step : testCase["steps"])
        {
            std::vector<int32_t> const tokens = flattenTokens(step["input_ids"], batchSize, seqLen);
            size_t const rowCount = static_cast<size_t>(batchSize) * seqLen * constants.params.ngramHeads;

            std::vector<int64_t> hostInPlaceRows(rowCount);
            computeNgramRowIds(tokens.data(), hostInPlace.data(), hostInPlaceRows.data(), hostInPlace.data(), batchSize,
                seqLen, constants.params);

            std::vector<int64_t> hostSeparateRows(rowCount);
            std::vector<int32_t> hostSeparateOut(historySize, -1);
            computeNgramRowIds(tokens.data(), hostSeparateIn.data(), hostSeparateRows.data(), hostSeparateOut.data(),
                batchSize, seqLen, constants.params);
            hostSeparateIn = hostSeparateOut;

            std::string const label = testCase["name"].get<std::string>();
            EXPECT_EQ(hostInPlaceRows, hostSeparateRows) << "rows differ in place vs separate, case " << label;
            EXPECT_EQ(hostInPlace, hostSeparateOut) << "history differs in place vs separate, case " << label;
        }
    }
}
