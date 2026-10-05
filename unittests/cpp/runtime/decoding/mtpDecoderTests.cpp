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

// MTP speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

Json makeMtpDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kVerifySize);
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

class MtpAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpAssemblyTests", makeSpecBaseConfig("mtp", kVerifySize), makeMtpDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }
};

TEST_F(MtpAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOnDisk)
{
    expectAssembledFromArtifactsAlone("mtp");
}

TEST_F(MtpAssemblyTest, RunsTheDraftChainThenOneBaseVerificationPerRound)
{
    expectChainProposalThenOneVerification();
}

TEST_F(MtpAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // MTP's per-slot state is the walked chain in its own draft cache. The other three decoders are held to this
    // too; leaving the mode that has been in production longest out of it would be the odd exception.
    expectSurvivorKeepsDecodingAfterCompaction(kVerifySize);
}

} // namespace
