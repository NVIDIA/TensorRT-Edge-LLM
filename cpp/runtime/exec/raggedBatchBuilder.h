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

#include "common/tensor.h"
#include "runtime/exec/scheduledStep.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class TokenLayoutBackend : int32_t
{
    kEntryPaddedCompatibility,
    kNativeCompactRagged,
};

//! The entry-padded backend requires mixedStepSupported == false; the compact backend requires it to be true.
struct RaggedEngineContract
{
    TokenLayoutBackend backend{TokenLayoutBackend::kEntryPaddedCompatibility};
    int32_t maxNumSequences{0};
    int32_t maxQueryLength{0};
    int32_t maxPhysicalTokens{0};
    int32_t recurrentPoolRows{0};
    bool mixedStepSupported{false};
};

struct RaggedStepShape
{
    int32_t numSequences{0};
    int32_t validTokens{0};
    int32_t physicalTokens{0};
    int32_t queryWidth{0};
    int32_t numContextSequences{0};
    int32_t numContextTokens{0};
    int32_t numLogits{0};
};

struct CompletionSequence
{
    RequestId requestId{0};
    ResidentRef resident;
    int32_t pastLength{0};
};

class IdentitySetScratch
{
public:
    void reserve(size_t maxItems);
    void clear();
    bool insert(uint64_t value);

private:
    std::vector<uint64_t> mKeys;
    std::vector<uint32_t> mGenerations;
    uint32_t mGeneration{0};
    size_t mMask{0};
};

struct RaggedExecutionBatch
{
    StepId stepId{0};
    TokenLayoutBackend layout{TokenLayoutBackend::kEntryPaddedCompatibility};
    RaggedStepShape shape;

    Tensor hostTokenIds;
    std::vector<SequenceIdentity> sequenceOrder;
    std::vector<int32_t> positions;
    std::vector<int32_t> queryStartOffsets;
    std::vector<int32_t> queryLengths;
    std::vector<int32_t> pastLengths;
    std::vector<int32_t> attentionSequenceLengths;
    std::vector<int32_t> stateIndices;
    std::vector<int64_t> logitsIndices;

    std::vector<int32_t> logitsToSequence;
    std::vector<SequenceWork> sequenceWorks;
    //! Compact layout only: logits rows after the logitsToSequence.size() routed ones, one per sequence that selects
    //! no logits (its last token row), in execution order. The exported graph sizes logits_indices by num_sequences,
    //! so a compact step always has L = N; the engine computes these rows and they are discarded.
    int32_t placeholderLogitsRows{0};

    void validateCommitSnapshot(StepId completedStepId, std::vector<CompletionSequence> const& current) const;
};

//! Lowers a ScheduledStep according to the contract backend.
//!
//! Entry-padded: every sequence occupies queryWidth physical rows; context and decode work cannot mix.
//! Compact (phase 8): the step must list every context sequence before every decode sequence. Execution row i is
//! step.sequences[i]; context rows [0, Np) own T_p tokens, decode rows [Np, N) own one token each, and tokens are
//! packed without padding (T = T_p + N - Np). logitsToSequence maps each logits row to its execution row.
class RaggedBatchBuilder
{
public:
    explicit RaggedBatchBuilder(RaggedEngineContract contract);

    void reserve(RaggedExecutionBatch& batch);
    void buildInto(ScheduledStep const& step, RaggedExecutionBatch& batch);
    void finalizeSubsetSelection(ScheduledStep const& sourceStep, std::vector<int32_t> const& keepStartOffsets,
        std::vector<int32_t> const& concatenatedKeepIndices, RaggedExecutionBatch& batch) const;

    static void validateRuntimeAdapterStep(ScheduledStep const& step, int32_t activeBatchSize,
        std::vector<RequestId> const& requestIds, std::vector<ResidentRef> const& residentRefs);
    static void validateExecutionBatch(RaggedExecutionBatch const& batch, RaggedEngineContract const& contract);

private:
    void buildCompactInto(ScheduledStep const& step, RaggedExecutionBatch& batch);

    RaggedEngineContract mContract;
    IdentitySetScratch mRequestIdentityScratch;
    IdentitySetScratch mResidentSlotScratch;
};

} // namespace rt
} // namespace trt_edgellm
