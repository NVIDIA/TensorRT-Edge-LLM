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

#pragma once

#include "runtime/config/llmEngineConfig.h"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

enum class DSparkSchedulerMode : int32_t
{
    kOff = 0,
    kThreshold = 1,
    kSPS = 2,
};

/*!
 * @brief User-supplied drafting parameters for speculative decoding.
 *
 * Caller-side input to `createDeploymentConfig`. The factory consumes this
 * together with the parsed engine configs to produce the consolidated
 * `SpecDecodeConfig` stored on `DeploymentConfig::specConfig`.
 */
struct SpecDecodeDraftingConfig
{
    //! Tokens to select from one predecessor during draft expansion:
    //! draftingTopK = 1 selects chain mode, draftingTopK > 1 selects tree
    //! drafting with this candidate fanout. DSpark tree mode drafts the full
    //! block and verifySize becomes the DDTree node budget.
    int32_t draftingTopK{0};
    int32_t draftingStep{0}; //!< Number of drafting steps with draft model
    int32_t verifySize{0};   //!< Number of tokens in the base verification input

    //! Optional DFlash/JetSpec draft-forward horizon. 0 means infer from the
    //! base/draft engine config. Linear DFlash/JetSpec treats this as the base
    //! verify window length and normalizes verifySize to this value.
    int32_t dflashBlockSize{0};

    DSparkSchedulerMode dsparkSchedulerMode{DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0}; //!< 0 means use the DSpark block proposal length.
};

/*!
 * @brief Consolidated speculative decoding deployment configuration.
 *
 * Holds every draft/verify speculative decoding value the runtime needs in one place, sourced
 * from three inputs:
 *   - the base engine's parsed config (`baseOutputHiddenDim`, `maxVerifySize`),
 *   - the draft engine's parsed config (`draftHiddenSize`, `maxDraftProposalSize`),
 *   - the caller-supplied `SpecDecodeDraftingConfig` drafting parameters
 *     (`draftingTopK`, `draftingStep`, `verifySize`).
 *
 * `createDeploymentConfig` populates this struct after both engine configs
 * are parsed and validates the requested drafting shape against the engine capacities.
 */
struct SpecDecodeConfig
{
    // --- Engine-derived capacities ---
    //! Base engine output hidden dim as seen by the draft (= the third dim of
    //! the draft's `hidden_states_input` binding). Sourced from the draft
    //! config's `base_model_hidden_size`: `base.hiddenSize * 3` for EAGLE-3,
    //! `base.hiddenSize` for MTP. NOT a `base.hiddenSize * 3` derivation.
    int32_t baseOutputHiddenDim{};
    //! Draft engine hidden dim (= draft.hiddenSize). Shared across all
    //! spec-decode strategies; the actual value differs per strategy (EAGLE-3
    //! draft has its own independent hidden size; MTP draft equals base hidden size).
    int32_t draftHiddenSize{};
    int32_t maxVerifySize{};        //!< Max seq_len the base engine accepts for proposal verification
    int32_t maxDraftProposalSize{}; //!< Max seq_len the draft engine accepts for proposal generation

    // --- User-supplied drafting parameters ---
    //! Tokens to select from one predecessor during draft expansion:
    //! draftingTopK = 1 selects chain mode, draftingTopK > 1 selects tree
    //! drafting with this candidate fanout.
    int32_t draftingTopK{};
    int32_t draftingStep{}; //!< Number of drafting steps with draft model
    int32_t verifySize{};   //!< Number of tokens in the base verification input

    //! DFlash/JetSpec draft-forward horizon. This is independent from draftingStep:
    //! these modes run one draft forward per iteration and that forward emits a full block.
    //! For linear DFlash/JetSpec, this is the base verify window length. DSpark uses
    //! its own block size as proposal length and therefore verifies block + 1 tokens.
    int32_t dflashBlockSize{};

    DSparkSchedulerMode dsparkSchedulerMode{DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0}; //!< 0 means use the DSpark block proposal length.
};

//! Complete deployment configuration: the base engine's config, optional draft
//! engine config, and optional consolidated speculative decoding settings.
//!
//! For non-speculative deployments `draft` and `specConfig` are both absent.
//! When `specConfig` is present `draft` must also be present — the factory
//! enforces this invariant.
struct DeploymentConfig
{
    LLMEngineConfig base;                       //!< Parsed base engine configuration
    std::optional<LLMEngineConfig> draft;       //!< Parsed draft engine configuration
    std::optional<SpecDecodeConfig> specConfig; //!< Consolidated speculative decoding settings

    //! Maximum runtime batch size across the bundle. Returns the base engine's
    //! `maxSupportedBatchSize` when there is no draft; otherwise returns the
    //! `min` of base and draft. Logs a warning if base and draft disagree.
    int32_t maxRuntimeBatchSize() const;

    //! Effective maximum proposal size across drafting and verification.
    //! Returns `max(specConfig->maxDraftProposalSize, specConfig->verifySize)`.
    //! Speculative decode only — throws `std::runtime_error` if `specConfig` is not set.
    int32_t effectiveMaxDraftProposalSize() const;

    //! Maximum tokens a single decode round can accept per slot: 1 (vanilla, no specConfig),
    //! draftingStep + 1 (chain/tree verify: EAGLE / MTP / Gemma4 MTP), verifySize
    //! for DSpark, or min(dflashBlockSize, verifySize) for DFlash/JetSpec block verify.
    //! New speculative modes must add a case to the switch in the implementation.
    int32_t maxAcceptedTokensPerRound() const;

    //! Return the concrete speculative decoding mode declared by the engine bundle.
    SpecDecodeMode specDecodeMode() const noexcept;
};

//! Create a `DeploymentConfig` from engine config paths and optional user-side drafting.
//!
//! - Parses `baseConfigPath` via `parseEngineConfig`.
//! - If `draftConfigPath` is set, parses it via `parseDraftEngineConfig`.
//! - If `draftingConfig` is set, `draftConfigPath` must also be set (else throws).
//! - If `draftingConfig` is set, builds `specConfig` by combining the engines'
//!   capacities with the user-supplied drafting parameters, and validates them
//!   against the engines' capacities:
//!     - `specConfig->verifySize <= specConfig->maxVerifySize`
//!     - non-DFlash/JetSpec: `specConfig->draftingStep * specConfig->draftingTopK <= specConfig->maxDraftProposalSize`
//!     - MTP: `draftingStep + 1 <= 9` (EAGLE utility kernel depth limit);
//!       `draftingTopK == 1` selects the linear chain and requires `verifySize == draftingStep + 1`,
//!       while `draftingTopK > 1` selects tree drafting and requires
//!       `draftingTopK < verifySize`, `draftingTopK <= 8`, and `verifySize <= 128`
//!       (MTP tree utility limit; unfillable verify nodes become padding)
//!     - DFlash/JetSpec: `draftingStep == 1`, `dflashBlockSize <= maxDraftProposalSize`, and
//!       `draftingTopK` is candidateTopK: 1 uses the linear-tree fast path while >1 uses branching DDTree.
//!       Branching DDTree supports `verifySize <= 128` and accepted path length <= 32; linear DFlash/JetSpec
//!       normalizes `verifySize` to `dflashBlockSize` for compatibility with the historical API.
//!     - DSpark: `verifySize == proposalLen + 1`; the DSpark block size/gamma denotes proposal tokens.
//!   Throws with named-fields message on violation.
//!
//! @throws std::runtime_error on any validation failure or parse failure.
DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::optional<int32_t> rank = std::nullopt,
    std::optional<int32_t> expectedWorldSize = std::nullopt);

} // namespace rt
} // namespace trt_edgellm
