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

#include "common/executionPhase.h"
#include "runtime/config/llmEngineConfig.h"

#include <cstdint>
#include <filesystem>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

enum class DSparkW2Backend : int32_t
{
    kAuto = 0,
    kFp16 = 1,
    kFp8 = 2,
    kW4A16 = 3,
};

char const* dsparkW2BackendName(DSparkW2Backend backend);
DSparkW2Backend parseDSparkW2Backend(std::string const& name);
//! Resolve storage and device capability without quantizing FP16 weights to NVFP4.
DSparkW2Backend resolveDSparkW2Backend(DSparkW2Backend requested, bool hasPackedW2, bool supportsW4A16);

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
    std::optional<int32_t> draftTopK;
    std::optional<int32_t> draftingStep;
    std::optional<int32_t> proposalTokens;
    std::optional<int32_t> verifyTokens;

    DSparkSchedulerMode dsparkSchedulerMode{DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0}; //!< 0 means use the DSpark block proposal length.
    DSparkW2Backend dsparkW2Backend{DSparkW2Backend::kAuto};
};

//! Resolve public S/P/Q/K defaults consistently for loaded and injected model artifacts.
SpecDecodeDraftingConfig resolveSpecDecodeDraftingConfig(
    LLMEngineConfig const& base, LLMEngineConfig const& draft, SpecDecodeDraftingConfig const& requested);

/*!
 * @brief Consolidated speculative decoding deployment configuration.
 *
 * Holds every draft/verify speculative decoding value the runtime needs in one place, sourced
 * from three inputs:
 *   - the base engine's parsed config (`baseOutputHiddenDim`, `maxVerifyTokens`),
 *   - the draft engine's parsed config (`draftHiddenSize`, `maxDraftTokens`),
 *   - the caller-supplied `SpecDecodeDraftingConfig` drafting parameters
 *     (`draftTopK`, `draftingStep`, `proposalTokens`, `verifyTokens`).
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
    int32_t maxVerifyTokens{}; //!< Max seq_len the base engine accepts for proposal verification
    int32_t maxDraftTokens{};  //!< Max seq_len the draft engine accepts for proposal generation

    // --- User-supplied drafting parameters ---
    //! Tokens to select from one predecessor during draft expansion:
    //! draftTopK = 1 selects chain mode, draftTopK > 1 selects tree
    //! drafting with this candidate fanout.
    int32_t draftTopK{};
    int32_t draftSteps{};         //!< Number of drafting steps with draft model
    int32_t verifyTokenCount{};   //!< Number of tokens in the base verification input
    int32_t proposalTokenCount{}; //!< Maximum usable continuation depth, excluding the verification root
    int32_t maxPathLength{};      //!< Maximum root-inclusive accepted path length
    SpecVerifyLayout verifyLayout{SpecVerifyLayout::kChain};

    // --- Jump-forward limits (a mode that cannot jump keeps the plain round's values) ---
    int32_t jumpVerifyTokenCount{}; //!< Verify width of a grafting round, at least verifyTokenCount
    int32_t maxJumpTokens{};        //!< Forced tokens one jump may add; 0 when the deployment cannot jump
    int32_t jumpMaxPathLength{};    //!< Accepted path capacity of a grafting round, at most 32

    //! DFlash/JetSpec draft-forward horizon. This is independent from draftSteps:
    //! these modes run one draft forward per iteration and that forward emits a full block.
    //! For linear DFlash/JetSpec, this is the base verify window length. DSpark uses
    //! its own block size as proposal length and therefore verifies block + 1 tokens.
    int32_t draftBlockSize{};

    DSparkSchedulerMode dsparkSchedulerMode{DSparkSchedulerMode::kOff};
    float dsparkConfidenceThreshold{0.0F};
    int32_t dsparkMinProposalLen{1};
    int32_t dsparkMaxProposalLen{0}; //!< 0 means use the DSpark block proposal length.
    DSparkW2Backend dsparkW2Backend{DSparkW2Backend::kAuto};
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

    //! File path for the TRT-RTX runtime cache. No effect on non-RTX builds.
    std::optional<std::filesystem::path> runtimeCachePath;

    //! Caller-shaped drafting values resolved for this deployment; absent for vanilla execution.
    std::optional<SpecDecodeDraftingConfig> draftingConfig() const;

    //! Maximum runtime batch size across the bundle. Returns the base engine's
    //! `maxSupportedBatchSize` when there is no draft; otherwise returns the
    //! `min` of base and draft. Logs a warning if base and draft disagree.
    int32_t maxRuntimeBatchSize() const;

    //! Effective maximum proposal size across drafting and verification.
    //! Returns `max(specConfig->maxDraftTokens, specConfig->verifyTokenCount)`.
    //! Speculative decode only — throws `std::runtime_error` if `specConfig` is not set.
    int32_t effectiveMaxDraftProposalSize() const;

    //! Maximum tokens a single decode round can accept per slot: 1 for vanilla, the diffusion canvas length for a
    //! diffusion backbone, or the validated root-inclusive maximum path length for speculative decoding, which is the
    //! capacity of a jump-forward grafting round when the deployment can jump.
    int32_t maxAcceptedTokensPerRound() const;

    //! Return the concrete speculative decoding mode declared by the engine bundle.
    SpecDecodeMode specDecodeMode() const noexcept;

    //! Select bounded SWA storage when its page budget is smaller than full storage. Context reuse selects full
    //! storage, because bounded storage can resume only from exact cached endpoints. Exported capability metadata
    //! remains unchanged.
    void selectSwaKVCacheMode(bool contextReuseEnabled) noexcept;
};

//! Create a `DeploymentConfig` from engine config paths and optional user-side drafting.
//!
//! - Parses `baseConfigPath` via `parseEngineConfig`.
//! - If `draftConfigPath` is set, parses it via `parseDraftEngineConfig`.
//! - If `draftingConfig` is set, `draftConfigPath` must also be set (else throws).
//! - If `draftingConfig` is set, builds `specConfig` by combining the engines'
//!   capacities with the user-supplied drafting parameters, and validates them
//!   against the engines' capacities:
//!     - `specConfig->verifyTokenCount <= specConfig->maxVerifyTokens`
//!     - non-DFlash/JetSpec: the resolved draft input width is at most `specConfig->maxDraftTokens`
//!     - MTP: `draftSteps + 1 <= 16` (EAGLE utility kernel depth limit);
//!       `draftTopK == 1` selects the linear chain and requires the active
//!       `verifyTokenCount == draftSteps + 1` (the engine's `maxVerifyTokens` profile may be larger),
//!       while `draftTopK > 1` selects tree drafting and requires
//!       `draftTopK < verifyTokenCount`, `draftTopK <= 8`, and `verifyTokenCount <= 128`
//!       (`kMAX_SPEC_VERIFY_TREE_TOKENS`, the tree budget shared by every tree verifier and the decode
//!       attention kernels; unfillable verify nodes become padding)
//!     - DFlash/JetSpec: `draftSteps == 1`, `draftBlockSize <= maxDraftTokens`, and
//!       `draftTopK` is candidateTopK: 1 uses the linear-tree fast path while >1 uses branching DDTree.
//!       Branching DDTree supports `verifyTokenCount <= 128` and accepted path length <= 32;
//!       chain verification requires `verifyTokenCount == proposalTokenCount + 1`.
//!     - DSpark chain mode verifies the anchor plus the selected proposal length; tree mode uses the configured
//!       verification-node budget. The DSpark block size/gamma denotes the maximum proposal depth.
//!     - Every speculative mode derives the jump-forward limits (`jumpVerifyTokenCount`, `maxJumpTokens`,
//!       `jumpMaxPathLength`) per drafting geometry from the engines' verify and proposal capacities, the draft block
//!       size and the node and accepted-path limits of the plain round's kernels. A tree widens only into the base's
//!       verify headroom; a chain needs none.
//!   Throws with named-fields message on violation.
//!
//! @throws std::runtime_error on any validation failure or parse failure.
DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::optional<int32_t> rank = std::nullopt,
    std::optional<int32_t> expectedWorldSize = std::nullopt);

} // namespace rt
} // namespace trt_edgellm
