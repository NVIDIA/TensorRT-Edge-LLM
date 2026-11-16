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

#include "runtime/llmInferenceRuntime.h"
#include "runtime/multiStage/stage.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/streaming.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief The backbone decode path, behind IStage.
 *
 * An adapter: the decode loop, KV management and sampling stay in LLMInferenceRuntime.
 * What lives here is the part the topology owns -- admission, cancellation, and
 * publishing the prefill window to whoever subscribed to it.
 *
 * Nothing here is tied to one model family: whether a deployment runs this as the speaking
 * topology's `thinker` or as a text-only backbone is a stage id and a set of config values,
 * never a branch. Whether hidden states are published at all follows from the ports the
 * deployment declares.
 *
 * Config keys (all optional; paths are relative to `artifact`). The deployment keys mirror
 * llm_inference's flags of the same meaning:
 *   multimodal_artifact   perception engine directory
 *   checkpoint            checkpoint directory for runtime-loaded weights
 *   draft_checkpoint      separate checkpoint of a paired speculative draft
 *   accept_hidden_layer   decoder layer this engine emits hidden states on; -1, the default, is the
 *                         post-norm output
 *   max_generate_length   decode cap; defaults to 256
 *   temperature           defaults to 1.0
 *   top_p                 defaults to 0.8
 *   top_k                 defaults to 50
 *   spec_decode           {draft_top_k, draft_step, proposal_tokens, verify_tokens, dspark_w2_backend,
 *                         dspark_scheduler, dspark_confidence_threshold, dspark_min_proposal_len,
 *                         dspark_max_proposal_len}; present only for a speculative build, which emits no
 *                         hidden port, as in llm_inference. An absent count is resolved from the engines.
 *   lora_adapters         {name: path} LoRA weights a request may select
 *   context_cache         {enable_reuse, max_records, max_snapshots, encoder_cache_budget_bytes}
 *   visual_pruning        {algorithm, reduction_ratio, min_visual_tokens, pivot_image_tokens,
 *                         pivot_text_tokens}; present to enable
 *   ple                   {file, residency, cache_slots}
 *   runtime_cache         TRT-RTX runtime cache file
 *   offload_embedding     keep the embedding table in pinned host memory; defaults to false
 * A request's `sampling` input overrides the four sampling keys for that request.
 */
class LlmRunner final : public multiStage::IStage
{
public:
    //! Builds the runtime preallocate() drives; the default loads the engines the config names.
    using RuntimeFactory
        = std::function<std::unique_ptr<LLMInferenceRuntime>(DeploymentOptions const& deployment, cudaStream_t stream)>;

    explicit LlmRunner(multiStage::StageInitContext const& init);
    LlmRunner(multiStage::StageInitContext const& init, RuntimeFactory makeRuntime);
    ~LlmRunner() override;

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void stepCohort(std::vector<multiStage::StageExecContext*> const& members,
        std::vector<multiStage::StageStatus>& statuses) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(multiStage::RequestSlot const& slot) const override;

    //! The runtime behind this stage, for its aggregate prefill, generation and encoder metrics.
    //! Valid once preallocate() has run.
    LLMInferenceRuntime const& runtime() const noexcept
    {
        return *mRuntime;
    }

private:
    /*!
     * @brief One cohort member's own state.
     *
     * Indexed by RequestSlot::batchSlot, never by the row a decode step happens to put the
     * member on -- the runtime compacts those, this does not move.
     */
    struct Row
    {
        LLMGenerationRequest::Request request;
        //! Pre-tokenized prompt, when the caller supplied ids rather than text.
        std::vector<int32_t> inputIds;
        //! The only public way to stop a decode loop already running. streaming.h forbids
        //! reusing a channel, so this one is rebuilt per request; see the note in prepare().
        std::shared_ptr<StreamChannel> cancelChannel;
        //! This member's batch-level settings, taken at admit(). Its requests and per-call vectors
        //! stay empty; the members of a cohort have to agree on the rest.
        LLMGenerationRequest settings;
        //! The request's own sampling seed, which the batching rule reads alongside the settings.
        std::optional<uint64_t> requestSeed;
        std::vector<int32_t> outputIds; //!< Host payload behind the text port.
        //! The runtime's decoded text, which can differ from the streamed pieces in leading spaces.
        std::string outputText;
        std::vector<int32_t> promptIds; //!< Host payload behind the prompt-id port.
        int64_t promptTokens{0};        //!< Tokens the runtime prefilled for this row.
        //! Speculative verifications and the tokens each accepted on average; 0 when not drafting.
        int32_t specVerifyCount{0};
        float specAcceptanceLength{0.0F};
        std::vector<std::vector<LogprobEntry>> logprobs;  //!< Per generated token, when asked for.
        LLMGenerationRequest::FormattedRequest formatted; //!< The prompt as the chat template rendered it.
        std::string finishReason{"not-finished"};
        bool ready{false};
        bool hiddenConsumed{false};
        bool generated{false};
    };

    Row& rowFor(multiStage::StageExecContext const& ctx);
    Row const& rowFor(multiStage::RequestSlot const& slot) const;

    //! Bind the hidden and embeds ports to this member's slice of the retained prefill window.
    //! @param decodeRow Row this member occupied in the batched step, not its batch slot.
    void publishPrefill(multiStage::StageExecContext& ctx, int32_t decodeRow);

    //! The batch-level settings @p ctx asks for, and the request's own seed in @p requestSeed.
    LLMGenerationRequest batchSettings(
        multiStage::StageExecContext const& ctx, std::optional<uint64_t>& requestSeed) const;

    //! Take a request the caller already parsed, media and per-request options included.
    void adoptRequest(multiStage::StageExecContext const& ctx, Row& row, multiStage::PortValue const& value);
    //! Build the request from the prompt, media and sampling inputs.
    void assembleRequest(multiStage::StageExecContext& ctx, Row& row);

    //! Attach the request's media to the request the encoders read, borrowing the payload.
    void bindMedia(multiStage::StageExecContext& ctx, LLMGenerationRequest::Request& request, Message& turn);
    void bindPixels(multiStage::StageExecContext& ctx, LLMGenerationRequest::Request& request, Message& turn);

    void bindText(multiStage::StageExecContext& ctx);
    //! Publish the prompt half of the token stream, captured by stepCohort(). The talker's prefill
    //! builders scan for the chat-template assistant segment, which lives in the prompt, not
    //! the generated tail.
    void bindPromptIds(multiStage::StageExecContext& ctx);

    std::string mStageId;
    std::string mArtifactDir;
    DeploymentOptions mDeployment;
    VisualPrunerConfig mVisualPruner;
    cudaStream_t mStream{nullptr};
    int32_t mMaxBatch{1};
    int32_t mMaxSeq{0};
    int32_t mAcceptHiddenLayer{-1};
    int64_t mMaxGenerateLength{0};
    std::vector<std::string> mHiddenPorts; //!< Hidden ports this deployment emits.

    RuntimeFactory mMakeRuntime;
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    int32_t mHiddenSize{0};

    //! Sized by max_batch at preallocate(); a member's batchSlot indexes it.
    std::vector<Row> mRows;
    //! Packed from the rows that actually step, so its index is a decode row, not a batchSlot.
    LLMGenerationRequest mRequest;
    LLMGenerationResponse mResponse;
    //! Index into the stepped members of each packed decode row. A member already done or
    //! cancelled is left out, so this is not the batchSlot.
    std::vector<size_t> mPacked;
    float mTemperature{1.0F};
    float mTopP{1.0F};
    int64_t mTopK{1};
};

std::unique_ptr<multiStage::IStage> createLlmRunner(multiStage::StageInitContext const& init);
//! What an llm.backbone stage offers a stage graph.
multiStage::RunnerDecl llmRunnerDeclaration();

} // namespace rt
} // namespace trt_edgellm
