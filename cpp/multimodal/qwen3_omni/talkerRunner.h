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

#include "chatTemplate/chatTemplate.h"
#include "common/tensor.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/multiStage/stage.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include "multimodal/qwen3_omni/cloneEncoderRunner.h"
#include "multimodal/qwen3_omni/codePredictor.h"
#include "multimodal/qwen3_omni/talkerTokenIds.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

/*!
 * @brief Codec generation, behind IStage.
 *
 * One step is one codec frame: the frame is appended to the codec port and published
 * before the next step runs, so a cancel lands within a frame rather than at the end
 * of the utterance. The port carries every frame produced so far, which is the shape
 * the vocoder downstream walks with its own cursor.
 *
 * The residual code predictor is a second engine inside this stage, not a stage of its
 * own: it runs once per codec frame from within stepCohort(), on this stage's stream, over
 * buffers this stage owns.
 *
 * Config keys:
 *   prefill_builder    required, no default. "segments", "chunked" or "tts"; see PrefillBuilder.
 *   code_predictor_artifact  required, code predictor engine directory relative to `artifact`
 *   code_predictor_checkpoint optional, checkpoint-backed code predictor weights,
 *                      relative to `code_predictor_artifact`
 *   tokenizer_dir      directory holding the backbone `embedding.safetensors`; defaults to the
 *                      parent of `artifact`. Every layout looks its tts tokens up in the
 *                      tokenizer here, else in `artifact`, and refuses to load with neither; the
 *                      segments and chunked layouts also look up their chat-template tokens, and
 *                      segments its media placeholders. A talker with no backbone renders
 *                      requests with the tokenizer and chat template of the first of the two
 *                      that has both
 *   checkpoint         optional, checkpoint-backed weights
 *   speaker            default speaker name or id; a request may override it via `voice`
 *   assistant_instruct style/emotion name resolved through `talker_assistant_prompt_id_mapping`
 *   language           tts layout language resolved through `codec_language_id`; a request
 *                      overrides it with the `language` style field
 *   talker_language    chunked layout language resolved through `talker_language_id`; a request
 *                      overrides it with the `talker_language` style field
 *   max_audio_length   default codec-frame cap; a request may raise it up to the KV room left
 *                      after its prefill
 *   temperature / top_k / top_p / repetition_penalty      talker sampling
 *   cp_temperature / cp_top_k / cp_top_p                  code predictor sampling
 *   cp_spec_verify_size      optional, code predictor verify window; 0 (default) keeps the
 *                            autoregressive RVQ loop
 *   clone_encoder_artifact   optional, reference-encoder engine directory relative to `artifact`
 *   report_codes       optional, false by default; true adds the request's codec frames to its
 *                      report as `rvq_codes`, one array of codes per frame
 */
class TalkerRunner final : public multiStage::IStage
{
public:
    explicit TalkerRunner(multiStage::StageInitContext const& init);
    ~TalkerRunner() override;

    void preallocate() override;
    bool admit(multiStage::StageExecContext const& ctx, std::string& reason) override;
    void prepare(multiStage::StageExecContext& ctx) override;
    multiStage::StageStatus step(multiStage::StageExecContext& ctx) override;
    void stepCohort(std::vector<multiStage::StageExecContext*> const& members,
        std::vector<multiStage::StageStatus>& statuses) override;
    void finish(multiStage::StageExecContext& ctx, multiStage::FinishReason reason) override;
    void reset() override;
    nlohmann::json diagnostics(rt::multiStage::RequestSlot const& slot) const override;

private:
    //! Talker sampling defaults when the config leaves a knob at 0.
    static constexpr float kTalkerTemperature = 0.9F;
    static constexpr int32_t kTalkerTopK = 50;
    static constexpr float kTalkerTopP = 1.0F;
    //! Seed for talker codec-token sampling.
    static constexpr uint64_t kTalkerSamplingSeed = 42;
    //! The per-position speaker-codec sum stages one code per group on the stack.
    static constexpr int32_t kMaxCodeGroups = 64;

    static std::string lowered(std::string value);

    /*!
     * @brief Which prefill layout this deployment assembles.
     *
     * The layouts are not interchangeable: they differ in row sequence, in which projection
     * runs over the backbone hidden states, and in how the assistant text reaches the
     * talker (per-frame residual addend versus periodic re-prefill). The config states
     * which one the artifacts were exported for.
     */
    enum class PrefillBuilder : uint8_t
    {
        kSegments, //!< User segments + restructured assistant preamble; text arrives per frame.
        kChunked,  //!< System/user/assistant row plan; text arrives by re-prefill.
        kTts,      //!< One assistant turn, no backbone: instruction, reference voice, language.
    };

    //! Talker configuration, read from the engine directory's config.json.
    struct TalkerConfig
    {
        int32_t thinkerHiddenSize{};
        int32_t talkerHiddenSize{};
        int32_t talkerVocabSize{};
        int32_t numCodeGroups{}; //!< Codes per frame; the residual codebook count is this minus one.
        int32_t maxSeqLen{};

        //! Codec special tokens, from the talker vocab, used directly.
        int32_t codecNothinkId{};
        //! -1 means the checkpoint has no think token and language conditioning stays off.
        int32_t codecThinkId{-1};
        int32_t codecThinkBosId{};
        int32_t codecThinkEosId{};
        int32_t codecPadId{};
        int32_t codecBosId{};
        int32_t codecEosId{};

        //! "voice_design" drops the speaker row from the assistant prefix entirely.
        std::string ttsModelType;
        //! The tts layout's own language table, distinct from talker_language_id.
        std::unordered_map<std::string, int32_t> codecLanguageIds;
        //! Speakers whose own dialect names the language when the request asks for none or for Chinese.
        std::unordered_map<std::string, std::string> speakerDialects;

        int32_t defaultSpeakerId{};

        //! PT _get_talker_user_parts subsamples multimodal positions to at most this many
        //! via torch.linspace(0, N-1, M).long(). 0 disables.
        int32_t maxThinkerToTalkerMmTokens{16};

        //! HF `talker_suppressed_tokens` range start, [vocabSize - K, vocabSize).
        int32_t talkerSuppressStart{};
    };

    //! One <|im_start|>-delimited span of the templated token stream.
    struct SegmentInfo
    {
        int64_t startPos;
        int64_t endPos;
        int32_t roleId;
    };

    /*!
     * @brief Chunked text feeding state.
     *
     * HF `generate_talker` interleaves chunkTokensPerCall text tokens with framesPerCall
     * codec frames through repeated KV-reset prefills; pure autoregressive decode past
     * chunk 0 degenerates. Stays inactive outside the kChunked layout.
     */
    struct ChunkStreamState
    {
        bool active{false};
        std::vector<int32_t> remainingTextTokens; //!< Token ids for chunks 1...
        bool hasTrailingTtsEos{false};            //!< tts_eos row still owed after all text chunks.
        int32_t chunkTokensPerCall{4};
        //! HF runs framesPerCall+1 forwards per forced call but its lagged predictor emits
        //! only framesPerCall frames; the extra sample is a discarded lookahead. Emitting it
        //! renders one text-starved orphan frame per chunk into the audio.
        int32_t framesPerCall{4};
        int32_t codecEmbedFrames{4}; //!< Frames fed back as codec embeds.
        int64_t cumulativeSeqLen{0}; //!< Rows written into mTalkerInputEmbeds so far.
        int32_t cursorToken{0};
        int32_t firstFrameOfCallIdx{0}; //!< Frame index of the first frame of the current call.
    };

    /*!
     * @brief One cohort member's own frame-loop state.
     *
     * Indexed by RequestSlot::batchSlot. The KV state index is the member's row in mCohortSlots,
     * frozen at prefill, so it stays the same for the whole utterance. Everything a request needs
     * only while its own prefill is being assembled stays a plain member below: prepare() runs
     * one member at a time, so that scratch cannot be seen by two requests at once.
     */
    struct Row
    {
        rt::Tensor trailingView; //!< This member's slice of mTrailingRows.
        //! [talkerVocab], 1 for each codec token already in the repetition-penalty window.
        std::vector<uint8_t> seenTokens;
        std::vector<int32_t> flatCodes; //!< [maxKVCacheCapacity * numCodesPerFrame], claimed once.
        int64_t prefillSeqLen{0};
        //! KV tokens already committed for this member's talker sequence.
        int32_t pastLen{0};
        int32_t numSeenTokens{0};
        int32_t codecToken{-1};
        int32_t frames{0};
        int32_t maxFrames{0};
        //! Batch-level in SamplingParams, so every member of a cohort has to agree.
        float temperature{0.0F};
        int32_t topK{0};
        float topP{0.0F};
        float cpTemperature{0.0F};
        int32_t cpTopK{0};
        float cpTopP{0.0F};
        float repetitionPenalty{1.05F}; //!< Applied per row.
        //! gTimer's talker generation total when the cohort's prefill ended. The timer accumulates
        //! across requests, and finish() runs after the whole cohort, so the RTF is this member's
        //! audio over the cohort's generation time.
        float generationMsAtStart{0.0F};
        float realTimeFactor{0.0F};
        float audioDurationSeconds{0.0F};
        bool ready{false};
        bool prepared{false};
        bool finished{false};
        bool hitEos{false};
    };

    Row& rowFor(multiStage::StageExecContext const& ctx);
    Row const& rowFor(multiStage::RequestSlot const& slot) const;
    //! View of @p buffer's slice for batch slot @p slot, shaped @p shape.
    static rt::Tensor sliceOf(rt::Tensor const& buffer, int32_t slot, rt::Coords const& shape);

    //! Prefill every member that prepare() staged, then sample each one's first codec token.
    void prefillCohort(std::vector<multiStage::StageExecContext*> const& members);
    //! One codec frame for the whole cohort. A finished member keeps its engine row until every
    //! row after it has finished too; nothing it produces is kept.
    bool decodeCohortFrame(cudaStream_t stream);
    //! Whether @p row decodes under the sampling @p founder's cohort shares.
    static bool sharesSampling(Row const& founder, Row const& row);
    //! Throws naming the first member whose sampling differs from the cohort's founder.
    void checkCohortSampling(
        std::vector<multiStage::StageExecContext*> const& members, std::vector<size_t> const& live) const;

    // ---- init ----
    bool validateAndFillConfig(std::string const& engineDir);
    bool initializeTalkerEngine(std::string const& engineDir, std::string const& checkpointDir);
    bool allocateBuffers();
    bool loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream);
    void loadTextEmbeddingTable(std::string const& weightsDir, cudaStream_t stream);
    void initializeTTSEmbeddings(cudaStream_t stream);
    void setUpCodePredictor(cudaStream_t stream);
    bool buildCodecEmbedPointerTable(cudaStream_t stream);
    void captureDecodeGraphs(cudaStream_t stream);

    // ---- speaker ----
    //! Exact name, then voice_map alias, then lower-cased name, then a speaker id spelled in
    //! digits. -1 when none of them hits.
    //! Admission and lookup must agree here: a near-miss that silently returns the default
    //! speaker swaps the voice and reads as a quality regression, not as a config error.
    int32_t lookupSpeakerId(std::string const& speakerName) const;
    int32_t getSpeakerIdByName(std::string const& speakerName) const;
    //! True when @p speaker resolves without landing on the default by accident.
    bool speakerResolves(std::string const& speaker) const;
    //! Index of the first speakable assistant token, recovered from the token stream when no
    //! backbone reports a hidden window.
    int32_t assistantContentStart() const;
    //! The `voice` binding when present, else the configured default.
    std::string requestedSpeaker(multiStage::StageExecContext const& ctx) const;

    // ---- prefill layouts, defined in talkerPrefill.cpp ----
    bool buildSegmentsPrefill(rt::Tensor const* prefillEmbeds, rt::Tensor const* prefillHidden, int32_t prefillLen,
        int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount, int64_t& outSeqLen,
        cudaStream_t stream);
    void finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream);

    bool buildChunkedPrefill(rt::Tensor const* prefillHidden, int32_t prefillLen, int32_t speakerId, int64_t& outSeqLen,
        cudaStream_t stream);
    //! Copy one embedding table row into dstBase[dstRow]. Fails on an out-of-range id.
    bool copyEmbedRow(rt::Tensor const& table, int32_t tokenId, __half* dstBase, int64_t dstRow, cudaStream_t stream);
    //! Copy a [talkerHidden] row into dstBase[dstRow].
    void copyRawRow(rt::Tensor const& src, __half* dstBase, int64_t dstRow, cudaStream_t stream);
    //! Write one per-position speaker-codec sum row. @p slot picks the pinned staging row the codes
    //! are uploaded from; a slot is reused only after a stream sync, so number the rows of one
    //! prefill from 0.
    void sumSpeakerCodecRow(
        int64_t const* hostCodes, int32_t slot, __half* dstBase, int64_t dstRow, cudaStream_t stream);
    //! Project prefillHidden[srcRow] through the single-Linear hidden projection.
    void projectHiddenRow(
        rt::Tensor const& prefillHidden, int64_t srcRow, __half* dstBase, int64_t dstRow, cudaStream_t stream);
    void resetChunkStream();
    enum class Reprefill : uint8_t
    {
        kNotDue,
        kDone,      //!< Re-prefilled, and the codec token resampled from it.
        kInputFull, //!< The next chunk would pass the engine's input length; nothing was written.
        kFailed,
    };
    Reprefill maybeReprefillChunk(cudaStream_t stream);
    Reprefill reprefillChunk(cudaStream_t stream);

    bool buildTtsPrefill(int32_t speakerId, int64_t& outSeqLen, cudaStream_t stream);
    //! Which codec token conditions the think block, or -1 to leave the layout language-free.
    int32_t resolveLanguageId(std::string const& languageName, std::string const& speakerName) const;
    //! Run the reference encoders over \p wav24k. Sets \p iclFrames to the ICL frame count, 0 when
    //! the request gave no transcript and only the x-vector conditions the voice.
    bool encodeVoiceCloneReference(rt::Tensor const& wav24k, int32_t& iclFrames, cudaStream_t stream);
    __half const* talkerEmbRow(int32_t tokenId) const;
    void pushPrefillRow(__half const* srcA, __half const* srcB);
    int64_t flushPrefillRows(rt::Tensor& output, cudaStream_t stream);

    // ---- engine steps ----
    //! @p seqLens is one entry per row; rows shorter than the padded extent are ragged.
    bool executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, std::vector<int32_t> const& seqLens,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);
    bool executeTalkerDecodingStep(rt::Tensor const& inputEmbeds, std::vector<int32_t> const& pastLengths,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);
    //! @p rowLengths gives each row's real length when the buffer still holds a padded prefill
    //! window; the last real token, not the last padded slot, is what the code predictor reads.
    bool extractTalkerLastHidden(rt::Tensor const& talkerHiddenStates, std::vector<int32_t> const& rowLengths,
        rt::Tensor& outputLastHidden, cudaStream_t stream);
    bool computeResidualConnection(Row const& row, std::vector<int32_t> const& codes, int32_t generationStep,
        rt::Tensor& outputResidual, rt::Tensor const& codecHiddens, cudaStream_t stream);

    //! Force logits[codecEosId] = -INF before sampling; enforces HF's min_new_tokens on a
    //! non-last chunked call.
    void suppressTalkerEosLogit(rt::Tensor& logits, int64_t row, cudaStream_t stream);
    void trackSeenToken(Row& row, int32_t slot, int32_t token, int32_t const* tokenDev, cudaStream_t stream);

    // ---- frame loop ----
    //! Cap the frame count at what the talker KV cache can still hold past \p assembledRows.
    void setFrameBudget(Row& row, int64_t assembledRows, int32_t framesSoFar);
    void bindCodes(multiStage::StageExecContext& ctx, bool complete);
    //! Fold this request's overrides onto the deployment's sampling config.
    void applySampling(nlohmann::json const& overrides);
    //! applySampling(), then record the result as @p row's own.
    void takeSampling(Row& row, nlohmann::json const& overrides);
    //! Fold this request's voice-clone codes and style names onto the deployment's.
    void applyStyle(nlohmann::json const& style, multiStage::PortValue const* speakerCodes);

    bool usesChunkedText() const noexcept
    {
        return mPrefillBuilder == PrefillBuilder::kChunked;
    }

    std::string mStageId;
    std::string mArtifactDir;
    //! Only a speech-only deployment needs these: with a backbone upstream the tokens arrive
    //! already rendered. Loaded in preallocate() from the talker's own directory.
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;
    std::unique_ptr<chat_template::ChatTemplate> mChatTemplate;
    //! Render a conversation into token ids. Empty when this deployment cannot.
    std::vector<int32_t> renderConversation(std::string const& envelope) const;
    std::string mCodePredictorDir; //!< Code predictor engine directory.
    std::string mCodePredictorCheckpoint;
    std::string mCloneEncoderDir; //!< Reference encoders; empty unless the graph names them.
    int32_t mCpSpecVerifySize{0}; //!< Verify window asked of the code predictor; 0 = off.
    std::string mBackboneDir;     //!< Where the backbone embedding table lives.
    std::string mCheckpointDir;
    cudaStream_t mStream{nullptr};
    int32_t mMaxBatch{1};
    int32_t mMaxSeq{0};
    PrefillBuilder mPrefillBuilder{PrefillBuilder::kSegments};
    bool mSpeechOnly{false}; //!< No backbone upstream, so this talker renders the conversation itself

    std::string mDefaultSpeaker;
    std::string mAssistantInstruct;
    std::string mLanguage;
    std::string mTalkerLanguage;
    bool mReportCodes{false};

    //! Per-request voice and style, empty unless the request bound them. Prefill scratch like
    //! mTextTokenIds: read only while prepare() assembles this member. A reference voice stands
    //! in for the built-in speaker rows; the instruct and language names override the deployment's.
    std::vector<std::vector<int32_t>> mPromptSpeakerCodes;
    std::string mRequestInstruct;
    std::string mRequestLanguage;
    std::string mRequestTalkerLanguage;
    std::vector<int32_t> mSystemInstructIds;
    //! Reference-voice conditioning for the tts layout. The ids form arrives assistant-wrapped;
    //! applyStyle() wraps the plain-text form when a tokenizer is loaded.
    std::vector<int32_t> mInstructIds;
    std::vector<int32_t> mRefTextIds;
    rt::Tensor mRefAudioView;
    std::string mRequestSpeakerName;
    //! What this deployment was configured with. Never mutated after construction; the
    //! per-request values below are derived from it every prepare(), so one request's
    //! overrides cannot leak into the next.
    struct SamplingConfig
    {
        int32_t maxAudioLength{0};
        float temperature{0.0F};
        int32_t topK{0};
        float topP{0.0F};
        float repetitionPenalty{1.05F};
        float cpTemperature{0.0F};
        int32_t cpTopK{0};
        float cpTopP{0.0F};
    };
    SamplingConfig mSamplingConfig;

    int32_t mMaxAudioLength{0};
    float mTemperature{0.0F};
    int32_t mTopK{0};
    float mTopP{0.0F};
    float mRepetitionPenalty{1.05F};
    float mCpTemperature{0.0F};
    int32_t mCpTopK{0};
    float mCpTopP{0.0F};

    TalkerConfig mTalkerConfig{};
    //! Resolved from the tokenizer at load; the tts layout reads only the tts tokens.
    TalkerTokenIds mTokenIds;
    int32_t mNumRvqLayers{0};
    int32_t mNumCodesPerFrame{0};
    std::unordered_map<std::string, int32_t> mSpeakerIdMap;
    //! Friendly name to internal speaker name, from voice_map.json. Keys are lower-cased.
    std::unordered_map<std::string, std::string> mVoiceAliasMap;
    //! Style/emotion name to token ids, from `talker_assistant_prompt_id_mapping`.
    std::unordered_map<std::string, std::vector<int32_t>> mAssistantPromptIds;
    //! Language name to codec-side language id, from `talker_language_id`.
    std::unordered_map<std::string, int32_t> mLanguageIds;
    //! speaker_id to text-token list, emitted between the system role trio and codec_bos.
    std::unordered_map<int32_t, std::vector<int32_t>> mSpeakerSystemPromptIds;

    LLMEngineConfig mTalkerLLMConfig;
    std::unique_ptr<EngineExecutor> mTalkerExec;
    //! Whether the talker engine emits hidden_states for the last token only rather than
    //! for the whole sequence. Probed from the engine's declared shape at load time.
    bool mTalkerHiddenIsGathered{false};

    //! Sequence extent the talker hidden-states buffer needs for a prefill of `seqLen`.
    int64_t talkerHiddenSeqDim(int64_t seqLen) const
    {
        return mTalkerHiddenIsGathered ? 1 : seqLen;
    }
    std::unique_ptr<SharedResources> mTalkerSharedRes;
    std::unique_ptr<PipelineIO> mTalkerPipelineIO;
    TensorMap mTalkerTensorMap;
    std::unique_ptr<StepPreparer> mTalkerStepPreparer;
    //! Token-major step layout; prepareRaggedStep() rewrites it before every talker execute().
    rt::RaggedExecutionBatch mTalkerRagged;
    //! Constant ragged lengths, sized at maxBatch so a step never allocates: all-1 queries for a
    //! decode step, all-0 past lengths for a prefill.
    std::vector<int32_t> mTalkerStepQueryLengths;
    std::vector<int32_t> mTalkerStepPastLengths;
    rt::Tensor mExecContextMemory; //!< kUSER_MANAGED context memory for the talker engine.

    CodePredictor mCodePredictor;

    // Weights. The kSegments layout projects text and multimodal rows through 2-layer MLPs;
    // kChunked looks text up in the talker's own embed_tokens and uses a single Linear for
    // the multimodal rows.
    rt::Tensor mTextFC1Weight;
    rt::Tensor mTextFC1Bias;
    rt::Tensor mTextFC2Weight;
    rt::Tensor mTextFC2Bias;
    rt::Tensor mHiddenFC1Weight;
    rt::Tensor mHiddenFC1Bias;
    rt::Tensor mHiddenFC2Weight;
    rt::Tensor mHiddenFC2Bias;
    rt::Tensor mHiddenProjLinearWeight;
    rt::Tensor mHiddenProjLinearBias;

    //! True when the text embeddings live with the backbone rather than in the talker's own
    //! artifact directory. Follows from the hidden projection: only checkpoints that carry
    //! one share the backbone's table.
    bool mTextEmbedFromBackbone{false};

    rt::Tensor mTextEmbeddingTable;   //!< [backboneVocab, thinkerHidden]
    rt::Tensor mTalkerEmbeddingTable; //!< [talkerVocab, talkerHidden]
    //! Talker codec embed table, distinct from mTalkerEmbeddingTable: for the kChunked
    //! layout embed_tokens is text-vocab and the codec table is a separate sidecar.
    rt::Tensor mTalkerCodecEmbedTable;
    //! [maxSpeakerNum, numCodeGroups, speakerEmbedLen] INT64, padded with -1.
    rt::Tensor mSpeakerCodecEmbeddings;

    //! Reference encoders, null unless the artifact directory carries their engines.
    std::unique_ptr<rt::CloneEncoderRunner> mCloneEncoders;
    rt::Tensor mVoiceCloneXVector; //!< [talkerHidden] FP16, the reference speaker row
    rt::Tensor mIclFrameSumBuffer; //!< [maxRefFrames, talkerHidden] FP16, per-frame code sums
    rt::Tensor mIclTablePtrsHost;  //!< Codec embedding table pointers, one per code group
    rt::Tensor mIclTablePtrsGpu;
    //! Row descriptors for the assembly kernel, staged host-side then uploaded once.
    std::vector<kernel::PrefillRowDesc> mPrefillRows;
    rt::Tensor mPrefillDescsHost; //!< Pinned [maxBatch, maxRows * descriptor bytes], one slice per batch slot.
    rt::Tensor mPrefillDescsDevice;

    rt::Tensor mTtsPadEmbed;
    rt::Tensor mTtsBosEmbed;
    rt::Tensor mTtsEosEmbed;
    //! Zero residual addend for the kChunked decode step: its text arrives by re-prefill,
    //! so there is no per-frame text row to add.
    rt::Tensor mZeroResidualAddend;
    //! Persistent FP32 -INF; source for suppressTalkerEosLogit's async D2D copy, which
    //! must not read from a stack variable.
    rt::Tensor mNegInfConst;

    //! invokeSpeakerCodecSum workspace. mCodecEmbPtrTable is INT8-typed because it carries
    //! a `__half const*[]`.
    rt::Tensor mCodecEmbPtrTable;
    rt::Tensor mCodecEmbVocabSizes;
    rt::Tensor mCodecRowCodes;
    //! Pinned [maxInputLength, numCodeGroups] INT64, one upload slot per row of a prefill.
    rt::Tensor mHostCodecRowCodes;
    //! Pinned [numCodeGroups * speakerEmbedLen] INT64, one built-in speaker's code matrix.
    rt::Tensor mHostSpeakerRow;

    // Workspace, all claimed in preallocate().
    rt::Tensor mThinkerEmbedBuffer;
    rt::Tensor mGpuTokenIdsBuffer;
    rt::Tensor mHostTokenIds;      //!< Pinned [maxBatch, maxSeqLen] upload source for mGpuTokenIdsBuffer.
    rt::Tensor mHostGatherIndices; //!< Pinned [maxBatch, maxSeqLen] upload source for mGatherIndicesBuffer.
    rt::Tensor mMLPWorkspace;
    rt::Tensor mProjectedBuffer;
    //! Rows past a segments prefill that its assistant preamble is assembled in.
    static constexpr int64_t kPreambleScratchRows = 11;
    //! [maxInputLength + kPreambleScratchRows, H] assembly scratch; the chunked re-prefill also runs on it.
    rt::Tensor mTalkerInputEmbeds;
    rt::Tensor mTalkerPrefillRows; //!< [maxBatch * maxInputLength, H]; what the prefill engine reads.
    rt::Tensor mGatherIndicesBuffer;
    rt::Tensor mTrailingTextHidden; //!< [max_seq + 1, talkerHidden], segments layout only; assembly scratch.
    rt::Tensor mTrailingRows;       //!< [maxBatch * (max_seq + 1), talkerHidden], one slice per member.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mTalkerLogits;
    rt::Tensor mTalkerSelectedIndices;
    rt::Tensor mHostSelectedTokenIds;
    rt::Tensor mSeenCodecTokensBuf;
    rt::Tensor mHostReuseKVCacheLengths;
    rt::Tensor mTalkerHiddenStatesBuffer;
    rt::Tensor mTalkerLastHidden;
    rt::Tensor mResidualEmbedBuffer;
    //! [maxBatch, numCodesPerFrame, talkerHidden]. Rows 1..numRvqLayers-1 are written by the code
    //! predictor; rows 0 and numRvqLayers are looked up by the residual kernel.
    rt::Tensor mCodecHiddensBuffer;

    std::vector<Row> mRows;
    //! Batch slots of the cohort, index-aligned with the engine's rows. Frozen at prefill:
    //! the row number is the KV state index, so it cannot move when a member finishes.
    std::vector<int32_t> mCohortSlots;
    //! Set once the cohort's prefill has run; the frame loop owns every step after it.
    bool mCohortPrefilled{false};
    //! Decode steps since the cohort's prefill: the index of the next frame the code predictor
    //! runs, from which the talker's and the code predictor's random offsets follow.
    int32_t mCohortStep{0};
    //! Index into stepCohort()'s members of each one still speaking, reused every frame.
    std::vector<size_t> mLive;
    //! The members a cohort's prefill covers, reused per cohort.
    std::vector<multiStage::StageExecContext*> mPrefillMembers;

    // Prefill scratch. prepare() runs one member at a time, so these are not per row.
    int32_t mPrepareSlot{0}; //!< Batch slot of the member prepare() is assembling.
    std::vector<int32_t> mTextTokenIds;
    rt::Tensor mEmbedsView; //!< Non-owning view of the upstream prefill embeddings.
    rt::Tensor mHiddenView; //!< Non-owning view of the tapped accept-layer window.
    //! Chunked text feeding, kept across frames; admit() limits the chunked layout to one member.
    ChunkStreamState mChunkStream;

    // Frame-loop state, indexed by engine row (mCohortSlots order).
    //! The code predictor's codes for the current frame, one entry per engine row.
    std::vector<std::vector<int32_t>> mFrameCodes;
    rt::Tensor mCodeZero; //!< Pinned [maxBatch], the code_0 each engine row is handed this frame.
    //! Per-row lengths of the prefill most recently written into mTalkerHiddenStatesBuffer.
    std::vector<int32_t> mQueryLengths;
    std::vector<int32_t> mPastLengths;
};

std::unique_ptr<multiStage::IStage> createTalkerRunner(multiStage::StageInitContext const& init);
//! What a talker stage offers a stage graph, and how request files spell what it reads.
multiStage::RunnerDecl talkerRunnerDeclaration();

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
