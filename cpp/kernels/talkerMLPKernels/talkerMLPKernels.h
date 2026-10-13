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

#include "common/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Two-layer MLP with SiLU activation (Talker projection layers)
//!
//! Performs: output = FC2(SiLU(FC1(input) + bias1)) + bias2
//! Where FC1: [inputDim → hiddenDim], FC2: [hiddenDim → outputDim]
//!
//! Architecture:
//!   input [N, 2048]
//!     ↓ FC1 (Linear)
//!   [N, 2048] + bias1
//!     ↓ SiLU
//!   [N, 2048]
//!     ↓ FC2 (Linear)
//!   [N, 1024] + bias2
//!     ↓
//!   output [N, 1024]
//!
//! GEMM is performed via CuTe DSL compiled kernels (FP16 I/O, FP32 accumulation).
//!
//! \param[in] input Input tensor with shape [numTokens, inputDim] (FP16)
//! \param[in] fc1Weight FC1 weight matrix with shape [hiddenDim, inputDim] (FP16, row-major)
//! \param[in] fc1Bias FC1 bias vector with shape [hiddenDim] (FP16)
//! \param[in] fc2Weight FC2 weight matrix with shape [outputDim, hiddenDim] (FP16, row-major)
//! \param[in] fc2Bias FC2 bias vector with shape [outputDim] (FP16)
//! \param[out] output Output tensor with shape [numTokens, outputDim] (FP16)
//! \param[in,out] workspace Workspace buffer for intermediate FC1 output [numTokens, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeTalkerMLP(rt::Tensor const& input, rt::Tensor const& fc1Weight, rt::Tensor const& fc1Bias,
    rt::Tensor const& fc2Weight, rt::Tensor const& fc2Bias, rt::Tensor& output, rt::Tensor& workspace,
    cudaStream_t stream);

//! \brief Single linear layer: output = input @ weight.T + bias
//!
//! GEMM is performed via CuTe DSL compiled kernels (FP16 I/O, FP32 accumulation).
//!
//! \param[in] input Input tensor with shape [N, inputDim] (FP16)
//! \param[in] weight Weight matrix with shape [outputDim, inputDim] (FP16, row-major)
//! \param[in] bias Bias vector with shape [outputDim] (FP16)
//! \param[out] output Output tensor with shape [N, outputDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeLinearLayer(
    rt::Tensor const& input, rt::Tensor const& weight, rt::Tensor const& bias, rt::Tensor& output, cudaStream_t stream);

//! \brief Gather operation: select rows from source tensor by indices
//!
//! Performs: output[i] = source[indices[i]]
//! where each row has hiddenDim elements.
//!
//! \param[in] source Source tensor with shape [srcNumTokens, hiddenDim] (FP16)
//! \param[in] indices Indices tensor with shape [numIndices] (INT32)
//! \param[out] output Output tensor with shape [numIndices, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeGather(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream);

//! \brief Scatter operation: place rows from source to output by indices
//!
//! Performs: output[indices[i]] = source[i]
//! where each row has hiddenDim elements.
//!
//! \param[in] source Source tensor with shape [numIndices, hiddenDim] (FP16)
//! \param[in] indices Indices tensor with shape [numIndices] (INT32)
//! \param[out] output Output tensor with shape [dstNumTokens, hiddenDim] (FP16)
//! \param[in] stream CUDA stream for execution
void invokeScatter(rt::Tensor const& source, rt::Tensor const& indices, rt::Tensor& output, cudaStream_t stream);

//! \brief Fused non-streaming assistant preamble construction for TTS input projection
//!
//! Builds the complete non-streaming prefill buffer in one pass.
//! Total rows written = P + textLen + 2, where P = 8 without language conditioning or
//! P = 9 when languageId >= 0 (CustomVoice language conditioning inserts one extra row).
//!
//! Row layout without language (P = 8, byte-identical to the historical layout):
//!   [0-2]:        projected[0-2]                            (role tokens)
//!   [3]:          ttsPadEmbed + talkerEmbTable[codecNothinkId]
//!   [4]:          ttsPadEmbed + talkerEmbTable[codecThinkBosId]
//!   [5]:          ttsPadEmbed + talkerEmbTable[codecThinkEosId]
//!   [6]:          ttsPadEmbed + talkerEmbTable[speakerId]
//!   [7]:          ttsBosEmbed + talkerEmbTable[codecPadId]
//!
//! Row layout with language (P = 9; think-token replaces the no-think token, language row inserted):
//!   [0-2]:        projected[0-2]
//!   [3]:          ttsPadEmbed + talkerEmbTable[codecThinkId]
//!   [4]:          ttsPadEmbed + talkerEmbTable[codecThinkBosId]
//!   [5]:          ttsPadEmbed + talkerEmbTable[languageId]
//!   [6]:          ttsPadEmbed + talkerEmbTable[codecThinkEosId]
//!   [7]:          ttsPadEmbed + talkerEmbTable[speakerId]
//!   [8]:          ttsBosEmbed + talkerEmbTable[codecPadId]
//!
//! Shared text/suffix rows:
//!   [P..P+N-2]:   projected[3+i] + talkerEmbTable[codecPadId]  (text tokens, N=textLen)
//!   [P+N-1]:      projected[3+N-1] + talkerEmbTable[codecBosId]  (last text = start-of-generation)
//!   [P+N]:        ttsEosEmbed + talkerEmbTable[codecPadId]
//!   [P+N+1]:      ttsPadEmbed + talkerEmbTable[codecBosId]
//!
//! \param projected      MLP output [seqLen, H] (FP16)
//! \param ttsPadEmbed/ttsBosEmbed/ttsEosEmbed  TTS special embeddings [H] (FP16)
//! \param talkerEmbTable Talker embedding table [vocabSize, H] (FP16)
//! \param codecNothinkId..codecBosId  Codec token IDs used in the prefix/suffix rows
//! \param speakerId      Speaker codec token ID
//! \param codecThinkId   Codec think token ID (used instead of codecNothinkId when languageId >= 0;
//!                       ignored otherwise, pass -1 if unavailable)
//! \param languageId     Language codec token ID; -1 disables language conditioning (8-row prefix)
//! \param textLen        Number of text token rows (N)
//! \param output         Full output buffer [P+N+2, H] (FP16)
//! \param stream         CUDA stream
void invokeAssistantPreamble(rt::Tensor const& projected, rt::Tensor const& ttsPadEmbed, rt::Tensor const& ttsBosEmbed,
    rt::Tensor const& ttsEosEmbed, rt::Tensor const& talkerEmbTable, int32_t codecNothinkId, int32_t codecThinkBosId,
    int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId, int32_t codecBosId, int32_t codecThinkId,
    int32_t languageId, int32_t textLen, rt::Tensor& output, cudaStream_t stream);

//! \brief One prefill output row = srcA (+ srcB when non-null), both device pointers to [H] rows.
//!
//! The per-row math is identical to invokeAssistantPreamble (vectorized half2 add, same order),
//! so rows described identically produce bit-identical output.
struct PrefillRowDesc
{
    half const* srcA; //!< Required source row [hiddenDim]
    half const* srcB; //!< Optional addend row [hiddenDim]; nullptr = copy srcA only
};

//! \brief Descriptor-driven prefill row assembly (generalization of invokeAssistantPreamble)
//!
//! Assembles arbitrary prefill layouts (instruction segments, no-speaker VoiceDesign prefixes,
//! continuous speaker embeddings, ICL segments) from a host-built row descriptor list.
//! One block per row; each row is srcA (+ srcB) with the same vectorized half2 add as
//! invokeAssistantPreamble.
//!
//! \param deviceDescs Device array of numRows descriptors (uploaded by the caller)
//! \param numRows     Number of output rows
//! \param hiddenDim   Row width (must be a multiple of 8)
//! \param output      Output buffer [numRows, hiddenDim] (FP16)
//! \param stream      CUDA stream
void invokePrefillRowAssemble(
    PrefillRowDesc const* deviceDescs, int32_t numRows, int32_t hiddenDim, rt::Tensor& output, cudaStream_t stream);

//! \brief Sum per-frame codec embeddings across all code groups (voice-clone ICL prompt)
//!
//! For each reference frame t: output[t] = sum_{g=0}^{numGroups-1} tables[g][codes[t][g]].
//! Group 0 uses the Talker codec embedding table; groups 1..numGroups-1 the CodePredictor
//! tables — the caller passes one device pointer per group. Accumulation in FP32.
//!
//! \param refCodes    Device codes [numFrames, numGroups] (INT32)
//! \param tablePtrs   Device array of numGroups table pointers, each [vocab, hiddenDim] (FP16)
//! \param numFrames   Reference frame count
//! \param numGroups   Code group count (16 for the 12Hz tokenizer)
//! \param hiddenDim   Embedding width
//! \param output      Output [numFrames, hiddenDim] (FP16)
//! \param stream      CUDA stream
//! refCodes is INT64 so the codec-encoder engine output is consumed in place (no host
//! round-trip or dtype conversion).
void invokeSumCodecEmbeddings(int64_t const* refCodes, half const* const* tablePtrs, int32_t numFrames,
    int32_t numGroups, int32_t hiddenDim, rt::Tensor& output, cudaStream_t stream);

//! \brief Score hidden rows against a per-row selection of stacked lm_heads.
//!
//! output[r] = heads[rowHeadIndices[r]] * hiddens[rowHiddenIndices[r]], for r in [0, rows).
//! The CodePredictor scores each verify position with the lm_head of its own RVQ
//! depth, which one plain GEMM per row cannot express.
//!
//! Both selectors are per-row tensors rather than a base plus offset, so a
//! tree-shaped verify window can pass node depths unchanged.
//!
//! \param hiddens          Hidden rows [*, hiddenDim] (FP16)
//! \param heads            Stacked heads [numHeads, outputDim, hiddenDim] (FP16, row-major)
//! \param rowHiddenIndices Row selector into hiddens [rows] (INT32)
//! \param rowHeadIndices   Row selector into heads [rows] (INT32)
//! \param rows             Output row count
//! \param hiddenDim        Reduction width (must be a multiple of 2)
//! \param outputDim        Head output width
//! \param output           Output [rows, outputDim] (FP16)
//! \param stream           CUDA stream
void invokeGroupedHeadLinear(half const* hiddens, rt::Tensor const& heads, int32_t const* rowHiddenIndices,
    int32_t const* rowHeadIndices, int32_t rows, int32_t hiddenDim, int32_t outputDim, rt::Tensor& output,
    cudaStream_t stream);

//! \brief Gather one codec embedding per verify position, each from its own RVQ table.
//!
//! output[b][i] = tables[tableIndices[b][i]][codeIds[b][i * ... ]] for i in [0, rows).
//! Collapses the per-position embeddingLookup + copy chain the CodePredictor verify
//! window would otherwise issue, in the same spirit as invokeResidualConnection.
//!
//! tableIndices is a per-row selector rather than a base+offset pair so a tree-shaped
//! verify window can pass its node depths unchanged.
//!
//! \param codeIds      Device code ids [batch, codeStride] (INT32); position i reads column i
//! \param tablePtrs    Device array of per-RVQ-depth table pointers, each [codebook, hiddenDim] (FP16)
//! \param tableIndices Per-row table selector [batch, rows] (INT32, device)
//! \param batchSize    Active batch size
//! \param rows         Verify positions per batch
//! \param codeStride   Column stride of codeIds
//! \param hiddenDim    Embedding width
//! \param output       Output [batch, rows, hiddenDim] (FP16), or the scatter target when
//!                     outputRowIndices is supplied
//! \param stream       CUDA stream
//! \param outputRowIndices Optional per-row destination selector [batch, rows] (INT32, device).
//!                     nullptr packs the rows contiguously; a negative entry drops its row, which
//!                     lets one launch cover a ragged per-batch row count.
void invokeGatherCodecEmbedRows(rt::Tensor const& codeIds, half const* const* tablePtrs, int32_t const* tableIndices,
    int32_t batchSize, int32_t rows, int32_t codeStride, int32_t hiddenDim, rt::Tensor& output, cudaStream_t stream,
    int32_t const* outputRowIndices = nullptr);

//! \brief Elementwise FP32 -> FP16 cast on device (small utility for engine-output adaptation)
void invokeCastFp32ToFp16(float const* input, half* output, int64_t numElements, cudaStream_t stream);

//! \brief Fused residual connection for TTS decode input
//!
//! Computes: output = embed0[code0] + embedLast[codeLast] + addend + sum(codecHiddens[1..N-1])
//! where N = numCodesPerFrame (inferred from codecHiddens shape).
//! Eliminates 7 separate dispatches (2x H→D, 2x embLookup, 2x D→D, sumReduce) in one kernel.
//!
//! \param codecHiddens   [1, numCodesPerFrame, H] buffer — inner rows pre-filled by CodePredictor (FP16)
//! \param embTable0      Talker embedding table [vocabSize, H] (FP16) — for embed(code0)
//! \param embTableLast   CodePredictor embedding table[-1] [vocabSize, H] (FP16) — for embed(codeLast)
//! \param code0/codeLast Token IDs passed as scalars (no H→D upload needed)
//! \param addend         Row pointer [H] — trailing_text_hidden[generationStep] or tts_pad_embed (FP16)
//! \param output         Output tensor [1, 1, H] (FP16)
//! \param stream         CUDA stream
void invokeResidualConnection(rt::Tensor const& codecHiddens, rt::Tensor const& embTable0,
    rt::Tensor const& embTableLast, int32_t code0, int32_t codeLast, half const* addend, rt::Tensor& output,
    cudaStream_t stream);

//! \brief Adjust Talker logits: suppress special tokens and apply repetition penalty.
//!
//! Performs two in-place modifications on the logits before sampling:
//!   1. Suppression: sets logits[i] = -inf for all i in [suppressStart, suppressEnd),
//!      except for codecEosId which is always preserved.
//!   2. Repetition penalty: for each token in seenTokens[], divides positive logits by
//!      repetitionPenalty and multiplies negative logits by repetitionPenalty, matching
//!      the HuggingFace repetition_penalty convention.
//!
//! Operates on FP32 logits tensor with shape [1, vocabSize].
//!
//! \param[in] seenTokens          GPU tensor of previously generated token IDs [maxAudioLength] INT32
//! \param[in,out] logits          Logits tensor [1, vocabSize] (FP32, in-place)
//! \param[in] suppressStart       Start of suppress range (inclusive)
//! \param[in] suppressEnd         End of suppress range (exclusive)
//! \param[in] codecEosId          Token ID exempt from suppression (EOS must remain samplable)
//! \param[in] numSeenTokens       Number of valid entries in seenTokens (0 to disable penalty)
//! \param[in] repetitionPenalty   Penalty factor >= 1.0 (1.0 = no penalty)
//! \param[in] stream              CUDA stream for execution
void invokeTalkerLogitAdjust(rt::Tensor const& seenTokens, rt::Tensor& logits, int32_t suppressStart,
    int32_t suppressEnd, int32_t codecEosId, int32_t numSeenTokens, float repetitionPenalty, cudaStream_t stream);

//! Per-position sum across codec groups: out[d] = sum_g embPtrTable[g][codes[g]][d];
//! codes[g] < 0 skips group g. embPtrTable is INT8-typed to carry `__half const*[numCodeGroups]`
//! (rt::Tensor has no pointer-array dtype). numCodeGroups = codes.shape[0]; hiddenSize = output.volume().
void invokeSpeakerCodecSum(rt::Tensor const& codes, rt::Tensor const& embPtrTable, rt::Tensor const& embVocabSizes,
    rt::Tensor& output, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
