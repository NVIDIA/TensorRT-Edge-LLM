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
//! Two layouts based on whether a language conditioning ID is provided:
//!
//! No-language path (langId < 0):
//!   Total rows = 8 + textLen + 2 (= seqLen + 2). Uses codecNothinkId at row 3.
//!     [0-2]:        projected[0-2]
//!     [3]:          ttsPadEmbed + talkerEmbTable[codecNothinkId]
//!     [4]:          ttsPadEmbed + talkerEmbTable[codecThinkBosId]
//!     [5]:          ttsPadEmbed + talkerEmbTable[codecThinkEosId]
//!     [6]:          ttsPadEmbed + talkerEmbTable[speakerId]
//!     [7]:          ttsBosEmbed + talkerEmbTable[codecPadId]
//!     [8..8+N-1]:   projected[3+i] + talkerEmbTable[codecPad/codecBos]  (last row uses codecBosId)
//!     [8+N]:        ttsEosEmbed + talkerEmbTable[codecPadId]
//!     [8+N+1]:      ttsPadEmbed + talkerEmbTable[codecBosId]
//!
//! Language path (langId >= 0, CustomVoice with language conditioning):
//!   Total rows = 9 + textLen + 2 (= seqLen + 3 = original-seqLen + 2, since N is also +1 upstream).
//!   Uses codecThinkId at row 3 and injects langId at row 5.
//!     [0-2]:        projected[0-2]
//!     [3]:          ttsPadEmbed + talkerEmbTable[codecThinkId]      (NOTE: think, not no-think)
//!     [4]:          ttsPadEmbed + talkerEmbTable[codecThinkBosId]
//!     [5]:          ttsPadEmbed + talkerEmbTable[langId]            (NEW row, language embed)
//!     [6]:          ttsPadEmbed + talkerEmbTable[codecThinkEosId]
//!     [7]:          ttsPadEmbed + talkerEmbTable[speakerId]
//!     [8]:          ttsBosEmbed + talkerEmbTable[codecPadId]
//!     [9..9+N-1]:   projected[3+i] + talkerEmbTable[codecPad/codecBos]  (last row uses codecBosId)
//!     [9+N]:        ttsEosEmbed + talkerEmbTable[codecPadId]
//!     [9+N+1]:      ttsPadEmbed + talkerEmbTable[codecBosId]
//!
//! \param projected      MLP output [seqLen, H] (FP16)
//! \param ttsPadEmbed/ttsBosEmbed/ttsEosEmbed  TTS special embeddings [H] (FP16)
//! \param talkerEmbTable Talker embedding table [vocabSize, H] (FP16)
//! \param codecNothinkId  Codec no-think control token (used when langId < 0)
//! \param codecThinkId    Codec think control token (used when langId >= 0)
//! \param codecThinkBosId/codecThinkEosId/codecPadId/codecBosId  Codec control IDs
//! \param speakerId       Speaker codec token ID
//! \param langId          Language codec token ID; if < 0, no-language path is used
//! \param textLen         Number of text token rows (N)
//! \param output          Full output buffer (FP16)
//! \param stream          CUDA stream
void invokeAssistantPreamble(rt::Tensor const& projected, rt::Tensor const& ttsPadEmbed, rt::Tensor const& ttsBosEmbed,
    rt::Tensor const& ttsEosEmbed, rt::Tensor const& talkerEmbTable, int32_t codecNothinkId, int32_t codecThinkId,
    int32_t codecThinkBosId, int32_t codecThinkEosId, int32_t speakerId, int32_t codecPadId, int32_t codecBosId,
    int32_t langId, int32_t textLen, rt::Tensor& output, cudaStream_t stream);

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

} // namespace kernel
} // namespace trt_edgellm
