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

#include "multimodal/qwen3_omni/talkerRunner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

namespace
{

//! Assistant prefix tokens stripped from a templated segment ("<|im_start|>assistant\n").
constexpr int32_t kAssistantPrefixLen = 3;
//! Fixed prefix rows the assistant preamble kernel writes.
constexpr int32_t kNonStreamingPrefixRows = 8;
//! "<|im_end|>\n<|im_start|>assistant\n" -- stripped from the tail before the text rows.
constexpr int32_t kAssistantTrailingSuffix = 5;
//! Language conditioning inserts one row into the think block.
constexpr int32_t kPrefixRowsWithLanguage = kNonStreamingPrefixRows + 1;

//! Text rows fed per chunked re-prefill call (talker_text_in_chunk_n).
constexpr int32_t kTextInChunkN = 4;

//! Async H2D of @p count INT32 values staged through row @p slot of the pinned [maxBatch, n]
//! @p staging. Cohort members prepare back to back with no sync between them, so each stages
//! through its own row; a row is reused only after the sync that ends the cohort's prefill.
//! @p device must already be shaped to hold them.
void uploadThroughPinned(
    int32_t const* values, int64_t count, rt::Tensor& staging, int32_t slot, rt::Tensor& device, cudaStream_t stream)
{
    auto const& shape = staging.getShape();
    ELLM_CHECK(shape.getNumDims() == 2 && slot >= 0 && slot < shape[0] && count <= shape[1],
        std::to_string(count) + " values for slot " + std::to_string(slot) + " exceed the pinned staging '"
            + staging.getName() + "'");
    auto* const row = staging.dataPointer<int32_t>() + static_cast<int64_t>(slot) * shape[1];
    auto const bytes = static_cast<size_t>(count) * sizeof(int32_t);
    std::memcpy(row, values, bytes);
    CUDA_CHECK(cudaMemcpyAsync(device.rawPointer(), row, bytes, cudaMemcpyHostToDevice, stream));
}

} // namespace

// ---------------------------------------------------------------------------
// Segments layout
// ---------------------------------------------------------------------------

bool TalkerRunner::buildSegmentsPrefill(rt::Tensor const* prefillEmbeds, rt::Tensor const* prefillHidden,
    int32_t prefillLen, int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount, int64_t& outSeqLen,
    cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::buildSegmentsPrefill", nvtx_colors::PALE_GREEN);

    auto const& textTokenIds = mTextTokenIds;
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    // Layer-0 embeddings: the prefill window comes from the backbone, the generated tail is
    // looked up here.
    ELLM_CHECK(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    if (prefillEmbeds != nullptr)
    {
        int32_t const copyLen = std::min(prefillLen, static_cast<int32_t>(seqLen));
        size_t const prefillBytes = copyLen * thinkerHiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(mThinkerEmbedBuffer.rawPointer(), prefillEmbeds->rawPointer(), prefillBytes,
            cudaMemcpyDeviceToDevice, stream));

        int32_t const genLen = static_cast<int32_t>(seqLen) - copyLen;
        if (genLen > 0)
        {
            ELLM_CHECK(mGpuTokenIdsBuffer.reshape({1, genLen}), "Tensor reshape failed");
            uploadThroughPinned(
                textTokenIds.data() + copyLen, genLen, mHostTokenIds, mPrepareSlot, mGpuTokenIdsBuffer, stream);
            __half* const genDst = static_cast<__half*>(mThinkerEmbedBuffer.rawPointer()) + copyLen * thinkerHiddenSize;
            rt::Tensor genEmbedView(genDst, rt::Coords{1, static_cast<int64_t>(genLen), thinkerHiddenSize},
                rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, genEmbedView, stream);
        }
    }
    else
    {
        ELLM_CHECK(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
        uploadThroughPinned(textTokenIds.data(), seqLen, mHostTokenIds, mPrepareSlot, mGpuTokenIdsBuffer, stream);
        rt::Tensor embedView(mThinkerEmbedBuffer.rawPointer(), rt::Coords{1, seqLen, thinkerHiddenSize},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, embedView, stream);
    }

    ELLM_CHECK(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    ELLM_CHECK(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mProjectedBuffer, mMLPWorkspace, stream);

    std::vector<SegmentInfo> segments;
    for (int64_t i = 0; i < seqLen; ++i)
    {
        if (textTokenIds[static_cast<size_t>(i)] == mTokenIds.imStart)
        {
            int32_t const roleId = (i + 1 < seqLen) ? textTokenIds[static_cast<size_t>(i) + 1] : -1;
            segments.push_back({i, seqLen, roleId});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].endPos = segments[s + 1].startPos;
    }

    std::vector<size_t> userSegmentIndices;
    int64_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == mTokenIds.systemRole)
        {
            continue;
        }
        if (segments[s].roleId == mTokenIds.userRole)
        {
            userSegmentIndices.push_back(s);
        }
        else if (segments[s].roleId == mTokenIds.assistantRole)
        {
            assistantSegIdx = static_cast<int64_t>(s);
        }
    }
    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildSegmentsPrefill: could not find the assistant segment");
        return false;
    }

    constexpr int32_t kAssistantRestructuredLen = kNonStreamingPrefixRows + 1;
    constexpr int32_t kAssistantTrailingOffset = kAssistantPrefixLen + 1;

    int64_t userTotalLen = 0;
    for (size_t sIdx : userSegmentIndices)
    {
        userTotalLen += (segments[sIdx].endPos - segments[sIdx].startPos);
    }
    // Before anything is gathered into the assembly scratch, which holds one engine input.
    int64_t const maxPrefillLen = mTalkerLLMConfig.maxSupportedInputLength;
    ELLM_CHECK(userTotalLen + kAssistantRestructuredLen <= maxPrefillLen,
        "stage '" + mStageId + "': talker prefill of " + std::to_string(userTotalLen + kAssistantRestructuredLen)
            + " rows exceeds the engine's max input length " + std::to_string(maxPrefillLen));

    bool const hasHiddenProjection = mHiddenFC1Weight.getShape().volume() > 0;
    if (hasHiddenProjection && prefillHidden != nullptr)
    {
        std::vector<int32_t> mmPositions;
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& segment = segments[sIdx];
            for (int64_t i = segment.startPos; i < segment.endPos && i < prefillLen; ++i)
            {
                int32_t const tid = textTokenIds[static_cast<size_t>(i)];
                if (tid == mTokenIds.audioPad || tid == mTokenIds.imagePad || tid == mTokenIds.videoPad)
                {
                    mmPositions.push_back(static_cast<int32_t>(i));
                }
            }
        }

        if (!mmPositions.empty())
        {
            int64_t const numMM = static_cast<int64_t>(mmPositions.size());
            // mThinkerEmbedBuffer is free again: its layer-0 contents were already projected
            // into mProjectedBuffer above, and it is only scratch for the gather/MLP/scatter
            // chain from here on.
            ELLM_CHECK(mThinkerEmbedBuffer.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            ELLM_CHECK(mMLPWorkspace.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            ELLM_CHECK(mTalkerInputEmbeds.reshape({numMM, hiddenSize}), "Tensor reshape failed");
            ELLM_CHECK(mGatherIndicesBuffer.reshape({numMM}), "Tensor reshape failed");
            uploadThroughPinned(
                mmPositions.data(), numMM, mHostGatherIndices, mPrepareSlot, mGatherIndicesBuffer, stream);

            rt::Tensor hiddenSource(const_cast<void*>(prefillHidden->rawPointer()),
                rt::Coords{static_cast<int64_t>(prefillLen), thinkerHiddenSize}, rt::DeviceType::kGPU,
                trt_edgellm::DataType::kHALF);
            kernel::invokeGather(hiddenSource, mGatherIndicesBuffer, mThinkerEmbedBuffer, stream);
            kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                mHiddenFC2Bias, mTalkerInputEmbeds, mMLPWorkspace, stream);
            kernel::invokeScatter(mTalkerInputEmbeds, mGatherIndicesBuffer, mProjectedBuffer, stream);
        }
    }
    else
    {
        // Without a projection the multimodal rows have no talker-space value; zero is the
        // documented fallback.
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& segment = segments[sIdx];
            for (int64_t i = segment.startPos; i < segment.endPos; ++i)
            {
                int32_t const tid = textTokenIds[static_cast<size_t>(i)];
                if (tid == mTokenIds.audioPad || tid == mTokenIds.imagePad || tid == mTokenIds.videoPad)
                {
                    __half* const rowPtr = static_cast<__half*>(mProjectedBuffer.rawPointer()) + i * hiddenSize;
                    CUDA_CHECK(cudaMemsetAsync(rowPtr, 0, hiddenSize * sizeof(__half), stream));
                }
            }
        }
    }

    auto const& assistantSeg = segments[static_cast<size_t>(assistantSegIdx)];
    outSeqLen = userTotalLen + kAssistantRestructuredLen;
    ELLM_CHECK(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(mTalkerInputEmbeds.rawPointer(), 0, outSeqLen * hiddenSize * sizeof(__half), stream));

    int64_t outOffset = 0;
    for (size_t sIdx : userSegmentIndices)
    {
        auto const& segment = segments[sIdx];
        int64_t const segLen = segment.endPos - segment.startPos;
        __half const* const src
            = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + segment.startPos * hiddenSize;
        __half* const dst = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outOffset * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, segLen * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        outOffset += segLen;
    }

    {
        __half const* const assistantProjPtr
            = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + assistantSeg.startPos * hiddenSize;
        constexpr int64_t kPreambleFullLen = kNonStreamingPrefixRows + 1 + 2;
        static_assert(kPreambleFullLen == kPreambleScratchRows, "allocateBuffers() sizes the scratch for this");
        int64_t const requiredCapacity = (outSeqLen + kPreambleFullLen) * hiddenSize * sizeof(__half);
        ELLM_CHECK(static_cast<int64_t>(mTalkerInputEmbeds.getMemoryCapacity()) >= requiredCapacity,
            "mTalkerInputEmbeds too small for the preamble scratch");
        __half* const scratchPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outSeqLen * hiddenSize;
        rt::Tensor preambleScratch(
            scratchPtr, rt::Coords{kPreambleFullLen, hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

        int64_t const assistantInputLen = kAssistantPrefixLen + 1;
        rt::Tensor assistantSlice(const_cast<__half*>(assistantProjPtr), rt::Coords{assistantInputLen, hiddenSize},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

        // This layout stays language-free: its checkpoints carry no codec_think_id.
        kernel::invokeAssistantPreamble(assistantSlice, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
            mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
            mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, /*codecThinkId=*/-1, /*languageId=*/-1, 1,
            preambleScratch, stream);

        __half* const preambleOut = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + userTotalLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(preambleOut, scratchPtr, kAssistantRestructuredLen * hiddenSize * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }

    int64_t const assistantSegLen = assistantSeg.endPos - assistantSeg.startPos;
    trailingCount = std::min(static_cast<int32_t>(assistantSegLen - kAssistantTrailingOffset),
        static_cast<int32_t>(trailingTextHidden.getShape()[0]) - 1);
    if (trailingCount > 0)
    {
        __half const* const trailSrc = static_cast<__half const*>(mProjectedBuffer.rawPointer())
            + (assistantSeg.startPos + kAssistantTrailingOffset) * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(trailingTextHidden.rawPointer(), trailSrc,
            trailingCount * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    else
    {
        trailingCount = 0;
    }

    LOG_INFO("buildSegmentsPrefill: outSeqLen=%ld (user=%ld + assistant=%d), trailing=%d", outSeqLen, userTotalLen,
        kAssistantRestructuredLen, trailingCount);
    return true;
}

void TalkerRunner::finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half* const dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mTtsEosEmbed.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

// ---------------------------------------------------------------------------
// Chunked layout
// ---------------------------------------------------------------------------

bool TalkerRunner::copyEmbedRow(
    rt::Tensor const& table, int32_t tokenId, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const vocab = table.getShape()[0];
    int64_t const hiddenSize = table.getShape()[1];
    if (tokenId < 0 || tokenId >= vocab)
    {
        LOG_ERROR("Embed lookup OOB: tokenId=%d vocab=%ld", tokenId, vocab);
        return false;
    }
    __half const* const src
        = static_cast<__half const*>(table.rawPointer()) + static_cast<int64_t>(tokenId) * hiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(dstBase + dstRow * hiddenSize, src, static_cast<size_t>(hiddenSize) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream));
    return true;
}

void TalkerRunner::copyRawRow(rt::Tensor const& src, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(dstBase + dstRow * hiddenSize, src.rawPointer(),
        static_cast<size_t>(hiddenSize) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

void TalkerRunner::sumSpeakerCodecRow(
    int64_t const* hostCodes, int32_t slot, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    int64_t const rowBytes = numCodeGroups * static_cast<int64_t>(sizeof(int64_t));
    ELLM_CHECK(slot >= 0 && (slot + 1) * rowBytes <= mHostCodecRowCodes.getMemoryCapacity(),
        "speaker-codec row slot " + std::to_string(slot) + " is outside the pinned staging");
    int64_t* const hostSlot = mHostCodecRowCodes.dataPointer<int64_t>() + static_cast<int64_t>(slot) * numCodeGroups;
    std::copy_n(hostCodes, numCodeGroups, hostSlot);
    // One device row suffices: stream order keeps each upload behind the previous row's kernel.
    CUDA_CHECK(cudaMemcpyAsync(
        mCodecRowCodes.rawPointer(), hostSlot, static_cast<size_t>(rowBytes), cudaMemcpyHostToDevice, stream));
    rt::Tensor rowOut(
        dstBase + dstRow * hiddenSize, rt::Coords{hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    kernel::invokeSpeakerCodecSum(mCodecRowCodes, mCodecEmbPtrTable, mCodecEmbVocabSizes, rowOut, stream);
}

void TalkerRunner::projectHiddenRow(
    rt::Tensor const& prefillHidden, int64_t srcRow, __half* dstBase, int64_t dstRow, cudaStream_t stream)
{
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half const* const src = static_cast<__half const*>(prefillHidden.rawPointer()) + srcRow * thinkerHiddenSize;
    rt::Tensor sourceSlice(
        const_cast<__half*>(src), rt::Coords{1, thinkerHiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor outSlice(dstBase + dstRow * talkerHiddenSize, rt::Coords{1, talkerHiddenSize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kHALF);
    kernel::invokeLinearLayer(sourceSlice, mHiddenProjLinearWeight, mHiddenProjLinearBias, outSlice, stream);
}

/*
 * Row sequence, mirroring HF _get_talker_{system,user,assistant}_parts:
 *   sys:  [im_start, system, nl] + system_instruct + speaker_system_prompt +
 *         codec_bos + speaker_codec_sum[..] + codec_eos + [im_end, nl]
 *   usr:  role tokens through the text embed table, content tokens through hidden_projection
 *   asst: [im_start, assistant, nl] + assistant_instruct
 *         + [codec_think, codec_think_bos, language, codec_think_eos] when a language resolves,
 *           else [codec_nothink, codec_think_bos, codec_think_eos]
 *         + tts_bos + codec_bos + the first text chunk (up to kTextInChunkN rows).
 * Remaining text chunks and the trailing tts_eos go to mChunkStream for the re-prefills.
 */
bool TalkerRunner::buildChunkedPrefill(
    rt::Tensor const* prefillHidden, int32_t prefillLen, int32_t speakerId, int64_t& outSeqLen, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::buildChunkedPrefill", nvtx_colors::PALE_GREEN);

    auto const& textTokenIds = mTextTokenIds;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    int32_t const codecBosId = mTalkerConfig.codecBosId;
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // A reference voice replaces the built-in speaker's rows outright, so a checkpoint that
    // ships no speaker table can still clone.
    bool const hasCustomVoice = !mPromptSpeakerCodes.empty();
    if (mTalkerCodecEmbedTable.rawPointer() == nullptr
        || (!hasCustomVoice && mSpeakerCodecEmbeddings.rawPointer() == nullptr))
    {
        LOG_ERROR("The chunked prefill needs the codec_embedding and speaker_codec_embeddings sidecars");
        return false;
    }

    int32_t speakerEmbedLen = 0;
    std::vector<int64_t> speakerRow;
    if (hasCustomVoice)
    {
        speakerEmbedLen = static_cast<int32_t>(mPromptSpeakerCodes.size());
        speakerRow.assign(static_cast<size_t>(numCodeGroups) * speakerEmbedLen, -1);
        for (int32_t p = 0; p < speakerEmbedLen; ++p)
        {
            auto const& frame = mPromptSpeakerCodes[static_cast<size_t>(p)];
            if (static_cast<int32_t>(frame.size()) != numCodeGroups)
            {
                LOG_ERROR("speaker_codes frame %d has %zu codes, expected %d", p, frame.size(), numCodeGroups);
                return false;
            }
            for (int32_t g = 0; g < numCodeGroups; ++g)
            {
                speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p] = frame[static_cast<size_t>(g)];
            }
        }
        LOG_INFO("Chunked prefill: cloning from %d reference frames; the built-in speaker rows are skipped",
            speakerEmbedLen);
    }
    else
    {
        auto const& speakerShape = mSpeakerCodecEmbeddings.getShape();
        int32_t const maxSpeakerNum = static_cast<int32_t>(speakerShape[0]);
        int32_t const numGroupsSpk = static_cast<int32_t>(speakerShape[1]);
        speakerEmbedLen = static_cast<int32_t>(speakerShape[2]);
        if (speakerId < 0 || speakerId >= maxSpeakerNum)
        {
            LOG_ERROR("speakerId %d out of range [0, %d)", speakerId, maxSpeakerNum);
            return false;
        }
        if (numGroupsSpk != numCodeGroups)
        {
            LOG_ERROR("speaker_codec_embeddings groups (%d) != numCodeGroups (%d)", numGroupsSpk, numCodeGroups);
            return false;
        }

        // Pull the speaker's code matrix to the host once and trim its -1 padding.
        int64_t const speakerRowBytes = static_cast<int64_t>(numCodeGroups) * speakerEmbedLen * sizeof(int64_t);
        CUDA_CHECK(cudaMemcpyAsync(mHostSpeakerRow.rawPointer(),
            static_cast<int64_t const*>(mSpeakerCodecEmbeddings.rawPointer())
                + static_cast<int64_t>(speakerId) * numCodeGroups * speakerEmbedLen,
            speakerRowBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const* const hostSpeakerRow = mHostSpeakerRow.dataPointer<int64_t>();
        speakerRow.assign(hostSpeakerRow, hostSpeakerRow + static_cast<size_t>(numCodeGroups) * speakerEmbedLen);
    }

    int32_t validSpeakerLen = 0;
    for (int32_t p = speakerEmbedLen - 1; p >= 0; --p)
    {
        bool any = false;
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            if (speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p] != -1)
            {
                any = true;
                break;
            }
        }
        if (any)
        {
            validSpeakerLen = p + 1;
            break;
        }
    }
    if (validSpeakerLen == 0)
    {
        validSpeakerLen = speakerEmbedLen;
    }

    constexpr int32_t kSysRolePrefix = 3;  // [im_start, system, nl]
    constexpr int32_t kSysSuffix = 2;      // [im_end, nl]
    constexpr int32_t kAsstRolePrefix = 3; // [im_start, assistant, nl]

    std::string const& assistantInstruct = mRequestInstruct.empty() ? mAssistantInstruct : mRequestInstruct;
    std::vector<int32_t> const* instructIds = nullptr;
    if (hasCustomVoice && !assistantInstruct.empty())
    {
        // The reference voice already carries the style; the two conditionings do not compose.
        LOG_WARNING("assistant_instruct '%s' ignored alongside speaker_codes", assistantInstruct.c_str());
    }
    else if (!assistantInstruct.empty())
    {
        if (auto const it = mAssistantPromptIds.find(assistantInstruct); it != mAssistantPromptIds.end())
        {
            instructIds = &it->second;
        }
        else
        {
            LOG_WARNING("assistant_instruct '%s' not in talker_assistant_prompt_id_mapping; ignored",
                assistantInstruct.c_str());
        }
    }
    int32_t const instructRows = instructIds ? static_cast<int32_t>(instructIds->size()) : 0;

    // A valid language swaps the codec-special trio for the 4-row think block
    // [codec_think, think_bos, LANGUAGE_ID, think_eos].
    int32_t languageCodecId = -1;
    std::string const& talkerLanguage = mRequestTalkerLanguage.empty() ? mTalkerLanguage : mRequestTalkerLanguage;
    if (!talkerLanguage.empty())
    {
        auto const lang = lowered(talkerLanguage);
        if (lang != "auto")
        {
            if (auto const it = mLanguageIds.find(lang); it != mLanguageIds.end() && mTalkerConfig.codecThinkId >= 0)
            {
                languageCodecId = it->second;
            }
            else
            {
                LOG_WARNING(
                    "talker_language '%s' not in talker_language_id mapping; using auto", talkerLanguage.c_str());
            }
        }
    }
    int32_t const codecSpecialRows = languageCodecId >= 0 ? 4 : 3;

    // The per-speaker prompt names a built-in voice, so it has nothing to say about a cloned one.
    std::vector<int32_t> const* speakerSysPrompt = nullptr;
    if (!hasCustomVoice)
    {
        if (auto const it = mSpeakerSystemPromptIds.find(speakerId); it != mSpeakerSystemPromptIds.end())
        {
            speakerSysPrompt = &it->second;
        }
    }
    int32_t const sysPromptLen = speakerSysPrompt ? static_cast<int32_t>(speakerSysPrompt->size()) : 0;
    int32_t const sysInstructRows = static_cast<int32_t>(mSystemInstructIds.size());
    int32_t const systemRows = kSysRolePrefix + sysInstructRows + sysPromptLen + 1 /*codec_bos*/ + validSpeakerLen
        + 1 /*codec_eos*/ + kSysSuffix;

    struct ChatSegment
    {
        int32_t start;
        int32_t end;
        int32_t roleId;
    };
    std::vector<ChatSegment> segments;
    for (size_t i = 0; i < textTokenIds.size(); ++i)
    {
        if (textTokenIds[i] == mTokenIds.imStart && i + 1 < textTokenIds.size())
        {
            segments.push_back(
                {static_cast<int32_t>(i), static_cast<int32_t>(textTokenIds.size()), textTokenIds[i + 1]});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].end = segments[s + 1].start;
    }
    int32_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == mTokenIds.assistantRole)
        {
            assistantSegIdx = static_cast<int32_t>(s);
        }
    }
    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildChunkedPrefill: assistant role marker not found in the token stream");
        return false;
    }

    // add_generation_prompt already emitted the assistant role and think block, so prefillLen
    // sits at the first generated content token.
    auto const& assistantSeg = segments[static_cast<size_t>(assistantSegIdx)];
    int32_t const assistantTextStart = prefillLen;
    int32_t assistantTextEnd = assistantSeg.end;
    for (int32_t i = assistantTextStart; i < assistantSeg.end; ++i)
    {
        if (textTokenIds[static_cast<size_t>(i)] == mTokenIds.imEnd)
        {
            assistantTextEnd = i;
            break;
        }
    }
    int32_t const assistantTextLen = std::max(0, assistantTextEnd - assistantTextStart);
    int32_t const totalTextLen = assistantTextLen + 1; // +1 for the appended tts_eos
    int32_t const firstChunkLen = std::min(totalTextLen, kTextInChunkN);

    int32_t const assistantRows
        = kAsstRolePrefix + instructRows + codecSpecialRows + 1 /*tts_bos*/ + 1 /*codec_bos*/ + firstChunkLen;

    auto const isUserRoleToken = [&](int32_t token) {
        return token == mTokenIds.imStart || token == mTokenIds.imEnd || token == mTokenIds.userRole
            || token == mTokenIds.newline;
    };
    std::vector<std::pair<int32_t, int32_t>> userPositions; // (tokenId, absolute position)
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (static_cast<int32_t>(s) == assistantSegIdx)
        {
            break;
        }
        if (segments[s].roleId == mTokenIds.systemRole)
        {
            continue; // the system role is covered by the speaker part above
        }
        std::vector<int32_t> rolePos;
        std::vector<int32_t> mmPos;
        for (int32_t i = segments[s].start; i < segments[s].end; ++i)
        {
            (isUserRoleToken(textTokenIds[static_cast<size_t>(i)]) ? rolePos : mmPos).push_back(i);
        }
        if (mTalkerConfig.maxThinkerToTalkerMmTokens > 0
            && static_cast<int32_t>(mmPos.size()) > mTalkerConfig.maxThinkerToTalkerMmTokens)
        {
            int32_t const total = static_cast<int32_t>(mmPos.size());
            int32_t const kept = mTalkerConfig.maxThinkerToTalkerMmTokens;
            std::vector<int32_t> subsampled;
            subsampled.reserve(kept);
            if (kept == 1)
            {
                // torch.linspace(0, N-1, 1) yields tensor([0.]).long() = [0].
                subsampled.push_back(mmPos.front());
            }
            else
            {
                for (int32_t i = 0; i < kept; ++i)
                {
                    // double avoids drifting off torch.linspace's ladder.
                    double const index
                        = static_cast<double>(i) * static_cast<double>(total - 1) / static_cast<double>(kept - 1);
                    subsampled.push_back(mmPos[static_cast<size_t>(static_cast<int32_t>(index))]);
                }
            }
            mmPos.swap(subsampled);
        }
        std::vector<int32_t> merged;
        merged.reserve(rolePos.size() + mmPos.size());
        merged.insert(merged.end(), rolePos.begin(), rolePos.end());
        merged.insert(merged.end(), mmPos.begin(), mmPos.end());
        std::sort(merged.begin(), merged.end());
        for (int32_t pos : merged)
        {
            userPositions.emplace_back(textTokenIds[static_cast<size_t>(pos)], pos);
        }
    }
    int32_t const userPartRows = static_cast<int32_t>(userPositions.size());

    int64_t const totalSeqLen = static_cast<int64_t>(systemRows) + userPartRows + assistantRows;
    if (totalSeqLen > mTalkerLLMConfig.maxSupportedInputLength)
    {
        LOG_ERROR("buildChunkedPrefill: assembled length %ld > maxInputLen %d", totalSeqLen,
            mTalkerLLMConfig.maxSupportedInputLength);
        return false;
    }
    outSeqLen = totalSeqLen;
    ELLM_CHECK(mTalkerInputEmbeds.reshape({1, totalSeqLen, hiddenSize}), "Tensor reshape failed");
    __half* const dstBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());

    int64_t row = 0;
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.imStart, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.systemRole, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.newline, dstBase, row++, stream))
    {
        return false;
    }
    for (int32_t tid : mSystemInstructIds)
    {
        if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
        {
            return false;
        }
    }
    if (speakerSysPrompt != nullptr)
    {
        for (int32_t tid : *speakerSysPrompt)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
            {
                return false;
            }
        }
    }
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecBosId, dstBase, row++, stream))
    {
        return false;
    }
    for (int32_t p = 0; p < validSpeakerLen; ++p)
    {
        int64_t codesAtPos[kMaxCodeGroups];
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            codesAtPos[g] = speakerRow[static_cast<size_t>(g) * speakerEmbedLen + p];
        }
        sumSpeakerCodecRow(codesAtPos, /*slot=*/p, dstBase, row++, stream);
    }
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecEosId, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.imEnd, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.newline, dstBase, row++, stream))
    {
        return false;
    }
    ELLM_CHECK(row == systemRows, "system row count mismatch");

    for (auto const& [tid, absPos] : userPositions)
    {
        if (isUserRoleToken(tid))
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
            {
                return false;
            }
        }
        else
        {
            if (prefillHidden == nullptr || absPos >= prefillLen)
            {
                LOG_ERROR("chunked user content pos %d out of range (prefillLen=%d) or hidden binding is null", absPos,
                    prefillLen);
                return false;
            }
            projectHiddenRow(*prefillHidden, absPos, dstBase, row++, stream);
        }
    }
    ELLM_CHECK(row == systemRows + userPartRows, "user row count mismatch");

    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.imStart, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.assistantRole, dstBase, row++, stream))
    {
        return false;
    }
    if (!copyEmbedRow(mTextEmbeddingTable, mTokenIds.newline, dstBase, row++, stream))
    {
        return false;
    }
    if (instructIds != nullptr)
    {
        for (int32_t tid : *instructIds)
        {
            if (!copyEmbedRow(mTextEmbeddingTable, tid, dstBase, row++, stream))
            {
                return false;
            }
        }
    }
    if (languageCodecId >= 0)
    {
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkId, dstBase, row++, stream))
        {
            return false;
        }
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkBosId, dstBase, row++, stream))
        {
            return false;
        }
        if (!copyEmbedRow(mTalkerCodecEmbedTable, languageCodecId, dstBase, row++, stream))
        {
            return false;
        }
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkEosId, dstBase, row++, stream))
        {
            return false;
        }
    }
    else
    {
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecNothinkId, dstBase, row++, stream))
        {
            return false;
        }
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkBosId, dstBase, row++, stream))
        {
            return false;
        }
        if (!copyEmbedRow(mTalkerCodecEmbedTable, mTalkerConfig.codecThinkEosId, dstBase, row++, stream))
        {
            return false;
        }
    }
    copyRawRow(mTtsBosEmbed, dstBase, row++, stream);
    if (!copyEmbedRow(mTalkerCodecEmbedTable, codecBosId, dstBase, row++, stream))
    {
        return false;
    }

    // First text chunk: tts_eos takes the last slot when the chunk reaches the end of the text.
    int32_t const chunkTextRows = std::min(firstChunkLen, assistantTextLen);
    for (int32_t i = 0; i < chunkTextRows; ++i)
    {
        if (!copyEmbedRow(
                mTextEmbeddingTable, textTokenIds[static_cast<size_t>(assistantTextStart + i)], dstBase, row++, stream))
        {
            return false;
        }
    }
    if (firstChunkLen > assistantTextLen)
    {
        copyRawRow(mTtsEosEmbed, dstBase, row++, stream);
    }
    ELLM_CHECK(row == totalSeqLen,
        "row count mismatch (got " + std::to_string(row) + ", expected " + std::to_string(totalSeqLen) + ")");

    int32_t const remainingTextStart = chunkTextRows;
    int32_t const remainingTextLen = assistantTextLen - remainingTextStart;
    int32_t const remainingHasEos = (firstChunkLen <= assistantTextLen) ? 1 : 0;

    mChunkStream.remainingTextTokens.clear();
    for (int32_t i = 0; i < remainingTextLen; ++i)
    {
        mChunkStream.remainingTextTokens.push_back(
            textTokenIds[static_cast<size_t>(assistantTextStart + remainingTextStart + i)]);
    }
    mChunkStream.hasTrailingTtsEos = (remainingHasEos != 0);
    mChunkStream.active = !mChunkStream.remainingTextTokens.empty() || mChunkStream.hasTrailingTtsEos;
    mChunkStream.chunkTokensPerCall = kTextInChunkN;
    mChunkStream.cumulativeSeqLen = outSeqLen;
    mChunkStream.cursorToken = 0;
    mChunkStream.firstFrameOfCallIdx = 0;

    LOG_INFO("buildChunkedPrefill: outSeqLen=%ld (sys=%d[+%d], usr=%d, asst=%d), moreChunks=%d", outSeqLen, systemRows,
        sysPromptLen, userPartRows, assistantRows, mChunkStream.active ? 1 : 0);
    return true;
}

void TalkerRunner::resetChunkStream()
{
    // Field-wise, so remainingTextTokens keeps the capacity claimed at init.
    mChunkStream.active = false;
    mChunkStream.remainingTextTokens.clear();
    mChunkStream.hasTrailingTtsEos = false;
    mChunkStream.cumulativeSeqLen = 0;
    mChunkStream.cursorToken = 0;
    mChunkStream.firstFrameOfCallIdx = 0;
}

TalkerRunner::Reprefill TalkerRunner::maybeReprefillChunk(cudaStream_t stream)
{
    if (!mChunkStream.active)
    {
        return Reprefill::kNotDue;
    }
    auto& row = mRows.front();
    if ((row.frames - mChunkStream.firstFrameOfCallIdx) < mChunkStream.framesPerCall)
    {
        return Reprefill::kNotDue;
    }
    return reprefillChunk(stream);
}

TalkerRunner::Reprefill TalkerRunner::reprefillChunk(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::reprefillChunk", nvtx_colors::PALE_GREEN);

    int32_t const numCodeGroups = mTalkerConfig.numCodeGroups;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    auto& row = mRows.front();
    int32_t const framesAvail = row.frames - mChunkStream.firstFrameOfCallIdx;
    if (framesAvail < mChunkStream.codecEmbedFrames)
    {
        LOG_ERROR("reprefillChunk: only %d frames since the last prefill, need at least %d", framesAvail,
            mChunkStream.codecEmbedFrames);
        return Reprefill::kFailed;
    }

    int32_t const textRemain = static_cast<int32_t>(mChunkStream.remainingTextTokens.size()) - mChunkStream.cursorToken;
    int32_t const textRows = std::min(mChunkStream.chunkTokensPerCall, textRemain);
    bool const textExhausted
        = mChunkStream.cursorToken + textRows >= static_cast<int32_t>(mChunkStream.remainingTextTokens.size());
    bool const appendEos
        = textExhausted && mChunkStream.hasTrailingTtsEos && textRows < mChunkStream.chunkTokensPerCall;
    int64_t const nextSeqLen
        = mChunkStream.cumulativeSeqLen + mChunkStream.codecEmbedFrames + textRows + (appendEos ? 1 : 0);
    // A re-prefill that fills the KV cache would leave no room to decode the frame it hands over.
    if (nextSeqLen > mTalkerLLMConfig.maxSupportedInputLength || nextSeqLen >= mTalkerLLMConfig.maxKVCacheCapacity)
    {
        return Reprefill::kInputFull;
    }

    // Codec-embed rows for the first codecEmbedFrames frames of this call. The reference
    // implementation drops the last lookahead frame from the handoff; feeding it breaks the
    // cumulative sequence-length alignment and degrades long utterances.
    __half* const dstBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    int64_t writeRow = mChunkStream.cumulativeSeqLen;
    for (int32_t f = 0; f < mChunkStream.codecEmbedFrames; ++f)
    {
        int32_t const frameIdx = mChunkStream.firstFrameOfCallIdx + f;
        int32_t const* const frameCodes
            = row.flatCodes.data() + static_cast<size_t>(frameIdx) * static_cast<size_t>(mNumCodesPerFrame);
        int64_t hostCodes[kMaxCodeGroups];
        for (int32_t g = 0; g < numCodeGroups; ++g)
        {
            hostCodes[g] = static_cast<int64_t>(frameCodes[g]);
        }
        sumSpeakerCodecRow(hostCodes, /*slot=*/f, dstBase, writeRow++, stream);
    }

    for (int32_t i = 0; i < textRows; ++i)
    {
        int32_t const tokenId = mChunkStream.remainingTextTokens[static_cast<size_t>(mChunkStream.cursorToken + i)];
        if (!copyEmbedRow(mTextEmbeddingTable, tokenId, dstBase, writeRow++, stream))
        {
            return Reprefill::kFailed;
        }
    }
    mChunkStream.cursorToken += textRows;

    if (appendEos)
    {
        copyRawRow(mTtsEosEmbed, dstBase, writeRow++, stream);
        mChunkStream.hasTrailingTtsEos = false;
    }
    mChunkStream.cumulativeSeqLen = writeRow;

    bool const exhaustedAfter = textExhausted && !mChunkStream.hasTrailingTtsEos;
    setFrameBudget(row, mChunkStream.cumulativeSeqLen, row.frames);
    LOG_INFO("Chunked re-prefill: cumSeq=%ld (added %d codec + %d text%s); more chunks=%s; maxFrames=%d",
        mChunkStream.cumulativeSeqLen, mChunkStream.codecEmbedFrames, textRows, appendEos ? " + tts_eos" : "",
        exhaustedAfter ? "no" : "yes", row.maxFrames);

    ELLM_CHECK(mTalkerInputEmbeds.reshape({1, mChunkStream.cumulativeSeqLen, hiddenSize}), "Tensor reshape failed");
    ELLM_CHECK(mTalkerHiddenStatesBuffer.reshape({1, talkerHiddenSeqDim(mChunkStream.cumulativeSeqLen), hiddenSize}),
        "Tensor reshape failed");
    mQueryLengths.assign(1, static_cast<int32_t>(mChunkStream.cumulativeSeqLen));
    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mQueryLengths, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
        {
            LOG_ERROR("Chunked re-prefill failed at cumSeq=%ld", mChunkStream.cumulativeSeqLen);
            return Reprefill::kFailed;
        }
    }

    // The reference reissues a fresh generate() per chunk call, so the repetition-penalty
    // window resets at every re-prefill; only the post-text stretch accumulates.
    std::fill(row.seenTokens.begin(), row.seenTokens.end(), 0);
    row.numSeenTokens = 0;
    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerSuppressStart,
        mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, row.numSeenTokens, row.repetitionPenalty, stream);
    if (!exhaustedAfter)
    {
        suppressTalkerEosLogit(mTalkerLogits, 0, stream);
    }

    float const temperature = (row.temperature > 0) ? row.temperature : kTalkerTemperature;
    int32_t const topK = (row.topK > 0) ? row.topK : kTalkerTopK;
    float const topP = (row.topP > 0) ? row.topP : kTalkerTopP;
    SamplingParams const talkerSampling(1, mTalkerConfig.talkerVocabSize, temperature, topK, topP);
    // Equal to decode step firstFrameOfCallIdx + 1's offset, so this sample reuses that step's random
    // number; llm_inference does the same, and this path matches it.
    auto const randomOffset = static_cast<uint64_t>(mChunkStream.firstFrameOfCallIdx) + 1;
    ELLM_CHECK(mTalkerLogits.reshape({1, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    ELLM_CHECK(mTalkerSelectedIndices.reshape({1, 1}), "Tensor reshape failed");
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSampling, mSamplingWorkspace,
        stream, kTalkerSamplingSeed, randomOffset);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    row.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
    row.pastLen = static_cast<int32_t>(mChunkStream.cumulativeSeqLen);

    trackSeenToken(row, /*slot=*/0, row.codecToken, mTalkerSelectedIndices.dataPointer<int32_t>(), stream);

    mChunkStream.firstFrameOfCallIdx = row.frames;
    if (exhaustedAfter)
    {
        mChunkStream.active = false;
    }
    return Reprefill::kDone;
}

// ---------------------------------------------------------------------------
// TTS layout
// ---------------------------------------------------------------------------

__half const* TalkerRunner::talkerEmbRow(int32_t tokenId) const
{
    return static_cast<__half const*>(mTalkerEmbeddingTable.rawPointer())
        + static_cast<int64_t>(tokenId) * mTalkerConfig.talkerHiddenSize;
}

void TalkerRunner::pushPrefillRow(__half const* srcA, __half const* srcB)
{
    ELLM_CHECK(srcA != nullptr, "pushPrefillRow: the required source row is null");
    mPrefillRows.push_back({srcA, srcB});
}

int64_t TalkerRunner::flushPrefillRows(rt::Tensor& output, cudaStream_t stream)
{
    auto const numRows = static_cast<int64_t>(mPrefillRows.size());
    ELLM_CHECK(numRows > 0, "flushPrefillRows: no rows queued");
    int64_t const bytes = numRows * static_cast<int64_t>(sizeof(kernel::PrefillRowDesc));
    int64_t const sliceBytes = mPrefillDescsHost.getShape()[1];
    ELLM_CHECK(
        bytes <= sliceBytes, "flushPrefillRows: " + std::to_string(numRows) + " rows exceed the staging capacity");
    // Members prepare back to back with no sync between them, so each stages through its own slice.
    auto* const staging = mPrefillDescsHost.dataPointer<int8_t>() + static_cast<int64_t>(mPrepareSlot) * sliceBytes;
    std::memcpy(staging, mPrefillRows.data(), static_cast<size_t>(bytes));
    CUDA_CHECK(cudaMemcpyAsync(
        mPrefillDescsDevice.rawPointer(), staging, static_cast<size_t>(bytes), cudaMemcpyHostToDevice, stream));
    kernel::invokePrefillRowAssemble(reinterpret_cast<kernel::PrefillRowDesc const*>(mPrefillDescsDevice.rawPointer()),
        static_cast<int32_t>(numRows), static_cast<int32_t>(mTalkerConfig.talkerHiddenSize), output, stream);
    return numRows;
}

int32_t TalkerRunner::resolveLanguageId(std::string const& languageName, std::string const& speakerName) const
{
    // An engine with no think token or no table has no language rows to place, whatever the
    // request asks for.
    if (mTalkerConfig.codecThinkId < 0 || mTalkerConfig.codecLanguageIds.empty())
    {
        if (!languageName.empty() && languageName != "auto")
        {
            LOG_WARNING(
                "language '%s' ignored: this checkpoint carries no codec_language_id map", languageName.c_str());
        }
        return -1;
    }

    auto const requested = lowered(languageName);
    int32_t languageId = -1;
    if (!requested.empty() && requested != "auto")
    {
        if (auto const it = mTalkerConfig.codecLanguageIds.find(requested); it != mTalkerConfig.codecLanguageIds.end())
        {
            languageId = it->second;
        }
        else
        {
            LOG_WARNING("language '%s' not in codec_language_id; using auto", languageName.c_str());
            return -1;
        }
    }

    // A dialect speaker names its own language when the request leaves it to auto or asks for
    // Chinese, the way the reference modeling does.
    if ((requested.empty() || requested == "auto" || requested == "chinese") && !speakerName.empty())
    {
        if (auto const it = mTalkerConfig.speakerDialects.find(lowered(speakerName));
            it != mTalkerConfig.speakerDialects.end())
        {
            if (auto const lang = mTalkerConfig.codecLanguageIds.find(it->second);
                lang != mTalkerConfig.codecLanguageIds.end())
            {
                languageId = lang->second;
            }
        }
    }
    return languageId;
}

bool TalkerRunner::encodeVoiceCloneReference(rt::Tensor const& wav24k, int32_t& iclFrames, cudaStream_t stream)
{
    iclFrames = 0;
    if (mCloneEncoders == nullptr)
    {
        LOG_ERROR("A reference voice was given but this artifact directory carries no clone encoders");
        return false;
    }
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    ELLM_CHECK(mCloneEncoders->speakerEmbeddingDim() == hiddenSize,
        "speaker encoder dimension does not match the talker hidden size");
    if (!mCloneEncoders->extractSpeakerEmbedding(wav24k, mVoiceCloneXVector, stream))
    {
        return false;
    }
    if (mRefTextIds.empty())
    {
        return true; // x-vector only: no transcript, so no in-context segment
    }
    ELLM_CHECK(mCloneEncoders->hasTokenizerEncoder() && mCloneEncoders->numQuantizers() == mTalkerConfig.numCodeGroups,
        "in-context cloning needs speech_tokenizer_encoder.engine with matching code groups");

    int32_t numFrames = 0;
    if (!mCloneEncoders->encodeReferenceCodes(wav24k, numFrames, stream))
    {
        return false;
    }
    // Against capacity, not the current shape: a previous request may have shrunk it.
    int64_t const maxRefFrames
        = static_cast<int64_t>(mIclFrameSumBuffer.getMemoryCapacity()) / (hiddenSize * sizeof(__half));
    ELLM_CHECK(numFrames <= maxRefFrames,
        "reference audio yields " + std::to_string(numFrames) + " frames, more than the workspace holds ("
            + std::to_string(maxRefFrames) + "); shorten it or drop the transcript for x-vector-only cloning");

    ELLM_CHECK(mIclFrameSumBuffer.reshape({numFrames, hiddenSize}), "Tensor reshape failed");
    kernel::invokeSumCodecEmbeddings(mCloneEncoders->refCodesDevice(),
        reinterpret_cast<__half const* const*>(mIclTablePtrsGpu.rawPointer()), numFrames, mTalkerConfig.numCodeGroups,
        static_cast<int32_t>(hiddenSize), mIclFrameSumBuffer, stream);
    iclFrames = numFrames;
    return true;
}

//! Standalone text-to-speech layout: the whole token stream is one assistant turn, so there
//! are no user segments to replay and no backbone hidden to project. Instruction control, a
//! reference voice and language conditioning all live here.
bool TalkerRunner::buildTtsPrefill(int32_t speakerId, int64_t& outSeqLen, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::buildTtsPrefill", nvtx_colors::PALE_GREEN);

    auto const& textTokenIds = mTextTokenIds;
    auto const seqLen = static_cast<int64_t>(textTokenIds.size());
    ELLM_CHECK(seqLen > 0, "buildTtsPrefill: empty token stream");

    bool const isVoiceDesign = (mTalkerConfig.ttsModelType == "voice_design");
    bool const hasClonePrompt = mRefAudioView.rawPointer() != nullptr;
    int32_t iclFrames = 0;
    if (hasClonePrompt && !encodeVoiceCloneReference(mRefAudioView, iclFrames, stream))
    {
        return false;
    }
    bool const iclMode = (iclFrames > 0);
    // The reference transcript arrives assistant-wrapped; the row plan reads it stripped,
    // which is ids[3 : size - 2].
    if (iclMode)
    {
        ELLM_CHECK(static_cast<int64_t>(mRefTextIds.size()) > 5, "voice clone ref_text_ids is too short to strip");
    }
    int64_t const refLen = iclMode ? static_cast<int64_t>(mRefTextIds.size()) - 5 : 0;
    auto const instructLen = static_cast<int64_t>(mInstructIds.size());
    // A style name resolves through talker_assistant_prompt_id_mapping, which only the chunked
    // layout carries; free-form instruction text reaches this layout as 'instruct' instead.
    if (instructLen == 0 && !mRequestInstruct.empty())
    {
        LOG_WARNING("assistant_instruct '%s' dropped: the tts prefill layout takes 'instruct' text or 'instruct_ids'",
            mRequestInstruct.c_str());
    }

    // One embed and one projection pass over [instruct][text][reference transcript].
    std::vector<int32_t> combinedIds;
    combinedIds.reserve(static_cast<size_t>(instructLen + seqLen + refLen));
    combinedIds.insert(combinedIds.end(), mInstructIds.begin(), mInstructIds.end());
    combinedIds.insert(combinedIds.end(), textTokenIds.begin(), textTokenIds.end());
    if (refLen > 0)
    {
        combinedIds.insert(combinedIds.end(), mRefTextIds.begin() + 3, mRefTextIds.end() - 2);
    }
    auto const combinedLen = static_cast<int64_t>(combinedIds.size());

    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    ELLM_CHECK(mGpuTokenIdsBuffer.reshape({1, combinedLen}), "Tensor reshape failed");
    uploadThroughPinned(combinedIds.data(), combinedLen, mHostTokenIds, mPrepareSlot, mGpuTokenIdsBuffer, stream);
    ELLM_CHECK(mThinkerEmbedBuffer.reshape({1, combinedLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, mThinkerEmbedBuffer, stream);
    ELLM_CHECK(mThinkerEmbedBuffer.reshape({combinedLen, thinkerHiddenSize}), "Tensor reshape failed");

    ELLM_CHECK(mProjectedBuffer.reshape({combinedLen, hiddenSize}), "Tensor reshape failed");
    ELLM_CHECK(mMLPWorkspace.reshape({combinedLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mProjectedBuffer, mMLPWorkspace, stream);

    int32_t const languageId
        = resolveLanguageId(mRequestLanguage.empty() ? mLanguage : mRequestLanguage, mRequestSpeakerName);
    int64_t const textRows = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
    ELLM_CHECK(textRows > 0, "buildTtsPrefill: the text is shorter than the assistant wrapper it is stripped of");

    // Checked before the rows land in the assembly scratch, which holds one engine input.
    auto const checkFitsEngine = [this](int64_t rows) {
        ELLM_CHECK(rows <= mTalkerLLMConfig.maxSupportedInputLength,
            "talker prefill of " + std::to_string(rows) + " rows exceeds the engine's max input length "
                + std::to_string(mTalkerLLMConfig.maxSupportedInputLength));
    };
    if (instructLen == 0 && !isVoiceDesign && !hasClonePrompt)
    {
        // Nothing to interleave, so the fused kernel lays the whole prefill out in one pass.
        outSeqLen = ((languageId >= 0) ? kPrefixRowsWithLanguage : kNonStreamingPrefixRows) + textRows + 2;
        checkFitsEngine(outSeqLen);
        ELLM_CHECK(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
        kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed,
            mTalkerEmbeddingTable, mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId,
            mTalkerConfig.codecThinkEosId, speakerId, mTalkerConfig.codecPadId, mTalkerConfig.codecBosId,
            mTalkerConfig.codecThinkId, languageId, static_cast<int32_t>(textRows), mTalkerInputEmbeds, stream);
    }
    else
    {
        auto const* const projBase = static_cast<__half const*>(mProjectedBuffer.rawPointer());
        auto const* const ttsPad = mTtsPadEmbed.dataPointer<__half>();
        auto const* const ttsBos = mTtsBosEmbed.dataPointer<__half>();
        auto const* const ttsEos = mTtsEosEmbed.dataPointer<__half>();

        mPrefillRows.clear();
        for (int64_t i = 0; i < instructLen; ++i)
        {
            pushPrefillRow(projBase + i * hiddenSize, nullptr);
        }
        for (int64_t i = 0; i < kAssistantPrefixLen; ++i)
        {
            pushPrefillRow(projBase + (instructLen + i) * hiddenSize, nullptr);
        }

        bool const hasLanguage = (languageId >= 0);
        ELLM_CHECK(!hasLanguage || mTalkerConfig.codecThinkId >= 0,
            "buildTtsPrefill: a language was resolved but this checkpoint has no codec think token");
        pushPrefillRow(ttsPad, talkerEmbRow(hasLanguage ? mTalkerConfig.codecThinkId : mTalkerConfig.codecNothinkId));
        pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecThinkBosId));
        if (hasLanguage)
        {
            pushPrefillRow(ttsPad, talkerEmbRow(languageId));
        }
        pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecThinkEosId));

        // A reference voice speaks through a continuous x-vector; a built-in one through its
        // codec row; voice design conditions on neither and gets no speaker row at all.
        if (hasClonePrompt)
        {
            pushPrefillRow(ttsPad, static_cast<__half const*>(mVoiceCloneXVector.rawPointer()));
        }
        else if (!isVoiceDesign)
        {
            pushPrefillRow(ttsPad, talkerEmbRow(speakerId));
        }
        pushPrefillRow(ttsBos, talkerEmbRow(mTalkerConfig.codecPadId));

        if (iclMode)
        {
            // Overlapped-add: row i pairs a text-side row (reference transcript, then the
            // target text, then tts_eos, then padding) with a codec-side row (codec_bos, then
            // the summed reference frames).
            int64_t const textLens = refLen + textRows + 1;
            int64_t const codecLens = 1 + iclFrames;
            ELLM_CHECK(textLens <= codecLens,
                "the target text needs " + std::to_string(textLens) + " rows but the reference supplies only "
                    + std::to_string(codecLens) + "; use a longer reference or drop the transcript");
            auto const* const frameSums = static_cast<__half const*>(mIclFrameSumBuffer.rawPointer());
            for (int64_t i = 0; i < codecLens; ++i)
            {
                __half const* textRow = nullptr;
                if (i < refLen)
                {
                    textRow = projBase + (instructLen + seqLen + i) * hiddenSize;
                }
                else if (i < refLen + textRows)
                {
                    textRow = projBase + (instructLen + kAssistantPrefixLen + (i - refLen)) * hiddenSize;
                }
                else
                {
                    textRow = (i == textLens - 1) ? ttsEos : ttsPad;
                }
                __half const* const codecRow
                    = (i == 0) ? talkerEmbRow(mTalkerConfig.codecBosId) : frameSums + (i - 1) * hiddenSize;
                pushPrefillRow(textRow, codecRow);
            }
        }
        else
        {
            for (int64_t i = 0; i < textRows; ++i)
            {
                int32_t const codecId = (i == textRows - 1) ? mTalkerConfig.codecBosId : mTalkerConfig.codecPadId;
                pushPrefillRow(projBase + (instructLen + kAssistantPrefixLen + i) * hiddenSize, talkerEmbRow(codecId));
            }
            pushPrefillRow(ttsEos, talkerEmbRow(mTalkerConfig.codecPadId));
            pushPrefillRow(ttsPad, talkerEmbRow(mTalkerConfig.codecBosId));
        }

        outSeqLen = static_cast<int64_t>(mPrefillRows.size());
        checkFitsEngine(outSeqLen);
        ELLM_CHECK(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
        flushPrefillRows(mTalkerInputEmbeds, stream);
    }

    LOG_INFO(
        "buildTtsPrefill: outSeqLen=%ld (instruct=%ld, text=%ld, languageId=%d, voiceDesign=%d, clone=%d, "
        "iclFrames=%d)",
        outSeqLen, instructLen, textRows, languageId, isVoiceDesign ? 1 : 0, hasClonePrompt ? 1 : 0, iclFrames);
    return true;
}

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
