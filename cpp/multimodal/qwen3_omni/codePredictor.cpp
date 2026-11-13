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

#include "multimodal/qwen3_omni/codePredictor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/cpSpecKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"

#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace qwen3_omni
{

namespace
{

//! Code predictor prefill is exactly two rows: the talker's hidden state and the code_0
//! embedding.
constexpr int32_t kCpPrefillSeqLen = 2;

//! Non-zero KV length the code predictor decode graphs are captured at, so capture takes the
//! non-empty-cache path a real decode step takes.
constexpr int32_t kCpGraphCaptureCacheLength = 128;

//! Seed of the draws outside speculative decoding. It differs from the talker's, so a talker draw and a
//! code predictor draw never share a random number.
constexpr uint64_t kCpSamplingSeed = 0xC0DE5EEDULL;

//! Random offset of the draw for code @p codeIdx (1..numRvqLayers) of frame @p frameIdx; one per
//! (frame, code) pair.
constexpr uint64_t cpSamplingOffset(int32_t frameIdx, int32_t codeIdx, int32_t numRvqLayers) noexcept
{
    return static_cast<uint64_t>(frameIdx) * static_cast<uint64_t>(numRvqLayers) + static_cast<uint64_t>(codeIdx - 1);
}

} // namespace

void prepareRaggedStep(rt::RaggedExecutionBatch& batch, rt::PipelineIO& io, rt::SharedResources& resources,
    rt::LLMEngineConfig const& config, std::vector<int32_t> const& queryLengths,
    std::vector<int32_t> const& pastLengths, int32_t batchSize, int32_t queryWidth, rt::ExecutionPhase phase,
    cudaStream_t stream)
{
    ELLM_CHECK(batchSize > 0 && batchSize <= config.maxSupportedBatchSize, "invalid ragged batch size");
    ELLM_CHECK(queryWidth > 0 && batchSize <= config.maxPhysicalTokens / queryWidth,
        "ragged step exceeds the engine token capacity");
    ELLM_CHECK(
        queryLengths.size() >= static_cast<size_t>(batchSize) && pastLengths.size() >= static_cast<size_t>(batchSize),
        "ragged step is missing host sequence lengths");

    bool const contextPhase
        = phase == rt::ExecutionPhase::kContextPrefill || phase == rt::ExecutionPhase::kContextChunk;
    int32_t const physicalTokens = batchSize * queryWidth;
    int32_t validTokens{0};
    batch.layout = rt::TokenLayoutBackend::kEntryPaddedCompatibility;
    batch.positions.assign(physicalTokens, -1);
    batch.queryStartOffsets.resize(batchSize + 1);
    batch.queryLengths.resize(batchSize);
    batch.pastLengths.resize(batchSize);
    batch.attentionSequenceLengths.resize(batchSize);
    batch.stateIndices.resize(batchSize);
    batch.logitsIndices.resize(batchSize);

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const queryLength = queryLengths[static_cast<size_t>(batchIdx)];
        int32_t const pastLength = pastLengths[static_cast<size_t>(batchIdx)];
        ELLM_CHECK(queryLength > 0 && queryLength <= queryWidth, "invalid ragged query length");
        ELLM_CHECK(pastLength >= 0 && pastLength + queryLength <= config.maxKVCacheCapacity,
            "ragged sequence exceeds the KV-cache capacity");

        int32_t const physicalStart = batchIdx * queryWidth;
        batch.queryStartOffsets[static_cast<size_t>(batchIdx)] = physicalStart;
        batch.queryLengths[static_cast<size_t>(batchIdx)] = queryLength;
        batch.pastLengths[static_cast<size_t>(batchIdx)] = pastLength;
        batch.attentionSequenceLengths[static_cast<size_t>(batchIdx)] = pastLength + queryLength;
        batch.stateIndices[static_cast<size_t>(batchIdx)] = batchIdx;
        batch.logitsIndices[static_cast<size_t>(batchIdx)] = physicalStart + queryLength - 1;
        validTokens += queryLength;
        for (int32_t tokenIdx = 0; tokenIdx < queryLength; ++tokenIdx)
        {
            batch.positions[static_cast<size_t>(physicalStart + tokenIdx)] = pastLength + tokenIdx;
        }
    }
    batch.queryStartOffsets[static_cast<size_t>(batchSize)] = physicalTokens;
    batch.shape = rt::RaggedStepShape{batchSize, validTokens, physicalTokens, queryWidth, contextPhase ? batchSize : 0,
        contextPhase ? validTokens : 0, batchSize};
    ++batch.stepId;

    rt::prepareRaggedExecutionBindings(io, resources, config, batch, /*kvCacheIndex=*/0, stream);
}

void CodePredictor::preallocate(std::string const& artifactDir, std::string const& checkpointDir, int32_t maxBatch,
    int32_t specVerifySize, cudaStream_t stream)
{
    ELLM_CHECK(!artifactDir.empty(), "code predictor needs an artifact directory");
    ELLM_CHECK(std::filesystem::is_directory(artifactDir),
        "code predictor artifact directory '" + artifactDir + "' does not exist");
    ELLM_CHECK(maxBatch >= 1, "code predictor max batch must be positive");

    mArtifactDir = artifactDir;
    mCheckpointDir = checkpointDir;
    mMaxBatch = maxBatch;
    mStream = stream;

    loadEngine();
    loadWeights();
    allocateWorkspaces();
    resolveSpeculation(specVerifySize);

    ELLM_CHECK(kCpPrefillSeqLen + mNumRvqLayers - 1 <= mEngineConfig.maxKVCacheCapacity,
        "a code predictor frame needs " + std::to_string(kCpPrefillSeqLen + mNumRvqLayers - 1)
            + " KV slots but the engine was built for " + std::to_string(mEngineConfig.maxKVCacheCapacity));
}

void CodePredictor::loadEngine()
{
    std::filesystem::path const enginePath = std::filesystem::path(mArtifactDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(mArtifactDir) / "config.json";
    ELLM_CHECK(std::filesystem::exists(enginePath), "no code predictor engine at '" + enginePath.string() + "'");
    ELLM_CHECK(std::filesystem::exists(configPath), "no code predictor config at '" + configPath.string() + "'");

    mEngineConfig = rt::parseEngineConfig(configPath);
    auto weights = std::make_unique<rt::ExternalWeightManager>();
    weights->load(mArtifactDir, configPath, mStream, mCheckpointDir);
    mExec = rt::EngineExecutor::createForLLM(
        enginePath, mEngineConfig, std::nullopt, nullptr, rt::EngineRefitOptions{weights->takeRefitWeights(), mStream});
    rt::validateAgainstEngine(mEngineConfig, *mExec, "omni_code_predictor");

    // The stacked lm_heads input is what lets one captured decode graph serve every codebook.
    ELLM_CHECK(mExec->hasIOTensor(binding_names::kLmHeads),
        std::string("the code predictor engine exposes no '") + binding_names::kLmHeads
            + "' input; re-export this artifact with the current frontend");

    std::unordered_map<std::string, std::string> const emptyLoraMap;
    mResources = rt::SharedResources::createForLLM(mEngineConfig, emptyLoraMap, mStream);
    mPipelineIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mEngineConfig, mStream));
    mStepPreparer = std::make_unique<rt::StepPreparer>(mEngineConfig);
    rt::buildTensorMap(mTensorMap, *mPipelineIO, *mResources, mEngineConfig, /*kvCacheIndex=*/0);
    mResources->externalWeightManager = std::move(weights);
    mResources->externalWeightManager->validateAgainstEngine(*mExec, "code_predictor");
    mResources->externalWeightManager->registerTensorMapEntries(mTensorMap);

    mHiddenSize = mEngineConfig.hiddenSize;
    ELLM_CHECK(mMaxBatch <= mEngineConfig.maxSupportedBatchSize,
        "code predictor batch " + std::to_string(mMaxBatch) + " exceeds what the engine was built for ("
            + std::to_string(mEngineConfig.maxSupportedBatchSize) + ")");

    // The engine allocates its context memory kUSER_MANAGED, so one arena is claimed here; the
    // talker may replace it with its own.
    mContextMemoryBytes = mExec->getRequiredContextMemorySize();
    if (mContextMemoryBytes > 0)
    {
        mExecContextMemory = rt::Tensor(
            {mContextMemoryBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kUINT8, "codePredictorContextMemory");
        ELLM_CHECK(mExec->setContextMemory(mExecContextMemory), "failed to set code predictor context memory");
    }

    LOG_INFO("Code predictor engine loaded: hiddenSize=%d, numLayers=%d, maxKVCacheCapacity=%d",
        mEngineConfig.hiddenSize, mEngineConfig.numDecoderLayers, mEngineConfig.maxKVCacheCapacity);
}

void CodePredictor::loadWeights()
{
    // The residual codebook count is what the artifacts say it is: one lm_head and one codec
    // embedding table per codebook. bind() checks it against the talker's own config.
    {
        std::filesystem::path const lmHeadPath = std::filesystem::path(mArtifactDir) / "lm_heads.safetensors";
        std::vector<rt::Tensor> heads;
        ELLM_CHECK(safetensors::loadSafetensors(lmHeadPath, heads, mStream),
            "failed to load lm_heads.safetensors from " + lmHeadPath.string());
        mNumRvqLayers = static_cast<int32_t>(heads.size());
        ELLM_CHECK(mNumRvqLayers >= 2,
            "lm_heads.safetensors carries " + std::to_string(mNumRvqLayers)
                + " heads; a frame needs a prefill head plus at least one decode head");

        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const weightKey = "lm_head_" + std::to_string(i) + ".weight";
            auto const head = std::find_if(heads.begin(), heads.end(),
                [&weightKey](rt::Tensor const& candidate) { return candidate.getName() == weightKey; });
            ELLM_CHECK(head != heads.end(), "missing key '" + weightKey + "' in lm_heads.safetensors");
            ELLM_CHECK(head->getShape().getNumDims() == 2, weightKey + " should be 2D [codebookSize, hiddenSize]");
            if (i == 0)
            {
                mCodebookSize = static_cast<int32_t>(head->getShape()[0]);
                mWeights.lmHeads = rt::Tensor({mNumRvqLayers, head->getShape()[0], head->getShape()[1]},
                    rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "codePredictorLmHeads");
            }
            ELLM_CHECK(head->getShape()[0] == mWeights.lmHeads.getShape()[1]
                    && head->getShape()[1] == mWeights.lmHeads.getShape()[2],
                "lm_head shape mismatch across heads");
            int64_t const headBytes = head->getMemoryCapacity();
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mWeights.lmHeads.rawPointer()) + i * headBytes,
                head->rawPointer(), headBytes, cudaMemcpyDeviceToDevice, mStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        ELLM_CHECK(mWeights.lmHeads.getShape()[2] == mHiddenSize,
            "lm_heads are " + std::to_string(mWeights.lmHeads.getShape()[2]) + " wide but the engine hidden size is "
                + std::to_string(mHiddenSize));
    }
    mNumCodesPerFrame = mNumRvqLayers + 1;

    {
        std::filesystem::path const embedPath = std::filesystem::path(mArtifactDir) / "codec_embeddings.safetensors";
        std::vector<rt::Tensor> tables;
        ELLM_CHECK(safetensors::loadSafetensors(embedPath, tables, mStream),
            "failed to load codec_embeddings.safetensors from " + embedPath.string());
        ELLM_CHECK(static_cast<int32_t>(tables.size()) == mNumRvqLayers,
            "codec_embeddings.safetensors has " + std::to_string(tables.size())
                + " tables but lm_heads.safetensors has " + std::to_string(mNumRvqLayers) + " heads");
        mWeights.codecEmbeddingTables.resize(static_cast<size_t>(mNumRvqLayers));
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const key = "embedding_" + std::to_string(i);
            auto const table = std::find_if(tables.begin(), tables.end(),
                [&key](rt::Tensor const& candidate) { return candidate.getName() == key; });
            ELLM_CHECK(table != tables.end(), "missing key '" + key + "' in codec_embeddings.safetensors");
            ELLM_CHECK(table->getShape().getNumDims() == 2, key + " should be 2D [codebookSize, talkerHiddenSize]");
            mWeights.codecEmbeddingTables[static_cast<size_t>(i)] = std::move(*table);
        }
        // Codec embeddings live in the talker's hidden space, which is where the talker's
        // residual reads them from; the projection below is what bridges the two widths.
        mTalkerHiddenSize = static_cast<int32_t>(mWeights.codecEmbeddingTables[0].getShape()[1]);
    }

    {
        std::filesystem::path const projPath
            = std::filesystem::path(mArtifactDir) / "small_to_mtp_projection.safetensors";
        if (std::filesystem::exists(projPath))
        {
            std::vector<rt::Tensor> projection;
            ELLM_CHECK(safetensors::loadSafetensors(projPath, projection, mStream),
                "failed to load small_to_mtp_projection.safetensors from " + projPath.string());
            for (auto& tensor : projection)
            {
                if (tensor.getName() == "weight")
                {
                    mWeights.smallToMtpWeight = std::move(tensor);
                }
                else if (tensor.getName() == "bias")
                {
                    mWeights.smallToMtpBias = std::move(tensor);
                }
            }
            ELLM_CHECK(
                mWeights.smallToMtpWeight.rawPointer() != nullptr && mWeights.smallToMtpBias.rawPointer() != nullptr,
                "small_to_mtp_projection.safetensors must contain both 'weight' and 'bias'");
            ELLM_CHECK(mWeights.smallToMtpWeight.getShape().getNumDims() == 2
                    && mWeights.smallToMtpWeight.getShape()[0] == mHiddenSize
                    && mWeights.smallToMtpWeight.getShape()[1] == mTalkerHiddenSize,
                "small_to_mtp_projection weight must be [" + std::to_string(mHiddenSize) + ", "
                    + std::to_string(mTalkerHiddenSize) + "]");
            mWeights.useSmallToMtpProjection = true;
        }
        else
        {
            ELLM_CHECK(mTalkerHiddenSize == mHiddenSize,
                "small_to_mtp_projection.safetensors is required when the talker hidden size ("
                    + std::to_string(mTalkerHiddenSize) + ") differs from the code predictor hidden size ("
                    + std::to_string(mHiddenSize) + ")");
            mWeights.useSmallToMtpProjection = false;
        }
    }

    LOG_INFO("Code predictor weights loaded: rvqLayers=%d, codebookSize=%d, talkerHidden=%d, projection=%s",
        mNumRvqLayers, mCodebookSize, mTalkerHiddenSize, mWeights.useSmallToMtpProjection ? "yes" : "no");
}

//! Widest window the kernels and this class are written for.
constexpr int32_t kCpSpecMaxK = 8;
//! Fixed seed for the speculative draws; the offset advances so no two rounds repeat.
constexpr uint64_t kCpSpecSamplingSeed = 0x51ED270BULL;

void CodePredictor::resolveSpeculation(int32_t requested)
{
    mSpec.verifySize = 0;
    if (requested <= 0)
    {
        return;
    }
    ELLM_CHECK(requested >= 2,
        "code predictor 'cp_spec_verify_size' of " + std::to_string(requested)
            + " leaves no draft slot; use 0 to disable speculation or at least 2");

    // Token-major: the verify window is the row count of the rank-2 generation profile, not a
    // sequence axis.
    auto const genMax = mExec->getProfileShape(
        binding_names::kInputsEmbeds, /*generationProfile=*/1, nvinfer1::OptProfileSelector::kMAX);
    mSpec.genProfileMaxTokens = genMax.nbDims == 2 ? static_cast<int32_t>(genMax.d[0]) : 1;

    int32_t const k = std::min(kCpSpecMaxK, requested);
    if (!kernel::cpSpecSupportsVocab(mCodebookSize))
    {
        LOG_WARNING("Code predictor speculation off: codebook %d exceeds what the accept kernels cover", mCodebookSize);
        return;
    }
    // A verify pass binds maxBatch * k rows at once, so the window has to fit the profile's
    // token budget for the whole batch, not for one sequence.
    if (mSpec.genProfileMaxTokens < mMaxBatch * k)
    {
        LOG_WARNING(
            "Code predictor speculation off: the generation profile admits %d token(s), a verify window of "
            "%d over %d sequence(s) needs %d. Rebuild the code predictor engine with a current builder.",
            mSpec.genProfileMaxTokens, k, mMaxBatch, mMaxBatch * k);
        return;
    }
    if (mNumRvqLayers < 2)
    {
        LOG_WARNING("Code predictor speculation off: %d RVQ layer(s) leave nothing to draft", mNumRvqLayers);
        return;
    }

    mSpec.verifySize = k;
    int64_t const maxBS = mMaxBatch;
    int64_t const cpH = mHiddenSize;
    int64_t const talkerH = mTalkerHiddenSize;
    int64_t const vocab = mCodebookSize;
    int64_t const rows = maxBS * k;

    mSpec.draftHidden = rt::Tensor({maxBS, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpDraftHidden");
    mSpec.verifyCodeIds
        = rt::Tensor({maxBS, k}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpVerifyCodeIds");
    mSpec.verifyInput = rt::Tensor({rows, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpVerifyInput");
    mSpec.verifyHidden = rt::Tensor({rows, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpVerifyHidden");
    mSpec.logitsFp16 = rt::Tensor({rows, vocab}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpSpecLogits");
    mSpec.draftProbs
        = rt::Tensor({maxBS * (k - 1), vocab}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "cpDraftProbs");
    mSpec.targetProbs = rt::Tensor({rows, vocab}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "cpTargetProbs");
    mSpec.draftTokenIds
        = rt::Tensor({maxBS, k - 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpDraftTokenIds");
    mSpec.acceptedTokenIds
        = rt::Tensor({maxBS, k}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpAcceptedIds");
    mSpec.acceptLength = rt::Tensor({maxBS}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpAcceptLength");
    mSpec.proposalLengths.device
        = rt::Tensor({maxBS}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpProposalLens");
    mSpec.rawCodecEmbedRows
        = rt::Tensor({rows, talkerH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpRawCodecRows");

    mSpec.hostAcceptLength
        = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostAcceptLength");
    mSpec.hostAcceptedTokenIds
        = rt::Tensor({maxBS, k}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostAcceptedIds");
    mSpec.proposalLengths.host
        = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostProposalLens");
    mSpec.hostM = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostSpecM");
    mSpec.hostBaseLen = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostSpecBaseLen");

    constexpr int32_t kRowIdxPlanes = 5;
    mSpec.rowIdx.device
        = rt::Tensor({kRowIdxPlanes * rows}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpRowIdx");
    mSpec.rowIdx.host
        = rt::Tensor({kRowIdxPlanes * rows}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostRowIdx");
    constexpr int32_t kStoreIdxPlanes = 2;
    mSpec.storeIdx.device
        = rt::Tensor({kStoreIdxPlanes * rows}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpStoreIdx");
    mSpec.storeIdx.host
        = rt::Tensor({kStoreIdxPlanes * rows}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostStoreIdx");

    // One codec table per RVQ depth. The talker's own table is not in this array: a verify
    // position embeds code_j through table j-1, never through the talker's.
    int64_t const tablePtrBytes = static_cast<int64_t>(mNumRvqLayers) * sizeof(void*);
    mSpec.tablePtrs.host
        = rt::Tensor({tablePtrBytes}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8, "cpTablePtrsHost");
    auto** tablePtrs = static_cast<void const**>(mSpec.tablePtrs.host.rawPointer());
    for (int32_t g = 0; g < mNumRvqLayers; ++g)
    {
        tablePtrs[g] = mWeights.codecEmbeddingTables[static_cast<size_t>(g)].rawPointer();
    }
    mSpec.tablePtrs.device
        = rt::Tensor({tablePtrBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "cpTablePtrsGpu");
    CUDA_CHECK(cudaMemcpyAsync(mSpec.tablePtrs.device.rawPointer(), tablePtrs, static_cast<size_t>(tablePtrBytes),
        cudaMemcpyHostToDevice, mStream));

    // A round draws at most B*(K-1) draft plus B*(2K-1) accept values and commits at least one
    // code, so one frame cannot outrun a pool this size.
    mSpec.uniformPoolSize = static_cast<int64_t>(mNumRvqLayers) * maxBS * 3 * k;
    mSpec.uniformPool
        = rt::Tensor({mSpec.uniformPoolSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "cpUniformPool");

    LOG_INFO("Code predictor speculative decoding enabled: verify window %d", mSpec.verifySize);
}

void CodePredictor::applyRowLmHeads(
    __half const* hiddens, int32_t hiddenPlane, int32_t headPlane, int32_t rows, cudaStream_t stream)
{
    int64_t const stride = static_cast<int64_t>(mMaxBatch) * mSpec.verifySize;
    auto const* const base = mSpec.rowIdx.device.dataPointer<int32_t>();
    ELLM_CHECK(mSpec.logitsFp16.reshape({rows, mCodebookSize}), "Tensor reshape failed");
    kernel::invokeGroupedHeadLinear(hiddens, mWeights.lmHeads, base + hiddenPlane * stride, base + headPlane * stride,
        rows, static_cast<int32_t>(mHiddenSize), mCodebookSize, mSpec.logitsFp16, stream);
}

void CodePredictor::uploadRowSelectors(int32_t activeBatchSize, int32_t const* hostM, int32_t n, cudaStream_t stream)
{
    int32_t const seqLen = n + 1;
    int64_t const stride = static_cast<int64_t>(mMaxBatch) * mSpec.verifySize;
    int32_t* const host = mSpec.rowIdx.host.dataPointer<int32_t>();

    // Producing code_k scores hidden k-1 through lm_head k-1 and embeds its input through
    // codec table k-2, which is where the three clamped ladders below come from.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        for (int32_t t = 0; t < n; ++t)
        {
            int64_t const row = static_cast<int64_t>(b) * n + t;
            host[0 * stride + row] = b; // every draft of a batch reads the one cached hidden
            host[1 * stride + row] = std::clamp(hostM[b] + t, 0, mNumRvqLayers - 1);
        }
        for (int32_t i = 0; i < seqLen; ++i)
        {
            int64_t const row = static_cast<int64_t>(b) * seqLen + i;
            host[2 * stride + row] = static_cast<int32_t>(row);
            host[3 * stride + row] = std::clamp(hostM[b] + i, 0, mNumRvqLayers - 1);
            host[4 * stride + row] = std::clamp(hostM[b] - 1 + i, 0, mNumRvqLayers - 1);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mSpec.rowIdx.device.rawPointer(), host,
        static_cast<size_t>(5 * stride) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

void CodePredictor::buildVerifyInput(int32_t activeBatchSize, int32_t n, cudaStream_t stream)
{
    int64_t const talkerH = mTalkerHiddenSize;
    int64_t const cpH = mHiddenSize;
    int32_t const seqLen = n + 1;
    int64_t const rows = static_cast<int64_t>(activeBatchSize) * seqLen;

    // Column 0 already holds the last committed code; the drafts fill 1..n.
    if (n > 0)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(mSpec.verifyCodeIds.dataPointer<int32_t>() + 1, mSpec.verifySize * sizeof(int32_t),
            mSpec.draftTokenIds.dataPointer<int32_t>(), n * sizeof(int32_t), n * sizeof(int32_t), activeBatchSize,
            cudaMemcpyDeviceToDevice, stream));
    }

    int64_t const stride = static_cast<int64_t>(mMaxBatch) * mSpec.verifySize;
    int32_t const* const rowTableIdx = mSpec.rowIdx.device.dataPointer<int32_t>() + 4 * stride;
    auto const* const tablePtrs = static_cast<__half const* const*>(mSpec.tablePtrs.device.rawPointer());

    if (mWeights.useSmallToMtpProjection)
    {
        ELLM_CHECK(mSpec.rawCodecEmbedRows.reshape({rows, talkerH}), "Tensor reshape failed");
        kernel::invokeGatherCodecEmbedRows(mSpec.verifyCodeIds, tablePtrs, rowTableIdx, activeBatchSize, seqLen,
            mSpec.verifySize, static_cast<int32_t>(talkerH), mSpec.rawCodecEmbedRows, stream);
        ELLM_CHECK(mSpec.verifyInput.reshape({rows, cpH}), "Tensor reshape failed");
        kernel::invokeLinearLayer(
            mSpec.rawCodecEmbedRows, mWeights.smallToMtpWeight, mWeights.smallToMtpBias, mSpec.verifyInput, stream);
        return;
    }
    ELLM_CHECK(mSpec.verifyInput.reshape({rows, cpH}), "Tensor reshape failed");
    kernel::invokeGatherCodecEmbedRows(mSpec.verifyCodeIds, tablePtrs, rowTableIdx, activeBatchSize, seqLen,
        mSpec.verifySize, static_cast<int32_t>(cpH), mSpec.verifyInput, stream);
}

void CodePredictor::storeCodecHiddensForAccepted(rt::Tensor const& codeIds, int32_t activeBatchSize, int32_t stride,
    int32_t const* hostM, int32_t offset, cudaStream_t stream, int32_t const* counts)
{
    int64_t const planeStride = static_cast<int64_t>(mMaxBatch) * mSpec.verifySize;
    int32_t* const host = mSpec.storeIdx.host.dataPointer<int32_t>();

    int32_t maxCount = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        maxCount = std::max(maxCount, counts != nullptr ? counts[b] : 1);
    }
    if (maxCount == 0)
    {
        return;
    }

    // Rows are [batch, maxCount]; a batch that committed fewer codes leaves a negative
    // destination, which the gather kernel drops rather than writing a stale row.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const count = counts != nullptr ? counts[b] : 1;
        for (int32_t i = 0; i < maxCount; ++i)
        {
            int64_t const row = static_cast<int64_t>(b) * maxCount + i;
            int32_t const p = hostM[b] + offset + i;
            bool const keep = i < count && p >= 1 && p <= mNumRvqLayers - 1;
            host[0 * planeStride + row] = keep ? b * mNumCodesPerFrame + p : -1;
            host[1 * planeStride + row] = keep ? p - 1 : 0;
        }
    }

    int64_t const usedRows = static_cast<int64_t>(activeBatchSize) * maxCount;
    CUDA_CHECK(cudaMemcpyAsync(mSpec.storeIdx.device.dataPointer<int32_t>(), host,
        static_cast<size_t>(planeStride + usedRows) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    kernel::invokeGatherCodecEmbedRows(codeIds, static_cast<__half const* const*>(mSpec.tablePtrs.device.rawPointer()),
        mSpec.storeIdx.device.dataPointer<int32_t>() + planeStride, activeBatchSize, maxCount, stride,
        static_cast<int32_t>(mTalkerHiddenSize), *mBinding.codecHiddens, stream,
        mSpec.storeIdx.device.dataPointer<int32_t>());
}

bool CodePredictor::executeVerifyStep(int32_t activeBatchSize, int32_t seqLen, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::CodePredictor::verify", nvtx_colors::ORANGE);

    mPipelineIO->waitForStepHostStaging();

    int32_t const* const hostBaseLen = mSpec.hostBaseLen.dataPointer<int32_t>();
    std::fill_n(mStepQueryLengths.begin(), activeBatchSize, seqLen);
    std::copy_n(hostBaseLen, activeBatchSize, mStepPastLengths.begin());

    bool const shapeChanged = mSpec.verifyBoundBatch != activeBatchSize || mSpec.verifyBoundSeqLen != seqLen;
    if (shapeChanged)
    {
        mTensorMap.set(binding_names::kInputsEmbeds, mSpec.verifyInput);
        mTensorMap.set(binding_names::kLogits, mFrame.logits);
        mTensorMap.set(binding_names::kOutputHiddenStates, mSpec.verifyHidden);
        mTensorMap.set(binding_names::kLmHeads, mWeights.lmHeads);
        mTensorMap.set(binding_names::kLmHeadIdx, mFrame.lmHeadIdx);
    }

    prepareRaggedStep(mRagged, *mPipelineIO, *mResources, mEngineConfig, mStepQueryLengths, mStepPastLengths,
        activeBatchSize, seqLen, rt::ExecutionPhase::kContextChunk, stream);

    // Context-chunk semantics over a multi-token query, but on the generation profile: the
    // prefill profile would drop the pass off the XQA path and cost more than the
    // autoregressive steps it stands in for.
    if (shapeChanged)
    {
        auto const dims = mEngineConfig.prefillDims(activeBatchSize, seqLen, rt::ExecutionPhase::kContextChunk);
        if (!mExec->prepare(/*decodeProfile=*/1, dims, mTensorMap, stream))
        {
            LOG_ERROR("Code predictor verify prepare failed (bs=%d, seqLen=%d)", activeBatchSize, seqLen);
            return false;
        }
        mSpec.verifyBoundBatch = activeBatchSize;
        mSpec.verifyBoundSeqLen = seqLen;
    }
    if (!mExec->execute(stream))
    {
        LOG_ERROR("Code predictor verify execute failed (bs=%d, seqLen=%d)", activeBatchSize, seqLen);
        return false;
    }
    return true;
}

void CodePredictor::logAndResetSpecStats()
{
    if (mSpec.verifySize == 0 || mSpec.rounds == 0)
    {
        return;
    }
    LOG_INFO("CP SPD: K=%d rounds=%llu proposed=%llu codes=%llu | %.3f codes/round, accept=%.3f", mSpec.verifySize,
        static_cast<unsigned long long>(mSpec.rounds), static_cast<unsigned long long>(mSpec.proposed),
        static_cast<unsigned long long>(mSpec.codes),
        static_cast<double>(mSpec.codes) / static_cast<double>(mSpec.rounds),
        mSpec.proposed ? static_cast<double>(mSpec.codes - mSpec.rounds) / static_cast<double>(mSpec.proposed) : 0.0);
    mSpec.rounds = 0;
    mSpec.proposed = 0;
    mSpec.codes = 0;
}

bool CodePredictor::runSpecDecodeFrame(FramePlan const& plan, int32_t activeBatchSize,
    std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream)
{
    int64_t const cpH = mHiddenSize;
    int64_t const vocab = mCodebookSize;
    int32_t const maxDraft = mSpec.verifySize - 1;
    auto& cacheManager = *mResources->cacheManagers[0];
    auto const& sampling = *plan.sampling;

    int32_t* const hostM = mSpec.hostM.dataPointer<int32_t>();
    int32_t* const hostBaseLen = mSpec.hostBaseLen.dataPointer<int32_t>();
    int32_t* const hostProposal = mSpec.proposalLengths.host.dataPointer<int32_t>();
    auto* const verifyIdBase = mSpec.verifyCodeIds.dataPointer<int32_t>();

    // The prefill left h_0 at row 1 of each batch slot and code_1 in mFrame.selectedIndices; the
    // per-batch state starts from there.
    auto const* const prefillHidden = static_cast<__half const*>(mFrame.hiddenStates.rawPointer());
    auto* const draftHidden = static_cast<__half*>(mSpec.draftHidden.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostM[b] = 1;
        hostBaseLen[b] = kCpPrefillSeqLen;
        CUDA_CHECK(cudaMemcpyAsync(verifyIdBase + b * mSpec.verifySize,
            mFrame.selectedIndices.dataPointer<int32_t>() + b, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(draftHidden + b * cpH, prefillHidden + (b * kCpPrefillSeqLen + 1) * cpH,
            cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    storeCodecHiddensForAccepted(mSpec.verifyCodeIds, activeBatchSize, mSpec.verifySize, hostM, 0, stream);

    kernel::dsparkFillUniforms(mSpec.uniformPool, static_cast<int32_t>(mSpec.uniformPoolSize), kCpSpecSamplingSeed,
        mSpec.randomOffset, stream);
    mSpec.randomOffset += static_cast<uint64_t>(mSpec.uniformPoolSize);
    auto* const uniformPool = static_cast<float*>(mSpec.uniformPool.rawPointer());
    int64_t uniformCursor = 0;
    auto takeUniforms = [&](int64_t count) {
        if (uniformCursor + count > mSpec.uniformPoolSize)
        {
            kernel::dsparkFillUniforms(mSpec.uniformPool, static_cast<int32_t>(mSpec.uniformPoolSize),
                kCpSpecSamplingSeed, mSpec.randomOffset, stream);
            mSpec.randomOffset += static_cast<uint64_t>(mSpec.uniformPoolSize);
            uniformCursor = 0;
        }
        float* const slice = uniformPool + uniformCursor;
        uniformCursor += count;
        return slice;
    };

    while (*std::min_element(hostM, hostM + activeBatchSize) < mNumRvqLayers)
    {
        // One engine call covers the batch, so the window is uniform; a batch that needs fewer
        // codes caps its own proposal length and drops the surplus on the host.
        int32_t maxRemaining{};
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            maxRemaining = std::max(maxRemaining, mNumRvqLayers - hostM[b]);
        }
        int32_t const n = std::max(0, std::min(maxDraft, maxRemaining - 1));
        int32_t const seqLen = n + 1;
        ++mSpec.rounds;
        mSpec.proposed += static_cast<uint64_t>(n);
        ELLM_CHECK(mSpec.acceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
        uploadRowSelectors(activeBatchSize, hostM, n, stream);

        if (n > 0)
        {
            float const* const draftUniforms = takeUniforms(static_cast<int64_t>(activeBatchSize) * n);
            applyRowLmHeads(draftHidden, /*hiddenPlane=*/0, /*headPlane=*/1, activeBatchSize * n, stream);
            ELLM_CHECK(mSpec.draftProbs.reshape({activeBatchSize * n, vocab}), "Tensor reshape failed");
            kernel::cpSpecTopKTopPProbs(mSpec.logitsFp16, mSpec.draftProbs, activeBatchSize * n,
                static_cast<int32_t>(vocab), sampling.temperature, sampling.topK, sampling.topP, stream);

            // Draft slot (b, t) is row b * n + t of both the probability and the uniform buffer,
            // so the whole block samples in one launch.
            ELLM_CHECK(mSpec.draftTokenIds.reshape({activeBatchSize, n}), "Tensor reshape failed");
            kernel::cpSpecSampleRows(mSpec.draftProbs, draftUniforms, mSpec.draftTokenIds, activeBatchSize * n,
                static_cast<int32_t>(vocab), stream);
        }

        buildVerifyInput(activeBatchSize, n, stream);

        ELLM_CHECK(mSpec.verifyInput.reshape({activeBatchSize * seqLen, cpH}), "Tensor reshape failed");
        ELLM_CHECK(mSpec.verifyHidden.reshape({activeBatchSize * seqLen, cpH}), "Tensor reshape failed");
        if (!executeVerifyStep(activeBatchSize, seqLen, stream))
        {
            return false;
        }

        applyRowLmHeads(static_cast<__half const*>(mSpec.verifyHidden.rawPointer()), /*hiddenPlane=*/2, /*headPlane=*/3,
            activeBatchSize * seqLen, stream);
        ELLM_CHECK(mSpec.targetProbs.reshape({activeBatchSize * seqLen, vocab}), "Tensor reshape failed");
        kernel::cpSpecTopKTopPProbs(mSpec.logitsFp16, mSpec.targetProbs, activeBatchSize * seqLen,
            static_cast<int32_t>(vocab), sampling.temperature, sampling.topK, sampling.topP, stream);

        if (n == 0)
        {
            ELLM_CHECK(mSpec.acceptedTokenIds.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            float const* const bonusUniforms = takeUniforms(activeBatchSize);
            ELLM_CHECK(mSpec.targetProbs.reshape({activeBatchSize, vocab}), "Tensor reshape failed");
            kernel::cpSpecSampleRows(mSpec.targetProbs, bonusUniforms, mSpec.acceptedTokenIds, activeBatchSize,
                static_cast<int32_t>(vocab), stream);
            CUDA_CHECK(cudaMemsetAsync(mSpec.acceptLength.rawPointer(), 0, activeBatchSize * sizeof(int32_t), stream));
            kernel::incrementLengthTensor(mSpec.acceptLength, 1, stream);
        }
        else
        {
            int32_t const uniformStride = 2 * n + 1;
            float const* const acceptUniforms = takeUniforms(static_cast<int64_t>(activeBatchSize) * uniformStride);

            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                hostProposal[b] = std::max(0, std::min(n, mNumRvqLayers - hostM[b] - 1));
            }
            ELLM_CHECK(mSpec.proposalLengths.device.reshape({activeBatchSize}), "Tensor reshape failed");
            CUDA_CHECK(
                cudaMemcpyAsync(mSpec.proposalLengths.device.rawPointer(), mSpec.proposalLengths.host.rawPointer(),
                    activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            ELLM_CHECK(mSpec.targetProbs.reshape({activeBatchSize, seqLen, vocab}), "Tensor reshape failed");
            ELLM_CHECK(mSpec.draftProbs.reshape({activeBatchSize, n, vocab}), "Tensor reshape failed");
            ELLM_CHECK(mSpec.acceptedTokenIds.reshape({activeBatchSize, seqLen}), "Tensor reshape failed");
            kernel::cpSpecProbabilisticAccept(mSpec.targetProbs, mSpec.draftProbs, mSpec.draftTokenIds,
                mSpec.proposalLengths.device, acceptUniforms, mSpec.acceptedTokenIds, mSpec.acceptLength,
                activeBatchSize, n, n, static_cast<int32_t>(vocab), stream);
        }

        CUDA_CHECK(cudaMemcpyAsync(mSpec.hostAcceptLength.rawPointer(), mSpec.acceptLength.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(mSpec.hostAcceptedTokenIds.rawPointer(), mSpec.acceptedTokenIds.rawPointer(),
            activeBatchSize * seqLen * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int32_t* const acceptLens = mSpec.hostAcceptLength.dataPointer<int32_t>();
        int32_t const* const acceptedCodes = mSpec.hostAcceptedTokenIds.dataPointer<int32_t>();
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int32_t const accepted = acceptLens[b];
            ELLM_CHECK(accepted >= 0 && accepted <= seqLen, "code predictor spec decode accept length out of range");
            int32_t const usable = std::min(accepted, mNumRvqLayers - hostM[b]);
            acceptLens[b] = usable;
            if (b == 0)
            {
                mSpec.codes += static_cast<uint64_t>(usable);
            }
            for (int32_t i = 0; i < usable; ++i)
            {
                outputCodesPerBatch[static_cast<size_t>(b)].push_back(acceptedCodes[b * seqLen + i]);
            }
            // Only the first `usable` verify positions were fed a surviving code; the rest of
            // the KV suffix the engine wrote is stale and gets overwritten next round.
            if (usable == 0)
            {
                continue;
            }
            hostBaseLen[b] += usable;
            CUDA_CHECK(cudaMemcpyAsync(verifyIdBase + b * mSpec.verifySize,
                mSpec.acceptedTokenIds.dataPointer<int32_t>() + b * seqLen + (usable - 1), sizeof(int32_t),
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(draftHidden + b * cpH,
                static_cast<__half const*>(mSpec.verifyHidden.rawPointer()) + (b * seqLen + usable - 1) * cpH,
                cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(mSpec.acceptLength.rawPointer(), mSpec.hostAcceptLength.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        cacheManager.commitSequenceLength(mSpec.acceptLength, stream);
        storeCodecHiddensForAccepted(mSpec.acceptedTokenIds, activeBatchSize, seqLen, hostM, 1, stream, acceptLens);
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            hostM[b] += acceptLens[b];
        }
    }
    // Speculation needs two RVQ layers, so at least one round ran, and its wait covered the
    // prefill's copy of code_1.
    int32_t const* const codeOne = mFrame.hostSelectedCodeIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        auto& row = outputCodesPerBatch[static_cast<size_t>(b)];
        row.insert(row.begin() + 1, codeOne[b]);
    }
    return true;
}

void CodePredictor::allocateWorkspaces()
{
    int64_t const maxBS = mMaxBatch;
    int64_t const cpH = mHiddenSize;
    int64_t const talkerH = mTalkerHiddenSize;

    // Everything is claimed at maxBatch; a frame reshapes down to its active batch size, which
    // is what lets the framework primitives stay batch-size agnostic without reallocating.
    mStepQueryLengths.assign(static_cast<size_t>(maxBS), 1);
    mStepPastLengths.assign(static_cast<size_t>(maxBS), 0);

    mFrame.logits = rt::Tensor({maxBS, mCodebookSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "cpLogits");
    mFrame.lmHeadIdx = rt::Tensor({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpLmHeadIdx");
    mFrame.prefillInput = rt::Tensor(
        {maxBS, kCpPrefillSeqLen, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpPrefillInput");
    mFrame.codecIds = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpCodecIds");
    mFrame.codecEmbed = rt::Tensor({maxBS, 1, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpCodecEmbed");
    mFrame.rawCodecEmbed
        = rt::Tensor({maxBS, 1, talkerH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpRawCodecEmbed");
    mFrame.projectedHidden
        = rt::Tensor({maxBS, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpProjectedHidden");
    mFrame.hiddenStates = rt::Tensor(
        {maxBS, mNumCodesPerFrame, cpH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "cpHiddenStates");
    mFrame.selectedIndices
        = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "cpSelectedIndices");
    mFrame.hostSelectedCodeIds
        = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostSelectedCodeIds");
    mFrame.hostGenCodes = rt::Tensor(
        {mNumRvqLayers - 1, maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostGenCodeBuf");
    mFrame.hostReuseKVCacheLengths
        = rt::Tensor({maxBS}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "cpHostReuseKVCacheLengths");

    // Sized for the widest sampler shape a request can ask for: the top-k path grows with topK
    // and the top-p path with the vocabulary, and neither is known until the frame runs.
    SamplingParams const topPParams(mMaxBatch, mCodebookSize, /*temperature=*/1.0F, /*topK=*/0, /*topP=*/0.9F);
    SamplingParams const topKParams(mMaxBatch, mCodebookSize, /*temperature=*/1.0F, /*topK=*/mCodebookSize,
        /*topP=*/1.0F);
    int64_t const workspaceBytes = static_cast<int64_t>(
        std::max(trt_edgellm::getTopKtopPSamplingWorkspaceSize(mMaxBatch, mCodebookSize, topPParams),
            trt_edgellm::getTopKtopPSamplingWorkspaceSize(mMaxBatch, mCodebookSize, topKParams)));
    mFrame.samplingWorkspace
        = rt::Tensor({workspaceBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8, "cpSamplingWorkspace");
}

void CodePredictor::bind(Binding binding)
{
    ELLM_CHECK(mExec != nullptr, "code predictor bind() before preallocate()");
    ELLM_CHECK(binding.talkerHiddenSize == mTalkerHiddenSize,
        "the talker binds " + std::to_string(binding.talkerHiddenSize)
            + "-wide hidden states but the code predictor artifacts were exported for "
            + std::to_string(mTalkerHiddenSize));
    ELLM_CHECK(binding.numCodesPerFrame == mNumCodesPerFrame,
        "the talker expects " + std::to_string(binding.numCodesPerFrame)
            + " codes per frame; the code predictor produces " + std::to_string(mNumCodesPerFrame));
    ELLM_CHECK(binding.maxBatchSize >= 1 && binding.maxBatchSize <= mMaxBatch,
        "the talker asks the code predictor for batch " + std::to_string(binding.maxBatchSize) + ", outside 1.."
            + std::to_string(mMaxBatch));
    ELLM_CHECK(binding.codeZeroEmbedTable != nullptr && binding.codeZeroEmbedTable->getShape().getNumDims() == 2
            && binding.codeZeroEmbedTable->getShape()[1] == mTalkerHiddenSize,
        "the talker bound no [codecVocabSize, " + std::to_string(mTalkerHiddenSize) + "] code_0 embedding table");
    int64_t const codecHiddenBytes = static_cast<int64_t>(binding.maxBatchSize) * mNumCodesPerFrame * mTalkerHiddenSize
        * static_cast<int64_t>(sizeof(__half));
    ELLM_CHECK(binding.codecHiddens != nullptr && binding.codecHiddens->getMemoryCapacity() >= codecHiddenBytes,
        "the talker bound a codec-hidden buffer smaller than the [" + std::to_string(binding.maxBatchSize) + ", "
            + std::to_string(mNumCodesPerFrame) + ", " + std::to_string(mTalkerHiddenSize)
            + "] FP16 window the code predictor writes into");

    mBinding = binding;
    mBound = true;
}

bool CodePredictor::setContextMemory(rt::Tensor& memory)
{
    if (memory.getMemoryCapacity() < mContextMemoryBytes)
    {
        LOG_ERROR("Code predictor: %lld bytes of context memory offered, %lld needed",
            static_cast<long long>(memory.getMemoryCapacity()), static_cast<long long>(mContextMemoryBytes));
        return false;
    }
    if (!mExec->setContextMemory(memory))
    {
        return false;
    }
    mExecContextMemory = rt::Tensor{};
    return true;
}

rt::Tensor const& CodePredictor::codecEmbedTable(int32_t layer) const
{
    ELLM_CHECK(layer >= 0 && layer < mNumRvqLayers,
        "codec embedding table " + std::to_string(layer) + " is outside 0.." + std::to_string(mNumRvqLayers - 1));
    return mWeights.codecEmbeddingTables[static_cast<size_t>(layer)];
}

void CodePredictor::resetKvCache(int32_t batchSize, cudaStream_t stream)
{
    auto& cacheManager = *mResources->cacheManagers[0];
    ELLM_CHECK(mFrame.hostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mFrame.hostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    cacheManager.resetForNewSequences(mFrame.hostReuseKVCacheLengths, stream);
}

void CodePredictor::projectToCp(rt::Tensor const& source, rt::Tensor& output, int32_t batchSize, cudaStream_t stream)
{
    ELLM_CHECK(mWeights.useSmallToMtpProjection, "the code predictor artifacts carry no small_to_mtp projection");
    ELLM_CHECK(output.reshape({batchSize, mHiddenSize}), "Tensor reshape failed");
    kernel::invokeLinearLayer(source, mWeights.smallToMtpWeight, mWeights.smallToMtpBias, output, stream);
}

bool CodePredictor::executePrefillStep(
    rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::CodePredictor::prefill", nvtx_colors::ORANGE);

    auto const inputShape = inputsEmbeds.getShape();
    ELLM_CHECK(inputShape.getNumDims() == 3, "code predictor prefill needs a 3D [batch, seqLen, hidden] input");
    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    ELLM_CHECK(batchSize <= mMaxBatch, "code predictor prefill batch exceeds max batch");

    auto& cacheManager = *mResources->cacheManagers[0];

    ELLM_CHECK(mPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mPipelineIO->hostContextLengths.dataPointer<int32_t>(), batchSize, static_cast<int32_t>(seqLen));

    CUDA_CHECK(cudaMemsetAsync(mFrame.lmHeadIdx.rawPointer(), 0, sizeof(int32_t), stream));

    mTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputsEmbeds));
    mTensorMap.set(binding_names::kLogits, outputLogits);
    mTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);
    mTensorMap.set(binding_names::kLmHeads, mWeights.lmHeads);
    mTensorMap.set(binding_names::kLmHeadIdx, mFrame.lmHeadIdx);

    mStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), cacheManager, *mPipelineIO, stream);

    // One code-predictor sequence per frame with its cache reset, so this is always a cold context.
    auto const phase = rt::ExecutionPhase::kContextPrefill;
    mPastLen = 0;
    std::fill_n(mStepQueryLengths.begin(), batchSize, static_cast<int32_t>(seqLen));
    std::fill_n(mStepPastLengths.begin(), batchSize, 0);
    prepareRaggedStep(mRagged, *mPipelineIO, *mResources, mEngineConfig, mStepQueryLengths, mStepPastLengths,
        static_cast<int32_t>(batchSize), static_cast<int32_t>(seqLen), phase, stream);
    auto const prefillDims = mEngineConfig.prefillDims(batchSize, seqLen, phase);
    mSpec.verifyBoundSeqLen = -1;
    if (!mExec->prepare(/*prefillProfile=*/0, prefillDims, mTensorMap, stream))
    {
        LOG_ERROR("Code predictor prefill prepare failed");
        return false;
    }
    if (!mExec->execute(stream))
    {
        LOG_ERROR("Code predictor prefill execute failed");
        return false;
    }
    cacheManager.commitSequenceLength(mPipelineIO->contextLengths, stream);
    mPastLen = static_cast<int32_t>(seqLen);
    return true;
}

bool CodePredictor::prepareDecodeBindings(int32_t activeBatchSize, cudaStream_t stream)
{
    int64_t const cpH = mHiddenSize;
    ELLM_CHECK(mFrame.codecEmbed.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
    ELLM_CHECK(mFrame.hiddenStates.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");

    mTensorMap.set(binding_names::kInputsEmbeds, mFrame.codecEmbed);
    mTensorMap.set(binding_names::kLogits, mFrame.logits);
    mTensorMap.set(binding_names::kOutputHiddenStates, mFrame.hiddenStates);
    mTensorMap.set(binding_names::kLmHeads, mWeights.lmHeads);
    mTensorMap.set(binding_names::kLmHeadIdx, mFrame.lmHeadIdx);

    auto const decodeDims = mEngineConfig.decodeDims(activeBatchSize);
    mSpec.verifyBoundSeqLen = -1;
    if (!mExec->prepare(/*decodeProfile=*/1, decodeDims, mTensorMap, stream))
    {
        LOG_ERROR("Code predictor decode prepare failed (bs=%d)", activeBatchSize);
        return false;
    }
    return true;
}

bool CodePredictor::runFrame(
    FramePlan const& plan, std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::CodePredictor::runFrame", nvtx_colors::ORANGE);

    ELLM_CHECK(mBound, "code predictor runFrame() before bind()");
    ELLM_CHECK(
        plan.codeZero != nullptr && plan.talkerLastHidden != nullptr && plan.sampling != nullptr && plan.frameIdx >= 0,
        "code predictor frame plan is incomplete");

    int32_t const activeBatchSize = plan.activeBatchSize;
    int64_t const talkerH = mTalkerHiddenSize;
    int64_t const cpH = mHiddenSize;
    ELLM_CHECK(activeBatchSize >= 1 && activeBatchSize <= mBinding.maxBatchSize,
        "code predictor frame batch size out of range");
    ELLM_CHECK(plan.codeZero->getDeviceType() == rt::DeviceType::kCPU
            && plan.codeZero->getDataType() == trt_edgellm::DataType::kINT32
            && plan.codeZero->getShape().volume() >= activeBatchSize,
        "code predictor frame needs a host INT32 code_0 for each of its " + std::to_string(activeBatchSize)
            + " batches");
    int32_t const* const codeZero = plan.codeZero->dataPointer<int32_t>();

    // code_0 comes from the talker; codes 1..numRvqLayers are produced below. Rows keep their
    // capacity across frames, so the frame loop stays free of allocation.
    ELLM_CHECK(outputCodesPerBatch.size() >= static_cast<size_t>(activeBatchSize),
        "code predictor frame has fewer output rows than batches");
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        auto& row = outputCodesPerBatch[static_cast<size_t>(b)];
        row.clear();
        row.reserve(static_cast<size_t>(mNumCodesPerFrame));
        row.push_back(codeZero[b]);
    }

    ELLM_CHECK(mFrame.codecIds.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(
        mFrame.codecIds.rawPointer(), codeZero, activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ELLM_CHECK(mFrame.rawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
    kernel::embeddingLookup(mFrame.codecIds, *mBinding.codeZeroEmbedTable, std::nullopt, mFrame.rawCodecEmbed, stream);

    // Prefill row 0 is the talker's hidden state, row 1 the code_0 embedding, both in CP space.
    ELLM_CHECK(mFrame.prefillInput.reshape({activeBatchSize, kCpPrefillSeqLen, cpH}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_PREFILL, stream);

        rt::Tensor talkerHidden2D(const_cast<__half*>(static_cast<__half const*>(plan.talkerLastHidden->rawPointer())),
            rt::Coords{activeBatchSize, talkerH}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        rt::Tensor rawCodec2D(mFrame.rawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);

        void const* talkerCp = talkerHidden2D.rawPointer();
        void const* codecCp = rawCodec2D.rawPointer();
        if (mWeights.useSmallToMtpProjection)
        {
            projectToCp(talkerHidden2D, mFrame.projectedHidden, activeBatchSize, stream);
            projectToCp(rawCodec2D, mFrame.codecEmbed, activeBatchSize, stream);
            talkerCp = mFrame.projectedHidden.rawPointer();
            codecCp = mFrame.codecEmbed.rawPointer();
        }

        // Both are packed [B, cpH]; each lands in every row's own prefill position.
        size_t const rowBytes = static_cast<size_t>(cpH) * sizeof(__half);
        size_t const prefillPitch = static_cast<size_t>(kCpPrefillSeqLen) * rowBytes;
        auto* const prefillBase = static_cast<__half*>(mFrame.prefillInput.rawPointer());
        CUDA_CHECK(cudaMemcpy2DAsync(prefillBase, prefillPitch, talkerCp, rowBytes, rowBytes,
            static_cast<size_t>(activeBatchSize), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(prefillBase + cpH, prefillPitch, codecCp, rowBytes, rowBytes,
            static_cast<size_t>(activeBatchSize), cudaMemcpyDeviceToDevice, stream));

        ELLM_CHECK(mFrame.hiddenStates.reshape({activeBatchSize, kCpPrefillSeqLen, cpH}), "Tensor reshape failed");
        if (!executePrefillStep(mFrame.prefillInput, mFrame.logits, mFrame.hiddenStates, stream))
        {
            return false;
        }

        ELLM_CHECK(mFrame.logits.reshape({activeBatchSize, mCodebookSize}), "Tensor reshape failed");
        ELLM_CHECK(mFrame.selectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        SamplingParams const perCallParams(
            activeBatchSize, mCodebookSize, plan.sampling->temperature, plan.sampling->topK, plan.sampling->topP);
        trt_edgellm::topKtopPSamplingFromLogits(mFrame.logits, mFrame.selectedIndices, perCallParams,
            mFrame.samplingWorkspace, stream, kCpSamplingSeed,
            cpSamplingOffset(plan.frameIdx, /*codeIdx=*/1, mNumRvqLayers));
        // Read on the host after the frame's next wait; the device keeps code_1 in mFrame.selectedIndices.
        ELLM_CHECK(mFrame.hostSelectedCodeIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mFrame.hostSelectedCodeIds.rawPointer(), mFrame.selectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }

    // The talker's residual reads codecHiddens[b]: positions 0 and numRvqLayers are its own
    // embedding lookups, positions 1..numRvqLayers-1 are the raw embeddings saved below.
    ELLM_CHECK(mBinding.codecHiddens->reshape({activeBatchSize, mNumCodesPerFrame, talkerH}), "Tensor reshape failed");

    if (mSpec.verifySize > 0)
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        return runSpecDecodeFrame(plan, activeBatchSize, outputCodesPerBatch, stream);
    }

    // Each step reads the previous sample straight from mFrame.selectedIndices and copies it into row
    // step - 2 of the pinned buffer. The predictor has no EOS, so one sync after the loop drains
    // the frame. Entry: mFrame.selectedIndices holds code_1 from the prefill above.
    int32_t* const hostGenBuf = mFrame.hostGenCodes.dataPointer<int32_t>();
    int64_t const hostGenBufStride = mMaxBatch;
    int32_t const numGenSteps = mNumRvqLayers - 1;
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        if (!prepareDecodeBindings(activeBatchSize, stream))
        {
            return false;
        }
        auto& cacheManager = *mResources->cacheManagers[0];
        for (int32_t step = 2; step <= mNumRvqLayers; ++step)
        {
            int32_t const embedIdx = step - 2; // step=2 reads codec_embedding[0]
            int32_t const savePos = step - 1;  // 1..numRvqLayers-1

            kernel::incrementLengthTensor(mFrame.lmHeadIdx, 1, stream);

            ELLM_CHECK(mFrame.selectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            ELLM_CHECK(mFrame.codecEmbed.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
            auto const& embedTable = mWeights.codecEmbeddingTables[static_cast<size_t>(embedIdx)];
            // Without a projection talkerH == cpH, so the lookup can land in the engine input
            // itself. The residual still needs the raw embedding, and in that branch the engine
            // input IS the raw embedding -- it is read, never written, by the decode graph.
            void const* rawEmbed = mFrame.codecEmbed.rawPointer();
            if (mWeights.useSmallToMtpProjection)
            {
                ELLM_CHECK(mFrame.rawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
                kernel::embeddingLookup(mFrame.selectedIndices, embedTable, std::nullopt, mFrame.rawCodecEmbed, stream);
                rawEmbed = mFrame.rawCodecEmbed.rawPointer();
            }
            else
            {
                kernel::embeddingLookup(mFrame.selectedIndices, embedTable, std::nullopt, mFrame.codecEmbed, stream);
            }

            size_t const talkerRowBytes = static_cast<size_t>(talkerH) * sizeof(__half);
            CUDA_CHECK(cudaMemcpy2DAsync(
                static_cast<__half*>(mBinding.codecHiddens->rawPointer()) + static_cast<int64_t>(savePos) * talkerH,
                static_cast<size_t>(mNumCodesPerFrame) * talkerRowBytes, rawEmbed, talkerRowBytes, talkerRowBytes,
                static_cast<size_t>(activeBatchSize), cudaMemcpyDeviceToDevice, stream));

            if (mWeights.useSmallToMtpProjection)
            {
                rt::Tensor rawCodec2D(mFrame.rawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH},
                    rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
                projectToCp(rawCodec2D, mFrame.codecEmbed, activeBatchSize, stream);
            }

            std::fill_n(mStepQueryLengths.begin(), activeBatchSize, 1);
            std::fill_n(mStepPastLengths.begin(), activeBatchSize, mPastLen);
            prepareRaggedStep(mRagged, *mPipelineIO, *mResources, mEngineConfig, mStepQueryLengths, mStepPastLengths,
                activeBatchSize, /*queryWidth=*/1, rt::ExecutionPhase::kAutoregressiveDecode, stream);
            if (!mExec->execute(stream))
            {
                LOG_ERROR("Code predictor decode execute failed (step=%d)", step);
                return false;
            }
            cacheManager.commitSequenceLength(/*increment=*/1, stream);
            ++mPastLen;

            ELLM_CHECK(mFrame.logits.reshape({activeBatchSize, mCodebookSize}), "Tensor reshape failed");
            SamplingParams const perCallParams(
                activeBatchSize, mCodebookSize, plan.sampling->temperature, plan.sampling->topK, plan.sampling->topP);
            trt_edgellm::topKtopPSamplingFromLogits(mFrame.logits, mFrame.selectedIndices, perCallParams,
                mFrame.samplingWorkspace, stream, kCpSamplingSeed,
                cpSamplingOffset(plan.frameIdx, /*codeIdx=*/step, mNumRvqLayers));

            CUDA_CHECK(cudaMemcpyAsync(hostGenBuf + (step - 2) * hostGenBufStride, mFrame.selectedIndices.rawPointer(),
                activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        int32_t const* const codeOne = mFrame.hostSelectedCodeIds.dataPointer<int32_t>();
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            outputCodesPerBatch[static_cast<size_t>(b)].push_back(codeOne[b]);
        }
        for (int32_t row = 0; row < numGenSteps; ++row)
        {
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                outputCodesPerBatch[static_cast<size_t>(b)].push_back(hostGenBuf[row * hostGenBufStride + b]);
            }
        }
    }

    return true;
}

bool CodePredictor::captureDecodeGraphs(cudaStream_t stream)
{
    ELLM_CHECK(mExec != nullptr, "code predictor graph capture before preallocate()");

    // The active head is gathered in-engine from the device head index, so the decode bindings
    // are step-invariant and one graph per batch size covers every codebook.
    bool captured = true;
    auto& cacheManager = *mResources->cacheManagers[0];
    for (int32_t bs = 1; bs <= mMaxBatch; ++bs)
    {
        int32_t const simulatedLength = std::min(kCpGraphCaptureCacheLength, mEngineConfig.maxKVCacheCapacity - 1);
        std::vector<int32_t> lengths(static_cast<size_t>(bs), simulatedLength);
        rt::Tensor simulatedReuse(lengths.data(), rt::Coords{bs}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        cacheManager.resetForNewSequences(simulatedReuse, stream);

        CUDA_CHECK(cudaMemsetAsync(mFrame.lmHeadIdx.rawPointer(), 0, sizeof(int32_t), stream));
        // Every batch size is attempted even after one fails: a missing graph slot only costs
        // the enqueueV3 fallback, so losing bs=1 must not cost the rest their capture.
        captured &= prepareDecodeBindings(bs, stream);
        std::vector<int32_t> const captureQuery(static_cast<size_t>(bs), 1);
        std::vector<int32_t> const capturePast(static_cast<size_t>(bs), simulatedLength);
        prepareRaggedStep(mRagged, *mPipelineIO, *mResources, mEngineConfig, captureQuery, capturePast, bs,
            /*queryWidth=*/1, rt::ExecutionPhase::kAutoregressiveDecode, stream);
        captured &= mExec->captureGraph(stream);
    }

    // A verify pass binds seqLen positions, a shape no decode graph above covers, and the
    // proposal shrinks as a frame runs out of codes, so every width in [1, K] occurs.
    for (int32_t bs = 1; mSpec.verifySize > 0 && bs <= mMaxBatch; ++bs)
    {
        for (int32_t seqLen = 1; seqLen <= mSpec.verifySize; ++seqLen)
        {
            int32_t const simulatedLength
                = std::min(kCpGraphCaptureCacheLength, mEngineConfig.maxKVCacheCapacity - 1 - mSpec.verifySize);
            std::vector<int32_t> lengths(static_cast<size_t>(bs), simulatedLength);
            rt::Tensor simulatedReuse(
                lengths.data(), rt::Coords{bs}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
            cacheManager.resetForNewSequences(simulatedReuse, stream);

            ELLM_CHECK(mSpec.verifyInput.reshape({bs * seqLen, mHiddenSize}), "Tensor reshape failed");
            ELLM_CHECK(mSpec.verifyHidden.reshape({bs * seqLen, mHiddenSize}), "Tensor reshape failed");
            std::fill_n(mSpec.hostBaseLen.dataPointer<int32_t>(), bs, simulatedLength);
            captured &= executeVerifyStep(bs, seqLen, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            captured &= mExec->captureGraph(stream);
        }
    }

    // The simulated lengths above leave both the per-batch lengths and the engine-written KV
    // contents mid-sequence; a real frame expects an empty cache.
    resetKvCache(mMaxBatch, stream);
    return captured;
}

} // namespace qwen3_omni
} // namespace rt
} // namespace trt_edgellm
