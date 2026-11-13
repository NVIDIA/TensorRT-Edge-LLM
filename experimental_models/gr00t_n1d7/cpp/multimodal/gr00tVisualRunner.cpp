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

#include "multimodal/gr00tVisualRunner.h"

#include "common/gr00tBindings.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

constexpr char const* kEmbeddingTensor = "embed_tokens.weight";
constexpr int64_t kChannels = 3;
constexpr int32_t kEmbeddingRank = 2; //!< [vocab, hidden]
constexpr int64_t kMaxPixel = 255;

} // namespace

Gr00tVisualRunner::Gr00tVisualRunner(std::string const& engineDir, int32_t maxSequenceLength, int32_t headDim,
    int32_t shortestEdge, float cropFraction, cudaStream_t stream)
    : mHeadDim(headDim)
    , mMaxSequenceLength(maxSequenceLength)
    , mShortestEdge(shortestEdge)
    , mCropFraction(cropFraction)
{
    std::filesystem::path const root(engineDir);
    std::filesystem::path const visualDir = root / "visual";
    mImageTokenId = loadJsonFile(visualDir / "config.json", "visual config").at("image_token_id").get<int32_t>();

    mViT = rt::MultimodalRunner::create(
        visualDir.string(), /*llmMaxBatchSize=*/1, maxSequenceLength, stream, /*checkpointDir=*/"");
    if (!mViT)
    {
        throw std::runtime_error("GR00T could not create the vision runner from " + visualDir.string());
    }
    int64_t const contextBytes = std::max<int64_t>(mViT->getRequiredContextMemorySize(), 1);
    mContextMemory
        = rt::Tensor({contextBytes}, rt::DeviceType::kGPU, trt_edgellm::DataType::kUINT8, "gr00t::visualContextMemory");
    if (!mViT->setContextMemory(mContextMemory))
    {
        throw std::runtime_error("GR00T could not bind the vision runner's context memory");
    }

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!mTokenizer->loadFromHF(root / "tokenizer"))
    {
        throw std::runtime_error("GR00T could not load the tokenizer from " + (root / "tokenizer").string());
    }
    auto const specialToken = [this](tokenizer::Rank id) {
        return id >= 0 ? mTokenizer->idToPiece(id, /*skipSpecialTokens=*/false) : std::string{};
    };
    if (!mChatTemplate.load(
            root / "tokenizer", specialToken(mTokenizer->getBosId()), specialToken(mTokenizer->getEosId())))
    {
        throw std::runtime_error("GR00T could not load the chat template from " + (root / "tokenizer").string());
    }

    std::vector<rt::Tensor> tables;
    if (!rt::safetensors::loadSafetensors(root / "embed_tokens.safetensors", tables, stream))
    {
        throw std::runtime_error("GR00T could not load embed_tokens.safetensors");
    }
    for (auto& table : tables)
    {
        if (table.getName() == kEmbeddingTensor)
        {
            mEmbeddingTable = std::move(table);
        }
    }
    if (mEmbeddingTable.getShape().getNumDims() != kEmbeddingRank)
    {
        throw std::runtime_error("GR00T embed_tokens.safetensors does not hold a [vocab, hidden] table");
    }
    mHiddenSize = static_cast<int32_t>(mEmbeddingTable.getShape()[1]);

    // Sized once at the engine's maximum context: a policy re-encodes every step,
    // so nothing here may allocate per observation.
    mInputs.inputsEmbeds = rt::Tensor({1, mMaxSequenceLength, mHiddenSize}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "gr00t::inputsEmbeds");
    mInputs.ropeCosSin = rt::Tensor(
        {1, mMaxSequenceLength, mHeadDim}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "gr00t::mropeCosSin");
    mInputs.positionIds = rt::Tensor(
        {1, mMaxSequenceLength}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "gr00t::positionIds");
    mPositionHost = rt::Tensor(
        {1, mMaxSequenceLength}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "gr00t::positionIdsHost");
    for (int32_t token = 0; token < mMaxSequenceLength; ++token)
    {
        mPositionHost.dataPointer<int32_t>()[token] = token;
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputs.positionIds.rawPointer(), mPositionHost.rawPointer(),
        static_cast<size_t>(mMaxSequenceLength) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mInputIds
        = rt::Tensor({1, mMaxSequenceLength}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "gr00t::inputIds");
    mInputIdsHost = rt::Tensor(
        {1, mMaxSequenceLength}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32, "gr00t::inputIdsHost");
    mMultimodalIndices = rt::Tensor(
        {1, mMaxSequenceLength}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "gr00t::multimodalIndices");
    int32_t const deepstackCount = static_cast<int32_t>(mViT->getDeepstackFeatures().size());
    for (int32_t index = 0; index < deepstackCount; ++index)
    {
        mInputs.deepstackEmbeds.emplace_back(rt::Coords{1, mMaxSequenceLength, mHiddenSize}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "gr00t::deepstackEmbeds");
    }
}

namespace
{

//! cv2's INTER_AREA along one axis: every output pixel averages the source
//! interval it covers, up- and downscale alike.
void areaTaps(int64_t inputSize, int64_t outputSize, AxisTaps& axis)
{
    if (axis.inputSize == inputSize && axis.outputSize == outputSize)
    {
        return;
    }
    axis.inputSize = inputSize;
    axis.outputSize = outputSize;
    std::vector<AreaTap>& taps = axis.taps;
    double const step = static_cast<double>(inputSize) / static_cast<double>(outputSize);
    taps.resize(static_cast<size_t>(outputSize));
    for (int64_t index = 0; index < outputSize; ++index)
    {
        double const low = static_cast<double>(index) * step;
        double const high = low + step;
        AreaTap& tap = taps[static_cast<size_t>(index)];
        tap.first = std::min<int64_t>(static_cast<int64_t>(low), inputSize - 1);
        tap.weights.clear();
        double total = 0.0;
        for (int64_t source = tap.first; source < inputSize && static_cast<double>(source) < high; ++source)
        {
            double const overlap
                = std::min(high, static_cast<double>(source + 1)) - std::max(low, static_cast<double>(source));
            tap.weights.push_back(static_cast<float>(overlap));
            total += overlap;
        }
        for (float& weight : tap.weights)
        {
            weight = static_cast<float>(weight / total);
        }
    }
}

} // namespace

void smallestMaxSize(
    std::vector<unsigned char>& pixels, int64_t& height, int64_t& width, int64_t channels, int64_t maxSize)
{
    ResizeScratch scratch;
    smallestMaxSize(pixels, height, width, channels, maxSize, scratch);
}

void smallestMaxSize(std::vector<unsigned char>& pixels, int64_t& height, int64_t& width, int64_t channels,
    int64_t maxSize, ResizeScratch& scratch)
{
    int64_t const shorter = std::min(height, width);
    if (shorter == maxSize || shorter <= 0)
    {
        return;
    }
    double const scale = static_cast<double>(maxSize) / static_cast<double>(shorter);
    auto const target = [scale](int64_t side) {
        return std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(static_cast<double>(side) * scale)));
    };
    int64_t const newHeight = target(height);
    int64_t const newWidth = target(width);

    areaTaps(width, newWidth, scratch.horizontal);
    areaTaps(height, newHeight, scratch.vertical);
    std::vector<AreaTap> const& horizontal = scratch.horizontal.taps;
    std::vector<AreaTap> const& vertical = scratch.vertical.taps;
    std::vector<float>& rows = scratch.rows;
    rows.assign(static_cast<size_t>(height * newWidth * channels), 0.0F);
    for (int64_t row = 0; row < height; ++row)
    {
        for (int64_t column = 0; column < newWidth; ++column)
        {
            AreaTap const& tap = horizontal[static_cast<size_t>(column)];
            for (int64_t channel = 0; channel < channels; ++channel)
            {
                float sum = 0.0F;
                for (size_t k = 0; k < tap.weights.size(); ++k)
                {
                    sum += tap.weights[k]
                        * static_cast<float>(pixels[static_cast<size_t>(
                            (row * width + tap.first + static_cast<int64_t>(k)) * channels + channel)]);
                }
                rows[static_cast<size_t>((row * newWidth + column) * channels + channel)] = sum;
            }
        }
    }
    std::vector<unsigned char>& out = scratch.out;
    out.resize(static_cast<size_t>(newHeight * newWidth * channels));
    for (int64_t row = 0; row < newHeight; ++row)
    {
        AreaTap const& tap = vertical[static_cast<size_t>(row)];
        for (int64_t column = 0; column < newWidth; ++column)
        {
            for (int64_t channel = 0; channel < channels; ++channel)
            {
                float sum = 0.0F;
                for (size_t k = 0; k < tap.weights.size(); ++k)
                {
                    sum += tap.weights[k]
                        * rows[static_cast<size_t>(
                            ((tap.first + static_cast<int64_t>(k)) * newWidth + column) * channels + channel)];
                }
                out[static_cast<size_t>((row * newWidth + column) * channels + channel)]
                    = static_cast<unsigned char>(std::clamp<int64_t>(std::lround(sum), 0, kMaxPixel));
            }
        }
    }
    pixels.swap(out);
    height = newHeight;
    width = newWidth;
}

void Gr00tVisualRunner::applyEvalTransform(Gr00tFrame const& frame, rt::imageUtils::ImageData& view)
{
    if (frame.rgb == nullptr || frame.height <= 0 || frame.width <= 0)
    {
        throw std::invalid_argument("GR00T camera frame carries no pixels");
    }
    int64_t height = frame.height;
    int64_t width = frame.width;
    std::vector<unsigned char>& pixels = mPixels;
    pixels.assign(frame.rgb, frame.rgb + static_cast<size_t>(height * width * kChannels));

    smallestMaxSize(pixels, height, width, kChannels, mShortestEdge, mResize);

    int64_t const cropHeight = std::max<int64_t>(1, static_cast<int64_t>(height * mCropFraction));
    int64_t const cropWidth = std::max<int64_t>(1, static_cast<int64_t>(width * mCropFraction));
    int64_t const top = (height - cropHeight) / 2;
    int64_t const left = (width - cropWidth) / 2;
    std::vector<unsigned char>& cropped = mCropped;
    cropped.resize(static_cast<size_t>(cropHeight * cropWidth * kChannels));
    for (int64_t row = 0; row < cropHeight; ++row)
    {
        std::copy_n(pixels.data() + ((top + row) * width + left) * kChannels, cropWidth * kChannels,
            cropped.data() + row * cropWidth * kChannels);
    }
    pixels.swap(cropped);
    height = cropHeight;
    width = cropWidth;

    smallestMaxSize(pixels, height, width, kChannels, mShortestEdge, mResize);

    // The view keeps its pinned buffer across frames so the tower's upload stays asynchronous.
    rt::Coords const shape{1, height, width, kChannels};
    if (!view.buffer || !view.buffer->reshape(shape))
    {
        view = rt::imageUtils::ImageData(
            rt::Tensor(shape, rt::DeviceType::kCPU, trt_edgellm::DataType::kUINT8, "gr00t::view"));
    }
    else
    {
        view.height = height;
        view.width = width;
        view.layout.pitchBytes[0] = width * kChannels;
        view.layout.frameStrideBytes[0] = view.frameBytes();
    }
    std::copy(pixels.begin(), pixels.end(), view.data());
    view.doResize = false;
}

Gr00tVlInputs const& Gr00tVisualRunner::encode(
    std::vector<Gr00tFrame> const& frames, std::string const& instruction, cudaStream_t stream)
{
    // The reference builds one user turn holding every view followed by the
    // instruction, and applies the chat template without a generation prompt.
    rt::Message message;
    message.role = "user";
    for (size_t view = 0; view < frames.size(); ++view)
    {
        message.contents.push_back({"image", ""});
    }
    message.contents.push_back({"text", instruction});

    if (mViews.size() < frames.size())
    {
        mViews.resize(frames.size());
    }
    rt::LLMGenerationRequest::Request one;
    one.messages.push_back(std::move(message));
    one.imageBuffers.reserve(frames.size());
    for (size_t view = 0; view < frames.size(); ++view)
    {
        applyEvalTransform(frames[view], mViews[view]);
        one.imageBuffers.push_back(mViews[view]);
    }

    rt::LLMGenerationRequest request;
    request.requests.push_back(std::move(one));
    request.applyChatTemplate = true;
    request.addGenerationPrompt = false;

    // The shared runner tokenizes the formatted prompt, so the checkpoint's own
    // Jinja template has to be rendered before it is handed the request.
    request.formattedRequests.resize(request.requests.size());
    if (!mChatTemplate.apply(request.requests.front(), request.formattedRequests.front(),
            chat_template::ChatTemplate::optionsFrom(request)))
    {
        throw std::runtime_error("GR00T could not apply the chat template to the observation");
    }

    Gr00tVlInputs& io = mInputs;
    std::vector<std::vector<int32_t>> batchedInputIds;
    if (!mViT->preprocess(request, batchedInputIds, mTokenizer.get(), io.ropeCosSin, stream))
    {
        throw std::runtime_error("GR00T vision preprocessing failed");
    }
    if (!mViT->infer(stream))
    {
        throw std::runtime_error("GR00T vision encoder execution failed");
    }
    io.inputIds = batchedInputIds.at(0);
    if (static_cast<int32_t>(io.inputIds.size()) > mMaxSequenceLength)
    {
        throw std::runtime_error("GR00T VL context is longer than the engines were built for");
    }
    buildEmbeddings(io, stream);
    return io;
}

void Gr00tVisualRunner::buildEmbeddings(Gr00tVlInputs& io, cudaStream_t stream)
{
    int32_t const seqLen = static_cast<int32_t>(io.inputIds.size());
    int64_t const vocab = mEmbeddingTable.getShape()[0];
    bool ok = mInputIdsHost.reshape({1, seqLen}) && mInputIds.reshape({1, seqLen})
        && mMultimodalIndices.reshape({1, seqLen}) && io.inputsEmbeds.reshape({1, seqLen, mHiddenSize})
        && io.positionIds.reshape({1, seqLen});
    for (auto& feature : io.deepstackEmbeds)
    {
        ok &= feature.reshape({1, seqLen, mHiddenSize});
    }
    if (!ok)
    {
        throw std::runtime_error("GR00T VL context is longer than the buffers were sized for");
    }

    io.textMask.assign(static_cast<size_t>(seqLen), 1);
    io.imageMask.assign(static_cast<size_t>(seqLen), 0);
    auto* ids = mInputIdsHost.dataPointer<int32_t>();
    for (int32_t token = 0; token < seqLen; ++token)
    {
        int32_t const id = io.inputIds[token];
        if (id < 0 || id >= vocab)
        {
            throw std::runtime_error("GR00T token id " + std::to_string(id) + " is outside the embedding table");
        }
        ids[token] = id;
        if (id == mImageTokenId)
        {
            io.textMask[token] = 0;
            io.imageMask[token] = 1;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mInputIdsHost.rawPointer(),
        static_cast<size_t>(seqLen) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::generateMultimodalIndices(mInputIds, mMultimodalIndices, mImageTokenId, std::nullopt, stream);
    kernel::embeddingLookup(mInputIds, mEmbeddingTable, std::nullopt, io.inputsEmbeds, stream, mMultimodalIndices,
        mImageTokenId, mViT->getOutputEmbedding());

    auto const deepstack = mViT->getDeepstackFeatures();
    if (deepstack.size() != io.deepstackEmbeds.size())
    {
        throw std::runtime_error("GR00T DeepStack tensor count changed between observations");
    }
    for (size_t feature = 0; feature < deepstack.size(); ++feature)
    {
        kernel::assembleDeepstackEmbedding(mInputIds, deepstack[feature].get(), io.deepstackEmbeds[feature], stream,
            mImageTokenId, mMultimodalIndices);
    }
}

} // namespace gr00t
} // namespace trt_edgellm
