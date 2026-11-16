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
#include "multimodal/common/multimodalRunner.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! \brief One camera frame, borrowed for the call: tightly packed [height, width, 3] 8-bit RGB.
struct Gr00tFrame
{
    unsigned char const* rgb{nullptr};
    int64_t height{0};
    int64_t width{0};
};

//! One output pixel's source interval, normalized so the weights sum to one.
struct AreaTap
{
    int64_t first{0};
    std::vector<float> weights;
};

//! One axis's taps with the sizes they were built for, so a steady stream of
//! same-sized frames never recomputes them.
struct AxisTaps
{
    int64_t inputSize{0};
    int64_t outputSize{0};
    std::vector<AreaTap> taps;
};

//! Scratch for smallestMaxSize, kept across frames so a policy step allocates nothing.
struct ResizeScratch
{
    AxisTaps horizontal;
    AxisTaps vertical;
    std::vector<float> rows;
    std::vector<unsigned char> out;
};

//! \brief albumentations' SmallestMaxSize with cv2's INTER_AREA resampling, in
//! place: scale so the shorter side becomes \p maxSize.
void smallestMaxSize(std::vector<unsigned char>& pixels, int64_t& height, int64_t& width, int64_t channels,
    int64_t maxSize, ResizeScratch& scratch);

//! Allocating overload for tests and tools.
void smallestMaxSize(
    std::vector<unsigned char>& pixels, int64_t& height, int64_t& width, int64_t channels, int64_t maxSize);

//! \brief One observation's vision-language inputs, ready for the backbone.
struct Gr00tVlInputs
{
    std::vector<int32_t> inputIds;
    rt::Tensor inputsEmbeds; //!< [1, S, hidden] fp16
    rt::Tensor ropeCosSin;   //!< [1, S, headDim] fp32, packed (cos|sin)
    rt::Tensor positionIds;  //!< [1, S] int32
    std::vector<rt::Tensor> deepstackEmbeds;
    std::vector<uint8_t> textMask;  //!< [S], 1 where a cross block may attend as text
    std::vector<uint8_t> imageMask; //!< [S], 1 where it may attend as image
};

//! \brief GR00T's vision-language front end: the eval transform, the shared
//! Qwen3-VL ViT runner, and the embedding lookup that turns token ids into
//! ``inputs_embeds`` with image and DeepStack rows at their placeholder positions.
class Gr00tVisualRunner
{
public:
    //! \param engineDir Bundle root; the tower lives in ``visual/``, the
    //!        embedding table and tokenizer beside it.
    Gr00tVisualRunner(std::string const& engineDir, int32_t maxSequenceLength, int32_t headDim, int32_t shortestEdge,
        float cropFraction, cudaStream_t stream);

    //! \brief Encode one observation into the backbone's inputs.
    //! \param frames One frame per camera, in the contract's view order.
    //! \param instruction Task text, already normalized by the policy.
    Gr00tVlInputs const& encode(
        std::vector<Gr00tFrame> const& frames, std::string const& instruction, cudaStream_t stream);

private:
    //! Gather token embeddings, then overwrite the placeholder rows with the tower's output.
    void buildEmbeddings(Gr00tVlInputs& io, cudaStream_t stream);

    //! GR00T's eval transform into \p view: shortest-edge resize, fractional centre
    //! crop, resize again; the tower's own resize is then skipped.
    void applyEvalTransform(Gr00tFrame const& frame, rt::imageUtils::ImageData& view);

    std::unique_ptr<rt::MultimodalRunner> mViT;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;
    chat_template::ChatTemplate mChatTemplate;
    rt::Tensor mEmbeddingTable; //!< [vocab, hidden] fp16, device
    rt::Tensor mContextMemory;
    Gr00tVlInputs mInputs;
    //! Pinned, one per camera; grown only when a larger frame than any before arrives.
    std::vector<rt::imageUtils::ImageData> mViews;
    std::vector<unsigned char> mPixels;  //!< eval-transform working copy of the frame
    std::vector<unsigned char> mCropped; //!< centre-crop destination, swapped with mPixels
    ResizeScratch mResize;
    rt::Tensor mPositionHost;
    rt::Tensor mInputIds;          //!< [1, S] int32, device
    rt::Tensor mInputIdsHost;      //!< [1, S] int32, pinned
    rt::Tensor mMultimodalIndices; //!< [1, S] int32, the image row each placeholder takes
    int32_t mHiddenSize{0};
    int32_t mHeadDim{0};
    int32_t mMaxSequenceLength{0};
    int32_t mImageTokenId{0};
    int32_t mShortestEdge{0};
    float mCropFraction{1.0F};
};

} // namespace gr00t
} // namespace trt_edgellm
