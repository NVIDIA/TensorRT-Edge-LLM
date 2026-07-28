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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tensor.h"
#include "runtime/llmRuntimeUtils.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
} // namespace tokenizer

namespace rt
{

//! Video tensor in Cosmos layout: [batch, channels, numFrames, height, width].
struct VideoBuffer
{
    std::shared_ptr<rt::Tensor> buffer;
    int64_t batch{1};
    int64_t channels{3};
    int64_t numFrames{0};
    int64_t height{0};
    int64_t width{0};
};

//! Audio waveform in Cosmos AVAE layout: [batch, 1, numSamples] or [batch, numSamples].
struct AudioBuffer
{
    std::shared_ptr<rt::Tensor> buffer;
    int64_t batch{1};
    int32_t sampleRate{48000};
    int64_t numSamples{0};
};

//! Export-time shapes and index tables (from Python ``build_cosmos_packed_static``).
struct CosmosPackedStatic
{
    int32_t undLen{0};
    int32_t sequenceLength{0};
    int32_t numNoisyTokens{0};
    std::vector<int32_t> inputIds;
    std::vector<int32_t> textIndexes;
    std::vector<int32_t> positionIds; //!< Flattened [3, sequenceLength] row-major
    std::vector<int32_t> visionSequenceIndexes;
    std::vector<int32_t> visionMseLossIndexes;
    std::vector<int32_t> visionNoisyFrameIndexes;
    std::vector<int32_t> visionTokenShape; //!< [latent_t, patch_h, patch_w]

    std::vector<int32_t> soundSequenceIndexes;
    std::vector<int32_t> soundMseLossIndexes;
    std::vector<int32_t> soundNoisySlotIndexes;
    std::vector<int32_t> soundTokenShape; //!< [sound_dim, latent_frames] or [latent_frames]
};

//! Loaded from ``config.json`` next to the exported Cosmos Edge engines.
struct CosmosEngineConfig
{
    int32_t numFrames{0};
    int32_t height{0};
    int32_t width{0};
    float fps{24.F};
    int32_t hiddenSize{0};
    int32_t numInferenceSteps{35};
    int32_t maxBatchSize{1};

    int32_t numTrainTimesteps{1000};
    float flowShift{5.F};
    int32_t schedulerSolverOrder{2};

    int32_t headDim{128};
    float ropeTheta{1000000.F};
    std::array<int32_t, 3> mropeSection{24, 20, 20};
    bool enableFpsModulation{false};
    int32_t unified3dMropeTemporalModalityMargin{15000};
    bool unified3dMropeResetSpatialIds{true};
    float baseFps{24.F};
    int32_t temporalCompressionFactor{4};
    std::string visionStartToken{"<|vision_start|>"};
    bool useChatTemplate{true};
    bool useSystemPrompt{true};
    std::string systemPromptVideo{"You are a helpful assistant who will generate videos from a give prompt."};

    bool enableSound{false};
    int32_t soundDim{64};
    int32_t sampleRate{48000};
    float soundLatentFps{25.F};
};

//! CPU/GPU tensors produced once per prompt before the denoise loop.
struct CosmosTextPhase0
{
    std::vector<int32_t> inputIds;
    int32_t undLen{0};
    int32_t genLen{0};
    int32_t sequenceLength{0};
    rt::Tensor undSeq;
    rt::Tensor cosUnd;
    rt::Tensor sinUnd;
    rt::Tensor cosGen;
    rt::Tensor sinGen;
};

struct WFMGenerationRequest
{
    std::string prompt;
    VideoBuffer pixels;
    AudioBuffer inputWaveform;    //!< Optional conditioning waveform (requires audio_encode engine).
    bool generateSound{false};    //!< When true, run sound denoise/decode path if engines are present.
    int32_t numInferenceSteps{0}; //!< 0 = use value from config.json
    int32_t seed{0};
};

struct WFMGenerationResponse
{
    VideoBuffer outputVideo;
    AudioBuffer outputWaveform;
};

//! Latent tensors updated in-place by the denoise loop (vision required, sound optional).
struct CosmosDenoiseLatents
{
    rt::Tensor* vision{nullptr};
    rt::Tensor* sound{nullptr};
};

//! Load export metadata written by the Python Cosmos Edge export script.
CosmosEngineConfig loadCosmosEngineConfig(std::filesystem::path const& configPath);

CosmosPackedStatic loadCosmosPackedStatic(std::filesystem::path const& packingPath);

//! Validate request against loaded export config (batch=1 for now).
bool examineWFMRequest(
    WFMGenerationRequest const& request, CosmosEngineConfig const& config, CosmosPackedStatic const& packed) noexcept;

//! Tokenize prompt and append Cosmos boundary tokens (eos + <|vision_start|>).
std::vector<int32_t> tokenizeCosmosPrompt(
    tokenizer::Tokenizer const& tokenizer, std::string const& prompt, CosmosEngineConfig const& config);

//! Build joint mRoPE position IDs [3, sequence_length] for text + vision tokens.
std::vector<int64_t> buildCosmosPositionIds(
    int32_t undLen, CosmosPackedStatic const& packed, CosmosEngineConfig const& config);

//! Lookup und_seq and build und/gen rotary tensors for the MoT backbone.
bool prepareCosmosTextPhase0(tokenizer::Tokenizer const& tokenizer, EmbeddingData const& embedding,
    CosmosEngineConfig const& config, CosmosPackedStatic const& packed, std::string const& prompt,
    CosmosTextPhase0& phase0, cudaStream_t stream);

//! Clone clean VAE latents and replace ``vision_noisy_frame_indexes`` slices with Gaussian noise.
bool seedNoisyVisionLatents(rt::Tensor const& cleanLatents, rt::Tensor& noisyLatents,
    std::vector<int32_t> const& noisyFrameIndexes, int32_t seed, float noiseScale, cudaStream_t stream);

//! Clone clean AVAE latents and replace ``sound_noisy_slot_indexes`` temporal slices with Gaussian noise.
//! When \p noisySlotIndexes is empty, all temporal slots are noised (full generation path).
bool seedNoisySoundLatents(rt::Tensor const& cleanLatents, rt::Tensor& noisyLatents,
    std::vector<int32_t> const& noisySlotIndexes, int32_t seed, float noiseScale, cudaStream_t stream);

//! CPU UniPC flow scheduler for vision latents (matches Cosmos3 Edge ``UniPCMultistepScheduler``).
class CosmosVisionScheduler
{
public:
    explicit CosmosVisionScheduler(CosmosEngineConfig const& config);

    void setTimesteps(int32_t numInferenceSteps);
    int32_t getNumInferenceSteps() const noexcept
    {
        return mNumInferenceSteps;
    }
    int32_t getStepIndex() const noexcept
    {
        return mStepIndex;
    }
    float getTimestepValue() const;
    bool stepLatents(rt::Tensor& latents, rt::Tensor const& modelOutput, cudaStream_t stream);

private:
    void resetState();
    std::vector<float> convertModelOutput(
        std::vector<float> const& sample, std::vector<float> const& modelOutput) const;
    std::vector<float> multistepUniPBhUpdate(
        std::vector<float> const& sample, std::vector<float> const& convertedOutput, int32_t order) const;
    std::vector<float> multistepUniCBhUpdate(std::vector<float> const& convertedOutput,
        std::vector<float> const& lastSample, std::vector<float> const& sample, int32_t order) const;
    static void sigmaToAlphaSigma(float sigma, float& alpha, float& sigmaOut) noexcept;

    CosmosEngineConfig mConfig{};
    int32_t mNumInferenceSteps{0};
    int32_t mStepIndex{0};
    int32_t mLowerOrderNums{0};
    std::vector<float> mTimesteps;
    std::vector<float> mSigmas;
    std::vector<std::vector<float>> mModelOutputs;
    std::vector<float> mTimestepList;
    std::vector<float> mLastSample;
    int32_t mThisOrder{1};
};

//! Alias: Omni uses independent scheduler copies per modality with the same UniPC config.
using CosmosFlowScheduler = CosmosVisionScheduler;

} // namespace rt
} // namespace trt_edgellm
