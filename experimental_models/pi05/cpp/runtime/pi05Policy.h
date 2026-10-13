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

#include "runtime/pi05Runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
}

namespace pi05
{

//! \brief Per-key normalization statistics from openpi's ``norm_stats.json``. Only the
//! quantiles are read: openpi derives ``use_quantile_norm`` from the model type, so every
//! pi0.5 configuration normalizes as
//! ``x_norm = 2 * (x - q01) / (q99 - q01 + kQuantileEpsilon) - 1``.
struct NormStats
{
    std::vector<float> q01;
    std::vector<float> q99;
};

//! openpi's divisor floor, applied to every quantile map whatever the spread.
constexpr float kQuantileEpsilon = 1e-6F;

//! Image slots the architecture carries: one base view and two wrist views.
constexpr int32_t kMaxCameraSlots = 3;

//! openpi's PaligemmaTokenizer bins the state over [-1, 1) with exactly this many
//! buckets; any other count yields a prompt no pi0.5 checkpoint was trained on.
constexpr int32_t kStateBins = 256;

//! \brief Which embodiment's observation and action math a bundle asks for. Every
//! step that differs between openpi's policy configurations keys off this.
enum class Pi05Adapter
{
    kLibero,
    kDroid,
    kAloha
};

//! \brief One image slot, as the prefix consumes it.
struct Pi05CameraSlot
{
    std::string name;
    //! An optional slot a request omits is left out of the prefix rather than sent as a
    //! masked black frame, so omitting it shifts every language position after it.
    bool required{true};
};

//! \brief Everything the exported ``policy.json`` declares about one embodiment.
//!
//! The camera list is ordered: the prefix graph carries no attention mask, so the
//! views must reach it in exactly this order or every language token's position
//! shifts.
struct Pi05Contract
{
    //! Provenance. \c policyConfig names the openpi config the checkpoint was trained
    //! under; \c checkpointFingerprint identifies the weights the engines came from.
    std::string policyConfig;
    std::string checkpointFingerprint;
    //! The export run that wrote this contract. The engines staged beside it carry the
    //! same stamp, and nothing else ties the two together.
    std::string exportId;

    Pi05Adapter adapter{Pi05Adapter::kLibero};

    std::vector<Pi05CameraSlot> cameras;
    //! Names the configuration accepts and drops, matching its openpi input transform.
    //! A name in neither list is rejected, since two swapped views produce plausible
    //! actions rather than an error.
    std::vector<std::string> ignoredCameras;

    int32_t stateDim{0};
    int32_t actionHorizon{0};  //!< steps per chunk; checked against the action engine
    int32_t modelActionDim{0}; //!< padded width of the action engine's output
    int32_t robotActionDim{0}; //!< the embodiment's own width, a prefix of the model's

    int32_t imageHeight{0};
    int32_t imageWidth{0};
    int32_t maxTokenLen{0};
    int32_t tokenizerVocabSize{0};
    int32_t numBins{256};
    //! Whether the discretized state goes into the prompt. openpi's pi05_libero says no,
    //! and pi0.5 has no state projection, so there the state reaches the model through
    //! neither path.
    bool discreteStateInput{false};
};

//! \brief One camera of an observation.
struct Pi05CameraView
{
    //! Contract camera name. Leave empty to place the view by position instead;
    //! naming it is checked against the contract, since two swapped views produce
    //! plausible actions rather than an error.
    std::string name;
    //! Exactly one source per view. A robot adapter already holding the frame sets
    //! \p rgb and skips the decode, which is most of the host cost of a request.
    std::string imagePath;
    //! Tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

//! \brief What one policy call sees, in the robot's own units.
struct Pi05Observation
{
    //! One entry per slot the request fills. Required slots must all be here; an
    //! optional one may be left out, and a name the contract ignores is dropped.
    std::vector<Pi05CameraView> cameras;
    //! Robot units, the embodiment's own width. The adapter converts it.
    std::vector<float> state;
    std::string task;
    //! Replicate the request across the batch axis; needs engines built for it.
    int32_t batch{1};
};

//! \brief Host cost of one call's observation work, summed over the views, in ms.
//! Decode is charged apart because it exists only for a client that hands over
//! encoded files; one holding decoded frames pays the other two.
struct Pi05ObservationTimes
{
    double decodeMs{0.0}; //!< PNG/JPG decode, and the decoded frame's pinned host buffer
    double resizeMs{0.0}; //!< resize-with-pad, normalize, fp16 staging convert
    double restMs{0.0};   //!< state normalize and discretize, prompt assembly, tokenize
    //! Wall clock over all of it. The three above are disjoint sub-intervals of it, so
    //! this is their sum plus the staging and logging no bucket claims.
    double totalMs{0.0};
};

//! \brief Where one policy call spent its time, in milliseconds. One member per clock:
//! stream events, host observation, host wall clock over the chain.
struct Pi05Timings
{
    Pi05StageTimes stages;            //!< per-stage clocks; cond is counted inside action
    Pi05ObservationTimes observation; //!< the host work the engine stages do not cover
    //! The engine chain alone -- generate() only. Timed on the host, not from the stage
    //! events, so it also carries the host gaps between enqueues and is not the sum of
    //! the four stages. Not comparable to a harness that times a whole policy call.
    double engineMs{0.0};
    //! The whole policy call: infer() entry through unnormalization, so it covers the
    //! observation front end and the unnormalize that engineMs excludes. From
    //! inferTensors() it starts at the canonical tensors, there being no front end.
    double policyMs{0.0};
};

//! \brief One policy call's output, with the request record that produced it.
struct Pi05ActionChunk
{
    //! Row-major [batch, horizon, modelActionDim], normalized and zero-padded as
    //! the action engine emits it.
    std::vector<float> normalizedActions;
    //! Row-major [horizon, robotActionDim], batch entry 0 only, and empty from
    //! inferTensors(): the ALOHA conversion reads the request's own state, which canonical
    //! tensors do not carry. Compare batch entries in normalizedActions above.
    std::vector<float> robotActions;
    //! What the towers were conditioned on, so a chunk can be reproduced without
    //! re-deriving the prompt.
    std::string prompt;
    std::vector<int32_t> tokenIds;
    int32_t batch{1};
    int32_t horizon{0};
    int32_t modelActionDim{0};
    Pi05Timings timings;
};

//! \brief The pi0.5 model policy: an observation in, robot commands out.
//!
//! Owns everything the checkpoint's numerical contract implies; nothing here knows a
//! simulator, a robot SDK or a transport. Normalization is quantile only, over the q01/q99
//! statistics staged under ``assets/``; any other set a checkpoint carries is ignored.
//!
//! NOT thread-safe: one instance, one stream, and infer() reuses staging buffers.
class Pi05Policy
{
public:
    //! \brief Load the contract alone, without the engines; infer() then throws.
    //! For tools that inspect an export -- listing its camera order, checking that a
    //! bundle is one this runtime supports.
    //! \param engineDir Engine root holding ``policy.json`` and ``assets/``.
    //! \throws std::runtime_error If either is missing, declares a configuration this
    //!         runtime does not implement, or lacks the statistics the declared modes require.
    explicit Pi05Policy(std::string const& engineDir);

    //! \brief Load the contract and the engines beside it, on \p stream.
    Pi05Policy(std::string const& engineDir, cudaStream_t stream);

    //! Out of line: the tokenizer and the runtime are only forward-declared here.
    ~Pi05Policy();

    //! \brief True if \p engineDir carries a contract, i.e. a policy can be built from it.
    static bool available(std::string const& engineDir);

    //! \brief Run one observation through the whole chain: camera slots, image
    //! preprocessing, prompt, tokenizer, engines, unnormalization and the slice to
    //! the robot's width.
    //! \throws std::runtime_error If this policy was built without engines.
    //! \throws std::invalid_argument If the observation does not fit the contract.
    Pi05ActionChunk infer(Pi05Observation const& observation);

    //! \brief Run already-canonical tensors, skipping the observation adapters.
    //! For accuracy comparison and profiling: the chunk carries the normalized actions
    //! alone, with no prompt and no robot-unit conversion. \p pixelValues
    //! may also carry the declared empty slots, which is the shape a full-prefix reference
    //! dumps -- but this runtime masks nothing, so those views attend and shift the language
    //! positions after them. Use that shape for shape-level benchmarks, not for parity.
    //! \param pixelValues Device FLOAT16 [views, 3, imageSize, imageSize] in [-1, 1].
    //! \throws std::invalid_argument If \p pixelValues is not that tensor, or \p batch is below 1.
    Pi05ActionChunk inferTensors(rt::Tensor const& pixelValues, std::vector<int32_t> const& tokenIds, int32_t batch);

    //! \brief The engines behind infer(). Exposed for the knobs a caller sets once per
    //! process -- seed, denoise steps, graph capture -- and for canonical-tensor runs
    //! that deliberately bypass the observation adapters.
    Pi05Runtime& runtime();

    Pi05Contract const& contract() const noexcept
    {
        return mContract;
    }

    //! \brief Image slots in the order the prefix expects them.
    std::vector<Pi05CameraSlot> const& cameras() const noexcept
    {
        return mContract.cameras;
    }

    //! The adapter API above is infer(), inferTensors(), runtime() and contract(). What
    //! follows are the contract steps, public because each is checked directly against an
    //! openpi golden: a wrong prompt, tokenization or unnormalization produces a
    //! plausible robot command rather than an error, so they are tested in isolation and
    //! not only through infer(). Callers should not need them.

    //! \brief Place the supplied views into the contract's slots, in prefix order, keeping
    //! the active ones alone. Unnamed views fill the leading slots by position instead.
    //! \throws std::invalid_argument On an unknown or repeated name, a partly named
    //!         request, or a required slot no view fills.
    std::vector<Pi05CameraView const*> resolveActiveViews(std::vector<Pi05CameraView> const& views) const;

    //! \brief Reject a view count the contract cannot explain: a pixel tensor either
    //! carries as many views as a request can activate, which is the policy path, or
    //! every image slot, which only a shape-level benchmark should use.
    //! \throws std::invalid_argument When \p views is neither.
    void validateViewCount(int32_t views) const;

    //! \brief Reject a pixel tensor the visual engine cannot be bound to: TensorRT is
    //! handed the contract's shape whatever the tensor holds, so a smaller frame or a
    //! host allocation is read out of bounds rather than rejected.
    //! \throws std::invalid_argument When it is not device FLOAT16 [views, 3, H, W].
    void validatePixelValues(rt::Tensor const& pixelValues) const;

    //! \brief Convert a robot-unit state into the space the checkpoint was trained in.
    //! Identity except under ALOHA, which flips joint signs and un-linearizes both grippers.
    //! \throws std::invalid_argument When \p state is not the contract's width.
    std::vector<float> adaptInputState(std::vector<float> const& state) const;

    //! \brief Assemble the pi0.5 prompt from a task string and an adapted state.
    //!
    //! Only a ``discrete_state_input`` contract puts the state in: normalized,
    //! discretized into ``num_bins`` bins over [-1, 1) and written in as decimal
    //! text. pi05_libero does not, so there the state is validated and dropped.
    std::string buildPrompt(std::string const& task, std::vector<float> const& adapted) const;

    //! \brief Encode \p prompt with the PaliGemma tokenizer staged beside the engines,
    //! appending the separately encoded start-of-answer newline unless the contract
    //! discretizes the state into \p prompt. The reference pads to the contract length and
    //! masks the padding out; the prefix graph has no mask, so the result is truncated.
    std::vector<int32_t> tokenize(std::string const& prompt);

    //! \brief Map a normalized chunk to robot units and drop the padded dims.
    //! \param normalized Row-major ``[horizon, actionDim]`` from the action engine.
    //! \param actionDim Padded model action dimension (32 for pi0.5).
    //! \return Row-major ``[horizon, robotActionDim()]``.
    std::vector<float> unnormalizeActions(
        std::vector<float> const& normalized, int32_t horizon, int32_t actionDim) const;

    //! \brief Unnormalize, then the embodiment's own conversion. ALOHA adds openpi's
    //! absolute-action step and the inverse input conversion; the others stop at the slice.
    //! \param adapted The same state adaptInputState() produced for this request.
    std::vector<float> postprocessActions(std::vector<float> const& normalized, int32_t horizon, int32_t actionDim,
        std::vector<float> const& adapted) const;

    int32_t robotActionDim() const noexcept
    {
        return mContract.robotActionDim;
    }

    //! \brief Slots the contract names, which is the most views a request can activate.
    int32_t numCameras() const noexcept
    {
        return static_cast<int32_t>(mContract.cameras.size());
    }

private:
    void loadContract(std::string const& engineDir);
    //! Decode and preprocess the ordered views into the bound pixel tensor.
    //! \param times Accumulates the decode and preprocess halves across the views.
    void stagePixelValues(std::vector<Pi05CameraView const*> const& ordered, Pi05ObservationTimes& times);
    //! Resize-with-pad \p rgb into view slot \p viewIdx of the pinned staging. Returns its ms.
    double stageOneView(unsigned char const* rgb, int32_t srcH, int32_t srcW, size_t viewIdx);

    std::string mEngineDir;
    Pi05Contract mContract;
    NormStats mStateStats;
    NormStats mActionStats;

    std::unique_ptr<Pi05Runtime> mRuntime;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;
    cudaStream_t mStream{nullptr};
    //! Sized once for every slot the contract declares; a request using fewer reshapes it down.
    rt::Tensor mPixelValues;
    rt::Tensor mPixelValuesHost;    //!< pinned staging for mPixelValues
    std::vector<float> mPlanarView; //!< one view's resized CHW float buffer, reused per request
};

//! \brief The contract's camera order on one line, for help text and diagnostics.
std::string cameraOrderSummary(std::vector<Pi05CameraSlot> const& slots);

//! \brief Preprocess one decoded RGB8 frame the way PaliGemma expects.
//!
//! Aspect-preserving resize to \p height x \p width with black padding, then
//! [0, 1] -> [-1, 1]. Must track ``openpi.transforms.ResizeImages``, which delegates to
//! ``openpi_client.image_tools.resize_with_pad``: Pillow's triangle filter, 8 bits between passes.
//! \param pixels Interleaved RGB8, \p srcH * \p srcW * 3 bytes.
//! \param out Planar CHW, \p height * \p width * 3 floats.
void resizeWithPad(unsigned char const* pixels, int32_t srcH, int32_t srcW, int32_t height, int32_t width, float* out);

} // namespace pi05
} // namespace trt_edgellm
