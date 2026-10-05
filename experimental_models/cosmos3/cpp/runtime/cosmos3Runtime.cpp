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

#include "runtime/cosmos3Runtime.h"
#include "common/cosmos3Bindings.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "profiling/nvtx_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace cosmos3
{

Cosmos3Runtime::Cosmos3Runtime(std::string const& engineDir, cudaStream_t stream)
{
    // Load the UND prefill engine (prefill-only tower emitting per-layer K/V directly).
    std::string const textEngine = engineDir + "/und_prefill/und_prefill.engine";
    mTextRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mTextRuntime, "Failed to create TensorRT runtime");
    mTextEngine = deserializeCudaEngineFromFile(*mTextRuntime, textEngine);
    mTextContext = std::unique_ptr<IExecutionContext>(
        mTextEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mTextContext, "Failed to create und_prefill execution context");
    ELLM_CHECK(mTextContext->setOptimizationProfileAsync(0, stream), "Failed to set und_prefill optimization profile");

    // Read the und_prefill component contract; every key is required (no fallback defaults).
    {
        std::string const configPath = engineDir + "/und_prefill/config.json";
        std::ifstream f(configPath);
        ELLM_CHECK(f.is_open(), "Failed to open und_prefill config: " + configPath);
        Json const j = Json::parse(f);
        auto requireKey = [&](char const* key) -> Json const& {
            ELLM_CHECK(j.contains(key), "und_prefill config missing required key: " + std::string(key));
            return j.at(key);
        };
        mNumLayers = requireKey("num_hidden_layers").get<int32_t>();
        mNumKVHeads = requireKey("num_key_value_heads").get<int32_t>();
        mHeadDim = requireKey("head_dim").get<int32_t>();
        mHiddenSize = requireKey("hidden_size").get<int32_t>();
        mRopeTheta = requireKey("rope_theta").get<float>();
    }

    mVaeRunner = std::make_unique<Cosmos3VaeEncoderRunner>(engineDir + "/vae_encoder", stream);
    mPolicyRunner = std::make_unique<Cosmos3PolicyRunner>(engineDir + "/gen", stream);

    allocateSharedContextMemory();
    allocateUndBuffers(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Cosmos3Runtime::allocateSharedContextMemory()
{
    // All three engines are created with ExecutionContextAllocationStrategy::kUSER_MANAGED and
    // bind the same shared pool through the TensorRT V2 user-managed-memory API: the UND prefill
    // context directly below, and the VAE/policy runners inside their setContextMemory() (which
    // call setDeviceMemoryV2 on their own contexts). No V1 context-memory API is used anywhere.
    int64_t poolBytes = mTextEngine->getDeviceMemorySizeV2();
    poolBytes = std::max(poolBytes, mVaeRunner->getRequiredContextMemorySize());
    poolBytes = std::max(poolBytes, mPolicyRunner->getRequiredContextMemorySize());

    mSharedContextMemory = rt::Tensor(
        std::vector<int64_t>{poolBytes}, rt::DeviceType::kGPU, DataType::kINT8, "cosmos3::sharedContextMemory");
    mTextContext->setDeviceMemoryV2(mSharedContextMemory.rawPointer(), mSharedContextMemory.getMemoryCapacity());
    mVaeRunner->setContextMemory(mSharedContextMemory);
    mPolicyRunner->setContextMemory(mSharedContextMemory);
}

void Cosmos3Runtime::allocateUndBuffers(cudaStream_t stream)
{
    // Everything below is allocated ONCE at the engine-profile maximum batch and prompt length;
    // per-request work is limited to shape updates (Tensor::reshape + setInputShape), a rope/pos-id
    // refill when the packed request shape changes, and the enqueue.
    Dims const maxEmbeds
        = mTextEngine->getProfileShape(trt_edgellm::binding_names::kInputsEmbeds, 0, OptProfileSelector::kMAX);
    mMaxUndBatch = static_cast<int32_t>(maxEmbeds.d[0]);
    mMaxUndLen = static_cast<int32_t>(maxEmbeds.d[1]);
    ELLM_CHECK(mMaxUndBatch > 0 && mMaxUndLen > 0, "und_prefill engine reports a non-positive profile maximum");
    int32_t const batch = mMaxUndBatch;

    // Only the conditional K/V set is allocated here. The second (unconditional, empty-prompt) K/V set
    // is allocated lazily by setGuidance -> allocateUncondBuffers, so the default guidance == 1 path does
    // not double UND K/V memory with an always-on buffer it never uses.
    mUndK.reserve(static_cast<size_t>(mNumLayers));
    mUndV.reserve(static_cast<size_t>(mNumLayers));
    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        mUndK.emplace_back(rt::Coords{batch, mMaxUndLen, mNumKVHeads, mHeadDim}, rt::DeviceType::kGPU, DataType::kHALF,
            "cosmos3::undK");
        mUndV.emplace_back(rt::Coords{batch, mMaxUndLen, mNumKVHeads, mHeadDim}, rt::DeviceType::kGPU, DataType::kHALF,
            "cosmos3::undV");
    }
    mUndHidden
        = rt::Tensor({batch, mMaxUndLen, mHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF, "cosmos3::undHidden");

    // Packed rope cos/sin [B,S,headDim] = [cos(D/2)|sin(D/2)] for text positions 0..S-1 and the iota
    // attention_pos_id [B,S]. The engine reads them PACKED at the request shape, so the buffers are
    // (re)filled in prefillUnd whenever the request (B,S) changes; capacity is allocated here once.
    mTextRopeCosSin
        = rt::Tensor({batch, mMaxUndLen, mHeadDim}, rt::DeviceType::kGPU, DataType::kFLOAT, "cosmos3::textRopeCosSin");
    mAttentionPosId = rt::Tensor({batch, mMaxUndLen}, rt::DeviceType::kGPU, DataType::kINT32, "cosmos3::attnPosId");
    mAttentionPosIdHost
        = rt::Tensor({batch, mMaxUndLen}, rt::DeviceType::kCPU, DataType::kINT32, "cosmos3::attnPosIdHost");

    // Bind the shared constant addresses once; inputs_embeds and the und_k/und_v OUTPUT addresses (which
    // target set to fill) are (re)bound per prefill call in prefillUnd.
    bool ok = true;
    ok &= mTextContext->setTensorAddress(trt_edgellm::binding_names::kRopeCosSin, mTextRopeCosSin.rawPointer());
    ok &= mTextContext->setTensorAddress(trt_edgellm::binding_names::kAttentionPosId, mAttentionPosId.rawPointer());
    ok &= mTextContext->setTensorAddress("hidden_states", mUndHidden.rawPointer());
    ELLM_CHECK(ok, "Cosmos3Runtime failed to bind und_prefill tensor addresses");
}

void Cosmos3Runtime::setGuidance(float guidance, float intervalLo, float intervalHi)
{
    mGuidance = guidance;
    // Allocate the second UND K/V set only when CFG is actually enabled; the default guidance == 1 path
    // must not carry a redundant always-on buffer (edge-target memory budget).
    if (guidance != 1.0F)
    {
        allocateUncondBuffers();
    }
    mPolicyRunner->setGuidance(guidance, intervalLo, intervalHi);
}

void Cosmos3Runtime::allocateUncondBuffers()
{
    // Guard against re-allocation if setGuidance is called more than once with CFG enabled.
    if (!mUndKUncond.empty())
    {
        return;
    }
    // Second K/V set for the CFG unconditional (empty-prompt) pass; written by a separate prefillUnd.
    // Same profile-maximum extent and layout as the conditional set (mUndK/mUndV).
    int32_t const batch = mMaxUndBatch;
    mUndKUncond.reserve(static_cast<size_t>(mNumLayers));
    mUndVUncond.reserve(static_cast<size_t>(mNumLayers));
    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        mUndKUncond.emplace_back(rt::Coords{batch, mMaxUndLen, mNumKVHeads, mHeadDim}, rt::DeviceType::kGPU,
            DataType::kHALF, "cosmos3::undKUncond");
        mUndVUncond.emplace_back(rt::Coords{batch, mMaxUndLen, mNumKVHeads, mHeadDim}, rt::DeviceType::kGPU,
            DataType::kHALF, "cosmos3::undVUncond");
    }
}

void Cosmos3Runtime::prefillUnd(
    rt::Tensor const& inputsEmbeds, std::vector<rt::Tensor>& undK, std::vector<rt::Tensor>& undV, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(undRange, "cosmos3::und_prefill");

    rt::Coords const& shape = inputsEmbeds.getShape();
    ELLM_CHECK(shape.getNumDims() == 3, "inputs_embeds must be [B,S,hidden]");
    int32_t const batch = static_cast<int32_t>(shape[0]);
    int32_t const seqLen = static_cast<int32_t>(shape[1]);
    ELLM_CHECK(batch >= 1 && batch <= mMaxUndBatch,
        "request batch " + std::to_string(batch) + " exceeds the und_prefill engine profile maximum "
            + std::to_string(mMaxUndBatch) + " (build with a larger --max-batch-size)");
    ELLM_CHECK(seqLen > 0 && seqLen <= mMaxUndLen, "prompt length exceeds the und_prefill engine profile maximum");
    ELLM_CHECK(shape[2] == mHiddenSize, "inputs_embeds hidden size does not match the contract");
    ELLM_CHECK(inputsEmbeds.getDataType() == DataType::kHALF, "inputs_embeds must be FLOAT16");

    // The rope cos/sin cache and iota position ids are read PACKED at [B,S,...]; refill them when the
    // request shape changes (identical text-only positions 0..S-1 for every batch element).
    if (batch != mRopeFilledBatch || seqLen != mRopeFilledLen)
    {
        kernel::initializeTextOnlyMRopeCosSin(static_cast<float*>(mTextRopeCosSin.rawPointer()), mRopeTheta,
            static_cast<int64_t>(mHeadDim), seqLen, batch, stream);
        auto* posHost = static_cast<int32_t*>(mAttentionPosIdHost.rawPointer());
        for (int32_t b = 0; b < batch; ++b)
        {
            std::iota(posHost + static_cast<size_t>(b) * seqLen, posHost + (static_cast<size_t>(b) + 1) * seqLen, 0);
        }
        // Member pinned buffer: upload is stream-ordered before enqueueV3 and not reused until the
        // next shape change, so no sync is needed here.
        CUDA_CHECK(cudaMemcpyAsync(mAttentionPosId.rawPointer(), posHost,
            static_cast<size_t>(batch) * seqLen * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        mRopeFilledBatch = batch;
        mRopeFilledLen = seqLen;
    }

    // Metadata-only reshape of the target K/V so downstream consumers see [B,S,H,D], and point the
    // engine's und_k/und_v OUTPUT bindings at this call's target set (conditional vs unconditional).
    bool ok = true;
    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        ELLM_CHECK(undK[static_cast<size_t>(i)].reshape(rt::Coords{batch, seqLen, mNumKVHeads, mHeadDim}),
            "und_k reshape failed");
        ELLM_CHECK(undV[static_cast<size_t>(i)].reshape(rt::Coords{batch, seqLen, mNumKVHeads, mHeadDim}),
            "und_v reshape failed");
        ok &= mTextContext->setTensorAddress(
            cosmos3::binding_names::formatUndKName(i).c_str(), undK[static_cast<size_t>(i)].rawPointer());
        ok &= mTextContext->setTensorAddress(
            cosmos3::binding_names::formatUndVName(i).c_str(), undV[static_cast<size_t>(i)].rawPointer());
    }

    ok &= mTextContext->setInputShape(trt_edgellm::binding_names::kInputsEmbeds, Dims{3, {batch, seqLen, mHiddenSize}});
    ok &= mTextContext->setInputShape(trt_edgellm::binding_names::kRopeCosSin, Dims{3, {batch, seqLen, mHeadDim}});
    ok &= mTextContext->setInputShape(trt_edgellm::binding_names::kAttentionPosId, Dims{2, {batch, seqLen}});
    ok &= mTextContext->setInputTensorAddress(trt_edgellm::binding_names::kInputsEmbeds, inputsEmbeds.rawPointer());
    ELLM_CHECK(ok, "Cosmos3Runtime::prefillUnd failed to set und_prefill input shapes/address");

    // One dynamic-shape enqueue per request: CUDA-graph capture is intentionally not used here — the
    // shape changes with every prompt (invalidating a capture), unlike the GEN denoise loop, which
    // replays a fixed-shape step many times and does use one (Cosmos3PolicyRunner::setUseCudaGraph).
    if (!mTextContext->enqueueV3(stream))
    {
        throw std::runtime_error("Cosmos3Runtime::prefillUnd und_prefill engine enqueueV3 failed");
    }
    // No sync: the GEN denoise runs on the same stream, so stream ordering already serializes these
    // UND K/V writes before generatePolicy's GEN pass consumes mUndK/mUndV.
}

std::vector<float> Cosmos3Runtime::generatePolicy(rt::Tensor const& pixelValues, rt::Tensor const& inputsEmbeds,
    rt::Tensor const* uncondEmbeds, std::vector<float> const& currentState, cudaStream_t stream)
{
    rt::Tensor const& condLatent = mVaeRunner->encode(pixelValues, stream);
    prefillUnd(inputsEmbeds, mUndK, mUndV, stream);

    // Guidance-interval CFG: prefill the empty-prompt (unconditional) context into the second K/V set
    // and hand both to the GEN loop. Without CFG the unconditional vectors are empty and generate()
    // takes its single-forward path.
    if (mGuidance != 1.0F && uncondEmbeds != nullptr)
    {
        prefillUnd(*uncondEmbeds, mUndKUncond, mUndVUncond, stream);
        return mPolicyRunner->generate(condLatent, mUndK, mUndV, mUndKUncond, mUndVUncond, currentState, stream);
    }
    if (mGuidance != 1.0F)
    {
        // Guidance was requested but no unconditional context was supplied, so CFG cannot run and the
        // single conditional forward below is guidance=1 behavior. Warn once so a misconfigured caller
        // (setGuidance(!=1) but uncondEmbeds == nullptr) is not silently served degraded actions.
        static bool warned = false;
        if (!warned)
        {
            LOG_WARNING(
                "Guidance %.2f requested but uncondEmbeds is null; running the single conditional forward "
                "(guidance=1 behavior). Provide unconditional embeddings to enable guidance-interval CFG.",
                mGuidance);
            warned = true;
        }
    }
    // Non-CFG path: bind empty uncond K/V sets to the const& parameters (no persistent buffer needed).
    return mPolicyRunner->generate(condLatent, mUndK, mUndV, {}, {}, currentState, stream);
}

} // namespace cosmos3
} // namespace trt_edgellm
