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

#include "kernels/compactPhase/compactPhaseSupport.h"
#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/tensor.h"

namespace trt_edgellm
{
namespace plugins
{

//! Internal context-attention implementation selected from build artifacts and runtime capabilities.
enum class ContextFMHABackend
{
    kNONE,
    kCUTE_DSL_FMHA_BLACKWELL,
    kCUTE_DSL_FMHA_V2
};

//! Compact input-slot indices of one plugin instance; defined with the plugin implementation.
struct AttentionInputLayout;

//! \brief TensorRT plugin for attention operations (V3 — IPluginV3).
//!
//! This plugin implements efficient attention mechanisms including context attention (prefill)
//! and decode attention with KV cache support.
class AttentionPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuildV2,
                        public nvinfer1::IPluginV3OneRuntime
{
public:
    //! \brief Constructor for attention plugin with configuration parameters
    //! \param[in] name Plugin instance name
    //! \param[in] numQHeads Number of query heads
    //! \param[in] numKVHeads Number of key-value heads
    //! \param[in] headSize Head dimension size
    //! \param[in] supportsSpecDecode Whether to support speculative decoding (Tree attention)
    //! \param[in] enableFp8KVCache Whether to enable FP8 KV cache
    //! \param[in] enableVisionBlockAttention Enable Gemma4 vision block attention
    //! \param[in] enableContextMaskSelector Whether to enable the optional context-mask selector input. When present,
    //! runtime shape [0] keeps the default causal/sliding context mask, while [batch] selects padding/non-causal
    //! context masking. The tensor value is ignored.
    //! \param[in] slidingWindowSize Sliding window size (-1 = no sliding window)
    //! \param[in] qkvScales Optional [q, k, v] FP8 dequant scales (required when enableFp8KVCache)
    //! \param[in] attentionScale Optional absolute QK^T multiplier; defaults to 1/sqrt(headSize)
    //! \param[in] externalKVPrefix Append a read-only prefix pool/table; main KV input/output is FP16 action scratch
    //! \param[in] fp8KVPrefix Prefix is E4M3 FP8; qkvScales K/V entries apply only to prefix loads
    //! \param[in] supportsBoundedKVCache Whether the engine supports runtime-selectable bounded O(W) KV storage
    AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
        int32_t supportsSpecDecode, int32_t enableFp8KVCache, int32_t enableVisionBlockAttention,
        int32_t enableContextMaskSelector, bool supportsBoundedKVCache = false, int32_t slidingWindowSize = -1,
        std::vector<float> const& qkvScales = {}, std::optional<float> attentionScale = std::nullopt,
        int32_t tpSize = 1, int32_t externalKVPrefix = 0, int32_t fp8KVPrefix = 0);
    AttentionPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    AttentionPlugin() = delete;
    AttentionPlugin(AttentionPlugin const&) = delete;
    ~AttentionPlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t getAliasedInput(int32_t outputIndex) noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! \brief NVRTC-compile the XQA decode kernels this layer needs and stage them for serialization.
    //! \note No-op when the configuration has no XQA decode kernel; the layer may still be
    //!       served by a prefill-only backend.
    void compileXQAJitKernelForBuild();

    //! \brief Register every serialized XQA kernel under the key it was compiled for.
    void loadSerializedXQAJitKernel();

    //! \brief Whether the engine carried any serialized XQA JIT kernel for this layer.
    bool hasSerializedXQAJitKernels() const noexcept
    {
        return !mXqaJitKernels.empty();
    }

private:
    //! Produce split K/V FP16 for independent prefill consumers that cannot read the paged pool directly
    //! (FMHA-v2 FP8, padding, and vision-block). Always device-gathers the page table into
    //! @p workspacePtr
    //! (no in-place alias): the gather follows any page table (identity or scrambled) correctly
    //! and identically in debug and release, and dequantizes an FP8 pool to FP16 using @p kScale /
    //! @p vScale so `dataPointer<half>()` consumers never reinterpret FP8 bytes. The V half lives
    //! at pool page id + numPages.
    //! When @p seqLen == 0, gathers the full capacity → output is [B, capPadded, Hkv, D].
    //! When @p seqLen > 0, gathers only the first seqLen tokens → output is [B, seqLen, Hkv, D];
    //! the compact form lets downstream kernels derive batch stride from the output's S dimension.
    //! An unmapped page zero-fills its destination span (@p kvSeqLens gives each slot's live
    //! length, which bounds the copied prefix); the runtime guarantees mapped coverage for live
    //! positions, so an in-range unmapped page cannot occur by construction.
    static std::pair<rt::Tensor, rt::Tensor> splitPagedKV(rt::Tensor const& poolTensor, int32_t const* pageTable,
        int32_t const* kvSeqLens, int32_t maxPagesPerSeq, std::byte*& workspacePtr, int32_t batchSize,
        int32_t numKVHeads, int32_t capPadded, int32_t headSize, int32_t seqLen, float kScale, float vScale,
        cudaStream_t stream);

    //! enqueue() body. enqueue() wraps it in a try/catch so a thrown error
    //! fails the call instead of terminating the process (enqueue is noexcept).
    int32_t enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    //! Phase 8: prefill rows [0, T_p) through ragged paged FMHA, decode rows [T_p, T) through enqueueDecodeRows.
    int32_t enqueueCompactImpl(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    //! Decode rows of one enqueue: B sequences of S rows each. Phase 3 passes the whole step, phase 8 views of its
    //! decode suffix.
    struct DecodeRows
    {
        rt::Tensor const& packedQKV;       //!< [B, S, Hq or Hq + 2 * Hkv, D]
        rt::Tensor const& ropeCosSin;      //!< [B, S, rotaryDim], token-aligned
        rt::Tensor const& sequenceLengths; //!< [B] KV lengths after this step
        rt::Tensor const& positionIds;     //!< [B, S] tree or explicit positions; empty otherwise
        rt::Tensor& treeMask;              //!< [B, S, maskWords] for tree decoding; empty otherwise
        rt::Tensor& kvCache;               //!< Donor input pool for shared-KV layers, the output pool otherwise
        rt::Tensor& output;                //!< [B, S, Hq, D]
        int32_t const* queryLengths;       //!< [B], read by tree decoding only
        int32_t const* pageTable;          //!< [B, 2, maxPagesPerSeq]
        int32_t maxPagesPerSeq;
        half const* qNormGamma;
        half const* kNormGamma;
        float const* attentionSinks;
        //! skip_softmax_decode_scale as flattened (bin, S) pairs; nullptr or len < 2 runs dense.
        int32_t const* skipSoftmaxDecodeScaleTable;
        int32_t skipSoftmaxDecodeScaleTableLen;
        bool explicitPositionIds;
    };

    //! Input-slot layout implied by this instance's static fields.
    AttentionInputLayout inputLayout() const noexcept;

    //! Decode emission of phase 3 and of the phase 8 decode rows: RoPE and KV write, then CTM or XQA.
    int32_t enqueueDecodeRows(DecodeRows const& rows, bool tree, std::byte* alignedWorkspacePtr, cudaStream_t stream);

    //! Ragged paged FMHA backend for compact prefill rows; kNONE when no runner covers this layer.
    ContextFMHABackend selectCompactPrefillBackend() const noexcept;

    //! Whether phase 8 runs for this layer given its packed QKV width @p qkvWidth (attentionCompactSupport); never for
    //! a two-pool external-prefix layer. getWorkspaceSize() reserves the compact workspace exactly when it holds.
    bool supportsCompactPhase(int64_t qkvWidth) const noexcept;

    //! This layer's kernel-deciding configuration, shared with the runtime's compact capability check.
    AttentionLayerTraits layerTraits(bool qOnlyInput = false) const noexcept;

    //! This layer's name and shape for error logs (describePluginLayer).
    std::string describeLayer() const;

    //! Whether the paged CuTe DSL D512 bidirectional-mask prefill kernel is available.
    bool canUseCuteDslBidirectionalForPrefill() const noexcept;

    //! Validate that a vision-block prefill backend and XQA decode backend are available.
    void enforceVisionBlockKernelSupport() const;

    //! Validate Ampere's FP8-storage/FP16-compute prefill and decode contract.

    //! Resolve one qk_norm gamma engine-weight input to a device pointer
    //! (nullptr when absent).
    half const* resolveNormGammaInput(
        nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs, int32_t inputIdx) const;

    float const* resolveAttentionSinkInput(
        nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs, int32_t inputIdx) const;

protected:
    trt_edgellm::XQAJitKey getXQAJitKey() const noexcept;
    bool canCompileXQAJitKernel() const noexcept;

    std::string mLayerName; //!< Plugin layer name
    std::string mNamespace; //!< Plugin namespace

    //! Number of query heads (specified by model, runtime constant)
    int32_t mNumQHeads{};
    //! Number of key-value heads (specified by model, runtime constant)
    int32_t mNumKVHeads{};
    //! Number of elements per head (head dimension)
    int32_t mHeadSize{};
    float mAttentionScale{}; //!< Absolute QK^T multiplier.
    //! Whether to enable tree attention for EAGLE speculative decoding
    int32_t mEnableTreeAttention{};
    //! Whether the optional token-aligned input carries [T_exec] Gemma4 image block IDs.
    int32_t mEnableVisionBlockAttention{};
    //! Whether the fused per-head q_norm / k_norm RMSNorm is enabled. When set, the q/k gamma
    //! engine-weight constants are wired as optional plugin inputs right after the required ones.
    int32_t mEnableQKNorm{};
    //! QK-norm order relative to RoPE: 0 = norm then rotate (Qwen3 convention),
    //! 1 = rotate then norm (HunYuan V1). Meaningful only when mEnableQKNorm.
    int32_t mQKNormPostRope{};
    //! Whether this layer reads K/V from a donated (shared) cache: the packed input carries
    //! Q only [T_exec, Hq*D] and the plugin skips the KV-cache write.
    int32_t mEnableKVShared{};
    //! Whether speculative query rows occupy consecutive positions, as required by the linear DSpark SWA path.
    int32_t mEnableContiguousQuerySwa{};
    //! Whether a learned per-query-head attention sink is supplied as an engine-weight input.
    int32_t mEnableAttentionSink{};
    //! QKV datatype is FP16; KV cache storage precision is configured separately.
    nvinfer1::DataType const mDataType{nvinfer1::DataType::kHALF};
    int32_t mSMVersion; //!< CUDA SM version

    int32_t mExternalKVPrefix{}; //!< Read an independent prefix without modifying it
    int32_t mFp8KVPrefix{};      //!< Prefix pool precision; the writable action pool remains FP16
    int32_t mEnableFp8KVCache{}; //!< Whether FP8 KV cache is enabled
    //! Whether the optional runtime context-mask selector input is present. Shape [0] keeps default causal/sliding
    //! context attention; shape [batch] selects padding/non-causal context attention. The tensor value is ignored.
    int32_t mEnableContextMaskSelector{};
    //! Host QKV dequant scales [q, k, v] (quant→orig).
    //! - q scale: used to quantize FP16 Q to FP8 (CuTe DSL path) and folded into softmaxScale.
    //! - k scale: used for FP8 KV cache quantization/dequantization and folded into softmaxScale.
    //! - v scale: used for FP8 KV cache quantization/dequantization and folded into scaleOutput.
    //! Attention output is always FP16; downstream Q/DQ for o_proj is handled by the TRT graph.
    std::vector<float> mQkvScales{1.f, 1.f, 1.f};

    //! Epsilon for the fused per-head q/k RMSNorm. The gamma weights themselves are
    //! optional engine-weight constant inputs (wired only when enable_qk_norm).
    float mRmsNormEps{1e-6f};

    //! Sliding window size for attention (-1 = no sliding window, >0 = window size)
    int32_t mSlidingWindowSize = -1;

    //! Whether this layer can use a sparse logical page table backed by an O(W) physical pool. The runtime shape
    //! of swa_kv_cache_mode selects bounded or full storage for the engine instance.
    bool mSupportsBoundedKVCache{false};

    //! Tensor-parallel world size of the rank this layer was exported for; TP>1 decodes on XQA on every SM.
    int32_t mTpSize{1};

    //! Whether the skip_softmax_prefill_scale / skip_softmax_decode_scale table inputs are wired; set only
    //! when that phase deploys a skip-softmax (BLASST) table. The tables' runtime contents drive the kernel.
    int32_t mSkipSoftmaxPrefillInput{};
    int32_t mSkipSoftmaxDecodeInput{};

    ContextFMHABackend mContextFMHABackend{ContextFMHABackend::kNONE};

    //! NVRTC-compiled XQA decode kernels, each paired with the key it was compiled for.
    //! One entry for a plain decode layer, two when tree attention also needs a spec-decode kernel.
    std::vector<trt_edgellm::XQAJitKernel> mXqaJitKernels;
    //! Serialized form of mXqaJitKernels. Held as a member because getFieldsToSerialize
    //! hands TensorRT a pointer into it.
    std::vector<uint8_t> mXqaJitBlob;
    //! Context-owned counters with a stable address for all captured batch shapes. Not serialized or cloned.
    rt::Tensor mXqaSemaphores;
    //! Whether FMHA context kernels are available for this configuration.
    bool mCanImplementFMHA{true};

    //! Whether the FP16 D512 paged CuTe DSL bidirectional-mask kernel is available.
    bool mCanImplementCuteDslBidirectionalFMHA{false};

    //! Whether the FMHA-v2 CuTe DSL d256/d512 vision-block context variant is active.
    bool mUseFMHAV2VisionBlockFMHA{false};

    //! Whether the selected CuTe DSL backend supports a dense PADDING context
    //! kernel for runtime-selected non-causal DiffusionGemma denoise attention.
    bool mCanImplementPaddingFMHA{false};

    //! Plugin-owned split-K arrival counters for the CTM decode kernel, zeroed at allocation and
    //! self-resetting across launches. Allocated once at the profile-max size so the address stays
    //! valid for every CUDA graph captured from this context. They cannot live in the TensorRT
    //! workspace, which other layers overwrite between enqueues.
    int32_t* mCtmSemaphores{nullptr};
    size_t mCtmSemaphoreElems{};
    //! Element count at the profile's max batch and query length; fixed in configurePlugin and serialized.
    //! Zero means the engine predates the field and must be rebuilt.
    int32_t mCtmSemaphoreCapacityElems{};
    //! Blackwell (SM100/101/103/110) at TP=1: the CTM kernels are the only decode backend; the engine carries no
    //! XQA blobs.
    bool mCtmDecodeOnly{false};

    //! \returns true when this layer's static configuration routes vanilla decode to the CTM kernel.
    bool useCtmDecode() const noexcept;
    //! \returns true when this layer's static configuration routes tree (spec-decode) decode to the CTM kernel.
    bool useCtmTreeDecode() const noexcept;
    //! int32 element count both CTM decode modules need for this batch and per-sequence query length.
    size_t ctmSemaphoreElems(int32_t batchSize, int32_t qSeqLen) const noexcept;
    bool allocateCtmSemaphores() noexcept;
    void releaseCtmSemaphores() noexcept;

    //! INT32 representation of mSupportsBoundedKVCache kept alive for TensorRT plugin serialization.
    int32_t mSupportsBoundedKVCacheToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! \brief Factory class for creating AttentionPlugin instances
class AttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    AttentionPluginCreator();
    ~AttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
