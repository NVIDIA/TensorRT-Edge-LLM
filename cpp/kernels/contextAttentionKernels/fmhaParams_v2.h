/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits.h>
#include <math.h>
#include <stdint.h>

//! \brief Parameters for ALiBi (Attention with Linear Biases) positional encoding
struct AlibiParams
{
    //! \brief Rounds down an integer to the nearest power of two
    //! \param x The input integer value
    //! \return The largest power of two less than or equal to x
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;

    //! \brief Constructor for ALiBi parameters
    //! \param h Number of attention heads
    //! \param scale_after_alibi Scaling factor to apply after ALiBi bias (default 1.0)
    AlibiParams(int h, float scale_after_alibi = 1.f)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    //! \brief Constructor for ALiBi parameters with tensor parallelism support
    //! \param h Number of attention heads per rank
    //! \param s Sequence length per rank
    //! \param tp_size Tensor parallelism size
    //! \param rank Current rank in tensor parallel group
    //! \param scale_after_alibi Scaling factor to apply after ALiBi bias (default 1.0)
    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi = 1.f)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};             //!< Number of heads rounded down to nearest power of two
    float alibi_neg4_div_h{};  //!< ALiBi slope computation: -4.0 / h_pow_2
    float scale_after_alibi{}; //!< Scaling factor to apply after ALiBi bias
    //! Head index offset for tensor parallelism.
    //! Could be simplified to `int rank` derive the others as `num_heads * rank, s * rank` at
    //! runtime, but this makes assumptions about the layout downstream
    //! (e.g. downstream may only split across the head dimension, so s would be the full sequence)
    int head_idx_offset = 0;
    int sequence_pos_offset = 0; //!< Sequence position offset for tensor parallelism
};

//! \brief TMA (Tensor Memory Accelerator) descriptor structure
//!
//! An opaque 64-byte aligned structure used for TensorRT Memory Accelerator descriptors
//! \cond INTERNAL
typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;
//! \endcond

//! \brief Array structure for managing paged KV cache blocks
struct KvBlockArray
{
    using PtrType = int32_t;

    int32_t mMaxSeqs;         //!< Current number of sequences
    int32_t mMaxBlocksPerSeq; //!< Max number of blocks per sequence
    int32_t mTokensPerBlock;  //!< Number of tokens per block. It must be power of 2.
    //! Exponent of number of tokens with base 2.
    //! E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
    int32_t mTokensPerBlockLog2;
    //! Table maps logical block idx to the data pointer of k/v cache block pool.
    //! Shape [B, W, 2, M], where 2 is table for K and V,
    //! B is current number of sequences,
    //! W is beam width,
    //! M is Max number of blocks per sequence

    int32_t mBytesPerBlock; //!< Size of KV cache blocks in bytes (H*D*T*sizeof(DataType))
    void* mPoolPtr;         //!< Pointer to beginning of pool
    PtrType* mBlockOffsets; //!< Pointer to block offsets

    KvBlockArray() = default;

    //! \brief Constructor for KV block array
    //! \param batchSize Current number of sequences
    //! \param maxBlocksPerSeq Maximum number of blocks per sequence
    //! \param tokensPerBlock Number of tokens per block (must be power of 2)
    //! \param bytesPerBlock Size of each KV cache block in bytes
    //! \param poolPtr Pointer to the beginning of the memory pool
    KvBlockArray(
        int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t bytesPerBlock, void* poolPtr)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , mBytesPerBlock{bytesPerBlock}
        , mPoolPtr{poolPtr}
        , mBlockOffsets{nullptr}
    {
        float const tokensPerBlockSeqLog2 = std::log2(mTokensPerBlock);
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

//! \brief Enumeration of context attention mask types
enum class ContextAttentionMaskType
{
    PADDING = 0,               //!< Mask the padded tokens
    CAUSAL,                    //!< Mask the padded tokens and all the tokens that come after in a sequence
    SLIDING_OR_CHUNKED_CAUSAL, //!< Causal mask + attend to the specific sliding window or chunk
    CUSTOM_MASK,               //!< The custom mask input
};

//! \brief Enumeration of attention input layout types
enum class AttentionInputLayout
{
    PACKED_QKV = 0,  //!< QKV are packed into [B, S, 3, H, D] layout
    CONTIGUOUS_Q_KV, //!< Q has contiguous [Compact_S, H, D] layout, while KV has contiguous [Compact_S, 2, H, D] layout
    //! Q has contiguous [B, S, H, D] layout, while paged KV layout are blocks of indices with shape
    //! of [B, 2, Blocks_per_Seq], and the indice indicates the block distance to the pool ptr in
    //! global memory
    Q_PAGED_KV,
    //! Q has [B, S, H, D] layout,
    //! K has [B, S, H_kv, D] layout,
    //! V has [B, S, H_kv, Dv] layout
    SEPARATE_Q_K_V,
};

//! \brief Parameters for fused multi-head attention version 2
struct FusedMultiheadAttentionParamsV2
{
    void* qkv_ptr;               //!< The packed QKV matrices
    void* q_ptr;                 //!< The separate Q matrix
    void* k_ptr;                 //!< The separate K matrix
    void* v_ptr;                 //!< The separate V matrix
    void* kv_ptr;                //!< The separate KV matrix (contiguous KV)
    KvBlockArray paged_kv_cache; //!< The separate paged kv cache
    void* packed_mask_ptr;       //!< The mask to implement drop-out
    float* attention_sinks;      //!< The attention sinks (per head)
    void* o_ptr;                 //!< The O matrix (output)
    //! The Softmax stats vector of layout [2, B, S, H], including softmax_sum and softmax_max
    void* softmax_stats_ptr;

    int64_t q_stride_in_bytes;             //!< The stride between rows of Q
    int64_t k_stride_in_bytes;             //!< The stride between rows of K
    int64_t v_stride_in_bytes;             //!< The stride between rows of V
    int64_t packed_mask_stride_in_bytes;   //!< The stride between matrices of packed mask
    int64_t o_stride_in_bytes;             //!< The stride between rows of O
    int64_t softmax_stats_stride_in_bytes; //!< The stride between rows of softmax_stats_ptr

    //! TMA descriptors on device.
    //! Either q in packed qkv [B, S, 3, H, D] or separate q layout [B, S, H, D].
    cudaTmaDesc tma_desc_q;
    //! TMA descriptors for packed/contiguous/paged kv cache.
    //! Kv in packed qkv layout: [B, S, 3, H, D]
    //! Contiguous kv layout: [B, 2, H, S, D].
    //! Paged kv layout: [UINT32_MAX, H, Tokens_per_block, D].
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v; //!< TMA descriptor for V
    cudaTmaDesc tma_desc_o; //!< TMA descriptor for O

    int blocks_per_tma_load;      //!< TMA load of paged kv cache
    int blocks_per_tma_load_log2; //!< Log2 of blocks per TMA load

    //! The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads.
    //! b: batch size, h: number of query heads, h_kv: number of key/value heads,
    //! h_q_per_kv: number of query heads per key/value head, s: sequence length,
    //! s_kv: key/value sequence length, d: head dimension
    int b, h, h_kv, h_q_per_kv, s, s_kv, d;
    int dv = 0;                //!< The dimension of V. If unset, dv = d
    int num_grouped_heads = 1; //!< The number of grouped heads in the seqlen dimension
    //! Sliding Window Attention.
    //! Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;
    int log2_chunked_attention_size
        = 0; //!< The chunked attention size in log2 (> 0 means that chunked attention is enabled)
    //! The scaling factors for the kernel
    uint32_t scale_bmm1, softcapping_scale_bmm1, scale_softmax, scale_bmm2;

    uint32_t* scale_bmm1_d; //!< The scaling factors in the device memory (required by TRT-LLM + FP8 FMHA)
    uint32_t* scale_bmm2_d; //!< The scaling factors in the device memory (required by TRT-LLM + FP8 FMHA)

    int* cu_q_seqlens;  //!< Array of length b+1 holding prefix sum of actual q sequence lengths
    int* cu_kv_seqlens; //!< Array of length b+1 holding prefix sum of actual kv sequence lengths
    //! Array of length b+1 holding prefix sum of actual mask sequence lengths.
    //! It might not be the same as cu_q_seqlens as the mask seqlens will be padded.
    int* cu_mask_rows;

    bool has_alibi = false;     //!< If the kernel is using ALiBi or not
    AlibiParams alibi_params{}; //!< ALiBi parameters

    uint32_t* tile_id_counter_ptr; //!< M tile id counter for dynamic scheduling
    uint32_t num_tiles;            //!< Total number of tiles
    uint32_t num_tiles_per_head;   //!< Number of tiles per head
    bool use_balanced_scheduling;  //!< Whether to use balanced scheduling

    bool is_s_padded = false; //!< Is input/output padded

    //! \brief SAGE attention parameters
    struct SageAttention
    {
        //! \brief Per-block quantization scales
        struct Scales
        {
            int max_nblock; //!< ceil(max_seqlen / block_size)
            float* scales;  //!< The scale of each block, layout: (B, H, max_nblock)
        } q, k, v;          //!< Scales for Q, K, V
    } sage;                 //!< SAGE attention configuration
};

//! \brief Flags to control kernel choice and launch parameters
struct LaunchParams
{
    //! Flags to control small batch kernel choice.
    //! true: never unroll
    bool ignore_b1opt = false;
    bool force_unroll = false;        //!< true: always unroll
    bool force_fp32_acc = false;      //!< Use FP32 accumulation
    bool interleaved = false;         //!< The C/32 format
    bool use_tma = false;             //!< By default TMA is not used
    int total_q_seqlen = 0;           //!< Total number of q tokens to set TMA descriptors
    int total_kv_seqlen = 0;          //!< Total number of kv tokens to set TMA descriptors
    bool flash_attention = false;     //!< If flash attention is used (only FP16)
    bool warp_specialization = false; //!< If warp-specialized kernels are used (only SM90 HGMMA + TMA)
    bool use_granular_tiling = false; //!< Granular tiling flash attention kernels
    //! Causal masking or sliding_or_chunked_causal masking or dense(padding) mask
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    AttentionInputLayout attention_input_layout = AttentionInputLayout::PACKED_QKV; //!< The attention input layout
    bool enable_attn_logit_softcapping
        = false;                   //!< Enable attention logit softcapping (choose kernels with softcapping_scale_bmm1)
    int multi_processor_count = 0; //!< Hardware properties to determine how to launch blocks
    int device_l2_cache_size = 0;  //!< Device L2 cache size in bytes
};