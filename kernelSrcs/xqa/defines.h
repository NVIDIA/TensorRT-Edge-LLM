/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include "mha_stdheaders.cuh"
#include "common/cudaMacros.h"
#include "kernels/decodeAttentionKernels/xqaKernelTypes.h"

#define STATIC_NB_K_HEADS 0
#if STATIC_NB_K_HEADS
#define NB_K_HEADS 2
#endif

// Allowed values are multiples of 16 in range [16, 256]. SM100+ also supports selected 512-wide kernels.
#ifndef HEAD_ELEMS
#define HEAD_ELEMS 128
#endif

// nbQHeads / nbKHeads for MQA/GQA
#ifndef HEAD_GRP_SIZE
#define HEAD_GRP_SIZE 8
#endif

#define IS_MLA (HEAD_GRP_SIZE == 128 && HEAD_ELEMS == 576)

#if IS_MLA
#if !SUPPORTS_FP8
#error "MLA XQA kernels require CUDA FP8 support."
#endif
#define INPUT_ELEM __nv_fp8_e4m3
#define INPUT_ELEM2 __nv_fp8x2_e4m3
#define HEAD_ELEMS_V 512
#else
// 1 means fp16 and 0 means bf16 input/output
#ifndef INPUT_FP16
#define INPUT_FP16 1
#endif

// Don't modify
#if INPUT_FP16
#define INPUT_ELEM half
#define INPUT_ELEM2 half2
#else
#define INPUT_ELEM __nv_bfloat16
#define INPUT_ELEM2 __nv_bfloat162
#endif
#endif

// For beam search. Allowed values: 1, 4
#ifndef BEAM_WIDTH
#define BEAM_WIDTH 1
#endif

#ifndef SPEC_DEC
#define SPEC_DEC 0
#endif

#ifndef XQA_2CTA_HEAD_DIM512
#define XQA_2CTA_HEAD_DIM512 0
#endif

#ifndef TILED_QKV_STAGING_HEAD_DIM512
#define TILED_QKV_STAGING_HEAD_DIM512 0
#endif

#if XQA_2CTA_HEAD_DIM512
static_assert(HEAD_ELEMS == 512, "XQA_2CTA_HEAD_DIM512 is only valid for head_dim=512.");
static_assert(BEAM_WIDTH == 1, "XQA_2CTA_HEAD_DIM512 supports beam width 1 only.");
#endif

#if TILED_QKV_STAGING_HEAD_DIM512
static_assert(HEAD_ELEMS == 512, "TILED_QKV_STAGING_HEAD_DIM512 is only valid for head_dim=512.");
static_assert(!XQA_2CTA_HEAD_DIM512, "TILED_QKV_STAGING_HEAD_DIM512 is a single-CTA path.");
#endif

#if SPEC_DEC
using MaskType = uint32_t;

#ifndef M_TILESIZE
#define M_TILESIZE 32
#endif
#endif

// Enables SWAP AB optimization for speculative decoding when using a small, fixed Q_SEQ_LEN.
// NOTE: Requires a uniform input sequence length for the entire batch.
#ifdef SPEC_Q_SEQ_LEN
static_assert(SPEC_DEC, "SPEC_Q_SEQ_LEN should only be used when SPEC_DEC is enabled.");
#endif

// 0: half/bf16 based on INPUT_FP16; 1: int8_t; 2: __nv_fp8_e4m3
#ifndef CACHE_ELEM_ENUM
#define CACHE_ELEM_ENUM TRT_EDGELLM_XQA_CACHE_ELEM_FP8_E4M3
#endif

#if CACHE_ELEM_ENUM == TRT_EDGELLM_XQA_CACHE_ELEM_FP8_E4M3 && !SUPPORTS_FP8
#error "FP8 XQA KV cache requires CUDA FP8 support."
#endif

// don't modify
#define USE_KV_CACHE true

// don't modify
#ifndef ALLOW_MULTI_BLOCK_MODE
#define ALLOW_MULTI_BLOCK_MODE true
#endif

#ifndef TOKENS_PER_PAGE
#define TOKENS_PER_PAGE 0
#endif

// Paged KV Cache Format
// 0 - XQA Original
// 1 - separate K and V cache pools, each with layout (batch, seq_len, head, head_elem) for VLLM/SGLang
#if TOKENS_PER_PAGE != 0
#ifndef PAGED_KV_CACHE_LAYOUT
#define PAGED_KV_CACHE_LAYOUT 0
#endif
#endif

// don't modify
#define USE_BEAM_SEARCH (BEAM_WIDTH > 1)

#if CACHE_ELEM_ENUM == TRT_EDGELLM_XQA_CACHE_ELEM_INPUT
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll")
#else
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll(1)")
#endif

// good for short sequence length but bad for long sequence length. Only for mha.cu.
#ifndef SHORT_SEQ_OPT
#define SHORT_SEQ_OPT 1
#endif

#ifndef SLIDING_WINDOW
#define SLIDING_WINDOW 0
#endif

// 0 - no PDL
// 1 - naive PDL
// 2 - aggressive PDL (implemented only in mha_sm90.cu for now)
#ifndef ENABLE_PDL
#define ENABLE_PDL 2
#endif

#ifndef USE_INPUT_KV
#define USE_INPUT_KV 0
#endif

#if USE_INPUT_KV
// 0 - no RoPE
// 1 - NEOX style
// 2 - GPTJ style
#ifndef ROPE_STYLE
#define ROPE_STYLE 0
#endif

#if SPEC_DEC
#error "SPEC_DEC is not supported for USE_INPUT_KV"
#endif
#endif

// Output element type:
//   0 - input element type
//   1 - KV cache element type
#ifndef LOW_PREC_OUTPUT
#define LOW_PREC_OUTPUT 0
#endif

#if LOW_PREC_OUTPUT
static_assert(CACHE_ELEM_ENUM != TRT_EDGELLM_XQA_CACHE_ELEM_INPUT);
#endif

// true should be better if warpTile.x * cacheElemSize < 128. otherwise use false.
#define GRP_LOAD_V                                                                                                    \
    (CACHE_ELEM_ENUM != 0)                                                                                            \
    || (HEAD_ELEMS == 256 && TOKENS_PER_PAGE != 0 && BEAM_WIDTH > 1)

// use custom barrier for NVRTC to avoid pulling in many headers
#ifndef USE_CUSTOM_BARRIER
#define USE_CUSTOM_BARRIER 1
#endif

#ifndef OPTIMIZE_FOR_LATENCY
#define OPTIMIZE_FOR_LATENCY 1
#endif

#ifndef IS_SPEC_DEC_TREE
#define IS_SPEC_DEC_TREE 1 // by default SPEC_DEC expect tree-based draft token structure
#endif

#define DBG_BATCH_SIZE 2
#define DBG_SEQ_LEN 256 * 4 + 3
#define DBG_NB_CTAS_PER_SEQ 8

#include <cuda_fp16.h>
#if SUPPORTS_FP8
template <int32_t elemTypeEnum>
using ElemType = mha::conditional_t<elemTypeEnum == TRT_EDGELLM_XQA_CACHE_ELEM_INPUT, INPUT_ELEM,
    mha::conditional_t<elemTypeEnum == TRT_EDGELLM_XQA_CACHE_ELEM_INT8, int8_t,
        mha::conditional_t<elemTypeEnum == TRT_EDGELLM_XQA_CACHE_ELEM_FP8_E4M3, __nv_fp8_e4m3, void>>>;
#else
template <int32_t elemTypeEnum>
using ElemType = mha::conditional_t<elemTypeEnum == TRT_EDGELLM_XQA_CACHE_ELEM_INPUT, INPUT_ELEM,
    mha::conditional_t<elemTypeEnum == TRT_EDGELLM_XQA_CACHE_ELEM_INT8, int8_t, void>>;
#endif
