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

// Alpamayo 2B prefill FFN on SM110: FP8 gate/up GEMM (M x 12288 x 2048, 2-CTA, 3 stages) with SwiGLU
// and FP8 output. Generated source: regenerate rather than hand-edit.

#include "alpamayoKernels.h"
#include "common/logger.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace trt_edgellm;

// Device code is SM110 (sm_110a) only. For any other target in CMAKE_CUDA_ARCHITECTURES the kernel body is a
// trap, so a launch on an unsupported build fails instead of returning stale output; the plugin's device check
// keeps it from being launched on other GPUs.
#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define ALPAMAYO_SM110_DEVICE 1
#else
#define ALPAMAYO_SM110_DEVICE 0
#endif

// Dynamic shared memory the kernel uses, and the alignment its 128B-swizzle TMA / UMMA buffers need. The launch
// requests kSMEM_BYTES + kSMEM_ALIGN so the kernel can align the pool at runtime.
constexpr uint32_t kSMEM_BYTES = 180224;
constexpr uint32_t kSMEM_ALIGN = 1024;

// kernel=alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh | mode=mma | grid=('grid_size', 1) | block=(192,)

// #define M 0  // (passed as kernel arg)
// #define N 0  // (passed as kernel arg)
// #define K 0  // (passed as kernel arg)
// #define grid_n 0  // (passed as kernel arg)
// #define grid_size 0  // (passed as kernel arg)

#if ALPAMAYO_SM110_DEVICE
__device__ __forceinline__ uint64_t _gen_smem_desc_swz_128b_lbo16_sbo1024(uint32_t addr)
{
    uint64_t desc = 0;
    desc |= ((uint64_t) (addr >> 4)) & 0x3FFF;       // bits [0:13] base
    desc |= ((uint64_t) (16 >> 4) & 0x3FFF) << 16;   // bits [16:29] LBO
    desc |= ((uint64_t) (1024 >> 4) & 0x3FFF) << 32; // bits [32:45] SBO
    desc |= (1ULL << 46);                            // bits [46:48] base mode
    desc |= ((uint64_t) 2) << 61;                    // bits [61:63] swz=128B
    return desc;                                     // LBO=16, SBO=1024, swz=128B
}

// Rounds a shared-memory pointer up to the next `align`-byte shared address (`align` is a power of two).
__device__ __forceinline__ char* alignSharedPtr(char* ptr, uint32_t align)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    return ptr + ((align - (addr & (align - 1))) & (align - 1));
}

namespace kl
{
__device__ __forceinline__ uint32_t elect_one()
{
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}
} // namespace kl
#endif // ALPAMAYO_SM110_DEVICE
__global__ void alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh(__grid_constant__ const CUtensorMap tma_A,
    __grid_constant__ const CUtensorMap tma_B, __grid_constant__ const CUtensorMap tma_C,
    float const* __restrict__ gate_decode, float const* __restrict__ up_encode, int M, int N, int K, int grid_n,
    int grid_size)
{
#if ALPAMAYO_SM110_DEVICE
    int const _kd_warp_idx = __shfl_sync(0xffffffffu, (int) (threadIdx.x >> 5), 0);
    int _cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;\n" : "=r"(_cta_rank));
    __shared__ __align__(8) uint64_t _kl_mbar_arena[8];
    uint32_t const _kl_mbar_arena_base = (uint32_t) __cvta_generic_to_shared(&_kl_mbar_arena[0]);
    // device_func(gemm_percta_tile_schedule)
    int _cluster_idx = ((int) blockIdx.x / 2);
    int _num_clusters = ((int) gridDim.x / 2);
    int _grid_m = ((M + 255) / 256);
    int _num_tiles = (_grid_m * grid_n);
    int _K_ITERS = (K / 128);
    // Where the dynamic segment starts is toolchain-dependent; align it inside the launch's kSMEM_ALIGN slack.
    extern __shared__ char _smem_raw[];
    char* _smem_pool = alignSharedPtr(_smem_raw, kSMEM_ALIGN);
    __nv_fp8_e4m3* a_smem = (__nv_fp8_e4m3*) (_smem_pool + 0);
    __nv_fp8_e4m3* b_smem_0 = (__nv_fp8_e4m3*) (_smem_pool + 49152);
    __nv_fp8_e4m3* b_smem_1 = (__nv_fp8_e4m3*) (_smem_pool + 98304);
    __nv_fp8_e4m3* _swiglu_c_smem_0 = (__nv_fp8_e4m3*) (_smem_pool + 147456);
    __nv_fp8_e4m3* _swiglu_c_smem_1 = (__nv_fp8_e4m3*) (_smem_pool + 163840);
    if (_kd_warp_idx == 0 && kl::elect_one())
    {
        for (int _s = 0; _s < 3; _s++)
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(_kl_mbar_arena_base + (uint32_t) (8 * (0 + _s))),
                "r"(2));
        for (int _s = 0; _s < 3; _s++)
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(_kl_mbar_arena_base + (uint32_t) (8 * (3 + _s))),
                "r"(1));
        for (int _s = 0; _s < 1; _s++)
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(_kl_mbar_arena_base + (uint32_t) (8 * (6 + _s))),
                "r"(1));
        for (int _s = 0; _s < 1; _s++)
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(_kl_mbar_arena_base + (uint32_t) (8 * (7 + _s))),
                "r"(256));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    int _phase_tma_bar[3] = {0, 0, 0};
    int _phase_mma_bar[3] = {1, 1, 1};
    int _phase_epilogue_bar[1] = {1};
    __syncthreads();
    // tmem alloc: acc (512 cols, cta_group::2)
    __shared__ uint32_t _tmem_acc_smem[1];
    if (((threadIdx.x / 32) == 1))
    {
        uint32_t _sa = (uint32_t) __cvta_generic_to_shared(&_tmem_acc_smem[0]);
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :
            : "r"(_sa), "r"(512)
            : "memory");
    }
    __syncthreads();
    uint32_t _tmem_acc = _tmem_acc_smem[0];
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    // warp_role(tma, warps=[0])
    if ((_kd_warp_idx) == 0)
    {
#pragma unroll 1
        for (int _tile = _cluster_idx; _tile < _num_tiles; _tile += _num_clusters)
        {
            int _group_id = (_tile / (grid_n * 4));
            int _first_m = (_group_id * 4);
            int _group_m = (((_grid_m - _first_m) < 4) ? (_grid_m - _first_m) : 4);
            int _bid_m = (_first_m + (_tile % _group_m));
            int _bid_n = ((_tile % (_group_m * grid_n)) / _group_m);
            uint32_t _tma_a_y = ((_bid_m * 256) + (_cta_rank * 128));
            uint32_t _tma_b_y_0 = ((_bid_n * 512) + (_cta_rank * 128));
            uint32_t _tma_b_y_1 = (((_bid_n * 512) + 256) + (_cta_rank * 128));
            for (int _k = 0; _k < _K_ITERS; _k += 1)
            {
                asm volatile(
                    "{\n"
                    ".reg .pred _p;\n"
                    "_wait_mma_bar_0:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 _p, [%0], %1, 0x989680;\n"
                    "@!_p bra _wait_mma_bar_0;\n"
                    "}\n"
                    :
                    : "r"((_kl_mbar_arena_base + (uint32_t) (8 * (3 + (_k % 3))))), "r"(_phase_mma_bar[_k % 3])
                    : "memory");
                _phase_mma_bar[_k % 3] ^= 1;
                bool const _gen_issue_leader_21 = kl::elect_one();
                if (_gen_issue_leader_21)
                {
                    // TMA load: tma_A → smem (stage=(_k % 3), tx=16384B)
                    {
                        uint32_t _smem = (uint32_t) __cvta_generic_to_shared((char*) a_smem + (_k % 3) * 16384);
                        asm volatile(
                            "{\n"
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
                            "[%0], [%1, {%2, %3, %4}], [%5];\n"
                            "}\n"
                            :
                            : "r"(_smem), "l"(&tma_A), "r"((uint32_t) (0)), "r"((uint32_t) (_tma_a_y)),
                            "r"((uint32_t) (_k)),
                            "r"(((_kl_mbar_arena_base + (uint32_t) (8 * (0 + (_k % 3))))) & 0xFEFFFFFFu)
                            : "memory");
                    }
                    // TMA load: tma_B → smem (stage=(_k % 3), tx=16384B)
                    {
                        uint32_t _smem = (uint32_t) __cvta_generic_to_shared((char*) b_smem_0 + (_k % 3) * 16384);
                        asm volatile(
                            "{\n"
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
                            "[%0], [%1, {%2, %3, %4}], [%5];\n"
                            "}\n"
                            :
                            : "r"(_smem), "l"(&tma_B), "r"((uint32_t) (0)), "r"((uint32_t) (_tma_b_y_0)),
                            "r"((uint32_t) (_k)),
                            "r"(((_kl_mbar_arena_base + (uint32_t) (8 * (0 + (_k % 3))))) & 0xFEFFFFFFu)
                            : "memory");
                    }
                    // TMA load: tma_B → smem (stage=(_k % 3), tx=16384B)
                    {
                        uint32_t _smem = (uint32_t) __cvta_generic_to_shared((char*) b_smem_1 + (_k % 3) * 16384);
                        asm volatile(
                            "{\n"
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
                            "[%0], [%1, {%2, %3, %4}], [%5];\n"
                            "}\n"
                            :
                            : "r"(_smem), "l"(&tma_B), "r"((uint32_t) (0)), "r"((uint32_t) (_tma_b_y_1)),
                            "r"((uint32_t) (_k)),
                            "r"(((_kl_mbar_arena_base + (uint32_t) (8 * (0 + (_k % 3))))) & 0xFEFFFFFFu)
                            : "memory");
                        asm volatile(
                            "{\n"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n"
                            "}\n"
                            :
                            : "r"(((_kl_mbar_arena_base + (uint32_t) (8 * (0 + (_k % 3))))) & 0xFEFFFFFFu),
                            "r"((uint32_t) 49152)
                            : "memory");
                    }
                }
            }
        }
    }
    // warp_role(mma, warps=[1])
    else if (_cta_rank == 0 && (_kd_warp_idx) == 1)
    {
#pragma unroll 1
        for (int _tile = _cluster_idx; _tile < _num_tiles; _tile += _num_clusters)
        {
            asm volatile(
                "{\n"
                ".reg .pred _p;\n"
                "_wait_epilogue_bar_1:\n"
                "mbarrier.try_wait.parity.shared::cta.b64 _p, [%0], %1, 0x989680;\n"
                "@!_p bra _wait_epilogue_bar_1;\n"
                "}\n"
                :
                : "r"((_kl_mbar_arena_base + 56u)), "r"(_phase_epilogue_bar[0])
                : "memory");
            _phase_epilogue_bar[0] ^= 1;
            for (int _k = 0; _k < _K_ITERS; _k += 1)
            {
                asm volatile(
                    "{\n"
                    ".reg .pred _p;\n"
                    "_wait_tma_bar_2:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 _p, [%0], %1, 0x989680;\n"
                    "@!_p bra _wait_tma_bar_2;\n"
                    "}\n"
                    :
                    : "r"((_kl_mbar_arena_base + (uint32_t) (8 * (0 + (_k % 3))))), "r"(_phase_tma_bar[_k % 3])
                    : "memory");
                _phase_tma_bar[_k % 3] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                bool const _gen_issue_leader_100 = kl::elect_one();
                if (_gen_issue_leader_100)
                {
                    // tcgen05.dot: acc += a_smem @ b_smem_0  [256x256x32, f8f6f4] (bk=128) (zero=(_k == 0))
                    // tcgen05.mma.cta_group::2.kind::f8f6f4 idesc=0x10400010 (bk=128, 4 inner iters, desc_iter=a+2/b+2)
                    {
                        uint32_t _tc_acc = (_tmem_acc + 0);
                        uint64_t _tc_a_desc = _gen_smem_desc_swz_128b_lbo16_sbo1024(
                            __cvta_generic_to_shared((char*) (a_smem + ((_k % 3)) * 128 * 128)));
                        uint64_t _tc_b_desc = _gen_smem_desc_swz_128b_lbo16_sbo1024(
                            __cvta_generic_to_shared((char*) (b_smem_0 + ((_k % 3)) * 128 * 128)));
                        {
                            asm volatile(
                                "{\n"
                                ".reg .pred p;\n"
                                "setp.ne.u32 p, %4, 0;\n"
                                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                                "}\n"
                                :
                                : "r"(_tc_acc), "l"(_tc_a_desc), "l"(_tc_b_desc), "r"(272629776),
                                "r"(((_k == 0)) ? 0 : 1)
                                : "memory");
                        }
#pragma unroll
                        for (int _ki = 1; _ki < 4; _ki++)
                        {
                            _tc_a_desc += 2;
                            _tc_b_desc += 2;
                            {
                                asm volatile(
                                    "{\n"
                                    ".reg .pred p;\n"
                                    "setp.ne.u32 p, %4, 0;\n"
                                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                                    "}\n"
                                    :
                                    : "r"(_tc_acc), "l"(_tc_a_desc), "l"(_tc_b_desc), "r"(272629776), "r"(1)
                                    : "memory");
                            }
                        }
                    }
                    // tcgen05.dot: acc += a_smem @ b_smem_1  [256x256x32, f8f6f4] (bk=128) (zero=(_k == 0))
                    // tcgen05.mma.cta_group::2.kind::f8f6f4 idesc=0x10400010 (bk=128, 4 inner iters, desc_iter=a+2/b+2)
                    {
                        uint32_t _tc_acc = (_tmem_acc + 256);
                        uint64_t _tc_a_desc = _gen_smem_desc_swz_128b_lbo16_sbo1024(
                            __cvta_generic_to_shared((char*) (a_smem + ((_k % 3)) * 128 * 128)));
                        uint64_t _tc_b_desc = _gen_smem_desc_swz_128b_lbo16_sbo1024(
                            __cvta_generic_to_shared((char*) (b_smem_1 + ((_k % 3)) * 128 * 128)));
                        {
                            asm volatile(
                                "{\n"
                                ".reg .pred p;\n"
                                "setp.ne.u32 p, %4, 0;\n"
                                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                                "}\n"
                                :
                                : "r"(_tc_acc), "l"(_tc_a_desc), "l"(_tc_b_desc), "r"(272629776),
                                "r"(((_k == 0)) ? 0 : 1)
                                : "memory");
                        }
#pragma unroll
                        for (int _ki = 1; _ki < 4; _ki++)
                        {
                            _tc_a_desc += 2;
                            _tc_b_desc += 2;
                            {
                                asm volatile(
                                    "{\n"
                                    ".reg .pred p;\n"
                                    "setp.ne.u32 p, %4, 0;\n"
                                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                                    "}\n"
                                    :
                                    : "r"(_tc_acc), "l"(_tc_a_desc), "l"(_tc_b_desc), "r"(272629776), "r"(1)
                                    : "memory");
                            }
                        }
                    }
                    // tcgen05.commit + mbarrier arrive (pipeline=mma_pipe, barrier=mma_bar)
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
                        "[%0], %1;\n"
                        :
                        : "r"((_kl_mbar_arena_base + (uint32_t) (8 * (3 + (_k % 3))))), "h"((int16_t) (0x3))
                        : "memory");
                }
            }
            bool const _gen_issue_leader_101 = kl::elect_one();
            if (_gen_issue_leader_101)
            {
                // tcgen05.commit + mbarrier arrive (pipeline=mma_pipe, barrier=mainloop_bar)
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], "
                    "%1;\n"
                    :
                    : "r"((_kl_mbar_arena_base + 48u)), "h"((int16_t) (0x3))
                    : "memory");
            }
        }
    }
    // warp_role(epilogue, warps=[2, 3, 4, 5])
    else if (((_kd_warp_idx) >= 2 && (_kd_warp_idx) <= 5))
    {
        // tmem distributed preamble
        int _kd_tmem_addr_slab = _cta_rank * 128 + (threadIdx.x / 32 % 4) * 32;
        int _kd_tmem_col_base_raw = _kd_tmem_addr_slab << 16;
        int _kd_tmem_col_base = __shfl_sync(0xffffffffu, _kd_tmem_col_base_raw, 0);
        int _accumulator_stage = 0;
        uint32_t _phase_epilogue_mainloop = 0;
        uint32_t const _gen_cursor_epilogue_mainloop_bar_base = (_kl_mbar_arena_base + 48u);
        float _gate_decode = gate_decode[0];
        float _up_encode = up_encode[0];
#pragma unroll 1
        for (int _tile = _cluster_idx; _tile < _num_tiles; _tile += _num_clusters)
        {
            int _group_id = (_tile / (grid_n * 4));
            int _first_m = (_group_id * 4);
            int _group_m = (((_grid_m - _first_m) < 4) ? (_grid_m - _first_m) : 4);
            int _bid_m = (_first_m + (_tile % _group_m));
            int _bid_n = ((_tile % (_group_m * grid_n)) / _group_m);
            if (((threadIdx.x / 32) == 2))
            {
                asm volatile(
                    "{\n"
                    ".reg .pred _p;\n"
                    "_wait_mainloop_bar_3:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 _p, [%0], %1, 0x989680;\n"
                    "@!_p bra _wait_mainloop_bar_3;\n"
                    "}\n"
                    :
                    : "r"(_gen_cursor_epilogue_mainloop_bar_base), "r"(_phase_epilogue_mainloop)
                    : "memory");
            }
            // TmemTileIter: acc → _swiglu_c_smem_0 (128 cols, pack=8, cast=fp8e4m3)
            {
                int _warp_id = threadIdx.x / 32;
                int _lane_id = threadIdx.x % 32;
                int _warp_row_base = (_warp_id % 4) * 32;
                int _smem_row = _warp_row_base + _lane_id;
                int _tmem_row = (_cta_rank * 128 + _warp_row_base + _lane_id) << 16;
                asm volatile("bar.sync %0, %1;\n" : : "r"(1), "r"(128) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                int _tmem_base = _tmem_acc;
#pragma unroll
                for (int _col = 0; _col < 128; _col += 8)
                {
                    float _gate_0[8];
                    {
                        int _rd = _tmem_base + _tmem_row + _col;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                            "[%8];\ntcgen05.wait::ld.sync.aligned;\n"
                            : "=f"(_gate_0[0]), "=f"(_gate_0[1]), "=f"(_gate_0[2]), "=f"(_gate_0[3]), "=f"(_gate_0[4]),
                            "=f"(_gate_0[5]), "=f"(_gate_0[6]), "=f"(_gate_0[7])
                            : "r"(_rd));
                    }
                    float _up_0[8];
                    {
                        int _rd = _tmem_base + _tmem_row + _col + 128;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                            "[%8];\ntcgen05.wait::ld.sync.aligned;\n"
                            : "=f"(_up_0[0]), "=f"(_up_0[1]), "=f"(_up_0[2]), "=f"(_up_0[3]), "=f"(_up_0[4]),
                            "=f"(_up_0[5]), "=f"(_up_0[6]), "=f"(_up_0[7])
                            : "r"(_rd));
                    }
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _gate_0[_i] = _gate_0[_i] * _gate_decode;
                    }
// SiLU formula: tanh; policy: fast; _gate_0 = silu(_gate_0)
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        float _x = _gate_0[_i];
                        float _half = 0.5f * _x;
                        _gate_0[_i] = fmaf(_half, __tanhf(_half), _half);
                    }
                    float _out_0[8];
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _out_0[_i] = _gate_0[_i] * _up_0[_i];
                    }
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _out_0[_i] = _out_0[_i] * _up_encode;
                    }
                    __nv_fp8_storage_t _smem_q_0_0_0 = __nv_cvt_float_to_fp8(_out_0[0], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_1 = __nv_cvt_float_to_fp8(_out_0[1], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_2 = __nv_cvt_float_to_fp8(_out_0[2], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_3 = __nv_cvt_float_to_fp8(_out_0[3], __NV_SATFINITE, __NV_E4M3);
                    uint32_t _smem_p_0_0 = ((uint32_t) _smem_q_0_0_0 << 0) | ((uint32_t) _smem_q_0_0_1 << 8)
                        | ((uint32_t) _smem_q_0_0_2 << 16) | ((uint32_t) _smem_q_0_0_3 << 24);
                    __nv_fp8_storage_t _smem_q_0_1_0 = __nv_cvt_float_to_fp8(_out_0[4], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_1 = __nv_cvt_float_to_fp8(_out_0[5], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_2 = __nv_cvt_float_to_fp8(_out_0[6], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_3 = __nv_cvt_float_to_fp8(_out_0[7], __NV_SATFINITE, __NV_E4M3);
                    uint32_t _smem_p_0_1 = ((uint32_t) _smem_q_0_1_0 << 0) | ((uint32_t) _smem_q_0_1_1 << 8)
                        | ((uint32_t) _smem_q_0_1_2 << 16) | ((uint32_t) _smem_q_0_1_3 << 24);
                    uint32_t _smem_abs_0 = (uint32_t) __cvta_generic_to_shared((char*) _swiglu_c_smem_0)
                        + (uint32_t) (_smem_row * 128 + ((0) + _col + 0) * 1);
                    uint32_t _smem_swz_0 = _smem_abs_0 ^ (((_smem_abs_0 >> 7) & 0x7u) << 4);
                    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
                        :
                        : "r"(_smem_swz_0), "r"(_smem_p_0_0), "r"(_smem_p_0_1)
                        : "memory");
                }
            }
            // TmemTileIter: acc → _swiglu_c_smem_1 (128 cols, pack=8, cast=fp8e4m3)
            {
                int _warp_id = threadIdx.x / 32;
                int _lane_id = threadIdx.x % 32;
                int _warp_row_base = (_warp_id % 4) * 32;
                int _smem_row = _warp_row_base + _lane_id;
                int _tmem_row = (_cta_rank * 128 + _warp_row_base + _lane_id) << 16;
                asm volatile("bar.sync %0, %1;\n" : : "r"(1), "r"(128) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                int _tmem_base = _tmem_acc + 256;
#pragma unroll
                for (int _col = 0; _col < 128; _col += 8)
                {
                    float _gate_1[8];
                    {
                        int _rd = _tmem_base + _tmem_row + _col;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                            "[%8];\ntcgen05.wait::ld.sync.aligned;\n"
                            : "=f"(_gate_1[0]), "=f"(_gate_1[1]), "=f"(_gate_1[2]), "=f"(_gate_1[3]), "=f"(_gate_1[4]),
                            "=f"(_gate_1[5]), "=f"(_gate_1[6]), "=f"(_gate_1[7])
                            : "r"(_rd));
                    }
                    float _up_1[8];
                    {
                        int _rd = _tmem_base + _tmem_row + _col + 128;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                            "[%8];\ntcgen05.wait::ld.sync.aligned;\n"
                            : "=f"(_up_1[0]), "=f"(_up_1[1]), "=f"(_up_1[2]), "=f"(_up_1[3]), "=f"(_up_1[4]),
                            "=f"(_up_1[5]), "=f"(_up_1[6]), "=f"(_up_1[7])
                            : "r"(_rd));
                    }
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _gate_1[_i] = _gate_1[_i] * _gate_decode;
                    }
// SiLU formula: tanh; policy: fast; _gate_1 = silu(_gate_1)
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        float _x = _gate_1[_i];
                        float _half = 0.5f * _x;
                        _gate_1[_i] = fmaf(_half, __tanhf(_half), _half);
                    }
                    float _out_1[8];
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _out_1[_i] = _gate_1[_i] * _up_1[_i];
                    }
#pragma unroll
                    for (int _i = 0; _i < 8; _i += 1)
                    {
                        _out_1[_i] = _out_1[_i] * _up_encode;
                    }
                    __nv_fp8_storage_t _smem_q_0_0_0 = __nv_cvt_float_to_fp8(_out_1[0], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_1 = __nv_cvt_float_to_fp8(_out_1[1], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_2 = __nv_cvt_float_to_fp8(_out_1[2], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_0_3 = __nv_cvt_float_to_fp8(_out_1[3], __NV_SATFINITE, __NV_E4M3);
                    uint32_t _smem_p_0_0 = ((uint32_t) _smem_q_0_0_0 << 0) | ((uint32_t) _smem_q_0_0_1 << 8)
                        | ((uint32_t) _smem_q_0_0_2 << 16) | ((uint32_t) _smem_q_0_0_3 << 24);
                    __nv_fp8_storage_t _smem_q_0_1_0 = __nv_cvt_float_to_fp8(_out_1[4], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_1 = __nv_cvt_float_to_fp8(_out_1[5], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_2 = __nv_cvt_float_to_fp8(_out_1[6], __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_storage_t _smem_q_0_1_3 = __nv_cvt_float_to_fp8(_out_1[7], __NV_SATFINITE, __NV_E4M3);
                    uint32_t _smem_p_0_1 = ((uint32_t) _smem_q_0_1_0 << 0) | ((uint32_t) _smem_q_0_1_1 << 8)
                        | ((uint32_t) _smem_q_0_1_2 << 16) | ((uint32_t) _smem_q_0_1_3 << 24);
                    uint32_t _smem_abs_0 = (uint32_t) __cvta_generic_to_shared((char*) _swiglu_c_smem_1)
                        + (uint32_t) (_smem_row * 128 + ((0) + _col + 0) * 1);
                    uint32_t _smem_swz_0 = _smem_abs_0 ^ (((_smem_abs_0 >> 7) & 0x7u) << 4);
                    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
                        :
                        : "r"(_smem_swz_0), "r"(_smem_p_0_0), "r"(_smem_p_0_1)
                        : "memory");
                }
            }
            asm volatile(
                "{\n"
                ".reg .u32 _bar_remote_44;\n"
                "mapa.shared::cluster.u32 _bar_remote_44, %0, 0;\n"
                "mbarrier.arrive.shared::cluster.b64 _, [_bar_remote_44];\n"
                "}\n"
                :
                : "r"((_kl_mbar_arena_base + 56u))
                : "memory");
            // async source publish: _swiglu_c_smem_0 -> swiglu_tma_store (named barrier 1+0/128)
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("bar.sync %0, %1;\n" : : "r"((1 + (0))), "r"(128) : "memory");
            // async source publish: _swiglu_c_smem_1 -> swiglu_tma_store (named barrier 2+0/128)
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("bar.sync %0, %1;\n" : : "r"((2 + (0))), "r"(128) : "memory");
            if (((threadIdx.x / 32) == 2))
            {
                // TMA store: smem -> tma_C
                if (_kd_warp_idx == 2 && kl::elect_one())
                {
                    uint32_t _smem = (uint32_t) __cvta_generic_to_shared(_swiglu_c_smem_0);
                    asm volatile(
                        "{\n"
                        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];\n"
                        "}\n"
                        :
                        : "l"(&tma_C), "r"((uint32_t) (0)), "r"((uint32_t) (((_bid_m * 256) + (_cta_rank * 128)))),
                        "r"((uint32_t) (((_bid_n * 256)) / 128)), "r"(_smem)
                        : "memory");
                }
                // TMA store: smem -> tma_C
                if (_kd_warp_idx == 2 && kl::elect_one())
                {
                    uint32_t _smem = (uint32_t) __cvta_generic_to_shared(_swiglu_c_smem_1);
                    asm volatile(
                        "{\n"
                        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];\n"
                        "}\n"
                        :
                        : "l"(&tma_C), "r"((uint32_t) (0)), "r"((uint32_t) (((_bid_m * 256) + (_cta_rank * 128)))),
                        "r"((uint32_t) ((((_bid_n * 256) + 128)) / 128)), "r"(_smem)
                        : "memory");
                }
                asm volatile(
                    "{\n"
                    "cp.async.bulk.commit_group;\n"
                    "cp.async.bulk.wait_group.read 0;\n"
                    "}\n"
                    :
                    :
                    : "memory");
            }
            asm volatile("bar.sync %0, %1;\n" : : "r"((3 + (0))), "r"(128) : "memory");
            ++_accumulator_stage;
            if (_accumulator_stage == 1)
            {
                _accumulator_stage = 0;
                _phase_epilogue_mainloop ^= 1u;
            }
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    // tmem dealloc: acc
    __syncthreads();
    if (((threadIdx.x / 32) == 1))
    {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" : : "r"(_tmem_acc), "r"(512) : "memory");
    }
#else
    __trap();
#endif // ALPAMAYO_SM110_DEVICE
}

namespace
{
int tensorMapError(CUresult status, char const* tensor)
{
    char const* message = nullptr;
    CUresult const stringStatus = cuGetErrorString(status, &message);
    LOG_ERROR("AlpamayoFusedGateUp: cuTensorMapEncodeTiled(%s) failed (%d): %s", tensor, static_cast<int>(status),
        stringStatus == CUDA_SUCCESS && message != nullptr ? message : "unknown CUDA driver error");
    return static_cast<int>(status);
}
} // namespace

extern "C" int initializeAlpamayoFusedGateUp()
{
    cudaError_t status = cudaFuncSetAttribute(alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM_BYTES + kSMEM_ALIGN);
    if (status != cudaSuccess)
    {
        LOG_ERROR("AlpamayoFusedGateUp: shared memory setup failed: %s", cudaGetErrorString(status));
        return static_cast<int>(status);
    }
    status = cudaFuncSetAttribute(
        alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (status != cudaSuccess)
    {
        LOG_ERROR("AlpamayoFusedGateUp: cluster setup failed: %s", cudaGetErrorString(status));
    }
    return static_cast<int>(status);
}

extern "C" int launch_alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh(
    AlpamayoFusedGateUpParams const& params, cudaStream_t stream)
{
    int const M = params.m;
    int const N = params.n;
    int const K = params.k;
    int const grid_n = params.gridN;
    int const grid_size = params.gridSize;
    CUtensorMap tma_A;
    {
        uint64_t globalDim[3] = {(uint64_t) 128, (uint64_t) M, (uint64_t) K / 128};
        uint64_t globalStrides[2] = {(uint64_t) K * sizeof(__nv_fp8_e4m3), 128};
        uint32_t boxDim[3] = {128, 128, 128 / 128};
        uint32_t elemStrides[3] = {1, 1, 1};
        CUresult _tma_status = cuTensorMapEncodeTiled(&tma_A, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
            const_cast<void*>(params.x), globalDim, globalStrides, boxDim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (_tma_status != CUDA_SUCCESS)
        {
            return tensorMapError(_tma_status, "A");
        }
    }
    CUtensorMap tma_B;
    {
        uint64_t globalDim[3] = {(uint64_t) 128, (uint64_t) N, (uint64_t) K / 128};
        uint64_t globalStrides[2] = {(uint64_t) K * sizeof(__nv_fp8_e4m3), 128};
        uint32_t boxDim[3] = {128, 128, 128 / 128};
        uint32_t elemStrides[3] = {1, 1, 1};
        CUresult _tma_status = cuTensorMapEncodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
            const_cast<void*>(params.w), globalDim, globalStrides, boxDim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (_tma_status != CUDA_SUCCESS)
        {
            return tensorMapError(_tma_status, "B");
        }
    }
    CUtensorMap tma_C;
    {
        uint64_t globalDim[3] = {(uint64_t) 128, (uint64_t) M, (uint64_t) N / 2 / 128};
        uint64_t globalStrides[2] = {(uint64_t) N / 2 * sizeof(__nv_fp8_e4m3), 128};
        uint32_t boxDim[3] = {128, 128, 128 / 128};
        uint32_t elemStrides[3] = {1, 1, 1};
        CUresult _tma_status = cuTensorMapEncodeTiled(&tma_C, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, params.out, globalDim,
            globalStrides, boxDim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (_tma_status != CUDA_SUCCESS)
        {
            return tensorMapError(_tma_status, "C");
        }
    }
    cudaLaunchConfig_t _config = {};
    _config.gridDim = dim3(grid_size, 1);
    _config.blockDim = dim3(192);
    _config.dynamicSmemBytes = kSMEM_BYTES + kSMEM_ALIGN;
    _config.stream = stream;
    cudaLaunchAttribute _attrs[1];
    _attrs[0].id = cudaLaunchAttributeClusterDimension;
    _attrs[0].val.clusterDim = {2, 1, 1};
    _config.attrs = _attrs;
    _config.numAttrs = 1;
    auto _err = cudaLaunchKernelEx(&_config, alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh, tma_A, tma_B, tma_C,
        params.gateDecode, params.upEncode, M, N, K, grid_n, grid_size);
    if (_err != cudaSuccess)
    {
        LOG_ERROR("AlpamayoFusedGateUp: kernel launch failed: %s", cudaGetErrorString(_err));
    }
    return static_cast<int>(_err);
}
