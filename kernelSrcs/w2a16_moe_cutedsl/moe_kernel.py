# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from b12x 2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88.
# See README.md for the supported IQ2_XXS/BF16 contract.

from w2a16_moe_cutedsl.intrinsics import atomic_add_global_i32, bf16_mma_m16n8k16_f32, bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32, bfloat2_mul, bfloat2_to_float2_scaled, broadcast_f32_to_bfloat2, cp_async4_shared_global, cp_async4_shared_global_pred, get_ptr_as_int64, iq2_xxs_descriptor_pair, ld_global_acquire_i32, ld_global_v4_f32, ld_shared_i32_relaxed, ld_shared_u32, ld_shared_v2_u32, ld_shared_v4_f32, ld_shared_v4_u32, ldmatrix_m8n8x2_b16, ldmatrix_m8n8x4_b16, pack_f32x2_to_bfloat2, packed_decode_iq2_xs_to_bfloat2x4, red_add_global_release_i32, shared_ptr_to_u32, st_global_i32, st_global_v4_f32, st_global_v4_u32, st_shared_bf16_from_f32, st_shared_i32, st_shared_u32, st_shared_v4_f32, threadfence
from w2a16_moe_cutedsl.codec import block_codec
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_PACK_FACTOR = 8
_STAGES = 4
_DEVICE_MAX_REG_BYTES = 256 * 1024
_DEFAULT_MAX_SHARED_MEM = 101376
_SCALAR_ACC_FRAGMENT_WIDTH = 1
_IQ2_LUT_BYTES = 2048


def _covering_count(total: int, quantum: int) -> int:
    return (total + quantum - 1) // quantum


def _w2a16_num_regs(*,
                    cta_threads: int,
                    cta_m_blocks: int,
                    cta_n_blocks: int,
                    cta_k_blocks: int,
                    uses_m_block_8: bool,
                    weight_layout: str = 'packed') -> int:
    if uses_m_block_8 and (cta_threads == 256 or
                           (cta_n_blocks == 4 and cta_k_blocks == 8)):
        return 128
    return 255


def _iq2_xs_stage_bytes(tile_k: int,
                        tile_n: int,
                        codec: str = 'iq2_xs') -> int:
    return tile_k * tile_n // block_codec(codec).pack_factor + _covering_count(
        tile_k,
        block_codec(
            codec).block_weights) * tile_n * 2 + tile_k // 32 * tile_n * (
                block_codec(codec).subscale_bytes // 8)


def _w2a16_pipeline_stages(*,
                           weight_layout: str,
                           tile_n: int,
                           tile_k: int,
                           uses_m_block_8: bool,
                           pipeline_stages: int | None = None,
                           scale_format: str = 'e4m3_k16',
                           small_m_occupancy: int | None = None) -> int:
    if pipeline_stages is not None:
        if type(pipeline_stages) is not int or pipeline_stages not in (2, 3, 4,
                                                                       5):
            raise ValueError('W2A16 pipeline_stages must be 2, 3, 4 or 5')
        return pipeline_stages
    if tile_n == 128 and tile_k == 128:
        return 2
    if uses_m_block_8 and tile_n == 64 and (tile_k == 128):
        return 3
    return _STAGES


def _reduction_shared_bytes(tile_n: int, tile_k: int,
                            uses_m_block_8: bool) -> int:
    """Extent of the FP32 stores in the CTA partial reduction.

    M8 uses four vec4 stores with a two-vector stride; larger M uses eight
    contiguous vec4 stores. The K128 first pass also uses a second row eight
    strides away. This is larger than the dense BF16 output tile and may
    extend into the dead activation staging region.
    It must never overlap a codec LUT appended after the GEMM's shared region.
    """
    reduction_rows = int(tile_k) // 64
    if reduction_rows == 0:
        return 0
    vectors = (7 if uses_m_block_8 else 8) + (8 if reduction_rows == 2 else 0)
    return int(tile_n) // 2 * vectors * 16


def _shared_memory_footprint(*,
                             cta_m_blocks: int,
                             tile_n: int,
                             tile_k: int,
                             scale_format: str = 'e4m3_k16',
                             weight_layout: str = 'packed',
                             weight_bits: int = 4,
                             uses_m_block_8: bool = False,
                             pipeline_stages: int | None = None,
                             small_m_occupancy: int | None = None,
                             csf_slab_rows: int = 128,
                             problem_k: int = 0) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    stages = _w2a16_pipeline_stages(weight_layout='iq2_xxs',
                                    tile_n=cta_n,
                                    tile_k=cta_k,
                                    uses_m_block_8=uses_m_block_8,
                                    pipeline_stages=pipeline_stages,
                                    scale_format='iq2_xxs',
                                    small_m_occupancy=None)
    activation_rows = 8 if uses_m_block_8 and stages == 3 else cta_m
    sh_block_meta_size = activation_rows * 16
    sh_a_size = stages * (activation_rows * cta_k) * 2
    staged_weight_bits = int(weight_bits)
    sh_b_size = stages * (cta_k * cta_n * staged_weight_bits // 8)
    sh_b_size = stages * _iq2_xs_stage_bytes(cta_k, cta_n, 'iq2_xxs')
    sh_red_size = cta_m * (cta_n + 8) * 2
    sh_bias_size = cta_n * 2
    tmp_size = min(sh_b_size, sh_red_size) + sh_bias_size
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size)
    sh_s_size = 0
    lut_size = block_codec('iq2_xxs').lut_bytes(selectors=True)
    gemm_size = tmp_size + sh_a_size + sh_s_size + sh_block_meta_size
    route_rows = 8 if uses_m_block_8 else cta_m
    gemm_size = max(
        gemm_size, route_rows * 16 +
        _reduction_shared_bytes(cta_n, cta_k, uses_m_block_8))
    return gemm_size + lut_size


def _determine_blocks_per_sm(*,
                             cta_threads: int,
                             cta_m_blocks: int,
                             tile_n: int,
                             tile_k: int,
                             uses_m_block_8: bool,
                             max_shared_mem: int,
                             scale_format: str = 'e4m3_k16',
                             weight_layout: str = 'packed',
                             weight_bits: int = 4,
                             pipeline_stages: int | None = None,
                             small_m_occupancy: int | None = None,
                             problem_k: int = 0) -> int:
    num_regs = _w2a16_num_regs(cta_threads=cta_threads,
                               cta_m_blocks=cta_m_blocks,
                               cta_n_blocks=tile_n // 16,
                               cta_k_blocks=tile_k // 16,
                               uses_m_block_8=uses_m_block_8,
                               weight_layout='iq2_xxs')
    register_bytes = max(num_regs, 1) * int(cta_threads) * 4
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_format='iq2_xxs',
        weight_layout='iq2_xxs',
        weight_bits=weight_bits,
        uses_m_block_8=uses_m_block_8,
        pipeline_stages=pipeline_stages,
        small_m_occupancy=None,
        problem_k=problem_k)
    blocks_per_sm_limit = min(_DEVICE_MAX_REG_BYTES // register_bytes,
                              int(max_shared_mem) // (smem_bytes + 1536))
    if uses_m_block_8:
        block_limit = 4 if tile_n == 64 and tile_k == 128 else 2
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, block_limit), 1)
    elif cta_m_blocks == 1:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 4), 1)
    else:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 2), 1)
    return int(blocks_per_sm_limit)


class _Iq2GemmKernel:

    def __init__(self,
                 *,
                 num_experts: int,
                 top_k: int,
                 mul_topk_weights: bool,
                 tile_n: int,
                 tile_k: int,
                 moe_block_size: int,
                 max_m_blocks: int,
                 epilogue_activation: str | None = None,
                 single_token_route_fast_path: bool = False,
                 direct_topk_routes: bool = False,
                 fused_sum_topk: int = 1,
                 dynamic_num_experts: bool = False,
                 schedule_route_block_factor: int = 1,
                 pipeline_stages: int | None = None,
                 skip_empty_m_blocks: bool | None = None):
        if epilogue_activation not in (None, 'relu2'):
            raise ValueError(
                'W2A16 GEMM epilogue activation currently supports only relu2')
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError('tile_n/tile_k must be multiples of 16')
        if tile_k > 128:
            raise ValueError('IQ2_XXS warp reduction requires tile_k <= 128')
        scale_group_size = _scale_group_size('iq2_xxs')
        if moe_block_size not in _ALLOWED_ROUTED_SIZES:
            raise ValueError(f'unsupported moe_block_size {moe_block_size}')
        if moe_block_size != 8 and moe_block_size % 16 != 0:
            raise ValueError('moe_block_size must be 8 or a multiple of 16')
        cta_threads = tile_n * tile_k // 64
        if cta_threads not in (128, 256):
            raise ValueError('W2A16 GEMM expects 128 or 256 CTA threads')
        self.dynamic_num_experts = bool(dynamic_num_experts)
        self.top_k = int(top_k)
        self.mul_topk_weights = bool(mul_topk_weights)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.cta_n_blocks = int(tile_n // 16)
        self.cta_k_blocks = int(tile_k // 16)
        self.cta_threads = int(cta_threads)
        self.moe_block_size = int(moe_block_size)
        self.epilogue_relu2 = epilogue_activation == 'relu2'
        self.skip_empty_m_blocks = True
        self.small_m_splitk = moe_block_size == 8
        self.block_pack_factor = block_codec('iq2_xxs').pack_factor
        self.block_k = block_codec('iq2_xxs').block_weights
        self.block_metadata_bytes = block_codec('iq2_xxs').metadata_bytes
        self.block_tile_bytes = 256 // self.block_pack_factor
        self.iq2_xs_smem_lut = False
        self.b_bundle_rows = 2
        self.single_token_route_fast_path = bool(single_token_route_fast_path)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.schedule_route_block_factor = int(schedule_route_block_factor)
        if self.schedule_route_block_factor < 1:
            raise ValueError('schedule_route_block_factor must be >= 1')
        if self.schedule_route_block_factor != 1:
            raise ValueError(
                'grouped route-block scheduling requires route-packed whole-tile execution'
            )
        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))
        self.uses_m_block_8 = moe_block_size == 8
        self.stages = _w2a16_pipeline_stages(
            weight_layout='iq2_xxs',
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            pipeline_stages=pipeline_stages,
            scale_format='iq2_xxs',
            small_m_occupancy=None)
        self.sms = 48
        max_shared_mem = 101376
        self.blocks_per_sm = _determine_blocks_per_sm(
            cta_threads=self.cta_threads,
            cta_m_blocks=self.cta_m_blocks,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            max_shared_mem=max_shared_mem,
            scale_format='iq2_xxs',
            weight_layout='iq2_xxs',
            weight_bits=4,
            pipeline_stages=self.stages,
            small_m_occupancy=None)
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (8 if self.uses_m_block_8
                                              and self.stages == 3 else 16 *
                                              self.cta_m_blocks)
        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8
        self.a_sh_wr_delta = self.a_sh_stride * (self.cta_threads //
                                                 self.a_gl_rd_delta_o)
        self.a_sh_wr_iters = _covering_count(self.a_sh_stage,
                                             self.a_sh_wr_delta)
        self.a_sh_rd_delta_i = self.a_sh_stride * 16
        self.b_sh_stride = self.cta_n_blocks * 16 * 16 // _PACK_FACTOR // 4
        self.b_sh_stride_threads = self.b_sh_stride
        self.b_sh_stage = self.b_sh_stride * self.cta_k_blocks
        self.b_sh_wr_iters = self.b_sh_stage // self.cta_threads
        self.b_unit_bytes = 16
        self.b_sh_stage_bytes = self.b_sh_stage * self.b_unit_bytes
        self.b_sh_stage_bytes = _iq2_xs_stage_bytes(self.tile_k, self.tile_n,
                                                    'iq2_xxs')
        if self.b_sh_stage_bytes % 16 != 0:
            raise ValueError(
                f'IQ2 B stage bytes must be a multiple of 16; got {self.b_sh_stage_bytes}'
            )
        self.s_sh_stage = 0
        self.tb_n_warps = self.cta_n_blocks // 4
        sh_block_route_indices = self.moe_block_size // 4
        sh_rd_block_route_indices = self.moe_block_size // 4
        sh_block_topk_weights = self.moe_block_size // 2
        self.sh_valid_count_off = sh_block_route_indices + sh_rd_block_route_indices + sh_block_topk_weights
        self.sh_route_off = 0
        self.sh_rd_route_off = sh_block_route_indices
        self.sh_topk_off = sh_block_route_indices + sh_rd_block_route_indices
        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        sh_b_size = _covering_count(self.stages * self.b_sh_stage_bytes, 16)
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + self.stages * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + self.stages * self.a_sh_stage
        self.shared_int4 = max(
            self.shared_int4, self.sh_red_off + _reduction_shared_bytes(
                self.tile_n, self.tile_k, self.uses_m_block_8) // 16)
        self.shared_words = self.shared_int4 * 4
        if self.shared_words * 4 > int(max_shared_mem):
            raise ValueError(
                f"W2A16 shared-memory footprint exceeds device opt-in limit: {self.shared_words * 4} > {int(max_shared_mem)} bytes (layout={'iq2_xxs'})"
            )

    @cute.jit
    def _activation_smem_permuted_offset(self, i: Int32) -> Int32:
        row = i // Int32(self.a_gl_rd_delta_o)
        if cutlass.const_expr(self.a_gl_rd_delta_o < 8):
            return Int32(self.a_gl_rd_delta_o) * row + (
                i - row * Int32(self.a_gl_rd_delta_o)
                ^ row >> Int32(1) & Int32(self.a_gl_rd_delta_o - 1))
        return Int32(self.a_gl_rd_delta_o) * row + (
            i - row * Int32(self.a_gl_rd_delta_o) ^ row & Int32(7))

    @cute.jit
    def _int4_addr(self, smem_base: Int32, int4_off: Int32) -> Int32:
        return smem_base + int4_off * Int32(16)

    @cute.jit
    def _iq2_xs_descriptor_offset(self, problem_shape, expert_idx: Int32,
                                  global_k16: Int32, global_n16: Int32):
        size_n, size_k = problem_shape
        tile = (Int64(expert_idx) * Int64(size_n // 64) +
                Int64(global_n16 // Int32(4))) * Int64(size_k // 128) + Int64(
                    global_k16 // Int32(8))
        return tile * Int64(32 * self.block_tile_bytes // 4) + (
            Int64(global_k16 % Int32(8)) * Int64(4) +
            Int64(global_n16 % Int32(4))) * Int64(self.block_tile_bytes // 4)

    @cute.jit
    def _elem2_mul(self, a: Uint32, b: Uint32) -> Uint32:
        return bfloat2_mul(a, b)

    @cute.jit
    def _broadcast_f32_to_elem2(self, x: cutlass.Float32) -> Uint32:
        return broadcast_f32_to_bfloat2(x)

    @cute.jit
    def _pack_f32x2_to_elem2(self, x0: cutlass.Float32,
                             x1: cutlass.Float32) -> Uint32:
        return pack_f32x2_to_bfloat2(x0, x1)

    @cute.jit
    def _elem2_to_f32x2(self, packed: Uint32):
        return bfloat2_to_float2_scaled(packed, cutlass.Float32(1.0))

    @cute.jit
    def _relu2_elem2(self, packed: Uint32) -> Uint32:
        x0, x1 = self._elem2_to_f32x2(packed)
        if x0 < cutlass.Float32(0.0):
            x0 = cutlass.Float32(0.0)
        if x1 < cutlass.Float32(0.0):
            x1 = cutlass.Float32(0.0)
        return self._pack_f32x2_to_elem2(x0 * x0, x1 * x1)

    @cute.jit
    def _st_shared_elem_from_f32(self, addr: Int32, val: cutlass.Float32):
        st_shared_bf16_from_f32(addr, val)

    @cute.jit
    def _mma_m16n8k16_f32(self, d0: cutlass.Float32, d1: cutlass.Float32,
                          d2: cutlass.Float32, d3: cutlass.Float32, a0: Uint32,
                          a1: Uint32, a2: Uint32, a3: Uint32, b0: Uint32,
                          b1: Uint32):
        return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    @cute.jit
    def _mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            self, d0: cutlass.Float32, d1: cutlass.Float32,
            d2: cutlass.Float32, d3: cutlass.Float32, b0_0: Uint32,
            b1_0: Uint32, b0_1: Uint32, b1_1: Uint32, a0: Uint32, a1: Uint32):
        return bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1)

    @cute.jit
    def _run_persistent_gemm(self,
                             problem_shape,
                             a_bf16_flat: cute.Tensor,
                             b_i32_flat: cute.Tensor,
                             c_bf16_flat: cute.Tensor,
                             scales_i32_flat: cute.Tensor,
                             global_scale: cute.Tensor,
                             packed_route_indices: cute.Tensor,
                             block_expert_ids: cute.Tensor,
                             packed_route_count: cute.Tensor,
                             topk_weights_flat: cute.Tensor,
                             c_tmp_f32_flat: cute.Tensor,
                             locks_i32_flat: cute.Tensor,
                             iq2_lut_addr: Int64,
                             smem_base: Int32,
                             tid: Int32,
                             cta: Int32,
                             grid_x: Int32,
                             active_size_m: Int32,
                             emit_tile: cutlass.Constexpr = None):
        size_n, size_k = problem_shape
        n_tiles = Int32(size_n // self.tile_n)
        route_blocks = active_size_m * Int32(self.top_k)
        if cutlass.const_expr(not self.direct_topk_routes):
            route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(
                self.moe_block_size * self.schedule_route_block_factor)
        k_tiles = Int32(size_k // self.tile_k)
        global_mn_tiles = route_blocks * n_tiles
        tail_mn_tiles = global_mn_tiles
        full_grid_mn_iters = Int32(0)
        force_one_tile_per_cta = Int32(0)
        if cutlass.const_expr(self.uses_m_block_8
                              and (not self.small_m_splitk)):
            if global_mn_tiles <= grid_x:
                force_one_tile_per_cta = Int32(1)
        whole_k_waves = Int32(0)
        if cutlass.const_expr(self.direct_topk_routes):
            if global_mn_tiles >= grid_x:
                whole_k_waves = Int32(1)
        if whole_k_waves != Int32(0):
            tail_mn_tiles = Int32(0)
            full_grid_mn_iters = (global_mn_tiles + grid_x -
                                  Int32(1)) // grid_x
        elif force_one_tile_per_cta != Int32(0):
            tail_mn_tiles = Int32(0)
            full_grid_mn_iters = Int32(1)
        elif global_mn_tiles > grid_x:
            tail_mn_tiles = global_mn_tiles - global_mn_tiles // grid_x * grid_x
            if tail_mn_tiles * Int32(3) <= grid_x:
                tail_mn_tiles += grid_x
            full_grid_mn_iters = (global_mn_tiles - tail_mn_tiles) // grid_x
        iters = (k_tiles * tail_mn_tiles + grid_x - Int32(1)) // grid_x
        lock_slot = Int32(0)
        if tail_mn_tiles >= grid_x:
            lock_slot = cta
        else:
            lock_slot = iters * cta // k_tiles - Int32(1)
        in_tail_region = Int32(0)
        has_work = Int32(1)
        work_mn_tile = cta
        reduce_k_tile = Int32(0)
        route_block_idx = Int32(0)
        output_n_tile = Int32(0)
        if iters == Int32(0) and full_grid_mn_iters == Int32(0):
            has_work = Int32(0)
        while has_work != Int32(0):
            reduce_tile_count = Int32(0)
            reduce_slice_count = Int32(1)
            reduce_slice_idx = Int32(0)
            if in_tail_region == Int32(0) and full_grid_mn_iters > Int32(0):
                route_block_idx = work_mn_tile // n_tiles
                output_n_tile = work_mn_tile - route_block_idx * n_tiles
                reduce_k_tile = Int32(0)
                reduce_tile_count = k_tiles
                full_grid_mn_iters -= Int32(1)
            else:
                if in_tail_region == Int32(0):
                    in_tail_region = Int32(1)
                    tail_mn_base = global_mn_tiles - tail_mn_tiles
                    cta_iter_start = iters * cta
                    work_mn_tile = cta_iter_start // k_tiles
                    reduce_k_tile = cta_iter_start - work_mn_tile * k_tiles
                    global_mn_tile = work_mn_tile + tail_mn_base
                    route_block_idx = global_mn_tile // n_tiles
                    output_n_tile = global_mn_tile - route_block_idx * n_tiles
                if work_mn_tile < tail_mn_tiles and iters > Int32(0):
                    reduce_tile_count = iters * (cta + Int32(1)) - (
                        k_tiles * work_mn_tile + reduce_k_tile)
                    if reduce_tile_count < Int32(0):
                        reduce_tile_count = Int32(0)
                    if reduce_k_tile + reduce_tile_count > k_tiles:
                        reduce_tile_count = k_tiles - reduce_k_tile
                    if reduce_tile_count > Int32(0):
                        first_reduce_boundary = iters * (
                            (k_tiles * work_mn_tile + iters - Int32(1)) //
                            iters)
                        if first_reduce_boundary <= k_tiles * (work_mn_tile +
                                                               Int32(1)):
                            reduce_boundary_offset = first_reduce_boundary - k_tiles * work_mn_tile
                            reduce_slice_count = (k_tiles -
                                                  reduce_boundary_offset +
                                                  iters - Int32(1)) // iters
                            if reduce_boundary_offset > Int32(0):
                                reduce_slice_count += Int32(1)
                            reduce_boundary_delta = iters * cta - first_reduce_boundary
                            if reduce_boundary_delta < Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(
                                    1)
                            elif reduce_boundary_offset == Int32(
                                    0) and reduce_boundary_delta == Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(
                                    1)
                            else:
                                reduce_slice_idx = reduce_slice_count - Int32(
                                    1) - reduce_boundary_delta // iters
                                if reduce_boundary_offset > Int32(0):
                                    reduce_slice_idx -= Int32(1)
                        if tail_mn_tiles >= grid_x:
                            if reduce_slice_count > Int32(
                                    1
                            ) and reduce_slice_idx == reduce_slice_count - Int32(
                                    1):
                                lock_slot += Int32(1)
                        else:
                            lock_slot += Int32(1)
                    else:
                        has_work = Int32(0)
                else:
                    has_work = Int32(0)
            if has_work != Int32(0) and reduce_tile_count > Int32(0) and (
                    route_block_idx < route_blocks):
                if cutlass.const_expr(emit_tile is not None):
                    emit_tile(route_block_idx, output_n_tile, reduce_k_tile,
                              reduce_tile_count, reduce_slice_count,
                              reduce_slice_idx, lock_slot)
                else:
                    if cutlass.const_expr(self.direct_topk_routes):
                        expert_idx = packed_route_indices[route_block_idx].to(
                            Int32)
                    else:
                        expert_idx = block_expert_ids[route_block_idx].to(
                            Int32)
                    if expert_idx >= Int32(0):
                        self._run_tile(
                            problem_shape, a_bf16_flat, b_i32_flat, c_bf16_flat,
                            scales_i32_flat, global_scale, packed_route_indices,
                            topk_weights_flat, c_tmp_f32_flat, locks_i32_flat,
                            iq2_lut_addr, smem_base, tid, route_block_idx,
                            expert_idx, output_n_tile, reduce_k_tile,
                            reduce_tile_count, reduce_slice_count,
                            reduce_slice_idx, lock_slot, active_size_m)
            if has_work != Int32(0):
                if in_tail_region == Int32(0):
                    work_mn_tile += grid_x
                else:
                    reduce_k_tile = Int32(0)
                    work_mn_tile += Int32(1)
                    output_n_tile += Int32(1)
                    if output_n_tile == n_tiles:
                        output_n_tile = Int32(0)
                        route_block_idx += Int32(1)

    @cute.jit
    def _read_moe_block_data(self, packed_route_indices: cute.Tensor,
                             topk_weights_flat: cute.Tensor, smem_base: Int32,
                             tid: Int32, route_block_idx: Int32,
                             global_scale_f32: cutlass.Float32,
                             active_size_m: Int32) -> Int32:
        if cutlass.const_expr(self.direct_topk_routes):
            if tid == Int32(0):
                idx = route_block_idx
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16),
                              rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    topk = topk_weights_flat[idx].to(
                        cutlass.Float32) * global_scale_f32
                    st_shared_u32(smem_base + Int32(self.sh_topk_off * 16),
                                  self._broadcast_f32_to_elem2(topk))
            cute.arch.sync_threads()
            return Int32(1)
        if cutlass.const_expr(self.single_token_route_fast_path):
            if tid == Int32(0):
                idx = packed_route_indices[route_block_idx * Int32(
                    self.moe_block_size)].to(Int32)
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16),
                              rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    safe_idx = idx
                    if idx >= active_size_m * Int32(self.top_k):
                        safe_idx = Int32(0)
                    topk = topk_weights_flat[safe_idx].to(
                        cutlass.Float32) * global_scale_f32
                    st_shared_u32(smem_base + Int32(self.sh_topk_off * 16),
                                  self._broadcast_f32_to_elem2(topk))
            cute.arch.sync_threads()
            return Int32(1)
        route_indices_int4_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_route_off) + tid)
        route_indices_gmem = get_ptr_as_int64(
            packed_route_indices,
            route_block_idx * Int32(self.moe_block_size) + tid * Int32(4))
        cp_async4_shared_global_pred(route_indices_int4_addr,
                                     route_indices_gmem, (tid < Int32(
                                         self.moe_block_size // 4)).to(Int32))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if tid >= Int32(self.cta_threads - 32):
            size_per_thread = _covering_count(self.moe_block_size, 32)
            lane = tid - Int32(self.cta_threads - 32)
            local_count = Int32(0)
            for i in cutlass.range_constexpr(size_per_thread):
                j = lane * Int32(size_per_thread) + Int32(i)
                if j < Int32(self.moe_block_size):
                    idx = ld_shared_i32_relaxed(smem_base +
                                                Int32(self.sh_route_off * 16) +
                                                j * Int32(4))
                    if idx < active_size_m * Int32(self.top_k):
                        local_count += Int32(1)
            valid = cute.arch.warp_redux_sync(local_count, 'add')
            if lane == Int32(0):
                st_shared_i32(smem_base + Int32(self.sh_valid_count_off * 16),
                              valid)
        if tid < Int32(self.moe_block_size):
            idx = ld_shared_i32_relaxed(smem_base +
                                        Int32(self.sh_route_off * 16) +
                                        tid * Int32(4))
            rd_row = idx // Int32(self.top_k)
            st_shared_i32(
                smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                rd_row)
            if cutlass.const_expr(self.mul_topk_weights):
                safe_idx = idx
                if idx >= active_size_m * Int32(self.top_k):
                    safe_idx = Int32(0)
                topk = topk_weights_flat[safe_idx].to(
                    cutlass.Float32) * global_scale_f32
                packed_topk = self._broadcast_f32_to_elem2(topk)
                topk_word_addr = smem_base + Int32(
                    self.sh_topk_off * 16) + tid * Int32(4)
                st_shared_u32(topk_word_addr, packed_topk)
        cute.arch.sync_threads()
        valid_count = ld_shared_i32_relaxed(smem_base +
                                            Int32(self.sh_valid_count_off *
                                                  16))
        cute.arch.sync_threads()
        return valid_count

    @cute.jit
    def _run_tile(self, problem_shape, a_bf16_flat: cute.Tensor,
                  b_i32_flat: cute.Tensor, c_bf16_flat: cute.Tensor,
                  scales_i32_flat: cute.Tensor, global_scale: cute.Tensor,
                  packed_route_indices: cute.Tensor,
                  topk_weights_flat: cute.Tensor, c_tmp_f32_flat: cute.Tensor,
                  locks_i32_flat: cute.Tensor, iq2_lut_addr: Int64,
                  smem_base: Int32, tid: Int32, route_block_idx: Int32,
                  expert_idx: Int32, output_n_tile: Int32, reduce_k_tile: Int32,
                  reduce_tile_count: Int32, reduce_slice_count: Int32,
                  reduce_slice_idx: Int32, lock_slot: Int32,
                  active_size_m: Int32):
        self._run_tile_with_pair_override(
            problem_shape, a_bf16_flat, b_i32_flat, c_bf16_flat,
            scales_i32_flat, global_scale, packed_route_indices,
            topk_weights_flat, c_tmp_f32_flat, locks_i32_flat, iq2_lut_addr,
            smem_base, tid, route_block_idx, expert_idx, output_n_tile,
            reduce_k_tile, reduce_tile_count, reduce_slice_count,
            reduce_slice_idx, lock_slot, active_size_m)

    @cute.jit
    def _run_tile_with_pair_override(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, c_bf16_flat: cute.Tensor,
            scales_i32_flat: cute.Tensor, global_scale: cute.Tensor,
            packed_route_indices: cute.Tensor, topk_weights_flat: cute.Tensor,
            c_tmp_f32_flat: cute.Tensor, locks_i32_flat: cute.Tensor,
            iq2_lut_addr: Int64, smem_base: Int32, tid: Int32,
            route_block_idx: Int32, expert_idx: Int32, output_n_tile: Int32,
            reduce_k_tile: Int32, reduce_tile_count: Int32,
            reduce_slice_count: Int32, reduce_slice_idx: Int32,
            lock_slot: Int32, active_size_m: Int32):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(problem_shape, a_bf16_flat, b_i32_flat,
                              c_bf16_flat, scales_i32_flat, global_scale,
                              packed_route_indices, topk_weights_flat,
                              c_tmp_f32_flat, locks_i32_flat, iq2_lut_addr,
                              smem_base, tid, route_block_idx, expert_idx,
                              output_n_tile, reduce_k_tile, reduce_tile_count,
                              reduce_slice_count, reduce_slice_idx, lock_slot,
                              active_size_m)
        else:
            self._run_tile_large_m(problem_shape, a_bf16_flat, b_i32_flat,
                                   c_bf16_flat, scales_i32_flat, global_scale,
                                   packed_route_indices, topk_weights_flat,
                                   c_tmp_f32_flat, locks_i32_flat, iq2_lut_addr,
                                   smem_base, tid, route_block_idx, expert_idx,
                                   output_n_tile, reduce_k_tile,
                                   reduce_tile_count, reduce_slice_count,
                                   reduce_slice_idx, lock_slot, active_size_m)

    @cute.jit
    def _tile_common_prologue(self, problem_shape, global_scale: cute.Tensor,
                              packed_route_indices: cute.Tensor,
                              topk_weights_flat: cute.Tensor, smem_base: Int32,
                              tid: Int32, route_block_idx: Int32,
                              expert_idx: Int32, output_n_tile: Int32,
                              active_size_m: Int32):
        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)
        block_valid_rows = self._read_moe_block_data(
            packed_route_indices, topk_weights_flat, smem_base, tid,
            route_block_idx, global_scale_f32, active_size_m)
        a_gl_stride, b_gl_stride, s_gl_stride, scales_expert_off, b_gl_rd_base, a_gl_rd_row, a_gl_rd_col0, a_sh_wr, a_rows_per_iter, b_sh_rd, s_sh_rd = self._tile_stream_offsets(
            problem_shape, tid, expert_idx, output_n_tile)
        return (global_scale_f32, block_valid_rows, a_gl_stride, b_gl_stride,
                s_gl_stride, scales_expert_off, b_gl_rd_base, a_gl_rd_row,
                a_gl_rd_col0, a_sh_wr, a_rows_per_iter, b_sh_rd, s_sh_rd)

    @cute.jit
    def _tile_stream_offsets(self, problem_shape, tid: Int32, expert_idx: Int32,
                             output_n_tile: Int32):
        size_n, size_k = problem_shape
        a_gl_stride = Int32(size_k // 8)
        b_gl_stride = Int32(16 * size_n // (_PACK_FACTOR * 4))
        s_gl_stride = Int32(size_n // 16)
        scales_expert_stride = Int32(size_n * size_k // (16 * 16))
        b_expert_off = Int32(size_n * size_k // (_PACK_FACTOR * 4)) * expert_idx
        scales_expert_off = scales_expert_stride * expert_idx
        a_gl_rd_row = tid // Int32(self.a_gl_rd_delta_o)
        a_gl_rd_col0 = tid - a_gl_rd_row * Int32(self.a_gl_rd_delta_o)
        a_sh_wr = Int32(self.a_sh_stride) * (
            tid // Int32(self.a_gl_rd_delta_o)) + (tid - tid // Int32(
                self.a_gl_rd_delta_o) * Int32(self.a_gl_rd_delta_o))
        a_rows_per_iter = Int32(self.cta_threads // self.a_gl_rd_delta_o)
        if cutlass.const_expr(self.cta_threads <= self.b_sh_stride):
            b_gl_rd_base = tid
        else:
            b_gl_rd_base = b_gl_stride * (
                tid // Int32(self.b_sh_stride)) + tid % Int32(self.b_sh_stride)
        b_gl_rd_base += b_expert_off + Int32(self.b_sh_stride) * output_n_tile
        b_sh_rd = tid
        b_sh_rd += b_sh_rd // Int32(self.b_sh_stride) * Int32(
            self.b_sh_stride * (self.b_sh_wr_iters - 1))
        s_sh_rd = Int32(8) * (tid // Int32(32) % Int32(self.tb_n_warps)) + (
            tid & Int32(31)) // Int32(4)
        return (a_gl_stride, b_gl_stride, s_gl_stride, scales_expert_off,
                b_gl_rd_base, a_gl_rd_row, a_gl_rd_col0, a_sh_wr,
                a_rows_per_iter, b_sh_rd, s_sh_rd)

    @cute.jit
    def _a_shared_read_offset(self, tid: Int32,
                              lanes_per_row: cutlass.Constexpr[int]):
        a_sh_rd = Int32(self.a_sh_stride) * (
            (tid & Int32(31)) %
            Int32(lanes_per_row)) + (tid & Int32(31)) // Int32(lanes_per_row)
        a_sh_rd += Int32(2) * (tid // Int32(32) // Int32(
            self.tb_n_warps)) * Int32(self.b_sh_wr_iters)
        return a_sh_rd

    @cute.jit
    def _run_tile_m8(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, c_bf16_flat: cute.Tensor,
            scales_i32_flat: cute.Tensor, global_scale: cute.Tensor,
            packed_route_indices: cute.Tensor, topk_weights_flat: cute.Tensor,
            c_tmp_f32_flat: cute.Tensor, locks_i32_flat: cute.Tensor,
            iq2_lut_addr: Int64, smem_base: Int32, tid: Int32,
            route_block_idx: Int32, expert_idx: Int32, output_n_tile: Int32,
            reduce_k_tile: Int32, reduce_tile_count: Int32,
            reduce_slice_count: Int32, reduce_slice_idx: Int32,
            lock_slot: Int32, active_size_m: Int32):
        global_scale_f32, block_valid_rows, a_gl_stride, b_gl_stride, s_gl_stride, scales_expert_off, b_gl_rd_base, a_gl_rd_row, a_gl_rd_col0, a_sh_wr, a_rows_per_iter, b_sh_rd, s_sh_rd = self._tile_common_prologue(
            problem_shape, global_scale, packed_route_indices,
            topk_weights_flat, smem_base, tid, route_block_idx, expert_idx,
            output_n_tile, active_size_m)
        a_sh_rd = self._a_shared_read_offset(tid, 8)
        acc = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH, ),
                                  cutlass.Float32)
            for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc[frag].fill(0.0)
        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(problem_shape, a_bf16_flat, b_i32_flat,
                                     scales_i32_flat, smem_base, tid, k_tiles,
                                     reduce_k_tile, block_valid_rows,
                                     a_gl_stride, a_gl_rd_row, a_gl_rd_col0,
                                     a_sh_wr, a_rows_per_iter, output_n_tile,
                                     expert_idx)
        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle()
        a_regs_cur = cute.make_rmem_tensor((2, ), Uint32)
        a_regs_next = cute.make_rmem_tensor((2, ), Uint32)
        self._load_a_register_bundle(a_regs_cur, smem_base, a_sh_rd, Int32(0),
                                     Int32(0), True)
        self._run_mma_pipeline(problem_shape, a_bf16_flat, b_i32_flat,
                               scales_i32_flat, iq2_lut_addr, smem_base, tid,
                               acc, acc, acc, acc, b_scale_cur, b_scale_next,
                               a_regs_cur, a_regs_next, a_sh_rd, k_tiles,
                               reduce_k_tile, block_valid_rows, a_gl_stride,
                               a_gl_rd_row, a_gl_rd_col0, a_sh_wr,
                               a_rows_per_iter, output_n_tile, expert_idx, True)
        self._finish_tile(problem_shape, acc, acc, acc, acc, c_bf16_flat,
                          c_tmp_f32_flat, locks_i32_flat, smem_base, tid,
                          output_n_tile, block_valid_rows, global_scale_f32,
                          reduce_slice_count, reduce_slice_idx, lock_slot, True)

    @cute.jit
    def _run_tile_large_m(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, c_bf16_flat: cute.Tensor,
            scales_i32_flat: cute.Tensor, global_scale: cute.Tensor,
            packed_route_indices: cute.Tensor, topk_weights_flat: cute.Tensor,
            c_tmp_f32_flat: cute.Tensor, locks_i32_flat: cute.Tensor,
            iq2_lut_addr: Int64, smem_base: Int32, tid: Int32,
            route_block_idx: Int32, expert_idx: Int32, output_n_tile: Int32,
            reduce_k_tile: Int32, reduce_tile_count: Int32,
            reduce_slice_count: Int32, reduce_slice_idx: Int32,
            lock_slot: Int32, active_size_m: Int32):
        global_scale_f32, block_valid_rows, a_gl_stride, b_gl_stride, s_gl_stride, scales_expert_off, b_gl_rd_base, a_gl_rd_row, a_gl_rd_col0, a_sh_wr, a_rows_per_iter, b_sh_rd, s_sh_rd = self._tile_common_prologue(
            problem_shape, global_scale, packed_route_indices,
            topk_weights_flat, smem_base, tid, route_block_idx, expert_idx,
            output_n_tile, active_size_m)
        a_sh_rd = self._a_shared_read_offset(tid, 16)
        acc0 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH, ),
                                  cutlass.Float32)
            for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc0[frag].fill(0.0)
        acc1 = acc0
        acc2 = acc0
        acc3 = acc0
        if cutlass.const_expr(self.cta_m_blocks > 1):
            acc1 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH, ),
                                      cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 //
                                                _SCALAR_ACC_FRAGMENT_WIDTH):
                acc1[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 2):
            acc2 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH, ),
                                      cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 //
                                                _SCALAR_ACC_FRAGMENT_WIDTH):
                acc2[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 3):
            acc3 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH, ),
                                      cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 //
                                                _SCALAR_ACC_FRAGMENT_WIDTH):
                acc3[frag].fill(0.0)
        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(problem_shape, a_bf16_flat, b_i32_flat,
                                     scales_i32_flat, smem_base, tid, k_tiles,
                                     reduce_k_tile, block_valid_rows,
                                     a_gl_stride, a_gl_rd_row, a_gl_rd_col0,
                                     a_sh_wr, a_rows_per_iter, output_n_tile,
                                     expert_idx)
        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle()
        a_regs = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(a_regs, smem_base, a_sh_rd, Int32(0),
                                     Int32(0), False)
        self._run_mma_pipeline(
            problem_shape, a_bf16_flat, b_i32_flat, scales_i32_flat,
            iq2_lut_addr, smem_base, tid, acc0, acc1, acc2, acc3, b_scale_cur,
            b_scale_next, a_regs, a_regs_next, a_sh_rd, k_tiles, reduce_k_tile,
            block_valid_rows, a_gl_stride, a_gl_rd_row, a_gl_rd_col0, a_sh_wr,
            a_rows_per_iter, output_n_tile, expert_idx, False)
        self._finish_tile(problem_shape, acc0, acc1, acc2, acc3, c_bf16_flat,
                          c_tmp_f32_flat, locks_i32_flat, smem_base, tid,
                          output_n_tile, block_valid_rows, global_scale_f32,
                          reduce_slice_count, reduce_slice_idx, lock_slot,
                          False)

    @cute.jit
    def _run_mma_pipeline(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, scales_i32_flat: cute.Tensor,
            iq2_lut_addr: Int64, smem_base: Int32, tid: Int32, acc0, acc1, acc2,
            acc3, b_scale_cur: cute.Tensor, b_scale_next: cute.Tensor,
            a_regs_cur: cute.Tensor, a_regs_next: cute.Tensor, a_sh_rd: Int32,
            k_tiles: Int32, reduce_k_tile: Int32, block_valid_rows: Int32,
            a_gl_stride: Int32, a_gl_rd_row: Int32, a_gl_rd_col0: Int32,
            a_sh_wr: Int32, a_rows_per_iter: Int32, output_n_tile: Int32,
            expert_idx: Int32, uses_m_block_8: cutlass.Constexpr[bool]):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        live_m_blocks = (block_valid_rows + Int32(15)) // Int32(16)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range(self.stages,
                                      unroll=1,
                                      unroll_full=not True):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
                            self._load_next_fragment_bundle(
                                b_scale_next, a_regs_next, smem_base, a_sh_rd,
                                pipe, kk, tile_idx, k_tiles, uses_m_block_8)
                        self._prefetch_pipeline_step(
                            problem_shape, a_bf16_flat, b_i32_flat,
                            scales_i32_flat, smem_base, tid, pipe, kk, tile_idx,
                            k_tiles, reduce_k_tile, block_valid_rows,
                            a_gl_stride, a_gl_rd_row, a_gl_rd_col0, a_sh_wr,
                            a_rows_per_iter, output_n_tile, expert_idx)
                        self._dequant_and_accumulate_bundle(
                            acc0, acc1, acc2,
                            acc3, b_frag, a_regs_cur, tid, smem_base,
                            Int32(pipe), kk, iq2_lut_addr, uses_m_block_8,
                            live_m_blocks)
                        if cutlass.const_expr(kk + 1 == self.b_sh_wr_iters):
                            cute.arch.cp_async_wait_group(self.stages - 2)
                            cute.arch.sync_threads()
                            self._load_next_fragment_bundle(
                                b_scale_next, a_regs_next, smem_base, a_sh_rd,
                                pipe, kk, tile_idx, k_tiles, uses_m_block_8)
                        if cutlass.const_expr(uses_m_block_8):
                            self._copy_a_register_bundle(
                                a_regs_cur, a_regs_next, uses_m_block_8)
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next)
                        else:
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next)
                            self._copy_a_register_bundle(
                                a_regs_cur, a_regs_next, uses_m_block_8)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle()
                self._load_a_register_bundle(a_regs_cur, smem_base, a_sh_rd,
                                             Int32(0), Int32(0),
                                             uses_m_block_8)

    @cute.jit
    def _dequant_and_accumulate_bundle(
            self, acc0, acc1, acc2, acc3, b_frag: cute.Tensor,
            a_regs_cur: cute.Tensor, tid: Int32, smem_base: Int32, pipe: Int32,
            kk: cutlass.Constexpr[int], iq2_lut_addr: Int64,
            uses_m_block_8: cutlass.Constexpr[bool], live_m_blocks: Int32):
        for jj in cutlass.range_constexpr(4):
            q0, q1, base_pair, subscale_pair = self._load_iq2_xs_fragment(
                smem_base, tid, pipe, Int32(kk), jj)
            self._scaled_dequant_b_fragment_iq2_xs(b_frag, q0, q1, base_pair,
                                                   subscale_pair, iq2_lut_addr,
                                                   tid)
            if cutlass.const_expr(uses_m_block_8):
                self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            elif cutlass.const_expr(self.skip_empty_m_blocks
                                    and self.cta_m_blocks > 1):
                self._mma_accumulate_large_m(acc0, a_regs_cur, 0, jj, b_frag)
                if live_m_blocks > Int32(1):
                    self._mma_accumulate_large_m(acc1, a_regs_cur, 1, jj,
                                                 b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 2):
                    if live_m_blocks > Int32(2):
                        self._mma_accumulate_large_m(acc2, a_regs_cur, 2, jj,
                                                     b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 3):
                    if live_m_blocks > Int32(3):
                        self._mma_accumulate_large_m(acc3, a_regs_cur, 3, jj,
                                                     b_frag)
            else:
                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                    if cutlass.const_expr(mb == 0):
                        self._mma_accumulate_large_m(acc0, a_regs_cur, mb, jj,
                                                     b_frag)
                    elif cutlass.const_expr(mb == 1):
                        self._mma_accumulate_large_m(acc1, a_regs_cur, mb, jj,
                                                     b_frag)
                    elif cutlass.const_expr(mb == 2):
                        self._mma_accumulate_large_m(acc2, a_regs_cur, mb, jj,
                                                     b_frag)
                    else:
                        self._mma_accumulate_large_m(acc3, a_regs_cur, mb, jj,
                                                     b_frag)

    @cute.jit
    def _finish_tile(self, problem_shape, acc0, acc1, acc2, acc3,
                     c_bf16_flat: cute.Tensor, c_tmp_f32_flat: cute.Tensor,
                     locks_i32_flat: cute.Tensor, smem_base: Int32, tid: Int32,
                     output_n_tile: Int32, block_valid_rows: Int32,
                     global_scale_f32: cutlass.Float32,
                     reduce_slice_count: Int32, reduce_slice_idx: Int32,
                     lock_slot: Int32, uses_m_block_8: cutlass.Constexpr[bool]):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc0, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(acc0, acc1, acc2, acc3, smem_base,
                                            tid)
        if reduce_slice_count > Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._combine_iq2_xs_splitk_partials(acc0, c_tmp_f32_flat,
                                                     locks_i32_flat, lock_slot,
                                                     reduce_slice_idx,
                                                     reduce_slice_count, tid)
            else:
                self._wait_for_reduction_turn(locks_i32_flat, lock_slot,
                                              reduce_slice_idx, tid)
                self._combine_splitk_accumulators(acc0, acc1, acc2, acc3,
                                                  c_tmp_f32_flat,
                                                  block_valid_rows, lock_slot,
                                                  reduce_slice_idx,
                                                  reduce_slice_count, tid,
                                                  uses_m_block_8)
                self._publish_reduction_turn(
                    locks_i32_flat, lock_slot,
                    reduce_slice_idx == reduce_slice_count - Int32(1), tid)
        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(problem_shape, acc0, c_bf16_flat, smem_base,
                                    tid, output_n_tile, block_valid_rows,
                                    global_scale_f32)
            else:
                self._store_tile_large_m(problem_shape, acc0, acc1, acc2, acc3,
                                         c_bf16_flat, smem_base, tid,
                                         output_n_tile, block_valid_rows,
                                         global_scale_f32)

    @cute.jit
    def _combine_iq2_xs_splitk_partials(self, acc, c_tmp_f32_flat: cute.Tensor,
                                        locks_i32_flat: cute.Tensor,
                                        lock_slot: Int32,
                                        reduce_slice_idx: Int32,
                                        reduce_slice_count: Int32, tid: Int32):
        cta, _, _ = cute.arch.block_idx()
        active_threads = Int32(32 * self.tb_n_warps)
        tile_elements = Int64(8 * self.tile_n)
        lock_addr = get_ptr_as_int64(locks_i32_flat, Int64(lock_slot))
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    offset = Int64(cta) * tile_elements + (
                        Int64(active_threads) * Int64(jj) +
                        Int64(tid)) * Int64(4)
                    st_global_v4_f32(
                        get_ptr_as_int64(c_tmp_f32_flat, offset),
                        acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                            jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH])
            cute.arch.sync_threads()
            if tid == Int32(0):
                red_add_global_release_i32(lock_addr, Int32(1))
        else:
            if tid == Int32(0):
                arrived = Int32(-1)
                while arrived != reduce_slice_count - Int32(1):
                    arrived = ld_global_acquire_i32(lock_addr)
            cute.arch.sync_threads()
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    s0 = cutlass.Float32(0.0)
                    s1 = cutlass.Float32(0.0)
                    s2 = cutlass.Float32(0.0)
                    s3 = cutlass.Float32(0.0)
                    for part in cutlass.range(reduce_slice_count - Int32(1),
                                              unroll=1):
                        source_cta = Int64(cta) + Int64(reduce_slice_count -
                                                        Int32(1) - part)
                        offset = source_cta * tile_elements + (
                            Int64(active_threads) * Int64(jj) +
                            Int64(tid)) * Int64(4)
                        v0, v1, v2, v3 = ld_global_v4_f32(
                            get_ptr_as_int64(c_tmp_f32_flat, offset))
                        s0 = v0 + s0
                        s1 = v1 + s1
                        s2 = v2 + s2
                        s3 = v3 + s3
                    acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] += s0
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] += s1
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] += s2
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] += s3
            cute.arch.sync_threads()
            if tid == Int32(0):
                st_global_i32(lock_addr, Int32(0))

    @cute.jit
    def _wait_for_reduction_turn(self, locks_i32_flat: cute.Tensor,
                                 lock_slot: Int32, count: Int32, tid: Int32):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        if tid == Int32(0):
            state = Int32(-1)
            while state != count:
                state = ld_global_acquire_i32(lock_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_reduction_turn(self, locks_i32_flat: cute.Tensor,
                                lock_slot: Int32, reset, tid: Int32):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        cute.arch.sync_threads()
        if tid == Int32(0):
            if reset:
                st_global_i32(lock_addr, Int32(0))
            else:
                red_add_global_release_i32(lock_addr, Int32(1))

    @cute.jit
    def _merge_splitk_vec4(self, c_tmp_f32_flat: cute.Tensor, f32_off: Int32,
                           reduce_slice_idx: Int32, reduce_slice_count: Int32,
                           c0: cutlass.Float32, c1: cutlass.Float32,
                           c2: cutlass.Float32, c3: cutlass.Float32):
        if reduce_slice_idx != Int32(0):
            r0, r1, r2, r3 = ld_global_v4_f32(
                get_ptr_as_int64(c_tmp_f32_flat, f32_off))
            c0 = c0 + r0
            c1 = c1 + r1
            c2 = c2 + r2
            c3 = c3 + r3
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            st_global_v4_f32(get_ptr_as_int64(c_tmp_f32_flat, f32_off), c0, c1,
                             c2, c3)
        return (c0, c1, c2, c3)

    @cute.jit
    def _combine_splitk_accumulators(self, acc0, acc1, acc2, acc3,
                                     c_tmp_f32_flat: cute.Tensor,
                                     block_valid_rows: Int32, lock_slot: Int32,
                                     reduce_slice_idx: Int32,
                                     reduce_slice_count: Int32, tid: Int32,
                                     uses_m_block_8: cutlass.Constexpr[bool]):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32(self.cta_m_blocks * 16 * self.cta_n_blocks * 16 //
                            4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    acc0[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH], acc0[
                            (jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][(
                                jj * 4 + 1
                            ) % _SCALAR_ACC_FRAGMENT_WIDTH], acc0[(
                                jj * 4 + 2
                            ) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2)
                                % _SCALAR_ACC_FRAGMENT_WIDTH], acc0[
                                    (jj * 4 + 3)
                                    // _SCALAR_ACC_FRAGMENT_WIDTH][
                                        (jj * 4 + 3)
                                        %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = self._merge_splitk_slot(
                                            c_tmp_f32_flat,
                                            c_cur_offset, active_threads,
                                            Int32(k), tid, reduce_slice_idx,
                                            reduce_slice_count,
                                            acc0[jj * 4 //
                                                 _SCALAR_ACC_FRAGMENT_WIDTH]
                                            [jj * 4 %
                                             _SCALAR_ACC_FRAGMENT_WIDTH],
                                            acc0[(jj * 4 + 1) //
                                                 _SCALAR_ACC_FRAGMENT_WIDTH]
                                            [(jj * 4 + 1) %
                                             _SCALAR_ACC_FRAGMENT_WIDTH],
                                            acc0[(jj * 4 + 2) //
                                                 _SCALAR_ACC_FRAGMENT_WIDTH]
                                            [(jj * 4 + 2) %
                                             _SCALAR_ACC_FRAGMENT_WIDTH],
                                            acc0[(jj * 4 + 3) //
                                                 _SCALAR_ACC_FRAGMENT_WIDTH]
                                            [(jj * 4 + 3) %
                                             _SCALAR_ACC_FRAGMENT_WIDTH])
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                    if cutlass.const_expr(mb == 0):
                        self._combine_splitk_accumulator_block(
                            acc0, mb, c_tmp_f32_flat, c_cur_offset,
                            active_threads, block_valid_rows, lane_row, tid,
                            reduce_slice_idx, reduce_slice_count)
                    elif cutlass.const_expr(mb == 1):
                        self._combine_splitk_accumulator_block(
                            acc1, mb, c_tmp_f32_flat, c_cur_offset,
                            active_threads, block_valid_rows, lane_row, tid,
                            reduce_slice_idx, reduce_slice_count)
                    elif cutlass.const_expr(mb == 2):
                        self._combine_splitk_accumulator_block(
                            acc2, mb, c_tmp_f32_flat, c_cur_offset,
                            active_threads, block_valid_rows, lane_row, tid,
                            reduce_slice_idx, reduce_slice_count)
                    else:
                        self._combine_splitk_accumulator_block(
                            acc3, mb, c_tmp_f32_flat, c_cur_offset,
                            active_threads, block_valid_rows, lane_row, tid,
                            reduce_slice_idx, reduce_slice_count)

    @cute.jit
    def _combine_splitk_accumulator_block(
            self, acc, mb: cutlass.Constexpr[int], c_tmp_f32_flat: cute.Tensor,
            c_cur_offset: Int32, active_threads: Int32,
            block_valid_rows: Int32, lane_row: Int32, tid: Int32,
            reduce_slice_idx: Int32, reduce_slice_count: Int32):
        for flat_j in cutlass.range_constexpr(8):
            row_valid = Int32(mb * 16) + lane_row < block_valid_rows
            if row_valid:
                acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                    flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH], acc[(
                        flat_j * 4 + 1
                    ) // _SCALAR_ACC_FRAGMENT_WIDTH][(
                        flat_j * 4 + 1
                    ) % _SCALAR_ACC_FRAGMENT_WIDTH], acc[(
                        flat_j *
                        4 +
                        2
                    ) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2)
                        % _SCALAR_ACC_FRAGMENT_WIDTH], acc[(
                            flat_j *
                            4 +
                            3
                        ) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3)
                            %
                            _SCALAR_ACC_FRAGMENT_WIDTH] = self._merge_splitk_slot(
                                c_tmp_f32_flat, c_cur_offset, active_threads,
                                Int32(mb * 8 + flat_j), tid, reduce_slice_idx,
                                reduce_slice_count,
                                acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                                    flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                                acc[(flat_j * 4 + 1) //
                                    _SCALAR_ACC_FRAGMENT_WIDTH][
                                        (flat_j * 4 + 1) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH],
                                acc[(flat_j * 4 + 2) //
                                    _SCALAR_ACC_FRAGMENT_WIDTH][
                                        (flat_j * 4 + 2) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH],
                                acc[(flat_j * 4 + 3) //
                                    _SCALAR_ACC_FRAGMENT_WIDTH][
                                        (flat_j * 4 + 3) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH])

    @cute.jit
    def _merge_splitk_slot(self, c_tmp_f32_flat: cute.Tensor,
                           c_cur_offset: Int32, active_threads: Int32,
                           slot: Int32, tid: Int32, reduce_slice_idx: Int32,
                           reduce_slice_count: Int32, c0: cutlass.Float32,
                           c1: cutlass.Float32, c2: cutlass.Float32,
                           c3: cutlass.Float32):
        int4_off = c_cur_offset + active_threads * slot + tid
        return self._merge_splitk_vec4(c_tmp_f32_flat, int4_off * Int32(4),
                                       reduce_slice_idx, reduce_slice_count,
                                       c0, c1, c2, c3)

    @cute.jit
    def _load_a_registers_large_m(self, smem_base: Int32, a_sh_rd: Int32,
                                  pipe: Int32, kk: Int32, m_block: Int32):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off) + pipe * Int32(self.a_sh_stage) +
            self._activation_smem_permuted_offset(
                Int32(2) * kk + m_block * Int32(self.a_sh_rd_delta_i) +
                a_sh_rd))
        return ldmatrix_m8n8x4_b16(a_addr)

    @cute.jit
    def _load_a_registers_m8(self, smem_base: Int32, a_sh_rd: Int32,
                             pipe: Int32, kk: Int32):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off) + pipe * Int32(self.a_sh_stage) +
            self._activation_smem_permuted_offset(Int32(2) * kk + a_sh_rd))
        return ldmatrix_m8n8x2_b16(a_addr)

    @cute.jit
    def _load_a_registers_large_m_bundle(self, regs: cute.Tensor,
                                         smem_base: Int32, a_sh_rd: Int32,
                                         pipe: Int32, kk: Int32):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base, a_sh_rd, pipe, kk, Int32(mb))
            regs[mb, 0] = a0
            regs[mb, 1] = a1
            regs[mb, 2] = a2
            regs[mb, 3] = a3

    @cute.jit
    def _load_a_registers_m8_bundle(self, regs: cute.Tensor, smem_base: Int32,
                                    a_sh_rd: Int32, pipe: Int32, kk: Int32):
        a0, a1 = self._load_a_registers_m8(smem_base, a_sh_rd, pipe, kk)
        regs[0] = a0
        regs[1] = a1

    @cute.jit
    def _clear_a_register_bundle_large_m(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _clear_a_register_bundle_m8(self, regs: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            regs[reg] = Uint32(0)

    @cute.jit
    def _copy_a_register_bundle_large_m(self, dst: cute.Tensor,
                                        src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _copy_a_register_bundle_m8(self, dst: cute.Tensor, src: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            dst[reg] = src[reg]

    @cute.jit
    def _load_a_register_bundle(self, regs: cute.Tensor, smem_base: Int32,
                                a_sh_rd: Int32, pipe: Int32, kk: Int32,
                                uses_m_block_8: cutlass.Constexpr[bool]):
        if cutlass.const_expr(uses_m_block_8):
            self._load_a_registers_m8_bundle(regs, smem_base, a_sh_rd, pipe,
                                             kk)
        else:
            self._load_a_registers_large_m_bundle(regs, smem_base, a_sh_rd,
                                                  pipe, kk)

    @cute.jit
    def _clear_a_register_bundle(self, regs: cute.Tensor,
                                 uses_m_block_8: cutlass.Constexpr[bool]):
        if cutlass.const_expr(uses_m_block_8):
            self._clear_a_register_bundle_m8(regs)
        else:
            self._clear_a_register_bundle_large_m(regs)

    @cute.jit
    def _copy_a_register_bundle(self, dst: cute.Tensor, src: cute.Tensor,
                                uses_m_block_8: cutlass.Constexpr[bool]):
        if cutlass.const_expr(uses_m_block_8):
            self._copy_a_register_bundle_m8(dst, src)
        else:
            self._copy_a_register_bundle_large_m(dst, src)

    @cute.jit
    def _load_iq2_xs_fragment(self, smem_base: Int32, tid: Int32, pipe: Int32,
                              kk: Int32, jj: cutlass.Constexpr[int]):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        warp_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        tc_col = lane // Int32(4)
        b_region = smem_base + Int32(
            self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        base_region = b_region + Int32(
            self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes)
        local_n16 = Int32(4) * warp_n + Int32(jj)
        base_addr = base_region + (
            (kt_local // Int32(self.block_k // 16) * Int32(self.cta_n_blocks) +
             local_n16) * Int32(8) + tc_col) * Int32(4)
        base_pair = ld_shared_u32(base_addr)
        tile_base = (kt_local // Int32(2) * Int32(2) * Int32(self.cta_n_blocks)
                     + local_n16) * Int32(16)
        record_addr = b_region + (tile_base + tc_col * Int32(2)) * Int32(4)
        grid0, grid1 = ld_shared_v2_u32(record_addr)
        signs0, signs1 = ld_shared_v2_u32(record_addr +
                                          Int32(self.cta_n_blocks *
                                                self.block_tile_bytes))
        q0 = iq2_xxs_descriptor_pair(grid0, signs0, kt_local % Int32(2))
        q1 = iq2_xxs_descriptor_pair(grid1, signs1, kt_local % Int32(2))
        subscale_pair = signs0 >> 28 | signs1 >> 28 << 8
        return (q0, q1, base_pair, subscale_pair)

    @cute.jit
    def _load_b_scale_register_bundle(self):
        return

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
        for row in cutlass.range_constexpr(self.b_bundle_rows):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor,
                                      src: cute.Tensor):
        for row in cutlass.range_constexpr(self.b_bundle_rows):
            for col in cutlass.range_constexpr(4):
                dst[row, col] = src[row, col]

    @cute.jit
    def _load_next_fragment_bundle(self, b_scale_next: cute.Tensor,
                                   a_regs_next: cute.Tensor, smem_base: Int32,
                                   a_sh_rd: Int32, pipe: Int32,
                                   kk: cutlass.Constexpr[int], tile_idx: Int32,
                                   k_tiles: Int32,
                                   uses_m_block_8: cutlass.Constexpr[bool]):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle(a_regs_next, uses_m_block_8)
        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle()
                self._load_a_register_bundle(a_regs_next, smem_base, a_sh_rd,
                                             Int32(pipe), Int32(kk + 1),
                                             uses_m_block_8)
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_b_scale_register_bundle()
                self._load_a_register_bundle(a_regs_next, smem_base, a_sh_rd,
                                             Int32((pipe + 1) % self.stages),
                                             Int32(0), uses_m_block_8)

    @cute.jit
    def _scaled_dequant_b_fragment_iq2_xs(self, frag: cute.Tensor,
                                          q_row0: Uint32, q_row1: Uint32,
                                          base_pair: Uint32,
                                          subscale_pair: Uint32,
                                          execution_lut_addr: Int64,
                                          tid: Int32):
        pair_byte_offset = (tid & Int32(3)) * Int32(2)
        b0_0, b0_1, b1_0, b1_1 = packed_decode_iq2_xs_to_bfloat2x4(
            q_row0,
            q_row1,
            base_pair,
            subscale_pair,
            execution_lut_addr,
            pair_byte_offset,
            shared_lut=self.iq2_xs_smem_lut,
            selector_lut=True)
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _mma_accumulate_m8(self, acc, jj: cutlass.Constexpr[int],
                           a_regs: cute.Tensor, b_frag: cute.Tensor):
        d0, d1, d2, d3 = self._mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            acc[jj * 4 //
                _SCALAR_ACC_FRAGMENT_WIDTH][jj * 4 %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 4 + 1) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 1) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 4 + 2) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 2) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 4 + 3) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 3) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            b_frag[0, 0], b_frag[1, 0], b_frag[0, 1], b_frag[1, 1], a_regs[0],
            a_regs[1])
        acc[jj * 4 //
            _SCALAR_ACC_FRAGMENT_WIDTH][jj * 4 %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d0
        acc[(jj * 4 + 1) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 1) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d1
        acc[(jj * 4 + 2) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 2) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d2
        acc[(jj * 4 + 3) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + 3) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d3

    @cute.jit
    def _mma_accumulate_large_m(self, acc, a_regs: cute.Tensor,
                                mb: cutlass.Constexpr[int],
                                jj: cutlass.Constexpr[int],
                                b_frag: cute.Tensor):
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[jj * 8 //
                _SCALAR_ACC_FRAGMENT_WIDTH][jj * 8 %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 1) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 1) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 2) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 2) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 3) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 3) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            a_regs[mb, 0], a_regs[mb, 1], a_regs[mb, 2], a_regs[mb, 3],
            b_frag[0, 0], b_frag[0, 1])
        acc[jj * 8 //
            _SCALAR_ACC_FRAGMENT_WIDTH][jj * 8 %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d0
        acc[(jj * 8 + 1) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 1) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d1
        acc[(jj * 8 + 2) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 2) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d2
        acc[(jj * 8 + 3) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 3) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[(jj * 8 + 4) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 4) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 5) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 5) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 6) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 6) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            acc[(jj * 8 + 7) //
                _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 7) %
                                            _SCALAR_ACC_FRAGMENT_WIDTH],
            a_regs[mb, 0], a_regs[mb, 1], a_regs[mb, 2], a_regs[mb, 3],
            b_frag[1, 0], b_frag[1, 1])
        acc[(jj * 8 + 4) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 4) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d0
        acc[(jj * 8 + 5) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 5) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d1
        acc[(jj * 8 + 6) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 6) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d2
        acc[(jj * 8 + 7) //
            _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 7) %
                                        _SCALAR_ACC_FRAGMENT_WIDTH] = d3

    @cute.jit
    def _stage_k_tile_async(self, problem_shape, a_bf16_flat: cute.Tensor,
                            b_i32_flat: cute.Tensor,
                            scales_i32_flat: cute.Tensor, smem_base: Int32,
                            tid: Int32, pipe: Int32, tile_idx: Int32,
                            block_valid_rows: Int32, a_gl_stride: Int32,
                            a_gl_rd_row: Int32, a_gl_rd_col0: Int32,
                            a_sh_wr: Int32, a_rows_per_iter: Int32,
                            output_n_tile: Int32, expert_idx: Int32):
        size_n, size_k = problem_shape
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_rd_route_off * 16) +
                    row * Int32(4))
            a_int4 = Int64(route_index) * Int64(a_gl_stride) + Int64(
                tile_idx) * Int64(self.a_gl_rd_delta_o) + Int64(a_gl_rd_col0)
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off) + pipe * Int32(self.a_sh_stage) +
                self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr))
            a_src = get_ptr_as_int64(a_bf16_flat, a_int4 * Int64(8))
            cp_async4_shared_global_pred(a_dst, a_src,
                                         (row < block_valid_rows).to(Int32))
        n16_total = size_n // 16
        chunks_per_k16 = self.cta_n_blocks * (self.block_tile_bytes // 16)
        total_chunks = self.cta_k_blocks * chunks_per_k16
        b_region = smem_base + Int32(
            self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        packed_tile = Int64(0)
        if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
            packed_tile = (Int64(expert_idx) * Int64(size_n // 64) +
                           Int64(output_n_tile)) * Int64(
                               size_k // 128) + Int64(tile_idx)
        for i in cutlass.range_constexpr(
                _covering_count(total_chunks, self.cta_threads)):
            chunk = Int32(i * self.cta_threads) + tid
            local_tile = chunk // Int32(self.block_tile_bytes // 16)
            local_k16 = local_tile // Int32(self.cta_n_blocks)
            local_n16 = local_tile % Int32(self.cta_n_blocks)
            tile_chunk = chunk % Int32(self.block_tile_bytes // 16)
            global_k16 = tile_idx * Int32(self.cta_k_blocks) + local_k16
            global_n16 = output_n_tile * Int32(self.cta_n_blocks) + local_n16
            if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                source_u32 = packed_tile * Int64(
                    32 * self.block_tile_bytes // 4) + Int64(chunk) * Int64(4)
            else:
                source_u32 = self._iq2_xs_descriptor_offset(
                    problem_shape, expert_idx, global_k16,
                    global_n16) + Int64(tile_chunk) * Int64(4)
            cp_async4_shared_global_pred(
                b_region + chunk * Int32(16),
                get_ptr_as_int64(b_i32_flat, source_u32),
                (chunk < Int32(total_chunks)).to(Int32))
        metadata_addr = get_ptr_as_int64(scales_i32_flat, Int64(0))
        expert_count = Int64(cute.size(b_i32_flat)) // Int64(
            size_k * size_n // (4 * self.block_pack_factor))
        base_chunks = _covering_count(self.cta_k_blocks * 16,
                                      self.block_k) * self.cta_n_blocks * 2
        base_region = b_region + Int32(
            self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes)
        for i in cutlass.range_constexpr(
                _covering_count(base_chunks, self.cta_threads)):
            chunk = Int32(i * self.cta_threads) + tid
            if chunk < Int32(base_chunks):
                if cutlass.const_expr(self.tile_n == 64
                                      and self.tile_k == 128):
                    base_tile = (
                        Int64(expert_idx) * Int64(size_k // self.block_k) +
                        Int64(tile_idx // Int32(2))) * Int64(
                            size_n // 64) + Int64(output_n_tile)
                    base_offset = base_tile * Int64(128) + Int64(
                        chunk) * Int64(16)
                else:
                    local_tile = chunk // Int32(2)
                    global_k256 = tile_idx * Int32(self.cta_k_blocks) // Int32(
                        self.block_k // 16) + local_tile // Int32(
                            self.cta_n_blocks)
                    global_n16 = output_n_tile * Int32(
                        self.cta_n_blocks) + local_tile % Int32(
                            self.cta_n_blocks)
                    block = (Int64(expert_idx) * Int64(size_k // self.block_k) +
                             Int64(global_k256)) * Int64(n16_total) + Int64(
                                 global_n16)
                    base_offset = block * Int64(32) + Int64(
                        chunk % Int32(2)) * Int64(16)
                cp_async4_shared_global(base_region + chunk * Int32(16),
                                        metadata_addr + base_offset)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _prefetch_pipeline_step(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, scales_i32_flat: cute.Tensor,
            smem_base: Int32, tid: Int32, pipe: Int32,
            kk: cutlass.Constexpr[int], tile_idx: Int32, k_tiles: Int32,
            reduce_k_tile: Int32, block_valid_rows: Int32, a_gl_stride: Int32,
            a_gl_rd_row: Int32, a_gl_rd_col0: Int32, a_sh_wr: Int32,
            a_rows_per_iter: Int32, output_n_tile: Int32, expert_idx: Int32):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                problem_shape, a_bf16_flat, b_i32_flat, scales_i32_flat,
                smem_base, tid, pipe, tile_idx, k_tiles, reduce_k_tile,
                block_valid_rows, a_gl_stride, a_gl_rd_row, a_gl_rd_col0,
                a_sh_wr, a_rows_per_iter, output_n_tile, expert_idx)

    @cute.jit
    def _prefetch_initial_tiles(
            self, problem_shape, a_bf16_flat: cute.Tensor,
            b_i32_flat: cute.Tensor, scales_i32_flat: cute.Tensor,
            smem_base: Int32, tid: Int32, k_tiles: Int32, reduce_k_tile: Int32,
            block_valid_rows: Int32, a_gl_stride: Int32, a_gl_rd_row: Int32,
            a_gl_rd_col0: Int32, a_sh_wr: Int32, a_rows_per_iter: Int32,
            output_n_tile: Int32, expert_idx: Int32):
        for pipe in cutlass.range_constexpr(self.stages - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    problem_shape,
                    a_bf16_flat, b_i32_flat, scales_i32_flat, smem_base, tid,
                    Int32(pipe), reduce_k_tile + Int32(pipe), block_valid_rows,
                    a_gl_stride, a_gl_rd_row, a_gl_rd_col0, a_sh_wr,
                    a_rows_per_iter, output_n_tile, expert_idx)
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(self, problem_shape, a_bf16_flat: cute.Tensor,
                                 b_i32_flat: cute.Tensor,
                                 scales_i32_flat: cute.Tensor, smem_base: Int32,
                                 tid: Int32, pipe: Int32, tile_idx: Int32,
                                 k_tiles: Int32, reduce_k_tile: Int32,
                                 block_valid_rows: Int32, a_gl_stride: Int32,
                                 a_gl_rd_row: Int32, a_gl_rd_col0: Int32,
                                 a_sh_wr: Int32, a_rows_per_iter: Int32,
                                 output_n_tile: Int32, expert_idx: Int32):
        fetch_tile = tile_idx + Int32(self.stages - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                problem_shape, a_bf16_flat, b_i32_flat, scales_i32_flat,
                smem_base, tid, Int32((pipe + self.stages - 1) % self.stages),
                reduce_k_tile + fetch_tile, block_valid_rows, a_gl_stride,
                a_gl_rd_row, a_gl_rd_col0, a_sh_wr, a_rows_per_iter,
                output_n_tile, expert_idx)
        else:
            cute.arch.cp_async_commit_group()

    @cute.jit
    def _reduction_offsets(self, tid: Int32):
        red_idx = tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * (tid // Int32(
            self.b_sh_stride_threads)) + tid % Int32(self.b_sh_stride_threads)
        return (red_idx, red_sh_stride, red_sh_delta, red_sh_rd)

    @cute.jit
    def _fold_cta_partials_m8(self, acc, smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid)
            if cutlass.const_expr(red_off == 2):
                if Int32(2) <= red_idx and red_idx < Int32(4):
                    for jj in cutlass.range_constexpr(4):
                        red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                            red_sh_rd - red_sh_stride * Int32(2))
                        st_shared_v4_f32(
                            self._int4_addr(smem_base,
                                            Int32(self.sh_red_off) +
                                            red_sh_wr),
                            acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                                jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH],
                            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH],
                            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH])
                cute.arch.sync_threads()
            if Int32(1) <= red_idx and red_idx < Int32(2):
                for jj in cutlass.range_constexpr(4):
                    red_sh_wr = red_sh_delta * Int32(jj * 2) + (red_sh_rd -
                                                                red_sh_stride)
                    if cutlass.const_expr(red_off > 1):
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off) +
                            red_sh_delta * Int32(jj * 2) + red_sh_rd)
                        wr_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off) + red_sh_wr)
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                        acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                            jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                                jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                                    jj * 4 %
                                    _SCALAR_ACC_FRAGMENT_WIDTH] + r0 + w0
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                                (jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                    (jj * 4 + 1) %
                                    _SCALAR_ACC_FRAGMENT_WIDTH] + r1 + w1
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                                (jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                    (jj * 4 + 2) %
                                    _SCALAR_ACC_FRAGMENT_WIDTH] + r2 + w2
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                                (jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                    (jj * 4 + 3) %
                                    _SCALAR_ACC_FRAGMENT_WIDTH] + r3 + w3
                    st_shared_v4_f32(
                        self._int4_addr(smem_base,
                                        Int32(self.sh_red_off) + red_sh_wr),
                        acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                            jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH])
            cute.arch.sync_threads()
            if red_idx == Int32(0):
                for jj in cutlass.range_constexpr(4):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off) + red_sh_delta * Int32(jj * 2) +
                        red_sh_rd)
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                                jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] + r0
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] + r1
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] + r2
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] + r3
            cute.arch.sync_threads()

    @cute.jit
    def _output_store_cursor(self, problem_shape, tid: Int32,
                             output_n_tile: Int32):
        size_n, size_k = problem_shape
        c_gl_stride = Int32(size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride * Int32(self.cta_threads //
                                            (2 * self.cta_n_blocks))
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads //
                                            (2 * self.cta_n_blocks))
        c_gl_wr = c_gl_stride * (tid //
                                 Int32(2 * self.cta_n_blocks)) + tid % Int32(
                                     2 * self.cta_n_blocks) + Int32(
                                         2 * self.cta_n_blocks) * output_n_tile
        c_sh_rd = c_sh_stride * (tid // Int32(
            2 * self.cta_n_blocks)) + tid % Int32(2 * self.cta_n_blocks)
        return (c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta,
                c_gl_wr, c_sh_rd)

    @cute.jit
    def _drain_output_smem(self, c_bf16_flat: cute.Tensor, smem_base: Int32,
                           c_gl_stride: Int32, c_gl_wr: Int32,
                           c_gl_wr_delta: Int32, c_sh_rd: Int32,
                           c_sh_rd_delta: Int32, block_valid_rows: Int32,
                           store_iters: cutlass.Constexpr[int]):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride
            if row < block_valid_rows:
                route_index = ld_shared_i32_relaxed(smem_base +
                                                    Int32(self.sh_route_off *
                                                          16) + row * Int32(4))
                true_idx = Int64(route_index) * Int64(c_gl_stride) + Int64(
                    c_gl_wr % c_gl_stride)
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base,
                                    Int32(self.sh_red_off) + c_sh_rd))
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(smem_base +
                                              Int32(self.sh_topk_off * 16) +
                                              row * Int32(4))
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                st_global_v4_u32(
                    get_ptr_as_int64(c_bf16_flat, true_idx * Int64(8)), q0, q1,
                    q2, q3)
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile_m8(self, problem_shape, acc, c_bf16_flat: cute.Tensor,
                       smem_base: Int32, tid: Int32, output_n_tile: Int32,
                       block_valid_rows: Int32,
                       global_scale_f32: cutlass.Float32):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = self._output_store_cursor(
            problem_shape, tid, output_n_tile)
        c_sh_wr = Int32(8) * c_sh_stride * (
            (tid & Int32(31)) % Int32(4) * Int32(2)) + (
                tid & Int32(31)) // Int32(4) + Int32(64) * (tid // Int32(32))
        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for jj in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jj)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) + wr * Int32(2),
                    acc[jj * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        jj * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] * write_scale)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) +
                    (wr + Int32(8) * c_sh_stride) * Int32(2),
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] *
                    write_scale)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) +
                    (wr + Int32(8)) * Int32(2),
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] *
                    write_scale)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) +
                    (wr + Int32(8) + Int32(8) * c_sh_stride) * Int32(2),
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] *
                    write_scale)
        cute.arch.sync_threads()
        store_iters = _covering_count(
            16, self.cta_threads // (2 * self.cta_n_blocks))
        self._drain_output_smem(c_bf16_flat, smem_base, c_gl_stride, c_gl_wr,
                                c_gl_wr_delta, c_sh_rd, c_sh_rd_delta,
                                block_valid_rows, store_iters)

    @cute.jit
    def _fold_cta_partials_large_m(self, acc0, acc1, acc2, acc3,
                                   smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid)
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._fold_cta_partials_large_m_block(
                        acc0, smem_base, red_off, red_idx, red_sh_stride,
                        red_sh_delta, red_sh_rd)
                elif cutlass.const_expr(mb == 1):
                    self._fold_cta_partials_large_m_block(
                        acc1, smem_base, red_off, red_idx, red_sh_stride,
                        red_sh_delta, red_sh_rd)
                elif cutlass.const_expr(mb == 2):
                    self._fold_cta_partials_large_m_block(
                        acc2, smem_base, red_off, red_idx, red_sh_stride,
                        red_sh_delta, red_sh_rd)
                else:
                    self._fold_cta_partials_large_m_block(
                        acc3, smem_base, red_off, red_idx, red_sh_stride,
                        red_sh_delta, red_sh_rd)

    @cute.jit
    def _fold_cta_partials_large_m_block(self, acc, smem_base: Int32,
                                         red_off: cutlass.Constexpr[int],
                                         red_idx: Int32, red_sh_stride: Int32,
                                         red_sh_delta: Int32,
                                         red_sh_rd: Int32):
        if cutlass.const_expr(red_off == 2):
            if Int32(2) <= red_idx and red_idx < Int32(4):
                for flat_j in cutlass.range_constexpr(8):
                    red_sh_wr = red_sh_delta * Int32(flat_j) + (
                        red_sh_rd - red_sh_stride * Int32(2))
                    st_shared_v4_f32(
                        self._int4_addr(smem_base,
                                        Int32(self.sh_red_off) + red_sh_wr),
                        acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                            flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH],
                        acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH])
            cute.arch.sync_threads()
        if Int32(1) <= red_idx and red_idx < Int32(2):
            for flat_j in cutlass.range_constexpr(8):
                red_sh_wr = red_sh_delta * Int32(flat_j) + (red_sh_rd -
                                                            red_sh_stride)
                if cutlass.const_expr(red_off > 1):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off) + red_sh_delta * Int32(flat_j) +
                        red_sh_rd)
                    wr_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off) + red_sh_wr)
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                    acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                                flat_j * 4 %
                                _SCALAR_ACC_FRAGMENT_WIDTH] + r0 + w0
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (flat_j * 4 + 1) %
                                _SCALAR_ACC_FRAGMENT_WIDTH] + r1 + w1
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (flat_j * 4 + 2) %
                                _SCALAR_ACC_FRAGMENT_WIDTH] + r2 + w2
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                            (flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (flat_j * 4 + 3) %
                                _SCALAR_ACC_FRAGMENT_WIDTH] + r3 + w3
                st_shared_v4_f32(
                    self._int4_addr(smem_base,
                                    Int32(self.sh_red_off) + red_sh_wr),
                    acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                        flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH])
        cute.arch.sync_threads()
        if red_idx == Int32(0):
            for flat_j in cutlass.range_constexpr(8):
                rd_addr = self._int4_addr(
                    smem_base,
                    Int32(self.sh_red_off) + red_sh_delta * Int32(flat_j) +
                    red_sh_rd)
                r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                acc[flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                    flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                        flat_j * 4 // _SCALAR_ACC_FRAGMENT_WIDTH][
                            flat_j * 4 % _SCALAR_ACC_FRAGMENT_WIDTH] + r0
                acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                        (flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH] + r1
                acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                        (flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH] + r2
                acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] = acc[
                        (flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH] + r3
        cute.arch.sync_threads()

    @cute.jit
    def _write_bf16x2_shared(self, smem_base: Int32, half2_idx: Int32,
                             c0: cutlass.Float32, c1: cutlass.Float32,
                             write_scale: cutlass.Float32):
        packed = self._pack_f32x2_to_elem2(c0 * write_scale, c1 * write_scale)
        st_shared_u32(
            smem_base + Int32(self.sh_red_off * 16) + half2_idx * Int32(4),
            packed)

    @cute.jit
    def _store_tile_large_m(self, problem_shape, acc0, acc1, acc2, acc3,
                            c_bf16_flat: cute.Tensor, smem_base: Int32,
                            tid: Int32, output_n_tile: Int32,
                            block_valid_rows: Int32,
                            global_scale_f32: cutlass.Float32):
        c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd = self._output_store_cursor(
            problem_shape, tid, output_n_tile)
        c_sh_wr = Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4)) + (
            tid & Int32(31)) % Int32(4) + Int32(32) * (tid // Int32(32))
        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._store_tile_large_m_block(acc0, smem_base, c_sh_wr,
                                                   c_sh_stride, write_scale)
                elif cutlass.const_expr(mb == 1):
                    self._store_tile_large_m_block(acc1, smem_base, c_sh_wr,
                                                   c_sh_stride, write_scale)
                elif cutlass.const_expr(mb == 2):
                    self._store_tile_large_m_block(acc2, smem_base, c_sh_wr,
                                                   c_sh_stride, write_scale)
                else:
                    self._store_tile_large_m_block(acc3, smem_base, c_sh_wr,
                                                   c_sh_stride, write_scale)
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()
        store_iters = _covering_count(
            16 * self.cta_m_blocks,
            self.cta_threads // (2 * self.cta_n_blocks))
        self._drain_output_smem(c_bf16_flat, smem_base, c_gl_stride, c_gl_wr,
                                c_gl_wr_delta, c_sh_rd, c_sh_rd_delta,
                                block_valid_rows, store_iters)

    @cute.jit
    def _store_tile_large_m_block(self, acc, smem_base: Int32, c_sh_wr: Int32,
                                  c_sh_stride: Int32,
                                  write_scale: cutlass.Float32):
        for jj in cutlass.range_constexpr(4):
            wr = c_sh_wr + Int32(8 * jj)
            self._write_bf16x2_shared(
                smem_base, wr,
                acc[jj * 8 //
                    _SCALAR_ACC_FRAGMENT_WIDTH][jj * 8 %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                acc[(jj * 8 + 1) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 1) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                write_scale)
            self._write_bf16x2_shared(
                smem_base, wr + Int32(4) * c_sh_stride * Int32(8) + Int32(0),
                acc[(jj * 8 + 2) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 2) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                acc[(jj * 8 + 3) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 3) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                write_scale)
            self._write_bf16x2_shared(
                smem_base, wr + Int32(4),
                acc[(jj * 8 + 4) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 4) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                acc[(jj * 8 + 5) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 5) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                write_scale)
            self._write_bf16x2_shared(
                smem_base, wr + Int32(4) * c_sh_stride * Int32(8) + Int32(4),
                acc[(jj * 8 + 6) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 6) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                acc[(jj * 8 + 7) //
                    _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 8 + 7) %
                                                _SCALAR_ACC_FRAGMENT_WIDTH],
                write_scale)


class W2A16MoeKernel:

    def __init__(self, decode: bool, row_tile: int):
        if row_tile not in _ALLOWED_ROUTED_SIZES or (decode and row_tile != 8):
            raise ValueError('Unsupported W2A16 row tile for the selected mode')
        block_size = row_tile
        self.hidden_size = 1024
        self.top_k = 9
        self.tc_decode_fused_sum = decode
        self.tc_zero_output = True
        self.zero_fc2_output = False
        common = dict(num_experts=512,
                      tile_n=128 if row_tile <= 16 else 256,
                      tile_k=128 if row_tile <= 16 else 64,
                      moe_block_size=block_size,
                      max_m_blocks=1,
                      direct_topk_routes=decode,
                      dynamic_num_experts=True,
                      pipeline_stages=2 if row_tile <= 16 else 4)
        self.fc1 = _Iq2GemmKernel(top_k=9,
                                  mul_topk_weights=False,
                                  epilogue_activation='relu2',
                                  **common)
        self.fc2 = _Iq2GemmKernel(top_k=1, mul_topk_weights=True, **common)
        self.cta_threads = 256
        self.sms = self.fc1.sms
        self.blocks_per_sm = min(self.fc1.blocks_per_sm,
                                 self.fc2.blocks_per_sm)
        self.shared_words = max(self.fc1.shared_words, self.fc2.shared_words)
        self.iq2_lut_bytes = block_codec('iq2_xxs').lut_bytes(selectors=True)
        self.iq2_xs_lut_off = self.shared_words * 4
        self.shared_words += self.iq2_lut_bytes // 4
        self.fc1.iq2_xs_smem_lut = True
        self.fc2.iq2_xs_smem_lut = True
        self.barrier_count_off = self.sms * 4
        self.barrier_sense_off = self.sms * 4 + 1
        self.smem_carveout = 100

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
            self, a_bf16_ptr: cute.Pointer, w13_ptr: cute.Pointer,
            w2_ptr: cute.Pointer, w13_elements: cutlass.Int64,
            w2_elements: cutlass.Int64, fc1_bf16_flat: cute.Tensor,
            activated_bf16_flat: cute.Tensor, fc2_bf16_flat: cute.Tensor,
            w13_scales_ptr: cute.Pointer, w2_scales_ptr: cute.Pointer,
            w13_global_scale_ptr: cute.Pointer,
            w2_global_scale_ptr: cute.Pointer,
            packed_route_indices: cute.Tensor, block_expert_ids: cute.Tensor,
            packed_route_count: cute.Tensor, topk_weights_ptr: cute.Pointer,
            fc1_c_tmp_f32_flat: cute.Tensor, fc2_c_tmp_f32_flat: cute.Tensor,
            locks_i32_flat: cute.Tensor, fc1_iq2_lut_ptr: cute.Pointer,
            weight_num_experts: cutlass.Int32, active_m: cutlass.Int32,
            grid_x: cutlass.Int32, intermediate_size: cutlass.Int32,
            stream: cuda.CUstream):
        expert_count = Int64(weight_num_experts)
        w13_i32_flat = cute.make_tensor(w13_ptr,
                                        layout=cute.make_layout(
                                            (Int64(w13_elements), ),
                                            stride=(1, )))
        w2_i32_flat = cute.make_tensor(w2_ptr,
                                       layout=cute.make_layout(
                                           (Int64(w2_elements), ),
                                           stride=(1, )))
        w13_metadata_elements = expert_count * Int64(
            self.hidden_size // self.fc1.block_k * Int64(intermediate_size) *
            self.fc1.block_metadata_bytes // 4)
        w2_metadata_elements = expert_count * Int64(
            Int64(intermediate_size) // self.fc2.block_k * self.hidden_size *
            self.fc2.block_metadata_bytes // 4)
        w13_scales_i32_flat = cute.make_tensor(w13_scales_ptr,
                                               layout=cute.make_layout(
                                                   (w13_metadata_elements, ),
                                                   stride=(1, )))
        w2_scales_i32_flat = cute.make_tensor(w2_scales_ptr,
                                              layout=cute.make_layout(
                                                  (w2_metadata_elements, ),
                                                  stride=(1, )))
        w13_global_scale = cute.make_tensor(w13_global_scale_ptr,
                                            layout=cute.make_layout(
                                                (expert_count, ),
                                                stride=(1, )))
        w2_global_scale = cute.make_tensor(w2_global_scale_ptr,
                                           layout=cute.make_layout(
                                               (expert_count, ), stride=(1, )))
        a_rows = Int64(active_m)
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((a_rows * Int64(self.hidden_size), ),
                                    stride=(1, )))
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((Int64(active_m) * Int64(self.top_k), ),
                                    stride=(1, )))
        fc1_iq2_lut_flat = cute.make_tensor(fc1_iq2_lut_ptr,
                                            layout=cute.make_layout(
                                                (Int64(_IQ2_LUT_BYTES), ),
                                                stride=(1, )))
        grid = (grid_x, 1, 1)
        self.kernel(a_bf16_flat, w13_i32_flat, w2_i32_flat, fc1_bf16_flat,
                    activated_bf16_flat, fc2_bf16_flat, w13_scales_i32_flat,
                    w2_scales_i32_flat, w13_global_scale, w2_global_scale,
                    packed_route_indices, block_expert_ids, packed_route_count,
                    topk_weights_flat, fc1_c_tmp_f32_flat, fc2_c_tmp_f32_flat,
                    locks_i32_flat, fc1_iq2_lut_flat, weight_num_experts,
                    active_m, intermediate_size).launch(
                        grid=grid,
                        block=[self.cta_threads, 1, 1],
                        min_blocks_per_mp=self.blocks_per_sm,
                        cooperative=True,
                        preferred_smem_carveout=self.smem_carveout,
                        stream=stream)

    @cute.kernel
    def kernel(self, a_bf16_flat: cute.Tensor, w13_i32_flat: cute.Tensor,
               w2_i32_flat: cute.Tensor, fc1_bf16_flat: cute.Tensor,
               activated_bf16_flat: cute.Tensor, fc2_bf16_flat: cute.Tensor,
               w13_scales_i32_flat: cute.Tensor,
               w2_scales_i32_flat: cute.Tensor, w13_global_scale: cute.Tensor,
               w2_global_scale: cute.Tensor, packed_route_indices: cute.Tensor,
               block_expert_ids: cute.Tensor, packed_route_count: cute.Tensor,
               topk_weights_flat: cute.Tensor, fc1_c_tmp_f32_flat: cute.Tensor,
               fc2_c_tmp_f32_flat: cute.Tensor, locks_i32_flat: cute.Tensor,
               fc1_iq2_lut_flat: cute.Tensor, weight_num_experts: cutlass.Int32,
               active_m: cutlass.Int32, intermediate_size: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32,
                                                          self.shared_words],
                                     1024]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        fc1_iq2_lut_addr = get_ptr_as_int64(fc1_iq2_lut_flat, Int32(0))
        for i in cutlass.range_constexpr(
                _covering_count(self.iq2_lut_bytes // 16, self.cta_threads)):
            chunk = Int32(i * self.cta_threads) + tid
            if chunk < Int32(self.iq2_lut_bytes // 16):
                cp_async4_shared_global(
                    smem_base + Int32(self.iq2_xs_lut_off) + chunk * Int32(16),
                    fc1_iq2_lut_addr + Int64(chunk) * Int64(16))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        fc1_emit_tile = None
        fc2_emit_tile = None
        self._moe_body(a_bf16_flat, w13_i32_flat, w2_i32_flat, fc1_bf16_flat,
                       activated_bf16_flat, fc2_bf16_flat, w13_scales_i32_flat,
                       w2_scales_i32_flat, w13_global_scale, w2_global_scale,
                       packed_route_indices, block_expert_ids,
                       packed_route_count, topk_weights_flat,
                       fc1_c_tmp_f32_flat, fc2_c_tmp_f32_flat, locks_i32_flat,
                       weight_num_experts, smem_base, tid, cta, grid_x,
                       active_m, intermediate_size, fc1_emit_tile,
                       fc2_emit_tile)

    @cute.jit
    def _moe_body(self,
                  a_bf16_flat: cute.Tensor,
                  w13_i32_flat: cute.Tensor,
                  w2_i32_flat: cute.Tensor,
                  fc1_bf16_flat: cute.Tensor,
                  activated_bf16_flat: cute.Tensor,
                  fc2_bf16_flat: cute.Tensor,
                  w13_scales_i32_flat: cute.Tensor,
                  w2_scales_i32_flat: cute.Tensor,
                  w13_global_scale: cute.Tensor,
                  w2_global_scale: cute.Tensor,
                  packed_route_indices: cute.Tensor,
                  block_expert_ids: cute.Tensor,
                  packed_route_count: cute.Tensor,
                  topk_weights_flat: cute.Tensor,
                  fc1_c_tmp_f32_flat: cute.Tensor,
                  fc2_c_tmp_f32_flat: cute.Tensor,
                  locks_i32_flat: cute.Tensor,
                  weight_num_experts: Int32,
                  smem_base: Int32,
                  tid: Int32,
                  cta: Int32,
                  grid_x: Int32,
                  active_m: cutlass.Int32,
                  intermediate_size: cutlass.Int32,
                  fc1_emit_tile: cutlass.Constexpr = None,
                  fc2_emit_tile: cutlass.Constexpr = None):
        table_addr = Int64(smem_base + Int32(self.iq2_xs_lut_off))
        fc1_phase_lut = table_addr
        fc2_phase_lut = table_addr
        self.fc1._run_persistent_gemm(
            (intermediate_size, self.hidden_size), a_bf16_flat, w13_i32_flat,
            activated_bf16_flat, w13_scales_i32_flat, w13_global_scale,
            packed_route_indices, block_expert_ids, packed_route_count,
            topk_weights_flat, fc1_c_tmp_f32_flat, locks_i32_flat,
            fc1_phase_lut, smem_base, tid, cta, grid_x, active_m, fc1_emit_tile)
        self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(self.zero_fc2_output):
            self._zero_fc2_output(fc2_bf16_flat, tid, cta, grid_x, active_m)
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        fc2_store = fc2_bf16_flat
        if cutlass.const_expr(self.tc_decode_fused_sum):
            fc2_store = fc1_bf16_flat
        self.fc2._run_persistent_gemm(
            (self.hidden_size, intermediate_size), activated_bf16_flat,
            w2_i32_flat, fc2_store, w2_scales_i32_flat, w2_global_scale,
            packed_route_indices, block_expert_ids, packed_route_count,
            topk_weights_flat, fc2_c_tmp_f32_flat, locks_i32_flat,
            fc2_phase_lut, smem_base, tid, cta, grid_x,
            active_m * Int32(self.top_k), fc2_emit_tile)
        if cutlass.const_expr(self.tc_decode_fused_sum):
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            self._sum_iq2_xs_routes(fc1_bf16_flat, fc2_bf16_flat,
                                    packed_route_indices, weight_num_experts,
                                    tid, cta, grid_x, active_m)

    @cute.jit
    def _sum_iq2_xs_routes(self, route_output: cute.Tensor,
                           output: cute.Tensor, topk_ids: cute.Tensor,
                           weight_num_experts: Int32, tid: Int32, cta: Int32,
                           grid_x: Int32, active_m: Int32):
        idx = Int64(cta) * Int64(self.cta_threads) + Int64(tid)
        stride = Int64(grid_x) * Int64(self.cta_threads)
        total = Int64(active_m) * Int64(self.hidden_size)
        while idx < total:
            token = idx // Int64(self.hidden_size)
            col = idx % Int64(self.hidden_size)
            value = cutlass.Float32(0.0)
            partials = cute.make_rmem_tensor((4, ), cutlass.Float32)
            for group in cutlass.range(self.top_k // 4, unroll=1):
                partials.fill(0.0)
                for j in cutlass.range_constexpr(4):
                    slot = group * Int32(4) + Int32(j)
                    route = token * Int64(self.top_k) + Int64(slot)
                    expert = topk_ids[route].to(Int32)
                    if expert >= Int32(0) and expert < weight_num_experts:
                        offset = route * Int64(self.hidden_size) + col
                        partials[j] = route_output[offset].to(cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    value += partials[j]
            for slot in cutlass.range_constexpr(self.top_k // 4 * 4,
                                                self.top_k):
                route = token * Int64(self.top_k) + Int64(slot)
                expert = topk_ids[route].to(Int32)
                if expert >= Int32(0) and expert < weight_num_experts:
                    offset = route * Int64(self.hidden_size) + col
                    value += route_output[offset].to(cutlass.Float32)
            if cutlass.const_expr(not self.tc_zero_output):
                value += output[idx].to(cutlass.Float32)
            output[idx] = cutlass.BFloat16(value)
            idx += stride

    @cute.jit
    def _grid_barrier(self, locks_i32_flat: cute.Tensor, tid: Int32,
                      grid_x: Int32):
        cute.arch.sync_threads()
        if tid == Int32(0):
            count_addr = get_ptr_as_int64(locks_i32_flat,
                                          Int32(self.barrier_count_off))
            sense_addr = get_ptr_as_int64(locks_i32_flat,
                                          Int32(self.barrier_sense_off))
            old_sense = ld_global_acquire_i32(sense_addr)
            old_count = atomic_add_global_i32(count_addr, Int32(1))
            if old_count == grid_x - Int32(1):
                st_global_i32(count_addr, Int32(0))
                threadfence()
                red_add_global_release_i32(sense_addr, Int32(1))
            else:
                sense = old_sense
                while sense == old_sense:
                    sense = ld_global_acquire_i32(sense_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _zero_fc2_output(self, fc2_bf16_flat: cute.Tensor, tid: Int32,
                         cta: Int32, grid_x: Int32, active_m: cutlass.Int32):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k * self.hidden_size)
        zero = self._cast_elem(cutlass.Float32(0.0))
        while idx < total:
            fc2_bf16_flat[idx] = zero
            idx += stride


def _scale_group_size(scale_format: str) -> int:
    return 16
