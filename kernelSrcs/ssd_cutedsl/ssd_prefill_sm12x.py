# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packed Mamba2 SSD using warp MMA and register-carried state on SM120/121.

The caller owns all allocations and supplies an explicit stream. Inputs/output
are FP16 [T,H,D], B/C are FP16 [T,G,N], A is FP32 [H], and dt/bias/D are FP16.
State is an in-place FP16 [slots,H,D,N] pool addressed by unique state_indices.
cu_seqlens is monotonic INT32 [batch+1], starts at zero and ends at T. Empty
sequences preserve their state. dt_processed and prefix are FP32 [T,H] scratch.

The runner keeps cumsum separate and fuses the remaining SSD computation without TMEM or global CB /
per-chunk state buffers. The mathematical decomposition follows the existing
ssd_prefill_blackwell.py; warp MMA follows ssd_prefill.py.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp, warpgroup


class PackedSSD:
    """Compile-time shape specialization for two-kernel packed SSD."""

    def __init__(self,
                 dim,
                 dstate,
                 chunk_size=64,
                 dim_tile=64,
                 dt_softplus=True):
        if dim not in (64, 80, 128) or dstate not in (64, 128):
            raise ValueError("dim must be 64/80/128 and dstate must be 64/128")
        if chunk_size not in (64, 128) or dim_tile not in (32, 64):
            raise ValueError(
                "chunk_size must be 64/128 and dim_tile must be 32/64")
        self.dim = dim
        self.dstate = dstate
        self.chunk_size = chunk_size
        self.dim_tile = dim_tile
        self.dt_softplus = dt_softplus
        self.threads = 256
        self.k_tile = 16
        self.scan_chunks = 8

    @cute.jit
    def x_layout(self):
        kind = (warpgroup.SmemLayoutAtomKind.K_SW128 if self.dim_tile == 64
                else warpgroup.SmemLayoutAtomKind.K_SW64)
        atom = warpgroup.make_smem_layout_atom(kind, cutlass.Float16)
        layout = cute.coalesce(cute.tile_to_shape(
            atom, (self.chunk_size, self.dim_tile), order=(0, 1)),
                               target_profile=(1, 1))
        return cute.select(layout, [1, 0])

    @cute.jit
    def bc_layout(self):
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
        return cute.coalesce(cute.tile_to_shape(atom,
                                                (self.chunk_size, self.dstate),
                                                order=(0, 1)),
                             target_profile=(1, 1))

    @cute.jit
    def copy_bc_tma(self, atom: cute.CopyAtom, source: cute.Tensor,
                    destination: cute.Tensor, offset: cutlass.Int32,
                    group: cutlass.Int32, barrier: cute.Pointer):
        view = cute.domain_offset((0, offset), source[None, None, group])
        tile = cute.zipped_divide(view,
                                  (self.dstate, self.chunk_size))[((None,
                                                                    None),
                                                                   (0, 0))]
        dest, src = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                          cute.group_modes(destination, 0, 2),
                                          cute.group_modes(tile, 0, 2))
        cute.copy(atom, src, dest, tma_bar_ptr=barrier)

    @cute.jit
    def __call__(self, x: cute.Tensor, dt: cute.Tensor, a: cute.Tensor,
                 b: cute.Tensor, c: cute.Tensor, d: cute.Tensor,
                 dt_bias: cute.Tensor, cu_seqlens: cute.Tensor,
                 state_indices: cute.Tensor, state: cute.Tensor,
                 output: cute.Tensor, dt_processed: cute.Tensor,
                 prefix: cute.Tensor, stream: cuda.CUstream):
        if cutlass.const_expr(not cute.is_static(x.shape[0])
                              or x.shape[0] != 0):
            if x.shape[0] > 0:
                batch = state_indices.shape[0]
                heads = x.shape[1]
                self.cumsum(
                    dt, a, dt_bias, cu_seqlens, dt_processed, prefix).launch(
                        grid=(heads, x.shape[0] //
                              (self.chunk_size * self.scan_chunks) + batch, 1),
                        block=(self.chunk_size, 1, 1),
                        smem=self.chunk_size * 4,
                        stream=stream)
                mma = cute.make_tiled_mma(warp.MmaF16BF16Op(
                    cutlass.Float16, cutlass.Float32, (16, 8, 16)),
                                          atom_layout_mnk=(2, 4, 1))
                x_view = cute.make_tensor(
                    x.iterator,
                    cute.make_layout((self.dim, x.shape[0], heads),
                                     stride=(1, heads * self.dim, self.dim)))
                x_layout = self.x_layout()
                tma_x, tensor_x = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(), x_view, x_layout,
                    (self.dim_tile, self.chunk_size))
                bc_layout = cute.select(self.bc_layout(), [1, 0])
                bc_view_layout = cute.make_layout(
                    (self.dstate, b.shape[0], b.shape[1]),
                    stride=(1, b.shape[1] * self.dstate, self.dstate))
                b_view = cute.make_tensor(b.iterator, bc_view_layout)
                c_view = cute.make_tensor(c.iterator, bc_view_layout)
                tma_b, tensor_b = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(), b_view, bc_layout,
                    (self.dstate, self.chunk_size))
                tma_c, tensor_c = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(), c_view, bc_layout,
                    (self.dstate, self.chunk_size))
                shared_bytes = (
                    self.dim_tile * self.k_tile + self.chunk_size * self.dstate
                    + self.chunk_size * max(self.dstate, self.chunk_size) +
                    self.dim_tile *
                    self.chunk_size) * 2 + 2048 + self.chunk_size * 8
                self.fused(x, b, c, d, cu_seqlens, state_indices, state,
                           output, dt_processed, prefix, mma, tma_x, tensor_x,
                           tma_b, tma_c, tensor_b,
                           tensor_c).launch(grid=(cute.ceil_div(
                               self.dim, self.dim_tile), batch, heads),
                                            block=(self.threads, 1, 1),
                                            smem=shared_bytes,
                                            min_blocks_per_mp=2,
                                            stream=stream)

    @cute.kernel
    def cumsum(self, dt: cute.Tensor, a: cute.Tensor, bias: cute.Tensor,
               cu_seqlens: cute.Tensor, dt_processed: cute.Tensor,
               prefix: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        head, tile, _ = cute.arch.block_idx()
        group_size = self.chunk_size * self.scan_chunks
        low = cutlass.Int32(0)
        high = cu_seqlens.shape[0] - 1
        # These monotonic keys reserve at most one extra scan group per sequence.
        while low + 1 < high:
            middle = (low + high) // 2
            key = cu_seqlens[middle] // group_size + middle
            if key <= tile:
                low = middle
            else:
                high = middle
        start = cu_seqlens[low]
        end = cu_seqlens[low + 1]
        first_tile = start // group_size + low
        group_start = start + (tile - first_tile) * group_size
        allocator = cutlass.utils.SmemAllocator()
        scan = allocator.allocate_tensor(cutlass.Float32,
                                         cute.make_layout(self.chunk_size), 16)
        for offset in range(group_start,
                            cutlass.min(end, group_start + group_size),
                            self.chunk_size):
            token = offset + tid
            delta = cutlass.Float32(0.0)
            if token < end:
                delta = cutlass.Float32(dt[token, head]) + cutlass.Float32(
                    bias[head])
                if cutlass.const_expr(self.dt_softplus):
                    if delta <= cutlass.Float32(20.0):
                        delta = cute.log(
                            cutlass.Float32(1.0) + cute.exp(delta))
                dt_processed[token, head] = delta
            scan[tid] = delta * a[head]
            cute.arch.barrier()
            for step in cutlass.range_constexpr(
                    0,
                    self.chunk_size.bit_length() - 1):
                distance = 1 << step
                previous = cutlass.Float32(0.0)
                if tid >= distance:
                    previous = scan[tid - distance]
                cute.arch.barrier()
                scan[tid] = scan[tid] + previous
                cute.arch.barrier()
            if token < end:
                prefix[token, head] = scan[tid]
            cute.arch.barrier()

    @cute.jit
    def multiply(self,
                 mma: cute.TiledMma,
                 lhs: cute.Tensor,
                 rhs: cute.Tensor,
                 accumulator: cute.Tensor,
                 lhs_transpose: cutlass.Constexpr = False,
                 rhs_transpose: cutlass.Constexpr = False,
                 post_barrier: cutlass.Constexpr = True,
                 pre_barrier: cutlass.Constexpr = True):
        tid, _, _ = cute.arch.thread_idx()
        thread_mma = mma.get_slice(tid)
        lhs_part = thread_mma.partition_A(lhs)
        rhs_part = thread_mma.partition_B(rhs)
        lhs_regs = thread_mma.make_fragment_A(lhs_part)
        rhs_regs = thread_mma.make_fragment_B(rhs_part)
        atom_a = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=lhs_transpose, num_matrices=4),
            cutlass.Float16)
        atom_b = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=rhs_transpose, num_matrices=4),
            cutlass.Float16)
        copy_a = cute.make_tiled_copy_A(atom_a, mma)
        copy_b = cute.make_tiled_copy_B(atom_b, mma)
        if cutlass.const_expr(pre_barrier):
            cute.arch.barrier()
        cute.copy(copy_a,
                  copy_a.get_slice(tid).partition_S(lhs),
                  copy_a.retile(lhs_regs))
        cute.copy(copy_b,
                  copy_b.get_slice(tid).partition_S(rhs),
                  copy_b.retile(rhs_regs))
        cute.gemm(mma, accumulator, lhs_regs, rhs_regs, accumulator)
        if cutlass.const_expr(post_barrier):
            cute.arch.barrier()

    @cute.jit
    def multiply_pair(self, mma: cute.TiledMma, lhs: cute.Tensor,
                      rhs_cb: cute.Tensor, rhs_state: cute.Tensor,
                      cb_acc: cute.Tensor, out_acc: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        thread_mma = mma.get_slice(tid)
        lhs_regs = thread_mma.make_fragment_A(thread_mma.partition_A(lhs))
        cb_regs = thread_mma.make_fragment_B(thread_mma.partition_B(rhs_cb))
        state_regs = thread_mma.make_fragment_B(
            thread_mma.partition_B(rhs_state))
        atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float16)
        copy_a = cute.make_tiled_copy_A(atom, mma)
        copy_b = cute.make_tiled_copy_B(atom, mma)
        cute.copy(copy_a,
                  copy_a.get_slice(tid).partition_S(lhs),
                  copy_a.retile(lhs_regs))
        cute.copy(copy_b,
                  copy_b.get_slice(tid).partition_S(rhs_cb),
                  copy_b.retile(cb_regs))
        cute.gemm(mma, cb_acc, lhs_regs, cb_regs, cb_acc)
        cute.copy(copy_b,
                  copy_b.get_slice(tid).partition_S(rhs_state),
                  copy_b.retile(state_regs))
        cute.gemm(mma, out_acc, lhs_regs, state_regs, out_acc)

    @cute.kernel
    def fused(self, x: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
              d: cute.Tensor, cu_seqlens: cute.Tensor,
              state_indices: cute.Tensor, state: cute.Tensor,
              output: cute.Tensor, dt_processed: cute.Tensor,
              prefix: cute.Tensor, mma: cute.TiledMma, tma_x: cute.CopyAtom,
              tensor_x: cute.Tensor, tma_b: cute.CopyAtom,
              tma_c: cute.CopyAtom, tensor_b: cute.Tensor,
              tensor_c: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        d_tile, seq, head = cute.arch.block_idx()
        start = cu_seqlens[seq]
        end = cu_seqlens[seq + 1]
        slot = state_indices[seq]
        group = head // (x.shape[1] // b.shape[1])
        d_base = d_tile * self.dim_tile
        allocator = cutlass.utils.SmemAllocator()
        state_slice_layout = cute.coalesce(
            cute.tile_to_shape(warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW32, cutlass.Float16),
                               (self.dim_tile, self.k_tile),
                               order=(0, 1)),
            target_profile=(1, 1))
        state_half = allocator.allocate_tensor(
            cutlass.Float16,
            state_slice_layout.outer,
            1024,
            swizzle=state_slice_layout.inner)
        cb_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
        cb_layout = cute.coalesce(cute.tile_to_shape(
            cb_atom, (self.chunk_size, self.chunk_size), order=(0, 1)),
                                  target_profile=(1, 1))
        bc_layout = self.bc_layout()
        c_storage_layout = cute.coalesce(cute.tile_to_shape(
            cb_atom, (self.chunk_size, max(self.dstate, self.chunk_size)),
            order=(0, 1)),
                                         target_profile=(1, 1))
        c_storage = allocator.allocate_tensor(cutlass.Float16,
                                              c_storage_layout.outer,
                                              1024,
                                              swizzle=c_storage_layout.inner)
        full_c = cute.make_tensor(c_storage.iterator, bc_layout.outer)
        full_b = allocator.allocate_tensor(cutlass.Float16,
                                           bc_layout.outer,
                                           1024,
                                           swizzle=bc_layout.inner)
        # C is dead before weighted CB and the state operand reuse its storage.
        cb_half = cute.make_tensor(full_c.iterator, cb_layout.outer)
        rhs_storage = full_c
        full_c_tma = cute.make_tensor(full_c.iterator,
                                      cute.select(full_c.layout, [1, 0]))
        full_b_tma = cute.make_tensor(full_b.iterator,
                                      cute.select(full_b.layout, [1, 0]))
        b_values = self.k_tile * self.dstate // self.threads
        copy_b_vector = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float16,
                                num_bits_per_copy=16 * b_values),
            cute.make_layout(
                (self.threads //
                 (self.dstate // b_values), self.dstate // b_values),
                stride=(self.dstate // b_values, 1)),
            cute.make_layout((1, b_values)))
        thread_b_vector = copy_b_vector.get_slice(tid)
        b_vector_coords = thread_b_vector.partition_S(
            cute.make_identity_tensor((self.k_tile, self.dstate)))
        state_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
        state_layout = cute.coalesce(cute.tile_to_shape(
            state_atom, (self.k_tile, self.dstate), order=(0, 1)),
                                     target_profile=(1, 1))
        rhs_state = cute.make_tensor(rhs_storage.iterator,
                                     cute.select(state_layout.outer, [1, 0]))

        x_shared = allocator.allocate_tensor(cutlass.Float16,
                                             self.x_layout().outer,
                                             1024,
                                             swizzle=self.x_layout().inner)
        x_barrier = allocator.allocate_array(cutlass.Int64, 1)
        bc_barrier = allocator.allocate_array(cutlass.Int64, 1)
        prefix_shared = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout(self.chunk_size), 16)
        delta_shared = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout(self.chunk_size), 16)
        if tid == 0:
            cute.arch.mbarrier_init(x_barrier, 1)
            cute.arch.mbarrier_init(bc_barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        x_phase = cutlass.Int32(0)
        bc_phase = cutlass.Int32(0)

        thread_mma = mma.get_slice(tid)
        state_coords = thread_mma.partition_C(
            cute.make_identity_tensor((self.dim_tile, self.dstate)))
        cb_coords = thread_mma.partition_C(
            cute.make_identity_tensor((self.chunk_size, self.chunk_size)))
        out_coords = thread_mma.partition_C(
            cute.make_identity_tensor((self.chunk_size, self.dim_tile)))
        copy_x_output = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float16,
                                num_bits_per_copy=32), mma)
        x_output_view = cute.make_tensor(x_shared.iterator,
                                         cute.select(x_shared.layout, [1, 0]))
        running = cute.make_rmem_tensor(state_coords.shape, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(running)):
            row, col = state_coords[i]
            running[i] = cutlass.Float32(0.0)
            if d_base + row < self.dim:
                running[i] = cutlass.Float32(state[slot, head, d_base + row,
                                                   col])

        skip_d = cutlass.Float32(d[head])

        for offset in range(start, end, self.chunk_size):
            valid = cutlass.min(self.chunk_size, end - offset)
            if valid == self.chunk_size:
                if tid < 32:
                    source = cute.domain_offset((d_base, offset),
                                                tensor_x[None, None, head])
                    source_tile = cute.zipped_divide(
                        source,
                        (self.dim_tile, self.chunk_size))[((None, None), (0,
                                                                          0))]
                    destination, source_tile = cpasync.tma_partition(
                        tma_x, 0, cute.make_layout(1),
                        cute.group_modes(x_shared, 0, 2),
                        cute.group_modes(source_tile, 0, 2))
                    if tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            x_barrier, self.dim_tile * self.chunk_size * 2)
                    cute.copy(tma_x,
                              source_tile,
                              destination,
                              tma_bar_ptr=x_barrier)
            else:
                # A packed sequence boundary is not a tensor-map OOB boundary.
                for idx in range(tid, self.chunk_size * self.dim_tile,
                                 self.threads):
                    token = idx // self.dim_tile
                    channel = idx % self.dim_tile
                    value = cutlass.Float16(0.0)
                    if token < valid and d_base + channel < self.dim:
                        value = x[offset + token, head, d_base + channel]
                    x_shared[channel, token] = value
            if valid == self.chunk_size:
                if tid < 32:
                    if tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bc_barrier, self.chunk_size * self.dstate * 4)
                    self.copy_bc_tma(tma_c, tensor_c, full_c_tma, offset,
                                     group, bc_barrier)
                    self.copy_bc_tma(tma_b, tensor_b, full_b_tma, offset,
                                     group, bc_barrier)
            else:
                for idx in range(tid, self.chunk_size * self.dstate,
                                 self.threads):
                    token = idx // self.dstate
                    col = idx % self.dstate
                    cv = cutlass.Float16(0.0)
                    bv = cutlass.Float16(0.0)
                    if token < valid:
                        cv = c[offset + token, group, col]
                        bv = b[offset + token, group, col]
                    full_c[token, col] = cv
                    full_b[token, col] = bv

            if tid < self.chunk_size:
                prefix_value = cutlass.Float32(0.0)
                delta_value = cutlass.Float32(0.0)
                if tid < valid:
                    prefix_value = prefix[offset + tid, head]
                    delta_value = dt_processed[offset + tid, head]
                prefix_shared[tid] = prefix_value
                delta_shared[tid] = delta_value

            out_acc = cute.make_rmem_tensor(out_coords.shape, cutlass.Float32)
            out_acc.fill(0.0)
            cb_acc = cute.make_rmem_tensor(cb_coords.shape, cutlass.Float32)
            cb_acc.fill(0.0)
            if valid == self.chunk_size:
                cute.arch.mbarrier_wait(bc_barrier, bc_phase)
                bc_phase = bc_phase ^ 1
            cute.arch.barrier()
            # Each FP16 state slice is consumed before its storage is reused.
            for kt in cutlass.range_constexpr(self.dstate // self.k_tile):
                for i in cutlass.range_constexpr(cute.size(running)):
                    row, col = state_coords[i]
                    if col // self.k_tile == kt:
                        state_half[row, col % self.k_tile] = cutlass.Float16(
                            running[i])
                cute.arch.barrier()
                lhs_cb = cute.local_tile(full_c,
                                         (self.chunk_size, self.k_tile),
                                         (0, kt))
                rhs_cb = cute.local_tile(full_b,
                                         (self.chunk_size, self.k_tile),
                                         (0, kt))
                self.multiply_pair(mma, lhs_cb, rhs_cb, state_half, cb_acc,
                                   out_acc)
                cute.arch.barrier()
            for i in cutlass.range_constexpr(cute.size(cb_acc)):
                row, col = cb_coords[i]
                weighted = cutlass.Float32(0.0)
                if row < valid and col <= row:
                    decay = cute.exp(prefix_shared[row] - prefix_shared[col])
                    weighted = cb_acc[i] * decay * delta_shared[col]
                cb_half[row, col] = cutlass.Float16(weighted)
            cute.arch.barrier()

            for i in cutlass.range_constexpr(cute.size(out_acc)):
                row, _ = out_coords[i]
                if row < valid:
                    out_acc[i] = out_acc[i] * cute.exp(prefix_shared[row])
            if valid == self.chunk_size:
                cute.arch.mbarrier_wait(x_barrier, x_phase)
                x_phase = x_phase ^ 1
            cute.arch.barrier()
            # Load the epilogue operand ahead of the output MMA.
            x_output_regs = cute.make_rmem_tensor(out_coords.shape,
                                                  cutlass.Float16)
            cute.copy(copy_x_output,
                      copy_x_output.get_slice(tid).partition_S(x_output_view),
                      copy_x_output.retile(x_output_regs))
            # CB and X stay read-only for this entire reduction.
            for k_base in range(0, self.chunk_size, self.k_tile):
                lhs_out = cute.local_tile(cb_half,
                                          (self.chunk_size, self.k_tile),
                                          (0, k_base // self.k_tile))
                x_part = cute.local_tile(x_shared,
                                         (self.dim_tile, self.k_tile),
                                         (0, k_base // self.k_tile))
                self.multiply(mma,
                              lhs_out,
                              x_part,
                              out_acc,
                              rhs_transpose=True,
                              post_barrier=False,
                              pre_barrier=False)
            cute.arch.barrier()
            output_regs = cute.make_rmem_tensor(out_coords.shape,
                                                cutlass.Float16)
            for i in cutlass.range_constexpr(cute.size(out_acc)):
                value = out_acc[i] + skip_d * cutlass.Float32(x_output_regs[i])
                output_regs[i] = cutlass.Float16(value)
            if valid == self.chunk_size and d_base + self.dim_tile <= self.dim:
                output_view = cute.domain_offset((offset, d_base),
                                                 output[None, head, None])
                output_tile = cute.local_tile(output_view,
                                              (self.chunk_size, self.dim_tile),
                                              (0, 0))
                # Packed token/head strides and d_base are 16-byte aligned.
                output_tile = cute.make_tensor(output_tile.iterator.align(16),
                                               output_tile.layout)
                cute.copy(
                    copy_x_output, copy_x_output.retile(output_regs),
                    copy_x_output.get_slice(tid).partition_D(output_tile))
            else:
                for i in cutlass.range_constexpr(cute.size(out_acc)):
                    row, col = out_coords[i]
                    if row < valid and d_base + col < self.dim:
                        output[offset + row, head,
                               d_base + col] = output_regs[i]

            last = prefix_shared[valid - 1]
            lane = tid % 32
            weight0 = cutlass.Float32(0.0)
            weight1 = cutlass.Float32(0.0)
            if lane < valid:
                weight0 = cute.exp(last -
                                   prefix_shared[lane]) * delta_shared[lane]
            if lane + 32 < valid:
                weight1 = cute.exp(
                    last - prefix_shared[lane + 32]) * delta_shared[lane + 32]
            if cutlass.const_expr(self.chunk_size == 128):
                weight2 = cutlass.Float32(0.0)
                weight3 = cutlass.Float32(0.0)
                if lane + 64 < valid:
                    weight2 = cute.exp(last - prefix_shared[lane + 64]
                                       ) * delta_shared[lane + 64]
                if lane + 96 < valid:
                    weight3 = cute.exp(last - prefix_shared[lane + 96]
                                       ) * delta_shared[lane + 96]
            contribution = cute.make_rmem_tensor(state_coords.shape,
                                                 cutlass.Float32)
            contribution.fill(0.0)
            for k_base in range(0, self.chunk_size, self.k_tile):
                warp_weights = weight0
                if k_base >= 32:
                    warp_weights = weight1
                if cutlass.const_expr(self.chunk_size == 128):
                    if k_base >= 64:
                        warp_weights = weight2
                    if k_base >= 96:
                        warp_weights = weight3
                x_part = cute.local_tile(x_shared,
                                         (self.dim_tile, self.k_tile),
                                         (0, k_base // self.k_tile))
                if valid == self.chunk_size:
                    b_view = cute.domain_offset((k_base, 0), full_b)
                    b_tile = cute.local_tile(b_view,
                                             (self.k_tile, self.dstate),
                                             (0, 0))
                    b_source = thread_b_vector.partition_S(b_tile)
                    b_regs = cute.make_rmem_tensor(b_source.shape,
                                                   cutlass.Float16)
                    cute.copy(copy_b_vector, b_source, b_regs)
                    for i in cutlass.range_constexpr(cute.size(b_regs)):
                        col, row = b_vector_coords[i]
                        weight = cute.arch.shuffle_sync(
                            warp_weights, (k_base + col) % 32)
                        b_regs[i] = cutlass.Float16(
                            cutlass.Float32(b_regs[i]) * weight)
                    b_destination = thread_b_vector.partition_D(
                        cute.make_tensor(rhs_storage.iterator,
                                         state_layout.outer))
                    cute.copy(copy_b_vector, b_regs, b_destination)
                else:
                    for idx in range(tid, self.dstate * self.k_tile,
                                     self.threads):
                        row = idx % self.dstate
                        col = idx // self.dstate
                        weight = cute.arch.shuffle_sync(
                            warp_weights, (k_base + col) % 32)
                        bv = cutlass.Float16(0.0)
                        if k_base + col < valid:
                            bv = cutlass.Float16(
                                cutlass.Float32(full_b[k_base + col, row]) *
                                weight)
                        rhs_state[row, col] = bv
                self.multiply(mma,
                              x_part,
                              rhs_state,
                              contribution,
                              lhs_transpose=True,
                              rhs_transpose=True)
            decay = cute.exp(last)
            for i in cutlass.range_constexpr(cute.size(running)):
                running[i] = decay * running[i] + contribution[i]

        if start < end:
            for i in cutlass.range_constexpr(cute.size(running)):
                row, col = state_coords[i]
                if d_base + row < self.dim:
                    state[slot, head, d_base + row,
                          col] = cutlass.Float16(running[i])


def export_packed_ssd(dim, dstate, output_dir, file_name, function_prefix,
                      gpu_arch):
    """Export a softplus/D/bias-enabled packed runner with dynamic T/H/G/batch."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from cutedsl_utils import aot_placeholders

    def tensor(dtype, shape, divisibility=1):
        value = aot_placeholders.make_compact_tensor(dtype,
                                                     shape,
                                                     assumed_align=16)
        value = value.mark_layout_dynamic(leading_dim=len(shape) - 1)
        return value.mark_compact_shape_dynamic(len(shape) - 1,
                                                divisibility=divisibility)

    half = cutlass.Float16
    float32 = cutlass.Float32
    int32 = cutlass.Int32
    args = [
        tensor(half, (1, 8, dim), dim),
        tensor(half, (1, 8)),
        tensor(float32, (8, )),
        tensor(half, (1, 1, dstate), dstate),
        tensor(half, (1, 1, dstate), dstate),
        tensor(half, (8, )),
        tensor(half, (8, )),
        tensor(int32, (2, )),
        tensor(int32, (1, )),
        tensor(half, (1, 8, dim, dstate), dstate),
        tensor(half, (1, 8, dim), dim),
        tensor(float32, (1, 8)),
        tensor(float32, (1, 8)),
        aot_placeholders.make_stream(),
    ]
    options = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else "")
    kernel = PackedSSD(dim, dstate)
    compiled = cute.compile(kernel, *args, options=options)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=output_dir,
                         file_name=file_name,
                         function_prefix=function_prefix)
    # Bind the runner's launch bound to the configuration compiled into this AOT variant.
    header_path = pathlib.Path(output_dir) / f"{file_name}.h"
    with header_path.open("a") as header:
        header.write(
            "\n#ifdef __cplusplus\n"
            f"static constexpr int32_t {function_prefix}_kSCAN_GROUP_SIZE = "
            f"{kernel.chunk_size * kernel.scan_chunks};\n"
            "#endif\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_only", action="store_true", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--dstate", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name")
    parser.add_argument("--function_prefix")
    parser.add_argument("--gpu_arch", default="")
    args = parser.parse_args()
    name = f"ssd_prefill_sm12x_d{args.dim}_n{args.dstate}"
    export_packed_ssd(args.dim, args.dstate, args.output_dir, args.file_name
                      or name, args.function_prefix or name, args.gpu_arch)
