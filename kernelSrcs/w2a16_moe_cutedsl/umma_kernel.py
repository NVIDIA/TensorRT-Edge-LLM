# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM110 IQ2_XXS/BF16 MoE projections with UMMA and TMEM operands."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from w2a16_moe_cutedsl import intrinsics
from w2a16_moe_cutedsl import umma_intrinsics as ops


class W2A16UmmaKernel:
    """M128/N{tokens}/K32 projections with the decoded weight operand in TMEM."""

    def __init__(self,
                 tokens,
                 stages=2,
                 specialized=False,
                 groups=1,
                 b_stages=None,
                 pair=False,
                 fused=False,
                 lag=4,
                 group_chunks=1,
                 lut_copies=1,
                 prefetch=1):
        if specialized and stages not in (2, 4):
            raise ValueError(
                "Warp specialization requires a two- or four-stage tile")
        if groups not in (1, 2) or (groups == 2 and not specialized):
            raise ValueError("Two decode groups require warp specialization")
        # One decode thread stages each token's route and weight.
        if tokens % 16 or not 16 <= tokens <= 128 * groups:
            raise ValueError(
                "UMMA token tile must be a multiple of 16 in [16, 128 * groups]")
        if stages not in (1, 2) and not specialized:
            raise ValueError("UMMA pipeline supports one or two stages")
        self.tokens = tokens
        self.stages = stages
        self.specialized = specialized
        self.groups = groups
        # A CTA pair issues M256 MMAs: each CTA decodes its own 128 weight rows and
        # loads half of the token rows; the even CTA issues the MMAs.
        self.pair = pair
        if pair and not specialized:
            raise ValueError("CTA pairs require warp specialization")
        self.load_tokens = tokens // 2 if pair else tokens
        # A fused tile runs FC1 and FC2 in one persistent launch. An item is the 128 output
        # columns of each CTA in a pair, or of a single CTA, so FC1 has intermediate / 128 CTA
        # tiles per unit and FC2 has eight.
        self.fused = fused
        if fused and not specialized:
            raise ValueError("Fused projections require warp specialization")
        self.item_ctas = 2 if pair else 1
        self.fc2_items = 8 // self.item_ctas
        # Fused work units are groups of up to group_chunks token chunks of one expert; FC2 of a
        # unit waits only for that unit's FC1 items, placed lag units earlier.
        self.lag = lag
        self.group_chunks = group_chunks
        # Resident CTAs on Thor's 20 SMs: TMEM fits two pair CTAs or four single CTAs per SM.
        self.fused_ctas = 40 if pair else 80
        # Routing header: per-expert counts, the active count and list, two claim counters at
        # 1277-1279, then one FC1 completion counter per fused work unit.
        self.header = 2304
        self.ready_offset = 1280
        self.claim_index = 1277
        # Specialized: 128 * groups decode/epilogue threads, then one activation-load warp per decode
        # group and one MMA warp. Each load lane keeps its rows' source offsets in registers.
        self.decoders = 128 * groups
        self.loaders = 32 * groups
        self.threads = self.decoders + self.loaders + 32 if specialized else 128
        # A single-stage tile double-buffers activations so the next K tile's copy overlaps this one;
        # a specialized tile may hold more activation stages than TMEM weight stages.
        self.b_stages = b_stages or (2 if stages == 1 else stages)
        if self.b_stages != (2 if stages == 1 else stages) and not specialized:
            raise ValueError("Extra activation stages require warp specialization")
        # With as many activation as weight stages, a stage's MMA commit also frees its activations.
        self.b_released = self.b_stages != stages
        # Specialized barriers: a_full and mma_done per weight stage, b_full (and b_empty when
        # released separately) per activation stage, then acc_full and acc_empty.
        barriers = (2 * stages + self.b_stages *
                    (2 if self.b_released else 1) + 2) if specialized else stages
        self.tile_k = 32
        self.b_bytes = self.load_tokens * 128
        # Activation stages start at offset zero.
        self.bar_offset = self.b_bytes * self.b_stages
        self.tmem_offset = self.bar_offset + 8 * barriers
        # The TMEM address, then two slots for the next claimed work item.
        self.ids_offset = self.tmem_offset + 16
        # The specialized epilogue reads staged row offsets with 16-byte loads.
        assert not specialized or self.ids_offset % 16 == 0
        self.weights_offset = self.ids_offset + self.tokens * 4
        self.lut_offset = (self.weights_offset + self.tokens * 2 +
                           15) // 16 * 16
        # The selector LUT holds lut_copies interleaved copies: entry e of copy c sits at
        # e * 8 * lut_copies + c * 8, so copy c occupies bank pairs c (mod lut_copies) and each
        # lane reads copy lane % lut_copies; 64-bit loads from different copies never conflict.
        self.lut_copies = lut_copies
        # Non-specialized tiles issue a K tile's weight loads this many tiles before decoding it.
        self.prefetch = prefetch
        # Fused tiles: two item_full barriers, a two-slot ring of claimed work items, the
        # exclusive prefix of work units over 512 expert slots, and 32 per-lane partial sums.
        self.item_offset = self.lut_offset + 2048 * lut_copies
        self.prefix_offset = self.item_offset + 32
        self.partial_offset = self.prefix_offset + 2064
        self.shared_bytes = (self.item_offset +
                             (32 + 2064 + 128 if fused else 0) +
                             1023) // 1024 * 1024
        # Each stage's K32 weight operand takes sixteen TMEM columns after the accumulator.
        used = self.tokens + 16 * stages
        self.tmem_columns = max(32, 1 << (used - 1).bit_length())
        # CTAs whose expert has at most 16 routes use a 16-column accumulator,
        # so the smaller TMEM allocation lets more CTAs share an SM.
        self.adaptive = not specialized and tokens > 16
        small = 16 + 16 * stages
        self.small_tmem_columns = max(32, 1 << (small - 1).bit_length())
        # Non-specialized CTAs claim (expert, 128-column) items, so experts
        # without routes launch no CTAs. Tuned on Thor's 20 SMs.
        self.persistent_ctas = 200

    @cute.jit
    def __call__(self, x: cute.Pointer, w1: cute.Pointer, w2: cute.Pointer,
                 s1: cute.Pointer, s2: cute.Pointer, lut: cute.Pointer,
                 lists: cute.Pointer, counts: cute.Pointer,
                 probability: cute.Pointer, activated: cute.Pointer,
                 slots: cute.Pointer, output: cute.Pointer, ids: cute.Pointer,
                 m: Int32, e: Int32, intermediate: Int32,
                 stream: cuda.CUstream):
        routes = m * 9
        self.pack(ids, counts, lists, routes,
                  e).launch(grid=((routes + 255) // 256, 1, 1),
                            block=(256, 1, 1),
                            stream=stream)
        projection = self.projection
        grid_y = cute.min(routes, e)
        if cutlass.const_expr(self.specialized):
            projection = self.projection_ws
        for i in cutlass.range_constexpr(5):
            size = (1536, 2048, 2304, 2560, 2816)[i]
            if intermediate == size:
                fc1_grid = (size // 128, grid_y, 1)
                fc2_grid = (8, grid_y, 1)
                if cutlass.const_expr(not self.specialized):
                    fc1_grid = (cute.min(grid_y * (size // 128),
                                         self.persistent_ctas), 1, 1)
                    fc2_grid = (cute.min(grid_y * 8,
                                         self.persistent_ctas), 1, 1)
                if cutlass.const_expr(self.fused and self.pair):
                    self.fused_ws(x, w1, s1, w2, s2, lut, probability,
                                  activated, slots, counts, lists, routes, e,
                                  size).launch(grid=(self.fused_ctas, 1, 1),
                                               block=(self.threads, 1, 1),
                                               cluster=(2, 1, 1),
                                               stream=stream)
                elif cutlass.const_expr(self.fused):
                    self.fused_ws(x, w1, s1, w2, s2, lut, probability,
                                  activated, slots, counts, lists, routes, e,
                                  size).launch(grid=(self.fused_ctas, 1, 1),
                                               block=(self.threads, 1, 1),
                                               stream=stream)
                elif cutlass.const_expr(self.pair):
                    projection(x, w1, s1, lut, probability, activated, counts,
                               lists, routes, e, size,
                               True).launch(grid=fc1_grid,
                                            block=(self.threads, 1, 1),
                                            cluster=(2, 1, 1),
                                            stream=stream)
                    projection(activated, w2, s2, lut, probability, slots,
                               counts, lists, routes, e, size,
                               False).launch(grid=fc2_grid,
                                             block=(self.threads, 1, 1),
                                             cluster=(2, 1, 1),
                                             stream=stream)
                else:
                    projection(x, w1, s1, lut, probability, activated, counts,
                               lists, routes, e, size,
                               True).launch(grid=fc1_grid,
                                            block=(self.threads, 1, 1),
                                            stream=stream)
                    projection(activated, w2, s2, lut, probability, slots,
                               counts, lists, routes, e, size,
                               False).launch(grid=fc2_grid,
                                             block=(self.threads, 1, 1),
                                             stream=stream)
        self.sum_routes(slots, output,
                        m * 1024).launch(grid=((m * 1024 + 255) // 256, 1, 1),
                                         block=(256, 1, 1),
                                         stream=stream)

    @cute.kernel
    def pack(self, ids: cute.Pointer, counts: cute.Pointer,
             lists: cute.Pointer, routes: Int32, experts: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        r = Int32(bx) * 256 + Int32(tx)
        id_tensor = cute.make_tensor(ids, cute.make_layout(routes))
        count_tensor = cute.make_tensor(counts, cute.make_layout(self.header))
        list_tensor = cute.make_tensor(lists,
                                       cute.make_layout(experts * routes))
        # FC1 and FC2 work-item claim counters.
        if r < 3:
            count_tensor[1277 + r] = 0
        if r < 256:
            for j in cutlass.range_constexpr(4):
                count_tensor[self.ready_offset + 256 * j + r] = 0
        if r < routes:
            expert = id_tensor[r]
            if expert >= 0 and expert < experts:
                slot = intrinsics.atomic_add_global_i32(
                    intrinsics.get_ptr_as_int64(count_tensor, expert),
                    Int32(1))
                list_tensor[expert * routes + slot] = r
                if slot == 0:
                    active = intrinsics.atomic_add_global_i32(
                        intrinsics.get_ptr_as_int64(count_tensor, experts),
                        Int32(1))
                    count_tensor[experts + 1 + active] = expert

    @cute.jit
    def load_records(self, q: cute.Tensor, bases: cute.Tensor, expert: Int32,
                     first_row: Int32, tid: Int32, start: Int32,
                     n: cutlass.Constexpr, k: cutlass.Constexpr):
        data = cute.make_rmem_tensor((128 * self.tile_k // 32 // 128, 3),
                                     Uint32)
        for record in cutlass.range_constexpr(128 * self.tile_k // 32 // 128):
            group = tid + record * 128
            row = first_row + group // (self.tile_k // 32)
            column = start + group % (self.tile_k // 32) * 32
            row_lane = row % 8 * 2 + row % 16 // 8
            offset = (expert * (n // 64) + row // 64) * (
                k * 4) + row // 16 % 4 * 16 + row_lane + column * 4
            address = intrinsics.get_ptr_as_int64(q, offset)
            data[record,
                 0] = ops.ptx('ld.global.nc.L1::evict_first.u32 $0, [$1];',
                              '=r,l', (address, ), ('u32', ))
            data[record, 1] = ops.ptx(
                'ld.global.nc.L1::evict_first.u32 $0, [$1+256];', '=r,l',
                (address, ), ('u32', ))
            base_index = (expert * (k // 256) +
                          start // 256) * n + row // 16 * 16 + row_lane
            data[record, 2] = ops.ptx(
                'ld.global.u16 $0, [$1];', '=r,l',
                (intrinsics.get_ptr_as_int64(bases, base_index), ), ('u32', ))
        return data

    @cute.jit
    def copy_activations(self, b_smem: Uint32, inputs: cute.Tensor,
                         tile: Int32, width: Int32, route_start: Int32,
                         count: Int32, tid: Int32):
        for record in cutlass.range_constexpr(
            (self.tokens * self.tile_k // 8 + 127) // 128):
            group = tid + record * 128
            if group < width * (self.tile_k // 8):
                row = group // (self.tile_k // 8)
                column = group % (self.tile_k // 8) * 8
                valid = Int32(0)
                if route_start + row < count:
                    valid = Int32(16)
                dest = b_smem + (row * 64 + (column ^ ((row & 7) * 8))) * 2
                ops.ptx('cp.async.cg.shared.global [$0], [$1], 16, $2;',
                        'r,l,r', (dest, inputs[record] +
                                  Int64(tile * self.tile_k * 2), valid))
        ops.ptx('cp.async.commit_group;', '')

    @cute.jit
    def wait_stage(self, shared: Uint32, phases: Int32, stage: Int32):
        """Wait for the oldest committed MMA of a stage; return phases with that stage's parity flipped."""
        ops.ptx(
            '{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
            'r,r', (shared + self.bar_offset + Uint32(stage) * 8,
                    (phases >> stage) & 1))
        ops.ptx('tcgen05.fence::after_thread_sync;', '')
        return phases ^ (Int32(1) << stage)

    @cute.kernel
    def projection(self, x: cute.Pointer, q_ptr: cute.Pointer,
                   base_ptr: cute.Pointer, lut_ptr: cute.Pointer,
                   probability: cute.Pointer, y: cute.Pointer,
                   counts: cute.Pointer, lists: cute.Pointer, routes: Int32,
                   experts: Int32, intermediate: cutlass.Constexpr,
                   first: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        n = intermediate if first else 1024
        k = 1024 if first else intermediate
        count_tensor = cute.make_tensor(counts, cute.make_layout(self.header))
        active = count_tensor[experts]
        items = active * (n // 128)
        if Int32(bx) < items:
            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate_tensor(Uint32,
                                                cute.make_layout(
                                                    self.shared_bytes // 4),
                                                byte_alignment=1024)
            shared = Uint32(intrinsics.shared_ptr_to_u32(storage.iterator))
            bar = shared + self.bar_offset
            lut = shared + self.lut_offset
            lookup = cute.make_tensor(lut_ptr, cute.make_layout(512))
            for index in cutlass.range_constexpr(4):
                j = tid + index * 128
                for copy in cutlass.range_constexpr(self.lut_copies):
                    ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                            (lut + j // 2 * 8 * self.lut_copies + copy * 8 +
                             j % 2 * 4, Uint32(lookup[j])))
            if tid == 0:
                for stage in cutlass.range_constexpr(self.stages):
                    ops.ptx('mbarrier.init.shared::cta.b64 [$0], 1;', 'r',
                            (bar + stage * 8, ))
            cute.arch.sync_threads()
            list_tensor = cute.make_tensor(lists,
                                           cute.make_layout(experts * routes))
            prob = cute.make_tensor(probability, cute.make_layout(routes))
            inp = cute.make_tensor(x, cute.make_layout(Int64(routes) * k))
            out = cute.make_tensor(y, cute.make_layout(Int64(routes) * n))
            q = cute.make_tensor(
                q_ptr, cute.make_layout(Int64(experts) * n * k // 16))
            bases = cute.make_tensor(
                base_ptr, cute.make_layout(Int64(experts) * n * k // 256))
            phases = Int32(0)
            gx, _, _ = cute.arch.grid_dim()
            # Every route count is at most M, so M <= 16 needs only the
            # 16-column accumulator.
            columns = Int32(self.tmem_columns)
            if cutlass.const_expr(self.adaptive):
                if routes <= 16 * 9:
                    columns = Int32(self.small_tmem_columns)
            if tid < 32:
                ops.ptx(
                    'tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;',
                    'r,r', (shared + self.tmem_offset, columns))
                ops.ptx(
                    'tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;',
                    '')
            cute.arch.sync_threads()
            tmem = ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                           (shared + self.tmem_offset, ), ('u32', ))
            cute.arch.sync_threads()
            claim = intrinsics.get_ptr_as_int64(count_tensor,
                                                1278 if first else 1279)
            item = Int32(bx)
            parity = Int32(0)
            while item < items:
                next_slot = shared + self.tmem_offset + 4 + Uint32(parity) * 4
                if tid == 0:
                    claimed = intrinsics.atomic_add_global_i32(claim, Int32(1))
                    ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                            (next_slot, Uint32(claimed + Int32(gx))))
                column0 = item % (n // 128) * 128
                expert = count_tensor[experts + 1 + item // (n // 128)]
                count = count_tensor[expert]
                accumulator = Int32(self.tokens)
                if cutlass.const_expr(self.adaptive):
                    if count <= 16:
                        accumulator = Int32(16)
                for route_start in cutlass.range(0, count, self.tokens):
                    if tid < self.tokens and route_start + tid < count:
                        route = list_tensor[expert * routes + route_start +
                                            tid]
                        ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                                (shared + self.ids_offset + tid * 4,
                                 Uint32(route)))
                    if cutlass.const_expr(not first):
                        if tid < self.tokens:
                            weight = Float32(0)
                            if route_start + tid < count:
                                route = list_tensor[expert * routes +
                                                    route_start + tid]
                                weight = Float32(prob[route])
                            ops.ptx(
                                '{ .reg .b16 value; cvt.rn.bf16.f32 value, $1; st.shared.b16 [$0], value; }',
                                'r,f', (shared + self.weights_offset + tid * 2,
                                        weight))
                    cute.arch.sync_threads()
                    # MMA N, activation rows and epilogue columns cover this chunk's routes rounded to 16.
                    width = Int32(self.tokens)
                    if cutlass.const_expr(self.adaptive):
                        width = cute.min(width,
                                         (count - route_start + 15) // 16 * 16)
                    inputs = cute.make_rmem_tensor(
                        ((self.tokens * self.tile_k // 8 + 127) // 128, ),
                        Int64)
                    for record in cutlass.range_constexpr(
                        (self.tokens * self.tile_k // 8 + 127) // 128):
                        group = tid + record * 128
                        row = group // (self.tile_k // 8)
                        column = group % (self.tile_k // 8) * 8
                        route = Int32(0)
                        if group < self.tokens * self.tile_k // 8 and route_start + row < count:
                            route = Int32(
                                ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                                        (shared + self.ids_offset + row * 4, ),
                                        ('u32', )))
                        input_row = route // 9 if first else route
                        inputs[record] = intrinsics.get_ptr_as_int64(
                            inp,
                            Int64(input_row) * k + column)
                    tiles = k // self.tile_k
                    assert tiles % 2 == 0, "drain assumes an even K-tile count"
                    # K tile t uses register buffer t % buffers, so loads issued prefetch tiles
                    # ahead are first read where they are decoded.
                    buffers = self.prefetch + 1
                    assert tiles % buffers == 0
                    records = []
                    for d in cutlass.range_constexpr(buffers):
                        buffer = cute.make_rmem_tensor(
                            (128 * self.tile_k // 32 // 128, 3), Uint32)
                        if d < self.prefetch:
                            buffer.store(
                                self.load_records(q, bases, expert, column0,
                                                  tid, d * self.tile_k, n,
                                                  k).load())
                        records.append(buffer)
                    if cutlass.const_expr(self.stages == 1):
                        # The even K-tile count ends each route chunk on stage 1, leaving stage 0 free.
                        self.copy_activations(shared, inputs, Int32(0), width,
                                              route_start, count, tid)
                    for base in cutlass.range(0, tiles, buffers, unroll=1):
                        for j in cutlass.range_constexpr(buffers):
                            tile = base + j
                            stage = tile % self.stages
                            b_smem = shared + Uint32(
                                tile % self.b_stages) * self.b_bytes
                            a_tmem = tmem + Uint32(
                                accumulator) + Uint32(stage) * 16
                            if tile + self.prefetch < tiles:
                                records[(j + self.prefetch) % buffers].store(
                                    self.load_records(
                                        q, bases, expert, column0, tid,
                                        (tile + self.prefetch) * self.tile_k,
                                        n, k).load())
                            current = records[j]
                            if cutlass.const_expr(self.stages == 2):
                                if tile >= 2:
                                    phases = self.wait_stage(
                                        shared, phases, stage)
                            if cutlass.const_expr(self.stages == 1):
                                # The previous tile's MMA has completed, so its activation stage is free.
                                if tile + 1 < tiles:
                                    self.copy_activations(
                                        shared + Uint32(
                                            (tile + 1) % 2) * self.b_bytes,
                                        inputs, tile + 1, width, route_start,
                                        count, tid)
                                else:
                                    ops.ptx('cp.async.commit_group;', '')
                            else:
                                self.copy_activations(b_smem, inputs, tile,
                                                      width, route_start,
                                                      count, tid)
                            for record in cutlass.range_constexpr(
                                    128 * self.tile_k // 32 // 128):
                                lane_lut = lut + Uint32(
                                    tid % self.lut_copies * 8)
                                values = ops.decode_record(
                                    current[record, 0], current[record, 1],
                                    current[record,
                                            2], lane_lut, self.lut_copies)
                                address = a_tmem + ((tid // 32 * 32) << 16)
                                ops.ptx(
                                    'tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {'
                                    + ','.join('$' + str(v)
                                               for v in range(1, 17)) + '};',
                                    ','.join(['r'] * 17), (address, *values))
                            if cutlass.const_expr(self.stages == 1):
                                ops.ptx(
                                    'cp.async.wait_group 1; fence.proxy.async.shared::cta;',
                                    '')
                            else:
                                ops.ptx(
                                    'cp.async.wait_group 0; fence.proxy.async.shared::cta;',
                                    '')
                            ops.ptx('tcgen05.wait::st.sync.aligned;', '')
                            cute.arch.sync_threads()
                            if tid == 0:
                                ops.ptx('tcgen05.fence::after_thread_sync;',
                                        '')
                                b = Uint64(b_smem >> 4) | Uint64(
                                    1 << 16) | Uint64(64 << 32) | Uint64(
                                        2 << 61)
                                instruction = Uint32(
                                    (1 << 4) | (1 << 7) | (1 << 10)
                                    | (8 << 24)) | (Uint32(width // 8) << 17)
                                for step in cutlass.range_constexpr(
                                        self.tile_k // 16):
                                    accumulate = Int32(1)
                                    if tile == 0 and step == 0:
                                        accumulate = Int32(0)
                                    ops.ptx(
                                        '{ .reg .pred p; .reg .b32 mask; mov.b32 mask, 0; setp.ne.b32 p, $4, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], $2, $3, {mask,mask,mask,mask}, p; }',
                                        'r,r,l,r,r', (tmem, a_tmem + step * 8,
                                                      b + Uint64(step * 2),
                                                      instruction, accumulate))
                                ops.ptx(
                                    'tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];',
                                    'r', (bar + Uint32(stage) * 8, ))
                            if cutlass.const_expr(self.stages == 1):
                                phases = self.wait_stage(
                                    shared, phases, Int32(0))
                    if cutlass.const_expr(self.stages == 2):
                        # Both stages hold one unwaited commit after an even number of K tiles.
                        phases = self.wait_stage(shared, phases, Int32(0))
                        phases = self.wait_stage(shared, phases, Int32(1))
                    ops.ptx('tcgen05.fence::after_thread_sync;', '')
                    for chunk in cutlass.range(0, width, 8):
                        address = tmem + ((tid // 32 * 32) << 16) + chunk
                        values = ops.ptx(
                            'tcgen05.ld.sync.aligned.32x32b.x8.b32 {' +
                            ','.join('$' + str(v)
                                     for v in range(8)) + '}, [$8];',
                            ','.join(['=f'] * 8 + ['r']), (address, ),
                            ('f32', ) * 8)
                        ops.ptx('tcgen05.wait::ld.sync.aligned;', '')
                        for j in cutlass.range_constexpr(0, 8, 2):
                            column = chunk + j
                            value = ops.packed_epilogue(
                                values[j], values[j + 1],
                                shared + self.weights_offset + column * 2,
                                first)
                            for part in cutlass.range_constexpr(2):
                                if route_start + column + part < count:
                                    route = Int32(
                                        ops.ptx('ld.shared.u32 $0, [$1];',
                                                '=r,r',
                                                (shared + self.ids_offset +
                                                 (column + part) * 4, ),
                                                ('u32', )))
                                    output_address = intrinsics.get_ptr_as_int64(
                                        out,
                                        Int64(route) * n + column0 + tid)
                                    ops.ptx(
                                        '{ .reg .b16 lo, hi; mov.b32 {lo,hi}, $1; st.global.b16 [$0], '
                                        + ('hi' if part else 'lo') + '; }',
                                        'l,r', (output_address, value))
                    cute.arch.sync_threads()
                item = Int32(
                    ops.ptx('ld.shared.u32 $0, [$1];', '=r,r', (next_slot, ),
                            ('u32', )))
                parity = parity ^ 1
            if tid < 32:
                ops.ptx(
                    'tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;',
                    'r,r', (tmem, columns))

    @cute.jit
    def wait_barrier(self, barrier: Uint32, parity: Int32):
        ops.ptx(
            '{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
            'r,r', (barrier, parity & 1))

    @cute.jit
    def arrive_barrier(self, barrier: Uint32):
        ops.ptx('mbarrier.arrive.shared::cta.b64 _, [$0];', 'r', (barrier, ))

    @cute.jit
    def arrive_leader(self, barrier: Uint32):
        ops.ptx(
            '{ .reg .b32 remote; mapa.shared::cluster.u32 remote, $0, 0; mbarrier.arrive.shared::cluster.b64 _, [remote]; }',
            'r', (barrier, ))

    @cute.jit
    def commit(self, barrier: Uint32):
        if cutlass.const_expr(self.pair):
            ops.ptx(
                '{ .reg .b16 mask; mov.b16 mask, 3; tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], mask; }',
                'r', (barrier, ))
        else:
            ops.ptx(
                'tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];',
                'r', (barrier, ))

    def ws_barriers(self, shared):
        """Barrier addresses: a_full, mma_done, b_full, b_empty, acc_full, acc_empty."""
        a_full = shared + self.bar_offset
        mma_done = a_full + 8 * self.stages
        b_full = mma_done + 8 * self.stages
        b_empty = mma_done
        acc_full = b_full + 8 * self.b_stages
        if cutlass.const_expr(self.b_released):
            b_empty = b_full + 8 * self.b_stages
            acc_full = b_empty + 8 * self.b_stages
        return a_full, mma_done, b_full, b_empty, acc_full, acc_full + 8

    @cute.jit
    def ws_relayed(self):
        # The even CTA's a_full, b_full and acc_empty also count the odd CTA's relay.
        relayed = Int32(0)
        if cutlass.const_expr(self.pair):
            if cute.arch.block_idx_in_cluster() == 0:
                relayed = Int32(1)
        return relayed

    @cute.jit
    def ws_setup(self, shared: Uint32, lut_ptr: cute.Pointer, tid: Int32,
                 relayed: Int32):
        """Stage the LUT, allocate TMEM and initialize barriers; return the TMEM address."""
        a_full, mma_done, b_full, b_empty, acc_full, acc_empty = self.ws_barriers(
            shared)
        lut = shared + self.lut_offset
        lookup = cute.make_tensor(lut_ptr, cute.make_layout(512))
        if tid < 128:
            for index in cutlass.range_constexpr(4):
                j = tid + index * 128
                for copy in cutlass.range_constexpr(self.lut_copies):
                    ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                            (lut + j // 2 * 8 * self.lut_copies + copy * 8 +
                             j % 2 * 4, Uint32(lookup[j])))
        cta_group = 2 if self.pair else 1
        if tid < 32:
            ops.ptx(
                f'tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32 [$0], $1;',
                'r,r', (shared + self.tmem_offset, Int32(self.tmem_columns)))
            ops.ptx(
                f'tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned;',
                '')
        if tid == 0:
            for stage in cutlass.range_constexpr(self.stages):
                ops.ptx('mbarrier.init.shared::cta.b64 [$0], $1;', 'r,r',
                        (a_full + stage * 8, 128 + relayed))
                ops.ptx('mbarrier.init.shared::cta.b64 [$0], 1;', 'r',
                        (mma_done + stage * 8, ))
            for stage in cutlass.range_constexpr(self.b_stages):
                ops.ptx('mbarrier.init.shared::cta.b64 [$0], $1;', 'r,r',
                        (b_full + stage * 8, self.loaders + relayed))
                if cutlass.const_expr(self.b_released):
                    ops.ptx('mbarrier.init.shared::cta.b64 [$0], 1;', 'r',
                            (b_empty + stage * 8, ))
            ops.ptx('mbarrier.init.shared::cta.b64 [$0], 1;', 'r',
                    (acc_full, ))
            ops.ptx('mbarrier.init.shared::cta.b64 [$0], $1;', 'r,r',
                    (acc_empty, self.decoders + relayed))
            if cutlass.const_expr(self.fused):
                for slot in cutlass.range_constexpr(2):
                    ops.ptx('mbarrier.init.shared::cta.b64 [$0], 1;', 'r',
                            (shared + self.item_offset + slot * 8, ))
            if cutlass.const_expr(self.pair):
                ops.ptx('fence.mbarrier_init.release.cluster;', '')
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pair):
            ops.ptx(
                'barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;',
                '')
        return ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                       (shared + self.tmem_offset, ), ('u32', ))

    @cute.jit
    def ws_teardown(self, tmem: Uint32, tid: Int32):
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pair):
            # Neither CTA may free TMEM or exit while its peer still uses it.
            ops.ptx(
                'barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;',
                '')
        if tid < 32:
            ops.ptx(
                f'tcgen05.dealloc.cta_group::{2 if self.pair else 1}.sync.aligned.b32 $0, $1;',
                'r,r', (tmem, Int32(self.tmem_columns)))

    @cute.kernel
    def projection_ws(self, x: cute.Pointer, q_ptr: cute.Pointer,
                      base_ptr: cute.Pointer, lut_ptr: cute.Pointer,
                      probability: cute.Pointer, y: cute.Pointer,
                      counts: cute.Pointer, lists: cute.Pointer,
                      routes: Int32, experts: Int32,
                      intermediate: cutlass.Constexpr,
                      first: cutlass.Constexpr):
        """Prefill projection with decode, activation-load and MMA warps linked by mbarriers.

        Global K-tile g uses stage g % S; its use index g // S selects the barrier parity.
        With two decode groups, group h decodes the K tiles with g % 2 == h, so it owns the stages
        with s % 2 == h.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        n = intermediate if first else 1024
        k = 1024 if first else intermediate
        count_tensor = cute.make_tensor(counts, cute.make_layout(self.header))
        if Int32(by) < count_tensor[experts]:
            expert = count_tensor[experts + 1 + Int32(by)]
            count = count_tensor[expert]
            chunks = (count + self.tokens - 1) // self.tokens
            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate_tensor(Uint32,
                                                cute.make_layout(
                                                    self.shared_bytes // 4),
                                                byte_alignment=1024)
            shared = Uint32(intrinsics.shared_ptr_to_u32(storage.iterator))
            a_full, mma_done, b_full, b_empty, acc_full, acc_empty = self.ws_barriers(
                shared)
            lut = shared + self.lut_offset
            relayed = self.ws_relayed()
            tmem = self.ws_setup(shared, lut_ptr, tid, relayed)
            list_tensor = cute.make_tensor(lists,
                                           cute.make_layout(experts * routes))
            prob = cute.make_tensor(probability, cute.make_layout(routes))
            inp = cute.make_tensor(x, cute.make_layout(Int64(routes) * k))
            out = cute.make_tensor(y, cute.make_layout(Int64(routes) * n))
            q = cute.make_tensor(
                q_ptr, cute.make_layout(Int64(experts) * n * k // 16))
            bases = cute.make_tensor(
                base_ptr, cute.make_layout(Int64(experts) * n * k // 256))
            if tid < 128:
                self.decode_role(q, bases, list_tensor, prob, out, shared,
                                 tmem, lut, a_full, mma_done, acc_full,
                                 acc_empty, expert, count, chunks, tid,
                                 Int32(bx), routes, n, k, first, 0, Int32(0),
                                 Int32(0), Int32(0))
            elif tid < self.decoders:
                self.decode_role(q, bases, list_tensor, prob, out, shared,
                                 tmem, lut, a_full, mma_done, acc_full,
                                 acc_empty, expert, count, chunks, tid,
                                 Int32(bx), routes, n, k, first, 1, Int32(0),
                                 Int32(0), Int32(0))
            elif tid < self.decoders + self.loaders:
                self.load_role(inp, list_tensor, shared, b_full, b_empty,
                               expert, count, chunks, tid - self.decoders,
                               routes, k, first, Int32(0), Int32(0))
            elif tid == self.decoders + self.loaders:
                if cutlass.const_expr(self.pair):
                    if relayed == 1:
                        self.mma_role(shared, tmem, a_full, b_full, mma_done,
                                      b_empty, acc_full, acc_empty, chunks, k,
                                      Int32(0), Int32(0))
                    else:
                        self.relay_role(a_full, b_full, acc_empty, chunks, k,
                                        Int32(0), Int32(0))
                else:
                    self.mma_role(shared, tmem, a_full, b_full, mma_done,
                                  b_empty, acc_full, acc_empty, chunks, k,
                                  Int32(0), Int32(0))
            self.ws_teardown(tmem, tid)

    @cute.jit
    def fused_item(self, index: Int32, units: Int32,
                   fc1_items: cutlass.Constexpr):
        """Map a work-item index to (is FC1, work unit, column item).

        Unit u contributes fc1_items FC1 items, then fc2_items FC2 items placed after the FC1 items of
        unit u + lag, so every FC2 item follows all FC1 items it reads.
        """
        lag = Int32(self.lag)
        if units < lag:
            lag = units
        block = fc1_items + self.fc2_items
        head = lag * fc1_items
        body = (units - lag) * block
        fc1 = Int32(1)
        unit = index // fc1_items
        column = index % fc1_items
        if index >= head:
            rest = index - head
            if rest < body:
                unit = lag + rest // block
                column = rest % block
                if column >= fc1_items:
                    fc1 = Int32(0)
                    unit = unit - lag
                    column = column - fc1_items
            else:
                fc1 = Int32(0)
                unit = units - lag + (rest - body) // self.fc2_items
                column = (rest - body) % self.fc2_items
        return fc1, unit, column

    @cute.jit
    def fused_prefix(self, shared: Uint32, count_tensor: cute.Tensor,
                     experts: Int32, tid: Int32):
        """Store each active expert slot's first work unit; entries past the last slot hold the unit count."""
        if tid < 32:
            active = count_tensor[experts]
            units = cute.make_rmem_tensor((16, ), Int32)
            total = Int32(0)
            for j in cutlass.range_constexpr(16):
                slot = tid * 16 + j
                units[j] = Int32(0)
                if slot < active:
                    chunks = (count_tensor[count_tensor[experts + 1 + slot]] +
                              self.tokens - 1) // self.tokens
                    units[j] = (chunks + self.group_chunks -
                                1) // self.group_chunks
                total += units[j]
            ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                    (shared + self.partial_offset + tid * 4, total))
            ops.ptx('bar.warp.sync -1;', '')
            offset = Int32(0)
            for lane in cutlass.range_constexpr(32):
                if lane < tid:
                    offset += Int32(
                        ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                                (shared + self.partial_offset + lane * 4, ),
                                ('u32', )))
            for j in cutlass.range_constexpr(16):
                ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                        (shared + self.prefix_offset +
                         (tid * 16 + j) * 4, offset))
                offset += units[j]
            if tid == 31:
                ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                        (shared + self.prefix_offset + 512 * 4, offset))
        cute.arch.sync_threads()

    @cute.jit
    def prefix_at(self, shared: Uint32, slot: Int32):
        return Int32(
            ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                    (shared + self.prefix_offset + Uint32(slot) * 4, ),
                    ('u32', )))

    @cute.jit
    def fused_work(self, shared: Uint32, count_tensor: cute.Tensor,
                   index: Int32, units: Int32, experts: Int32,
                   fc1_items: cutlass.Constexpr):
        """Return (is FC1, unit, column item, expert, count, first chunk, chunks) of a work item."""
        fc1, unit, column = self.fused_item(index, units, fc1_items)
        # Active slots have at least one unit, so the prefix strictly increases up to the last slot.
        slot = Int32(0)
        for step in cutlass.range_constexpr(9):
            probe = slot + (256 >> step)
            if self.prefix_at(shared, probe) <= unit:
                slot = probe
        expert = count_tensor[experts + 1 + slot]
        count = count_tensor[expert]
        chunk0 = (unit - self.prefix_at(shared, slot)) * self.group_chunks
        chunks = (count + self.tokens - 1) // self.tokens - chunk0
        if chunks > self.group_chunks:
            chunks = Int32(self.group_chunks)
        return fc1, unit, column, expert, count, chunk0, chunks

    @cute.jit
    def publish_item(self, shared: Uint32, sequence: Int32, index: Int32):
        """Write a claimed item into each CTA's ring slot and signal its item_full barriers."""
        slot = Uint32(sequence % 2)
        value = shared + self.item_offset + 16 + slot * 4
        barrier = shared + self.item_offset + slot * 8
        if cutlass.const_expr(self.pair):
            ops.ptx(
                '{ .reg .b32 remote; mapa.shared::cluster.u32 remote, $0, 1; st.shared::cluster.u32 [remote], $2; '
                'mapa.shared::cluster.u32 remote, $1, 1; mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]; }',
                'r,r,r', (value, barrier, index))
        ops.ptx('st.shared.u32 [$0], $1;', 'r,r', (value, index))
        self.arrive_barrier(barrier)

    @cute.jit
    def next_item(self, shared: Uint32, sequence: Int32):
        slot = Uint32(sequence % 2)
        ops.ptx(
            '{ .reg .pred p; WAIT: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
            'r,r', (shared + self.item_offset + slot * 8, (sequence // 2) & 1))
        return Int32(
            ops.ptx('ld.shared.u32 $0, [$1];', '=r,r',
                    (shared + self.item_offset + 16 + slot * 4, ), ('u32', )))

    @cute.jit
    def fused_decode(self, q1: cute.Tensor, b1: cute.Tensor, q2: cute.Tensor,
                     b2: cute.Tensor, list_tensor: cute.Tensor,
                     prob: cute.Tensor, out1: cute.Tensor, out2: cute.Tensor,
                     count_tensor: cute.Tensor, shared: Uint32, tmem: Uint32,
                     tid: Int32, routes: Int32, experts: Int32, units: Int32,
                     intermediate: cutlass.Constexpr,
                     group: cutlass.Constexpr):
        a_full, mma_done, b_full, b_empty, acc_full, acc_empty = self.ws_barriers(
            shared)
        lut = shared + self.lut_offset
        rank = Int32(0)
        if cutlass.const_expr(self.pair):
            rank = cute.arch.block_idx_in_cluster()
        total = units * (intermediate //
                         (128 * self.item_ctas) + self.fc2_items)
        g0 = Int32(0)
        c0 = Int32(0)
        sequence = Int32(0)
        index = self.next_item(shared, sequence)
        while index < total:
            fc1, unit, column, expert, count, chunk0, chunks = self.fused_work(
                shared, count_tensor, index, units, experts,
                intermediate // (128 * self.item_ctas))
            bx = column * self.item_ctas + rank
            if fc1 == 1:
                self.decode_role(q1, b1, list_tensor, prob, out1, shared, tmem,
                                 lut, a_full, mma_done, acc_full, acc_empty,
                                 expert, count, chunks, tid, bx, routes,
                                 intermediate, 1024, True, group, g0, c0,
                                 chunk0)
                # Publish this CTA's 128 FC1 columns once every decode thread has stored them.
                ops.ptx('bar.sync 2, $0;', 'r', (Int32(self.decoders), ))
                if tid == 0:
                    ops.ptx(
                        'fence.acq_rel.gpu; red.relaxed.gpu.global.add.s32 [$0], 1;',
                        'l', (intrinsics.get_ptr_as_int64(
                            count_tensor, self.ready_offset + unit), ))
                g0 += chunks * (1024 // self.tile_k)
            else:
                self.decode_role(q2, b2, list_tensor, prob, out2, shared, tmem,
                                 lut, a_full, mma_done, acc_full, acc_empty,
                                 expert, count, chunks, tid, bx, routes, 1024,
                                 intermediate, False, group, g0, c0, chunk0)
                g0 += chunks * (intermediate // self.tile_k)
            c0 += chunks
            sequence += 1
            index = self.next_item(shared, sequence)

    @cute.kernel
    def fused_ws(self, x: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w2: cute.Pointer, s2: cute.Pointer, lut_ptr: cute.Pointer,
                 probability: cute.Pointer, activated: cute.Pointer,
                 slots: cute.Pointer, counts: cute.Pointer,
                 lists: cute.Pointer, routes: Int32, experts: Int32,
                 intermediate: cutlass.Constexpr):
        """Persistent FC1 and FC2 over work units; FC2 items wait on their unit's FC1 count.

        The even CTA's MMA thread claims items in increasing order one item ahead and broadcasts
        them through a two-slot ring. A slot is rewritten only after the MMA thread finishes the
        item two claims back, which every consumer must have read to feed those MMAs.
        """
        tidx, _, _ = cute.arch.thread_idx()
        tid = Int32(tidx)
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate_tensor(Uint32,
                                            cute.make_layout(
                                                self.shared_bytes // 4),
                                            byte_alignment=1024)
        shared = Uint32(intrinsics.shared_ptr_to_u32(storage.iterator))
        a_full, mma_done, b_full, b_empty, acc_full, acc_empty = self.ws_barriers(
            shared)
        relayed = self.ws_relayed()
        tmem = self.ws_setup(shared, lut_ptr, tid, relayed)
        count_tensor = cute.make_tensor(counts, cute.make_layout(self.header))
        self.fused_prefix(shared, count_tensor, experts, tid)
        units = self.prefix_at(shared, Int32(512))
        list_tensor = cute.make_tensor(lists,
                                       cute.make_layout(experts * routes))
        prob = cute.make_tensor(probability, cute.make_layout(routes))
        inp1 = cute.make_tensor(x, cute.make_layout(Int64(routes) * 1024))
        out1 = cute.make_tensor(activated,
                                cute.make_layout(Int64(routes) * intermediate))
        inp2 = cute.make_tensor(activated,
                                cute.make_layout(Int64(routes) * intermediate))
        out2 = cute.make_tensor(slots, cute.make_layout(Int64(routes) * 1024))
        weights = Int64(experts) * intermediate * 1024
        q1 = cute.make_tensor(w1, cute.make_layout(weights // 16))
        b1 = cute.make_tensor(s1, cute.make_layout(weights // 256))
        q2 = cute.make_tensor(w2, cute.make_layout(weights // 16))
        b2 = cute.make_tensor(s2, cute.make_layout(weights // 256))
        total = units * (intermediate //
                         (128 * self.item_ctas) + self.fc2_items)
        if tid < 128:
            self.fused_decode(q1, b1, q2, b2, list_tensor, prob, out1, out2,
                              count_tensor, shared, tmem, tid, routes, experts,
                              units, intermediate, 0)
        elif tid < self.decoders:
            self.fused_decode(q1, b1, q2, b2, list_tensor, prob, out1, out2,
                              count_tensor, shared, tmem, tid, routes, experts,
                              units, intermediate, 1)
        elif tid < self.decoders + self.loaders:
            g0 = Int32(0)
            sequence = Int32(0)
            index = self.next_item(shared, sequence)
            while index < total:
                fc1, unit, column, expert, count, chunk0, chunks = self.fused_work(
                    shared, count_tensor, index, units, experts,
                    intermediate // (128 * self.item_ctas))
                if fc1 == 1:
                    self.load_role(inp1, list_tensor, shared, b_full, b_empty,
                                   expert, count, chunks, tid - self.decoders,
                                   routes, 1024, True, g0, chunk0)
                    g0 += chunks * (1024 // self.tile_k)
                else:
                    ops.ptx(
                        '{ .reg .pred p; .reg .s32 v; WAIT: ld.acquire.gpu.global.s32 v, [$0]; setp.lt.s32 p, v, $1; @p bra WAIT; }',
                        'l,r', (intrinsics.get_ptr_as_int64(
                            count_tensor, self.ready_offset + unit),
                                Int32(intermediate // 128)))
                    self.load_role(inp2, list_tensor, shared, b_full, b_empty,
                                   expert, count, chunks, tid - self.decoders,
                                   routes, intermediate, False, g0, chunk0)
                    g0 += chunks * (intermediate // self.tile_k)
                sequence += 1
                index = self.next_item(shared, sequence)
        elif tid == self.decoders + self.loaders:
            g0 = Int32(0)
            c0 = Int32(0)
            sequence = Int32(0)
            claim = intrinsics.get_ptr_as_int64(count_tensor, self.claim_index)
            # The even CTA of a pair, or every single CTA, claims items and issues MMAs.
            leader = Int32(1)
            if cutlass.const_expr(self.pair):
                leader = relayed
            if leader == 1:
                self.publish_item(
                    shared, sequence,
                    intrinsics.atomic_add_global_i32(claim, Int32(1)))
            index = self.next_item(shared, sequence)
            while index < total:
                if leader == 1:
                    self.publish_item(
                        shared, sequence + 1,
                        intrinsics.atomic_add_global_i32(claim, Int32(1)))
                fc1, unit, column, expert, count, chunk0, chunks = self.fused_work(
                    shared, count_tensor, index, units, experts,
                    intermediate // (128 * self.item_ctas))
                if leader == 1:
                    if fc1 == 1:
                        self.mma_role(shared, tmem, a_full, b_full, mma_done,
                                      b_empty, acc_full, acc_empty, chunks,
                                      1024, g0, c0)
                    else:
                        self.mma_role(shared, tmem, a_full, b_full, mma_done,
                                      b_empty, acc_full, acc_empty, chunks,
                                      intermediate, g0, c0)
                else:
                    if fc1 == 1:
                        self.relay_role(a_full, b_full, acc_empty, chunks,
                                        1024, g0, c0)
                    else:
                        self.relay_role(a_full, b_full, acc_empty, chunks,
                                        intermediate, g0, c0)
                if fc1 == 1:
                    g0 += chunks * (1024 // self.tile_k)
                else:
                    g0 += chunks * (intermediate // self.tile_k)
                c0 += chunks
                sequence += 1
                index = self.next_item(shared, sequence)
        self.ws_teardown(tmem, tid)

    @cute.jit
    def decode_role(self, q: cute.Tensor, bases: cute.Tensor,
                    list_tensor: cute.Tensor, prob: cute.Tensor,
                    out: cute.Tensor, shared: Uint32, tmem: Uint32,
                    lut: Uint32, a_full: Uint32, mma_done: Uint32,
                    acc_full: Uint32, acc_empty: Uint32, expert: Int32,
                    count: Int32, chunks: Int32, tid: Int32, bx: Int32,
                    routes: Int32, n: cutlass.Constexpr, k: cutlass.Constexpr,
                    first: cutlass.Constexpr, group: cutlass.Constexpr,
                    g0: Int32, c0: Int32, chunk0: Int32):
        stages = self.stages
        groups = self.groups
        tiles = k // self.tile_k
        row = tid - 128 * group
        current = self.load_records(q, bases, expert, bx * 128, row,
                                    group * self.tile_k, n, k)
        for chunk in cutlass.range(chunks, unroll=1):
            for step in cutlass.range(tiles // groups, unroll=2):
                tile = step * groups + group
                g = g0 + chunk * tiles + tile
                stage = g % stages
                upcoming = cute.make_rmem_tensor(current.layout, Uint32)
                if tile + groups < tiles:
                    upcoming.store(
                        self.load_records(q, bases, expert, bx * 128, row,
                                          (tile + groups) * self.tile_k, n,
                                          k).load())
                elif chunk + 1 < chunks:
                    upcoming.store(
                        self.load_records(q, bases, expert, bx * 128, row,
                                          group * self.tile_k, n, k).load())
                if g >= stages:
                    self.wait_barrier(mma_done + Uint32(stage) * 8,
                                      g // stages - 1)
                values = ops.decode_record(
                    current[0, 0], current[0, 1], current[0, 2],
                    lut + Uint32(tid % self.lut_copies * 8), self.lut_copies)
                address = tmem + self.tokens + Uint32(stage) * 16 + Uint32(
                    (row // 32 * 32) << 16)
                ops.ptx(
                    'tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {' +
                    ','.join('$' + str(v) for v in range(1, 17)) + '};',
                    ','.join(['r'] * 17), (address, *values))
                ops.ptx(
                    'tcgen05.wait::st.sync.aligned; tcgen05.fence::before_thread_sync;',
                    '')
                self.arrive_barrier(a_full + Uint32(stage) * 8)
                current.store(upcoming.load())
            route_start = (chunk0 + chunk) * self.tokens
            # The previous chunk's epilogue must finish reading the staged routes.
            ops.ptx('bar.sync 1, $0;', 'r', (Int32(self.decoders), ))
            if tid < self.tokens:
                offset = Uint32(0)
                weight = Float32(0)
                if route_start + tid < count:
                    route = list_tensor[expert * routes + route_start + tid]
                    weight = Float32(prob[route])
                    offset = Uint32(route) * (n * 2)
                # Byte offset of the route's output row (routes * n * 2 < 2^32).
                ops.ptx('st.shared.u32 [$0], $1;', 'r,r',
                        (shared + self.ids_offset + tid * 4, offset))
                if cutlass.const_expr(not first):
                    ops.ptx(
                        '{ .reg .b16 value; cvt.rn.bf16.f32 value, $1; st.shared.b16 [$0], value; }',
                        'r,f', (shared + self.weights_offset + tid * 2,
                                weight))
            ops.ptx('bar.sync 1, $0;', 'r', (Int32(self.decoders), ))
            self.wait_barrier(acc_full, c0 + chunk)
            ops.ptx('tcgen05.fence::after_thread_sync;', '')
            # Each decode group writes back its own share of the chunk's token columns.
            share = self.tokens // groups
            valid = count - route_start
            row_address = intrinsics.get_ptr_as_int64(out, bx * 128 + row)
            for column_chunk in cutlass.range(group * share,
                                              (group + 1) * share, 16):
                address = tmem + Uint32((row // 32 * 32) << 16) + Uint32(
                    column_chunk)
                values = ops.ptx(
                    'tcgen05.ld.sync.aligned.32x32b.x16.b32 {' +
                    ','.join('$' + str(v) for v in range(16)) + '}, [$16];',
                    ','.join(['=f'] * 16 + ['r']), (address, ), ('f32', ) * 16)
                offsets = []
                for quad in cutlass.range_constexpr(4):
                    offsets.extend(
                        ops.ptx(
                            'ld.shared.v4.u32 {$0,$1,$2,$3}, [$4];',
                            '=r,=r,=r,=r,r',
                            (shared + self.ids_offset +
                             (column_chunk + quad * 4) * 4, ), ('u32', ) * 4))
                ops.ptx('tcgen05.wait::ld.sync.aligned;', '')
                for j in cutlass.range_constexpr(0, 16, 2):
                    column = column_chunk + j
                    value = ops.packed_epilogue(
                        values[j], values[j + 1],
                        shared + self.weights_offset + column * 2, first)
                    for part in cutlass.range_constexpr(2):
                        if column + part < valid:
                            ops.ptx(
                                '{ .reg .b16 lo, hi; .reg .b64 a; mov.b32 {lo,hi}, $2; cvt.u64.u32 a, $1; add.u64 a, a, $0; st.global.b16 [a], '
                                + ('hi' if part else 'lo') + '; }', 'l,r,r',
                                (row_address, offsets[j + part], value))
            ops.ptx('tcgen05.fence::before_thread_sync;', '')
            self.arrive_barrier(acc_empty)

    @cute.jit
    def load_role(self, inp: cute.Tensor, list_tensor: cute.Tensor,
                  shared: Uint32, b_full: Uint32, b_empty: Uint32,
                  expert: Int32, count: Int32, chunks: Int32, lane: Int32,
                  routes: Int32, k: cutlass.Constexpr,
                  first: cutlass.Constexpr, g0: Int32, chunk0: Int32):
        stages = self.b_stages
        tiles = k // self.tile_k
        copies = self.load_tokens * self.tile_k // 8 // self.loaders
        half = Int32(0)
        if cutlass.const_expr(self.pair):
            half = cute.arch.block_idx_in_cluster() * self.load_tokens
        g = g0
        for chunk in cutlass.range(chunks, unroll=1):
            route_start = (chunk0 + chunk) * self.tokens + half
            # Element offsets fit in Int32 (routes * k < 2^31); -1 marks a padding row.
            offsets = cute.make_rmem_tensor((copies, ), Int32)
            for record in cutlass.range_constexpr(copies):
                group = lane + record * self.loaders
                row = group // (self.tile_k // 8)
                column = group % (self.tile_k // 8) * 8
                offsets[record] = Int32(-1)
                if route_start + row < count:
                    route = list_tensor[expert * routes + route_start + row]
                    input_row = route // 9 if first else route
                    offsets[record] = input_row * k + column
            for tile in cutlass.range(tiles, unroll=2):
                stage = g % stages
                if g >= stages:
                    self.wait_barrier(b_empty + Uint32(stage) * 8,
                                      g // stages - 1)
                b_smem = shared + Uint32(stage) * self.b_bytes
                for record in cutlass.range_constexpr(copies):
                    group = lane + record * self.loaders
                    row = group // (self.tile_k // 8)
                    column = group % (self.tile_k // 8) * 8
                    dest = b_smem + (row * 64 + (column ^ ((row & 7) * 8))) * 2
                    offset = offsets[record]
                    valid = Int32(16)
                    if offset < 0:
                        offset = Int32(0)
                        valid = Int32(0)
                    ops.ptx('cp.async.cg.shared.global [$0], [$1], 16, $2;',
                            'r,l,r', (dest,
                                      intrinsics.get_ptr_as_int64(
                                          inp, offset + tile * self.tile_k),
                                      valid))
                # The stage is published when this lane's copies land, so later
                # tiles never wait behind it.
                ops.ptx(
                    'cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];',
                    'r', (b_full + Uint32(stage) * 8, ))
                g += 1

    @cute.jit
    def mma_role(self, shared: Uint32, tmem: Uint32, a_full: Uint32,
                 b_full: Uint32, mma_done: Uint32, b_empty: Uint32,
                 acc_full: Uint32, acc_empty: Uint32, chunks: Int32,
                 k: cutlass.Constexpr, g0: Int32, c0: Int32):
        stages = self.stages
        tiles = k // self.tile_k
        g = g0
        m = 256 if self.pair else 128
        instruction = Uint32((1 << 4) | (1 << 7) | (1 << 10)
                             | ((self.tokens // 8) << 17) | ((m // 16) << 24))
        cta_group = 2 if self.pair else 1
        masks = 8 if self.pair else 4
        for chunk in cutlass.range(chunks, unroll=1):
            if c0 + chunk > 0:
                self.wait_barrier(acc_empty, c0 + chunk - 1)
            for tile in cutlass.range(tiles, unroll=2):
                stage = g % stages
                b_stage = g % self.b_stages
                self.wait_barrier(a_full + Uint32(stage) * 8, g // stages)
                self.wait_barrier(b_full + Uint32(b_stage) * 8,
                                  g // self.b_stages)
                ops.ptx('tcgen05.fence::after_thread_sync;', '')
                b_smem = shared + Uint32(b_stage) * self.b_bytes
                b = Uint64(b_smem >> 4) | Uint64(1 << 16) | Uint64(
                    64 << 32) | Uint64(2 << 61)
                a_tmem = tmem + self.tokens + Uint32(stage) * 16
                for step in cutlass.range_constexpr(self.tile_k // 16):
                    accumulate = Int32(1)
                    if tile == 0 and step == 0:
                        accumulate = Int32(0)
                    ops.ptx(
                        '{ .reg .pred p; .reg .b32 mask; mov.b32 mask, 0; setp.ne.b32 p, $4, 0; '
                        f'tcgen05.mma.cta_group::{cta_group}.kind::f16 [$0], [$1], $2, $3, {{'
                        + ','.join(['mask'] * masks) + '}, p; }', 'r,r,l,r,r',
                        (tmem, a_tmem + step * 8, b + Uint64(step * 2),
                         instruction, accumulate))
                self.commit(mma_done + Uint32(stage) * 8)
                if cutlass.const_expr(self.b_released):
                    self.commit(b_empty + Uint32(b_stage) * 8)
                g += 1
            self.commit(acc_full)

    @cute.jit
    def relay_role(self, a_full: Uint32, b_full: Uint32, acc_empty: Uint32,
                   chunks: Int32, k: cutlass.Constexpr, g0: Int32, c0: Int32):
        """Forward the odd CTA's operand and accumulator releases to the even CTA's barriers."""
        stages = self.stages
        tiles = k // self.tile_k
        g = g0
        for chunk in cutlass.range(chunks, unroll=1):
            if c0 + chunk > 0:
                self.wait_barrier(acc_empty, c0 + chunk - 1)
                self.arrive_leader(acc_empty)
            for tile in cutlass.range(tiles, unroll=2):
                stage = g % stages
                b_stage = g % self.b_stages
                self.wait_barrier(a_full + Uint32(stage) * 8, g // stages)
                self.arrive_leader(a_full + Uint32(stage) * 8)
                self.wait_barrier(b_full + Uint32(b_stage) * 8,
                                  g // self.b_stages)
                self.arrive_leader(b_full + Uint32(b_stage) * 8)
                g += 1

    @cute.kernel
    def sum_routes(self, slots: cute.Pointer, output: cute.Pointer,
                   elements: Int32):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        i = Int32(bx) * 256 + Int32(tx)
        inp = cute.make_tensor(slots, cute.make_layout(Int64(elements) * 9))
        out = cute.make_tensor(output, cute.make_layout(elements))
        if i < elements:
            value = Float32(0)
            for j in cutlass.range_constexpr(9):
                value += Float32(inp[Int64(i // 1024 * 9 + j) * 1024 +
                                     i % 1024])
            out[i] = cutlass.BFloat16(value)
