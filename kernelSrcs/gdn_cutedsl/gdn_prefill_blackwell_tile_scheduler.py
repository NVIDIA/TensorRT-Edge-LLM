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

# Ported from FlashInfer PR #2742:
#   flashinfer/gdn_kernels/blackwell_prefill/gdn_tile_scheduler.py
# Original copyright: Copyright (c) 2026 by FlashInfer team. Apache-2.0.

import cutlass
import cutlass.cute as cute

from cutlass.cute.typing import Int32, Boolean


class GdnWorkTileInfo:
    """Work-tile state supporting GDN's nested ``(m, 0, (b, h))`` coordinate."""

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self):
        tile_values = cutlass.extract_mlir_values(self._tile_idx)
        self._tile_value_count = len(tile_values)
        return tile_values + cutlass.extract_mlir_values(self._is_valid_tile)

    def __new_from_mlir_values__(self, values):
        tile_values = values[:self._tile_value_count]
        valid_values = values[self._tile_value_count:]
        tile_idx = cutlass.new_from_mlir_values(self._tile_idx, tile_values)
        is_valid_tile = cutlass.new_from_mlir_values(
            self._is_valid_tile, valid_values)
        return GdnWorkTileInfo(tile_idx, is_valid_tile)

    @property
    def is_valid_tile(self):
        return self._is_valid_tile

    @property
    def tile_idx(self):
        return self._tile_idx


class GdnStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.is_persistent, self.problem_shape_mbh]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.is_persistent, self.problem_shape_mbh], self._values_pos, strict=True
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return GdnStaticTileSchedulerParams(
            *(tuple(obj_list)), loc=self._loc, ip=self._ip
        )


def create_gdn_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_mbh: cute.Shape,
) -> GdnStaticTileSchedulerParams:
    return GdnStaticTileSchedulerParams(is_persistent, problem_shape_mbh)


class GdnStaticTileScheduler:
    def __init__(
        self,
        params: GdnStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(
            params.problem_shape_mbh, loc=loc, ip=ip
        )
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    # called by host
    @staticmethod
    def get_grid_shape(
        params: GdnStaticTileSchedulerParams,
        sm_count=None,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        if params.is_persistent:
            # sm_count is a runtime value supplied by the caller (AOT wrappers
            # thread it from a kernel argument). HardwareInfo probe of the
            # local device remains as a JIT-mode fallback only.
            if sm_count is None:
                hardware_info = cutlass.utils.HardwareInfo()
                sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                cutlass.min(
                    sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)
                ),
                1,
                1,
            )
        else:
            return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> GdnWorkTileInfo:
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        # cur_tile_coord is (mid, 0, (bid, hid))
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return GdnWorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values = []
        self._value_counts = []
        for obj in (
            self._params,
            self._current_work_linear_idx,
            self._blk_coord,
            self._grid_shape,
        ):
            obj_values = cutlass.extract_mlir_values(obj)
            values.extend(obj_values)
            self._value_counts.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        reconstructed = []
        start = 0
        for obj, value_count in zip(
            (
                self._params,
                self._current_work_linear_idx,
                self._blk_coord,
                self._grid_shape,
            ),
            self._value_counts,
            strict=True,
        ):
            reconstructed.append(
                cutlass.new_from_mlir_values(
                    obj, values[start:start + value_count]))
            start += value_count
        new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape = reconstructed
        return GdnStaticTileScheduler(
            new_params,
            new_current_work_linear_idx,
            new_blk_coord,
            new_grid_shape,
            loc=self._loc,
            ip=self._ip,
        )


def create_gdn_static_tile_scheduler(
    params: GdnStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> GdnStaticTileScheduler:
    return GdnStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)
