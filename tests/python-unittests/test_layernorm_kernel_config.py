# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Launch-geometry tests for the CuTe DSL LayerNorm kernel.

``layernorm_config`` is deliberately free of CuTe DSL, cupy and CUDA imports,
so the geometry every exported artifact will be compiled with is asserted here
on any runner, GPU or not. This is the check that catches a target SM resolving
to the build host's GPU: the geometry would silently change and nothing else in
the suite would notice.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (_REPO_ROOT / "kernelSrcs" / "layernorm_cutedsl" /
                "layernorm_config.py")

_SPEC = importlib.util.spec_from_file_location("layernorm_config",
                                               _CONFIG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
layernorm_config = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = layernorm_config
_SPEC.loader.exec_module(layernorm_config)

_FP16_WIDTH = 16

# The geometry every W-schedule artifact is compiled with, keyed by hidden size:
# (vec_size, threads_per_row, num_threads, rows_per_block, num_vec_blocks,
#  cols_per_tile, smem_bytes). One CTA per row, CTA sized to the row and
# rounded up to a power of two capped at 1024.
_W_GEOMETRY = {
    4096: (8, 512, 512, 1, 1, 4096, 128),
    4097: (1, 1024, 1024, 1, 5, 5120, 256),
    5120: (8, 1024, 1024, 1, 1, 8192, 256),
    7168: (8, 1024, 1024, 1, 1, 8192, 256),
    8192: (8, 1024, 1024, 1, 1, 8192, 256),
}

# The staged schedule. Note cols_per_tile == hidden_size at all four production
# sizes, so this schedule has no masked lanes at all.
_S_GEOMETRY = {
    4096: (8, 64, 128, 2, 8, 4096, 16416),
    5120: (8, 64, 128, 2, 10, 5120, 20512),
    7168: (8, 128, 128, 1, 7, 7168, 14368),
    8192: (8, 128, 128, 1, 8, 8192, 16416),
}


def _geometry(config):
    return (config.vec_size, config.threads_per_row, config.num_threads,
            config.rows_per_block, config.num_vec_blocks, config.cols_per_tile,
            config.smem_bytes)


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
@pytest.mark.parametrize("hidden_size",
                         layernorm_config.SUPPORTED_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype_width", [16])
def test_resolved_geometry_matches_the_published_table(target_sm, hidden_size,
                                                       dtype_width):
    """Every artifact's launch geometry is exactly the documented one."""
    config = layernorm_config.layernorm_config(target_sm, dtype_width,
                                               hidden_size)
    expected = (_W_GEOMETRY
                if config.schedule == "W" else _S_GEOMETRY)[hidden_size]

    assert _geometry(config) == expected


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
@pytest.mark.parametrize("hidden_size", [4096, 5120, 7168, 8192])
def test_staged_schedule_has_no_masked_lanes(target_sm, hidden_size):
    """The staged tile covers each production row exactly.

    Every lane does useful work, so the padding machinery is dead code under
    this schedule and the variance reduction cannot be poisoned by a lane that
    contributes ``mean ** 2``.
    """
    config = layernorm_config.layernorm_config(target_sm,
                                               _FP16_WIDTH,
                                               hidden_size,
                                               schedule="S")

    assert config.cols_per_tile == hidden_size
    assert not config.needs_padding
    assert config.use_async_copy


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
def test_odd_hidden_size_never_uses_the_staged_schedule(target_sm):
    """H=4097 stays on the wide schedule on every SM.

    Its largest power-of-two divisor is 1, so the vector width collapses to a
    single element and the 16-bit copy cannot drive the asynchronous
    global-to-shared copy the staged schedule depends on.
    """
    assert layernorm_config.select_schedule(target_sm, 4097) == "W"

    config = layernorm_config.layernorm_config(target_sm, _FP16_WIDTH, 4097)
    assert config.schedule == "W"
    assert config.vec_size == 1


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
@pytest.mark.parametrize("hidden_size",
                         layernorm_config.SUPPORTED_HIDDEN_SIZES)
@pytest.mark.parametrize("schedule", ["S", "W"])
def test_shared_memory_fits_the_target_limit(target_sm, hidden_size, schedule):
    """No artifact asks for more shared memory than its board can opt into."""
    if schedule == "S" and hidden_size == 4097:
        pytest.skip("H=4097 has no staged variant")
    config = layernorm_config.layernorm_config(target_sm,
                                               _FP16_WIDTH,
                                               hidden_size,
                                               schedule=schedule)

    limit = layernorm_config.SHARED_MEMORY_PER_BLOCK_OPTIN[target_sm]
    assert config.smem_bytes <= limit


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
def test_both_schedules_are_constructible_on_every_target(target_sm):
    """Either schedule can be exported for any board, for validation.

    Without this the staged schedule would only ever be exercised on the boards
    that select it, leaving each code path covered by a single class of runner.
    """
    staged = layernorm_config.layernorm_config(target_sm,
                                               _FP16_WIDTH,
                                               8192,
                                               schedule="S")
    wide = layernorm_config.layernorm_config(target_sm,
                                             _FP16_WIDTH,
                                             8192,
                                             schedule="W")

    assert staged.schedule == "S"
    assert wide.schedule == "W"
    assert staged.num_threads != wide.num_threads


@pytest.mark.parametrize("target_sm", [87, 110])
@pytest.mark.parametrize("hidden_size", [4096, 5120, 7168, 8192])
def test_measured_edge_targets_use_the_selected_staged_schedule(
        target_sm, hidden_size):
    """Orin and Thor keep the schedule validated over the full perf matrix.

    The selected-source plugin was compared with the decomposed TensorRT path.
    This pins the validated selection without claiming the cross-source result
    is a controlled same-source S/W experiment.
    """
    assert layernorm_config.select_schedule(target_sm, hidden_size) == "S"
    assert layernorm_config.layernorm_config(target_sm, _FP16_WIDTH,
                                             hidden_size).schedule == "S"

    # H=4097 has no staged variant, so it must stay behind on the wide one.
    assert layernorm_config.select_schedule(target_sm, 4097) == "W"


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
@pytest.mark.parametrize("hidden_size",
                         layernorm_config.SUPPORTED_HIDDEN_SIZES)
@pytest.mark.parametrize("schedule", ["S", "W"])
def test_staging_is_restricted_to_exactly_covered_rows(target_sm, hidden_size,
                                                       schedule):
    """A staged tile is only used when every lane holds a real column.

    The kernel does not zero-fill shared memory before the asynchronous copy, so
    staging a padded tile would feed uninitialized bytes into the mean
    reduction rather than the zeros the register-resident path relies on.
    """
    if schedule == "S" and hidden_size == 4097:
        pytest.skip("H=4097 has no staged variant")
    config = layernorm_config.layernorm_config(target_sm,
                                               _FP16_WIDTH,
                                               hidden_size,
                                               schedule=schedule)

    if config.use_async_copy:
        assert config.cols_per_tile == hidden_size
        assert not config.needs_padding


@pytest.mark.parametrize("target_sm", layernorm_config.SUPPORTED_TARGET_SMS)
@pytest.mark.parametrize("hidden_size",
                         layernorm_config.SUPPORTED_HIDDEN_SIZES)
@pytest.mark.parametrize("schedule", ["S", "W"])
def test_every_row_owns_a_whole_number_of_warps(target_sm, hidden_size,
                                                schedule):
    """No warp may straddle two rows of a multi-row block.

    The reduction buffer is indexed ``(warp_idx // warps_per_row,
    warp_idx % warps_per_row)``, which only identifies the row a warp belongs to
    when each row owns complete warps and the rows tile the block exactly. A
    hidden size needing, say, 320 threads per row would break that silently and
    mix two rows' statistics together.
    """
    if schedule == "S" and hidden_size == 4097:
        pytest.skip("H=4097 has no staged variant")
    config = layernorm_config.layernorm_config(target_sm,
                                               _FP16_WIDTH,
                                               hidden_size,
                                               schedule=schedule)

    assert config.threads_per_row % layernorm_config.WARP_SIZE == 0
    assert config.threads_per_row * config.rows_per_block == config.num_threads
    assert (config.threads_per_row //
            layernorm_config.WARP_SIZE) == config.warps_per_row


def test_every_supported_sm_has_an_explicit_schedule():
    """No SM may fall through to a default.

    A rule of the form "staged unless SM121" would silently change the schedule
    on architectures where the kernel has never been measured.
    """
    assert (set(layernorm_config.SCHEDULE_BY_SM) == set(
        layernorm_config.SUPPORTED_TARGET_SMS))
    assert (set(layernorm_config.SHARED_MEMORY_PER_BLOCK_OPTIN) == set(
        layernorm_config.SUPPORTED_TARGET_SMS))
    assert all(schedule in ("S", "W")
               for schedule in layernorm_config.SCHEDULE_BY_SM.values())


@pytest.mark.parametrize("target_sm,hidden_size", [(89, 4096), (103, 4096),
                                                   (110, 4095), (110, 8193)])
def test_unsupported_targets_are_rejected(target_sm, hidden_size):
    """An unsupported SM or hidden size fails loudly rather than guessing."""
    with pytest.raises(ValueError):
        layernorm_config.layernorm_config(target_sm, _FP16_WIDTH, hidden_size)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        layernorm_config.layernorm_config(110, _FP16_WIDTH, 8192, schedule="X")
