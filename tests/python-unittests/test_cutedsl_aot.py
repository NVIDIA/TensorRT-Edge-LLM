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
"""Unit tests for GPU-free CuTe DSL artifact orchestration."""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KERNEL_SRCS = _REPO_ROOT / "kernelSrcs"


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "build_cutedsl", _KERNEL_SRCS / "build_cutedsl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_kernel_module(relative_path):
    pytest.importorskip("cutlass")
    kernel_dir = _KERNEL_SRCS / "gdn_cutedsl"
    for path in (_KERNEL_SRCS, kernel_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = kernel_dir / relative_path
    module_name = f"test_cutedsl_aot_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qsa_module():
    pytest.importorskip("cutlass")
    qsa_dir = _KERNEL_SRCS / "qsa_cutedsl"
    for path in (_KERNEL_SRCS, qsa_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = qsa_dir / "qsa_sparse_gqa.py"
    spec = importlib.util.spec_from_file_location("test_cutedsl_aot_qsa",
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_registered_exporters_exist():
    build_cutedsl = _load_build_module()
    assert build_cutedsl.KERNEL_VARIANTS
    for variant in build_cutedsl.KERNEL_VARIANTS:
        assert (_KERNEL_SRCS / variant.script).is_file()


@pytest.mark.parametrize(
    "sm,group,expected_arch",
    [(80, "gdn", "sm_80"), (121, "f16_moe", "sm_121a"),
     (121, "nvfp4_fused_moe", "sm_121a")],
)
def test_compile_command_forwards_explicit_target(tmp_path, sm, group,
                                                  expected_arch):
    """Every kernel export is pinned to the explicit offline target arch."""
    build_cutedsl = _load_build_module()
    variant = build_cutedsl.KernelVariant(
        name="offline_test",
        group=group,
        supported_sms=[sm],
        script="unused.py",
    )
    compile_gpu_arch = build_cutedsl.default_compile_gpu_arch(sm)
    assert compile_gpu_arch == expected_arch

    command = build_cutedsl._compile_command(
        variant, tmp_path, compile_gpu_arch,
        build_cutedsl.default_host_target_for_arch("x86_64", "x86_64"), sm)

    arch_index = command.index("--gpu-arch")
    assert command[arch_index + 1] == expected_arch


def test_build_cli_requires_target_architecture():
    result = subprocess.run(
        [sys.executable,
         str(_KERNEL_SRCS / "build_cutedsl.py")],
        capture_output=True,
        text=True,
        env={
            **os.environ, "CUDA_VISIBLE_DEVICES": ""
        },
    )
    assert result.returncode != 0
    assert "--gpu_arch" in result.stderr


def test_storage_free_placeholder_smoke():
    pytest.importorskip("cutlass")
    if str(_KERNEL_SRCS) not in sys.path:
        sys.path.insert(0, str(_KERNEL_SRCS))
    import cutlass
    from cutedsl_utils import aot_placeholders

    pointer = aot_placeholders.make_ptr(cutlass.Float16, assumed_align=16)
    tensor = aot_placeholders.make_compact_tensor(
        cutlass.Float16,
        (2, 4),
        stride_order=(1, 0),
        assumed_align=16,
    )

    assert pointer is not None
    assert tensor is not None
    assert tuple(tensor.stride) == (4, 1)
    assert tensor.mark_layout_dynamic(leading_dim=1) is tensor
    assert tensor.mark_compact_shape_dynamic(mode=0,
                                             stride_order=(0, 1)) is tensor
    assert int(aot_placeholders.make_stream()) == 0


def test_marker_emulation_layout_dynamic_frees_all_shapes():
    """mark_layout_dynamic must match from_dlpack: every shape mode dynamic,
    every stride dynamic except the leading one."""
    pytest.importorskip("cutlass")
    if str(_KERNEL_SRCS) not in sys.path:
        sys.path.insert(0, str(_KERNEL_SRCS))
    import cutlass
    from cutedsl_utils import aot_placeholders

    tensor = aot_placeholders.make_compact_tensor(cutlass.Float16, (2, 4, 8),
                                                  assumed_align=16)
    tensor = tensor.mark_layout_dynamic(leading_dim=2)
    assert tensor.dynamic_shapes_mask == (1, 1, 1)
    assert tensor.dynamic_strides_mask == (1, 1, 0)


def test_marker_emulation_rejects_inconsistent_stride_order():
    """An explicit stride_order must agree with the recorded layout."""
    pytest.importorskip("cutlass")
    if str(_KERNEL_SRCS) not in sys.path:
        sys.path.insert(0, str(_KERNEL_SRCS))
    import cutlass
    from cutedsl_utils import aot_placeholders

    # After mark_layout_dynamic, the stride_order's innermost mode must be
    # the recorded leading dim.
    tensor = aot_placeholders.make_compact_tensor(cutlass.Float16, (2, 4),
                                                  assumed_align=16)
    tensor = tensor.mark_layout_dynamic(leading_dim=1)
    with pytest.raises(ValueError, match="leading_dim"):
        tensor.mark_compact_shape_dynamic(mode=0, stride_order=(1, 0))

    # On a compact tensor, the stride_order must match the construction-time
    # packing (here: C-contiguous), like the real marker's consistency check.
    compact = aot_placeholders.make_compact_tensor(cutlass.Float16, (2, 4),
                                                   assumed_align=16)
    with pytest.raises(ValueError, match="inconsistent"):
        compact.mark_compact_shape_dynamic(mode=0, stride_order=(1, 0))

    # leading_dim naming a non-stride-1 mode of the recorded packing fails.
    tensor2 = aot_placeholders.make_compact_tensor(cutlass.Float16, (2, 4),
                                                   assumed_align=16)
    with pytest.raises(ValueError, match="stride-1"):
        tensor2.mark_layout_dynamic(leading_dim=0)


@pytest.mark.parametrize(
    "module_name,factory_name,args",
    [
        ("gdn_decode.py", "_make_aot_cute_tensors",
         (4, 8, 8, 128, 128, False)),
        ("gdn_prefill.py", "_make_aot_cute_tensors", (4, 8, 8, 128, 128, 8)),
        ("gdn_prefill_blackwell.py", "_make_aot_cute_tensors_bw",
         (4, 8, 8, 128, 128, 128)),
        ("gdn_prefill_sm12x.py", "_make_aot_cute_tensors", (4, 8, 8, 8)),
    ],
)
def test_gdn_aot_tensors_include_state_indices(module_name, factory_name,
                                               args):
    module = _load_kernel_module(module_name)
    tensors = getattr(module, factory_name)(*args)

    state_indices = tensors["state_indices"]
    assert state_indices.element_type == module.cutlass.Int32
    assert state_indices.dynamic_shapes_mask == (1, )
    assert state_indices.dynamic_strides_mask == (0, )


def test_gdn_mtp_aot_tensors_do_not_add_state_indices():
    module = _load_kernel_module("gdn_decode_mtp.py")
    tensors = module._make_aot_mtp_cute_tensors(4, 8, 8, 128, 128, 4)

    assert "state_indices" not in tensors


def test_qsa_decode_aot_tensors_are_storage_free():
    module = _load_qsa_module()
    tensors = module._make_decode_aot_tensors(
        module.cutlass.Float16,
        batch_size=2,
        num_head=24,
        num_kv_head=2,
        head_dim=256,
        pool_head_dim=384,
        num_flat_pages=6,
        max_pages=1,
        topk=2051,
        max_splits=8,
        m_block_size=16,
    )

    assert tensors["q"].element_type == module.cutlass.Float16
    assert tensors["pool"].element_type == module.cutlass.Float16
    assert tensors["page_table"].element_type == module.cutlass.Int32
    assert tensors["indices"].element_type == module.cutlass.Int32
    assert tensors["context_lengths"].element_type == module.cutlass.Int32
    assert tensors["partial_output"].element_type == module.cutlass.Float32
    assert tensors["partial_stats"].element_type == module.cutlass.Float32
    assert tensors["counters"].element_type == module.cutlass.Int32
