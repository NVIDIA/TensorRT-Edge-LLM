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

"""Storage-free placeholders for CuTe DSL ahead-of-time compilation.

The kernel scripts trace and export with descriptors that carry layout and
dynamism information but no device storage, so artifact generation never
needs a physical GPU.  CuTe DSL >= 4.7 expresses dynamism through
:class:`~cutlass.cute.typing.SymInt` shape entries at construction time and
no longer implements ``mark_layout_dynamic`` / ``mark_compact_shape_dynamic``
on fake tensors.  The kernel scripts, however, share those marker calls with
the real ``from_dlpack`` path, so this module installs marker emulation on
the fake-tensor type: each call rebuilds the descriptor with the matching
``SymInt`` entries and returns the tensor for fluent chaining.

Conventions:
  * ``make_compact_tensor``'s ``stride_order`` uses the native
    ``cute.runtime.make_fake_compact_tensor`` RANK convention:
    ``stride_order[i]`` is the rank of mode ``i`` and rank 0 is the
    innermost (stride-1) mode.  A C-contiguous tensor is
    ``(n-1, ..., 1, 0)`` — the default when omitted.
  * The emulated markers keep the ``mark_compact_shape_dynamic`` API's
    ``torch.Tensor.dim_order`` convention (modes listed outermost to
    innermost; C-contiguous is ``(0, 1, ..., n-1)``); the translation to the
    rank convention happens internally.
  * Dynamic shape scalars are 32-bit (matching the ``int32
    dynamic_shapes[]`` members of exported descriptor structs); derived
    dynamic strides are 64-bit (matching ``int64 dynamic_strides[]``).
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import SymInt


def _rank_order_from_dim_order(dim_order):
    """Translate a dim_order (outermost..innermost) into the rank order.

    ``make_fake_compact_tensor``'s ``stride_order[i]`` is the rank of mode
    ``i`` with rank 0 the innermost (stride-1) mode.
    """
    n = len(dim_order)
    return tuple((n - 1) - dim_order.index(mode) for mode in range(n))


def _extent_divisibility(extent):
    """Divisibility factor an extent contributes to outer compact strides."""
    return extent.divisibility if isinstance(extent, SymInt) else extent


def _rebuild(tensor):
    """Recreate the underlying typed tensor after a dynamism update."""
    dtype = tensor.element_type
    shape = tuple(tensor._edgellm_shape)
    align = tensor._edgellm_assumed_align
    leading = tensor._edgellm_layout_dynamic_leading
    if leading is None:
        rebuilt = cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tensor._edgellm_rank_order,
            assumed_align=align,
        )
    else:
        # Layout-dynamic: every stride except the leading one is a free
        # runtime value, but the from_dlpack markers still propagate the
        # divisibility contracts of inner shape modes onto it (a compact
        # mode-m stride is the product of the extents inner to m). Without
        # this, alignment proofs that hold on the real path — e.g. the
        # 128-bit cp.async source alignment of the Ampere GEMM tiles, which
        # follows from the K extent's divisibility — fail IR verification.
        rank_order = tensor._edgellm_rank_order
        stride = []
        for mode in range(len(shape)):
            if mode == leading:
                stride.append(1)
                continue
            divisibility = 1
            if rank_order is not None:
                for inner_mode in range(len(shape)):
                    if rank_order[inner_mode] < rank_order[mode]:
                        divisibility *= _extent_divisibility(shape[inner_mode])
            stride.append(SymInt(width=64, divisibility=divisibility))
        rebuilt = cute.runtime.make_fake_tensor(
            dtype, shape, tuple(stride), assumed_align=align
        )
    tensor._typed_tensor = rebuilt._typed_tensor
    return tensor


def _fake_mark_compact_shape_dynamic(self, mode, stride_order=None,
                                     divisibility=1):
    """Emulate the from_dlpack marker on a storage-free descriptor.

    ``stride_order`` here follows the marker API's dim_order convention and
    is translated to the rank convention used at construction time.

    Like the real marker, an explicit ``stride_order`` is validated against
    the recorded layout instead of silently re-deriving it: after
    ``mark_layout_dynamic`` the strides are already free (all dynamic except
    the leading mode), so a ``stride_order`` whose innermost mode disagrees
    with the recorded leading dim is a caller bug, and on a compact tensor a
    ``stride_order`` that contradicts the construction-time packing would
    change the traced layout behind the caller's back.
    """
    if stride_order is not None:
        rank_order = _rank_order_from_dim_order(tuple(stride_order))
        leading = self._edgellm_layout_dynamic_leading
        if leading is not None:
            innermost = rank_order.index(0)
            if innermost != leading:
                raise ValueError(
                    f"stride_order {tuple(stride_order)} names mode "
                    f"{innermost} as innermost, but mark_layout_dynamic "
                    f"recorded leading_dim={leading}"
                )
        elif (self._edgellm_rank_order is not None
              and rank_order != self._edgellm_rank_order):
            raise ValueError(
                f"stride_order {tuple(stride_order)} (rank order "
                f"{rank_order}) is inconsistent with the construction-time "
                f"packing (rank order {self._edgellm_rank_order})"
            )
        self._edgellm_rank_order = rank_order
    elif self._edgellm_rank_order is None:
        raise ValueError(
            "mark_compact_shape_dynamic on a strided placeholder requires an "
            "explicit stride_order (the packing cannot be deduced)"
        )
    shape = list(self._edgellm_shape)
    shape[mode] = SymInt(width=32, divisibility=divisibility)
    self._edgellm_shape = tuple(shape)
    return _rebuild(self)


def _fake_mark_layout_dynamic(self, leading_dim=None):
    """Emulate ``mark_layout_dynamic``.

    Matches the from_dlpack semantics: every shape mode becomes dynamic and
    every stride becomes dynamic except the leading (stride-1) mode.  Modes
    already carrying a divisibility contract keep it.
    """
    if leading_dim is None:
        if self._edgellm_rank_order is None:
            raise ValueError(
                "mark_layout_dynamic on a strided placeholder requires an "
                "explicit leading_dim (the stride-1 mode cannot be deduced)"
            )
        # Deduce as the innermost (rank-0) mode of the tracked ordering.
        leading_dim = self._edgellm_rank_order.index(0)
    elif (self._edgellm_rank_order is not None
          and self._edgellm_rank_order[leading_dim] != 0):
        raise ValueError(
            f"leading_dim={leading_dim} does not name the stride-1 mode of "
            f"the recorded packing (rank order {self._edgellm_rank_order})"
        )
    self._edgellm_layout_dynamic_leading = leading_dim
    self._edgellm_shape = tuple(
        extent if isinstance(extent, SymInt) else SymInt(width=32)
        for extent in self._edgellm_shape
    )
    return _rebuild(self)


def _install_fake_tensor_markers(tensor_type):
    if getattr(tensor_type, "_edgellm_marker_emulation", False):
        return
    # If a future CuTe DSL release defines working markers on the fake-tensor
    # class again (4.6.1 had them; 4.7.0 only inherits the abstract
    # NotImplementedError stubs from the Tensor base class), prefer the
    # library's own implementations and skip the emulation entirely.
    if ("mark_compact_shape_dynamic" in tensor_type.__dict__
            and "mark_layout_dynamic" in tensor_type.__dict__):
        return
    # The emulation patches the class, so make sure this really is the DSL's
    # storage-free descriptor type and not a class shared with real
    # from_dlpack tensors (whose markers must keep their native behavior).
    if tensor_type.__name__ != "_FakeTensor":
        raise TypeError(
            f"marker emulation expects cute.runtime._FakeTensor, got "
            f"{tensor_type.__module__}.{tensor_type.__name__}"
        )
    tensor_type.mark_compact_shape_dynamic = _fake_mark_compact_shape_dynamic
    tensor_type.mark_layout_dynamic = _fake_mark_layout_dynamic
    tensor_type._edgellm_marker_emulation = True


def make_ptr(dtype, assumed_align: int = 16):
    """Create an aligned global-memory pointer carrying no backing storage."""
    return cute.runtime.make_ptr(
        dtype,
        assumed_align,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def make_compact_tensor(
    dtype,
    shape,
    *,
    stride_order=None,
    assumed_align: int | None = None,
):
    """Create a compact tensor descriptor carrying no backing storage.

    ``stride_order`` uses the native rank convention (``stride_order[i]`` is
    the rank of mode ``i``, rank 0 innermost); it defaults to C-contiguous
    ``(n-1, ..., 1, 0)``.
    """
    if stride_order is None:
        stride_order = tuple(reversed(range(len(shape))))
    stride_order = tuple(stride_order)
    tensor = cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=assumed_align,
    )
    _install_fake_tensor_markers(type(tensor))
    tensor._edgellm_shape = tuple(shape)
    tensor._edgellm_rank_order = stride_order
    tensor._edgellm_assumed_align = assumed_align
    tensor._edgellm_layout_dynamic_leading = None
    return tensor


def make_tensor(
    dtype,
    shape,
    stride,
    *,
    assumed_align: int | None = None,
):
    """Create a strided tensor descriptor carrying no backing storage."""
    tensor = cute.runtime.make_fake_tensor(
        dtype, shape, stride, assumed_align=assumed_align
    )
    _install_fake_tensor_markers(type(tensor))
    tensor._edgellm_shape = tuple(shape)
    # Track the packing implied by the explicit strides so later marker calls
    # deduce the correct stride-1 mode. Only a verified compact packing
    # yields a rank order: symbolic or non-compact strides admit no compact
    # ordering, and markers then require explicit stride_order / leading_dim
    # arguments (or reject rebuilds that would alter the layout).
    rank_order = None
    if (all(isinstance(extent, int) for extent in stride)
            and all(isinstance(extent, int) for extent in shape)):
        by_magnitude = sorted(range(len(stride)), key=lambda m: stride[m])
        candidate = tuple(by_magnitude.index(m) for m in range(len(stride)))
        compact = [None] * len(candidate)
        product = 1
        for rank in range(len(candidate)):
            mode = candidate.index(rank)
            compact[mode] = product
            product *= shape[mode]
        if tuple(compact) == tuple(stride):
            rank_order = candidate
    tensor._edgellm_rank_order = rank_order
    tensor._edgellm_assumed_align = assumed_align
    tensor._edgellm_layout_dynamic_leading = None
    return tensor


def make_stream():
    """Create the null-stream ABI placeholder without initializing CUDA."""
    return cuda.CUstream(0)


def runtime_int32(value: int = 1):
    """Create a scalar placeholder for an integer supplied at runtime."""
    return cutlass.Int32(value)


def compile_options(default: str = "") -> str | None:
    """Return compile options for the explicit offline target architecture.

    Honors the same environment pass-throughs as the retired
    ``cute_dsl_utils.cute_compile_options`` helper: an explicit target
    architecture (``EDGE_LLM_CUTE_DSL_GPU_ARCH``) and extra ptxas flags
    (``EDGE_LLM_CUTE_DSL_PTXAS_OPTIONS``).
    """
    options = default.strip()
    gpu_arch = os.environ.get("EDGE_LLM_CUTE_DSL_GPU_ARCH", "").strip()
    if gpu_arch and "--gpu-arch" not in options:
        options = f"{options} --gpu-arch={gpu_arch}".strip()
    ptxas_options = os.environ.get(
        "EDGE_LLM_CUTE_DSL_PTXAS_OPTIONS", "").strip()
    if ptxas_options and "--ptxas-options" not in options:
        options = f"{options} --ptxas-options='{ptxas_options}'".strip()
    return options or None
