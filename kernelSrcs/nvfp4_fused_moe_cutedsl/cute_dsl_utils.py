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

from __future__ import annotations

import ctypes
import functools
import importlib.util
from typing import Union, Tuple

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


def get_cutlass_dtype(dtype: str) -> cutlass.dtype:
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
    }
    return dtype_map[dtype]


def _cuda_device_id(device: int | str | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    device_str = str(device)
    if device_str == "cuda":
        return None
    if device_str.startswith("cuda:"):
        return int(device_str.split(":", 1)[1])
    return int(device)


def _ensure_cuda_context(device: int | str | None = None):
    import cupy as cp

    device_id = _cuda_device_id(device)
    if device_id is None:
        device_id = cp.cuda.runtime.getDevice()
    cp.cuda.Device(device_id).use()
    # HardwareInfo calls CUDA Driver APIs that require a current context.
    cp.cuda.runtime.free(0)
    return cp, device_id


@functools.cache
def get_compute_capability(device: int | str | None = None) -> int:
    cp, device_id = _ensure_cuda_context(device)
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return int(props["major"]) * 10 + int(props["minor"])


@functools.cache
def get_num_sm(device: int | str | None = None) -> int:
    # Get the streaming multiprocessor count of the device.
    cp, device_id = _ensure_cuda_context(device)
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props["multiProcessorCount"]


# Cache for HardwareInfo - it's expensive to create on every call
_hardware_info_cache: "cutlass.utils.HardwareInfo | None" = None


def get_hardware_info() -> "cutlass.utils.HardwareInfo":
    """Get cached HardwareInfo singleton.

    HardwareInfo queries CUDA device capabilities, which can be expensive.
    This function caches the singleton to avoid repeated queries.
    """
    global _hardware_info_cache
    if _hardware_info_cache is None:
        _, device_id = _ensure_cuda_context()
        _hardware_info_cache = cutlass.utils.HardwareInfo(device_id)
    return _hardware_info_cache


@functools.cache
def get_max_active_clusters(cluster_size: int) -> int:
    """Get max active clusters for a given cluster size (cached).

    Args:
        cluster_size: Product of cluster_shape_mn dimensions.

    Returns:
        Maximum number of active clusters supported by hardware.
    """
    return get_hardware_info().get_max_active_clusters(cluster_size)


# WAR for CuTeDSL make_ptr implementation
class _Pointer(Pointer):
    """Runtime representation of a pointer that can inter-operate with
    various data structures, including numpy arrays and device memory.

    :param pointer: The pointer to the data
    :type pointer: int or pointer-like object
    :param dtype: Data type of the elements pointed to
    :type dtype: Type
    :param mem_space: Memory space where the pointer resides, defaults generic
    :type mem_space: _cute_ir.AddressSpace, optional
    :param assumed_align: Alignment of input pointer in bytes, defaults None
    :type assumed_align: int, optional

    :ivar _pointer: The underlying pointer
    :ivar _dtype: Data type of the elements
    :ivar _addr_space: Memory space of the pointer
    :ivar _assumed_align: Alignment of the pointer in bytes
    :ivar _desc: C-type descriptor for the pointer
    :ivar _c_pointer: C-compatible pointer representation
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        # if expected_py_type is Pointer:
        #     return True
        # elif isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer:
        #     return True
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address

    :param dtype: Data type of the pointer elements
    :type dtype: Type[Numeric]
    :param value: Memory address as integer or ctypes pointer
    :type value: Union[int, ctypes._Pointer]
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :type mem_space: AddressSpace, optional
    :param assumed_align: Alignment in bytes, defaults to None
    :type assumed_align: int, optional
    :return: A pointer object
    :rtype: Pointer

    .. code-block:: python

        import numpy as np
        import ctypes

        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)

        # Get pointer address as integer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)


def get_mma_sf_shape(
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> Tuple[int, int, int, int, int, int]:
    """Get the 6D MMA-compatible scale factor shape.

    Args:
        m: The M dimension (rows) of the matrix.
        k: The K dimension (columns) of the matrix.
        num_groups: Number of groups. Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Shape tuple: (32, 4, m_tiles, 4, k_tiles, num_groups)
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)
    return (32, 4, m_tiles, 4, k_tiles, num_groups)


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make shared memory layout for scale factors A on SM120.

    Constructs the SMEM layout for SFA based on BlockScaledBasicChunk,
    MMA tiler shape, scale factor vector size, and number of pipeline stages.

    Args:
        tiled_mma: The tiled MMA operation.
        tile_shape_mnk: The tile shape (M, N, K).
        sf_vec_size: Scale factor vector size (16 or 32).
        num_stages: Number of pipeline stages.

    Returns:
        Staged shared memory layout for SFA.
    """
    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[0] must be divisible by 64"
    )

    # Scale-factor tiles are quantized in 128-row blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfa_tile_m = max(blk_mn, ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make shared memory layout for scale factors B on SM120.

    Constructs the SMEM layout for SFB based on BlockScaledBasicChunk,
    MMA tiler shape, scale factor vector size, and number of pipeline stages.

    Args:
        tiled_mma: The tiled MMA operation.
        tile_shape_mnk: The tile shape (M, N, K).
        sf_vec_size: Scale factor vector size (16 or 32).
        num_stages: Number of pipeline stages.

    Returns:
        Staged shared memory layout for SFB.
    """
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    assert tile_shape_mnk[1] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[1] must be divisible by 64"
    )

    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    # Scale-factor tiles are quantized in 128-column blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfb_tile_n = max(blk_mn, ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFB_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFB_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )

    sSFB_shape = (sSFB_shapeN, sSFB_shapeK)
    sSFB_stride = (sSF_strideN, sSF_strideK)

    smem_layout = cute.make_layout(sSFB_shape, stride=sSFB_stride)

    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged
