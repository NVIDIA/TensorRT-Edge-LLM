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
"""Minimal checkpoint reader built on NumPy and mmap.

The safetensors file format is a tiny binary container:
    [ 8-byte LE uint64 header_len ][ header_len bytes of JSON ][ raw tensor data ]

The JSON header maps tensor name -> ``{"dtype", "shape", "data_offsets"}`` plus
an optional ``__metadata__`` entry.  This reader returns raw bytes / numpy
views so the caller can decode exotic dtypes (BF16, F8_E4M3, F4) that the
official ``safetensors.numpy`` backend cannot represent.

Hugging Face checkpoints that only provide ``pytorch_model.bin`` are loaded
lazily with PyTorch's restricted, memory-mapped weights-only reader.
"""

import glob
import json
import mmap
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .numpy_dtypes import bf16_bytes_to_f32, fp8_e4m3_bytes_to_f32

__all__ = [
    "SafetensorsStore",
    "load_safetensors_tensor",
    "read_safetensors_metadata",
]

# safetensors dtype string -> (numpy dtype for raw view, element byte size).
# Sub-byte / exotic dtypes are read as raw uint8 and decoded by helpers.
_NP_DTYPE = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "BOOL": np.bool_,
}


class _ShardFile:
    """Memory-maps a single .safetensors shard and exposes raw tensor bytes."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = int.from_bytes(self._mm[0:8], "little")
        header = json.loads(self._mm[8:8 + header_len].decode("utf-8"))
        self._data_start = 8 + header_len
        self._header: Dict[str, dict] = {
            k: v
            for k, v in header.items() if k != "__metadata__"
        }

    def keys(self) -> List[str]:
        return list(self._header.keys())

    def info(self, name: str) -> dict:
        return self._header[name]

    def raw_bytes(self, name: str) -> bytes:
        meta = self._header[name]
        begin, end = meta["data_offsets"]
        return self._mm[self._data_start + begin:self._data_start + end]

    def raw_size(self, name: str) -> int:
        begin, end = self._header[name]["data_offsets"]
        return int(end) - int(begin)

    def raw_slice(self, name: str, offset: int, length: int) -> bytes:
        begin, end = self._header[name]["data_offsets"]
        size = int(end) - int(begin)
        if offset < 0 or length < 0 or offset > size - length:
            raise ValueError(
                f"{name}: byte range [{offset}, {offset + length}) exceeds "
                f"tensor storage ({size} bytes)")
        start = self._data_start + int(begin) + offset
        return self._mm[start:start + length]

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()


class SafetensorsStore:
    """Read-only view over checkpoint shards in a model directory.

    Resolves the per-tensor shard via ``model.safetensors.index.json`` when
    present, else falls back to a single ``model.safetensors``. PyTorch pickle
    shards are supported when no standard safetensors checkpoint is present.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._weight_map: Dict[str, str] = {}
        self._shards: Dict[str, _ShardFile] = {}
        self._torch_metadata_shards: Dict[str, Dict[str, Any]] = {}
        self._torch_shards: Dict[str, Dict[str, Any]] = {}
        self._raw_files: Dict[str, Any] = {}

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        single_path = os.path.join(model_dir, "model.safetensors")
        torch_index_path = os.path.join(model_dir,
                                        "pytorch_model.bin.index.json")
        torch_single_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self._weight_map = dict(index.get("weight_map", {}))
        elif os.path.exists(single_path):
            shard = _ShardFile(single_path)
            self._shards["model.safetensors"] = shard
            for k in shard.keys():
                self._weight_map[k] = "model.safetensors"
        elif os.path.exists(torch_index_path):
            with open(torch_index_path) as f:
                index = json.load(f)
            self._weight_map = dict(index.get("weight_map", {}))
        elif os.path.exists(torch_single_path):
            metadata = self._load_torch_shard(torch_single_path, "meta")
            self._torch_metadata_shards["pytorch_model.bin"] = metadata
            for key in metadata:
                self._weight_map[key] = "pytorch_model.bin"
        else:
            shard_paths = sorted(
                glob.glob(os.path.join(model_dir, "*.safetensors")))
            if not shard_paths:
                raise FileNotFoundError(
                    f"No safetensors or PyTorch checkpoint files found in {model_dir}"
                )
            for shard_path in shard_paths:
                shard_name = os.path.basename(shard_path)
                shard = _ShardFile(shard_path)
                self._shards[shard_name] = shard
                for key in shard.keys():
                    if key in self._weight_map:
                        raise ValueError(
                            f"duplicate tensor {key!r} in {model_dir}")
                    self._weight_map[key] = shard_name

    # -- shard handling -----------------------------------------------------

    @staticmethod
    def _load_torch_shard(path: str, map_location: str) -> Dict[str, Any]:
        import torch

        kwargs = {
            "map_location": map_location,
            "weights_only": True,
        }
        if map_location == "cpu":
            kwargs["mmap"] = True
        state = torch.load(path, **kwargs)
        if isinstance(state, dict) and isinstance(state.get("state_dict"),
                                                  dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"expected a tensor dictionary in {path}")
        return state

    def _torch_tensor(self, name: str):
        shard_name = self._weight_map[name]
        state = self._torch_shards.get(shard_name)
        if state is None:
            state = self._load_torch_shard(
                os.path.join(self.model_dir, shard_name), "cpu")
            self._torch_shards[shard_name] = state
        return state[name]

    def _torch_metadata(self, name: str):
        shard_name = self._weight_map[name]
        state = self._torch_metadata_shards.get(shard_name)
        if state is None:
            state = self._load_torch_shard(
                os.path.join(self.model_dir, shard_name), "meta")
            self._torch_metadata_shards[shard_name] = state
        return state[name]

    @staticmethod
    def _torch_dtype_name(dtype) -> str:
        import torch

        names = {
            torch.float64: "F64",
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.int64: "I64",
            torch.int32: "I32",
            torch.int16: "I16",
            torch.int8: "I8",
            torch.uint8: "U8",
            torch.bool: "BOOL",
        }
        for attribute, name in (("uint16", "U16"), ("uint32", "U32"),
                                ("float8_e4m3fn", "F8_E4M3FN")):
            value = getattr(torch, attribute, None)
            if value is not None:
                names[value] = name
        if dtype not in names:
            raise TypeError(f"unsupported PyTorch checkpoint dtype {dtype}")
        return names[dtype]

    def _shard(self, name: str) -> _ShardFile:
        shard_name = self._weight_map[name]
        if shard_name.endswith(".bin"):
            raise TypeError(f"{name} belongs to a PyTorch checkpoint shard")
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = _ShardFile(os.path.join(self.model_dir, shard_name))
            self._shards[shard_name] = shard
        return shard

    def keys(self) -> List[str]:
        return list(self._weight_map.keys())

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def dtype(self, name: str) -> str:
        if self._weight_map[name].endswith(".bin"):
            return self._torch_dtype_name(self._torch_metadata(name).dtype)
        return self._shard(name).info(name)["dtype"]

    def shape(self, name: str) -> Tuple[int, ...]:
        if self._weight_map[name].endswith(".bin"):
            return tuple(self._torch_metadata(name).shape)
        return tuple(self._shard(name).info(name)["shape"])

    def checkpoint_location(self, name: str) -> Optional[dict]:
        """Describe one tensor's byte range in a PyTorch ZIP checkpoint.

        ``torch.load(..., map_location="meta")`` parses only the restricted
        weights metadata and records the archive offset on each fake storage.
        The C++ runtime can therefore mmap the original ``.bin`` directly,
        without materializing the externalized tensor during engine build.
        """
        if not self.has(name):
            return None
        shard_name = self._weight_map[name]
        if not shard_name.endswith(".bin"):
            return None

        tensor = self._torch_metadata(name)
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name}: external PyTorch checkpoint tensor must be contiguous"
            )
        storage_offset = getattr(tensor.untyped_storage(),
                                 "_checkpoint_offset", None)
        if storage_offset is None:
            raise RuntimeError(
                "The installed PyTorch package does not expose checkpoint byte "
                f"offsets required to externalize {name!r} from {shard_name}")
        byte_offset = (int(storage_offset) +
                       int(tensor.storage_offset()) * tensor.element_size())
        byte_count = int(tensor.numel()) * tensor.element_size()
        return {
            "file": shard_name,
            "offset": byte_offset,
            "bytes": byte_count,
            "dtype": self._torch_dtype_name(tensor.dtype),
            "shape": [int(dimension) for dimension in tensor.shape],
        }

    def checkpoint_files(self, names: List[str]) -> dict:
        """Return a cheap manifest for checkpoint shards owning ``names``."""
        files = {self._weight_map[name] for name in names if self.has(name)}
        for index_name in ("model.safetensors.index.json",
                           "pytorch_model.bin.index.json"):
            if os.path.isfile(os.path.join(self.model_dir, index_name)):
                files.add(index_name)
        manifest = {}
        for filename in sorted(files):
            stat = os.stat(os.path.join(self.model_dir, filename))
            manifest[filename] = {
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        return manifest

    def tensor_identity(self, name: str, sample_bytes: int = 16) -> dict:
        """Return a bounded content identity for one checkpoint tensor.

        Positive ``sample_bytes`` reads at most three ranges regardless of
        tensor size. Zero records only structural metadata without touching
        payload pages.
        """
        if sample_bytes < 0:
            raise ValueError("sample_bytes must be nonnegative")
        if not self.has(name):
            raise KeyError(name)

        shard_name = self._weight_map[name]
        if shard_name.endswith(".bin"):
            location = self.checkpoint_location(name)
            if location is None:
                raise RuntimeError(
                    f"missing checkpoint location for tensor {name!r}")
            total_bytes = int(location["bytes"])

        else:
            shard = self._shard(name)
            total_bytes = shard.raw_size(name)

        identity = {
            "dtype": self.dtype(name),
            "shape": [int(dimension) for dimension in self.shape(name)],
            "bytes": total_bytes,
            "samples": [],
        }
        if sample_bytes == 0 or total_bytes == 0:
            return identity

        if shard_name.endswith(".bin"):
            raw_file = self._raw_files.get(shard_name)
            if raw_file is None:
                raw_file = open(os.path.join(self.model_dir, shard_name), "rb")
                self._raw_files[shard_name] = raw_file

            def read_sample(offset: int, length: int) -> bytes:
                raw_file.seek(int(location["offset"]) + offset)
                data = raw_file.read(length)
                if len(data) != length:
                    raise OSError(
                        f"short read while sampling {name!r} from {shard_name}"
                    )
                return data

        else:

            def read_sample(offset: int, length: int) -> bytes:
                return shard.raw_slice(name, offset, length)

        width = min(sample_bytes, total_bytes)
        offsets = sorted(
            set((0, max(0, (total_bytes - width) // 2),
                 max(0, total_bytes - width))))
        identity["samples"] = [{
            "offset": offset,
            "data": read_sample(offset, width).hex(),
        } for offset in offsets]
        return identity

    # -- typed accessors ----------------------------------------------------

    def _raw_array(self, name: str) -> Tuple[np.ndarray, str, Tuple[int, ...]]:
        if self._weight_map[name].endswith(".bin"):
            import torch

            tensor = self._torch_tensor(name).detach().cpu().contiguous()
            dtype_str = self._torch_dtype_name(tensor.dtype)
            shape = tuple(tensor.shape)
            if dtype_str == "BF16":
                tensor = tensor.view(torch.uint16)
            elif dtype_str in ("F8_E4M3", "F8_E4M3FN"):
                tensor = tensor.view(torch.uint8)
            raw = tensor.numpy().view(np.uint8).reshape(-1).copy()
            return raw, dtype_str, shape
        shard = self._shard(name)
        meta = shard.info(name)
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        raw = shard.raw_bytes(name)
        return np.frombuffer(raw, dtype=np.uint8).copy(), dtype_str, shape

    def get_f32(self, name: str) -> np.ndarray:
        """Return tensor as float32, decoding F16/BF16/F8_E4M3/F32."""
        buf, dtype_str, shape = self._raw_array(name)
        if dtype_str == "F32":
            return buf.view(np.float32).reshape(shape)
        if dtype_str == "F16":
            return buf.view(np.float16).reshape(shape).astype(np.float32)
        if dtype_str == "BF16":
            return bf16_bytes_to_f32(buf.view(np.uint16)).reshape(shape)
        if dtype_str in ("F8_E4M3", "F8_E4M3FN"):
            return fp8_e4m3_bytes_to_f32(buf).reshape(shape)
        if dtype_str in _NP_DTYPE:
            return buf.view(_NP_DTYPE[dtype_str]).reshape(shape).astype(
                np.float32)
        raise TypeError(f"{name}: cannot decode dtype {dtype_str!r} to f32")

    def get_numpy(self, name: str) -> np.ndarray:
        """Return a copied NumPy array while preserving the stored dtype.

        BF16 and sub-byte FP4 have no portable NumPy dtype. BF16 is returned
        as float32 and FP4 is returned as its packed uint8 storage.
        """
        buf, dtype_str, shape = self._raw_array(name)
        if dtype_str == "BF16":
            return bf16_bytes_to_f32(buf.view(np.uint16)).reshape(shape)
        if dtype_str in ("F8_E4M3", "F8_E4M3FN"):
            return buf.reshape(shape)
        if dtype_str in ("F4", "F4_E2M1"):
            return buf
        if dtype_str not in _NP_DTYPE:
            raise TypeError(
                f"{name}: unsupported safetensors dtype {dtype_str!r}")
        return buf.view(_NP_DTYPE[dtype_str]).reshape(shape)

    def get_f16(self, name: str) -> np.ndarray:
        """Return tensor as float16 (decoding BF16/F32/F16)."""
        return self.get_f32(name).astype(np.float16)

    def get_f16_row(self, name: str, row: int) -> np.ndarray:
        """Read one row of a dense rank-two tensor as float16."""
        shape = self.shape(name)
        if len(shape) != 2 or not 0 <= row < shape[0]:
            raise ValueError(f"{name}: row {row} is outside shape {shape}")
        dtype_str = self.dtype(name)
        if dtype_str not in ("F16", "BF16", "F32"):
            raise TypeError(
                f"{name}: cannot decode dtype {dtype_str!r} to an f16 row")
        if self._weight_map[name].endswith(".bin"):
            return (self._torch_tensor(name)
                    [row].detach().cpu().float().numpy().astype(np.float16))

        item_sizes = {"F16": 2, "BF16": 2, "F32": 4}
        columns = int(shape[1])
        row_bytes = columns * item_sizes[dtype_str]
        raw = self._shard(name).raw_slice(name, row * row_bytes, row_bytes)
        buffer = np.frombuffer(raw, dtype=np.uint8).copy()
        if dtype_str == "F16":
            return buffer.view(np.float16)
        if dtype_str == "F32":
            return buffer.view(np.float32).astype(np.float16)
        return bf16_bytes_to_f32(buffer.view(np.uint16)).astype(np.float16)

    def get_fp8_bytes(self, name: str) -> np.ndarray:
        """Return raw FP8 E4M3 bytes (uint8) with the stored shape."""
        buf, dtype_str, shape = self._raw_array(name)
        if dtype_str not in ("F8_E4M3", "F8_E4M3FN", "I8", "U8"):
            raise TypeError(
                f"{name}: expected FP8/byte tensor, got {dtype_str}")
        return buf.reshape(shape)

    def get_packed_fp4(self, name: str) -> np.ndarray:
        """Return a packed-FP4 weight as uint8 ``[out, in//2]``.

        ModelOpt may store the packed weight as ``U8``/``I8`` with shape
        ``[out, in//2]`` or as a sub-byte ``F4`` dtype with logical shape
        ``[out, in]``.  Both are normalised to a ``[out, in//2]`` uint8 array of
        two FP4 nibbles per byte (low nibble = even index).
        """
        buf, dtype_str, shape = self._raw_array(name)
        out = int(shape[0])
        total_bytes = buf.size
        if total_bytes % out != 0:
            raise ValueError(
                f"{name}: packed-fp4 byte count {total_bytes} not divisible by "
                f"out={out} (dtype={dtype_str}, shape={shape})")
        cols = total_bytes // out
        return buf.reshape(out, cols)

    def get_scalar_f32(self, name: str) -> float:
        return float(self.get_f32(name).reshape(-1)[0])

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()
        for raw_file in self._raw_files.values():
            raw_file.close()
        self._raw_files.clear()
        self._torch_metadata_shards.clear()
        self._torch_shards.clear()

    def __enter__(self) -> "SafetensorsStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_safetensors_tensor(path: str, name: str) -> np.ndarray:
    """Load one tensor from an explicitly named safetensors file."""
    shard = _ShardFile(path)
    try:
        if name not in shard.keys():
            raise KeyError(
                f"{name!r} not found in {path}; available: {shard.keys()}")
        meta = shard.info(name)
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        raw = np.frombuffer(shard.raw_bytes(name), dtype=np.uint8).copy()
        if dtype == "BF16":
            return bf16_bytes_to_f32(raw.view(np.uint16)).reshape(shape)
        if dtype in ("F8_E4M3", "F8_E4M3FN"):
            return raw.reshape(shape)
        if dtype not in _NP_DTYPE:
            raise TypeError(f"{name}: unsupported safetensors dtype {dtype!r}")
        return raw.view(_NP_DTYPE[dtype]).reshape(shape)
    finally:
        shard.close()


def read_safetensors_metadata(path: str) -> Dict[str, dict]:
    """Read tensor dtype and shape headers without touching payload bytes."""
    shard = _ShardFile(path)
    try:
        return {
            name: {
                "dtype": shard.info(name)["dtype"],
                "shape": tuple(int(dim) for dim in shard.info(name)["shape"]),
            }
            for name in shard.keys()
        }
    finally:
        shard.close()
