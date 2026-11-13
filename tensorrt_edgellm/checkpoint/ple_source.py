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
"""Bounded checkpoint metadata for PLE row pools, native F16 or sharded FP8."""

import json
import math
import os
import re
import struct
from pathlib import Path
from typing import BinaryIO, Tuple

GEMMA_PLE_TENSOR_SUFFIX = "embed_tokens_per_layer.weight"
NGRAM_SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.*\.ngram_embedding)(?:\.shard_(?P<index>\d+))?\.weight$")
_MAX_HEADER_BYTES = 16 * 1024 * 1024


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


def _tensor_record(source: BinaryIO,
                   tensor_name: str) -> Tuple[dict, int, int]:
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError("Truncated safetensors header length")
    header_bytes = struct.unpack("<Q", prefix)[0]
    file_bytes = os.fstat(source.fileno()).st_size
    if not 0 < header_bytes <= _MAX_HEADER_BYTES:
        raise ValueError("Safetensors header exceeds the bounded header limit")
    if header_bytes > file_bytes - 8:
        raise ValueError("Truncated safetensors header")
    header = json.loads(source.read(header_bytes),
                        object_pairs_hook=_unique_json_object)
    if not isinstance(header, dict) or tensor_name not in header:
        raise ValueError(f"Missing safetensors tensor: {tensor_name}")
    record = header[tensor_name]
    if not isinstance(record, dict):
        raise ValueError("Invalid safetensors tensor record")
    offsets = record.get("data_offsets")
    if (not isinstance(offsets, list) or len(offsets) != 2
            or any(type(value) is not int for value in offsets)):
        raise ValueError("Invalid safetensors data_offsets")
    begin, end = offsets
    if not 0 <= begin <= end <= file_bytes - 8 - header_bytes:
        raise ValueError("Safetensors tensor range exceeds the source file")
    shape = record.get("shape")
    if (not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)):
        raise ValueError("Invalid safetensors tensor shape")
    return record, 8 + header_bytes + begin, end - begin


def checkpoint_shard_path(model_dir, filename):
    """Validate an index shard name without resolving checkpoint symlinks."""
    if not isinstance(filename, str) or not filename or "\0" in filename:
        raise ValueError(f"Invalid PLE checkpoint shard path: {filename!r}")
    relative = Path(filename)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid PLE checkpoint shard path: {filename!r}")
    base = os.path.abspath(model_dir)
    candidate = os.path.abspath(os.path.join(base, filename))
    if os.path.commonpath((base, candidate)) != base or candidate == base:
        raise ValueError(f"Invalid PLE checkpoint shard path: {filename!r}")
    return Path(candidate)


def select_gemma_ple_tensor(keys, diffusion=False):
    """Resolve the PLE table without confusing a diffusion encoder with its decoder."""
    matches = [
        key for key in keys if key == GEMMA_PLE_TENSOR_SUFFIX
        or key.endswith("." + GEMMA_PLE_TENSOR_SUFFIX)
    ]
    if diffusion:
        decoder = [key for key in matches if key.startswith("model.decoder.")]
        matches = decoder or [
            key for key in matches if not key.startswith("model.encoder.")
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one checkpoint PLE tensor, found {matches}")
    return matches[0]


def make_gemma_ple_source_metadata(tensor_key,
                                   identity,
                                   row_count,
                                   row_width,
                                   hidden_size,
                                   location=None,
                                   weight_scale=None):
    """Describe raw checkpoint rows and the cast-then-scale operation into the pool."""
    sizes = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1}
    dtype = identity["dtype"]
    if (dtype not in sizes or identity["shape"] != [row_count, row_width]
            or row_count <= 0 or row_width <= 0 or hidden_size <= 0
            or identity["bytes"] != row_count * row_width * sizes[dtype]
            or not identity["samples"]):
        raise ValueError(
            "Invalid PLE checkpoint dtype, shape, byte size, or samples")
    embed_scale = struct.unpack("<e", struct.pack("<e",
                                                  math.sqrt(hidden_size)))[0]
    if dtype == "F8_E4M3":
        if weight_scale is None or not math.isfinite(
                weight_scale) or weight_scale <= 0:
            raise ValueError(
                "Quantized Gemma4 PLE requires a positive FP8 weight scale")
        embed_scale = struct.unpack("<f",
                                    struct.pack("<f",
                                                math.sqrt(hidden_size)))[0]
        scale = struct.unpack("<f",
                              struct.pack("<f", embed_scale * weight_scale))[0]
    else:
        if weight_scale is not None:
            raise ValueError(
                "Unquantized Gemma4 PLE must not carry an FP8 scale")
        scale = embed_scale
    source = {
        "tensor_key": tensor_key,
        "dtype": dtype,
        "bytes": identity["bytes"],
        "scale": 1.0 if dtype == "F8_E4M3" else scale,
        "samples": identity["samples"]
    }
    if location is not None:
        source["location"] = {
            key: location[key]
            for key in ("file", "offset", "bytes")
        }
    return {
        "row_count": row_count,
        "row_width": row_width,
        "dtype": "F8_E4M3" if dtype == "F8_E4M3" else "F16",
        "scale": scale if dtype == "F8_E4M3" else 1.0,
        "source": source
    }


def read_ple_samples(path, byte_offset, byte_count):
    """Read at most three 16-byte ranges, using checkpoint-identity sample offsets."""
    width = min(16, byte_count)
    offsets = sorted(
        {0, max(0, (byte_count - width) // 2),
         max(0, byte_count - width)})
    result = []
    with open(path, "rb") as source:
        for offset in offsets:
            source.seek(byte_offset + offset)
            data = source.read(width)
            if len(data) != width:
                raise ValueError("Truncated PLE checkpoint tensor")
            result.append({"offset": offset, "data": data.hex()})
    return result


def safetensors_weight_map(model_dir):
    """Map every checkpoint tensor to its shard file, from the index or the headers."""
    directory = Path(model_dir)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        with index.open() as source:
            return json.load(source)["weight_map"]
    weight_map = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            count = struct.unpack("<Q", prefix)[0]
            if not 0 < count <= _MAX_HEADER_BYTES:
                raise ValueError("Safetensors header exceeds bounded limit")
            records = json.loads(source.read(count),
                                 object_pairs_hook=_unique_json_object)
        for key in records:
            if key != "__metadata__":
                if key in weight_map:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                weight_map[key] = path.name
    return weight_map


def gemma_ple_identity(model_dir, diffusion=False):
    """Locate one raw PLE tensor using headers and a bounded payload sample."""
    directory = Path(model_dir)
    weight_map = safetensors_weight_map(directory)
    key = select_gemma_ple_tensor(weight_map, diffusion)
    path = checkpoint_shard_path(directory, weight_map[key])
    with path.open("rb") as source:
        record, offset, byte_count = _tensor_record(source, key)
    identity = {
        "dtype": record["dtype"],
        "shape": record["shape"],
        "bytes": byte_count,
        "samples": read_ple_samples(path, offset, byte_count)
    }
    return key, identity


def gemma_ple_weight_scale(model_dir, tensor_key):
    """Read ModelOpt's per-tensor FP8 PLE scale from the same checkpoint."""
    key = tensor_key.removesuffix(".weight") + ".weight_scale"
    weight_map = safetensors_weight_map(model_dir)
    if key not in weight_map:
        raise ValueError(f"Missing Gemma4 PLE weight scale: {key}")
    path = checkpoint_shard_path(model_dir, weight_map[key])
    with path.open("rb") as source:
        record, offset, byte_count = _tensor_record(source, key)
        if record["dtype"] != "F32" or record["shape"] not in ([], [
                1
        ]) or byte_count != 4:
            raise ValueError(
                "Gemma4 PLE weight scale must be a scalar FP32 tensor")
        source.seek(offset)
        scale = struct.unpack("<f", source.read(4))[0]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Gemma4 PLE weight scale must be finite and positive")
    return scale


def _read_bf16_scalar(path, byte_offset):
    """Widen one BF16 value to float by zero-extending it into the FP32 mantissa."""
    with open(path, "rb") as source:
        source.seek(byte_offset)
        raw = source.read(2)
    if len(raw) != 2:
        raise ValueError("Truncated PLE dequantization scale")
    return struct.unpack("<f", b"\x00\x00" + raw)[0]


def ngram_table_keys(weight_map):
    """Order the table's row blocks: ``shard_<i>.weight`` by index, or one unsplit ``weight``.

    The shard suffix does not sort lexicographically -- ``shard_10`` precedes ``shard_2`` -- so
    ordering by the trailing integer is what keeps the concatenated row space correct.
    """
    shards = {}
    prefixes = set()
    for key in weight_map:
        match = NGRAM_SHARD_PATTERN.match(key)
        if match:
            index = int(match.group("index") or 0)
            if index in shards:
                raise ValueError(f"Duplicate n-gram table shard {index}")
            shards[index] = key
            prefixes.add(match.group("prefix"))
    if not shards:
        raise ValueError("Checkpoint has no n-gram table shards")
    if len(prefixes) != 1:
        raise ValueError(
            f"Expected one n-gram table, found {sorted(prefixes)}")
    if sorted(shards) != list(range(len(shards))):
        raise ValueError(
            f"N-gram table shards are not contiguous from 0: {sorted(shards)}")
    return prefixes.pop(), [shards[index] for index in range(len(shards))]


def ngram_shard_identities(model_dir):
    """Read the bounded identity samples of the n-gram table shards, in row order."""
    directory = Path(model_dir)
    weight_map = safetensors_weight_map(directory)
    prefix, keys = ngram_table_keys(weight_map)
    scale_key = prefix + ".weight_scale"
    if scale_key not in weight_map:
        raise ValueError(f"Missing n-gram dequantization scale: {scale_key}")

    identities = []
    for key in keys:
        path = checkpoint_shard_path(directory, weight_map[key])
        with path.open("rb") as source:
            record, offset, byte_count = _tensor_record(source, key)
        identities.append({
            "tensor_key":
            key,
            "dtype":
            record["dtype"],
            "shape":
            record["shape"],
            "bytes":
            byte_count,
            "samples":
            read_ple_samples(path, offset, byte_count),
        })
    scale_path = checkpoint_shard_path(directory, weight_map[scale_key])
    with scale_path.open("rb") as source:
        scale_record, scale_offset, scale_bytes = _tensor_record(
            source, scale_key)
    # Released checkpoints store the scale as BF16; the quantizer writes F32.
    if scale_record["dtype"] == "BF16" and scale_bytes == 2:
        return identities, _read_bf16_scalar(scale_path, scale_offset)
    if scale_record["dtype"] == "F32" and scale_bytes == 4:
        with scale_path.open("rb") as source:
            source.seek(scale_offset)
            return identities, struct.unpack("<f", source.read(4))[0]
    raise ValueError(
        "N-gram dequantization scale must be a BF16 or F32 scalar")


def ngram_hash_constant_keys(model_dir):
    """Name the checkpoint tensors holding the row-ID hash parameters.

    Only the keys travel in the engine config; the runtime reads the values from the same
    checkpoint it opens the table from, so the multipliers, moduli and row offsets cannot drift out
    of step with the rows they index.
    """
    directory = Path(model_dir)
    weight_map = safetensors_weight_map(directory)
    prefix = ngram_table_keys(weight_map)[0][:-len("ngram_embedding")]

    keys = {
        "layer_multipliers": prefix + "layer_multipliers",
        "heads_vocab_sizes": prefix + "ngram_heads_vocab_sizes",
        "heads_offsets": prefix + "ngram_heads_offsets",
    }
    for key in keys.values():
        if key not in weight_map:
            raise ValueError(f"Checkpoint has no n-gram hash constant: {key}")
    return keys


def make_ngram_pool_metadata(model_dir, embed_dim, heads):
    """Describe the sharded FP8 n-gram table as ``ple_pool`` metadata.

    The pool holds the checkpoint's FP8 rows verbatim, so the source conversion is a raw copy and
    the checkpoint's ``weight_scale`` is what the gather applies after dequantizing.
    """
    if heads <= 0 or embed_dim <= 0 or embed_dim % heads:
        raise ValueError(
            f"Invalid n-gram pool geometry: embed_dim {embed_dim}, heads {heads}"
        )
    row_width = embed_dim // heads
    identities, scale = ngram_shard_identities(model_dir)
    shards = []
    row_count = 0
    total_bytes = 0
    for identity in identities:
        if (identity["dtype"] != "F8_E4M3" or len(identity["shape"]) != 2
                or identity["shape"][1] != row_width
                or identity["shape"][0] <= 0
                or identity["bytes"] != identity["shape"][0] * row_width):
            raise ValueError(f"Invalid n-gram shard {identity['tensor_key']}: "
                             f"{identity['dtype']} {identity['shape']}")
        shard = {
            "tensor_key": identity["tensor_key"],
            "rows": identity["shape"][0],
            "bytes": identity["bytes"],
            "samples": identity["samples"],
        }
        shards.append(shard)
        row_count += shard["rows"]
        total_bytes += shard["bytes"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid n-gram dequantization scale: {scale}")
    return {
        "row_count": row_count,
        "row_width": row_width,
        "dtype": "F8_E4M3",
        "scale": scale,
        "heads": heads,
        "hash_constants": ngram_hash_constant_keys(model_dir),
        "source": {
            "dtype": "F8_E4M3",
            "bytes": total_bytes,
            "scale": 1.0,
            "shards": shards,
        },
    }
