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
"""Sharded FP8 n-gram pool metadata, read without torch.

The checkpoint-direct builder emits this metadata during a cold build, and its ``[builder]`` extra
installs neither torch nor safetensors. These tests therefore write the fixture with the standard
library only, and the residency test proves the reader stays importable when torch is absent.
"""

import json
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tensorrt_edgellm.checkpoint.ple_source import (make_ngram_pool_metadata,
                                                    ngram_hash_constant_keys)

REPO_ROOT = Path(__file__).resolve().parents[2]
_PREFIX = "model.language_model.layers.1.ple.ple_embedding"
_HEADS = 4
_ROW_WIDTH = 8
_ROWS_PER_SHARD = 5


def _write_safetensors(path, tensors):
    """Write a safetensors file from ``{name: (dtype, shape, payload)}`` using no third party."""
    header = {}
    offset = 0
    body = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        body += payload
        offset += len(payload)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(bytes(body))


def _write_checkpoint(directory,
                      shards=3,
                      scale_bits=0x3F00,
                      split=False,
                      scale_dtype="BF16"):
    """A minimal checkpoint: ``shards`` FP8 row blocks, a BF16 scale and the three hash vectors.

    ``split`` reproduces the released layout, where the table and its scale sit in their own
    ``model-fp8-mtp-ple.safetensors`` while the hash constants stay in the ordinary weight shards.
    """
    tensors = {}
    for index in range(shards):
        payload = bytes((index * 7 + byte) % 251
                        for byte in range(_ROWS_PER_SHARD * _ROW_WIDTH))
        tensors[f"{_PREFIX}.ngram_embedding.shard_{index}.weight"] = (
            "F8_E4M3", (_ROWS_PER_SHARD, _ROW_WIDTH), payload)
    scale_value = struct.unpack("<f", struct.pack("<I", scale_bits << 16))[0]
    tensors[f"{_PREFIX}.ngram_embedding.weight_scale"] = (
        ("BF16", (1, ),
         struct.pack("<H", scale_bits)) if scale_dtype == "BF16" else
        ("F32", (1, ), struct.pack("<f", scale_value)))
    vocab = [11, 13, 17, 19]
    offsets = [0, 11, 24, 41]
    tensors[f"{_PREFIX}.layer_multipliers"] = ("I64", (3, ),
                                               struct.pack(
                                                   "<3q", 23703573157769,
                                                   20109073645365,
                                                   8052911324071))
    tensors[f"{_PREFIX}.ngram_heads_vocab_sizes"] = ("I64", (_HEADS, ),
                                                     struct.pack(
                                                         f"<{_HEADS}q",
                                                         *vocab))
    tensors[f"{_PREFIX}.ngram_heads_offsets"] = ("I64", (_HEADS, ),
                                                 struct.pack(
                                                     f"<{_HEADS}q", *offsets))
    if split:
        table_keys = [key for key in tensors if ".ngram_embedding." in key]
        table = {key: tensors.pop(key) for key in table_keys}
        _write_safetensors(directory / "model-fp8-mtp-ple.safetensors", table)
    _write_safetensors(directory / "model-00001.safetensors", tensors)
    return vocab, offsets


@pytest.mark.parametrize("scale_dtype", ["BF16", "F32"])
def test_pool_metadata_describes_every_shard_in_row_order(
        tmp_path, scale_dtype):
    # Given a checkpoint whose n-gram table is split across shards,
    # When the pool metadata is produced,
    # Then the shards appear in row order and their rows and bytes account for the whole pool --
    # the invariant the runtime relies on to turn a global row ID into a shard and an offset.
    _write_checkpoint(tmp_path, shards=3, scale_dtype=scale_dtype)
    pool = make_ngram_pool_metadata(tmp_path,
                                    embed_dim=_HEADS * _ROW_WIDTH,
                                    heads=_HEADS)

    assert pool["row_count"] == 3 * _ROWS_PER_SHARD
    assert pool["row_width"] == _ROW_WIDTH
    assert pool["heads"] == _HEADS
    assert pool["dtype"] == "F8_E4M3"
    assert pool["source"]["scale"] == 1.0
    shards = pool["source"]["shards"]
    assert [entry["tensor_key"] for entry in shards] == [
        f"{_PREFIX}.ngram_embedding.shard_{index}.weight" for index in range(3)
    ]
    assert sum(entry["rows"] for entry in shards) == pool["row_count"]
    assert sum(entry["bytes"] for entry in shards) == pool["source"]["bytes"]
    assert all(entry["samples"] for entry in shards)


def test_shards_order_numerically_not_lexicographically(tmp_path):
    # Given more than ten shards, whose keys sort as shard_0, shard_1, shard_10, shard_2 as strings,
    # When the pool metadata is produced,
    # Then the array is in numeric order, because array order *is* row order and a lexicographic
    # sort would silently misplace every shard past the ninth.
    _write_checkpoint(tmp_path, shards=12)
    shards = make_ngram_pool_metadata(tmp_path,
                                      embed_dim=_HEADS * _ROW_WIDTH,
                                      heads=_HEADS)["source"]["shards"]
    suffixes = [
        int(entry["tensor_key"].rsplit("shard_", 1)[1].split(".")[0])
        for entry in shards
    ]
    assert suffixes == list(range(12))


def test_unsplit_table_is_read_as_one_shard(tmp_path):
    """An unsplit ``ngram_embedding.weight`` is the table's single shard and still anchors the
    hash-constant keys."""
    _write_checkpoint(tmp_path, shards=1)
    path = tmp_path / "model-00001.safetensors"
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        body = handle.read()
    header[f"{_PREFIX}.ngram_embedding.weight"] = header.pop(
        f"{_PREFIX}.ngram_embedding.shard_0.weight")
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(body)

    pool = make_ngram_pool_metadata(tmp_path,
                                    embed_dim=_HEADS * _ROW_WIDTH,
                                    heads=_HEADS)
    assert [entry["tensor_key"] for entry in pool["source"]["shards"]
            ] == [f"{_PREFIX}.ngram_embedding.weight"]
    assert pool["row_count"] == _ROWS_PER_SHARD
    assert ngram_hash_constant_keys(tmp_path)["layer_multipliers"] == (
        f"{_PREFIX}.layer_multipliers")


def test_hash_constant_keys_name_checkpoint_tensors(tmp_path):
    # Given a checkpoint carrying the row-ID hash parameters,
    # When the pool metadata is produced,
    # Then it names those tensors rather than copying their values: the runtime reads them from the
    # same checkpoint it opens the table from, so nothing can drift out of step with the rows.
    _write_checkpoint(tmp_path)
    pool = make_ngram_pool_metadata(tmp_path,
                                    embed_dim=_HEADS * _ROW_WIDTH,
                                    heads=_HEADS)

    assert pool["hash_constants"] == {
        "layer_multipliers": f"{_PREFIX}.layer_multipliers",
        "heads_vocab_sizes": f"{_PREFIX}.ngram_heads_vocab_sizes",
        "heads_offsets": f"{_PREFIX}.ngram_heads_offsets",
    }
    assert ngram_hash_constant_keys(tmp_path) == pool["hash_constants"]
    # No numeric value of a hash parameter is duplicated into the config.
    assert "23703573157769" not in json.dumps(pool)


@pytest.mark.parametrize("missing", [
    "layer_multipliers",
    "ngram_heads_vocab_sizes",
    "ngram_heads_offsets",
])
def test_missing_hash_constant_is_rejected(tmp_path, missing):
    # Given a checkpoint with one hash constant absent,
    # When the constants are read,
    # Then it fails rather than defaulting, because a silently absent multiplier would produce row
    # IDs that address real rows and so look plausible.
    _write_checkpoint(tmp_path)
    path = tmp_path / "model-00001.safetensors"
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        body = handle.read()
    del header[f"{_PREFIX}.{missing}"]
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(body)

    with pytest.raises(ValueError, match="hash constant"):
        ngram_hash_constant_keys(tmp_path)


def test_reader_works_without_torch_installed(tmp_path):
    # Given an interpreter where importing torch fails, as in the builder's own `[builder]` extra,
    # When the pool metadata and hash constants are read,
    # Then both succeed: a checkpoint-direct cold build must not need PyTorch, and this path used
    # to import it only to widen three small integer vectors.
    _write_checkpoint(tmp_path)
    script = textwrap.dedent(f"""
        import sys
        sys.modules["torch"] = None          # any `import torch` now raises ImportError
        sys.modules["safetensors"] = None
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from tensorrt_edgellm.checkpoint.ple_source import (
            make_ngram_pool_metadata, ngram_hash_constant_keys)
        pool = make_ngram_pool_metadata({str(tmp_path)!r}, {_HEADS * _ROW_WIDTH}, {_HEADS})
        constants = ngram_hash_constant_keys({str(tmp_path)!r})
        print(pool["row_count"], len(constants))
    """)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [str(3 * _ROWS_PER_SHARD), "3"]


def test_table_and_constants_may_live_in_different_files(tmp_path):
    # Given the released layout, where the table sits in model-fp8-mtp-ple.safetensors while the
    # hash constants stay in the ordinary weight shards,
    # When the pool metadata and constants are read,
    # Then both resolve, because each tensor is located through the checkpoint's own weight map
    # rather than by assuming the n-gram tensors share one file.
    _write_checkpoint(tmp_path, shards=3, split=True)
    assert (tmp_path / "model-fp8-mtp-ple.safetensors").is_file()
    assert (tmp_path / "model-00001.safetensors").is_file()

    pool = make_ngram_pool_metadata(tmp_path,
                                    embed_dim=_HEADS * _ROW_WIDTH,
                                    heads=_HEADS)
    constants = ngram_hash_constant_keys(tmp_path)

    assert pool["row_count"] == 3 * _ROWS_PER_SHARD
    assert len(pool["source"]["shards"]) == 3
    assert constants[
        "heads_vocab_sizes"] == f"{_PREFIX}.ngram_heads_vocab_sizes"
    assert constants["heads_offsets"] == f"{_PREFIX}.ngram_heads_offsets"
