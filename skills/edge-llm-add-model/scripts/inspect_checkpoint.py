#!/usr/bin/env python3
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
"""Inspect checkpoint metadata without importing or executing model code."""

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _nested_model_types(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if not isinstance(value, dict):
        return found
    model_type = value.get("model_type")
    if isinstance(model_type, str):
        found[prefix or "<root>"] = model_type
    for key, child in value.items():
        if isinstance(child, dict):
            child_prefix = f"{prefix}.{key}" if prefix else key
            found.update(_nested_model_types(child, child_prefix))
    return found


def _config_keys(value: Any) -> set[str]:
    """Return every key name at any depth of a parsed config."""
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys |= _config_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _config_keys(child)
    return keys


def _tensor_pattern(name: str) -> str:
    return re.sub(r"(?<=\.)\d+(?=\.)", "N", name)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size > 100_000_000:
            raise ValueError(f"{path}: unreasonable header size {header_size}")
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(header)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return parsed


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")

    shards = sorted(checkpoint.glob("*.safetensors"))
    patterns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tensor_names: set[str] = set()
    for shard in shards:
        for name, metadata in _read_safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"{shard}: invalid metadata for {name}")
            tensor_names.add(name)
            patterns[_tensor_pattern(name)].append({
                "dtype":
                metadata.get("dtype"),
                "shape":
                metadata.get("shape"),
            })

    index_path = checkpoint / "model.safetensors.index.json"
    missing_shards: list[str] = []
    missing_index_tensors: list[str] = []
    extra_shard_tensors: list[str] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(
                "model.safetensors.index.json weight_map is invalid")
        missing_shards = sorted({
            shard
            for shard in weight_map.values()
            if isinstance(shard, str) and not (checkpoint / shard).is_file()
        })
        missing_index_tensors = sorted(set(weight_map) - tensor_names)
        extra_shard_tensors = sorted(tensor_names - set(weight_map))

    normalized_patterns = []
    for pattern in sorted(patterns):
        variants = Counter((entry["dtype"], tuple(entry["shape"] or []))
                           for entry in patterns[pattern])
        normalized_patterns.append({
            "pattern":
            pattern,
            "count":
            len(patterns[pattern]),
            "variants": [{
                "dtype": dtype,
                "shape": list(shape),
                "count": count,
            } for (dtype, shape), count in sorted(variants.items())],
        })

    config_keys = _config_keys(config)
    markers = [
        key for key in (
            "quantization_config",
            "draft_vocab_size",
            "dflash_config",
            "mtp_num_hidden_layers",
            "num_nextn_predict_layers",
            "auto_map",
        ) if key in config_keys
    ]
    sidecars = sorted(path.name for path in checkpoint.iterdir()
                      if not path.name.endswith(".safetensors"))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "model_types": _nested_model_types(config),
        "architectures": config.get("architectures", []),
        "markers": markers,
        "sidecars": sidecars,
        "safetensors_shards": [path.name for path in shards],
        "tensor_count": len(tensor_names),
        "tensor_patterns": normalized_patterns,
        "missing_shards": missing_shards,
        "missing_index_tensors": missing_index_tensors,
        "extra_shard_tensors": extra_shard_tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a local Hugging Face checkpoint without imports")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if not report["safetensors_shards"]:
        print(
            "error: no .safetensors shards found; no tensors were inspected "
            "(config-only directory or .bin weights)",
            file=sys.stderr)
        raise SystemExit(2)
    if report["missing_shards"] or report["missing_index_tensors"]:
        print(
            "error: index names missing shards "
            f"{report['missing_shards'][:5]} or tensors "
            f"{report['missing_index_tensors'][:5]}",
            file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
