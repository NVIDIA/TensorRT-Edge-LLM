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
"""Deterministic and atomic Hugging Face checkpoint publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import (ConversionError, ConversionReport, InvalidMappingPlan,
                     load_json_asset)
from .transforms import MappingPlan, MappingRule, convert_tensor

_CONVERSION_SCHEMA_VERSION = 2
_PUBLIC_FILE_MODE = 0o644


@dataclass(frozen=True)
class CheckpointFilePlan:
    """Rules assigned to one deterministically named safetensors file.

    Attributes:
        name: Canonical checkpoint filename.
        rules: Source rules kept whole within the file.
        target_bytes: Planned uncompressed BF16 payload bytes.
    """

    name: str
    rules: tuple[MappingRule, ...]
    target_bytes: int

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return this file's target tensor names in rule order."""
        return tuple(name for rule in self.rules for name in rule.target_names)


@dataclass(frozen=True)
class ConversionMetadata:
    """Family provenance and numeric representation written to the manifest."""

    source_family: str
    source_revision: str
    hf_reference_family: str
    hf_reference_revision: str
    source_dtype: str
    target_dtype: str
    rounding_policy: str


class CheckpointWritePolicy(ABC):
    """Family policy consumed by the generic HF checkpoint writer."""

    @property
    @abstractmethod
    def required_tokenizer_files(self) -> tuple[str, ...]:
        """Return the exact tokenizer filenames to copy from the request."""

    @property
    @abstractmethod
    def generated_checkpoint_files(self) -> Mapping[str, bytes]:
        """Return immutable family files generated without request input."""

    @abstractmethod
    def validate_tokenizer_asset(self, name: str, value: dict[str,
                                                              Any]) -> None:
        """Validate one family tokenizer asset."""

    @abstractmethod
    def requires_dedicated_file(self, rule: MappingRule) -> bool:
        """Return whether one mapping rule must occupy its own file."""

    @property
    @abstractmethod
    def conversion_metadata(self) -> ConversionMetadata:
        """Return family provenance for conversion.json."""


@dataclass(frozen=True)
class CheckpointFileReport:
    """Serialized checkpoint file integrity record.

    Attributes:
        name: Checkpoint filename relative to the output directory.
        sha256: Hexadecimal SHA-256 digest of serialized bytes.
        file_size: Serialized file size in bytes.
        tensor_bytes: Uncompressed tensor payload bytes.
    """

    name: str
    sha256: str
    file_size: int
    tensor_bytes: int


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) +
            "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))
    path.chmod(_PUBLIC_FILE_MODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_bytes(shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * 2


def _validate_mapping_rule(rule: MappingRule) -> None:
    if (not isinstance(rule.source_path, tuple) or not rule.source_path
            or any(not isinstance(component, str) or not component
                   for component in rule.source_path)):
        raise InvalidMappingPlan(
            f"invalid checkpoint source path: {rule.source_path!r}")
    if (not isinstance(rule.target_names, tuple)
            or not isinstance(rule.target_shapes, tuple)
            or not rule.target_names
            or len(rule.target_names) != len(rule.target_shapes)
            or len(rule.target_names) != len(set(rule.target_names))
            or any(not isinstance(name, str) or not name
                   for name in rule.target_names)):
        raise InvalidMappingPlan(
            f"invalid checkpoint targets for {rule.source_path}")
    if rule.transform.output_shapes != rule.target_shapes:
        raise InvalidMappingPlan(
            f"transform/target shapes differ for {rule.source_path}")
    for shape in (rule.source_shape, ) + rule.target_shapes:
        if (not isinstance(shape, tuple) or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                or dimension <= 0 for dimension in shape)):
            raise InvalidMappingPlan(
                f"invalid checkpoint shape for {rule.source_path}: {shape}")


def _validate_checkpoint_file_plan(
        checkpoint_file: CheckpointFilePlan) -> None:
    if (not checkpoint_file.name
            or Path(checkpoint_file.name).name != checkpoint_file.name
            or not checkpoint_file.name.endswith(".safetensors")):
        raise InvalidMappingPlan(
            f"invalid checkpoint filename: {checkpoint_file.name!r}")
    if not checkpoint_file.rules:
        raise InvalidMappingPlan(
            f"checkpoint file has no rules: {checkpoint_file.name}")
    for rule in checkpoint_file.rules:
        _validate_mapping_rule(rule)
    target_names = checkpoint_file.target_names
    if len(target_names) != len(set(target_names)):
        raise InvalidMappingPlan(
            f"duplicate targets in {checkpoint_file.name}")
    expected_bytes = sum(rule.target_bytes for rule in checkpoint_file.rules)
    if checkpoint_file.target_bytes != expected_bytes:
        raise InvalidMappingPlan(
            f"checkpoint byte count differs for {checkpoint_file.name}: "
            f"plan={checkpoint_file.target_bytes}, expected={expected_bytes}")


def _validate_publishable_output(output: Path) -> None:
    if not isinstance(output, Path):
        raise ConversionError(
            f"output path must be a pathlib.Path, found {type(output).__name__}"
        )
    if output.is_symlink():
        raise ConversionError(f"output path cannot be a symlink: {output}")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ConversionError(
            f"output parent must be an existing non-symlink directory: {parent}"
        )
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ConversionError(
                f"output directory must be absent or empty: {output}")


def _remove_staging(staging: Path) -> None:
    try:
        if staging.is_symlink():
            staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging)
    except OSError:
        pass


def _lookup_source(source_tree: Mapping[str, Any],
                   source_path: tuple[str, ...]) -> Any:
    value: Any = source_tree
    try:
        for component in source_path:
            value = value[component]
    except (KeyError, TypeError) as error:
        raise InvalidMappingPlan(
            f"mapping source disappeared before write: {source_path}"
        ) from error
    return value


def _mapping_digest(plan: MappingPlan) -> str:
    entries = []
    for rule in sorted(plan.rules,
                       key=lambda item: (item.source_path, item.target_names)):
        entries.append({
            "source_path":
            list(rule.source_path),
            "source_shape":
            list(rule.source_shape),
            "target_names":
            list(rule.target_names),
            "target_shapes": [list(shape) for shape in rule.target_shapes],
            "transform": {
                "axes":
                list(rule.transform.axes),
                "kind":
                rule.transform.kind.value,
                "output_shapes":
                [list(shape) for shape in rule.transform.output_shapes],
                "split_axis":
                rule.transform.split_axis,
            },
        })
    return hashlib.sha256(_json_bytes(entries)).hexdigest()


@dataclass(frozen=True)
class HFCheckpointWriter:
    """Write one validated plan into an atomic HF-style directory.

    Attributes:
        output_path: Final checkpoint directory.
        max_checkpoint_file_bytes: Planned tensor-byte bound per file.
    """

    output_path: Path
    max_checkpoint_file_bytes: int
    policy: CheckpointWritePolicy

    def write(self, tokenizer_path: Path, source_tree: Mapping[str, Any],
              plan: MappingPlan, config: dict[str, Any]) -> ConversionReport:
        """Serialize metadata and weights, then atomically publish them.

        Args:
            tokenizer_path: Validated local tokenizer asset directory.
            source_tree: Canonical nested source parameter tree.
            plan: Complete validated source-to-target plan.
            config: Validated JSON-compatible root configuration.

        Returns:
            Published conversion summary.

        Raises:
            ConversionError: If output preconditions or serialization fail.
            InvalidMappingPlan: If plan metadata is internally inconsistent.
        """
        output_path = self.output_path
        _validate_publishable_output(output_path)
        parent = output_path.parent
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.",
                             suffix=".tmp",
                             dir=parent))
        try:
            checkpoint_files = self.partition_tensors_into_checkpoint_files(
                plan)
            self._write_checkpoint_files(staging, checkpoint_files,
                                         source_tree)
            self.write_checkpoint_metadata(staging, checkpoint_files, plan,
                                           config, tokenizer_path)
            self.atomic_publish(staging, output_path)
        except BaseException:
            _remove_staging(staging)
            raise

        return ConversionReport(
            output_path=output_path,
            tensor_count=len(plan.target_names),
            checkpoint_files=tuple(item.name for item in checkpoint_files),
            total_size=sum(item.target_bytes for item in checkpoint_files),
        )

    def _write_checkpoint_files(self, staging: Path,
                                checkpoint_files: tuple[CheckpointFilePlan,
                                                        ...],
                                source_tree: Mapping[str, Any]) -> None:
        """Serialize checkpoint files in deterministic filename order."""
        for checkpoint_file in sorted(checkpoint_files,
                                      key=lambda item: item.name):
            self.write_checkpoint_file(staging, checkpoint_file, source_tree)

    def partition_tensors_into_checkpoint_files(
            self, plan: MappingPlan) -> tuple[CheckpointFilePlan, ...]:
        """Partition complete rules without materializing source weights.

        Family policy may isolate indivisible rules and permit explicitly
        reviewed rules to exceed a caller-selected bound.

        Args:
            plan: Validated mapping plan.
        Returns:
            Canonically named checkpoint file plans.

        Raises:
            ConversionError: If the byte bound is invalid or another rule is
                individually larger than it.
        """
        ordered = self._validate_partition_inputs(plan)
        groups = self._group_mapping_rules(ordered)
        return self._name_checkpoint_files(groups)

    def _validate_partition_inputs(
            self, plan: MappingPlan) -> tuple[MappingRule, ...]:
        """Validate a mapping plan before deterministic partitioning."""
        limit = self.max_checkpoint_file_bytes
        if (isinstance(limit, bool) or not isinstance(limit, int)
                or limit <= 0):
            raise ConversionError("max_checkpoint_file_bytes must be positive")
        if not plan.rules:
            raise InvalidMappingPlan("cannot serialize an empty mapping plan")
        if not isinstance(plan.variant, str) or not plan.variant:
            raise InvalidMappingPlan(
                "mapping plan must declare a non-empty variant")

        for rule in plan.rules:
            _validate_mapping_rule(rule)
        all_targets = [
            name for rule in plan.rules for name in rule.target_names
        ]
        if len(all_targets) != len(set(all_targets)):
            raise InvalidMappingPlan(
                "mapping plan contains duplicate target names")
        ordered = tuple(
            sorted(plan.rules, key=lambda rule: rule.target_names[0]))
        for rule in ordered:
            if (rule.target_bytes > limit
                    and not self.policy.requires_dedicated_file(rule)):
                raise ConversionError(
                    "ordinary mapping rule exceeds checkpoint file bound: "
                    f"{rule.target_names} uses {rule.target_bytes} bytes; "
                    f"bound={limit}")
        return ordered

    def _group_mapping_rules(
        self, ordered: tuple[MappingRule,
                             ...]) -> tuple[tuple[MappingRule, ...], ...]:
        """Pack ordinary rules and isolate policy-declared dedicated rules."""
        limit = self.max_checkpoint_file_bytes
        groups: list[tuple[MappingRule, ...]] = []
        current: list[MappingRule] = []
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if current:
                groups.append(tuple(current))
                current = []
                current_bytes = 0

        for rule in ordered:
            if self.policy.requires_dedicated_file(rule):
                flush()
                groups.append((rule, ))
                continue
            if current and current_bytes + rule.target_bytes > limit:
                flush()
            current.append(rule)
            current_bytes += rule.target_bytes
        flush()
        return tuple(groups)

    def _name_checkpoint_files(
        self, groups: tuple[tuple[MappingRule, ...], ...]
    ) -> tuple[CheckpointFilePlan, ...]:
        """Assign canonical names after the group count is known."""
        file_count = len(groups)
        checkpoint_files = tuple(
            CheckpointFilePlan(
                name=f"model-{index:05d}-of-{file_count:05d}.safetensors",
                rules=rules,
                target_bytes=sum(rule.target_bytes for rule in rules),
            ) for index, rules in enumerate(groups, start=1))
        for checkpoint_file in checkpoint_files:
            _validate_checkpoint_file_plan(checkpoint_file)
        return checkpoint_files

    def write_checkpoint_file(
            self, staging: Path, checkpoint_file: CheckpointFilePlan,
            source_tree: Mapping[str, Any]) -> CheckpointFileReport:
        """Materialize and serialize only one planned checkpoint file.

        Args:
            staging: Private sibling staging directory.
            checkpoint_file: Rules assigned to this file.
            source_tree: Canonical nested source parameter tree.

        Returns:
            Serialized file size and digest record.

        Raises:
            ConversionError: If target names collide or serialization fails.
        """
        import torch
        from safetensors.torch import save_file

        _validate_checkpoint_file_plan(checkpoint_file)
        path = staging / checkpoint_file.name
        tensors: dict[str, Any] = {}
        try:
            for rule in checkpoint_file.rules:
                converted = convert_tensor(
                    _lookup_source(source_tree, rule.source_path), rule)
                if set(converted) != set(rule.target_names):
                    raise InvalidMappingPlan(
                        f"converted targets differ for {rule.source_path}")
                expected_shapes = dict(
                    zip(rule.target_names, rule.target_shapes))
                for name, tensor in converted.items():
                    if (tuple(tensor.shape) != expected_shapes[name]
                            or tensor.dtype != torch.bfloat16
                            or tensor.device.type != "cpu"
                            or not tensor.is_contiguous()):
                        raise ConversionError(
                            f"invalid BF16 host tensor for target {name}")
                duplicate = set(tensors) & set(converted)
                if duplicate:
                    raise InvalidMappingPlan(
                        f"duplicate target during write: {sorted(duplicate)}")
                tensors.update(converted)
            if tuple(sorted(tensors)) != tuple(
                    sorted(checkpoint_file.target_names)):
                raise InvalidMappingPlan(
                    f"checkpoint target set differs for {checkpoint_file.name}"
                )
            save_file(tensors, str(path), metadata={"format": "pt"})
            path.chmod(_PUBLIC_FILE_MODE)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return CheckpointFileReport(
            name=checkpoint_file.name,
            sha256=_sha256(path),
            file_size=path.stat().st_size,
            tensor_bytes=checkpoint_file.target_bytes,
        )

    def write_checkpoint_metadata(
        self,
        staging: Path,
        checkpoint_files: tuple[CheckpointFilePlan, ...],
        plan: MappingPlan,
        config: dict[str, Any],
        tokenizer_path: Path,
    ) -> None:
        """Write the weight index, config, tokenizer, and conversion record.

        Args:
            staging: Private sibling staging directory.
            checkpoint_files: Deterministic weight file plans.
            plan: Complete target mapping plan.
            config: Validated target config.
            tokenizer_path: Local tokenizer asset directory.

        Raises:
            InvalidMappingPlan: If weight metadata is inconsistent.
            ConversionError: If tokenizer assets are unsafe.
        """
        for checkpoint_file in checkpoint_files:
            _validate_checkpoint_file_plan(checkpoint_file)
            path = staging / checkpoint_file.name
            if path.is_symlink() or not path.is_file():
                raise InvalidMappingPlan(
                    f"checkpoint file is missing or unsafe: {path}")
        weight_map, total_size = self._build_weight_map(checkpoint_files, plan)
        self._write_hf_metadata_files(staging, config, weight_map, total_size)
        self.copy_required_tokenizer_files(tokenizer_path, staging)
        self.write_conversion_manifest(staging, plan)

    def _build_weight_map(self, checkpoint_files: tuple[CheckpointFilePlan,
                                                        ...],
                          plan: MappingPlan) -> tuple[dict[str, str], int]:
        """Build a complete unique target-to-shard map."""
        weight_map: dict[str, str] = {}
        total_size = 0
        for checkpoint_file in sorted(checkpoint_files,
                                      key=lambda item: item.name):
            for name in checkpoint_file.target_names:
                if name in weight_map:
                    raise InvalidMappingPlan(
                        f"duplicate target in checkpoint files: {name}")
                weight_map[name] = checkpoint_file.name
            total_size += checkpoint_file.target_bytes
        expected = set(plan.target_names)
        actual = set(weight_map)
        if actual != expected:
            raise InvalidMappingPlan(
                "weight index does not cover mapping plan; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}")
        return weight_map, total_size

    def _write_hf_metadata_files(self, staging: Path, config: dict[str, Any],
                                 weight_map: dict[str, str],
                                 total_size: int) -> None:
        """Write canonical HF config and safetensors index files."""
        if not isinstance(config, dict):
            raise InvalidMappingPlan("checkpoint config must be a JSON object")
        if not weight_map:
            raise InvalidMappingPlan("checkpoint weight map cannot be empty")
        _write_json(staging / "config.json", config)
        _write_json(staging / "model.safetensors.index.json", {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map,
        })

    def copy_required_tokenizer_files(self, tokenizer_path: Path,
                                      staging: Path) -> None:
        """Copy exactly the allowlisted local tokenizer assets.

        Args:
            tokenizer_path: Directory containing exactly the required files.
            staging: Private sibling staging directory.

        Raises:
            ConversionError: If assets are missing, symlinked, empty, invalid
                JSON where applicable, or accompanied by unexpected entries.
        """
        required = tuple(sorted(self.policy.required_tokenizer_files))
        self._validate_tokenizer_directory(tokenizer_path, required)
        for name in required:
            self._copy_tokenizer_asset(tokenizer_path, staging, name)
        self._write_generated_checkpoint_files(staging, set(required))

    def _validate_tokenizer_directory(self, tokenizer_path: Path,
                                      required: tuple[str, ...]) -> None:
        """Require exactly the policy tokenizer allowlist."""
        if tokenizer_path.is_symlink() or not tokenizer_path.is_dir():
            raise ConversionError(
                f"tokenizer path must be a non-symlink directory: {tokenizer_path}"
            )
        actual_names = {entry.name for entry in tokenizer_path.iterdir()}
        expected_names = set(required)
        if actual_names != expected_names:
            raise ConversionError(
                "tokenizer directory must contain exactly the required assets; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}")

    def _copy_tokenizer_asset(self, tokenizer_path: Path, staging: Path,
                              name: str) -> None:
        """Validate and byte-preserve one tokenizer asset."""
        source = tokenizer_path / name
        if source.is_symlink() or not source.is_file():
            raise ConversionError(
                f"tokenizer asset must be a regular non-symlink file: {source}"
            )
        if source.suffix == ".json":
            value, data = load_json_asset(source)
            self.policy.validate_tokenizer_asset(name, value)
        else:
            data = source.read_bytes()
            if not data:
                raise ConversionError(
                    f"tokenizer asset must not be empty: {source}")
        destination = staging / name
        destination.write_bytes(data)
        destination.chmod(_PUBLIC_FILE_MODE)

    def _write_generated_checkpoint_files(self, staging: Path,
                                          reserved_names: set[str]) -> None:
        """Write safe policy-generated files without collisions."""
        generated = self.policy.generated_checkpoint_files
        if not isinstance(generated, Mapping):
            raise ConversionError(
                "generated checkpoint files must be a mapping")
        for name, data in sorted(generated.items()):
            if (not isinstance(name, str) or not name
                    or Path(name).name != name or name in reserved_names
                    or name in {
                        "config.json", "conversion.json",
                        "model.safetensors.index.json"
                    }):
                raise ConversionError(
                    f"invalid generated checkpoint filename: {name!r}")
            if not isinstance(data, bytes) or not data:
                raise ConversionError(
                    f"generated checkpoint file must contain bytes: {name}")
            destination = staging / name
            if destination.exists():
                raise ConversionError(
                    f"generated checkpoint file collides with output: {name}")
            destination.write_bytes(data)
            destination.chmod(_PUBLIC_FILE_MODE)

    def write_conversion_manifest(self, staging: Path,
                                  plan: MappingPlan) -> None:
        """Write a deterministic converter-owned provenance manifest.

        Args:
            staging: Private sibling staging directory.
            plan: Complete target mapping plan.
        """
        weight_map = self._load_weight_map(staging, plan)
        weight_files = self._build_weight_file_records(staging, plan,
                                                       weight_map)
        files = self._build_file_records(staging)
        metadata = self.policy.conversion_metadata
        _write_json(
            staging / "conversion.json", {
                "files": files,
                "hf_reference": {
                    "family": metadata.hf_reference_family,
                    "revision": metadata.hf_reference_revision,
                },
                "mapping_sha256": _mapping_digest(plan),
                "rounding_policy": metadata.rounding_policy,
                "schema_version": _CONVERSION_SCHEMA_VERSION,
                "source": {
                    "family": metadata.source_family,
                    "revision": metadata.source_revision,
                },
                "source_dtype": metadata.source_dtype,
                "source_tensor_count": len(plan.rules),
                "target_dtype": metadata.target_dtype,
                "target_tensor_count": len(plan.target_names),
                "variant": plan.variant,
                "weight_files": weight_files,
            })

    def _load_weight_map(self, staging: Path,
                         plan: MappingPlan) -> dict[str, str]:
        """Load and validate complete target coverage from the HF index."""
        index_path = staging / "model.safetensors.index.json"
        try:
            index, _ = load_json_asset(index_path)
            weight_map = index["weight_map"]
        except (ConversionError, KeyError, TypeError) as error:
            raise InvalidMappingPlan(
                "checkpoint index is missing or invalid") from error
        if (not isinstance(weight_map, dict)
                or len(weight_map) != len(plan.target_names)
                or set(weight_map) != set(plan.target_names)):
            raise InvalidMappingPlan(
                "checkpoint index does not cover the mapping plan")
        if any(not isinstance(name, str) or not isinstance(filename, str)
               for name, filename in weight_map.items()):
            raise InvalidMappingPlan(
                "checkpoint index names and filenames must be strings")

        for filename in weight_map.values():
            if (Path(filename).name != filename
                    or not filename.endswith(".safetensors")):
                raise InvalidMappingPlan(
                    f"invalid checkpoint index filename: {filename!r}")
        return dict(sorted(weight_map.items()))

    def _build_weight_file_records(
            self, staging: Path, plan: MappingPlan,
            weight_map: dict[str, str]) -> list[dict[str, Any]]:
        """Hash indexed shards and record their tensor payload sizes."""
        for rule in plan.rules:
            _validate_mapping_rule(rule)
        target_bytes = {
            name: _shape_bytes(shape)
            for rule in plan.rules
            for name, shape in zip(rule.target_names, rule.target_shapes)
        }
        weight_files = []
        for name in sorted(set(weight_map.values())):
            path = staging / name
            if path.is_symlink() or not path.is_file():
                raise InvalidMappingPlan(
                    f"indexed checkpoint file is missing or unsafe: {path}")
            weight_files.append({
                "file_size":
                path.stat().st_size,
                "name":
                name,
                "sha256":
                _sha256(path),
                "tensor_bytes":
                sum(target_bytes[target_name]
                    for target_name, filename in weight_map.items()
                    if filename == name),
            })
        return weight_files

    def _build_file_records(self, staging: Path) -> list[dict[str, Any]]:
        """Hash every staged output except the manifest itself."""
        files = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.name == "conversion.json":
                continue
            if path.is_symlink() or not path.is_file():
                raise InvalidMappingPlan(
                    f"staged output is not a plain file: {path}")
            files.append({
                "name": path.name,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            })
        return files

    def atomic_publish(self,
                       staging: Path,
                       output_path: Path | None = None) -> None:
        """Publish staging while preserving an existing empty destination.

        Args:
            staging: Complete sibling directory to publish.
            output_path: Optional destination overriding the writer attribute.

        Raises:
            ConversionError: If staging or destination is unsafe or nonempty.
        """
        output = self.output_path if output_path is None else output_path
        if not isinstance(staging, Path):
            raise ConversionError(
                f"staging path must be a pathlib.Path, found {type(staging).__name__}"
            )
        if staging.is_symlink() or not staging.is_dir():
            raise ConversionError(
                f"staging path must be a non-symlink directory: {staging}")
        _validate_publishable_output(output)
        if staging.parent != output.parent:
            raise ConversionError(
                "staging and output must be sibling paths on the destination filesystem"
            )
        os.replace(staging, output)
