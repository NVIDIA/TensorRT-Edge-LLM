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
"""Resolve test checkpoints from local storage or Hugging Face."""

import fcntl
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_FILES = ("*.safetensors", "*.bin")
CHECKPOINTS_BY_ID = {}
PRIVATE_CHECKPOINT_IDS = frozenset()



def _build_alias_index(*, quantized: bool) -> dict[str, str]:
    index: dict[str, str] = {}
    for checkpoint_id, checkpoint in CHECKPOINTS_BY_ID.items():
        if quantized:
            metadata = checkpoint.get("quantized")
            aliases = () if not metadata else tuple(
                f"{metadata['name']}-{suffix}"
                for suffix in metadata["quantizations"])
        else:
            paths = checkpoint.get("torch", ())
            metadata = checkpoint.get("quantized")
            family_aliases = () if not metadata else (metadata["name"], )
            aliases = (checkpoint_id, checkpoint_id.rsplit("/",
                                                           maxsplit=1)[-1],
                       *paths, *(Path(path).name
                                 for path in paths), *family_aliases)

        for alias in aliases:
            previous = index.setdefault(alias, checkpoint_id)
            if previous != checkpoint_id:
                kind = "quantized" if quantized else "direct"
                raise RuntimeError(
                    f"Ambiguous {kind} checkpoint alias {alias!r}: "
                    f"{previous!r} and {checkpoint_id!r}")
    return index


DIRECT_CHECKPOINT_ID_BY_ALIAS = _build_alias_index(quantized=False)
QUANTIZED_CHECKPOINT_ID_BY_ALIAS = _build_alias_index(quantized=True)


def is_direct_checkpoint_alias(alias: str) -> bool:
    """Return whether alias identifies a public or TRT-LLM checkpoint."""
    return alias in DIRECT_CHECKPOINT_ID_BY_ALIAS


def is_quantized_checkpoint_alias(alias: str) -> bool:
    """Return whether alias identifies a managed quantized checkpoint."""
    return alias in QUANTIZED_CHECKPOINT_ID_BY_ALIAS


def is_registered_checkpoint_alias(alias: str) -> bool:
    """Return whether alias identifies any registered checkpoint."""
    return (is_direct_checkpoint_alias(alias)
            or is_quantized_checkpoint_alias(alias))


def is_hf_checkpoint_id(value: str) -> bool:
    """Return whether value is an explicit Hugging Face repository ID."""
    parts = value.split("/")
    return (len(parts) == 2
            and all(part not in ("", ".", "..") for part in parts))


def is_checkpoint_dir(path: Path) -> bool:
    """Return whether path contains a complete Hugging Face checkpoint."""
    return (path.is_dir() and (path / "config.json").is_file()
            and any(any(path.glob(pattern)) for pattern in _CHECKPOINT_FILES))


def _find_in_roots(relative_paths: Sequence[str],
                   roots: Sequence[str]) -> Optional[str]:
    for root in roots:
        if not root:
            continue
        root_path = Path(root).expanduser()
        for relative_path in relative_paths:
            candidate = root_path / relative_path
            if is_checkpoint_dir(candidate):
                return str(candidate)
    return None


def _quantized_relative_paths(checkpoint_id: str,
                              variant: str) -> tuple[str, ...]:
    metadata = CHECKPOINTS_BY_ID[checkpoint_id].get("quantized")
    if not metadata:
        return ()
    prefix = f"{metadata['name']}-"
    if not variant.startswith(prefix):
        return ()
    quantization = variant[len(prefix):]
    if quantization not in metadata["quantizations"]:
        return ()
    scopes = (("private", ) if checkpoint_id in PRIVATE_CHECKPOINT_IDS else
              ("public", "private"))
    return tuple(f"{scope}/{variant}" for scope in scopes)


def _quantized_managed_paths(checkpoint_id: str,
                             variant: str) -> tuple[str, ...]:
    metadata = CHECKPOINTS_BY_ID[checkpoint_id].get("quantized")
    if not metadata:
        return ()
    prefix = f"{metadata['name']}-"
    if not variant.startswith(prefix):
        return ()
    quantization = variant[len(prefix):]
    return tuple(metadata.get("paths", {}).get(quantization, ()))


def _managed_download_path(hf_id: str,
                           download_root: str) -> tuple[Path, Path]:
    root = Path(download_root).expanduser()
    if not root.is_absolute():
        raise ValueError("HF_CHECKPOINT_DOWNLOAD_DIR must be absolute")
    root = root.resolve()
    if root == _REPO_ROOT or _REPO_ROOT in root.parents:
        raise ValueError(
            "HF_CHECKPOINT_DOWNLOAD_DIR must be outside the Git checkout")
    parts = hf_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid Hugging Face checkpoint ID: {hf_id}")
    return root, root.joinpath(*parts)


def _download_from_hf(hf_id: str, download_root: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required when checkpoint downloading is enabled"
        ) from error

    root, local_dir = _managed_download_path(hf_id, download_root)
    namespace, model = hf_id.split("/")
    lock_path = root / ".locks" / namespace / f"{model}.lock"
    cache_dir = root / ".cache"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if is_checkpoint_dir(local_dir):
            return str(local_dir)
        snapshot_download(repo_id=hf_id,
                          local_dir=local_dir,
                          cache_dir=cache_dir)

    if not is_checkpoint_dir(local_dir):
        raise RuntimeError(
            f"Hugging Face download did not produce a checkpoint: {local_dir}")
    return str(local_dir)


def resolve_checkpoint(
    alias: str,
    *,
    torch_roots: Sequence[str],
    quantized_root: Optional[str] = None,
    managed_roots: Sequence[str] = (),
    quantized_variant: Optional[str] = None,
    allow_download: bool = False,
    download_root: Optional[str] = None,
) -> str:
    """Resolve a local name, registered checkpoint or explicit Hugging Face ID.

    Direct checkpoints use declared TRT-LLM paths, an existing managed
    download, then an optional Hugging Face download. An explicit Hugging Face
    ID may be downloaded without an inventory entry. Quantized variants are
    resolved only under ``quantized_root`` so a base checkpoint cannot be
    mistaken for a missing quantized checkpoint.
    """
    if quantized_variant:
        checkpoint_id = DIRECT_CHECKPOINT_ID_BY_ALIAS.get(alias)
        variant = quantized_variant
    elif alias in DIRECT_CHECKPOINT_ID_BY_ALIAS:
        quantized_checkpoint_id = QUANTIZED_CHECKPOINT_ID_BY_ALIAS.get(alias)
        if quantized_checkpoint_id:
            relative_paths = _quantized_relative_paths(quantized_checkpoint_id,
                                                       alias)
            if quantized_root:
                found = _find_in_roots(relative_paths, (quantized_root, ))
                if found:
                    return found
            managed_paths = _quantized_managed_paths(quantized_checkpoint_id,
                                                     alias)
            found = _find_in_roots(managed_paths, managed_roots)
            if found:
                return found
        checkpoint_id = DIRECT_CHECKPOINT_ID_BY_ALIAS[alias]
        variant = None
    elif is_hf_checkpoint_id(alias):
        checkpoint_id = alias
        variant = None
    else:
        checkpoint_id = QUANTIZED_CHECKPOINT_ID_BY_ALIAS.get(alias)
        variant = alias if checkpoint_id else None

    if checkpoint_id is None:
        if (not quantized_variant and alias not in ("", ".", "..")
                and "/" not in alias):
            found = _find_in_roots((alias, ), torch_roots)
            if found:
                return found
        raise ValueError(f"Checkpoint is not registered: {alias}")

    if variant:
        relative_paths = _quantized_relative_paths(checkpoint_id, variant)
        if not relative_paths:
            raise ValueError("Quantized checkpoint is not registered for "
                             f"{checkpoint_id}: {variant}")
        if quantized_root:
            found = _find_in_roots(relative_paths, (quantized_root, ))
            if found:
                return found
        managed_paths = _quantized_managed_paths(checkpoint_id, variant)
        found = _find_in_roots(managed_paths, managed_roots)
        if found:
            return found
        raise ValueError("Registered quantized checkpoint not found: "
                         f"{list(relative_paths)} "
                         f"under {quantized_root}; managed paths "
                         f"{list(managed_paths)} under {list(managed_roots)}")

    checkpoint = CHECKPOINTS_BY_ID.get(checkpoint_id, {})
    declared_paths = checkpoint.get("torch", ())
    if not checkpoint:
        declared_paths = (checkpoint_id, checkpoint_id.rsplit("/",
                                                              maxsplit=1)[-1])
    found = _find_in_roots(declared_paths, torch_roots)
    if found:
        return found

    if checkpoint_id in PRIVATE_CHECKPOINT_IDS:
        raise ValueError(
            f"Private CI checkpoint {checkpoint_id} was not found in its "
            "declared managed roots")

    if download_root:
        _, managed_path = _managed_download_path(checkpoint_id, download_root)
        if is_checkpoint_dir(managed_path):
            return str(managed_path)
    if allow_download:
        if not download_root:
            raise ValueError("HF_CHECKPOINT_DOWNLOAD_DIR is required when "
                             "EDGE_LLM_ALLOW_HF_DOWNLOAD is enabled")
        return _download_from_hf(checkpoint_id, download_root)

    raise ValueError(
        f"Checkpoint {checkpoint_id} was not found in TRT-LLM roots "
        f"{list(torch_roots)}. "
        "Set EDGE_LLM_ALLOW_HF_DOWNLOAD=1 to download it into the managed "
        "HF_CHECKPOINT_DOWNLOAD_DIR.")


def resolve_checkpoint_alias(
    alias: str,
    *,
    torch_roots: Sequence[str],
    quantized_root: Optional[str] = None,
    managed_roots: Sequence[str] = (),
    allow_download: bool = False,
    download_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve a draft or paired checkpoint by a registered alias."""
    if not is_registered_checkpoint_alias(alias):
        return None
    return resolve_checkpoint(
        alias,
        torch_roots=torch_roots,
        quantized_root=quantized_root,
        managed_roots=managed_roots,
        allow_download=allow_download,
        download_root=download_root,
    )
