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
"""Behavioral contracts for Flax converter request validation."""

import json

import pytest

from experimental.checkpoint_converters.models.gemma4 import schema
from experimental.checkpoint_converters.models.gemma4.schema import (
    GEMMA_REVISION, Gemma4ConversionRequest)
from experimental.checkpoint_converters.models.gemma4.variant import \
    A4B_26B_IT_VARIANT
from experimental.checkpoint_converters.schema import (ConversionError,
                                                       RevisionMismatch,
                                                       UnsupportedConversion)

from ...tokenizer_fixtures import create_conversion_paths


@pytest.fixture
def conversion_paths(tmp_path):
    """Create compatible local paths for request validation."""
    return create_conversion_paths(tmp_path)


def _request(conversion_paths, **changes):
    """Build a request with any explicitly varied contract fields."""
    checkpoint, tokenizer, output = conversion_paths
    values = {
        "checkpoint_path": checkpoint,
        "tokenizer_path": tokenizer,
        "output_path": output,
    }
    values.update(changes)
    return Gemma4ConversionRequest(**values)


def test_request_accepts_only_safe_pinned_first_cut_inputs(
        conversion_paths, monkeypatch):
    """Validation accepts a safe local request only at the pinned Gemma revision."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)

    _request(conversion_paths, variant_name=A4B_26B_IT_VARIANT.name).validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({
            "variant_name": "other"
        }, "unsupported variant"),
        ({
            "output_dtype": "float16"
        }, "text-only BF16"),
        ({
            "text_only": False
        }, "text-only BF16"),
    ],
)
def test_request_rejects_modes_outside_the_first_cut(conversion_paths,
                                                     monkeypatch, changes,
                                                     message):
    """Unknown variants, target dtype, and modality cannot silently broaden."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)

    with pytest.raises(UnsupportedConversion, match=message):
        _request(conversion_paths, **changes).validate()


def test_request_rejects_unpinned_requested_revision(conversion_paths,
                                                     monkeypatch):
    """The requested source revision must match the reviewed Gemma commit."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)

    with pytest.raises(RevisionMismatch, match="pinned revision"):
        _request(conversion_paths, gemma_revision="unversioned").validate()


def test_request_rejects_unknown_installed_revision(conversion_paths,
                                                    monkeypatch):
    """Missing install provenance cannot bypass the pinned-source boundary."""
    monkeypatch.setattr(schema, "_installed_gemma_revision", lambda: None)

    with pytest.raises(RevisionMismatch, match="found unknown"):
        _request(conversion_paths).validate()


def test_request_preserves_existing_nonempty_output(conversion_paths,
                                                    monkeypatch):
    """Validation rejects an occupied destination without modifying its files."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)
    output = conversion_paths[2]
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(ConversionError, match="absent or empty"):
        _request(conversion_paths).validate()

    assert marker.read_text(encoding="utf-8") == "original"


def test_request_rejects_symlinked_tokenizer_asset(conversion_paths,
                                                   monkeypatch):
    """Tokenizer validation refuses assets that escape the supplied directory."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)
    tokenizer = conversion_paths[1]
    target = conversion_paths[0] / "real.json"
    target.write_text("{}", encoding="utf-8")
    asset = tokenizer / "tokenizer.json"
    asset.unlink()
    asset.symlink_to(target)

    with pytest.raises(ConversionError, match="non-symlink"):
        _request(conversion_paths).validate()


def test_request_rejects_unexpected_tokenizer_asset_before_restore(
        conversion_paths, monkeypatch):
    """Request preflight rejects extra tokenizer content before model restore."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)
    unexpected = conversion_paths[1] / "unreviewed.json"
    unexpected.write_text("{}", encoding="utf-8")

    with pytest.raises(ConversionError, match="unexpected"):
        _request(conversion_paths).validate()


@pytest.mark.parametrize(
    ("name", "field_path", "invalid_value"),
    [
        ("generation_config.json", ("bos_token_id", ), 7),
        ("tokenizer_config.json", ("backend", ), "sentencepiece"),
        ("tokenizer.json", ("model", "type"), "Unigram"),
    ],
)
def test_request_rejects_incompatible_tokenizer_asset(conversion_paths,
                                                      monkeypatch, name,
                                                      field_path,
                                                      invalid_value):
    """Preflight rejects semantically incompatible tokenizer JSON assets."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)
    _, tokenizer, _ = conversion_paths
    asset = tokenizer / name
    value = json.loads(asset.read_text(encoding="utf-8"))
    parent = value
    for field in field_path[:-1]:
        parent = parent[field]
    parent[field_path[-1]] = invalid_value
    asset.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConversionError, match="incompatible"):
        _request(conversion_paths).validate()


@pytest.mark.parametrize("defect", ["sparse", "duplicate"])
def test_request_rejects_incompatible_tokenizer_vocabulary(
        conversion_paths, monkeypatch, defect):
    """Preflight requires one unique contiguous ID for every Gemma4 vocab item."""
    monkeypatch.setattr(schema, "_installed_gemma_revision",
                        lambda: GEMMA_REVISION)
    asset = conversion_paths[1] / "tokenizer.json"
    value = json.loads(asset.read_text(encoding="utf-8"))
    vocab = value["model"]["vocab"]
    if defect == "sparse":
        vocab.pop("token_100")
    else:
        vocab["token_0"] = 1
    asset.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConversionError, match="vocabulary"):
        _request(conversion_paths).validate()
