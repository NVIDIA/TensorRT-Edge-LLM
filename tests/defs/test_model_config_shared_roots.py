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
"""Unit tests for ${ROOT} expansion in model-owned YAML checkpoint references."""

import pytest
import yaml

from . import model_test_config, shared_roots


@pytest.fixture
def load_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(model_test_config, "MODEL_CONFIG_ROOT", tmp_path)

    def load(reference):
        document = {
            "schema_version":
            1,
            "model_id":
            "Qwen/Qwen3-0.6B",
            "inputs": ["text"],
            "outputs": ["text"],
            "cases": [{
                "name":
                "shared-root",
                "mode":
                "server",
                "platform":
                "l0_e2e_a30",
                "quantized_model_id_or_dir":
                reference,
                "requirements": {
                    "min_sm": 100,
                    "min_gpu_memory_gb": 5
                },
                "engine": {},
                "inference": [{
                    "name": "request",
                    "scenario": "openai",
                    "input_file": "tests/test_cases/llm_basic.json"
                }],
            }],
        }
        description = "## Summary\n<!-- EDGE_LLM_L0_TESTS -->\n```yaml\n" + yaml.safe_dump(
            document) + "```\n"
        tests = model_test_config.load_model_tests("l0_e2e_a30", description)
        return tests[0].model_parameters["checkpoint_dir"]

    return load


def test_checkpoint_reference_expands_root_from_the_environment(
        monkeypatch, load_checkpoint):
    """Environment roots select the checkpoint used by generated tests."""
    # Given a shared root pointed at this machine's mount by its variable
    monkeypatch.setenv("LLM_MODELS_DIR", "/mnt/local/llm-models")

    # When a model config references a checkpoint under that root
    resolved = load_checkpoint("${LLM_MODELS_DIR}/Qwen3/Qwen3-0.6B")

    # Then the root is replaced and the checkpoint-relative suffix is preserved,
    # which is what lets one config run against CI and developer mounts alike
    assert resolved == "/mnt/local/llm-models/Qwen3/Qwen3-0.6B"


def test_checkpoint_reference_falls_back_to_a_mounted_candidate(
        monkeypatch, load_checkpoint):
    """An available mount supplies a root when its variable is absent."""
    # Given no variable set, but one of the known mounts present on this machine
    monkeypatch.delenv("LLM_MODELS_DIR", raising=False)
    monkeypatch.setattr(shared_roots, "SHARED_ROOT_CANDIDATES",
                        {"LLM_MODELS_DIR": ("/definitely/absent", "/tmp")})

    # When the same reference is resolved
    resolved = load_checkpoint("${LLM_MODELS_DIR}/Qwen3/Qwen3-0.6B")

    # Then the first candidate that actually exists wins, so an unconfigured
    # checkout still runs without anyone exporting anything
    assert resolved == "/tmp/Qwen3/Qwen3-0.6B"


def test_checkpoint_reference_keeps_an_unresolvable_root_visible(
        monkeypatch, load_checkpoint):
    """Missing mounts do not prevent collection."""
    # Given a root with neither a variable nor any mounted candidate
    monkeypatch.delenv("LLM_MODELS_DIR", raising=False)
    monkeypatch.setattr(shared_roots, "SHARED_ROOT_CANDIDATES",
                        {"LLM_MODELS_DIR": ("/definitely/absent", )})

    # When the reference is resolved
    resolved = load_checkpoint("${LLM_MODELS_DIR}/Qwen3/Qwen3-0.6B")

    # Then it is returned unexpanded rather than raised on: collection still
    # succeeds on a machine missing one export, and the surviving variable name
    # is what tells the reader which one to set
    assert resolved == "${LLM_MODELS_DIR}/Qwen3/Qwen3-0.6B"


def test_checkpoint_reference_rejects_an_unknown_root(load_checkpoint):
    """Unknown root names are rejected during configuration loading."""
    # Given a reference naming a root the test framework does not define
    # When it is resolved
    # Then it fails loudly at parse time, instead of being treated as a relative
    # path and skipped later as a mysteriously missing checkpoint
    with pytest.raises(ValueError, match="unknown shared root"):
        load_checkpoint("${NO_SUCH_ROOT}/Qwen3/Qwen3-0.6B")


@pytest.mark.parametrize("reference", [
    "/scratch.edge_llm_data/onnx_models/public/Qwen3-8B-NVFP4",
    "Qwen/Qwen3.5-0.8B",
])
def test_checkpoint_reference_still_accepts_the_pre_existing_forms(
        reference, load_checkpoint):
    """Absolute paths and provider IDs reach test execution unchanged."""
    # Given a plain absolute path or a Hugging Face repository ID
    # When it is resolved
    # Then it round-trips untouched, so adding ${ROOT} support does not change
    # how references that do not use it behave
    assert load_checkpoint(reference) == reference
