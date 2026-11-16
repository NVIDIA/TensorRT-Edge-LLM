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
"""Contract of the jump-forward L0 checks: the fixture must have jumped, and its bounded guides must finish."""

import json
from pathlib import Path

import pytest
from defs.utils.command_execution import (_check_guided_decoding_conformance,
                                          _check_jump_forward_ran)

_FIXTURE = Path(__file__).resolve(
).parents[2] / "tests/test_cases/llm_guided_decoding_jump_forward.json"

# One finished, guide-satisfying answer per fixture request, in fixture order.
_VALID_TEXTS = [
    '{"name": "Ada", "age": 36}',
    '{"ok": true}',
    "ada@example.com",
    "a",
    "The weather in Paris is mild.",
    "yes",
    '{"intent": "book_flight", "origin": "SFO", "destination": "LAX", '
    '"date": "2026-01-02", "passengers": 2}',
]


class _Config:
    """The two paths the conformance check reads."""

    def __init__(self, output_file: Path):
        self._output_file = output_file

    def get_test_case_file(self) -> str:
        return str(_FIXTURE)

    def get_output_json_file(self) -> str:
        return str(self._output_file)


def _config_with(tmp_path: Path, responses: list) -> _Config:
    output_file = tmp_path / "output.json"
    output_file.write_text(json.dumps({"responses": responses}))
    return _Config(output_file)


def test_bounded_guides_must_finish(tmp_path):
    """Jumping once and then hitting the budget on every request is a failure, not a pass."""
    config = _config_with(tmp_path, [{
        "output_text": "INVALID",
        "finish_reason": "max-length"
    }] * 7)
    with pytest.raises(RuntimeError, match="generation budget"):
        _check_guided_decoding_conformance(config, require_completion=True)


def test_free_form_guides_may_hit_the_budget(tmp_path):
    """json_object and structural_tag answers are the model's text, so the budget may end them."""
    responses = [{
        "output_text": text,
        "finish_reason": "end-of-sequence"
    } for text in _VALID_TEXTS]
    for index in (1, 4):
        responses[index]["finish_reason"] = "max-length"
    _check_guided_decoding_conformance(_config_with(tmp_path, responses),
                                       require_completion=True)


_JUMPED = """[INFO] Request 0 jump-forward: jumps=2 forced=2 rolledBack=0 chunkSteps=3 skips{}
[INFO] Request 1 jump-forward: jumps=0 forced=0 rolledBack=0 chunkSteps=0 skips{}
"""


def test_passes_when_a_request_jumped():
    """A run with at least one jump and no refusal is accepted."""
    _check_jump_forward_ran(_JUMPED)


def test_fails_when_no_request_jumped():
    """Statistics lines with zero jumps everywhere mean the grafting did not run."""
    with pytest.raises(RuntimeError, match="none jumped"):
        _check_jump_forward_ran(
            _JUMPED.replace("jumps=2 forced=2", "jumps=0 forced=0"))


def test_fails_without_statistics_lines():
    """Output without a jump-forward line means the requests ran without the flag."""
    with pytest.raises(RuntimeError, match="did not run with jump-forward"):
        _check_jump_forward_ran("[INFO] Request 0 finished\n")


def test_fails_when_the_flag_was_refused():
    """A refusal warning fails the check even when other lines report jumps."""
    output = (
        "[WARNING] Jump-forward is not supported on speculative engines without verify "
        "headroom; guided requests run without it.\n" + _JUMPED)
    with pytest.raises(RuntimeError, match="refused"):
        _check_jump_forward_ran(output)
