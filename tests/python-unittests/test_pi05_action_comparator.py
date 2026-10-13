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
"""The pi0.5 action comparator must not let one batch entry carry another."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_COMPARATOR = (Path(__file__).resolve().parents[2] / "experimental_models" /
               "pi05" / "scripts" / "compare_pi05_actions.py")


def _score(tmp_path, reference, actual):
    np.save(tmp_path / "ref.npy", reference)
    (tmp_path / "act.json").write_text(json.dumps({"actions":
                                                   actual.tolist()}))
    done = subprocess.run(
        [
            sys.executable,
            str(_COMPARATOR),
            "--reference",
            str(tmp_path / "ref.npy"),
            "--actual",
            str(tmp_path / "act.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout


def test_a_bad_second_entry_is_not_diluted_by_a_good_first(tmp_path):
    """Flattening the batch would average the two; the gate is worst-of-entries."""
    reference = np.random.RandomState(0).randn(10, 32).astype(np.float32)
    wrong = reference.copy()
    wrong[:, :4] *= -1.0
    code, out = _score(tmp_path, reference, np.stack([reference, wrong]))
    assert code == 1, out
    assert "FAIL" in out
    assert "entry 1" in out


def test_a_single_entry_passes(tmp_path):
    reference = np.random.RandomState(1).randn(10, 32).astype(np.float32)
    code, out = _score(tmp_path, reference, reference)
    assert code == 0, out
    assert "PASS" in out


def test_every_entry_must_clear_the_ceiling(tmp_path):
    """Entry 0 exact, entry 1 over max-abs but still well correlated."""
    reference = np.random.RandomState(2).randn(10, 32).astype(np.float32)
    over = reference.copy()
    over[0, 0] += 0.02
    code, out = _score(tmp_path, reference, np.stack([reference, over]))
    assert code == 1, out
    assert "FAIL" in out
