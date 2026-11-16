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
"""Action-chunk numerical parity metrics for policy evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare_actions(actual, reference):
    """Compare every batch, timestep and action dimension in matching chunks."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.ndim == 2:
        actual = actual[None]
    if reference.ndim == 2:
        reference = reference[None]
    if (actual.shape != reference.shape or actual.ndim != 3
            or actual.size == 0):
        raise ValueError(
            f"Action shape mismatch: {actual.shape}, {reference.shape}")
    if not (np.isfinite(actual).all() and np.isfinite(reference).all()):
        raise ValueError("Action chunks must contain finite values")

    def cosine(left, right):
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator == 0:
            return float(np.array_equal(left, right))
        return float(np.clip(left.ravel() @ right.ravel() / denominator, -1,
                             1))

    error = np.abs(actual - reference)
    return {
        "cosine":
        cosine(actual, reference),
        "worst_timestep_cosine":
        min(
            cosine(left, right) for left, right in zip(
                actual.reshape(-1, actual.shape[-1]),
                reference.reshape(-1, reference.shape[-1]))),
        "max_abs_error":
        float(error.max()),
        "mean_abs_error":
        float(error.mean()),
        "rmse":
        float(np.sqrt(np.mean(error**2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--json-field",
                        default="actions",
                        choices=("actions", "action", "robot_actions"))
    parser.add_argument("--cosine-floor", type=float, required=True)
    parser.add_argument("--max-abs-ceiling", type=float, required=True)
    args = parser.parse_args()

    def load(path):
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=False)
        return json.loads(Path(path).read_text())[args.json_field]

    metrics = compare_actions(load(args.actual), load(args.reference))
    print(json.dumps(metrics, indent=2))
    return int(metrics["cosine"] < args.cosine_floor
               or metrics["max_abs_error"] > args.max_abs_ceiling)


if __name__ == "__main__":
    raise SystemExit(main())
