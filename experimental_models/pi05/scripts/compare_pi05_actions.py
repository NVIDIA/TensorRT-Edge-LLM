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
"""Score a pi0.5 action chunk against an openpi reference.

Numerical parity only: it says the TensorRT chunk matches the PyTorch one for a
fixed observation and a fixed x_0. Task success needs a simulator and is not this.
"""
import argparse
import json
import struct
import sys

import numpy as np

#: All three configurations clear these on their own embodiment's frames. Scored on another
#: embodiment's frames they run several times looser, so the comparison is only meaningful in
#: distribution; the model guide records the measurements.
COSINE_FLOOR = 0.99999
MAX_ABS_CEILING = 5e-3


def _read_safetensors(path):
    """First tensor of a .safetensors file, as float64."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        name = next(k for k in header if k != "__metadata__")
        entry = header[name]
        start, end = entry["data_offsets"]
        handle.seek(8 + header_len + start)
        raw = handle.read(end - start)
    dtypes = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
    if entry["dtype"] not in dtypes:
        raise ValueError(f"{path}: unsupported dtype {entry['dtype']}")
    return np.frombuffer(raw, dtype=dtypes[entry["dtype"]]).reshape(
        entry["shape"]).astype(np.float64)


def _read_json_actions(path, field):
    with open(path) as handle:
        response = json.load(handle)
    if field not in response:
        raise ValueError(
            f"{path}: no \"{field}\"; it holds {sorted(response)}")
    return np.asarray(response[field], dtype=np.float64)


def read_chunk(path, field="actions"):
    """An action chunk from .safetensors, .npy, or the CLI's JSON response.

    ``field`` picks between the JSON response's normalized ``actions`` and its
    ``robot_actions``, which are the same chunk in robot units.
    """
    if path.endswith(".safetensors"):
        return _read_safetensors(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return _read_json_actions(path, field)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",
                        required=True,
                        help="openpi chunk: .npy, .safetensors or .json")
    parser.add_argument("--actual",
                        required=True,
                        help="pi05_policy_inference --output")
    parser.add_argument("--cosine-floor", type=float, default=COSINE_FLOOR)
    parser.add_argument("--max-abs-ceiling",
                        type=float,
                        default=MAX_ABS_CEILING)
    parser.add_argument(
        "--json-field",
        default="actions",
        choices=["actions", "robot_actions"],
        help="which chunk to read from a JSON response (default: actions)")
    args = parser.parse_args()

    reference = read_chunk(args.reference, args.json_field)
    actual = read_chunk(args.actual, args.json_field)
    # Only a single-request dump's leading [1, ...] axis is dropped: squeezing a real
    # [batch, horizon, dim] run would let entries 1..N-1 be wrong and still pass.
    while reference.ndim > 2 and reference.shape[0] == 1:
        reference = reference[0]
    while actual.ndim > 2 and actual.shape[0] == 1:
        actual = actual[0]
    if actual.shape != reference.shape:
        if actual.ndim == reference.ndim + 1 and actual.shape[
                1:] == reference.shape:
            reference = np.broadcast_to(reference, actual.shape)
            print(
                f"scoring all {actual.shape[0]} batch entries against the reference"
            )
        else:
            print(
                f"FAIL shape {actual.shape} against reference {reference.shape}"
            )
            return 1
    if not np.isfinite(actual).all():
        print(f"FAIL {int((~np.isfinite(actual)).sum())} non-finite elements")
        return 1

    # Per entry, then worst-of: one flat cosine over a batch lets a good entry carry a bad one.
    entries_ref = reference if reference.ndim > 2 else reference[None]
    entries_act = actual if actual.ndim > 2 else actual[None]
    cosines = []
    for ref_entry, act_entry in zip(entries_ref, entries_act):
        flat_ref, flat_act = ref_entry.ravel(), act_entry.ravel()
        cosines.append(
            float(flat_ref @ flat_act /
                  (np.linalg.norm(flat_ref) * np.linalg.norm(flat_act))))
    cosine = min(cosines)
    max_abs = float(np.abs(reference - actual).max())
    ok = cosine >= args.cosine_floor and max_abs <= args.max_abs_ceiling

    print(f"shape       {actual.shape}")
    if len(cosines) > 1:
        worst = int(np.argmin(cosines))
        print(f"cosine      {cosine:.8f}  (floor {args.cosine_floor}, "
              f"worst of {len(cosines)} entries, entry {worst})")
    else:
        print(f"cosine      {cosine:.8f}  (floor {args.cosine_floor})")
    print(f"max abs     {max_abs:.3e}  (ceiling {args.max_abs_ceiling:.0e})")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
