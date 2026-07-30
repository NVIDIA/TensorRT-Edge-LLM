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
"""Build a ~10% BFCL calibration set for on-device quantization.

Reads the BFCL v4 single-turn parquet (materialized by the mlcommons
inference-endpoints harness at dataset_cache/bfcl_v4/bfcl_v4_single_turn.parquet
after any BFCL run) and writes a JSONL of {messages, tools} records, stratified
~10% per subset. tensorrt_edgellm.quantization.quantize renders these through the
tokenizer chat template (tool-aware, reasoning-off) for calibration.

Usage: python make_calibration.py <out.jsonl> <bfcl_v4_single_turn.parquet> [pct]
"""
import json
import random
import sys

import pandas as pd

out = sys.argv[1]
parquet = sys.argv[2]
pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

df = pd.read_parquet(parquet)
rows = []
for subset, g in df.groupby("subset"):
    g = g.reset_index(drop=True)
    n = max(1, round(len(g) * pct))
    step = max(1, len(g) // n)
    for _, r in g.iloc[::step].head(n).iterrows():
        rows.append({
            "messages": json.loads(r["messages"]),
            "tools": json.loads(r["tools"]),
            "subset": r["subset"]
        })
random.seed(42)
random.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} calibration samples (~{pct:.0%}) to {out}")
