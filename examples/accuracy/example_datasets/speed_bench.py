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
"""Download and convert SPEED-Bench throughput data to Edge-LLM requests."""

import os
import sys
from typing import Any, Dict, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset

SPEED_BENCH_REPO_ID = "nvidia/SPEED-Bench"
SPEED_BENCH_CONFIGS = (
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
_MASKED_PROMPT = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)


class SPEEDBenchDataset(EdgeLLMDataset):
    """Edge-LLM request formatter for a SPEED-Bench throughput split."""

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        return data["turns"][0]


def _has_available_prompt(entry: Dict[str, Any]) -> bool:
    turns = entry["turns"]
    return (not entry["multiturn"] and len(turns) == 1 and bool(turns[0])
            and not turns[0].startswith(_MASKED_PROMPT))


def convert_speed_bench_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str = SPEED_BENCH_REPO_ID,
    output_dir: Union[str, os.PathLike] = "speed_bench_dataset",
    config_name: str = "throughput_1k",
) -> SPEEDBenchDataset:
    """Download and convert one SPEED-Bench throughput configuration."""
    if config_name not in SPEED_BENCH_CONFIGS:
        supported = ", ".join(SPEED_BENCH_CONFIGS)
        raise ValueError(
            f"Unsupported SPEED-Bench config {config_name!r}; use {supported}")

    print(f"Loading SPEED-Bench {config_name} from {dataset_name_or_dir}")
    dataset = load_dataset(dataset_name_or_dir, config_name, split="test")
    available_dataset = dataset.filter(_has_available_prompt)
    print(f"Skipping {len(dataset) - len(available_dataset)} masked prompts")
    if len(available_dataset) == 0:
        raise ValueError(
            f"SPEED-Bench config {config_name} has no available prompts")

    dataset = available_dataset.rename_column("question_id", "id")
    dataset = dataset.rename_column("sub_category", "question_type")
    edge_dataset = SPEEDBenchDataset(dataset=dataset,
                                     config=config,
                                     output_dir=output_dir)
    edge_dataset.process_and_save_dataset(f"speed_bench_{config_name}.json")
    return edge_dataset
