# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
from typing import Any, Dict, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class MTBenchDataset(EdgeLLMDataset):
    """
    Example implementation for MTBench dataset. Supports the following datasets:
    https://huggingface.co/datasets/philschmid/mt-bench
    
    MTBench data format:
    {
        'question_id': '80',
        'category': "writing",
        'turns': [
            "Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as smells, sounds, and visual elements to create an immersive experience for the reader.", 
            "Rework your previous response. Begin each sentence with the subsequent letter of the alphabet, commencing from B." 
        ],
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MTBench prompt with the first turn of the conversation."""
        assert "turns" in data, "turns is required"
        prompt = f"{data['turns'][0]}"
        return prompt


def convert_mtbench_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "philschmid/mt-bench",
        output_dir: Union[str, os.PathLike] = "mtbench_dataset"):
    """
    Convert MTBench dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/philschmid/mt-bench
    if "philschmid/mt-bench" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MTBench dataset from {dataset_name_or_dir} to {output_dir}"
    )
    mtbench_dataset = load_dataset("philschmid/mt-bench", split="train")
    print(f"Loaded MTBench dataset with {len(mtbench_dataset)} examples")

    # Use provided config

    edge_llm_mtbench_dataset = MTBenchDataset(dataset=mtbench_dataset,
                                              config=config,
                                              output_dir=output_dir)

    print(f"Processing MTBench dataset with config: {config}")
    edge_llm_mtbench_dataset.process_and_save_dataset("mtbench_dataset.json")

    print(f"Successfully converted MTBench dataset to {output_dir}")
    return edge_llm_mtbench_dataset
