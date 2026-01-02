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
from typing import Any, Dict, Optional, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class HumanEvalDataset(EdgeLLMDataset):
    """
    Example implementation for HumanEval dataset. Supports the following datasets:
    https://huggingface.co/datasets/openai/openai_humaneval
    
    HumanEval data format:
    {
        'task_id': 'HumanEval/0',
        'prompt': 'Python function prompt with docstring and examples',
        'canonical_solution': 'Correct implementation of the function',
        'test': 'Test cases to validate the function',
        'entry_point': 'Function name to be implemented'
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format HumanEval prompt with code completion task."""

        assert "prompt" in data, "prompt is required"
        prompt = f"{data['prompt']}\n\n"

        return prompt

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from HumanEval data."""
        assert "canonical_solution" in data, "canonical_solution is required"
        return data["canonical_solution"].strip()


def convert_humaneval_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "openai/openai_humaneval",
        output_dir: Union[str, os.PathLike] = "humaneval_dataset"):
    """
    Convert HumanEval dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/openai/openai_humaneval
    if "humaneval" not in dataset_name_or_dir.lower():
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting HumanEval dataset from {dataset_name_or_dir} to {output_dir}"
    )

    # Load dataset
    humaneval_dataset = load_dataset("openai/openai_humaneval", split="test")
    print(f"Loaded HumanEval dataset with {len(humaneval_dataset)} examples")

    # Use provided config

    # Create dataset processor
    edge_llm_humaneval_dataset = HumanEvalDataset(dataset=humaneval_dataset,
                                                  config=config,
                                                  output_dir=output_dir)

    print(f"Processing HumanEval dataset with config: {config}")
    edge_llm_humaneval_dataset.process_and_save_dataset(
        "humaneval_dataset.json")

    print(f"Successfully converted HumanEval dataset to {output_dir}")
    return edge_llm_humaneval_dataset
