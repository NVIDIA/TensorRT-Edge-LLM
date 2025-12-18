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


class GSM8KDataset(EdgeLLMDataset):
    """
    Example implementation for GSM8K dataset. Supports the following datasets:
    https://huggingface.co/datasets/openai/gsm8k
    
    GSM8K data format:
    {
        'question': 'Natalia sold clips to 48 of her friends in April...?',
        'answer': "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format GSM8K prompt with question."""

        assert "question" in data, "question is required"
        prompt = f"{data['question']}\n\n"

        return prompt

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from GSM8K data."""
        assert "answer" in data, "answer is required"
        answer = data["answer"]
        return answer


def convert_gsm8k_dataset(config: DatasetConfig,
                          dataset_name_or_dir: str = "openai/gsm8k",
                          output_dir: Union[str,
                                            os.PathLike] = "gsm8k_dataset"):
    """
    Convert GSM8K dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/openai/gsm8k
    if "gsm8k" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting GSM8K dataset from {dataset_name_or_dir} to {output_dir}")

    # Load dataset
    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded GSM8K dataset with {len(gsm8k_dataset)} examples")

    # Use provided config

    # Create dataset processor
    edge_llm_gsm8k_dataset = GSM8KDataset(dataset=gsm8k_dataset,
                                          config=config,
                                          output_dir=output_dir)

    print(f"Processing GSM8K dataset with config: {config}")
    edge_llm_gsm8k_dataset.process_and_save_dataset("gsm8k_dataset.json")

    print(f"Successfully converted GSM8K dataset to {output_dir}")
    return edge_llm_gsm8k_dataset
