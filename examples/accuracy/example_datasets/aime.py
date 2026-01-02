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


class AIMEDataset(EdgeLLMDataset):
    """
    Example implementation for AIME 2024 dataset. Supports the following datasets:
    https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
    
    AIME data format:
    {
        'Problem': 'Find the number of integers n with 1 ≤ n ≤ 2024 such that...',
        'Answer': '123'
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format AIME prompt with problem statement."""

        assert "Problem" in data, "Problem is required"
        prompt = f"{data['Problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"

        return prompt

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from AIME data."""
        assert "Answer" in data, "Answer is required"
        return str(data["Answer"]).strip()


def convert_aime_dataset(config: DatasetConfig,
                         dataset_name_or_dir: str = "Maxwell-Jia/AIME_2024",
                         output_dir: Union[str, os.PathLike] = "aime_dataset"):
    """
    Convert AIME dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
    if "AIME" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting AIME dataset from {dataset_name_or_dir} to {output_dir}")

    # Load dataset
    aime_dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    print(f"Loaded AIME dataset with {len(aime_dataset)} examples")

    # Use provided config

    # Create dataset processor
    edge_llm_aime_dataset = AIMEDataset(dataset=aime_dataset,
                                        config=config,
                                        output_dir=output_dir)

    print(f"Processing AIME dataset with config: {config}")
    edge_llm_aime_dataset.process_and_save_dataset("aime_dataset.json")

    print(f"Successfully converted AIME dataset to {output_dir}")
    return edge_llm_aime_dataset
