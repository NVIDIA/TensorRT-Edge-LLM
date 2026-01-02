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


class Math500Dataset(EdgeLLMDataset):
    """
    Example implementation for MATH-500 dataset. Supports the following datasets:
    https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    
    MATH-500 data format:
    {
        'problem': 'Find the sum of all values of $x$ such that $2^{x} + 4^{x} = 6^{x}$.',
        'solution': 'We have $2^{x} + 4^{x} = 6^{x}$.\n\nDividing both sides by $2^{x}$, we get $1 + 2^{x} = 3^{x}$.\n\nLet $y = 2^{x}$, so we have $1 + y = 3^{x}$.\n\nSince $y = 2^{x}$, we have $x = \\log_{2}(y)$, so $3^{x} = 3^{\\log_{2}(y)} = y^{\\log_{2}(3)}$.\n\nThus, $1 + y = y^{\\log_{2}(3)}$.\n\nLet $a = \\log_{2}(3)$, so we have $1 + y = y^{a}$.\n\nRearranging, we get $y^{a} - y - 1 = 0$.\n\nLet $f(y) = y^{a} - y - 1$. We have $f(0) = -1 < 0$ and $f(2) = 2^{a} - 2 - 1 = 2^{\\log_{2}(3)} - 3 = 3 - 3 = 0$.\n\nSo $y = 2$ is a root of $f(y) = 0$.\n\nSince $f\'(y) = ay^{a-1} - 1$ and $a = \\log_{2}(3) > 1$, we have $f\'(y) > 0$ for $y > (1/a)^{1/(a-1)}$.\n\nSince $a > 1$, we have $(1/a)^{1/(a-1)} < 1 < 2$, so $f\'(2) > 0$.\n\nAlso, $f\'\'(y) = a(a-1)y^{a-2} > 0$ for $y > 0$ since $a > 1$.\n\nSo $f$ is convex for $y > 0$.\n\nSince $f(2) = 0$ and $f$ is convex, $y = 2$ is the unique positive root of $f(y) = 0$.\n\nThus, $2^{x} = 2$, so $x = 1$.\n\nTherefore, the sum of all values of $x$ is $\\boxed{1}$.',
        'answer': '1',
        'subject': 'Intermediate Algebra',
        'level': 'Level 1'
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MATH-500 prompt with problem statement."""

        assert "problem" in data, "problem is required"
        prompt = f"{data['problem']}\n\n"

        return prompt

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from MATH-500 data."""
        assert "answer" in data, "answer is required"
        return str(data["answer"]).strip()


def convert_math500_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "HuggingFaceH4/MATH-500",
        output_dir: Union[str, os.PathLike] = "math500_dataset"):
    """
    Convert MATH-500 dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    if "MATH" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MATH-500 dataset from {dataset_name_or_dir} to {output_dir}"
    )

    # Load dataset
    math500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded MATH-500 dataset with {len(math500_dataset)} examples")

    # Use provided config

    # Create dataset processor
    edge_llm_math500_dataset = Math500Dataset(dataset=math500_dataset,
                                              config=config,
                                              output_dir=output_dir)

    print(f"Processing MATH-500 dataset with config: {config}")
    edge_llm_math500_dataset.process_and_save_dataset("math500_dataset.json")

    print(f"Successfully converted MATH-500 dataset to {output_dir}")
    return edge_llm_math500_dataset
