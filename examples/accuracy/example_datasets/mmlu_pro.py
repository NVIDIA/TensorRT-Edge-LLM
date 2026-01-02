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
from typing import Any, Dict, List, Optional, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class MMLUProDataset(EdgeLLMDataset):
    """
    Example implementation for MMLU-Pro dataset. Supports the following datasets:
    https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    
    MMLU-Pro data format:
    {
        'question_id': 1,
        'question': 'What is the capital of France?',
        'options': ['Option A', 'Option B', ..., 'N/A', 'N/A'],  # Up to 10 options
        'answer': 'A',  # Correct answer letter (A-J)
        'answer_index': 0,  # Index of correct answer (0-9)
        'cot_content': 'Chain of thought reasoning...',
        'category': 'geography',
        'src': 'subcategory'
    }
    """

    def __init__(self,
                 dataset: Dataset,
                 config: DatasetConfig,
                 dev_dataset: Optional[Dataset] = None,
                 num_shot: int = 5,
                 **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.dev_dataset = dev_dataset
        self.num_shot = num_shot

    def _format_single_example(self,
                               data: Dict[str, Any],
                               include_answer: bool = True) -> str:
        """Format a single MMLU-Pro example with or without the answer."""
        question = data["question"]
        options = data["options"]

        example = question

        # Filter out 'N/A' options and format remaining ones
        valid_options = [opt for opt in options if opt != "N/A"]

        for i, choice in enumerate(valid_options):
            letter = chr(ord('A') + i)
            example += f"\n{letter}. {choice}"

        example += "\nAnswer:"
        if include_answer:
            answer_letter = data["answer"]
            example += f" {answer_letter}\n\n"
        else:
            example += ""

        return example

    def _get_few_shot_examples(self, category: str) -> List[Dict[str, Any]]:
        """Get few-shot examples for a given category from the dev dataset."""
        if not self.dev_dataset or self.num_shot <= 0:
            return []

        # Filter dev dataset by category
        category_examples = [
            ex for ex in self.dev_dataset if ex.get("category") == category
        ]

        # Return up to num_shot examples
        return category_examples[:self.num_shot]

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMLU-Pro prompt with question and multiple choice options."""

        assert "question" in data, "question is required"
        assert "options" in data, "options is required"
        assert "answer" in data, "answer is required"
        assert "category" in data, "category is required"

        # Build user prompt with few-shot examples prepended
        user_prompt = ""

        # Format category name and add header
        category_fmt = data["category"].replace("_", " ")
        user_prompt += f"The following are multiple choice questions (with answers) about {category_fmt}.\n\n"

        # Add few-shot examples if available
        few_shot_examples = self._get_few_shot_examples(data["category"])
        for example in few_shot_examples:
            user_prompt += self._format_single_example(example,
                                                       include_answer=True)

        # Add the current question
        user_prompt += self._format_single_example(data, include_answer=False)
        return user_prompt

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from MMLU-Pro data."""
        assert "answer" in data, "answer is required"
        answer = data["answer"]
        assert isinstance(answer, str), "answer must be a string (letter A-J)"
        return answer


def convert_mmlu_pro_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "TIGER-Lab/MMLU-Pro",
        output_dir: Union[str, os.PathLike] = "mmlu_pro_dataset",
        num_shot: int = 5):
    """
    Convert MMLU-Pro dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        num_shot: Number of examples to include for few-shot learning (5 = matches C++ implementation)
    """
    # https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    if "TIGER-Lab/MMLU-Pro" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MMLU-Pro dataset from {dataset_name_or_dir} to {output_dir}"
    )
    mmlu_pro_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"Loaded MMLU-Pro dataset with {len(mmlu_pro_dataset)} examples")

    # Load validation dataset for few-shot examples if needed
    dev_dataset = None
    if num_shot > 0:
        dev_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        print(
            f"Loaded MMLU-Pro validation dataset with {len(dev_dataset)} examples for few-shot learning"
        )

    # Use provided config
    edge_llm_mmlu_pro_dataset = MMLUProDataset(dataset=mmlu_pro_dataset,
                                               config=config,
                                               dev_dataset=dev_dataset,
                                               num_shot=num_shot,
                                               output_dir=output_dir,
                                               apply_chat_template=False)

    print(f"Processing MMLU-Pro dataset with config: {config}")
    edge_llm_mmlu_pro_dataset.process_and_save_dataset("mmlu_pro_dataset.json")

    print(f"Successfully converted MMLU-Pro dataset to {output_dir}")
    return edge_llm_mmlu_pro_dataset
