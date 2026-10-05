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

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


def select_subject_prefixes(requests, sample_count):
    """Select near-equal per-subject prefixes while preserving source order."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(requests):
        raise ValueError(
            f"sample_count ({sample_count}) exceeds request count "
            f"({len(requests)})")

    subject_rows = defaultdict(list)
    for source_index, request in enumerate(requests):
        subject = request.get("subject")
        if not subject:
            raise ValueError(
                f"Request at source index {source_index} has no subject")
        subject_rows[subject].append((source_index, request))

    subjects = sorted(subject_rows)
    if sample_count < len(subjects):
        raise ValueError(f"sample_count ({sample_count}) must cover all "
                         f"{len(subjects)} subjects")

    per_subject, remainder = divmod(sample_count, len(subjects))
    selected_rows = []
    subject_counts = {}
    for subject_index, subject in enumerate(subjects):
        quota = per_subject + (1 if subject_index < remainder else 0)
        available = subject_rows[subject]
        if len(available) < quota:
            raise ValueError(
                f"Subject {subject!r} has {len(available)} rows, fewer than "
                f"the required prefix quota of {quota}")
        selected_rows.extend(available[:quota])
        subject_counts[subject] = quota

    selected_rows.sort(key=lambda row: row[0])
    return [request for _, request in selected_rows], subject_counts


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_mmlu_lite_dataset(input_file, output_dir, sample_count=2000):
    """Create deterministic Lite data from an existing MMLU Full JSON file."""
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    source = json.loads(input_file.read_text(encoding="utf-8"))
    requests = source.get("requests")
    if not isinstance(requests, list):
        raise ValueError("Input dataset must contain a requests list")

    selected, subject_counts = select_subject_prefixes(requests, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "mmlu_dataset.json"
    lite = dict(source)
    lite["requests"] = selected
    dataset_path.write_text(json.dumps(lite, indent=4) + "\n",
                            encoding="utf-8")

    manifest = {
        "source_file": input_file.name,
        "source_sha256": _sha256(input_file),
        "source_request_count": len(requests),
        "sample_count": sample_count,
        "subject_count": len(subject_counts),
        "selection_rule": "first_n_per_subject_in_source_order",
        "remainder_rule":
        "first_subjects_in_lexicographic_order_receive_one_extra",
        "subject_samples": dict(sorted(subject_counts.items())),
        "dataset_file": dataset_path.name,
        "dataset_sha256": _sha256(dataset_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    return dataset_path, manifest_path


class MMLUDataset(EdgeLLMDataset):
    """
    Example implementation for MMLU dataset. Supports the following datasets:
    https://huggingface.co/datasets/cais/mmlu
    
    MMLU data format:
    {
        'question': 'What is the capital of France?',
        'choices': "['Option A', 'Option B', ...]",
        'answer': '1' (index of the correct answer),
        'subject': 'abstract_algebra',
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
        """Format a single MMLU example with or without the answer."""
        question = data["question"]
        choices = data["choices"]

        example = question
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            example += f"\n{letter}. {choice}"

        example += "\nAnswer:"
        if include_answer:
            answer_letter = chr(ord('A') + data["answer"])
            example += f" {answer_letter}\n\n"
        else:
            example += " "

        return example

    def _get_few_shot_examples(self, subject: str) -> List[Dict[str, Any]]:
        """Get few-shot examples for a given subject from the dev dataset."""
        if not self.dev_dataset or self.num_shot <= 0:
            return []

        # Filter dev dataset by subject
        subject_examples = [
            ex for ex in self.dev_dataset if ex.get("subject") == subject
        ]

        # Return up to num_shot examples
        return subject_examples[:self.num_shot]

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMLU prompt with question and multiple choice options."""

        assert "question" in data, "question is required"
        assert "choices" in data, "choices is required"
        assert "answer" in data, "answer is required"
        assert "subject" in data, "subject is required"

        # Build user prompt with few-shot examples prepended
        user_prompt = ""

        # Format subject name and add header
        subject_fmt = data["subject"].replace("_", " ")
        user_prompt += f"The following are multiple choice questions (with answers) about {subject_fmt}.\n\n"

        # Add few-shot examples if available
        few_shot_examples = self._get_few_shot_examples(data["subject"])
        for example in few_shot_examples:
            user_prompt += self._format_single_example(example,
                                                       include_answer=True)

        # Add the current question
        user_prompt += self._format_single_example(data, include_answer=False)
        return user_prompt

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        """No system prompt for MMLU."""
        return ""

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from MMLU data."""
        assert "answer" in data, "answer is required"
        answer = data["answer"]
        assert isinstance(answer, int), "answer must be an integer"
        return chr(ord('A') + answer)


def convert_mmlu_dataset(config: DatasetConfig,
                         dataset_name_or_dir: str = "cais/mmlu",
                         output_dir: Union[str, os.PathLike] = "mmlu_dataset",
                         num_shot: int = 5):
    """
    Convert MMLU dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        num_shot: Number of examples to include for few-shot learning (5 = matches C++ implementation)
    """
    # https://huggingface.co/datasets/cais/mmlu
    if "cais/mmlu" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MMLU dataset from {dataset_name_or_dir} to {output_dir}")
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"Loaded MMLU dataset with {len(mmlu_dataset)} examples")

    # Load dev dataset for few-shot examples if needed
    dev_dataset = None
    if num_shot > 0:
        dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
        print(
            f"Loaded MMLU dev dataset with {len(dev_dataset)} examples for few-shot learning"
        )

    # Use provided config

    edge_llm_mmlu_dataset = MMLUDataset(dataset=mmlu_dataset,
                                        config=config,
                                        dev_dataset=dev_dataset,
                                        num_shot=num_shot,
                                        output_dir=output_dir)

    print(f"Processing MMLU dataset with config: {config}")
    edge_llm_mmlu_dataset.process_and_save_dataset("mmlu_dataset.json")

    print(f"Successfully converted MMLU dataset to {output_dir}")
    return edge_llm_mmlu_dataset
