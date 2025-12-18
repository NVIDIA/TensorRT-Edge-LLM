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

import ast
import os
import sys
from typing import Any, Dict, List, Optional, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import (Dataset, concatenate_datasets, get_dataset_config_names,
                      load_dataset)
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class MMMUDataset(EdgeLLMDataset):
    """
    Example implementation for MMMU dataset and MMMU_Pro dataset. Supports the following datasets:
    https://huggingface.co/datasets/MMMU/MMMU/
    https://huggingface.co/datasets/MMMU/MMMU_Pro
    
    MMMU data format:
    {
        'id': 'test_History_1',
        'image_1': <PIL.Image>,
        'options': "['Option A', 'Option B', ...]",
        'answer': 'B',
        'subject': 'History'
    }
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMMU system prompt"""
        # vlm_accuracy.cpp doesn't use a system prompt, just user prompt
        return ""

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMMU prompt with question and multiple choice options"""
        # Start with the question
        if "question" in data:
            user_prompt = data["question"]
        else:
            # MMMU_Pro vision does not have a question. The question is in the image.
            user_prompt = "Please answer the question in the image."

        options_str = data["options"].strip("[]") if "options" in data else ""
        if options_str:
            options_list = ast.literal_eval(options_str)
            for i, option in enumerate(options_list):
                letter = chr(ord('A') + i)
                user_prompt += f"\n{letter}. {option}"  # Space after letter

            user_prompt += "\n\nAnswer with the option's letter from the given choices directly."
        else:
            user_prompt += "\n\nAnswer the question using a single word or phrase."

        return user_prompt

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save MMMU image and return relative path."""
        image_paths = []

        def save_image_for_key(data: Dict[str, Any], image_key: str):
            if image_key in data and data[image_key] is not None:
                image_filename = f"{data['id']}_{image_key}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                image = data[image_key]
                # Ensure 3-channel RGB format for JPEG
                image = image.convert('RGB')
                image.save(image_path, 'JPEG')
                image_paths.append(image_path)

        save_image_for_key(data, "image")
        # images are stored in the data with the keys image_1, image_2, ..., image_7
        valid_image_ids = [1, 2, 3, 4, 5, 6, 7]
        for image_id in valid_image_ids:
            save_image_for_key(data, f"image_{image_id}")

        return image_paths

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from MMMU data."""
        assert "answer" in data, "answer is required"
        return data["answer"]


def convert_mmmu_pro_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "MMMU/MMMU_Pro",
        output_dir: Union[str, os.PathLike] = "mmmu_pro_dataset",
        subset: str = "vision"):
    """
    Convert MMMU_Pro dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        subset: Dataset subset (vision, 'standard (10 options)', 'standard (4 options)')
    """
    # https://huggingface.co/datasets/MMMU/MMMU_Pro
    if "MMMU/MMMU_Pro" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )
    if subset not in [
            "vision", "standard (10 options)", "standard (4 options)"
    ]:
        raise ValueError(f"Unsupported subset: {subset}")

    print(
        f"Converting MMMU_Pro dataset from {dataset_name_or_dir} to {output_dir} with subset {subset}"
    )
    mmmu_dataset = load_dataset(dataset_name_or_dir, subset, split="test")
    print(f"Loaded MMMU_Pro dataset with {len(mmmu_dataset)} examples")

    # Use provided config

    edge_llm_mmmu_dataset = MMMUDataset(dataset=mmmu_dataset,
                                        config=config,
                                        output_dir=output_dir)

    print(f"Processing MMMU_Pro dataset with config: {config}")
    subset_name = subset.replace(' ', '_').replace('(', '_').replace(')', '_')
    edge_llm_mmmu_dataset.process_and_save_dataset(
        f"mmmu_pro_{subset_name}_dataset.json")

    print(f"Successfully converted MMMU_Pro dataset to {output_dir}")
    return edge_llm_mmmu_dataset


def convert_mmmu_dataset(config: DatasetConfig,
                         dataset_name_or_dir: str = "MMMU/MMMU",
                         output_dir: Union[str, os.PathLike] = "mmmu_dataset"):
    """
    Convert MMMU dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/MMMU/MMMU
    if "MMMU/MMMU" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MMMU dataset from {dataset_name_or_dir} to {output_dir}")
    configs = get_dataset_config_names("MMMU/MMMU")
    mmmu_datasets = []
    for config_name in configs:
        mmmu_dataset = load_dataset("MMMU/MMMU",
                                    config_name,
                                    split="validation")
        mmmu_datasets.append(mmmu_dataset)
    # concat the datasets
    concat_mmmu_dataset = concatenate_datasets(mmmu_datasets)
    print(f"Loaded MMMU dataset with {len(concat_mmmu_dataset)} examples")

    # Use provided config

    edge_llm_mmmu_dataset = MMMUDataset(dataset=concat_mmmu_dataset,
                                        config=config,
                                        output_dir=output_dir)

    print(f"Processing MMMU dataset with config: {config}")
    edge_llm_mmmu_dataset.process_and_save_dataset("mmmu_dataset.json")

    print(f"Successfully converted MMMU dataset to {output_dir}")
    return edge_llm_mmmu_dataset
