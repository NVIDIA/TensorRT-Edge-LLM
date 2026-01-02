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


class MMStarDataset(EdgeLLMDataset):
    """
    Example implementation for MMStar dataset. Supports the following datasets:
    https://huggingface.co/datasets/Lin-Chen/MMStar
    
    MMStar data format:
    {
        'question': 'What is the main subject of this image?',
        'answer': 'A',
        'category': 'Image Recognition',
        'l2-category': 'Visual Recognition',
        'meta_info': {
            'image_path': 'images/001.jpg'
        },
        'image': <PIL Image object>
    }
    
    Note: This is a multimodal dataset that requires image processing.
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMStar prompt with question and image."""

        assert "question" in data, "question is required"
        assert "image" in data, "image is required"
        prompt = f"{data['question']}\n\n"
        return prompt

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save MMStar image and return relative path."""
        image_paths = []

        if "image" in data and data["image"] is not None:
            # Fallback to hash-based naming
            image_filename = f"image_{data['index']}.jpg"

            image_path = os.path.join(self.images_dir, image_filename)
            image = data["image"]
            # Ensure 3-channel RGB format for JPEG
            image = image.convert('RGB')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

        return image_paths

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer from MMStar data."""
        assert "answer" in data, "answer is required"
        return str(data["answer"]).strip()


def convert_mmstar_dataset(config: DatasetConfig,
                           dataset_name_or_dir: str = "Lin-Chen/MMStar",
                           output_dir: Union[str,
                                             os.PathLike] = "mmstar_dataset"):
    """
    Convert MMStar dataset to TensorRT Edge-LLM format.
    
    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/Lin-Chen/MMStar
    if "MMStar" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MMStar dataset from {dataset_name_or_dir} to {output_dir}"
    )

    # Load dataset
    mmstar_dataset = load_dataset("Lin-Chen/MMStar", split="val")
    print(f"Loaded MMStar dataset with {len(mmstar_dataset)} examples")

    # Use provided config

    # Create dataset processor
    edge_llm_mmstar_dataset = MMStarDataset(dataset=mmstar_dataset,
                                            config=config,
                                            output_dir=output_dir)

    print(f"Processing MMStar dataset with config: {config}")
    edge_llm_mmstar_dataset.process_and_save_dataset("mmstar_dataset.json")

    print(f"Successfully converted MMStar dataset to {output_dir}")
    return edge_llm_mmstar_dataset
