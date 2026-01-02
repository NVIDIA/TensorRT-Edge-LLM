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
"""
Base class for formatting datasets for TensorRT Edge-LLM inference.

This module provides a base class EdgeLLMDataset that can be extended to format
various datasets into the JSON structure expected by the TensorRT Edge-LLM runtime.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration class for dataset processing parameters."""
    batch_size: int
    temperature: float
    top_p: float
    top_k: int
    max_generate_length: int

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_generate_length <= 0:
            raise ValueError("max_generate_length must be positive")


class EdgeLLMDataset:
    """
    Base class for formatting datasets for TensorRT Edge-LLM inference.
    
    This class provides a framework for converting various datasets into the JSON format
    expected by the TensorRT Edge-LLM runtime, as specified in INPUT_FORMAT.md.
    
    Attributes:
        dataset_name (str): Name of the dataset
        output_dir (Path): Directory where images and output JSON will be saved
        batch_size (int): Number of messages to process in a single batch
        temperature (float): Controls randomness in generation
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        max_generate_length (int): Maximum number of tokens to generate
        formatted_data (List[Dict]): Processed dataset entries
    """

    def __init__(self,
                 dataset: Dataset,
                 config: DatasetConfig,
                 output_dir: Union[str, os.PathLike] = "./output",
                 apply_chat_template: Optional[bool] = None,
                 **kwargs):
        """
        Initialize the EdgeLLMDataset.
        
        Args:
            dataset: Dataset object
            config: DatasetConfig object with processing parameters (required)
            output_dir: Directory where images and output JSON will be saved
            apply_chat_template: Whether to apply chat template formatting (optional, defaults to None which means use default behavior)
            **kwargs: Additional parameters to override config values
        """
        self.dataset = dataset
        self.output_dir = str(output_dir)
        self.apply_chat_template = apply_chat_template

        # Override config with kwargs if provided
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        self.config = DatasetConfig(**config_dict)

        # Set attributes for backward compatibility
        self.batch_size = self.config.batch_size
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.top_k = self.config.top_k
        self.max_generate_length = self.config.max_generate_length

        self.formatted_data: List[Dict[str, Any]] = []

        # Create output directory and images subdirectory
        os.makedirs(self.output_dir, exist_ok=True)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format the user prompt from dataset entry.
        
        Args:
            data: Single dataset entry
            
        Returns:
            Formatted user prompt string
        """
        raise NotImplementedError("format_user_prompt is not implemented")

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """
        Save images from dataset entry and return their paths.
        
        Args:
            data: Single dataset entry containing image data
            
        Returns:
            List of relative paths to saved images
        """
        raise NotImplementedError("save_image is not implemented")

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract the reference answer from dataset entry.
        
        Args:
            data: Single dataset entry
            
        Returns:
            Reference answer string, or None if not available
        """
        raise NotImplementedError("extract_answer is not implemented")

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format the system prompt from dataset entry.
        """
        return ""

    def process_and_save_dataset(
            self,
            output_filename: Optional[str] = None,
            overwrite_formatted_prompts: bool = True) -> str:
        """
        Process the entire dataset and save it as JSON file compatible with TensorRT Edge-LLM.
        
        The output format follows the structure defined in INPUT_FORMAT.md:
        {
            "batch_size": <int>,
            "temperature": <float>,
            "top_p": <float>, 
            "top_k": <int>,
            "max_generate_length": <int>,
            "requests": [
                {
                    "messages": [
                        {"role": "user", "content": "<string or array>"}
                    ],
                    "answer": "<string>",  // optional
                    "id": "<string>",      // optional
                    "subject": "<string>"  // optional
                }
            ]
        }
        
        Args:
            output_filename: Name of the output JSON file. If None, uses dataset.json
            
        Returns:
            Path to the saved JSON file
        """
        self.formatted_data = []
        print(f"Processing {len(self.dataset)} examples...")
        pbar = tqdm(self.dataset, desc="Processing dataset")

        for idx, data_entry in enumerate(pbar):
            try:
                # Format the prompt
                user_prompt = self.format_user_prompt(data_entry)

                # Extract reference answer
                try:
                    answer = self.extract_answer(data_entry)
                except NotImplementedError:
                    answer = None

                # Save images and get paths
                try:
                    image_paths = self.save_image(data_entry)
                except NotImplementedError:
                    image_paths = None

                # Create request entry with messages array
                request = {}

                # Build messages array in OpenAI format
                messages = []

                # Add user message with content
                if image_paths:
                    # Multimodal: content is array with images and text
                    content = []
                    for img_path in image_paths:
                        content.append({"type": "image", "image": img_path})
                    content.append({"type": "text", "text": user_prompt})
                else:
                    # Text-only: content is string
                    content = user_prompt

                messages.append({"role": "user", "content": content})

                request["messages"] = messages

                # Add reference answer if available
                if answer:
                    request["answer"] = answer

                # Add any additional metadata
                if "id" in data_entry:
                    request["id"] = data_entry["id"]

                # Unified subject or category
                if "subject" in data_entry:
                    request["subject"] = data_entry["subject"]
                elif "category" in data_entry:
                    request["subject"] = data_entry["category"]

                if "question_type" in data_entry:
                    request["question_type"] = data_entry["question_type"]

                self.formatted_data.append(request)

            except Exception as e:
                print(f"Warning: Failed to process entry {idx}: {e}")
                continue

        print(f"Processed {len(self.formatted_data)} examples")

        # Save the processed data to JSON file
        if output_filename is None:
            output_filename = f"dataset.json"

        output_path = os.path.join(self.output_dir, output_filename)

        # Create the output structure
        output_data = {
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_generate_length": self.max_generate_length,
            "requests": self.formatted_data
        }

        # Add apply_chat_template if specified
        if self.apply_chat_template is not None:
            output_data["apply_chat_template"] = self.apply_chat_template

        # Save to JSON file with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(
            f"Successfully saved {len(self.formatted_data)} messages to {output_path}"
        )
        return str(output_path)
