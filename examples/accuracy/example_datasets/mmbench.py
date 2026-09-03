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

import os
import sys
from typing import Any, Dict, List, Optional, Union

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset

OPTION_LETTERS = ["A", "B", "C", "D"]


def _valid_option(value) -> bool:
    """MMBench pads missing options with NaN / 'nan' — treat those as absent."""
    if value is None:
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


class MMBenchDataset(EdgeLLMDataset):
    """
    Example implementation for the MMBench EN dev split:
    https://huggingface.co/datasets/lmms-lab/MMBench_EN

    MMBench data format:
    {
        'index': 241,
        'question': 'Which of the following ...?',
        'hint': 'Optional context paragraph (may be NaN)',
        'A': 'option text', 'B': '...', 'C': '...', 'D': '...',  # C/D may be NaN
        'answer': 'B',
        'category': 'identity_reasoning',
        'image': <PIL Image object>
    }

    Note: plain letter-match accuracy on the dev split. The official MMBench metric adds
    circular evaluation and GPT-assisted answer matching (VLMEvalKit); expect small
    differences from officially reported numbers.
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format MMBench prompt: optional hint, question, available options."""
        assert "question" in data, "question is required"
        assert "image" in data, "image is required"

        parts = []
        if _valid_option(data.get("hint")):
            parts.append(f"Hint: {str(data['hint']).strip()}")
        parts.append(str(data["question"]).strip())
        options = [
            f"{letter}. {str(data[letter]).strip()}"
            for letter in OPTION_LETTERS if _valid_option(data.get(letter))
        ]
        parts.append("Options:\n" + "\n".join(options))
        parts.append(
            "Answer with the option's letter from the given choices directly.")
        return "\n".join(parts) + "\n"

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save MMBench image and return its path."""
        image_paths = []
        if "image" in data and data["image"] is not None:
            image_filename = f"image_{data['index']}.jpg"
            image_path = os.path.join(self.images_dir, image_filename)
            data["image"].convert('RGB').save(image_path, 'JPEG')
            image_paths.append(image_path)
        return image_paths

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer letter from MMBench data."""
        assert "answer" in data, "answer is required"
        return str(data["answer"]).strip()


def convert_mmbench_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "lmms-lab/MMBench_EN",
        output_dir: Union[str, os.PathLike] = "mmbench_dataset"):
    """
    Convert the MMBench EN dev split to TensorRT Edge-LLM format.

    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    # https://huggingface.co/datasets/lmms-lab/MMBench_EN — or any local dataset directory
    # with the same schema (checked case-insensitively so local paths are not rejected).
    if "mmbench" not in dataset_name_or_dir.lower(
    ) and not os.path.isdir(dataset_name_or_dir):
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(
        f"Converting MMBench dataset from {dataset_name_or_dir} to {output_dir}"
    )

    mmbench_dataset = load_dataset(dataset_name_or_dir, split="dev")
    print(f"Loaded MMBench dataset with {len(mmbench_dataset)} examples")

    edge_llm_mmbench_dataset = MMBenchDataset(dataset=mmbench_dataset,
                                              config=config,
                                              output_dir=output_dir)

    print(f"Processing MMBench dataset with config: {config}")
    edge_llm_mmbench_dataset.process_and_save_dataset("mmbench_dataset.json")

    print(f"Successfully converted MMBench dataset to {output_dir}")
    return edge_llm_mmbench_dataset
