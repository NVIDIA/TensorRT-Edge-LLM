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
Dataset preparation script for TensorRT Edge-LLM.

This script converts various datasets (AIME, GSM8K, HumanEval, MATH500, MMLU, MMLU_Pro, MMMU, MMMU_Pro, MMStar, MTBench) 
into the format expected by the TensorRT Edge-LLM runtime.

Usage:
    # Using default dataset names
    python prepare_dataset.py --dataset AIME --output_dir aime_output
    python prepare_dataset.py --dataset GSM8K --output_dir gsm8k_output
    python prepare_dataset.py --dataset HumanEval --output_dir humaneval_output
    python prepare_dataset.py --dataset MATH500 --output_dir math500_output
    python prepare_dataset.py --dataset MMStar --output_dir mmstar_output
    
    # Explicit dataset names or local directory
    python prepare_dataset.py --dataset GSM8K --dataset_name_or_dir openai/gsm8k --output_dir gsm8k_output
"""

import argparse
import os
import sys

# Add the parent directory to the Python path to import example_datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_datasets.aime import convert_aime_dataset
from example_datasets.edgellm_dataset import DatasetConfig
from example_datasets.gsm8k import convert_gsm8k_dataset
from example_datasets.humaneval import convert_humaneval_dataset
from example_datasets.math500 import convert_math500_dataset
from example_datasets.mmlu import convert_mmlu_dataset
from example_datasets.mmlu_pro import convert_mmlu_pro_dataset
from example_datasets.mmmu import (convert_mmmu_dataset,
                                   convert_mmmu_pro_dataset)
from example_datasets.mmstar import convert_mmstar_dataset
from example_datasets.mtbench import convert_mtbench_dataset

# Default dataset mappings
DEFAULT_DATASETS = {
    "AIME": "Maxwell-Jia/AIME_2024",
    "GSM8K": "openai/gsm8k",
    "HumanEval": "openai/openai_humaneval",
    "MATH500": "HuggingFaceH4/MATH-500",
    "MMLU": "cais/mmlu",
    "MMLU_Pro": "TIGER-Lab/MMLU-Pro",
    "MMMU": "MMMU/MMMU",
    "MMMU_VLMEvalkit": "MMMU/MMMU",
    "MMMU_Pro": "MMMU/MMMU_Pro",
    "MMStar": "Lin-Chen/MMStar",
    "MTBench": "philschmid/mt-bench"
}

# Default max_generate_length for each dataset
DEFAULT_MAX_GENERATE_LENGTHS = {
    "AIME": 512,
    "GSM8K": 512,
    "HumanEval": 512,
    "MATH500": 512,
    "MMLU": 1,
    "MMLU_Pro": 1,
    "MMMU": 20,
    "MMMU_VLMEvalkit": 8192,
    "MMMU_Pro": 1,
    "MMStar": 512,
    "MTBench": 512
}


def get_dataset_path(dataset_type: str,
                     dataset_name_or_dir: str = None) -> str:
    """
    Determine the dataset path to use, checking for local cache first.
    
    Args:
        dataset_type: Type of dataset (AIME, GSM8K, HumanEval, MATH500, MMLU, MMLU_Pro, MMMU, MMMU_Pro, MMStar, MTBench)
        dataset_name_or_dir: User-specified dataset name or directory
        
    Returns:
        Dataset path to use (local directory or HuggingFace dataset name)
    """
    # If user explicitly provided a dataset name/dir, use it
    if dataset_name_or_dir:
        return dataset_name_or_dir

    # Get default dataset name
    default_name = DEFAULT_DATASETS.get(dataset_type)
    if not default_name:
        raise ValueError(f"No default dataset name for {dataset_type}")

    # Fall back to default HuggingFace dataset name
    print(f"Using HuggingFace dataset: {default_name}")
    return default_name


def main():
    """Main function to parse arguments and call appropriate conversion function."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for TensorRT Edge-LLM inference", )

    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        choices=[
                            "AIME", "GSM8K", "HumanEval", "MATH500", "MMLU",
                            "MMLU_Pro", "MMMU", "MMMU_VLMEvalkit", "MMMU_Pro",
                            "MMStar", "MTBench"
                        ],
                        help="Dataset type to convert")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory to save converted dataset")

    parser.add_argument(
        "--dataset_name_or_dir",
        type=str,
        required=False,
        help=
        "HuggingFace dataset name or local directory path. If not provided, uses default dataset name and automatically checks for local cache."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="vision",
        required=False,
        help=
        "Subset for MMMU_Pro dataset (vision, 'standard (10 options)', 'standard (4 options)')"
    )

    parser.add_argument(
        "--num_shot",
        type=int,
        default=5,
        required=False,
        help=
        "Number of examples to include for few-shot learning (0 = zero-shot). Currently only supported for MMLU dataset. Default is 5."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Number of messages per batch (for configuration)")

    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        required=False,
                        help="Sampling temperature (0.0-2.0)")

    parser.add_argument("--top_p",
                        type=float,
                        default=1.0,
                        required=False,
                        help="Nucleus sampling parameter (0.0-1.0)")

    parser.add_argument("--top_k",
                        type=int,
                        default=50,
                        required=False,
                        help="Top-k sampling parameter")

    parser.add_argument(
        "--max_generate_length",
        type=int,
        required=False,
        help=
        "Maximum number of tokens to generate (uses dataset default if not provided)"
    )

    args = parser.parse_args()

    try:
        print(f"Converting {args.dataset} dataset...")

        # Determine dataset path (local cache or HuggingFace name)
        dataset_path = get_dataset_path(
            dataset_type=args.dataset,
            dataset_name_or_dir=args.dataset_name_or_dir)

        print(f"Dataset source: {dataset_path}")
        print(f"Output directory: {args.output_dir}")

        if args.dataset == "AIME":
            # Create config for AIME
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "AIME"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_aime_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir)

        elif args.dataset == "GSM8K":
            # Create config for GSM8K
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "GSM8K"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_gsm8k_dataset(config=config,
                                  dataset_name_or_dir=dataset_path,
                                  output_dir=args.output_dir)

        elif args.dataset == "MMLU":
            # Create config for MMLU
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMLU"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmlu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir,
                                 num_shot=args.num_shot)

        elif args.dataset == "MMLU_Pro":
            # Create config for MMLU-Pro
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMLU_Pro"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmlu_pro_dataset(config=config,
                                     dataset_name_or_dir=dataset_path,
                                     output_dir=args.output_dir,
                                     num_shot=args.num_shot)

        elif args.dataset == "MMMU":
            # Create config for MMMU
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMMU"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmmu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir)

        elif args.dataset == "MMMU_VLMEvalkit":
            # Create config for MMMU
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMMU_VLMEvalkit"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmmu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir,
                                 vlmevalkit=True)

        elif args.dataset == "MMMU_Pro":
            print(f"Using subset: {args.subset}")
            # Create config for MMMU_Pro
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMMU_Pro"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmmu_pro_dataset(config=config,
                                     dataset_name_or_dir=dataset_path,
                                     output_dir=args.output_dir,
                                     subset=args.subset)

        elif args.dataset == "HumanEval":
            # Create config for HumanEval
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "HumanEval"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_humaneval_dataset(config=config,
                                      dataset_name_or_dir=dataset_path,
                                      output_dir=args.output_dir)

        elif args.dataset == "MATH500":
            # Create config for MATH500
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MATH500"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_math500_dataset(config=config,
                                    dataset_name_or_dir=dataset_path,
                                    output_dir=args.output_dir)

        elif args.dataset == "MMStar":
            # Create config for MMStar
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MMStar"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mmstar_dataset(config=config,
                                   dataset_name_or_dir=dataset_path,
                                   output_dir=args.output_dir)

        elif args.dataset == "MTBench":
            # Create config for MTBench
            max_generate_length = args.max_generate_length or DEFAULT_MAX_GENERATE_LENGTHS[
                "MTBench"]
            config = DatasetConfig(batch_size=args.batch_size,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   max_generate_length=max_generate_length)
            convert_mtbench_dataset(config=config,
                                    dataset_name_or_dir=dataset_path,
                                    output_dir=args.output_dir)
    except Exception as e:
        print(f"Error converting dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
