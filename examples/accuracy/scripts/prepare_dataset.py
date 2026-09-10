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
from typing import Tuple

# Add the parent directory to the Python path to import example_datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_datasets.aime import convert_aime_dataset
from example_datasets.coco import convert_coco_dataset
from example_datasets.edgellm_dataset import DatasetConfig
from example_datasets.gsm8k import convert_gsm8k_dataset
from example_datasets.humaneval import convert_humaneval_dataset
from example_datasets.librispeech import convert_librispeech_dataset
from example_datasets.math500 import convert_math500_dataset
from example_datasets.mmbench import convert_mmbench_dataset
from example_datasets.mmlu import convert_mmlu_dataset
from example_datasets.mmlu_pro import convert_mmlu_pro_dataset
from example_datasets.mmmu import (convert_mmmu_dataset,
                                   convert_mmmu_pro_dataset)
from example_datasets.mmstar import convert_mmstar_dataset
from example_datasets.mtbench import convert_mtbench_dataset
from example_datasets.omnibench import convert_omnibench_dataset
from example_datasets.tts_eval import (convert_minimax_multilingual_dataset,
                                       convert_seed_tts_eval_dataset)

# Default dataset mappings (HuggingFace repo ID or local path)
DEFAULT_DATASETS = {
    "AIME": "Maxwell-Jia/AIME_2024",
    "COCO": "lmms-lab/COCO-Caption2017",
    "GSM8K": "openai/gsm8k",
    "HumanEval": "openai/openai_humaneval",
    "LibriSpeech": "openslr/librispeech_asr",
    "MATH500": "HuggingFaceH4/MATH-500",
    "MMLU": "cais/mmlu",
    "MMLU_Pro": "TIGER-Lab/MMLU-Pro",
    "MMMU": "MMMU/MMMU",
    "MMMU_VLMEvalkit": "MMMU/MMMU",
    "MMMU_Pro": "MMMU/MMMU_Pro",
    "MMBench": "lmms-lab/MMBench_EN",
    "MMStar": "Lin-Chen/MMStar",
    "MTBench": "philschmid/mt-bench",
    "MiniMaxMultilingual": "MiniMaxAI/TTS-Multilingual-Test-Set",
    "OmniBench": "m-a-p/OmniBench",
}

# Datasets that require manual download — no HuggingFace auto-download.
# Values are user-facing error messages with download instructions.
LOCAL_ONLY_DATASETS = {
    "SeedTTSEval":
    ("SeedTTSEval requires --dataset_name_or_dir pointing to a .lst file "
     "(e.g. zh/meta.lst, en/meta.lst). Download from: "
     "https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"),
}

# Default max_generate_length for each dataset
DEFAULT_MAX_GENERATE_LENGTHS = {
    "AIME": 512,
    "COCO": 64,
    "GSM8K": 512,
    "HumanEval": 512,
    "LibriSpeech": 256,
    "MATH500": 512,
    "MMLU": 1,
    "MMLU_Pro": 1,
    "MMMU": 20,
    "MMMU_VLMEvalkit": 8192,
    "MMMU_Pro": 1,
    "MMBench": 20,
    "MMStar": 512,
    "MTBench": 512,
    "SeedTTSEval": 2048,
    "MiniMaxMultilingual": 2048,
    "OmniBench": 128,
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

    # Local-only datasets have no HF default — give a helpful error
    if dataset_type in LOCAL_ONLY_DATASETS:
        raise ValueError(LOCAL_ONLY_DATASETS[dataset_type])

    # Get default dataset name
    default_name = DEFAULT_DATASETS.get(dataset_type)
    if not default_name:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Fall back to default HuggingFace dataset name
    print(f"Using HuggingFace dataset: {default_name}")
    return default_name


def get_default_sampling_params(dataset_type: str) -> Tuple[float, float, int]:
    """Return (temperature, top_p, top_k) defaults for a given dataset."""
    if dataset_type in {"OmniBench", "SeedTTSEval", "MiniMaxMultilingual"}:
        return 0.0, 1.0, 1
    return 1.0, 1.0, 50


def build_dataset_config(args, dataset_type: str) -> DatasetConfig:
    """Build DatasetConfig with dataset-specific defaults and CLI overrides."""
    default_temperature, default_top_p, default_top_k = get_default_sampling_params(
        dataset_type)
    max_generate_length = (args.max_generate_length
                           or DEFAULT_MAX_GENERATE_LENGTHS[dataset_type])
    apply_chat_template = None
    if dataset_type not in {"SeedTTSEval", "MiniMaxMultilingual"}:
        apply_chat_template = not args.disable_chat_template
    return DatasetConfig(
        batch_size=args.batch_size,
        temperature=(default_temperature
                     if args.temperature is None else args.temperature),
        top_p=(default_top_p if args.top_p is None else args.top_p),
        top_k=(default_top_k if args.top_k is None else args.top_k),
        max_generate_length=max_generate_length,
        max_samples=args.max_samples,
        apply_chat_template=apply_chat_template,
    )


def main():
    """Main function to parse arguments and call appropriate conversion function."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for TensorRT Edge-LLM inference", )

    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        choices=[
                            "AIME", "COCO", "GSM8K", "HumanEval",
                            "LibriSpeech", "MATH500", "MMBench", "MMLU",
                            "MMLU_Pro", "MMMU", "MMMU_VLMEvalkit", "MMMU_Pro",
                            "MMStar", "MTBench", "SeedTTSEval",
                            "MiniMaxMultilingual", "OmniBench"
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
                        default=None,
                        required=False,
                        help="Sampling temperature (0.0-2.0). Uses the "
                        "dataset-specific default if not provided.")

    parser.add_argument("--top_p",
                        type=float,
                        default=None,
                        required=False,
                        help="Nucleus sampling parameter (0.0-1.0). Uses the "
                        "dataset-specific default if not provided.")

    parser.add_argument("--top_k",
                        type=int,
                        default=None,
                        required=False,
                        help="Top-k sampling parameter (>0). Uses the "
                        "dataset-specific default if not provided.")

    parser.add_argument(
        "--max_generate_length",
        type=int,
        required=False,
        help=
        "Maximum number of tokens to generate (uses dataset default if not provided)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        required=False,
        help="Language for audio benchmarks. SeedTTSEval accepts "
        "'zh'/'en' or 'chinese'/'english'; MiniMaxMultilingual also accepts "
        "short codes like 'ja' and dataset names like 'japanese'.")

    parser.add_argument("--max_samples",
                        type=int,
                        default=None,
                        required=False,
                        help="Limit number of samples for quick testing")

    parser.add_argument(
        "--disable_chat_template",
        action="store_true",
        required=False,
        help=("Disable chat template formatting for Q/A datasets. This is "
              "useful for granularity analysis against the default "
              "chat-template mode."))

    args = parser.parse_args()

    try:
        print(f"Converting {args.dataset} dataset...")

        # Determine dataset path (local cache or HuggingFace name)
        dataset_path = get_dataset_path(
            dataset_type=args.dataset,
            dataset_name_or_dir=args.dataset_name_or_dir)

        print(f"Dataset source: {dataset_path}")
        print(f"Output directory: {args.output_dir}")
        config = build_dataset_config(args, args.dataset)

        if args.dataset == "AIME":
            convert_aime_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir)

        elif args.dataset == "GSM8K":
            convert_gsm8k_dataset(config=config,
                                  dataset_name_or_dir=dataset_path,
                                  output_dir=args.output_dir)

        elif args.dataset == "MMLU":
            convert_mmlu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir,
                                 num_shot=args.num_shot)

        elif args.dataset == "MMLU_Pro":
            convert_mmlu_pro_dataset(config=config,
                                     dataset_name_or_dir=dataset_path,
                                     output_dir=args.output_dir,
                                     num_shot=args.num_shot)

        elif args.dataset == "MMMU":
            convert_mmmu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir)

        elif args.dataset == "MMMU_VLMEvalkit":
            convert_mmmu_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir,
                                 vlmevalkit=True)

        elif args.dataset == "MMMU_Pro":
            print(f"Using subset: {args.subset}")
            convert_mmmu_pro_dataset(config=config,
                                     dataset_name_or_dir=dataset_path,
                                     output_dir=args.output_dir,
                                     subset=args.subset)

        elif args.dataset == "HumanEval":
            convert_humaneval_dataset(config=config,
                                      dataset_name_or_dir=dataset_path,
                                      output_dir=args.output_dir)

        elif args.dataset == "MATH500":
            convert_math500_dataset(config=config,
                                    dataset_name_or_dir=dataset_path,
                                    output_dir=args.output_dir)

        elif args.dataset == "MMBench":
            convert_mmbench_dataset(config=config,
                                    dataset_name_or_dir=dataset_path,
                                    output_dir=args.output_dir)

        elif args.dataset == "MMStar":
            convert_mmstar_dataset(config=config,
                                   dataset_name_or_dir=dataset_path,
                                   output_dir=args.output_dir)

        elif args.dataset == "MTBench":
            convert_mtbench_dataset(config=config,
                                    dataset_name_or_dir=dataset_path,
                                    output_dir=args.output_dir)

        elif args.dataset == "SeedTTSEval":
            if not args.dataset_name_or_dir:
                raise ValueError(
                    "SeedTTSEval requires --dataset_name_or_dir pointing to "
                    "a .lst file (e.g. zh/meta.lst, en/meta.lst)")
            # SeedTTSEval does not extend EdgeLLMDataset, so it never reads
            # config.max_samples; forward --max_samples explicitly.
            convert_seed_tts_eval_dataset(
                config=config,
                dataset_name_or_dir=dataset_path,
                output_dir=args.output_dir,
                language=args.language,
                max_samples=args.max_samples,
            )

        elif args.dataset == "MiniMaxMultilingual":
            # Auto-download from HuggingFace when no local path is given.
            if not args.dataset_name_or_dir:
                from huggingface_hub import snapshot_download
                dataset_path = snapshot_download(
                    DEFAULT_DATASETS["MiniMaxMultilingual"],
                    repo_type="dataset")
                print(f"Auto-downloaded MiniMaxMultilingual to: "
                      f"{dataset_path}")
            # MiniMaxMultilingual also does not extend EdgeLLMDataset; forward
            # --max_samples explicitly.
            convert_minimax_multilingual_dataset(
                config=config,
                dataset_name_or_dir=dataset_path,
                output_dir=args.output_dir,
                language=args.language,
                max_samples=args.max_samples)

        elif args.dataset == "OmniBench":
            convert_omnibench_dataset(config=config,
                                      dataset_name_or_dir=dataset_path,
                                      output_dir=args.output_dir)

        elif args.dataset == "COCO":
            convert_coco_dataset(config=config,
                                 dataset_name_or_dir=dataset_path,
                                 output_dir=args.output_dir)

        elif args.dataset == "LibriSpeech":
            convert_librispeech_dataset(config=config,
                                        dataset_name_or_dir=dataset_path,
                                        output_dir=args.output_dir)
    except Exception as e:
        print(f"Error converting dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
