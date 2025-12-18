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
This script provides a command-line interface for reducing vocabulary size
based on token frequency analysis in a calibration dataset.

Supports two algorithms:
1. Frequency-based approach (default) - analyzes token frequency in input articles
2. Input-aware approach - Count the frequency of tokens in both input articles and output summaries and apply input-aware filtering

Both algorithms use CNN/DailyMail dataset.

Usage:
    # Reduce vocabulary to 16k tokens with frequency approach
    python reduce_vocab.py --model_dir /path/to/model --output_dir /path/to/output --reduced_vocab_size 16384
    
    # Use input-aware algorithm for summarization
    python reduce_vocab.py --model_dir /path/to/model --output_dir /path/to/output --reduced_vocab_size 8192 --method input_aware
"""

import argparse
import json
import os
import sys
import traceback

from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from tensorrt_edgellm.vocab_reduction.vocab_reduction import reduce_vocab_size, get_vocab_size  # isort: skip


def main() -> None:
    """
    Parse command line arguments and reduce vocabulary.
    
    This function sets up argument parsing for the vocabulary reduction script,
    loads the model tokenizer and config, processes the dataset, and saves the
    vocabulary mapping and vocabulary information.
    """
    parser = argparse.ArgumentParser(
        description="Reduce vocabulary size based on token frequency analysis")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the input model directory containing tokenizer and config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the vocabulary mapping and vocabulary info")
    parser.add_argument(
        "--reduced_vocab_size",
        type=int,
        required=True,
        help=
        "Target reduced vocabulary size (must be less than original vocab size)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["input_aware", "frequency"],
        default="input_aware",
        help="Vocabulary reduction method: 'input_aware' ( algorithm) or "
        "'frequency' (input frequency-based). Both use CNN/DailyMail dataset.")
    parser.add_argument(
        "--max_samples",
        type=int,
        required=False,
        default=50000,
        help="Maximum number of samples to use from dataset (default: 50000)")
    parser.add_argument(
        "--d2t_path",
        type=str,
        required=False,
        default=None,
        help="Path to EAGLE d2t tensor file (safetensors format). "
        "If provided, all tokens referenced in d2t mapping will be included in reduced vocabulary."
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Loading tokenizer and config from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)

        vocab_size = get_vocab_size(config)
        print(f"Original vocabulary size: {vocab_size}")
        print(f"Target reduced vocabulary size: {args.reduced_vocab_size}")
        print(f"Method: {args.method}")

        # Load CNN/DailyMail dataset
        print(f"Loading example dataset: cnn_dailymail")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        print(f"Using {len(dataset)} samples for vocabulary analysis")

        # Load d2t tensor if provided
        d2t_tensor = None
        if args.d2t_path:
            print(f"\nLoading d2t tensor from {args.d2t_path}...")
            d2t_data = load_file(args.d2t_path)
            assert "d2t" in d2t_data, "d2t tensor not found in d2t.safetensors"
            d2t_tensor = d2t_data["d2t"]
            print(f"Loaded d2t tensor with shape {d2t_tensor.shape}")

        # Reduce vocabulary using selected method
        print(f"\n{'=' * 70}")
        print(f"Reducing vocabulary with '{args.method}' method...")
        print(f"{'=' * 70}\n")

        vocab_map = reduce_vocab_size(
            tokenizer=tokenizer,
            config=config,
            dataset=dataset,
            reduced_vocab_size=args.reduced_vocab_size,
            d2t_tensor=d2t_tensor,
            method=args.method)

        # Get actual reduced vocabulary size from vocab_map
        actual_reduced_vocab_size = len(vocab_map)

        # Save vocabulary map as safetensors
        vocab_map_path = os.path.join(args.output_dir, "vocab_map.safetensors")
        print(f"Saving vocabulary map to {vocab_map_path}...")
        save_file({"vocab_map": vocab_map}, str(vocab_map_path))

        # Save vocabulary info as JSON
        vocab_info = {
            "vocab_size": vocab_size,
            "reduced_vocab_size": actual_reduced_vocab_size,
            "method": args.method,
            "dataset": "cnn_dailymail",
            "max_samples": min(args.max_samples, len(dataset)),
        }
        if args.d2t_path:
            vocab_info["d2t_tensor_size"] = len(d2t_tensor)

        vocab_info_path = os.path.join(args.output_dir, "reduced_vocab.json")
        print(f"Saving vocabulary info to {vocab_info_path}...")
        with open(vocab_info_path, "w") as f:
            json.dump(vocab_info, f, indent=2)

        print("Vocabulary reduction completed successfully!")
        print(f"Output files saved to: {args.output_dir}")
        print(
            f"  - vocab_map.safetensors: Vocabulary mapping tensor [{actual_reduced_vocab_size}]"
        )
        print(f"  - reduced_vocab.json: Vocabulary size information")
        print(f"  - Method used: {args.method}")
        print(f"  - Dataset: cnn_dailymail")

    except Exception as e:
        print(f"Error during vocabulary reduction: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
