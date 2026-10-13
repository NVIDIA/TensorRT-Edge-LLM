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
"""CLI entry point for generating reduced-vocabulary artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from safetensors.torch import load_file

from tensorrt_edgellm._safetensors_io import save_file
from tensorrt_edgellm.vocab_reduction.constants import (VOCAB_INFO_NAME,
                                                        VOCAB_MAP_NAME)
from tensorrt_edgellm.vocab_reduction.selection import (get_vocab_size,
                                                        reduce_vocab_size)


def main() -> None:
    """Generate ``vocab_map.safetensors`` from calibration data."""
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Reduce vocabulary size from calibration data")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing tokenizer and config")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory to save reduced-vocab artifacts")
    parser.add_argument(
        "--reduced_vocab_size",
        type=int,
        required=True,
        help="Target reduced vocabulary size, less than the original vocab")
    parser.add_argument("--method",
                        type=str,
                        choices=["input_aware", "frequency"],
                        default="input_aware",
                        help="Vocabulary reduction method")
    parser.add_argument("--max_samples",
                        type=int,
                        default=50000,
                        help="Maximum CNN/DailyMail samples to use")
    parser.add_argument(
        "--d2t_path",
        type=str,
        default=None,
        help="Optional EAGLE d2t.safetensors path. Referenced base tokens "
        "are always included.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Loading tokenizer and config from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)

        vocab_size = get_vocab_size(config)
        print(f"Original vocabulary size: {vocab_size}")
        print(f"Target reduced vocabulary size: {args.reduced_vocab_size}")
        print(f"Method: {args.method}")

        print("Loading example dataset: cnn_dailymail")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples for vocabulary analysis")

        d2t_tensor = None
        if args.d2t_path:
            print(f"\nLoading d2t tensor from {args.d2t_path}...")
            d2t_data = load_file(args.d2t_path)
            if "d2t" not in d2t_data:
                raise KeyError("d2t tensor not found in d2t.safetensors")
            d2t_tensor = d2t_data["d2t"]
            print(f"Loaded d2t tensor with shape {d2t_tensor.shape}")

        print(f"\n{'=' * 70}")
        print(f"Reducing vocabulary with {args.method!r} method...")
        print(f"{'=' * 70}\n")

        # The runtime unions the config.json and generation_config.json EOS
        # sets; every stop token must survive vocabulary reduction so guided
        # decoding can stop on it.
        generation_eos_ids = set()
        generation_config_path = os.path.join(args.model_dir,
                                              "generation_config.json")
        if os.path.isfile(generation_config_path):
            with open(generation_config_path) as generation_file:
                eos_value = json.load(generation_file).get("eos_token_id")
            if isinstance(eos_value, int):
                generation_eos_ids = {eos_value}
            elif isinstance(eos_value, list):
                generation_eos_ids = {int(item) for item in eos_value}

        vocab_map = reduce_vocab_size(
            tokenizer=tokenizer,
            config=config,
            dataset=dataset,
            reduced_vocab_size=args.reduced_vocab_size,
            d2t_tensor=d2t_tensor,
            method=args.method,
            extra_required_tokens=generation_eos_ids)

        vocab_map_path = os.path.join(args.output_dir, VOCAB_MAP_NAME)
        print(f"Saving vocabulary map to {vocab_map_path}...")
        save_file({"vocab_map": vocab_map}, str(vocab_map_path))

        vocab_info = {
            "vocab_size": vocab_size,
            "reduced_vocab_size": int(vocab_map.numel()),
            "method": args.method,
            "dataset": "cnn_dailymail",
            "max_samples": min(args.max_samples, len(dataset)),
        }
        if args.d2t_path:
            vocab_info["d2t_tensor_size"] = len(d2t_tensor)

        vocab_info_path = os.path.join(args.output_dir, VOCAB_INFO_NAME)
        print(f"Saving vocabulary info to {vocab_info_path}...")
        with open(vocab_info_path, "w") as f:
            json.dump(vocab_info, f, indent=2)

        print("Vocabulary reduction completed successfully!")
        print(f"Output files saved to: {args.output_dir}")
        print(f"  - {VOCAB_MAP_NAME}: Vocabulary mapping tensor "
              f"[{int(vocab_map.numel())}]")
        print(f"  - {VOCAB_INFO_NAME}: Vocabulary size information")

    except Exception as exc:
        print(f"Error during vocabulary reduction: {exc}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
