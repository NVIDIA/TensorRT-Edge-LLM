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
Command-line script for merging LoRA weights into a base HF model.

Usage:
    python merge_lora.py --model_dir /path/to/base_model --lora_dir /path/to/lora --output_dir /path/to/output
"""

import argparse
import os
import shutil

from peft import PeftModel

from tensorrt_edgellm.llm_models.model_utils import load_hf_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Merge LoRA weights into a base HF model and save the merged model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Base model directory",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="LoRA checkpoint directory (e.g. vision-lora)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the merged model",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=
        "Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)"
    )
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # 1. load base model
    model, tokenizer, processor = load_hf_model(args.model_dir, "fp16",
                                                args.device)

    # 2. attach LoRA and merge into base model
    lora_model = PeftModel.from_pretrained(model, args.lora_dir)
    print("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()

    # 3. save merged model + tokenizer + processor
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if processor is not None:
        if model.config.model_type == "phi4mm":
            # remove audio_tokenizer for Phi-4-multimodal to avoid error when saving processor
            processor.audio_tokenizer = None
        processor.save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
