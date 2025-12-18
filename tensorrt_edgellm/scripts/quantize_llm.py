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
This script provides a command-line interface for quantizing HuggingFace models
using various quantization schemes supported by NVIDIA ModelOpt.

Usage:
    # Quantize with FP8 quantization
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
    
    # Quantize without quantization (default)
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output
    
    # Quantize with different quantization for LM head
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8 --lm_head_quantization fp8
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.quantization.llm_quantization import \
    quantize_and_save_llm


def main() -> None:
    """
    Main function that parses command line arguments and quantizes the model.
    
    This function sets up argument parsing for the quantization script and calls
    the quantize_and_save_llm function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Quantize a model using NVIDIA ModelOpt")
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to the input model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the quantized model")
    parser.add_argument("--quantization",
                        type=str,
                        required=False,
                        choices=["fp8", "int4_awq", "nvfp4", "int8_sq"],
                        default=None,
                        help="Quantization method to use")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["fp16"],
                        required=False,
                        default="fp16",
                        help="Model data type for loading")
    parser.add_argument("--dataset_dir",
                        type=str,
                        required=False,
                        default="cnn_dailymail",
                        help="Dataset name or path for calibration data")
    parser.add_argument(
        "--lm_head_quantization",
        type=str,
        required=False,
        choices=["fp8", "nvfp4"],
        default=None,
        help=
        "Quantization method for language model head (only fp8 and nvfp4 are currently supported)"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to use for model loading and quantization")

    args = parser.parse_args()

    try:
        quantize_and_save_llm(model_dir=args.model_dir,
                              output_dir=args.output_dir,
                              quantization=args.quantization,
                              dtype=args.dtype,
                              dataset_dir=args.dataset_dir,
                              lm_head_quantization=args.lm_head_quantization,
                              device=args.device)
        print("Model quantization completed successfully!")
    except Exception as e:
        print(f"Error during model quantization: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
