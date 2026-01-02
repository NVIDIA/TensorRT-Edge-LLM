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
Command-line script for exporting visual models to ONNX format using TensorRT Edge-LLM.

This script provides a command-line interface for exporting visual components of
multimodal models (Qwen2-VL, Qwen2.5-VL, InternVL3) to ONNX format with optional
quantization support.

Usage:
    # Export without quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output
    
    # Export with FP8 quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
    
    # Export with specific device
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --device cuda:1
    
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.visual_export import visual_export


def main() -> None:
    """
    Main function that parses command line arguments and exports the visual model.
    
    This function sets up argument parsing for the visual export script and calls
    the visual_export function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Export visual model to ONNX format using TensorRT Edge-LLM"
    )
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to the input model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the exported ONNX model")
    parser.add_argument("--dtype",
                        type=str,
                        required=False,
                        choices=["fp16"],
                        default="fp16",
                        help="Data type for export (only fp16 supported)")
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        choices=["fp8"],
        default=None,
        help="Quantization method to use (fp8 for FP8 quantization)")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="lmms-lab/MMMU",
        help=
        "Dataset directory to use for quantization (default: lmms-lab/MMMU)")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=
        "Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)"
    )

    args = parser.parse_args()

    try:
        visual_export(model_dir=args.model_dir,
                      output_dir=args.output_dir,
                      dtype=args.dtype,
                      quantization=args.quantization,
                      dataset_dir=args.dataset_dir,
                      device=args.device)
        print("Visual model export completed successfully!")
    except Exception as e:
        print(f"Error during visual model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
