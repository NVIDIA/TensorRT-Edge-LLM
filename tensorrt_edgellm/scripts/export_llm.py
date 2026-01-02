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
This script provides a command-line interface for exporting language models to ONNX format
with support for standard models and EAGLE base models.

Usage:
    # Standard model export
    python export_llm.py --model_dir /path/to/model --output_dir /path/to/output
    
    # EAGLE base model export
    python export_llm.py --model_dir /path/to/base_model --output_dir /path/to/output --is_eagle_base
    
    # Export with reduced vocabulary
    python export_llm.py --model_dir /path/to/model --output_dir /path/to/output --reduced_vocab_dir /path/to/reduced_vocab
    
    # Export with a provided chat template (validates and copies the template instead of inferring from model)
    python export_llm.py --model_dir /path/to/model --output_dir /path/to/output --chat_template /path/to/chat_template.json
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.llm_export import export_llm_model


def main() -> None:
    """
    Main function that parses command line arguments and exports the LLM model.
    
    This function sets up argument parsing for the LLM export script and calls
    the export_llm_model function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Export standard/Eagle3 base LLM model to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the input model directory (base model for EAGLE)")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the exported ONNX model")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=
        "Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument("--is_eagle_base",
                        required=False,
                        action='store_true',
                        help="Whether to export the base model")
    parser.add_argument(
        "--reduced_vocab_dir",
        type=str,
        required=False,
        default=None,
        help=
        "Path to directory containing vocab_map.safetensors for vocabulary reduction (optional)"
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        required=False,
        default=None,
        dest="chat_template_path",
        help=
        "Path to chat template JSON file. When provided, validates and uses this template instead of inferring from the model (optional)"
    )

    args = parser.parse_args()

    try:
        # Export model(s)
        export_llm_model(model_dir=args.model_dir,
                         output_dir=args.output_dir,
                         device=args.device,
                         is_eagle_base=args.is_eagle_base,
                         reduced_vocab_dir=args.reduced_vocab_dir,
                         chat_template_path=args.chat_template_path)

        print("LLM model export completed successfully!")

    except Exception as e:
        print(f"Error during LLM model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
