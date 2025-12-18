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
This script provides a command-line interface for exporting EAGLE3 draft models to ONNX format
with support for EAGLE draft models.

Usage:
    # EAGLE draft model export
    python export_draft.py --draft_model_dir /path/to/draft_model --base_model_dir /path/to/base_model --output_dir /path/to/output (--use_prompt_tuning)
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.llm_export import export_draft_model


def main() -> None:
    """
    Main function that parses command line arguments and exports the EAGLE3 draft model.
    
    This function sets up argument parsing for the EAGLE3 draft model export script and calls
    the export_draft_model function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description=
        "Export EAGLE3 Draft model to ONNX format using TensorRT Edge-LLM")
    parser.add_argument("--draft_model_dir",
                        type=str,
                        required=True,
                        help="Path to the draft model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to save the exported ONNX model")
    parser.add_argument("--use_prompt_tuning",
                        action="store_true",
                        help="Whether to use prompt tuning")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=False,
        help=
        "Path to the base model directory. Used to copy weights from if the draft weights are incomplete."
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

    try:
        # Export model(s)
        export_draft_model(draft_model_dir=args.draft_model_dir,
                           output_dir=args.output_dir,
                           use_prompt_tuning=args.use_prompt_tuning,
                           base_model_dir=args.base_model_dir,
                           device=args.device)

        print("EAGLE3 Draft model export completed successfully!")

    except Exception as e:
        print(f"Error during EAGLE3 draft model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
