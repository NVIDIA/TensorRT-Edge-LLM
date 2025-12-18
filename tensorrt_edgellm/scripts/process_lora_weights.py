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
Command-line script for processing LoRA weights.

This script provides a command-line interface for processing LoRA weights according
to specified requirements including tensor name processing, shape corrections, and
format conversions.

Usage:
    python process_lora_weights.py --input_dir /path/to/adapter --output_dir /path/to/output
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.lora import process_lora_weights_and_save


def main() -> None:
    """
    Main function that parses command line arguments and processes LoRA weights.
    
    This function sets up argument parsing for the LoRA weight processing script
    and calls the process_lora_weights_and_save function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Process LoRA weights according to specifications")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Directory containing input adapter files")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory where processed files will be saved")

    args = parser.parse_args()

    try:
        # Process LoRA weights
        process_lora_weights_and_save(input_dir=args.input_dir,
                                      output_dir=args.output_dir)

        print("LoRA weight processing completed successfully!")

    except Exception as e:
        print(f"Error during LoRA weight processing: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
