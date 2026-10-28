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
Command-line script for inserting LoRA patterns into ONNX models.

This script provides a command-line interface for inserting LoRA patterns into
ONNX models in-place. A lora_model.onnx will be created in the same directory
as the original model.onnx. The LoRA model will share the same data as the
original model.

Usage:
    tensorrt-edgellm-insert-lora --onnx-dir /path/to/onnx_model
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.scripts import _dependency_errors


@_dependency_errors.optional_dependencies("tools")
def main(argv=None, *, prog="tensorrt-edgellm-insert-lora") -> None:
    """
    Main function that parses command line arguments and inserts LoRA patterns.

    This function sets up argument parsing for the LoRA insertion script and calls
    the insert_lora_and_save function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        prog=prog, description="Insert LoRA patterns into ONNX models")
    parser.add_argument(
        "--onnx-dir",
        "--onnx_dir",
        type=str,
        required=True,
        help="Directory containing the ONNX model (model.onnx and config.json)"
    )

    args = parser.parse_args(argv)

    from tensorrt_edgellm.lora.lora import insert_lora_and_save

    try:
        # Insert LoRA patterns
        insert_lora_and_save(onnx_dir=args.onnx_dir)

        print("LoRA insertion completed successfully!")

    except ModuleNotFoundError:
        raise
    except Exception as e:
        print(f"Error during LoRA insertion: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
