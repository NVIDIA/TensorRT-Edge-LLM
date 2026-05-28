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
Command-line script for exporting audio models to ONNX format using TensorRT Edge-LLM.

This script provides a command-line interface for exporting audio components of
multimodal models (Qwen3-Omni, Qwen3-ASR) to ONNX format with optional quantization
support.

Usage:
    # Export without quantization
    python export_audio.py --model_dir /path/to/model --output_dir /path/to/output

    # Export audio encoder with FP8 quantization
    python export_audio.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8

    # Export with specific calibration dataset
    python export_audio.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8 --dataset_dir openslr/librispeech_asr
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.audio_export import audio_export


def main() -> None:
    """
    Main function that parses command line arguments and exports the audio model.
    
    This function sets up argument parsing for the audio export script and calls
    the audio_export function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Export audio model to ONNX format using TensorRT Edge-LLM"
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
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=
        "Device to load the model on (default: cuda, options: cpu, cuda, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        choices=["fp8"],
        default=None,
        help="Quantization method for audio encoder (fp8 for FP8 quantization). "
        "Only applies to audio_encoder; code2wav is always exported in FP16.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="openslr/librispeech_asr",
        help=
        "Dataset directory to use for audio encoder quantization calibration (default: openslr/librispeech_asr)"
    )
    parser.add_argument(
        "--export_models",
        type=str,
        required=False,
        default=None,
        help=
        "Comma-separated list of models to export for Qwen3-Omni (e.g., 'audio_encoder', 'code2wav', or both. Default is to export both models)"
    )

    args = parser.parse_args()

    try:
        audio_export(model_dir=args.model_dir,
                     output_dir=args.output_dir,
                     dtype=args.dtype,
                     device=args.device,
                     quantization=args.quantization,
                     export_models=args.export_models,
                     dataset_dir=args.dataset_dir)
    except Exception as e:
        print(f"Error during audio export: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
        print("Audio model export completed successfully!")
    except Exception as e:
        print(f"Error during audio model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
