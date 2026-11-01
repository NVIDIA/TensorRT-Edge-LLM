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
"""CLI for merging a PEFT LoRA adapter into a HuggingFace checkpoint."""

import argparse
import logging
import sys

from tensorrt_edgellm.scripts import _dependency_errors

logger = logging.getLogger(__name__)


@_dependency_errors.optional_dependencies("tools")
def main(argv=None, *, prog="tensorrt-edgellm-merge-lora") -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(
        prog=prog, description="Merge LoRA weights into a base HF model")
    parser.add_argument("--model-dir",
                        "--model_dir",
                        required=True,
                        help="Base model checkpoint directory")
    parser.add_argument("--lora-dir",
                        "--lora_dir",
                        required=True,
                        help="PEFT LoRA adapter directory")
    parser.add_argument("--output-dir",
                        "--output_dir",
                        required=True,
                        help="Directory for the merged checkpoint")
    parser.add_argument("--device",
                        default="cuda",
                        help="Device used while merging (default: cuda)")
    parser.add_argument("--torch-dtype",
                        "--torch_dtype",
                        dest="torch_dtype",
                        default="float16",
                        help="Model dtype used while merging")
    args = parser.parse_args(argv)

    from tensorrt_edgellm.lora.lora import merge_lora_and_save

    try:
        merge_lora_and_save(model_dir=args.model_dir,
                            lora_dir=args.lora_dir,
                            output_dir=args.output_dir,
                            device=args.device,
                            torch_dtype=args.torch_dtype)
        logger.info("LoRA merge completed successfully")
    except ModuleNotFoundError:
        raise
    except Exception:
        logger.exception("Error during LoRA merge")
        sys.exit(1)


if __name__ == "__main__":
    main()
