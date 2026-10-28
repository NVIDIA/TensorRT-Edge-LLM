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
"""Common command entry point with imports deferred until dispatch."""

import argparse
import importlib
import sys

from . import cli_version

COMMANDS = {
    "chat": ("experimental.frontend.cli", "Chat with a local checkpoint"),
    "serve": ("experimental.server.cli", "Launch an OpenAI-compatible server"),
    "build":
    ("experimental.builder.cli", "Build engines directly from a checkpoint"),
    "export":
    ("tensorrt_edgellm.scripts.export", "Export a checkpoint to ONNX"),
    "quantize": ("tensorrt_edgellm.scripts.quantize",
                 "Quantize a checkpoint or draft model"),
    "insert-lora":
    ("tensorrt_edgellm.scripts.insert_lora", "Insert LoRA patterns into ONNX"),
    "process-lora": ("tensorrt_edgellm.scripts.process_lora_weights",
                     "Process LoRA adapter weights"),
    "merge-lora": ("tensorrt_edgellm.scripts.merge_lora",
                   "Merge a LoRA adapter into a checkpoint"),
    "reduce-vocab": ("tensorrt_edgellm.scripts.reduce_vocab",
                     "Generate reduced-vocabulary artifacts"),
}


def main(argv=None):
    """Dispatch a command and preserve its arguments and exit status."""
    parser = argparse.ArgumentParser(
        prog="trtedge",
        description="NVIDIA TensorRT Edge-LLM command line tools.",
        epilog="Use trtedge COMMAND --help for command options.\n\n" +
        "\n".join(f"  {name:<14} {description}"
                  for name, (_, description) in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action=cli_version.VersionAction,
        nargs=0,
        help="Show the package version and checkout revision, when available")
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    parser.add_argument("arguments",
                        nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    module_name, _ = COMMANDS[args.command]
    module = importlib.import_module(module_name)
    return module.main(args.arguments, prog=f"{parser.prog} {args.command}")


if __name__ == "__main__":
    sys.exit(main())
