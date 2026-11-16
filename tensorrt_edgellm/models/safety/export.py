# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Safety export CLI and model selection."""

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv=None, *, prog="tensorrt-edgellm-export") -> None:
    """Parse Safety options and dispatch to the selected model exporter."""
    parser = argparse.ArgumentParser(
        prog=prog,
        allow_abbrev=False,
        description="Export for the Safety runtime.")
    parser.add_argument(
        "model", help="Local checkpoint directory or Hugging Face model ID.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--safety",
                        action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dtype",
                        choices=("float16", "fp16"),
                        default="float16",
                        help="Visual weight dtype.")
    parser.add_argument("--skip-visual",
                        action="store_true",
                        help="Skip visual encoder export.")
    parser.add_argument(
        "--num-decoder-layer",
        type=int,
        help="Export only the first N decoder layers for debugging.")
    parser.add_argument(
        "--reduced-vocab-dir",
        default="",
        help="Directory containing the reduced vocabulary mapping.")
    args = parser.parse_args(argv)
    if args.num_decoder_layer is not None and args.num_decoder_layer < 1:
        parser.error("--num-decoder-layer must be >= 1")
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        import huggingface_hub
        model_dir = Path(huggingface_hub.snapshot_download(args.model))
    with (model_dir / "config.json").open() as stream:
        config = json.load(stream)
    model_type = config.get("model_type")
    if model_type in ("internvl_chat", "internvl"):
        from .internvl import export as model_export
    else:
        parser.error(f"--safety does not support model type {model_type!r}")

    os.umask(0o022)
    output_dir = Path(args.output_dir)
    model_export.export_model(model_dir,
                              output_dir,
                              config,
                              skip_visual=args.skip_visual,
                              num_decoder_layers=args.num_decoder_layer,
                              reduced_vocab_dir=args.reduced_vocab_dir)
    logger.info("Safety export complete: %s", output_dir)
