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
"""CLI for standalone quantization.

Usage::

    # LLM quantization
    tensorrt-edgellm-quantize llm \\
        --model-dir /path/to/model \\
        --output-dir /path/to/output \\
        --quantization nvfp4 \\
        --lm-head-quantization nvfp4

    # Speculative draft quantization (Eagle3 or DFlash, auto-detected)
    tensorrt-edgellm-quantize draft \\
        --base-model-dir /path/to/base \\
        --draft-model-dir /path/to/draft \\
        --output-dir /path/to/output \\
        --quantization fp8

    # Qwen3-Omni is auto-detected by the `llm` subcommand and dispatched
    # to the joint Thinker+Talker driver (both MoE and non-MoE variants).
    tensorrt-edgellm-quantize llm \\
        --model-dir Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --output-dir /path/to/output \\
        --quantization nvfp4
"""

import argparse
import json
import os

from tensorrt_edgellm.scripts import _dependency_errors

from ..quantization.datasets import (DEFAULT_AUDIO_DATASET,
                                     DEFAULT_IMAGE_DATASET,
                                     DEFAULT_TEXT_DATASET, available_datasets)


def _add_common_args(parser):
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"],
        help="Backbone quantization method.")
    parser.add_argument("--lm-head-quantization",
                        "--lm_head_quantization",
                        default=None,
                        choices=["fp8", "int4_awq", "nvfp4", "mxfp8"],
                        help="LM-head quantization method.")
    parser.add_argument(
        "--visual-quantization",
        "--visual_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the visual tower (visual / vision_tower / "
              "multi_modal_projector). Only fp8 is exposed today; other "
              "precisions are deferred until per-recipe accuracy is "
              "validated. When unset the visual tower stays at fp16 "),
    )
    parser.add_argument(
        "--visual-mha-quantization",
        "--visual_mha_quantization",
        default=None,
        choices=["fp8"],
        help=("Run the visual attention (Q*K^T and P*V matmuls) in FP8."),
    )
    parser.add_argument(
        "--audio-quantization",
        "--audio_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the audio tower (audio_tower / audio_embed). Only "
              "fp8 is exposed today; lower-bit recipes are deferred until "
              "WER-validated. For Qwen3-ASR this triggers an ASR-shaped "
              "multimodal calibration loop (LibriSpeech audio + transcript "
              "pairs streamed through audio_tower + text decoder). When "
              "unset the audio tower stays at fp16."),
    )
    parser.add_argument(
        "--cp-quantization",
        "--cp_quantization",
        default=None,
        choices=["fp8", "nvfp4"],
        help=("Quantize the Talker CodePredictor "
              "(talker.code_predictor.*) of Qwen3-Omni or Qwen3-TTS, dense "
              "and MoE.  down_proj, the per-codebook lm_heads and "
              "talker_projection are kept unquantized in both modes (see "
              "_cp_entries in quantization_configs.py).  When unset CP stays "
              "at fp16."),
    )
    parser.add_argument("--kv-cache-quantization",
                        "--kv_cache_quantization",
                        default=None,
                        choices=["fp8"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text-dataset",
        "--text_dataset",
        default=DEFAULT_TEXT_DATASET,
        help=("Registered text calibration dataset name (LLM, LM-head, "
              "KV-cache, CodePredictor, MTP, EAGLE3, DFlash). "
              f"Default: {DEFAULT_TEXT_DATASET}. Available: "
              f"{', '.join(available_datasets('text'))}. Unknown names fail "
              "with a pointer to the customization guide."))
    parser.add_argument(
        "--image-dataset",
        "--image_dataset",
        default=DEFAULT_IMAGE_DATASET,
        help=("Registered image calibration dataset name (used with "
              "--visual-quantization). "
              f"Default: {DEFAULT_IMAGE_DATASET}. Available: "
              f"{', '.join(available_datasets('image'))}."))
    parser.add_argument(
        "--audio-dataset",
        "--audio_dataset",
        default=DEFAULT_AUDIO_DATASET,
        help=("Registered audio calibration dataset name (Qwen3-ASR / audio "
              "tower). "
              f"Default: {DEFAULT_AUDIO_DATASET}. Available: "
              f"{', '.join(available_datasets('audio'))}."))
    parser.add_argument("--num-samples",
                        "--num_samples",
                        type=int,
                        default=512)


@_dependency_errors.optional_dependencies("tools")
def main(argv=None, *, prog="tensorrt-edgellm-quantize"):
    parser = argparse.ArgumentParser(
        prog=prog, description="TensorRT Edge-LLM standalone quantization")
    sub = parser.add_subparsers(dest="command", required=True)

    llm_parser = sub.add_parser("llm", help="Quantize an LLM")
    llm_parser.add_argument("--model-dir", "--model_dir", required=True)
    llm_parser.add_argument(
        "--fuse-gdn-qkvzba-scales",
        "--fuse_gdn_qkvzba_scales",
        action="store_true",
        help=("NVFP4 hybrid-GDN models only: also quantize the small GDN "
              "in_proj_b/in_proj_a projections (disabled by stock ModelOpt "
              ">=0.45 configs) and share in_proj_qkv's per-tensor scales so "
              "export fuses qkv/z/b/a into a single NVFP4 GEMM."))
    llm_parser.add_argument(
        "--quantize-gemma4-down-proj",
        "--quantize_gemma4_down_proj",
        action="store_true",
        help=("Gemma4 int4_awq/nvfp4 only: force-quantize down_proj to 4-bit. "
              "Off by default -- down_proj stays fp16 to preserve accuracy."))
    llm_parser.add_argument(
        "--mtp-draft-dir",
        "--mtp_draft_dir",
        default=None,
        help=
        ("Optional separate checkpoint providing the MTP draft weights "
         "(``mtp.*``). Use when the MTP head lives in a different checkpoint "
         "than the calibration base -- e.g. a BF16 base that can run the "
         "forward pass plus a checkpoint that carries the MTP weights. "
         "Defaults to --model-dir (MTP embedded in the base)."))
    _add_common_args(llm_parser)

    draft_parser = sub.add_parser(
        "draft", help="Quantize an Eagle3, DFlash, or JetSpec draft model")
    draft_parser.add_argument("--base-model-dir",
                              "--base_model_dir",
                              required=True)
    draft_parser.add_argument("--draft-model-dir",
                              "--draft_model_dir",
                              required=True)
    _add_common_args(draft_parser)

    args = parser.parse_args(argv)

    if args.command == "llm":
        from ..quantization.quantize import quantize_and_export
        quantize_and_export(
            model_dir=args.model_dir,
            mtp_draft_dir=args.mtp_draft_dir,
            output_dir=args.output_dir,
            quantization=args.quantization,
            lm_head_quantization=args.lm_head_quantization,
            visual_quantization=args.visual_quantization,
            visual_mha_quantization=args.visual_mha_quantization,
            audio_quantization=args.audio_quantization,
            cp_quantization=args.cp_quantization,
            kv_cache_quantization=args.kv_cache_quantization,
            dtype=args.dtype,
            device=args.device,
            text_dataset=args.text_dataset,
            image_dataset=args.image_dataset,
            audio_dataset=args.audio_dataset,
            num_samples=args.num_samples,
            fuse_gdn_qkvzba_scales=args.fuse_gdn_qkvzba_scales,
            quantize_gemma4_down_proj=args.quantize_gemma4_down_proj,
        )
    elif args.command == "draft":
        if _is_dflash_family_draft(args.draft_model_dir):
            _validate_dflash_quant_args(parser, args)
            from ..quantization.models.dflash_draft import \
                quantize_and_export_dflash_draft
            quantize_and_export_dflash_draft(
                base_model_dir=args.base_model_dir,
                draft_model_dir=args.draft_model_dir,
                output_dir=args.output_dir,
                quantization=args.quantization or "nvfp4",
                lm_head_quantization=args.lm_head_quantization,
                kv_cache_quantization=args.kv_cache_quantization,
                dtype=args.dtype,
                device=args.device,
                text_dataset=args.text_dataset,
                num_samples=args.num_samples,
            )
        else:
            from ..quantization.models.eagle3_draft import \
                quantize_and_export_draft
            quantize_and_export_draft(
                base_model_dir=args.base_model_dir,
                draft_model_dir=args.draft_model_dir,
                output_dir=args.output_dir,
                quantization=args.quantization or "fp8",
                lm_head_quantization=args.lm_head_quantization,
                kv_cache_quantization=args.kv_cache_quantization,
                dtype=args.dtype,
                device=args.device,
                text_dataset=args.text_dataset,
                num_samples=args.num_samples,
            )


def _is_dflash_family_draft(draft_model_dir: str) -> bool:
    cfg_path = os.path.join(draft_model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    # DSpark drafts share the DFlash backbone; the Markov/confidence sidecar
    # tensors pass through the same quantization path unquantized.
    # DeepSpec checkpoints (Qwen3DSparkModel, Gemma4DSparkModel) publish
    # block_size/target_layer_ids at the top level without a dflash_config.
    is_dspark = bool(
        cfg.get("dspark_config") or cfg.get("markov_head_type") or any(
            str(arch).endswith("DSparkModel")
            for arch in (cfg.get("architectures") or [])))
    return bool(
        cfg.get("dflash_config") or cfg.get("jetspec_config") or is_dspark)


def _validate_dflash_quant_args(parser, args) -> None:
    if args.kv_cache_quantization is not None:
        parser.error(
            "DFlash/JetSpec draft KV-cache quantization is not validated yet.")


if __name__ == "__main__":
    main()
