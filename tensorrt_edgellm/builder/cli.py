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
"""Build TensorRT engines directly from an Edge-LLM checkpoint."""

import argparse
import logging
import os
from dataclasses import replace
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

from tensorrt_edgellm._native import NativeManifestNotFoundError
from tensorrt_edgellm._native.dependencies import require_tensorrt
from tensorrt_edgellm._native.load import resolve_payload
from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    validate_external_speculative_target
from tensorrt_edgellm.dflash import LILICORR_PROJECTOR_TYPE, DFlashVersion

LOGGER = logging.getLogger("tensorrt_edgellm.builder")


def _value_or_default(value, default):
    return default if value is None else value


def _build_args(args: argparse.Namespace, component: str):
    from .core import contracts
    from .core.builder import BuildArgs

    supports_refit = bool(
        contracts.component_spec(component).external_weight_kinds)

    return BuildArgs(
        model_dir=args.model_dir,
        engine_dir=args.engine_dir,
        component=component,
        spec_role=args.spec_role,
        spec_type=args.spec_type,
        dflash_version=getattr(args, "dflash_version", DFlashVersion.V1),
        max_input_len=args.max_input_len,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        max_batch_size=args.max_batch_size,
        max_lora_rank=args.max_lora_rank,
        num_decoder_layers=args.num_decoder_layer,
        max_spec_verify_tokens=_value_or_default(args.max_spec_verify_tokens,
                                                 60),
        max_spec_draft_tokens=_value_or_default(args.max_spec_draft_tokens,
                                                60),
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_image_tokens_per_image=args.max_image_tokens_per_image,
        min_time_steps=args.min_time_steps,
        max_time_steps=args.max_time_steps,
        min_code_len=args.min_code_len,
        opt_code_len=args.opt_code_len,
        max_code_len=args.max_code_len,
        reuse_llm_kv_cache=args.reuse_llm_kv_cache == "true",
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        reduced_vocab_dir=args.reduced_vocab_dir or "",
        draft_reduced_vocab_dir=args.draft_reduced_vocab_dir or "",
        draft_model_dir=args.draft_model_dir or "",
        target_model_dir=args.target_model_dir or "",
        fp8_embedding=args.fp8_embedding,
        plugin_path=args.plugin_path,
        profiling_detailed=args.profiling_detailed,
        dense_quant=args.dense,
        quantize_activations=args.quantize_activations,
        int4_gemm_plugin_version=args.int4_gemm_plugin_version,
        externalize_weights=tuple(args.externalize_weights or ()),
        pi05_policy_config=args.pi05_policy_config or "",
        pi05_max_prefix_len=args.pi05_max_prefix_len,
        pi05_denoise_steps=args.pi05_denoise_steps,
        trt_native_moe=args.trt_native_moe,
        draft_dense_head=args.draft_dense_head,
        weightless_engine=args.weightless_engine and supports_refit,
        gr00t_embodiment_id=args.gr00t_embodiment_id,
        minimax_h3_num_steps=args.minimax_h3_num_steps,
        minimax_h3_max_video_tokens=args.minimax_h3_max_video_tokens,
    )


def _component_tokens(values: Union[Iterable[str], str]) -> Tuple[str, ...]:
    if isinstance(values, str):
        values = (values, )
    tokens = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(","))
    return tuple(token for token in tokens if token)


def _copy_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _resolve_build_selection(args: argparse.Namespace) -> Tuple[object, Tuple]:
    from .core import contracts
    from .core.config import BundleConfig
    from .models import registry as model_registry

    bundle = BundleConfig.from_pretrained(args.model_dir)
    requested = _component_tokens(args.components)
    components = contracts.resolve_components(bundle.root_model_type,
                                              requested,
                                              available=bundle.components)
    if not components:
        raise ValueError(f"{bundle.root_model_type!r} has no buildable "
                         "components")
    if args.tp_size > 1:
        if components != (contracts.Component.LLM, ):
            raise ValueError(
                "tensor-parallel direct builds currently support only the llm component"
            )
        if args.spec_type != "none":
            raise ValueError(
                "tensor-parallel speculative decoding is not supported; omit --spec-type"
            )
    configuration = model_registry.configuration_module_for(
        bundle.root_model_type)
    validate_build = getattr(configuration, "validate_build", None)
    if validate_build is not None:
        validate_build(args, components)
    return bundle, components


def _apply_dflash_contract(args: argparse.Namespace,
                           contract) -> argparse.Namespace:
    """Set the DFlash version and, for V2, the verify and draft profiles.

    Both default to the checkpoint block size (chain verification). A larger
    base verify profile serves DDTree verification with the same draft.
    """
    from .core import contracts

    if contract.version == DFlashVersion.V2:
        block_size = contract.block_size
        verify_size = _value_or_default(args.max_spec_verify_tokens,
                                        block_size)
        draft_size = _value_or_default(args.max_spec_draft_tokens, block_size)
        if not block_size <= draft_size <= 16:
            raise ValueError(
                "DFlash V2 draft profile size must include checkpoint "
                f"block_size={block_size} and not exceed 16")
        if contract.is_lilicorr and draft_size != block_size:
            raise ValueError(
                "LiLiCorr drafts require --max-spec-draft-tokens equal to "
                f"checkpoint block_size={block_size}: the candidate-lattice "
                "head has a fixed block_size - 1 slot extent")
        max_verify = contracts.MAX_SPEC_VERIFY_TREE_TOKENS
        if not block_size <= verify_size <= max_verify:
            raise ValueError(
                "DFlash V2 verify profile size must include checkpoint "
                f"block_size={block_size} and not exceed {max_verify}")
        args = _copy_args(args,
                          max_spec_verify_tokens=verify_size,
                          max_spec_draft_tokens=draft_size)
    return _copy_args(args, dflash_version=contract.version)


def _speculative_build_plan(args: argparse.Namespace, bundle, components):
    """Expand one model build into its base, draft, and component builds."""
    from .core import contracts
    from .core.config import BundleConfig, draft_dflash_contract

    if contracts.Component.LLM not in components:
        raise ValueError("--spec-type requires the llm component")
    dflash_contract = (draft_dflash_contract(args.draft_model_dir)
                       if args.spec_type == "dflash" and args.draft_model_dir
                       else None)
    if args.spec_type in ("eagle3", "dflash", "jetspec", "domino", "dspark"):
        lilicorr = dflash_contract is not None and dflash_contract.is_lilicorr
        validate_external_speculative_target(
            bundle.component_dict(contracts.Component.LLM),
            mode=LILICORR_PROJECTOR_TYPE if lilicorr else args.spec_type)
    if args.spec_type == "mtp" and args.draft_model_dir:
        raise ValueError(
            "native MTP reads draft layers from --model-dir; do not pass "
            "--draft-model-dir")
    if args.spec_type != "mtp" and not args.draft_model_dir:
        raise ValueError(f"{args.spec_type} requires --draft-model-dir")

    draft_model_dir = args.draft_model_dir or args.model_dir
    draft_bundle = BundleConfig.from_pretrained(draft_model_dir)
    if dflash_contract is not None:
        args = _apply_dflash_contract(args, dflash_contract)
    if args.spec_type == "dspark":
        from .models.dspark.configuration import resolve_build_profile
        draft = draft_bundle.component_dict(contracts.Component.LLM)
        verify_tokens, draft_tokens = resolve_build_profile(
            draft, args.max_spec_draft_tokens, args.max_spec_verify_tokens)
        args = _copy_args(args,
                          max_spec_verify_tokens=verify_tokens,
                          max_spec_draft_tokens=draft_tokens)
    plan = []
    for component in components:
        if component != contracts.Component.LLM:
            component_args = _copy_args(args,
                                        spec_type="none",
                                        draft_model_dir=None,
                                        target_model_dir=None)
            plan.append((component.value, component_args, bundle, component))
            continue

        base_args = _copy_args(
            args,
            spec_role=contracts.SpecRole.BASE.value,
            draft_model_dir=(draft_model_dir if args.spec_type
                             in ("eagle3", "dflash", "jetspec", "domino",
                                 "dspark") else None),
            target_model_dir=None,
        )
        draft_args = _copy_args(
            args,
            model_dir=draft_model_dir,
            spec_role=contracts.SpecRole.DRAFT.value,
            draft_model_dir=None,
            target_model_dir=args.model_dir,
        )
        plan.extend((
            ("spec-base", base_args, bundle, component),
            ("spec-draft", draft_args, draft_bundle, component),
        ))
    return plan


def _role_dflash_contract(args: argparse.Namespace):
    """Resolve the draft's DFlash contract for a per-role build.

    A ``--spec-role`` build skips the paired plan, so read the draft checkpoint
    (its own for a draft engine, ``--draft-model-dir`` for a base) to keep the
    version and V2 tree profile consistent with the paired build.
    """
    from .core import contracts
    from .core.config import draft_dflash_contract

    draft_dir = (args.model_dir if args.spec_role
                 == contracts.SpecRole.DRAFT.value else args.draft_model_dir)
    if not draft_dir:
        return None
    return draft_dflash_contract(draft_dir)


def _build_plan(args: argparse.Namespace, bundle, components):
    from .core import contracts

    if (args.spec_role == contracts.SpecRole.NONE.value
            and args.spec_type != "none"):
        return _speculative_build_plan(args, bundle, components)
    if args.spec_type == "dflash":
        contract = _role_dflash_contract(args)
        if contract is not None:
            args = _apply_dflash_contract(args, contract)
    return [(component.value, args, bundle, component)
            for component in components]


def _preflight_plan(args: argparse.Namespace, plan) -> None:
    """Run model-owned bundle checks before plugin load or engine writes."""
    from .models import registry as model_registry

    checked = set()
    for _, component_args, bundle, component in plan:
        build_args = _build_args(component_args, component.value)
        artifact_writer = model_registry.artifact_writer_for(
            bundle.root_model_type,
            build_args.spec_type,
            build_args.resolved_spec_role,
            build_args.dflash_version,
            eagle4_draft=build_args.eagle4_draft)
        preflight = getattr(artifact_writer, "preflight_bundle", None)
        key = (artifact_writer.__name__, os.path.abspath(bundle.model_dir),
               os.path.abspath(args.engine_dir))
        if preflight is None or key in checked:
            continue
        preflight(bundle, build_args, args.engine_dir)
        checked.add(key)


def _build_one(args: argparse.Namespace, bundle, component,
               plugin_handle) -> str:
    from .core import contracts
    from .core.builder import build_engine, load_device_config
    from .core.bundle import LLM_COMPONENTS
    from .models import registry as model_registry

    build_args = _build_args(args, component.value)
    if args.weightless_engine and not build_args.weightless_engine:
        LOGGER.info(
            "Keeping %s as a weighted engine; the component has no "
            "checkpoint-backed engine weights", component.value)
    cfg = None
    if component in LLM_COMPONENTS:
        cfg = load_device_config(build_args)
    result = build_engine(build_args,
                          cfg,
                          bundle=bundle,
                          plugin_handle=plugin_handle)
    if build_args.tp_size > 1 and build_args.tp_rank != 0:
        return result.engine_path

    artifact_cfg = cfg
    if build_args.tp_size > 1:
        artifact_cfg = load_device_config(
            replace(build_args, tp_size=1, tp_rank=0))
    artifact_writer = model_registry.artifact_writer_for(
        bundle.root_model_type,
        build_args.spec_type,
        build_args.resolved_spec_role,
        build_args.dflash_version,
        eagle4_draft=build_args.eagle4_draft)
    artifact_writer.write_artifacts(bundle, artifact_cfg, build_args,
                                    args.engine_dir)
    if result.compact_execution or result.checkpoint_weight_bindings:
        config_path = contracts.component_spec(component).config_path(
            args.engine_dir, build_args.resolved_spec_role, build_args.tp_size)
    if result.compact_execution:
        from .core.compact_build_report import write_compact_execution
        write_compact_execution(config_path)
    if result.checkpoint_weight_bindings:
        from .core.artifacts import patch_external_weight_config
        patch_external_weight_config(
            config_path,
            result.checkpoint_weight_bindings,
            result.checkpoint_identity,
            checkpoint_dir=(result.checkpoint_dir
                            or os.path.abspath(build_args.model_dir)),
            weightless_engine=result.weightless_engine,
        )
    return result.engine_path


def _write_multistage_config(args: argparse.Namespace, bundle,
                             components) -> Optional[str]:
    """Emit the stage graph, or log why this build has none.

    A graph naming a stage this build lacks loads, then fails at that stage.
    Which component subsets stand alone is the model's call; its topology hook
    returns None for the rest.
    """
    from .core.artifacts.multistage import (drop_stale_config,
                                            write_multistage_config)

    if not components:
        return None
    if getattr(args, "tp_size", 1) > 1:
        LOGGER.info("Skipping multistage config: tensor-parallel engines are "
                    "named per rank, which the graph cannot address")
        drop_stale_config(bundle, components, args)
        return None
    speculative = getattr(args, "spec_type", "none") != "none"
    return write_multistage_config(bundle,
                                   components,
                                   args,
                                   speculative=speculative)


def _write_graph_only(args: argparse.Namespace) -> None:
    from .core.artifacts.multistage import write_graph_for_engines
    from .core.config import BundleConfig

    bundle = BundleConfig.from_pretrained(args.model_dir)
    path = write_graph_for_engines(bundle, args.engine_dir)
    if path is not None:
        print(f"Multistage config: {path}")


def _build(args: argparse.Namespace,
           on_progress: Optional[Callable[[str], None]] = None) -> None:
    require_tensorrt()
    from .core.builder import load_plugin_library

    plugin_path = args.plugin_path
    if plugin_path is None:
        try:
            plugin_path = str(resolve_payload().plugin)
        except NativeManifestNotFoundError:
            plugin_path = "build/lib/libNvInfer_edgellm_plugin.so"
    args = _copy_args(args, plugin_path=plugin_path)
    bundle, components = _resolve_build_selection(args)
    plan = _build_plan(args, bundle, components)
    _preflight_plan(args, plan)
    plugin_handle = load_plugin_library(plugin_path)
    LOGGER.info("Building %s engines for %s: %s", len(plan),
                bundle.root_model_type, ", ".join(label for label, *_ in plan))
    results = []
    for index, (label, build_args, build_bundle,
                component) in enumerate(plan, 1):
        if on_progress is not None:
            name = {
                "llm": "text",
                "spec-base": "base",
                "spec-draft": "draft",
                "visual": "vision"
            }.get(label, label.replace("-", " "))
            on_progress(f"Building {name} engine ({index}/{len(plan)})")
        LOGGER.info("Building component %s", label)
        engine_path = _build_one(build_args, build_bundle, component,
                                 plugin_handle)
        results.append((label, engine_path, os.path.getsize(engine_path)))

    config_path = _write_multistage_config(args, bundle, components)

    print("Engine outputs:")
    for component, engine_path, size in results:
        print(f"  {component}: {engine_path} ({size} bytes)")
    if config_path is not None:
        print(f"Multistage config: {config_path}")


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-dir",
                        required=True,
                        help="Output directory for the compiled engine bundle")
    parser.add_argument(
        "--plugin-path",
        help=("Plugin DSO override. By default, use the payload selected from "
              "the installed TensorRT Edge-LLM wheel."))
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose build logging")


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    from .core import contracts, weight_policy

    parser.add_argument("--model-dir",
                        required=True,
                        help="Local checkpoint directory")
    parser.add_argument(
        "--components",
        default="all",
        help=(
            "Comma-separated component list or 'all'. Default builds every "
            "component available in the checkpoint except the Cosmos3 policy "
            "engines (und-prefill, gen, vae-encoder) of a checkpoint with an "
            "LLM, which must be named. "
            "Build order: " + ", ".join(contracts.component_build_order()) +
            "."))
    parser.add_argument(
        "--spec-type",
        choices=("none", "eagle3", "mtp", "dflash", "jetspec", "domino",
                 "dspark", "gemma4_mtp"),
        default="none",
        help=
        ("Build base and draft engines. EAGLE3, DFlash, JetSpec, Domino, dSpark, "
         "and Gemma4 MTP require --draft-model-dir."))
    parser.add_argument(
        "--draft-dense-head",
        action="store_true",
        help=("EAGLE-4 drafts with a low-rank candidate head: build the "
              "full-vocabulary lm_head instead, as tensorrt-edgellm-export "
              "--draft-dense-head does."))
    parser.add_argument("--dense",
                        choices=("auto", "nvfp4-qdq", "fp16"),
                        default="auto",
                        help="Dense-layer build mode (default: %(default)s)")
    parser.add_argument(
        "--no-quantize-activations",
        dest="quantize_activations",
        action="store_false",
        help=("Keep activations in FP16. For packed NVFP4 checkpoints this "
              "selects the NVFP4-A16 GEMM/MoE path instead of W4A4."))
    parser.add_argument(
        "--int4-gemm-plugin-version",
        type=int,
        choices=(1, 2),
        default=2,
        help="INT4 GEMM implementation: 2 (default) or 1 (legacy)")
    parser.add_argument(
        "--trt-native-moe",
        "--trt_native_moe",
        dest="trt_native_moe",
        action="store_true",
        help=("Build Qwen3-family MoE experts through TensorRT's native MoE "
              "layer instead of the Edge-LLM MoE plugins. Opt-in; the "
              "precision follows the checkpoint: FP16/BF16 experts use the "
              "FP16 layer, NVFP4 experts use W4A8 (FP4 weights, static FP8 "
              "activations). INT4, NVFP4-A16, and Nemotron-H experts are "
              "rejected. Needs TensorRT >= 11.3 and an SM 10.x or 11.x "
              "device."))
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help=(
            "Build nothing: write multistage_config.json for the engines "
            "already under --engine-dir, such as tensorrt-edgellm-export and "
            "the C++ builders produce. Budgets come from each engine's own "
            "config."))
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=32,
        help=("Maximum prompt tokens, including conversation history "
              "(default: %(default)s)"))
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=96,
        help=("Maximum total tokens per sequence, including prompt and "
              "generated tokens (default: %(default)s)"))
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=("Maximum number of sequences in an engine batch (default: "
              "%(default)s)"))
    parser.add_argument(
        "--reuse-llm-kv-cache",
        choices=("true", "false"),
        default="true",
        help=
        "Reuse the LLM KV prefix for Alpamayo action (default: true; use false for LLM TP>1)."
    )
    parser.add_argument(
        "--pi05-policy-config",
        choices=("pi05_libero", "pi05_droid", "pi05_aloha"),
        help=("pi0.5 only: openpi policy contract served by the checkpoint. "
              "Required when the checkpoint feature schema is ambiguous."))
    parser.add_argument(
        "--pi05-max-prefix-len",
        type=int,
        default=0,
        help=("pi0.5 only: maximum compacted image-plus-language prefix. "
              "0 uses three 256-token views plus the 208-token prompt."))
    parser.add_argument(
        "--pi05-denoise-steps",
        type=int,
        default=0,
        help=("pi0.5 only: default flow-matching step count. 0 reads the "
              "checkpoint or uses 10."))
    parser.add_argument(
        "--minimax-h3-num-steps",
        type=int,
        default=50,
        help=("MiniMax-H3 only: denoising steps baked into the GEN engine "
              "(default: %(default)s)"))
    parser.add_argument(
        "--minimax-h3-max-video-tokens",
        type=int,
        default=40960,
        help=("MiniMax-H3 only: maximum video latent rows per request; "
              "the default covers 1344x768 at 124 frames (default: "
              "%(default)s)"))
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=0,
        help=("Maximum supported LoRA rank. 0 disables LoRA (default: "
              "%(default)s)"))
    parser.add_argument(
        "--num-decoder-layer",
        type=int,
        default=0,
        help=("Build only the first N decoder layers (few-layer numeric "
              "validation). 0 keeps the checkpoint's own layer count."))
    parser.add_argument(
        "--max-spec-verify-tokens",
        dest="max_spec_verify_tokens",
        help=("Maximum verification token capacity compiled into the base "
              "engine (default: 60; DFlash V2 and dSpark derive it from the "
              "draft checkpoint)"),
        type=int)
    parser.add_argument(
        "--max-spec-draft-tokens",
        dest="max_spec_draft_tokens",
        help=("Maximum draft token capacity compiled into the draft engine "
              "(default: 60; DFlash V2 and dSpark derive it from the draft "
              "checkpoint)"),
        type=int)
    parser.add_argument(
        "--min-image-tokens",
        type=int,
        default=4,
        help=("Minimum visual tokens in the visual engine profile (default: "
              "%(default)s)"))
    parser.add_argument(
        "--max-image-tokens",
        type=int,
        default=1024,
        help=("Maximum total visual tokens in the visual engine profile "
              "(default: %(default)s)"))
    parser.add_argument(
        "--max-image-tokens-per-image",
        type=int,
        default=512,
        help=("Maximum visual tokens per image in the visual engine profile "
              "(default: %(default)s)"))
    parser.add_argument(
        "--min-time-steps",
        type=int,
        default=100,
        help=("Minimum audio time steps in the engine profile (default: "
              "%(default)s)"))
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=6000,
        help=("Maximum audio time steps in the engine profile (default: "
              "%(default)s)"))
    parser.add_argument(
        "--min-code-len",
        type=int,
        default=1,
        help="Minimum audio code sequence length (default: %(default)s)")
    parser.add_argument(
        "--opt-code-len",
        type=int,
        default=300,
        help=("Audio code sequence length used for profile optimization "
              "(default: %(default)s)"))
    parser.add_argument(
        "--max-code-len",
        type=int,
        default=2000,
        help="Maximum audio code sequence length (default: %(default)s)")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Number of tensor-parallel ranks (default: %(default)s)")
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=0,
        help=("Tensor-parallel rank to build, starting at 0 (default: "
              "%(default)s)"))
    parser.add_argument(
        "--reduced-vocab-dir",
        help=("Directory containing reduced-vocabulary artifacts for the base "
              "model"))
    parser.add_argument(
        "--draft-reduced-vocab-dir",
        help=("Directory containing reduced-vocabulary artifacts for the "
              "draft model"))
    parser.add_argument(
        "--draft-model-dir",
        help="Local draft checkpoint directory for paired speculative builds")
    parser.add_argument("--fp8-embedding",
                        action="store_true",
                        help="Store external embedding weights in FP8")
    parser.add_argument(
        "--weightless-engine",
        action="store_true",
        help=
        ("Experimental weightless constants for supported LLM and audio components. "
         "Requires TensorRT null-weight placeholder support and "
         "checkpoint refit before inference."))
    parser.add_argument(
        "--externalize-weights",
        action="append",
        choices=weight_policy.EXTERNAL_WEIGHT_CHOICES,
        help=("Limit which weights leave the engine, repeatable. Defaults to "
              "every supported kind, except that TP builds keep small FP16 "
              "parameters baked for runtime performance. FP8 and MXFP8 "
              "weights are always baked in."))
    parser.add_argument(
        "--gr00t-embodiment-id",
        type=int,
        help=(
            "GR00T N1.7 only: the embodiment whose category-specific encoder "
            "weights are baked in. Defaults to the checkpoint's own id."))
    parser.add_argument(
        "--profiling-detailed",
        action="store_true",
        help="Include detailed TensorRT profiling information in the engine")
    parser.set_defaults(spec_role=contracts.SpecRole.NONE.value,
                        target_model_dir=None)


def _parser(prog="tensorrt-edgellm-build") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Build every TensorRT engine required by a checkpoint "
                     "without an intermediate ONNX export."))
    _add_common_engine_args(parser)
    _add_build_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None,
         *,
         prog="tensorrt-edgellm-build",
         on_progress: Optional[Callable[[str], None]] = None) -> None:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.graph_only:
        _write_graph_only(args)
        return
    _build(args, on_progress=on_progress)


if __name__ == "__main__":
    main()
