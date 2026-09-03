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
from typing import Iterable, Optional, Sequence, Tuple, Union

from tensorrt_edgellm._native import NativeManifestNotFoundError
from tensorrt_edgellm._native.load import resolve_payload

LOGGER = logging.getLogger("experimental.builder")


def _value_or_default(value, default):
    return default if value is None else value


def _build_args(args: argparse.Namespace, component: str):
    from .core.builder import BuildArgs

    return BuildArgs(
        model_dir=args.model_dir,
        engine_dir=args.engine_dir,
        component=component,
        spec_role=args.spec_role,
        spec_type=args.spec_type,
        max_input_len=args.max_input_len,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        max_batch_size=args.max_batch_size,
        max_lora_rank=args.max_lora_rank,
        max_verify_tree_size=_value_or_default(args.max_verify_tree_size, 60),
        max_draft_tree_size=_value_or_default(args.max_draft_tree_size, 60),
        tree_base=args.tree_base,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_image_tokens_per_image=args.max_image_tokens_per_image,
        min_time_steps=args.min_time_steps,
        max_time_steps=args.max_time_steps,
        min_code_len=args.min_code_len,
        opt_code_len=args.opt_code_len,
        max_code_len=args.max_code_len,
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
        int4_gemm_plugin_version=args.int4_gemm_plugin_version,
        externalize_weights=tuple(args.externalize_weights or ()),
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


def _speculative_build_plan(args: argparse.Namespace, bundle, components):
    """Expand one model build into its base, draft, and component builds."""
    from .core import contracts
    from .core.config import BundleConfig

    if contracts.Component.LLM not in components:
        raise ValueError("--spec-type requires the llm component")
    if args.spec_type == "mtp" and args.draft_model_dir:
        raise ValueError(
            "native MTP reads draft layers from --model-dir; do not pass "
            "--draft-model-dir")
    if args.spec_type != "mtp" and not args.draft_model_dir:
        raise ValueError(f"{args.spec_type} requires --draft-model-dir")

    draft_model_dir = args.draft_model_dir or args.model_dir
    draft_bundle = BundleConfig.from_pretrained(draft_model_dir)
    if args.spec_type == "dspark":
        draft = draft_bundle.component_dict(contracts.Component.LLM)
        dspark = draft.get("dspark_config") or {}
        block_size = int(dspark.get("block_size", draft.get("block_size", 7)))
        draft_tree_size = _value_or_default(args.max_draft_tree_size,
                                            block_size)
        verify_tree_size = _value_or_default(args.max_verify_tree_size,
                                             draft_tree_size + 1)
        args = _copy_args(args,
                          max_verify_tree_size=verify_tree_size,
                          max_draft_tree_size=draft_tree_size)
    plan = []
    for component in components:
        if component != contracts.Component.LLM:
            component_args = _copy_args(args,
                                        spec_type="none",
                                        tree_base=False,
                                        draft_model_dir=None,
                                        target_model_dir=None)
            plan.append((component.value, component_args, bundle, component))
            continue

        base_args = _copy_args(
            args,
            spec_role=contracts.SpecRole.BASE.value,
            draft_model_dir=(draft_model_dir if args.spec_type
                             in ("eagle3", "dflash", "jetspec",
                                 "dspark") else None),
            target_model_dir=None,
        )
        draft_args = _copy_args(
            args,
            model_dir=draft_model_dir,
            spec_role=contracts.SpecRole.DRAFT.value,
            tree_base=False,
            draft_model_dir=None,
            target_model_dir=args.model_dir,
        )
        plan.extend((
            ("spec-base", base_args, bundle, component),
            ("spec-draft", draft_args, draft_bundle, component),
        ))
    return plan


def _build_plan(args: argparse.Namespace, bundle, components):
    from .core import contracts

    if (args.spec_role == contracts.SpecRole.NONE.value
            and args.spec_type != "none"):
        return _speculative_build_plan(args, bundle, components)
    return [(component.value, args, bundle, component)
            for component in components]


def _build_one(args: argparse.Namespace, bundle, component,
               plugin_handle) -> str:
    from .core import contracts
    from .core.builder import build_engine, load_device_config
    from .core.bundle import LLM_COMPONENTS
    from .models import registry as model_registry

    build_args = _build_args(args, component.value)
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
        bundle.root_model_type, build_args.spec_type,
        build_args.resolved_spec_role)
    artifact_writer.write_artifacts(bundle, artifact_cfg, build_args,
                                    args.engine_dir)
    if result.checkpoint_weight_bindings:
        from .core.artifacts import patch_external_weight_config
        config_path = contracts.component_spec(component).config_path(
            args.engine_dir, build_args.resolved_spec_role, build_args.tp_size)
        patch_external_weight_config(
            config_path,
            result.checkpoint_weight_bindings,
            result.checkpoint_identity,
            checkpoint_dir=(result.checkpoint_dir
                            or os.path.abspath(build_args.model_dir)),
        )
    return result.engine_path


def _build(args: argparse.Namespace) -> None:
    from .core.builder import load_plugin_library

    plugin_path = args.plugin_path
    if plugin_path is None:
        try:
            plugin_path = str(resolve_payload().plugin)
        except NativeManifestNotFoundError:
            plugin_path = "build/libNvInfer_edgellm_plugin.so"
    args = _copy_args(args, plugin_path=plugin_path)
    bundle, components = _resolve_build_selection(args)
    plugin_handle = load_plugin_library(plugin_path)
    plan = _build_plan(args, bundle, components)
    LOGGER.info("Building %s engines for %s: %s", len(plan),
                bundle.root_model_type, ", ".join(label for label, *_ in plan))
    results = []
    for label, build_args, build_bundle, component in plan:
        LOGGER.info("Building component %s", label)
        engine_path = _build_one(build_args, build_bundle, component,
                                 plugin_handle)
        results.append((label, engine_path, os.path.getsize(engine_path)))

    print("Engine outputs:")
    for component, engine_path, size in results:
        print(f"  {component}: {engine_path} ({size} bytes)")


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument(
        "--plugin-path",
        help=("Plugin DSO override. By default, use the payload selected from "
              "the installed TensorRT Edge-LLM wheel."))
    parser.add_argument("--verbose", action="store_true")


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    from .core import contracts, weight_policy

    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--components",
        default="all",
        help=("Comma-separated component list or 'all'. Default builds every "
              "component available in the checkpoint. Build order: " +
              ", ".join(contracts.component_build_order()) + "."))
    parser.add_argument(
        "--spec-type",
        choices=("none", "eagle3", "mtp", "dflash", "jetspec", "dspark",
                 "gemma4_mtp"),
        default="none",
        help=("Build both speculative engines in this single invocation. "
              "EAGLE3, DFlash, JetSpec, dSpark, and Gemma4 MTP also require "
              "--draft-model-dir."))
    parser.add_argument("--dense",
                        choices=("auto", "nvfp4-qdq", "fp16"),
                        default="auto")
    parser.add_argument(
        "--int4-gemm-plugin-version",
        type=int,
        choices=(1, 2),
        default=2,
        help=("Dense INT4 implementation. V2 is the default CuTeDSL fragment "
              "layout; V1 is the legacy plugin-packed fallback."))
    parser.add_argument("--max-input-len", type=int, default=32)
    parser.add_argument("--max-kv-cache-capacity", type=int, default=96)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=0)
    parser.add_argument("--max-verify-tree-size", type=int)
    parser.add_argument("--max-draft-tree-size", type=int)
    parser.add_argument(
        "--tree-base",
        action="store_true",
        help=("Build an MTP or DFlash base with DDTree parent/depth metadata "
              "for hybrid recurrent-state verification."))
    parser.add_argument("--min-image-tokens", type=int, default=4)
    parser.add_argument("--max-image-tokens", type=int, default=1024)
    parser.add_argument("--max-image-tokens-per-image", type=int, default=512)
    parser.add_argument("--min-time-steps", type=int, default=100)
    parser.add_argument("--max-time-steps", type=int, default=6000)
    parser.add_argument("--min-code-len", type=int, default=1)
    parser.add_argument("--opt-code-len", type=int, default=300)
    parser.add_argument("--max-code-len", type=int, default=2000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--reduced-vocab-dir")
    parser.add_argument("--draft-reduced-vocab-dir")
    parser.add_argument("--draft-model-dir")
    parser.add_argument("--fp8-embedding", action="store_true")
    parser.add_argument(
        "--externalize-weights",
        action="append",
        choices=weight_policy.EXTERNAL_WEIGHT_CHOICES,
        help=("Limit which weights leave the engine, repeatable. Defaults to "
              "every supported kind, except that TP builds keep small FP16 "
              "parameters baked for runtime performance. FP8 and MXFP8 "
              "weights are always baked in."))
    parser.add_argument("--profiling-detailed", action="store_true")
    parser.set_defaults(spec_role=contracts.SpecRole.NONE.value,
                        target_model_dir=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build every TensorRT engine required by a checkpoint "
                     "without an intermediate ONNX export."))
    _add_common_engine_args(parser)
    _add_build_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _build(args)


if __name__ == "__main__":
    main()
