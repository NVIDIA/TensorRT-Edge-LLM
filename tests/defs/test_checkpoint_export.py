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
"""Checkpoint export test suite for pre-quantized models.

Uses `tensorrt-edgellm-export` to export pre-quantized
models (e.g. Nemotron 4B NVFP4, Phi-4) to ONNX format. The exported ONNX
is placed in the same ONNX directory structure used by the standard export
pipeline, so downstream engine build and inference tests can consume it.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from tensorrt_edgellm import config as edgellm_config

from .config import (GEMMA4_MTP_ASSISTANT_MODELS_MAP, ModelType, TaskType,
                     TestConfig, infer_checkpoint_export_model_type,
                     strip_model_quant_suffixes)
from .utils.command_generation import resolve_lora_model_name

# --externalize-weights wiring. extw_<token> -> CLI kinds for
# tensorrt_edgellm.scripts.export --externalize-weights.
_EXTW_TOKEN_MAP = {
    "ffn": ["int4_ffn"],
    "lm": ["lm_head"],
    "moe": ["int4_moe"],
    "nvfp4_moe": ["nvfp4_moe"],
    "ffn_lm": ["int4_ffn", "lm_head"],
    "all": ["all"],
}

_EXTW_FILE_BY_KIND = {
    "int4_ffn": "external_int4_ffn_weights.safetensors",
    "int4_moe": "external_int4_moe_weights.safetensors",
    "nvfp4_moe": "external_nvfp4_moe_weights.safetensors",
    "lm_head": "external_lm_head_weight.safetensors",
}

_DFLASH_TARGET_KV_UPDATE_OP = "DFlashTargetKVCacheUpdate"
_DFLASH_TARGET_KV_UPDATE_INPUT_COUNT = 7
_DFLASH_TARGET_KV_PAGE_TABLE_INPUT_INDEX = 6


def _extw_cli_kinds(extw_token):
    """Resolve a test_param ``extw_<value>`` token to CLI kinds."""
    if extw_token is None:
        return []
    try:
        return list(_EXTW_TOKEN_MAP[extw_token])
    except KeyError:
        raise ValueError(
            f"Unknown externalize_weights token 'extw_{extw_token}'. "
            f"Valid: {sorted('extw_' + k for k in _EXTW_TOKEN_MAP)}")


def _verify_externalized_outputs(out_dir, requested_cli_kinds):
    """Assert externalization produced the expected safetensors + manifest.

    ``all`` passes if at least one kind was externalized; explicit kinds must
    each produce both a safetensors file and a config.json manifest entry.
    """
    if not requested_cli_kinds:
        return
    config_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(config_path):
        pytest.fail(
            f"Externalize-weights check: missing config.json in {out_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    manifest = cfg.get("external_weight_files") or []
    manifest_files = {entry.get("file", "") for entry in manifest}
    manifest_kinds = {entry.get("kind", "") for entry in manifest}

    def _has(kind):
        filename = _EXTW_FILE_BY_KIND[kind]
        return (os.path.isfile(os.path.join(out_dir, filename))
                and filename in manifest_files)

    if "all" in requested_cli_kinds:
        if any(_has(k) for k in _EXTW_FILE_BY_KIND):
            return
        pytest.fail(
            f"--externalize-weights all produced no output in {out_dir}. "
            f"Expected at least one of {sorted(_EXTW_FILE_BY_KIND)} on disk "
            f"and in the config.json manifest. manifest={manifest!r}")

    missing = []
    for kind in requested_cli_kinds:
        filename = _EXTW_FILE_BY_KIND[kind]
        on_disk = os.path.isfile(os.path.join(out_dir, filename))
        in_manifest = filename in manifest_files
        if not (on_disk and in_manifest):
            missing.append(
                f"{kind}: file_on_disk={on_disk}, in_manifest={in_manifest}")
    if missing:
        pytest.fail(f"Externalize-weights check failed in {out_dir} "
                    f"(requested={requested_cli_kinds}):\n" +
                    "\n".join(f"  - {m}" for m in missing) +
                    f"\n  manifest_kinds={sorted(manifest_kinds)}")


def _onnx_graph_input_names(onnx_path):
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    return {value_info.name for value_info in model.graph.input}


def _requires_ddtree_state_inputs(model_dir):
    cfg = edgellm_config.ModelConfig.from_pretrained(
        model_dir, lambda head_dim: 1.0 / (float(head_dim)**0.5))
    return cfg.is_hybrid


def _verify_tree_base_inputs(mode, onnx_path, model_dir):
    input_names = _onnx_graph_input_names(onnx_path)
    expected_attention_inputs = {"attention_pos_id", "attention_mask"}
    missing = expected_attention_inputs - input_names
    if missing:
        pytest.fail(
            f"{mode} tree-base ONNX missing inputs {sorted(missing)}: {onnx_path}"
        )

    if not _requires_ddtree_state_inputs(model_dir):
        return

    expected_state_inputs = {"tree_parent_ids", "tree_depths"}
    missing = expected_state_inputs - input_names
    if missing:
        pytest.fail(
            f"{mode} hybrid tree-base ONNX missing inputs {sorted(missing)}: {onnx_path}"
        )


def _assert_dflash_target_kv_update_contract(onnx_path):
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    nodes = [
        node for node in model.graph.node
        if node.op_type == _DFLASH_TARGET_KV_UPDATE_OP
    ]
    if not nodes:
        pytest.fail(
            f"{onnx_path} contains no {_DFLASH_TARGET_KV_UPDATE_OP} nodes")

    for node in nodes:
        if len(node.input) != _DFLASH_TARGET_KV_UPDATE_INPUT_COUNT:
            pytest.fail(
                f"{_DFLASH_TARGET_KV_UPDATE_OP} node {node.name!r} has "
                f"{len(node.input)} inputs; expected "
                f"{_DFLASH_TARGET_KV_UPDATE_INPUT_COUNT}")
        page_table_input = node.input[_DFLASH_TARGET_KV_PAGE_TABLE_INPUT_INDEX]
        if page_table_input != "kv_page_table":
            pytest.fail(
                f"{_DFLASH_TARGET_KV_UPDATE_OP} node {node.name!r} input "
                f"{_DFLASH_TARGET_KV_PAGE_TABLE_INPUT_INDEX} must be "
                f"'kv_page_table', got {page_table_input!r}")


def test_checkpoint_export(test_param: str, test_logger,
                           env_config: EnvironmentConfig):
    """Export a pre-quantized model via tensorrt_edgellm.scripts.export."""

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)

    # Locate source model checkpoint
    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    # Determine the final ONNX output directory (same layout as standard export)
    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    # Use a temporary directory for the raw export output
    tmp_dir = tempfile.mkdtemp(prefix="checkpoint_export_")

    try:
        # Build the export command
        # PYTHONPATH must include repository root so the source package imports.
        # The CI job sets this via PYTHONPATH=$LLM_SDK_DIR:$PYTHONPATH.
        export_cmd = [
            sys.executable,
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
        ]

        if config.model_name.startswith("Cosmos3-Edge-reasoning"):
            export_cmd += ["--task", "reasoning"]

        extw_kinds = _extw_cli_kinds(config.externalize_weights)
        if extw_kinds:
            export_cmd += ["--externalize-weights", *extw_kinds]

        if config.int4_gemm_plugin_v1:
            export_cmd += ["--int4-gemm-plugin-version", "1"]

        env_vars = {}
        if config.trt_native_attn:
            env_vars["USE_TRT_NATIVE_ATTN"] = "1"
        env_vars.update(config.get_export_env_vars())

        with timer_context(
                f"Exporting {config.model_name} via the checkpoint exporter",
                test_logger):
            result = run_command(export_cmd,
                                 timeout=600,
                                 remote_config=None,
                                 logger=test_logger,
                                 env_vars=env_vars or None)
            if not result['success']:
                pytest.fail(
                    f"checkpoint export failed: {result.get('error', 'Unknown error')}"
                )

        # Move the LLM ONNX output to the expected directory.
        # TTS exports two LLM-class sub-models (talker + code_predictor); build
        # expects them under ``<llm_onnx_dir>/{talker,code_predictor}/``. The
        # standard non-TTS export emits a single ``tmp/llm/`` that maps
        # directly to ``<llm_onnx_dir>/``.
        llm_output = os.path.join(tmp_dir, "llm")
        cp_output = os.path.join(tmp_dir, "code_predictor")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"the checkpoint exporter did not produce llm/ output directory in {tmp_dir}"
            )
        if os.path.isdir(cp_output):
            # TTS layout: tmp/llm/         -> onnx/llm-<prec>/talker/
            #             tmp/code_predictor/ -> onnx/llm-<prec>/code_predictor/
            shutil.copytree(llm_output,
                            os.path.join(llm_onnx_dir, "talker"),
                            dirs_exist_ok=True)
            shutil.copytree(cp_output,
                            os.path.join(llm_onnx_dir, "code_predictor"),
                            dirs_exist_ok=True)
        else:
            # Standard layout: tmp/llm/ -> onnx/llm-<prec>/
            shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        # If the model also has a visual encoder output, move that too
        visual_output = os.path.join(tmp_dir, "visual")
        if os.path.isdir(visual_output):
            visual_onnx_dir = config.get_visual_onnx_dir(
                config.visual_precision or "fp16")
            shutil.copytree(visual_output, visual_onnx_dir, dirs_exist_ok=True)

        # Same for audio encoder (ASR / Omni).
        audio_output = os.path.join(tmp_dir, "audio")
        if config.model_type == ModelType.ASR:
            missing_audio = [
                path for path in (
                    os.path.join(audio_output, "model.onnx"),
                    os.path.join(audio_output, "config.json"),
                ) if not os.path.isfile(path)
            ]
            if missing_audio:
                pytest.fail("ASR checkpoint export missing audio outputs:\n" +
                            "\n".join(f"  - {p}" for p in missing_audio))
        if os.path.isdir(audio_output):
            audio_onnx_dir = config.get_audio_onnx_dir(config.audio_precision
                                                       or "fp16")
            shutil.copytree(audio_output, audio_onnx_dir, dirs_exist_ok=True)

        # Same for Code2Wav vocoder.
        code2wav_output = os.path.join(tmp_dir, "code2wav")
        if os.path.isdir(code2wav_output):
            code2wav_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                             "code2wav-fp16")
            shutil.copytree(code2wav_output,
                            code2wav_onnx_dir,
                            dirs_exist_ok=True)

        # Alpamayo VLA action expert.
        action_output = os.path.join(tmp_dir, "action")
        if os.path.isdir(action_output):
            shutil.copytree(action_output,
                            config.get_action_onnx_dir(),
                            dirs_exist_ok=True)

        _verify_externalized_outputs(llm_onnx_dir, extw_kinds)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Validate the exported ONNX model exists. TTS produces model.onnx under
    # ``talker/`` (the talker is the LLM-class submodel); non-TTS produces it
    # at the root of llm_onnx_dir.
    onnx_candidates = [
        os.path.join(llm_onnx_dir, "model.onnx"),
        os.path.join(llm_onnx_dir, "talker", "model.onnx"),
    ]
    if not any(os.path.exists(p) for p in onnx_candidates):
        pytest.fail(
            f"LLM ONNX model not found after export at any of: {onnx_candidates}"
        )


def _run_checkpoint_export(cmd, timeout, test_logger, label, env_vars=None):
    """Run an checkpoint export command and fail on error."""
    with timer_context(label, test_logger):
        result = run_command(cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"{label} failed: {result.get('error', 'Unknown error')}")


def test_checkpoint_eagle_export(test_param: str, test_logger,
                                 env_config: EnvironmentConfig):
    """Export EAGLE base + draft models via tensorrt_edgellm.scripts.export.

    Uses pre-quantized unified checkpoints for both base and draft models
    so that no on-the-fly quantization is needed during CI.

    Two tensorrt-edgellm-export invocations:
      1. Base model with --eagle-base  -> llm-base ONNX (+ visual for VLMs)
      2. Draft model -> draft ONNX
    """

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    # Locate pre-quantized base model checkpoint
    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    # Locate pre-quantized draft model checkpoint (hub name or local quant output)
    draft_torch_dir = config.get_eagle_draft_checkpoint_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"Draft model checkpoint not found: {draft_torch_dir}")

    # Export params include quantized hub suffixes to locate the checkpoint, but
    # downstream ONNX paths are keyed by the base model name.
    config.model_name = strip_model_quant_suffixes(config.model_name)

    # Output directories
    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="eagle_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="eagle_draft_export_")

    try:
        # --- Export base model with --eagle-base ---
        base_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_base,
            "--eagle-base",
        ]
        extw_kinds = _extw_cli_kinds(config.externalize_weights)
        if extw_kinds:
            base_cmd += ["--externalize-weights", *extw_kinds]
        _run_checkpoint_export(
            base_cmd,
            600,
            test_logger,
            f"Exporting EAGLE base {config.model_name} via the checkpoint exporter",
            env_vars=export_env_vars)

        # Copy base LLM ONNX
        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(f"Base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)
        _verify_externalized_outputs(llm_onnx_dir, extw_kinds)

        # Copy visual encoder if present (VLM models)
        base_vis_out = os.path.join(tmp_base, "visual")
        if os.path.isdir(base_vis_out):
            visual_onnx_dir = config.get_visual_onnx_dir(
                config.visual_precision or "fp16")
            shutil.copytree(base_vis_out, visual_onnx_dir, dirs_exist_ok=True)

        # --- Export draft model ---
        draft_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            draft_torch_dir,
            tmp_draft,
        ]
        if extw_kinds:
            draft_cmd += ["--externalize-weights", *extw_kinds]
        _run_checkpoint_export(
            draft_cmd,
            600,
            test_logger,
            f"Exporting EAGLE draft {config.draft_model_id} via the checkpoint exporter",
            env_vars=export_env_vars)

        # Copy draft ONNX
        draft_llm_out = os.path.join(tmp_draft, "llm")
        if not os.path.isdir(draft_llm_out):
            pytest.fail(f"Draft export did not produce llm/ in {tmp_draft}")
        shutil.copytree(draft_llm_out, draft_onnx_dir, dirs_exist_ok=True)
        _verify_externalized_outputs(draft_onnx_dir, extw_kinds)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    # Validate outputs
    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"Base ONNX model not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"Draft ONNX model not found: {draft_onnx}")


def test_checkpoint_dflash_export(test_param: str, test_logger,
                                  env_config: EnvironmentConfig):
    """Export DFlash base + draft models via tensorrt_edgellm.scripts.export."""

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    draft_torch_dir = config.get_dflash_draft_model_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"DFlash draft model checkpoint not found: {draft_torch_dir}")

    # Export test params may include a quantized checkpoint suffix in the model
    # name (for example "Qwen3.5-4B-NVFP4") to locate the pre-quantized base
    # checkpoint.  Downstream build/inference tests use the canonical base model
    # name plus precision ("Qwen3.5-4B-nvfp4"), so normalize the ONNX path after
    # both checkpoint locations have been resolved.
    _QUANT_SUFFIXES = ("-NVFP4", "-FP8", "-FP8-KV", "-INT8-SQ", "-INT4-AWQ")
    for suffix in _QUANT_SUFFIXES:
        if config.model_name.endswith(suffix):
            config.model_name = config.model_name[:-len(suffix)]
            break

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="dflash_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="dflash_draft_export_")
    use_tree_base = config.is_dflash_tree
    base_export_flag = "--dflash-tree-base" if use_tree_base else "--dflash-base"

    try:
        base_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_base,
            base_export_flag,
            "--dflash-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            base_cmd,
            1200,
            test_logger,
            f"Exporting DFlash base {config.model_name} via the checkpoint exporter",
            env_vars=export_env_vars)

        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(
                f"DFlash base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)

        draft_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_draft,
            "--dflash-draft",
            "--dflash-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            draft_cmd,
            1200,
            test_logger,
            f"Exporting DFlash draft {config.draft_model_id} via the checkpoint exporter",
            env_vars=export_env_vars)

        draft_output = os.path.join(tmp_draft, "dflash_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(
                f"DFlash draft export did not produce dflash_draft/ in {tmp_draft}"
            )
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"DFlash base ONNX not found: {base_onnx}")
    if use_tree_base:
        _verify_tree_base_inputs("DFlash", base_onnx, base_torch_dir)

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"DFlash draft ONNX not found: {draft_onnx}")
    _assert_dflash_target_kv_update_contract(draft_onnx)


def test_checkpoint_jetspec_export(test_param: str, test_logger,
                                   env_config: EnvironmentConfig):
    """Export JetSpec base + draft models via tensorrt_edgellm.scripts.export."""

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    draft_torch_dir = config.get_jetspec_draft_model_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"JetSpec draft model checkpoint not found: {draft_torch_dir}")

    config.model_name = strip_model_quant_suffixes(config.model_name)

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="jetspec_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="jetspec_draft_export_")
    use_tree_base = config.is_jetspec_tree
    base_export_flag = "--jetspec-tree-base" if use_tree_base else "--jetspec-base"

    try:
        base_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_base,
            base_export_flag,
            "--jetspec-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            base_cmd,
            1200,
            test_logger,
            f"Exporting JetSpec base {config.model_name} via the checkpoint exporter",
            env_vars=export_env_vars)

        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(
                f"JetSpec base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)

        draft_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_draft,
            "--jetspec-draft",
            "--jetspec-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            draft_cmd,
            1200,
            test_logger,
            f"Exporting JetSpec draft {config.draft_model_id} via the checkpoint exporter",
            env_vars=export_env_vars)

        draft_output = os.path.join(tmp_draft, "jetspec_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(
                f"JetSpec draft export did not produce jetspec_draft/ in {tmp_draft}"
            )
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"JetSpec base ONNX not found: {base_onnx}")
    if use_tree_base:
        _verify_tree_base_inputs("JetSpec", base_onnx, base_torch_dir)

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"JetSpec draft ONNX not found: {draft_onnx}")
    _assert_dflash_target_kv_update_contract(draft_onnx)


def test_checkpoint_dspark_export(test_param: str, test_logger,
                                  env_config: EnvironmentConfig):
    """Export DSpark base + draft models via tensorrt_edgellm.scripts.export."""

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    draft_torch_dir = config.get_dspark_draft_model_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"DSpark draft model checkpoint not found: {draft_torch_dir}")

    # Export test params may include a quantized checkpoint suffix in the model
    # name. Normalize the output path after checkpoint resolution so downstream
    # build/inference tests use the canonical base model name plus precision.
    config.model_name = strip_model_quant_suffixes(config.model_name)

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="dspark_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="dspark_draft_export_")

    try:
        base_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_base,
            "--dspark-base",
            "--dspark-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            base_cmd, 1200, test_logger,
            f"Exporting DSpark base {config.model_name} via the checkpoint exporter"
        )

        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(
                f"DSpark base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)

        draft_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_draft,
            "--dspark-draft",
            "--dspark-draft-dir",
            draft_torch_dir,
        ]
        _run_checkpoint_export(
            draft_cmd, 1200, test_logger,
            f"Exporting DSpark draft {config.draft_model_id} via the checkpoint exporter"
        )

        draft_output = os.path.join(tmp_draft, "dspark_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(
                f"DSpark draft export did not produce dspark_draft/ in {tmp_draft}"
            )
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"DSpark base ONNX not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"DSpark draft ONNX not found: {draft_onnx}")
    _assert_dflash_target_kv_update_contract(draft_onnx)

    heads_path = os.path.join(draft_onnx_dir, "dspark_heads.safetensors")
    if not os.path.exists(heads_path):
        pytest.fail(f"DSpark draft sidecar not found: {heads_path}")


def test_checkpoint_mtp_export(test_param: str, test_logger,
                               env_config: EnvironmentConfig):
    """Export MTP base + draft from a single checkpoint via --mtp flag."""

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mtp_export_")

    try:
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
            "--mtp",
        ]
        base_model_name = strip_model_quant_suffixes(config.model_name)
        if base_model_name in GEMMA4_MTP_ASSISTANT_MODELS_MAP:
            export_cmd += [
                "--mtp-draft-dir",
                config.get_gemma4_mtp_assistant_model_dir(),
            ]

        extw_kinds = _extw_cli_kinds(config.externalize_weights)
        if extw_kinds:
            export_cmd += ["--externalize-weights", *extw_kinds]

        with timer_context(
                f"Exporting MTP {config.model_name} via the checkpoint exporter",
                test_logger):
            result = run_command(export_cmd,
                                 timeout=600,
                                 remote_config=None,
                                 logger=test_logger,
                                 env_vars=export_env_vars)
            if not result['success']:
                pytest.fail(
                    f"MTP export failed: {result.get('error', 'Unknown error')}"
                )

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(f"MTP export did not produce llm/ in {tmp_dir}")
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)
        _verify_externalized_outputs(llm_onnx_dir, extw_kinds)

        draft_output = os.path.join(tmp_dir, "mtp_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(f"MTP export did not produce mtp_draft/ in {tmp_dir}")
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)
        # The paired Gemma4 assistant draft is a small fp16 dense model (tied
        # embeddings, no int4/moe/lm_head weights), so --externalize-weights is
        # a no-op for it. Only quantized derived drafts carry externalizable
        # weights, so only verify the draft externalization in that case.
        if base_model_name not in GEMMA4_MTP_ASSISTANT_MODELS_MAP:
            _verify_externalized_outputs(draft_onnx_dir, extw_kinds)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"MTP base ONNX not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"MTP draft ONNX not found: {draft_onnx}")


def test_checkpoint_lora_export(test_param: str, test_logger,
                                env_config: EnvironmentConfig):
    """Export a model and insert LoRA patterns via the checkpoint exporter.

    Exports a model ONNX, then runs LoRA insertion and verifies the
    lora_model.onnx has additional LoRA nodes.
    """

    config = TestConfig.from_param_string(
        test_param, infer_checkpoint_export_model_type(test_param),
        TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="lora_export_")

    try:
        # Step 1: Export the model
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
        ]

        _run_checkpoint_export(
            export_cmd,
            600,
            test_logger,
            f"Exporting {config.model_name} for LoRA via the checkpoint exporter",
            env_vars=export_env_vars)

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"the checkpoint exporter did not produce llm/ output directory in {tmp_dir}"
            )

        orig_onnx = os.path.join(llm_output, "model.onnx")
        if not os.path.exists(orig_onnx):
            pytest.fail(f"model.onnx not found: {orig_onnx}")

        # Step 2: Insert LoRA patterns
        lora_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.insert_lora",
            "--onnx_dir",
            llm_output,
        ]

        _run_checkpoint_export(lora_cmd, 120, test_logger,
                               f"Inserting LoRA into {config.model_name}")

        lora_onnx = os.path.join(llm_output, "lora_model.onnx")
        if not os.path.exists(lora_onnx):
            pytest.fail(f"lora_model.onnx not found: {lora_onnx}")

        # Verify LoRA model has more nodes than original
        import onnx
        orig_model = onnx.load(orig_onnx, load_external_data=False)
        lora_model = onnx.load(lora_onnx, load_external_data=False)

        if len(lora_model.graph.node) <= len(orig_model.graph.node):
            pytest.fail(f"LoRA model should have more nodes than original. "
                        f"Original: {len(orig_model.graph.node)}, "
                        f"LoRA: {len(lora_model.graph.node)}")

        # Verify LoRA-specific inputs exist
        lora_inputs = [
            i.name for i in lora_model.graph.input if 'lora' in i.name.lower()
        ]
        if not lora_inputs:
            pytest.fail("No LoRA weight inputs found in lora_model.onnx")

        # Step 3: Process LoRA adapter weights for runtime tests.
        lora_model_name = resolve_lora_model_name(config.model_name)
        if lora_model_name is None:
            pytest.fail(f"No LoRA weights configured for {config.model_name}")
        lora_weights_dir = config.find_lora_adapter_weights_dir(
            lora_model_name)
        if not lora_weights_dir:
            pytest.fail(
                f"LoRA weights directory '{lora_model_name}' not found under "
                f"edgellm_data_dir={config.edgellm_data_dir}, "
                f"llm_models_dir={config.llm_models_dir}")

        process_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.process_lora_weights",
            "--input_dir",
            lora_weights_dir,
            "--output_dir",
            config.get_lora_weights_dir(),
        ]
        _run_checkpoint_export(
            process_cmd, 120, test_logger,
            f"Processing LoRA weights for {config.model_name}")

        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_llm_loader_tp_export(test_param: str, test_logger,
                              env_config: EnvironmentConfig):
    """Export per-rank ONNX for TP=2 via tensorrt_edgellm.scripts.export --tp-size 2.

    Validates:
      - Per-rank files exist: model_world2_rank{0,1}.onnx and matching .data
      - Each rank's external-data file has distinct content (regression
        guard for the per-rank filename collision in onnx/export.py
        _fix_initializer_dtypes — without the fix both ranks share
        model.onnx.data and the second-written wins).
    """
    import hashlib

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)
    export_env_vars = config.get_export_env_vars() or None

    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="llm_loader_tp_export_")
    try:
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
            "--tp-size",
            "2",
        ]
        _run_checkpoint_export(export_cmd,
                               600,
                               test_logger,
                               f"TP=2 export for {config.model_name}",
                               env_vars=export_env_vars)

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"llm_loader did not produce llm/ output directory in {tmp_dir}"
            )
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Validate per-rank files exist and external-data files are distinct.
    rank_files = []
    for rank in (0, 1):
        onnx_path = os.path.join(llm_onnx_dir, f"model_world2_rank{rank}.onnx")
        data_path = onnx_path + ".data"
        if not os.path.exists(onnx_path):
            pytest.fail(f"Missing per-rank ONNX: {onnx_path}")
        if not os.path.exists(data_path):
            pytest.fail(f"Missing per-rank external data: {data_path}")
        rank_files.append((onnx_path, data_path))

    # Distinct .data content — catches the model.onnx.data collision bug.
    md5s = []
    for _, data_path in rank_files:
        with open(data_path, "rb") as f:
            md5s.append(hashlib.md5(f.read()).hexdigest())
    if md5s[0] == md5s[1]:
        pytest.fail(
            f"TP=2 rank0 and rank1 external-data files have identical content "
            f"({md5s[0]}); per-rank sharding collapsed (likely the "
            f"model.onnx.data filename collision in onnx/export.py)")
