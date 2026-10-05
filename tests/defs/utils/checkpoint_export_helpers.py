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
"""Shared helpers for exporting models via ``tensorrt_edgellm.scripts.export``.

Used by both ``test_model_export.py`` and standalone exporter tests.

The checkpoint exporter handles chat templates internally. It materializes the
model-owned ``chat_template.jinja`` (or an explicit manual renderer marker)
alongside the exported graph, so no template argument is needed.
"""

import logging
import os
import shutil
import tempfile
from typing import Optional

import pytest
from pytest_helpers import run_command, timer_context

from ..config import ModelType, TestConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model directory selection
# ---------------------------------------------------------------------------


def get_export_model_dir(config: TestConfig) -> str:
    """Determine which checkpoint directory to feed to the exporter.

    After an optional quantization step the weights live in a derived
    directory; otherwise we use the original torch checkpoint.

    This is the single source of truth for export tests and pipeline
    pre-export steps.
    """
    if config.fp8_kv_cache and config.llm_precision == "fp16":
        return config.get_kv_cache_quantized_model_dir()
    if config.visual_precision == "fp8" or config.audio_precision == "fp8":
        return config.get_quantized_model_dir()
    if config.llm_precision != "fp16" and not config.is_prequantized():
        return config.get_quantized_model_dir()
    if config.merge_lora:
        return config.get_merged_model_dir()
    return config.get_torch_model_dir()


# ---------------------------------------------------------------------------
# Output directory mapping
# ---------------------------------------------------------------------------

# Mapping from tensorrt_edgellm output sub-directory names to a callable that
# returns the target directory given a TestConfig.  Add new entries here
# when tensorrt_edgellm starts emitting additional top-level output dirs — no
# other copy logic needs to change.
#
# Special cases handled outside this map:
#   - TTS talker/code_predictor: talker is written at ``llm/`` in the
#     exporter output, but downstream engine build expects
#     ``<llm_onnx_dir>/talker/``; code_predictor is copied to
#     ``<llm_onnx_dir>/code_predictor/``.
#   - EAGLE draft: handled by ``run_tensorrt_edgellm_draft_export`` which copies
#     the draft output to ``config.get_draft_onnx_dir()``.

_OUTPUT_DIR_MAP = {
    # subdir_name: (target_dir_func, description)
    "visual":
    (lambda cfg: cfg.get_visual_onnx_dir(cfg.visual_precision or "fp16"),
     "visual encoder"),
    "audio":
    (lambda cfg: cfg.get_audio_onnx_dir(cfg.audio_precision or "fp16"),
     "audio encoder"),
    "code2wav":
    (lambda cfg: cfg.get_code2wav_onnx_dir("fp16"), "fp16 Code2Wav"),
    "action": (lambda cfg: cfg.get_action_onnx_dir(), "action expert"),
}


def _copy_subdir(tmp_dir: str, subdir: str, target_dir: str) -> bool:
    """Copy ``tmp_dir/subdir`` → ``target_dir`` if it exists. Return True if copied."""
    src = os.path.join(tmp_dir, subdir)
    if not os.path.isdir(src):
        return False
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(src, target_dir, dirs_exist_ok=True)
    return True


def _log_dir_contents(dir_path: str, label: str, test_logger=None) -> None:
    """Log the files in *dir_path* with sizes in MB. Used to sanity-check
    the checkpoint export output before it is copied to the final location.
    """
    log = test_logger.info if test_logger else logger.info
    if not os.path.isdir(dir_path):
        return
    entries = sorted(os.listdir(dir_path))
    log(f"{label} export produced {len(entries)} files in {dir_path}:")
    for name in entries:
        path = os.path.join(dir_path, name)
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.isfile(
            path) else 0.0
        log(f"  {size_mb:7.2f} MB  {name}")


def copy_tensorrt_edgellm_output(tmp_dir: str,
                                 config: TestConfig,
                                 test_logger=None) -> None:
    """Copy checkpoint export output from *tmp_dir* into the directory
    layout expected by downstream engine-build and inference tests.

    Raises ``pytest.fail`` if the mandatory ``llm/`` output is missing.
    """

    def _log(msg, *args):
        if test_logger:
            test_logger.info(msg, *args)
        else:
            logger.info(msg, *args)

    llm_onnx_dir = config.get_llm_onnx_dir()
    llm_output = os.path.join(tmp_dir, "llm")

    if not os.path.isdir(llm_output):
        pytest.fail(
            f"the checkpoint exporter did not produce llm/ output in {tmp_dir}"
        )

    _log_dir_contents(llm_output, "LLM", test_logger=test_logger)

    # --- LLM / Talker ---
    if config.model_type == ModelType.TTS:
        # TTS: exporter writes Talker at llm/model.onnx; downstream
        # engine build expects get_llm_onnx_dir()/talker/.
        dst = os.path.join(llm_onnx_dir, "talker")
        _copy_subdir(tmp_dir, "llm", dst)
        _log("Copied llm/ → %s (talker)", dst)
    else:
        os.makedirs(llm_onnx_dir, exist_ok=True)
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)
        _log("Copied llm/ → %s", llm_onnx_dir)

    # --- CodePredictor (TTS) ---
    # Placed under get_llm_onnx_dir()/code_predictor/ to match engine build.
    cp_src = os.path.join(tmp_dir, "code_predictor")
    if os.path.isdir(cp_src):
        _log_dir_contents(cp_src, "CodePredictor", test_logger=test_logger)
    cp_dst = os.path.join(llm_onnx_dir, "code_predictor")
    if _copy_subdir(tmp_dir, "code_predictor", cp_dst):
        _log("Copied code_predictor/ → %s", cp_dst)

    # --- Generic mapped outputs (visual, audio, future additions) ---
    for subdir, (target_fn, desc) in _OUTPUT_DIR_MAP.items():
        src = os.path.join(tmp_dir, subdir)
        if os.path.isdir(src):
            _log_dir_contents(src, desc.capitalize(), test_logger=test_logger)
        dst = target_fn(config)
        if _copy_subdir(tmp_dir, subdir, dst):
            _log("Copied %s/ → %s (%s)", subdir, dst, desc)


# ---------------------------------------------------------------------------
# Export execution
# ---------------------------------------------------------------------------


def _resolve_repo_root() -> Optional[str]:
    """Locate the TensorRT Edge-LLM repository root.

    Checks ``LLM_SDK_DIR`` first, then falls back to a path
    relative to this file so the helper works when LLM_SDK_DIR is unset.
    """
    sdk_dir = os.environ.get("LLM_SDK_DIR")
    if sdk_dir and os.path.isdir(os.path.join(sdk_dir, "tensorrt_edgellm")):
        return os.path.abspath(sdk_dir)

    # Fallback: <repo>/tests/defs/utils/checkpoint_export_helpers.py -> <repo>
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if os.path.isdir(os.path.join(candidate, "tensorrt_edgellm")):
        return candidate

    return None


def get_tensorrt_edgellm_root() -> Optional[str]:
    """TensorRT Edge-LLM repository root on disk.

    Used so subprocesses can run ``tensorrt-edgellm-quantize ...`` with
    a ``PYTHONPATH``/cwd that resolves the source checkout package.
    """
    return _resolve_repo_root()


def _checkpoint_export_env(test_logger=None) -> dict:
    """Return env_vars dict that makes ``tensorrt_edgellm`` importable."""
    repo_root = _resolve_repo_root()
    if not repo_root:
        if test_logger:
            test_logger.warning(
                "Could not locate repository root for tensorrt_edgellm. "
                "Set LLM_SDK_DIR or install the package.")
        return {}
    existing = os.environ.get("PYTHONPATH", "")
    new_pp = f"{repo_root}{os.pathsep}{existing}" if existing else repo_root
    return {"PYTHONPATH": new_pp}


def _run_export_subprocess(model_dir: str,
                           tmp_dir: str,
                           label: str,
                           test_logger,
                           timeout: int,
                           extra_args: Optional[list] = None,
                           extra_env: Optional[dict] = None,
                           failure_prefix: str = "checkpoint export") -> None:
    """Invoke ``python3 -m tensorrt_edgellm.scripts.export`` and fail on error.

    Centralizes PYTHONPATH setup, timing, and error handling shared by
    base and draft exports.
    """
    # Optional: EDGELLM_EXPORT_PYTHON (e.g. export venv vs quant venv) set by test harness.
    ex_py = os.environ.get("EDGELLM_EXPORT_PYTHON", "python3")
    export_cmd = [
        ex_py, "-m", "tensorrt_edgellm.scripts.export", model_dir, tmp_dir
    ]
    if extra_args:
        export_cmd.extend(extra_args)

    env_vars = _checkpoint_export_env(test_logger) or {}
    if extra_env:
        env_vars.update(extra_env)

    with timer_context(label, test_logger):
        result = run_command(export_cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            pytest.fail(f"{failure_prefix} failed: "
                        f"{result.get('error', 'Unknown error')}")


def run_checkpoint_export(config: TestConfig,
                          test_logger,
                          model_dir: Optional[str] = None,
                          timeout: int = 1200,
                          eagle_base: bool = False) -> None:
    """Run ``tensorrt_edgellm.scripts.export`` and copy output to final dirs.

    Args:
        config: Test configuration (determines output directories).
        test_logger: Logger for test output.
        model_dir: Checkpoint directory to export. Defaults to
            ``get_export_model_dir(config)``.
        timeout: Export command timeout in seconds.
        eagle_base: When True, pass ``--eagle-base`` to export an EAGLE3
            base model (adds tree-attention I/O and hidden_states output).
    """
    if model_dir is None:
        model_dir = get_export_model_dir(config)

    tmp_dir = tempfile.mkdtemp(prefix="checkpoint_export_")
    try:
        label = f"Exporting {config.model_name} via the checkpoint exporter"
        extra_args = []
        if eagle_base:
            extra_args.append("--eagle-base")
            label += " (EAGLE base)"
        if config.fp8_embedding:
            extra_args.append("--fp8-embedding")
            label += " (FP8 embedding)"
        if config.reduced_vocab_size:
            extra_args.append(
                f"--reduced-vocab-dir={config.get_reduced_vocab_dir()}")
            label += f" (rvs{config.reduced_vocab_size})"
        if (config.model_type == ModelType.VLA
                and config.max_kv_cache_capacity is not None):
            extra_args.append(
                f"--max-kv-cache-capacity={config.max_kv_cache_capacity}")
            label += f" (mxkvc{config.max_kv_cache_capacity})"

        extra_env = {}
        if config.trt_native_attn:
            extra_env["USE_TRT_NATIVE_ATTN"] = "1"
            label += " (TRT-native attn)"
        extra_env.update(config.get_export_env_vars())

        _run_export_subprocess(model_dir,
                               tmp_dir,
                               label,
                               test_logger,
                               timeout,
                               extra_args=extra_args,
                               extra_env=extra_env or None)

        copy_tensorrt_edgellm_output(tmp_dir, config, test_logger=test_logger)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_tensorrt_edgellm_draft_export(config: TestConfig,
                                      test_logger,
                                      timeout: int = 1200) -> None:
    """Export the EAGLE draft model via ``tensorrt_edgellm.scripts.export``.

    Uses the quantized draft checkpoint when ``draft_llm_precision`` is a
    quantization that actually produces a quantized dir (i.e. non-fp16 and
    non-int4_gptq — int4_gptq ships pre-quantized so no intermediate dir
    is created). Otherwise uses the original torch draft checkpoint. The
    ONNX output is copied to ``config.get_draft_onnx_dir()``.
    """
    draft_model_dir = config.get_eagle_draft_checkpoint_dir()

    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="tensorrt_edgellm_draft_export_")
    try:
        _run_export_subprocess(
            draft_model_dir,
            tmp_dir,
            f"Exporting EAGLE draft {config.draft_model_id} via the checkpoint exporter",
            test_logger,
            timeout,
            extra_env=config.get_export_env_vars() or None,
            failure_prefix="tensorrt_edgellm draft export")

        draft_llm_out = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(draft_llm_out):
            pytest.fail(
                f"Draft export did not produce llm/ output in {tmp_dir}")

        _log_dir_contents(draft_llm_out, "Draft", test_logger=test_logger)

        shutil.copytree(draft_llm_out, draft_onnx_dir, dirs_exist_ok=True)
        if test_logger:
            test_logger.info(f"Copied draft/ → {draft_onnx_dir}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_tensorrt_edgellm_mtp_export(config: TestConfig,
                                    test_logger,
                                    timeout: int = 600) -> None:
    """Export MTP base + draft from a single checkpoint via ``--mtp``.

    Calls ``tensorrt_edgellm.scripts.export --mtp`` on the base model checkpoint and
    copies outputs:
      - ``llm/``       → ``config.get_llm_onnx_dir()``
      - ``mtp_draft/`` → ``config.get_draft_onnx_dir()``
      - ``visual/``    → ``config.get_visual_onnx_dir("fp16")`` (VLM only,
                          copied when present)
    """
    model_dir = get_export_model_dir(config)

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)
    for subdir, (target_fn, _) in _OUTPUT_DIR_MAP.items():
        os.makedirs(target_fn(config), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mtp_export_")
    try:
        _run_export_subprocess(
            model_dir,
            tmp_dir,
            f"Exporting MTP {config.model_name} via the checkpoint exporter",
            test_logger,
            timeout,
            extra_args=["--mtp"],
            extra_env=config.get_export_env_vars() or None,
            failure_prefix="MTP export")

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(f"MTP export did not produce llm/ in {tmp_dir}")
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        draft_output = os.path.join(tmp_dir, "mtp_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(f"MTP export did not produce mtp_draft/ in {tmp_dir}")
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

        for subdir, (target_fn, _) in _OUTPUT_DIR_MAP.items():
            _copy_subdir(tmp_dir, subdir, target_fn(config))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Command runner helper
# ---------------------------------------------------------------------------


def run_command_list(commands, label, test_logger, env_vars=None):
    """Execute a list of ``(cmd, timeout)`` tuples, failing on first error."""
    for i, (cmd, timeout) in enumerate(commands):
        result = run_command(cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            pytest.fail(f"{label} step {i+1}/{len(commands)} failed: "
                        f"{result.get('error', 'Unknown error')}")
