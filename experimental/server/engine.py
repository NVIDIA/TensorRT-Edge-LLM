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
vLLM-style one-line API for TensorRT Edge-LLM.

Pipeline: HuggingFace model ID or local path -> ONNX export -> TensorRT engine
build -> inference.

Example::

    from experimental.server import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-1.7B")
    outputs = llm.generate(
        ["What is the capital of France?"],
        SamplingParams(temperature=0.7, max_tokens=256),
    )
    for output in outputs:
        print(output.text)

    # Or start an OpenAI-compatible server:
    llm.serve(port=8000)
"""

import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from .tool_calling import (ToolConfig, parse_assistant_output,
                           validate_tool_request)
from .tool_chat_template import (ToolChatTemplateFormatter,
                                 needs_tool_chat_template)

logger = logging.getLogger("edgellm.server")

_PLUGIN_LIB_NAME = "libNvInfer_edgellm_plugin.so"
_MAX_LOGIT_BIAS_TOKENS = 1024
_MAX_LOGIT_BIAS_TOKEN_ID = (1 << 31) - 1
_MIN_LOGIT_BIAS = -100.0
_MAX_LOGIT_BIAS = 100.0

_LOGIT_BIAS_SPEC_DECODE_ERROR = (
    "logit_bias is not supported while speculative decoding is enabled; "
    "set disable_spec_decode=true or use a vanilla engine")

_VLM_MODEL_TYPES = frozenset([
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_5",
    "qwen2_5_vl",
    "internvl",
    "internvl_chat",
    "phi4mm",
    "phi4_multimodal",
])

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Sampling parameters (mirrors vLLM's SamplingParams)."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    enable_thinking: bool = False
    disable_spec_decode: bool = False
    num_logprobs: int = 0
    stop: List[str] = field(default_factory=list)
    logit_bias: Dict[int, float] = field(default_factory=dict)


@dataclass
class LogprobEntry:
    """One top-K log-probability entry for a single generated token.

    ``token`` is the piece decoded as UTF-8 with ``errors="replace"`` (a
    byte-level BPE token may be only part of a multi-byte character, so it can
    contain U+FFFD); ``bytes`` carries the raw token bytes losslessly.
    """

    token_id: int
    logprob: float
    token: str
    bytes: List[int]


def _convert_logprobs(raw) -> List[List[LogprobEntry]]:
    """Convert the C++/pybind logprobs (list of list of native LogprobEntry with
    a raw-bytes ``piece``) into engine LogprobEntry dataclasses."""
    return [[
        LogprobEntry(token_id=e.token_id,
                     logprob=e.logprob,
                     token=e.piece.decode("utf-8", "replace"),
                     bytes=list(e.piece)) for e in step
    ] for step in raw]


@dataclass
class CompletionOutput:
    """Output of a single generation request."""

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class StreamDelta:
    """Single delta from a streaming generation."""

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_model_dir(model: str) -> str:
    """Resolve a HuggingFace model ID or local path to a local directory."""
    if os.path.isdir(model):
        return os.path.abspath(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with: "
            "pip install huggingface_hub") from exc
    logger.info("Downloading %s from Hugging Face Hub ...", model)
    return snapshot_download(model)


def _artifacts_dir_for_model(model_dir: str) -> str:
    """Return a deterministic directory for ONNX/engine artifacts.

    Stored under ``<model_dir>/.edgellm/``.  If that is not writable
    (e.g. shared filesystem), falls back to ``~/.cache/edgellm/<hash>/``.
    """
    preferred = os.path.join(model_dir, ".edgellm")
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except OSError:
        digest = hashlib.sha256(
            os.path.abspath(model_dir).encode()).hexdigest()[:12]
        fallback = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "edgellm",
            digest,
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _engine_config_tag(
    max_input_len: int,
    max_batch_size: int,
    max_kv_cache_capacity: int,
) -> str:
    """Return a short tag encoding the engine build config."""
    return f"i{max_input_len}_b{max_batch_size}_kv{max_kv_cache_capacity}"


def _is_vlm(model_dir: str) -> bool:
    """Check if the model is a VLM by reading config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type", "") in _VLM_MODEL_TYPES


def _read_model_type(model_dir: str) -> str:
    """Read model_type from config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return ""
    with open(cfg_path) as f:
        return json.load(f).get("model_type", "")


def _read_vision_config(model_dir: str) -> dict:
    """Read vision_config from config.json for visual builder params."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return json.load(f).get("vision_config", {})


def _ensure_plugin_path() -> None:
    """Set EDGELLM_PLUGIN_PATH if not already set.

    Searches relative to this package and common build locations.
    """
    if os.environ.get("EDGELLM_PLUGIN_PATH"):
        return
    project_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = [
        project_root / "build" / "core",
        project_root / "build" / "lib",
    ]
    for d in search_dirs:
        candidate = d / _PLUGIN_LIB_NAME
        if candidate.is_file():
            os.environ["EDGELLM_PLUGIN_PATH"] = str(candidate)
            return


def _import_runtime():
    """Import the C++ pybind module."""
    _ensure_plugin_path()
    try:
        from tensorrt_edgellm import _edgellm_runtime as _rt
        return _rt
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = []
    if os.environ.get("BUILD_DIR"):
        search_dirs.append(Path(os.environ["BUILD_DIR"]) / "pybind")
    search_dirs.extend([
        project_root / "experimental" / "pybind" / "build",
        project_root / "build" / "pybind",
    ])
    search_dirs.extend(project_root.glob("build/lib.*"))
    for cand_dir in search_dirs:
        if not cand_dir.is_dir():
            continue
        so_files = list(cand_dir.glob("*_edgellm_runtime*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location(
                "_edgellm_runtime", so_files[0])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["tensorrt_edgellm._edgellm_runtime"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Could not import _edgellm_runtime. Build the C++ extension first:\n"
        "  TRT_PACKAGE_DIR=/path/to/tensorrt python experimental/server/setup_pybind.py build_ext --inplace"
    )


def _ensure_export_package() -> None:
    """Ensure the installed checkpoint export package is importable."""
    try:
        import tensorrt_edgellm  # noqa: F401
        return
    except ImportError:
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


def _normalize_logit_bias(
        logit_bias: Optional[Dict[Any, Any]]) -> Dict[int, float]:
    """Validate and normalize an OpenAI-compatible logit_bias map."""
    if logit_bias is None:
        return {}
    if not isinstance(logit_bias, dict):
        raise ValueError(
            "'logit_bias' must be an object mapping token IDs to bias values")
    if len(logit_bias) > _MAX_LOGIT_BIAS_TOKENS:
        raise ValueError(f"'logit_bias' has {len(logit_bias)} entries; max is "
                         f"{_MAX_LOGIT_BIAS_TOKENS}")

    normalized: Dict[int, float] = {}
    for token, bias in logit_bias.items():
        if isinstance(token, bool):
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if isinstance(token, int):
            token_id = token
        elif isinstance(token, str):
            try:
                token_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"'logit_bias' token ID {token!r} is not an integer"
                ) from exc
        else:
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if token_id < 0 or token_id > _MAX_LOGIT_BIAS_TOKEN_ID:
            raise ValueError(
                f"'logit_bias' token ID must be in "
                f"[0, {_MAX_LOGIT_BIAS_TOKEN_ID}], got {token_id}")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be a number")
        try:
            bias_value = float(bias)
        except OverflowError as exc:
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias}") from exc
        if (not math.isfinite(bias_value) or bias_value < _MIN_LOGIT_BIAS
                or bias_value > _MAX_LOGIT_BIAS):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias_value}")
        normalized[token_id] = bias_value
    return normalized


def _validate_logit_bias_spec_decode(logit_bias: Dict[int, float], *,
                                     disable_spec_decode: bool,
                                     has_draft_model: bool) -> None:
    """Reject logit bias unless speculative decoding is absent or explicitly disabled."""
    if logit_bias and has_draft_model and not disable_spec_decode:
        raise ValueError(_LOGIT_BIAS_SPEC_DECODE_ERROR)


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------


class LLM:
    """vLLM-style entry point for TensorRT Edge-LLM inference.

    Three initialization modes (exactly one of ``model``, ``onnx_dir``,
    or ``engine_dir`` must be provided):

    1. **HuggingFace checkpoint** — exports ONNX, builds engine, loads::

           llm = LLM(model="Qwen/Qwen3-1.7B")

    2. **ONNX directory** — builds engine from ONNX, loads::

           llm = LLM(onnx_dir="/path/to/onnx")

    3. **Pre-built engine** — loads directly::

           llm = LLM(engine_dir="/path/to/engine")
           llm = LLM(engine_dir="...", visual_engine_dir="...")

    See :mod:`experimental.server.engine_layout` for the expected
    directory layouts.
    """

    def __init__(
        self,
        model: str = "",
        *,
        onnx_dir: str = "",
        visual_onnx_dir: str = "",
        engine_dir: str = "",
        visual_engine_dir: str = "",
        max_input_len: int = 4096,
        max_batch_size: int = 1,
        max_kv_cache_capacity: int = 8192,
        eagle_engine_dir: str = "",
        draft_top_k: int = 10,
        draft_step: int = 6,
        verify_tree_size: int = 60,
        enable_context_reuse: bool = False,
        context_cache_max_records: int = 1024,
        context_cache_recurrent_snapshot_pool_bytes: int = 134217728,
        context_cache_partial_kv_snapshot_pool_bytes: int = 16777216,
        recurrent_capture_interval: int = 0,
        context_cache_prefill_state_only: bool = False,
    ):
        sources = sum(bool(s) for s in (model, onnx_dir, engine_dir))
        if sources != 1:
            raise ValueError(
                "Exactly one of 'model', 'onnx_dir', or 'engine_dir' "
                "must be provided.")
        if engine_dir and eagle_engine_dir:
            raise ValueError(
                "'engine_dir' already identifies the complete pre-built "
                "engine bundle; do not also provide 'eagle_engine_dir'.")

        self._model_id = (model or os.path.basename(onnx_dir)
                          or os.path.basename(engine_dir))
        self._eagle_engine_dir = eagle_engine_dir
        self._draft_top_k = draft_top_k
        self._draft_step = draft_step
        self._verify_tree_size = verify_tree_size
        self._enable_context_reuse = enable_context_reuse
        self._context_cache_max_records = context_cache_max_records
        self._context_cache_recurrent_snapshot_pool_bytes = (
            context_cache_recurrent_snapshot_pool_bytes)
        self._context_cache_partial_kv_snapshot_pool_bytes = (
            context_cache_partial_kv_snapshot_pool_bytes)
        self._recurrent_capture_interval = recurrent_capture_interval
        self._context_cache_prefill_state_only = context_cache_prefill_state_only
        self._tool_template_formatter: Optional[
            ToolChatTemplateFormatter] = None

        if engine_dir:
            self._init_from_engine(engine_dir, visual_engine_dir)
        elif onnx_dir:
            self._init_from_onnx(
                onnx_dir,
                visual_onnx_dir=visual_onnx_dir,
                max_input_len=max_input_len,
                max_batch_size=max_batch_size,
                max_kv_cache_capacity=max_kv_cache_capacity,
            )
        else:
            self._init_from_model(
                model,
                max_input_len=max_input_len,
                max_batch_size=max_batch_size,
                max_kv_cache_capacity=max_kv_cache_capacity,
            )

        self._load_runtime()

    # ------------------------------------------------------------------
    # Initialization paths
    # ------------------------------------------------------------------

    def _init_from_engine(self, engine_dir: str,
                          visual_engine_dir: str) -> None:
        """Load from pre-built engine directories (no export, no build)."""
        from .engine_layout import (EngineType, detect_engine_type,
                                    find_visual_engine_dir,
                                    validate_llm_engine_dir,
                                    validate_spec_decode_engine_dir,
                                    validate_visual_engine_dir)

        engine_type = detect_engine_type(engine_dir)
        if engine_type == EngineType.SPEC_DECODE:
            if not validate_spec_decode_engine_dir(engine_dir):
                raise ValueError(
                    f"spec_base.engine/spec_draft.engine not found in: {engine_dir}"
                )
            if visual_engine_dir:
                logger.warning(
                    "visual_engine_dir=%r is ignored for spec-decode engines",
                    visual_engine_dir)
            # Spec-decode dir (spec_base.engine + spec_draft.engine): route through
            # the spec-decode path by promoting engine_dir to eagle_engine_dir.
            self._engine_dir = engine_dir
            self._model_dir = engine_dir
            self._eagle_engine_dir = engine_dir
            self._visual_engine_dir = ""
            self._is_vlm = False
            logger.info("Using pre-built spec-decode engine: %s",
                        self._engine_dir)
            return
        if not validate_llm_engine_dir(engine_dir):
            raise ValueError(f"llm.engine not found in: {engine_dir}")
        self._engine_dir = engine_dir
        self._model_dir = engine_dir
        self._is_vlm = False

        if visual_engine_dir:
            if not validate_visual_engine_dir(visual_engine_dir):
                raise ValueError(
                    f"visual.engine not found in: {visual_engine_dir}")
            self._visual_engine_dir = visual_engine_dir
            self._is_vlm = True
        else:
            auto = find_visual_engine_dir(engine_dir)
            self._visual_engine_dir = auto or ""
            if auto:
                self._is_vlm = True
                logger.info("Auto-detected visual engine: %s", auto)

        logger.info("Using pre-built engine: %s", self._engine_dir)

    def _init_from_onnx(
        self,
        onnx_dir: str,
        *,
        visual_onnx_dir: str,
        max_input_len: int,
        max_batch_size: int,
        max_kv_cache_capacity: int,
    ) -> None:
        """Build engine from ONNX directories (no export)."""
        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity
        self._onnx_dir = onnx_dir
        self._visual_onnx_dir = visual_onnx_dir
        self._model_dir = onnx_dir
        self._is_vlm = bool(visual_onnx_dir)

        cfg_tag = _engine_config_tag(max_input_len, max_batch_size,
                                     max_kv_cache_capacity)
        artifacts = _artifacts_dir_for_model(onnx_dir)

        spec_decode_engine_dir = self._eagle_engine_dir
        self._engine_dir = (spec_decode_engine_dir
                            if spec_decode_engine_dir else os.path.join(
                                artifacts, "engine", cfg_tag, "llm"))
        if not spec_decode_engine_dir and not os.path.exists(
                os.path.join(self._engine_dir, "llm.engine")):
            self._build_engine()
        else:
            logger.info("Using cached engine: %s", self._engine_dir)

        self._visual_engine_dir = ""
        if self._is_vlm:
            self._visual_engine_dir = os.path.join(artifacts, "engine",
                                                   cfg_tag, "visual")
            if not os.path.exists(
                    os.path.join(self._visual_engine_dir, "visual.engine")):
                self._build_visual_engine()
            else:
                logger.info("Using cached visual engine: %s",
                            self._visual_engine_dir)

    def _init_from_model(
        self,
        model: str,
        *,
        max_input_len: int,
        max_batch_size: int,
        max_kv_cache_capacity: int,
    ) -> None:
        """Export ONNX + build engine from HuggingFace checkpoint."""
        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity

        logger.info("Resolving model: %s", model)
        self._model_dir = _resolve_model_dir(model)
        artifacts = _artifacts_dir_for_model(self._model_dir)
        self._is_vlm = _is_vlm(self._model_dir)
        self._model_type = _read_model_type(self._model_dir)
        if self._is_vlm:
            logger.info("Detected VLM model (type=%s)", self._model_type)

        self._onnx_dir = os.path.join(artifacts, "onnx", "llm")
        if not os.path.exists(os.path.join(self._onnx_dir, "model.onnx")):
            self._export_onnx()
        else:
            logger.info("Using cached ONNX: %s", self._onnx_dir)

        self._visual_onnx_dir = ""
        if self._is_vlm:
            self._visual_onnx_dir = os.path.join(artifacts, "onnx", "visual")
            if not os.path.exists(
                    os.path.join(self._visual_onnx_dir, "model.onnx")):
                self._export_visual_onnx()
            else:
                logger.info("Using cached visual ONNX: %s",
                            self._visual_onnx_dir)

        # Delegate to _init_from_onnx for the build step
        self._init_from_onnx(
            self._onnx_dir,
            visual_onnx_dir=self._visual_onnx_dir,
            max_input_len=max_input_len,
            max_batch_size=max_batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
        )

    def _load_runtime(self) -> None:
        """Load the C++ runtime from engine directories."""
        self._rt = _import_runtime()
        logger.info("Loading TensorRT engine from %s ...", self._engine_dir)
        if self._visual_engine_dir:
            logger.info("Loading visual engine from %s ...",
                        self._visual_engine_dir)
        ctx_cache = self._rt.ContextCacheConfig()
        ctx_cache.enabled = self._enable_context_reuse
        if self._enable_context_reuse:
            ctx_cache.max_records = self._context_cache_max_records
            ctx_cache.recurrent_snapshot_pool_bytes = (
                self._context_cache_recurrent_snapshot_pool_bytes)
            ctx_cache.partial_kv_snapshot_pool_bytes = (
                self._context_cache_partial_kv_snapshot_pool_bytes)
            logger.info(
                "Context reuse (KV cache reuse) ENABLED "
                "(max_records=%d, recurrent_pool=%d B, partial_kv_pool=%d B)",
                self._context_cache_max_records,
                self._context_cache_recurrent_snapshot_pool_bytes,
                self._context_cache_partial_kv_snapshot_pool_bytes,
            )
        spec_decode_engine_dir = self._eagle_engine_dir
        if spec_decode_engine_dir:
            logger.info(
                "Speculative decoding enabled (top_k=%d, step=%d, tree=%d)",
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
            )
            self._runtime = self._rt.LLMRuntime(
                self._engine_dir,
                self._visual_engine_dir,
                {},
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
                ctx_cache,
            )
        else:
            self._runtime = self._rt.LLMRuntime(
                self._engine_dir,
                self._visual_engine_dir,
                {},
                ctx_cache,
            )
        self._runtime.capture_decoding_cuda_graph()
        # Metrics (LLMPrefillMetrics.recordRun) only accumulate when profiling
        # is enabled; it is off by default. Turn it on so per-request token
        # cache reuse counters are populated.
        self._rt.set_profiling_enabled(True)
        # KV/token cache reuse accounting. get_prefill_metrics() returns
        # server-lifetime cumulative counters; we snapshot the previous values
        # to derive per-request deltas. The lock keeps the delta exact under
        # the single-stream (concurrency 1) MLPerf load; with concurrent
        # in-flight requests the per-request split is only approximate.
        self._cache_metric_lock = threading.Lock()
        self._prev_reused_tokens = 0
        self._prev_computed_tokens = 0
        logger.info("Engine loaded and ready.")

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _export_onnx(self) -> None:
        """Export the model checkpoint to ONNX via tensorrt_edgellm."""
        logger.info("Exporting ONNX to %s ...", self._onnx_dir)
        os.makedirs(self._onnx_dir, exist_ok=True)

        _ensure_export_package()
        from tensorrt_edgellm import AutoModel, export_onnx

        model = AutoModel.from_pretrained(self._model_dir, device="cpu")
        output_path = os.path.join(self._onnx_dir, "model.onnx")
        export_onnx(model, output_path, model_dir=self._model_dir)

        # Patch image_token_id for VLM models
        if self._is_vlm:
            _ensure_export_package()
            from tensorrt_edgellm.scripts.export import _find_token_id
            image_token_id = _find_token_id(self._model_dir, "<|image_pad|>")
            if image_token_id is not None:
                cfg_path = os.path.join(self._onnx_dir, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    cfg["image_token_id"] = image_token_id
                    with open(cfg_path, "w") as f:
                        json.dump(cfg, f, indent=2)
                    logger.info(
                        "Patched image_token_id=%d into LLM config",
                        image_token_id,
                    )

        logger.info("ONNX export complete: %s", output_path)

    def _export_visual_onnx(self) -> None:
        """Export the visual encoder to ONNX via tensorrt_edgellm."""
        logger.info(
            "Exporting visual ONNX to %s ...",
            self._visual_onnx_dir,
        )
        os.makedirs(self._visual_onnx_dir, exist_ok=True)

        import torch

        _ensure_export_package()
        from tensorrt_edgellm.scripts.export import (_export_visual,
                                                     _load_all_weights,
                                                     _load_config)

        config = _load_config(self._model_dir)
        weights = _load_all_weights(self._model_dir)
        _export_visual(
            self._model_dir,
            self._visual_onnx_dir,
            weights,
            config,
            self._model_type,
            torch.float16,
        )
        logger.info(
            "Visual ONNX export complete: %s",
            self._visual_onnx_dir,
        )

    def _build_engine(self) -> None:
        """Build a TensorRT engine from the ONNX directory."""
        logger.info(
            "Building TensorRT engine: %s -> %s",
            self._onnx_dir,
            self._engine_dir,
        )
        os.makedirs(self._engine_dir, exist_ok=True)

        rt = _import_runtime()
        config = rt.LLMBuilderConfig()
        config.max_input_len = self._max_input_len
        config.max_batch_size = self._max_batch_size
        config.max_kv_cache_capacity = self._max_kv_cache_capacity

        builder = rt.LLMBuilder(self._onnx_dir, self._engine_dir, config)
        if not builder.build():
            raise RuntimeError(
                f"TensorRT engine build failed. "
                f"ONNX dir: {self._onnx_dir}, engine dir: {self._engine_dir}")
        logger.info("Engine build complete: %s", self._engine_dir)

    def _build_visual_engine(self) -> None:
        """Build a TensorRT engine for the visual encoder."""
        logger.info(
            "Building visual TensorRT engine: %s -> %s",
            self._visual_onnx_dir,
            self._visual_engine_dir,
        )
        os.makedirs(self._visual_engine_dir, exist_ok=True)

        rt = _import_runtime()
        config = rt.VisualBuilderConfig()

        # Derive image token counts from vision_config
        vis_cfg = _read_vision_config(self._model_dir)
        image_size = vis_cfg.get("image_size", 448)
        patch_size = vis_cfg.get("patch_size", 14)
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        tokens_per_tile = (image_size // patch_size)**2
        # Round up to nearest multiple of tokens_per_tile
        config.min_image_tokens = tokens_per_tile
        config.max_image_tokens = tokens_per_tile * 4
        config.max_image_tokens_per_image = tokens_per_tile * 2

        builder = rt.VisualBuilder(
            self._visual_onnx_dir,
            self._visual_engine_dir,
            config,
        )
        if not builder.build():
            raise RuntimeError(f"Visual TensorRT engine build failed. "
                               f"ONNX dir: {self._visual_onnx_dir}, "
                               f"engine dir: {self._visual_engine_dir}")
        logger.info(
            "Visual engine build complete: %s",
            self._visual_engine_dir,
        )

    def _tool_template_dirs(self) -> List[str]:
        dirs = [self._model_dir, self._engine_dir]
        if hasattr(self, "_onnx_dir"):
            dirs.append(self._onnx_dir)
        return dirs

    def _get_tool_template_formatter(self) -> ToolChatTemplateFormatter:
        if self._tool_template_formatter is None:
            self._tool_template_formatter = ToolChatTemplateFormatter(
                self._tool_template_dirs())
        return self._tool_template_formatter

    def _tool_choice_for_template(
            self, tool_config: ToolConfig) -> Union[str, Dict[str, Any]]:
        if tool_config.forced_name:
            return {
                "type": "function",
                "function": {
                    "name": tool_config.forced_name
                },
            }
        return tool_config.tool_choice

    def _prepare_messages_for_runtime(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        enable_thinking: bool = False,
    ):
        """Prepare messages for the C++ runtime."""
        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        template_tools = (tool_config.tools
                          if tool_config.tool_choice != "none" else [])
        image_buffers = _load_image_buffers(self._rt, messages)

        if needs_tool_chat_template(messages, template_tools,
                                    tool_config.tool_choice):
            template_tool_choice = None
            if tool_config.tool_choice != "none":
                template_tool_choice = self._tool_choice_for_template(
                    tool_config)
            formatter = self._get_tool_template_formatter()
            replay_tail_length = 0
            if (self._enable_context_reuse and self.has_draft_model
                    and self._context_cache_prefill_state_only):
                prompt, replay_tail_length = formatter.format_with_replay_tail(
                    messages,
                    tools=template_tools,
                    tool_choice=template_tool_choice,
                    enable_thinking=enable_thinking,
                )
            else:
                prompt = formatter.format(
                    messages,
                    tools=template_tools,
                    tool_choice=template_tool_choice,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            cpp_messages = _convert_messages_to_cpp(
                self._rt,
                [{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return (cpp_messages, image_buffers, False, False,
                    replay_tail_length)

        cpp_messages = _convert_messages_to_cpp(self._rt, messages)
        return cpp_messages, image_buffers, True, True, 0

    def _make_generation_request(
        self,
        messages: List[Dict[str, Any]],
        params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        stream_channel: Optional[Any] = None,
        raw_prompt: Optional[str] = None,
    ):
        normalized_logit_bias = _normalize_logit_bias(params.logit_bias)
        _validate_logit_bias_spec_decode(
            normalized_logit_bias,
            disable_spec_decode=params.disable_spec_decode,
            has_draft_model=self.has_draft_model,
        )

        if raw_prompt is not None:
            # Raw-prompt path (/v1/completions): the caller supplies the
            # already-formatted prompt; feed it verbatim with no chat template
            # so both engines can be driven with an identical pre-rendered
            # prompt for apples-to-apples comparison.
            request = self._rt.LLMGenerationRequest()
            req = self._rt.Request(messages=_convert_messages_to_cpp(
                self._rt, [{
                    "role": "user",
                    "content": raw_prompt
                }]))
            req.stop_strings = params.stop
            req.logit_bias = normalized_logit_bias
            request.requests = [req]
            if stream_channel is not None:
                request.stream_channels = [stream_channel]
            request.temperature = params.temperature
            request.top_p = params.top_p
            request.top_k = params.top_k
            request.max_generate_length = params.max_tokens
            request.apply_chat_template = False
            request.add_generation_prompt = False
            request.enable_thinking = params.enable_thinking
            request.disable_spec_decode = params.disable_spec_decode
            request.num_logprobs = params.num_logprobs
            if (self._enable_context_reuse
                    and self._recurrent_capture_interval > 0):
                request.recurrent_capture_interval = (
                    self._recurrent_capture_interval)
            if (self._enable_context_reuse
                    and self._context_cache_prefill_state_only):
                request.context_cache_commit_policy = (
                    self._rt.ContextCacheCommitPolicy.PREFILL_STATE_ONLY)
            return request

        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        (cpp_messages, image_buffers, apply_template, add_prompt,
         replay_tail_length) = (self._prepare_messages_for_runtime(
             messages,
             tools=tool_config.tools,
             tool_choice=tool_config.tool_choice,
             tool_config=tool_config,
             enable_thinking=params.enable_thinking,
         ))

        audio_buffers = _load_audio_buffers(self._rt, messages)

        request = self._rt.LLMGenerationRequest()
        req = self._rt.Request(messages=cpp_messages)
        req.image_buffers = image_buffers
        req.audio_buffers = audio_buffers
        req.stop_strings = params.stop
        req.logit_bias = normalized_logit_bias
        request.requests = [req]
        if stream_channel is not None:
            request.stream_channels = [stream_channel]
        request.temperature = params.temperature
        request.top_p = params.top_p
        request.top_k = params.top_k
        request.max_generate_length = params.max_tokens
        request.apply_chat_template = apply_template
        request.add_generation_prompt = add_prompt
        request.enable_thinking = params.enable_thinking
        request.disable_spec_decode = params.disable_spec_decode
        request.num_logprobs = params.num_logprobs
        request.context_cache_replay_tail_length = replay_tail_length
        if (self._enable_context_reuse
                and self._recurrent_capture_interval > 0):
            # Periodically snapshot recurrent (GDN) state during prefill so a
            # later request sharing this prefix can reuse it instead of
            # recomputing. Required for hybrid (linear-attn) context reuse.
            request.recurrent_capture_interval = self._recurrent_capture_interval
        if self._enable_context_reuse and self._context_cache_prefill_state_only:
            request.context_cache_commit_policy = (
                self._rt.ContextCacheCommitPolicy.PREFILL_STATE_ONLY)
        return request

    def _parse_generation_output(
        self,
        text: str,
        token_ids: List[int],
        finish_reason: Optional[str],
        tool_config: ToolConfig,
    ) -> CompletionOutput:
        if not tool_config.parse_output:
            return CompletionOutput(text=text,
                                    token_ids=token_ids,
                                    finish_reason=finish_reason)

        parsed = parse_assistant_output(text, tool_config, self._model_dir)
        tool_calls = [call.to_openai() for call in parsed.tool_calls]
        return CompletionOutput(
            text=parsed.content,
            token_ids=token_ids,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            tool_calls=tool_calls,
            reasoning=parsed.reasoning or None,
        )

    # ------------------------------------------------------------------
    # Inference API (vLLM-style)
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_rate(reused: int, computed: int) -> float:
        total = reused + computed
        return reused / total if total > 0 else 0.0

    def _log_prefill_cache_hit_rate(self) -> None:
        """Log token-cache reuse for the request that just finished plus the
        server-lifetime cumulative total.

        ``get_prefill_metrics()`` accumulates ``reused_tokens`` /
        ``computed_tokens`` across the whole server run, so the per-request
        figure is the delta since the previous request.
        """
        with self._cache_metric_lock:
            pm = self._runtime.get_prefill_metrics()
            cum_reused = int(pm.reused_tokens)
            cum_computed = int(pm.computed_tokens)
            req_reused = cum_reused - self._prev_reused_tokens
            req_computed = cum_computed - self._prev_computed_tokens
            self._prev_reused_tokens = cum_reused
            self._prev_computed_tokens = cum_computed
        logger.info(
            "[token-cache] request: reused=%d computed=%d hit_rate=%.4f | "
            "cumulative: reused=%d computed=%d hit_rate=%.4f",
            req_reused,
            req_computed,
            self._hit_rate(req_reused, req_computed),
            cum_reused,
            cum_computed,
            self._hit_rate(cum_reused, cum_computed),
        )

    def generate(
        self,
        prompts: Union[str, List[str], List[List[Dict[str, Any]]]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> List[CompletionOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string, a list of prompt strings, or
                a list of OpenAI-style message lists.
            sampling_params: Sampling configuration. Defaults to
                ``SamplingParams()``.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            List of ``CompletionOutput`` objects, one per prompt.
        """
        params = sampling_params or SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]
        message_batches = []
        for p in prompts:
            if isinstance(p, str):
                message_batches.append([{"role": "user", "content": p}])
            elif isinstance(p, list):
                message_batches.append(p)
            else:
                raise TypeError(f"Unsupported prompt type: {type(p)}")

        outputs = []
        for messages in message_batches:
            tool_config = validate_tool_request(messages, tools, tool_choice)
            request = self._make_generation_request(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                tool_config=tool_config,
            )

            response = self._runtime.handle_request(request)
            self._log_prefill_cache_hit_rate()
            text = response.output_texts[0] if response.output_texts else ""
            ids = response.output_ids[0] if response.output_ids else []
            reason = finish_reason_name(self._rt, response.finish_reasons[0]) \
                if response.finish_reasons else "stop"
            lps = _convert_logprobs(response.logprobs[0]) if (
                params.num_logprobs > 0 and response.logprobs) else []
            out = self._parse_generation_output(text, ids, reason, tool_config)
            out.logprobs = lps
            outputs.append(out)

        return outputs

    def generate_raw(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> CompletionOutput:
        """Generate from an already-formatted prompt (no chat template).

        Used by ``/v1/completions`` so both engines can be driven with an
        identical pre-rendered prompt.
        """
        params = sampling_params or SamplingParams()
        request = self._make_generation_request([], params, raw_prompt=prompt)
        response = self._runtime.handle_request(request)
        self._log_prefill_cache_hit_rate()
        text = response.output_texts[0] if response.output_texts else ""
        ids = list(response.output_ids[0]) if response.output_ids else []
        reason = finish_reason_name(self._rt, response.finish_reasons[0]) \
            if response.finish_reasons else "stop"
        return CompletionOutput(text=text, token_ids=ids, finish_reason=reason)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> CompletionOutput:
        """Single-turn chat completion (convenience wrapper).

        Args:
            messages: OpenAI-style message list.
            sampling_params: Sampling configuration.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            A single ``CompletionOutput``.
        """
        return self.generate([messages],
                             sampling_params,
                             tools=tools,
                             tool_choice=tool_choice)[0]

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Stream generation deltas for a single message list.

        Runs ``handleRequest`` in a background thread with a
        ``StreamChannel`` attached, yielding ``StreamDelta`` objects as
        tokens are produced.
        """
        params = sampling_params or SamplingParams()

        channel = self._rt.StreamChannel.create()
        channel.set_skip_special_tokens(True)

        request = self._make_generation_request(
            messages,
            params,
            tools=tools,
            tool_choice=tool_choice,
            stream_channel=channel,
        )

        error_holder = [None]

        def _run():
            try:
                self._runtime.handle_request(request)
            except Exception as exc:
                error_holder[0] = exc
                channel.cancel()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        try:
            while True:
                chunk = channel.wait_pop(timeout_ms=200)
                if chunk is None:
                    if channel.is_finished() or channel.is_cancelled():
                        break
                    continue
                reason = finish_reason_name(
                    self._rt, chunk.reason) if chunk.finished else None
                yield StreamDelta(
                    text=chunk.text,
                    token_ids=list(chunk.token_ids),
                    finished=chunk.finished,
                    finish_reason=reason,
                    logprobs=_convert_logprobs(chunk.logprobs),
                )
                if chunk.finished:
                    break
        finally:
            worker.join(timeout=5.0)

        if error_holder[0] is not None:
            raise error_holder[0]

        self._log_prefill_cache_hit_rate()

    # ------------------------------------------------------------------
    # Server API
    # ------------------------------------------------------------------

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start an OpenAI-compatible HTTP server.

        Args:
            host: Bind address.
            port: Bind port.
        """
        from .api_server import run_server

        run_server(self, host=host, port=port)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> str:
        """Path to the resolved model checkpoint."""
        return self._model_dir

    @property
    def engine_dir(self) -> str:
        """Path to the TensorRT engine directory."""
        return self._engine_dir

    @property
    def has_draft_model(self) -> bool:
        """Whether Eagle speculative decoding is active."""
        return self._runtime.has_draft_model()


# ---------------------------------------------------------------------------
# Message conversion & image loading
# ---------------------------------------------------------------------------


def finish_reason_name(rt_module, reason) -> Optional[str]:
    """Map a C++ FinishReason enum value to its OpenAI-compatible string.

    NOT_FINISHED maps to None — reaching this function with a non-terminal
    reason indicates a bug; surfacing None instead of silently returning "stop"
    makes it visible. The fallback "stop" catches truly-unknown enum values
    (e.g. future C++ enum additions). STOP_WORDS and END_ID both map to "stop"
    since OpenAI does not distinguish them.
    """
    return {
        rt_module.FinishReason.NOT_FINISHED: None,
        rt_module.FinishReason.END_ID: "stop",
        rt_module.FinishReason.LENGTH: "length",
        rt_module.FinishReason.CANCELLED: "cancelled",
        rt_module.FinishReason.ERROR: "error",
        rt_module.FinishReason.STOP_WORDS: "stop",
    }.get(reason, "stop")


def _convert_messages_to_cpp(rt_module, messages: List[Dict[str, Any]]):
    """Convert Python message dicts to C++ Message objects."""
    cpp_messages = []
    for msg in messages:
        cpp_msg = rt_module.Message()
        cpp_msg.role = msg["role"]
        content = msg["content"]
        contents_list = []
        if isinstance(content, str):
            contents_list.append(rt_module.MessageContent("text", content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    contents_list.append(rt_module.MessageContent(
                        "text", item))
                elif isinstance(item, dict):
                    ct = item.get("type", "text")
                    if ct == "text":
                        contents_list.append(
                            rt_module.MessageContent(
                                "text",
                                item.get("text", ""),
                            ))
                    elif ct == "image":
                        contents_list.append(
                            rt_module.MessageContent(
                                "image",
                                item.get("image", ""),
                            ))
                    elif ct in ("audio", "input_audio", "audio_url"):
                        # Audio bytes are decoded out-of-band by
                        # `_load_audio_buffers`; the chat template just emits
                        # an opaque audio placeholder here. The per-model
                        # audio runner expands that into model-specific
                        # special tokens (Qwen3: <|audio_start|> +
                        # N×<|audio_pad|> + <|audio_end|>; Nemotron-Omni:
                        # N×<so_embedding>).
                        contents_list.append(
                            rt_module.MessageContent("audio", ""))
                    else:
                        raise ValueError(f"Unsupported content type: {ct}")
        cpp_msg.contents = contents_list
        cpp_messages.append(cpp_msg)
    return cpp_messages


def _load_image_buffers(rt_module, messages: List[Dict[str, Any]]):
    """Load image files referenced in messages into ImageData buffers."""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                path = item.get("image", "")
                if path and os.path.isfile(path):
                    images.append(rt_module.load_image_from_path(path))
    return images


def _load_audio_buffers(rt_module, messages: List[Dict[str, Any]]):
    """Load audio content from messages into AudioData buffers.

    Returns an empty list when no audio is present, keeping the byte-identical
    fast path for text-only and image-only requests.
    """
    from .audio_preprocess import load_audio_buffers
    return load_audio_buffers(rt_module, messages)
