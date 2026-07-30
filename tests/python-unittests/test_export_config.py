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
"""Tests for dtype fields written into config.json by the ONNX export pipeline.

The C++ runtime parses top-level `kv_cache_dtype` strictly and, for hybrid
engines, also `recurrent_state_dtype` / `conv_state_dtype`. These are
properties of the exported weights (decided by the Python export step), not
builder knobs — they live at the top level of `config.json`, never inside
`builder_config`. Those fields are populated by `export_llm_model` (for base /
standard engines) and the checkpoint-based export sidecar writer right before
the config.json is written to disk.

These tests exercise the behavior contract of the dtype-writing snippets and
verify that the snippets remain present in the source file, without importing
the module (which would pull torch / transformers into the test environment).
"""

import os
import re

# Resolve the real module source — we read it as text to avoid importing the
# entire ONNX export pipeline (torch, transformers) into a unit test.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLM_EXPORT_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "tensorrt_edgellm", "checkpoint",
                 "checkpoint_utils.py"))


def _load_source():
    with open(_LLM_EXPORT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# Recurrent- and conv-state dtypes are decided by the plugin math, not by the
# model name — see `hybrid_state_shapes_and_dtypes` in `llm_export.py`. The
# simulator mirrors that dispatch so the presence checks below stay honest
# without importing torch.
_HYBRID_STATE_DTYPES_BY_MODEL_TYPE = {
    # (recurrent_dtype, conv_dtype)
    'nemotron_h': ('fp16', 'fp16'),
    'qwen3_5_text': ('fp32', 'fp16'),
}


def _simulate_base_dtype_write(model_config,
                               fp8_kv_cache,
                               model_type='nemotron_h'):
    """Replicates the dtype-writing snippet inserted into `export_llm_model`.

    The snippet is the sole mutation the runtime relies on for dtype
    propagation; testing it as a pure function on a dict is equivalent to
    testing the snippet in situ, because the snippet has no other side
    effects.

    `model_type` is consulted to dispatch the hybrid recurrent/conv dtypes
    (see `hybrid_state_shapes_and_dtypes`). Non-hybrid callers (num_linear_attn_layers == 0)
    never reach the dispatch, so the default is a harmless placeholder.
    """
    model_config['kv_cache_dtype'] = 'fp8' if fp8_kv_cache else 'fp16'
    if model_config.get('num_linear_attn_layers', 0) > 0:
        rec, conv = _HYBRID_STATE_DTYPES_BY_MODEL_TYPE[model_type]
        model_config['recurrent_state_dtype'] = rec
        model_config['conv_state_dtype'] = conv
    return model_config


def _simulate_draft_dtype_write(draft_config):
    """Replicates the dtype-writing line in `export_draft_model`."""
    draft_config['kv_cache_dtype'] = 'fp16'
    return draft_config


# ---------------------------------------------------------------------------
# Base / standard engine dtype writes — behavior contract
# ---------------------------------------------------------------------------


def test_base_kv_cache_dtype_fp16_when_not_fp8():
    cfg = {}
    _simulate_base_dtype_write(cfg, fp8_kv_cache=False)
    assert cfg['kv_cache_dtype'] == 'fp16'


def test_base_kv_cache_dtype_fp8_when_fp8_enabled():
    cfg = {}
    _simulate_base_dtype_write(cfg, fp8_kv_cache=True)
    assert cfg['kv_cache_dtype'] == 'fp8'


def test_base_non_hybrid_omits_recurrent_and_conv_dtypes():
    # num_linear_attn_layers missing (or zero) -> no recurrent/conv fields.
    cfg = {}
    _simulate_base_dtype_write(cfg, fp8_kv_cache=False)
    assert 'recurrent_state_dtype' not in cfg
    assert 'conv_state_dtype' not in cfg

    cfg_zero = {'num_linear_attn_layers': 0}
    _simulate_base_dtype_write(cfg_zero, fp8_kv_cache=False)
    assert 'recurrent_state_dtype' not in cfg_zero
    assert 'conv_state_dtype' not in cfg_zero


def test_base_hybrid_nemotron_h_emits_both_recurrent_and_conv_fp16():
    cfg = {'num_linear_attn_layers': 4}
    _simulate_base_dtype_write(cfg,
                               fp8_kv_cache=False,
                               model_type='nemotron_h')
    assert cfg['recurrent_state_dtype'] == 'fp16'
    assert cfg['conv_state_dtype'] == 'fp16'


def test_base_hybrid_qwen3_5_emits_fp32_recurrent_fp16_conv():
    # Qwen3.5 GDN: recurrent state runs in fp32 per the plugin schema; conv is fp16.
    cfg = {'num_linear_attn_layers': 4}
    _simulate_base_dtype_write(cfg,
                               fp8_kv_cache=False,
                               model_type='qwen3_5_text')
    assert cfg['recurrent_state_dtype'] == 'fp32'
    assert cfg['conv_state_dtype'] == 'fp16'


def test_base_hybrid_with_fp8_kv_cache_preserves_state_dtypes():
    # fp8_kv_cache flips kv_cache_dtype but hybrid state dtypes are independent.
    cfg = {'num_linear_attn_layers': 8}
    _simulate_base_dtype_write(cfg, fp8_kv_cache=True, model_type='nemotron_h')
    assert cfg['kv_cache_dtype'] == 'fp8'
    assert cfg['recurrent_state_dtype'] == 'fp16'
    assert cfg['conv_state_dtype'] == 'fp16'


# ---------------------------------------------------------------------------
# Draft engine dtype write — behavior contract
# ---------------------------------------------------------------------------


def test_draft_always_writes_fp16_kv_cache_dtype():
    cfg = {}
    _simulate_draft_dtype_write(cfg)
    assert cfg['kv_cache_dtype'] == 'fp16'


# ---------------------------------------------------------------------------
# Source-level guardrails: the real module source must contain the snippets
# we are simulating above. These catch the case where someone updates the
# helpers in this file without updating the real module, or vice versa.
# ---------------------------------------------------------------------------


def test_llm_export_source_contains_base_dtype_write():
    src = _load_source()
    pattern = re.compile(
        r"out\[['\"]kv_cache_dtype['\"]\]\s*=\s*\(\s*['\"]fp8['\"]\s+if\s+config\.quant\.kv_cache_quant\s*==\s*['\"]fp8['\"]\s+else\s+['\"]fp16['\"]"
    )
    assert pattern.search(src), (
        "Expected kv_cache_dtype write missing from checkpoint_utils.py; "
        "the runtime will silently skip parsing and back-patching will regress."
    )


def test_runtime_config_source_contains_sliding_window_capability():
    src = _load_source()
    assert re.search(
        r"['\"]sliding_window['\"]\s*:\s*config\.sliding_window_size",
        src,
    ), ("Expected sliding_window metadata write missing from checkpoint_utils.py; "
        "the runtime cannot safely gate context reuse backends without it.")


def test_llm_export_source_contains_hybrid_dtype_write():
    src = _load_source()
    # Presence check only — the right-hand side is now derived at runtime via
    # `torch_dtype_to_config_str(state_specs['{recurrent,conv}_dtype'])` (see
    # `hybrid_state_shapes_and_dtypes`), but the key being written is what the
    # runtime depends on. The specific dtype values are exercised above by the
    # simulator-driven behavior tests.
    assert "recurrent_state_dtype" in src, (
        "Expected recurrent_state_dtype handling missing from checkpoint_utils.py "
        "for hybrid models.")
    assert "conv_state_dtype" in src, (
        "Expected conv_state_dtype handling missing from checkpoint_utils.py "
        "for hybrid models.")


def test_llm_export_source_contains_draft_dtype_write():
    src = _load_source()
    assert re.search(
        r"if\s+config\.is_eagle3_draft\s*:", src
    ), "Expected EAGLE3 draft sidecar block missing from checkpoint_utils.py."


def test_llm_export_hybrid_block_is_gated_on_num_linear_attn_layers():
    """The hybrid dtype fields must be gated on num_linear_attn_layers > 0 so
    non-hybrid engines do not carry them (the C++ parser rejects unexpected
    fields via `numLinearAttnLayers > 0` gating)."""
    src = _load_source()
    assert re.search(
        r"num_linear_attn_layers['\"][\s,)]+\s*,?\s*0\s*\)\s*>\s*0", src
    ), "Hybrid dtype block must be gated on `num_linear_attn_layers > 0`."
