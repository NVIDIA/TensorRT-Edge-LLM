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
"""Quantization smoke tests: actually quantize small models with the configs
built by :func:`build_quant_config`, and assert the resulting quantizers.

Coverage spans the module families the builder reasons about:

* dense LLM (FP8 / NVFP4 / INT4-AWQ / MXFP8 / INT8-SQ), LM-head, KV cache;
* MoE (real Qwen3-MoE) — experts quantized, router left in high precision;
* VLM visual towers under every supported prefix, with mixed precision;
* ASR audio tower;
* Omni Talker (CodePredictor per-channel + excludes, Code2Wav off);
* a full quantize -> ``export_hf_checkpoint`` round-trip.

Everything runs on CPU in a few seconds. Skips when ModelOpt is unavailable.
``quantization_configs`` only imports ``modelopt.torch.quantization``, so it is
loaded directly (bypassing the ``tensorrt_edgellm.quantization`` package
``__init__``, which pulls in torch/transformers/datasets).
"""

import glob
import importlib
import importlib.util
import json
import os
import tempfile
from types import SimpleNamespace

import pytest

pytest.importorskip("modelopt.torch.quantization")
pytest.importorskip("transformers")

import modelopt.torch.quantization as mtq  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "tensorrt_edgellm", "quantization",
                 "quantization_configs.py"))


def _load_build_quant_config():
    spec = importlib.util.spec_from_file_location("_qc_under_test",
                                                  _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_quant_config


build_quant_config = _load_build_quant_config()

_DIM = 16


# --------------------------------------------------------------------------- #
# Introspection helpers
# --------------------------------------------------------------------------- #
def _wq(module):
    """(enabled, num_bits, calibrated) for a Linear's weight quantizer."""
    q = module.weight_quantizer
    bits = tuple(q.num_bits) if hasattr(q.num_bits, "__iter__") else q.num_bits
    return q.is_enabled, bits, q.amax is not None


def _enabled_weight_quantizer_names(model):
    """Names of every enabled + calibrated weight quantizer in ``model``.

    Handles both the standard ``*.weight_quantizer`` and the fused-MoE
    ``*.<name>_weight_quantizers.<expert>`` layouts ModelOpt produces.
    """
    names = set()
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer") or "_weight_quantizers." in name:
            if getattr(module, "is_enabled", False) and getattr(
                    module, "amax", None) is not None:
                names.add(name)
    return names


def _quantized(names, needle):
    return any(needle in n for n in names)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=_DIM,
                      intermediate_size=2 * _DIM,
                      num_hidden_layers=2,
                      num_attention_heads=4,
                      num_key_value_heads=4,
                      vocab_size=64,
                      max_position_embeddings=32)
    cfg.architectures = ["LlamaForCausalLM"]
    return LlamaForCausalLM(cfg).eval()


def _tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=_DIM,
                         intermediate_size=2 * _DIM,
                         moe_intermediate_size=_DIM,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         num_experts=4,
                         num_experts_per_tok=2,
                         decoder_sparse_step=1,
                         vocab_size=64,
                         max_position_embeddings=32)
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    return Qwen3MoeForCausalLM(cfg).eval()


def _tiny_draft_config():
    return SimpleNamespace(hidden_size=_DIM,
                           intermediate_size=2 * _DIM,
                           num_hidden_layers=1,
                           num_attention_heads=4,
                           num_key_value_heads=2,
                           head_dim=4,
                           rms_norm_eps=1e-6,
                           attention_bias=False,
                           bias=False,
                           target_hidden_size=_DIM,
                           vocab_size=64,
                           draft_vocab_size=64,
                           pad_token_id=0,
                           rope_theta=10000.0,
                           model_type="qwen3")


def _calib_ids(model):
    for _ in range(2):
        model(torch.randint(0, 64, (2, 8)))


class _Tower(nn.Module):
    """Stand-in for a visual / audio encoder tower: a couple of Linears."""

    def __init__(self, dim=_DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _CodePredictor(nn.Module):

    def __init__(self, dim=_DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(self.q_proj(x))


class _MultiModalModel(nn.Module):
    """Body Linears + optional towers named exactly as the builder targets:
    a visual tower under ``visual_prefix``, an ``audio_tower``, and an Omni
    ``talker`` with ``code_predictor`` + ``code2wav``."""

    def __init__(self,
                 visual_prefix=None,
                 with_audio=False,
                 with_talker=False):
        super().__init__()
        self._visual_prefix = visual_prefix
        if visual_prefix:
            setattr(self, visual_prefix, _Tower())
        if with_audio:
            self.audio_tower = _Tower()
        self.q_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.o_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.lm_head = nn.Linear(_DIM, _DIM, bias=False)
        if with_talker:
            self.talker = nn.Module()
            self.talker.add_module("code_predictor", _CodePredictor())
            self.talker.add_module("code2wav", _Tower())

    def forward(self, x):
        if self._visual_prefix:
            x = getattr(self, self._visual_prefix)(x)
        if hasattr(self, "audio_tower"):
            x = self.audio_tower(x)
        x = self.o_proj(self.q_proj(x))
        if hasattr(self, "talker"):
            x = self.talker.code_predictor(x)
            x = self.talker.code2wav(x)
        return self.lm_head(x)


class _TinyTokenizer:

    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, texts, **_):
        rows = []
        for text in texts:
            ids = [(ord(ch) % 61) + 1 for ch in text[:8]]
            ids = ids or [1]
            rows.append(ids)
        width = max(len(row) for row in rows)
        padded = [row + [0] * (width - len(row)) for row in rows]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    def save_pretrained(self, output_dir):
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"model_max_length": 8}, f)


class _TinyBaseWithHidden(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5",
                                      mtp_num_hidden_layers=1)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, _DIM)
        self.proj = nn.Linear(_DIM, _DIM, bias=False)
        self.lm_head = nn.Linear(_DIM, 64, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, output_hidden_states=False, **_):
        embeds = self.model.embed_tokens(input_ids)
        hidden = self.proj(embeds)
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return {"hidden_states": (embeds, hidden)}
        return logits


def _tiny_text_dataset():
    yield "edge llm quantization"
    yield "qwen3.5 mtp draft"


def _calib_hidden(model):
    for _ in range(3):
        model(torch.randn(2, _DIM))


# --------------------------------------------------------------------------- #
# Dense LLM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,bits", [
    ("fp8", (4, 3)),
    ("nvfp4", (2, 1)),
    ("mxfp8", (4, 3)),
    ("int8_sq", 8),
])
def test_llm_backbone_precision(method, bits):
    # mxfp8 uses dynamic block scales (no per-tensor amax), so calibration is
    # asserted in the fp8/nvfp4 paths rather than here.
    model = _tiny_llama()
    mtq.quantize(model, build_quant_config(method), forward_loop=_calib_ids)
    en, got, _ = _wq(model.model.layers[0].self_attn.q_proj)
    assert en and got == bits
    assert not _wq(model.lm_head)[0], "lm_head stays high precision by default"


def test_llm_int4_awq_weight_only():
    model = _tiny_llama()
    mtq.quantize(model,
                 build_quant_config("int4_awq"),
                 forward_loop=_calib_ids)
    en, bits, cal = _wq(model.model.layers[0].self_attn.q_proj)
    assert en and cal and bits == 4
    assert not model.model.layers[0].self_attn.q_proj.input_quantizer.is_enabled


def test_llm_lm_head_override():
    model = _tiny_llama()
    mtq.quantize(model,
                 build_quant_config("nvfp4", lm_head_quantization="fp8"),
                 forward_loop=_calib_ids)
    assert _wq(model.model.layers[0].self_attn.q_proj)[1] == (2, 1)
    lm_en, lm_bits, lm_cal = _wq(model.lm_head)
    assert lm_en and lm_cal and lm_bits == (4, 3)


def test_llm_kv_cache_quantized():
    model = _tiny_llama()
    mtq.quantize(model,
                 build_quant_config("fp8", kv_cache_quantization="fp8"),
                 forward_loop=_calib_ids)
    attn = model.model.layers[0].self_attn
    assert attn.k_bmm_quantizer.is_enabled and attn.v_bmm_quantizer.is_enabled
    assert attn.q_bmm_quantizer.is_enabled, "q_bmm kept for attention parity"
    assert attn.q_bmm_quantizer.amax is not None


# --------------------------------------------------------------------------- #
# MoE (Qwen3-MoE family; the Qwen3.5 / A3B models share this structure)
# --------------------------------------------------------------------------- #
def test_moe_experts_quantized_router_preserved():
    model = _tiny_qwen3_moe()
    mtq.quantize(model, build_quant_config("fp8"), forward_loop=_calib_ids)
    names = _enabled_weight_quantizer_names(model)
    assert _quantized(names, "experts"), "expert Linears are quantized"
    assert _quantized(names, "self_attn"), "attention is quantized"
    assert not _quantized(names, "mlp.gate"), "the MoE router stays FP16"


# --------------------------------------------------------------------------- #
# VLM visual towers (every supported family prefix) + mixed precision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("visual_prefix", [
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "mlp1",
    "image_embed",
    "embed_vision",
])
def test_visual_tower_quantized_per_prefix(visual_prefix):
    model = _MultiModalModel(visual_prefix=visual_prefix)
    mtq.quantize(model,
                 build_quant_config("fp8", visual_quantization="fp8"),
                 forward_loop=_calib_hidden)
    assert _wq(getattr(model, visual_prefix).fc1)[0], visual_prefix


def test_vlm_mixed_precision():
    """NVFP4 backbone + FP8 visual tower, on one model."""
    model = _MultiModalModel(visual_prefix="visual")
    mtq.quantize(model,
                 build_quant_config("nvfp4", visual_quantization="fp8"),
                 forward_loop=_calib_hidden)
    assert _wq(model.q_proj)[1] == (2, 1)
    assert _wq(model.visual.fc1)[1] == (4, 3)


def test_visual_off_unless_requested():
    model = _MultiModalModel(visual_prefix="visual")
    mtq.quantize(model, build_quant_config("fp8"), forward_loop=_calib_hidden)
    assert _wq(model.q_proj)[0] and not _wq(model.visual.fc1)[0]


# --------------------------------------------------------------------------- #
# ASR audio tower
# --------------------------------------------------------------------------- #
def test_asr_audio_tower():
    model = _MultiModalModel(with_audio=True)
    mtq.quantize(model,
                 build_quant_config("fp8", audio_quantization="fp8"),
                 forward_loop=_calib_hidden)
    assert _wq(
        model.audio_tower.fc1)[0], "audio tower quantized when requested"

    model_off = _MultiModalModel(with_audio=True)
    mtq.quantize(model_off,
                 build_quant_config("fp8"),
                 forward_loop=_calib_hidden)
    assert not _wq(model_off.audio_tower.fc1)[0], "audio off by default"


# --------------------------------------------------------------------------- #
# Omni: visual + audio + Talker CodePredictor, Code2Wav always off
# --------------------------------------------------------------------------- #
def test_omni_mixed_all_modalities():
    model = _MultiModalModel(visual_prefix="visual",
                             with_audio=True,
                             with_talker=True)
    cfg = build_quant_config("fp8",
                             visual_quantization="fp8",
                             audio_quantization="fp8",
                             cp_quantization="fp8")
    mtq.quantize(model, cfg, forward_loop=_calib_hidden)

    cp = model.talker.code_predictor
    assert _wq(model.visual.fc1)[0], "visual quantized"
    assert _wq(model.audio_tower.fc1)[0], "audio quantized"
    cp_en, _, cp_cal = _wq(cp.q_proj)
    assert cp_en and cp_cal and cp.q_proj.weight_quantizer.axis == 0, \
        "CodePredictor weight is per-channel (axis=0)"
    assert not _wq(cp.down_proj)[0], "CodePredictor down_proj excluded"
    assert not _wq(model.talker.code2wav.fc1)[0], "Code2Wav always off"


def test_code_predictor_quantization():
    """cp_quantization on its own: the CodePredictor is quantized per-channel
    (down_proj excluded, Code2Wav off), and stays untouched when not requested."""
    model = _MultiModalModel(with_talker=True)
    mtq.quantize(model,
                 build_quant_config("fp8", cp_quantization="fp8"),
                 forward_loop=_calib_hidden)
    cp = model.talker.code_predictor
    assert _wq(cp.q_proj)[0] and cp.q_proj.weight_quantizer.axis == 0
    assert not _wq(cp.down_proj)[0]
    assert not _wq(model.talker.code2wav.fc1)[0]

    off = _MultiModalModel(with_talker=True)
    mtq.quantize(off, build_quant_config("fp8"), forward_loop=_calib_hidden)
    assert not _wq(off.talker.code_predictor.q_proj)[0], "cp off by default"


# --------------------------------------------------------------------------- #
# Speculative draft models: standalone draft and Qwen3.5-style base+MTP
# --------------------------------------------------------------------------- #
def test_eagle3_draft_model_quantized():
    draft_mod = importlib.import_module(
        "tensorrt_edgellm.quantization.models.eagle3_draft")
    model = draft_mod.Eagle3DraftModel(_tiny_draft_config()).eval()

    def _calib_draft(draft):
        for _ in range(2):
            input_ids = torch.randint(0, 64, (2, 6))
            target_hidden = torch.randn(2, 6, 3 * _DIM)
            draft_hidden = torch.zeros(2, 6, _DIM)
            draft(input_ids, target_hidden, draft_hidden)

    mtq.quantize(model,
                 build_quant_config("fp8", lm_head_quantization="fp8"),
                 forward_loop=_calib_draft)

    assert _wq(model.layers[0].self_attn.q_proj)[0], "draft attention"
    assert _wq(model.layers[0].mlp.gate_proj)[0], "draft MLP"
    assert _wq(model.lm_head)[0], "draft LM head override"


def test_base_and_mtp_quantized_together(monkeypatch):
    quantize_mod = importlib.import_module(
        "tensorrt_edgellm.quantization.quantize")
    mtp_mod = importlib.import_module(
        "tensorrt_edgellm.quantization.models.mtp_draft")
    base = _TinyBaseWithHidden().eval()
    tokenizer = _TinyTokenizer()
    seen = {}

    def _load_model(_model_dir, _dtype, device):
        base.to(device)
        return base, tokenizer, None

    def _quantize_mtp_from_base(base_model,
                                tokenizer,
                                model_dir,
                                quantization,
                                lm_head_quantization=None,
                                kv_cache_quantization=None,
                                **_):
        assert not hasattr(base_model.proj, "weight_quantizer"), \
            "MTP must calibrate before base quantization mutates the model"
        draft = mtp_mod.MtpDraftModel(_tiny_draft_config()).eval().to(
            base_model.device)
        mtp_mod._share_embed_tokens(base_model, draft)

        def _calib_mtp(mtp_draft):
            ids = tokenizer(["qwen3.5 mtp draft"
                             ])["input_ids"].to(base_model.device)
            with torch.no_grad():
                outputs = base_model(ids, output_hidden_states=True)
            mtp_draft(ids, outputs["hidden_states"][-1])

        mtq.quantize(draft,
                     build_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization),
                     forward_loop=_calib_mtp)
        seen["mtp_quantized"] = _wq(draft.layers[0].self_attn.q_proj)[0]
        return draft

    def _export_hf_checkpoint(model, export_dir, extra_state_dict=None):
        os.makedirs(export_dir, exist_ok=True)
        seen["extra_keys"] = set(extra_state_dict or {})
        with open(os.path.join(export_dir, "hf_quant_config.json"), "w") as f:
            json.dump({"quantization": {"quant_algo": "FP8"}}, f)

    monkeypatch.setattr(quantize_mod, "_load_model", _load_model)
    monkeypatch.setattr(mtp_mod, "quantize_mtp_from_base",
                        _quantize_mtp_from_base)
    monkeypatch.setattr(quantize_mod, "export_hf_checkpoint",
                        _export_hf_checkpoint)

    with tempfile.TemporaryDirectory() as model_dir, \
            tempfile.TemporaryDirectory() as output_dir:
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"model_type": "qwen3_5"}, f)
        quantize_mod.quantize_and_export(
            model_dir,
            output_dir,
            quantization="fp8",
            lm_head_quantization="fp8",
            dtype="fp16",
            device="cuda" if torch.cuda.is_available() else "cpu",
            text_dataset=_tiny_text_dataset,
            num_samples=2,
        )

    assert seen["mtp_quantized"], "MTP draft was quantized"
    assert _wq(base.proj)[0], "base model was quantized"
    assert any(key.startswith("mtp.") for key in seen["extra_keys"])


def _write_index(model_dir, keys):
    with open(os.path.join(model_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in keys}}, f)


def test_resolve_mtp_dir_distinguishes_absent_from_unreadable():
    """Absence of ``mtp.*`` only counts when the tensor names were readable.

    A directory with no safetensors at all is unknowable, not empty — callers
    that patch loading never write one, so skipping there would silently drop
    MTP quantization for them.
    """
    quantize_mod = importlib.import_module(
        "tensorrt_edgellm.quantization.quantize")
    with tempfile.TemporaryDirectory() as with_mtp, \
            tempfile.TemporaryDirectory() as without_mtp, \
            tempfile.TemporaryDirectory() as unreadable:
        _write_index(with_mtp, ["thinker.mtp.layers.0.fc.weight", "a.weight"])
        _write_index(without_mtp, ["thinker.model.layers.0.fc.weight"])

        assert quantize_mod._has_mtp_weights(with_mtp) is True
        assert quantize_mod._has_mtp_weights(without_mtp) is False
        assert quantize_mod._has_mtp_weights(unreadable) is None

        # Implicit: skip only on confirmed absence; proceed when unknowable.
        assert quantize_mod._resolve_mtp_dir(None, with_mtp) == with_mtp
        assert quantize_mod._resolve_mtp_dir(None, without_mtp) is None
        assert quantize_mod._resolve_mtp_dir(None, unreadable) == unreadable

        # Explicit --mtp_draft_dir asserts the weights are there.
        assert quantize_mod._resolve_mtp_dir(with_mtp, without_mtp) == with_mtp
        for bad in (without_mtp, unreadable, "/nonexistent"):
            with pytest.raises(ValueError, match="mtp_draft_dir"):
                quantize_mod._resolve_mtp_dir(bad, with_mtp)


def test_mtp_moe_detected_under_either_expert_key():
    """``num_experts`` is an attribute_map alias for ``num_local_experts`` on
    some HF configs and absent on others, so both spellings must select the
    sparse-MoE block — otherwise a MoE MTP head is silently built as dense.
    """
    mtp_mod = importlib.import_module(
        "tensorrt_edgellm.quantization.models.mtp_draft")
    base = dict(hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                rms_norm_eps=1e-6,
                vocab_size=64,
                attention_bias=False,
                moe_intermediate_size=16,
                shared_expert_intermediate_size=8,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                rope_theta=10000.0,
                max_position_embeddings=128)
    for key in ("num_experts", "num_local_experts"):
        cfg = SimpleNamespace(**{**base, key: 4})
        mlp = mtp_mod.MtpDraftModel(cfg).layers[0].mlp
        assert isinstance(mlp, mtp_mod.MtpSparseMoeBlock), key
        assert mlp.num_experts == 4, key
    dense = mtp_mod.MtpDraftModel(SimpleNamespace(**base)).layers[0].mlp
    assert not isinstance(dense, mtp_mod.MtpSparseMoeBlock)


def test_sub_llm_quantized_detection_covers_awq_gptq():
    """AWQ/GPTQ pack the weight as ``qweight`` and ship no ``weight_scale``.

    Keying only on ``weight_scale`` reads those sub-LLMs as fully FP16 and
    silently drops their quantization on export.
    """
    export_mod = importlib.import_module("tensorrt_edgellm.scripts.export")
    cases = {
        "nvfp4": (["talker.mlp.down_proj.weight_scale"], True),
        "awq": (["talker.mlp.down_proj.qweight"], True),
        "fp16": (["talker.mlp.down_proj.weight"], False),
        "other_submodel": (["thinker.mlp.down_proj.qweight"], False),
    }
    for name, (keys, expected) in cases.items():
        with tempfile.TemporaryDirectory() as d:
            _write_index(d, keys)
            got = export_mod._sub_llm_has_quantized_weights(d, "talker.")
            assert got is expected, f"{name}: expected {expected}, got {got}"


# --------------------------------------------------------------------------- #
# Full quantize -> export round-trip on a real HF model
# --------------------------------------------------------------------------- #
def test_quantize_and_export_hf_checkpoint():
    export_hf_checkpoint = pytest.importorskip(
        "modelopt.torch.export").export_hf_checkpoint
    model = _tiny_llama()
    mtq.quantize(model, build_quant_config("fp8"), forward_loop=_calib_ids)
    with tempfile.TemporaryDirectory() as export_dir:
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=export_dir)
        files = {os.path.basename(p) for p in glob.glob(export_dir + "/*")}
        assert "hf_quant_config.json" in files
        assert any(f.endswith(".safetensors") for f in files)
        with open(os.path.join(export_dir, "hf_quant_config.json")) as f:
            meta = json.load(f)
        assert meta["quantization"]["quant_algo"] == "FP8"
        assert "lm_head" in meta["quantization"]["exclude_modules"]


# --------------------------------------------------------------------------- #
# Contract guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {
        "quantization": "bogus"
    },
    {
        "quantization": "fp8",
        "lm_head_quantization": "bogus"
    },
    {
        "quantization": "fp8",
        "kv_cache_quantization": "int4_awq"
    },
    {
        "quantization": "fp8",
        "visual_quantization": "nvfp4"
    },
    {
        "quantization": "fp8",
        "cp_quantization": "nvfp4"
    },
])
def test_unsupported_methods_raise(kwargs):
    with pytest.raises(ValueError):
        build_quant_config(**kwargs)


# --------------------------------------------------------------------------- #
# GDN qkvzba scale sharing (hybrid NVFP4)
# --------------------------------------------------------------------------- #
class _TinyGdnMixer(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(_DIM, 3 * _DIM, bias=False)
        self.in_proj_z = nn.Linear(_DIM, _DIM, bias=False)
        self.in_proj_b = nn.Linear(_DIM, _DIM, bias=False)
        self.in_proj_a = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, x):
        return (self.in_proj_qkv(x).sum() + self.in_proj_z(x).sum() +
                self.in_proj_b(x).sum() + self.in_proj_a(x).sum())


class _TinyGdnModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_attn = _TinyGdnMixer()

    def forward(self, x):
        return self.linear_attn(x)


def test_fuse_gdn_qkvzba_scales_unifies_group_amax():
    from tensorrt_edgellm.quantization.quantize import _share_gdn_qkvzba_scales
    model = _TinyGdnModel().eval()
    cfg = build_quant_config("nvfp4", fuse_gdn_qkvzba_scales=True)
    mtq.quantize(model, cfg, forward_loop=_calib_hidden)
    mixer = model.linear_attn
    projs = (mixer.in_proj_qkv, mixer.in_proj_z, mixer.in_proj_b,
             mixer.in_proj_a)
    for proj in (mixer.in_proj_b, mixer.in_proj_a):
        en, bits, cal = _wq(proj)
        assert en and cal and bits == (2, 1)
    group_wmax = torch.max(
        torch.stack([p.weight_quantizer.amax for p in projs]))
    assert _share_gdn_qkvzba_scales(model) == 1
    for proj in projs:
        assert torch.equal(proj.weight_quantizer.amax, group_wmax)
        assert torch.equal(proj.input_quantizer.amax,
                           mixer.in_proj_qkv.input_quantizer.amax)


def test_fuse_gdn_qkvzba_scales_requires_nvfp4():
    with pytest.raises(ValueError, match="requires --quantization nvfp4"):
        build_quant_config("fp8", fuse_gdn_qkvzba_scales=True)
