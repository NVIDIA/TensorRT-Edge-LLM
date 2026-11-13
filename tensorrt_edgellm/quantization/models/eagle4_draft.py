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
"""Standalone EAGLE-4 draft model for quantization.

An EAGLE-4 drafter is an EAGLE3-style draft with ``fc_norm``, ``norm_output``
or a low-rank candidate head. Only the calibration forward path is
implemented; the inference graph belongs in the ONNX export layer
(``tensorrt_edgellm.models.eagle4``).

The drafter follows the DFlash draft conventions: ``fc`` stays FP16 and is
recorded in ``exclude_modules``, KV-cache quantization is rejected (and
``lm_head`` quantization with a candidate head), and the candidate head factors
pass through unquantized under their checkpoint names, so the ONNX exporter
loads the quantized checkpoint like the original one.
"""

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from safetensors.torch import load_file, safe_open
from torch import nn
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoTokenizer)

from ...checkpoint.loader import iter_checkpoint_keys
from ...config import (_EAGLE_CANDIDATE_HEAD_KEY_PREFIXES,
                       _EAGLE_SKIPPED_CHECKPOINT_KEYS,
                       _eagle_draft_feature_keys, _parse_eagle_draft_features,
                       _parse_eagle_target_layer_ids)
from ..datasets import TextDataset, dataset_name, resolve_dataset
from ..quantization_configs import append_quant_cfg_entries, build_quant_config
from .attention_scale import resolve_attention_scale
from .layers import (RMSNorm, RotaryEmbedding, SwiGLUMLP, apply_rotary_pos_emb,
                     config_rope_theta, repeat_kv)

# Projectors that stay FP16, as in the DFlash draft quantizer.
_FP16_PROJECTORS = ("fc", )
_QUANTIZERS = ("input_quantizer", "weight_quantizer", "output_quantizer")


class Eagle4CalibAttention(nn.Module):
    """Causal GQA attention over the concatenated ``[embeds, hidden]`` input (``2 * hidden_size``)."""

    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 bias,
                 attention_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.attention_scale = (resolve_attention_scale({}, "", head_dim) if
                                attention_scale is None else attention_scale)

        in_dim = hidden_size * 2
        self.q_proj = nn.Linear(in_dim, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(in_dim, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(in_dim, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, x, cos, sin):
        B, L, _ = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.kv_groups)
        v = repeat_kv(v, self.kv_groups)

        w = torch.matmul(q, k.transpose(2, 3)) * self.attention_scale
        if L > 1:
            mask = torch.triu(
                torch.ones(1, 1, L, L, device=x.device, dtype=torch.bool), 1)
            w.masked_fill_(mask, torch.finfo(w.dtype).min)
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().reshape(B, L, -1)
        return self.o_proj(out)


class Eagle4CalibDecoderLayer(nn.Module):
    """Normalises ``hidden_states`` and ``inputs_embeds`` separately and attends over their concatenation."""

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 bias,
                 attention_scale=None):
        super().__init__()
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Eagle4CalibAttention(hidden_size, num_heads,
                                              num_kv_heads, head_dim, bias,
                                              attention_scale)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, cos, sin, inputs_embeds):
        residual = hidden_states
        h = self.hidden_norm(hidden_states)
        e = self.input_layernorm(inputs_embeds)
        h = self.self_attn(torch.cat((e, h), dim=-1), cos, sin)
        h = residual + h
        residual = h
        h = self.mlp(self.post_attention_layernorm(h))
        return residual + h


class Eagle4CalibDraftModel(nn.Module):
    """Standalone EAGLE-4 draft model for quantization.

    Fully self-contained: uses only basic ``torch.nn`` modules, no
    ``transformers`` model classes. ``forward`` returns the decoder output
    before the final norm at every position.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hs = config.hidden_size
        head_dim = getattr(config, "head_dim",
                           hs // config.num_attention_heads)
        bias = getattr(config, "attention_bias", False)
        target_hidden = getattr(config, "target_hidden_size", hs)
        attention_scale = resolve_attention_scale(
            config, getattr(config, "model_type", ""), head_dim)

        self.fc = nn.Linear(target_hidden * 3,
                            hs,
                            bias=getattr(config, "bias", False))
        self.fc_norm = (nn.ModuleList([
            RMSNorm(target_hidden, eps=config.rms_norm_eps) for _ in range(3)
        ]) if getattr(config, "fc_norm", False) else None)
        self.target_hidden_size = target_hidden
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         hs,
                                         padding_idx=getattr(
                                             config, "pad_token_id", None))
        self.layers = nn.ModuleList([
            Eagle4CalibDecoderLayer(hs, config.intermediate_size,
                                    config.num_attention_heads,
                                    config.num_key_value_heads, head_dim,
                                    config.rms_norm_eps, bias, attention_scale)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hs, eps=config.rms_norm_eps)
        draft_vocab = getattr(config, "draft_vocab_size", config.vocab_size)
        self.lm_head = nn.Linear(hs, draft_vocab, bias=False)
        self.register_buffer("d2t", torch.empty(draft_vocab,
                                                dtype=torch.int32))

        self.rotary_emb = RotaryEmbedding(head_dim,
                                          base=config_rope_theta(config))

    def _fuse_target_hidden(self, hidden_states):
        if self.fc_norm is not None:
            chunks = hidden_states.split(self.target_hidden_size, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1)
        return self.fc(hidden_states)

    def forward(self, input_ids, hidden_states, hidden_states_from_draft):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = (self._fuse_target_hidden(hidden_states) +
                         hidden_states_from_draft)
        pos_ids = torch.arange(input_ids.shape[1],
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, pos_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, inputs_embeds)
        return hidden_states

    @classmethod
    def from_pretrained(cls, draft_model_dir, base_model_dir, device):
        """Load a draft checkpoint, filling a missing embedding from the base.

        Rejects a checkpoint that lacks a module's weights (an absent ``d2t``
        is allowed) or holds names no module reads, other than the candidate
        head factors, instead of keeping those modules at their initial
        values.
        """
        config = AutoConfig.from_pretrained(draft_model_dir,
                                            trust_remote_code=True)
        if hasattr(config, "text_config"):
            config = config.text_config
        model = cls(config)

        sd = _load_draft_weights(draft_model_dir, device)
        sd = _remap_keys(sd)
        _fill_embedding(sd, base_model_dir, device)
        loaded = model.load_state_dict(sd, strict=False)
        prefixes = tuple(prefix
                         for prefix, _ in _EAGLE_CANDIDATE_HEAD_KEY_PREFIXES)
        missing = [key for key in loaded.missing_keys if key != "d2t"]
        unexpected = [
            key for key in loaded.unexpected_keys
            if not key.startswith(prefixes)
        ]
        if missing or unexpected:
            raise ValueError(
                f"The draft checkpoint {draft_model_dir} does not match "
                f"its config.json: missing {missing[:8]}, unexpected "
                f"{unexpected[:8]}. Fix the checkpoint's tensor names.")
        return model


@dataclass(frozen=True)
class _Eagle4Draft:
    """What calibration needs from the drafter config."""

    aux_layer_ids: List[int]  # HF hidden-state indices, draft-ID order
    norm_output: bool
    candidate_head: bool


def quantize_and_export_eagle4_draft(
    base_model_dir: str,
    draft_model_dir: str,
    output_dir: str,
    quantization: str = "fp8",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    *,
    text_dataset: Union[str, TextDataset, None] = None,
    num_samples: int = 512,
) -> str:
    """Load base + draft models, quantize the draft, and export.

    ``fc`` stays FP16, KV-cache quantization is rejected, and a candidate head
    also keeps ``lm_head`` FP16.
    """
    import shutil
    import time

    from modelopt.torch.export.quant_utils import get_quant_config
    from modelopt.torch.export.unified_export_hf import (
        QUANTIZATION_NONE, _export_quantized_weight, get_quantization_format,
        is_quantlinear, postprocess_state_dict)

    from tensorrt_edgellm._safetensors_io import save_file

    _check_recipe(_parse_draft(draft_model_dir), quantization,
                  lm_head_quantization, kv_cache_quantization)
    draft_spec = _parse_draft(draft_model_dir,
                              _base_num_hidden_layers(base_model_dir))
    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    draft = Eagle4CalibDraftModel.from_pretrained(draft_model_dir,
                                                  base_model_dir, device)
    draft = draft.eval().to(device).to(torch_dtype)

    base, tokenizer = _load_base_for_calib(base_model_dir, dtype, device,
                                           max(draft_spec.aux_layer_ids) + 1)
    quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                   kv_cache_quantization)
    _disable_quantizers(quant_cfg, _FP16_PROJECTORS)
    from ..quantize import _text_calib_dataloader
    text_ds = resolve_dataset(text_dataset, "text")
    print(f"Draft text calibration: {dataset_name(text_ds)}")
    loader = _text_calib_dataloader(
        tokenizer,
        text_ds,
        batch_size=16 if "int4" in quantization else 1,
        num_samples=num_samples)

    def _calib(dm):
        for data in tqdm(loader, desc="Calibrating draft"):
            data = data.to(device)
            with torch.no_grad():
                out = base(input_ids=data, output_hidden_states=True)
            _calibrate(dm, data, out["hidden_states"], draft_spec)

    mtq.quantize(draft, quant_cfg, forward_loop=_calib)
    mtq.print_quant_summary(draft)
    _check_finite_amax(draft, "EAGLE draft")
    del base

    print(f"Quantization: {time.time() - t0:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        for _, m in draft.named_modules():
            if (get_quantization_format(m) != QUANTIZATION_NONE
                    and is_quantlinear(m)):
                _export_quantized_weight(m, torch_dtype)

    qc = get_quant_config(draft)
    sd = postprocess_state_dict(draft.state_dict(), 0,
                                qc["quantization"]["kv_cache_quant_algo"])
    sd = _export_state(sd, draft_model_dir, draft_spec)
    _record_excluded_modules(qc, _FP16_PROJECTORS)
    save_file(sd, os.path.join(output_dir, "model.safetensors"))

    src_cfg = os.path.join(draft_model_dir, "config.json")
    if os.path.isfile(src_cfg):
        shutil.copy2(src_cfg, os.path.join(output_dir, "config.json"))
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(qc, f)

    print(f"Saved draft to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir


def _base_num_hidden_layers(base_model_dir) -> int:
    config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    return getattr(config, "text_config", config).num_hidden_layers


def _parse_draft(draft_model_dir, base_num_layers=None) -> _Eagle4Draft:
    """Parse the draft config as the ONNX exporter does.

    ``base_num_layers`` range-checks the aux layers against the base.
    """
    config_path = os.path.join(draft_model_dir, "config.json")
    with open(config_path, encoding="utf-8") as config_file:
        llm_dict = json.load(config_file)
    if not _eagle_draft_feature_keys(llm_dict):
        raise ValueError(
            f"{config_path} sets none of fc_norm, norm_output and a low-rank "
            "candidate head, so it is not an EAGLE-4 drafter. Quantize an "
            "EAGLE3 draft with quantize_and_export_draft.")
    aux_layer_ids = _parse_eagle_target_layer_ids(llm_dict, base_num_layers)
    features = _parse_eagle_draft_features(llm_dict, aux_layer_ids)
    if not aux_layer_ids:
        raise ValueError(
            "EAGLE drafts with fc_norm, norm_output or a candidate head must "
            "name the base layers they read: set "
            "eagle_config.eagle_aux_hidden_state_layer_ids, "
            "eagle_aux_hidden_state_layer_ids or target_layer_ids in "
            f"{config_path}.")
    return _Eagle4Draft(aux_layer_ids=aux_layer_ids,
                        norm_output=features.norm_output,
                        candidate_head=features.candidate_count > 0)


def _check_recipe(draft_spec, quantization, lm_head_quantization,
                  kv_cache_quantization):
    if quantization == "nvfp4_a16_gptq":
        raise ValueError(
            "--quantization nvfp4_a16_gptq is not supported for EAGLE drafts "
            "with fc_norm, norm_output or a candidate head: the export cannot "
            "load its weight-only checkpoint yet. For NVFP4-A16, quantize "
            "with --quantization nvfp4 and export it with "
            "tensorrt-edgellm-export --no-quantize-activations.")
    if kv_cache_quantization is not None:
        raise ValueError(
            "EAGLE drafts with fc_norm, norm_output or a candidate head keep "
            "their KV cache in FP16: the exported draft graph has no "
            "quantized KV cache. Drop --kv-cache-quantization.")
    if draft_spec.candidate_head and lm_head_quantization is not None:
        raise ValueError(
            "The low-rank candidate head gathers fp16 lm_head rows, so this "
            "draft keeps lm_head in FP16. Drop --lm-head-quantization.")


def _calibrate(draft, input_ids, base_hidden_states, draft_spec):
    """Run the context pass and one draft-level pass, with runtime pairing.

    Context entry ``p`` pairs the base features of token ``p`` with the
    embedding of token ``p + 1``. A draft level feeds back entry ``p - 1``'s
    hidden state (``norm(h)`` under ``norm_output``) with zero base features,
    so ``fc`` sees zeros there as it does at runtime.
    """
    aux = torch.cat([base_hidden_states[i] for i in draft_spec.aux_layer_ids],
                    dim=-1)
    tokens, aux = input_ids[:, 1:], aux[:, :-1]
    if tokens.shape[1] < 2:
        return
    hidden = draft(tokens, aux,
                   aux.new_zeros(*aux.shape[:-1], draft.config.hidden_size))
    nrm = draft.norm(hidden)
    if not draft_spec.candidate_head:
        # A dense head reads norm(h); this calibrates a requested lm_head recipe.
        draft.lm_head(nrm)
    feedback = nrm if draft_spec.norm_output else hidden
    draft(tokens[:, 1:], torch.zeros_like(aux[:, 1:]), feedback[:, :-1])


def _export_state(state_dict, draft_model_dir, draft_spec):
    """Drop what the exporter never reads and pass the candidate head through.

    The exporter takes the embedding from the base and needs no ``d2t`` when
    the checkpoint has none, so the calibration model's copies are dropped.
    The candidate head factors keep their checkpoint names and precision.
    """
    source_keys = set(iter_checkpoint_keys(draft_model_dir))
    dropped = set(_EAGLE_SKIPPED_CHECKPOINT_KEYS)
    if "d2t" not in source_keys:
        dropped.add("d2t")
    out = {
        key: value
        for key, value in state_dict.items() if key not in dropped
    }
    prefixes = tuple(prefix
                     for prefix, _ in _EAGLE_CANDIDATE_HEAD_KEY_PREFIXES)
    for sf_path in sorted(Path(draft_model_dir).glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefixes) and key not in out:
                    out[key] = f.get_tensor(key)
    if draft_spec.candidate_head and not all(
            any(key.startswith(prefix) for key in out) for prefix in prefixes):
        raise ValueError(
            "The draft config declares a low-rank candidate head, but "
            f"{draft_model_dir} has no safetensors file with its factors. "
            "Convert the draft checkpoint to safetensors and quantize again.")
    return out


def _disable_quantizers(quant_cfg: dict, module_names: Iterable[str]) -> None:
    """Keep the named modules unquantized: their input, weight and output quantizers are disabled."""
    append_quant_cfg_entries(quant_cfg, [{
        "quantizer_name": f"{name}.{quantizer}",
        "enable": False
    } for name in module_names for quantizer in _QUANTIZERS])


def _check_finite_amax(model: nn.Module, label: str) -> None:
    """Reject calibration that observed FP16 overflow (non-finite amax)."""
    bad = []
    for name, module in model.named_modules():
        amax = getattr(module, "amax", None)
        if (isinstance(amax, torch.Tensor)
                and not torch.isfinite(amax.float()).all()):
            bad.append(name)
    if bad:
        raise RuntimeError(f"{label} calibration produced non-finite "
                           f"amax (activation overflow) in: {bad[:8]}")


def _record_excluded_modules(quant_config: dict,
                             module_names: Iterable[str]) -> None:
    """Add ``module_names`` to the ``exclude_modules`` list that the exporters read."""
    section = quant_config.setdefault("quantization", {})
    excluded = list(section.get("exclude_modules") or [])
    excluded += [name for name in module_names if name not in excluded]
    section["exclude_modules"] = excluded


def _load_draft_weights(model_dir, device="cpu"):
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(bin_path):
        return torch.load(bin_path, weights_only=True, map_location=device)
    if os.path.exists(st_path):
        return load_file(st_path, device=device)
    raise FileNotFoundError(f"No weights at {bin_path} or {st_path}")


def _remap_keys(sd):
    out = {}
    for k, v in sd.items():
        if "t2d" in k:
            continue
        if "midlayer" in k:
            k = k.replace("midlayer", "layers.0")
        out[k] = v
    return out


def _fill_embedding(sd, base_dir, device):
    if "embed_tokens.weight" in sd or not base_dir:
        return
    if not os.path.isdir(base_dir):
        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(base_dir)
    candidates = (
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    )
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device=device) as sf:
            for cand in candidates:
                if cand in sf.keys():
                    sd["embed_tokens.weight"] = sf.get_tensor(cand)
                    return
    raise ValueError("embed_tokens.weight not found in base model")


def _is_vlm_model_type(config):
    """Return True if ``config`` needs AutoModelForImageTextToText."""
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return False
    causal_types = {
        str(k)
        for k in AutoModelForCausalLM._model_mapping._model_mapping
    }
    vlm_types = {
        str(k)
        for k in AutoModelForImageTextToText._model_mapping._model_mapping
    }
    return model_type in vlm_types and model_type not in causal_types


def _load_base_for_calib(model_dir, dtype, device, num_layers):
    """Load the calibration base with only its first ``num_layers`` decoder layers.

    HF normalizes the last ``hidden_states`` entry, so keep one layer past the
    deepest index read.
    """
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    auto_cls = (AutoModelForImageTextToText
                if _is_vlm_model_type(config) else AutoModelForCausalLM)
    text_config = getattr(config, "text_config", config)
    if num_layers > text_config.num_hidden_layers:
        raise ValueError(
            f"Draft calibration reads base hidden state {num_layers - 1}, "
            f"but {model_dir} has {text_config.num_hidden_layers} decoder "
            "layers. Check that the draft's aux layer IDs belong to this "
            "base.")
    text_config.num_hidden_layers = num_layers
    if getattr(text_config, "layer_types", None):
        text_config.layer_types = text_config.layer_types[:num_layers]
    model = auto_cls.from_pretrained(model_dir,
                                     torch_dtype=torch_dtype,
                                     trust_remote_code=True,
                                     low_cpu_mem_usage=True,
                                     config=config).to(device)
    gc.collect()  # release safetensor mmap handles after GPU transfer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
