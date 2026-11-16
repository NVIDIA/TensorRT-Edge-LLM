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
"""Standalone EAGLE3 Draft Model for quantization.

EAGLE3 is a speculative-decoding draft architecture that is not part of
HuggingFace ``transformers``.  This module provides a fully self-contained
PyTorch implementation with no dependency on ``transformers`` model classes.

Only the calibration forward path is implemented; the full inference
forward with KV-cache and GatherND belongs in the ONNX export layer.
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional, Union

import modelopt.torch.quantization as mtq
import torch
from safetensors.torch import load_file, safe_open
from torch import nn
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoTokenizer)

from ..datasets import TextDataset, dataset_name, resolve_dataset
from ..quantization_configs import (build_quant_config,
                                    split_lm_head_calibration)
from .attention_scale import resolve_attention_scale
from .layers import (RMSNorm, RotaryEmbedding, SwiGLUMLP, apply_rotary_pos_emb,
                     config_rope_theta, repeat_kv)


class Eagle3DraftAttention(nn.Module):
    """Eagle3 draft attention (input dim = 2 * hidden_size)."""

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


class Eagle3DraftDecoderLayer(nn.Module):
    """Eagle3 draft decoder layer."""

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
        self.self_attn = Eagle3DraftAttention(hidden_size, num_heads,
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


class Eagle3DraftModel(nn.Module):
    """Standalone EAGLE3 draft model for quantization.

    Fully self-contained — uses only basic ``torch.nn`` modules.
    No dependency on ``transformers`` model classes.
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
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         hs,
                                         padding_idx=getattr(
                                             config, "pad_token_id", None))
        self.layers = nn.ModuleList([
            Eagle3DraftDecoderLayer(hs, config.intermediate_size,
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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, hidden_states, hidden_states_from_draft):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(hidden_states) + hidden_states_from_draft
        pos_ids = torch.arange(input_ids.shape[1],
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, pos_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, inputs_embeds)
        return self.lm_head(self.norm(hidden_states[:, -1]))

    @classmethod
    def from_pretrained(cls,
                        draft_model_dir,
                        base_model_dir=None,
                        device="cuda"):
        config = AutoConfig.from_pretrained(draft_model_dir,
                                            trust_remote_code=True)
        if hasattr(config, "text_config"):
            config = config.text_config
        model = cls(config)

        sd = _load_draft_weights(draft_model_dir, device)
        sd = _remap_keys(sd)
        _fill_embedding(sd, base_model_dir, device)
        model.load_state_dict(sd, strict=False)
        return model


def quantize_and_export_draft(
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
    """Load base + draft models, quantize the draft, and export."""
    import shutil
    import time

    from modelopt.torch.export.quant_utils import get_quant_config
    from modelopt.torch.export.unified_export_hf import (
        QUANTIZATION_NONE, _export_quantized_weight, get_quantization_format,
        is_quantlinear, postprocess_state_dict)
    from modelopt.torch.quantization.utils import is_quantized

    from tensorrt_edgellm._safetensors_io import save_file

    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    draft = Eagle3DraftModel.from_pretrained(draft_model_dir, base_model_dir,
                                             device)
    draft = draft.eval().to(device).to(torch_dtype)

    if not is_quantized(draft):
        base, tokenizer = _load_for_draft_calib(base_model_dir, dtype, device)
        quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                       kv_cache_quantization)
        text_ds = resolve_dataset(text_dataset, "text")
        print(f"Draft text calibration: {dataset_name(text_ds)}")
        loader = _draft_text_loader(
            tokenizer,
            text_ds,
            batch_size=16 if "int4" in quantization else 1,
            num_samples=num_samples)

        def _calib(dm):
            for data in tqdm(loader, desc="Calibrating draft"):
                data = data.to(device)
                out = base(input_ids=data, output_hidden_states=True)
                hs = out["hidden_states"]
                idx = [2, (len(hs) - 1) // 2, len(hs) - 4]
                cat_hs = torch.cat([hs[i] for i in idx], dim=-1)
                dm(data, cat_hs, torch.zeros_like(hs[idx[0]]))

        lm_head_cfg = split_lm_head_calibration(quant_cfg,
                                                lm_head_quantization)
        mtq.quantize(draft, quant_cfg, forward_loop=_calib)
        if lm_head_cfg:
            mtq.quantize(draft, lm_head_cfg, forward_loop=_calib)
        mtq.print_quant_summary(draft)

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
    save_file(sd, os.path.join(output_dir, "model.safetensors"))

    src_cfg = os.path.join(draft_model_dir, "config.json")
    if os.path.isfile(src_cfg):
        shutil.copy2(src_cfg, os.path.join(output_dir, "config.json"))
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(qc, f)

    print(f"Saved draft to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir


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


def _resolve_model_dir(model_dir):
    """Resolve a HuggingFace hub ID or local path to a local directory."""
    if os.path.isdir(model_dir):
        return model_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(model_dir)


def _fill_embedding(sd, base_dir, device):
    if "embed_tokens.weight" in sd or not base_dir:
        return
    base_dir = _resolve_model_dir(base_dir)
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


def _is_vlm_model_type(model_dir):
    """Return True if model_dir is a VLM requiring AutoModelForImageTextToText."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
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


def _load_for_draft_calib(model_dir, dtype, device):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    is_vlm = _is_vlm_model_type(model_dir)
    auto_cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    model = auto_cls.from_pretrained(model_dir,
                                     torch_dtype=torch_dtype,
                                     trust_remote_code=True,
                                     low_cpu_mem_usage=True).to(device)
    gc.collect()  # release safetensor mmap handles after GPU transfer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _draft_text_loader(tokenizer, text_dataset, batch_size, num_samples):
    from ..quantize import _text_calib_dataloader
    return _text_calib_dataloader(tokenizer,
                                  text_dataset,
                                  batch_size=batch_size,
                                  num_samples=num_samples)
