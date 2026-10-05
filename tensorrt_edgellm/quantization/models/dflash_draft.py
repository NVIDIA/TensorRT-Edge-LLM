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
"""Standalone DFlash Draft Model for quantization.

DFlash draft is a speculative-decoding draft architecture with a separate
checkpoint directory. This module provides a fully self-contained PyTorch
implementation for ModelOpt calibration — no TensorRT custom op stubs.

Architecture::

    target_hidden_concat [B, L, Nl*H]
        -> fc(Nl*H, H)
        -> hidden_norm
        -> h_delta [B, L, H]

    For each decoder layer:
        target path: k_proj(h_delta), v_proj(h_delta)  -> target K/V
        proposal path: q/k/v/o_proj(proposal_hidden)   -> self-attention
        + MLP

    -> norm -> lm_head -> logits

The calibration forward drives all Linear modules with realistic
activations produced from the unquantized base model's hidden states.
"""

import gc
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import modelopt.torch.quantization as mtq
import torch
from safetensors.torch import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorrt_edgellm._safetensors_io import save_file

from ..datasets import TextDataset, dataset_name, resolve_dataset
from ..quantization_configs import append_quant_cfg_entries, build_quant_config
from .attention_scale import resolve_attention_scale
from .layers import (RMSNorm, RotaryEmbedding, SwiGLUMLP, apply_rotary_pos_emb,
                     repeat_kv, rotate_half)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DFlash draft calibration building blocks
# ---------------------------------------------------------------------------


class DFlashCalibAttention(nn.Module):
    """DFlash draft attention for calibration (no KV cache, no plugins)."""

    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 attention_scale=None,
                 is_causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.attention_scale = (resolve_attention_scale({}, "", head_dim) if
                                attention_scale is None else attention_scale)
        self.is_causal = bool(is_causal)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size,
                                num_kv_heads * head_dim,
                                bias=False)
        self.v_proj = nn.Linear(hidden_size,
                                num_kv_heads * head_dim,
                                bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, proposal_hidden, target_k, target_v, target_cos,
                target_sin, proposal_cos, proposal_sin):
        """Calibration forward: simple self-attention on proposal tokens,
        with target K/V prepended for cross-attention effect.

        Args:
            proposal_hidden: [B, BS, H] proposal embeddings
            target_k: [B, L, Hkv, D] target key from h_delta projection
            target_v: [B, L, Hkv, D] target value from h_delta projection
            target_cos/target_sin: RoPE for target cache positions
            proposal_cos/proposal_sin: RoPE for proposal positions
        """
        B, BS, _ = proposal_hidden.shape

        # Proposal Q/K/V
        q = self.q_proj(proposal_hidden).view(B, BS, self.num_heads,
                                              self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, BS, D]

        k_self = self.k_proj(proposal_hidden).view(B, BS, self.num_kv_heads,
                                                   self.head_dim)
        k_self = self.k_norm(k_self).transpose(1, 2)  # [B, Hkv, BS, D]

        v_self = self.v_proj(proposal_hidden).view(B, BS, self.num_kv_heads,
                                                   self.head_dim)
        v_self = v_self.transpose(1, 2)  # [B, Hkv, BS, D]

        # Apply RoPE to proposal and target keys. Runtime updates target
        # K through DFlashTargetKVCacheUpdate, which applies RoPE before the
        # target K is visible to proposal attention.
        q, k_self = apply_rotary_pos_emb(q, k_self, proposal_cos, proposal_sin)

        # Concatenate target + proposal K/V
        # target_k/v: [B, L, Hkv, D] -> [B, Hkv, L, D]
        target_k_t = target_k.transpose(1, 2)
        target_v_t = target_v.transpose(1, 2)
        target_k_t = (target_k_t * target_cos) + (rotate_half(target_k_t) *
                                                  target_sin)

        k_full = torch.cat([target_k_t, k_self], dim=2)  # [B, Hkv, L+BS, D]
        v_full = torch.cat([target_v_t, v_self], dim=2)  # [B, Hkv, L+BS, D]

        # GQA repeat
        k_full = repeat_kv(k_full, self.kv_groups)
        v_full = repeat_kv(v_full, self.kv_groups)

        # Scaled dot-product attention with causal mask for proposal tokens
        total_len = k_full.shape[2]
        w = torch.matmul(q, k_full.transpose(2, 3)) * self.attention_scale

        # Proposal tokens can attend to all target tokens + causal self
        if self.is_causal and BS > 1:
            # Causal mask: proposal position i can attend to target[:L] +
            # proposal[:i+1]
            mask = torch.ones(BS, total_len, device=q.device, dtype=torch.bool)
            for i in range(BS):
                # Can see all target positions + proposal 0..i
                mask[i, target_k.shape[1] + i + 1:] = False
            mask = (~mask).unsqueeze(0).unsqueeze(0)  # [1, 1, BS, L+BS]
            w.masked_fill_(mask, torch.finfo(w.dtype).min)

        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(w, v_full)  # [B, H, BS, D]
        attn_output = attn_output.transpose(1,
                                            2).contiguous().reshape(B, BS, -1)

        return self.o_proj(attn_output)


class DFlashCalibDecoderLayer(nn.Module):
    """DFlash draft decoder layer for calibration."""

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 attention_scale=None,
                 is_causal=True):
        super().__init__()
        self.self_attn = DFlashCalibAttention(hidden_size, num_heads,
                                              num_kv_heads, head_dim,
                                              rms_norm_eps, attention_scale,
                                              is_causal)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, h_delta, target_cos, target_sin,
                proposal_cos, proposal_sin):
        """
        Args:
            hidden_states: [B, BS, H] proposal hidden
            h_delta: [B, L, H] target hidden delta (after fc + norm)
            target_cos/target_sin: RoPE for target cache positions
            proposal_cos/proposal_sin: RoPE for proposal positions
        """
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # Target K/V from h_delta
        B, L, _ = h_delta.shape
        num_kv = self.self_attn.num_kv_heads
        hd = self.self_attn.head_dim
        target_k = self.self_attn.k_proj(h_delta).view(B, L, num_kv, hd)
        target_k = self.self_attn.k_norm(target_k)
        target_v = self.self_attn.v_proj(h_delta).view(B, L, num_kv, hd)

        attn_out = self.self_attn(normed, target_k, target_v, target_cos,
                                  target_sin, proposal_cos, proposal_sin)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states


class DFlash2CalibGroupedConv(nn.Module):
    """ModelOpt-visible equivalent of the production grouped-conv plugin."""

    def __init__(self, hidden_size, block_size, taps, group_size):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")
        if block_size <= 0 or taps <= 0 or group_size <= 0:
            raise ValueError(
                "DFlash2 grouped-conv dimensions must be positive")
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size
        base = torch.zeros(2, self.taps, int(hidden_size))
        base[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base)
        self.kernel_projection = nn.Linear(int(hidden_size),
                                           2 * self.taps * self.num_groups,
                                           bias=False)

    def _convolve(self, hidden_states, delta, side, residual=None):
        batch, sequence, hidden = hidden_states.shape
        blocks = hidden_states.reshape(batch, sequence, self.num_groups,
                                       self.group_size)
        base = self.base_kernel[side].reshape(1, 1, self.taps, self.num_groups,
                                              self.group_size)
        coefficients = (base + delta.unsqueeze(-1)).to(torch.float32)
        blocks = blocks.to(torch.float32)
        output = coefficients[:, :, 0] * blocks
        positions = torch.arange(sequence, device=hidden_states.device)
        positions = positions.remainder(self.block_size)
        for tap in range(1, self.taps):
            padding = torch.zeros_like(blocks[:, :tap])
            shifted = torch.cat((padding, blocks[:, :-tap]), dim=1)
            valid = (positions >= tap).reshape(1, sequence, 1, 1)
            output = output + coefficients[:, :, tap] * shifted * valid
        output = output.reshape(batch, sequence, hidden)
        if residual is not None:
            return output + residual.to(torch.float32)
        if hidden_states.dtype == torch.float16:
            output = output.clamp(min=-torch.finfo(torch.float16).max,
                                  max=torch.finfo(torch.float16).max)
        return output.to(hidden_states.dtype)

    def prepare(self, hidden_states):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups)
        return (self._convolve(hidden_states, coefficients[..., 0, :, :],
                               0), coefficients[..., 1, :, :])

    def finish(self, hidden_states, coefficients, residual):
        return self._convolve(hidden_states, coefficients, 1, residual)


class DFlash2CalibDecoderLayer(DFlashCalibDecoderLayer):
    """DFlash2 calibration layer with both trained convolution wrappers."""

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 block_size,
                 taps,
                 group_size,
                 attention_scale=None,
                 is_causal=False):
        super().__init__(hidden_size, intermediate_size, num_heads,
                         num_kv_heads, head_dim, rms_norm_eps, attention_scale,
                         is_causal)
        self.attention_conv = DFlash2CalibGroupedConv(hidden_size, block_size,
                                                      taps, group_size)
        self.mlp_conv = DFlash2CalibGroupedConv(hidden_size, block_size, taps,
                                                group_size)

    def forward(self, hidden_states, h_delta, target_cos, target_sin,
                proposal_cos, proposal_sin):
        residual = hidden_states
        normed, attention_kernel = self.attention_conv.prepare(
            self.input_layernorm(hidden_states))

        batch, length, _ = h_delta.shape
        num_kv = self.self_attn.num_kv_heads
        head_dim = self.self_attn.head_dim
        target_k = self.self_attn.k_proj(h_delta).view(batch, length, num_kv,
                                                       head_dim)
        target_k = self.self_attn.k_norm(target_k)
        target_v = self.self_attn.v_proj(h_delta).view(batch, length, num_kv,
                                                       head_dim)
        attention = self.self_attn(normed, target_k, target_v, target_cos,
                                   target_sin, proposal_cos, proposal_sin)
        hidden_states = self.attention_conv.finish(attention, attention_kernel,
                                                   residual)

        residual = hidden_states
        normed, mlp_kernel = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden_states))
        return self.mlp_conv.finish(self.mlp(normed), mlp_kernel, residual)


class DFlash2CalibCandidateSelector(nn.Module):
    """Materialize and calibrate every trained DFlash2 selector parameter."""

    def __init__(self, hidden_size, vocab_size, rank):
        super().__init__()
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.predecessor_codebook = nn.Parameter(torch.zeros(vocab_size, rank),
                                                 requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.zeros(vocab_size, rank),
                                               requires_grad=False)

    def forward(self, hidden_states):
        return self.hidden_projection(hidden_states)


class DFlashCalibDraftModel(nn.Module):
    """Standalone DFlash draft model for quantization calibration.

    Module names match the checkpoint-native builder model.s state_dict keys so
    quantized weights load without remapping.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hs = config.hidden_size
        head_dim = getattr(config, "head_dim",
                           hs // config.num_attention_heads)
        attention_scale = resolve_attention_scale(
            config, getattr(config, "model_type", ""), head_dim)

        dflash_cfg = getattr(config, "dflash_config", {}) or {}
        architectures = tuple(getattr(config, "architectures", ()) or ())
        self.is_dflash2 = "DFlash2DraftModel" in architectures
        self.target_layer_ids = dflash_cfg.get("target_layer_ids",
                                               [1, 8, 15, 22, 29])
        self.block_size = dflash_cfg.get("block_size", 16)
        self.mask_token_id = dflash_cfg.get("mask_token_id", 248070)
        num_target_layers = len(self.target_layer_ids)

        self.fc = nn.Linear(num_target_layers * hs, hs, bias=False)
        self.hidden_norm = RMSNorm(hs, eps=config.rms_norm_eps)

        if self.is_dflash2:
            taps = int(dflash_cfg["conv_kernel_size"])
            group_size = int(dflash_cfg["conv_group_size"])
            self.layers = nn.ModuleList([
                DFlash2CalibDecoderLayer(
                    hs, config.intermediate_size, config.num_attention_heads,
                    config.num_key_value_heads, head_dim, config.rms_norm_eps,
                    self.block_size, taps, group_size, attention_scale,
                    bool(getattr(config, "is_causal", False)))
                for _ in range(config.num_hidden_layers)
            ])
            self.candidate_selector = DFlash2CalibCandidateSelector(
                hs, config.vocab_size, int(dflash_cfg["selector_rank"]))
        else:
            self.layers = nn.ModuleList([
                DFlashCalibDecoderLayer(hs, config.intermediate_size,
                                        config.num_attention_heads,
                                        config.num_key_value_heads, head_dim,
                                        config.rms_norm_eps, attention_scale)
                for _ in range(config.num_hidden_layers)
            ])
            self.candidate_selector = None
        self.norm = RMSNorm(hs, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(hs, config.vocab_size, bias=False)

        rope_theta = getattr(config, "rope_theta", None) or 10000.0
        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_theta)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, proposal_embeds, target_hidden_concat):
        """Calibration forward.

        Args:
            proposal_embeds: [B, BS, H] — embeddings of [last_token, mask...]
            target_hidden_concat: [B, L, Nl*H] — concatenated target hidden
                states from the base model.
        """
        B, BS, _ = proposal_embeds.shape

        # fc + norm: keep the precision-sensitive target-hidden projector and
        # RMSNorm input in FP32. The exported runtime model uses the same
        # single full-FP32 projector path.
        h_delta_acc = nn.functional.linear(
            target_hidden_concat.to(torch.float32),
            self.fc.weight.to(torch.float32))
        h_delta = self.hidden_norm(h_delta_acc).to(proposal_embeds.dtype)

        # RoPE positions mirror the runtime first-round semantics:
        # target delta is written at positions [0, L), then proposal tokens are
        # read at positions [L, L + BS).
        L = h_delta.shape[1]
        target_position_ids = torch.arange(
            L, dtype=torch.long, device=proposal_embeds.device).unsqueeze(0)
        proposal_position_ids = (L + torch.arange(
            BS, dtype=torch.long, device=proposal_embeds.device)).unsqueeze(0)
        target_cos, target_sin = self.rotary_emb(h_delta, target_position_ids)
        proposal_cos, proposal_sin = self.rotary_emb(proposal_embeds,
                                                     proposal_position_ids)

        # DFlash2 is trained in BF16 and its residual stream exceeds FP16
        # range. Match the TensorRT graph: residuals stay FP32 while RMSNorm
        # emits the FP16 activations consumed by attention/MLP/conv Linears.
        hidden_states = (proposal_embeds.to(torch.float32)
                         if self.is_dflash2 else proposal_embeds)
        for layer in self.layers:
            hidden_states = layer(hidden_states, h_delta, target_cos,
                                  target_sin, proposal_cos, proposal_sin)

        hidden_states = self.norm(hidden_states)
        if self.candidate_selector is not None:
            prediction_hidden = hidden_states[:, 1:self.block_size]
            # Execute the projection so ModelOpt observes its real activation
            # distribution. Codebook lookups do not require quantization.
            self.candidate_selector(prediction_hidden)
            hidden_states = prediction_hidden
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls,
                        draft_model_dir,
                        base_model_dir=None,
                        device="cuda"):
        """Load DFlash draft weights from checkpoint."""
        config = AutoConfig.from_pretrained(draft_model_dir,
                                            trust_remote_code=True)
        if hasattr(config, "text_config"):
            config = config.text_config

        # Attach dflash_config to the config object. JetSpec and DSpark drafts
        # share the DFlash backbone; their raw checkpoints (and some raw
        # DFlash ones) publish target_layer_ids/block_size/mask_token_id at
        # the top level.
        cfg_path = os.path.join(draft_model_dir, "config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        raw_spec_cfg = cfg_dict.get("dflash_config")
        if raw_spec_cfg is None:
            raw_spec_cfg = cfg_dict.get("jetspec_config")
        if raw_spec_cfg is None:
            raw_spec_cfg = cfg_dict.get("dspark_config")
        spec_cfg = dict(raw_spec_cfg or {})
        for key in ("target_layer_ids", "block_size", "mask_token_id"):
            if key not in spec_cfg and key in cfg_dict:
                spec_cfg[key] = cfg_dict[key]
        config.dflash_config = spec_cfg

        model = cls(config)

        # Load draft weights
        draft_sd = {}
        for sf_path in sorted(Path(draft_model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt", device=device) as f:
                for key in f.keys():
                    if "rotary_emb" in key:
                        continue
                    draft_sd[key] = f.get_tensor(key)

        # Load lm_head from base if not in draft checkpoint. The fallback
        # decision must use the base model config, not the draft config.
        if "lm_head.weight" not in draft_sd and base_model_dir:
            base_config = AutoConfig.from_pretrained(base_model_dir,
                                                     trust_remote_code=True)
            if hasattr(base_config, "text_config"):
                base_config = base_config.text_config
            _fill_lm_head_from_base(
                draft_sd, base_model_dir, device,
                bool(getattr(base_config, "tie_word_embeddings", True)))

        missing, _ = model.load_state_dict(draft_sd, strict=False)
        real_missing = [k for k in missing if not k.startswith("rotary_emb")]
        if real_missing:
            logger.warning("Missing keys in DFlash draft model: %s",
                           real_missing)

        model.to(device)
        return model


def _fill_lm_head_from_base(sd, base_dir, device, tie_word_embeddings):
    """Load lm_head.weight from the base model checkpoint."""
    tie = bool(tie_word_embeddings)
    lm_head_keys = [
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
        "model.language_model.lm_head.weight",
    ]
    embed_keys = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ]

    for sf_path in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device=device) as f:
            keys = f.keys()
            # Priority 1: explicit lm_head
            for cand in lm_head_keys:
                if cand in keys:
                    sd["lm_head.weight"] = f.get_tensor(cand)
                    return
            # Priority 2: tied embedding fallback
            if tie:
                for cand in embed_keys:
                    if cand in keys:
                        sd["lm_head.weight"] = f.get_tensor(cand)
                        return

    if not tie:
        raise ValueError(
            "DFlash draft lm_head: base model has tie_word_embeddings=False "
            "but no lm_head.weight found. Cannot fall back to embeddings.")
    raise ValueError("lm_head.weight not found in base model")


# ---------------------------------------------------------------------------
# Quantization & export
# ---------------------------------------------------------------------------


def quantize_and_export_dflash_draft(
    base_model_dir: str,
    draft_model_dir: str,
    output_dir: str,
    quantization: str = "nvfp4",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    *,
    text_dataset: Union[str, TextDataset, None] = None,
    num_samples: int = 512,
) -> str:
    """Load base + DFlash draft models, quantize draft, and export."""
    import shutil
    import time

    from modelopt.torch.export.quant_utils import get_quant_config
    from modelopt.torch.export.unified_export_hf import (
        QUANTIZATION_NONE, _export_quantized_weight, get_quantization_format,
        is_quantlinear, postprocess_state_dict)
    from modelopt.torch.quantization.utils import is_quantized

    if kv_cache_quantization is not None:
        raise ValueError("DFlash draft KV-cache quantization is not "
                         "validated yet.")

    t0 = time.time()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    # Load draft model
    draft = DFlashCalibDraftModel.from_pretrained(draft_model_dir,
                                                  base_model_dir, device)
    draft = draft.eval().to(device).to(torch_dtype)

    if not is_quantized(draft):
        # Load base model for calibration hidden states
        base, tokenizer = _load_base_for_calib(base_model_dir, dtype, device)
        embed_layer = _resolve_base_embed_layer(base)

        quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                       kv_cache_quantization)
        _disable_dflash_fc_quantization(quant_cfg)
        from ..quantize import _text_calib_dataloader
        text_ds = resolve_dataset(text_dataset, "text")
        print(f"DFlash text calibration: {dataset_name(text_ds)}")
        loader = _text_calib_dataloader(
            tokenizer,
            text_ds,
            batch_size=16 if "int4" in quantization else 1,
            num_samples=num_samples)

        target_layer_ids = draft.target_layer_ids
        block_size = draft.block_size
        mask_token_id = draft.mask_token_id

        def _calib(dm):
            for data in tqdm(loader, desc="Calibrating DFlash draft"):
                data = data.to(device)
                with torch.no_grad():
                    out = base(input_ids=data, output_hidden_states=True)
                hs = out["hidden_states"]

                # Build target_hidden_concat matching Edge-LLM DFlash base
                # HF: hs[0] = embedding, hs[i+1] = layer i output
                selected = []
                for lid in target_layer_ids:
                    idx = lid + 1  # HF offset
                    if idx < len(hs):
                        selected.append(hs[idx])
                    else:
                        selected.append(hs[-1])
                target_hidden_concat = torch.cat(selected, dim=-1)

                # Build proposal input: [last_token, mask, mask, ...]
                B, S = data.shape
                last_tokens = data[:, -1:]
                mask_ids = torch.full((B, block_size - 1),
                                      mask_token_id,
                                      dtype=torch.long,
                                      device=device)
                proposal_ids = torch.cat([last_tokens, mask_ids], dim=1)

                proposal_embeds = embed_layer(proposal_ids).to(torch_dtype)

                # Use a single delta position for calibration
                dm(proposal_embeds, target_hidden_concat)

        mtq.quantize(draft, quant_cfg, forward_loop=_calib)
        mtq.print_quant_summary(draft)

    logger.info("Quantization: %.1fs", time.time() - t0)

    # Export quantized weights
    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        for _, m in draft.named_modules():
            if (get_quantization_format(m) != QUANTIZATION_NONE
                    and is_quantlinear(m)):
                _export_quantized_weight(m, torch_dtype)

    qc = get_quant_config(draft)
    sd = postprocess_state_dict(draft.state_dict(), 0,
                                qc["quantization"]["kv_cache_quant_algo"])

    # Remove rotary_emb from saved state
    sd = {k: v for k, v in sd.items() if not k.startswith("rotary_emb")}

    # Pass DSpark sidecar heads through unquantized: the exporter reads
    # markov/confidence tensors from this checkpoint's safetensors.
    passthrough_prefixes = ("markov_head.", "confidence_head.")
    for sf_path in sorted(Path(draft_model_dir).glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(passthrough_prefixes) and key not in sd:
                    sd[key] = f.get_tensor(key)

    save_file(sd, os.path.join(output_dir, "model.safetensors"))

    # Copy config.json from draft dir, preserving dflash_config
    src_cfg = os.path.join(draft_model_dir, "config.json")
    if os.path.isfile(src_cfg):
        shutil.copy2(src_cfg, os.path.join(output_dir, "config.json"))

    # Record the precision-sensitive projectors that stay in FP16 while the
    # heavy draft linears, including lm_head when requested, are quantized.
    q_section = qc.setdefault("quantization", {})
    exclude = list(q_section.get("exclude_modules", []))
    if "fc" not in exclude:
        exclude.append("fc")
    if draft.is_dflash2 and "candidate_selector.hidden_projection" not in exclude:
        exclude.append("candidate_selector.hidden_projection")
    q_section["exclude_modules"] = exclude

    # Write hf_quant_config.json
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(qc, f)

    logger.info("Saved DFlash draft to %s (total %.1fs)", output_dir,
                time.time() - t0)
    return output_dir


def _load_base_for_calib(model_dir, dtype, device):
    """Load unquantized base model and tokenizer for calibration."""
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True).to(device)
    gc.collect()  # release safetensor mmap handles after GPU transfer
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _resolve_base_embed_layer(base):
    """Resolve the base model embedding layer once for DFlash calibration."""
    if hasattr(base, "model"):
        base_inner = base.model
        if hasattr(base_inner, "language_model"):
            return base_inner.language_model.embed_tokens
        if hasattr(base_inner, "embed_tokens"):
            return base_inner.embed_tokens
        raise AttributeError(
            f"Cannot locate embed_tokens on {type(base_inner).__name__}")
    if hasattr(base, "embed_tokens"):
        return base.embed_tokens
    raise AttributeError(
        f"Cannot locate embed_tokens on {type(base).__name__}")


def _disable_dflash_fc_quantization(quant_cfg):
    """Exclude DFlash projectors whose quantization changes proposal semantics."""
    names = (
        "fc.input_quantizer",
        "fc.weight_quantizer",
        "fc.output_quantizer",
        "candidate_selector.hidden_projection.input_quantizer",
        "candidate_selector.hidden_projection.weight_quantizer",
        "candidate_selector.hidden_projection.output_quantizer",
    )
    append_quant_cfg_entries(quant_cfg, [{
        "quantizer_name": name,
        "enable": False
    } for name in names])
