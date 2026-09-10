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
"""Standalone Qwen3.5 MTP Draft Model for quantization.

MTP (Multi-Token Prediction) is a speculative-decoding draft architecture
embedded in Qwen3.5 checkpoints under the ``mtp.*`` weight prefix.  Unlike
EAGLE3 (which lives in a separate checkpoint directory), MTP weights are
stored alongside the base model in the same safetensors files.

This module provides a fully self-contained PyTorch implementation with no
dependency on ``transformers`` model classes.  Only the calibration forward
path is implemented; the full inference forward with KV-cache and TensorRT
plugins belongs in the ONNX export layer.

Architecture::

    input_ids ---------> embed_tokens --> pre_fc_norm_embedding --+
                                                                  |--> cat --> fc --> decoder_layer --> norm --> lm_head
    hidden_states_from_base -----------> pre_fc_norm_hidden ------+

The single decoder layer is a Qwen3.5 gated-attention block with QK-norm
and SwiGLU MLP.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from safetensors.torch import load_file, safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorrt_edgellm._safetensors_io import save_file
from tensorrt_edgellm.checkpoint.fused_weights import (
    split_fused_moe_experts, split_fused_shared_expert)

from ..datasets import TextDataset, dataset_name, resolve_dataset
from ..quantization_configs import build_quant_config
from .attention_scale import resolve_attention_scale
from .layers import RotaryEmbedding, SwiGLUMLP, apply_rotary_pos_emb, repeat_kv

# ---------------------------------------------------------------------------
# Qwen3.5-specific building blocks
# ---------------------------------------------------------------------------


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with Qwen3.5 residual-weight convention.

    HuggingFace Qwen3.5 stores weights initialised to **zero** and computes
    ``(1 + weight) * RMSNorm(x)``.  This differs from Llama-style RMSNorm
    (``weight * RMSNorm(x)`` with weight initialised to one).
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        hidden = x.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * hidden).to(input_dtype)


# ---------------------------------------------------------------------------
# Qwen3.5 Gated Attention (calibration-only, no KV-cache)
# ---------------------------------------------------------------------------


class Qwen3_5GatedAttention(nn.Module):
    """Qwen3.5 gated attention for MTP calibration.

    ``q_proj`` packs both query and gate: output shape is
    ``[batch, seq, num_heads * head_dim * 2]``.  Per head, the first
    ``head_dim`` values are the query, the second ``head_dim`` are the gate.
    After attention: ``output = o_proj(attn_out * sigmoid(gate))``.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 bias,
                 attention_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.attention_scale = (resolve_attention_scale({}, "", head_dim) if
                                attention_scale is None else attention_scale)

        self.q_proj = nn.Linear(hidden_size,
                                num_heads * head_dim * 2,
                                bias=bias)
        self.k_proj = nn.Linear(hidden_size,
                                num_kv_heads * head_dim,
                                bias=bias)
        self.v_proj = nn.Linear(hidden_size,
                                num_kv_heads * head_dim,
                                bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, x, cos, sin):
        B, L, _ = x.size()

        # Q projection with query/gate split
        q_output = self.q_proj(x)
        q_output = q_output.view(B, L, self.num_heads, self.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)
        # query_states, gate_states: [B, L, num_heads, head_dim]

        key_states = self.k_proj(x).view(B, L, self.num_kv_heads,
                                         self.head_dim)
        value_states = self.v_proj(x).view(B, L, self.num_kv_heads,
                                           self.head_dim)

        # QK-norm (per-head, residual-weight convention)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to [B, H, L, D] for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        gate_states = gate_states.transpose(1, 2)

        # RoPE
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # GQA: repeat KV heads
        key_states = repeat_kv(key_states, self.kv_groups)
        value_states = repeat_kv(value_states, self.kv_groups)

        # Scaled dot-product attention with causal mask
        w = (torch.matmul(query_states, key_states.transpose(2, 3)) *
             self.attention_scale)
        if L > 1:
            mask = torch.triu(
                torch.ones(1, 1, L, L, device=x.device, dtype=torch.bool), 1)
            w.masked_fill_(mask, torch.finfo(w.dtype).min)
        w = nn.functional.softmax(w, dim=-1,
                                  dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(w, value_states)

        # Gating: attn_output * sigmoid(gate)
        attn_output = attn_output * torch.sigmoid(gate_states)

        attn_output = attn_output.transpose(1,
                                            2).contiguous().reshape(B, L, -1)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# MTP MLPs, Decoder Layer, and Model
# ---------------------------------------------------------------------------


def _num_experts(config) -> int:
    """Routed-expert count under whichever key the checkpoint family uses.

    ``num_experts`` is an ``attribute_map`` alias for ``num_local_experts`` on
    some HF configs, but not all of them declare the map — ``Qwen3OmniNextTextConfig``
    ships an empty one — so reading a single key silently reports a sparse-MoE
    head as dense. Mirrors the parsing in :mod:`tensorrt_edgellm.config`.
    """
    for key in ("num_experts", "num_local_experts"):
        value = getattr(config, key, None)
        if value:
            return int(value)
    return 0


class MtpSparseMoeBlock(nn.Module):
    """Qwen3.5 sparse-MoE block used by the MTP calibration model."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = int(config.moe_intermediate_size)
        self.num_experts = _num_experts(config)
        self.top_k = int(config.num_experts_per_tok)

        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")

        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size)
            for _ in range(self.num_experts)
        ])

        shared_size = int(
            getattr(config, "shared_expert_intermediate_size", 0) or 0)
        self.has_shared_expert = shared_size > 0
        if self.has_shared_expert:
            self.shared_expert = SwiGLUMLP(hidden_size, shared_size)
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        router_probs = torch.softmax(self.gate(flat_states).float(), dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat_states.dtype)

        routed_states = torch.zeros_like(flat_states)
        for expert_id in torch.unique(selected_experts).tolist():
            token_indices, expert_slots = torch.where(
                selected_experts == expert_id)
            expert_states = self.experts[expert_id](flat_states[token_indices])
            expert_states *= routing_weights[token_indices, expert_slots, None]
            routed_states.index_add_(0, token_indices,
                                     expert_states.to(routed_states.dtype))

        if self.has_shared_expert:
            shared_states = self.shared_expert(flat_states)
            shared_states *= torch.sigmoid(
                self.shared_expert_gate(flat_states))
            routed_states += shared_states

        return routed_states.reshape(batch_size, sequence_length, hidden_size)


class MtpDecoderLayer(nn.Module):
    """Single Qwen3.5 MTP decoder layer (gated attention + dense/MoE MLP)."""

    def __init__(self,
                 config,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rms_norm_eps,
                 bias,
                 attention_scale=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.input_layernorm = Qwen3_5RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3_5GatedAttention(hidden_size, num_heads,
                                               num_kv_heads, head_dim,
                                               rms_norm_eps, bias,
                                               attention_scale)
        self.post_attention_layernorm = Qwen3_5RMSNorm(hidden_size,
                                                       eps=rms_norm_eps)
        if _num_experts(config) > 0:
            self.mlp = MtpSparseMoeBlock(config)
        else:
            self.mlp = SwiGLUMLP(hidden_size, config.intermediate_size)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states),
                                       cos, sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class MtpDraftModel(nn.Module):
    """Standalone Qwen3.5 MTP draft model for quantization.

    Fully self-contained — uses only basic ``torch.nn`` modules.
    Module names match HuggingFace checkpoint keys (after stripping
    the ``mtp.`` prefix) so weights load without remapping.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hs = config.hidden_size
        head_dim = getattr(config, "head_dim",
                           hs // config.num_attention_heads)
        bias = getattr(config, "attention_bias", False)
        attention_scale = resolve_attention_scale(
            config, getattr(config, "model_type", ""), head_dim)

        # MTP fusion layers
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hs,
                                                    eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hs, eps=config.rms_norm_eps)
        self.fc = nn.Linear(hs * 2, hs, bias=False)

        # Single decoder layer
        self.layers = nn.ModuleList([
            MtpDecoderLayer(config, config.num_attention_heads,
                            config.num_key_value_heads, head_dim,
                            config.rms_norm_eps, bias, attention_scale)
        ])

        self.norm = Qwen3_5RMSNorm(hs, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(hs, config.vocab_size, bias=False)

        # Shared with base model — set by calibration code
        self.embed_tokens: Optional[nn.Embedding] = None

        rope_theta = getattr(config, "rope_theta", None) or 10000.0
        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_theta)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, hidden_states_from_base):
        """Calibration forward (no KV cache, no TRT plugins).

        Args:
            input_ids: ``(batch, seq_len)`` — used to look up shared embeddings.
            hidden_states_from_base: ``(batch, seq_len, hidden_size)`` — last
                hidden state from the unquantized base model.

        Returns:
            Logits of shape ``(batch, vocab_size)``.
        """
        assert self.embed_tokens is not None, \
            "embed_tokens must be set (shared from base model) before forward"

        inputs_embeds = self.embed_tokens(input_ids)

        # Fuse embeddings and base hidden states
        normed_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden = self.pre_fc_norm_hidden(hidden_states_from_base)
        hidden_states = self.fc(
            torch.cat((normed_embeds, normed_hidden), dim=-1))

        # RoPE
        position_ids = torch.arange(input_ids.shape[1],
                                    dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        # Final norm + lm_head on last token
        return self.lm_head(self.norm(hidden_states[:, -1]))

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        """Load MTP draft weights from a Qwen3.5 checkpoint.

        Reads ``mtp.*`` keys from safetensors, strips the ``mtp.`` prefix,
        and loads them into the model.  ``lm_head`` is loaded from the
        checkpoint's ``lm_head.weight`` (or tied from ``embed_tokens``).
        """
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Descend to the text config that declares the MTP head: Omni nests it
        # under ``thinker_config.text_config``; other multimodal models expose
        # a direct ``text_config``; plain LLMs are already at that level.
        if hasattr(config, "thinker_config"):
            config = config.thinker_config
        if hasattr(config, "text_config"):
            config = config.text_config
        model = cls(config)

        mtp_state_dict = {}
        for sf_path in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt", device=device) as f:
                for key in f.keys():
                    # Standalone LLMs store the head as ``mtp.*``; Omni
                    # checkpoints nest it under the Thinker (``thinker.mtp.*``).
                    prefix = next((p for p in ("mtp.", "thinker.mtp.")
                                   if key.startswith(p)), None)
                    if prefix is not None:
                        mtp_state_dict[key[len(prefix):]] = f.get_tensor(key)
                    elif key == "lm_head.weight":
                        mtp_state_dict["lm_head.weight"] = f.get_tensor(key)
                    elif key in ("model.embed_tokens.weight",
                                 "model.language_model.embed_tokens.weight"):
                        if "lm_head.weight" not in mtp_state_dict:
                            mtp_state_dict["lm_head.weight"] = f.get_tensor(
                                key)

        if not any(k != "lm_head.weight" for k in mtp_state_dict):
            raise RuntimeError(
                f"No 'mtp.*' tensors found under {model_dir} — nothing to "
                "load for the draft head; check that the checkpoint is "
                "complete and carries the MTP draft weights.")

        _split_fused_mtp_weights(mtp_state_dict, model)

        missing, unexpected = model.load_state_dict(mtp_state_dict,
                                                    strict=False)
        # Omni checkpoints ship no ``mtp.lm_head``: the draft borrows the
        # base head at export, or --lm_head_quantization copies it in.
        expected_absent = ("rotary_emb", "lm_head")
        real_missing = [
            k for k in missing if not k.startswith(expected_absent)
        ]
        if real_missing:
            raise RuntimeError(
                f"MTP draft has {len(real_missing)} unloaded weight(s); "
                f"first keys: {real_missing[:10]}")
        if unexpected:
            raise RuntimeError(
                f"MTP draft checkpoint has {len(unexpected)} unsupported "
                f"weight(s); first keys: {unexpected[:10]}")

        model.to(device)
        return model


def _split_fused_mtp_weights(state_dict: Dict[str, torch.Tensor],
                             model: nn.Module) -> None:
    """Expand fused checkpoint tensors into the calibration module layout."""
    mlp = model.layers[0].mlp
    shared_expert = getattr(mlp, "shared_expert", None)
    for key in list(state_dict):
        tensor = state_dict[key]
        split_weights = split_fused_moe_experts(key, tensor)
        if split_weights is None and shared_expert is not None:
            split_weights = split_fused_shared_expert(
                key, tensor, shared_expert.gate_proj.in_features,
                shared_expert.gate_proj.out_features)
        if split_weights is None:
            continue
        del state_dict[key]
        state_dict.update(split_weights)


# ---------------------------------------------------------------------------
# Quantization & export
# ---------------------------------------------------------------------------


def _share_embed_tokens(base_model, mtp_draft):
    """Share the base model's embedding table with the MTP draft model."""
    base_embed = None
    if hasattr(base_model, "model"):
        base_inner = base_model.model
        if hasattr(base_inner, "language_model"):
            base_embed = base_inner.language_model.embed_tokens
        elif hasattr(base_inner, "embed_tokens"):
            base_embed = base_inner.embed_tokens
    if base_embed is None:
        raise ValueError("Could not find embed_tokens in the base model")
    mtp_draft.embed_tokens = _EmbeddingProxy(base_embed)


class _EmbeddingProxy(nn.Module):
    """Read-only embedding view used so MTP quantization does not mutate base."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.padding_idx = embedding.padding_idx
        self.register_buffer("weight",
                             embedding.weight.detach(),
                             persistent=False)

    def forward(self, input_ids):
        return nn.functional.embedding(input_ids,
                                       self.weight,
                                       padding_idx=self.padding_idx)


def quantize_mtp_from_base(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_dir: str,
    quantization: str,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    *,
    text_dataset: Union[str, TextDataset, None] = None,
    num_samples: int = 512,
) -> "MtpDraftModel":
    """Load and quantize the MTP draft using the unquantized base model.

    Called *before* base model quantization so the unquantized base can
    generate calibration hidden states.

    Returns:
        Quantized MTP draft model.
    """
    from modelopt.torch.quantization.utils import is_quantized

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    mtp_draft = MtpDraftModel.from_pretrained(model_dir, device)
    mtp_draft = mtp_draft.eval().to(torch_dtype)

    if is_quantized(mtp_draft):
        print("MTP draft model is already quantized, skipping.")
        return mtp_draft

    _share_embed_tokens(base_model, mtp_draft)

    # Omni checkpoints ship no ``mtp.lm_head``, so the draft's copy is still at
    # its random init here. Quantizing that would calibrate on noise; the real
    # head is the base's, which the draft borrows at export anyway.
    if lm_head_quantization is not None:
        base_lm = getattr(base_model, "lm_head", None) or getattr(
            getattr(base_model, "model", None), "lm_head", None)
        if base_lm is not None and hasattr(mtp_draft, "lm_head"):
            with torch.no_grad():
                mtp_draft.lm_head.weight.copy_(
                    base_lm.weight.to(mtp_draft.lm_head.weight))
            print("Loaded base LM head into the MTP draft for quantization.")

    quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                   kv_cache_quantization)
    is_moe = isinstance(mtp_draft.layers[0].mlp, MtpSparseMoeBlock)
    if is_moe:
        quant_cfg["quant_cfg"].extend({
            "quantizer_name": pattern,
            "enable": False,
        } for pattern in (
            "*mlp.gate.weight_quantizer",
            "*mlp.gate.input_quantizer",
            "*shared_expert_gate.weight_quantizer",
            "*shared_expert_gate.input_quantizer",
        ))
    from ..quantize import _text_calib_dataloader
    text_ds = resolve_dataset(text_dataset, "text")
    print(f"MTP text calibration: {dataset_name(text_ds)}")
    loader = _text_calib_dataloader(
        tokenizer,
        text_ds,
        num_samples=num_samples,
        batch_size=16 if "int4" in quantization else 1)

    def _calib(draft):
        for data in tqdm(loader, desc="Calibrating MTP draft"):
            data = data.to(device)
            with torch.no_grad():
                outputs = base_model(data, output_hidden_states=True)
            last_hidden = outputs["hidden_states"][-1]
            draft(data, last_hidden)

    mtq.quantize(mtp_draft, quant_cfg, forward_loop=_calib)

    if is_moe:
        # Top-k routing leaves un-fired experts without a calibration amax, and
        # NVFP4 export reads weight_scale_2 from it; derive one from the weights.
        from ..qwen3_omni import _backfill_missing_amax
        n_filled = _backfill_missing_amax(mtp_draft)
        if n_filled:
            print(f"Backfilled {n_filled} MTP amax buffer(s) from weights "
                  "(top-k routing left them uncalibrated).")

    mtq.print_quant_summary(mtp_draft)
    return mtp_draft


def export_quantized_mtp_state_dict(mtp_draft: "MtpDraftModel",
                                    dtype: str) -> Dict[str, torch.Tensor]:
    """Return quantized MTP tensors in unified-checkpoint key format."""
    from modelopt.torch.export.quant_utils import get_quant_config
    from modelopt.torch.export.unified_export_hf import (
        QUANTIZATION_NONE, _export_quantized_weight, get_quantization_format,
        is_quantlinear, postprocess_state_dict)

    model_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    with torch.inference_mode():
        for _, m in mtp_draft.named_modules():
            if (get_quantization_format(m) != QUANTIZATION_NONE
                    and is_quantlinear(m)):
                _export_quantized_weight(m, model_dtype)

    qc = get_quant_config(mtp_draft)
    kv_algo = qc["quantization"]["kv_cache_quant_algo"]
    sd = postprocess_state_dict(mtp_draft.state_dict(), 0, kv_algo)

    # Re-prefix with mtp.* and exclude shared tensors
    mtp_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if key.startswith("embed_tokens") or key.startswith("rotary_emb"):
            continue
        mtp_tensors[f"mtp.{key}"] = tensor.detach().cpu()

    return mtp_tensors


def save_quantized_mtp(mtp_draft: "MtpDraftModel", output_dir: str,
                       dtype: str) -> None:
    """Extract quantized MTP weights and merge into the base checkpoint.

    Quantized weights are prefixed with ``mtp.`` and merged into the
    safetensors file(s) in *output_dir*.  Shared tensors (embed_tokens,
    rotary_emb) are excluded.
    """
    mtp_tensors = export_quantized_mtp_state_dict(mtp_draft, dtype)

    _merge_tensors_into_safetensors(output_dir, mtp_tensors)
    print(f"Merged {len(mtp_tensors)} quantized MTP weight(s) into "
          f"{output_dir}.")


def copy_unquantized_mtp(model_dir: str, output_dir: str) -> None:
    """Copy unquantized MTP tensors from *model_dir* into *output_dir*.

    Used as fallback when MTP quantization is not applicable.  Marks MTP
    modules as excluded from quantization in config files.
    """
    mtp_tensors: Dict[str, torch.Tensor] = {}
    for sf_path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    mtp_tensors[key] = f.get_tensor(key)

    if not mtp_tensors:
        return

    _merge_tensors_into_safetensors(output_dir, mtp_tensors)

    mtp_linear_names = sorted(
        {k.rsplit(".", 1)[0]
         for k, v in mtp_tensors.items() if v.dim() >= 2})
    _patch_quant_exclude_list(output_dir, mtp_linear_names)

    print(f"Copied {len(mtp_tensors)} unquantized MTP weight(s) into "
          f"{output_dir}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_tensors_into_safetensors(output_dir, tensors):
    """Merge *tensors* into the safetensors file(s) in *output_dir*."""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    single_path = os.path.join(output_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        first_shard = sorted(set(index["weight_map"].values()))[0]
        shard_path = os.path.join(output_dir, first_shard)
        existing = load_file(shard_path)
        existing.update(tensors)
        save_file(existing, shard_path)
        for key in tensors:
            index["weight_map"][key] = first_shard
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    elif os.path.exists(single_path):
        existing = load_file(single_path)
        existing.update(tensors)
        save_file(existing, single_path)
    else:
        save_file(tensors, os.path.join(output_dir, "mtp.safetensors"))


def _patch_quant_exclude_list(output_dir, module_names):
    """Add *module_names* to quantization exclude lists in the output dir."""
    for cfg_file, path_to_list in [
        ("config.json", ("quantization_config", )),
        ("hf_quant_config.json", ("quantization", )),
    ]:
        cfg_path = os.path.join(output_dir, cfg_file)
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        section = cfg
        for key in path_to_list:
            section = section.get(key, {})
        if not section:
            continue
        list_key = "ignore" if "ignore" in section else "exclude_modules"
        exclude = list(section.get(list_key, []))
        for name in module_names:
            if name not in exclude:
                exclude.append(name)
        section[list_key] = exclude
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
