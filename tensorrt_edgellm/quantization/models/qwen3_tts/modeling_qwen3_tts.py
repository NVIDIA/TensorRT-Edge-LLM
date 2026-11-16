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
"""Qwen3-TTS Talker + CodePredictor, rebuilt for quantization calibration.

The Qwen3-TTS checkpoint ships no modeling code and no released
``transformers`` registers ``model_type="qwen3_tts"``, so ModelOpt has nothing
to run calibration through. This module rebuilds just enough of the model from
stock ``transformers`` Qwen3 blocks: every Talker and CodePredictor layer is a
plain :class:`Qwen3DecoderLayer` whose parameter names already match the
checkpoint, leaving only the TTS-specific parts to implement here -- the split
text/codec embeddings, ``text_projection``, the Talker's interleaved mrope, and
the CodePredictor's per-codebook generation walk.

The module tree mirrors the checkpoint key layout exactly (``talker.model.*``,
``talker.text_projection.*``, ``talker.code_predictor.*``), so ``state_dict``
round-trips and ModelOpt's ``export_hf_checkpoint`` writes a directory the
ONNX exporter reads back unchanged.

Scope is deliberately narrow: prefill for the Talker and a greedy per-codebook
walk for the CodePredictor, which is all
:func:`tensorrt_edgellm.quantization.qwen3_cp_loader.qwen3_cp_calibration_loop`
exercises. Speaker encoding, Code2Wav and sampling are absent -- this model is
never used for generation.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (Qwen3DecoderLayer,
                                                      Qwen3RMSNorm)


class _SubConfig:
    """Attribute view over a checkpoint sub-config dict.

    Deliberately not a ``PretrainedConfig``: the calibration config serialises
    the raw ``config.json`` verbatim, so transformers never needs to introspect
    or reconstruct these.
    """

    def __init__(self, raw: dict):
        for key, value in raw.items():
            setattr(self, key,
                    _SubConfig(value) if isinstance(value, dict) else value)


class Qwen3TTSCalibConfig(PretrainedConfig):
    """Verbatim passthrough of the checkpoint ``config.json``.

    ``to_dict`` returns the original mapping, so ``save_pretrained`` reproduces
    the input config byte-for-byte and the exporter keeps reading the fields it
    expects. This also sidesteps ``Qwen3TTSSpeakerEncoderConfig``, whose
    ``__init__`` never chains to ``PretrainedConfig.__init__`` and makes
    ``to_diff_dict`` raise ``KeyError: 'dtype'``.
    """

    model_type = "qwen3_tts"

    def __init__(self, raw: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._raw = copy.deepcopy(raw or {})
        for key, value in self._raw.items():
            setattr(self, key,
                    _SubConfig(value) if isinstance(value, dict) else value)

    def to_dict(self):
        return copy.deepcopy(self._raw)

    def to_diff_dict(self):
        return self.to_dict()


def _qwen3_layer_config(raw: dict, num_layers: int) -> Qwen3Config:
    """Map a TTS sub-config onto the stock Qwen3 layer hyper-parameters."""
    return Qwen3Config(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        attention_bias=raw.get("attention_bias", False),
        attention_dropout=raw.get("attention_dropout", 0.0),
        max_position_embeddings=raw["max_position_embeddings"],
        vocab_size=raw["vocab_size"],
        sliding_window=None,
        use_sliding_window=False,
        attn_implementation="eager",
    )


def _inv_freq(head_dim: int, theta: float, device) -> torch.Tensor:
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (theta**(exponent / head_dim))


def _causal_mask(bsz: int, q_len: int, kv_len: int, dtype, device):
    """Additive float mask; row i may attend to keys up to its own position."""
    mask = torch.full((q_len, kv_len),
                      torch.finfo(dtype).min,
                      dtype=dtype,
                      device=device)
    offset = kv_len - q_len
    col = torch.arange(kv_len, device=device)
    row = torch.arange(q_len, device=device)[:, None] + offset
    mask.masked_fill_(col[None, :] <= row, 0.0)
    return mask[None, None].expand(bsz, 1, q_len, kv_len)


@dataclass
class _TalkerOutput:
    """Mirrors the reference Talker's ``(per_layer_states, residual)`` tuple."""

    hidden_states: tuple


class _TalkerInner(nn.Module):
    """``talker.model``: split text/codec embeddings + Qwen3 decoder stack."""

    def __init__(self, raw: dict):
        super().__init__()
        cfg = _qwen3_layer_config(raw, raw["num_hidden_layers"])
        self.text_embedding = nn.Embedding(raw["text_vocab_size"],
                                           raw["text_hidden_size"])
        self.codec_embedding = nn.Embedding(raw["vocab_size"],
                                            raw["hidden_size"])
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg, i) for i in range(raw["num_hidden_layers"]))
        self.norm = Qwen3RMSNorm(raw["hidden_size"], eps=raw["rms_norm_eps"])
        self._head_dim = raw["head_dim"]
        self._theta = raw["rope_theta"]
        rope_scaling = raw.get("rope_scaling") or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._interleaved = bool(rope_scaling.get("interleaved"))

    def _position_embeddings(self, position_ids, dtype, device):
        """Interleaved mrope: sections 1 and 2 overwrite strided slices of the
        section-0 frequencies. With identical position rows -- which is what a
        contiguous text prefill produces -- this reduces to plain rope."""
        inv = _inv_freq(self._head_dim, self._theta, device)
        if position_ids.dim() == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        inv_expanded = inv[None, None, :,
                           None].expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_expanded @ pos).transpose(2, 3)

        def collapse(x):
            out = x[0].clone()
            modality_num = len(self._mrope_section)
            for index, size in enumerate(self._mrope_section[1:], 1):
                sl = slice(index, size * modality_num, modality_num)
                out[..., sl] = x[index][..., sl]
            return out

        if self._interleaved:
            cos = torch.cat([collapse(freqs.cos())] * 2, dim=-1)
            sin = torch.cat([collapse(freqs.sin())] * 2, dim=-1)
        else:
            emb = torch.cat((freqs[0], freqs[0]), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype), sin.to(dtype)

    def forward(self, inputs_embeds, position_ids):
        bsz, seq_len, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)
            position_ids = position_ids[None, None].expand(3, bsz, -1)
        pos_emb = self._position_embeddings(position_ids, dtype, device)
        mask = _causal_mask(bsz, seq_len, seq_len, dtype, device)

        hidden = inputs_embeds
        per_layer = []
        for layer in self.layers:
            hidden = layer(hidden,
                           attention_mask=mask,
                           position_embeddings=pos_emb,
                           use_cache=False)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            per_layer.append(hidden)
        per_layer[-1] = self.norm(hidden)
        return tuple(per_layer)


class _CodePredictorInner(nn.Module):
    """``talker.code_predictor.model``: per-codebook embeddings + Qwen3 stack."""

    def __init__(self, raw: dict):
        super().__init__()
        cfg = _qwen3_layer_config(raw, raw["num_hidden_layers"])
        groups = raw["num_code_groups"]
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg, i) for i in range(raw["num_hidden_layers"]))
        self.norm = Qwen3RMSNorm(raw["hidden_size"], eps=raw["rms_norm_eps"])
        self.codec_embedding = nn.ModuleList(
            nn.Embedding(raw["vocab_size"], raw["hidden_size"])
            for _ in range(groups - 1))
        self._head_dim = raw["head_dim"]
        self._theta = raw["rope_theta"]

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(self, inputs_embeds, cache, cache_position):
        bsz, seq_len, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        inv = _inv_freq(self._head_dim, self._theta, device)
        freqs = cache_position[None, :, None].float() * inv[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).expand(bsz, -1, -1)
        pos_emb = (emb.cos().to(dtype), emb.sin().to(dtype))
        kv_len = int(cache_position[-1]) + 1
        mask = _causal_mask(bsz, seq_len, kv_len, dtype, device)

        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden,
                           attention_mask=mask,
                           position_embeddings=pos_emb,
                           past_key_values=cache,
                           use_cache=True,
                           cache_position=cache_position)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return self.norm(hidden)


@dataclass
class _CodePredictorOutput:
    sequences: torch.Tensor


class _CodePredictor(nn.Module):
    """``talker.code_predictor``: the module FP8_CP actually quantizes."""

    def __init__(self, raw: dict):
        super().__init__()
        groups = raw["num_code_groups"]
        self.model = _CodePredictorInner(raw)
        self.lm_head = nn.ModuleList(
            nn.Linear(raw["hidden_size"], raw["vocab_size"], bias=False)
            for _ in range(groups - 1))

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def generate(self, inputs_embeds=None, max_new_tokens=1, **kwargs):
        """Greedy per-codebook walk matching the reference ``generate``.

        Prefill emits codebook 0 through ``lm_head[0]``; step ``k`` embeds the
        previous token with ``codec_embedding[k - 1]`` and reads
        ``lm_head[k]``, so a full walk touches every head and all but the last
        embedding -- the activation pattern the CP quantizers must see.
        """
        device = inputs_embeds.device
        cache = DynamicCache()
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        hidden = self.model(inputs_embeds, cache, cache_position)
        token = self.lm_head[0](hidden[:, -1]).argmax(-1, keepdim=True)
        sampled = [token]

        for step in range(1, max_new_tokens):
            embeds = self.model.get_input_embeddings()[step - 1](token)
            cache_position = torch.tensor([seq_len + step - 1], device=device)
            hidden = self.model(embeds, cache, cache_position)
            token = self.lm_head[step](hidden[:, -1]).argmax(-1, keepdim=True)
            sampled.append(token)
        return _CodePredictorOutput(sequences=torch.cat(sampled, dim=1))


class _ResizeMLP(nn.Module):
    """``talker.text_projection``: text_hidden_size -> Talker hidden_size."""

    def __init__(self, raw: dict):
        super().__init__()
        text_dim = raw["text_hidden_size"]
        self.linear_fc1 = nn.Linear(text_dim, text_dim, bias=True)
        self.linear_fc2 = nn.Linear(text_dim, raw["hidden_size"], bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class _Talker(nn.Module):

    def __init__(self, raw: dict):
        super().__init__()
        self.model = _TalkerInner(raw)
        self.text_projection = _ResizeMLP(raw)
        self.codec_head = nn.Linear(raw["hidden_size"],
                                    raw["vocab_size"],
                                    bias=False)
        self.code_predictor = _CodePredictor(raw["code_predictor_config"])

    @property
    def dtype(self):
        return self.codec_head.weight.dtype

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def get_text_embeddings(self):
        return self.model.text_embedding

    def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
        per_layer = self.model(inputs_embeds, position_ids)
        return _TalkerOutput(hidden_states=(per_layer, None))


class Qwen3TTSForCalibration(PreTrainedModel):
    """Root wrapper: exposes ``talker`` and keeps ``talker.*`` state-dict keys."""

    config_class = Qwen3TTSCalibConfig
    base_model_prefix = "talker"
    _supports_sdpa = False

    def __init__(self, config: Qwen3TTSCalibConfig):
        super().__init__(config)
        self.talker = _Talker(config.to_dict()["talker_config"])

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        torch_dtype=torch.float16,
                        device="cuda",
                        **kwargs):
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        # Build on meta so the ~0.9B parameters are never initialised on CPU
        # only to be overwritten; the strict key check below guarantees every
        # one of them is replaced by a real tensor.
        with torch.device("meta"):
            model = cls(Qwen3TTSCalibConfig(raw=raw))

        state = {}
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        for shard in shards:
            state.update(
                load_file(os.path.join(model_dir, shard), device="cpu"))

        missing, unexpected = model.load_state_dict(state,
                                                    strict=False,
                                                    assign=True)
        # The checkpoint is talker-only; anything else means the tree drifted
        # from the key layout the exporter later reads back. A missing key
        # would also leave a meta tensor behind.
        if missing or unexpected:
            raise RuntimeError(
                f"Qwen3-TTS calibration model does not match the checkpoint: "
                f"{len(missing)} missing (e.g. {missing[:3]}), "
                f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}).")
        return model.to(device=device, dtype=torch_dtype).eval()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3-TTS calibration runs through qwen3_cp_calibration_loop, "
            "which drives model.talker and model.talker.code_predictor "
            "directly.")


__all__ = ["Qwen3TTSCalibConfig", "Qwen3TTSForCalibration"]
