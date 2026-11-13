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
Qwen3-Omni audio tower: ViTAttentionPlugin path vs the dense-mask path.

The audio tower's attention mask is block diagonal over chunk windows, which is
exactly what a ragged kernel's cu_seqlens encodes. These tests pin that
equivalence and the export I/O contract that goes with it, so the two paths
cannot silently drift apart.

Run:
    python3 -m pytest tests/python-unittests/test_qwen3_omni_audio_attention_plugin.py -v
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.models.qwen3_asr import modeling_qwen3_asr_audio

_D_MODEL = 1280
_NUM_HEADS = 20
_HEAD_DIM = _D_MODEL // _NUM_HEADS
_SCALE = _HEAD_DIM**-0.5
_N_WINDOW = 50
_N_WINDOW_INFER = 800
_EXPORT_CONFIG = {
    "num_mel_bins": 128,
    "n_window": _N_WINDOW,
    "n_window_infer": _N_WINDOW_INFER,
}


def _with_plugin(monkeypatch, enabled: bool):
    """Select the arm for the next module built.

    The flag is read from the environment inside __init__, so setting it is
    enough — both arms can be built in one test by flipping it in between.
    """
    monkeypatch.setenv("USE_AUDIO_ATTN_PLUGIN", "1" if enabled else "0")
    return modeling_qwen3_asr_audio


def _model_config() -> ModelConfig:
    return ModelConfig(model_type="qwen3_asr",
                       hidden_size=_D_MODEL,
                       num_hidden_layers=1,
                       num_attention_heads=_NUM_HEADS,
                       num_key_value_heads=_NUM_HEADS,
                       intermediate_size=5120,
                       head_dim=_HEAD_DIM,
                       rms_norm_eps=1e-6,
                       vocab_size=1024,
                       rope_theta=1e6,
                       max_position_embeddings=4096,
                       default_attention_scale=_SCALE)


def _encoder(encoder_cls, **overrides):
    kwargs = dict(num_mel_bins=128,
                  d_model=_D_MODEL,
                  num_layers=1,
                  num_heads=_NUM_HEADS,
                  ffn_dim=5120,
                  output_dim=2048,
                  downsample_hidden=480)
    kwargs.update(overrides)
    return encoder_cls(**kwargs,
                       attention_scale=_SCALE,
                       model_config=_model_config())


def _attention(monkeypatch, enabled: bool):
    module = _with_plugin(monkeypatch, enabled)
    attn = module.QwenAudioAttention(
        model_config=_model_config(),
        attention_scale=_SCALE,
        d_model=_D_MODEL,
        num_heads=_NUM_HEADS,
        name_prefix="audio_tower.layers.0.self_attn")
    return attn.to(torch.float16).eval()


def _fill_weights(module: torch.nn.Module, seed: int = 42) -> None:
    """make_linear allocates with torch.empty, so give every tensor real
    values."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if not tensor.is_floating_point():
                continue
            scale = 0.02 if tensor.dim() > 1 else 0.0
            tensor.copy_((torch.randn(tensor.shape, generator=generator) *
                          scale).to(tensor.dtype))


def _window_lens(after_cnn_lens: list[int]) -> list[int]:
    """Port of audioUtils::computeChunkwiseWindowLens."""
    width = max(after_cnn_lens) * (_N_WINDOW_INFER // (_N_WINDOW * 2))
    total = sum(after_cnn_lens)
    lens = [width] * (total // width)
    if total % width:
        lens.append(total % width)
    return lens


def _cu_seqlens(window_lens: list[int]) -> list[int]:
    cu = [0]
    for length in window_lens:
        cu.append(cu[-1] + length)
    return cu


def _block_diagonal_mask(cu: list[int]) -> torch.Tensor:
    total = cu[-1]
    mask = torch.full((total, total), -65504.0, dtype=torch.float16)
    for start, end in zip(cu[:-1], cu[1:]):
        mask[start:end, start:end] = 0.0
    return mask


# 28 chunks with a 7-frame tail: a 27.54 s clip, T=358.
@pytest.mark.parametrize("after_cnn_lens, expected_windows", [
    ([13] * 27 + [7], 4),
    ([13] * 60, 8),
    ([13, 13, 5], 1),
])
def test_plugin_matches_dense_mask(monkeypatch, after_cnn_lens,
                                   expected_windows):
    """Attention through ViTAttentionPlugin with cu_seqlens matches the dense
    block-diagonal mask to FP16 precision."""
    plugin_attn = _attention(monkeypatch, True)
    dense_attn = _attention(monkeypatch, False)
    assert plugin_attn._use_attn_plugin and not dense_attn._use_attn_plugin
    _fill_weights(plugin_attn)
    dense_attn.load_state_dict(plugin_attn.state_dict())

    window_lens = _window_lens(after_cnn_lens)
    assert len(window_lens) == expected_windows
    cu = _cu_seqlens(window_lens)
    assert cu[-1] == sum(after_cnn_lens)

    hidden = (torch.randn(
        cu[-1], _D_MODEL, generator=torch.Generator().manual_seed(7)) *
              0.1).to(torch.float16)

    with torch.no_grad():
        plugin_out = plugin_attn(
            hidden,
            cu_seqlens=torch.tensor(cu, dtype=torch.int32),
            max_seqlen_carrier=torch.zeros(max(window_lens),
                                           dtype=torch.int32))
        dense_out = dense_attn(hidden, _block_diagonal_mask(cu))

    assert torch.isfinite(plugin_out).all()
    relative = (plugin_out.float() - dense_out.float()).norm().item()
    assert relative / dense_out.float().norm().item() < 5e-3


def test_plugin_attends_only_inside_its_window(monkeypatch):
    """A token attends only within its cu_seqlens window, so a signal in one
    window never reaches another."""
    attn = _attention(monkeypatch, True)
    _fill_weights(attn)
    cu = [0, 16, 32]
    base = (
        torch.randn(32, _D_MODEL, generator=torch.Generator().manual_seed(3)) *
        0.1).to(torch.float16)
    perturbed = base.clone()
    perturbed[16:] += 1.0

    kwargs = dict(cu_seqlens=torch.tensor(cu, dtype=torch.int32),
                  max_seqlen_carrier=torch.zeros(16, dtype=torch.int32))
    with torch.no_grad():
        out_base = attn(base, **kwargs)
        out_perturbed = attn(perturbed, **kwargs)

    assert torch.equal(out_base[:16], out_perturbed[:16])
    assert not torch.equal(out_base[16:], out_perturbed[16:])


def test_plugin_call_carries_the_head_geometry(monkeypatch):
    """The plugin call carries the head count, head size and scale verbatim,
    with q/k/v as FP16 [T, heads, head_dim]."""
    module = _with_plugin(monkeypatch, True)
    attn = module.QwenAudioAttention(model_config=_model_config(),
                                     attention_scale=_SCALE,
                                     d_model=_D_MODEL,
                                     num_heads=_NUM_HEADS,
                                     name_prefix="a").to(torch.float16).eval()
    _fill_weights(attn)

    recorded = {}
    original = module.ops.vit_attention_plugin

    def recorder(query, key, value, cu_seqlens, carrier, **kwargs):
        recorded.update(kwargs)
        recorded["shapes"] = [tuple(t.shape) for t in (query, key, value)]
        recorded["dtypes"] = [t.dtype for t in (query, key, value)]
        recorded["cu_seqlens"] = cu_seqlens.tolist()
        recorded["carrier_len"] = carrier.shape[0]
        return original(query, key, value, cu_seqlens, carrier, **kwargs)

    module.ops.vit_attention_plugin = recorder
    try:
        hidden = (torch.randn(
            32, _D_MODEL, generator=torch.Generator().manual_seed(5)) *
                  0.1).to(torch.float16)
        with torch.no_grad():
            attn(hidden,
                 cu_seqlens=torch.tensor([0, 16, 32], dtype=torch.int32),
                 max_seqlen_carrier=torch.zeros(16, dtype=torch.int32))
    finally:
        module.ops.vit_attention_plugin = original

    assert recorded["num_heads"] == _NUM_HEADS
    assert recorded["head_size"] == _HEAD_DIM
    assert recorded["attention_scale"] == _SCALE
    assert recorded["shapes"] == [(32, _NUM_HEADS, _HEAD_DIM)] * 3
    assert recorded["dtypes"] == [torch.float16] * 3
    assert recorded["cu_seqlens"] == [0, 16, 32]
    assert recorded["carrier_len"] == 16


def test_export_uses_ragged_descriptors_by_default(monkeypatch):
    """With USE_AUDIO_ATTN_PLUGIN unset, the export takes cu_seqlens and
    max_seqlen_carrier instead of attention_mask, each with a dynamic axis."""
    monkeypatch.delenv("USE_AUDIO_ATTN_PLUGIN", raising=False)
    encoder = _encoder(modeling_qwen3_asr_audio.QwenAudioEncoder)
    inputs, names, _, dynamic_shapes = encoder.get_onnx_export_args(
        _EXPORT_CONFIG, "cpu")

    assert "attention_mask" not in names
    assert names[-2:] == ["cu_seqlens", "max_seqlen_carrier"]
    assert names == list(inputs.keys())
    assert set(dynamic_shapes) == set(names)
    # A key mapped to {} is a static shape, which the key-set check above would
    # still accept, so require every input to declare at least one dynamic axis.
    for name in names:
        assert dynamic_shapes[name], f"{name} has no dynamic axis"
    assert inputs["cu_seqlens"].dtype == torch.int32
    assert inputs["max_seqlen_carrier"].dtype == torch.int32


def test_export_keeps_dense_mask_when_opted_out(monkeypatch):
    """USE_AUDIO_ATTN_PLUGIN=0 exports the dense attention_mask graph without
    the ragged inputs."""
    module = _with_plugin(monkeypatch, False)
    encoder = _encoder(module.QwenAudioEncoder)
    _, names, _, dynamic_shapes = encoder.get_onnx_export_args(
        _EXPORT_CONFIG, "cpu")

    assert "attention_mask" in names
    assert "cu_seqlens" not in names
    assert "max_seqlen_carrier" not in names
    assert set(dynamic_shapes) == set(names)
