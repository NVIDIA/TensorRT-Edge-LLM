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

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tensorrt_edgellm.checkpoint.repacking import (_quantize_nvfp4_moe_weight,
                                                   decode_modelopt_nvfp4)
from tensorrt_edgellm.config import QUANT_NVFP4_A16, ModelConfig, QuantConfig
from tensorrt_edgellm.models.lilicorr import (LiLiCorrHead,
                                              build_lattice_attention_bias)
from tensorrt_edgellm.models.linear import FP16Linear, NVFP4A16Linear

_D = 32
_HEADS = 4
_H = 48
_BLOCK = 8
_S = _BLOCK - 1
_K = 8
_B = 2
_FACTOR = 24
_LAYERS = 2
_SIGMA = 3.0
_VEC_EPS = 1e-3
_RMS_EPS = 1e-6
_VOCAB = 64

# Dense head tensor shapes of nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr
# @ abd47806 under "lilicorr.", with NVFP4 [out, in/2] tensors at their decoded
# [out, in] shape.
_CHECKPOINT_HEAD_SHAPES = {
    "anchor_norm.weight": (1024, ),
    "anchor_out_head.bias": (1024, ),
    "anchor_out_head.weight": (1024, 1024),
    "context_proj.bias": (1024, ),
    "context_proj.weight": (1024, 4096),
    "factor_input_proj.bias": (1024, ),
    "factor_input_proj.weight": (1024, 3072),
    "feature_mlp.down_proj.bias": (1024, ),
    "feature_mlp.down_proj.weight": (1024, 1024),
    "feature_mlp.up_proj.bias": (1024, ),
    "feature_mlp.up_proj.weight": (1024, 5),
    "feature_norm.bias": (5, ),
    "feature_norm.weight": (5, ),
    "in_head.bias": (1024, ),
    "in_head.weight": (1024, 1024),
    "out_head.bias": (1024, ),
    "out_head.weight": (1024, 1024),
    "output_norm.weight": (1024, ),
    "pass_hidden_proj.bias": (1024, ),
    "pass_hidden_proj.weight": (1024, 4096),
    "rank_embedding": (1, 1, 1, 8, 1024),
    "relative_slot_bias": (8, 15),
    "same_slot_bias": (8, ),
    "slot_embedding": (1, 1, 7, 1, 1024),
    "token_proj.bias": (1024, ),
    "token_proj.weight": (1024, 4096),
    **{
        f"layers.{i}.{name}": shape
        for i in range(2)
        for name, shape in {
            "attn.in_proj_bias": (3072, ),
            "attn.in_proj_weight": (3072, 1024),
            "attn.out_proj.bias": (1024, ),
            "attn.out_proj.weight": (1024, 1024),
            "attn_norm.weight": (1024, ),
            "mlp.down_proj.bias": (1024, ),
            "mlp.down_proj.weight": (1024, 2048),
            "mlp.up_proj.bias": (2048, ),
            "mlp.up_proj.weight": (2048, 1024),
            "mlp_norm.weight": (1024, ),
        }.items()
    },
}

# Head modules the real draft stores as BF16 (its hf_quant_config exclude list).
_CHECKPOINT_HEAD_EXCLUDES = [
    "lilicorr.anchor_out_head", "lilicorr.factor_input_proj",
    "lilicorr.feature_mlp*", "lilicorr.in_head", "lilicorr.out_head"
]
_NVFP4_HEAD_LINEARS = {"token_proj", "pass_hidden_proj", "context_proj"} | {
    f"layers.{i}.{name}"
    for i in range(_LAYERS)
    for name in ("attn.out_proj", "mlp.up_proj", "mlp.down_proj")
}
_GROUP = 16


def _w4a16_quant() -> QuantConfig:
    return QuantConfig(quant_type=QUANT_NVFP4_A16,
                       group_size=_GROUP,
                       excluded=list(_CHECKPOINT_HEAD_EXCLUDES),
                       quantize_activations=False)


def _config(**overrides) -> ModelConfig:
    values = dict(hidden=_H,
                  d=_D,
                  heads=_HEADS,
                  layers=_LAYERS,
                  topk=_K,
                  factor=_FACTOR,
                  block=_BLOCK,
                  eps=_VEC_EPS,
                  sigma=_SIGMA)
    quant = overrides.pop("quant", None)
    values.update(overrides)
    extra = {} if quant is None else {"quant": quant}
    return ModelConfig(model_type="qwen3",
                       **extra,
                       hidden_size=values["hidden"],
                       num_hidden_layers=5,
                       num_attention_heads=4,
                       num_key_value_heads=2,
                       intermediate_size=96,
                       head_dim=12,
                       rms_norm_eps=_RMS_EPS,
                       vocab_size=_VOCAB,
                       rope_theta=10_000.0,
                       max_position_embeddings=4096,
                       default_attention_scale=1.0 / math.sqrt(12.0),
                       dflash2_block_size=values["block"],
                       dflash_projector_type="lilicorr",
                       lilicorr_candidate_topk=values["topk"],
                       lilicorr_hidden_size=values["d"],
                       lilicorr_num_layers=values["layers"],
                       lilicorr_num_heads=values["heads"],
                       lilicorr_mlp_ratio=2.0,
                       lilicorr_factor_dim=values["factor"],
                       lilicorr_vector_eps=values["eps"],
                       lilicorr_logit_scale=values["sigma"])


def _random_weights(seed: int = 0) -> dict:
    """FP32 head weights keyed by head-relative checkpoint names."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    mlp = 2 * _D
    w = {
        "token_proj.weight": rnd(_D, _H, scale=_H**-0.5),
        "token_proj.bias": rnd(_D, scale=0.1),
        "pass_hidden_proj.weight": rnd(_D, _H, scale=_H**-0.5),
        "pass_hidden_proj.bias": rnd(_D, scale=0.1),
        "feature_norm.weight": 1.0 + rnd(5, scale=0.1),
        "feature_norm.bias": rnd(5, scale=0.1),
        "feature_mlp.up_proj.weight": rnd(_D, 5, scale=0.5),
        "feature_mlp.up_proj.bias": rnd(_D, scale=0.1),
        "feature_mlp.down_proj.weight": rnd(_D, _D, scale=_D**-0.5),
        "feature_mlp.down_proj.bias": rnd(_D, scale=0.1),
        "slot_embedding": rnd(1, 1, _S, 1, _D, scale=0.5),
        "rank_embedding": rnd(1, 1, 1, _K, _D, scale=0.5),
        "relative_slot_bias": rnd(_HEADS, 2 * _BLOCK - 1),
        "same_slot_bias": rnd(_HEADS),
        "context_proj.weight": rnd(_D, _H, scale=_H**-0.5),
        "context_proj.bias": rnd(_D, scale=0.1),
        "output_norm.weight": 1.0 + rnd(_D, scale=0.1),
        "anchor_norm.weight": 1.0 + rnd(_D, scale=0.1),
        "factor_input_proj.weight": rnd(_D, 3 * _D, scale=(3 * _D)**-0.5),
        "factor_input_proj.bias": rnd(_D, scale=0.1),
        "out_head.weight": rnd(_FACTOR, _D, scale=_D**-0.5),
        "out_head.bias": rnd(_FACTOR, scale=0.1),
        "in_head.weight": rnd(_FACTOR, _D, scale=_D**-0.5),
        "in_head.bias": rnd(_FACTOR, scale=0.1),
        "anchor_out_head.weight": rnd(_FACTOR, _D, scale=_D**-0.5),
        "anchor_out_head.bias": rnd(_FACTOR, scale=0.1),
    }
    for i in range(_LAYERS):
        p = f"layers.{i}."
        w[p + "attn_norm.weight"] = 1.0 + rnd(_D, scale=0.1)
        w[p + "attn.in_proj_weight"] = rnd(3 * _D, _D, scale=_D**-0.5)
        w[p + "attn.in_proj_bias"] = rnd(3 * _D, scale=0.1)
        w[p + "attn.out_proj.weight"] = rnd(_D, _D, scale=_D**-0.5)
        w[p + "attn.out_proj.bias"] = rnd(_D, scale=0.1)
        w[p + "mlp_norm.weight"] = 1.0 + rnd(_D, scale=0.1)
        w[p + "mlp.up_proj.weight"] = rnd(mlp, _D, scale=_D**-0.5)
        w[p + "mlp.up_proj.bias"] = rnd(mlp, scale=0.1)
        w[p + "mlp.down_proj.weight"] = rnd(_D, mlp, scale=mlp**-0.5)
        w[p + "mlp.down_proj.bias"] = rnd(_D, scale=0.1)
    return w


def _random_inputs(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(_B, _S, _K, generator=g) * 2.0
    logits, _ = logits.sort(dim=-1, descending=True)
    return {
        "embedding": torch.randn(_VOCAB, _H, generator=g),
        "ids": torch.randint(0, _VOCAB, (_B, _S, _K), generator=g),
        "log_probs": logits - 1.5,
        "pass_hidden": torch.randn(_B, _S, _H, generator=g),
        "anchor_hidden": torch.randn(_B, _H, generator=g),
        "anchor_valid": torch.tensor([True, True]),
    }


def _reference_lilicorr_scores(w, embedding, ids, log_probs, pass_hidden,
                               anchor_hidden, anchor_valid):
    """Reference lattice-head transition scores, written out step by step in FP32."""

    def linear(name, x):
        return x @ w[name + ".weight"].T + w[name + ".bias"]

    def rms_norm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) +
                               _RMS_EPS) * weight

    def normalize(x):
        return F.normalize(x, dim=-1, eps=_VEC_EPS)

    batch, slots, topk = ids.shape
    lattice = slots * topk

    # Step 5: node states.
    x = linear("token_proj", embedding[ids])
    x = x + linear("pass_hidden_proj", pass_hidden)[:, :, None, :]
    rank = torch.arange(topk, dtype=torch.float32) / (topk - 1)
    top1 = (torch.arange(topk) == 0).to(torch.float32)
    features = torch.stack([
        log_probs,
        log_probs.exp(),
        log_probs - log_probs.max(dim=-1, keepdim=True).values,
        rank.expand_as(log_probs),
        top1.expand_as(log_probs),
    ],
                           dim=-1)
    features = F.layer_norm(features, (5, ), w["feature_norm.weight"],
                            w["feature_norm.bias"])
    x = x + linear("feature_mlp.down_proj",
                   F.silu(linear("feature_mlp.up_proj", features)))
    x = x + w["slot_embedding"].reshape(slots, 1, _D)
    x = x + w["rank_embedding"].reshape(1, topk, _D)

    # Step 6: lattice transformer.
    x = x.reshape(batch, lattice, _D)
    bias = torch.zeros(_HEADS, lattice, lattice)
    for h in range(_HEADS):
        for q in range(lattice):
            for k in range(lattice):
                sq, sk = q // topk, k // topk
                bias[h, q, k] = w["relative_slot_bias"][h,
                                                        sq - sk + _BLOCK - 1]
                if sq == sk:
                    bias[h, q, k] += w["same_slot_bias"][h]
    head_dim = _D // _HEADS
    for i in range(_LAYERS):
        p = f"layers.{i}."
        y = rms_norm(x, w[p + "attn_norm.weight"])
        qkv = y @ w[p + "attn.in_proj_weight"].T + w[p + "attn.in_proj_bias"]
        q, k, v = (t.reshape(batch, lattice, _HEADS, head_dim).transpose(1, 2)
                   for t in qkv.split(_D, dim=-1))
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim) +
                             bias,
                             dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(batch, lattice, _D)
        x = x + linear(p + "attn.out_proj", y)
        y = rms_norm(x, w[p + "mlp_norm.weight"])
        x = x + linear(p + "mlp.down_proj", F.silu(linear(
            p + "mlp.up_proj", y)))
    states = rms_norm(x,
                      w["output_norm.weight"]).reshape(batch, slots, topk, _D)

    # Step 7: fusion with the anchor.
    valid = anchor_valid.to(torch.float32)[:, None]
    anchor = rms_norm(
        linear("context_proj", anchor_hidden) * valid, w["anchor_norm.weight"])
    anchor_b = anchor[:, None, None, :].expand_as(states)
    fused = linear("factor_input_proj",
                   torch.cat([states, anchor_b, states * anchor_b], dim=-1))
    factor = F.silu(fused)

    # Step 8: outgoing and incoming vectors.
    out_vec = normalize(linear("out_head", factor))
    in_vec = normalize(linear("in_head", factor))
    anchor_out = normalize(linear("anchor_out_head", anchor))

    # Step 9: transition scores [b, slot, prev, next].
    start = _SIGMA * torch.einsum("bf,bkf->bk", anchor_out, in_vec[:, 0])
    pair = _SIGMA * torch.einsum("bspf,bskf->bspk", out_vec[:, :-1],
                                 in_vec[:, 1:])
    first = start[:, None, None, :].expand(batch, 1, topk, topk)
    return torch.cat([first, pair], dim=1)


def _fp32_head(weights: dict) -> LiLiCorrHead:
    head = LiLiCorrHead(_config()).float()
    head.load_state_dict(weights, strict=True)
    head.materialize()
    return head


def _token_embeddings(inputs: dict) -> torch.Tensor:
    return inputs["embedding"][inputs["ids"]]


def _run_head(head, weights, inputs, dtype=torch.float32):
    return head(
        _token_embeddings(inputs).to(dtype), inputs["log_probs"],
        inputs["pass_hidden"].to(dtype), inputs["anchor_hidden"].to(dtype),
        inputs["anchor_valid"])


def _run_reference(weights, inputs):
    return _reference_lilicorr_scores(weights, inputs["embedding"],
                                      inputs["ids"], inputs["log_probs"],
                                      inputs["pass_hidden"],
                                      inputs["anchor_hidden"],
                                      inputs["anchor_valid"])


# Rows whose reference top-1/top-2 gap is at most this are near-ties that FP16 rounding may legitimately flip.
_NEAR_TIE_GAP = 1e-3


def _assert_top1_matches_off_near_ties(out, ref):
    """Require the reference argmax on every row that is not a near-tie."""
    top2 = ref.topk(2, dim=-1).values
    decided = (top2[..., 0] - top2[..., 1]) > _NEAR_TIE_GAP
    assert decided.float().mean().item() > 0.9
    assert torch.equal(out.argmax(-1)[decided], ref.argmax(-1)[decided])


def test_head_matches_reference():
    """An FP32 head reproduces the reference transition scores and their argmax."""
    weights, inputs = _random_weights(), _random_inputs()
    with torch.no_grad():
        out = _run_head(_fp32_head(weights), weights, inputs)
        ref = _run_reference(weights, inputs)

    assert out.shape == (_B, _S, _K, _K)
    assert out.dtype == torch.float32
    assert (out - ref).abs().max().item() <= 1e-4
    assert torch.equal(out.argmax(-1), ref.argmax(-1))


def test_fp16_head_emits_fp32_scores():
    """A production-dtype FP16 head returns FP32 scores that track the FP32 reference and keep its decided argmax.

    The measured FP16 error reaches 4.9e-3 over 32 input seeds on CPU; 1e-2 leaves headroom for GPU kernels.
    """
    weights, inputs = _random_weights(), _random_inputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = _fp32_head(weights).half().to(device)
    head.materialize()
    inputs16 = {
        k: v.to(device)
        for k, v in inputs.items() if k not in ("embedding", "ids")
    }
    token_embeddings = _token_embeddings(inputs).half().to(device)
    with torch.no_grad():
        out = head(token_embeddings, inputs16["log_probs"],
                   inputs16["pass_hidden"].half(),
                   inputs16["anchor_hidden"].half(), inputs16["anchor_valid"])
        ref = _run_reference(weights, inputs)

    assert out.dtype == torch.float32
    _assert_top1_matches_off_near_ties(out.cpu(), ref)
    assert (out.cpu() - ref).abs().max().item() <= 1e-2


def test_slot0_rows_identical_over_prev():
    """Slot 0 broadcasts the anchor start scores over every predecessor row."""
    weights, inputs = _random_weights(), _random_inputs()
    with torch.no_grad():
        out = _run_head(_fp32_head(weights), weights, inputs)

    expected = out[:, 0, :1, :].expand(_B, _K, _K)
    assert torch.equal(out[:, 0], expected)
    assert not torch.equal(out[:, 1], out[:, 1, :1, :].expand(_B, _K, _K))


def _one_hot_rank_weights() -> dict:
    """Weights where the fused factor of candidate k is silu(sqrt(D)) e_k.

    Rank embedding k is a one-hot on dim k and every other contribution is
    zero, so output_norm(x[s, k]) = sqrt(D) e_k.
    """
    weights = {
        name: torch.zeros_like(t)
        for name, t in _random_weights().items()
    }
    for name in weights:
        if name.endswith("norm.weight"):
            weights[name] = torch.ones_like(weights[name])
    weights["rank_embedding"][0, 0, 0] = torch.eye(_K, _D)
    weights["factor_input_proj.weight"][:, :_D] = torch.eye(_D)
    return weights


def test_pair_layout_is_prev_then_next():
    """Pair scores are indexed [slot, prev candidate of slot s, next candidate of slot s+1]."""
    weights = _one_hot_rank_weights()
    # o[s, p] = e_p; i[s + 1, k] = e_{(k + 1) % K}: asymmetric under transpose.
    weights["out_head.weight"] = torch.eye(_FACTOR, _D)
    perm = torch.zeros(_FACTOR, _D)
    for k in range(_K):
        perm[(k + 1) % _K, k] = 1.0
    weights["in_head.weight"] = perm
    inputs = _random_inputs()
    with torch.no_grad():
        out = _run_head(_fp32_head(weights), weights, inputs)

    expected = torch.zeros(_K, _K)
    for p in range(_K):
        for k in range(_K):
            expected[p, k] = _SIGMA * float(p == (k + 1) % _K)
    for s in range(1, _S):
        assert torch.allclose(out[:, s],
                              expected.expand(_B, _K, _K),
                              atol=1e-5)


def test_vector_eps_clamps_short_vectors():
    """normalize() divides by max(||v||, lilicorr_vector_eps), so short vectors are not inflated."""
    weights = _one_hot_rank_weights()
    weights["out_head.weight"] = torch.eye(_FACTOR, _D)
    weights["in_head.bias"][0] = 0.1 * _VEC_EPS
    inputs = _random_inputs()
    with torch.no_grad():
        out = _run_head(_fp32_head(weights), weights, inputs)

    # o[s, p] = e_p and i[s + 1, k] = 0.1 e_0 for every k.
    expected = torch.zeros(_K, _K)
    expected[0, :] = 0.1 * _SIGMA
    for s in range(1, _S):
        assert torch.allclose(out[:, s],
                              expected.expand(_B, _K, _K),
                              atol=1e-5)


def test_invalid_anchor_zeroes_context():
    """anchor_valid=False zeroes the projected anchor before anchor_norm, independent of anchor_hidden."""
    weights, inputs = _random_weights(), _random_inputs()
    inputs["anchor_valid"] = torch.tensor([True, False])
    head = _fp32_head(weights)
    invalid_row_zeroed = dict(inputs)
    invalid_row_zeroed["anchor_hidden"] = inputs["anchor_hidden"].clone()
    invalid_row_zeroed["anchor_hidden"][1] = 0.0
    all_zeroed = dict(inputs)
    all_zeroed["anchor_hidden"] = torch.zeros_like(inputs["anchor_hidden"])
    with torch.no_grad():
        out = _run_head(head, weights, inputs)
        out_all_zeroed = _run_head(head, weights, all_zeroed)
        ref = _run_reference(weights, invalid_row_zeroed)

    assert (out - ref).abs().max().item() <= 1e-4
    assert torch.equal(out[1], out_all_zeroed[1])
    assert not torch.equal(out[0], out_all_zeroed[0])


def test_attention_bias_relative_and_same_slot():
    """Lattice bias is rel[h, slot(q)-slot(k)+BS-1] + same[h]*[slot(q)==slot(k)] with slot(i)=i//K."""
    g = torch.Generator().manual_seed(3)
    rel = torch.randn(_HEADS, 2 * _BLOCK - 1, generator=g).half()
    same = torch.randn(_HEADS, generator=g).half()
    bias = build_lattice_attention_bias(rel, same, _S, _K, _BLOCK)

    assert bias.shape == (_HEADS, _S * _K, _S * _K)
    assert bias.dtype == torch.float32
    rel32, same32 = rel.float(), same.float()
    for h in range(_HEADS):
        for q in range(_S * _K):
            for k in range(_S * _K):
                sq, sk = q // _K, k // _K
                expected = rel32[h, sq - sk + _BLOCK - 1]
                if sq == sk:
                    expected = expected + same32[h]
                assert bias[h, q, k].item() == pytest.approx(expected.item())


def test_forward_requires_materialized_bias():
    """forward refuses an unmaterialized lattice bias and uses the bias rebuilt by materialize()."""
    weights, inputs = _random_weights(), _random_inputs()
    head = LiLiCorrHead(_config()).float()
    head.load_state_dict(weights, strict=True)
    with pytest.raises(RuntimeError, match="materialize"):
        _run_head(head, weights, inputs)

    head.materialize()
    weights["relative_slot_bias"] = weights["relative_slot_bias"] + 5.0 * (
        torch.arange(2 * _BLOCK - 1, dtype=torch.float32) - _BLOCK + 1)
    with torch.no_grad():
        head.relative_slot_bias.copy_(weights["relative_slot_bias"])
        head.materialize()
        out = _run_head(head, weights, inputs)
        ref = _run_reference(weights, inputs)
    assert (out - ref).abs().max().item() <= 1e-4

    head.half()
    with pytest.raises(RuntimeError, match="materialize"):
        _run_head(head, weights, inputs, dtype=torch.float16)


def test_rejects_indivisible_head_width():
    """A lattice width that the head count does not divide is rejected at construction."""
    with pytest.raises(ValueError, match="divisible"):
        LiLiCorrHead(_config(heads=5))


def test_parameter_names_mirror_checkpoint():
    """Head parameters match the checkpoint 'lilicorr.' names and dense shapes."""
    config = _config(hidden=4096,
                     d=1024,
                     heads=8,
                     topk=8,
                     factor=1024,
                     block=8,
                     eps=1e-4,
                     sigma=8.0)
    with torch.device("meta"):
        head = LiLiCorrHead(config)

    shapes = {name: tuple(t.shape) for name, t in head.state_dict().items()}
    assert shapes == _CHECKPOINT_HEAD_SHAPES
    assert all(t.dtype == torch.float16 for t in head.state_dict().values())


def test_checkpoint_nvfp4_linears_stay_packed():
    """Under the checkpoint's W4A16 recipe the NVFP4-stored head linears are packed NVFP4A16Linear; BF16 ones are FP16."""
    head = LiLiCorrHead(_config(quant=_w4a16_quant()))

    linears = {
        name: type(module)
        for name, module in head.named_modules()
        if isinstance(module, (FP16Linear, NVFP4A16Linear))
    }
    packed = {name for name, kind in linears.items() if kind is NVFP4A16Linear}
    assert packed == _NVFP4_HEAD_LINEARS
    assert all(kind is FP16Linear for name, kind in linears.items()
               if name not in packed)
    assert "factor_input_proj" in linears


def _nvfp4_pack(weight: torch.Tensor) -> dict:
    """Quantize an FP32 ``[out, in]`` weight to ModelOpt NVFP4 buffers."""
    dense = weight.numpy().astype(np.float32)
    global_scale = float(np.abs(dense).max()) / (6.0 * 448.0)
    qweight, sf_bytes = _quantize_nvfp4_moe_weight(dense, _GROUP, global_scale)
    return {
        "weight": torch.from_numpy(qweight),
        "weight_scale": torch.from_numpy(sf_bytes).view(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor([global_scale], dtype=torch.float32),
    }


def _w4a16_eager_forward(self, hidden_states):
    """Eager stand-in for the W4A16 plugin: FP16 activation, NVFP4 weight, FP32 accumulate, FP16 output."""
    if hidden_states.dtype != torch.float16:
        raise TypeError("NVFP4A16Linear expects float16 input")
    weight = torch.from_numpy(
        decode_modelopt_nvfp4(self.weight, self.weight_scale,
                              self.weight_scale_2, self.group_size))
    out = F.linear(hidden_states.to(torch.float32), weight)
    return (out.to(torch.float16) + self.bias.to(torch.float16))


def test_w4a16_head_matches_reference_on_decoded_weights(monkeypatch):
    """A head with packed NVFP4 linears tracks the reference evaluated on the decoded NVFP4 weights.

    The reference (vLLM) runs these checkpoint linears as W4A16, so the oracle
    uses the decoded NVFP4 weights and the remaining error is FP16 activation
    rounding, the same 1e-2 bound as the dense FP16 head. The W4A16 GEMM runs
    as an eager stand-in; the plugin's own numerics are covered by its tests,
    not here.
    """
    monkeypatch.setattr(NVFP4A16Linear, "forward", _w4a16_eager_forward)
    weights, inputs = _random_weights(), _random_inputs()
    head = LiLiCorrHead(_config(quant=_w4a16_quant()))
    decoded = dict(weights)
    state = {}
    for name, value in weights.items():
        module_name, _, attr = name.rpartition(".")
        if module_name in _NVFP4_HEAD_LINEARS and attr == "weight":
            packed = _nvfp4_pack(value)
            for key, tensor in packed.items():
                state[f"{module_name}.{key}"] = tensor
            decoded[name] = torch.from_numpy(
                decode_modelopt_nvfp4(packed["weight"], packed["weight_scale"],
                                      packed["weight_scale_2"], _GROUP))
        elif module_name in _NVFP4_HEAD_LINEARS:
            state[name] = value
        else:
            state[name] = value.half()
    head.load_state_dict(state, strict=True)
    head.materialize()
    with torch.no_grad():
        out = _run_head(head, weights, inputs, dtype=torch.float16)
        ref = _run_reference(decoded, inputs)
        unquantized = _run_reference(weights, inputs)

    assert out.dtype == torch.float32
    _assert_top1_matches_off_near_ties(out, ref)
    assert (out - ref).abs().max().item() <= 1e-2
    assert (unquantized - ref).abs().max().item() > 1e-2


def test_direct_builder_head_matches_reference(tmp_path):
    """The checkpoint-direct LiLiCorr head network reproduces the reference transition scores and their decided argmax.

    FP16 checkpoint weights; the tolerance is the FP16 head test's.
    """
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires a CUDA device")
    from safetensors.torch import save_file

    from tensorrt_edgellm.builder.core.config import DeviceConfig
    from tensorrt_edgellm.builder.core.weights import Weights
    from tensorrt_edgellm.builder.models.lilicorr import \
        modeling_lilicorr_head as direct_head
    from tensorrt_edgellm.builder.ops import BuildContext, BuildOptions, Tensor
    from tensorrt_edgellm.builder.ops.backend import Net

    weights, inputs = _random_weights(), _random_inputs()
    save_file(
        {
            f"lilicorr.{name}": value.half().contiguous()
            for name, value in weights.items()
        }, str(tmp_path / "model.safetensors"))
    cfg = DeviceConfig(model_type="qwen3",
                       hidden_size=_H,
                       num_hidden_layers=5,
                       num_attention_heads=4,
                       num_key_value_heads=2,
                       head_dim=12,
                       intermediate_size=96,
                       vocab_size=_VOCAB,
                       rms_norm_eps=_RMS_EPS,
                       rope_theta=10_000.0,
                       max_position_embeddings=4096,
                       dflash2_block_size=_BLOCK,
                       lilicorr_candidate_topk=_K,
                       lilicorr_hidden_size=_D,
                       lilicorr_num_layers=_LAYERS,
                       lilicorr_num_heads=_HEADS,
                       lilicorr_mlp_ratio=2.0,
                       lilicorr_factor_dim=_FACTOR,
                       lilicorr_vector_eps=_VEC_EPS,
                       lilicorr_logit_scale=_SIGMA)
    feeds = {
        "token_embeddings": _token_embeddings(inputs).half(),
        "log_probs": inputs["log_probs"],
        "pass_hidden": inputs["pass_hidden"].half(),
        "anchor_hidden": inputs["anchor_hidden"].half(),
        "anchor_valid": inputs["anchor_valid"].float()[:, None],
    }
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    net = Net(builder, network)
    checkpoint = Weights(str(tmp_path))
    try:
        head = direct_head.LiLiCorrHead(
            BuildContext(net, cfg, checkpoint, BuildOptions(), None, None),
            "lilicorr")
        scores = head(*(Tensor(
            net.add_input(
                name, trt.float16 if value.dtype ==
                torch.float16 else trt.float32, tuple(value.shape)))
                        for name, value in feeds.items()))
        net.mark_output(scores, "scores")
    finally:
        checkpoint.close()
    plan = builder.build_serialized_network(network,
                                            builder.create_builder_config())
    context = trt.Runtime(logger).deserialize_cuda_engine(
        plan).create_execution_context()
    device_feeds = {name: value.cuda() for name, value in feeds.items()}
    out = torch.empty(_B, _S, _K, _K, device="cuda")
    for name, value in device_feeds.items():
        context.set_tensor_address(name, value.data_ptr())
    context.set_tensor_address("scores", out.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ref = _run_reference(weights, inputs)

    _assert_top1_matches_off_near_ties(out.cpu(), ref)
    assert (out.cpu() - ref).abs().max().item() <= 1e-2
