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
"""Direct-builder contracts for the opt-in TRT-native MoE arm.

GPU-free tests of the Edge-LLM side of ``--trt-native-moe``: the flag's
plumbing from the CLI to ``BuildOptions`` and ``build_engine``, the routing
graph built around ``INetworkDefinition.add_moe`` (rank-3 wrapping, softmax ->
top-k -> renormalisation), dispatch on the checkpoint's expert precision, the
``ValueError`` guards for what the layer cannot express, the weight
preparation (per-expert transpose and stacking for FP16, block-scale and
activation-scale folding for NVFP4 W4A8), the FP4 block-dequantize lowering
and the CUDA Tile workaround scoping. The TensorRT network is a recording
stub, so no engine is built and the MoE layer itself is not exercised: its
numerics are covered by TensorRT's own tests and by the few-layer validation
(``scripts/few-layer-validation.sh --trt-native-moe``).
"""

import contextlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder import cli, models
from tensorrt_edgellm.builder.core import builder, quantization
from tensorrt_edgellm.builder.core.builder import BuildArgs
from tensorrt_edgellm.builder.core.numpy_dtypes import fp8_e4m3_bytes_to_f32
from tensorrt_edgellm.builder.models import registry as model_registry
from tensorrt_edgellm.builder.models.nemotron_h.modeling_nemotron_h import \
    NemotronHMoE
from tensorrt_edgellm.builder.models.qwen3_5_moe import \
    modeling_qwen3_5_moe_sparse_moe
from tensorrt_edgellm.builder.models.qwen3_moe import modeling_qwen3_moe
from tensorrt_edgellm.builder.ops import functional as functional_pkg
from tensorrt_edgellm.builder.ops import moe as moe_ops
from tensorrt_edgellm.builder.ops.backend import Net
from tensorrt_edgellm.builder.ops.functional import moe as functional_moe
from tensorrt_edgellm.builder.ops.module import (BuildContext, BuildOptions,
                                                 NetworkModule)
from tensorrt_edgellm.builder.ops.scope import build_scope
from tensorrt_edgellm.builder.ops.tensor import Tensor

# Tiny expert bank: E experts, top-K, hidden H, intermediate I. H != I so a
# transposed weight cannot pass on shape alone.
E, K, H, I = 4, 2, 32, 16
DIMS = (E, K, H, I)
SOFTMAX_TOPK = functional_moe.MoeRouting.SOFTMAX_TOPK


def _fp16_bank():
    return (np.zeros((E, H, I), np.float16), np.zeros(
        (E, H, I), np.float16), np.zeros((E, I, H), np.float16))


def _w4a8_bank():
    bank = {
        "fc1_activation_scale": np.array(0.5, np.float16),
        "fc_down_activation_scales": np.ones(E, np.float32)
    }
    for name, (out, inp) in (("gate", (I, H)), ("up", (I, H)), ("down", (H,
                                                                         I))):
        bank[f"fc_{name}_packed"] = np.zeros((E, out, inp // 2), np.uint8)
        bank[f"fc_{name}_block_scale"] = np.ones((E, out, inp // 16),
                                                 np.float32)
    return bank


def _cfg(**overrides):
    cfg = dict(num_experts=E,
               num_experts_per_tok=K,
               hidden_size=H,
               moe_intermediate_size=I,
               group_size=16,
               n_group=1,
               topk_group=1,
               routed_scaling_factor=1.0)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _experts(prefix):
    """GatedExperts stand-in: the blocks and the native arm key the
    expert precision through ``experts.key("0.up_proj")``."""
    return SimpleNamespace(prefix=prefix,
                           key=lambda suffix: f"{prefix}.{suffix}")


def _weights(quant_type, fallback=False):
    return SimpleNamespace(module_quant_type=lambda name: quant_type,
                           uses_w4a16_fallback=lambda kind: fallback)


class _FakeTrtTensor:

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


def _unwrap(value):
    return value._as_trt() if hasattr(value, "_as_trt") else value


class _RoutingNet:
    """Records the graph the functional ops build around the native layer."""

    def __init__(self):
        self.calls, self.moe = [], None

    def _like(self, x, shape=None, dtype=None):
        x = _unwrap(x)
        return _FakeTrtTensor(x.shape if shape is None else shape,
                              x.dtype if dtype is None else dtype)

    def softmax(self, x, axis):
        self.calls.append(("softmax", axis))
        return self._like(x)

    def topk(self, x, k, axis):
        self.calls.append(("topk", k, axis))
        shape = _unwrap(x).shape[:axis] + (k, )
        return self._like(x, shape), self._like(x, shape, trt.int32)

    def reduce(self, x, op, axes, keep_dims):
        self.calls.append(("reduce", op, axes, keep_dims))
        return self._like(x, _unwrap(x).shape[:-1] + (1, ))

    def elementwise(self, a, b, op):
        self.calls.append(("elementwise", op))
        return self._like(a)

    def reshape(self, x, shape):
        return self._like(x, shape)

    def trt_native_moe(self, hidden, indices, scores, gate, up, down, **kw):
        self.moe = dict(hidden=_unwrap(hidden),
                        indices=_unwrap(indices),
                        scores=_unwrap(scores),
                        gate=gate,
                        up=up,
                        down=down,
                        **kw)
        return self._like(hidden)


@pytest.fixture
def routing_net(monkeypatch):
    """A recording Net; Tensor arithmetic goes straight to it because its
    fake tensors are not trt.ITensor."""
    net = _RoutingNet()
    monkeypatch.setattr(
        Tensor, "_binary", lambda self, other, op: Tensor(
            net.elementwise(self._value, Tensor.unwrap(other), op)))
    return net


@pytest.mark.parametrize("variant, norm_topk_prob", [("fp16", 1), ("fp16", 0),
                                                     ("w4a8", 1)])
def test_trt_native_moe_builds_rank3_routing_around_the_layer(
        routing_net, variant, norm_topk_prob):
    """Softmax -> top-k (-> renormalise when norm_topk_prob) over the experts
    axis, rank-3 [1, T, *] wrapping for the layer, token-major [T, H] out.
    FP16 passes [E, H, I] / [E, I, H] arrays; W4A8 passes (FP4, scale) pairs
    plus the FC1 and per-expert FC2 activation scales."""
    router = Tensor(_FakeTrtTensor((-1, E), trt.float32))
    hidden = Tensor(_FakeTrtTensor((-1, H), trt.float16))
    ctx = BuildContext(net=routing_net,
                       cfg=None,
                       weights=None,
                       options=BuildOptions(),
                       bundle=None,
                       args=None)
    bank = _w4a8_bank()
    kwargs = dict(norm_topk_prob=norm_topk_prob, weight_prefix="experts")
    with build_scope(ctx):
        if variant == "fp16":
            output = functional_moe.trt_native_moe_fp16(
                router, hidden, *_fp16_bank(), *DIMS, **kwargs)
        else:
            output = functional_moe.trt_native_moe_w4a8(
                router, hidden, bank, *DIMS, **kwargs)

    calls, moe = routing_net.calls, routing_net.moe
    assert ("softmax", 1) in calls and ("topk", K, 1) in calls
    renormalised = bool(norm_topk_prob)
    assert (("reduce", trt.ReduceOperation.SUM, 1 << 1, True)
            in calls) is renormalised
    assert (("elementwise", trt.ElementWiseOperation.DIV)
            in calls) is renormalised
    assert moe["hidden"].shape == (1, -1, H)
    assert (moe["indices"].shape, moe["indices"].dtype) == ((1, -1, K),
                                                            trt.int32)
    assert (moe["scores"].shape, moe["scores"].dtype) == ((1, -1, K),
                                                          trt.float32)
    assert moe["name"] == "experts.trt_native_moe"
    assert output.shape == (-1, H)
    if variant == "fp16":
        assert moe["gate"].shape == (E, H, I)
        assert moe["down"].shape == (E, I, H)
        assert "fc1_activation_scale" not in moe
    else:
        assert moe["gate"][0] is bank["fc_gate_packed"]
        assert moe["gate"][1] is bank["fc_gate_block_scale"]
        assert moe["down"][0].shape == (E, H, I // 2)
        assert moe["fc1_activation_scale"].dtype == np.float16
        assert moe["fc_down_activation_scales"].shape == (E, )


@pytest.mark.parametrize("quant_type, prepare, op", [
    ("fp16", "prepare_gated_trt_native_fp16_weights", "trt_native_moe_fp16"),
    ("nvfp4", "prepare_gated_trt_native_w4a8_weights", "trt_native_moe_w4a8"),
])
def test_trt_native_gated_moe_dispatches_on_expert_precision(
        monkeypatch, quant_type, prepare, op):
    """FP16 experts take the FP16 layer and NVFP4 experts the W4A8 form; both
    receive the block's norm_topk_prob and the expert prefix."""
    experts = _experts("model.layers.3.mlp.experts")
    router, hidden = object(), object()
    bank = {
        name: object()
        for name in ("fc_gate_weights", "fc_up_weights", "fc_down_weights")
    }
    seen = {}
    monkeypatch.setattr(moe_ops, prepare, lambda ctx, e: bank)
    monkeypatch.setattr(
        moe_ops.F, op, lambda r, h, *args, **kwargs: seen.update(
            router=r, hidden=h, args=args, kwargs=kwargs) or "routed")
    ctx = SimpleNamespace(cfg=_cfg(),
                          weights=_weights(quant_type),
                          options=BuildOptions(trt_native_moe=True))

    result = moe_ops.trt_native_gated_moe(ctx,
                                          experts,
                                          router,
                                          hidden,
                                          norm_topk_prob=0)

    assert result == "routed"
    assert seen["router"] is router and seen["hidden"] is hidden
    assert seen["args"][-4:] == DIMS
    assert seen["kwargs"] == {
        "norm_topk_prob": 0,
        "weight_prefix": experts.prefix
    }
    expected = (bank, ) if quant_type == "nvfp4" else tuple(bank.values())
    assert seen["args"][:-4] == expected


@pytest.mark.parametrize(
    "quant_type, fallback, overrides, routing, needs", [
        ("int4_gptq", False, {}, SOFTMAX_TOPK,
         "only FP16 and NVFP4 experts have a TRT_MoE form"),
        ("nvfp4", True, {}, SOFTMAX_TOPK, "executed as W4A16 on this device"),
        ("fp16", False, {}, functional_moe.MoeRouting.SIGMOID_GROUP_TOPK,
         "routing_mode=SIGMOID_GROUP_TOPK"),
        ("fp16", False, {
            "n_group": 8,
            "topk_group": 4
        }, SOFTMAX_TOPK, "n_group=8, topk_group=4"),
        ("fp16", False, {
            "routed_scaling_factor": 2.5
        }, SOFTMAX_TOPK, "routed_scaling_factor=2.5"),
    ],
    ids=["int4", "nvfp4_w4a16_fallback", "sigmoid", "group_topk", "scaled"])
def test_trt_native_gated_moe_rejects_what_it_does_not_build(
        monkeypatch, quant_type, fallback, overrides, routing, needs):
    """Only FP16 / activation-quantized NVFP4 experts and plain softmax ->
    top-k routing have a native form; anything else fails before any expert
    weight is read, naming what it needs. Never a silent plugin fallback."""
    for prepare in ("prepare_gated_trt_native_fp16_weights",
                    "prepare_gated_trt_native_w4a8_weights"):
        monkeypatch.setattr(moe_ops, prepare,
                            lambda ctx, e: pytest.fail("weights read"))
    ctx = SimpleNamespace(cfg=_cfg(**overrides),
                          weights=_weights(quant_type, fallback),
                          options=BuildOptions(trt_native_moe=True))

    with pytest.raises(ValueError, match=re.escape(needs)):
        moe_ops.trt_native_gated_moe(ctx,
                                     _experts("experts"),
                                     object(),
                                     object(),
                                     norm_topk_prob=1,
                                     routing_mode=routing)


class _StubGatedExperts:
    """GatedExperts stand-in serving known [out, in] matrices and scales."""

    prefix = "model.layers.0.mlp.experts"

    def key(self, suffix):
        return f"{self.prefix}.{suffix}"

    def __init__(self, dense=None, raw=None, fc1_scales=(), fc2_scales=()):
        self._dense, self._raw = dense, raw
        self._scales = (np.asarray(fc1_scales, np.float32),
                        np.asarray(fc2_scales, np.float32))

    def load_expert_dense(self, expert_index):
        return self._dense[expert_index]

    def load_expert_raw_nvfp4(self, expert_index):
        return self._raw[expert_index]

    def input_scales(self):
        return self._scales


def _prepare_context(group_size=16):
    return SimpleNamespace(
        cfg=_cfg(group_size=group_size),
        weights=SimpleNamespace(
            module_quant_group_size=lambda name: group_size))


def test_prepare_trt_native_fp16_weights_transposes_and_stacks_each_expert():
    """The checkpoint stores [out, in]; the native layer wants [E, H, I] for
    gate/up and [E, I, H] for down."""
    rng = np.random.default_rng(0)
    dense = [{
        "gate": rng.standard_normal((I, H), dtype=np.float32),
        "up": rng.standard_normal((I, H), dtype=np.float32),
        "down": rng.standard_normal((H, I), dtype=np.float32),
    } for _ in range(E)]

    bank = moe_ops.prepare_gated_trt_native_fp16_weights(
        _prepare_context(), _StubGatedExperts(dense=dense))

    assert bank["fc_gate_weights"].shape == (E, H, I)
    assert bank["fc_down_weights"].shape == (E, I, H)
    for name in ("gate", "up", "down"):
        assert bank[f"fc_{name}_weights"].dtype == np.float16
        np.testing.assert_array_equal(
            bank[f"fc_{name}_weights"],
            np.stack([dense[e][name].T for e in range(E)]).astype(np.float16))


# FP8 E4M3 encodings of 1.0 and 2.0.
_E4M3_ONE, _E4M3_TWO = 0x38, 0x40


def _raw_nvfp4_experts(rng, alphas, fc1_scales):
    """Provider-packed NVFP4 experts as load_expert_raw_nvfp4 returns them:
    [out, in // 2] nibbles (int8 storage on odd experts), [out, in // 16] raw
    E4M3 block scales, the per-expert alpha and FC1 input scale."""

    def projection(expert, out, inp):
        packed = rng.integers(0, 256, (out, inp // 2), dtype=np.uint8)
        return {
            "packed":
            packed.view(np.int8) if expert % 2 else packed,
            "sf":
            rng.choice(np.array([_E4M3_ONE, _E4M3_TWO], np.uint8),
                       (out, inp // 16)),
            "alpha":
            alphas[expert],
            "input_scale":
            fc1_scales[expert]
        }

    return [{
        "gate": projection(e, I, H),
        "up": projection(e, I, H),
        "down": projection(e, H, I)
    } for e in range(E)]


def test_prepare_trt_native_w4a8_weights_folds_block_and_activation_scales():
    """Nibbles pass through (int8 storage viewed as uint8); block scale =
    e4m3(sf) * alpha per expert; ModelOpt input scales (amax / (6 * 448)) x 6
    give the FP8 scales: FC1 shared as the max over experts, FC2 per expert."""
    alphas = [0.5, 0.25, 2.0, 1.0]
    fc1_scales = [0.010, 0.020, 0.015, 0.020]
    fc2_scales = [0.1, 0.2, 0.3, 0.4]
    raw = _raw_nvfp4_experts(np.random.default_rng(1), alphas, fc1_scales)

    bank = moe_ops.prepare_gated_trt_native_w4a8_weights(
        _prepare_context(),
        _StubGatedExperts(raw=raw,
                          fc1_scales=fc1_scales,
                          fc2_scales=fc2_scales))

    assert bank["fc_gate_packed"].shape == (E, I, H // 2)
    assert bank["fc_down_packed"].shape == (E, H, I // 2)
    assert bank["fc_gate_block_scale"].shape == (E, I, H // 16)
    assert bank["fc_down_block_scale"].shape == (E, H, I // 16)
    for expert in range(E):
        for name in ("gate", "up", "down"):
            packed = bank[f"fc_{name}_packed"]
            assert packed.dtype == np.uint8
            np.testing.assert_array_equal(
                packed[expert], raw[expert][name]["packed"].view(np.uint8))
            block_scale = bank[f"fc_{name}_block_scale"]
            assert block_scale.dtype == np.float32
            np.testing.assert_array_equal(
                block_scale[expert],
                fp8_e4m3_bytes_to_f32(raw[expert][name]["sf"]) *
                np.float32(alphas[expert]))
    assert bank["fc1_activation_scale"].dtype == np.float16
    assert bank["fc1_activation_scale"] == np.float16(
        float(np.asarray(fc1_scales, np.float32).max()) * 6.0)
    assert bank["fc_down_activation_scales"].dtype == np.float32
    np.testing.assert_array_equal(bank["fc_down_activation_scales"],
                                  (np.asarray(fc2_scales, np.float32) *
                                   6.0).astype(np.float32))


def test_prepare_trt_native_w4a8_weights_names_the_expert_with_another_block():
    """A mixed-precision checkpoint can leave one expert in another format;
    its stored scale shape says so and the error names the tensor."""
    raw = _raw_nvfp4_experts(np.random.default_rng(3), [1.0] * E, [0.02] * E)
    raw[1]["up"]["sf"] = raw[1]["up"]["sf"][:, ::2]  # [I, H // 32]

    with pytest.raises(ValueError,
                       match=r"experts\.1\.up_proj: FP8 block scale shape .* "
                       r"block size 32, need 16"):
        moe_ops.prepare_gated_trt_native_w4a8_weights(
            _prepare_context(),
            _StubGatedExperts(raw=raw,
                              fc1_scales=[0.02] * E,
                              fc2_scales=[0.1] * E))


class _Sym:
    """Stands in for a symbolic Tensor where a block only combines results."""

    def __add__(self, other):
        return self

    __mul__ = __add__

    def sigmoid(self):
        return self


@pytest.mark.parametrize("module, block_cls", [
    (modeling_qwen3_moe, modeling_qwen3_moe.Qwen3MoeSparseMoeBlock),
    (modeling_qwen3_5_moe_sparse_moe,
     modeling_qwen3_5_moe_sparse_moe.Qwen3_5MoeSparseMoeBlock),
],
                         ids=["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("flag", [True, False])
def test_sparse_moe_blocks_take_the_native_arm_only_with_the_flag(
        monkeypatch, module, block_cls, flag):
    """The Qwen3 MoE blocks hand router logits, hidden states, norm_topk_prob
    and routing mode to trt_native_gated_moe when the flag is on and keep
    their plugin dispatch when it is off."""
    hidden, logits = object(), object()
    block = block_cls.__new__(block_cls)
    block.ctx = SimpleNamespace(options=BuildOptions(trt_native_moe=flag),
                                cfg=_cfg(),
                                weights=_weights(quantization.QUANT_NVFP4_A16))
    block.prefix = "model.layers.0.mlp"
    block.gate = lambda hidden_states: logits
    block.experts = _experts("model.layers.0.mlp.experts")
    block.activation_type = functional_moe.MoeActivation.SWIGLU
    block.routing_mode = SOFTMAX_TOPK
    block.n_group = block.topk_group = 1
    block.routed_scaling_factor = 1.0
    block.norm_topk_prob = 0
    block.shared_expert = block.shared_expert_gate = lambda x: _Sym()
    native, plugin = [], []
    monkeypatch.setattr(
        module, "trt_native_gated_moe", lambda *args, **kwargs: native.append(
            (args, kwargs)) or _Sym())
    monkeypatch.setattr(module, "prepare_gated_nvfp4_a16_weights",
                        lambda ctx, experts: ({
                            "padded_intermediate": I
                        }, {}))
    monkeypatch.setattr(functional_pkg, "nvfp4_a16_moe",
                        lambda *args, **kwargs: plugin.append(args) or _Sym())

    block.forward(hidden)

    if flag:
        assert native == [((block.ctx, block.experts, logits, hidden), {
            "norm_topk_prob": 0,
            "routing_mode": SOFTMAX_TOPK
        })]
        assert plugin == []
    else:
        assert native == []
        assert len(plugin) == 1 and plugin[0][:2] == (logits, hidden)


def test_nemotron_h_moe_rejects_trt_native_moe():
    """Nemotron-H's ReLU2 / sigmoid-group-top-k MoE has no native form: the
    flag is an error at module construction, before any weight is read."""
    ctx = SimpleNamespace(options=BuildOptions(trt_native_moe=True),
                          cfg=SimpleNamespace(quant_type="nvfp4"),
                          weights=SimpleNamespace())

    with pytest.raises(ValueError, match="no TRT_MoE form"):
        NemotronHMoE(ctx, "backbone.layers.1.mixer")


class _RecordingNetwork:
    """INetworkDefinition stand-in for the FP4 block-dequantize lowering."""

    def __init__(self):
        self.constants, self.dequantize = [], []

    def add_constant(self, shape, weights):
        self.constants.append((tuple(shape), weights))
        return SimpleNamespace(get_output=lambda index: "const", name=None)

    def add_dequantize(self, tensor, scale, output_type):
        self.dequantize.append(
            SimpleNamespace(output_type=output_type,
                            get_output=lambda index: "dequantized"))
        return self.dequantize[-1]


def test_fp4_block_dequantize_lowers_fp16_block_scales_without_an_axis():
    """Myelin's MoE block-dequantize compiles only for FP4 x FP16 scales with
    block shape [1, 1, 16] and the default axis (fp32 scales or a per-axis
    setting fail with 'Could not infer output types'); the lowering casts
    the host-side fp32 scales accordingly."""
    network = _RecordingNetwork()
    net = Net(SimpleNamespace(), network)
    packed = np.zeros((E, I, H // 2), np.uint8)
    block_scale = np.full((E, I, H // 16), 0.0078125, np.float32)

    output = net._fp4_block_dequantize(packed, block_scale, "moe_gate_fp4")

    assert output == "dequantized"
    (weight_shape, weight), (scale_shape, scale) = network.constants
    assert weight_shape == (E, I, H) and weight.dtype == trt.DataType.FP4
    assert scale_shape == (E, I, H // 16) and scale.dtype == trt.float16
    layer, = network.dequantize
    assert layer.output_type == trt.float16
    assert list(layer.block_shape) == [1, 1, 16]
    assert not hasattr(layer, "axis")


def _native_moe_build_args():
    return BuildArgs(model_dir="ckpt", engine_dir="out", trt_native_moe=True)


def _trt_stub(with_native_moe=True, version="11.3.0.89"):
    """tensorrt stand-in: INetworkDefinition with or without add_moe."""
    network = SimpleNamespace(
        add_moe=object()) if with_native_moe else SimpleNamespace()
    return SimpleNamespace(__version__=version, INetworkDefinition=network)


def test_flag_reaches_build_options_through_cli_and_build_model(monkeypatch):
    """--trt-native-moe lands in BuildArgs (default off) and build_model
    copies it into the BuildOptions every Module reads via ctx.options."""
    parser = cli._parser()
    base = ["--model-dir", "ckpt", "--engine-dir", "out"]
    assert cli._build_args(parser.parse_args(base),
                           "llm").trt_native_moe is False
    args = cli._build_args(parser.parse_args(base + ["--trt-native-moe"]),
                           "llm")
    assert args.trt_native_moe is True

    seen = {}

    class _Probe(NetworkModule):

        def forward(self):
            seen["options"] = self.ctx.options

    monkeypatch.setattr(builder, "_active_cuda_compute_capability", lambda:
                        (10, 0))
    monkeypatch.setattr(model_registry, "definition_for",
                        lambda *args: SimpleNamespace(load=lambda: _Probe))
    models.build_model(SimpleNamespace(),
                       SimpleNamespace(root_model_type="qwen3_moe"),
                       SimpleNamespace(), SimpleNamespace(), args)

    assert seen["options"].trt_native_moe is True


@pytest.mark.parametrize("compute_capability", [(10, 0), (11, 0)])
def test_build_engine_scopes_the_workarounds_with_the_flag(
        monkeypatch, compute_capability):
    """On SM 10.x / 11.x with TensorRT >= 11.3 the flag builds and the
    compile-workaround scope is told the network uses the native layer so
    CUDA Tile stays on (flag set: test_direct_builder_checkpoint_contract)."""
    seen = {}

    @contextlib.contextmanager
    def fake_scope(max_batch_size, use_trt_native_moe=False):
        seen["use_trt_native_moe"] = use_trt_native_moe
        yield ""

    monkeypatch.setattr(builder, "trt", _trt_stub())
    monkeypatch.setattr(builder, "_active_cuda_compute_capability",
                        lambda: compute_capability)
    monkeypatch.setattr(builder, "_compile_workarounds", fake_scope)
    monkeypatch.setattr(builder, "_build_engine", lambda *args: "engine")

    assert builder.build_engine(_native_moe_build_args()) == "engine"
    assert seen["use_trt_native_moe"] is True


@pytest.mark.parametrize("trt_stub, compute_capability, needs", [
    (_trt_stub(with_native_moe=False, version="10.16.0.12"),
     (10, 0), r"TensorRT >= 11\.3.*found TensorRT 10\.16\.0\.12"),
    (_trt_stub(), (8, 9), r"SM 10\.x or 11\.x.*SM 8\.9"),
    (_trt_stub(), (12, 0), r"SM 10\.x or 11\.x.*SM 12\.0"),
],
                         ids=["trt_10", "sm89", "sm120"])
def test_build_engine_rejects_the_flag_without_the_native_layer(
        monkeypatch, trt_stub, compute_capability, needs):
    """TensorRT before 11.3 has no IMoELayer and the layer runs on SM 10.x /
    11.x only: the flag fails with a ValueError naming the requirement before
    any weight is read, not inside the TensorRT compiler."""
    monkeypatch.setattr(builder, "trt", trt_stub)
    monkeypatch.setattr(builder, "_active_cuda_compute_capability",
                        lambda: compute_capability)
    monkeypatch.setattr(builder, "_build_engine",
                        lambda *args: pytest.fail("build started"))

    with pytest.raises(ValueError, match=needs):
        builder.build_engine(_native_moe_build_args())
