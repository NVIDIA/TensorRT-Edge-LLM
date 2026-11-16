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
"""Behavioral contracts for the Gemma4 Flax checkpoint adapter."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from experimental.checkpoint_converters.models.gemma4 import \
    flax_adapter as adapter_module
from experimental.checkpoint_converters.models.gemma4.flax_adapter import \
    Gemma4FlaxAdapter
from experimental.checkpoint_converters.models.gemma4.variant import \
    A4B_26B_IT_VARIANT
from experimental.checkpoint_converters.schema import (ShapeMismatch,
                                                       UnmappedSourceTensor)
from experimental.checkpoint_converters.transforms import (MappingPlan,
                                                           Transform,
                                                           TransformKind)


def test_adapter_selectors_encode_split_and_transpose_layouts():
    """Fused attention/MLP and PLE leaves select the reviewed exact transforms."""
    adapter = Gemma4FlaxAdapter()
    kv = adapter.select_rule(("layer_0", "attn", "kv_einsum", "w"),
                             (2, 1, 1536, 256))[0]
    gate = adapter.select_rule(("layer_15", "mlp", "gating_einsum", "w"),
                               (2, 12288, 1536))[0]
    ple = adapter.select_rule(("layer_0", "per_layer_input_gate", "w"),
                              (1536, 256))[0]

    assert kv.transform.kind == TransformKind.SPLIT_TRANSPOSE_RESHAPE
    assert kv.target_shapes == ((256, 1536), (256, 1536))
    assert kv.target_names == (
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    )
    assert gate.transform.kind == TransformKind.SPLIT_TRANSPOSE_RESHAPE
    assert gate.target_shapes == ((12288, 1536), (12288, 1536))
    assert ple.transform.kind == TransformKind.TRANSPOSE
    assert ple.target_shapes == ((256, 1536), )


@pytest.mark.parametrize(
    ("source_path", "source_shape", "target_names", "target_shapes",
     "transform"),
    [
        (
            ("embedder", "per_layer_embeddings"),
            (262144, 35, 256),
            ("model.language_model.embed_tokens_per_layer.weight", ),
            ((262144, 8960), ),
            Transform(TransformKind.RESHAPE, output_shapes=((262144, 8960), )),
        ),
        (
            ("embedder", "per_layer_model_projection", "w"),
            (1536, 35, 256),
            ("model.language_model.per_layer_model_projection.weight", ),
            ((8960, 1536), ),
            Transform(TransformKind.TRANSPOSE_RESHAPE,
                      axes=(1, 2, 0),
                      output_shapes=((8960, 1536), )),
        ),
        (
            ("layer_0", "attn", "q_einsum", "w"),
            (8, 1536, 256),
            ("model.language_model.layers.0.self_attn.q_proj.weight", ),
            ((2048, 1536), ),
            Transform(TransformKind.TRANSPOSE_RESHAPE,
                      axes=(0, 2, 1),
                      output_shapes=((2048, 1536), )),
        ),
        (
            ("layer_0", "attn", "kv_einsum", "w"),
            (2, 1, 1536, 256),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "model.language_model.layers.0.self_attn.v_proj.weight",
            ),
            ((256, 1536), (256, 1536)),
            Transform(TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                      axes=(0, 2, 1),
                      output_shapes=((256, 1536), (256, 1536))),
        ),
        (
            ("layer_0", "attn", "attn_vec_einsum", "w"),
            (8, 256, 1536),
            ("model.language_model.layers.0.self_attn.o_proj.weight", ),
            ((1536, 2048), ),
            Transform(TransformKind.TRANSPOSE_RESHAPE,
                      axes=(2, 0, 1),
                      output_shapes=((1536, 2048), )),
        ),
        (
            ("layer_15", "mlp", "gating_einsum", "w"),
            (2, 12288, 1536),
            (
                "model.language_model.layers.15.mlp.gate_proj.weight",
                "model.language_model.layers.15.mlp.up_proj.weight",
            ),
            ((12288, 1536), (12288, 1536)),
            Transform(TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                      axes=(0, 1),
                      output_shapes=((12288, 1536), (12288, 1536))),
        ),
        (
            ("layer_15", "mlp", "linear", "w"),
            (12288, 1536),
            ("model.language_model.layers.15.mlp.down_proj.weight", ),
            ((1536, 12288), ),
            Transform(TransformKind.TRANSPOSE,
                      axes=(1, 0),
                      output_shapes=((1536, 12288), )),
        ),
        (
            ("layer_0", "per_layer_input_gate", "w"),
            (1536, 256),
            ("model.language_model.layers.0.per_layer_input_gate.weight", ),
            ((256, 1536), ),
            Transform(TransformKind.TRANSPOSE,
                      axes=(1, 0),
                      output_shapes=((256, 1536), )),
        ),
        (
            ("layer_0", "per_layer_projection", "w"),
            (256, 1536),
            ("model.language_model.layers.0.per_layer_projection.weight", ),
            ((1536, 256), ),
            Transform(TransformKind.TRANSPOSE,
                      axes=(1, 0),
                      output_shapes=((1536, 256), )),
        ),
    ],
)
def test_adapter_selectors_preserve_all_nonidentity_layouts(
        source_path, source_shape, target_names, target_shapes, transform):
    """Every non-identity selector fixes target names, shapes, kind, and axes."""
    rule = Gemma4FlaxAdapter().select_rule(source_path, source_shape)[0]

    assert (rule.target_names, rule.target_shapes,
            rule.transform) == (target_names, target_shapes, transform)


def test_adapter_plan_flattens_selects_and_validates_complete_tree(
        monkeypatch):
    """Plan maps the canonical flattened tree and returns validated source order."""
    adapter = Gemma4FlaxAdapter()
    source_path = ("final_norm", "scale")
    source_shape = (1536, )
    flat_tree = {source_path: SimpleNamespace(shape=source_shape)}
    expected = MappingPlan(
        tuple(adapter.select_rule(source_path, source_shape)),
        adapter.variant.name)
    source_tree = object()
    flatten_calls = []
    validation_calls = []
    traverse_util = ModuleType("flax.traverse_util")

    def flatten_dict(value):
        flatten_calls.append(value)
        return flat_tree

    def validate(plan, tree, variant):
        validation_calls.append((plan, tree, variant))
        return expected

    traverse_util.flatten_dict = flatten_dict
    flax = ModuleType("flax")
    flax.traverse_util = traverse_util
    monkeypatch.setitem(sys.modules, "flax", flax)
    monkeypatch.setitem(sys.modules, "flax.traverse_util", traverse_util)
    monkeypatch.setattr(adapter_module, "validate_mapping_plan", validate)

    plan = adapter.plan(source_tree)

    assert flatten_calls == [source_tree]
    assert len(validation_calls) == 1
    assert validation_calls[0][1:] == (flat_tree, adapter.variant)
    assert plan == expected


def test_adapter_rejects_unknown_paths_and_source_shapes():
    """Unknown or shape-incompatible Flax leaves fail before any output write."""
    adapter = Gemma4FlaxAdapter()

    with pytest.raises(UnmappedSourceTensor, match="unmapped source"):
        adapter.select_rule(("invented", "weight"), (1, ))
    with pytest.raises(ShapeMismatch, match="expected"):
        adapter.select_rule(("layer_4", "attn", "q_einsum", "w"),
                            (8, 1536, 256))


def test_26b_selectors_encode_optional_component_layouts():
    """The 26B-only attention, MoE, and norm layouts map exactly."""
    cases = (
        (
            ("layer_5", "attn", "k_einsum", "w"),
            (2, 2816, 512),
            ("model.language_model.layers.5.self_attn.k_proj.weight", ),
            Transform(TransformKind.TRANSPOSE_RESHAPE,
                      axes=(0, 2, 1),
                      output_shapes=((1024, 2816), )),
        ),
        (
            ("layer_0", "mlp", "gating_einsum", "w"),
            (128, 2, 704, 2816),
            ("model.language_model.layers.0.experts.gate_up_proj", ),
            Transform(TransformKind.RESHAPE,
                      output_shapes=((128, 1408, 2816), )),
        ),
        (
            ("layer_0", "mlp", "linear", "w"),
            (128, 704, 2816),
            ("model.language_model.layers.0.experts.down_proj", ),
            Transform(TransformKind.TRANSPOSE,
                      axes=(0, 2, 1),
                      output_shapes=((128, 2816, 704), )),
        ),
        (
            ("layer_0", "pre_ffw_norm", "scale"),
            (2816, ),
            ("model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
             ),
            Transform(TransformKind.IDENTITY, output_shapes=((2816, ), )),
        ),
        (
            ("layer_0", "pre_ffw2_norm", "scale"),
            (2816, ),
            ("model.language_model.layers.0.pre_feedforward_layernorm.weight",
             ),
            Transform(TransformKind.IDENTITY, output_shapes=((2816, ), )),
        ),
        (
            ("layer_0", "post_ffw1_norm", "scale"),
            (2816, ),
            ("model.language_model.layers.0.post_feedforward_layernorm_2.weight",
             ),
            Transform(TransformKind.IDENTITY, output_shapes=((2816, ), )),
        ),
        (
            ("layer_0", "post_ffw2_norm", "scale"),
            (2816, ),
            ("model.language_model.layers.0.post_feedforward_layernorm_1.weight",
             ),
            Transform(TransformKind.IDENTITY, output_shapes=((2816, ), )),
        ),
    )

    adapter = Gemma4FlaxAdapter(A4B_26B_IT_VARIANT)
    for source_path, source_shape, target_names, transform in cases:
        rule = adapter.select_rule(source_path, source_shape)[0]
        assert (rule.target_names, rule.transform) == (target_names, transform)
