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
"""Exact Gemma4 Flax-to-Hugging-Face mapping plan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ...adapter import CheckpointAdapter
from ...schema import InvalidMappingPlan, ShapeMismatch, UnmappedSourceTensor
from ...transforms import (MappingPlan, MappingRule, Shape, Transform,
                           TransformKind)
from .variant import (E2B_IT_VARIANT, VARIANTS, Gemma4VariantSpec, MoeSpec,
                      PleSpec)

_TARGET_PREFIX = "model.language_model"
_LAYER_PATTERN = re.compile(r"layer_(\d+)")


def _identity(shape: Shape) -> Transform:
    return Transform(TransformKind.IDENTITY, output_shapes=(shape, ))


def _transpose(axes: tuple[int, ...], shape: Shape) -> Transform:
    return Transform(TransformKind.TRANSPOSE,
                     axes=axes,
                     output_shapes=(shape, ))


def _reshape(shape: Shape) -> Transform:
    return Transform(TransformKind.RESHAPE, output_shapes=(shape, ))


def _transpose_reshape(axes: tuple[int, ...], shape: Shape) -> Transform:
    return Transform(TransformKind.TRANSPOSE_RESHAPE,
                     axes=axes,
                     output_shapes=(shape, ))


def _split_transpose_reshape(axes: tuple[int, ...],
                             shapes: tuple[Shape, ...]) -> Transform:
    return Transform(TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                     axes=axes,
                     output_shapes=shapes)


def _shape(value: Any) -> Shape:
    return tuple(int(dimension) for dimension in value.shape)


def _rule(source_path: tuple[str, ...], source_shape: Shape,
          expected_source_shape: Shape, target_names: tuple[str, ...],
          target_shapes: tuple[Shape,
                               ...], transform: Transform) -> MappingRule:
    if source_shape != expected_source_shape:
        raise ShapeMismatch(
            f"{source_path} has shape {source_shape}, expected "
            f"{expected_source_shape}")
    return MappingRule(source_path=source_path,
                       source_shape=expected_source_shape,
                       target_names=target_names,
                       target_shapes=target_shapes,
                       transform=transform)


def _base_layer_target_shapes(variant: Gemma4VariantSpec,
                              layer_index: int) -> dict[str, Shape]:
    dimensions = variant.dimensions
    hidden = dimensions["hidden_size"]
    head_dim = variant.head_dim(layer_index)
    q_width = dimensions["num_attention_heads"] * head_dim
    kv_width = variant.num_kv_heads(layer_index) * head_dim
    ffn = variant.ffn_width(layer_index)
    layer = f"{_TARGET_PREFIX}.layers.{layer_index}"
    targets = {
        f"{layer}.input_layernorm.weight": (hidden, ),
        f"{layer}.layer_scalar": (1, ),
        f"{layer}.mlp.down_proj.weight": (hidden, ffn),
        f"{layer}.mlp.gate_proj.weight": (ffn, hidden),
        f"{layer}.mlp.up_proj.weight": (ffn, hidden),
        f"{layer}.post_attention_layernorm.weight": (hidden, ),
        f"{layer}.post_feedforward_layernorm.weight": (hidden, ),
        f"{layer}.pre_feedforward_layernorm.weight": (hidden, ),
        f"{layer}.self_attn.k_norm.weight": (head_dim, ),
        f"{layer}.self_attn.k_proj.weight": (kv_width, hidden),
        f"{layer}.self_attn.o_proj.weight": (hidden, q_width),
        f"{layer}.self_attn.q_norm.weight": (head_dim, ),
        f"{layer}.self_attn.q_proj.weight": (q_width, hidden),
    }
    if not variant.attention_k_eq_v(layer_index):
        targets[f"{layer}.self_attn.v_proj.weight"] = (kv_width, hidden)
    return targets


def _ple_target_shapes(variant: Gemma4VariantSpec,
                       ple: PleSpec) -> dict[str, Shape]:
    dimensions = variant.dimensions
    hidden = dimensions["hidden_size"]
    layers = dimensions["num_hidden_layers"]
    vocab = dimensions["vocab_size"]
    flattened_ple = layers * ple.hidden_size
    targets = {
        f"{_TARGET_PREFIX}.embed_tokens_per_layer.weight":
        (vocab, flattened_ple),
        f"{_TARGET_PREFIX}.per_layer_model_projection.weight":
        (flattened_ple, hidden),
        f"{_TARGET_PREFIX}.per_layer_projection_norm.weight":
        (ple.hidden_size, ),
    }
    for layer_index in range(layers):
        layer = f"{_TARGET_PREFIX}.layers.{layer_index}"
        targets.update({
            f"{layer}.per_layer_input_gate.weight": (ple.hidden_size, hidden),
            f"{layer}.per_layer_projection.weight": (hidden, ple.hidden_size),
            f"{layer}.post_per_layer_input_norm.weight": (hidden, ),
        })
    return targets


def _moe_target_shapes(variant: Gemma4VariantSpec,
                       moe: MoeSpec) -> dict[str, Shape]:
    dimensions = variant.dimensions
    hidden = dimensions["hidden_size"]
    targets: dict[str, Shape] = {}
    for layer_index in range(dimensions["num_hidden_layers"]):
        layer = f"{_TARGET_PREFIX}.layers.{layer_index}"
        targets.update({
            f"{layer}.experts.down_proj":
            (moe.num_experts, hidden, moe.intermediate_size),
            f"{layer}.experts.gate_up_proj":
            (moe.num_experts, 2 * moe.intermediate_size, hidden),
            f"{layer}.post_feedforward_layernorm_1.weight": (hidden, ),
            f"{layer}.post_feedforward_layernorm_2.weight": (hidden, ),
            f"{layer}.pre_feedforward_layernorm_2.weight": (hidden, ),
            f"{layer}.router.per_expert_scale": (moe.num_experts, ),
            f"{layer}.router.proj.weight": (moe.num_experts, hidden),
            f"{layer}.router.scale": (hidden, ),
        })
    return targets


def _expected_target_shapes(variant: Gemma4VariantSpec) -> dict[str, Shape]:
    dimensions = variant.dimensions
    hidden = dimensions["hidden_size"]
    targets: dict[str, Shape] = {
        f"{_TARGET_PREFIX}.embed_tokens.weight":
        (dimensions["vocab_size"], hidden),
        f"{_TARGET_PREFIX}.norm.weight": (hidden, ),
    }
    for layer_index in range(dimensions["num_hidden_layers"]):
        targets.update(_base_layer_target_shapes(variant, layer_index))
    if variant.ple is not None:
        targets.update(_ple_target_shapes(variant, variant.ple))
    if variant.moe is not None:
        targets.update(_moe_target_shapes(variant, variant.moe))
    return targets


_RuleEntry = tuple[Shape, tuple[str, ...], tuple[Shape, ...], Transform]


def _select_entry(
        source_path: tuple[str, ...], source_shape: Shape, suffix: tuple[str,
                                                                         ...],
        entries: Mapping[tuple[str, ...], _RuleEntry]) -> list[MappingRule]:
    entry = entries.get(suffix)
    if entry is None:
        raise UnmappedSourceTensor(f"unmapped source tensor: {source_path}")
    expected_source, names, shapes, transform = entry
    return [
        _rule(source_path, source_shape, expected_source, names, shapes,
              transform)
    ]


def _base_embedding_entries(
        variant: Gemma4VariantSpec) -> dict[tuple[str, ...], _RuleEntry]:
    dimensions = variant.dimensions
    vocab = dimensions["vocab_size"]
    hidden = dimensions["hidden_size"]
    return {
        ("embedder", "input_embedding"):
        ((vocab, hidden), (f"{_TARGET_PREFIX}.embed_tokens.weight", ),
         ((vocab, hidden), ), _identity((vocab, hidden))),
        ("final_norm", "scale"):
        ((hidden, ), (f"{_TARGET_PREFIX}.norm.weight", ), ((hidden, ), ),
         _identity((hidden, ))),
    }


def _ple_embedding_entries(
    variant: Gemma4VariantSpec,
    ple: PleSpec,
) -> dict[tuple[str, ...], _RuleEntry]:
    dimensions = variant.dimensions
    vocab = dimensions["vocab_size"]
    hidden = dimensions["hidden_size"]
    layers = dimensions["num_hidden_layers"]
    flattened_ple = layers * ple.hidden_size
    return {
        ("embedder", "per_layer_embeddings"):
        ((vocab, layers, ple.hidden_size),
         (f"{_TARGET_PREFIX}.embed_tokens_per_layer.weight", ),
         ((vocab, flattened_ple), ), _reshape((vocab, flattened_ple))),
        ("embedder", "per_layer_model_projection", "w"):
        ((hidden, layers, ple.hidden_size),
         (f"{_TARGET_PREFIX}.per_layer_model_projection.weight", ),
         ((flattened_ple, hidden), ),
         _transpose_reshape((1, 2, 0), (flattened_ple, hidden))),
        ("embedder", "per_layer_projection_norm", "scale"):
        ((ple.hidden_size, ),
         (f"{_TARGET_PREFIX}.per_layer_projection_norm.weight", ),
         ((ple.hidden_size, ), ), _identity((ple.hidden_size, ))),
    }


def _embedding_entries(
        variant: Gemma4VariantSpec) -> dict[tuple[str, ...], _RuleEntry]:
    entries = _base_embedding_entries(variant)
    if variant.ple is not None:
        entries.update(_ple_embedding_entries(variant, variant.ple))
    return entries


def _base_attention_entries(
    variant: Gemma4VariantSpec,
    layer_index: int,
) -> dict[tuple[str, ...], _RuleEntry]:
    dimensions = variant.dimensions
    hidden = dimensions["hidden_size"]
    heads = dimensions["num_attention_heads"]
    head_dim = variant.head_dim(layer_index)
    q_width = heads * head_dim
    target = f"{_TARGET_PREFIX}.layers.{layer_index}.self_attn"
    return {
        ("q_einsum", "w"):
        ((heads, hidden, head_dim), (f"{target}.q_proj.weight", ),
         ((q_width, hidden), ), _transpose_reshape((0, 2, 1),
                                                   (q_width, hidden))),
        ("attn_vec_einsum", "w"):
        ((heads, head_dim, hidden), (f"{target}.o_proj.weight", ),
         ((hidden, q_width), ), _transpose_reshape((2, 0, 1),
                                                   (hidden, q_width))),
        ("query_norm", "scale"): ((head_dim, ), (f"{target}.q_norm.weight", ),
                                  ((head_dim, ), ), _identity((head_dim, ))),
        ("key_norm", "scale"): ((head_dim, ), (f"{target}.k_norm.weight", ),
                                ((head_dim, ), ), _identity((head_dim, ))),
    }


def _shared_kv_attention_entry(variant: Gemma4VariantSpec,
                               layer_index: int) -> _RuleEntry:
    hidden = variant.dimensions["hidden_size"]
    kv_heads = variant.num_kv_heads(layer_index)
    head_dim = variant.head_dim(layer_index)
    kv_width = kv_heads * head_dim
    target = f"{_TARGET_PREFIX}.layers.{layer_index}.self_attn"
    return ((kv_heads, hidden, head_dim), (f"{target}.k_proj.weight", ),
            ((kv_width, hidden), ),
            _transpose_reshape((0, 2, 1), (kv_width, hidden)))


def _fused_kv_attention_entry(variant: Gemma4VariantSpec,
                              layer_index: int) -> _RuleEntry:
    hidden = variant.dimensions["hidden_size"]
    kv_heads = variant.num_kv_heads(layer_index)
    head_dim = variant.head_dim(layer_index)
    kv_width = kv_heads * head_dim
    target = f"{_TARGET_PREFIX}.layers.{layer_index}.self_attn"
    target_shapes = ((kv_width, hidden), (kv_width, hidden))
    return ((2, kv_heads, hidden, head_dim), (f"{target}.k_proj.weight",
                                              f"{target}.v_proj.weight"),
            target_shapes, _split_transpose_reshape((0, 2, 1), target_shapes))


def _attention_entries(
    variant: Gemma4VariantSpec,
    layer_index: int,
) -> dict[tuple[str, ...], _RuleEntry]:
    entries = _base_attention_entries(variant, layer_index)
    if variant.attention_k_eq_v(layer_index):
        entries[("k_einsum",
                 "w")] = _shared_kv_attention_entry(variant, layer_index)
    else:
        entries[("kv_einsum",
                 "w")] = _fused_kv_attention_entry(variant, layer_index)
    return entries


def _dense_mlp_entries(
    variant: Gemma4VariantSpec,
    layer_index: int,
) -> dict[tuple[str, ...], _RuleEntry]:
    hidden = variant.dimensions["hidden_size"]
    ffn = variant.ffn_width(layer_index)
    target = f"{_TARGET_PREFIX}.layers.{layer_index}.mlp"
    target_shapes = ((ffn, hidden), (ffn, hidden))
    return {
        ("gating_einsum", "w"):
        ((2, ffn, hidden), (f"{target}.gate_proj.weight",
                            f"{target}.up_proj.weight"), target_shapes,
         _split_transpose_reshape((0, 1), target_shapes)),
        ("linear", "w"):
        ((ffn, hidden), (f"{target}.down_proj.weight", ), ((hidden, ffn), ),
         _transpose((1, 0), (hidden, ffn))),
    }


def _moe_entries(variant: Gemma4VariantSpec, moe: MoeSpec,
                 layer_index: int) -> dict[tuple[str, ...], _RuleEntry]:
    hidden = variant.dimensions["hidden_size"]
    experts = moe.num_experts
    expert_ffn = moe.intermediate_size
    target = f"{_TARGET_PREFIX}.layers.{layer_index}"
    return {
        ("gating_einsum", "w"): ((experts, 2, expert_ffn, hidden),
                                 (f"{target}.experts.gate_up_proj", ),
                                 ((experts, 2 * expert_ffn, hidden), ),
                                 _reshape((experts, 2 * expert_ffn, hidden))),
        ("linear", "w"):
        ((experts, expert_ffn, hidden), (f"{target}.experts.down_proj", ),
         ((experts, hidden, expert_ffn), ),
         _transpose((0, 2, 1), (experts, hidden, expert_ffn))),
        ("router_logits", "w"):
        ((hidden, experts), (f"{target}.router.proj.weight", ),
         ((experts, hidden), ), _transpose((1, 0), (experts, hidden))),
        ("per_expert_scale", ):
        ((experts, ), (f"{target}.router.per_expert_scale", ), ((experts, ), ),
         _identity((experts, ))),
        ("router_scale", ): ((hidden, ), (f"{target}.router.scale", ),
                             ((hidden, ), ), _identity((hidden, ))),
    }


def _base_norm_entries(variant: Gemma4VariantSpec,
                       layer_index: int) -> dict[tuple[str, ...], _RuleEntry]:
    hidden = variant.dimensions["hidden_size"]
    target = f"{_TARGET_PREFIX}.layers.{layer_index}"
    norm_shape = (hidden, )
    return {
        ("pre_attention_norm", "scale"):
        (norm_shape, (f"{target}.input_layernorm.weight", ), (norm_shape, ),
         _identity(norm_shape)),
        ("post_attention_norm", "scale"):
        (norm_shape, (f"{target}.post_attention_layernorm.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("pre_ffw_norm", "scale"):
        (norm_shape, (f"{target}.pre_feedforward_layernorm.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("post_ffw_norm", "scale"):
        (norm_shape, (f"{target}.post_feedforward_layernorm.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("skip_scale", ): ((1, ), (f"{target}.layer_scalar", ), ((1, ), ),
                           _identity((1, ))),
    }


def _moe_norm_entries(variant: Gemma4VariantSpec,
                      layer_index: int) -> dict[tuple[str, ...], _RuleEntry]:
    hidden = variant.dimensions["hidden_size"]
    target = f"{_TARGET_PREFIX}.layers.{layer_index}"
    norm_shape = (hidden, )
    return {
        ("pre_ffw_norm", "scale"):
        (norm_shape, (f"{target}.pre_feedforward_layernorm_2.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("post_ffw1_norm", "scale"):
        (norm_shape, (f"{target}.post_feedforward_layernorm_2.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("pre_ffw2_norm", "scale"):
        (norm_shape, (f"{target}.pre_feedforward_layernorm.weight", ),
         (norm_shape, ), _identity(norm_shape)),
        ("post_ffw2_norm", "scale"):
        (norm_shape, (f"{target}.post_feedforward_layernorm_1.weight", ),
         (norm_shape, ), _identity(norm_shape)),
    }


def _norm_entries(
    variant: Gemma4VariantSpec,
    layer_index: int,
) -> dict[tuple[str, ...], _RuleEntry]:
    entries = _base_norm_entries(variant, layer_index)
    if variant.moe is not None:
        entries.update(_moe_norm_entries(variant, layer_index))
    return entries


def _ple_entries(variant: Gemma4VariantSpec, ple: PleSpec,
                 layer_index: int) -> dict[tuple[str, ...], _RuleEntry]:
    hidden = variant.dimensions["hidden_size"]
    target = f"{_TARGET_PREFIX}.layers.{layer_index}"
    return {
        ("per_layer_input_gate", "w"):
        ((hidden, ple.hidden_size),
         (f"{target}.per_layer_input_gate.weight", ), ((ple.hidden_size,
                                                        hidden), ),
         _transpose((1, 0), (ple.hidden_size, hidden))),
        ("per_layer_projection", "w"):
        ((ple.hidden_size, hidden),
         (f"{target}.per_layer_projection.weight", ), ((hidden,
                                                        ple.hidden_size), ),
         _transpose((1, 0), (hidden, ple.hidden_size))),
        ("post_per_layer_input_norm", "scale"):
        ((hidden, ), (f"{target}.post_per_layer_input_norm.weight", ),
         ((hidden, ), ), _identity((hidden, ))),
    }


def validate_mapping_plan(
    plan: MappingPlan,
    flat_tree: Mapping[tuple[str, ...], Any],
    variant: Gemma4VariantSpec,
) -> MappingPlan:
    """Validate complete source coverage and the exact target manifest.

    Args:
        plan: Candidate mapping plan.
        flat_tree: Flattened canonical source parameter tree.
        variant: Declarative target model specification.

    Returns:
        Deterministically source-path-ordered immutable plan.

    Raises:
        InvalidMappingPlan: If coverage, uniqueness, or target shapes differ.
    """
    if plan.variant != variant.name:
        raise InvalidMappingPlan(
            f"mapping plan variant {plan.variant!r} does not match "
            f"{variant.name!r}")
    if not isinstance(plan.rules, tuple) or not plan.rules:
        raise InvalidMappingPlan("mapping plan cannot be empty")
    validate_rule_contracts(plan.rules)
    validate_source_coverage(plan.rules, flat_tree)
    validate_target_coverage(plan.rules, variant)

    ordered_rules = tuple(sorted(plan.rules,
                                 key=lambda rule: rule.source_path))
    return MappingPlan(rules=ordered_rules, variant=plan.variant)


def validate_rule_contracts(rules: tuple[MappingRule, ...]) -> None:
    """Validate the internal contract of every mapping rule."""
    for rule in rules:
        if (not isinstance(rule.source_path, tuple) or not rule.source_path
                or any(not isinstance(component, str) or not component
                       for component in rule.source_path)):
            raise InvalidMappingPlan(
                f"invalid source path in mapping rule: {rule.source_path!r}")
        if (not isinstance(rule.target_names, tuple)
                or not isinstance(rule.target_shapes, tuple)
                or not rule.target_names
                or any(not isinstance(name, str) or not name
                       for name in rule.target_names)):
            raise InvalidMappingPlan(
                f"invalid target name for {rule.source_path}")
        if len(rule.target_names) != len(rule.target_shapes):
            raise InvalidMappingPlan(
                f"target name/shape count differs for {rule.source_path}")
        if rule.transform.output_shapes != rule.target_shapes:
            raise InvalidMappingPlan(
                f"transform/target shapes differ for {rule.source_path}")


def validate_source_coverage(rules: tuple[MappingRule, ...],
                             flat_tree: Mapping[tuple[str, ...], Any]) -> None:
    """Validate exact source coverage and declared source shapes."""

    source_paths = [rule.source_path for rule in rules]
    source_path_set = set(source_paths)
    flat_path_set = set(flat_tree)
    if len(source_paths) != len(source_path_set):
        raise InvalidMappingPlan(
            "mapping plan contains duplicate source paths")
    if source_path_set != flat_path_set:
        missing = sorted(flat_path_set - source_path_set)
        unexpected = sorted(source_path_set - flat_path_set)
        raise InvalidMappingPlan(
            f"source coverage differs; missing={missing}, unexpected={unexpected}"
        )
    for rule in rules:
        actual_source_shape = _shape(flat_tree[rule.source_path])
        if actual_source_shape != rule.source_shape:
            raise InvalidMappingPlan(
                f"source shape differs for {rule.source_path}: "
                f"plan={rule.source_shape}, actual={actual_source_shape}")


def validate_target_coverage(rules: tuple[MappingRule, ...],
                             variant: Gemma4VariantSpec) -> None:
    """Validate exact target names, uniqueness, and shapes."""
    target_names = [name for rule in rules for name in rule.target_names]
    if len(target_names) != len(set(target_names)):
        raise InvalidMappingPlan(
            "mapping plan contains duplicate target names")
    expected = _expected_target_shapes(variant)
    actual = {
        name: shape
        for rule in rules
        for name, shape in zip(rule.target_names, rule.target_shapes)
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(name for name in set(actual) & set(expected)
                            if actual[name] != expected[name])
        raise InvalidMappingPlan(
            "target manifest differs; "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={mismatched}")


@dataclass(frozen=True)
class Gemma4FlaxAdapter(CheckpointAdapter):
    """Adapter for supported pinned public Gemma4 Flax parameters.

    Attributes:
        variant: Exact immutable target variant specification.
    """

    variant: Gemma4VariantSpec = E2B_IT_VARIANT

    def plan(self, source_tree: Any) -> MappingPlan:
        """Flatten the canonical tree and map every source leaf.

        Args:
            source_tree: Canonical nested tree returned by the pinned reader.

        Returns:
            Complete immutable mapping plan.

        Raises:
            UnmappedSourceTensor: If any leaf has no exact mapping.
            ShapeMismatch: If any source leaf has unexpected geometry.
            InvalidMappingPlan: If target coverage is not exact.
        """
        if self.variant.name not in VARIANTS or VARIANTS[
                self.variant.name] != self.variant:
            raise InvalidMappingPlan(
                f"unsupported Gemma4 variant: {self.variant.name}")

        from flax.traverse_util import flatten_dict

        flat_tree = flatten_dict(source_tree)
        rules = []
        for source_path in sorted(flat_tree):
            rules.extend(
                self.select_rule(source_path, _shape(flat_tree[source_path])))
        plan = MappingPlan(rules=tuple(rules), variant=self.variant.name)
        return validate_mapping_plan(plan, flat_tree, self.variant)

    def select_rule(self, source_path: tuple[str, ...],
                    source_shape: Shape) -> list[MappingRule]:
        """Dispatch one exact source path to a focused selector.

        Args:
            source_path: Flattened canonical source path.
            source_shape: Actual source tensor shape.

        Returns:
            One mapping rule, represented as a list for split-capable families.

        Raises:
            UnmappedSourceTensor: If the source path is not declared.
        """
        if source_path and source_path[0] in ("embedder", "final_norm"):
            return self.select_embedding_rule(source_path, source_shape)
        if source_path and _LAYER_PATTERN.fullmatch(source_path[0]):
            return self.select_layer_rule(source_path, source_shape)
        raise UnmappedSourceTensor(f"unmapped source tensor: {source_path}")

    def select_embedding_rule(self, source_path: tuple[str, ...],
                              source_shape: Shape) -> list[MappingRule]:
        """Map the token, PLE, top projection, and final norm leaves.

        Args:
            source_path: Exact embedding or final-norm path.
            source_shape: Actual source tensor shape.

        Returns:
            Single exact mapping rule.

        Raises:
            UnmappedSourceTensor: If the path is not a declared global leaf.
            ShapeMismatch: If the tensor geometry is not exact.
        """
        return _select_entry(source_path, source_shape, source_path,
                             _embedding_entries(self.variant))

    def select_layer_rule(self, source_path: tuple[str, ...],
                          source_shape: Shape) -> list[MappingRule]:
        """Dispatch one decoder-layer source leaf.

        Args:
            source_path: Exact layer-prefixed source path.
            source_shape: Actual source tensor shape.

        Returns:
            Single rule for the selected layer leaf.

        Raises:
            UnmappedSourceTensor: If its path family is undeclared.
        """
        if len(source_path) < 2:
            raise UnmappedSourceTensor(
                f"unmapped source tensor: {source_path}")
        if source_path[1] == "attn":
            return self.select_attention_rule(source_path, source_shape)
        if source_path[1] == "mlp":
            if self.variant.moe is not None:
                return self.select_moe_rule(source_path, source_shape)
            return self.select_mlp_rule(source_path, source_shape)
        if source_path[1] == "mlp2" and self.variant.moe is not None:
            return self.select_mlp_rule(source_path, source_shape)
        if source_path[1] in {
                "pre_attention_norm", "post_attention_norm", "pre_ffw_norm",
                "post_ffw_norm", "post_ffw1_norm", "pre_ffw2_norm",
                "post_ffw2_norm", "skip_scale"
        }:
            return self.select_norm_rule(source_path, source_shape)
        return self.select_ple_rule(source_path, source_shape)

    def _layer_index(self, source_path: tuple[str, ...]) -> int:
        match = _LAYER_PATTERN.fullmatch(source_path[0])
        if match is None:
            raise UnmappedSourceTensor(f"invalid layer path: {source_path}")
        layer_index = int(match.group(1))
        if not 0 <= layer_index < self.variant.dimensions["num_hidden_layers"]:
            raise UnmappedSourceTensor(f"layer outside variant: {source_path}")
        return layer_index

    def select_attention_rule(self, source_path: tuple[str, ...],
                              source_shape: Shape) -> list[MappingRule]:
        """Map Q, variant-specific KV, O, and Q/K normalization leaves.

        Args:
            source_path: Exact attention source path.
            source_shape: Actual source tensor shape.

        Returns:
            One rule, with two targets for the fused KV source.

        Raises:
            UnmappedSourceTensor: If the attention leaf is undeclared.
            ShapeMismatch: If its local/global head geometry is incorrect.
        """
        layer_index = self._layer_index(source_path)
        suffix = source_path[2:]
        return _select_entry(source_path, source_shape, suffix,
                             _attention_entries(self.variant, layer_index))

    def select_mlp_rule(self, source_path: tuple[str, ...],
                        source_shape: Shape) -> list[MappingRule]:
        """Map the fused gated input and dense down projection.

        Args:
            source_path: Exact MLP source path.
            source_shape: Actual source tensor shape.

        Returns:
            One rule, with two targets for the fused gated source.

        Raises:
            UnmappedSourceTensor: If the MLP leaf is undeclared.
            ShapeMismatch: If its declared layer width is incorrect.
        """
        layer_index = self._layer_index(source_path)
        suffix = source_path[2:]
        return _select_entry(source_path, source_shape, suffix,
                             _dense_mlp_entries(self.variant, layer_index))

    def select_moe_rule(self, source_path: tuple[str, ...],
                        source_shape: Shape) -> list[MappingRule]:
        """Map routed-expert weights, router weights, and their scales."""
        layer_index = self._layer_index(source_path)
        moe = self.variant.moe
        if moe is None:
            raise UnmappedSourceTensor(
                f"unmapped source tensor: {source_path}")
        suffix = source_path[2:]
        return _select_entry(source_path, source_shape, suffix,
                             _moe_entries(self.variant, moe, layer_index))

    def select_norm_rule(self, source_path: tuple[str, ...],
                         source_shape: Shape) -> list[MappingRule]:
        """Map decoder norms and the layer scalar.

        Args:
            source_path: Exact normalization or scalar source path.
            source_shape: Actual source tensor shape.

        Returns:
            Single identity mapping rule.

        Raises:
            UnmappedSourceTensor: If the leaf is undeclared.
            ShapeMismatch: If its shape is incorrect.
        """
        layer_index = self._layer_index(source_path)
        suffix = source_path[1:]
        return _select_entry(source_path, source_shape, suffix,
                             _norm_entries(self.variant, layer_index))

    def select_ple_rule(self, source_path: tuple[str, ...],
                        source_shape: Shape) -> list[MappingRule]:
        """Map the per-layer gate, projection, and post-input norm.

        Args:
            source_path: Exact PLE source path.
            source_shape: Actual source tensor shape.

        Returns:
            Single exact mapping rule.

        Raises:
            UnmappedSourceTensor: If the PLE leaf is undeclared.
            ShapeMismatch: If its shape is incorrect.
        """
        layer_index = self._layer_index(source_path)
        ple = self.variant.ple
        if ple is None:
            raise UnmappedSourceTensor(
                f"unmapped source tensor: {source_path}")
        suffix = source_path[1:]
        return _select_entry(source_path, source_shape, suffix,
                             _ple_entries(self.variant, ple, layer_index))

    def build_config(self) -> dict[str, Any]:
        """Build and locally validate the exact target configuration.

        Returns:
            JSON-compatible root configuration with nested text config.

        Raises:
            InvalidMappingPlan: If declarative fields violate variant invariants.
        """
        if self.variant.name not in VARIANTS or VARIANTS[
                self.variant.name] != self.variant:
            raise InvalidMappingPlan(
                f"unsupported Gemma4 variant: {self.variant.name}")
        return self.variant.build_hf_config()
