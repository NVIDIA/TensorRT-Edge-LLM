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
"""Component, profile, and output contracts for checkpoint-direct builds."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Tuple

from . import weight_policy

__all__ = [
    "Component",
    "ComponentSpec",
    "ProfileLimits",
    "SpecRole",
    "available_components",
    "component_build_order",
    "component_choices",
    "component_spec",
    "resolve_components",
]


class Component(str, Enum):
    """Engine component built by one runner invocation."""

    LLM = "llm"
    DLLM = "dllm"
    TALKER = "talker"
    CODE_PREDICTOR = "code-predictor"
    VISUAL = "visual"
    AUDIO = "audio"
    RNNT = "rnnt"
    CODE2WAV = "code2wav"
    SPEAKER_ENCODER = "speaker-encoder"
    SPEECH_TOKENIZER_ENCODER = "speech-tokenizer-encoder"
    ACTION = "action"
    UND_PREFILL = "und-prefill"
    GEN = "gen"
    VAE_ENCODER = "vae-encoder"


class SpecRole(str, Enum):
    """Speculative-decoding role for an LLM component."""

    NONE = "none"
    BASE = "base"
    DRAFT = "draft"


@dataclass(frozen=True)
class ProfileLimits:
    """Optimization-profile limits shared by all component builders."""

    max_input_len: int = 1024
    max_kv_cache_capacity: int = 4096
    max_batch_size: int = 4
    max_lora_rank: int = 0
    max_verify_tree_size: int = 60
    max_draft_tree_size: int = 60
    min_image_tokens: int = 4
    max_image_tokens: int = 1024
    max_image_tokens_per_image: int = 512
    min_time_steps: int = 100
    max_time_steps: int = 6000
    min_code_len: int = 1
    opt_code_len: int = 300
    max_code_len: int = 2000

    def generation_sequence_max(self, role: SpecRole) -> int:
        """Return the generation-profile sequence bound for an LLM role."""
        if role == SpecRole.BASE:
            return self.max_verify_tree_size
        if role == SpecRole.DRAFT:
            return self.max_draft_tree_size
        return 1

    def validate(self) -> None:
        """Validate limits before a TensorRT network is created."""
        positive = {
            "max_input_len": self.max_input_len,
            "max_kv_cache_capacity": self.max_kv_cache_capacity,
            "max_batch_size": self.max_batch_size,
            "max_verify_tree_size": self.max_verify_tree_size,
            "max_draft_tree_size": self.max_draft_tree_size,
            "min_image_tokens": self.min_image_tokens,
            "max_image_tokens": self.max_image_tokens,
            "max_image_tokens_per_image": self.max_image_tokens_per_image,
            "min_time_steps": self.min_time_steps,
            "max_time_steps": self.max_time_steps,
            "min_code_len": self.min_code_len,
            "opt_code_len": self.opt_code_len,
            "max_code_len": self.max_code_len,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("profile limits must be positive: " +
                             ", ".join(sorted(invalid)))
        if self.max_lora_rank < 0:
            raise ValueError("max_lora_rank must be non-negative")
        if not (self.min_image_tokens <= self.max_image_tokens_per_image <=
                self.max_image_tokens):
            raise ValueError(
                "image token limits must satisfy min <= per-image <= max")
        if self.min_time_steps > self.max_time_steps:
            raise ValueError("min_time_steps cannot exceed max_time_steps")
        if not self.min_code_len <= self.opt_code_len <= self.max_code_len:
            raise ValueError(
                "code lengths must satisfy min_code_len <= opt_code_len <= max_code_len"
            )


@dataclass(frozen=True)
class ComponentSpec:
    """Runtime-visible output and plugin contract for one component."""

    component: Component
    engine_subdir: str
    engine_filename: str
    required_plugins: Tuple[str, ...] = ()
    supports_spec_role: bool = False
    # Weight kinds this component's C++ runner binds at initialization.
    external_weight_kinds: Tuple[str, ...] = ()

    def output_dir(self, requested_dir: str) -> str:
        """Return the runtime directory for this component."""
        if not self.engine_subdir:
            return requested_dir
        return os.path.join(requested_dir, self.engine_subdir)

    def output_path(self,
                    requested_dir: str,
                    spec_role: SpecRole = SpecRole.NONE,
                    tp_size: int = 1,
                    tp_rank: int = 0) -> str:
        """Return the engine path for this component and speculative role."""
        filename = self.engine_filename
        if self.component == Component.LLM and spec_role == SpecRole.NONE and tp_size > 1:
            filename = f"llm_world{tp_size}_rank{tp_rank}.engine"
        if self.supports_spec_role:
            if spec_role == SpecRole.BASE:
                filename = "spec_base.engine"
            elif spec_role == SpecRole.DRAFT:
                filename = "spec_draft.engine"
        elif spec_role != SpecRole.NONE:
            raise ValueError(
                f"component {self.component.value!r} does not support speculative roles"
            )
        return os.path.join(self.output_dir(requested_dir), filename)

    def config_path(self,
                    requested_dir: str,
                    spec_role: SpecRole = SpecRole.NONE,
                    tp_size: int = 1) -> str:
        """Return the runtime config path for this component and role."""
        filename = "config.json"
        if self.component == Component.LLM and spec_role == SpecRole.NONE and tp_size > 1:
            filename = f"config_world{tp_size}.json"
        if self.supports_spec_role:
            if spec_role == SpecRole.BASE:
                filename = "base_config.json"
            elif spec_role == SpecRole.DRAFT:
                filename = "draft_config.json"
        elif spec_role != SpecRole.NONE:
            raise ValueError(
                f"component {self.component.value!r} does not support speculative roles"
            )
        return os.path.join(self.output_dir(requested_dir), filename)


_ENGINE_WEIGHT_KINDS = tuple(
    kind for kind in weight_policy.EXTERNAL_WEIGHT_KINDS
    if kind != weight_policy.EXTERNAL_WEIGHT_EMBEDDING)

_COMPONENT_SPECS: Dict[Component, ComponentSpec] = {
    Component.LLM:
    ComponentSpec(Component.LLM,
                  "",
                  "llm.engine",
                  required_plugins=("AttentionPlugin", ),
                  supports_spec_role=True,
                  external_weight_kinds=weight_policy.EXTERNAL_WEIGHT_KINDS),
    Component.DLLM:
    ComponentSpec(Component.DLLM,
                  "",
                  "dllm.engine",
                  required_plugins=("AttentionPlugin", ),
                  external_weight_kinds=weight_policy.EXTERNAL_WEIGHT_KINDS),
    Component.TALKER:
    ComponentSpec(Component.TALKER,
                  "talker",
                  "llm.engine",
                  required_plugins=("AttentionPlugin", ),
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.CODE_PREDICTOR:
    ComponentSpec(Component.CODE_PREDICTOR,
                  "code_predictor",
                  "llm.engine",
                  required_plugins=("AttentionPlugin", ),
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.VISUAL:
    ComponentSpec(Component.VISUAL,
                  "visual",
                  "visual.engine",
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.AUDIO:
    ComponentSpec(Component.AUDIO,
                  "audio",
                  "audio_encoder.engine",
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.RNNT:
    ComponentSpec(Component.RNNT, "rnnt", "rnnt_step.engine"),
    Component.CODE2WAV:
    ComponentSpec(Component.CODE2WAV,
                  "code2wav",
                  "code2wav.engine",
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.SPEAKER_ENCODER:
    ComponentSpec(Component.SPEAKER_ENCODER, "clone_encoders",
                  "speaker_encoder.engine"),
    Component.SPEECH_TOKENIZER_ENCODER:
    ComponentSpec(Component.SPEECH_TOKENIZER_ENCODER, "clone_encoders",
                  "speech_tokenizer_encoder.engine"),
    Component.ACTION:
    ComponentSpec(Component.ACTION,
                  "action",
                  "action.engine",
                  external_weight_kinds=_ENGINE_WEIGHT_KINDS),
    Component.UND_PREFILL:
    ComponentSpec(Component.UND_PREFILL, "und_prefill", "und_prefill.engine"),
    Component.GEN:
    ComponentSpec(Component.GEN, "gen", "gen.engine"),
    Component.VAE_ENCODER:
    ComponentSpec(Component.VAE_ENCODER, "vae_encoder", "vae_encoder.engine"),
}

_COMPONENT_BUILD_ORDER: Tuple[Component, ...] = (
    Component.DLLM,
    Component.LLM,
    Component.VISUAL,
    Component.AUDIO,
    Component.RNNT,
    Component.TALKER,
    Component.CODE_PREDICTOR,
    Component.CODE2WAV,
    Component.SPEAKER_ENCODER,
    Component.SPEECH_TOKENIZER_ENCODER,
    Component.ACTION,
    Component.UND_PREFILL,
    Component.GEN,
    Component.VAE_ENCODER,
)


def component_choices() -> Tuple[str, ...]:
    """Return stable CLI component choices."""
    return tuple(component.value for component in Component)


def component_build_order() -> Tuple[str, ...]:
    """Return the default one-command build order."""
    return tuple(component.value for component in _COMPONENT_BUILD_ORDER)


def component_spec(component: "Component | str") -> ComponentSpec:
    """Resolve the output contract for a component value."""
    resolved = component if isinstance(component,
                                       Component) else Component(component)
    return _COMPONENT_SPECS[resolved]


def _normalize_component_name(name: str) -> Component:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "codepredictor": Component.CODE_PREDICTOR,
        "code-predictor": Component.CODE_PREDICTOR,
        "text": Component.LLM,
        "language": Component.LLM,
        "diffusion": Component.DLLM,
        "backbone": Component.DLLM,
        "vision": Component.VISUAL,
        "image": Component.VISUAL,
        "speech": Component.AUDIO,
        "vocoder": Component.CODE2WAV,
        "speaker": Component.SPEAKER_ENCODER,
        "speech-tokenizer": Component.SPEECH_TOKENIZER_ENCODER,
        "understanding-prefill": Component.UND_PREFILL,
        "policy": Component.GEN,
        "vae": Component.VAE_ENCODER,
    }
    if normalized in aliases:
        return aliases[normalized]
    return Component(normalized)


def resolve_components(
        root_model_type: str,
        requested: Iterable[str],
        available: "Iterable[Component] | None" = None
) -> Tuple[Component, ...]:
    """Resolve an ``all``/list CLI selection to buildable components."""
    tokens = tuple(token.strip().lower().replace("_", "-")
                   for token in requested if token.strip())
    available = frozenset(available if available is not None else
                          available_components(root_model_type))
    if not tokens or tokens == ("all", ):
        selected = available
    elif "all" in tokens:
        raise ValueError("'all' cannot be combined with named components")
    else:
        selected = frozenset(
            _normalize_component_name(token) for token in tokens)
        missing = selected - available
        if missing:
            choices = ", ".join(component.value
                                for component in _ordered(available))
            invalid = ", ".join(component.value
                                for component in _ordered(missing))
            raise ValueError(
                f"{root_model_type!r} has no requested component(s): "
                f"{invalid}; available components: {choices}")
    return _ordered(selected)


def _ordered(components: Iterable[Component]) -> Tuple[Component, ...]:
    component_set = frozenset(components)
    return tuple(component for component in _COMPONENT_BUILD_ORDER
                 if component in component_set)


def available_components(root_model_type: str) -> FrozenSet[Component]:
    """Return components contained in a checkpoint root model type."""
    from ..models import registry
    return registry.components_for(root_model_type)
