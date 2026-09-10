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
"""Lazy registry of model-owned TensorRT component networks."""

import importlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Tuple

from ..core import contracts

Component = contracts.Component


@dataclass(frozen=True)
class ComponentDefinition:
    """Import path of one concrete model-owned ``NetworkModule``."""

    module: str
    class_name: str

    def load(self):
        module = importlib.import_module(f".{self.module}", __package__)
        try:
            model_class = getattr(module, self.class_name)
        except AttributeError as error:
            raise ValueError(
                f"{self.module!r} does not define {self.class_name!r}"
            ) from error

        from ..ops import NetworkModule
        if not isinstance(model_class, type) or not issubclass(
                model_class, NetworkModule):
            raise TypeError(f"{self.module}.{self.class_name} must inherit "
                            "NetworkModule")
        return model_class


@dataclass(frozen=True)
class ModelFamily:
    """HF model variants and their model-owned build entrypoints."""

    name: str
    variants: Mapping[str, FrozenSet[Component]]
    components: Mapping[Component, ComponentDefinition]
    configuration: str = ""
    artifact_writer: str = ""
    weight_conversion: str = ""


def _component(module: str, class_name: str) -> ComponentDefinition:
    return ComponentDefinition(module, class_name)


def _set(*components: Component) -> FrozenSet[Component]:
    return frozenset(components)


FAMILIES: Tuple[ModelFamily, ...] = (
    ModelFamily(
        "llama",
        {"llama": _set(Component.LLM)},
        {
            Component.LLM: _component("llama.modeling_llama",
                                      "LlamaForCausalLM")
        },
    ),
    ModelFamily(
        "mistral",
        {"mistral": _set(Component.LLM)},
        {
            Component.LLM:
            _component("mistral.modeling_mistral", "MistralForCausalLM")
        },
    ),
    ModelFamily(
        "qwen2",
        {"qwen2": _set(Component.LLM)},
        {
            Component.LLM: _component("qwen2.modeling_qwen2",
                                      "Qwen2ForCausalLM")
        },
    ),
    ModelFamily(
        "qwen3",
        {"qwen3": _set(Component.LLM)},
        {
            Component.LLM: _component("qwen3.modeling_qwen3",
                                      "Qwen3ForCausalLM")
        },
    ),
    ModelFamily(
        "qwen3_moe",
        {"qwen3_moe": _set(Component.LLM)},
        {
            Component.LLM:
            _component("qwen3_moe.modeling_qwen3_moe", "Qwen3MoeForCausalLM")
        },
    ),
    ModelFamily(
        "qwen2_5_vl",
        {"qwen2_5_vl": _set(Component.LLM, Component.VISUAL)},
        {
            Component.LLM:
            _component("qwen2_5_vl.modeling_qwen2_5_vl_text",
                       "Qwen25VLCausalLM"),
            Component.VISUAL:
            _component("qwen2_5_vl.modeling_qwen2_5_vl_visual",
                       "Qwen25VLVisualEncoder"),
        },
    ),
    ModelFamily(
        "qwen2_vl",
        {"qwen2_vl": _set(Component.LLM, Component.VISUAL)},
        {
            Component.LLM:
            _component("qwen2_vl.modeling_qwen2_vl_text",
                       "Qwen2VLForConditionalGeneration"),
            Component.VISUAL:
            _component("qwen2_vl.modeling_qwen2_vl_visual",
                       "Qwen2VLVisualEncoder"),
        },
    ),
    ModelFamily(
        "qwen3_vl",
        {"qwen3_vl": _set(Component.LLM, Component.VISUAL)},
        {
            Component.LLM:
            _component("qwen3_vl.modeling_qwen3_vl_text", "Qwen3VLCausalLM"),
            Component.VISUAL:
            _component("qwen3_vl.modeling_qwen3_vl_visual",
                       "Qwen3VLVisualEncoder"),
        },
    ),
    ModelFamily(
        "qwen3_5",
        {
            "qwen3_5": _set(Component.LLM, Component.VISUAL),
            "qwen3_5_text": _set(Component.LLM),
        },
        {
            Component.LLM:
            _component("qwen3_5.modeling_qwen3_5_text", "Qwen3_5ForCausalLM"),
            Component.VISUAL:
            _component("qwen3_5.modeling_qwen3_5_visual",
                       "Qwen3_5VisionModel"),
        },
    ),
    ModelFamily(
        "qwen3_5_moe",
        {
            "qwen3_5_moe": _set(Component.LLM, Component.VISUAL),
            "qwen3_5_moe_text": _set(Component.LLM),
        },
        {
            Component.LLM:
            _component("qwen3_5_moe.modeling_qwen3_5_moe",
                       "Qwen3_5MoeForCausalLM"),
            Component.VISUAL:
            _component("qwen3_5_moe.modeling_qwen3_5_moe_visual",
                       "Qwen3_5MoeVisionModel"),
        },
    ),
    ModelFamily(
        "qwen3_asr",
        {
            "qwen3_asr": _set(Component.LLM, Component.AUDIO),
            "qwen3_asr_thinker": _set(Component.LLM),
        },
        {
            Component.LLM:
            _component("qwen3_asr.modeling_qwen3_asr_text",
                       "Qwen3ASRForConditionalGeneration"),
            Component.AUDIO:
            _component("qwen3_asr.modeling_qwen3_asr_audio",
                       "Qwen3ASRAudioEncoder"),
        },
    ),
    ModelFamily(
        "qwen3_omni",
        {
            "qwen3_omni":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO,
                 Component.TALKER, Component.CODE_PREDICTOR,
                 Component.CODE2WAV),
            "qwen3_omni_thinker":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "qwen3_omni_text":
            _set(Component.LLM),
            "qwen3_omni_talker":
            _set(Component.TALKER),
            "qwen3_omni_talker_code_predictor":
            _set(Component.CODE_PREDICTOR),
            "qwen3_omni_vision_encoder":
            _set(Component.VISUAL),
            "qwen3_omni_audio_encoder":
            _set(Component.AUDIO),
            "qwen3_omni_code2wav":
            _set(Component.CODE2WAV),
        },
        {
            Component.LLM:
            _component("qwen3_omni.modeling_qwen3_omni_text",
                       "Qwen3OmniThinker"),
            Component.VISUAL:
            _component("qwen3_omni.modeling_qwen3_omni_visual",
                       "Qwen3OmniVisualEncoder"),
            Component.AUDIO:
            _component("qwen3_omni.modeling_qwen3_omni_audio",
                       "Qwen3OmniAudioEncoder"),
            Component.TALKER:
            _component("qwen3_omni.modeling_qwen3_omni_talker",
                       "Qwen3OmniTalker"),
            Component.CODE_PREDICTOR:
            _component("qwen3_omni.modeling_qwen3_omni_code_predictor",
                       "Qwen3OmniCodePredictor"),
            Component.CODE2WAV:
            _component("qwen3_omni.modeling_qwen3_omni_code2wav",
                       "Qwen3OmniCode2WavModel"),
        },
    ),
    ModelFamily(
        "qwen3_omni_moe",
        {
            "qwen3_omni_moe":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO,
                 Component.TALKER, Component.CODE_PREDICTOR,
                 Component.CODE2WAV),
            "qwen3_omni_moe_thinker":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "qwen3_omni_moe_text":
            _set(Component.LLM),
            "qwen3_omni_moe_talker":
            _set(Component.TALKER),
            "qwen3_omni_moe_talker_code_predictor":
            _set(Component.CODE_PREDICTOR),
        },
        {
            Component.LLM:
            _component("qwen3_omni.modeling_qwen3_omni_moe_text",
                       "Qwen3OmniMoeThinker"),
            Component.VISUAL:
            _component("qwen3_omni.modeling_qwen3_omni_visual",
                       "Qwen3OmniVisualEncoder"),
            Component.AUDIO:
            _component("qwen3_omni.modeling_qwen3_omni_audio",
                       "Qwen3OmniAudioEncoder"),
            Component.TALKER:
            _component("qwen3_omni.modeling_qwen3_omni_moe_talker",
                       "Qwen3OmniMoeTalker"),
            Component.CODE_PREDICTOR:
            _component("qwen3_omni.modeling_qwen3_omni_code_predictor",
                       "Qwen3OmniCodePredictor"),
            Component.CODE2WAV:
            _component("qwen3_omni.modeling_qwen3_omni_code2wav",
                       "Qwen3OmniCode2WavModel"),
        },
        configuration="qwen3_omni.configuration",
        artifact_writer="qwen3_omni.artifacts",
        weight_conversion="qwen3_omni.weights",
    ),
    ModelFamily(
        "qwen3_omni_next",
        {
            "qwen3_omni_next":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO,
                 Component.TALKER, Component.CODE_PREDICTOR,
                 Component.CODE2WAV),
            "qwen3_omni_next_thinker":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "qwen3_omni_next_text":
            _set(Component.LLM),
            "qwen3_omni_next_text_moe":
            _set(Component.LLM),
            "qwen3_omni_next_talker":
            _set(Component.CODE_PREDICTOR),
            "qwen3_omni_next_talker_text":
            _set(Component.TALKER),
            "qwen3_omni_next_code_predictor":
            _set(Component.CODE_PREDICTOR),
            "qwen3_omni_next_talker_code_predictor":
            _set(Component.CODE_PREDICTOR),
            "qwen3_omni_next_vision_encoder":
            _set(Component.VISUAL),
            "qwen3_omni_next_audio_encoder":
            _set(Component.AUDIO),
            "qwen3_omni_next_code2wav":
            _set(Component.CODE2WAV),
        },
        {
            Component.LLM:
            _component("qwen3_omni_next.modeling_qwen3_omni_next_text",
                       "Qwen3OmniNextThinker"),
            Component.VISUAL:
            _component("qwen3_omni_next.modeling_qwen3_omni_next_visual",
                       "Qwen3OmniNextVisualEncoder"),
            Component.AUDIO:
            _component("qwen3_omni_next.modeling_qwen3_omni_next_audio",
                       "Qwen3OmniNextAudioEncoder"),
            Component.TALKER:
            _component("qwen3_omni_next.modeling_qwen3_omni_next_talker",
                       "Qwen3OmniNextTalker"),
            Component.CODE_PREDICTOR:
            _component(
                "qwen3_omni_next.modeling_qwen3_omni_next_code_predictor",
                "Qwen3OmniNextCodePredictor"),
            Component.CODE2WAV:
            _component("qwen3_omni_next.modeling_qwen3_omni_next_code2wav",
                       "Qwen3OmniNextCode2WavModel"),
        },
    ),
    ModelFamily(
        "qwen3_tts",
        {
            "qwen3_tts":
            _set(Component.TALKER, Component.CODE_PREDICTOR,
                 Component.CODE2WAV, Component.SPEAKER_ENCODER,
                 Component.SPEECH_TOKENIZER_ENCODER),
            "qwen3_tts_talker":
            _set(Component.TALKER),
            "qwen3_tts_code_predictor":
            _set(Component.CODE_PREDICTOR),
            "qwen3_tts_code2wav":
            _set(Component.CODE2WAV),
        },
        {
            Component.TALKER:
            _component("qwen3_tts.modeling_qwen3_tts_talker",
                       "Qwen3TTSTalker"),
            Component.CODE_PREDICTOR:
            _component("qwen3_tts.modeling_qwen3_tts_code_predictor",
                       "Qwen3TTSCodePredictor"),
            Component.CODE2WAV:
            _component("qwen3_tts.modeling_qwen3_tts_code2wav",
                       "Qwen3TTSCode2WavModel"),
            Component.SPEAKER_ENCODER:
            _component("qwen3_tts.modeling_qwen3_tts_speaker_clone_encoder",
                       "Qwen3TTSSpeakerCloneEncoder"),
            Component.SPEECH_TOKENIZER_ENCODER:
            _component(
                "qwen3_tts.modeling_qwen3_tts_speech_tokenizer_clone_encoder",
                "Qwen3TTSSpeechTokenizerCloneEncoder"),
        },
    ),
    ModelFamily(
        "internvl3",
        {"internvl_chat": _set(Component.LLM, Component.VISUAL)},
        {
            Component.LLM:
            _component("internvl3.modeling_internvl3_text",
                       "InternVL3CausalLM"),
            Component.VISUAL:
            _component("internvl3.modeling_internvl3_visual",
                       "InternVL3VisualEncoder"),
        },
    ),
    ModelFamily(
        "internvl3_5",
        {"internvl": _set(Component.LLM, Component.VISUAL)},
        {
            Component.LLM:
            _component("internvl3_5.modeling_internvl3_5_text",
                       "InternVL35CausalLM"),
            Component.VISUAL:
            _component("internvl3_5.modeling_internvl3_5_visual",
                       "InternVL35VisualEncoder"),
        },
    ),
    ModelFamily(
        "phi4mm",
        {
            model_type: _set(Component.LLM, Component.VISUAL)
            for model_type in ("phi4mm", "phi4_multimodal")
        },
        {
            Component.LLM:
            _component("phi4mm.modeling_phi4mm_text",
                       "Phi4MultimodalForCausalLM"),
            Component.VISUAL:
            _component("phi4mm.modeling_phi4mm_visual",
                       "Phi4MultimodalVisionModel"),
        },
    ),
    ModelFamily(
        "nemotron_h",
        {"nemotron_h": _set(Component.LLM)},
        {
            Component.LLM:
            _component("nemotron_h.modeling_nemotron_h",
                       "NemotronHForCausalLM")
        },
    ),
    ModelFamily(
        "nemotron3_5_asr",
        {"nemotron3_5_asr": _set(Component.AUDIO, Component.RNNT)},
        {
            Component.AUDIO:
            _component("nemotron3_5_asr.modeling_nemotron3_5_asr_audio",
                       "Nemotron3_5AsrAudioEncoder"),
            Component.RNNT:
            _component("nemotron3_5_asr.modeling_nemotron3_5_asr_rnnt",
                       "Nemotron3_5AsrRNNTStep"),
        },
    ),
    ModelFamily(
        "nemotron_omni",
        {
            "NemotronH_Nano_VL_V2":
            _set(Component.LLM, Component.VISUAL),
            "NemotronH_Nano_Omni_Reasoning_V3":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "nemotron_omni_vision_encoder":
            _set(Component.VISUAL),
            "nemotron_omni_audio_encoder":
            _set(Component.AUDIO),
        },
        {
            Component.LLM:
            _component("nemotron_omni.modeling_nemotron_omni_text",
                       "NemotronOmniCausalLM"),
            Component.VISUAL:
            _component("nemotron_omni.modeling_nemotron_omni_visual",
                       "NemotronVisualEncoder"),
            Component.AUDIO:
            _component("nemotron_omni.modeling_nemotron_omni_audio",
                       "NemotronOmniAudioEncoder"),
        },
    ),
    ModelFamily(
        "gemma4",
        {
            "gemma4": _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "gemma4_text": _set(Component.LLM),
            "gemma4_vision": _set(Component.VISUAL),
            "gemma4_audio": _set(Component.AUDIO),
        },
        {
            Component.LLM:
            _component("gemma4.modeling_gemma4_text", "Gemma4ForCausalLM"),
            Component.VISUAL:
            _component("gemma4.modeling_gemma4_visual", "Gemma4VisionModel"),
            Component.AUDIO:
            _component("gemma4.modeling_gemma4_audio", "Gemma4AudioModel"),
        },
    ),
    ModelFamily(
        "gemma4_unified",
        {
            "gemma4_unified":
            _set(Component.LLM, Component.VISUAL, Component.AUDIO),
            "gemma4_unified_text":
            _set(Component.LLM),
            "gemma4_unified_vision":
            _set(Component.VISUAL),
            "gemma4_unified_audio":
            _set(Component.AUDIO),
        },
        {
            Component.LLM:
            _component("gemma4.modeling_gemma4_text", "Gemma4ForCausalLM"),
            Component.VISUAL:
            _component("gemma4.modeling_gemma4_unified_visual",
                       "Gemma4UnifiedVisualModel"),
            Component.AUDIO:
            _component("gemma4.modeling_gemma4_unified_audio",
                       "Gemma4UnifiedAudioModel"),
        },
        configuration="gemma4.configuration",
        artifact_writer="gemma4.artifacts",
        weight_conversion="gemma4.weights",
    ),
    ModelFamily(
        "diffusion_gemma",
        {
            "diffusion_gemma": _set(Component.DLLM, Component.VISUAL),
            "diffusiongemma": _set(Component.DLLM, Component.VISUAL),
            "diffusion_gemma_text": _set(Component.DLLM),
        },
        {
            Component.DLLM:
            _component("diffusion_gemma.modeling_diffusion_gemma",
                       "DiffusionGemmaForBlockDiffusion"),
            Component.VISUAL:
            _component("diffusion_gemma.modeling_diffusion_gemma_visual",
                       "DiffusionGemmaVisionModel"),
        },
    ),
    ModelFamily(
        "cosmos3",
        {
            model_type:
            _set(Component.LLM, Component.VISUAL, Component.UND_PREFILL,
                 Component.GEN, Component.VAE_ENCODER)
            for model_type in ("cosmos3_omni", "cosmos3_edge", "cosmos3")
        } | {"cosmos3_edge_text": _set(Component.LLM)},
        {
            Component.LLM:
            _component("cosmos3.modeling_cosmos3_reasoner_text",
                       "Cosmos3ReasonerForCausalLM"),
            Component.VISUAL:
            _component("cosmos3.modeling_cosmos3_reasoner_visual",
                       "Cosmos3ReasonerVisualModel"),
            Component.UND_PREFILL:
            _component("cosmos3.modeling_cosmos3_und_prefill",
                       "Cosmos3UndPrefillModel"),
            Component.GEN:
            _component("cosmos3.modeling_cosmos3_gen", "Cosmos3GenModel"),
            Component.VAE_ENCODER:
            _component("cosmos3.modeling_cosmos3_vae_encoder",
                       "Cosmos3VaeEncoder"),
        },
    ),
    ModelFamily(
        "gemma4_assistant",
        {
            "gemma4_assistant": _set(Component.LLM),
            "gemma4_unified_assistant": _set(Component.LLM),
        },
        {
            Component.LLM:
            _component("gemma4.modeling_gemma4_assistant",
                       "Gemma4AssistantForCausalLM")
        },
        configuration="gemma4.configuration",
        artifact_writer="gemma4.artifacts",
        weight_conversion="gemma4.weights",
    ),
    ModelFamily(
        "alpamayo",
        {
            "alpamayo_r1": _set(Component.LLM, Component.VISUAL,
                                Component.ACTION)
        },
        {
            Component.LLM:
            _component("alpamayo.modeling_alpamayo_text",
                       "AlpamayoForCausalLM"),
            Component.VISUAL:
            _component("alpamayo.modeling_alpamayo_visual",
                       "AlpamayoVisualEncoder"),
            Component.ACTION:
            _component("alpamayo.modeling_alpamayo_action",
                       "AlpamayoActionModel"),
        },
    ),
)

_BY_ROOT: Dict[str, Tuple[ModelFamily, FrozenSet[Component]]] = {}
for _family in FAMILIES:
    for _root_type, _components in _family.variants.items():
        if _root_type in _BY_ROOT:
            raise RuntimeError(
                f"duplicate model_type registration: {_root_type}")
        missing = _components.difference(_family.components)
        if missing:
            names = ", ".join(sorted(component.value for component in missing))
            raise RuntimeError(
                f"{_family.name} has no model class for {names}")
        _BY_ROOT[_root_type] = (_family, _components)

SPECULATIVE_DRAFTS = {
    "eagle3":
    _component("eagle3.modeling_eagle3_draft", "Eagle3DraftModel"),
    "mtp":
    _component("qwen3_5.modeling_qwen3_5_mtp", "Qwen35MtpDraftModel"),
    "dflash":
    _component("dflash.modeling_dflash_draft", "DFlashDraftModel"),
    "jetspec":
    _component("dflash.modeling_dflash_draft", "DFlashDraftModel"),
    "dspark":
    _component("dspark.modeling_dspark_draft", "DSparkDraftModel"),
    "gemma4_mtp":
    _component("gemma4.modeling_gemma4_assistant",
               "Gemma4AssistantForCausalLM"),
}

SPECULATIVE_WEIGHT_CONVERSIONS = {
    "eagle3": "eagle3.weights",
    "mtp": "qwen3_5.weights",
    "dflash": "dflash.weights",
    "jetspec": "dflash.weights",
    "dspark": "dspark.weights",
    "gemma4_mtp": "gemma4.weights",
}

SPECULATIVE_CONFIGURATIONS = {
    "eagle3": "eagle3.configuration",
    "mtp": "qwen3_5.configuration",
    "dflash": "dflash.configuration",
    "jetspec": "jetspec.configuration",
    "dspark": "dspark.configuration",
    "gemma4_mtp": "gemma4.configuration",
}

SPECULATIVE_ARTIFACT_WRITERS = {
    "dspark": "dspark.artifacts",
}

FAMILY_SPECULATIVE_DRAFTS = {
    ("qwen3_5_moe", "mtp"):
    _component("qwen3_5_moe.modeling_qwen3_5_moe_mtp",
               "Qwen3_5MoeMtpDraftModel"),
    ("qwen3_omni_moe", "mtp"):
    _component("qwen3_omni.modeling_qwen3_omni_mtp", "Qwen3OmniMtpDraftModel"),
    ("qwen3_omni_next", "mtp"):
    _component("qwen3_omni_next.modeling_qwen3_omni_next_mtp",
               "Qwen3OmniNextMtpDraftModel"),
}

FAMILY_SPECULATIVE_WEIGHT_CONVERSIONS = {
    ("qwen3_5_moe", "mtp"): "qwen3_5_moe.weights",
    ("qwen3_omni_moe", "mtp"): "qwen3_omni.weights",
    ("qwen3_omni_next", "mtp"): "qwen3_omni_next.weights",
}

FAMILY_SPECULATIVE_CONFIGURATIONS = {
    ("qwen3_5_moe", "mtp"): "qwen3_5_moe.configuration",
    ("qwen3_omni_moe", "mtp"): "qwen3_omni.configuration",
    ("qwen3_omni_next", "mtp"): "qwen3_omni_next.configuration",
}


def _registration(root_model_type: str):
    try:
        return _BY_ROOT[root_model_type]
    except KeyError as error:
        supported = ", ".join(sorted(_BY_ROOT))
        raise ValueError(f"unsupported model_type {root_model_type!r}; "
                         f"supported model types: {supported}") from error


def family_for(root_model_type: str) -> ModelFamily:
    """Return the registered family for one HF ``model_type``."""
    return _registration(root_model_type)[0]


def components_for(root_model_type: str) -> FrozenSet[Component]:
    """Return the exact components present in a checkpoint variant."""
    return _registration(root_model_type)[1]


def definition_for(root_model_type: str, component: Component, spec_type: str,
                   spec_role: contracts.SpecRole) -> ComponentDefinition:
    """Resolve one concrete model class for a component build."""
    if spec_role == contracts.SpecRole.DRAFT:
        if component != Component.LLM:
            raise ValueError(
                "speculative draft builds require --component llm")
        family = family_for(root_model_type)
        try:
            return FAMILY_SPECULATIVE_DRAFTS.get((family.name, spec_type),
                                                 SPECULATIVE_DRAFTS[spec_type])
        except KeyError as error:
            raise ValueError(
                f"unsupported speculative draft type {spec_type!r}") from error

    family, available = _registration(root_model_type)
    if component not in available:
        raise ValueError(
            f"{root_model_type!r} has no {component.value!r} component")
    return family.components[component]


def registered_root_types() -> FrozenSet[str]:
    """Return every explicit HF ``model_type`` alias."""
    return frozenset(_BY_ROOT)


def configuration_module_for(root_model_type: str):
    """Import the configuration owned by one registered model family."""
    family = family_for(root_model_type)
    module_name = family.configuration or f"{family.name}.configuration"
    return importlib.import_module(f".{module_name}", __package__)


def artifact_writer_for(
        root_model_type: str,
        spec_type: str = "none",
        spec_role: contracts.SpecRole = contracts.SpecRole.NONE):
    """Import the runtime artifact writer owned by one model family."""
    if spec_role == contracts.SpecRole.DRAFT:
        module_name = SPECULATIVE_ARTIFACT_WRITERS.get(spec_type)
        if module_name is not None:
            return importlib.import_module(f".{module_name}", __package__)
    family = family_for(root_model_type)
    module_name = family.artifact_writer or f"{family.name}.artifacts"
    return importlib.import_module(f".{module_name}", __package__)


def weight_conversion_for(
        root_model_type: str,
        spec_type: str = "none",
        spec_role: contracts.SpecRole = contracts.SpecRole.NONE):
    """Import checkpoint conversion rules owned by a model family."""
    if spec_role == contracts.SpecRole.DRAFT:
        family = family_for(root_model_type)
        try:
            module_name = FAMILY_SPECULATIVE_WEIGHT_CONVERSIONS.get(
                (family.name, spec_type),
                SPECULATIVE_WEIGHT_CONVERSIONS[spec_type])
        except KeyError as error:
            raise ValueError(
                f"unsupported speculative draft type {spec_type!r}") from error
    else:
        family = family_for(root_model_type)
        module_name = family.weight_conversion or f"{family.name}.weights"
    return importlib.import_module(f".{module_name}", __package__)


def configure_for_build(cfg,
                        spec_role: "contracts.SpecRole | str",
                        spec_type: str,
                        *,
                        paired_target=None,
                        paired_draft_dir: str = "",
                        build_args=None):
    """Apply graph-role details through the selected model strategy."""
    role = (spec_role if isinstance(spec_role, contracts.SpecRole) else
            contracts.SpecRole(spec_role))
    cfg.engine_role = ((
        "dllm" if cfg.component == contracts.Component.DLLM.value else "llm")
                       if role == contracts.SpecRole.NONE else role.value)
    cfg.spec_decode_type = spec_type
    if role == contracts.SpecRole.NONE:
        return cfg
    family = family_for(cfg.root_model_type)
    try:
        module_name = FAMILY_SPECULATIVE_CONFIGURATIONS.get(
            (family.name, spec_type), SPECULATIVE_CONFIGURATIONS[spec_type])
    except KeyError as error:
        raise ValueError(
            f"unsupported speculative configuration {spec_type!r}") from error
    module = importlib.import_module(f".{module_name}", __package__)
    function_name = ("configure_base"
                     if role == contracts.SpecRole.BASE else "configure_draft")
    configure = getattr(module, function_name, None)
    if configure is not None:
        configure(cfg,
                  paired_target=paired_target,
                  paired_draft_dir=paired_draft_dir,
                  build_args=build_args)
    return cfg
