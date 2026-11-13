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
"""Model documents, deployment cases, and rules connecting their fields."""

from pathlib import Path
from typing import Annotated, Literal, Optional

import pydantic

from .common import (CheckpointReference, ConfigModel, NonblankString,
                     NonemptyString, checkpoint_reference)
from .deployment import EngineConfig, ExportConfig, Requirements
from .inference import GenerateInference, Inference

InputModality = Literal["text", "image", "video", "audio", "trajectory"]
OutputModality = Literal["text", "audio", "action"]
Mode = Literal["onnx", "direct_builder", "server"]
Platform = Literal[
    "l0_e2e_a30",
    "l0_e2e_a30_trtrtx",
    "l0_e2e_b100",
    "l0_e2e_drive_thor",
    "l0_e2e_orin",
    "l0_e2e_rtx5090",
    "l0_e2e_spark",
    "l0_e2e_super_thor",
    "l0_nemo_eval_a30",
]


def unique_modalities(value):
    if len(value) != len(set(value)):
        raise ValueError("modalities must not contain duplicates")
    return value


Inputs = Annotated[list[InputModality],
                   pydantic.Field(min_length=1),
                   pydantic.AfterValidator(unique_modalities)]
Outputs = Annotated[list[OutputModality],
                    pydantic.Field(min_length=1),
                    pydantic.AfterValidator(unique_modalities)]


def model_type_from_modalities(inputs: tuple[str, ...],
                               outputs: tuple[str, ...]) -> str:
    if "action" in outputs:
        return "vla"
    if ("audio" in outputs and len(outputs) > 1
            or "audio" in inputs and set(inputs) & {"image", "video"}):
        return "omni"
    if outputs == ("audio", ):
        return "tts"
    if "audio" in inputs:
        return "asr"
    if set(inputs) & {"image", "video", "trajectory"}:
        return "vlm"
    return "llm"


# Text and Omni speech from llm_inference, or speech from qwen3_tts_inference.
MULTISTAGE_PARITY_MODEL_TYPES = {"llm", "vlm", "asr", "tts", "omni"}


class ModelCase(ConfigModel):
    """Common case and inference fields for one deployment."""

    name: NonemptyString
    model_name: Optional[NonemptyString] = None
    mode: Mode
    platform: Platform
    quantized_model_id_or_dir: CheckpointReference
    inputs: Optional[Inputs] = None
    outputs: Optional[Outputs] = None
    export: ExportConfig = pydantic.Field(default_factory=ExportConfig)
    engine: EngineConfig
    inference: Annotated[list[Inference], pydantic.Field(min_length=1)]
    requirements: Requirements
    spec_decode_config: Optional[Literal["mtp", "eagle3", "dflash", "jetspec",
                                         "domino", "dspark"]] = None
    draft_id_or_dir: Optional[NonblankString] = None
    draft_model_name: Optional[NonblankString] = None
    tree_speculative_decoding: Optional[bool] = None
    pipeline: Optional[Literal["cosmos3_policy"]] = None

    @pydantic.field_validator("mode",
                              "platform",
                              "spec_decode_config",
                              "pipeline",
                              mode="before")
    @classmethod
    def strip_selection(cls, value):
        return value.strip() if isinstance(value, str) else value

    @pydantic.field_validator("model_name", "inputs", "outputs", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None:
            raise ValueError(
                "omit this field to inherit its default; null is not allowed")
        return value

    @pydantic.field_validator("inference")
    @classmethod
    def unique_inference_names(cls, entries):
        names = set()
        for entry in entries:
            name = entry.name.strip()
            if name in names:
                raise ValueError(f"duplicate inference name {name!r}")
            names.add(name)
        return entries

    @pydantic.model_validator(mode="after")
    def compatible_deployment(self):
        if self.mode != "onnx" and self.export.model_fields_set:
            raise ValueError("export settings require mode: onnx")
        method = self.spec_decode_config
        if method is None:
            for name in ("draft_id_or_dir", "draft_model_name"):
                if name in self.model_fields_set:
                    raise ValueError(f"{name} requires spec_decode_config")
        elif method == "mtp":
            if "draft_model_name" in self.model_fields_set:
                raise ValueError("draft_model_name is not used by MTP")
        else:
            if self.draft_id_or_dir is None or self.draft_model_name is None:
                raise ValueError(
                    "draft_id_or_dir and draft_model_name are required for this spec_decode_config"
                )
        if self.draft_id_or_dir is not None and not (
                method == "mtp" and self.draft_id_or_dir == "self"):
            try:
                checkpoint_reference(self.draft_id_or_dir.strip())
            except ValueError as error:
                raise ValueError(f"draft_id_or_dir: {error}") from error
        if self.tree_speculative_decoding and method not in {
                "dflash", "jetspec", "domino", "dspark"
        }:
            raise ValueError(
                "tree_speculative_decoding requires spec_decode_config: dflash, jetspec, domino, or dspark"
            )
        if self.engine.in_flight_batching:
            if self.mode != "server":
                raise ValueError(
                    "engine.in_flight_batching requires mode: server")
            if any(entry.scenario not in {"openai", "ifb_server"}
                   for entry in self.inference):
                raise ValueError(
                    "engine.in_flight_batching is served by openai and ifb_server, not by nemo_evaluator"
                )
        if self.engine.spec_verify_tokens is not None and (
                method is None or self.mode == "server"):
            raise ValueError(
                "engine.spec_verify_tokens requires spec_decode_config and mode onnx or direct_builder"
            )
        if self.pipeline is not None:
            if self.mode != "onnx":
                raise ValueError("pipeline requires mode: onnx")
            ignored = self.engine.model_fields_set - {"max_batch_size"}
            if ignored:
                raise ValueError(
                    f"pipeline {self.pipeline} does not build with engine settings {sorted(ignored)}"
                )
            if any(
                    getattr(entry, "checks", None) is not None
                    for entry in self.inference):
                raise ValueError(
                    "pipeline tests enforce their own pass criteria and report no metrics; omit checks"
                )
        return self


# Mirrors command_execution._GUIDED_DECODING_TEST_CASES: these fixtures are checked for grammar
# conformance, and the jump-forward one for a jump, which no declared metric expresses.
_GUIDED_DECODING_FIXTURES = ("llm_guided_decoding",
                             "llm_guided_decoding_jump_forward")


def _checked_by_conformance(entry: GenerateInference) -> bool:
    stem = Path(
        entry.input_file).stem if entry.input_file else (entry.test_case or "")
    return stem in _GUIDED_DECODING_FIXTURES


class ModelDocument(ConfigModel):
    """A model's declared capabilities and named deployment cases."""

    schema_version: Annotated[int, pydantic.Field(ge=1, le=1)]
    model_id: CheckpointReference
    inputs: Inputs
    outputs: Outputs
    cases: Annotated[list[ModelCase], pydantic.Field(min_length=1)]

    @pydantic.field_validator("cases")
    @classmethod
    def unique_case_names(cls, cases):
        names = set()
        for case in cases:
            if case.name in names:
                raise ValueError(f"duplicate case name {case.name!r}")
            names.add(case.name)
        return cases

    @pydantic.model_validator(mode="after")
    def supported_scenarios(self):
        allowed = {
            "onnx": {
                "generate", "runtime_bench", "compact_engine_tests",
                "ifb_trace", "media_ipc_server"
            },
            "direct_builder": {"generate", "few_layer"},
            "server": {"openai", "ifb_server", "nemo_evaluator"},
        }
        for index, case in enumerate(self.cases):
            for name in ("inputs", "outputs"):
                override = getattr(case, name)
                if override is not None and not set(override).issubset(
                        getattr(self, name)):
                    raise ValueError(
                        f"cases[{index}].{name} must be a subset of the model's {name}"
                    )
            model_type = model_type_from_modalities(
                tuple(case.inputs or self.inputs),
                tuple(case.outputs or self.outputs))
            for entry_index, entry in enumerate(case.inference):
                location = f"cases[{index}].inference[{entry_index}]"
                if entry.scenario in {
                        "runtime_bench", "compact_engine_tests", "ifb_trace"
                }:
                    if case.mode != "onnx" or model_type != "llm":
                        raise ValueError(
                            f"{location}: {entry.scenario} requires an ONNX LLM deployment"
                        )
                if entry.scenario == "media_ipc_server" and (
                        case.mode != "onnx"
                        or model_type not in {"vlm", "asr", "omni"}):
                    raise ValueError(
                        f"{location}: media_ipc_server requires an ONNX image or audio deployment"
                    )
                if entry.scenario in {
                        "compact_engine_tests", "ifb_trace", "media_ipc_server"
                } and case.spec_decode_config:
                    raise ValueError(
                        f"{location}: {entry.scenario} requires a deployment without speculative decoding"
                    )
                if entry.scenario == "ifb_server":
                    if case.mode != "server" or model_type != "llm":
                        raise ValueError(
                            f"{location}: ifb_server requires a server LLM deployment"
                        )
                    if not case.engine.in_flight_batching or case.spec_decode_config:
                        raise ValueError(
                            f"{location}: ifb_server requires engine.in_flight_batching and no speculative decoding"
                        )
                if entry.scenario not in allowed[case.mode]:
                    raise ValueError(
                        f"{location}: {case.mode} scenario must be one of {sorted(allowed[case.mode])}"
                    )
                if isinstance(entry, GenerateInference):
                    if entry.multistage_parity and (
                            case.mode != "direct_builder"
                            or model_type not in MULTISTAGE_PARITY_MODEL_TYPES
                            or case.spec_decode_config or
                        (case.engine.tp_size or 1) > 1):
                        raise ValueError(
                            f"{location}.multistage_parity requires a direct_builder generate scenario of an LLM, "
                            "VLM, ASR, TTS or Omni case without speculative decoding or tensor parallelism"
                        )
                    if (entry.multistage_parity and model_type == "omni"
                            and (case.engine.max_batch_size or 1) > 1):
                        raise ValueError(
                            f"{location}.multistage_parity on an Omni case requires max_batch_size 1: "
                            "llm_inference names each reply's WAV by its batch"
                        )
                    guided = _checked_by_conformance(entry)
                    if (model_type != "tts" and entry.checks is None
                            and not guided and case.pipeline is None):
                        raise ValueError(
                            f"{location}.checks must guard accuracy or performance"
                        )
                    if case.spec_decode_config and not guided and (
                            entry.checks is None
                            or entry.checks.performance is None):
                        raise ValueError(
                            f"{location}.checks.performance is required for speculative decoding"
                        )
            scenarios = {entry.scenario for entry in case.inference}
            if case.mode == "server" and "nemo_evaluator" in scenarios and len(
                    scenarios) != 1:
                raise ValueError(
                    f"cases[{index}]: server cannot mix nemo_evaluator with openai or ifb_server scenarios"
                )
        return self
