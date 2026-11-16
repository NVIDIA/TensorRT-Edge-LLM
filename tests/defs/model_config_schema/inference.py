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
"""Inference scenarios and the settings each executor accepts."""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import pydantic

from ..utils.baseline import MODEL_TEST_METRICS
from .common import (ConfigModel, NonblankString, NonnegativeInt, Number,
                     PositiveInt, nonblank_string, reject_null)


def json_input_file(value: str) -> str:
    path = value.strip()
    if not path.endswith(".json"):
        raise ValueError("must name a JSON input file")
    if not Path(path).is_absolute() and ".." in Path(path).parts:
        raise ValueError("must not escape the repository")
    return value


JsonInputFile = Annotated[NonblankString,
                          pydantic.AfterValidator(json_input_file)]

Scalar = Union[str, pydantic.StrictInt, pydantic.StrictFloat, bool, None]


class MetricCheck(ConfigModel):
    required: Annotated[list[str], pydantic.Field(min_length=1)]
    max_regression: Annotated[float, pydantic.Field(ge=0, lt=1)]


class MetricChecks(ConfigModel):
    accuracy: Optional[MetricCheck] = None
    performance: Optional[MetricCheck] = None

    @pydantic.field_validator("accuracy", "performance")
    @classmethod
    def supported_metrics(cls, value, info):
        reject_null(value)
        names = value.required
        if len(names) != len(set(names)) or any(
                name not in MODEL_TEST_METRICS[info.field_name]
                for name in names):
            raise ValueError(
                "required must contain unique supported metric names")
        return value

    @pydantic.model_validator(mode="after")
    def require_checks(self):
        if not self.model_fields_set:
            raise ValueError("checks must declare at least one metric")
        return self


class InferenceScenario(ConfigModel):
    name: NonblankString


class GenerateInference(InferenceScenario):
    scenario: Literal["generate"] = "generate"
    input_file: Optional[JsonInputFile] = None
    test_case: Optional[str] = None
    checks: Optional[MetricChecks] = None
    warmup: Optional[NonnegativeInt] = None
    multistage_parity: bool = False

    @pydantic.field_validator("input_file", "checks")
    @classmethod
    def supplied_value(cls, value):
        return reject_null(value)

    @pydantic.model_validator(mode="after")
    def require_input(self):
        if self.input_file is None:
            if self.test_case is None or not self.test_case.strip():
                raise ValueError("requires input_file or test_case")
        elif self.test_case:
            nonblank_string(self.test_case)
        return self


class RuntimeBenchInference(InferenceScenario):
    scenario: Literal["runtime_bench"]
    bench_mode: Literal["fixed_round", "generate"]
    input_len: PositiveInt
    output_seq_len: Optional[PositiveInt] = None
    warmup: Optional[NonnegativeInt] = None


class FewLayerInference(InferenceScenario):
    scenario: Literal["few_layer"]
    input_file: JsonInputFile
    num_layers: PositiveInt
    cosine_threshold: Annotated[Number, pydantic.Field(gt=0, le=1)]
    minimum_compute_capability: NonnegativeInt = 0
    minimum_trt_version: Optional[str] = None
    timeout_seconds: PositiveInt = 1200
    arguments: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("minimum_trt_version")
    @classmethod
    def dotted_version(cls, value):
        if value is not None and (not value
                                  or not all(part.isdigit()
                                             for part in value.split("."))):
            raise ValueError('must be a dotted version string such as "11.3"')
        return value


class OpenAIInference(InferenceScenario):
    scenario: Literal["openai"]
    input_file: JsonInputFile


class Evaluator(ConfigModel):
    eval_type: NonblankString
    model_id: Optional[NonblankString] = None
    params: dict[NonblankString, Scalar] = pydantic.Field(default_factory=dict)
    overrides: dict[NonblankString,
                    Scalar] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("model_id")
    @classmethod
    def supplied_model_id(cls, value):
        return reject_null(value)


class EvaluatorScore(ConfigModel):
    key: NonblankString
    min: Number


class NemoEvaluatorInference(InferenceScenario):
    scenario: Literal["nemo_evaluator"]
    evaluator: Evaluator
    score: EvaluatorScore
    timeout_seconds: PositiveInt = 7200


class CompactEngineInference(InferenceScenario):
    scenario: Literal["compact_engine_tests"]


class IfbTraceInference(InferenceScenario):
    scenario: Literal["ifb_trace"]
    test_case: Literal["ifb_smoke", "ifb_priority",
                       "ifb_context_cache"] = "ifb_smoke"


class IfbServerInference(InferenceScenario):
    scenario: Literal["ifb_server"]


class MediaIpcServerInference(InferenceScenario):
    scenario: Literal["media_ipc_server"]


def select_scenario(value):
    if isinstance(value, dict):
        scenario = value.get("scenario", "generate")
        if isinstance(scenario, str):
            scenario = scenario.strip()
        return {**value, "scenario": scenario}
    return value


Inference = Annotated[
    Union[GenerateInference, RuntimeBenchInference, FewLayerInference,
          OpenAIInference, NemoEvaluatorInference, CompactEngineInference,
          IfbTraceInference, IfbServerInference, MediaIpcServerInference],
    pydantic.Field(discriminator="scenario"),
    pydantic.BeforeValidator(select_scenario),
]
