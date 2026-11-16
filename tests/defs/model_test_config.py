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
"""Map model-owned L0 YAML cases onto the existing pytest workflows."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pydantic
import yaml

from .model_config_schema import ModelDocument, model_type_from_modalities
from .model_config_schema.common import (SHARED_ROOT_PATTERN,
                                         checkpoint_reference_shape)
from .shared_roots import resolve_shared_root

_REPO_ROOT = Path(__file__).parents[2]
MODEL_CONFIG_ROOT = _REPO_ROOT / "tests" / "model_configs" / "l0"
CUSTOM_BLOCK_MARKER = "<!-- EDGE_LLM_L0_TESTS -->"

_INFERENCE_ONLY_ENGINE_KEYS = {
    "context_cache_max_snapshots",
    "context_reuse",
    "eagle_draft_step",
    "eagle_draft_top_k",
    "in_flight_batching",
    "spec_verify_tokens",
}
_ONNX_EXPORT_PLATFORM = {
    "l0_e2e_a30_trtrtx": "l0_onnx_export_1",
    "l0_e2e_b100": "l0_onnx_export_2",
    "l0_e2e_drive_thor": "l0_onnx_export_1",
    "l0_e2e_orin": "l0_onnx_export_1",
    "l0_e2e_spark": "l0_onnx_export_2",
    "l0_e2e_super_thor": "l0_onnx_export_1",
}
_DEFAULT_TEST_CASE = {
    "llm": "llm_basic",
    "vlm": "vlm_basic",
    "asr": "asr_basic",
    "tts": "tts_basic",
    "omni": "llm_basic",
    "vla": "vla_basic",
}

_BENCH_NODE = "tests/defs/test_llm_pipeline.py::TestLLMPipeline::test_kernel_bench"
_DIRECT_NODE = "tests/defs/test_builder_pipeline.py::test_build_and_run"
_DIRECT_WEIGHTLESS_NODE = (
    "tests/defs/test_builder_pipeline.py::test_build_and_run_weightless")
_FEW_LAYER_NODE = (
    "tests/defs/test_few_layer_validation.py::test_few_layer_validation")
_NEMO_NODE = "tests/defs/test_nemo_evaluator.py::test_nemo_eval"
_SERVER_NODE = (
    "tests/defs/test_server_pipeline.py::TestHLAPI::test_http_server_lifecycle"
)
_COMPACT_ENGINE_TESTS_NODE = (
    "tests/defs/test_llm_pipeline.py::TestLLMPipeline"
    "::test_compact_engine_tests")
_IFB_TRACE_NODE = (
    "tests/defs/test_server_pipeline.py::TestServerPipeline::test_ifb_trace")
_IFB_TRACE_TEST_CASE = "ifb_smoke"
_IFB_SERVER_NODE = (
    "tests/defs/test_server_pipeline.py::TestHLAPI::test_ifb_http_server")
_MEDIA_IPC_SERVER_NODE = (
    "tests/defs/test_media_ipc_server.py::test_media_ipc_server")
_PIPELINE_NODES = {
    "llm": (
        "tests/defs/test_llm_pipeline.py::TestLLMPipeline::test_engine_build",
        "tests/defs/test_llm_pipeline.py::TestLLMPipeline::test_inference",
    ),
    "vlm": (
        "tests/defs/test_vlm_pipeline.py::TestVLMPipeline::test_engine_build",
        "tests/defs/test_vlm_pipeline.py::TestVLMPipeline::test_inference",
    ),
    "asr": (
        "tests/defs/test_audio_pipeline.py::TestASRPipeline::test_engine_build",
        "tests/defs/test_audio_pipeline.py::TestASRPipeline::test_inference",
    ),
    "tts": (
        "tests/defs/test_audio_pipeline.py::TestTTSPipeline::test_engine_build",
        "tests/defs/test_audio_pipeline.py::TestTTSPipeline::test_inference",
    ),
    "vla": (
        "tests/defs/test_vla_pipeline.py::TestVLAPipeline::test_engine_build",
        "tests/defs/test_vla_pipeline.py::TestVLAPipeline::test_inference",
    ),
}
_EXPORT_NODES = {
    "eagle3":
    "tests/defs/test_checkpoint_export.py::test_checkpoint_eagle_export",
    "mtp":
    "tests/defs/test_checkpoint_export.py::test_checkpoint_mtp_export",
    "dflash":
    "tests/defs/test_checkpoint_export.py::test_checkpoint_dflash_export",
    "jetspec":
    "tests/defs/test_checkpoint_export.py::test_checkpoint_jetspec_export",
    "dspark":
    "tests/defs/test_checkpoint_export.py::test_checkpoint_dspark_export",
}
_DEFAULT_EXPORT_NODE = (
    "tests/defs/test_checkpoint_export.py::test_checkpoint_export")
_TP_EXPORT_NODE = (
    "tests/defs/test_checkpoint_export.py::test_llm_loader_tp_export")

_COSMOS3_POLICY_TESTS = "tests/defs/test_cosmos3_policy.py"
# A pipeline case exports, builds and runs in one job: its image must carry the
# exporter, and it must have no ONNX lane (ONNX_DIR) to write into.
_CASE_PIPELINE_PLATFORMS = {"l0_e2e_a30"}
_CASE_PIPELINE_NODES = {
    "cosmos3_policy": (
        f"{_COSMOS3_POLICY_TESTS}::test_checkpoint_export",
        f"{_COSMOS3_POLICY_TESTS}::test_engine_build",
        {
            "cosmos3_droid": f"{_COSMOS3_POLICY_TESTS}::test_accuracy",
            "cosmos3_droid_lite": f"{_COSMOS3_POLICY_TESTS}::test_accuracy",
        },
    ),
}


@dataclass(frozen=True, eq=False)
class ModelTest:
    """One executable phase derived from a model-owned YAML case."""

    node: str
    platform: str
    case_id: str
    model_type: str
    model_parameters: Mapping[str, Any]
    requirements: Mapping[str, int]
    inference: Any = None

    @property
    def runtime_id(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", self.case_id).strip("-")

    def __str__(self) -> str:
        return self.runtime_id


def _resolve_checkpoint_reference(reference: str, location: str) -> str:
    """Expand a leading ``${ROOT}`` against this machine's mount for that export.

    An unresolvable root is left in place rather than raised on: collection must
    still succeed on a machine missing one export, and the unexpanded name is what
    tells the reader which variable to set when the case later reports the
    checkpoint as missing.
    """
    match = SHARED_ROOT_PATTERN.match(reference)
    if match is None:
        return reference
    name = match.group(1)
    root = resolve_shared_root(name)
    if root is None:
        return reference
    resolved = root.rstrip("/") + reference[match.end():]
    try:
        return checkpoint_reference_shape(resolved)
    except ValueError as error:
        raise ValueError(f"{location}: {error}") from error


_COMPONENT_PRECISION_SUFFIXES = (
    ("-VITMHAFP8", "visual_mha_precision", "fp8"),
    ("-VITFP8", "visual_precision", "fp8"),
    ("-AUDFP8", "audio_precision", "fp8"),
    ("-FP8-KV", "fp8_kv_cache", True),
    ("-FP8-EMBEDDING", "fp8_embedding", True),
    ("-TOKEMBNVFP4", "token_embedding_precision", "nvfp4"),
    ("-TOKEMBFP8", "token_embedding_precision", "fp8"),
    ("-PLEFP8", "ple_embedding_precision", "fp8"),
    ("-LMINT4-AWQ", "lm_head_precision", "int4_awq"),
    ("-LMNVFP4", "lm_head_precision", "nvfp4"),
    ("-LMMXFP8", "lm_head_precision", "mxfp8"),
    ("-LMFP8", "lm_head_precision", "fp8"),
)
_LLM_PRECISION_SUFFIXES = (
    ("-GPTQ-INT8", "int8_gptq"),
    ("-INT8-GPTQ", "int8_gptq"),
    ("-GPTQ-INT4", "int4_gptq"),
    ("-INT4-GPTQ", "int4_gptq"),
    ("-INT4-AWQ", "int4_awq"),
    ("-INT8-SQ", "int8_sq"),
    ("-NVFP4", "nvfp4"),
    ("-MXFP8", "mxfp8"),
    ("-FP8", "fp8"),
    ("-FP16", "fp16"),
)


def _checkpoint_precisions(reference: str) -> dict[str, Any]:
    """Infer component precisions from the checkpoint directory/repository."""
    checkpoint_name = Path(reference.rstrip("/")).name.upper().replace(
        "_", "-")
    remaining = checkpoint_name
    result: dict[str, Any] = {"lm_head_precision": "fp16"}

    while True:
        for suffix, field, value in _COMPONENT_PRECISION_SUFFIXES:
            if remaining.endswith(suffix):
                result[field] = value
                remaining = remaining[:-len(suffix)]
                break
        else:
            break

    result["llm_precision"] = "fp16"
    for suffix, precision in _LLM_PRECISION_SUFFIXES:
        if remaining.endswith(suffix):
            result["llm_precision"] = precision
            break
    return result


def _parameters(model_name: str,
                case: Mapping[str, Any],
                inference: Optional[Mapping[str, Any]] = None,
                *,
                export: bool = False,
                server: bool = False) -> dict[str, Any]:
    """Parameters of one phase of a case.

    A build phase (no ``inference``) leaves the inference-only engine options
    out so that cases differing only in them share an engine. A server case
    has no separate build phase: its single lifecycle phase (``server``) takes
    every engine option.
    """
    checkpoint = case["quantized_model_id_or_dir"]
    result: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_dir": checkpoint,
        **_checkpoint_precisions(checkpoint),
    }

    method = case.get("spec_decode_config")
    if method is not None:
        result["speculative_decoding"] = method
        draft = case.get("draft_id_or_dir")
        if method == "mtp":
            if draft not in (None, "self"):
                result["draft_checkpoint_dir"] = _resolve_checkpoint_reference(
                    draft.strip(), "case.draft_id_or_dir")
        else:
            result["draft_checkpoint_dir"] = _resolve_checkpoint_reference(
                draft.strip(), "case.draft_id_or_dir")
            result["draft_model_id"] = case["draft_model_name"].strip()
            draft_precisions = _checkpoint_precisions(
                result["draft_checkpoint_dir"])
            result["draft_llm_precision"] = draft_precisions["llm_precision"]
            result["draft_lm_head_precision"] = draft_precisions[
                "lm_head_precision"]
        if case.get("tree_speculative_decoding"):
            result["tree_speculative_decoding"] = True

    export_options = case.get("export", {})
    engine = case["engine"]
    if export:
        for name, value in export_options.items():
            if name == "int4_gemm_plugin_version":
                result["int4_gemm_plugin_v1"] = value == 1
            elif name == "trt_native_attention":
                result["trt_native_attn"] = value
            else:
                result[name] = value
        if "tp_size" in engine:
            result["tp_size"] = engine["tp_size"]
    else:
        runtime_phase = inference is not None or server
        result.update({
            name: value
            for name, value in engine.items()
            if runtime_phase or name not in _INFERENCE_ONLY_ENGINE_KEYS
        })
        if inference and inference.get("input_file"):
            input_file = inference["input_file"].strip()
            result["input_file"] = input_file
            result["test_case"] = Path(input_file).stem
        if inference and inference.get("test_case"):
            result["test_case"] = inference["test_case"].strip()
        elif inference and inference.get("scenario") == "nemo_evaluator":
            result["test_case"] = "llm_basic"
        if inference and inference.get("checks"):
            result["metric_checks"] = inference["checks"]
        if inference and "warmup" in inference:
            result["warmup"] = inference["warmup"]
        if inference and inference.get("multistage_parity"):
            result["multistage_parity"] = True
        if inference and inference.get("scenario") == "runtime_bench":
            for name in ("bench_mode", "input_len", "output_seq_len"):
                if name in inference:
                    result[name] = inference[name]
        if inference and inference.get("scenario") in {
                "compact_engine_tests", "media_ipc_server"
        }:
            result["test_case"] = inference["scenario"]
        if inference and inference.get("scenario") == "ifb_trace":
            result["test_case"] = inference.get("test_case",
                                                _IFB_TRACE_TEST_CASE)
    return result


def _test(node: str,
          platform: str,
          case_id: str,
          model_type: str,
          parameters: Mapping[str, Any],
          inference: Any = None,
          requirements: Optional[Mapping[str, int]] = None) -> ModelTest:
    return ModelTest(node, platform, case_id, model_type, dict(parameters),
                     dict(requirements or {}), inference)


def _parse_model_document(document: Any, source: str) -> list[ModelTest]:
    try:
        validated = ModelDocument.model_validate(document)
    except pydantic.ValidationError as error:
        raise ValueError(f"{source}: {error}") from error
    data = validated.model_dump(exclude_unset=True)
    model_id = _resolve_checkpoint_reference(data.get("model_id"),
                                             f"{source}.model_id")
    if Path(model_id).is_absolute():
        raise ValueError(
            f"{source}.model_id must be a Hugging Face repository "
            "ID for the provider base model")
    default_name = model_id.rsplit("/", maxsplit=1)[-1]
    default_inputs = data["inputs"]
    default_outputs = data["outputs"]

    parsed: list[ModelTest] = []
    for index, case in enumerate(data["cases"]):
        location = f"{source}.cases[{index}]"
        name = case["name"]
        mode = case["mode"]
        platform = case["platform"]
        checkpoint = _resolve_checkpoint_reference(
            case.get("quantized_model_id_or_dir"),
            f"{location}.quantized_model_id_or_dir")
        case = {**case, "quantized_model_id_or_dir": checkpoint}
        draft = case.get("draft_id_or_dir")
        checkpoint_references = (checkpoint, )
        if isinstance(draft, str) and draft != "self":
            checkpoint_references += (draft, )
        requirements = case["requirements"]
        uses_nvfp4 = any("nvfp4" in _checkpoint_precisions(reference).values()
                         for reference in checkpoint_references)
        if uses_nvfp4 and requirements["min_sm"] < 100 and mode == "onnx":
            raise ValueError(
                f"{location}.requirements.min_sm must be at least 100 for NVFP4"
            )
        model_name = case.get("model_name", default_name)
        inputs = tuple(case.get("inputs", default_inputs))
        outputs = tuple(case.get("outputs", default_outputs))
        model_type = model_type_from_modalities(inputs, outputs)
        if model_type not in _PIPELINE_NODES and mode == "onnx":
            raise ValueError(
                f"{location} has no ONNX pipeline for {model_type}")

        normalized_inference = [{
            **item, "checks": item.get("checks", {})
        } for item in case["inference"]]

        case_id = f"{model_name}/{name}"

        if mode == "onnx":
            pipeline = case.get("pipeline")
            if pipeline is not None:
                if platform not in _CASE_PIPELINE_PLATFORMS:
                    raise ValueError(
                        f"{location}.platform {platform!r} cannot run the "
                        f"{pipeline} export")
                export_platform = platform
            else:
                export_platform = _ONNX_EXPORT_PLATFORM.get(platform)
            if export_platform is None:
                raise ValueError(
                    f"{location}.platform {platform!r} has no ONNX export "
                    "lane")
            export_parameters = _parameters(model_name, case, export=True)
            if pipeline is not None:
                export_node, build_node, inference_nodes = \
                    _CASE_PIPELINE_NODES[pipeline]
                for index, item in enumerate(normalized_inference):
                    if item.get("test_case") not in inference_nodes:
                        raise ValueError(
                            f"{location}.inference[{index}].test_case must be "
                            f"one of {sorted(inference_nodes)} for {pipeline}")
            else:
                export_node = _EXPORT_NODES.get(case.get("spec_decode_config"))
                if export_node is None:
                    export_node = (_TP_EXPORT_NODE
                                   if case.get("engine", {}).get("tp_size", 1)
                                   > 1 else _DEFAULT_EXPORT_NODE)
                build_node, inference_node = _PIPELINE_NODES[model_type]
            parsed.append(
                _test(export_node, export_platform, f"{case_id}/export",
                      model_type, export_parameters))

            build_parameters = _parameters(model_name, case)
            parsed.append(
                _test(build_node,
                      platform,
                      f"{case_id}/build",
                      model_type,
                      build_parameters,
                      requirements=requirements))
            for item in normalized_inference:
                if pipeline is not None:
                    inference_node = inference_nodes[item["test_case"]]
                node = {
                    "generate": inference_node,
                    "runtime_bench": _BENCH_NODE,
                    "compact_engine_tests": _COMPACT_ENGINE_TESTS_NODE,
                    "ifb_trace": _IFB_TRACE_NODE,
                    "media_ipc_server": _MEDIA_IPC_SERVER_NODE,
                }[item["scenario"]]
                parameters = _parameters(model_name, case, item)
                parsed.append(
                    _test(node, platform, f"{case_id}/{item['name']}",
                          model_type, parameters, item, requirements))
            continue

        if mode == "direct_builder":
            for item in normalized_inference:
                if item["scenario"] == "few_layer":
                    parameters = _parameters(model_name, case, item)
                    node = _FEW_LAYER_NODE
                else:
                    parameters = _parameters(model_name, case, item)
                    node = _DIRECT_NODE
                parsed.append(
                    _test(node, platform, f"{case_id}/{item['name']}",
                          model_type, parameters, item, requirements))
                if node == _DIRECT_NODE:
                    parsed.append(
                        _test(_DIRECT_WEIGHTLESS_NODE, platform,
                              f"{case_id}/{item['name']}", model_type,
                              parameters, item, requirements))
            continue

        scenarios = {item["scenario"] for item in normalized_inference}
        if scenarios == {"nemo_evaluator"}:
            for item in normalized_inference:
                parameters = _parameters(model_name, case, item)
                parsed.append(
                    _test(_NEMO_NODE, platform, f"{case_id}/{item['name']}",
                          model_type, parameters, item, requirements))
        else:
            parameters = _parameters(model_name, case, server=True)
            parameters["test_case"] = _DEFAULT_TEST_CASE[model_type]
            openai = tuple(item for item in normalized_inference
                           if item["scenario"] == "openai")
            if openai:
                parsed.append(
                    _test(_SERVER_NODE, platform, f"{case_id}/openai",
                          model_type, parameters, openai, requirements))
            for item in normalized_inference:
                if item["scenario"] == "ifb_server":
                    parsed.append(
                        _test(_IFB_SERVER_NODE, platform,
                              f"{case_id}/{item['name']}", model_type,
                              parameters, item, requirements))
    return parsed


def _custom_document(description: str) -> Any:
    summary = re.search(r"(?ms)^## Summary\s*$\n(.*?)(?=^##\s|\Z)",
                        description)
    if not summary or CUSTOM_BLOCK_MARKER not in summary.group(1):
        raise ValueError(
            f"{CUSTOM_BLOCK_MARKER} must appear in the merge request Summary")
    tail = summary.group(1).split(CUSTOM_BLOCK_MARKER, maxsplit=1)[1]
    blocks = re.findall(r"(?ms)```ya?ml\s*\n(.*?)^```\s*$", tail)
    if len(blocks) != 1:
        raise ValueError("custom L0 marker must be followed by one YAML block")
    return yaml.safe_load(blocks[0])


def load_model_tests(
        priority: str,
        custom_description: Optional[str] = None) -> list[ModelTest]:
    """Return checked-in or MR-provided model cases for one existing L0 job."""
    tests: list[ModelTest] = []
    if custom_description is not None:
        tests.extend(
            _parse_model_document(_custom_document(custom_description),
                                  "MR Summary custom L0 block"))
    if MODEL_CONFIG_ROOT.is_dir():
        for path in sorted(MODEL_CONFIG_ROOT.rglob("*.yml")):
            with path.open(encoding="utf-8") as stream:
                tests.extend(
                    _parse_model_document(yaml.safe_load(stream),
                                          str(path.relative_to(_REPO_ROOT))))
    selected = [test for test in tests if test.platform == priority]
    unique: list[ModelTest] = []
    seen: set[tuple[str, str]] = set()
    for test in selected:
        key = (test.node, test.runtime_id)
        if key in seen:
            raise ValueError(f"duplicate model test {test.case_id!r} for "
                             f"{test.node}")
        unique.append(test)
        seen.add(key)
    return unique
