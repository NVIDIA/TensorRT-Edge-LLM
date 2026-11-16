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
"""The checkpoint-direct builder's builder_config.compact_execution field.

GPU-free: a stand-in plugin library decides, and the TensorRT build
and model artifacts are stubbed. The real plugin library's decision is covered
by CompactBuildReportTest (C++) and the plugin tests
(``test_build_reports_whether_the_layer_reserved_the_compact_workspace``).
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.builder.core.compact_build_report import \
    build_with_compact_report


def _plugin_library(compact, calls):
    """A plugin library whose compact build report ends with ``compact``."""

    def begin():
        calls.append("begin")
        return 7

    def end(report_id):
        calls.append(("end", report_id))
        return compact

    return SimpleNamespace(edgellmBeginCompactBuildReport=begin,
                           edgellmEndCompactBuildReport=end)


def test_the_report_brackets_the_build_and_yields_the_library_decision():
    """The report opens before the TensorRT build and ends after it, also
    when the build raises, with the id the library returned; the engine is
    built for compact execution exactly when the library says so, and not when
    the library has no report."""
    calls = []
    assert build_with_compact_report(_plugin_library(True, calls),
                                     lambda: calls.append("build")) == (None,
                                                                        True)
    assert calls == ["begin", "build", ("end", 7)]

    assert build_with_compact_report(_plugin_library(False, []),
                                     lambda: "engine") == ("engine", False)
    assert build_with_compact_report(SimpleNamespace(),
                                     lambda: "engine") == ("engine", False)

    calls.clear()

    def failed_build():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        build_with_compact_report(_plugin_library(True, calls), failed_build)
    assert calls == ["begin", ("end", 7)]


@pytest.mark.parametrize("compact", [True, False],
                         ids=["reserved", "not_reserved"])
def test_llm_config_records_compact_execution_of_its_build(
        tmp_path, monkeypatch, compact):
    """An LLM engine's config.json has builder_config.compact_execution: true
    exactly when its build was found built for compact execution, beside the
    builder_config the model's artifact writer wrote."""
    pytest.importorskip("tensorrt")
    from tensorrt_edgellm.builder import cli
    from tensorrt_edgellm.builder.core import builder, contracts
    from tensorrt_edgellm.builder.models import registry

    engine_dir = tmp_path / "engine"
    args = cli._parser().parse_args(
        ["--model-dir",
         str(tmp_path), "--engine-dir",
         str(engine_dir)])
    config_path = contracts.component_spec(
        contracts.Component.LLM).config_path(str(engine_dir))

    def write_artifacts(*_):
        engine_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as config_file:
            json.dump({"builder_config": {"max_batch_size": 4}}, config_file)

    monkeypatch.setattr(builder, "load_device_config", lambda _: None)
    monkeypatch.setattr(
        builder, "build_engine",
        lambda *_, **__: builder.BuildResult(engine_path="llm.engine",
                                             checkpoint_weight_bindings=(),
                                             checkpoint_dir="",
                                             checkpoint_identity={},
                                             compact_execution=compact))
    monkeypatch.setattr(
        registry, "artifact_writer_for",
        lambda *_, **__: SimpleNamespace(write_artifacts=write_artifacts))

    cli._build_one(args,
                   SimpleNamespace(root_model_type="qwen3"),
                   contracts.Component.LLM,
                   plugin_handle=None)

    with open(config_path) as config_file:
        builder_config = json.load(config_file)["builder_config"]
    if compact:
        assert builder_config["compact_execution"] is True
    else:
        assert "compact_execution" not in builder_config
    assert builder_config["max_batch_size"] == 4
