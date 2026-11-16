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
"""Exercise the exporter's InternVL config transformation without loading model weights."""

import ast
import pathlib

import pytest


def internvl_config_transform():
    source = pathlib.Path(
        __file__).resolve().parents[3] / "tensorrt_edgellm/onnx/cli.py"
    tree = ast.parse(source.read_text())
    export = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_export_visual")
    branch = next(
        node for node in export.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id ==
        "model_type" and isinstance(node.test.comparators[0], ast.Tuple)
        and [item.value for item in node.test.comparators[0].elts
             ] == ["internvl", "internvl_chat"])
    return compile(ast.Module(body=[branch], type_ignores=[]), str(source),
                   "exec")


@pytest.mark.parametrize("model_type", ["internvl", "internvl_chat"])
@pytest.mark.parametrize("ratio", [0.5, 0.25, 0.375, 0.0, None])
def test_preserves_checkpoint_ratio(model_type, ratio):
    # Given an explicit ratio, when writing either family's config, then preserve it for downstream validation.
    config = {
        "downsample_ratio": ratio,
        "image_token_id": 123,
        "vision_config": {
            "image_size": 448,
            "patch_size": 14,
            "model_type": "internvl"
        }
    }
    output = {"vision_config": dict(config["vision_config"])}
    exec(internvl_config_transform(), {
        "config": config,
        "vis_cfg_out": output,
        "model_type": model_type
    })
    assert output["downsample_ratio"] == ratio


@pytest.mark.parametrize("model_type", ["internvl", "internvl_chat"])
def test_missing_ratio_writes_exporter_default(model_type):
    # Given missing checkpoint metadata, when writing the sidecar, then explicitly record the shared 0.5 default.
    output = {"vision_config": {}}
    exec(
        internvl_config_transform(), {
            "config": {
                "image_token_id": 123
            },
            "vis_cfg_out": output,
            "model_type": model_type
        })
    assert output["downsample_ratio"] == 0.5
