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
"""Dynamo-exported compressed PLE pools for checker and TensorRT probes."""

import json
import os
from pathlib import Path

import onnx
import pytest
import torch

from tensorrt_edgellm.models import ops
from tensorrt_edgellm.onnx import dynamo_translations
from tensorrt_edgellm.onnx.export import _permissive_inline_opset

_ROW_WIDTH = 160
_HEADS = 16
_SCALE = 1.1875
_BELOW_2G_CAPACITY = ((1 << 31) - 1) // _ROW_WIDTH
_LARGE_CAPACITY = (1 << 31) // _ROW_WIDTH + 1
_ABOVE_4G_CAPACITY = (1 << 32) // _ROW_WIDTH + 1


class _PlePoolFixture(torch.nn.Module):

    def __init__(self, output_dtype, scale=_SCALE):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=output_dtype))

    def forward(self, ple_cache_pool, ple_row_map):
        return ops.ple_pool_gather(ple_cache_pool, ple_row_map, self.scale)


class _PleNativePoolFixture(torch.nn.Module):

    def forward(self, ple_cache_pool, ple_row_map):
        return ops.ple_pool_lookup(ple_cache_pool, ple_row_map).flatten(-2)


def export_ple_pool_fixture(output_dir,
                            capacity,
                            output_dtype,
                            *,
                            scale=_SCALE,
                            dynamic_capacity=False,
                            pool_dtype=torch.float8_e4m3fn,
                            row_width=_ROW_WIDTH,
                            heads=_HEADS,
                            ragged=False):
    """Export the production helper without allocating the cache-pool input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype_name = "bf16" if output_dtype == torch.bfloat16 else "fp16"
    pool_dtype_name = str(pool_dtype).removeprefix("torch.")
    native = pool_dtype != torch.float8_e4m3fn
    if native and (pool_dtype not in (torch.float16, torch.bfloat16)
                   or output_dtype != pool_dtype or scale != 1.0):
        raise ValueError(
            "Native pool fixture requires matching output dtype and unit scale"
        )
    suffix = f"_{pool_dtype_name}_w{row_width}_h{heads}" if native else ""
    suffix += "_ragged" if ragged else ""
    output_path = output_dir / f"ple_pool_c{capacity}_{dtype_name}{suffix}.onnx"
    pool = torch.empty((capacity, row_width), dtype=pool_dtype, device="meta")
    row_map = torch.empty((6, heads) if ragged else (2, 3, heads),
                          dtype=torch.int32,
                          device="meta")
    map_shapes = ({
        0: torch.export.Dim("physical_tokens", min=1, max=16)
    } if ragged else {
        0: torch.export.Dim("batch", min=1, max=2),
        1: torch.export.Dim("sequence", min=1, max=8),
    })
    with _permissive_inline_opset():
        program = torch.onnx.export(
            (_PleNativePoolFixture() if native else _PlePoolFixture(
                output_dtype, scale)).eval(),
            (pool, row_map),
            dynamo=True,
            input_names=["ple_cache_pool", "ple_row_map"],
            output_names=["ple_embeddings"],
            dynamic_shapes=(
                ({
                    0: torch.export.Dim("capacity", min=1, max=capacity)
                } if dynamic_capacity else {}),
                map_shapes,
            ),
            opset_version=23,
            custom_translation_table=dynamo_translations.
            build_custom_translation_table(),
            optimize=True,
            external_data=False,
        )
    program.save(output_path, external_data=False)
    graph = onnx.load(output_path)
    onnx.checker.check_model(graph, full_check=True, check_custom_domain=True)
    manifest = {
        "onnx": output_path.name,
        "pool_binding": "ple_cache_pool",
        "map_binding": "ple_row_map",
        "output_binding": "ple_embeddings",
        "pool_dtype": pool_dtype_name,
        "map_dtype": "int32",
        "output_dtype": dtype_name,
        "capacity": capacity,
        "dynamic_capacity": dynamic_capacity,
        "ragged": ragged,
        "pool_profile": {
            "min": [1, row_width],
            "opt": [min(capacity, 4096), row_width],
            "max": [capacity, row_width]
        },
        "row_width": row_width,
        "heads": heads,
        "scale": scale,
        "pool_bytes": capacity * row_width * pool.element_size(),
        "opset": 23,
        "row_map_profile": {
            "min": [1, heads],
            "opt": [3, heads],
            "max": [16, heads],
        } if ragged else {
            "min": [1, 1, heads],
            "opt": [1, 3, heads],
            "max": [2, 8, heads],
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return output_path


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_ple_pool_gather_preserves_row_order_and_global_scale(output_dtype):
    pool = ((torch.arange(64 * _ROW_WIDTH).reshape(64, _ROW_WIDTH) % 31 - 15) /
            4).to(torch.float8_e4m3fn)
    row_map = torch.tensor([63, 0, 7, 7, 31, 1, 62, 8] * 12,
                           dtype=torch.int32).reshape(2, 3, _HEADS)
    actual = _PlePoolFixture(output_dtype)(pool, row_map)
    expected = (pool.float()[row_map.long()] *
                _SCALE).to(output_dtype).flatten(-2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == output_dtype
    assert actual.shape == (2, 3, _HEADS * _ROW_WIDTH)


@pytest.mark.parametrize("capacity", [
    pytest.param(64, id="small"),
    pytest.param(_BELOW_2G_CAPACITY, id="below2g"),
    pytest.param(_LARGE_CAPACITY, id="above2g"),
    pytest.param(_ABOVE_4G_CAPACITY, id="above4g"),
])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_dynamo_exports_fp8_pool_gather_before_dequantize(
        tmp_path, capacity, output_dtype):
    output_dir = os.environ.get("EDGELLM_PLE_FIXTURE_DIR", str(tmp_path))
    output_path = export_ple_pool_fixture(output_dir, capacity, output_dtype)
    model = onnx.load(output_path)
    inputs = {value.name: value for value in model.graph.input}
    assert set(inputs) == {"ple_cache_pool", "ple_row_map"}
    pool_type = inputs["ple_cache_pool"].type.tensor_type
    assert pool_type.elem_type == onnx.TensorProto.FLOAT8E4M3FN
    assert [d.dim_value for d in pool_type.shape.dim] == [capacity, _ROW_WIDTH]
    map_type = inputs["ple_row_map"].type.tensor_type
    assert map_type.elem_type == onnx.TensorProto.INT32
    assert all(d.dim_param for d in map_type.shape.dim[:2])
    assert map_type.shape.dim[2].dim_value == _HEADS
    gathers = [n for n in model.graph.node if n.op_type == "PleGatherPlugin"]
    assert len(gathers) == 1
    gather = gathers[0]
    assert gather.domain == "trt_edgellm"
    assert {
        a.name: onnx.helper.get_attribute_value(a)
        for a in gather.attribute
    } == {
        "plugin_version": b"1"
    }
    assert not any(n.op_type in ("Gather", "TRT_Gather")
                   for n in model.graph.node)
    assert list(gather.input) == ["ple_cache_pool", "ple_row_map"]
    dequant = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dequant) == 1
    assert dequant[0].input[0] == gather.output[0]
    constants = {value.name: value for value in model.graph.initializer}
    assert "ple_cache_pool" not in constants
    assert float(onnx.numpy_helper.to_array(
        constants[dequant[0].input[1]])) == _SCALE
    output_type = model.graph.output[0].type.tensor_type
    expected_type = (onnx.TensorProto.BFLOAT16 if output_dtype
                     == torch.bfloat16 else onnx.TensorProto.FLOAT16)
    assert output_type.elem_type == expected_type
    assert output_type.shape.dim[-1].dim_value == _HEADS * _ROW_WIDTH
    assert output_path.stat().st_size < 65536


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_pool_lookup_preserves_bits_and_dynamic_capacity(
        tmp_path, dtype):

    class Lookup(torch.nn.Module):

        def forward(self, pool, row_map):
            return ops.ple_pool_lookup(pool, row_map)

    pool = (torch.arange(32 * 24).reshape(32, 24) / 13).to(dtype)
    row_map = torch.tensor([31, 0, 7, 7, 19, 1],
                           dtype=torch.int32).reshape(2, 3, 1)
    torch.testing.assert_close(Lookup()(pool, row_map),
                               pool[row_map.long()],
                               rtol=0,
                               atol=0)
    with _permissive_inline_opset():
        program = torch.onnx.export(
            Lookup(),
            (pool, row_map),
            dynamo=True,
            opset_version=23,
            input_names=["ple_cache_pool", "ple_row_map"],
            dynamic_shapes=({
                0: torch.export.Dim("capacity", min=1, max=128)
            }, {
                0: torch.export.Dim("batch", min=1, max=8),
                1: torch.export.Dim("sequence", min=1, max=32)
            }),
            custom_translation_table=dynamo_translations.
            build_custom_translation_table(),
        )
    path = tmp_path / "native.onnx"
    program.save(path, external_data=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True, check_custom_domain=True)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    assert graph.graph.output[
        0].type.tensor_type.elem_type == graph.graph.input[
            0].type.tensor_type.elem_type
    assert [n.op_type for n in graph.graph.node] == ["PleGatherPlugin"]
    for capacity in (8, 64):
        values = torch.randn(capacity, 24).to(dtype)
        indices = torch.tensor([capacity - 1, 0],
                               dtype=torch.int32).reshape(1, 2, 1)
        torch.testing.assert_close(program.exported_program.module()(values,
                                                                     indices),
                                   values[indices.long()],
                                   rtol=0,
                                   atol=0)


def test_fp8_pool_export_has_dynamic_capacity(tmp_path):
    path = export_ple_pool_fixture(tmp_path,
                                   _ABOVE_4G_CAPACITY,
                                   torch.float16,
                                   dynamic_capacity=True)
    graph = onnx.load(path)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param


@pytest.mark.parametrize("pool_dtype",
                         [torch.float8_e4m3fn, torch.float16, torch.bfloat16])
def test_ragged_pool_export_preserves_token_axis(tmp_path, pool_dtype):
    native = pool_dtype != torch.float8_e4m3fn
    output_dtype = pool_dtype if native else torch.float16
    scale = 1.0 if native else _SCALE
    path = export_ple_pool_fixture(tmp_path,
                                   64,
                                   output_dtype,
                                   scale=scale,
                                   dynamic_capacity=True,
                                   pool_dtype=pool_dtype,
                                   heads=3,
                                   row_width=24,
                                   ragged=True)
    graph = onnx.load(path).graph
    inputs = {value.name: value for value in graph.input}
    pool_shape = inputs["ple_cache_pool"].type.tensor_type.shape.dim
    map_shape = inputs["ple_row_map"].type.tensor_type.shape.dim
    output_shape = graph.output[0].type.tensor_type.shape.dim
    assert len(map_shape) == 2 and map_shape[1].dim_value == 3
    assert len(output_shape) == 2 and output_shape[1].dim_value == 72
    assert map_shape[0].dim_param == output_shape[0].dim_param
    assert pool_shape[0].dim_param != map_shape[0].dim_param
    pool = (torch.arange(64 * 24).reshape(64, 24) / 13).to(pool_dtype)
    row_map = torch.tensor([[63, 0, 7], [7, 19, 1]], dtype=torch.int32)
    expected = (pool.float()[row_map.long()] *
                scale).to(output_dtype).flatten(-2)
    actual = (ops.ple_pool_lookup(pool, row_map).flatten(-2)
              if native else ops.ple_pool_gather(
                  pool, row_map, torch.tensor(scale, dtype=output_dtype)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
