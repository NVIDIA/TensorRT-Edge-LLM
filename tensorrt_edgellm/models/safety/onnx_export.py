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
"""ONNX export for Safety models."""

import contextlib
import gc
import logging
import os
from pathlib import Path

import numpy as np
import onnx
import torch

from ...checkpoint import checkpoint_utils

logger = logging.getLogger(__name__)
_OPSET_VERSION = 24


@contextlib.contextmanager
def _permissive_inline_opset():
    """Align torch's opset-18 functions with the opset-21 FP8 translations."""
    try:
        from onnx_ir.passes.common.inliner import InlinePass
    except ImportError:
        yield
        return

    original = InlinePass._instantiate_call

    def instantiate(self, node, call_site_id):
        function = self._functions.get(node.op_identifier())
        if function is not None:
            for domain, version in list(function.opset_imports.items()):
                merged = max(self._opset_imports.get(domain, version), version)
                self._opset_imports[domain] = merged
                function.opset_imports[domain] = merged
        return original(self, node, call_site_id)

    InlinePass._instantiate_call = instantiate
    try:
        yield
    finally:
        InlinePass._instantiate_call = original


def _export(model, inputs, output_path, input_names, output_names, shapes,
            translation_table):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    logger.info("Exporting Safety ONNX to %s", output_path)
    with _permissive_inline_opset():
        program = torch.onnx.export(
            model,
            inputs,
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=shapes,
            opset_version=_OPSET_VERSION,
            custom_translation_table=translation_table,
            external_data=True,
            optimize=True,
        )
    program.save(str(output_path), external_data=True)
    with output_path.open("rb") as stream:
        os.fsync(stream.fileno())
    del program
    gc.collect()


def _prepare_decoder_for_trt(output_path):
    """Normalize decoder constants and custom-op attributes for TensorRT."""
    output_path = Path(output_path)
    model = onnx.load(str(output_path), load_external_data=False)
    for node in model.graph.node:
        if node.domain == "trt":
            for attribute in list(node.attribute):
                if attribute.name.startswith("_"):
                    node.attribute.remove(attribute)

    value_info = {value.name: value for value in model.graph.value_info}
    for tensor in model.graph.initializer:
        if 0 in tensor.dims:
            tensor.ClearField("raw_data")
            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT
            continue
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        if not tensor.dims or (len(tensor.dims) == 1 and tensor.dims[0] <= 1):
            continue
        if tensor.name.endswith((".weight_scale", ".input_scale",
                                 ".pre_quant_scale", "_scale", "_scale_2")):
            continue
        # FP16 decoder weights can be folded into FP32 initializers by dynamo.
        data = onnx.numpy_helper.to_array(
            tensor, base_dir=str(output_path.parent)).astype(np.float16)
        tensor.CopyFrom(onnx.numpy_helper.from_array(data, name=tensor.name))
        if tensor.name in value_info:
            tensor_type = value_info[tensor.name].type.tensor_type
            tensor_type.elem_type = tensor.data_type

    # Keep existing external-data references while saving adjusted constants.
    onnx.save_model(model,
                    str(output_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=output_path.name + ".data")


def export_decoder(model,
                   output_path,
                   *,
                   translation_table,
                   model_dir,
                   reduced_vocab_dir=""):
    """Export a decoder using its model-specific ONNX specification."""
    spec = model.onnx_export_spec()
    _export(spec.wrapped, spec.args, output_path, spec.input_names,
            spec.output_names, spec.dynamic_shapes, translation_table)
    _prepare_decoder_for_trt(output_path)
    checkpoint_utils.write_runtime_artifacts(
        model,
        model_dir,
        str(Path(output_path).parent),
        reduced_vocab_dir=reduced_vocab_dir)


def export_visual(model, inputs, output_path, input_names, output_names,
                  shapes, *, translation_table):
    """Export a visual encoder using its model-specific inputs and shapes."""
    _export(model, inputs, output_path, input_names, output_names, shapes,
            translation_table)
