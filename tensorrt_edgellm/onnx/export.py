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
"""
ONNX export via ``torch.onnx.export(dynamo=True)``.


One graph covers prefill (``past_len=0``) and decode (``past_len>0``); custom
attention and Mamba ops expose state as I/O.

ONNX input / output layout - attention-only model
--------------------------------------------------
Inputs:
    inputs_embeds           [physical_tokens, hidden_size]           float16
    past_key_values_0..N    [2, num_pages, 128, num_kv_heads, head_dim] float16 (paged pool)
    rope_rotary_cos_sin     [physical_tokens, rotary_dim] float32
    token metadata          token rows [physical_tokens], sequence rows [batch]
    state_indices           [batch]                       int32
    execution_phase_marker  [phase_extent]                int32
    kv_page_table           [batch, 2, max_pages_per_seq] int32
    logits_indices          [selected_tokens]             int64

Outputs:
    logits                  [selected_tokens, vocab_size]            float32
    present_key_values_0..N [2, num_pages, 128, num_kv_heads, head_dim] float16 (aliases past)

Additional I/O for hybrid (Mamba) models
-----------------------------------------
Extra inputs:
    conv_state_0..M   [resident_rows, conv_dim, conv_kernel]        float16
    ssm_state_0..M    [resident_rows, num_heads, head_dim, ssm_state] float16

Extra outputs:
    present_conv_0..M   updated conv states
    present_ssm_0..M    updated ssm states
"""

import contextlib
import gc
import logging
import os

import onnx
import torch

from ..checkpoint.checkpoint_utils import write_runtime_artifacts
from ..external_weights import (externalize_model_weights,
                                patch_external_weight_manifest,
                                reject_quantized_lm_head_externalization,
                                resolve_externalize_weights)
from ..models.default.modeling_default import CausalLM
from ..models.ops import nvfp4_a16_export_target_sm
from .dynamo_translations import build_custom_translation_table
from .onnx_custom_schemas import _attention_plugin_schema

_ATTENTION_SINK_INIT_PREFIX = "attention_sinks_fp32"

logger = logging.getLogger(__name__)

__all__ = ["export_onnx"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_onnx(
    model: CausalLM,
    output_path: str,
    model_dir: str = "",
    fp8_embedding: bool = False,
    reduced_vocab_dir: str = "",
    externalize_weights=None,
    config_filename: str = "config.json",
    write_shared_artifacts: bool = True,
    fp8_ple: bool = False,
) -> None:
    """Export *model* to ONNX using the dynamo exporter.

    Writes ``model.onnx``, ``model.onnx.data``, the runtime config (named
    *config_filename*), ``embedding.safetensors``, and any tokenizer files
    present in *model_dir* to the same output directory.

    Args:
        model:       A :class:`~modules.CausalLM` with weights loaded.
        output_path: Destination ``.onnx`` file path.
        model_dir:   Checkpoint directory (for tokenizer file copying).
                     If empty, tokenizer files are skipped.
        fp8_embedding: Quantize embedding.safetensors to FP8 E4M3 with
                       one FP32 per-tensor scale.
        reduced_vocab_dir: Directory containing ``vocab_map.safetensors``
                           when reduced vocabulary is enabled.
        externalize_weights: Iterable of weight kinds to expose as fixed-shape
                             ONNX inputs and save to safetensors external
                             weight files.
                             Supported kinds: ``int4_ffn``, ``int4_moe``,
                             ``nvfp4_moe``, ``lm_head``, and ``all``.
        config_filename: Filename for the runtime config beside the ONNX.
                         Use ``"config.json"`` for single-device exports
                         or ``"config_world{N}.json"`` for multi-rank
                         exports.
        write_shared_artifacts: Emit shared embedding/tokenizer files. Set to
                                False on non-rank-0 per-rank exports to avoid
                                redundant rewrites of identical sidecar files.
        fp8_ple: Quantize the Gemma4 or Qwen4-Exp PLE table to per-tensor
                 FP8 at export time.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    if fp8_ple and not model_dir:
        raise ValueError(
            "FP8 PLE export requires the source checkpoint directory")
    if model_dir:
        from ..checkpoint.embedding_quantization import prepare_ple_export
        prepare_ple_export(model, model_dir, out_dir, fp8_ple)

    requested_external_weights = resolve_externalize_weights(
        externalize_weights)
    reject_quantized_lm_head_externalization(model, model_dir,
                                             requested_external_weights)

    external_weight_files = _export_model(
        model,
        output_path,
        externalize_weights=requested_external_weights,
    )
    write_runtime_artifacts(model,
                            model_dir,
                            out_dir,
                            fp8_embedding=fp8_embedding,
                            reduced_vocab_dir=reduced_vocab_dir,
                            config_filename=config_filename,
                            write_shared_artifacts=write_shared_artifacts)
    if external_weight_files:
        patch_external_weight_manifest(out_dir, external_weight_files)


# ---------------------------------------------------------------------------
# ONNX post-processing: TRT compatibility
# ---------------------------------------------------------------------------


def _fix_nvfp4_weight_dtype(onnx_path: str) -> None:
    """Reinterpret INT8 NVFP4 weight initializers as FLOAT4E2M1.

    Our model stores packed FP4 weights as int8 [out, in//2] (2 nibbles per byte).
    TRT's DequantizeLinear with block_size requires FLOAT4E2M1 (elem_type=23)
    type with logical shape [out, in] -- same bytes, different ONNX element type
    and shape declaration.

    This pass finds the int8 initialisers whose name ends with ``.weight``
    (NVFP4 linear weights) and that no INT4 GEMM plugin consumes, and rewrites
    them:
      elem_type  INT8  -> FLOAT4E2M1 (23)
      dims       [N, M] -> [N, M*2]   (double the last dim -- nibble unpacking)
    The value_info of each dequantize output fed by such a weight gets the
    unpacked dims and the element type the node produces (fp32 for the dense
    ``DequantizeLinear``, the ``output_dtype`` of ``TRT_BlockDequantize``).
    """
    _FLOAT4E2M1 = 23  # ONNX TensorProto.FLOAT4E2M1
    _INT8 = 3  # ONNX TensorProto.INT8
    _FLOAT = 1  # ONNX TensorProto.FLOAT
    _FLOAT16 = 10  # ONNX TensorProto.FLOAT16

    model = onnx.load(onnx_path, load_external_data=False)
    value_infos = {
        value_info.name: value_info
        for value_info in (list(model.graph.input) + list(model.graph.output) +
                           list(model.graph.value_info))
    }

    def _set_value_info(name: str, elem_type: int, dims: list[int]) -> bool:
        value_info = value_infos.get(name)
        if value_info is None:
            return False
        tensor_type = value_info.type.tensor_type
        tensor_type.elem_type = elem_type
        del tensor_type.shape.dim[:]
        for dim_value in dims:
            dim = tensor_type.shape.dim.add()
            dim.dim_value = int(dim_value)
        return True

    changed = 0
    fixed_value_infos = 0
    fp4_weight_dims = {}
    # Stacked NVFP4 MoE expert weights for the TRT-native W4A8 and W4A4 arms
    # (``Qwen3SparseMoeBlock._prepare_native_moe_weights`` and
    # ``_prepare_native_moe_w4a4_weights``,
    # ``NemotronHMoEMLP._prepare_for_export_native`` and
    # ``_prepare_for_export_native_w4a4``). They are plain buffers, so their
    # initializer names do not end in ".weight", but they are packed FP4
    # exactly like the dense linear weights and need the same INT8 ->
    # FLOAT4E2M1 reinterpretation.
    _MOE_FP4_BUFFERS = ("nvfp4_gate_weight", "nvfp4_up_weight",
                        "nvfp4_down_weight")
    # Plugins that read packed INT8 weights, e.g. an INT4 lm_head in an NVFP4
    # mixed-precision model.
    _PACKED_WEIGHT_OPS = ("Q8_0LinearPlugin", "Int4GroupwiseGemmPlugin",
                          "Int4GroupwiseGemmPluginV2")
    packed_linear_weights = {
        name
        for node in model.graph.node if node.op_type in _PACKED_WEIGHT_OPS
        for name in node.input[1:]
    }

    for init in model.graph.initializer:
        # Only NVFP4 tensors whose name ends with ".weight" (not scale /
        # qweight), plus the stacked MoE FP4 expert buffers.
        is_fp4_name = (init.name.endswith(".weight")
                       or init.name.endswith(_MOE_FP4_BUFFERS))
        if (not is_fp4_name or init.name in packed_linear_weights
                or "scale" in init.name or "qweight" in init.name):
            continue
        if len(init.dims) < 1:
            continue

        if init.data_type == _INT8:
            # Reinterpret: same raw bytes, element type -> FLOAT4E2M1, last dim *2.
            init.data_type = _FLOAT4E2M1
            old_dims = list(init.dims)
            init.dims[-1] = old_dims[-1] * 2
            changed += 1
        elif init.data_type != _FLOAT4E2M1:
            continue

        new_dims = list(init.dims)
        fp4_weight_dims[init.name] = new_dims
        fixed_value_infos += int(
            _set_value_info(init.name, _FLOAT4E2M1, new_dims))

    if not fp4_weight_dims:
        return

    initializer_dtypes = {
        init.name: init.data_type
        for init in model.graph.initializer
    }

    def _dequant_output_dtype(node) -> int:
        """Element type the dequantize node produces.

        The dense NVFP4 linears use the standard ``DequantizeLinear`` with an
        fp32 per-block scale (float32 out, cast to fp16 afterwards). The
        TRT-native MoE experts use ``TRT_BlockDequantize`` with an fp16 block
        scale and an explicit ``output_dtype`` (fp16 out, fed to ``TRT_MoE``
        directly), so their value_info must say fp16.
        """
        if node.op_type == "TRT_BlockDequantize":
            for attr in node.attribute:
                if attr.name == "output_dtype":
                    return int(attr.i)
            if len(node.input) > 1:  # the output type follows the scale
                return initializer_dtypes.get(node.input[1], _FLOAT16)
            return _FLOAT16
        return _FLOAT

    for node in model.graph.node:
        # TRT_BlockDequantize is the MoE weight path (it carries an explicit
        # block_shape, which Myelin's moe lowering requires).
        if node.op_type not in ("DequantizeLinear",
                                "TRT_BlockDequantize") or not node.input:
            continue
        weight_dims = fp4_weight_dims.get(node.input[0])
        if weight_dims is None:
            continue
        out_dtype = _dequant_output_dtype(node)
        for output_name in node.output:
            fixed_value_infos += int(
                _set_value_info(output_name, out_dtype, weight_dims))

    if changed:
        logger.info("TRT fix: reinterpreted %d NVFP4 weight(s) as FLOAT4E2M1",
                    changed)
    if fixed_value_infos:
        logger.info("TRT fix: updated %d NVFP4 weight value_info entries",
                    fixed_value_infos)
    if changed == 0 and fixed_value_infos == 0:
        return
    data_file = os.path.basename(onnx_path) + ".data"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )


def _encode_fp8_kv_cache_as_int8(onnx_path: str) -> None:
    """Represent FP8 KV-cache graph I/O as raw INT8 E4M3 bytes.

    TensorRT cannot expose FP8 bindings on Ampere. The attention plugin's
    ``enable_fp8_kv_cache`` attribute defines the byte encoding, so INT8 is a
    transport type and does not change cache quantization semantics.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    changed = 0
    for value in (list(model.graph.input) + list(model.graph.output) +
                  list(model.graph.value_info)):
        if not value.name.startswith(
            ("past_key_values_", "present_key_values_")):
            continue
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT8E4M3FN:
            tensor_type.elem_type = onnx.TensorProto.INT8
            changed += 1
    if not changed:
        return
    logger.info("TRT fix: represented %d FP8 KV-cache value(s) as INT8 bytes",
                changed)
    data_file = os.path.basename(onnx_path) + ".data"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )


def _strip_attention_plugin_optional_inputs(onnx_path: str) -> None:
    """Strip disabled optional inputs from AttentionPlugin ONNX nodes.

    The onnxscript translation always emits the full optional layout:
    q/k norm gammas, context-mask selector, tree/vision mask inputs, and runtime
    shape selectors. The C++ plugin expects those optional groups compacted in
    that relative order, with disabled groups removed from the ONNX node input
    list.
    """
    # Node input positions as the translation emits them: the schema's input
    # order, with the four token-metadata inputs and the two prefix inputs as
    # groups.
    position = {
        param.name: index
        for index, param in enumerate(_attention_plugin_schema.inputs)
    }
    _NUM_REQUIRED = position["q_norm_gamma"]
    _GAMMA_POSITIONS = (position["q_norm_gamma"], position["k_norm_gamma"])
    _CONTEXT_MASK_SELECTOR_POSITION = position["context_mask_selector"]
    _ATTENTION_MASK_POSITION = position["attention_mask"]
    _ATTENTION_POS_ID_POSITION = position["attention_pos_id"]
    _SKIP_PREFILL_SCALE_POSITION = position["skip_softmax_prefill_scale"]
    _SKIP_DECODE_SCALE_POSITION = position["skip_softmax_decode_scale"]
    _SWA_KV_CACHE_MODE_POSITION = position["swa_kv_cache_mode"]
    _ATTENTION_SINKS_POSITION = position["attention_sinks"]
    _TOKEN_METADATA_POSITIONS = range(
        position["query_start_offsets"],
        position["context_sequence_count_carrier"] + 1)
    _PREFIX_KV_POSITIONS = (position["prefix_key_value"],
                            position["prefix_page_table"])
    model = onnx.load(onnx_path, load_external_data=False)
    changed = 0
    mode_shape_changed = False
    dropped_const_tensors: set = set()
    for graph_input in model.graph.input:
        if graph_input.name != "swa_kv_cache_mode":
            continue
        tensor_type = graph_input.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.INT32:
            raise ValueError("swa_kv_cache_mode must be an INT32 tensor")
        if len(tensor_type.shape.dim) != 1:
            raise ValueError("swa_kv_cache_mode must be a 1-D tensor")
        mode_dim = tensor_type.shape.dim[0]
        if mode_dim.dim_param != "swa_kv_cache_mode_len":
            mode_dim.ClearField("dim_value")
            mode_dim.dim_param = "swa_kv_cache_mode_len"
            mode_shape_changed = True
    for node in model.graph.node:
        if node.op_type != "AttentionPlugin":
            continue
        tree_attn = next(
            (a.i for a in node.attribute if a.name == "enable_tree_attention"),
            0,
        )
        context_mask_selector = next(
            (a.i for a in node.attribute
             if a.name == "enable_context_mask_selector"),
            0,
        )
        vision_block_attn = next(
            (a.i for a in node.attribute
             if a.name == "enable_vision_block_attention"),
            0,
        )
        qk_norm = next(
            (a.i for a in node.attribute if a.name == "enable_qk_norm"),
            0,
        )
        attention_sink = next(
            (a.i for a in node.attribute if a.name == "enable_attention_sink"),
            0,
        )
        skip_prefill_input = next(
            (a.i for a in node.attribute
             if a.name == "skip_softmax_prefill_input"),
            0,
        )
        skip_decode_input = next(
            (a.i
             for a in node.attribute if a.name == "skip_softmax_decode_input"),
            0,
        )
        inputs = list(node.input)

        def get_input(index: int) -> str:
            return inputs[index] if index < len(inputs) else ""

        new_inputs = inputs[:_NUM_REQUIRED]
        if qk_norm:
            new_inputs += [get_input(i) for i in _GAMMA_POSITIONS]
        else:
            dropped_const_tensors.update(
                get_input(i) for i in _GAMMA_POSITIONS if get_input(i))
        if context_mask_selector:
            new_inputs.append(get_input(_CONTEXT_MASK_SELECTOR_POSITION))
        if tree_attn:
            new_inputs.extend([
                get_input(_ATTENTION_MASK_POSITION),
                get_input(_ATTENTION_POS_ID_POSITION),
            ])
        if vision_block_attn:
            new_inputs.append(get_input(_ATTENTION_MASK_POSITION))
        if skip_prefill_input:
            skip_prefill_table = get_input(_SKIP_PREFILL_SCALE_POSITION)
            if skip_prefill_table:
                new_inputs.append(skip_prefill_table)
        if skip_decode_input:
            skip_decode_table = get_input(_SKIP_DECODE_SCALE_POSITION)
            if skip_decode_table:
                new_inputs.append(skip_decode_table)
        # The final shape-only policy input is the single source of truth for
        # bounded SWA capability. Materialize the corresponding TensorRT plugin
        # field only after optional inputs have been normalized.
        swa_mode_input = get_input(_SWA_KV_CACHE_MODE_POSITION)
        supports_bounded_attr = next(
            (a
             for a in node.attribute if a.name == "supports_bounded_kv_cache"),
            None,
        )
        node_changed = False
        if swa_mode_input:
            new_inputs.append(swa_mode_input)
            if supports_bounded_attr is None:
                node.attribute.append(
                    onnx.helper.make_attribute("supports_bounded_kv_cache", 1))
                node_changed = True
            elif supports_bounded_attr.i != 1:
                supports_bounded_attr.i = 1
                node_changed = True
        elif supports_bounded_attr is not None:
            node.attribute.remove(supports_bounded_attr)
            node_changed = True
        sink_input = get_input(_ATTENTION_SINKS_POSITION)
        if attention_sink:
            if sink_input:
                new_inputs.append(sink_input)
        elif sink_input:
            dropped_const_tensors.add(sink_input)
        new_inputs.extend(get_input(i) for i in _TOKEN_METADATA_POSITIONS)
        if any(a.name == "external_kv_prefix" and a.i for a in node.attribute):
            new_inputs.extend(get_input(i) for i in _PREFIX_KV_POSITIONS)
        if new_inputs != inputs:
            del node.input[:]
            node.input.extend(new_inputs)
            node_changed = True
        changed += int(node_changed)

    # Prune disabled gamma/sink Constant and Cast chains that no longer feed a node.
    if dropped_const_tensors:
        consumed = {i for n in model.graph.node for i in n.input}
        graph_outputs = {o.name for o in model.graph.output}
        pruned = True
        while pruned:
            pruned = False
            for n in list(model.graph.node):
                if not n.output:
                    continue
                if all(o in dropped_const_tensors and o not in consumed
                       and o not in graph_outputs for o in n.output):
                    model.graph.node.remove(n)
                    dropped_const_tensors.update(n.input)
                    consumed = {
                        i
                        for node_ in model.graph.node
                        for i in node_.input
                    }
                    pruned = True
        # The dynamo exporter may lift the gamma Constants to graph
        # initializers instead of Constant nodes — drop those as well.
        for init in list(model.graph.initializer):
            if init.name in dropped_const_tensors and init.name not in consumed:
                model.graph.initializer.remove(init)

    # Drop the graph-level skip-softmax table inputs no node consumes any more
    # (skip_softmax_prefill_input == 0 / skip_softmax_decode_input == 0
    # everywhere): a dangling ONNX graph input still becomes a TRT network
    # input, so the dense engine would carry the binding it is supposed to be
    # free of.
    consumed_inputs = {i for n in model.graph.node for i in n.input}
    for graph_input in list(model.graph.input):
        if (graph_input.name
                in ("skip_softmax_prefill_scale", "skip_softmax_decode_scale")
                and graph_input.name not in consumed_inputs):
            model.graph.input.remove(graph_input)
            changed += 1

    if not changed and not mode_shape_changed:
        return
    logger.info(
        "TRT fix: normalized optional inputs on %d AttentionPlugin node(s)%s",
        changed,
        " and restored the dynamic SWA mode shape"
        if mode_shape_changed else "",
    )
    data_file = os.path.basename(onnx_path) + ".data"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )


def _fix_zero_volume_initializers(onnx_path: str) -> None:
    """Clear stray payload bytes on zero-volume initializers.

    A zero-volume initializer must carry zero payload bytes, otherwise the
    TRT ONNX parser rejects the model with a size mismatch. Only the model
    proto is rewritten; external-data references are left untouched.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    fixed = 0
    for init in model.graph.initializer:
        volume = 1
        for d in init.dims:
            volume *= d
        if volume != 0:
            continue
        has_payload = (len(init.raw_data) > 0 or len(init.external_data) > 0
                       or init.data_location != onnx.TensorProto.DEFAULT)
        if not has_payload:
            continue
        init.raw_data = b""
        del init.external_data[:]
        init.data_location = onnx.TensorProto.DEFAULT
        fixed += 1
    if not fixed:
        return
    logger.info(
        "TRT fix: cleared stray payload on %d zero-volume initializer(s)",
        fixed)
    onnx.save(model, onnx_path)


def _strip_onnxscript_internal_attrs(onnx_path: str) -> None:
    """Remove ``_outputs`` attributes injected by onnxscript multi-output ops.

    onnxscript emits ``_outputs=N`` on custom-domain nodes with multiple
    outputs (e.g. TRT_MXFP8DynamicQuantize).  TRT does not recognise this
    attribute and may reject the graph.  Strip all attrs whose name starts
    with ``_`` from ``trt::`` domain nodes.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    stripped = 0
    for node in model.graph.node:
        if node.domain != "trt":
            continue
        internal = [a for a in node.attribute if a.name.startswith("_")]
        for a in internal:
            node.attribute.remove(a)
            stripped += 1
    if not stripped:
        return
    logger.info("TRT fix: stripped %d internal onnxscript attr(s)", stripped)
    data_file = os.path.basename(onnx_path) + ".data"
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )


def _dedup_shared_dql_scales(model) -> int:
    """Duplicate shared DequantizeLinear scale initializers in-place.

    The dynamo exporter deduplicates identical scalar initializers (e.g.
    per-tensor NVFP4 global scales) into a single initializer referenced
    by DequantizeLinear nodes across many layers.  TRT's compiler backend
    segfaults when a single scalar initializer fans out to many DQL nodes
    spanning different transformer layers.

    Only small initializers (≤ 1 KB, i.e. scalars and small vectors) are
    duplicated.  In practice the shared tensors are 4-byte FP32 scalars so
    the total overhead is a few hundred bytes.  Large shared tensors are
    left untouched to avoid doubling model memory.

    Operates on an already-loaded ``onnx.ModelProto`` in-place and returns
    the number of duplicated references (0 means no changes).
    """
    _MAX_DUP_BYTES = 1024  # only duplicate initializers up to 1 KB

    # Collect all initializer names consumed by DQL nodes, counting
    # how many *distinct* DQL consumers each has.
    dql_consumers: dict[str, list] = {}  # init_name -> [node_indices]
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        for inp in node.input:
            dql_consumers.setdefault(inp, []).append(idx)

    # Only care about initializers with >1 DQL consumer
    init_map = {init.name: init for init in model.graph.initializer}
    shared = {
        name: indices
        for name, indices in dql_consumers.items()
        if name in init_map and len(indices) > 1
    }
    if not shared:
        return 0

    duplicated = 0
    skipped = 0
    for init_name, node_indices in shared.items():
        orig = init_map[init_name]
        nbytes = _initializer_data_nbytes(orig)
        if nbytes > _MAX_DUP_BYTES:
            skipped += 1
            logger.warning(
                "TRT fix: skipping large shared DQL initializer %s "
                "(%d bytes, %d consumers) — would double memory", init_name,
                nbytes, len(node_indices))
            continue

        # Keep first consumer using the original; duplicate for the rest
        for seq, nidx in enumerate(node_indices[1:], start=1):
            clone_name = f"{init_name}__dup{seq}"
            clone = onnx.TensorProto()
            clone.CopyFrom(orig)
            clone.name = clone_name
            model.graph.initializer.append(clone)

            # Patch the DQL node input to point at the clone
            node = model.graph.node[nidx]
            for i, inp in enumerate(node.input):
                if inp == init_name:
                    node.input[i] = clone_name
                    break
            duplicated += 1

    if duplicated:
        logger.info(
            "TRT fix: duplicated %d shared DQL scale ref(s) "
            "(%d unique, %d skipped as too large)", duplicated,
            len(shared) - skipped, skipped)
    return duplicated


def _external_data_value(init, key: str) -> "str | None":
    for entry in init.external_data:
        if entry.key == key:
            return entry.value
    return None


def _initializer_data_nbytes(init) -> int:
    if init.raw_data:
        return len(init.raw_data)
    length = _external_data_value(init, "length")
    if length is None:
        return 0
    try:
        return int(length)
    except ValueError:
        return 0


def _count_shared_dql_scale_duplicates(model) -> int:
    _MAX_DUP_BYTES = 1024
    dql_consumers: dict[str, list] = {}
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        for inp in node.input:
            dql_consumers.setdefault(inp, []).append(idx)

    init_map = {init.name: init for init in model.graph.initializer}
    duplicated = 0
    for init_name, node_indices in dql_consumers.items():
        if len(node_indices) <= 1:
            continue
        init = init_map.get(init_name)
        if init is None:
            continue
        if _initializer_data_nbytes(init) > _MAX_DUP_BYTES:
            continue
        duplicated += len(node_indices) - 1
    return duplicated


def _fp32_consumer_initializer_names(model, elem_types: dict) -> set:
    """Initializers consumed by a standard ONNX op whose exporter-typed output is FLOAT.

    ``Cast`` is excluded because its output dtype does not follow its input.
    """
    init_names = {init.name for init in model.graph.initializer}
    names: set = set()
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type == "Cast":
            continue
        if any(elem_types.get(name) == 1 for name in node.output):  # FLOAT
            names.update(name for name in node.input if name in init_names)
    return names


def _initializer_dtype_fixup_required(
    model,
    dedup_dql_scales: bool,
    cast_fp32_weights_to_fp16: bool,
    preserve_fp32_patterns: "tuple[str, ...]",
    match_fp32_matmul_initializers: bool,
    match_fp32_elementwise_initializers: bool,
    match_fp32_consumer_initializers: bool = False,
) -> bool:
    if dedup_dql_scales and _count_shared_dql_scale_duplicates(model) > 0:
        return True

    plugin_fp32_init_names: set = set()
    for node in model.graph.node:
        if node.op_type == "update_ssm_state" and len(node.input) > 1:
            plugin_fp32_init_names.add(node.input[1])
        if node.op_type == "gated_delta_net" and len(node.input) > 5:
            plugin_fp32_init_names.add(node.input[5])
        if node.op_type in ("Nvfp4MoePlugin", "NvFP4MoEPluginGeforce"):
            for input_idx in (4, 7, 8, 9, 10):
                if len(node.input) > input_idx:
                    plugin_fp32_init_names.add(node.input[input_idx])
        if node.op_type == "Nvfp4A16BlackwellMoePlugin":
            for input_idx in (4, 7, 8):
                if len(node.input) > input_idx:
                    plugin_fp32_init_names.add(node.input[input_idx])
        if node.op_type == "Fp16MoePlugin" and len(node.input) > 4:
            plugin_fp32_init_names.add(node.input[4])
        if node.op_type == "W2A16MoePlugin" and len(node.input) > 5:
            plugin_fp32_init_names.add(node.input[5])
        if node.op_type == "AttentionPlugin":
            plugin_fp32_init_names.update(
                name for name in node.input
                if name.startswith(_ATTENTION_SINK_INIT_PREFIX))

    init_map = {init.name: init for init in model.graph.initializer}
    elem_types: dict[str, int] = {}
    if (match_fp32_matmul_initializers or match_fp32_elementwise_initializers
            or match_fp32_consumer_initializers):
        for value in (list(model.graph.input) + list(model.graph.value_info) +
                      list(model.graph.output)):
            tensor_type = value.type.tensor_type
            if tensor_type.HasField("elem_type"):
                elem_types[value.name] = tensor_type.elem_type
        for init in model.graph.initializer:
            elem_types[init.name] = init.data_type

    matmul_fp32_init_names: set = set()
    if match_fp32_matmul_initializers:
        for node in model.graph.node:
            if node.op_type != "MatMul" or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None:
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    matmul_fp32_init_names.add(init.name)

    _EW_OPS = frozenset({"Mul", "Add", "Sub", "Div"})
    elementwise_fp32_init_names: set = set()
    if match_fp32_elementwise_initializers:
        for node in model.graph.node:
            if node.op_type not in _EW_OPS or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None:
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    elementwise_fp32_init_names.add(init.name)

    consumer_fp32_init_names: set = set()
    if match_fp32_consumer_initializers:
        consumer_fp32_init_names = _fp32_consumer_initializer_names(
            model, elem_types)

    def _is_preserved_fp32(init_name: str) -> bool:
        return any(p in init_name for p in preserve_fp32_patterns)

    for init in model.graph.initializer:
        if init.name in plugin_fp32_init_names and init.data_type == 10:
            return True

        if cast_fp32_weights_to_fp16 and init.data_type == 1:
            dims = list(init.dims)
            if len(dims) == 0 or (len(dims) == 1 and dims[0] <= 1):
                continue
            if (init.name.endswith(".weight_scale")
                    or init.name.endswith(".input_scale")
                    or init.name.endswith(".pre_quant_scale")
                    or init.name.endswith("_scale")
                    or init.name.endswith("_scale_2")):
                continue
            if init.name in plugin_fp32_init_names:
                continue
            if _is_preserved_fp32(init.name):
                continue
            if init.name in matmul_fp32_init_names:
                continue
            if init.name in elementwise_fp32_init_names:
                continue
            if init.name in consumer_fp32_init_names:
                continue
            return True

    if match_fp32_matmul_initializers:
        for node in model.graph.node:
            if node.op_type != "MatMul" or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None or init.data_type != 10:  # FLOAT16
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    return True

    if match_fp32_elementwise_initializers:
        for node in model.graph.node:
            if node.op_type not in _EW_OPS or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None or init.data_type != 10:  # FLOAT16
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    return True

    return False


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

_OPSET_VERSION = 24


@contextlib.contextmanager
def _permissive_inline_opset():
    """Patch onnx-ir InlinePass to resolve opset-version conflicts by taking max.

    torch TORCHLIB functions are compiled at opset 18; our custom onnxscript
    translation functions use opset 21 (required for FP8 ``QuantizeLinear``
    with ``output_dtype``).  ``InlinePass._instantiate_call`` raises
    ``ValueError: Opset mismatch: 18 != 21`` when it encounters both in the
    same model.

    The standard ONNX domain is strictly backwards-compatible, so taking the
    higher version is correct: opset 21 is a superset of opset 18.
    """
    try:
        from onnx_ir.passes.common.inliner import InlinePass
    except ImportError:
        yield
        return

    _orig = InlinePass._instantiate_call

    def _patched(self, node, call_site_id):
        # Pre-merge opset_imports taking max to avoid ValueError in original.
        # Also align function.opset_imports so _orig's equality check passes.
        op_id = node.op_identifier()
        function = self._functions.get(op_id)
        if function is not None:
            for key, value in list(function.opset_imports.items()):
                merged = max(self._opset_imports.get(key, value), value)
                self._opset_imports[key] = merged
                function.opset_imports[key] = merged
        return _orig(self, node, call_site_id)

    InlinePass._instantiate_call = _patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        InlinePass._instantiate_call = _orig  # type: ignore[method-assign]


def _buffer_scalar(buffer) -> float:
    return float(buffer.item()) if buffer is not None else 1.0


def _attach_shared_kv_fp8_scales(model: "torch.nn.Module") -> None:
    attention_modules = [
        module for module in model.modules()
        if getattr(module, "enable_fp8_kv_cache", False)
        and hasattr(module, "layer_idx") and hasattr(module, "attention_type")
        and hasattr(module, "is_kv_shared")
    ]
    if not attention_modules:
        return
    donors_by_type = {}
    for module in sorted(attention_modules,
                         key=lambda attn: int(attn.layer_idx)):
        attention_type = str(module.attention_type)
        if not getattr(module, "is_kv_shared", False):
            donors_by_type[attention_type] = module
            continue

        donor = donors_by_type.get(attention_type)
        if donor is None:
            raise RuntimeError(
                "FP8 shared-KV attention is missing its donor layer: "
                f"layer={module.layer_idx}, type={attention_type}.")
        donor_scales = getattr(donor, "_qkv_scales_float", None)
        consumer_scales = getattr(module, "_qkv_scales_float", None)
        if donor_scales is None or len(donor_scales) != 3:
            raise RuntimeError(
                "FP8 shared-KV donor is missing Q/K/V export scales: "
                f"layer={donor.layer_idx}, type={attention_type}.")
        if consumer_scales is None or len(consumer_scales) != 3:
            raise RuntimeError(
                "FP8 shared-KV consumer is missing Q/K/V export scales: "
                f"layer={module.layer_idx}, type={attention_type}.")
        module._qkv_scales_float = [
            float(consumer_scales[0]),
            float(donor_scales[1]),
            float(donor_scales[2]),
        ]


def setup_fp8_qkv_scales_for_export(model: "torch.nn.Module") -> None:
    """Pre-cache FP8 Q/K/V scales as Python floats before torch.export tracing.

    During tracing, calling ``.item()`` on a tensor buffer creates a
    data-dependent symbolic expression that ``torch.export`` cannot guard on.
    By extracting the float values here (before the trace) and storing them
    as plain Python attributes on each attention module, they appear as
    compile-time constants during export.

    Triggers on either ``enable_fp8_kv_cache`` (LLM) or ``enable_fp8_mha``
    (ViT visual MHA) — both signal a calibrated checkpoint with
    ``k_proj.k_scale`` / ``v_proj.v_scale`` to surface.

    Stored attribute: ``module._qkv_scales_float = [q, k, v]``
      - q_scale : module-level ``q_scale`` buffer if present (visual MHA;
                  surfaced by ``_surface_visual_q_scales`` in the
                  quantization frontend), else ``q_proj.q_scale`` (LLM
                  convention), else 1.0
      - k_scale : ``k_proj.k_scale`` buffer value if present, else 1.0
      - v_scale : ``v_proj.v_scale`` buffer value if present, else K scale
                  when ``attention_k_eq_v`` writes V from K, else 1.0
    """
    for module in model.modules():
        if not (getattr(module, "enable_fp8_kv_cache", False)
                or getattr(module, "enable_fp8_mha", False)):
            continue
        if hasattr(module, "qkv_proj_fused"):
            scales = getattr(module, "_qkv_scales_float", None)
            if scales is None or len(scales) != 3:
                raise RuntimeError(
                    "Fused FP8-KV attention is missing its Q/K/V scales")
            module._qkv_scales_float = [float(scale) for scale in scales]
            continue
        q_buf = getattr(module, "q_scale", None)
        if q_buf is None:
            q_buf = getattr(getattr(module, "q_proj", None), "q_scale", None)
        k_buf = getattr(getattr(module, "k_proj", None), "k_scale", None)
        v_buf = getattr(getattr(module, "v_proj", None), "v_scale", None)
        if getattr(module, "attention_k_eq_v",
                   False) and (v_buf is None or v_buf.isnan().any()):
            v_buf = k_buf
        module._qkv_scales_float = [
            _buffer_scalar(q_buf),
            _buffer_scalar(k_buf),
            _buffer_scalar(v_buf),
        ]
    _attach_shared_kv_fp8_scales(model)


def _capture_qk_norm_gammas_for_export(model: "CausalLM") -> None:
    """Populate qk_norm gamma lists on every attention module before tracing.

    Runs in the shared export path so every export entrypoint captures the
    loaded gamma weights. Raises if a module carries qk_norm weights but no
    gamma values were captured — the export must never silently drop the
    fused norm.
    """
    for module in model.modules():
        if not hasattr(module, "_capture_qk_norm_gamma_lists"):
            continue
        module._capture_qk_norm_gamma_lists()
        has_norm = (getattr(module, "q_norm", None) is not None
                    or getattr(module, "k_norm", None) is not None)
        captured = bool(
            getattr(module, "_q_norm_gamma_list", None)
            or getattr(module, "_k_norm_gamma_list", None))
        if has_norm and not captured:
            raise RuntimeError(
                "qk_norm gamma capture failed for "
                f"{type(module).__name__}: the module has q_norm/k_norm "
                "weights but no gamma values were captured — the export "
                "would silently drop the fused qk_norm.")
        # Same contract for a sparse-attention indexer's own norms, which the
        # check above does not reach. Gated on the submodule actually being
        # there so models without an indexer are unaffected.
        indexer = getattr(module, "indexer", None)
        indexer_has_norm = (getattr(indexer, "q_layernorm", None) is not None
                            or getattr(indexer, "k_layernorm",
                                       None) is not None)
        indexer_captured = bool(
            getattr(module, "_indexer_q_norm_gamma_list", None)
            or getattr(module, "_indexer_k_norm_gamma_list", None))
        if indexer_has_norm and not indexer_captured:
            raise RuntimeError(
                "indexer qk_norm gamma capture failed for "
                f"{type(module).__name__}: the indexer has q_layernorm/"
                "k_layernorm weights but no gamma values were captured — the "
                "export would silently drop the indexer norm.")


def _capture_attention_sinks_for_export(model: "CausalLM") -> None:
    for module in model.modules():
        if not hasattr(module, "_capture_attention_sink_list"):
            continue
        module._capture_attention_sink_list()
        has_sink = getattr(module, "attention_sink_bias", None) is not None
        captured = bool(getattr(module, "_attention_sinks_list", None))
        if has_sink and not captured:
            raise RuntimeError(
                "attention sink capture failed for "
                f"{type(module).__name__}: sink weights exist but no values were captured"
            )


def _fix_initializer_dtypes(
    onnx_path: str,
    dedup_dql_scales: bool = False,
    cast_fp32_weights_to_fp16: bool = True,
    preserve_fp32_patterns: "tuple[str, ...]" = (),
    match_fp32_matmul_initializers: bool = False,
    match_fp32_elementwise_initializers: bool = False,
    match_fp32_consumer_initializers: bool = False,
) -> None:
    """Single-pass ONNX initializer fixup for TRT compatibility.

    Performs up to three corrections in one ONNX load+save:

    1. **Shared DQL scales** (when *dedup_dql_scales* is True): duplicate
       shared scalar DequantizeLinear initializers so each DQL node gets
       its own copy (see :func:`_dedup_shared_dql_scales`).

    2. **FP32 weights → FP16** (when *cast_fp32_weights_to_fp16* is True):
       The dynamo exporter may emit FP32 constants for FP16 model weights
       (e.g. tied lm_head in BF16 checkpoints).  TRT requires uniform dtype
       in MatMul inputs.  Scalars and quantization scale tensors are left as
       FP32.  Disable this for graphs that legitimately keep FP32 constants
       (e.g. ``weight.float()`` inside a LayerNorm whose body is FP32).

       Initializers whose name contains any substring in
       ``preserve_fp32_patterns`` are kept FP32.  This is how a model opts
       out of the downgrade for weights that must stay FP32.  Some
       ``torch.export`` initializers are anonymous, so selected models can
       additionally request MatMul initializer dtype matching when the other
       input is known to be FP32.

    3. **Plugin FP32 inputs**: ONNX constant folding may collapse plugin
       FP32 input expressions into initializers.  Any such initializer is
       kept (or restored to) FP32 when the consuming plugin requires FP32.

    4. **Element-wise FP32 input matching** (when
       *match_fp32_elementwise_initializers* is True): promote FP16
       initializers to FP32 when they feed a ``Mul`` / ``Add`` / ``Sub``
       / ``Div`` node whose other input is FP32.  This fixes the ONNX
       dynamo exporter folding float32 buffers (e.g. RoPE ``inv_freq``)
       into FP16 initializers — TRT rejects mixed-type element-wise ops.
       Used by the DFlash draft model export.

    5. **FP32 consumer matching** (when *match_fp32_consumer_initializers*
       is True): skip the FP32→FP16 downgrade for initializers consumed by
       any standard ONNX op with an FP32 output (see
       :func:`_fp32_consumer_initializer_names`), e.g. folded FP32 constants
       feeding ``Expand`` or ``LayerNormalization`` in an FP32 head.
    """
    import numpy as np

    _onnx = __import__("onnx")
    metadata_model = _onnx.load(onnx_path, load_external_data=False)
    if not _initializer_dtype_fixup_required(
            metadata_model,
            dedup_dql_scales=dedup_dql_scales,
            cast_fp32_weights_to_fp16=cast_fp32_weights_to_fp16,
            preserve_fp32_patterns=preserve_fp32_patterns,
            match_fp32_matmul_initializers=match_fp32_matmul_initializers,
            match_fp32_elementwise_initializers=
            match_fp32_elementwise_initializers,
            match_fp32_consumer_initializers=match_fp32_consumer_initializers,
    ):
        logger.info(
            "_fix_initializer_dtypes: no dtype fixup required; skipped "
            "external tensor load")
        return
    del metadata_model
    gc.collect()

    model = _onnx.load(onnx_path)

    # --- Dedup shared DQL scale initializers (NVFP4 dynamo fix) ---
    n_deduped = 0
    if dedup_dql_scales:
        n_deduped = _dedup_shared_dql_scales(model)

    # Collect plugin initializer names that must stay FP32.
    # - Mamba2 update_ssm_state: input[1] = ssm_A
    # - gated_delta_net: input[5] = A_log
    # - Nvfp4MoePlugin / NvFP4MoEPluginGeforce: inputs[4,7,8,9] are FP32 scale
    #   vectors; input[10] is the FP32 router correction bias. Both plugins
    #   share the same 11-input ONNX surface.
    # - Nvfp4A16BlackwellMoePlugin: inputs[4,7] are the FP32 per-expert global
    #   scales; input[8] is the FP32 router correction bias.
    plugin_fp32_init_names: set = set()
    for node in model.graph.node:
        if node.op_type == "update_ssm_state" and len(node.input) > 1:
            plugin_fp32_init_names.add(node.input[1])
        if node.op_type == "gated_delta_net" and len(node.input) > 5:
            plugin_fp32_init_names.add(node.input[5])
        if node.op_type in ("Nvfp4MoePlugin", "NvFP4MoEPluginGeforce"):
            for input_idx in (4, 7, 8, 9, 10):
                if len(node.input) > input_idx:
                    plugin_fp32_init_names.add(node.input[input_idx])
        if node.op_type == "Nvfp4A16MoePlugin" and len(node.input) > 8:
            plugin_fp32_init_names.add(node.input[8])
        if node.op_type == "Nvfp4A16BlackwellMoePlugin":
            for input_idx in (4, 7, 8):
                if len(node.input) > input_idx:
                    plugin_fp32_init_names.add(node.input[input_idx])
        if node.op_type == "Fp16MoePlugin" and len(node.input) > 4:
            plugin_fp32_init_names.add(node.input[4])
        if node.op_type == "AttentionPlugin":
            plugin_fp32_init_names.update(
                name for name in node.input
                if name.startswith(_ATTENTION_SINK_INIT_PREFIX))

    init_map = {init.name: init for init in model.graph.initializer}
    elem_types: dict[str, int] = {}
    if (match_fp32_matmul_initializers or match_fp32_elementwise_initializers
            or match_fp32_consumer_initializers):
        for value in (list(model.graph.input) + list(model.graph.value_info) +
                      list(model.graph.output)):
            tensor_type = value.type.tensor_type
            if tensor_type.HasField("elem_type"):
                elem_types[value.name] = tensor_type.elem_type
        for init in model.graph.initializer:
            elem_types[init.name] = init.data_type

    matmul_fp32_init_names: set = set()
    if match_fp32_matmul_initializers:
        for node in model.graph.node:
            if node.op_type != "MatMul" or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None:
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    matmul_fp32_init_names.add(init.name)

    # Pre-collect element-wise FP32 init names so the downgrade pass skips them.
    _EW_OPS = frozenset({"Mul", "Add", "Sub", "Div"})
    elementwise_fp32_init_names: set = set()
    if match_fp32_elementwise_initializers:
        for node in model.graph.node:
            if node.op_type not in _EW_OPS or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None:
                    continue
                if elem_types.get(node.input[other_idx]) == 1:  # FLOAT
                    elementwise_fp32_init_names.add(init.name)

    consumer_fp32_init_names: set = set()
    if match_fp32_consumer_initializers:
        consumer_fp32_init_names = _fp32_consumer_initializer_names(
            model, elem_types)

    def _is_preserved_fp32(init_name: str) -> bool:
        """Does ``init_name`` match any caller-supplied preserve pattern?"""
        return any(p in init_name for p in preserve_fp32_patterns)

    n_to_fp16 = 0
    n_to_fp32 = 0
    for init in model.graph.initializer:
        # --- Plugin-required FP32 input: ensure FP32 ---
        if init.name in plugin_fp32_init_names and init.data_type == 10:  # FP16
            dims = list(init.dims)
            data = np.frombuffer(init.raw_data, dtype=np.float16).reshape(dims)
            init.data_type = 1  # FLOAT (FP32)
            init.raw_data = data.astype(np.float32).tobytes()
            n_to_fp32 += 1
            logger.info(
                "_fix_initializer_dtypes: %s %s FP16→FP32 (plugin FP32 input)",
                init.name, dims)
            continue

        # --- FP32 weight → FP16 ---
        if not cast_fp32_weights_to_fp16:
            continue
        if init.data_type != 1:  # not FP32
            continue
        if init.name in plugin_fp32_init_names:  # already FP32, must stay
            continue
        if _is_preserved_fp32(init.name):  # caller opted this init out
            logger.info(
                "_fix_initializer_dtypes: %s %s kept FP32 (preserve pattern)",
                init.name, list(init.dims))
            continue
        if init.name in matmul_fp32_init_names:
            logger.info(
                "_fix_initializer_dtypes: %s %s kept FP32 (MatMul FP32 input)",
                init.name, list(init.dims))
            continue
        if init.name in elementwise_fp32_init_names:
            logger.info(
                "_fix_initializer_dtypes: %s %s kept FP32 (element-wise FP32 input)",
                init.name, list(init.dims))
            continue
        if init.name in consumer_fp32_init_names:
            logger.info(
                "_fix_initializer_dtypes: %s %s kept FP32 (FP32 consumer)",
                init.name, list(init.dims))
            continue
        dims = list(init.dims)
        if len(dims) == 0 or (len(dims) == 1 and dims[0] <= 1):
            continue  # keep scalars as FP32
        if (init.name.endswith(".weight_scale")
                or init.name.endswith(".input_scale")
                or init.name.endswith(".pre_quant_scale")
                or init.name.endswith("_scale")
                or init.name.endswith("_scale_2")):
            continue  # keep quantization scales as FP32
        data = np.frombuffer(init.raw_data, dtype=np.float32).reshape(dims)
        init.data_type = 10  # FLOAT16
        init.raw_data = data.astype(np.float16).tobytes()
        n_to_fp16 += 1
        logger.info("_fix_initializer_dtypes: %s %s FP32→FP16", init.name,
                    dims)

    if match_fp32_matmul_initializers:
        for init in model.graph.initializer:
            elem_types[init.name] = init.data_type

        for node in model.graph.node:
            if node.op_type != "MatMul" or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None or init.data_type != 10:  # FLOAT16
                    continue
                if elem_types.get(node.input[other_idx]) != 1:  # FLOAT
                    continue
                data = _onnx.numpy_helper.to_array(init).astype(np.float32)
                init.CopyFrom(
                    _onnx.numpy_helper.from_array(data, name=init.name))
                elem_types[init.name] = init.data_type
                n_to_fp32 += 1
                logger.info(
                    "_fix_initializer_dtypes: %s %s FP16→FP32 "
                    "(MatMul FP32 input match)", init.name, list(init.dims))

    if match_fp32_elementwise_initializers:
        # Refresh elem_types after prior passes may have changed dtypes.
        for init in model.graph.initializer:
            elem_types[init.name] = init.data_type

        _EW_OPS = frozenset({"Mul", "Add", "Sub", "Div"})
        for node in model.graph.node:
            if node.op_type not in _EW_OPS or len(node.input) < 2:
                continue
            for init_idx, other_idx in ((0, 1), (1, 0)):
                init = init_map.get(node.input[init_idx])
                if init is None or init.data_type != 10:  # FLOAT16
                    continue
                if elem_types.get(node.input[other_idx]) != 1:  # FLOAT
                    continue
                data = _onnx.numpy_helper.to_array(init).astype(np.float32)
                init.CopyFrom(
                    _onnx.numpy_helper.from_array(data, name=init.name))
                elem_types[init.name] = init.data_type
                n_to_fp32 += 1
                logger.info(
                    "_fix_initializer_dtypes: %s %s FP16→FP32 "
                    "(%s FP32 input match)", init.name, list(init.dims),
                    node.op_type)

    if n_to_fp16 == 0 and n_to_fp32 == 0 and n_deduped == 0:
        return

    # Update matching value_info entries
    vi_map = {vi.name: vi for vi in model.graph.value_info}
    for init in model.graph.initializer:
        if init.name in vi_map:
            vi_map[init.name].type.tensor_type.elem_type = init.data_type

    logger.info(
        "_fix_initializer_dtypes: %d→FP16, %d→FP32, %d DQL deduped, "
        "saving...", n_to_fp16, n_to_fp32, n_deduped)
    # Delete existing external data file before re-saving.  onnx.save_model
    # opens the file in r+b mode and appends new tensors at the end, so the
    # old data would remain as unreferenced garbage, doubling the file size.
    # Derive the data filename from onnx_path so per-rank TP exports
    # (model_world{N}_rank{R}.onnx) get distinct .data files instead of
    # all overwriting the same model.onnx.data.
    data_file = os.path.basename(onnx_path) + ".data"
    ext_path = os.path.join(os.path.dirname(onnx_path), data_file)
    if os.path.isfile(ext_path):
        old_size = os.path.getsize(ext_path)
        logger.info("Removing stale external data %s (%.2f GB) before re-save",
                    ext_path, old_size / 1e9)
        os.remove(ext_path)
    _onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        convert_attribute=True,
    )


def _export_model(
    model: "CausalLM",
    output_path: str,
    optimize: bool = True,
    externalize_weights=None,
) -> "list[dict[str, object]]":
    setup_fp8_qkv_scales_for_export(model)
    _capture_qk_norm_gammas_for_export(model)
    _capture_attention_sinks_for_export(model)
    spec = model.onnx_export_spec()

    translation_table = build_custom_translation_table()

    logger.info("Exporting ONNX to %s (opset %d, dynamo) ...", output_path,
                _OPSET_VERSION)
    with _permissive_inline_opset():
        prog = torch.onnx.export(
            spec.wrapped,
            spec.args,
            dynamo=True,
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_shapes=spec.dynamic_shapes,
            opset_version=_OPSET_VERSION,
            custom_translation_table=translation_table,
            external_data=True,
            optimize=optimize,
        )
    prog.save(output_path, external_data=True)
    with open(output_path, "rb+") as _f:
        os.fsync(_f.fileno())
    del prog
    gc.collect()
    nvfp4 = model.config.quant.uses_nvfp4_weights
    mxfp8 = model.config.quant.uses_mxfp8_weights
    if nvfp4:
        _fix_nvfp4_weight_dtype(output_path)
    if mxfp8:
        _strip_onnxscript_internal_attrs(output_path)
    # Models may opt specific initializer names out of the FP32→FP16
    # downgrade via a class attribute (see e.g. CodePredictorCausalLM).
    preserve_patterns = tuple(
        getattr(model, "preserve_fp32_initializer_patterns", ()))
    _fix_initializer_dtypes(output_path,
                            dedup_dql_scales=(nvfp4 or mxfp8),
                            preserve_fp32_patterns=preserve_patterns,
                            match_fp32_matmul_initializers=bool(
                                getattr(model,
                                        "match_fp32_matmul_initializers",
                                        False)),
                            match_fp32_elementwise_initializers=bool(
                                getattr(model,
                                        "match_fp32_elementwise_initializers",
                                        False)),
                            match_fp32_consumer_initializers=bool(
                                getattr(model,
                                        "match_fp32_consumer_initializers",
                                        False)))
    _strip_attention_plugin_optional_inputs(output_path)
    if nvfp4_a16_export_target_sm() in (80, 86, 87):
        _encode_fp8_kv_cache_as_int8(output_path)
    # Must run after every pass that re-saves with save_as_external_data
    # (which can re-materialize stray payloads on zero-volume tensors).
    _fix_zero_volume_initializers(output_path)
    external_weight_files = externalize_model_weights(
        output_path, model, externalize_weights=externalize_weights)
    logger.info("Export complete: %s", output_path)
    return external_weight_files
