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
    inputs_embeds           [batch, seq_len, hidden_size]            float16
    past_key_values_0..N    [batch, 2, num_kv_heads, past, head_dim] float16
    rope_rotary_cos_sin     [batch, max_pos, rotary_dim]  float32
    context_lengths         [batch]                       int32
    kvcache_start_index     [batch]                       int32
    kv_page_table           [batch, 2, max_pages_per_seq] int32
    last_token_ids          [batch, 1]                    int64

Outputs:
    logits                  [batch, seq_len, vocab_size]             float32
    present_key_values_0..N [batch, 2, num_kv_heads, past+seq_len, head_dim] float16

Additional I/O for hybrid (Mamba) models
-----------------------------------------
Extra inputs:
    conv_state_0..M   [batch, conv_dim, conv_kernel-1]        float16
    ssm_state_0..M    [batch, num_heads, head_dim, ssm_state] float16

Extra outputs:
    present_conv_0..M   updated conv states
    present_ssm_0..M    updated ssm states
"""

import contextlib
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
from .dynamo_translations import build_custom_translation_table

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
                       per-row block scales.
        reduced_vocab_dir: Directory containing ``vocab_map.safetensors``
                           when reduced vocabulary is enabled.
        externalize_weights: Iterable of weight kinds to expose as fixed-shape
                             ONNX inputs and save to safetensors external
                             weight files.
                             Supported kinds: ``int4_ffn``, ``int4_moe``,
                             ``nvfp4_moe``, ``lm_head``, and ``all``.
        config_filename: Filename for the runtime config beside the ONNX.
                         Use ``"config.json"`` for single-device exports
                         or ``"config_tp{N}_rank{R}.json"`` for per-rank
                         TP exports so each rank is self-describing.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

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
                            config_filename=config_filename)
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

    This pass finds all int8 initialisers whose name ends with ``.weight``
    (NVFP4 linear weights) and rewrites them:
      elem_type  INT8  -> FLOAT4E2M1 (23)
      dims       [N, M] -> [N, M*2]   (double the last dim -- nibble unpacking)
    """
    _FLOAT4E2M1 = 23  # ONNX TensorProto.FLOAT4E2M1
    _INT8 = 3  # ONNX TensorProto.INT8
    _FLOAT = 1  # ONNX TensorProto.FLOAT

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
    for init in model.graph.initializer:
        # Only NVFP4 tensors whose name ends with ".weight" (not scale / qweight).
        if (not init.name.endswith(".weight") or "scale" in init.name
                or "qweight" in init.name):
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

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        weight_dims = fp4_weight_dims.get(node.input[0])
        if weight_dims is None:
            continue
        for output_name in node.output:
            fixed_value_infos += int(
                _set_value_info(output_name, _FLOAT, weight_dims))

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


def _strip_attention_plugin_optional_inputs(onnx_path: str) -> None:
    """Strip trailing empty optional inputs from AttentionPlugin ONNX nodes.

    ``torch.export`` emits the two optional inputs (``attention_mask``,
    ``attention_pos_id``) on every node, as empty strings when unused.  The
    TRT AttentionPlugin C++ requires exactly ``kNUM_REQUIRED_INPUTS=8``
    inputs (``[..., kvcache_start_index, kv_page_table]``; optionals start
    at index 8) for vanilla mode and raises ``(input) != nullptr`` when it
    encounters the extra null entries via
    ``INetworkDefinition::addPluginV2``.

    This pass trims each ``AttentionPlugin`` node to its expected input
    count: 8 for vanilla nodes, 9 for vision-block-attention nodes (the
    real ``attention_mask`` input carrying block IDs is kept, the empty
    ``attention_pos_id`` placeholder is dropped), and 10 for tree-attention
    nodes (both optional inputs are real and kept).
    """
    _REQUIRED = 8
    model = onnx.load(onnx_path, load_external_data=False)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "AttentionPlugin":
            continue
        tree_attn = next(
            (a.i for a in node.attribute if a.name == "enable_tree_attention"),
            0,
        )
        vision_block_attn = next(
            (a.i for a in node.attribute
             if a.name == "enable_vision_block_attention"),
            0,
        )
        optional_count = 2 if tree_attn else (1 if vision_block_attn else 0)
        keep = _REQUIRED + optional_count
        trailing = list(node.input)[keep:]
        if not trailing or any(i != "" for i in trailing):
            continue
        # Keep the real vision-block-ID input while dropping its unused
        # attention_pos_id placeholder.  Vanilla nodes retain only required
        # inputs; tree nodes retain both optional inputs.
        del node.input[keep:]
        changed += len(trailing)

    if not changed:
        return
    logger.info(
        "TRT fix: stripped %d empty optional input(s) from AttentionPlugin nodes",
        changed,
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
    by DequantizeLinear nodes across many layers.  TRT's Myelin compiler
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
        nbytes = len(orig.raw_data) if orig.raw_data else 0
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


def _setup_fp8kv_scales_for_export(model: "CausalLM") -> None:
    """Pre-cache FP8 KV scales as Python floats before torch.export tracing.

    During tracing, calling ``.item()`` on a tensor buffer creates a
    data-dependent symbolic expression that ``torch.export`` cannot guard on.
    By extracting the float values here (before the trace) and storing them
    as plain Python attributes on each attention module, they appear as
    compile-time constants during export.

    Stored attribute: ``module._qkv_scales_float = [q, k, v]``
      - q_scale : ``q_proj.q_scale`` buffer value if present, else 1.0
      - k_scale : ``k_proj.k_scale`` buffer value if present, else 1.0
      - v_scale : ``v_proj.v_scale`` buffer value if present, else 1.0
    """
    for module in model.modules():
        if not getattr(module, "enable_fp8_kv_cache", False):
            continue
        q_buf = getattr(getattr(module, "q_proj", None), "q_scale", None)
        k_buf = getattr(getattr(module, "k_proj", None), "k_scale", None)
        v_buf = getattr(getattr(module, "v_proj", None), "v_scale", None)
        if v_buf is None and getattr(module, "attention_k_eq_v", False):
            v_buf = k_buf
        module._qkv_scales_float = [
            float(q_buf.item()) if q_buf is not None else 1.0,
            float(k_buf.item()) if k_buf is not None else 1.0,
            float(v_buf.item()) if v_buf is not None else 1.0,
        ]


def _fix_initializer_dtypes(
    onnx_path: str,
    dedup_dql_scales: bool = False,
    cast_fp32_weights_to_fp16: bool = True,
    preserve_fp32_patterns: "tuple[str, ...]" = (),
    match_fp32_matmul_initializers: bool = False,
    match_fp32_elementwise_initializers: bool = False,
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
    """
    import numpy as np

    _onnx = __import__("onnx")
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

    init_map = {init.name: init for init in model.graph.initializer}
    elem_types: dict[str, int] = {}
    if match_fp32_matmul_initializers or match_fp32_elementwise_initializers:
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
    # (model_tp{N}_rank{R}.onnx) get distinct .data files instead of
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
    _setup_fp8kv_scales_for_export(model)
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
    with open(output_path, "rb") as _f:
        os.fsync(_f.fileno())
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
                                        False)))
    _strip_attention_plugin_optional_inputs(output_path)
    external_weight_files = externalize_model_weights(
        output_path, model, externalize_weights=externalize_weights)
    logger.info("Export complete: %s", output_path)
    return external_weight_files
