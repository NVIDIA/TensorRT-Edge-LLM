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
Tests for the shared_kv mode of AttentionPlugin.

Verifies that a shared-KV attention layer:
1. Applies RoPE to Q correctly
2. Does NOT modify the KV cache (read-only)
3. Produces correct attention output by reading the donor's KV cache

Usage:
    python3 -m pytest tests/defs/test_shared_kv.py -v

Requirements:
    - Custom plugin library (libNvInfer_edgellm_plugin.so)
    - torch (with CUDA), numpy, tensorrt
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

# Conditional imports
try:
    import tensorrt as trt
    import torch

    DEPENDENCIES_AVAILABLE = torch.cuda.is_available()
    IMPORT_ERROR = None if DEPENDENCIES_AVAILABLE else "No CUDA GPU available"
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

    class _Dummy:

        def __getattr__(self, name):
            return None

    trt = _Dummy()
    torch = _Dummy()

# Tokens per page of the paged-KV pool. Must match kTOKENS_PER_PAGE
# (cpp/common/pagedKvTypes.h).
PAGE_SIZE = 128


def _pool_geometry(params: "SharedKVTestParams"):
    """Paged-pool geometry for an identity page table.

    The logical per-slot capacity is padded up to whole pages; every batch
    slot owns a fixed page range (``mpps`` pages). ``num_pages`` is sized for
    the max batch so the pool never resizes.
    Returns (cap_padded, mpps, num_pages).
    """
    cap_padded = -(-params.kv_cache_capacity // PAGE_SIZE) * PAGE_SIZE
    mpps = cap_padded // PAGE_SIZE  # maxPagesPerSeq
    num_pages = params.max_batch_size * mpps
    return cap_padded, mpps, num_pages


@dataclass
class SharedKVTestParams:
    """Parameters for shared-KV attention test."""
    batch_size: int = 2
    seq_len: int = 4
    num_q_heads: int = 8
    num_kv_heads: int = 4  # GQA
    head_size: int = 64
    kv_cache_capacity: int = 32
    max_batch_size: int = 4
    max_seq_len: int = 8
    max_position_embeddings: int = 64


def _find_plugin_library() -> str:
    """Locate the plugin .so file."""
    candidates = [
        "build/libNvInfer_edgellm_plugin.so",
        os.path.join(os.path.dirname(__file__),
                     "../../build/libNvInfer_edgellm_plugin.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise RuntimeError(
        "Could not find libNvInfer_edgellm_plugin.so. Build the project first."
    )


def _apply_rope_numpy(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                      position_ids: np.ndarray) -> np.ndarray:
    """Apply non-interleaved RoPE to x [B, S, H, D].

    cos/sin: [max_pos, D/2], position_ids: [B, S].
    """
    B, S, H, D = x.shape
    half = cos.shape[1]  # D/2
    out = x.copy()
    for b in range(B):
        for s in range(S):
            pos = position_ids[b, s]
            c = cos[pos]
            si = sin[pos]
            for h in range(H):
                x_left = x[b, s, h, :half]
                x_right = x[b, s, h, half:half * 2]
                out[b, s, h, :half] = x_left * c - x_right * si
                out[b, s, h, half:half * 2] = x_right * c + x_left * si
    return out


def _numpy_attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
                     kv_seq_len: int, attention_scale: float) -> np.ndarray:
    """Compute scaled dot-product attention against KV cache.

    Args:
        q: [B, S_q, H_q, D]
        k_cache: [B, H_kv, capacity, D]
        v_cache: [B, H_kv, capacity, D]
        kv_seq_len: number of valid entries in cache
        attention_scale: absolute multiplier applied to QK scores
    Returns:
        output: [B, S_q, H_q, D]
    """
    B, S_q, H_q, D = q.shape
    H_kv = k_cache.shape[1]
    repeats = H_q // H_kv

    k = k_cache[:, :, :kv_seq_len, :]  # [B, H_kv, kv_len, D]
    v = v_cache[:, :, :kv_seq_len, :]

    # GQA: expand KV heads
    k = np.repeat(k, repeats, axis=1)  # [B, H_q, kv_len, D]
    v = np.repeat(v, repeats, axis=1)

    # q: [B, S_q, H_q, D] -> [B, H_q, S_q, D]
    q_t = q.transpose(0, 2, 1, 3)

    scores = np.matmul(q_t, k.transpose(
        0, 1, 3, 2)) * attention_scale  # [B, H_q, S_q, kv_len]

    # Causal mask
    for sq in range(S_q):
        q_pos = kv_seq_len - S_q + sq
        scores[:, :, sq, q_pos + 1:] = -1e9

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    out = np.matmul(attn_weights, v).transpose(0, 2, 1, 3)  # [B, S_q, H_q, D]
    return out


def _build_attention_engine(logger,
                            params: SharedKVTestParams,
                            attention_scale: Optional[float] = None):
    """Build a TRT engine with the AttentionPlugin.

    Shared-KV mode is detected at runtime by passing K/V with seq_len=0.
    The optimization profile allows K/V seq_len range [0, max_seq_len].
    """
    p = params
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03,
                               True)

    q_hidden = p.num_q_heads * p.head_size
    kv_hidden = p.num_kv_heads * p.head_size

    cap_padded, mpps, num_pages = _pool_geometry(p)

    q_input = network.add_input("q", trt.float16, (-1, -1, q_hidden))
    k_input = network.add_input("k", trt.float16, (-1, -1, kv_hidden))
    v_input = network.add_input("v", trt.float16, (-1, -1, kv_hidden))
    # Paged-KV pool binding: [2, numPages, PAGE_SIZE, Hkv, D].
    kv_cache_input = network.add_input(
        "kv_cache", trt.float16,
        (2, -1, PAGE_SIZE, p.num_kv_heads, p.head_size))
    context_lengths = network.add_input("context_lengths", trt.int32, (-1, ))
    rope_cos_sin = network.add_input(
        "rope_cos_sin", trt.float32,
        (1, p.max_position_embeddings, p.head_size))
    kv_cache_indices = network.add_input("kv_cache_indices", trt.int32, (-1, ))
    kv_page_table = network.add_input("kv_page_table", trt.int32,
                                      (-1, 2, mpps))

    # Plugin creation (V3 API)
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_creator("AttentionPlugin", "1", "")
    if plugin_creator is None:
        raise RuntimeError("AttentionPlugin not found in registry")

    plugin_fields = [
        trt.PluginField("num_q_heads", np.array([p.num_q_heads],
                                                dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("num_kv_heads",
                        np.array([p.num_kv_heads], dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([p.head_size], dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("enable_tree_attention", np.array([0], dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("enable_fp8_kv_cache", np.array([0], dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("sliding_window_size", np.array([-1], dtype=np.int32),
                        trt.PluginFieldType.INT32),
    ]
    if attention_scale is not None:
        plugin_fields.append(
            trt.PluginField("attention_scale",
                            np.array([attention_scale], dtype=np.float32),
                            trt.PluginFieldType.FLOAT32))

    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin = plugin_creator.create_plugin("attention", plugin_field_collection,
                                          trt.TensorRTPhase.BUILD)

    plugin_inputs = [
        q_input, k_input, v_input, kv_cache_input, context_lengths,
        rope_cos_sin, kv_cache_indices, kv_page_table
    ]
    plugin_layer = network.add_plugin_v3(plugin_inputs, [], plugin)

    plugin_layer.get_output(0).name = "attention_output"
    network.mark_output(plugin_layer.get_output(0))
    plugin_layer.get_output(1).name = "kv_cache_output"
    network.mark_output(plugin_layer.get_output(1))

    # Optimization profile — K/V seq_len min=0 to support shared-KV mode
    profile = builder.create_optimization_profile()
    profile.set_shape("q", (1, 1, q_hidden),
                      (p.batch_size, p.seq_len, q_hidden),
                      (p.max_batch_size, p.max_seq_len, q_hidden))
    profile.set_shape("k", (1, 0, kv_hidden),
                      (p.batch_size, p.seq_len, kv_hidden),
                      (p.max_batch_size, p.max_seq_len, kv_hidden))
    profile.set_shape("v", (1, 0, kv_hidden),
                      (p.batch_size, p.seq_len, kv_hidden),
                      (p.max_batch_size, p.max_seq_len, kv_hidden))
    # The pool never resizes: numPages is fixed per engine (min=opt=max).
    pool_shape = (2, num_pages, PAGE_SIZE, p.num_kv_heads, p.head_size)
    profile.set_shape("kv_cache", pool_shape, pool_shape, pool_shape)
    profile.set_shape("context_lengths", (1, ), (p.batch_size, ),
                      (p.max_batch_size, ))
    profile.set_shape("rope_cos_sin",
                      (1, p.max_position_embeddings, p.head_size),
                      (1, p.max_position_embeddings, p.head_size),
                      (1, p.max_position_embeddings, p.head_size))
    profile.set_shape("kv_cache_indices", (0, ), (p.batch_size, ),
                      (p.max_batch_size, ))
    profile.set_shape("kv_page_table", (1, 2, mpps), (p.batch_size, 2, mpps),
                      (p.max_batch_size, 2, mpps))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TRT engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    return engine, context


def _execute_plugin(context, stream_handle: int, tensors: dict, shapes: dict):
    """Set shapes, bind tensor addresses, execute."""
    for name, shape in shapes.items():
        if not context.set_input_shape(name, shape):
            raise RuntimeError(
                f"Failed to set input shape for '{name}' to {shape}")

    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Failed to set tensor address for '{name}'")

    if not context.execute_async_v3(stream_handle):
        raise RuntimeError("execute_async_v3 failed")


@pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"Required dependencies not available: {IMPORT_ERROR}")
class TestSharedKVAttention:
    """Test shared-KV mode of the AttentionPlugin."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load plugin library and set up parameters."""
        self.params = SharedKVTestParams()
        self.rng = np.random.default_rng(123)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.device = torch.device("cuda:0")

        plugin_path = _find_plugin_library()
        ctypes.CDLL(plugin_path)
        trt.init_libnvinfer_plugins(self.logger, "")

    def _to_gpu(self, arr: np.ndarray) -> torch.Tensor:
        """Move numpy array to GPU as a contiguous torch tensor."""
        return torch.from_numpy(arr).contiguous().to(self.device)

    def _run_donor_then_shared(self,
                               donor_seq_len: int,
                               shared_seq_len: int,
                               is_decode: bool = False,
                               attention_scale: Optional[float] = None):
        """Run donor layer (populates KV cache), then shared layer (reads it).

        Returns:
            (shared_attn_output_np, kv_cache_before_np, kv_cache_after_np,
             numpy_reference_output)
        """
        p = self.params
        expected_scale = (attention_scale if attention_scale is not None else
                          p.head_size**-0.5)
        B = p.batch_size
        D = p.head_size
        H_q = p.num_q_heads
        H_kv = p.num_kv_heads

        stream = torch.cuda.Stream(device=self.device)

        # --- RoPE cache ---
        rope_cos = self.rng.standard_normal(
            (p.max_position_embeddings, D // 2)).astype(np.float32)
        rope_sin = self.rng.standard_normal(
            (p.max_position_embeddings, D // 2)).astype(np.float32)
        rope_cos_sin_np = np.zeros((1, p.max_position_embeddings, D),
                                   dtype=np.float32)
        rope_cos_sin_np[0, :, :D // 2] = rope_cos
        rope_cos_sin_np[0, :, D // 2:] = rope_sin
        d_rope = self._to_gpu(rope_cos_sin_np)

        # --- Step 1: Donor layer fills KV cache (K/V have normal seq_len) ---
        donor_engine, donor_ctx = _build_attention_engine(
            self.logger, p, attention_scale)

        donor_q_np = self.rng.standard_normal(
            (B, donor_seq_len, H_q * D)).astype(np.float16)
        donor_k_np = self.rng.standard_normal(
            (B, donor_seq_len, H_kv * D)).astype(np.float16)
        donor_v_np = self.rng.standard_normal(
            (B, donor_seq_len, H_kv * D)).astype(np.float16)

        d_q = self._to_gpu(donor_q_np)
        d_k = self._to_gpu(donor_k_np)
        d_v = self._to_gpu(donor_v_np)
        # Paged-KV pool [2, numPages, PAGE_SIZE, Hkv, D]; the donor plugin call
        # fills it in place under an identity page table.
        cap_padded, mpps, num_pages = _pool_geometry(p)
        cap = p.kv_cache_capacity
        d_kv_cache = torch.zeros(2,
                                 num_pages,
                                 PAGE_SIZE,
                                 H_kv,
                                 D,
                                 dtype=torch.float16,
                                 device=self.device)
        # Identity page table [B, 2, mpps]: K row b -> pages [b*mpps, (b+1)*mpps),
        # V row = K row + num_pages (the pool's V half).
        k_page_ids = torch.arange(B * mpps,
                                  dtype=torch.int32,
                                  device=self.device).reshape(B, mpps)
        d_page_table = torch.stack((k_page_ids, k_page_ids + num_pages), dim=1)
        d_ctx_len = self._to_gpu(np.full(B, donor_seq_len, dtype=np.int32))
        # Initial prefill: kv_cache_indices has shape [0] (plugin sentinel).
        # Allocate 1 element so TRT gets a non-null address; shape (0,) is set separately.
        d_cache_idx = torch.zeros(1, dtype=torch.int32, device=self.device)
        d_attn_out = torch.zeros(B,
                                 donor_seq_len,
                                 H_q,
                                 D,
                                 dtype=torch.float16,
                                 device=self.device)

        tensors_donor = {
            "q": d_q,
            "k": d_k,
            "v": d_v,
            "kv_cache": d_kv_cache,
            "context_lengths": d_ctx_len,
            "rope_cos_sin": d_rope,
            "kv_cache_indices": d_cache_idx,
            "kv_page_table": d_page_table,
            "attention_output": d_attn_out,
            "kv_cache_output": d_kv_cache,  # in-place
        }
        shapes_donor = {
            "q": (B, donor_seq_len, H_q * D),
            "k": (B, donor_seq_len, H_kv * D),
            "v": (B, donor_seq_len, H_kv * D),
            "kv_cache": (2, num_pages, PAGE_SIZE, H_kv, D),
            "context_lengths": (B, ),
            "rope_cos_sin": (1, p.max_position_embeddings, D),
            "kv_cache_indices": (0, ),
            "kv_page_table": (B, 2, mpps),
        }

        with torch.cuda.stream(stream):
            _execute_plugin(donor_ctx, stream.cuda_stream, tensors_donor,
                            shapes_donor)
        stream.synchronize()

        # Snapshot donor's populated KV cache, converting the pool back to the
        # logical per-slot view [B, 2, Hkv, cap, D] for the numpy reference.
        # Under the identity page table the K half [numPages, PAGE_SIZE, Hkv, D]
        # is slot-major/token-major, so viewing it as [max_batch, cap_padded,
        # Hkv, D] gives slot b's token t at [b, t] (NHD); permute to HND.
        pool_k = d_kv_cache[0].view(p.max_batch_size, cap_padded, H_kv, D)
        pool_v = d_kv_cache[1].view(p.max_batch_size, cap_padded, H_kv, D)
        k_logical = pool_k[:B, :cap].permute(0, 2, 1, 3).contiguous()
        v_logical = pool_v[:B, :cap].permute(0, 2, 1, 3).contiguous()
        kv_cache_after_donor = torch.stack((k_logical, v_logical),
                                           dim=1).cpu().numpy().copy()

        # --- Step 2: Shared layer reads donor's cache (K/V have seq_len=0) ---
        shared_engine, shared_ctx = _build_attention_engine(
            self.logger, p, attention_scale)

        shared_q_np = self.rng.standard_normal(
            (B, shared_seq_len, H_q * D)).astype(np.float16)
        d_q_shared = self._to_gpu(shared_q_np)
        # Empty K/V tensors signal shared-KV mode (seq_len=0).
        # Allocate 1 element so TRT gets a non-null address; shape (B,0,kv_hidden) is set separately.
        d_k_shared = torch.zeros(1, dtype=torch.float16, device=self.device)
        d_v_shared = torch.zeros(1, dtype=torch.float16, device=self.device)

        total_ctx = donor_seq_len + shared_seq_len
        # In prefill (FMHA path): context_lengths feeds cuQSeqLens which must equal the
        # actual Q token count (shared_seq_len). In decode (XQA path): context_lengths is
        # the total KV length to attend to.
        ctx_len_val = total_ctx if is_decode else shared_seq_len
        d_ctx_len_shared = self._to_gpu(np.full(B, ctx_len_val,
                                                dtype=np.int32))

        if is_decode:
            cache_idx_np = np.full(B, total_ctx, dtype=np.int32)
        else:
            cache_idx_np = np.full(B, donor_seq_len, dtype=np.int32)
        d_cache_idx_shared = self._to_gpu(cache_idx_np)

        d_attn_out_shared = torch.zeros(B,
                                        shared_seq_len,
                                        H_q,
                                        D,
                                        dtype=torch.float16,
                                        device=self.device)

        # Record KV cache state before shared layer
        kv_cache_before_shared = d_kv_cache.cpu().numpy().copy()

        tensors_shared = {
            "q": d_q_shared,
            "k": d_k_shared,
            "v": d_v_shared,
            "kv_cache": d_kv_cache,  # same buffer as donor wrote to
            "context_lengths": d_ctx_len_shared,
            "rope_cos_sin": d_rope,
            "kv_cache_indices": d_cache_idx_shared,
            "kv_page_table": d_page_table,
            "attention_output": d_attn_out_shared,
            "kv_cache_output": d_kv_cache,  # in-place (should NOT be modified)
        }
        shapes_shared = {
            "q": (B, shared_seq_len, H_q * D),
            "k": (B, 0, H_kv * D),
            "v": (B, 0, H_kv * D),
            "kv_cache": (2, num_pages, PAGE_SIZE, H_kv, D),
            "context_lengths": (B, ),
            "rope_cos_sin": (1, p.max_position_embeddings, D),
            "kv_cache_indices": (B, ),
            "kv_page_table": (B, 2, mpps),
        }

        with torch.cuda.stream(stream):
            _execute_plugin(shared_ctx, stream.cuda_stream, tensors_shared,
                            shapes_shared)
        stream.synchronize()

        # Read outputs
        shared_attn_out_np = d_attn_out_shared.cpu().numpy().astype(np.float32)
        kv_cache_after_shared = d_kv_cache.cpu().numpy()

        # --- Numpy reference ---
        q_reshaped = shared_q_np.astype(np.float32).reshape(
            B, shared_seq_len, H_q, D)
        position_ids = np.arange(donor_seq_len, total_ctx,
                                 dtype=np.int32)[None, :].repeat(B, axis=0)
        q_roped = _apply_rope_numpy(q_reshaped, rope_cos, rope_sin,
                                    position_ids)

        k_cache = kv_cache_after_donor[:, 0, :, :, :].astype(np.float32)
        v_cache = kv_cache_after_donor[:, 1, :, :, :].astype(np.float32)
        np_ref = _numpy_attention(q_roped, k_cache, v_cache, total_ctx,
                                  expected_scale)

        return (shared_attn_out_np, kv_cache_before_shared,
                kv_cache_after_shared, np_ref)

    def test_shared_kv_engine_builds(self):
        """Engine builds successfully (shared-KV detected via empty K/V at runtime)."""
        engine, ctx = _build_attention_engine(self.logger, self.params)
        assert engine is not None
        assert ctx is not None
        print("PASS: engine built (shared-KV via empty K/V)")

    def test_shared_kv_cache_not_modified(self):
        """Shared-KV layer must NOT write to the KV cache."""
        _, kv_before, kv_after, _ = self._run_donor_then_shared(
            donor_seq_len=4, shared_seq_len=4)

        np.testing.assert_array_equal(
            kv_before,
            kv_after,
            err_msg=
            "Shared-KV plugin modified the KV cache (expected read-only)")
        print("PASS: KV cache unchanged after shared-KV layer")

    @pytest.mark.parametrize("attention_scale", [None, 1.0, 0.37],
                             ids=["legacy-default", "identity", "custom"])
    def test_shared_kv_attention_correctness(self, attention_scale):
        """Shared-KV layer output matches numpy reference (prefill)."""
        shared_out, _, _, np_ref = self._run_donor_then_shared(
            donor_seq_len=4, shared_seq_len=4, attention_scale=attention_scale)

        atol, rtol = 2e-2, 2e-2
        max_abs = np.abs(shared_out - np_ref).max()
        print(f"Shared-KV prefill: max_abs_diff={max_abs:.6f}")

        assert np.allclose(shared_out, np_ref, atol=atol, rtol=rtol), (
            f"Output mismatch: max_abs={max_abs:.6f} (atol={atol})")
        print("PASS: Shared-KV prefill output matches reference")

    def test_shared_kv_decode_cache_not_modified(self):
        """Shared-KV decode must NOT write to the KV cache."""
        _, kv_before, kv_after, _ = self._run_donor_then_shared(
            donor_seq_len=4, shared_seq_len=1, is_decode=True)

        np.testing.assert_array_equal(
            kv_before,
            kv_after,
            err_msg="Shared-KV decode modified the KV cache")
        print("PASS: KV cache unchanged after shared-KV decode")

    @pytest.mark.parametrize("attention_scale", [None, 1.0, 0.37],
                             ids=["legacy-default", "identity", "custom"])
    def test_shared_kv_decode_correctness(self, attention_scale):
        """Shared-KV decode output matches numpy reference."""
        shared_out, _, _, np_ref = self._run_donor_then_shared(
            donor_seq_len=4,
            shared_seq_len=1,
            is_decode=True,
            attention_scale=attention_scale)

        atol, rtol = 2e-2, 2e-2
        max_abs = np.abs(shared_out - np_ref).max()
        print(f"Shared-KV decode: max_abs_diff={max_abs:.6f}")

        assert np.allclose(shared_out, np_ref, atol=atol, rtol=rtol), (
            f"Decode output mismatch: max_abs={max_abs:.6f} (atol={atol})")
        print("PASS: Shared-KV decode output matches reference")
