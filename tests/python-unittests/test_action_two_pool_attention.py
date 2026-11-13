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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native attention across a read-only prefix and independent FP16 action KV."""
import pytest
import test_action_attention_plugin_contract as contract
from test_action_attention_plugin_contract import (ATTENTION_SCALE, DEV,
                                                   HEAD_SIZE, KV_HEADS,
                                                   KV_HIDDEN, Q_HEADS,
                                                   Q_HIDDEN, QKV_HIDDEN,
                                                   SEQ_LEN, STEPS)
from test_plugin_base import (PluginRunner, _device_sm, pf_float32, pf_int32,
                              torch, trt)


@pytest.mark.skipif(_device_sm()
                    not in (80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121),
                    reason="Two-pool attention requires SM80 or newer")
@pytest.mark.parametrize("fp8", [False, True],
                         ids=["fp16-prefix", "fp8-prefix"])
@pytest.mark.parametrize("starts", [(127, 129), (4095, 4096), (0, 64)])
def test_separate_action_pool(starts, fp8):
    """Independent tables retain read-only prefixes and exclusive FP16 action values."""
    _run_two_pool_case(starts, fp8)


def _run_two_pool_case(starts, fp8, scrambled=True, capacity=None):
    # Given read-only, scrambled prefix pages (with poisoned unused slots) and
    # a separate FP16 scratch page per request,
    # When ten denoising steps overwrite action positions starting at zero,
    # Then one non-causal softmax matches prefix+action, action KV stays FP16,
    # and every byte of the prefix pool remains unchanged.
    batch = len(starts)
    capacity = capacity or max(128, ((max(starts) + 127) // 128) * 128)
    pages_per_prefix = capacity // 128
    prefix_n = batch * pages_per_prefix
    prefix_dtype = torch.float8_e4m3fn if fp8 else torch.float16
    scales = (0.03125, 0.015625)
    generator = torch.Generator(device=DEV).manual_seed(2651)
    dense = torch.full((batch, 2, KV_HEADS, capacity + SEQ_LEN, HEAD_SIZE),
                       float('nan'),
                       dtype=torch.float16,
                       device=DEV)
    contract._seed_prefix(dense, starts, generator)
    # Give prefix values a substantial contribution so a missing scale fails.
    dense *= 8
    prefix = torch.full((2, prefix_n, 128, KV_HEADS, HEAD_SIZE),
                        float('nan'),
                        dtype=torch.float16,
                        device=DEV).to(prefix_dtype)
    k_ids = torch.arange(prefix_n, dtype=torch.int32, device=DEV)
    if scrambled:
        k_ids = k_ids.flip(0)
    k_ids = k_ids.reshape(batch, -1)
    v_ids = (k_ids.roll(1, dims=1) if scrambled else k_ids) + prefix_n
    prefix_table = torch.stack((k_ids, v_ids), 1).contiguous()
    flat = prefix.view(2 * prefix_n, 128, KV_HEADS, HEAD_SIZE)
    for b, start in enumerate(starts):
        for page in range(pages_per_prefix):
            begin = page * 128
            count = max(0, min(128, start - begin))
            for kv in range(2):
                values = dense[b, kv, :, begin:begin + count].permute(1, 0, 2)
                encoded = (values.float() /
                           scales[kv]).to(prefix_dtype) if fp8 else values
                flat[prefix_table[b, kv, page], :count] = encoded
                if fp8:
                    dense[b, kv, :,
                          begin:begin + count] = (encoded.float() *
                                                  scales[kv]).half().permute(
                                                      1, 0, 2)
    before_bytes = prefix.view(torch.uint8).clone()
    action = torch.full((2, batch, 128, KV_HEADS, HEAD_SIZE),
                        float('nan'),
                        dtype=torch.float16,
                        device=DEV)
    # Scratch IDs are independent from the prefix's physical numbering.
    action_ids = torch.arange(batch, dtype=torch.int32, device=DEV).flip(0)
    action_table = torch.stack((action_ids, action_ids + batch),
                               1).unsqueeze(-1)
    cos, sin, combined = contract._make_rope(capacity + SEQ_LEN)
    positions = (torch.tensor(starts, device=DEV)[:, None] +
                 torch.arange(SEQ_LEN, device=DEV)).flatten()
    qkv = contract._finite_qkv(batch, 0, generator)
    tensors = {
        "qkv":
        qkv.reshape(batch * SEQ_LEN, QKV_HIDDEN),
        "action":
        action,
        "query_lengths":
        torch.full((batch, ), SEQ_LEN, dtype=torch.int32, device=DEV),
        "rope":
        combined[0].index_select(0, positions).contiguous(),
        "prefix_lengths":
        torch.tensor(starts, dtype=torch.int32, device=DEV),
        "action_table":
        action_table,
        "selector":
        torch.zeros(batch, dtype=torch.int32, device=DEV),
        "query_offsets":
        torch.arange(0, (batch + 1) * SEQ_LEN,
                     SEQ_LEN,
                     dtype=torch.int32,
                     device=DEV),
        "attention_lengths":
        torch.tensor([s + SEQ_LEN for s in starts],
                     dtype=torch.int32,
                     device=DEV),
        "phase":
        torch.zeros(6, dtype=torch.int32, device=DEV),
        "context_count":
        torch.zeros(batch, dtype=torch.int32, device=DEV),
        "prefix":
        prefix.view(torch.int8)
        if fp8 and _device_sm() in (80, 86, 87) else prefix,
        "prefix_table":
        prefix_table,
    }
    dtype_map = {
        torch.float16: trt.float16,
        torch.float32: trt.float32,
        torch.int32: trt.int32,
        torch.int8: trt.int8,
        torch.float8_e4m3fn: trt.fp8
    }
    specs = [(name, dtype_map[t.dtype], tuple(t.shape))
             for name, t in tensors.items()]
    fields = [
        pf_int32("num_q_heads", Q_HEADS),
        pf_int32("num_kv_heads", KV_HEADS),
        pf_int32("head_size", HEAD_SIZE),
        pf_int32("enable_context_mask_selector", 1),
        pf_int32("external_kv_prefix", 1),
        pf_int32("fp8_kv_prefix", int(fp8)),
        pf_float32("qkv_scales", [1., *scales]),
        pf_float32("attention_scale", ATTENTION_SCALE)
    ]
    runner = PluginRunner()
    runner.build(input_specs=specs,
                 output_names=["out", "action_out"],
                 plugin_name="AttentionPlugin",
                 plugin_version="1",
                 plugin_fields=fields,
                 profiles={})
    tensors["out"] = torch.empty((batch * SEQ_LEN, Q_HIDDEN),
                                 dtype=torch.float16,
                                 device=DEV)
    tensors["action_out"] = action
    for step in range(STEPS):
        qkv = contract._finite_qkv(batch, step, generator)
        qkv[:, :, Q_HIDDEN + KV_HIDDEN:] += torch.linspace(-6.,
                                                           6.,
                                                           SEQ_LEN,
                                                           device=DEV)[None, :,
                                                                       None]
        tensors["qkv"] = qkv.reshape(batch * SEQ_LEN, QKV_HIDDEN)
        expected, expected_kv = contract._dense_reference(
            qkv, dense, starts, cos, sin)
        runner.execute(tensors)
        actual = tensors["out"].reshape(batch, SEQ_LEN, Q_HIDDEN)
        assert torch.equal(prefix.view(torch.uint8), before_bytes)
        contract._assert_close(expected, actual)
        for b, start in enumerate(starts):
            for kv in range(2):
                page = action.view(2 * batch, 128, KV_HEADS,
                                   HEAD_SIZE)[action_table[b, kv, 0]]
                contract._assert_close(
                    expected_kv[b, kv, :,
                                start:start + SEQ_LEN].permute(1, 0, 2),
                    page[:SEQ_LEN])
                assert torch.isnan(page[SEQ_LEN:]).all()
            v = qkv[b, :,
                    Q_HIDDEN + KV_HIDDEN:].reshape(SEQ_LEN, KV_HEADS,
                                                   HEAD_SIZE)
            assert torch.equal(action[1, action_ids[b], :SEQ_LEN], v)
        if step == 0:
            causal, _ = contract._dense_reference(qkv,
                                                  dense,
                                                  starts,
                                                  cos,
                                                  sin,
                                                  causal=True)
            contract._assert_mutation_fails(causal, actual)
            wrong_scales = dense.clone()
            if fp8:
                for b, start in enumerate(starts):
                    for kv in range(2):
                        wrong_scales[b, kv, :, :start] /= scales[kv]
                wrong, _ = contract._dense_reference(qkv, wrong_scales, starts,
                                                     cos, sin)
                contract._assert_mutation_fails(wrong, actual)
