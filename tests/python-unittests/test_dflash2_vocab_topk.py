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
"""TensorRT execution test for the DFlash2 draft's two-stage vocabulary TopK."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_plugin_base import DEPENDENCIES_AVAILABLE, IMPORT_ERROR

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

    from tensorrt_edgellm.builder.models.dflash2 import modeling_dflash2_draft
    from tensorrt_edgellm.builder.ops import functional as F
    from tensorrt_edgellm.builder.ops.backend import Net
    from tensorrt_edgellm.builder.ops.module import BuildContext, BuildOptions
    from tensorrt_edgellm.builder.ops.scope import build_scope

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"needs TensorRT + CUDA: {IMPORT_ERROR}")

TOP_K, ROWS = 16, 7
# 10 x 2048 (power-of-two chunks) and 9 x 2296 (the Muse-Glimmer vocabulary's chunk width).
VOCABS = (2048 * 10, 2296 * 9)


def _run_topk(logits: np.ndarray, two_stage: bool):
    """Build [batch, ROWS, vocab] -> TopK as the draft does and run it."""
    vocab = logits.shape[-1]
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    net = Net(builder, network)
    context = BuildContext(net=net,
                           cfg=SimpleNamespace(),
                           weights=SimpleNamespace(),
                           options=BuildOptions(),
                           bundle=SimpleNamespace(),
                           args=SimpleNamespace())
    source = network.add_input("logits", trt.float32, (-1, -1, vocab))
    with build_scope(context):
        x = F.tensor(source)
        values, ids = (modeling_dflash2_draft._vocab_topk(x, TOP_K)
                       if two_stage else F.topk(x, TOP_K, 2))
    for name, out in (("values", values), ("ids", ids)):
        out = F.unwrap(out)
        out.name = name
        network.mark_output(out)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("logits", (1, 1, vocab), (1, ROWS, vocab),
                      (2, ROWS, vocab))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    execution = engine.create_execution_context()
    source_gpu = torch.from_numpy(logits).cuda()
    execution.set_input_shape("logits", tuple(logits.shape))
    out_values = torch.empty((*logits.shape[:2], TOP_K),
                             dtype=torch.float32,
                             device="cuda")
    out_ids = torch.empty((*logits.shape[:2], TOP_K),
                          dtype=torch.int32,
                          device="cuda")
    for name, buf in (("logits", source_gpu), ("values", out_values),
                      ("ids", out_ids)):
        execution.set_tensor_address(name, buf.data_ptr())
    stream = torch.cuda.current_stream()
    assert execution.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out_values.cpu().numpy(), out_ids.cpu().numpy()


@pytest.mark.parametrize("vocab", VOCABS)
def test_two_stage_vocab_topk_matches_one_topk_on_distinct_logits(vocab):
    """Given distinct logits over a vocabulary split into 2048- or 2296-wide
    chunks, when the draft's two-stage TopK runs in TensorRT, then it returns
    the same top-16 values and ids, in the same order, as one TensorRT TopK and
    as a NumPy sort."""
    rng = np.random.default_rng(1290)
    logits = np.stack([
        rng.permutation(vocab).reshape(1, vocab) / 8.0 for _ in range(2 * ROWS)
    ]).reshape(2, ROWS, vocab).astype(np.float32)

    values, ids = _run_topk(logits, two_stage=True)
    single_values, single_ids = _run_topk(logits, two_stage=False)

    reference_ids = np.argsort(-logits, axis=-1)[..., :TOP_K]
    np.testing.assert_array_equal(ids, reference_ids)
    np.testing.assert_array_equal(
        values, np.take_along_axis(logits, reference_ids, axis=-1))
    np.testing.assert_array_equal(ids, single_ids)
    np.testing.assert_array_equal(values, single_values)
