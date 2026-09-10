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

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 62), (3, 7)])
def test_qkv_concat_preserves_every_token_row(batch_size, seq_len):
    widths = (64, 16, 16)
    input_specs = [(name, trt.float16, (-1, -1, width))
                   for name, width in zip(("q", "k", "v"), widths)]
    profiles = {
        name: ((1, 1, width), (2, 8, width), (3, 64, width))
        for name, width in zip(("q", "k", "v"), widths)
    }

    runner = PluginRunner()
    runner.build(input_specs=input_specs,
                 output_names=["qkv"],
                 plugin_name="QkvConcatPlugin",
                 plugin_version="1",
                 plugin_fields=[],
                 profiles=profiles)

    generator = torch.Generator().manual_seed(1701 + batch_size + seq_len)
    inputs = {
        name:
        torch.randn((batch_size, seq_len, width),
                    generator=generator,
                    dtype=torch.float16).to("cuda")
        for name, width in zip(("q", "k", "v"), widths)
    }
    output = torch.empty((batch_size, seq_len, sum(widths)),
                         dtype=torch.float16,
                         device="cuda")
    runner.execute({**inputs, "qkv": output})

    torch.testing.assert_close(output,
                               torch.cat(tuple(inputs.values()), dim=-1),
                               rtol=0,
                               atol=0)


def test_qkv_concat_rejects_mismatched_leading_shape_profiles():
    widths = (64, 16, 16)
    input_specs = [(name, trt.float16, (-1, -1, width))
                   for name, width in zip(("q", "k", "v"), widths)]
    profiles = {
        "q": ((1, 1, 64), (1, 8, 64), (2, 64, 64)),
        "k": ((1, 1, 16), (2, 8, 16), (2, 64, 16)),
        "v": ((1, 1, 16), (1, 8, 16), (2, 64, 16)),
    }

    with pytest.raises(PluginUnsupportedError):
        PluginRunner().build(input_specs=input_specs,
                             output_names=["qkv"],
                             plugin_name="QkvConcatPlugin",
                             plugin_version="1",
                             plugin_fields=[],
                             profiles=profiles,
                             expect_unsupported=True)
