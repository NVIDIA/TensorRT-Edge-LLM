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
Patches the Gemma 4 Unified vision embedder so image calibration can run.

The BF16-trained patch projection has a far larger dynamic range than
ordinary transformer weights. Loaded as FP16, ``patch_dense`` overflows and
the following LayerNorm turns that inf into an all-NaN visual embedding, so
ModelOpt discards every multimodal calibration batch for NaN amax and
calibration cannot run.

``build_gemma4_unified_visual`` already keeps the visual graph in FP32 for
inference; ``apply()`` mirrors that for calibration, casting the embedder
output back to the model dtype so only the overflow-prone projection runs
wider.
"""

import torch


def apply(model, model_type: str) -> bool:
    """Run the vision embedder in FP32. Returns whether it was applied.

    Restricted to ``gemma4_unified``: plain ``gemma4`` has no such overflow,
    and its embedder feeds a weightless (still FP16) RMSNorm into the
    projection, which an FP32 projection would reject on dtype.
    """
    if model_type != "gemma4_unified":
        return False
    embedder = getattr(getattr(model, "model", model), "embed_vision", None)
    if embedder is None:
        return False
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float32:
        return False

    embedder.to(torch.float32)

    def _cast_back(_module, _args, output):
        if isinstance(output, torch.Tensor):
            return output.to(model_dtype)
        return output

    embedder.register_forward_hook(_cast_back)
    print("[WAR] Running the Gemma4 Unified vision embedder in FP32 for "
          f"calibration (model dtype {model_dtype})")
    return True
