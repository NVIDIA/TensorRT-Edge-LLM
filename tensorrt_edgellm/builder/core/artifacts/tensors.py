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
"""Tensor file writers used by runtime artifacts."""

import json
from typing import Any, Dict, Optional

import numpy as np

_NP_TO_ST = {
    np.dtype(np.float16): "F16",
    np.dtype(np.float32): "F32",
    np.dtype(np.int32): "I32",
    np.dtype(np.int64): "I64",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
}

_WRITE_CHUNK_BYTES = 64 << 20


def save_safetensors(path: str,
                     tensors: Dict[str, np.ndarray],
                     dtype_overrides: Optional[Dict[str, str]] = None) -> None:
    """Minimal safetensors writer."""
    header: Dict[str, Any] = {}
    arrays = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = int(arr.nbytes)
        header[name] = {
            "dtype": (dtype_overrides or {}).get(name, _NP_TO_ST[arr.dtype]),
            "shape": list(int(d) for d in arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        arrays.append(arr)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        for arr in arrays:
            view = memoryview(arr).cast("B")
            for start in range(0, view.nbytes, _WRITE_CHUNK_BYTES):
                f.write(view[start:start + _WRITE_CHUNK_BYTES])
