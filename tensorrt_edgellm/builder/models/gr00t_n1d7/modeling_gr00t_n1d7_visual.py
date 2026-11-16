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
"""GR00T N1.7 vision tower: the shared Qwen3-VL encoder, whose modules discover
the ``backbone.model.model.`` prefix by scanning checkpoint keys."""

from ..qwen3_vl.modeling_qwen3_vl_visual import Qwen3VLVisualEncoder


class Gr00tVisualEncoder(Qwen3VLVisualEncoder):
    """Qwen3-VL vision transformer over a GR00T checkpoint."""
