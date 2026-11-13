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
"""Standalone model implementations for quantization.

Some model architectures used by TensorRT Edge-LLM (e.g. EAGLE3 draft
models for speculative decoding) are not available in HuggingFace
``transformers`` and cannot be loaded via ``AutoModel``.  This package
provides minimal PyTorch re-implementations of those models — just
enough to run ModelOpt quantization and export a unified checkpoint.
"""

from .dflash_draft import DFlashCalibDraftModel  # noqa: F401
from .dflash_draft import quantize_and_export_dflash_draft  # noqa: F401
from .eagle3_draft import Eagle3DraftModel  # noqa: F401
from .eagle3_draft import quantize_and_export_draft  # noqa: F401
from .eagle4_draft import Eagle4CalibDraftModel  # noqa: F401
from .eagle4_draft import quantize_and_export_eagle4_draft  # noqa: F401
from .mtp_draft import Qwen3_5MtpDraftModel  # noqa: F401
from .mtp_draft import quantize_mtp_from_base  # noqa: F401
