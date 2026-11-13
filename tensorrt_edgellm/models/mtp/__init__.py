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
"""Third-party MTP draft heads for dense base models.

Model-native MTP heads (Qwen3.5, Nemotron-H) live with their model family;
heads trained separately and merged into a dense checkpoint live here, one
module per validated base family.
"""
from .modeling_qwen3_mtp import Qwen3MtpDraftModel

__all__ = ["Qwen3MtpDraftModel"]
