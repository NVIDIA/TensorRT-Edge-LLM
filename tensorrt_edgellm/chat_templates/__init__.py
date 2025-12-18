# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Chat template processing and templates for TensorRT Edge-LLM.

This module provides:
- Chat template extraction from HuggingFace tokenizers
- Chat template validation
- Chat templates for models that require explicit templates
"""

import os
from typing import Optional

# Re-export main functions from chat_template.py
from .chat_template import process_chat_template, validate_chat_template

# Directory containing templates
_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "templates")


def get_template_path(model_identifier: str) -> Optional[str]:
    """
    Get the path to a chat template for a model identifier.
    
    Args:
        model_identifier: Model identifier (e.g., "phi4mm")
        
    Returns:
        Path to the template file, or None if not found
    """
    template_path = os.path.join(_TEMPLATES_DIR, f"{model_identifier}.json")
    if os.path.exists(template_path):
        return template_path
    return None


__all__ = [
    'process_chat_template',
    'validate_chat_template',
    'get_template_path',
]
