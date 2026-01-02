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
Quantization utilities for TensorRT Edge-LLM.

This module provides core quantization functionality using NVIDIA ModelOpt.
"""

from typing import Any, Dict

import modelopt.torch.quantization as mtq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def enable_huggingface_checkpointing_patch() -> None:
    from modelopt.torch.opt.plugins.huggingface import (
        _LIBRARY_CLASSES_FOR_PATCHING, _PATCHED_CLASSES,
        patch_pretrained_methods)
    """Enables automatic save/restore of ModelOpt state with HuggingFace checkpointing APIs.
    This is adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.37.0/modelopt/torch/opt/plugins/huggingface.py#L127
    Edge-LLM finds that _from_config() should not be patched.

    """
    for name, (classes, methods_list) in _LIBRARY_CLASSES_FOR_PATCHING.items():
        for cls, patch_methods in zip(classes, methods_list):
            if cls in _PATCHED_CLASSES:
                continue
            patch_methods = [
                method for method in patch_methods
                if method[0] != "_from_config"
            ]  # Edge-LLM finds that _from_config() should not be patched.
            patch_pretrained_methods(cls, patch_methods)
            _PATCHED_CLASSES.add(cls)
        print(f"ModelOpt save/restore enabled for `{name}` library.")


def quantize_model(
    model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        model: PyTorch model to quantize
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        
    Returns:
        Quantized PyTorch model
    """

    # Define calibration loop
    def calibrate_loop(model: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:
            model: Model to calibrate
        """
        # Create progress bar for calibration
        print(f"Calibrating model on {len(calib_dataloader)} samples...")
        pbar = tqdm(calib_dataloader, desc="Calibrating", unit="num_samples")

        # Add extra necessary kwargs for Phi-4-Multimodal
        kwargs = {}
        if hasattr(model, "config") and "phi4mm" in getattr(
                model.config, "model_type", "").lower():
            # Have already merged the vision LoRA, so set input_mode=0 (LANGUAGE) during quantization
            kwargs["input_mode"] = 0
            # Work around a transformers version mismatch between Phi-4MM and Edge-LLM
            kwargs["use_cache"] = False

        for data in pbar:
            if isinstance(data, dict):
                data = {k: v.to(model.device) for k, v in data.items()}
                model(**data, **kwargs)
            else:
                data = data.to(model.device)
                model(data, **kwargs)

    # Get quantization config and perform quantization
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(model)
    return model


def quantize_draft_model(
    base_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        base_model: Base model which is used to generate inputs for the draft model.
        draft_model: The draft model to quantize
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        
    Returns:
        Quantized PyTorch model
    """

    # Define calibration loop
    def calibrate_loop(draft_model: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:
            draft_model: Model to calibrate
        """
        # Create progress bar for calibration
        print(f"Calibrating model on {len(calib_dataloader)} samples...")
        pbar = tqdm(calib_dataloader, desc="Calibrating", unit="num_samples")
        assert base_model.device == draft_model.device, "Base model and draft model must be on the same device"

        for data in pbar:
            if isinstance(data, dict):
                data = {k: v.to(draft_model.device) for k, v in data.items()}
                base_model(**data)
            else:
                data = data.to(base_model.device)
                outputs = base_model(data, output_hidden_states=True)
            all_hidden_states = outputs['hidden_states']
            idx = [
                2, ((len(all_hidden_states) - 1) // 2),
                len(all_hidden_states) - 4
            ]
            hidden_states_0 = all_hidden_states[idx[0]]
            hidden_states_1 = all_hidden_states[idx[1]]
            hidden_states_2 = all_hidden_states[idx[2]]
            hidden_states = torch.cat(
                [hidden_states_0, hidden_states_1, hidden_states_2], dim=-1)
            hidden_states_from_draft = torch.zeros_like(hidden_states_0)
            draft_model.quant_forward(hidden_states,
                                      hidden_states_from_draft,
                                      input_ids=data)

    # Get quantization config and perform quantization
    mtq.quantize(draft_model, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(draft_model)
    return draft_model
