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
Model utility functions for loading and setting up LLM models for models quantization and ONNX export.

This module contains functions for loading Hugging Face models,
checking model types, and setting up quantization.
"""

import gc
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from modelopt.torch.quantization.utils import is_quantized_linear
from safetensors.torch import safe_open
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer, PreTrainedModel)

from .models.eagle3_draft import Eagle3DraftModel
from .models.llm_model import EdgeLLMModelForCausalLM


def is_nvfp4_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with NVFP4 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (4, 3)
    return False


def is_mxfp8_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with MXFP8 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (8, 0)
    return False


def set_dynamic_quant(model: nn.Module, dtype: str) -> None:
    """Set quantization for nvfp4 and mxfp8 quantization."""
    for module in model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == "fp16" else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
        elif is_mxfp8_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"


def is_vlm(model_dir: str) -> bool:
    """Check if the model is a VLM."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg_dict = cfg.to_dict()
    has_vision = "vision_config" in cfg_dict
    has_phi4_vision = "image_embd_layer" in cfg_dict.get("embd_layer", {})
    if (has_vision or has_phi4_vision):
        print("Set use_prompt_tuning to True")
        return True
    else:
        print("Set use_prompt_tuning to False")
        return False


def is_gptq_model(model: PreTrainedModel) -> bool:
    """Check if the model is a GPTQ model by config."""
    config = model.config.to_dict()
    quant_config = config.get("quantization_config", None)
    return quant_config and quant_config.get("quant_method") == "gptq"


def _check_model_type(model_dir: str, model_identifier: str) -> bool:
    """
    Check if a model matches a given identifier by checking model_type and architectures.
    
    Args:
        model_dir: Path to the model directory
        model_identifier: String to match against model_type or architectures (case-insensitive)
        
    Returns:
        True if model matches the identifier
    """
    try:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        return False

    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_identifier in model_type:
        return True

    archs = getattr(cfg, "architectures", []) or []
    return any(model_identifier in str(a).lower() for a in archs)


def _is_phi4mm_model(dir_path: str) -> bool:
    """Check if the model is a Phi4MM model."""
    return _check_model_type(dir_path, "phi4mm")


# Models that require explicit chat template because auto-extraction fails
INCOMPATIBLE_CHAT_TEMPLATE_MODELS = [
    "phi4mm",  # Phi-4-multimodal: tokenizer lacks proper chat template
]


def is_incompatible_chat_template_model(model_dir: str) -> Tuple[bool, str]:
    """
    Check if the model requires an explicit chat template file.
    
    Some models have tokenizers that don't contain proper chat templates
    or have incompatible formats that cannot be auto-extracted.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Tuple of (is_incompatible, model_identifier):
            - is_incompatible: True if model requires explicit chat template
            - model_identifier: String identifying the incompatible model type (empty if compatible)
    """
    for model_identifier in INCOMPATIBLE_CHAT_TEMPLATE_MODELS:
        if _check_model_type(model_dir, model_identifier):
            return True, model_identifier

    return False, ""


def _load_phi4mm_war(model_dir: str):
    """
    Dynamically import local modeling_phi4mm.py as a synthetic package so that
    relative imports work, then inject a no-op prepare_inputs_for_generation
    on Phi4MMModel to satisfy PEFT checks during initialization.
    """
    package_name = "local_phi4mm"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [model_dir]
        sys.modules[package_name] = pkg

    # Preload configuration module if present (support both relative and absolute imports)
    cfg_path = os.path.join(model_dir, "configuration_phi4mm.py")
    if os.path.exists(cfg_path):
        cfg_name_local = f"{package_name}.configuration_phi4mm"
        if cfg_name_local not in sys.modules:
            cfg_spec = importlib.util.spec_from_file_location(
                cfg_name_local, cfg_path)
            cfg_mod = importlib.util.module_from_spec(cfg_spec)
            sys.modules[cfg_name_local] = cfg_mod
            sys.modules["configuration_phi4mm"] = cfg_mod
            cfg_mod.__package__ = package_name
            assert cfg_spec is not None and cfg_spec.loader is not None
            cfg_spec.loader.exec_module(cfg_mod)

    module_name = f"{package_name}.modeling_phi4mm"
    mdl_path = os.path.join(model_dir, "modeling_phi4mm.py")
    spec = importlib.util.spec_from_file_location(module_name, mdl_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    sys.modules["modeling_phi4mm"] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    lora_dir = os.path.join(model_dir, "vision-lora")
    if os.path.exists(lora_dir):
        print(f"Loading LoRA models into the PEFT framework.")
        if hasattr(module, "Phi4MMModel"):

            def _fake_prepare_inputs_for_generation(self, *args, **kwargs):
                pass

            module.Phi4MMModel.prepare_inputs_for_generation = _fake_prepare_inputs_for_generation
    else:
        # WAR: Override Phi4MMForCausalLM.__init__ to prevent the model from being
        # converted into a PEFT model, which modelopt and transformers cannot handle correctly.
        # The LoRA weights have already been merged into the base model.
        if (hasattr(module, "Phi4MMForCausalLM")
                and hasattr(module, "Phi4MMModel")
                and hasattr(module, "Phi4MMPreTrainedModel")):

            def _phi4mm_init_war(self, config):
                module.Phi4MMPreTrainedModel.__init__(self, config)
                self.model = module.Phi4MMModel(config)
                self.vocab_size = config.vocab_size
                self.lm_head = nn.Linear(config.hidden_size,
                                         config.vocab_size,
                                         bias=False)

            module.Phi4MMForCausalLM.__init__ = _phi4mm_init_war

        if hasattr(module, "Phi4MMImageAudioEmbedding"):

            def _phi4mm_image_audio_embedding_init_text_only(
                    self, config, **kwargs):
                nn.Module.__init__(self)
                self.vocab_size = config.vocab_size

                # Keep token ids consistent for assertions/BC.
                self.image_input_id = kwargs.get("image_input_id", -1)
                self.audio_input_id = kwargs.get("audio_input_id", -10000)
                assert self.image_input_id != self.audio_input_id, (
                    "image_input_id and audio_input_id should be different")
                self.image_embed = None
                self.audio_embed = None
                self.input_image_embeds = None
                self.image_sizes = None
                self.image_attention_mask = None
                self.input_audio_embeds = None
                self.audio_embed_sizes = None

            # Override Phi4MMImageAudioEmbedding.__init__ to set `image_embed` and `audio_embed` to None.
            # This avoids creating the image/audio towers in the LLM export/quantization pipeline, which
            # is not compatible with ModelOpt currently. We export the visual encoder with a
            # dedicated script (`tensorrt-edgellm-export-visual`) that handles the visual model separately.
            module.Phi4MMImageAudioEmbedding.__init__ = _phi4mm_image_audio_embedding_init_text_only

    return module


def load_hf_model(
    model_dir: str, dtype: str, device: str
) -> Tuple[Union[AutoModelForCausalLM, AutoModelForImageTextToText],
           AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load a HuggingFace model, tokenizer, and optional processor with automatic model type detection.
    
    Args:
        model_dir: Directory containing the model files
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        
    Returns:
        Tuple of (model, tokenizer, processor)
        processor will be None if AutoProcessor cannot be loaded from the model directory
        
    Raises:
        ValueError: If dtype is not supported or model loading fails
    """
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)

    # Due to a known loading issue with Phi4MM on recent transformers, special handling is required.
    # See: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/75.
    if _is_phi4mm_model(model_dir):
        module = _load_phi4mm_war(model_dir)
        model = module.Phi4MMForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            _attn_implementation="eager").to(device)
    else:
        # Try loading as AutoModelForCausalLM first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                _attn_implementation="eager").to(device)
        except Exception:
            # If that fails, try AutoModelForImageTextToText
            try:
                # TODO: Need a WAR to quantize only the language model.
                # In VLMs, the model has both model.language_model and model.vision_model.
                model = AutoModelForImageTextToText.from_pretrained(
                    model_dir, torch_dtype=torch_dtype,
                    trust_remote_code=True).to(device)
            except Exception as e:
                raise ValueError(
                    f"Could not load model from {model_dir}. Error: {e}")
    if not is_gptq_model(model):
        model.to(torch_dtype)

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load processor if available
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            # The fields are required because during quantization it may OOM due to large images in the dataset.
            min_pixels=128 * 28 * 28,
            max_pixels=2048 * 32 * 32)
        print(
            f"Warning: Loaded processor from {model_dir}. The processor will skip image processing for images smaller than 128x28x28 or bigger than 2048x32x32 due to excessive memory usage during image quntization."
        )
    except Exception:
        # Processor not available for this model
        pass

    return model, tokenizer, processor


def load_llm_model(
    model_dir: str,
    dtype: str,
    device: str,
    is_eagle_base: bool,
    reduced_vocab_size: Optional[int] = None,
    vocab_map: Optional[torch.Tensor] = None
) -> tuple[nn.Module, bool, AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load a language model (standard or EAGLE base).
    
    Args:
        model_dir: Directory containing the torch model
        dtype: Model dtype
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        is_eagle_base: Whether this is an EAGLE3 base model
        reduced_vocab_size: Size of the reduced vocabulary (optional)
        vocab_map: Tensor of shape (reduced_vocab_size,) with int32 indices for vocabulary reduction (optional)
        
    Returns:
        tuple: (model, use_prompt_tuning, tokenizer, processor)
        processor will be None if AutoProcessor cannot be loaded from the model directory
    """
    # Determine model type and print message
    if is_eagle_base:
        print(f"Loading eagle3 base model from {model_dir}")
    else:
        print(f"Loading standard model from {model_dir}")

    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)
    use_prompt_tuning = is_vlm(model_dir)
    set_dynamic_quant(model, dtype)

    # Create EdgeLLMModelForCausalLM wrapper.
    edge_model = EdgeLLMModelForCausalLM(model, is_eagle_base,
                                         use_prompt_tuning, reduced_vocab_size,
                                         vocab_map)

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return edge_model, use_prompt_tuning, tokenizer, processor


def load_eagle3_draft_model(draft_model_dir: str, base_model_dir: str,
                            use_prompt_tuning: bool, dtype: str,
                            device: str) -> nn.Module:
    """
    Load an EAGLE draft model with base model for weight copying.
    
    Args:
        draft_model_dir: Directory containing the draft model
        base_model_dir: Directory containing the base model 
        use_prompt_tuning: Whether the model uses prompt tuning
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        
    Returns:
        nn.Module: Draft model
    """
    print(f"Loading eagle3 draft model from {draft_model_dir}")
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Load draft model using from_pretrained. Draft model only support fp16.
    draft_model = Eagle3DraftModel.from_pretrained(
        draft_model_dir=draft_model_dir,
        base_model_dir=base_model_dir,
        use_prompt_tuning=use_prompt_tuning,
        device=device).eval().to(device)
    if not is_gptq_model(draft_model):
        draft_model.to(torch_dtype)

    set_dynamic_quant(draft_model, dtype)

    return draft_model


def load_tensor_by_candidate_keys(model_dir: str, keys_candidate: List[str],
                                  device: str) -> Optional[torch.Tensor]:
    """
    Search all .safetensors shards in `model_dir` and lazily load
    the first matching tensor in `candidate_keys`.

    Returns
    -------
    tensor : Optional[torch.Tensor]
        The requested tensor moved to `device`.
    """
    model_dir = Path(model_dir)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for shard_path in safetensor_files:
        # Lazy/MMAP open – only metadata is read
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in keys_candidate:
                # try candidates in order
                if key in keys:
                    print(f"Using {key} from {model_dir}/{shard_path.name}")
                    tensor = f.get_tensor(key)  # actually loads data
                    return tensor.to(device)  # move to desired device

    return None


def load_reduced_vocab_map(reduced_vocab_dir: str,
                           device: str) -> Tuple[int, torch.Tensor]:
    """
    Load the reduced vocabulary map from a directory.
    
    The directory should contain a vocab_map.safetensors file with a 'vocab_map' tensor.
    
    Args:
        reduced_vocab_dir: Directory containing vocab_map.safetensors
        device: Device to load the tensor on
        
    Returns:
        Tuple of (reduced_vocab_size, vocab_map)
        
    Raises:
        FileNotFoundError: If vocab_map.safetensors is not found
        KeyError: If 'vocab_map' key is not found in the file
    """
    reduced_vocab_dir = Path(reduced_vocab_dir)
    vocab_map_file = reduced_vocab_dir / "vocab_map.safetensors"

    if not vocab_map_file.exists():
        raise FileNotFoundError(
            f"vocab_map.safetensors not found in {reduced_vocab_dir}")

    print(f"Loading vocab_map from {vocab_map_file}")

    with safe_open(vocab_map_file, framework="pt", device="cpu") as f:
        if "vocab_map" not in f.keys():
            raise KeyError(
                f"'vocab_map' key not found in {vocab_map_file}. Available keys: {list(f.keys())}"
            )
        vocab_map = f.get_tensor("vocab_map")

    vocab_map = vocab_map.to(device)
    reduced_vocab_size = vocab_map.shape[0]

    print(f"Loaded vocab_map with reduced_vocab_size={reduced_vocab_size}")

    return reduced_vocab_size, vocab_map
