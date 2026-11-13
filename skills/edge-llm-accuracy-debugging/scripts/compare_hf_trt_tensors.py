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
"""Compare intermediate tensors between a HuggingFace model and TRT-EdgeLLM debug dumps.

Overview
--------
An instrumented TRT-EdgeLLM build can dump unfused intermediate tensors to
safetensors files. The following flags belong to that diagnostic build, not
the standard CLI:

  * ``llm_build --debugTensors`` / ``visual_build --debugTensors``  (build-time)
  * ``llm_inference --dumpTensors <dir> --dumpMultimodalTensors <dir>``  (runtime)

This script provides the complementary HuggingFace side:

1. Runs the **same prompt + image** through a HuggingFace model, capturing hidden
   states at every transformer layer via ``register_forward_hook``.

   * **Prefill** hook fires once during ``generate()`` on the full prompt.
   * **Decode** hooks fire once per generated token.
   * **Vision-encoder** hooks fire before the LLM, once per image.

2. Saves captured tensors as safetensors alongside the TRT dumps.

3. Loads both sets of tensors and computes per-step, per-tensor comparison:
   - Cosine similarity          (shape-independent)
   - Max absolute difference
   - Mean absolute difference
   - Relative L∞ error
   - NaN / Inf flags on either side

Step semantics
--------------
  step_000_prefill  →  prompt prefill (full sequence, one forward pass)
  step_001_decode   →  first generated token
  step_002_decode   →  second generated token
  ...
  step_000_vision   →  visual encoder run (multimodal-only)

Usage
-----
Text-only::

    python compare_hf_trt_tensors.py \\
        --model-dir  /path/to/hf-model \\
        --trt-dump   /path/to/trt-llm-dump \\
        --prompt     "Tell me about NVIDIA." \\
        --dtype      fp16

Vision-language (Qwen3-VL)::

    python compare_hf_trt_tensors.py \\
        --model-dir  /path/to/Qwen3-VL-2B-Instruct \\
        --trt-dump   /tmp/qwen3vl_llm_tensors \\
        --multimodal-trt-dump /tmp/qwen3vl_vis_tensors \\
        --prompt     "Describe this image in one word." \\
        --image      /path/to/image.jpg \\
        --dtype      fp16 \\
        --max-new-tokens 4 \\
        --output-json /tmp/comparison_report.json

Requirements
------------
torch, transformers, safetensors, numpy, Pillow (optional, for --image)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — clear error messages if missing
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    sys.exit("ERROR: 'torch' is required. Install via: pip install torch")

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as st_save_file
except ImportError:
    sys.exit(
        "ERROR: 'safetensors' is required. Install via: pip install safetensors"
    )

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    # AutoModelForVision2Seq was renamed / reorganised across transformers versions
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None  # type: ignore[assignment,misc]
    # Qwen3-VL specific class (transformers >= 5.x)
    try:
        from transformers.models.qwen3_vl import \
            Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    try:
        from transformers.models.internvl.modeling_internvl import \
            InternVLForConditionalGeneration
    except ImportError:
        InternVLForConditionalGeneration = None  # type: ignore[assignment,misc]
except ImportError:
    sys.exit(
        "ERROR: 'transformers' is required. Install via: pip install transformers"
    )

# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------


def load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Return all tensors from a safetensors file as numpy float32 arrays."""
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key).float().numpy()
    return out


def collect_trt_steps(dump_dir: Path) -> List[Path]:
    """Return sorted safetensors step files from *dump_dir*."""
    files = sorted(dump_dir.glob("step_*.safetensors"))
    if not files:
        print(f"[WARN] No safetensors step files found in {dump_dir}")
    return files


# ---------------------------------------------------------------------------
# HuggingFace runner — captures hidden states at every forward call
# ---------------------------------------------------------------------------


def resolve_model_directory(model_dir: str, revision: Optional[str]) -> str:
    """Resolve a local checkpoint or download one immutable Hub snapshot."""
    path = Path(model_dir).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if not revision or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError(
            "Hub models require --revision with a full immutable "
            "revision; otherwise provide a local checkpoint directory")
    import huggingface_hub

    return huggingface_hub.snapshot_download(repo_id=model_dir,
                                             revision=revision)


class HFTensorCapture:
    """Runs a HuggingFace model and captures layer outputs.

    Per-forward-call captures are accumulated in ``self.step_captures``:
        step_captures[0]  →  prefill (full prompt)
        step_captures[1]  →  first decode step
        ...

    Vision-encoder captures are in ``self.vision_captures``.
    """

    def __init__(self,
                 model_dir: str,
                 dtype: str = "float32",
                 device: str = "cuda",
                 revision: Optional[str] = None,
                 trust_remote_code: bool = False):
        model_dir = resolve_model_directory(model_dir, revision)
        self.model_dir = model_dir
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.device = device
        self._dtype_map = {
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = self._dtype_map[dtype]
        print(f"[HF] Loading model from {model_dir} (dtype={dtype}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            revision=revision,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code)

        # Read model_type from config.json so we can choose the right loader
        # before calling from_pretrained (avoids loading the wrong architecture).
        _cfg_path = Path(model_dir) / "config.json"
        _model_type: str = ""
        if _cfg_path.exists():
            with open(_cfg_path) as _f:
                _model_type = json.load(_f).get("model_type", "")

        # Models with custom code that are not directly supported by a standard
        # AutoModel class should go through tensorrt_edgellm's load_hf_model which
        # has the necessary WARs (e.g. Phi-4mm requires SlidingWindowCache shim,
        # _tied_weights_keys fix, etc.)
        _use_edgellm_loader = _model_type in ("phi4mm", )
        _use_qwen3asr_loader = _model_type in ("qwen3_asr", )

        if _use_qwen3asr_loader:
            try:
                from qwen_asr.core.transformers_backend import (  # noqa: PLC0415
                    Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor)
                self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
                    model_dir,
                    revision=revision,
                    local_files_only=True,
                    torch_dtype=self.torch_dtype,
                    device_map=device,
                ).eval()
                # Override tokenizer with the ASR-specific processor's tokenizer
                self.tokenizer = Qwen3ASRProcessor.from_pretrained(
                    model_dir, revision=revision, local_files_only=True)
                print(
                    f"[HF] Loaded via qwen_asr.Qwen3ASRForConditionalGeneration ({_model_type})"
                )
            except ImportError as e:
                sys.exit(
                    f"ERROR: qwen_asr package is required for qwen3_asr models. "
                    f"Install via: pip install qwen-asr (in a dedicated venv). Error: {e}"
                )
            except Exception as e:
                sys.exit(
                    f"ERROR: Could not load Qwen3-ASR model from {model_dir}. Error: {e}"
                )
        elif _use_edgellm_loader:
            try:
                from tensorrt_edgellm.llm_models.model_utils import \
                    load_hf_model
                dtype_str = dtype if dtype != "float32" else "fp32"
                self.model, _, _ = load_hf_model(model_dir, dtype_str, device)
                self.model = self.model.eval()
                print(
                    f"[HF] Loaded via tensorrt_edgellm.load_hf_model ({_model_type})"
                )
            except Exception as e:
                sys.exit(f"ERROR: Could not load model from {model_dir} via "
                         f"load_hf_model. Error: {e}")
        else:
            # Try several auto-model classes in order; VLMs (e.g. Qwen3-VL) are
            # registered under AutoModelForVision2Seq or AutoModel depending on
            # the transformers version.
            load_kwargs = dict(torch_dtype=self.torch_dtype,
                               device_map=device,
                               revision=revision,
                               local_files_only=True,
                               trust_remote_code=self.trust_remote_code)
            # For known text-only model types, skip VLM-specific loaders to avoid
            # misloading (e.g. Qwen3VLForConditionalGeneration succeeding on a
            # text-only Qwen3 checkpoint but returning incorrect behaviour).
            _TEXT_ONLY_TYPES = {
                "qwen3",
                "qwen2",
                "llama",
                "mistral",
                "gemma",
                "gemma2",
                "falcon",
                "gpt2",
                "gpt_neox",
                "opt",
                "bloom",
                "mpt",
                "starcoder",
                "deepseek_v2",
                "deepseek_v3",
            }
            if _model_type in _TEXT_ONLY_TYPES:
                candidates = [
                    c for c in (AutoModelForCausalLM, ) if c is not None
                ]
            else:
                candidates = [
                    c for c in (
                        Qwen3VLForConditionalGeneration,
                        Qwen2_5_VLForConditionalGeneration,
                        Qwen2VLForConditionalGeneration,
                        InternVLForConditionalGeneration,
                        AutoModelForCausalLM,
                        AutoModelForVision2Seq,
                    ) if c is not None
                ]
            for cls in candidates:
                try:
                    self.model = cls.from_pretrained(model_dir,
                                                     **load_kwargs).eval()
                    print(f"[HF] Loaded with {cls.__name__}")
                    break
                except (ValueError, TypeError, OSError, AttributeError,
                        RuntimeError, ImportError):
                    continue
            else:
                sys.exit(
                    f"ERROR: Could not load model from {model_dir} with any "
                    "AutoModel class. Check your transformers version.")

        # step_captures[i] maps label → numpy array for forward call i
        self.step_captures: List[Dict[str, np.ndarray]] = []
        self.vision_captures: Dict[str, np.ndarray] = {}

        self._hooks: list = []
        self._current_step: int = 0
        self._call_counter: int = 0  # counts model.forward() calls during generate()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _capture(self, label: str, t: torch.Tensor) -> None:
        """Store a tensor in the current step's capture dict."""
        while len(self.step_captures) <= self._current_step:
            self.step_captures.append({})
        self.step_captures[self._current_step][label] = (
            t.detach().float().cpu().numpy())

    def _make_layer_hook(self, layer_idx: int):
        """Hook for a full decoder layer — captures the residual hidden state."""

        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                self._capture(f"layers.{layer_idx}.hidden_states", t)

        return hook

    def _make_submodule_hook(self, label: str):
        """Hook for attention / MLP submodules."""

        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                self._capture(label, t)

        return hook

    def _make_step_counter_hook(self):
        """Pre-forward hook on the model itself — increments step counter."""

        def hook(module, inp):
            # The first call is prefill; subsequent calls are decode steps.
            self._current_step = self._call_counter
            self._call_counter += 1

        return hook

    def register_llm_hooks(self) -> None:
        """Attach hooks to decoder layers and the model-level step counter."""
        # Step counter on the whole model's forward.
        # For Qwen3-ASR, the outer model.generate() delegates to thinker.generate()
        # internally; the outer model.__call__ only fires once.  Register the step
        # counter on thinker so it increments per decode token.
        _cfg = getattr(getattr(self.model, "config", None), "model_type", "")
        _step_module = (self.model.thinker if _cfg == "qwen3_asr"
                        and hasattr(self.model, "thinker") else self.model)
        h = _step_module.register_forward_pre_hook(
            self._make_step_counter_hook())
        self._hooks.append(h)

        # lm_head hook — captures per-step logits for models where per-layer
        # TRT names are mangled and cannot be matched (e.g. Nemotron-H).
        # The captured tensor is named "logits" to match the TRT dump key.
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:

            def _logits_hook(module, inp, out):
                t = out if isinstance(out, torch.Tensor) else (
                    out[0] if isinstance(out, (tuple, list)) else None)
                if t is not None:
                    # Keep only the last-position logits: [B, 1, vocab]
                    self._capture("logits", t[:, -1:, :].float())

            self._hooks.append(lm_head.register_forward_hook(_logits_hook))

        # Per-layer hooks
        layers = self._find_layers()
        if layers is None:
            print(
                "[WARN] Could not locate decoder layers — no per-layer hooks registered."
            )
            return

        for i, layer in enumerate(layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_hook(i)))
            for attr, suffix in [("self_attn", "attn"), ("attn", "attn"),
                                 ("mlp", "mlp"), ("feed_forward", "mlp")]:
                sub = getattr(layer, attr, None)
                if sub is not None:
                    self._hooks.append(
                        sub.register_forward_hook(
                            self._make_submodule_hook(f"layers.{i}.{suffix}")))
                    break  # only first match per category

    def register_vision_hooks(self) -> None:
        """Attach hooks to the vision encoder."""
        vit = self._find_vit()
        if vit is None:
            print(
                "[WARN] Could not locate vision encoder — no ViT hooks registered."
            )
            return

        # Try to find individual transformer blocks
        blocks = self._find_blocks(vit)
        if blocks is None:
            # Fallback: hook the whole encoder
            def hook_all(module, inp, out):
                # Handle plain tensors, tuples/lists, and transformers ModelOutput objects
                if isinstance(out, torch.Tensor):
                    t = out
                elif isinstance(out, (tuple, list)):
                    t = out[0]
                elif hasattr(out, "last_hidden_state"):
                    t = out.last_hidden_state
                elif hasattr(out, "hidden_states") and out.hidden_states:
                    t = out.hidden_states[-1]
                else:
                    t = None
                if isinstance(t, torch.Tensor):
                    self.vision_captures["vision_encoder_output"] = (
                        t.detach().float().cpu().numpy())

            self._hooks.append(vit.register_forward_hook(hook_all))
            return

        for i, block in enumerate(blocks):

            def hook_block(module, inp, out, _i=i):
                if isinstance(out, torch.Tensor):
                    t = out
                elif isinstance(out, (tuple, list)):
                    t = out[0]
                elif hasattr(out, "last_hidden_state"):
                    t = out.last_hidden_state
                else:
                    t = None
                if isinstance(t, torch.Tensor):
                    self.vision_captures[
                        f"vision.blocks.{_i}.hidden_states"] = (
                            t.detach().float().cpu().numpy())

            self._hooks.append(block.register_forward_hook(hook_block))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Model attribute discovery helpers
    # ------------------------------------------------------------------

    def _find_layers(self):
        # Try common paths — VLMs like Qwen3-VL nest the decoder under
        # model.language_model.layers (Qwen2.5-VL/Qwen3-VL) or
        # model.language_model.model.layers (other VLMs).
        for path in (
                "model.language_model.layers",
                "model.layers",
                "model.language_model.model.layers",
                "language_model.model.layers",
                "language_model.layers",
                "model.model.layers",
                "transformer.h",
                "model.decoder.layers",
                # Qwen3-ASR: decoder layers live under thinker.model.layers
                "thinker.model.layers",
                # Nemotron-H: decoder lives under backbone.layers
                "backbone.layers",
        ):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if obj is not None and hasattr(obj,
                                               "__len__") and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        return None

    def _find_vit(self):
        for path in (
                "model.visual",
                "visual",
                "model.vision_model",
                "visual_model",
                "model.vision_tower",
                "model.image_encoder",
                # Phi-4mm: vision encoder lives inside embed_tokens_extend
                "model.model.embed_tokens_extend.image_embed",
                # Qwen3-ASR: audio encoder lives under thinker.audio_tower
                "thinker.audio_tower",
        ):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if obj is not None:
                    return obj
            except AttributeError:
                continue
        return None

    def _find_blocks(self, vit):
        for attr in ("blocks", "layers", "encoder.layers", "encoder.layer"):
            obj = vit
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def run_text(self,
                 prompt: str,
                 max_new_tokens: int = 1) -> Tuple[str, List[Dict]]:
        self.step_captures = []
        self._current_step = 0
        self._call_counter = 0
        # Apply chat template when the tokenizer supports it so that the HF
        # prompt matches the formatted prompt used by TRT-EdgeLLM.
        messages = [{"role": "user", "content": prompt}]
        formatted = None
        if hasattr(self.tokenizer,
                   "apply_chat_template") and self.tokenizer.chat_template:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
        text_input = formatted if formatted is not None else prompt
        inputs = self.tokenizer(text_input,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      do_sample=False)
        generated = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated, list(self.step_captures)

    def run_multimodal(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 1,
    ) -> Tuple[str, List[Dict], Dict[str, np.ndarray]]:
        self.step_captures = []
        self.vision_captures = {}
        self._current_step = 0
        self._call_counter = 0

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_dir,
                revision=self.revision,
                local_files_only=True,
                trust_remote_code=self.trust_remote_code)
        except Exception:
            print(
                "[WARN] AutoProcessor failed; falling back to text-only run.")
            gen, caps = self.run_text(prompt, max_new_tokens)
            return gen, caps, {}

        _model_type_str = getattr(getattr(self.model, "config", None),
                                  "model_type", "")

        if image_path:
            try:
                from PIL import Image  # noqa: PLC0415
                img = Image.open(image_path).convert("RGB")
            except ImportError:
                sys.exit(
                    "ERROR: 'Pillow' is required for --image. pip install Pillow"
                )

            if _model_type_str == "phi4mm":
                # Phi-4mm uses a string-based chat template with <|image_1|> placeholder.
                # apply_chat_template does NOT accept list-of-dicts content.
                text = processor.apply_chat_template(
                    [{
                        "role": "user",
                        "content": f"<|image_1|>\n{prompt}"
                    }],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=text,
                    images=img,
                    return_tensors="pt",
                )
                # Move all tensor fields to device/dtype; drop empty audio tensors
                # (the processor returns shape-[0] input_audio_embeds even for
                # image-only input, but the model's audio_embed is None so calling
                # it with a non-None value would raise TypeError).
                inputs = {
                    k:
                    (v.to(device=self.device,
                          dtype=self.torch_dtype if v.is_floating_point() else
                          v.dtype) if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                    if not (isinstance(v, torch.Tensor) and v.numel() == 0 and
                            k in ("input_audio_embeds", "audio_embed_sizes"))
                }
            else:
                # Qwen3-VL / Qwen2-VL style conversation with image
                messages = [{
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }]
                # enable_thinking=False matches TRT-EdgeLLM's non-thinking prompt
                try:
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # qwen_vl_utils.process_vision_info is the recommended way for
                # Qwen2-VL / Qwen3-VL to build the image_inputs/video_inputs dicts.
                try:
                    from qwen_vl_utils import \
                        process_vision_info  # noqa: PLC0415
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device, dtype=self.torch_dtype)
                except ImportError:
                    # Fallback: pass image directly (older transformers)
                    inputs = processor(
                        text=text,
                        images=img,
                        return_tensors="pt",
                    ).to(self.device, dtype=self.torch_dtype)
        elif audio_path:
            try:
                import soundfile as sf  # noqa: PLC0415
                audio_data, sr = sf.read(audio_path)
            except ImportError:
                sys.exit(
                    "ERROR: 'soundfile' is required for --audio. pip install soundfile"
                )
            if _model_type_str == "qwen3_asr":
                # Qwen3-ASR uses Qwen3ASRProcessor with a specific prompt format.
                # The processor has apply_chat_template; build messages the same way
                # Qwen3ASRModel._infer_asr_transformers does.
                msgs = [
                    {
                        "role": "system",
                        "content": ""
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "audio",
                            "audio": ""
                        }]
                    },
                ]
                text = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[text],
                                   audio=[audio_data],
                                   return_tensors="pt",
                                   padding=True)
                inputs = {
                    k:
                    v.to(device=self.device,
                         dtype=self.torch_dtype if isinstance(v, torch.Tensor)
                         and v.is_floating_point() else v.dtype) if isinstance(
                             v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    out = self.model.generate(**inputs,
                                              max_new_tokens=max_new_tokens,
                                              do_sample=False)
                input_len = inputs["input_ids"].shape[1]
                raw = processor.batch_decode(
                    out.sequences[:, input_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)[0]
                # Use qwen_asr's parser to correctly split "language {lang}<asr_text>{text}"
                # (naive regex strips first word of transcription when <asr_text> decodes to "")
                from qwen_asr import parse_asr_output  # noqa: PLC0415
                _, generated = parse_asr_output(raw)
                return generated, list(self.step_captures), dict(
                    self.vision_captures)
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": audio_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }]
            try:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=text,
                audios=audio_data,
                sampling_rate=sr,
                return_tensors="pt",
            ).to(self.device, dtype=self.torch_dtype)
        else:
            inputs = processor(text=prompt,
                               return_tensors="pt").to(self.device,
                                                       dtype=self.torch_dtype)

        with torch.no_grad():
            out = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = processor.decode(out[0][input_len:],
                                     skip_special_tokens=True)
        return generated, list(self.step_captures), dict(self.vision_captures)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def _try_squeeze(hf: np.ndarray, trt: np.ndarray):
    """Try to squeeze a leading batch dimension from hf so shapes align with trt.

    HuggingFace models often return tensors with a leading batch dimension of 1
    (e.g. shape (1, seq_len, hidden) ) while TRT drops that dimension
    (shape (seq_len, hidden)).  Strip the leading 1 if it helps alignment.
    """
    if hf.shape == trt.shape:
        return hf, trt
    if hf.ndim == trt.ndim + 1 and hf.shape[0] == 1:
        return hf.squeeze(0), trt
    if trt.ndim == hf.ndim + 1 and trt.shape[0] == 1:
        return hf, trt.squeeze(0)
    return hf, trt


def compare_tensors(hf: np.ndarray, trt: np.ndarray) -> dict:
    """Compute element-wise accuracy metrics between two arrays."""
    hf, trt = _try_squeeze(hf, trt)
    hf_f = hf.astype(np.float64).ravel()
    trt_f = trt.astype(np.float64).ravel()

    if hf.shape != trt.shape:
        # Cannot compute element-wise metrics; skip cosine too (sizes may differ)
        return {
            "shape_match": False,
            "hf_shape": list(hf.shape),
            "trt_shape": list(trt.shape),
            "cosine_similarity": float("nan"),
        }

    hf_norm = np.linalg.norm(hf_f)
    trt_norm = np.linalg.norm(trt_f)
    cos = float(np.dot(hf_f, trt_f) /
                (hf_norm * trt_norm)) if (hf_norm > 0
                                          and trt_norm > 0) else float("nan")

    diff = np.abs(hf_f - trt_f)
    ref = np.maximum(np.abs(trt_f), 1e-8)
    return {
        "shape_match": True,
        "shape": list(hf.shape),
        "cosine_similarity": cos,
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "rel_linf_error": float(np.max(diff / ref)),
        "hf_has_nan": bool(np.any(np.isnan(hf_f))),
        "trt_has_nan": bool(np.any(np.isnan(trt_f))),
    }


# ---------------------------------------------------------------------------
# Name-matching heuristic (HF hook name ↔ TRT internal tensor name)
# ---------------------------------------------------------------------------


def _layer_num(name: str) -> Optional[str]:
    import re  # noqa: PLC0415
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None


def best_match(hf_name: str, trt_names: List[str]) -> Optional[str]:
    """Return the TRT tensor name that best matches the HF hook name.

    HF hooks produce names like:
        layers.N.hidden_states  →  post-layer residual output
        layers.N.attn           →  attention module output
        layers.N.mlp            →  MLP module output
        vision.blocks.N.hidden_states  →  post-ViT-block residual

    TRT internal names follow the ONNX graph, e.g.:
        /model/layers.N/Add_1_output_0            (LLM post-block residual)
        /model/layers.N/self_attn/AttentionPlugin_output_0
        /model/layers.N/mlp/Mul_output_0.*
        /blocks.N/Add_1_output_0                  (ViT post-block residual)
        /blocks.N/attn/ViTAttentionPlugin_output_0
    """
    import re  # noqa: PLC0415

    # Exact match takes priority (e.g. "logits" == "logits")
    if hf_name in trt_names:
        return hf_name
    # Profile-suffixed match: TRT sometimes appends " [profile N]" to tensor names
    # (e.g. decode steps have "logits [profile 1]" while prefill has "logits").
    for trt_n in trt_names:
        base = trt_n.split(" [profile")[0].strip()
        if base == hf_name:
            return trt_n

    ln = _layer_num(hf_name)
    hf_lower = hf_name.lower()

    # Build a list of (score, trt_name) and return the best match
    def score(trt: str) -> int:
        s = 0
        trt_lower = trt.lower()
        # Penalise shape-input tensors (Nemotron ForeignNode scalars): they carry
        # layer numbers in their names but are size-1 constants, not activations.
        if "shapeinput" in trt_lower:
            return -1
        # Layer number match
        if ln and (f".{ln}/" in trt or f"/{ln}/" in trt
                   or f"layers.{ln}" in trt or f"blocks.{ln}" in trt):
            s += 10
        # Semantic type match — ordered from most specific to least
        if "hidden_states" in hf_lower or hf_lower.endswith(".hidden_states"):
            # Post-layer residual = final Add before next layer
            if re.search(r"/Add_\d+_output_0$", trt):
                s += 5
        if ".attn" in hf_lower or hf_lower.endswith(".attn"):
            if "attentionplugin_output_0" in trt_lower or "vitattentionplugin" in trt_lower:
                s += 5
            elif "self_attn" in trt_lower or "/attn/" in trt_lower:
                s += 3
        if ".mlp" in hf_lower or hf_lower.endswith(".mlp"):
            if "/mlp/" in trt_lower:
                s += 5
        if "vision" in hf_lower and "/blocks." in trt_lower:
            s += 2
        return s

    scored = [(score(t), t) for t in trt_names]
    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL = {"hf": 42, "trt": 52, "step": 14}


def _row(hf: str, trt: str, step: str, metrics: dict) -> str:
    cos = metrics.get("cosine_similarity", float("nan"))
    mabs = metrics.get("max_abs_diff", float("nan"))
    rel = metrics.get("rel_linf_error", float("nan"))
    sm = "YES" if metrics.get("shape_match", False) else "NO "
    return (f"{step:<{_COL['step']}} {hf:<{_COL['hf']}} {trt:<{_COL['trt']}} "
            f"{cos:>8.5f}  {mabs:>10.4e}  {rel:>10.4e}  {sm}")


def print_header():
    h = (f"{'step':<{_COL['step']}} {'HF tensor':<{_COL['hf']}} "
         f"{'TRT tensor':<{_COL['trt']}} {'cos_sim':>8}  {'max_abs':>10}  "
         f"{'rel_Linf':>10}  {'shape'}")
    print("\n" + "=" * len(h))
    print(h)
    print("=" * len(h))


def print_footer(rows: list):
    matched = [
        r for r in rows if r["metrics"].get("shape_match")
        and not np.isnan(r["metrics"].get("cosine_similarity", float("nan")))
    ]
    if matched:
        cos_vals = [r["metrics"]["cosine_similarity"] for r in matched]
        print(f"\n  Matched pairs : {len(matched)}"
              f"  |  mean cos={np.mean(cos_vals):.5f}"
              f"  |  min cos={np.min(cos_vals):.5f}"
              f"  |  max cos={np.max(cos_vals):.5f}")


def print_step_summary(step_rows: List[dict], step_label: str):
    """Print a one-line per-layer cosine similarity table for one step."""
    matched = [
        r for r in step_rows if r["metrics"].get("shape_match")
        and not np.isnan(r["metrics"].get("cosine_similarity", float("nan")))
    ]
    shape_mismatch = [
        r for r in step_rows
        if r["metrics"] and not r["metrics"].get("shape_match", True)
    ]
    no_match = [r for r in step_rows if not r["metrics"]]
    if matched:
        cos_vals = [r["metrics"]["cosine_similarity"] for r in matched]
        max_diff = max(r["metrics"].get("max_abs_diff", 0.0) for r in matched)
        print(f"  [{step_label}]  matched={len(matched):3d}  "
              f"cos mean={np.mean(cos_vals):.5f}  min={np.min(cos_vals):.5f}  "
              f"max_abs_diff={max_diff:.4e}" +
              (f"  shape_mismatch={len(shape_mismatch)}"
               if shape_mismatch else ""))
    elif shape_mismatch:
        # Report shape mismatch info (common for prefill with VLMs)
        example = shape_mismatch[0]
        hf_s = example["metrics"].get("hf_shape", "?")
        trt_s = example["metrics"].get("trt_shape", "?")
        print(f"  [{step_label}]  matched=  0  (shape mismatch: "
              f"HF {hf_s} vs TRT {trt_s}, "
              f"n_mismatch={len(shape_mismatch)})")
    elif no_match:
        print(
            f"  [{step_label}]  matched=  0  (no TRT tensor name matched for {len(no_match)} HF tensors)"
        )


# ---------------------------------------------------------------------------
# Save HF captures
# ---------------------------------------------------------------------------


def save_hf_step(captures: Dict[str, np.ndarray], path: Path) -> None:
    tensors = {k: torch.tensor(v) for k, v in captures.items()}
    st_save_file(tensors, str(path))
    print(f"[HF] Saved {len(tensors)} tensors → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HuggingFace vs TRT-EdgeLLM intermediate tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local HuggingFace checkpoint or Hub ID (requires --revision)")
    parser.add_argument(
        "--revision",
        help="Full immutable Hub revision; not required for local checkpoints")
    parser.add_argument(
        "--trt-dump",
        required=True,
        help="Directory with TRT LLM tensor dumps (step_*.safetensors)")
    parser.add_argument(
        "--multimodal-trt-dump",
        default=None,
        help="Directory with TRT multimodal encoder dumps (optional)")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--image", default=None, help="Image file for VLMs")
    parser.add_argument("--audio",
                        default=None,
                        help="Audio file for audio-LMs")
    parser.add_argument(
        "--dtype",
        choices=["float32", "fp16", "bf16"],
        default="float32",
        help="HF model dtype (default: float32; match TRT precision)")
    parser.add_argument("--device",
                        default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Tokens to generate (default: 4; one per decode step dump)")
    parser.add_argument(
        "--save-hf-captures",
        default=None,
        metavar="DIR",
        help="Directory to save HF captures as safetensors files")
    parser.add_argument("--output-json",
                        default=None,
                        metavar="PATH",
                        help="Write JSON comparison report to this file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-tensor rows (default: step-summary only)")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Python code shipped inside the checkpoint directory to "
        "run (needed only for checkpoints with custom modeling code). "
        "Off by default; enable only for a checkpoint you have reviewed.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load TRT dumps
    # ------------------------------------------------------------------
    trt_dump_dir = Path(args.trt_dump)
    if not trt_dump_dir.is_dir():
        sys.exit(f"ERROR: --trt-dump does not exist: {trt_dump_dir}")

    trt_llm_steps = collect_trt_steps(trt_dump_dir)
    print(
        f"[TRT] Found {len(trt_llm_steps)} LLM step file(s) in {trt_dump_dir}")
    for f in trt_llm_steps:
        print(f"       {f.name}")

    trt_mm_steps: List[Path] = []
    if args.multimodal_trt_dump:
        mm_dir = Path(args.multimodal_trt_dump)
        trt_mm_steps = collect_trt_steps(mm_dir)
        print(
            f"[TRT] Found {len(trt_mm_steps)} multimodal step file(s) in {mm_dir}"
        )

    # ------------------------------------------------------------------
    # Run HuggingFace model
    # ------------------------------------------------------------------
    capture = HFTensorCapture(args.model_dir,
                              dtype=args.dtype,
                              device=args.device,
                              revision=args.revision,
                              trust_remote_code=args.trust_remote_code)
    capture.register_llm_hooks()

    is_multimodal = bool(args.image or args.audio)
    if is_multimodal:
        capture.register_vision_hooks()
        generated, hf_steps, hf_vis = capture.run_multimodal(
            args.prompt,
            image_path=args.image,
            audio_path=args.audio,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        hf_vis = {}
        generated, hf_steps = capture.run_text(
            args.prompt, max_new_tokens=args.max_new_tokens)

    capture.remove_hooks()
    print(f"[HF] Generated: {generated!r}")
    print(f"[HF] Captured {len(hf_steps)} forward call(s) "
          f"({len(hf_vis)} vision tensor(s))")

    # Optionally save HF captures
    if args.save_hf_captures:
        save_dir = Path(args.save_hf_captures)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, caps in enumerate(hf_steps):
            stem = f"hf_step_{i:03d}_{'prefill' if i == 0 else 'decode'}"
            save_hf_step(caps, save_dir / f"{stem}.safetensors")
        if hf_vis:
            save_hf_step(hf_vis, save_dir / "hf_step_vision.safetensors")

    # ------------------------------------------------------------------
    # Per-step comparison
    # ------------------------------------------------------------------
    all_rows: List[dict] = []

    print_header()

    # LLM steps: prefill + decode
    # HF captures: step_captures[0] = prefill, [1..N] = decode steps
    # TRT step files: step_000_prefill, step_001_decode, step_002_decode, ...
    # Mapping: TRT prefill → HF[0], TRT decode N → HF[N] (1-indexed)
    hf_decode_counter = 1  # first decode step maps to hf_steps[1]
    for step_idx, trt_file in enumerate(trt_llm_steps):
        trt_tensors = load_safetensors(trt_file)
        trt_names = list(trt_tensors.keys())
        step_label = trt_file.stem  # e.g. "step_000_prefill"
        is_prefill = "prefill" in step_label
        if is_prefill:
            hf_step_idx = 0
        else:
            hf_step_idx = hf_decode_counter
            hf_decode_counter += 1

        if hf_step_idx >= len(hf_steps):
            print(f"  [{step_label}]  No matching HF forward call (only "
                  f"{len(hf_steps)} captured).")
            continue

        hf_caps = hf_steps[hf_step_idx]
        step_rows: List[dict] = []

        for hf_name, hf_tensor in sorted(hf_caps.items()):
            matched = best_match(hf_name, trt_names)
            if matched is None:
                row = {
                    "step": step_label,
                    "hf_name": hf_name,
                    "trt_name": "—",
                    "metrics": {}
                }
            else:
                metrics = compare_tensors(hf_tensor, trt_tensors[matched])
                row = {
                    "step": step_label,
                    "hf_name": hf_name,
                    "trt_name": matched,
                    "metrics": metrics
                }
                if args.verbose:
                    print(
                        _row(hf_name[:_COL["hf"] - 1],
                             matched[:_COL["trt"] - 1], step_label, metrics))
            step_rows.append(row)
            all_rows.append(row)

        print_step_summary(step_rows, step_label)

    # Vision encoder steps
    for trt_file in trt_mm_steps:
        trt_tensors = load_safetensors(trt_file)
        trt_names = list(trt_tensors.keys())
        step_label = trt_file.stem

        if not hf_vis:
            print(f"  [{step_label}]  No HF vision captures — "
                  "did you pass --image and register_vision_hooks?")
            continue

        step_rows = []
        for hf_name, hf_tensor in sorted(hf_vis.items()):
            matched = best_match(hf_name, trt_names)
            if matched is None:
                row = {
                    "step": step_label,
                    "hf_name": hf_name,
                    "trt_name": "—",
                    "metrics": {}
                }
            else:
                metrics = compare_tensors(hf_tensor, trt_tensors[matched])
                row = {
                    "step": step_label,
                    "hf_name": hf_name,
                    "trt_name": matched,
                    "metrics": metrics
                }
                if args.verbose:
                    print(
                        _row(hf_name[:_COL["hf"] - 1],
                             matched[:_COL["trt"] - 1], step_label, metrics))
            step_rows.append(row)
            all_rows.append(row)

        print_step_summary(step_rows, step_label)

    print_footer(all_rows)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    if args.output_json:
        report = {
            "model_dir": args.model_dir,
            "trt_dump": args.trt_dump,
            "multimodal_trt_dump": args.multimodal_trt_dump,
            "prompt": args.prompt,
            "image": args.image,
            "dtype": args.dtype,
            "generated_text": generated,
            "num_hf_forward_calls": len(hf_steps),
            "comparisons": all_rows,
        }
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\n[report] JSON written → {args.output_json}")


if __name__ == "__main__":
    main()
