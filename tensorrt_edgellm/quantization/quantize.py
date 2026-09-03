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
"""Quantize a HuggingFace LLM and export a unified checkpoint.

Loads a model via ``AutoModelForCausalLM`` (with ``AutoModelForImageTextToText``
fallback for VLMs), runs ModelOpt quantization, and writes a unified safetensors
checkpoint consumable by the checkpoint-based ``tensorrt_edgellm`` exporter.
"""

import gc
import json
import os
import shutil
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForImageTextToText,
                          AutoModelForTextToWaveform, AutoProcessor,
                          AutoTokenizer)

from .datasets import (AudioDataset, ImageDataset, TextDataset, dataset_name,
                       resolve_dataset)
from .quantization_configs import _VISUAL_PREFIXES, build_quant_config
from .qwen3_asr_loader import (asr_calibration_dataloader, is_qwen3_asr_model,
                               load_qwen3_asr_joint_for_calibration,
                               postprocess_qwen3_asr_checkpoint)
from .qwen3_cp_loader import (has_code_predictor, is_qwen3_next_omni,
                              qwen3_cp_calibration_loop,
                              qwen3_next_cp_calibration_loop)
from .qwen3_omni import (_load_omni_model, is_omni_model_dir,
                         quantize_and_export_omni)


def _text_calib_dataloader(tokenizer,
                           text_dataset: TextDataset,
                           batch_size=1,
                           num_samples=512,
                           max_length=512):
    """Return a DataLoader of tokenised ``input_ids`` for calibration.

    ``text_dataset`` is a text generator function; the first ``num_samples``
    non-empty strings it yields are tokenized.
    """
    texts = list(islice(text_dataset(), num_samples))
    if not texts:
        raise ValueError(
            f"Text calibration dataset {dataset_name(text_dataset)!r} yielded "
            f"no samples. Check dataset access / fields.")

    enc = tokenizer(texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length)
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


def _is_nemotron_h_model(model_dir: str) -> bool:
    """True if ``<model_dir>/config.json`` declares ``model_type == "nemotron_h"``."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return json.load(f).get("model_type") == "nemotron_h"
    except (OSError, ValueError):
        return False


def _is_phi4mm_model(model_dir: str) -> bool:
    """True if ``<model_dir>/config.json`` declares a Phi-4MM checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            model_type = json.load(f).get("model_type")
        return model_type in ("phi4mm", "phi4_multimodal")
    except (OSError, ValueError):
        return False


def _pre_register_phi4mm_attention_for_kv_quant(
        model: torch.nn.Module) -> None:
    # ModelOpt's register_hf_attentions_on_the_fly short-circuits when ANY
    # attention in the model uses the new ALL_ATTENTION_FUNCTIONS interface
    # (SiglipAttention in the visual encoder), so Phi4MMAttention (text
    # decoder, older-style trust_remote_code module) is never registered for
    # KV-cache BMM quantization.  Pre-register it explicitly before mtq.quantize.
    import logging

    from modelopt.torch.quantization.conversion import QuantModuleRegistry
    from modelopt.torch.quantization.plugins.attention import \
        register_attention_for_kv_quant
    logger = logging.getLogger(__name__)
    registered: set[type] = set()
    for _, module in model.named_modules():
        attn_type = type(module)
        # Match Phi4MMAttention by class name: the Phi4MM text decoder fuses
        # q/k/v into a single qkv_proj, so a ``hasattr(module, "k_proj")`` gate
        # never fires for it and it is left unregistered (hf_quant_config.json
        # is then omitted). Keep a generic trust_remote_code fallback for other
        # custom attentions that expose separate q/k/v projections.
        is_phi4mm_attn = attn_type.__name__ == "Phi4MMAttention"
        is_trc_kv_attn = (getattr(attn_type, "__module__",
                                  "").startswith("transformers_modules")
                          and hasattr(module, "k_proj"))
        if ((is_phi4mm_attn or is_trc_kv_attn) and attn_type not in registered
                and QuantModuleRegistry.get(attn_type) is None):
            register_attention_for_kv_quant(attn_type)
            registered.add(attn_type)
    if not registered:
        logger.warning(
            "_pre_register_phi4mm_attention_for_kv_quant: no trust_remote_code "
            "attention modules found; KV-cache pre-registration skipped")


def _copy_phi4mm_processor_files(model_dir: str, output_dir: str) -> None:
    for name in ("preprocessor_config.json", "processor_config.json",
                 "processing_phi4mm.py"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))


def _multimodal_calib_dataloader(processor,
                                 image_dataset: ImageDataset,
                                 num_samples: int = 128,
                                 max_length: int = 512,
                                 is_phi4mm: bool = False):
    """Yield ``BatchFeature`` dicts with ``input_ids`` + ``pixel_values``.

    Streams ``(image, question)`` pairs from ``image_dataset`` through the
    model's own ``AutoProcessor`` chat template so the visual tower receives
    real activations.  Used when ``visual_quantization`` is set — text-only
    calibration would leave visual quantizers with uninitialised scales.

    Drops down to 128 samples at batch_size=1 — VLM calibration is
    GPU-memory bound; small batches are safest.
    """
    batches: list[dict[str, Any]] = []
    for image, question in image_dataset():
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            if not is_phi4mm:
                raise
            # Phi-4MM's remote processor has a custom chat-template signature
            # that rejects ``tokenize`` / ``return_dict`` / ``return_tensors``.
            # It also uses textual image placeholders, so keep this fallback
            # Phi-4MM-only instead of applying ``<|image_1|>`` to other VLMs.
            fallback_messages = [{
                "role": "user",
                "content": f"<|image_1|>{question}",
            }]
            template_owner = processor if hasattr(
                processor, "apply_chat_template") else processor.tokenizer
            text = template_owner.apply_chat_template(
                fallback_messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text,
                               images=[image],
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=max_length)

        batches.append({
            k: v
            for k, v in inputs.items() if v is not None
            and not (isinstance(v, torch.Tensor) and v.numel() == 0)
        })
        if len(batches) >= num_samples:
            break

    if not batches:
        raise RuntimeError(
            f"No usable multimodal samples from "
            f"{dataset_name(image_dataset)!r}. Check dataset access, fields, "
            "and the processor chat template.")
    return batches


def _load_model(model_dir, dtype="fp16", device="cuda"):
    """Load model + tokenizer + optional processor via Auto* classes."""
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True,
                                                  min_pixels=128 * 28 * 28,
                                                  max_pixels=2048 * 32 * 32)
    except Exception:
        processor = None

    # NemotronH (hybrid Mamba+Attention): the custom modeling code imports
    # ``mamba_ssm.ops.triton.layernorm_gated`` and (when available)
    # ``causal_conv1d``. Apply the in-package patch BEFORE
    # AutoModelForCausalLM.from_pretrained so the modeling import resolves
    # against pure-PyTorch substitutes. This mirrors the
    # tensorrt-edgellm-quantize flow, which applies the same patch before
    # model loading.
    if _is_phi4mm_model(model_dir):
        from tensorrt_edgellm.lora import load_phi4mm_model
        model = load_phi4mm_model(model_dir, torch_dtype)
        model.to(device)
    elif is_qwen3_asr_model(model_dir):
        # Qwen3-ASR HF ckpt declares model_type="qwen3_asr" but ships no
        # modeling code, so the AutoModel factories below would fail. We
        # build a *joint* calibration model: a vanilla Qwen3ForCausalLM
        # text decoder with the from-scratch Qwen3ASRAudioEncoder attached
        # as an ``audio_tower`` submodule plus a custom forward that
        # splices audio embeddings into the text input embedding stream
        # at <|audio_pad|> positions. ModelOpt's mtq.quantize walks the
        # joint module tree, so the same forward_loop drives both halves
        # under realistic ASR-shaped activations -- the equivalent of
        # _calibrate_multimodal for VLMs.
        model, tokenizer, processor = load_qwen3_asr_joint_for_calibration(
            model_dir, torch_dtype, device)
    else:
        if _is_nemotron_h_model(model_dir):
            from .nemotron_h_patch import apply as _apply_nemotron_h_patch
            _apply_nemotron_h_patch()

        # Try ImageTextToText, then CausalLM, then the generic AutoModel.
        # ImageTextToText goes first because Qwen3.5 / Qwen3-VL register both
        # a CausalLM (text-only) and an ImageTextToText (multimodal)
        # architecture for the same checkpoint; AutoModelForCausalLM happily
        # resolves to the text-only entry and silently drops the visual tower
        # from the loaded model, breaking visual quantization downstream.
        # Some VLMs (e.g. InternVL3) are custom architectures registered only
        # under ``AutoModel``; the more specific factories raise
        # ``ValueError: Unrecognized configuration class``.  We only fall back
        # for *recognition* failures — not for ImportError or other runtime
        # errors, which would otherwise be silently masked by a misleading
        # "Unrecognized configuration class" exception.
        # Most-specific factory first: TextToWaveform > ImageTextToText >
        # CausalLM > AutoModel (fallback for custom architectures).
        factories = [
            f
            for f in (AutoModelForTextToWaveform, AutoModelForImageTextToText,
                      AutoModelForCausalLM, AutoModel) if f is not None
        ]
        last_err: Optional[Exception] = None
        for factory in factories:
            try:
                model = factory.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to(device)
                gc.collect(
                )  # release safetensor mmap handles after GPU transfer
                break
            except (ValueError, KeyError) as e:
                last_err = e
        else:
            raise RuntimeError(
                f"Could not load {model_dir} via any AutoModel factory"
            ) from last_err

    model.to(torch_dtype)

    # modelopt export_hf_checkpoint crashes when architectures is None
    # (e.g. Qwen3.5 resolves to text_config with architectures=None).
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor


def _collect_mtp_weight_map(
        weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Group MTP tensor names by safetensors shard from an HF weight map."""
    mtp_weight_map: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if "mtp" in key or "mtp" in filename:
            mtp_weight_map.setdefault(filename, []).append(key)
    return {
        filename: sorted(keys)
        for filename, keys in sorted(mtp_weight_map.items())
    }


def _extract_mtp_layer_prefixes(keys: Iterable[str]) -> list[str]:
    """Return MTP prefixes for the unquantized export fallback."""
    prefixes: set[str] = set()
    for key in keys:
        parts = key.split(".")
        if parts:
            prefixes.add(parts[0])
        for idx, part in enumerate(parts[:-1]):
            if part == "layers" and parts[idx + 1].isdigit():
                prefixes.add(".".join(parts[:idx + 2]))
                break
    return sorted(prefixes)


def _load_mtp_weights_for_unified_export(
        model: torch.nn.Module,
        model_dir: str) -> "tuple[list[str], dict[str, torch.Tensor]]":
    """Load unquantized Qwen3.5 MTP tensors for unified export fallback.

    This mirrors ModelOpt's ``load_mtp_weights`` + ``export_hf_checkpoint``
    usage while keeping the checkpoint parsing deterministic and explicit:
    tensors already present on the loaded HF model are copied into the model
    state, and tensors absent from ``model.state_dict()`` are returned for
    ``export_hf_checkpoint(extra_state_dict=...)``.
    """
    ckpt_dir = Path(model_dir)
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return [], {}

    with index_path.open(encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})

    mtp_weight_map = _collect_mtp_weight_map(weight_map)
    if not mtp_weight_map:
        return [], {}

    mtp_keys = [key for keys in mtp_weight_map.values() for key in keys]
    mtp_layer_prefixes = _extract_mtp_layer_prefixes(mtp_keys)
    model_state_keys = set(model.state_dict())
    extra_state_dict: dict[str, torch.Tensor] = {}
    loaded_count = 0

    for filename, keys in mtp_weight_map.items():
        path = ckpt_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"MTP shard referenced by index not found: "
                f"{path}")
        print(f"Loading {len(keys)} MTP tensor(s) from {filename}...")
        key_set = set(keys)
        tensors = load_file(str(path), device="cpu")
        tensors = {
            key: tensor
            for key, tensor in tensors.items() if key in key_set
        }
        in_state_dict = {
            key: tensor
            for key, tensor in tensors.items() if key in model_state_keys
        }
        if in_state_dict:
            model.load_state_dict(in_state_dict, strict=False)
            loaded_count += len(in_state_dict)
        extra_state_dict.update({
            key: tensor
            for key, tensor in tensors.items() if key not in model_state_keys
        })

    if loaded_count or extra_state_dict:
        print("Loaded MTP tensors for unified export: "
              f"{loaded_count} in model state, "
              f"{len(extra_state_dict)} via extra_state_dict.")
    if mtp_layer_prefixes:
        print(f"Detected unquantized MTP prefixes for quantization ignore: "
              f"{mtp_layer_prefixes}")

    return mtp_layer_prefixes, extra_state_dict


def _mtp_num_hidden_layers(model: torch.nn.Module) -> int:
    """Return the number of dense MTP draft layers declared by a HF config."""
    text_config = getattr(model.config, "text_config", model.config)
    return int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)


def _has_mtp_weights(model_dir: str) -> Optional[bool]:
    """Whether *model_dir* ships ``mtp.*`` draft tensors, or None if unknowable.

    ``mtp_num_hidden_layers`` describes the architecture, not the file: an Omni
    checkpoint can declare the head and ship none of its weights. Quantizing on
    the config alone would then calibrate a randomly initialised draft.

    None means the tensor names could not be read at all — no safetensors in the
    directory. That is not the same as "no MTP": the weights may arrive by some
    other route (a caller that patches loading, a non-safetensors format), so
    callers must not treat it as absence.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            keys = json.load(f).get("weight_map", {})
        return any(".mtp." in k or k.startswith("mtp.") for k in keys)

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return any(".mtp." in k or k.startswith("mtp.") for k in f.keys())
    return None


def _resolve_mtp_dir(mtp_draft_dir: Optional[str],
                     model_dir: str) -> Optional[str]:
    """Return the directory holding the MTP weights, or None to skip.

    An explicit ``--mtp_draft_dir`` is the caller asserting the weights are
    there, so a missing head is a user error and raises. Falling back to
    *model_dir* only means the config declared a head, which checkpoints do
    without shipping one, so that case skips with a diagnostic instead.
    """
    mtp_dir = mtp_draft_dir or model_dir
    present = _has_mtp_weights(mtp_dir)
    if mtp_draft_dir is not None:
        # The caller named this directory, so anything short of a confirmed
        # ``mtp.*`` set — including an unreadable or mistyped path — is theirs
        # to fix.
        if present:
            return mtp_dir
        raise ValueError(
            f"--mtp_draft_dir: {mtp_dir} ships no readable 'mtp.*' weights")
    if present is False:
        print(f"Skipping MTP quantization: {mtp_dir} declares an MTP head "
              "but ships no 'mtp.*' weights.")
        return None
    return mtp_dir


def _calibrate(model, dataloader):
    """Forward-loop calibration pass."""
    for data in tqdm(dataloader, desc="Calibrating"):
        data = data.to(model.device)
        if getattr(getattr(model, "config", None), "model_type",
                   "") in ("phi4mm", "phi4_multimodal"):
            model(input_ids=data, input_mode=0, use_cache=False)
        else:
            model(data)


def _collect_attention_q_scales_for_export(
        model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Preserve calibrated FP8 Q-BMM scales in the exported checkpoint.

    ModelOpt retains the calibrated Q-BMM amax on the quantizer, but checkpoint
    export does not emit it as ``q_proj.q_scale``. Edge-LLM needs that scale to
    quantize prefill Q to E4M3 without saturation.
    """
    q_scales: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        quantizer = getattr(module, "q_bmm_quantizer", None)
        if quantizer is None or not getattr(quantizer, "is_enabled", False):
            continue
        if getattr(module, "q_proj", None) is None:
            continue

        amax = getattr(quantizer, "_amax", None)
        if amax is None:
            raise RuntimeError(
                f"Enabled Q-BMM quantizer {module_name}.q_bmm_quantizer "
                "has no calibrated amax")
        if amax.numel() != 1:
            raise RuntimeError(
                f"Q-BMM quantizer {module_name}.q_bmm_quantizer must use "
                f"a per-tensor scale, got shape {tuple(amax.shape)}")

        maxbound = float(quantizer.maxbound)
        if maxbound <= 0.0:
            raise RuntimeError(
                f"Invalid Q-BMM maxbound for {module_name}: {maxbound}")
        scale = amax.detach().float().reshape(1).cpu() / maxbound
        if not torch.isfinite(scale).all() or scale.item() <= 0.0:
            raise RuntimeError(
                f"Invalid calibrated Q-BMM scale for {module_name}: "
                f"amax={amax.item()}, maxbound={maxbound}")
        prefix = f"{module_name}." if module_name else ""
        q_scales[f"{prefix}q_proj.q_scale"] = scale

    return q_scales


def _normalize_tied_weights_keys(model) -> None:
    """WAR for transformers >= 5.x ``_tied_weights_keys`` format change.

    Newer transformers expects each submodule's ``_tied_weights_keys``
    attribute to be a *dict-like* (so ``modeling_utils._get_tied_weight_keys``
    can call ``.keys()``). Older custom modeling code (e.g. NemotronH's
    ``modeling_nemotron_h.py``) still declares it as a *list*, which
    crashes ``model.save_pretrained`` with::

        AttributeError: 'list' object has no attribute 'keys'

    Convert list-shaped attributes to ``{key: key}`` dicts in place. The
    dict's keys exactly match the original list, preserving behavior for
    downstream tied-weight tracking. No-op for modules that already use
    the dict format or have no ``_tied_weights_keys`` set.
    """
    for module in model.modules():
        attr = getattr(module, "_tied_weights_keys", None)
        if isinstance(attr, list):
            module._tied_weights_keys = {k: k for k in attr}


def _remove_stale_safetensors_index(output_dir: str) -> None:
    """Remove a stale shard index when export produced a single safetensors file."""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    single_path = os.path.join(output_dir, "model.safetensors")
    if not (os.path.exists(index_path) and os.path.exists(single_path)):
        return
    try:
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, json.JSONDecodeError):
        return
    if any(not os.path.exists(os.path.join(output_dir, shard))
           for shard in set(weight_map.values())):
        os.remove(index_path)


def _surface_visual_q_scales(model: torch.nn.Module) -> None:
    """Save visual attention Q's calibrated scale as a top-level buffer.

    ModelOpt's ``postprocess_state_dict`` (``modelopt/torch/export/quant_utils.py``)
    renames ``k_bmm_quantizer._amax -> k_proj.k_scale`` (and v_bmm analogue) at
    save time, but has no Q rename — so ``q_bmm_quantizer._amax`` (matched by the
    ``_amax`` skip-key) is silently dropped from the saved state_dict. Register a
    ``q_scale = amax / 448`` buffer directly on the parent attention module so
    it survives ``save_pretrained -> safetensors -> load_state_dict``. The name
    "q_scale" contains none of modelopt's skip-keys, so it passes through
    ``postprocess_state_dict`` unchanged.

    LLM attention modules (whose ``q_bmm_quantizer`` is enabled by
    ``FP8_ATTN`` when ``--kv_cache_quantization fp8``) follow the legacy
    ``qScale=1.0`` convention, so we filter by visual prefix to avoid
    materializing buffers the LLM loader doesn't expect.
    """
    FP8_E4M3_MAX = 448.0
    for name, module in model.named_modules():
        if not any(p in name for p in _VISUAL_PREFIXES):
            continue
        q_q = getattr(module, "q_bmm_quantizer", None)
        if q_q is None:
            continue
        amax = getattr(q_q, "_amax", None)
        if amax is None:
            continue
        scale = (amax.detach().float() / FP8_E4M3_MAX).reshape(())
        module.register_buffer("q_scale", scale)


def _fix_generation_config_for_strict_validate(model) -> None:
    """WAR for transformers >= 5.x ``GenerationConfig.validate(strict=True)``.

    ModelOpt's ``export_hf_checkpoint`` -> ``model.save_pretrained`` ->
    ``generation_config.save_pretrained`` runs ``validate(strict=True)``
    which rejects HF model checkpoints whose ``generation_config.json``
    sets sampling-only kwargs (``top_p`` / ``top_k`` / ``temperature``)
    without setting ``do_sample = True``. NVIDIA-Nemotron-3-Nano-* and
    similar checkpoints ship that exact mismatch.

    Force ``do_sample = True`` when any sampling kwarg is present. This
    only changes the saved ``generation_config.json``; the C++ runtime
    (llm_inference / llm_bench) reads its own runtime params and does
    not depend on this file.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    sampling_set = (getattr(gc, "top_p", None) not in (None, 1.0)
                    or getattr(gc, "top_k", None) not in (None, 0, 50)
                    or getattr(gc, "temperature", None) not in (None, 1.0))
    if sampling_set and not getattr(gc, "do_sample", False):
        gc.do_sample = True


def _is_moe_model(model):
    """Return True if the model has MoE (mixture-of-experts) layers."""
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    if getattr(config, "num_experts", None) or getattr(
            config, "num_local_experts", None):
        return True
    return any("experts" in n for n, _ in model.named_modules())


def _is_hybrid_model(model):
    """Return True if the model has hybrid Mamba+Attention layers.

    Checks multiple signals: ``layers_block_type`` in config (NemotronH),
    ``mamba_ssm_dtype`` in config (Qwen3.5), or ``linear_attn`` submodules.
    """
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    if getattr(config, "layers_block_type", None) is not None:
        return True
    if getattr(config, "mamba_ssm_dtype", None) is not None:
        return True
    if any("linear_attn" in n for n, _ in model.named_modules()):
        return True
    return False


def _share_gdn_qkvzba_scales(model) -> int:
    """Unify per-tensor scales across the 4 GDN input projections.

    Groups ``in_proj_qkv``/``z``/``b``/``a`` of every GDN mixer and runs
    ModelOpt's :func:`preprocess_linear_fusion` on the group — the same
    scale-unification ModelOpt export applies to fused layers (input and
    weight amax each unified to the group max).  Identical per-tensor
    scales are the precondition for the exporter to concatenate the four
    projections into a single NVFP4 GEMM.  ModelOpt's own shared-input
    detection cannot be used here: its dummy forward misgroups projections
    on hybrid Mamba/GDN models (see ``_skip_resmooth_for_hybrid``), so the
    groups are formed explicitly by module structure.  Returns the number
    of mixers updated.
    """
    from modelopt.torch.export.quant_utils import preprocess_linear_fusion

    shared = 0
    for _, module in model.named_modules():
        projs = [
            getattr(module, n, None)
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        ]
        if any(p is None for p in projs):
            continue
        if not all(
                getattr(getattr(p, "weight_quantizer", None), "is_enabled",
                        False) for p in projs):
            continue
        preprocess_linear_fusion(projs)
        shared += 1
    return shared


@contextmanager
def _skip_resmooth_for_hybrid(model, quantization: str = ""):
    """WAR for ModelOpt resmoothing bugs on selected custom models.

    ``export_hf_checkpoint`` calls ``requantize_resmooth_fused_llm_layers``
    which averages AWQ pre_quant_scales across all linear modules that share
    the same input and re-quantizes their weights.  For hybrid models the
    dummy forward used to detect shared inputs does not propagate through
    Mamba layers correctly, and the Mamba projections (qkv, z, a, b) get
    incorrectly fused, corrupting the int4 weights.  For Phi-4 multimodal,
    the dummy forward is incompatible with the required ``input_mode``.

    For dense INT4-AWQ models resmoothing is lossy: replacing each linear's
    calibrated pre_quant_scale with the q/k/v (and gate/up) average and
    re-quantizing the int4 weights against it deviates from the calibrated
    model enough to collapse small-model accuracy (Qwen3-0.6B answers
    English prompts in Chinese; ROUGE-1 0.09 vs 0.42 without resmoothing).
    The exported checkpoint keeps each linear's own pre_quant_scale, which
    the Edge-LLM export/runtime path fully supports.  MoE INT4-AWQ models
    are exempt: expert weight stacking at export requires the shared scales
    that resmoothing produces.

    NVFP4 is exempt: resmoothing works correctly for NVFP4 and is required
    to equalise per-tensor scales across GDN input projections that share
    the same input activation, enabling fusion into a single GEMM.

    This context manager patches the resmoothing function to a no-op when the
    model needs it.  Standard transformer models are unaffected.

    TODO: Remove once ModelOpt fixes these model paths upstream.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    quantization = quantization.lower()
    # NVFP4 on hybrid models: resmoothing is safe and required for GDN
    # input projection fusion — do NOT skip.
    is_nvfp4 = quantization in ("nvfp4", "fp4")
    is_int4_awq = quantization == "int4_awq"
    # Multimodal wrappers have no top-level ``forward``; resmooth's dummy
    # ``model(fake_input)`` crashes on them. Resmooth is a no-op without
    # AWQ pre_quant_scales, so skipping is safe here.
    should_skip = ((_is_hybrid_model(model) and not is_nvfp4)
                   or model_type in ("phi4mm", "phi4_multimodal", "qwen3_omni",
                                     "qwen3_omni_moe", "qwen3_omni_next")
                   or (is_int4_awq and not _is_moe_model(model)))
    if not should_skip:
        yield
        return

    import modelopt.torch.export.unified_export_hf as _ueh
    _orig = _ueh.requantize_resmooth_fused_llm_layers

    def _noop(m):
        print("[WAR] Skipping requantize_resmooth_fused_llm_layers "
              "for this model (ModelOpt bug workaround)")

    _ueh.requantize_resmooth_fused_llm_layers = _noop
    try:
        yield
    finally:
        _ueh.requantize_resmooth_fused_llm_layers = _orig


def _is_image_blind_calibration(model, quant_cfg: dict) -> bool:
    """Return True if a text-only calibration would miss image activations.

    Needs a visual tower and an algorithm that *reshapes* weights against the
    calibration distribution rather than only recording ranges. ``awq_lite``
    searches per-channel pre_quant_scales, so text-only data optimizes the
    image-token ranges away and the model degenerates on image input.
    ``max`` (fp8, nvfp4) only records ranges and is modality-blind either way.
    """
    if not any(
            any(p in name for p in _VISUAL_PREFIXES)
            for name, _ in model.named_modules()):
        return False
    return quant_cfg.get("algorithm") not in (None, "max")


def _calibrate_multimodal(model, batches):
    """Forward-loop calibration pass for multimodal ``BatchFeature`` dicts."""
    device = model.device
    valid_batches = 0
    skipped_nan_batches = 0
    for batch in tqdm(batches, desc="Calibrating (multimodal)"):
        kwargs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                # Float inputs (pixel_values) inherit the model's dtype;
                # int inputs (input_ids, attention_mask) stay untouched.
                if v.dtype.is_floating_point:
                    v = v.to(next(model.parameters()).dtype)
            kwargs[k] = v
        kwargs.setdefault("use_cache", False)
        with torch.no_grad():
            try:
                model(**kwargs)
                valid_batches += 1
            except AssertionError as exc:
                # Some multimodal samples can produce non-finite activations
                # during calibrator collection. ModelOpt 0.44 reports this
                # through an internal assert on the calibrator amax value; skip
                # those samples so calibration can complete with valid batches.
                if "detected nan values in amax" in str(exc):
                    skipped_nan_batches += 1
                    continue
                raise
    if valid_batches == 0:
        raise RuntimeError(
            "All multimodal calibration batches were skipped due to NaN amax.")
    if skipped_nan_batches > 0:
        print(
            f"[WAR] Skipped {skipped_nan_batches} multimodal calibration batch(es) with NaN amax."
        )


def _calibrate_asr_multimodal(model, batch_iter):
    """Forward-loop calibration pass for joint ASR (audio + text) batches.

    Drives the joint Qwen3-ASR calibration model -- :func:`Qwen3ASR
    audio_tower` + vanilla ``Qwen3ForCausalLM`` text decoder + audio splice
    -- with real (audio, transcript) pairs so quantizers on both the audio
    and text paths see realistic activations. Mirror of
    :func:`_calibrate_multimodal` but consumes a generator (the LibriSpeech
    streaming dataloader is finite-but-streamed, not pre-materialised).
    """
    device = model.device
    model_dtype = next(model.parameters()).dtype
    for batch in tqdm(batch_iter, desc="Calibrating (ASR multimodal)"):
        kwargs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.dtype.is_floating_point:
                    v = v.to(model_dtype)
            kwargs[k] = v
        with torch.no_grad():
            model(**kwargs)


def quantize_and_export(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    cp_quantization: Optional[str] = None,
    visual_mha_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    audio_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    *,
    mtp_draft_dir: Optional[str] = None,
    text_dataset: Union[str, TextDataset, None] = None,
    image_dataset: Union[str, ImageDataset, None] = None,
    audio_dataset: Union[str, AudioDataset, None] = None,
    num_samples: int = 512,
    fuse_gdn_qkvzba_scales: bool = False,
) -> str:
    """Load a HuggingFace model, quantize it, and export a unified checkpoint.

    ``visual_quantization`` turns on quantization of the visual tower
    (``visual.*`` / ``vision_tower.*`` / ``multi_modal_projector.*``); when
    ``None`` (default) the visual tower stays in fp16.  Quantizing the visual
    tower with text-only calibration produces uninitialised activation scales
    on the visual path — a multimodal calibration loader is required for
    accurate visual stats (see ``A3``).

    Image calibration is selected automatically for a quantized backbone under
    an unquantized visual tower -- see ``_is_image_blind_calibration``.

    ``text_dataset`` / ``image_dataset`` / ``audio_dataset`` each accept a
    registered dataset name (str), a dataset generator function, or ``None``
    for that modality's default. Only the dataset for the modality a run
    actually calibrates is resolved, so an unknown name for an unused modality
    never fails the run; an unknown name for the modality in use fails out
    with a pointer to the customization guide.
    """
    from ..chat_template import _get_model_type
    from .models.eagle3_draft import _resolve_model_dir

    # ``model_dir`` may be a HuggingFace hub id; resolve it once so the
    # path-based consumers below (weight globs, processor-file copies, the
    # unified-export index probe) see a real directory.
    model_dir = _resolve_model_dir(model_dir)

    # Qwen3-Omni needs a joint Thinker+Talker multimodal calibration chain
    # the generic single-model path below can't express; delegate the full
    # quant+export pipeline to the dedicated driver (auto-branches MoE vs
    # non-MoE from ``config.json``). ``qwen3_omni_next`` is deliberately
    # excluded: the shared chat-template tuple lumps all three variants,
    # which would send Next into this driver whose HF classes cannot load
    # a Next checkpoint — Next dispatches to its orchestrator below.
    if _get_model_type(model_dir) in ("qwen3_omni", "qwen3_omni_moe"):
        from .qwen3_omni import quantize_qwen3_omni

        # Split num_samples across audio/image/text roughly evenly.
        third = max(1, num_samples // 3)
        quantize_qwen3_omni(
            model_dir=model_dir,
            output_dir=output_dir,
            quantization=quantization,
            lm_head_quantization=lm_head_quantization,
            kv_cache_quantization=kv_cache_quantization,
            visual_quantization=visual_quantization,
            audio_quantization=audio_quantization,
            cp_quantization=cp_quantization,
            dtype=dtype,
            device=device,
            text_dataset=text_dataset,
            num_samples=num_samples,
            talker_num_audio=third,
            talker_num_image=third,
            talker_num_text=num_samples - 2 * third,
        )
        return output_dir

    # Qwen3-Omni Next: dedicated orchestrator (transformers patch +
    # thinker/talker multimodal calib + amax backfill). Shares the ``llm``
    # flag surface; unsupported sub-encoder flags fail loudly there. EXCEPT
    # CP-only quantization (``--cp_quantization`` without ``--quantization``),
    # which the orchestrator doesn't support and the generic flow below does.
    omni_dir = is_omni_model_dir(model_dir)
    cp_only = cp_quantization is not None and quantization is None
    if omni_dir and not cp_only:
        if cp_quantization is not None:
            raise ValueError(
                "Joint --quantization + --cp_quantization on a "
                "Qwen3-Omni Next root is not supported. Run "
                "`--cp_quantization fp8` alone (CP-only checkpoint), and "
                "quantize the backbone in a separate pass.")
        kwargs = {}
        if text_dataset is not None:
            kwargs["text_dataset"] = text_dataset
        return quantize_and_export_omni(
            model_dir=model_dir,
            output_dir=output_dir,
            mtp_draft_dir=mtp_draft_dir,
            quantization=quantization,
            lm_head_quantization=lm_head_quantization,
            kv_cache_quantization=kv_cache_quantization,
            visual_quantization=visual_quantization,
            audio_quantization=audio_quantization,
            cp_quantization=cp_quantization,
            dtype=dtype,
            device=device,
            num_samples=num_samples,
            **kwargs,
        )

    t0 = time.time()
    if omni_dir:
        # CP-only on an Omni root: the ForConditionalGeneration classes need
        # the dedicated loader (transformers workarounds + delegating
        # forward for ModelOpt's dummy walk); it loads bf16-declared
        # checkpoints as bf16 regardless of the requested dtype.
        model, tokenizer, processor = _load_omni_model(model_dir, dtype,
                                                       device)
    else:
        model, tokenizer, processor = _load_model(model_dir, dtype, device)

    base_already_quantized = is_quantized(model)
    mtp_layers = _mtp_num_hidden_layers(model)
    mtp_quantized = False
    mtp_state_dict: dict[str, torch.Tensor] = {}
    mtp_dir = None
    if (mtp_layers > 0 and quantization is not None
            and not base_already_quantized):
        mtp_dir = _resolve_mtp_dir(mtp_draft_dir, model_dir)
    if mtp_dir is not None:
        text_ds = resolve_dataset(text_dataset, "text")
        from .models.mtp_draft import (export_quantized_mtp_state_dict,
                                       quantize_mtp_from_base)

        print(f"Detected {mtp_layers} MTP layer(s); quantizing MTP "
              f"draft (weights from {mtp_dir}) before base model.")
        quantized_mtp_draft = quantize_mtp_from_base(
            base_model=model,
            tokenizer=tokenizer,
            model_dir=mtp_dir,
            quantization=quantization,
            lm_head_quantization=lm_head_quantization,
            kv_cache_quantization=kv_cache_quantization,
            dtype=dtype,
            device=device,
            text_dataset=text_ds,
            num_samples=num_samples,
        )
        mtp_state_dict = export_quantized_mtp_state_dict(
            quantized_mtp_draft, dtype)
        print(f"Prepared {len(mtp_state_dict)} quantized MTP tensor(s) for "
              "unified export.")
        mtp_quantized = True
        del quantized_mtp_draft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Quantize base model ----------------------------------------------
    if base_already_quantized:
        print("Model already quantized — skipping.")
    else:
        # Fail fast when the user asks for CP quantization on a model that
        # has no CodePredictor — otherwise the cp_quantization argument
        # silently no-ops (build_quant_config still adds *code_predictor*
        # wildcards but they match nothing).
        if cp_quantization is not None and not has_code_predictor(model):
            raise ValueError(
                f"--cp_quantization={cp_quantization} requires a model with "
                "talker.code_predictor (Qwen3-Omni / Qwen3-TTS); the loaded "
                "checkpoint has none.")
        # The generic path can't dummy-walk a MoE thinker wrapper; joint
        # mode here would silently produce a Thinker-unquantized checkpoint.
        # (Qwen3-Omni never reaches here — it is delegated above.)
        if (cp_quantization is not None and quantization is not None
                and getattr(model, "thinker", None) is not None
                and "Moe" in type(model).__name__):
            raise ValueError(
                "Joint --quantization + --cp_quantization is not supported "
                "on this MoE thinker wrapper via the generic path.")
        quant_cfg = build_quant_config(
            quantization,
            lm_head_quantization,
            kv_cache_quantization,
            visual_quantization=visual_quantization,
            visual_mha_quantization=visual_mha_quantization,
            audio_quantization=audio_quantization,
            cp_quantization=cp_quantization,
            fuse_gdn_qkvzba_scales=fuse_gdn_qkvzba_scales,
        )
        # When INT4 is exported to the cuteDSL GEMM kernel's fragment layout, repack
        # requires N%64==0 && K%64==0. Small hybrid/GDN projections (e.g. Qwen3.5
        # in_proj_a / in_proj_b with out_features=16/32) cannot be repacked, so
        # exclude any 64-misaligned Linear from int4 -- it stays fp16 and exports
        # as a plain GEMM.
        if quantization == "int4_awq":
            for name, module in model.named_modules():
                if isinstance(
                        module,
                        torch.nn.Linear) and (module.out_features % 64 != 0
                                              or module.in_features % 64 != 0):
                    quant_cfg["quant_cfg"].append({
                        "quantizer_name": f"*{name}.weight_quantizer",
                        "enable": False,
                    })
                    print(
                        f"[int4] skipping {name}: weight [{module.out_features}, "
                        f"{module.in_features}] not 64-aligned (kept fp16)")
        if kv_cache_quantization is not None and _is_phi4mm_model(model_dir):
            _pre_register_phi4mm_attention_for_kv_quant(model)
        if cp_quantization is not None and is_qwen3_next_omni(model):
            # Qwen3-Omni Next (dense + MoE share the class): the Talker fires
            # the CP inside its own generate loop, so calibration drives the
            # checkpoint's reference generation path with ChatML prompts
            # instead of the hand-built Thinker->Talker chain below.
            text_ds = resolve_dataset(text_dataset, "text")
            print(f"Text calibration dataset: {dataset_name(text_ds)}")
            cp_n = min(num_samples, 64)
            # 2x margin: samples can be skipped (short thinker replies).
            texts = list(islice(text_ds(), cp_n * 2))
            mtq.quantize(
                model,
                quant_cfg,
                forward_loop=lambda m: qwen3_next_cp_calibration_loop(
                    m, tokenizer, texts, num_cp_samples=cp_n),
            )
        elif cp_quantization is not None and has_code_predictor(model):
            # CP is only reached via the Thinker->Talker->CP generation path,
            # so a dedicated loop drives that chain (bs=1: Talker uses 3D
            # RoPE, no batch-mixing). When backbone is co-quantized, prepend
            # a standard text pass; backbone forward doesn't fire CP
            # quantizers so CP amax matches standalone mode.
            text_ds = resolve_dataset(text_dataset, "text")
            print(f"Text calibration dataset: {dataset_name(text_ds)}")
            cp_loader = _text_calib_dataloader(tokenizer,
                                               text_ds,
                                               batch_size=1,
                                               num_samples=num_samples)
            cp_n = min(num_samples, 64)
            if quantization is not None:
                bb_loader = _text_calib_dataloader(tokenizer,
                                                   text_ds,
                                                   batch_size=16,
                                                   num_samples=num_samples)

                def _joint_cp_loop(m):
                    _calibrate(m, bb_loader)
                    qwen3_cp_calibration_loop(m,
                                              cp_loader,
                                              num_cp_samples=cp_n)

                mtq.quantize(model, quant_cfg, forward_loop=_joint_cp_loop)
            else:
                mtq.quantize(
                    model,
                    quant_cfg,
                    forward_loop=lambda m: qwen3_cp_calibration_loop(
                        m, cp_loader, num_cp_samples=cp_n),
                )
        elif is_qwen3_asr_model(model_dir):
            audio_ds = resolve_dataset(audio_dataset, "audio")
            print(f"Audio calibration dataset: {dataset_name(audio_ds)}")
            # ASR multimodal calibration: stream real (audio, transcript)
            # pairs through the joint audio_tower + text decoder so the
            # text quantizers see audio-embedding-spliced inputs (the
            # distribution they actually see at runtime). Mirror of the
            # visual_quantization branch below for VLMs.
            asr_samples = min(num_samples, 128)
            audio_n_window = int(model.audio_tower.config.get("n_window", 100))
            num_mel_bins = int(
                model.audio_tower.config.get("num_mel_bins", 128))
            audio_token_id = int(model._asr_audio_token_id)
            # Materialize the generator so ModelOpt can re-iterate
            # forward_loop (e.g. AutoQuantize algorithm selection).
            # Mirrors the visual path's _multimodal_calib_dataloader,
            # which returns a list for the same reason.
            batches = list(
                asr_calibration_dataloader(
                    tokenizer=tokenizer,
                    audio_token_id=audio_token_id,
                    audio_n_window=audio_n_window,
                    num_mel_bins=num_mel_bins,
                    audio_dataset=audio_ds,
                    num_samples=asr_samples,
                ))
            mtq.quantize(
                model,
                quant_cfg,
                forward_loop=lambda m: _calibrate_asr_multimodal(m, batches),
            )
        elif (visual_quantization is not None
              or _is_image_blind_calibration(model, quant_cfg)):
            image_ds = resolve_dataset(image_dataset, "image")
            print(f"Image calibration dataset: {dataset_name(image_ds)}")
            # Multimodal calibration: feed (image, text) pairs through the
            # whole VLM so visual + LLM quantizers both see real activations.
            if visual_quantization is None:
                from .gemma4_patch import apply as _apply_gemma4_patch
                _apply_gemma4_patch(model, _get_model_type(model_dir))
            processor = AutoProcessor.from_pretrained(model_dir,
                                                      trust_remote_code=True)
            mm_samples = min(num_samples, 128)
            batches = _multimodal_calib_dataloader(
                processor,
                image_dataset=image_ds,
                num_samples=mm_samples,
                is_phi4mm=_is_phi4mm_model(model_dir))
            # Mixing text batches in was tried and reverted: it wins back
            # some text accuracy but costs more on image benchmarks.
            mtq.quantize(
                model,
                quant_cfg,
                forward_loop=lambda m: _calibrate_multimodal(m, batches),
            )
        else:
            text_ds = resolve_dataset(text_dataset, "text")
            print(f"Text calibration dataset: {dataset_name(text_ds)}")
            batch_size = 16 if quantization in (None, "int4_awq") else 1
            loader = _text_calib_dataloader(tokenizer,
                                            text_ds,
                                            batch_size=batch_size,
                                            num_samples=num_samples)
            mtq.quantize(model,
                         quant_cfg,
                         forward_loop=lambda m: _calibrate(m, loader))
        if fuse_gdn_qkvzba_scales:
            n_shared = _share_gdn_qkvzba_scales(model)
            print(f"GDN qkvzba scale sharing: {n_shared} layer(s)")
        mtq.print_quant_summary(model)
        if visual_mha_quantization == "fp8":
            _surface_visual_q_scales(model)

    print(f"Quantization: {time.time() - t0:.1f}s")

    _fix_generation_config_for_strict_validate(model)
    _normalize_tied_weights_keys(model)

    if mtp_layers > 0 and not mtp_quantized:
        mtp_layer_prefixes, mtp_state_dict = (
            _load_mtp_weights_for_unified_export(model, model_dir))
        if mtp_layer_prefixes:
            model._mtp_layer_prefixes = mtp_layer_prefixes

    attention_q_scales = _collect_attention_q_scales_for_export(model)
    extra_state_dict = dict(mtp_state_dict)
    extra_state_dict.update(attention_q_scales)

    os.makedirs(output_dir, exist_ok=True)
    # MoE wrapper has no top-level ``forward`` → ``export_hf_checkpoint``'s
    # dummy walk crashes. Route CP-only quantization on such wrappers through
    # ``qwen3_omni._export_submodel(model, "talker", ...)``.
    cp_only_moe_wrapper = (cp_quantization is not None and quantization is None
                           and has_code_predictor(model)
                           and getattr(model, "thinker", None) is not None
                           and ("Moe" in type(model).__name__
                                or is_qwen3_next_omni(model)))
    if cp_only_moe_wrapper:
        from .qwen3_omni import _export_submodel
        _export_submodel(model, "talker", output_dir)
        # HF Talker configs carry ``model_type=""`` and would fail component
        # dispatch. Patch in the Talker-root type — deliberately NOT the
        # full-root ``qwen3_omni_next``, which AutoConfig would default-fill
        # with the fork's non-JSON-serializable thinker/code2wav sections;
        # an unregistered type falls back to the raw config.json everywhere.
        # The CP exporter detects Talker-root via ``code_predictor_config``.
        if is_qwen3_next_omni(model):
            cfg_path = os.path.join(output_dir, "config.json")
            with open(cfg_path) as f:
                exported_cfg = json.load(f)
            if not exported_cfg.get("model_type"):
                exported_cfg["model_type"] = "qwen3_omni_next_talker"
                with open(cfg_path, "w") as f:
                    json.dump(exported_cfg, f, indent=2)
    else:
        with torch.inference_mode(), _skip_resmooth_for_hybrid(
                model, quantization or ""):
            export_hf_checkpoint(model,
                                 export_dir=output_dir,
                                 extra_state_dict=extra_state_dict)
        if attention_q_scales:
            print("Exported calibrated Q-BMM scales for "
                  f"{len(attention_q_scales)} attention layer(s).")
    _remove_stale_safetensors_index(output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        if _is_phi4mm_model(model_dir):
            _copy_phi4mm_processor_files(model_dir, output_dir)
        else:
            processor.save_pretrained(output_dir)

    # Copy preprocessor / processor configs so downstream tools (tensorrt_edgellm's
    # tensorrt-edgellm-export, the C++ visual builder) can find image preprocessing
    # parameters (patch_size, image_mean, image_std, ...).  ``export_hf_checkpoint``
    # only writes the model + hf_quant_config; processor metadata is part of the
    # source HF directory and must be carried over explicitly.
    for fname in ("preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json", "chat_template.jinja"):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # Qwen3-ASR: convert the vanilla-Qwen3-shaped output back into the
    # qwen3_asr layout the runtime expects (re-prefix safetensors keys with
    # ``thinker.``, restore the qwen3_asr config.json + chat_template +
    # preprocessor). ``audio_tower.*`` weights are already in the exported
    # safetensors -- the joint calibration model carries them as a submodule.
    if is_qwen3_asr_model(model_dir):
        postprocess_qwen3_asr_checkpoint(model_dir, output_dir)

    print(f"Saved to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
