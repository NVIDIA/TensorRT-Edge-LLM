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
"""Qwen3-ASR quantization helpers.

Qwen3-ASR's HF checkpoint declares ``model_type="qwen3_asr"`` but ships
no in-tree HF modeling code, so the vanilla ``AutoModel`` factories in
:func:`tensorrt_edgellm.quantization.quantize._load_model` raise
``ValueError: Unrecognized configuration class``. This module wraps the
standalone joint calibration model in
:mod:`tensorrt_edgellm.quantization.models.qwen3_asr` with the bookkeeping
``_load_model`` expects, plus a LibriSpeech audio+transcript calibration
dataloader and a post-process step to re-prefix the saved unified
checkpoint back to the qwen3_asr namespace.

The actual modeling (audio encoder, joint factory) lives under
:mod:`tensorrt_edgellm.quantization.models.qwen3_asr` -- pure ``nn.Linear``
throughout, no imports from :mod:`tensorrt_edgellm`. The runtime
side has its own quant-aware modeling tree under
:mod:`tensorrt_edgellm.models.qwen3_asr` for ONNX export.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
from typing import Dict, Iterator

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tensorrt_edgellm._safetensors_io import save_file

from .datasets import AudioDataset, dataset_name
from .models.qwen3_asr import Qwen3ASRForConditionalGeneration
from .models.qwen3_asr.modeling_qwen3_asr_audio import prepare_audio_inputs

logger = logging.getLogger(__name__)


def is_qwen3_asr_model(model_dir: str) -> bool:
    """Return True if *model_dir* holds a qwen3_asr HF checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return json.load(f).get("model_type") == "qwen3_asr"
    except (OSError, ValueError):
        return False


def load_qwen3_asr_joint_for_calibration(model_dir: str, torch_dtype, device):
    """Build a joint Qwen3-ASR calibration model.

    Thin wrapper around
    :meth:`Qwen3ASRForConditionalGeneration.from_pretrained` that also
    builds the tokenizer, matching the ``(model, tokenizer, processor)``
    contract of :func:`tensorrt_edgellm.quantization.quantize._load_model`.

    ``processor`` is always ``None`` -- the ASR calibration loader builds
    its own tokenized inputs from raw audio + transcripts, so no HF
    processor is required.
    """
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    return model, tokenizer, None


def asr_calibration_dataloader(
    tokenizer,
    audio_token_id: int,
    audio_n_window: int,
    num_mel_bins: int = 128,
    *,
    audio_dataset: AudioDataset,
    num_samples: int = 128,
    max_audio_seconds: float = 20.0,
    sample_rate: int = 16000,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Stream (audio, transcript) pairs as joint-forward batches.

    Yields one batch dict per sample; each dict is consumable directly by
    the joint forward in
    :meth:`Qwen3ASRForConditionalGeneration.from_pretrained` (and by
    :func:`tensorrt_edgellm.quantization.quantize._calibrate_asr_multimodal`).

    The chat-template prompt mirrors the runtime layout (system / user
    with a single ``<|audio_pad|>`` between ``<|audio_start|>`` and
    ``<|audio_end|>`` / assistant prefix), with the assistant content
    filled in with the ground-truth transcript so the LLM's calibration
    sees realistic generation activations as well as prompt-side ones.
    """
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    def extract_mel_spectrogram(audio: np.ndarray, sample_rate: int,
                                feature_extractor_type: str):
        """Offline-only helper: build mel via HF WhisperFeatureExtractor for
        quantization calibration data.

        The runtime audio path itself does NOT use Python — it's pure C++
        via miniaudio + the in-tree mel extractor in
        ``cpp/runtime/melSpectrogram``. This calibration path stays in
        Python because (a) calibration runs once at quantize-time on CPU
        and (b) the calibrator wants a numpy tensor it can feed into the
        model's forward. Parameters match Qwen3-Omni / Qwen3-ASR
        ``preprocessor_config.json`` (``feature_size=128``,
        ``hop_length=160``, ``n_fft=400``); HF's default ``feature_size``
        is 80, which would produce a mel shape the audio encoder cannot
        consume.
        """
        if feature_extractor_type != "whisper":
            raise ValueError(
                "qwen3_asr_loader: only whisper FE supported here")
        fe = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=sample_rate,
            hop_length=160,
            n_fft=400,
            return_attention_mask=True,
            padding_value=0.0,
        )
        out = fe(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np",
            return_attention_mask=True,
            padding=False,
            truncation=False,
        )
        mel = out["input_features"][0]
        valid_len = int(out["attention_mask"][0].sum())
        return mel, valid_len

    def _build_prompt(transcript: str) -> str:
        # Match the model-owned runtime contract. The single ``<|audio_pad|>``
        # is what the joint forward splices the audio embeddings into.
        return ("<|im_start|>system\n<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\n"
                f"language English{transcript}<|im_end|>")

    yielded = 0
    for audio_bytes, transcript in audio_dataset():
        if yielded >= num_samples:
            break
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                continue
            duration = audio.shape[0] / sr
            if duration > max_audio_seconds:
                continue
            transcript = (transcript or "").strip()
            if not transcript:
                continue
        except (OSError, ValueError, RuntimeError) as exc:
            # ``RuntimeError`` covers ``soundfile.LibsndfileError`` for
            # corrupt audio bytes. Real bugs (``TypeError``,
            # ``AttributeError``, ...) propagate. The ``yielded == 0`` check
            # below turns the all-samples-skipped case into a loud failure.
            logger.debug("Skipping calib sample (read error): %s", exc)
            continue

        mel, mel_len = extract_mel_spectrogram(
            audio, sample_rate=sr, feature_extractor_type="whisper")
        # ``extract_mel_spectrogram`` returns ``[mel_bins, T_padded]``
        # numpy; the encoder ``forward`` expects ``[B, mel_bins, T]``
        # float16, so unsqueeze the batch dim and convert.
        input_features = torch.from_numpy(mel).unsqueeze(0).to(torch.float32)
        feature_lens = torch.tensor([mel_len], dtype=torch.int64)

        ragged = prepare_audio_inputs(
            input_features,
            feature_lens,
            n_window=audio_n_window,
            dtype=input_features.dtype,
        )

        prompt = _build_prompt(transcript)
        token_ids = tokenizer(prompt,
                              return_tensors="pt",
                              add_special_tokens=False)["input_ids"]
        if (token_ids == audio_token_id).sum().item() != 1:
            # Defensive: tokenizer changes that re-split <|audio_pad|>
            # would break the joint forward's splice.
            logger.debug("Skipping sample without exactly one audio_pad token")
            continue

        yield {
            "input_ids": token_ids,
            "audio_padded_feature": ragged["padded_feature"],
            "audio_indices": ragged["padded_mask_after_cnn_indices"],
            "audio_attn_mask": ragged["attention_mask"],
        }
        yielded += 1

    if yielded == 0:
        raise RuntimeError(
            f"ASR calibration dataloader produced 0 samples from "
            f"{dataset_name(audio_dataset)!r}. Check dataset access / "
            f"streaming.")
    logger.info("ASR calibration: %d samples streamed from %s", yielded,
                dataset_name(audio_dataset))


def postprocess_qwen3_asr_checkpoint(model_dir: str, output_dir: str) -> None:
    """Convert eo1's vanilla-Qwen3-shaped output back into qwen3_asr layout.

    Mutates *output_dir* in place:

      * ``model.safetensors`` -- prepends ``thinker.`` to every saved key
        (so ``model.embed_tokens.weight`` becomes
        ``thinker.model.embed_tokens.weight``,
        ``audio_tower.layers.0.self_attn.q_proj.weight`` becomes
        ``thinker.audio_tower.layers.0.self_attn.q_proj.weight``, etc.).
      * ``config.json`` -- replaced with the original qwen3_asr config
        (preserves ``thinker_config.audio_config`` /
        ``thinker_config.text_config`` and ``model_type="qwen3_asr"``
        which eo2 routes on).
      * Optional ``chat_template.jinja`` and ``preprocessor_config.json`` --
        copied from *model_dir* if present (Qwen3 tokenizer save does not
        produce these). Standard Qwen3-ASR checkpoints use the native runtime
        renderer selected later during engine artifact generation.

    ``hf_quant_config.json`` is untouched: ModelOpt writes short module
    names (``model.layers.0.self_attn.q_proj`` /
    ``audio_tower.layers.0.self_attn.q_proj`` / ``lm_head``), and
    ``tensorrt_edgellm.config._normalize_module_name`` strips the ``model.``
    prefix during lookup, so keys remain valid against the re-prefixed
    safetensors.
    """
    saved_path = os.path.join(output_dir, "model.safetensors")
    if not os.path.exists(saved_path):
        raise FileNotFoundError(
            f"Expected eo1 output at {saved_path}; nothing to post-process.")

    saved_weights = load_file(saved_path, device="cpu")
    merged: Dict[str, torch.Tensor] = {
        f"thinker.{k}": v
        for k, v in saved_weights.items()
    }

    # The joint calibration model always carries audio_tower.* in its
    # state_dict (whether quantized or kept FP16). If those keys are
    # missing here, something upstream dropped them -- fail loud rather
    # than silently merge from the source ckpt.
    has_audio = any(
        k.startswith("thinker.audio_tower.") for k in merged.keys())
    if not has_audio:
        raise RuntimeError(
            f"No thinker.audio_tower.* keys in {saved_path} after "
            "export_hf_checkpoint. The joint calibration model should "
            "always include audio_tower.* in its state_dict -- "
            "investigate whether mtq.quantize / export_hf_checkpoint "
            "dropped them.")

    logger.info("Re-prefixed %d tensors with 'thinker.' in %s",
                len(saved_weights), saved_path)
    save_file(merged, saved_path)

    shutil.copy2(os.path.join(model_dir, "config.json"),
                 os.path.join(output_dir, "config.json"))
    logger.info("Replaced config.json with original qwen3_asr config")

    for name in ("chat_template.jinja", "preprocessor_config.json"):
        src = os.path.join(model_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info("Copied %s from original checkpoint", name)


__all__ = [
    "is_qwen3_asr_model",
    "load_qwen3_asr_joint_for_calibration",
    "asr_calibration_dataloader",
    "postprocess_qwen3_asr_checkpoint",
]
