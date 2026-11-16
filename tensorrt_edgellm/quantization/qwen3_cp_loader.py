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
"""CP calibration loop: drives Thinker → Talker → CP so CP quantizers see
real activations (a plain ``model(input_ids=...)`` never hits CP, leaving
its input_quantizer amax un-initialised → degenerate FP8 scales).
Dispatches by ``has_thinker``: Omni runs Thinker + projection → Talker;
Qwen3-TTS has no Thinker and projects its own text embeddings instead.

The two families keep their codec tokens in the same place but their text
tokens in different ones, so the paths are not interchangeable:
``talker.get_input_embeddings()`` is the *codec* table in both, while
Qwen3-TTS holds text in a separate ``get_text_embeddings()`` that must be
resized by ``text_projection``.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ModelOpt <=0.44's ``get_expert_linear_names`` falls back to Mixtral's
# ``[w1,w2,w3]`` for Qwen3-Omni-MoE / Qwen3-Omni Next experts (which use
# ``gate/up/down_proj``), crashing the CP-only exporter that still
# enumerates every MoE block.
def _install_modelopt_expert_name_patch() -> None:
    try:
        from modelopt.torch.export import layer_utils, unified_export_hf
    except ImportError:
        return
    orig = layer_utils.get_expert_linear_names

    def patched(module):
        name = type(module).__name__.lower()
        if (("qwen3omnimoe" in name or "qwen3omninext" in name)
                and "sparsemoeblock" in name):
            return ["gate_proj", "down_proj", "up_proj"]
        return orig(module)

    if getattr(layer_utils.get_expert_linear_names, "__is_cp_patch__", False):
        return  # already installed
    patched.__is_cp_patch__ = True  # type: ignore[attr-defined]
    layer_utils.get_expert_linear_names = patched
    unified_export_hf.get_expert_linear_names = patched


_install_modelopt_expert_name_patch()


def has_code_predictor(model) -> bool:
    """True if ``model.talker.code_predictor`` exists (Qwen3-Omni + Qwen3-TTS)."""
    talker = getattr(model, "talker", None)
    if talker is None:
        return False
    return getattr(talker, "code_predictor", None) is not None


def is_qwen3_next_omni(model) -> bool:
    """True for Qwen3-Omni Next (``Qwen3OmniNextForConditionalGeneration``),
    dense and sparse-MoE alike (they share the class; MoE-ness only shows
    in ``num_experts``)."""
    return "Qwen3OmniNext" in type(model).__name__


def _talker_inputs_from_thinker(model, input_ids, talker_cfg, accept_layer,
                                audio_tok, image_tok, video_tok):
    """Omni path: Thinker hidden -> Talker input embeds. Returns None if the
    Thinker didn't yield enough hidden layers (caller skips the sample)."""
    talker = model.talker
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    with torch.no_grad():
        thinker_out = model.thinker(input_ids=input_ids,
                                    output_hidden_states=True,
                                    use_cache=False)
    all_hidden = thinker_out.hidden_states
    if all_hidden is None or len(all_hidden) <= accept_layer:
        return None

    thinker_hidden = all_hidden[accept_layer]
    thinker_embed = all_hidden[0]
    mm_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tok in (audio_tok, image_tok, video_tok):
        if tok >= 0:
            mm_mask |= (input_ids == tok)
    talker_inputs = torch.empty(bsz,
                                seq_len,
                                talker_cfg.text_config.hidden_size,
                                dtype=talker.dtype,
                                device=device)
    text_mask = ~mm_mask
    if mm_mask.any():
        talker_inputs[mm_mask] = talker.hidden_projection(
            thinker_hidden[mm_mask].to(talker.dtype))
    if text_mask.any():
        talker_inputs[text_mask] = talker.text_projection(
            thinker_embed[text_mask].to(talker.dtype))
    return talker_inputs


def _talker_inputs_from_text(model, input_ids):
    """Qwen3-TTS path: text ids -> text embedding -> ``text_projection``.

    ``get_input_embeddings()`` on the TTS Talker is the codec table
    (``vocab_size`` rows), so feeding it text-vocabulary ids indexes out of
    bounds. Text lives in ``get_text_embeddings()`` (``text_vocab_size``
    rows, ``text_hidden_size`` wide) and ``text_projection`` resizes it to
    the Talker hidden size, mirroring the reference ``generate``.
    """
    talker = model.talker
    text_embed = talker.get_text_embeddings()
    return talker.text_projection(text_embed(input_ids).to(talker.dtype))


def qwen3_cp_calibration_loop(model, dataloader, num_cp_samples: int = 64):
    """Drive Talker → CodePredictor calibration.

    Per batch: build Talker inputs (Thinker+projection for Omni,
    text_projection for Qwen3-TTS), Talker prefill, then ``cp.generate`` for
    ``num_code_groups - 1`` steps so every per-codebook lm_head sees real
    activations (matches the production inference call).  ``num_cp_samples``
    caps the batches actually processed; 64 typically suffices for stable
    per-tensor amax.
    """
    device = next(model.parameters()).device
    talker = model.talker
    cp = talker.code_predictor
    has_thinker = getattr(model, "thinker", None) is not None

    talker_cfg = model.config.talker_config
    cp_cfg = talker_cfg.code_predictor_config
    accept_layer = getattr(talker_cfg, "accept_hidden_layer", 14)
    num_code_groups = cp_cfg.num_code_groups
    # Match the production generate() call (modeling_qwen3_omni.py
    # ``code_predictor.generate(max_new_tokens=num_code_groups - 1)``).
    # HF generate() counts the prefill output as one of max_new_tokens,
    # so this exercises every ``lm_head[0..num_code_groups - 2]``.
    cp_max_new_tokens = num_code_groups - 1

    # Omni-only multimodal token IDs used to route between
    # text_projection (text) and hidden_projection (audio/image/video).
    # TTS path is purely text and ignores these.
    text_cfg = (model.config.thinker_config.text_config
                if has_thinker else None)
    audio_tok = (getattr(text_cfg, "audio_token_id", -1)
                 if text_cfg is not None else -1)
    image_tok = (getattr(text_cfg, "image_token_id", -1)
                 if text_cfg is not None else -1)
    video_tok = (getattr(text_cfg, "video_token_id", -1)
                 if text_cfg is not None else -1)

    talker_embed = talker.get_input_embeddings()
    label = "Omni+CP" if has_thinker else "TTS+CP"

    samples_done = 0
    for batch in tqdm(dataloader, desc=f"Calibrating {label}"):
        if samples_done >= num_cp_samples:
            break
        input_ids = batch.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        bsz, seq_len = input_ids.shape

        # 1-2. Build Talker inputs_embeds (model-specific path).
        if has_thinker:
            talker_inputs = _talker_inputs_from_thinker(
                model, input_ids, talker_cfg, accept_layer, audio_tok,
                image_tok, video_tok)
            if talker_inputs is None:
                continue
        else:
            talker_inputs = _talker_inputs_from_text(model, input_ids)

        # 3. Talker prefill — pass position_ids directly to skip
        # ``get_rope_index`` (which on Qwen3-Omni needs ``talker_input_ids``
        # + ``image_grid_thw``).  The TTS Talker recomputes them from
        # ``attention_mask`` anyway, which for an all-ones mask is the same
        # ``arange``.
        position_ids = torch.arange(seq_len,
                                    device=device).view(1, 1, -1).expand(
                                        3, bsz, -1).contiguous()
        attention_mask = torch.ones(bsz,
                                    seq_len,
                                    dtype=torch.long,
                                    device=device)
        with torch.no_grad():
            talker_out = talker(inputs_embeds=talker_inputs,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                output_hidden_states=True,
                                use_cache=False,
                                return_dict=True)
        # Talker returns ``hidden_states = (per_layer_tuple, residual_codes)``;
        # ``[0][-1]`` is the last layer.
        talker_hidden_states = getattr(talker_out, "hidden_states", None)
        if talker_hidden_states is None or talker_hidden_states[0] is None:
            continue
        talker_last_hidden = talker_hidden_states[0][-1][:, -1:, :]

        # 4. Drive CP through prefill + (num_code_groups - 2) generation
        # steps.  ``cp.generate`` walks every lm_head / codec_embedding
        # internally via ``_update_model_kwargs_for_generation``.
        # Bound the seed token by the codec table itself: Qwen3-Omni nests the
        # Talker decoder config (whose ``vocab_size`` is the codec vocab) under
        # ``text_config``, while Qwen3-TTS keeps a flat codec ``vocab_size``.
        random_token = torch.randint(0,
                                     talker_embed.num_embeddings, (bsz, 1),
                                     device=device)
        last_token_embed = talker_embed(random_token).to(talker.dtype)
        cp_inputs = torch.cat([talker_last_hidden, last_token_embed], dim=1)
        with torch.no_grad():
            cp.generate(inputs_embeds=cp_inputs,
                        max_new_tokens=cp_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict_in_generate=True)

        samples_done += bsz

    # Without at least one CP forward, every input_quantizer keeps its
    # un-initialised amax and FP8 scales degenerate — fail loudly here
    # rather than silently producing a broken engine.
    if samples_done == 0:
        raise RuntimeError(
            "CP calibration produced 0 samples. Dataloader exhausted or "
            "all batches were skipped (e.g. Thinker yielded fewer than "
            f"accept_layer={accept_layer} hidden layers).")


def _call_without_inference_mode(bound_method, *args, **kwargs):
    """Call *bound_method* bypassing a ``@torch.inference_mode()`` decorator.

    ModelOpt calibrators create/update amax state inside forward hooks;
    tensors created under inference mode poison later quantizer math
    ("Inference tensors cannot be used outside InferenceMode"). The HF
    decorator preserves the undecorated function as ``__wrapped__``.
    """
    fn = getattr(type(bound_method.__self__), bound_method.__name__)
    inner = getattr(fn, "__wrapped__", None)
    if inner is not None:
        with torch.no_grad():
            return inner(bound_method.__self__, *args, **kwargs)
    return bound_method(*args, **kwargs)


def _next_chat_input_ids(tokenizer, text, device):
    """Tokenize *text* as a single-turn ChatML user prompt.

    ``generate_talker`` locates ``im_start`` role tokens in ``input_ids``
    to build the Talker row plan, so raw text (no chat structure) breaks
    it. Prefer the checkpoint's chat template; fall back to literal ChatML.
    """
    try:
        ids = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": text
            }],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except (ValueError, TypeError, AttributeError):
        prompt = (f"<|im_start|>user\n{text}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids.to(device)


def qwen3_next_cp_calibration_loop(model,
                                   tokenizer,
                                   texts,
                                   num_cp_samples: int = 64,
                                   thinker_max_new_tokens: int = 64,
                                   talker_max_new_tokens: int = 16):
    """Qwen3-Omni Next (qwen3_omni_next) CP calibration via the reference
    generation path.

    The Next Talker invokes ``code_predictor.generate`` inside
    ``prepare_inputs_for_generation`` on every decode step, so driving the
    checkpoint's own ``generate`` → ``generate_talker`` chain fires the CP
    with exactly the production activation distribution (talker last-layer
    hidden + codec embedding through ``talker_projection``). Hand-building
    Talker inputs (the Omni loop's approach) would mean re-implementing the
    Next row plan (speaker codec rows, tts special tokens, chunked text) —
    the generation path already does all of it.

    Cost per sample is bounded by ``thinker_max_new_tokens`` (text reply)
    and ``talker_max_new_tokens`` (codec frames; each frame = 1 CP
    prefill + ``num_code_groups - 2`` CP decode steps).
    """
    device = next(model.parameters()).device
    speaker_map = getattr(model.config.talker_config, "speaker_id", None) or {}
    speaker = sorted(speaker_map)[0] if speaker_map else "f245"

    samples_done = 0
    errors = 0
    progress = tqdm(texts, desc="Calibrating Next+CP")
    for text in progress:
        if samples_done >= num_cp_samples:
            break
        if not text or not text.strip():
            continue
        input_ids = _next_chat_input_ids(tokenizer, text.strip()[:512], device)
        try:
            with torch.no_grad():
                thinker_result, talker_kwargs = model.generate(
                    input_ids=input_ids,
                    return_audio=True,
                    thinker_max_new_tokens=thinker_max_new_tokens,
                    talker_max_new_tokens=talker_max_new_tokens,
                    talker_do_sample=False,
                    subtalker_dosample=False,
                )
            _call_without_inference_mode(
                model.generate_talker,
                input_ids,
                thinker_result,
                talker_kwargs,
                speaker=speaker,
            )
        except Exception as error:
            # One bad sample (e.g. thinker reply too short for the talker
            # row plan) shouldn't kill the whole calibration; but log it —
            # a 100% failure rate is caught below.
            errors += 1
            logger.warning("Next CP calibration sample failed: %s", error)
            if errors >= 3 and samples_done == 0:
                raise
            continue
        samples_done += 1
        progress.set_postfix(done=samples_done)

    if samples_done == 0:
        raise RuntimeError(
            "Next CP calibration produced 0 samples — the CP quantizers "
            "hold uninitialised amax. Check the chat template / speaker "
            "config and the warnings above.")
