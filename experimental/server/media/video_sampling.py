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
#
# Adapted from third-party references (both Apache-2.0) so the sampled frame
# count / indices / fps match the per-model HF pipelines the C++ runner aligns to:
#   - `smart_nframes` + uniform `sample_indices` mirror the Qwen vision utility
#     (`qwen_vl_utils.vision_process`, Alibaba Cloud, Apache-2.0):
#     https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
#   - `sample_indices_internvl` mirrors HuggingFace `transformers`
#     `InternVLVideoProcessor.sample_frames` (initial-shift midpoint sampling,
#     HuggingFace Inc., Apache-2.0):
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/video_processing_internvl.py
"""Server-side video decode + frame sampling: resolves an mp4 /
``video_url`` / ``data:`` source into a native-resolution uint8
``[T, H, W, 3]`` array plus the effective sample fps (``T / duration``),
for ``load_video_from_array``. The C++ runner expects pre-sampled frames
and only resizes; the fps drives its temporal MRoPE positions.

Sampling mirrors ``qwen_vl_utils`` (links above). Deliberate deviations:

* ``max_frames`` below the frame factor wins over the factor floor (HF
  cannot express "at most 1 frame").
* pre-sampled ``frames`` lists default to ``fps=1.0`` when the request
  omits it.
* a positive container-reported frame count is trusted (single-pass
  decode); explicit ``nframes`` that cannot be met is rejected after
  decoding.
* single-frame sources stay videos (HF rejects sources below the factor).
* the reference is ``qwen_vl_utils``, not the transformers video loader;
  the two differ in factor rounding and default fps.
"""
from __future__ import annotations

import io
import math
import os
from typing import List, Optional, Tuple

from .media_source import (decode_base64_data_url, fetch_remote_media,
                           resolve_file_url)

# qwen_vl_utils defaults (kept in sync with the HF reference).
FRAME_FACTOR = 2  # frames rounded to a multiple of temporal_patch_size
DEFAULT_FPS = 2.0  # target sampling rate
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
# Hard ceiling on decoded video held at native resolution: true peak is
# ~2x the RGB bytes (assembly + load_video_from_array copies), so
# 256M px = ~1.5 GiB peak.
MAX_DECODE_PIXELS = 256 * 1024 * 1024
# Ceilings on the input itself (decode work, not just retained frames):
# uniform sampling picks the last frame, so nearly the whole source is
# sequentially decoded regardless of how few frames are kept.
MAX_SOURCE_BYTES = 256 * 1024 * 1024  # encoded payload (data: URLs)
MAX_SOURCE_FRAMES = 54000  # ~30 min @ 30 fps

# Nemotron-Omni video defaults (checkpoint video_io.py / configuration.py).
NEMOTRON_DEFAULT_FPS = 1.0
NEMOTRON_TEMPORAL_PATCH = 2  # T frames packed per tubelet


def _round_by_factor(n: float, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by_factor(n: float, factor: int) -> int:
    return math.ceil(n / factor) * factor


def _floor_by_factor(n: float, factor: int) -> int:
    return math.floor(n / factor) * factor


def smart_nframes(total_frames: int,
                  video_fps: float,
                  *,
                  target_fps: float = DEFAULT_FPS,
                  nframes: Optional[int] = None,
                  min_frames: int = FPS_MIN_FRAMES,
                  max_frames: Optional[int] = None,
                  frame_factor: int = FRAME_FACTOR) -> int:
    """Frames to sample, mirroring ``qwen_vl_utils.smart_nframes``: explicit
    ``nframes`` (rounded to ``frame_factor``), or ``fps`` bounded by
    ``[min_frames, max_frames]`` and ``total_frames``, floored to the factor."""
    if total_frames <= 0 or video_fps <= 0:
        raise ValueError(
            f"invalid video: total_frames={total_frames}, fps={video_fps}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    if nframes is not None:
        # Explicit nframes is exact-or-reject (qwen_vl_utils contract): the
        # rounded count must exist in the source; nothing clamps it silently.
        if nframes > FPS_MAX_FRAMES:
            raise ValueError(
                f"nframes={nframes} exceeds the supported maximum "
                f"{FPS_MAX_FRAMES}")
        out = _round_by_factor(nframes, frame_factor)
        if not frame_factor <= out <= total_frames:
            raise ValueError(
                f"nframes should be in the interval [{frame_factor}, "
                f"{total_frames}], got {nframes}")
        return int(out)
    lo = _ceil_by_factor(min_frames, frame_factor)
    # FPS_MAX_FRAMES is a hard decode/memory ceiling; max_frames may only
    # lower it, never raise it.
    hi_cap = min(FPS_MAX_FRAMES, total_frames) if max_frames is None \
        else min(max_frames, FPS_MAX_FRAMES, total_frames)
    hi = _floor_by_factor(hi_cap, frame_factor)
    est = total_frames / video_fps * target_fps
    out = _floor_by_factor(min(max(est, lo), hi), frame_factor)
    out = max(frame_factor,
              min(out, _floor_by_factor(total_frames, frame_factor)))
    # A user max_frames below frame_factor wins over the factor floor.
    if max_frames is not None:
        out = min(out, max_frames)
    return int(min(out, FPS_MAX_FRAMES))


def sample_indices(total_frames: int, nframes: int) -> List[int]:
    """Uniformly-spaced ``nframes`` indices in ``[0, total_frames-1]`` (HF
    uses ``linspace(...).round()``)."""
    if nframes <= 0 or total_frames <= 0:
        return []
    if nframes == 1:
        return [0]
    step = (total_frames - 1) / (nframes - 1)
    return [
        min(total_frames - 1, int(round(i * step))) for i in range(nframes)
    ]


def sample_indices_internvl(total_frames: int,
                            nframes: int,
                            initial_shift=True) -> List[int]:
    """InternVL frame indices, mirroring ``InternVLVideoProcessor.sample_frames``:
    uniform with an initial shift (segment midpoints), unlike Qwen's
    endpoint-aligned linspace."""
    if nframes <= 0 or total_frames <= 0:
        return []
    # The HF rejection for nframes > total lives in internvl_nframes; this
    # clamp only guards direct callers against duplicate indices.
    nframes = min(nframes, total_frames)
    if initial_shift is True:
        initial_shift = total_frames / nframes / 2
    step = total_frames / nframes
    out: List[int] = []
    v = float(initial_shift)
    while v < total_frames and len(out) < nframes:
        out.append(int(v))
        v += step
    return out


def sample_indices_nemotron(total_frames: int,
                            video_fps: float,
                            *,
                            target_fps: Optional[float] = None,
                            nframes: Optional[int] = None,
                            max_frames: Optional[int] = None) -> List[int]:
    """Nemotron-Omni frame indices, mirroring the checkpoint's
    ``video_io.sample_video_frames_to_data_urls``: sample ``int(duration * fps)``
    frames (``fps`` defaults to ``NEMOTRON_DEFAULT_FPS``; explicit ``nframes``
    overrides the fps rule), uniformly via ``linspace`` + dedup."""
    import numpy as np
    if total_frames <= 0:
        return []
    if nframes is not None:
        desired = int(nframes)
    else:
        fps = target_fps if target_fps is not None else NEMOTRON_DEFAULT_FPS
        duration = total_frames / video_fps if video_fps > 0 else 0.0
        desired = int(duration * fps)
    if max_frames is not None:
        desired = min(desired, int(max_frames))
    if desired >= total_frames:
        return list(range(total_frames))
    if desired <= 1:
        return [0]
    return list(
        np.unique(
            np.round(np.linspace(0, total_frames - 1, desired)).astype(int)))


def internvl_nframes(total_frames: int,
                     video_fps: float,
                     *,
                     target_fps: Optional[float] = None,
                     nframes: Optional[int] = None,
                     max_frames: Optional[int] = None) -> int:
    """InternVL frame count — HF semantics: explicit ``num_frames`` wins, else
    ``int(total / video_fps * fps)``. The HF class ships no default count, so
    absent both we fall back to the server-wide DEFAULT_FPS target."""
    if nframes is not None:
        # Exact-or-reject (HF sample_frames raises when num_frames exceeds
        # the source).
        n = int(nframes)
        if n > FPS_MAX_FRAMES:
            raise ValueError(f"nframes={n} exceeds the supported maximum "
                             f"{FPS_MAX_FRAMES}")
        if n > total_frames:
            raise ValueError(f"nframes={n} exceeds the video's "
                             f"{total_frames} frames")
        return max(1, n)
    fps = target_fps if target_fps is not None else DEFAULT_FPS
    n = int(total_frames / video_fps * fps) if video_fps > 0 else 1
    if max_frames is not None:
        n = min(n, int(max_frames))
    return max(1, min(n, total_frames, FPS_MAX_FRAMES))


def effective_fps(nframes: int, total_frames: int, video_fps: float) -> float:
    """Sample fps = sampled frames / duration. Drives the runner's MRoPE
    ``second_per_grid``."""
    if total_frames <= 0 or video_fps <= 0:
        return float(DEFAULT_FPS)
    duration = total_frames / video_fps
    return nframes / duration if duration > 0 else float(DEFAULT_FPS)


def _frame_timestamps(family: str, idxs, video_fps: float) -> List[float]:
    """Per-frame label timestamps in seconds. Nemotron mirrors vLLM's
    ``idx * floor(1000/fps)`` ms quantization; other families use ``idx/fps``."""
    if family == "nemotron":
        per_ms = int(1000.0 / video_fps) if video_fps > 0 else 0
        return [int(i) * per_ms / 1000.0 for i in idxs]
    return [i / video_fps for i in idxs]


def _describe_source(source: str) -> str:
    """Bounded description of a video source for error messages: a data URL
    can be hundreds of MB and must never be echoed back."""
    if not source:
        return "<bytes>"
    if source.startswith("data:") or len(source) > 256:
        return f"<data url, {len(source)} chars>"
    return repr(source)


def _pixel_cap(pixel_budget) -> int:
    """Effective decode pixel ceiling: the hard host cap, lowered by the
    request's remaining pixel budget (0 means exhausted, not unlimited)."""
    if pixel_budget is None:
        return MAX_DECODE_PIXELS
    return min(MAX_DECODE_PIXELS, pixel_budget)


def _raw_video_frame_tokens(family: str, count: int, width: int, height: int,
                            limits: dict) -> int:
    """Visual tokens of a do_resize=false video: InternVL frames must be exactly
    one 448 block (C++ derives frames as tokens/256); Qwen frames must be
    factor-aligned, accounted by input size. Nemotron always resizes, so it
    never reaches this path."""
    if family == "internvl":
        if width != 448 or height != 448:
            raise ValueError("do_resize=false InternVL video frames must be "
                             f"448x448, got {width}x{height}")
        return count * 256
    factor = (limits.get("patch_size", 0) or 1) * \
        (limits.get("merge_size", 0) or 1)
    if factor <= 1:
        return 0
    if width % factor or height % factor:
        raise ValueError(
            f"do_resize=false video frames must be {factor}-aligned, "
            f"got {width}x{height}")
    tps = max(1, limits.get("temporal_patch_size", 2))
    t_bar = -(-count // tps) * tps
    # Each temporal group is one entry of the C++ TRT carrier, so its
    # spatial tokens must fit the engine's per-image profile even when the
    # request total does.
    group_tokens = height * width // (factor * factor)
    per_image = limits.get("max_image_tokens_per_image", 0)
    if per_image and group_tokens > per_image:
        raise ValueError(
            f"do_resize=false video frames yield {group_tokens} tokens per "
            f"temporal group, over the engine's per-image capacity "
            f"{per_image}; lower the resolution")
    return max(1, t_bar // tps * group_tokens)


def _check_cu_budget(frames: int, family: str, limits: dict,
                     cu_budget) -> None:
    """Reject when this media's cu_seqlens entries (one per Qwen temporal group)
    exceed the remaining request budget. Applies to every Qwen intake incl.
    do_resize=false; InternVL/Nemotron have no cu_seqlens binding and are
    exempt."""
    if cu_budget is None or family in ("internvl", "nemotron"):
        return
    tps = max(1, (limits or {}).get("temporal_patch_size", 2))
    groups = -(-frames // tps)
    if groups > cu_budget:
        raise ValueError(
            f"video needs {groups} cu_seqlens entries but only "
            f"{max(0, cu_budget)} remain in the visual engine's capacity; "
            "reduce the media in the request")


def _check_aspect_ratio(width: int, height: int) -> None:
    """Mirror the C++ maxRatio=200 rejection so extreme aspect ratios fail
    with a 400 before decode/request construction, not a runtime 500."""
    if width > 0 and height > 0 and \
            max(width, height) / min(width, height) > 200:
        raise ValueError(
            f"absolute aspect ratio of {width}x{height} exceeds the "
            "supported maximum of 200")


def _estimate_qwen2d_frame_tokens(width: int, height: int,
                                  limits: dict) -> int:
    """Per-frame visual tokens after the C++ 2D smart_resize (HF parity:
    round-to-factor, then min-pixels upscale or max-pixels downscale)."""
    patch = limits.get("patch_size", 0)
    merge = limits.get("merge_size", 0)
    if patch <= 0 or merge <= 0 or width <= 0 or height <= 0:
        return 0
    factor = patch * merge
    # C++ reuses the global min_image_tokens as the per-image minimum.
    min_px = limits.get("min_image_tokens", 0) * factor * factor
    max_px = limits.get("max_image_tokens_per_image", 0) * factor * factor
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if max_px > 0 and h_bar * w_bar > max_px:
        beta = math.sqrt((height * width) / max_px)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif min_px > 0 and h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return max(1, (h_bar * w_bar) // (factor * factor))


def _estimate_internvl_image_tokens(width: int, height: int,
                                    limits: dict) -> int:
    """Tokens the C++ InternVL image path produces: mirrors imageUtils
    ``computeBestBlockGridForResize`` plus the runner's thumbnail rule (a
    thumbnail block when the grid or the engine minimum exceeds one block)."""
    per_image = limits.get("max_image_tokens_per_image", 0)
    min_per_image = limits.get("min_image_tokens", 0)
    min_tiles = max(1, min_per_image // 256 - 1)
    max_tiles = max(1, per_image // 256 - 1)
    grids = [(cols, rows) for cols in range(1, max_tiles + 1)
             for rows in range(1, max_tiles + 1)
             if min_tiles <= cols * rows <= max_tiles]
    grids.sort(key=lambda g: g[0] * g[1])
    aspect = width / height
    area = width * height
    best, best_diff = (1, 1), float("inf")
    for cols, rows in grids:
        diff = abs(aspect - cols / rows)
        if diff < best_diff:
            best_diff, best = diff, (cols, rows)
        elif diff == best_diff and area > (448 * 448 // 2) * cols * rows:
            best = (cols, rows)
    blocks = best[0] * best[1]
    if blocks > 1 or min_per_image // 256 > 1:
        blocks += 1  # thumbnail block
    return blocks * 256


def estimate_image_tokens(path: str,
                          family: str,
                          limits: dict,
                          do_resize: bool = True) -> int:
    """Visual tokens a single image will consume after the C++ resize, for
    request-level budget accounting. Falls back to the per-media cap when the
    image cannot be probed."""
    if not limits:
        return 0
    per_image = limits.get("max_image_tokens_per_image", 0)
    try:
        width, height = _probe_image_size(path)
    except ValueError:
        return per_image
    if width <= 0 or height <= 0:
        return per_image
    if family == "internvl":
        if not do_resize:
            # Raw input skips the C++ grid resize: each 448x448 tile is one
            # block, so the dimensions must be tile-aligned.
            if width % 448 or height % 448:
                raise ValueError(
                    "do_resize=false InternVL images must be 448-aligned, "
                    f"got {width}x{height}")
            blocks = (width // 448) * (height // 448)
            # C++ appends a thumbnail block when the main image spans more
            # than one block or the engine minimum requires it
            # (internViTRunner formatPatch).
            min_blocks = max(1, limits.get("min_image_tokens", 0) // 256)
            if blocks > 1 or min_blocks > 1:
                blocks += 1
            return blocks * 256
        return _estimate_internvl_image_tokens(width, height, limits)
    if not do_resize:
        # Raw input skips the C++ smart resize: tokens follow the input
        # dimensions, which must be factor-aligned.
        factor = (limits.get("patch_size", 0) or 1) * \
            (limits.get("merge_size", 0) or 1)
        if factor <= 1:
            return per_image  # no patch geometry available; reserve the cap
        if width % factor or height % factor:
            raise ValueError(
                f"do_resize=false images must be {factor}-aligned, "
                f"got {width}x{height}")
        tokens = max(1, width * height // (factor * factor))
        # A raw image is one TRT carrier entry: it must fit the per-image
        # profile on its own, not just the request total.
        if per_image and tokens > per_image:
            raise ValueError(
                f"do_resize=false image yields {tokens} visual tokens, over "
                f"the engine's per-image capacity {per_image}; lower the "
                "resolution")
        return tokens
    _check_aspect_ratio(width, height)
    model_type = limits.get("model_type", "")
    if "qwen3_vl" in model_type or "qwen3_5" in model_type:
        # Still image on a 3D family: the C++ resize routes stills through
        # qwenSmartResize3D with isVideo=false (temporal factor 1), which
        # includes the factor-grid fallback the 2D estimate lacks.
        return _estimate_qwen3d_video_tokens(1,
                                             width,
                                             height,
                                             limits,
                                             is_video=False)
    return _estimate_qwen2d_frame_tokens(width, height, limits)


def _estimate_qwen3d_video_tokens(nframes: int,
                                  width: int,
                                  height: int,
                                  limits: dict,
                                  is_video: bool = True) -> int:
    """Visual tokens after the Qwen3-VL 3D smart resize (temporal budget
    folded in; ``is_video=False`` is the still-image form with a temporal
    factor of 1), so budget accounting uses the tokens the C++ resize
    actually produces rather than the per-media cap."""
    patch = limits.get("patch_size", 0)
    merge = limits.get("merge_size", 0)
    if patch <= 0 or merge <= 0 or width <= 0 or height <= 0 or nframes <= 0:
        return 0
    factor = patch * merge
    tps = max(1, limits.get("temporal_patch_size", 2))
    temporal_factor = tps if is_video else 1
    min_px = limits.get("min_image_tokens", 0) * temporal_factor * factor \
        * factor
    max_px = limits.get("max_image_tokens_per_image", 0) * temporal_factor \
        * factor * factor
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    t_bar = _ceil_by_factor(nframes, tps) if is_video else 1
    budget = t_bar * h_bar * w_bar
    if max_px > 0 and budget > max_px:
        beta = math.sqrt((nframes * height * width) / max_px)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif min_px > 0 and budget < min_px:
        beta = math.sqrt(min_px / (nframes * height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    if max_px > 0 and t_bar * h_bar * w_bar > max_px:
        beta = math.sqrt((t_bar * height * width) / max_px)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    # Factor-grid fallback (mirrors the C++ resize): quantization can land
    # outside [min_px, max_px]; pick the aspect-closest feasible grid shape.
    if (max_px > 0 and t_bar * h_bar * w_bar > max_px) or \
            (min_px > 0 and t_bar * h_bar * w_bar < min_px):
        unit = t_bar * factor * factor
        pq_min = max(1, -(-min_px // unit)) if min_px > 0 else 1
        pq_max = max_px // unit if max_px > 0 else 0
        if max_px > 0 and (pq_max < 1 or pq_min > pq_max):
            # Mirrors the C++ resize: no factor-grid shape fits the profile.
            raise ValueError(
                f"no resized visual shape fits the engine profile "
                f"(frames={nframes}, {height}x{width}); reduce the frame "
                "count/resolution or rebuild the visual engine with wider "
                "--minImageTokens/--maxImageTokens bounds")
        if pq_max >= 1 and pq_min <= pq_max:
            target = height / width
            pq_target = pq_min if t_bar * h_bar * w_bar < min_px else pq_max
            best = None
            for pp in range(1, pq_max + 1):
                q_lo = max(1, -(-pq_min // pp))
                q_hi = pq_max // pp
                if q_lo > q_hi:
                    continue
                qq = min(max(round(pp / target), q_lo), q_hi)
                dist = abs(math.log((pp / qq) / target))
                gap = abs(pp * qq - pq_target)
                if best is None or dist < best[0] - 1e-12 or (
                        abs(dist - best[0]) <= 1e-12 and gap < best[1]):
                    best = (dist, gap, pp, qq)
            h_bar, w_bar = best[2] * factor, best[3] * factor
    return max(1,
               (t_bar * h_bar * w_bar) // (temporal_factor * factor * factor))


def _nemotron_tubelet_geometry(limits: dict):
    """Nemotron-Omni per-tubelet geometry from engine limits: ``(T frames per
    tubelet, tokens per tubelet, EVS keep rate q)``, or ``None`` when the limits
    lack the video fields. Tokens per tubelet = ``video_target_num_patches /
    downsample^2``."""
    t = max(1, limits.get("video_temporal_patch_size",
                          NEMOTRON_TEMPORAL_PATCH))
    target = limits.get("video_target_num_patches", 0)
    ratio = limits.get("downsample_ratio", 0)
    if target <= 0 or ratio <= 0:
        return None
    scale = max(1, int(round(1.0 / ratio)))
    tokens_per_tubelet = target // (scale * scale)
    if tokens_per_tubelet <= 0:
        return None
    q = limits.get("video_pruning_rate", 0.0) or 0.0
    return t, tokens_per_tubelet, q


def _estimate_nemotron_video_tokens(nframes: int, limits: dict) -> int:
    """Coarse estimate of a Nemotron-Omni video's EVS-pruned visual tokens:
    ceil(frames / T) tubelets, each ``video_target_num_patches / downsample^2``
    tokens, scaled by the ``(1 - video_pruning_rate)`` EVS keep fraction."""
    geom = _nemotron_tubelet_geometry(limits)
    if geom is None or nframes <= 0:
        return 0
    t, tokens_per_tubelet, q = geom
    tubelets = -(-nframes // t)
    # EVS keeps at least one full tubelet's worth of tokens (matches the C++ numKeep floor).
    return max(tokens_per_tubelet,
               int(tubelets * tokens_per_tubelet * (1.0 - q)))


def clamp_nframes_to_profile(
        nframes: int,
        family: str,
        width: int,
        height: int,
        limits: dict,
        budget: Optional[int] = None,
        cu_budget: Optional[int] = None) -> Tuple[int, int]:
    """Clamp the sampled frame count to the visual engine's token profile and
    estimate the tokens consumed. ``budget`` is the request's remaining
    token budget (all media share one profile); defaults to the engine
    total. Returns ``(nframes, estimated_tokens)``, or ``(nframes, 0)``
    when ``limits`` is empty."""
    if family not in ("internvl", "nemotron"):
        # Only the Qwen smart resize enforces maxRatio=200 in C++; the InternVL
        # grid resize and the aspect-preserving Nemotron resize accept any ratio.
        _check_aspect_ratio(width, height)
    if not limits:
        return nframes, 0
    max_total = limits.get("max_image_tokens", 0)
    if max_total <= 0:
        return nframes, 0
    cap = min(budget, max_total) if budget is not None else max_total
    if cap <= 0:
        raise ValueError(
            "the request's other media already consume the engine's visual "
            f"token budget ({max_total}); no room left for this video")
    if family == "internvl":
        # kBlockLength: an InternVL ViT block is 448x448 = 256 tokens. The
        # engine minimum is a request-wide bound (all media accumulate), so
        # it is checked by the caller after all buffers are loaded.
        block_tokens = 256
        max_blocks = max(1, cap // block_tokens)
        n = min(nframes, max_blocks)
        return n, n * block_tokens
    if family == "nemotron":
        # Aspect-preserving frames pack T per tubelet; each tubelet is
        # video_target_num_patches/downsample^2 tokens, EVS-pruned to
        # (1 - video_pruning_rate). Clamp frames so the estimate fits the cap.
        geom = _nemotron_tubelet_geometry(limits)
        if geom is None:
            return nframes, 0
        t, tokens_per_tubelet, q = geom
        # The engine processes every tubelet pre-EVS, so the tubelet count is bounded by the block
        # profile (max_image_tokens / tokens_per_tubelet), independent of the post-EVS keep rate.
        engine_tubelets = max(1, max_total // tokens_per_tubelet)
        kept = max(1, int(tokens_per_tubelet * (1.0 - q)))
        max_tubelets = min(engine_tubelets, max(1, cap // kept))
        n = min(nframes, max_tubelets * t)
        return n, _estimate_nemotron_video_tokens(n, limits)
    model_type = limits.get("model_type", "")
    per_image = limits.get("max_image_tokens_per_image", 0)
    if "qwen3_vl" in model_type or "qwen3_5" in model_type:
        # The 3D resize fits the whole video per media, but each temporal group
        # needs >= 1 token, so frames are bounded by temporalPatchSize * budget;
        # charge the tokens the resize actually produces.
        tps = max(1, limits.get("temporal_patch_size", 2))
        media_cap = min(per_image, cap) if per_image > 0 else cap
        # The cu_seqlens profile holds one entry per temporal group, shared
        # request-wide: consume the remaining group budget. Prefer the
        # builder-recorded capacity; fall back to the pre-recording formula.
        cu_groups = cu_budget if cu_budget is not None else (
            limits.get("max_cu_seqlen_groups")
            or max_total // max(1, limits.get("min_image_tokens", 1)))
        if cu_groups <= 0:
            raise ValueError(
                "the request's other media already consume the visual "
                "engine's cu_seqlens capacity; reduce the media count")
        n = min(nframes, tps * media_cap, tps * cu_groups)
        est = _estimate_qwen3d_video_tokens(n, width, height, limits)
        if est > cap:
            raise ValueError(
                f"video needs ~{est} visual tokens but only {cap} remain in "
                "the engine budget; reduce other media in the request")
        return n, est
    frame_tokens = _estimate_qwen2d_frame_tokens(width, height, limits)
    if frame_tokens <= 0:
        return nframes, 0
    tps = max(1, limits.get("temporal_patch_size", 2))
    max_groups = cap // frame_tokens
    # Each temporal group is one cu_seqlens entry; clamp gracefully to the
    # shared group budget (like the 3D branch) instead of erroring downstream.
    if cu_budget is not None:
        max_groups = min(max_groups, cu_budget)
    if max_groups < 1:
        raise ValueError(
            "video frames need "
            f"{frame_tokens} visual tokens each but only {cap} remain in "
            "the engine budget; reduce other media in the request or "
            "rebuild the visual engine with a larger --maxImageTokens")
    n = min(nframes, max_groups * tps)
    groups = (n + tps - 1) // tps
    return n, groups * frame_tokens


def resolve_video_source(url: str) -> Tuple[str, Optional[bytes]]:
    """Resolve an OpenAI video reference: ``(path, None)`` for ``file://`` /
    bare paths, ``("", bytes)`` for HTTP(S) and ``data:`` URLs."""
    url = (url or "").strip()
    if not url:
        raise ValueError("video content has empty url")
    if url.startswith(("http://", "https://")):
        return "", fetch_remote_media(url, "video", MAX_SOURCE_BYTES)
    if url.startswith("data:"):
        return "", decode_base64_data_url(url,
                                          "video",
                                          strict=True,
                                          max_bytes=MAX_SOURCE_BYTES)
    if url.startswith("file:"):
        url = resolve_file_url(url)
    if not os.path.isfile(url):
        raise ValueError(f"video file not found: {url}")
    return url, None


def sample_video(source: str,
                 *,
                 target_fps: float = DEFAULT_FPS,
                 nframes: Optional[int] = None,
                 min_frames: int = FPS_MIN_FRAMES,
                 max_frames: Optional[int] = None,
                 family: str = "qwen",
                 frame_limits: Optional[dict] = None,
                 budget: Optional[int] = None,
                 do_resize: bool = True,
                 pixel_budget: Optional[int] = None,
                 cu_budget: Optional[int] = None):
    """Decode + sample a video: returns ``(frames[T,H,W,3] uint8, fps,
    timestamps, est_tokens, decoded_px)``. Frames stay at native resolution
    (the C++ runner resizes); ``timestamps`` are the sampled frames' source
    times (original_index / source_fps); ``fps`` is the effective sample fps."""
    if target_fps <= 0:
        raise ValueError(f"fps must be positive, got {target_fps}")
    if nframes is not None and nframes <= 0:
        raise ValueError(f"nframes must be positive, got {nframes}")
    if min_frames <= 0:
        raise ValueError(f"min_frames must be positive, got {min_frames}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    # Resolve (and validate) the source before touching the optional decode
    # dependency, so a rejected URL is a ValueError even when av is absent.
    path, data = resolve_video_source(source)

    import av  # lazy: decode dependency
    import numpy as np

    try:
        container = av.open(path if path else io.BytesIO(data))
    except av.error.FFmpegError as exc:  # corrupt container -> client error
        raise ValueError(f"failed to open video source: {exc}") from exc
    try:
        if not container.streams.video:
            raise ValueError(
                f"no video stream in input {_describe_source(source)} "
                "(audio-only or "
                "non-video container)")
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate) if stream.average_rate else 0.0
        # A positive reported count is trusted as-is: single-pass sampling, a
        # deliberate trade-off vs HF's decode-everything (see module docstring).
        total = stream.frames or 0
        if total <= 0:  # some containers don't report; count by decoding
            try:
                total = 0
                for _ in container.decode(stream):
                    total += 1
                    if total > MAX_SOURCE_FRAMES:
                        raise ValueError(
                            "video exceeds the supported maximum of "
                            f"{MAX_SOURCE_FRAMES} source frames")
            except av.error.FFmpegError as exc:
                raise ValueError(f"failed to decode video: {exc}") from exc
            container.seek(0)
        if total > MAX_SOURCE_FRAMES:
            raise ValueError(
                f"video reports {total} frames, over the supported maximum "
                f"{MAX_SOURCE_FRAMES}; trim the clip")
        if video_fps <= 0:
            video_fps = DEFAULT_FPS

        def _raw_video_tokens(count, width, height):
            # do_resize=false skips the C++ resize entirely, so the sampling
            # plan is never resize-clamped; the raw-size tokens must simply
            # fit the remaining budget.
            est = _raw_video_frame_tokens(family, count, width, height,
                                          frame_limits or {})
            max_total = (frame_limits or {}).get("max_image_tokens", 0)
            cap = None
            if budget is not None:
                cap = min(budget, max_total) if max_total else budget
            elif max_total:
                cap = max_total
            if cap is not None and est > cap:
                raise ValueError(
                    f"pre-resized video needs ~{est} visual tokens but only "
                    f"{cap} remain in the engine budget; reduce frames or "
                    "other media in the request")
            return est

        if family == "internvl":
            n = internvl_nframes(total,
                                 video_fps,
                                 target_fps=target_fps,
                                 nframes=nframes,
                                 max_frames=max_frames)
        elif family == "nemotron":
            n = len(
                sample_indices_nemotron(total,
                                        video_fps,
                                        target_fps=target_fps,
                                        nframes=nframes,
                                        max_frames=max_frames))
        else:
            n = smart_nframes(total,
                              video_fps,
                              target_fps=target_fps,
                              nframes=nframes,
                              min_frames=min_frames,
                              max_frames=max_frames)
        if do_resize:
            planned = n
            n, est_tokens = clamp_nframes_to_profile(n, family, stream.width
                                                     or 0, stream.height or 0,
                                                     frame_limits or {},
                                                     budget, cu_budget)
            if nframes is not None and n < planned:
                # Explicit nframes is exact-or-reject: the profile clamp only
                # auto-shrinks fps-derived counts.
                raise ValueError(
                    f"nframes={nframes} needs more visual tokens than the "
                    "engine profile allows; lower nframes or rebuild the "
                    "visual engine with a larger --maxImageTokens")
        else:
            est_tokens = _raw_video_tokens(n, stream.width or 0, stream.height
                                           or 0)
        # Planned-count group check before any decode work (a request with
        # exhausted capacity must not scan the whole source first); the
        # decoded count can only be <= n, so no post-decode recheck needed.
        _check_cu_budget(n, family, frame_limits or {}, cu_budget)
        if family == "internvl":
            wanted = set(sample_indices_internvl(total, n))
        elif family == "nemotron":
            wanted = set(sample_indices_nemotron(total, video_fps, nframes=n))
        else:
            wanted = set(sample_indices(total, n))
        # Charge planned decode work by the distinct sampled positions (a
        # single-frame video plans n=2 but decodes one frame).
        decode_px = len(wanted) * (stream.width or 0) * (stream.height or 0)
        px_cap = _pixel_cap(pixel_budget)
        if decode_px > px_cap:
            raise ValueError(
                f"decoding {n} frames of {stream.width}x{stream.height} "
                f"needs {decode_px} pixels, over the request's remaining "
                f"host limit {px_cap}; lower nframes/fps, the resolution, "
                "or the number of videos")
        picked: dict = {}
        retained_px = 0
        try:
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    arr = frame.to_ndarray(format="rgb24")
                    # Accumulate actual pixels as frames are retained, so a
                    # source whose metadata under-reports its size cannot
                    # blow past the cap before the post-decode check.
                    retained_px += arr.shape[0] * arr.shape[1]
                    if retained_px > px_cap:
                        raise ValueError(
                            f"decoded frames exceed the request's remaining "
                            f"host pixel limit {px_cap}")
                    picked[i] = arr
                    if len(picked) == len(wanted):
                        break
        except av.error.FFmpegError as exc:  # mid-stream corruption
            raise ValueError(f"failed to decode video: {exc}") from exc
    finally:
        container.close()

    idxs = sorted(picked)
    if not idxs:
        raise ValueError("decoded no frames from video source: "
                         f"{_describe_source(source)}")
    if len(idxs) != len(wanted):
        # Over-reporting containers yield fewer frames than sampled positions:
        # trusting them would skew the effective fps and under-deliver explicit
        # nframes. Checked before the assembly allocation.
        raise ValueError(
            f"the container reports {total} frames but only {len(idxs)} of "
            f"the {len(wanted)} sampled positions decoded; re-encode the "
            "video or pass frames explicitly")
    # Preallocate and free per-frame arrays as they are copied (transient
    # ~2x peak at the allocation); the in-loop accumulation already capped
    # the total decoded pixels.
    fh, fw = picked[idxs[0]].shape[:2]
    for i in idxs:
        if picked[i].shape[:2] != (fh, fw):
            raise ValueError(
                f"video decodes frames of differing sizes "
                f"({fw}x{fh} vs {picked[i].shape[1]}x{picked[i].shape[0]}); "
                "mid-stream resolution changes are not supported")
    stacked = np.empty((len(idxs), fh, fw, 3), dtype=np.uint8)
    for pos, i in enumerate(idxs):
        stacked[pos] = picked[i]
        picked[i] = None
    if len(idxs) != n or fw != (stream.width or 0) or fh != (stream.height
                                                             or 0):
        # Short clips and rotation/filter chains decode a different count/size
        # than the plan's metadata; re-estimate so token accounting matches what
        # the C++ preprocessor really sees.
        if do_resize:
            kept, est_tokens = clamp_nframes_to_profile(
                len(idxs), family, fw, fh, frame_limits or {}, budget,
                cu_budget)
            if kept < len(idxs):
                raise ValueError(
                    f"video decodes to {len(idxs)} frames of {fw}x{fh}, "
                    "over the engine's remaining visual token budget; lower "
                    "nframes/fps or the resolution")
        else:
            est_tokens = _raw_video_tokens(len(idxs), fw, fh)
    timestamps = _frame_timestamps(family, idxs, video_fps)
    # Charge by the actual decoded frame size (rotation/filter chains can
    # differ from the container metadata used for the pre-decode check).
    decoded_px = len(idxs) * fh * fw
    return (stacked, effective_fps(len(idxs), total, video_fps), timestamps,
            est_tokens, decoded_px)


# --- server content -> ImageData ------------------------------------------


def _extract_video_source(item: dict) -> str:
    """Pull the clip url/path from a content item: accepts ``{"type": "video",
    "video": url}`` and the ``{"type": "video_url", "video_url": {"url": ...}}``
    spelling (mirrors ``image_url`` / ``audio_url``)."""
    if item.get("type") == "video_url":
        ref = item.get("video_url") or {}
    else:  # "video"
        ref = item.get("video")
    url = (ref.get("url") if isinstance(ref, dict) else ref) or ""
    if not isinstance(url, str):
        raise ValueError(
            f"video url must be a string, got {type(url).__name__}")
    url = url.strip()
    if not url:
        raise ValueError("video content needs a url/path "
                         '("video": "..." or "video_url": {"url": "..."})')
    return url


def _positive_float(value, name: str) -> float:
    """Coerce a request parameter to a positive float; type errors and
    non-positive values both map to ValueError (HTTP 400)."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, "
                         f"got {value!r}")
    return out


def _probe_image_size(path: str) -> Tuple[int, int]:
    """(width, height) of an image file via a header-only av probe."""
    import av  # lazy: decode dependency
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            return int(stream.width or 0), int(stream.height or 0)
    except (av.error.FFmpegError, IndexError) as exc:
        raise ValueError(
            f"cannot read video frame image {path!r}: {exc}") from exc


def load_video_buffer(rt_module,
                      item: dict,
                      family: str = "qwen",
                      frame_limits: Optional[dict] = None,
                      budget: Optional[int] = None,
                      pixel_budget: Optional[int] = None,
                      cu_budget: Optional[int] = None):
    """Build a video ``ImageData`` (``isVideo``) from a content item. Two source
    forms: pre-sampled frame paths ``{"frames": [...], "fps": f}`` (no
    decode; CLI ``requestFileParser`` schema) via ``load_video_from_paths``,
    or a clip reference (``video`` / ``video_url``) decoded + sampled here.
    Returns ``(buffer, est_tokens, used_px, used_groups)``."""
    do_resize = bool(item.get("do_resize", True))
    if family == "nemotron" and not do_resize:
        # The Nemotron-Omni runner always smart-resizes to the target grid, so
        # do_resize=false cannot be honored; reject rather than silently resize.
        raise ValueError(
            "Nemotron-Omni video does not support do_resize=false; the runner "
            "always resizes frames to the target patch grid")
    frame_paths = item.get("frames")
    if frame_paths is not None:
        if (not isinstance(frame_paths, (list, tuple)) or not frame_paths
                or not all(isinstance(f, str) for f in frame_paths)):
            raise ValueError(
                "video 'frames' must be a non-empty list of file paths")
        missing = [f for f in frame_paths if not os.path.isfile(f)]
        if missing:
            raise ValueError(f"video frame file(s) not found: {missing[:3]}")
        if len(frame_paths) > FPS_MAX_FRAMES:
            raise ValueError(
                f"{len(frame_paths)} pre-sampled frames exceed the supported "
                f"maximum {FPS_MAX_FRAMES}")
        # Probe every frame: the pixel ceiling counts actual sizes (a small
        # first frame cannot exempt followers); mismatches fail as client
        # errors, not deep in the C++ loader.
        sizes = [_probe_image_size(p) for p in frame_paths]
        width, height = sizes[0]
        for p, size in zip(frame_paths, sizes):
            if size != (width, height):
                raise ValueError(
                    "video frames must share one size; frame 0 is "
                    f"{width}x{height} but {p!r} is {size[0]}x{size[1]}")
        if do_resize and family not in ("internvl", "nemotron"):
            # The C++ smart resize enforces maxRatio=200 regardless of
            # whether profile limits are available for estimation.
            _check_aspect_ratio(width, height)
        frames_px = sum(w * h for w, h in sizes)
        px_cap = _pixel_cap(pixel_budget)
        if frames_px > px_cap:
            raise ValueError(
                f"{len(frame_paths)} frames of {width}x{height} exceed the "
                f"request's remaining host pixel limit {px_cap}; lower the "
                "frame count, resolution, or number of videos")
        fps_val = _positive_float(item.get("fps", 1.0), "fps")
        limits = frame_limits or {}
        est = 0
        cap = None
        if limits.get("max_image_tokens"):
            cap = (min(budget, limits["max_image_tokens"])
                   if budget is not None else limits["max_image_tokens"])
            if cap <= 0:
                raise ValueError(
                    "the request's other media already consume the engine's "
                    "visual token budget; no room left for these frames")
        if not do_resize:
            # Pre-resized frames skip the C++ resize: validate alignment and
            # account by input size; no resize-based frame clamping applies.
            est = _raw_video_frame_tokens(family, len(frame_paths), width,
                                          height, limits)
            _check_cu_budget(len(frame_paths), family, limits, cu_budget)
            if cap is not None and est > cap:
                raise ValueError(
                    f"pre-resized video needs ~{est} visual tokens but only "
                    f"{cap} remain in the engine budget; reduce frames or "
                    "other media in the request")
        elif cap is not None:
            if family == "nemotron":
                # Aspect-preserving tubelet estimate; no cu_seqlens binding.
                geom = _nemotron_tubelet_geometry(limits)
                if geom is not None:
                    temporal, tokens_per_tubelet, _ = geom
                    tubelets = -(-len(frame_paths) // temporal)
                    engine_tubelets = max(
                        1, limits["max_image_tokens"] // tokens_per_tubelet)
                    if tubelets > engine_tubelets:
                        raise ValueError(
                            f"{len(frame_paths)} pre-sampled frames need "
                            f"{tubelets} tubelets but the visual engine "
                            f"profile holds {engine_tubelets}; reduce the "
                            "frame count")
                est = _estimate_nemotron_video_tokens(len(frame_paths), limits)
                if est > cap:
                    raise ValueError(
                        f"{len(frame_paths)} pre-sampled frames need ~{est} "
                        f"visual tokens but only {cap} remain in the engine "
                        "budget; reduce the frame count or other media")
            elif family == "internvl":
                max_blocks = cap // 256
                if len(frame_paths) > max_blocks:
                    raise ValueError(
                        f"{len(frame_paths)} pre-sampled frames exceed the "
                        f"engine's remaining budget of {max_blocks} InternVL "
                        "blocks")
                _check_cu_budget(len(frame_paths), family, limits, cu_budget)
                est = len(frame_paths) * 256
            elif ("qwen3_vl" in limits.get("model_type", "")
                  or "qwen3_5" in limits.get("model_type", "")):
                # 3D families use the whole-video 3D estimate, not the
                # per-frame 2D one (same constraints as the clip path).
                _check_aspect_ratio(width, height)
                tps = max(1, limits.get("temporal_patch_size", 2))
                per_image = limits.get("max_image_tokens_per_image", 0)
                media_cap = min(per_image, cap) if per_image > 0 else cap
                cu_groups = cu_budget if cu_budget is not None else (
                    limits.get("max_image_tokens", 0) //
                    max(1, limits.get("min_image_tokens", 1)))
                if cu_groups <= 0:
                    raise ValueError(
                        "the request's other media already consume the "
                        "visual engine's cu_seqlens capacity")
                media_cap = min(media_cap, cu_groups)
                if len(frame_paths) > tps * media_cap:
                    raise ValueError(
                        f"{len(frame_paths)} pre-sampled frames exceed the "
                        f"engine's per-media capacity of {tps * media_cap} "
                        "frames")
                est = _estimate_qwen3d_video_tokens(len(frame_paths), width,
                                                    height, limits)
                if est > cap:
                    raise ValueError(
                        f"{len(frame_paths)} pre-sampled frames need ~{est} "
                        f"visual tokens but only {cap} remain in the engine "
                        "budget; reduce the frame count or other media")
            else:
                # Estimate the post-resize per-frame tokens like the clip
                # path does (the C++ smart resize enforces maxRatio too).
                _check_aspect_ratio(width, height)
                frame_tokens = _estimate_qwen2d_frame_tokens(
                    width, height, limits)
                tps = max(1, limits.get("temporal_patch_size", 2))
                _check_cu_budget(len(frame_paths), family, limits, cu_budget)
                if frame_tokens > 0:
                    groups = (len(frame_paths) + tps - 1) // tps
                    est = groups * frame_tokens
                    if est > cap:
                        max_frames_fit = (cap // frame_tokens) * tps
                        raise ValueError(
                            f"{len(frame_paths)} pre-sampled {width}x{height}"
                            f" frames need ~{est} visual tokens but only "
                            f"{cap} remain in the engine budget (max "
                            f"~{max_frames_fit} frames); reduce the frame "
                            "count or other media in the request")
                else:
                    est = min(
                        limits.get("max_image_tokens_per_image", 0) or cap,
                        cap)
        ts = _frame_timestamps(family, range(len(frame_paths)), fps_val)
        try:
            buffer = rt_module.load_video_from_paths(list(frame_paths),
                                                     fps_val,
                                                     timestamps=ts)
        except RuntimeError as exc:
            # The C++ loader rejects undecodable frames or mismatched sizes;
            # user-supplied files -> client error, not a 500.
            raise ValueError(f"failed to load video frames: {exc}") from exc
        buffer.do_resize = do_resize
        if family in ("internvl", "nemotron"):
            cu_used = 0  # no cu_seqlens binding on InternVL / Nemotron engines
        else:
            tps = max(1, limits.get("temporal_patch_size", 2))
            cu_used = -(-len(frame_paths) // tps)
        return buffer, est, frames_px, cu_used

    nframes = item.get("nframes")
    if nframes is not None and "fps" in item:
        # qwen_vl_utils rejects requests that pin both; a silent winner would
        # diverge from the HF sampling contract.
        raise ValueError("provide either fps or nframes for a video, not both")
    max_frames = item.get("max_frames")
    try:
        nframes = int(nframes) if nframes is not None else None
        max_frames = int(max_frames) if max_frames is not None else None
        min_frames = int(item.get("min_frames", FPS_MIN_FRAMES))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"nframes/min_frames/max_frames must be integers: {exc}") from exc
    default_fps = NEMOTRON_DEFAULT_FPS if family == "nemotron" else DEFAULT_FPS
    frames, fps, timestamps, est, decoded_px = sample_video(
        _extract_video_source(item),
        target_fps=_positive_float(item.get("fps", default_fps), "fps"),
        nframes=nframes,
        min_frames=min_frames,
        max_frames=max_frames,
        family=family,
        frame_limits=frame_limits,
        budget=budget,
        do_resize=do_resize,
        pixel_budget=pixel_budget,
        cu_budget=cu_budget,
    )
    buffer = rt_module.load_video_from_array(frames, fps, timestamps)
    buffer.do_resize = do_resize
    lim = frame_limits or {}
    n_frames = len(timestamps)
    if family in ("internvl", "nemotron"):
        cu_used = 0  # no cu_seqlens binding on InternVL / Nemotron engines
    else:
        tps = max(1, lim.get("temporal_patch_size", 2))
        cu_used = -(-n_frames // tps)  # one entry per temporal group
    return buffer, est, decoded_px, cu_used
