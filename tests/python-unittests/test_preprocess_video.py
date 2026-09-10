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
"""
Unit coverage for the server-side video frame sampler
(``experimental/server/media/video_sampling.py``): sampling math vs the HF
pipelines,
source resolution, PyAV decode on synthetic clips, and request-budget/estimator
checks. Loaded standalone; only the pybind-marked tests need the C++ runtime.

Usage:
    python3 -m pytest tests/python-unittests/test_preprocess_video.py -v
"""
from __future__ import annotations

import base64
import os

import pytest

import experimental.server.media.video_sampling as vs


def _load_edgellm_runtime():
    """Resolve the pybind extension the way the server does: a bare import,
    else glob EDGELLM_PYBIND_DIR (CI sets that, not PYTHONPATH). Returns the
    module or None when the extension is not built anywhere."""
    import importlib
    import importlib.util
    importlib.invalidate_caches()
    try:
        import _edgellm_runtime as rt
        return rt
    except ImportError:
        pass
    pybind_dir = os.environ.get("EDGELLM_PYBIND_DIR")
    if not pybind_dir:
        return None
    import glob
    so_files = glob.glob(os.path.join(pybind_dir, "*_edgellm_runtime*.so"))
    if not so_files:
        return None
    spec = importlib.util.spec_from_file_location("_edgellm_runtime",
                                                  so_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- smart_nframes (frame-count selection) ---------------------------------

# --- sample_indices (uniform selection) ------------------------------------


def test_sample_indices():
    assert vs.sample_indices(10, 4) == [0, 3, 6, 9]
    assert vs.sample_indices(100, 1) == [0]
    idx = vs.sample_indices(1000, 16)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 999
    assert idx == sorted(idx)


# --- resolve_video_source --------------------------------------------------


def test_resolve_video_source(tmp_path):
    payload = base64.b64encode(b"\x00\x00\x00\x18ftypmp4").decode()
    path, data = vs.resolve_video_source(f"data:video/mp4;base64,{payload}")
    assert path == "" and data == b"\x00\x00\x00\x18ftypmp4"
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert vs.resolve_video_source(f"file://{f}") == (str(f), None)
    assert vs.resolve_video_source(str(f)) == (str(f), None)
    for url in (
            "data:video/mp4,rawbytes",  # non-base64 data URL
            "/no/such/clip.mp4",  # missing file (ValueError -> 400, not 500)
            "   ",  # empty
    ):
        with pytest.raises(ValueError):
            vs.resolve_video_source(url)


@pytest.mark.parametrize("url", ["http://h/v.mp4", "https://h/v.mp4"])
def test_resolve_video_source_fetches_remote(url, monkeypatch):
    monkeypatch.setattr(vs, "fetch_remote_media",
                        lambda source, kind, limit: b"video")
    assert vs.resolve_video_source(url) == ("", b"video")


# --- parameter validation & profile clamping --------------------------------


def test_frame_count_param_validation(tmp_path):
    # Invalid source metadata / request params must raise (400), and the
    # FPS_MAX_FRAMES decode ceiling can never be bypassed.
    with pytest.raises(ValueError):
        vs.smart_nframes(0, 30.0)
    with pytest.raises(ValueError):
        vs.smart_nframes(100, 0.0)
    with pytest.raises(ValueError):
        vs.smart_nframes(100, 30.0, max_frames=0)
    for fn in (vs.smart_nframes, vs.internvl_nframes):
        with pytest.raises(ValueError, match="maximum"):
            fn(100000, 30.0, nframes=10000)
    # max_frames raises the fps-derived count but never past the hard cap...
    assert vs.smart_nframes(100000, 30.0, max_frames=10000) <= \
        vs.FPS_MAX_FRAMES
    # ...and a max_frames below the frame factor wins over the factor floor.
    assert vs.smart_nframes(100, 30.0, max_frames=1) == 1
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    # fps and nframes together are rejected (qwen_vl_utils contract).
    with pytest.raises(ValueError, match="not both"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "video": str(f),
            "fps": 2.0,
            "nframes": 4
        }, "qwen")
    for kw in ({
            "target_fps": -2
    }, {
            "nframes": -7
    }, {
            "min_frames": 0
    }, {
            "max_frames": 0
    }):
        with pytest.raises(ValueError):
            vs.sample_video(str(f), **kw)


def test_sample_video_rejects_audio_only_container(tmp_path):
    pytest.importorskip("av")
    import struct
    import wave
    wav = tmp_path / "sound.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<160h", *([0] * 160)))
    with pytest.raises(ValueError, match="no video stream"):
        vs.sample_video(str(wav))


_QWEN25_LIMITS = {
    "model_type": "qwen2_5_vl",
    "min_image_tokens": 128,
    "max_image_tokens": 4096,
    "max_image_tokens_per_image": 4096,
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
}

_INTERNVL_LIMITS = {
    "model_type": "internvl",
    "min_image_tokens": 1024,  # 4 blocks
    "max_image_tokens": 4096,  # 16 blocks
    "max_image_tokens_per_image": 512,
}

_NEMOTRON_LIMITS = {
    "model_type": "nemotron_omni_vision_encoder",
    "min_image_tokens": 256,
    "max_image_tokens": 4096,
    "max_image_tokens_per_image": 4096,
    "video_pruning_rate": 0.0,
    "video_temporal_patch_size": 2,
    "video_target_num_patches": 1024,  # 256 tokens/tubelet after /downsample^2
    "downsample_ratio": 0.5,
}


def test_clamp_qwen25_total_budget():
    # 720x1280 resizes to ~1196 tokens/frame under a 4096-per-image cap, so a
    # 4096-token engine holds 3 temporal groups = 6 frames; 8 must be clamped.
    n, est = vs.clamp_nframes_to_profile(8, "qwen", 1280, 720, _QWEN25_LIMITS)
    assert n < 8
    tokens_per_frame = 1196
    assert (n + 1) // 2 * tokens_per_frame <= 4096
    assert 0 < est <= 4096


def test_clamp_qwen25_min_upscale_counted():
    # A small frame is UPSCALED by the C++ minimum branch (256x256 -> 896x896
    # = 1024 tokens under min_image_tokens=1024/f=28), so a 4096-token engine
    # holds 4 groups = 8 frames, not 100.
    limits = dict(_QWEN25_LIMITS, min_image_tokens=1024)
    n, est = vs.clamp_nframes_to_profile(100, "qwen", 256, 256, limits)
    assert n == 8
    assert est <= 4096


def test_clamp_budget_shared_across_videos():
    # A second video sees only the REMAINING budget: 16 blocks total, first
    # video takes 16, the second cannot reach the 4-block minimum.
    n2, est2 = vs.clamp_nframes_to_profile(32, "internvl", 640, 360,
                                           _INTERNVL_LIMITS)
    assert (n2, est2) == (16, 4096)
    with pytest.raises(ValueError, match="no room left"):
        vs.clamp_nframes_to_profile(16,
                                    "internvl",
                                    640,
                                    360,
                                    _INTERNVL_LIMITS,
                                    budget=4096 - est2)
    with pytest.raises(ValueError, match="no room left"):
        vs.clamp_nframes_to_profile(8,
                                    "qwen",
                                    1280,
                                    720,
                                    _QWEN25_LIMITS,
                                    budget=-1)


def test_clamp_qwen3d_frame_feasibility():
    # 4096 frames exceed BOTH 3D bounds: the per-media token budget
    # (tps * 512 = 1024 frames) and the tighter cu_seqlens group capacity
    # (max_image_tokens/min_image_tokens = 32 groups -> 64 frames wins).
    limits = dict(_QWEN25_LIMITS,
                  model_type="qwen3_vl",
                  max_image_tokens_per_image=512)
    n, est = vs.clamp_nframes_to_profile(4096, "qwen", 1280, 720, limits)
    assert n == 2 * (4096 // 128)
    assert est <= 512


def test_frames_path_qwen_size_estimate_rejects_overflow(tmp_path):
    # Pre-sampled qwen frames are size-probed: 12 x 256x256 frames upscale to
    # 1024 tokens each (min_image_tokens=1024), 6 temporal groups = 6144
    # tokens > the 4096-token engine -> reject with a 400-mappable error.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "f.mp4"
    _write_synthetic_clip(frame, n_frames=1, size=256, fps=1)
    limits = dict(_QWEN25_LIMITS, min_image_tokens=1024)
    rt = _FakeRt()
    with pytest.raises(ValueError, match="engine budget"):
        vs.load_video_buffer(rt, {
            "type": "video",
            "frames": [str(frame)] * 12,
            "fps": 1.0
        },
                             "qwen",
                             frame_limits=limits)
    # A fitting count passes and returns the probed estimate.
    buffer, est, _, _ = vs.load_video_buffer(rt, {
        "type": "video",
        "frames": [str(frame)] * 4,
        "fps": 1.0
    },
                                             "qwen",
                                             frame_limits=limits)
    assert est == 2 * 1024


def test_load_video_buffer_rejects_bad_fps(tmp_path):
    # Wrong type and non-finite values are both client errors (400).
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    for bad, msg in (([], "must be a number"), ("inf", "finite"), ("nan",
                                                                   "finite")):
        with pytest.raises(ValueError, match=msg):
            vs.load_video_buffer(_FakeRt(), {
                "type": "video",
                "video": str(f),
                "fps": bad
            }, "qwen")


def test_clamp_internvl_block_range():
    # One 256-token block per frame: [4, 16] blocks for this engine.
    assert vs.clamp_nframes_to_profile(32, "internvl", 640, 360,
                                       _INTERNVL_LIMITS)[0] == 16
    assert vs.clamp_nframes_to_profile(4, "internvl", 640, 360,
                                       _INTERNVL_LIMITS)[0] == 4
    # The engine minimum is request-wide (two short videos can jointly reach
    # it), so a single short video is NOT rejected here; the caller validates
    # the accumulated total.
    assert vs.clamp_nframes_to_profile(1, "internvl", 640, 360,
                                       _INTERNVL_LIMITS) == (1, 256)


_QWEN3VL_LIMITS = {
    "model_type": "qwen3_vl",
    "min_image_tokens": 128,
    "max_image_tokens": 8192,
    "max_image_tokens_per_image": 4096,
    "patch_size": 16,
    "merge_size": 2,
    "temporal_patch_size": 2,
}


def test_clamp_qwen3d_uses_real_resize_tokens():
    # The 3D estimate must be the tokens the C++ resize actually produces
    # (8 frames of 720x1280 -> 704x1280 -> 3520 tokens), NOT the remaining
    # cap, or multi-video accounting drifts from reality.
    n, est = vs.clamp_nframes_to_profile(8, "qwen", 1280, 720, _QWEN3VL_LIMITS)
    assert n == 8
    assert est == 3520
    # Two such videos exhaust the 8192 budget; a third must be rejected.
    with pytest.raises(ValueError, match="budget"):
        vs.clamp_nframes_to_profile(8,
                                    "qwen",
                                    1280,
                                    720,
                                    _QWEN3VL_LIMITS,
                                    budget=8192 - 2 * 3520)


def test_frames_path_qwen3d_uses_3d_estimate(tmp_path):
    # Pre-sampled frames for a 3D-family model must use the whole-video 3D
    # estimate (100 frames of 256x256 fit a 4096-token per-media budget after
    # the 3D shrink), not the per-frame 2D math.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "f.mp4"
    _write_synthetic_clip(frame, n_frames=1, size=256, fps=1)
    buffer, est, _, _ = vs.load_video_buffer(_FakeRt(), {
        "type": "video",
        "frames": [str(frame)] * 100,
        "fps": 1.0
    },
                                             "qwen",
                                             frame_limits=_QWEN3VL_LIMITS)
    assert 0 < est <= 4096


def test_sample_video_corrupt_container_maps_to_value_error(tmp_path):
    # PyAV 17 raises av.error.FFmpegError; it must surface as ValueError
    # (HTTP 400), not AttributeError/RuntimeError (HTTP 500).
    pytest.importorskip("av")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\xde\xad" * 64)
    with pytest.raises(ValueError):
        vs.sample_video(str(bad))


def test_clamp_qwen3d_estimate_matches_cpp_fallback():
    # 146x640x360 under [256, 512]: the naive shrink lands below the minimum
    # and the factor-grid fallback (mirroring the C++ resize) yields 438
    # tokens -- NOT the pre-fallback 219. Multi-video accounting must use it.
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=256,
                  max_image_tokens=1024,
                  max_image_tokens_per_image=512)
    # The estimator itself must reproduce the factor-grid fallback result.
    assert vs._estimate_qwen3d_video_tokens(146, 640, 360, limits) == 438
    # End to end, the cu_seqlens group capacity (1024/256 = 4 groups) caps
    # the video to 8 frames first; the estimate then reflects 8 frames.
    n, est = vs.clamp_nframes_to_profile(146, "qwen", 640, 360, limits)
    assert n == 8
    assert est == 480


def test_sample_video_single_frame_est_uses_actual_count(tmp_path):
    # A 1-frame clip plans nframes=2 (frame factor) but decodes 1; the
    # budget estimate must reflect the actually decoded frame count.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "one.mp4"
    _write_synthetic_clip(clip, n_frames=1, size=64, fps=1)
    frames, _, _, est, _ = vs.sample_video(str(clip),
                                           frame_limits=_QWEN3VL_LIMITS)
    assert frames.shape[0] == 1
    assert est == vs._estimate_qwen3d_video_tokens(1, 64, 64, _QWEN3VL_LIMITS)


def test_load_video_buffer_sets_do_resize(tmp_path, monkeypatch):
    # do_resize:false must reach the buffer AND switch the estimate to the
    # input dimensions (the C++ resize is skipped for pre-resized input).
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "pre.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    rt = _FakeRt()
    buffer, est, _, _ = vs.load_video_buffer(rt, {
        "type": "video",
        "video": str(clip),
        "do_resize": False
    },
                                             "qwen",
                                             frame_limits=_QWEN3VL_LIMITS)
    assert buffer.do_resize is False
    # input-size tokens: tBar*64*64/(2*32*32) = 4*4096/2048 = 8
    assert est == 8
    buffer2, _, _, _ = vs.load_video_buffer(rt, {
        "type": "video",
        "video": str(clip)
    },
                                            "qwen",
                                            frame_limits=_QWEN3VL_LIMITS)
    assert buffer2.do_resize is True


def test_estimate_qwen_rejects_extreme_aspect_ratio():
    # Mirror the C++ maxRatio=200 rejection before decode, not a runtime 500.
    with pytest.raises(ValueError, match="aspect ratio"):
        vs.clamp_nframes_to_profile(8, "qwen", 2010, 10, _QWEN3VL_LIMITS)


def test_internvl_ratio_not_limited():
    # The C++ InternVL grid resize has NO maxRatio=200 gate (only the Qwen
    # smart resize does), so a 201:1 video must not be rejected up front.
    n, est = vs.clamp_nframes_to_profile(4, "internvl", 448 * 201, 448,
                                         _INTERNVL_LIMITS)
    assert (n, est) == (4, 1024)


def test_estimate_image_tokens_internvl_golden(tmp_path):
    # Golden values from the C++ computeBestBlockGridForResize semantics.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    cap = {"max_image_tokens_per_image": 3072}
    cases = [
        # exactly one block, no thumbnail
        (448, 448, cap, lambda est: est == 256),
        # (11+1)*256 max: C++ reserves one grid block for the thumbnail
        (5376, 448, cap, lambda est: est <= 3072),
        # area tie-break upgrades 1x1 -> 2x2 (not 3x3), + thumbnail = (4+1)*256
        (896, 896, cap, lambda est: est == 1280),
        # grid stays 1x1 but the 2-block engine minimum forces the thumbnail
        (448, 448, dict(cap, min_image_tokens=512), lambda est: est == 512),
    ]
    for idx, (w, h, limits, ok) in enumerate(cases):
        f = tmp_path / f"g{idx}.mp4"
        _write_synthetic_clip_hw(f, n_frames=1, width=w, height=h, fps=1)
        est = vs.estimate_image_tokens(str(f), "internvl", limits)
        assert ok(est), f"case {w}x{h} {limits}: got {est}"


def test_estimate_image_tokens_raw_input(tmp_path):
    # do_resize=false images skip the C++ resize: tokens follow the raw
    # dimensions, which must be tile-aligned (InternVL) / factor-aligned
    # (Qwen); misalignment is a client error, not a runtime 500.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    aligned = tmp_path / "a.mp4"
    _write_synthetic_clip(aligned, n_frames=1, size=896, fps=1)
    limits = {"max_image_tokens_per_image": 3072}
    # 896x896 = 4 main blocks + the thumbnail the C++ path appends = 5.
    assert vs.estimate_image_tokens(str(aligned),
                                    "internvl",
                                    limits,
                                    do_resize=False) == 5 * 256
    # A single 448 block normally has no thumbnail, but an engine minimum
    # above one block forces it (matches C++ minNumBlocks semantics).
    single = tmp_path / "s448.mp4"
    _write_synthetic_clip(single, n_frames=1, size=448, fps=1)
    assert vs.estimate_image_tokens(str(single),
                                    "internvl",
                                    limits,
                                    do_resize=False) == 256
    limits_min = dict(limits, min_image_tokens=512)
    assert vs.estimate_image_tokens(str(single),
                                    "internvl",
                                    limits_min,
                                    do_resize=False) == 512
    qwen_ok = tmp_path / "q.mp4"
    _write_synthetic_clip(qwen_ok, n_frames=1, size=64, fps=1)
    assert vs.estimate_image_tokens(
        str(qwen_ok), "qwen", _QWEN3VL_LIMITS,
        do_resize=False) == 4  # 64*64 / 32^2
    bad = tmp_path / "b.mp4"
    _write_synthetic_clip(bad, n_frames=1, size=100, fps=1)
    with pytest.raises(ValueError, match="448-aligned"):
        vs.estimate_image_tokens(str(bad), "internvl", limits, do_resize=False)
    with pytest.raises(ValueError, match="32-aligned"):
        vs.estimate_image_tokens(str(bad),
                                 "qwen",
                                 _QWEN3VL_LIMITS,
                                 do_resize=False)


def test_raw_clip_not_resize_clamped(tmp_path):
    # do_resize=false clips keep the sampling plan's frame count: the
    # resize-based clamp does not apply, and only a genuine budget
    # overflow rejects.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "raw.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    frames, _, _, est, _ = vs.sample_video(str(clip),
                                           frame_limits=_QWEN3VL_LIMITS,
                                           budget=8,
                                           do_resize=False)
    assert frames.shape[0] == 4
    assert est == 8  # tBar*64*64 // (2*32*32)
    with pytest.raises(ValueError, match="engine budget"):
        vs.sample_video(str(clip),
                        frame_limits=_QWEN3VL_LIMITS,
                        budget=7,
                        do_resize=False)


def test_raw_frames_path_matrix(tmp_path):
    # Pre-sampled frames with do_resize=false: InternVL frames must be
    # exactly 448x448 (one block per frame); Qwen frames account by raw
    # size without resize-based truncation.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    limits = {
        "model_type": "internvl",
        "min_image_tokens": 256,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 512,
    }
    big = tmp_path / "big.mp4"
    _write_synthetic_clip(big, n_frames=1, size=896, fps=1)
    with pytest.raises(ValueError, match="448x448"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(big)] * 2,
            "fps": 1.0,
            "do_resize": False
        },
                             "internvl",
                             frame_limits=limits)
    tile = tmp_path / "tile.mp4"
    _write_synthetic_clip(tile, n_frames=1, size=448, fps=1)
    buffer, est, _, _ = vs.load_video_buffer(_FakeRt(), {
        "type": "video",
        "frames": [str(tile)] * 2,
        "fps": 1.0,
        "do_resize": False
    },
                                             "internvl",
                                             frame_limits=limits)
    assert buffer.do_resize is False
    assert est == 2 * 256
    small = tmp_path / "small.mp4"
    _write_synthetic_clip(small, n_frames=1, size=64, fps=1)
    buffer, est, _, _ = vs.load_video_buffer(_FakeRt(), {
        "type": "video",
        "frames": [str(small)] * 4,
        "fps": 1.0,
        "do_resize": False
    },
                                             "qwen",
                                             frame_limits=_QWEN3VL_LIMITS)
    assert est == 8  # tBar*64*64 // (2*32*32)


def test_frames_path_fills_timestamps(tmp_path):
    # The frames-path must attach per-frame label timestamps so the runner
    # bounds its Frame labels; check both an even and an odd frame count for
    # the Nemotron (quantized) and default (idx/fps) formulas.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "f.mp4"
    _write_synthetic_clip(frame, n_frames=1, size=64, fps=1)
    for family, count, fps in (("nemotron", 3, 2.0), ("qwen", 4, 2.0)):
        buffer, _, _, _ = vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(frame)] * count,
            "fps": fps,
        }, family)
        expected = vs._frame_timestamps(family, range(count), fps)
        assert list(buffer.timestamps) == expected
        assert len(buffer.timestamps) == count


def test_frames_path_rejects_mismatched_sizes(tmp_path):
    # Mixed frame sizes fail up front with a client error (the C++ loader
    # would reject them later with an opaque RuntimeError), and the pixel
    # ceiling sums ACTUAL per-frame sizes rather than assuming frame 0.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_synthetic_clip(a, n_frames=1, size=64, fps=1)
    _write_synthetic_clip(b, n_frames=1, size=128, fps=1)
    with pytest.raises(ValueError, match="share one size"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(a), str(b)],
            "fps": 1.0
        },
                             "qwen",
                             frame_limits=_QWEN3VL_LIMITS)


def test_frames_path_qwen3d_respects_per_media_frame_cap(tmp_path):
    # 768 frames under a per-media budget of 128 tokens exceed the
    # tps * media_cap = 256-frame capacity; must reject up front (the C++
    # resize would find "no feasible shape").
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "f.mp4"
    _write_synthetic_clip(frame, n_frames=1, size=256, fps=1)
    limits = dict(_QWEN3VL_LIMITS,
                  max_image_tokens=8192,
                  max_image_tokens_per_image=128)
    with pytest.raises(ValueError, match="per-media"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(frame)] * 768,
            "fps": 1.0
        },
                             "qwen",
                             frame_limits=limits)


def test_frames_path_pixel_ceiling(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "f.mp4"
    _write_synthetic_clip(frame, n_frames=1, size=64, fps=1)
    monkeypatch.setattr(vs, "MAX_DECODE_PIXELS", 1000)
    with pytest.raises(ValueError, match="pixel limit"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(frame)] * 4,
            "fps": 1.0
        },
                             "qwen",
                             frame_limits=_QWEN3VL_LIMITS)


def test_raw_video_input_validation(tmp_path):
    # do_resize=false inputs bypass the C++ resize, so misaligned dimensions
    # must be rejected up front: InternVL video frames must be exactly
    # 448x448 and Qwen frames factor-aligned.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "c.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=100, fps=2)  # not aligned
    limits = {
        "model_type": "internvl",
        "min_image_tokens": 256,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 512,
    }
    with pytest.raises(ValueError, match="448x448"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "video": str(clip),
            "do_resize": False
        },
                             "internvl",
                             frame_limits=limits)
    with pytest.raises(ValueError, match="aligned"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "video": str(clip),
            "do_resize": False
        },
                             "qwen",
                             frame_limits=_QWEN3VL_LIMITS)


def test_request_wide_pixel_budget(tmp_path, monkeypatch):
    # Two videos each under the per-video pixel ceiling must not jointly
    # exceed it: the second sees only the REQUEST remaining pixel budget.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "px.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)  # 16384 px
    monkeypatch.setattr(vs, "MAX_DECODE_PIXELS", 20000)
    rt = _FakeRt()
    item = {"type": "video", "video": str(clip)}
    buffer, est, px, _ = vs.load_video_buffer(rt, item, "qwen")
    assert px == 4 * 64 * 64
    with pytest.raises(ValueError, match="remaining host"):
        vs.load_video_buffer(rt, item, "qwen", pixel_budget=20000 - px)


def test_frames_path_ratio_checked_without_limits(tmp_path):
    # An extreme-ratio pre-sampled qwen video must be rejected even when no
    # profile limits are available (the C++ resize enforces maxRatio anyway).
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    frame = tmp_path / "wide.mp4"
    _write_synthetic_clip_hw(frame, n_frames=1, width=2010, height=10, fps=1)
    with pytest.raises(ValueError, match="aspect ratio"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(frame)] * 2,
            "fps": 1.0
        }, "qwen")


def test_clamp_qwen3d_respects_cu_seqlens_group_capacity():
    # 768 frames fit the token budget after the 3D shrink but produce 384
    # temporal groups vs the 64-entry cu_seqlens profile
    # (max_image_tokens/min_image_tokens); frames must cap at tps * 64 = 128.
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=128,
                  max_image_tokens=8192,
                  max_image_tokens_per_image=4096)
    n, est = vs.clamp_nframes_to_profile(768, "qwen", 640, 360, limits)
    assert n == 2 * (8192 // 128)
    assert est <= 4096


def test_request_wide_cu_group_budget(tmp_path):
    # Two videos each within the 32-group capacity (8192/256) must share it:
    # the first consumes its groups, the second sees only the remainder.
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=256,
                  max_image_tokens=8192,
                  max_image_tokens_per_image=512)
    n1, _ = vs.clamp_nframes_to_profile(768, "qwen", 640, 360, limits)
    assert n1 == 2 * 32  # full capacity when the request is empty
    n2, _ = vs.clamp_nframes_to_profile(768,
                                        "qwen",
                                        640,
                                        360,
                                        limits,
                                        cu_budget=32 - 20)
    assert n2 == 2 * 12  # only the remaining groups
    with pytest.raises(ValueError, match="cu_seqlens capacity"):
        vs.clamp_nframes_to_profile(8, "qwen", 640, 360, limits, cu_budget=0)


def test_source_decode_work_ceilings(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    # Reported-frame ceiling: a long source is rejected before decoding.
    clip = tmp_path / "long.mp4"
    _write_synthetic_clip(clip, n_frames=30, size=64, fps=30)
    monkeypatch.setattr(vs, "MAX_SOURCE_FRAMES", 10)
    with pytest.raises(ValueError, match="maximum"):
        vs.sample_video(str(clip))
    # Encoded-byte ceiling for data: URLs, checked BEFORE base64 decode.
    monkeypatch.setattr(vs, "MAX_SOURCE_BYTES", 16)
    with pytest.raises(ValueError, match="bytes"):
        vs.resolve_video_source("data:video/mp4;base64," + "A" * 400)


def test_cu_budget_zero_blocks_every_intake_path(tmp_path):
    # With the group capacity exhausted, every path must
    # reject — resized clip, raw clip, resized frames, raw frames — not just
    # the clamp helper.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "c.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    frame448 = tmp_path / "f448.mp4"
    _write_synthetic_clip(frame448, n_frames=1, size=448, fps=1)
    rt = _FakeRt()
    qwen3 = dict(_QWEN3VL_LIMITS)
    internvl = {
        "model_type": "internvl",
        "min_image_tokens": 256,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 512,
    }
    qwen_frame = tmp_path / "qf.mp4"
    _write_synthetic_clip(qwen_frame, n_frames=1, size=64, fps=1)
    cases = [
        ({
            "type": "video",
            "video": str(clip)
        }, qwen3),
        ({
            "type": "video",
            "video": str(clip),
            "do_resize": False
        }, qwen3),
        ({
            "type": "video",
            "frames": [str(qwen_frame)] * 2,
            "fps": 1.0
        }, qwen3),
        ({
            "type": "video",
            "frames": [str(qwen_frame)] * 2,
            "fps": 1.0,
            "do_resize": False
        }, qwen3),
    ]
    for item, limits in cases:
        with pytest.raises(ValueError, match="capacity"):
            vs.load_video_buffer(rt,
                                 item,
                                 "qwen",
                                 frame_limits=limits,
                                 cu_budget=0)
    # InternVL has NO cu_seqlens binding (block-profile engine): it must be
    # exempt from the group budget and only bounded by its token profile —
    # 8 frames (2048 tokens) fit an 8192-token engine even with cu_budget=0.
    buffer, est, _, _ = vs.load_video_buffer(rt, {
        "type": "video",
        "frames": [str(frame448)] * 8,
        "fps": 1.0
    },
                                             "internvl",
                                             frame_limits=internvl,
                                             cu_budget=0)
    assert est == 8 * 256


def test_actual_decoded_pixels_enforced(tmp_path, monkeypatch):
    # The pixel cap must be enforced on the actual decoded size as frames are
    # retained, not only via container metadata before the decode.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "px.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    with pytest.raises(ValueError, match="host limit"):
        vs.sample_video(str(clip), pixel_budget=4 * 64 * 64 - 1)
    # Metadata says 64x64 but frames decode at 512x512: the pre-decode check
    # passes, so the in-loop actual-pixel accumulation must abort instead.
    _patch_av_decoded_sizes(monkeypatch, (512, 512))
    with pytest.raises(ValueError, match="host pixel limit"):
        vs.sample_video(str(clip), pixel_budget=4 * 64 * 64)


def _patch_av_decoded_sizes(monkeypatch, *sizes):
    """Decode a real container but fake each frame's (height, width) from the
    cycled ``sizes`` sequence, mimicking rotation/filter chains and
    mis-reporting metadata."""
    import av as av_mod
    np = pytest.importorskip("numpy")
    real_open = av_mod.open

    class _Frame:

        def __init__(self, hw):
            self._hw = hw

        def to_ndarray(self, format):
            return np.zeros((*self._hw, 3), np.uint8)

    class _Container:

        def __init__(self, inner):
            self._inner = inner
            self.streams = inner.streams

        def decode(self, stream):
            for i, _ in enumerate(self._inner.decode(stream)):
                yield _Frame(sizes[i % len(sizes)])

        def seek(self, *args, **kwargs):
            return self._inner.seek(*args, **kwargs)

        def close(self):
            return self._inner.close()

    monkeypatch.setattr(av_mod, "open",
                        lambda *a, **k: _Container(real_open(*a, **k)))


def _patch_av_frame_counts(monkeypatch, reported, decoded, size=32):
    """Fully fake container whose metadata reports ``reported`` frames while
    ``decoded`` frames actually come out (mis-reporting sources)."""
    import av as av_mod
    np = pytest.importorskip("numpy")

    class _Frame:

        def to_ndarray(self, format):
            return np.zeros((size, size, 3), np.uint8)

    class _Stream:
        frames = reported
        average_rate = 2
        width = size
        height = size

    class _Streams:
        video = [_Stream()]

    class _Container:
        streams = _Streams()

        def decode(self, stream):
            for _ in range(decoded):
                yield _Frame()

        def seek(self, *args, **kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(av_mod, "open", lambda *a, **k: _Container())


def test_actual_decoded_size_reestimates_tokens(tmp_path, monkeypatch):
    # Metadata says 64x64 (passes the pre-decode estimate) but frames decode
    # at 512x512: the estimate must be redone with the actual size so this
    # fails 400 here, not at the visual engine profile.
    pytest.importorskip("av")
    clip = tmp_path / "re.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    limits = dict(_QWEN25_LIMITS, max_image_tokens=512)
    # sanity: at the metadata size the plan fits
    vs.sample_video(str(clip), family="qwen", frame_limits=limits, budget=512)
    _patch_av_decoded_sizes(monkeypatch, (512, 512))
    with pytest.raises(ValueError, match="token"):
        vs.sample_video(str(clip),
                        family="qwen",
                        frame_limits=limits,
                        budget=512)


def test_actual_decoded_size_rechecks_raw_alignment(tmp_path, monkeypatch):
    # do_resize=false alignment must hold for the ACTUAL decoded size, not
    # just the container metadata (C++ derives InternVL frame count from
    # exact 448x448 blocks).
    pytest.importorskip("av")
    clip = tmp_path / "raw.mp4"
    _write_synthetic_clip_hw(clip, n_frames=4, height=448, width=448, fps=2)
    _patch_av_decoded_sizes(monkeypatch, (512, 512))
    with pytest.raises(ValueError, match="448"):
        vs.sample_video(str(clip),
                        family="internvl",
                        frame_limits=_INTERNVL_LIMITS,
                        do_resize=False)


class _FakeRt:
    """Records which load_* binding the buffer loader chose + its args."""

    def __init__(self):
        self.calls = []

    class _Buf(tuple):
        """Tuple-comparable stub buffer that accepts attribute writes
        (do_resize etc.)."""

        def __new__(cls, *items):
            return super().__new__(cls, items)

    def load_video_from_array(self, frames, fps, timestamps=()):
        self.calls.append(("array", frames, fps))
        buf = self._Buf("array", frames, fps)
        buf.timestamps = list(timestamps)
        return buf

    def load_video_from_paths(self, paths, fps, timestamps=()):
        self.calls.append(("paths", list(paths), fps))
        buf = self._Buf("paths", list(paths), fps)
        buf.timestamps = list(timestamps)
        return buf


def test_content_item_source_validation():
    # Accepted spellings map to the url; malformed items (empty, missing,
    # non-string, bad base64) are ValueError (400), never AttributeError /
    # binascii.Error (500).
    accepts = [
        ({
            "type": "video",
            "video": "/a/clip.mp4"
        }, "/a/clip.mp4"),
        ({
            "type": "video",
            "video": {
                "url": "file:///a/clip.mp4"
            }
        }, "file:///a/clip.mp4"),
        ({
            "type": "video_url",
            "video_url": {
                "url": "/b/clip.mp4"
            }
        }, "/b/clip.mp4"),
    ]
    for item, expected in accepts:
        assert vs._extract_video_source(item) == expected
    for item in ({"type": "video", "video": ""}, {"type": "video"}):
        with pytest.raises(ValueError):
            vs._extract_video_source(item)
    with pytest.raises(ValueError, match="base64"):
        vs.resolve_video_source("data:video/mp4;base64,a")
    for item in ({
            "type": "video",
            "video": 123
    }, {
            "type": "video_url",
            "video_url": {
                "url": 123
            }
    }):
        with pytest.raises(ValueError, match="string"):
            vs.load_video_buffer(_FakeRt(), item, "qwen")


@pytest.mark.parametrize(
    "total,video_fps,ele",
    [
        (300, 30.0, {
            "fps": 2.0
        }),  # 10 s clip, default target fps
        (50, 30.0, {
            "fps": 2.0
        }),  # short, odd-ish totals
        (121, 30.0, {
            "fps": 2.0
        }),
        (6, 30.0, {
            "fps": 2.0
        }),  # barely above min
        (90, 30.0, {
            "fps": 2.0
        }),  # 3 s
        (5, 30.0, {
            "fps": 2.0
        }),  # shorter than min_frames -> clamps
        (37, 30.0, {
            "fps": 2.0
        }),  # odd total
        (10000, 30.0, {
            "fps": 2.0
        }),  # long clip -> hits max cap
        (600, 24.0, {
            "fps": 1.0
        }),  # lower target fps, 24 fps source
        (300, 30.0, {
            "nframes": 8
        }),  # explicit nframes
        (300, 30.0, {
            "nframes": 7
        }),  # explicit, rounded to factor
        (300, 30.0, {
            "fps": 2.0,
            "max_frames": 16
        }),  # max cap
        (300, 30.0, {
            "fps": 2.0,
            "min_frames": 12
        }),  # min floor
    ],
)
def test_smart_nframes_matches_qwen_vl_utils(total, video_fps, ele):
    vp = pytest.importorskip("qwen_vl_utils.vision_process")
    ref = vp.smart_nframes(ele, total_frames=total, video_fps=video_fps)
    ours = vs.smart_nframes(
        total,
        video_fps,
        target_fps=ele.get("fps", vs.DEFAULT_FPS),
        nframes=ele.get("nframes"),
        min_frames=ele.get("min_frames", vs.FPS_MIN_FRAMES),
        max_frames=ele.get("max_frames"),
    )
    assert ours == ref, f"ours={ours} ref={ref} for total={total} ele={ele}"


# --- decode round-trip on a synthetic clip (shape + fps) -------------------
# Exercises the real PyAV decode path of sample_video. Skips without PyAV + numpy.


def _write_synthetic_clip_hw(path, *, n_frames, width, height, fps):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        img = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_synthetic_clip(path, *, n_frames=30, size=64, fps=30):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        img = np.full((size, size, 3), (i * 8) % 255, dtype=np.uint8)
        for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_sample_video_decode(tmp_path):
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    clip = tmp_path / "syn.mp4"
    _write_synthetic_clip(clip, n_frames=30, size=64, fps=30)
    # Explicit nframes: native-resolution uint8 [T, H, W, 3], T as requested.
    frames, fps, ts, _, _ = vs.sample_video(str(clip), nframes=6)
    assert frames.shape == (6, 64, 64, 3)
    assert frames.dtype == np.uint8
    # 6 frames sampled from a 1 s clip -> 6 fps.
    assert fps == pytest.approx(6.0, abs=0.5)
    # Source timestamps (original_index / source_fps), monotonic 0..duration.
    assert len(ts) == 6 and ts == sorted(ts)
    assert ts[0] == pytest.approx(0.0, abs=0.05)
    assert ts[-1] == pytest.approx(29 / 30, abs=0.05)
    # fps-derived path: 2 s clip @ target 2 fps -> even count, native size.
    clip2 = tmp_path / "syn2.mp4"
    _write_synthetic_clip(clip2, n_frames=60, size=48, fps=30)
    frames, fps, ts, _, _ = vs.sample_video(str(clip2), target_fps=2.0)
    assert frames.ndim == 4 and frames.shape[1:] == (48, 48, 3)
    assert frames.shape[0] % vs.FRAME_FACTOR == 0
    assert fps > 0


def test_internvl_indices_clamp_and_edges():
    assert vs.sample_indices_internvl(3, 8) == [0, 1, 2]  # clamped to total
    assert vs.sample_indices_internvl(1, 1) == [0]
    assert vs.sample_indices_internvl(0, 4) == []


def test_internvl_nframes():
    # HF semantics: explicit nframes is exact-or-reject; else fps-derived
    # with max_frames/total_frames caps; never exceeds FPS_MAX_FRAMES.
    assert vs.internvl_nframes(100, 25.0, nframes=8) == 8
    assert vs.internvl_nframes(100, 25.0, target_fps=2.0) == 8
    assert vs.internvl_nframes(100, 25.0, target_fps=2.0, max_frames=4) == 4
    with pytest.raises(ValueError, match="exceeds the video"):
        vs.internvl_nframes(6, 25.0, nframes=16)
    assert vs.internvl_nframes(100000, 30.0) <= vs.FPS_MAX_FRAMES


@pytest.mark.parametrize("total,kw", [
    (300, {
        "num_frames": 8
    }),
    (300, {
        "num_frames": 7
    }),
    (37, {
        "num_frames": 5
    }),
    (1000, {
        "num_frames": 16
    }),
    (300, {
        "fps": 2.0
    }),
    (240, {
        "fps": 1.0
    }),
])
def test_sample_indices_internvl_matches_hf(total, kw):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # transformers' video-processor import chain needs torchvision.
    pytest.importorskip("torchvision")
    from transformers.models.internvl.video_processing_internvl import \
        InternVLVideoProcessor
    from transformers.video_utils import VideoMetadata
    video_fps = 30.0
    meta = VideoMetadata(total_num_frames=total,
                         fps=video_fps,
                         duration=total / video_fps,
                         frames_indices=list(range(total)))
    ref = InternVLVideoProcessor().sample_frames(meta, **kw).tolist()
    n = vs.internvl_nframes(total,
                            video_fps,
                            target_fps=kw.get("fps"),
                            nframes=kw.get("num_frames"))
    ours = vs.sample_indices_internvl(total, n)
    assert ours == ref, f"ours={ours} ref={ref} kw={kw}"


def _sample_indices_nemotron_ref(total_frames,
                                 video_fps,
                                 *,
                                 target_fps=None,
                                 nframes=None,
                                 max_frames=None):
    """Inline port of the checkpoint video_io.py index selection."""
    np = pytest.importorskip("numpy")
    if total_frames <= 0:
        return []
    if nframes is not None:
        desired = int(nframes)
    else:
        fps = target_fps if target_fps is not None else 1.0
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


@pytest.mark.parametrize(
    "total,video_fps,kw",
    [
        (300, 30.0, {}),  # 10 s @ default 1 fps -> 10 frames
        (300, 30.0, {
            "target_fps": 2.0
        }),
        (37, 30.0, {}),  # sub-2 s odd total
        (25, 30.0, {}),  # <1 s -> desired 0 -> [0]
        (30, 30.0, {
            "nframes": 6
        }),  # explicit nframes overrides fps
        (30, 30.0, {
            "nframes": 100
        }),  # nframes > total -> every frame
        (300, 30.0, {
            "target_fps": 1.0,
            "max_frames": 4
        }),  # max cap
    ])
def test_sample_indices_nemotron_matches_hf(total, video_fps, kw):
    pytest.importorskip("numpy")
    ref = _sample_indices_nemotron_ref(total, video_fps, **kw)
    ours = vs.sample_indices_nemotron(total, video_fps, **kw)
    assert ours == ref, f"ours={ours} ref={ref} kw={kw}"


def test_sample_indices_nemotron_dedup_cap_and_explicit():
    pytest.importorskip("numpy")
    # <1 s source -> desired 0 -> single leading frame.
    assert vs.sample_indices_nemotron(25, 30.0) == [0]
    # desired == total returns every frame.
    assert vs.sample_indices_nemotron(10, 1.0) == list(range(10))
    # Explicit nframes overrides the fps rule and is uniform + sorted.
    idx = vs.sample_indices_nemotron(30, 30.0, nframes=6)
    assert len(idx) == 6 and idx[0] == 0 and idx[-1] == 29 and idx == sorted(
        idx)
    # nframes above the source count collapses (dedup) to every frame.
    assert vs.sample_indices_nemotron(4, 1.0, nframes=100) == [0, 1, 2, 3]
    # max_frames caps the fps-derived count.
    assert len(
        vs.sample_indices_nemotron(300, 30.0, target_fps=1.0,
                                   max_frames=4)) == 4


def test_clamp_nemotron_video_budget():
    # 8 frames -> 4 tubelets (T=2); 1024 patches / downsample^2 (4) = 256
    # tokens/tubelet, no pruning -> 1024 tokens; fits the 4096-token engine.
    n, est = vs.clamp_nframes_to_profile(8, "nemotron", 640, 360,
                                         _NEMOTRON_LIMITS)
    assert n == 8
    assert est == 4 * 256
    # A tight budget clamps the frame count: 512 tokens holds 2 tubelets = 4.
    small = dict(_NEMOTRON_LIMITS, max_image_tokens=512)
    n2, est2 = vs.clamp_nframes_to_profile(100, "nemotron", 640, 360, small)
    assert (n2, est2) == (4, 512)
    # EVS pruning does NOT raise the tubelet count: the engine processes every
    # tubelet pre-EVS, so the block budget is unchanged by the pruning rate.
    pruned = dict(_NEMOTRON_LIMITS,
                  max_image_tokens=512,
                  video_pruning_rate=0.5)
    n3, _ = vs.clamp_nframes_to_profile(100, "nemotron", 640, 360, pruned)
    assert n3 == 4


def test_clamp_nemotron_missing_video_keys():
    # Absent Nemotron geometry -> no clamp (like the empty-limits case).
    limits = {
        "model_type": "nemotron_omni_vision_encoder",
        "min_image_tokens": 1,
        "max_image_tokens": 4096,
    }
    assert vs.clamp_nframes_to_profile(8, "nemotron", 640, 360,
                                       limits) == (8, 0)


def test_nemotron_ratio_not_limited():
    # Nemotron resize is aspect-preserving: extreme ratios must not be rejected
    # up front (exempt from maxRatio=200, like InternVL).
    n, est = vs.clamp_nframes_to_profile(4, "nemotron", 2010, 10,
                                         _NEMOTRON_LIMITS)
    assert n == 4 and est == 2 * 256


def test_sample_video_nemotron_timestamps(tmp_path):
    # Nemotron timestamps use the vLLM formula int(idx)*int(1000/fps)/1000,
    # NOT qwen's idx/fps; both sample the same uniform indices.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "n.mp4"
    _write_synthetic_clip(clip, n_frames=30, size=64, fps=30)
    _, _, ts_n, _, _ = vs.sample_video(str(clip), nframes=6, family="nemotron")
    _, _, ts_q, _, _ = vs.sample_video(str(clip), nframes=6, family="qwen")
    idxs = vs.sample_indices_nemotron(30, 30.0, nframes=6)
    per_ms = int(1000.0 / 30.0)
    assert ts_n == pytest.approx([int(i) * per_ms / 1000.0 for i in idxs])
    assert ts_n != pytest.approx(ts_q)


def test_load_video_buffer_nemotron_rejects_do_resize_false(tmp_path):
    # The Nemotron-Omni runner always smart-resizes, so do_resize=false is a
    # client error rather than a silently-resized clip.
    pytest.importorskip("av")
    clip = tmp_path / "n.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    with pytest.raises(ValueError, match="do_resize=false"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "video": str(clip),
            "do_resize": False
        },
                             "nemotron",
                             frame_limits=_NEMOTRON_LIMITS)


def test_load_video_buffer_nemotron_bounds_presampled_tubelets(
        tmp_path, monkeypatch):
    frame_paths = []
    for index in range(6):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(b"image")
        frame_paths.append(str(path))
    monkeypatch.setattr(vs, "_probe_image_size", lambda _path: (64, 64))
    limits = dict(_NEMOTRON_LIMITS,
                  max_image_tokens=512,
                  video_pruning_rate=0.9)
    with pytest.raises(ValueError, match="visual engine profile holds 2"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": frame_paths,
            "fps": 1.0,
        },
                             "nemotron",
                             frame_limits=limits)


def test_mid_stream_resolution_change_rejected(tmp_path, monkeypatch):
    # The (T, H, W, 3) stack requires uniform frames; a source whose frames
    # decode at differing sizes must be a client error, not a numpy
    # broadcast crash at assembly.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "mix.mp4"
    _write_synthetic_clip(clip, n_frames=4, size=64, fps=2)
    _patch_av_decoded_sizes(monkeypatch, (64, 64), (32, 32))
    with pytest.raises(ValueError, match="differing sizes"):
        vs.sample_video(str(clip))


def test_single_frame_video_pixel_budget_not_inflated(tmp_path):
    # A single-frame video plans n=2 (frame-factor rounding) but decodes one
    # frame; the decode-pixel precheck must charge the one sampled position,
    # not the padded plan.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "one.mp4"
    _write_synthetic_clip(clip, n_frames=1, size=64, fps=1)
    frames, _, timestamps, _, px = vs.sample_video(str(clip),
                                                   pixel_budget=64 * 64)
    assert len(timestamps) == 1
    assert px == 64 * 64


def test_underreported_frame_count_samples_reported_range(
        tmp_path, monkeypatch):
    # A positive reported frame count is trusted as-is (single-pass sampling):
    # an under-reporting container samples within the reported range instead
    # of decoding everything like HF — deliberate performance trade-off.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "under.mp4"
    clip.write_bytes(b"x")
    _patch_av_frame_counts(monkeypatch, reported=4, decoded=8)
    frames, fps, timestamps, _, _ = vs.sample_video(str(clip))
    assert len(timestamps) == 4  # bounded by the reported count


def test_overreported_frame_count_rejects_explicit_nframes(
        tmp_path, monkeypatch):
    # Metadata reports 8 frames but only 4 decode: an explicit nframes=6 must
    # reject (exact-or-reject), never silently return 3.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    clip = tmp_path / "over.mp4"
    clip.write_bytes(b"x")
    _patch_av_frame_counts(monkeypatch, reported=8, decoded=4)
    with pytest.raises(ValueError, match="sampled positions decoded"):
        vs.sample_video(str(clip), nframes=6)
    # The default (fps-derived) path rejects too: trusting the short decode
    # would report a wrong effective fps into temporal MRoPE.
    with pytest.raises(ValueError, match="sampled positions decoded"):
        vs.sample_video(str(clip))


def test_error_messages_never_echo_data_urls(tmp_path):
    # A data: video source can be hundreds of MB of base64; error messages
    # must describe it boundedly instead of echoing the payload.
    pytest.importorskip("av")
    import base64 as b64
    import io
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<h", 0) * 800)
    payload = b64.b64encode(buf.getvalue()).decode()
    url = "data:video/wav;base64," + payload
    with pytest.raises(ValueError) as exc:
        vs.sample_video(url)
    assert payload[:48] not in str(exc.value)
    assert len(str(exc.value)) < 512


def test_pybind_video_rejects_non_finite_time_values():
    # Direct pybind API: inf fps / nan timestamps must be rejected at the
    # binding (the HTTP server validates earlier; this is defense-in-depth
    # for the raw C++/pybind surface).
    rt = _load_edgellm_runtime()
    if rt is None:
        pytest.skip("_edgellm_runtime pybind extension is not importable; "
                    "build with -DBUILD_PYTHON_BINDINGS=ON")
    np = pytest.importorskip("numpy")
    frames = np.zeros((2, 4, 4, 3), np.uint8)
    rt.load_video_from_array(frames, 2.0, [0.0, 0.5])  # sane baseline
    with pytest.raises(Exception, match="finite"):
        rt.load_video_from_array(frames, float("inf"), [0.0, 0.5])
    with pytest.raises(Exception, match="finite"):
        rt.load_video_from_array(frames, 2.0, [0.0, float("nan")])
    # The path-based loader shares the binding-level guard (fps is validated
    # before any file access, so a placeholder path suffices).
    with pytest.raises(Exception, match="finite"):
        rt.load_video_from_paths(["/nonexistent.png"], float("inf"))


def test_qwen3d_infeasible_profile_raises():
    # min=max=256 tokens with 400 frames of
    # 1280x720 leaves no feasible factor-grid shape — the C++ resize raises,
    # so the estimator must too instead of returning an out-of-profile size.
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=256,
                  max_image_tokens=256,
                  max_image_tokens_per_image=256)
    with pytest.raises(ValueError, match="no resized visual shape"):
        vs._estimate_qwen3d_video_tokens(400, 1280, 720, limits)


def test_qwen3_still_image_uses_3d_exact_shape(tmp_path):
    # Still images route through qwenSmartResize3D(isVideo=false) whose
    # factor-grid fallback finds the exact in-profile shape; the plain 2D
    # estimate would land outside a fixed narrow profile.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=4,
                  max_image_tokens=4,
                  max_image_tokens_per_image=4)
    assert vs._estimate_qwen3d_video_tokens(1,
                                            1024,
                                            1024,
                                            limits,
                                            is_video=False) == 4
    img = tmp_path / "still.mp4"
    _write_synthetic_clip(img, n_frames=1, size=1024, fps=1)
    assert vs.estimate_image_tokens(str(img), "qwen", limits) == 4


def test_cu_capacity_from_builder_recording():
    # An engine that records max_cu_seqlen_groups=512 must allow 66 frames
    # (33 temporal groups); the pre-recording formula (8192/4096 = 2 groups)
    # would truncate the same request to 4 frames.
    limits = dict(_QWEN3VL_LIMITS,
                  min_image_tokens=4096,
                  max_image_tokens=8192,
                  max_image_tokens_per_image=8192,
                  max_cu_seqlen_groups=512)
    n, _ = vs.clamp_nframes_to_profile(66,
                                       "qwen",
                                       640,
                                       360,
                                       limits,
                                       budget=8192)
    assert n == 66
    legacy = dict(limits)
    legacy.pop("max_cu_seqlen_groups")
    n_legacy, _ = vs.clamp_nframes_to_profile(66,
                                              "qwen",
                                              640,
                                              360,
                                              legacy,
                                              budget=8192)
    assert n_legacy == 4


def test_raw_per_group_token_cap(tmp_path):
    # do_resize=false media bypass the resize clamp but each temporal group
    # (or single raw image) is still one TRT carrier entry, so its spatial
    # tokens must fit max_image_tokens_per_image on all three raw intakes.
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    limits = {
        "model_type": "qwen2_5_vl",
        "min_image_tokens": 4,
        "max_image_tokens": 8192,
        "max_image_tokens_per_image": 512,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    over = tmp_path / "over.mp4"  # 532x756 -> 19*27 = 513 tokens/group
    _write_synthetic_clip_hw(over, n_frames=2, width=532, height=756, fps=1)
    ok = tmp_path / "ok.mp4"  # 504x756 -> 18*27 = 486 tokens/group
    _write_synthetic_clip_hw(ok, n_frames=2, width=504, height=756, fps=1)
    # Raw clip intake.
    with pytest.raises(ValueError, match="per-image"):
        vs.sample_video(str(over), frame_limits=limits, do_resize=False)
    _, _, _, est, _ = vs.sample_video(str(ok),
                                      frame_limits=limits,
                                      do_resize=False)
    assert est == 486
    # Raw pre-sampled frames intake.
    with pytest.raises(ValueError, match="per-image"):
        vs.load_video_buffer(_FakeRt(), {
            "type": "video",
            "frames": [str(over)] * 2,
            "fps": 1.0,
            "do_resize": False
        },
                             "qwen",
                             frame_limits=limits)
    buffer, est, _, _ = vs.load_video_buffer(_FakeRt(), {
        "type": "video",
        "frames": [str(ok)] * 2,
        "fps": 1.0,
        "do_resize": False
    },
                                             "qwen",
                                             frame_limits=limits)
    assert est == 486
    # Raw still image intake (group = the image itself).
    with pytest.raises(ValueError, match="per-image"):
        vs.estimate_image_tokens(str(over), "qwen", limits, do_resize=False)
    assert vs.estimate_image_tokens(str(ok), "qwen", limits,
                                    do_resize=False) == 486


def test_resolve_video_cfg_precedence():
    """Video sizing prefers vision_config, then top level, then the default;
    export and the runtime model build share this so their T agree."""
    pytest.importorskip("torch")
    from tensorrt_edgellm.models.nemotron_omni.modeling_nemotron_omni_visual import \
        resolve_video_cfg

    # vision_config wins over a stale top-level value (official checkpoint).
    cfg = {
        "video_temporal_patch_size": 2,
        "vision_config": {
            "video_temporal_patch_size": 4
        },
    }
    assert resolve_video_cfg(cfg, "video_temporal_patch_size", None) == 4
    # top-level fallback when vision_config omits the key (older artifact).
    assert resolve_video_cfg({"video_target_num_patches": 2048},
                             "video_target_num_patches", 1024) == 2048
    # default when neither level carries it.
    assert resolve_video_cfg({}, "video_maintain_aspect_ratio", True) is True
    # vision_config present but missing the key -> top level / default.
    assert resolve_video_cfg({"vision_config": {}}, "video_target_num_patches",
                             1024) == 1024
