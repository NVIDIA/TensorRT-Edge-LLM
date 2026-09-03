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
Accuracy alignment of the C++ mel-spectrogram extractor (whisper / parakeet)
against the HuggingFace ``transformers`` feature extractors that the
runtime is meant to mirror byte-for-byte.

Each accuracy test does:

    REF  = transformers.<FeatureExtractor>(audio)
    OURS = _edgellm_runtime.extract_mel_to_numpy(<bytes>, "<fe>")

and asserts max abs diff + cosine similarity under tight thresholds.

The smoke tests exercise the public
``_edgellm_runtime.load_audio_buffer_from_bytes`` path (miniaudio decode
→ host PCM) for plumbing coverage. Mel extraction itself happens inside
the audio runner; the standalone ``extract_mel_to_numpy`` binding above
is the test-only entrypoint used by the accuracy tests.

Usage:
    python3 -m pytest tests/python-unittests/test_preprocess_audio.py -v
"""
from __future__ import annotations

import importlib
import os
from typing import Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         ".."))
FIXTURE_FLAC = os.path.join(REPO_ROOT, "examples", "multimodal", "audio",
                            "6930-75918-0000.flac")


@pytest.fixture(scope="module")
def runtime():
    # The pybind module may be built by test_build_project_with_pybind earlier in the same
    # pytest process; refresh the import caches so a path entry that was empty
    # at startup is rescanned.
    importlib.invalidate_caches()
    try:
        import _edgellm_runtime  # type: ignore
    except ImportError:
        pytest.skip("_edgellm_runtime pybind extension is not importable; "
                    "build with -DBUILD_PYTHON_BINDINGS=ON")
    return _edgellm_runtime


@pytest.fixture(scope="module")
def audio_pair() -> Tuple[bytes, np.ndarray]:
    """Return (raw flac bytes, decoded float32 mono samples at 16 kHz)."""
    sf = pytest.importorskip("soundfile")
    if not os.path.isfile(FIXTURE_FLAC):
        pytest.skip(f"audio fixture missing: {FIXTURE_FLAC}")
    with open(FIXTURE_FLAC, "rb") as f:
        flac_bytes = f.read()
    samples, sr = sf.read(FIXTURE_FLAC, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    assert sr == 16000, f"fixture must be 16 kHz; got {sr}"
    return flac_bytes, samples


def _diff_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (max_abs, mean_abs, cos_sim) of a vs b. Assumes equal shape."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    a_flat, b_flat = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
    cos = float(np.dot(a_flat, b_flat) / denom)
    return float(diff.max()), float(diff.mean()), cos


# ---------------------------------------------------------------------------
# Accuracy: C++ pybind vs HuggingFace transformers feature extractors
# ---------------------------------------------------------------------------


def test_mel_accuracy_whisper(runtime, audio_pair):
    transformers = pytest.importorskip("transformers")
    flac_bytes, samples = audio_pair

    fe = transformers.WhisperFeatureExtractor(feature_size=128,
                                              sampling_rate=16000)
    # padding=False -> no time-pad to 30s, output T matches natural frame count
    # so we can compare directly to our extractor without truncation noise.
    out = fe(samples, sampling_rate=16000, return_tensors="np", padding=False)
    ref = np.asarray(out["input_features"][0], dtype=np.float32)  # [n_mel, T]

    ours = runtime.extract_mel_to_numpy(flac_bytes, "whisper")
    assert ours.shape == ref.shape, (
        f"whisper shape mismatch: ours {ours.shape} vs ref {ref.shape}")

    max_abs, mean_abs, cos = _diff_stats(ref, ours)
    print(
        f"[whisper]  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  cos={cos:.6f}"
    )
    # C++ FFT vs HF torch.stft: diff is in FP32 ULP territory.
    assert max_abs < 1e-3, f"whisper max_abs {max_abs:.3e} above threshold"
    assert cos > 0.9999, f"whisper cos {cos:.6f} below threshold"


def test_mel_accuracy_parakeet(runtime, audio_pair):
    transformers = pytest.importorskip("transformers")
    # ParakeetFeatureExtractor declares a torch backend requirement
    # (requires_backends), unlike WhisperFeatureExtractor's numpy path.
    pytest.importorskip("torch")
    flac_bytes, samples = audio_pair

    # NVIDIA Nemotron-Omni uses num_mel_bins=128 (NOT the HF library default 80).
    fe = transformers.ParakeetFeatureExtractor(feature_size=128,
                                               sampling_rate=16000)
    out = fe(samples, sampling_rate=16000, return_tensors="np", padding=False)
    ref = np.asarray(out["input_features"][0], dtype=np.float32)  # [T, n_mel]

    ours = runtime.extract_mel_to_numpy(flac_bytes, "parakeet")
    # HF Parakeet emits T_natural frames but zeros the trailing frames past
    # ``features_lengths = audio_len // hop`` (mask multiply). Our extractor
    # drops them entirely. Trim ref to ours' shape so we compare the active
    # frames only -- the trimmed ref frames are the masked zeros.
    if ref.shape[0] == ours.shape[0] + 1:
        ref = ref[:ours.shape[0]]

    max_abs, mean_abs, cos = _diff_stats(ref, ours)
    print(
        f"[parakeet] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  cos={cos:.6f}"
    )
    # C++ FFT + additive log floor vs HF Parakeet: diff at the ~1e-3 scale.
    assert max_abs < 1e-2, f"parakeet max_abs {max_abs:.3e} above threshold"
    assert cos > 0.9999, f"parakeet cos {cos:.6f} below threshold"


# ---------------------------------------------------------------------------
# End-to-end smoke: full pybind path (decode + extract + GPU upload)
# ---------------------------------------------------------------------------


def test_load_audio_buffer_from_bytes_smoke(runtime, audio_pair):
    flac_bytes, _ = audio_pair
    audio = runtime.load_audio_buffer_from_bytes(flac_bytes)
    assert audio.sample_rate == 16000


def test_load_audio_buffer_from_bytes_rejects_garbage(runtime):
    with pytest.raises(Exception):
        runtime.load_audio_buffer_from_bytes(b"not-a-real-audio-file")


def test_extract_mel_to_numpy_unknown_fe(runtime, audio_pair):
    flac_bytes, _ = audio_pair
    with pytest.raises(Exception):
        runtime.extract_mel_to_numpy(flac_bytes, "this_fe_does_not_exist")


# ---------------------------------------------------------------------------
# Basic sanity: shape + dtype contract per FE. No HF dep -- fast smoke for
# the host pipeline that runs even without ``transformers`` available.
# ---------------------------------------------------------------------------


def test_extract_mel_to_numpy_whisper_shape(runtime, audio_pair):
    flac_bytes, _ = audio_pair
    mel = runtime.extract_mel_to_numpy(flac_bytes, "whisper")
    assert mel.ndim == 2
    assert mel.shape[0] == 128
    assert mel.shape[1] > 0
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()


def test_extract_mel_to_numpy_parakeet_shape(runtime, audio_pair):
    flac_bytes, _ = audio_pair
    mel = runtime.extract_mel_to_numpy(flac_bytes, "parakeet")
    assert mel.ndim == 2
    assert mel.shape[1] == 128
    assert mel.shape[0] > 0
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()


@pytest.mark.parametrize("container,subtype", [
    ("WAV", "PCM_16"),
    ("FLAC", "PCM_16"),
    ("MP3", None),
])
@pytest.mark.parametrize("fe_type", ["whisper", "parakeet"])
def test_extract_mel_round_trip_synthetic(runtime, fe_type, container,
                                          subtype):
    """Round-trip: synthetic sine -> container bytes -> C++ mel for each FE.

    Exercises the miniaudio decode path × every FE × every supported
    container, without depending on bundled audio fixtures.
    """
    sf = pytest.importorskip("soundfile")
    if container not in sf.available_formats():
        pytest.skip(f"soundfile build lacks {container} support")
    import io
    sr = 16000
    n = sr  # 1 second
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(n) / sr)).astype(
        np.float32)
    buf = io.BytesIO()
    if subtype is not None:
        sf.write(buf, audio, sr, format=container, subtype=subtype)
    else:
        sf.write(buf, audio, sr, format=container)
    encoded = buf.getvalue()
    assert len(encoded) > 0, f"{container} encode produced no bytes"

    mel = runtime.extract_mel_to_numpy(encoded, fe_type)
    assert mel.ndim == 2
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()
    # Per-FE shape contract.
    if fe_type == "whisper":
        assert mel.shape[0] == 128  # [n_mel=128, T]
    elif fe_type == "parakeet":
        assert mel.shape[1] == 128  # [T, n_mel=128]


# ---------------------------------------------------------------------------
# Source resolution (pure python, no runtime): OpenAI audio content -> local
# path / bounded bytes.
# ---------------------------------------------------------------------------


def _resolve(item):
    import sys
    sys.path.insert(0, REPO_ROOT)
    from experimental.server.media import audio_preprocess
    return audio_preprocess.resolve_audio_message(item)


def test_resolve_input_audio_base64():
    import base64
    raw = b"RIFF0000WAVEfmt "
    out = _resolve({
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(raw).decode(),
            "format": "wav"
        },
    })
    assert out == raw


def test_resolve_input_audio_missing_data():
    with pytest.raises(ValueError):
        _resolve({"type": "input_audio", "input_audio": {}})


@pytest.mark.parametrize("url", ["http://h/a.wav", "https://h/a.wav"])
def test_resolve_audio_url_fetches_remote(url, monkeypatch):
    import sys
    sys.path.insert(0, REPO_ROOT)
    from experimental.server.media import audio_preprocess

    calls = []
    monkeypatch.setattr(
        audio_preprocess, "fetch_remote_media",
        lambda source, kind, limit: calls.append(
            (source, kind, limit)) or b"wav")
    assert audio_preprocess.resolve_audio_message({
        "type": "audio_url",
        "audio_url": {
            "url": url
        }
    }) == b"wav"
    assert calls == [(url, "audio", audio_preprocess.MAX_AUDIO_UPLOAD_BYTES)]


def test_resolve_audio_url_data_and_file(tmp_path):
    import base64
    raw = b"ID3\x03"
    out = _resolve({
        "type": "audio_url",
        "audio_url": {
            "url": "data:audio/mpeg;base64," + base64.b64encode(raw).decode()
        },
    })
    assert out == raw
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert _resolve({
        "type": "audio_url",
        "audio_url": {
            "url": f"file://{f}"
        }
    }) == str(f)


def test_resolve_audio_url_rejects_non_base64_data():
    with pytest.raises(ValueError):
        _resolve({
            "type": "audio_url",
            "audio_url": {
                "url": "data:audio/wav,notb64"
            }
        })


def test_resolve_unknown_type_raises():
    with pytest.raises(ValueError):
        _resolve({"type": "image"})
