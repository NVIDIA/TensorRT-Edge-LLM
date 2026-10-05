# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate unittests/resources/mel_parity_golden.txt for melSpectrogramTests.cpp.

The goldens are HF feature-extractor output, not a replica of the C++ math: each
matrix comes from the transformers class the corresponding
``cpp/runtime/melSpectrogram.cpp`` factory says it mirrors.

    whisper  -> WhisperFeatureExtractor    (Qwen3-Omni / Qwen3-ASR front end)
    parakeet -> ParakeetFeatureExtractor
    gemma4   -> Gemma4AudioFeatureExtractor

nemotron_asr has no transformers counterpart, so it stays pinned by difference
against parakeet in the test file rather than against a golden here.

Each matrix is written in the layout its C++ factory emits (whisper and gemma4
mel-major, parakeet time-major), so the test compares element for element.

The input waveform is not stored: it is a 32-bit LCG with no floating-point
state, so the test rebuilds it exactly. Keep the two definitions in step or
every matrix stops matching at once.

Run from the repo root with a venv that has transformers >= 5.x installed:
    ./venv/bin/python unittests/resources/gen_mel_parity_golden.py
"""

import numpy as np
import transformers

OUT_PATH = "unittests/resources/mel_parity_golden.txt"

# Prepended to the golden so a reviewer meeting a wall of numbers is told what
# they are before deciding whether to read any of them.
HEADER = """\
# Mel-spectrogram parity goldens for unittests/cpp/runtime/melSpectrogramTests.cpp.
#
# Each block is one HuggingFace feature extractor's output for a fixed waveform that
# the test rebuilds from an LCG. These are HF's numbers, not ours and not hand-written,
# and they are literals because the unit tests are host C++ and cannot run transformers.
# Format: `name rows cols`, then `rows` lines, in the layout that preset's factory emits.
#
# Regenerate with unittests/resources/gen_mel_parity_golden.py; do not edit by hand.
"""

# 60 ms at 16 kHz. Long enough that every framing mode produces both boundary
# frames and interior ones, short enough that the golden stays readable.
NUM_SAMPLES = 960
SAMPLE_RATE = 16000
NUM_MEL = 128


def make_waveform(count: int) -> np.ndarray:
    """Broadband deterministic noise, so every mel band receives energy.

    A tone would leave most of the filter bank at the log floor, where a wrong
    filter row is indistinguishable from a right one.

    Mirrored by `makeWaveform` in melSpectrogramTests.cpp, which is why the
    waveform is not stored in the golden. Only the top 24 bits of the state are
    used and the scale is a power of two, so both sides reach the same float32
    without libm and without a signed conversion.
    """
    state = 12345
    samples = np.empty(count, dtype=np.float32)
    for i in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        samples[i] = np.float32(float(state >> 8) * (2.0**-23) - 1.0)
    return samples


def whisper_golden(wave: np.ndarray) -> np.ndarray:
    extractor = transformers.WhisperFeatureExtractor(feature_size=NUM_MEL,
                                                     sampling_rate=SAMPLE_RATE,
                                                     hop_length=160,
                                                     n_fft=400,
                                                     chunk_length=30)
    # padding=False: the C++ extractor has TimePadding::kNone, so it must be
    # compared against the unpadded feature, not against Whisper's 30 s frame.
    out = extractor(wave,
                    sampling_rate=SAMPLE_RATE,
                    padding=False,
                    return_tensors="np")
    return np.asarray(out["input_features"])[0]  # [n_mel, T]


def parakeet_golden(wave: np.ndarray) -> np.ndarray:
    extractor = transformers.ParakeetFeatureExtractor(feature_size=NUM_MEL)
    out = extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="np")
    features = np.asarray(out["input_features"])[0]  # [T, n_mel]
    valid = int(np.asarray(out["attention_mask"])[0].sum())
    # HF emits one trailing padding frame; the C++ config drops it via
    # dropLastStftFrame, so the golden is the valid prefix.
    return features[:valid]


def gemma4_golden(wave: np.ndarray) -> np.ndarray:
    extractor = transformers.Gemma4AudioFeatureExtractor()
    out = extractor([wave], sampling_rate=SAMPLE_RATE, return_tensors="np")
    features = np.asarray(out["input_features"])[0]  # [T, n_mel]
    valid = int(np.asarray(out["input_features_mask"])[0].sum())
    return features[:valid].T  # C++ gemma4 factory emits MelLayout::kMelTime


def write_block(handle, name: str, values: np.ndarray) -> None:
    matrix = np.asarray(values, dtype=np.float32)
    handle.write(f"{name} {matrix.shape[0]} {matrix.shape[1]}\n")
    for row in matrix:
        handle.write(" ".join(f"{v:.9g}" for v in row) + "\n")


def main() -> None:
    wave = make_waveform(NUM_SAMPLES)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"{HEADER}# transformers {transformers.__version__}\n")
        write_block(handle, "whisper", whisper_golden(wave))
        write_block(handle, "parakeet", parakeet_golden(wave))
        write_block(handle, "gemma4", gemma4_golden(wave))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
