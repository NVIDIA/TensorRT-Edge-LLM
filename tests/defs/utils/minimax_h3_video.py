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
"""Checks of the text-to-video outputs that multistage_inference writes for MiniMax-H3.

A request file names a prompt, a size, a frame count and a seed per request.
The runtime writes `video_req<i>.rgb` (raw 8-bit RGB frames),
`audio_req<i>.wav` (16-bit stereo) and a predictions file. There is no
reference output, so the checks cover what the runtime promises: the sizes it
reports, media that is neither empty nor constant, and the same output for the
same seed.
"""

import hashlib
import math
import os
import wave
from typing import Any, Dict, List, Sequence, Tuple

MULTISTAGE_INFERENCE = "multistage_inference"
SAMPLE_RATE = 32000
CHANNELS = 2
FPS = 24
# A frame count is rounded up to 17n + 5 frames.
_FRAME_STEP = 17
_FRAME_BASE = 5
# Distinct byte values a picture needs before it counts as more than a flat fill.
_MIN_DISTINCT_BYTES = 16
_MIN_NONZERO_SAMPLE_RATIO = 0.5
# Distinct sample values a waveform needs before it counts as more than a
# constant level.
_MIN_DISTINCT_SAMPLES = 16
_SAMPLE_STRIDE = 997


def expected_frames(num_frames: int) -> int:
    """Return the frame count the runtime produces for *num_frames*."""
    if num_frames <= _FRAME_BASE:
        return _FRAME_BASE
    return _FRAME_BASE + _FRAME_STEP * math.ceil(
        (num_frames - _FRAME_BASE) / _FRAME_STEP)


def validate_command(executable: str, engine_dir: str) -> List[str]:
    """Check the stage graph against the runners of the binary."""
    return [executable, f"--config={engine_dir}", "--validate-config"]


def generate_command(executable: str, engine_dir: str, input_file: str,
                     output_file: str, output_dir: str) -> List[str]:
    """Generate every request of *input_file* with one model load."""
    return [
        executable,
        f"--config={engine_dir}",
        f"--input-file={input_file}",
        f"--output-file={output_file}",
        f"--output-video-dir={output_dir}",
        f"--output-audio-dir={output_dir}",
        "--dump-profile",
    ]


def media_paths(output: Dict[str, Any]) -> List[str]:
    """Return every media file the predictions name."""
    paths: List[str] = []
    for response in output.get("responses", []):
        for key in ("video_file", "audio_file"):
            if response.get(key):
                paths.append(response[key])
    return paths


def _wav_mismatches(path: str, label: str, response: Dict[str,
                                                          Any]) -> List[str]:
    try:
        with wave.open(path, "rb") as stream:
            channels = stream.getnchannels()
            rate = stream.getframerate()
            width = stream.getsampwidth()
            samples = stream.getnframes()
            data = stream.readframes(samples)
    except (wave.Error, EOFError, OSError) as error:
        return [f"{label}: {path} is not a readable wav file: {error}"]
    problems = []
    if (channels, rate, width) != (CHANNELS, SAMPLE_RATE, 2):
        problems.append(f"{label}: wav is {channels} channels at {rate} Hz, "
                        f"{width * 8}-bit; expected {CHANNELS} channels at "
                        f"{SAMPLE_RATE} Hz, 16-bit")
    if samples == 0:
        return problems + [f"{label}: wav has no samples"]
    if response.get("audio_samples") not in (None, samples):
        problems.append(f"{label}: predictions report "
                        f"{response['audio_samples']} samples, the wav has "
                        f"{samples}")
    values = memoryview(data).cast("h")
    nonzero = sum(1 for value in values if value) / len(values)
    if nonzero < _MIN_NONZERO_SAMPLE_RATIO:
        problems.append(f"{label}: only {nonzero:.1%} of the audio samples "
                        f"are non-zero")
    distinct = len(set(values))
    if distinct < _MIN_DISTINCT_SAMPLES:
        problems.append(f"{label}: audio has {distinct} distinct sample "
                        f"values, which is a constant level")
    return problems


def _video_mismatches(path: str, label: str, frames: int, height: int,
                      width: int) -> List[str]:
    expected_bytes = frames * height * width * 3
    size = os.path.getsize(path)
    if size != expected_bytes:
        return [
            f"{label}: video is {size} bytes; {frames} frames of "
            f"{width}x{height} RGB are {expected_bytes}"
        ]
    with open(path, "rb") as stream:
        sample = stream.read()[::_SAMPLE_STRIDE]
    if len(set(sample)) < _MIN_DISTINCT_BYTES:
        return [
            f"{label}: video has {len(set(sample))} distinct byte "
            f"values, which is a flat fill"
        ]
    return []


def _digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_key(request: Dict[str, Any]) -> Tuple[Any, ...]:
    return (request["prompt"], request.get("height"), request.get("width"),
            request.get("num_frames"), request.get("sampling", {}).get("seed"))


def _prompt_key(request: Dict[str, Any]) -> Tuple[Any, ...]:
    return _seed_key(request)[:4]


def output_mismatches(requests: Sequence[Dict[str, Any]],
                      output: Dict[str, Any]) -> List[str]:
    """Return what differs between the requests and the files written for them."""
    responses = output.get("responses", [])
    if len(responses) != len(requests):
        return [f"{len(responses)} responses for {len(requests)} requests"]
    problems: List[str] = []
    digests: List[Tuple[str, str]] = []
    for index, (request, response) in enumerate(zip(requests, responses)):
        label = f"request {index}"
        if not response.get("ok") or response.get("error"):
            problems.append(f"{label}: failed: {response.get('error')!r}")
            digests.append(("", ""))
            continue
        height, width = request["height"], request["width"]
        frames = expected_frames(request["num_frames"])
        reported = (response.get("video_frames"), response.get("video_height"),
                    response.get("video_width"), response.get("video_fps"))
        if reported != (frames, height, width, FPS):
            problems.append(f"{label}: predictions report frames, height, "
                            f"width, fps {reported}; expected "
                            f"{(frames, height, width, FPS)}")
        video, audio = response.get("video_file"), response.get("audio_file")
        if not video or not os.path.isfile(video):
            problems.append(f"{label}: video file {video!r} was not written")
            digests.append(("", ""))
            continue
        if not audio or not os.path.isfile(audio):
            problems.append(f"{label}: audio file {audio!r} was not written")
            digests.append(("", ""))
            continue
        problems += _video_mismatches(video, label, frames, height, width)
        problems += _wav_mismatches(audio, label, response)
        digests.append((_digest(video), _digest(audio)))

    # The same prompt, size and seed must give the same output; the same
    # prompt and size with another seed must not.
    for first in range(len(requests)):
        for second in range(first + 1, len(requests)):
            if not all(digests[first]) or not all(digests[second]):
                continue
            if _seed_key(requests[first]) == _seed_key(requests[second]):
                if digests[first] != digests[second]:
                    problems.append(f"requests {first} and {second} share "
                                    f"prompt, size and seed but differ")
            elif _prompt_key(requests[first]) == _prompt_key(requests[second]):
                if digests[first][0] == digests[second][0]:
                    problems.append(f"requests {first} and {second} use "
                                    f"different seeds but the same video")
    return problems
