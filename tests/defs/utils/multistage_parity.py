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
"""Parity of multistage_inference with the binary a model case already runs.

Both binaries read the same input file on the same engines, so a request must
come out the same: the same RVQ codes for TTS, the same text otherwise, and for
speech, which both vocode in one pass, a waveform within 1 of each sample.

By default multistage_inference vocodes in windows while the talker runs, which
neither existing binary does, so a run in those windows is compared with the
same binary's whole-utterance run: the same codes and length, and in every 2 s
that either run makes audible, a waveform of the same shape and level.
"""

import json
import math
import operator
import os
import struct
import wave
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MULTISTAGE_INFERENCE = "multistage_inference"

# Flags of the existing run that multistage_inference reads the same way.
_MIRRORED_FLAGS = ("--inputFile=", "--checkpointDir=", "--batchSize=",
                   "--maxGenerateLength=", "--warmup=")
_EXCERPT_CHARACTERS = 60
# With --wholeUtterance both binaries vocode an utterance in one pass, and the
# existing binary truncates each float sample where multistage_inference rounds it.
_SAMPLE_ERROR = 1

_WHOLE_RUN = "the whole-utterance run"
_WINDOWED_RUN = "the windowed run"
# The default vocoder window: 25 codec frames of 1920 samples at 24 kHz.
_WINDOW_SECONDS = 2
# A 16-bit RMS of about -50 dBFS.
_SILENT_RMS = 100
# Measured windowed vs whole: correlation >= 0.857, RMS ratio 0.881-1.124.
_MIN_CORRELATION = 0.5
_RMS_RATIO = (0.5, 2)


def command(executable: str,
            existing_command: Sequence[str],
            engine_dir: str,
            output_file: str,
            audio_dir: Optional[str] = None,
            whole_utterance: bool = False) -> List[str]:
    """Run the stage graph in *engine_dir* as *existing_command* ran its engines.

    The input file, checkpoint, batch size, generation cap and warmup are the
    existing run's. The graph names every engine, and a talker finds its own
    tokenizer, so no engine or tokenizer directory is passed. *whole_utterance*
    vocodes each utterance in one pass, as the existing binaries do.
    """
    arguments = [
        executable, f"--config={engine_dir}", f"--outputFile={output_file}"
    ]
    if audio_dir is not None:
        arguments.append(f"--outputAudioDir={audio_dir}")
    if whole_utterance:
        arguments.append("--wholeUtterance")
    arguments.extend(argument for argument in existing_command[1:]
                     if argument.startswith(_MIRRORED_FLAGS))
    return arguments


def read_rvq_codes(path: str) -> Tuple[Tuple[int, int], List[int]]:
    """Shape and row-major values of the ``rvq_codes`` tensor of an RVQ dump.

    Raises:
        ValueError: The dump has no readable 2-D ``rvq_codes`` tensor, or it
            is not int32, the type both binaries write.
    """
    try:
        with open(path, "rb") as stream:
            (header_size, ) = struct.unpack("<Q", stream.read(8))
            entry = json.loads(stream.read(header_size))["rvq_codes"]
            data = stream.read()
        dtype, (frames, codes) = entry["dtype"], entry["shape"]
        begin, end = entry["data_offsets"]
        if dtype == "I32":
            values = struct.unpack(f"<{frames * codes}i", data[begin:end])
    except (KeyError, TypeError, ValueError, struct.error) as error:
        raise ValueError(
            f"{path} holds no readable 2-D rvq_codes tensor: {error}"
        ) from error
    if dtype != "I32":
        raise ValueError(f"{path} holds {dtype} rvq_codes, not I32")
    return (frames, codes), list(values)


def read_pcm16(path: str) -> Tuple[int, Tuple[int, ...]]:
    """Sample rate and samples of a 16-bit mono PCM WAV, the form both binaries write.

    Raises:
        ValueError: The file is not a 16-bit mono PCM WAV.
    """
    try:
        with wave.open(path, "rb") as stream:
            channels, width = stream.getnchannels(), stream.getsampwidth()
            rate = stream.getframerate()
            data = stream.readframes(stream.getnframes())
    except EOFError as error:
        raise ValueError(f"{path} ends inside its WAV header") from error
    except wave.Error as error:
        raise ValueError(f"{path} is not a PCM WAV: {error}") from error
    if (channels, width) != (1, 2):
        raise ValueError(f"{path} holds {channels}-channel {8 * width}-bit "
                         "samples, not 16-bit mono")
    return rate, struct.unpack_from(f"<{len(data) // 2}h", data)


def _pair(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    reference_name: str,
    candidate_name: str = MULTISTAGE_INFERENCE
) -> Tuple[List[tuple], List[str]]:
    """Responses to the same request, and why the others cannot be compared."""
    expected = {
        response["request_idx"]: response
        for response in reference["responses"]
    }
    expected_order = sorted(response["request_idx"]
                            for response in reference["responses"])
    actual_order = sorted(response["request_idx"]
                          for response in candidate["responses"])
    mismatches = []
    if actual_order != expected_order:
        mismatches.append(f"{candidate_name} answered requests "
                          f"{actual_order}, {reference_name} {expected_order}")
    pairs = []
    for actual in candidate["responses"]:
        index = actual["request_idx"]
        if actual.get("ok") is not True:
            mismatches.append(f"request {index}: {candidate_name} "
                              f"failed: {actual.get('error')}")
        elif index in expected:
            pairs.append((index, expected[index], actual))
    return pairs, mismatches


def _read_both(name: str,
               read: Callable[[str], Any],
               reference_dir: str,
               candidate_dir: str,
               reference_name: str,
               candidate_name: str = MULTISTAGE_INFERENCE) -> List[Any]:
    """*read* of file *name* of the reference run, then of the candidate run.

    Raises:
        ValueError: A run wrote no such file, or *read* rejects it.
    """
    contents = []
    for binary, directory in ((reference_name, reference_dir),
                              (candidate_name, candidate_dir)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise ValueError(f"{binary} wrote no {name}")
        contents.append(read(path))
    return contents


def _rvq_mismatch(index: int,
                  reference_audio_dir: str,
                  candidate_audio_dir: str,
                  reference_name: str,
                  candidate_name: str = MULTISTAGE_INFERENCE) -> Optional[str]:
    try:
        (expected_shape, expected), (actual_shape, actual) = _read_both(
            f"rvq_req{index}.safetensors", read_rvq_codes, reference_audio_dir,
            candidate_audio_dir, reference_name, candidate_name)
    except ValueError as error:
        return str(error)
    if expected_shape[1] == actual_shape[1]:
        for position, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                frame, code = divmod(position, expected_shape[1])
                return (f"RVQ codes differ at frame {frame}, code {code}: "
                        f"{reference_name} {left}, {candidate_name} {right}")
    if actual_shape != expected_shape:
        return (f"RVQ codes of shape {list(actual_shape)}, {reference_name} "
                f"{list(expected_shape)}")
    return None


def _sample_mismatch(index: int, reference_audio_dir: str,
                     candidate_audio_dir: str, reference_name: str,
                     reference_wav: str) -> Optional[str]:
    names = (reference_wav, f"audio_req{index}.wav")
    waveforms = []
    for binary, directory, name in ((reference_name, reference_audio_dir,
                                     names[0]),
                                    (MULTISTAGE_INFERENCE, candidate_audio_dir,
                                     names[1])):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            return f"{binary} wrote no {name}"
        try:
            waveforms.append(read_pcm16(path)[1])
        except ValueError as error:
            return str(error)
    expected, actual = waveforms
    if len(actual) != len(expected):
        return (f"{MULTISTAGE_INFERENCE}'s {names[1]} holds {len(actual)} "
                f"samples, {reference_name}'s {names[0]} {len(expected)}")
    for position, (left, right) in enumerate(zip(expected, actual)):
        if abs(right - left) > _SAMPLE_ERROR:
            return (f"{names[1]} differs from {reference_name}'s {names[0]} "
                    f"by {right - left} at sample {position}, more than "
                    f"{_SAMPLE_ERROR}")
    return None


def speech_mismatches(reference: Dict[str, Any], reference_audio_dir: str,
                      candidate: Dict[str, Any], candidate_audio_dir: str,
                      reference_name: str) -> List[str]:
    """Requests that *candidate* (multistage_inference) speaks differently.

    Each request needs the reference's RVQ codes and sample count, and a
    waveform of the reference's length whose every 16-bit sample is within 1
    of the reference's. Its talker must also stop at EOS, so two runs that both
    decode to the frame cap cannot pass as a match. A missing or unreadable
    dump or WAV is a mismatch.
    """
    pairs, mismatches = _pair(reference, candidate, reference_name)
    for index, expected, actual in pairs:
        for mismatch in (_rvq_mismatch(index, reference_audio_dir,
                                       candidate_audio_dir, reference_name),
                         _sample_mismatch(index, reference_audio_dir,
                                          candidate_audio_dir, reference_name,
                                          f"audio_req{index}.wav")):
            if mismatch:
                mismatches.append(f"request {index}: {mismatch}")
        if actual.get("audio_samples") != expected.get("audio_samples"):
            mismatches.append(
                f"request {index}: {actual.get('audio_samples')} audio "
                f"samples, {reference_name} {expected.get('audio_samples')}")
        talker = actual.get("stages", {}).get("talker", {})
        if talker.get("exit_reason") != "eos":
            mismatches.append(f"request {index}: the talker stopped with "
                              f"{talker.get('exit_reason')!r}, not at EOS")
    return mismatches


def omni_mismatches(reference: Dict[str, Any], reference_audio_dir: str,
                    candidate: Dict[str, Any], candidate_audio_dir: str,
                    reference_name: str) -> List[str]:
    """Requests that *candidate* (multistage_inference) answers or speaks differently.

    Each request needs what text_mismatches requires, a talker that stopped at
    EOS, and a waveform of the reference's length whose every 16-bit sample is
    within 1 of the reference's. A missing or unreadable WAV is a mismatch.
    """
    mismatches = text_mismatches(reference, candidate, reference_name)
    pairs, _ = _pair(reference, candidate, reference_name)
    for index, _, actual in pairs:
        # llm_inference names a reply's WAV by its batch, 0 at one request per batch.
        mismatch = _sample_mismatch(index, reference_audio_dir,
                                    candidate_audio_dir, reference_name,
                                    f"audio_req{index}_batch0.wav")
        if mismatch:
            mismatches.append(f"request {index}: {mismatch}")
        talker = actual.get("stages", {}).get("talker", {})
        if talker.get("exit_reason") != "eos":
            mismatches.append(f"request {index}: the talker stopped with "
                              f"{talker.get('exit_reason')!r}, not at EOS")
    return mismatches


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(map(operator.mul, left, right))


def _rms(samples: Sequence[int]) -> float:
    return math.sqrt(_dot(samples, samples) / len(samples))


def _correlation(left: Sequence[int], right: Sequence[int]) -> float:
    """Pearson correlation of two spans of one length, 0 when either is constant."""
    count = len(left)
    left_sum, right_sum = sum(left), sum(right)
    # Integer sums, scaled by count squared, give a constant span exactly 0.
    covariance = count * _dot(left, right) - left_sum * right_sum
    left_variance = count * _dot(left, left) - left_sum**2
    right_variance = count * _dot(right, right) - right_sum**2
    if not left_variance or not right_variance:
        return 0.0
    return covariance / (math.sqrt(left_variance) * math.sqrt(right_variance))


def _span_mismatches(name: str, rate: int, whole: Sequence[int],
                     windowed: Sequence[int]) -> List[str]:
    """2 s spans in which *windowed* departs from *whole*, a waveform as long."""
    mismatches = []
    span = _WINDOW_SECONDS * rate
    for start in range(0, len(whole), span):
        expected = whole[start:start + span]
        actual = windowed[start:start + span]
        loudness = _rms(expected), _rms(actual)
        if max(loudness) < _SILENT_RMS:
            continue
        where = (f"{_WINDOWED_RUN}'s {name} at {start / rate:.2f}-"
                 f"{(start + len(expected)) / rate:.2f} s")
        correlation = _correlation(expected, actual)
        if correlation < _MIN_CORRELATION:
            mismatches.append(
                f"{where} correlates {correlation:.2f} with {_WHOLE_RUN}'s, "
                f"below {_MIN_CORRELATION}")
        # A correlated span varies on both sides, so neither RMS is 0.
        elif not _RMS_RATIO[0] <= loudness[1] / loudness[0] <= _RMS_RATIO[1]:
            mismatches.append(
                f"{where} has {loudness[1] / loudness[0]:.2f} times the RMS "
                f"of {_WHOLE_RUN}'s, outside {list(_RMS_RATIO)}")
    return mismatches


def windowed_mismatches(whole: Dict[str, Any], whole_audio_dir: str,
                        windowed: Dict[str, Any],
                        windowed_audio_dir: str) -> List[str]:
    """Requests that multistage_inference speaks differently in its default vocoder windows.

    *whole* is the same binary's whole-utterance run of the same input. Each
    request needs its RVQ codes and WAV sample count, and a windowed talker that
    stopped at EOS. Both waveforms are cut into 2 s spans, the length of a
    default window, at the WAV's sample rate. A span where both runs are quieter
    than an RMS of 100 is skipped; any other needs a correlation of at least 0.5,
    which a constant span never has, and a windowed RMS of 0.5 to 2 times the
    whole one. Some request must be longer than one window, or no seam was
    compared. A missing or unreadable dump or WAV is a mismatch.
    """
    pairs, mismatches = _pair(whole, windowed, _WHOLE_RUN, _WINDOWED_RUN)
    crossed_a_seam = False
    for index, _, actual in pairs:
        problems = [
            _rvq_mismatch(index, whole_audio_dir, windowed_audio_dir,
                          _WHOLE_RUN, _WINDOWED_RUN)
        ]
        name = f"audio_req{index}.wav"
        try:
            (rate, expected), (_, samples) = _read_both(
                name, read_pcm16, whole_audio_dir, windowed_audio_dir,
                _WHOLE_RUN, _WINDOWED_RUN)
        except ValueError as error:
            problems.append(str(error))
        else:
            crossed_a_seam |= len(expected) > _WINDOW_SECONDS * rate
            if len(samples) != len(expected):
                problems.append(
                    f"{_WINDOWED_RUN}'s {name} holds {len(samples)} "
                    f"samples, {_WHOLE_RUN}'s {len(expected)}")
            else:
                problems.extend(_span_mismatches(name, rate, expected,
                                                 samples))
        talker = actual.get("stages", {}).get("talker", {})
        if talker.get("exit_reason") != "eos":
            problems.append(f"{_WINDOWED_RUN}'s talker stopped with "
                            f"{talker.get('exit_reason')!r}, not at EOS")
        mismatches.extend(f"request {index}: {problem}" for problem in problems
                          if problem)
    if not crossed_a_seam:
        mismatches.append(
            f"no request is longer than one {_WINDOW_SECONDS} s vocoder "
            "window, so no window seam was compared")
    return mismatches


def text_mismatches(reference: Dict[str, Any], candidate: Dict[str, Any],
                    reference_name: str) -> List[str]:
    """Requests that *candidate* (multistage_inference) answers differently.

    Each request needs the reference's output text, character for character,
    and its generated token count when both binaries report one.
    """
    pairs, mismatches = _pair(reference, candidate, reference_name)
    for index, expected, actual in pairs:
        expected_text = expected["output_text"]
        actual_text = actual["output_text"]
        if actual_text != expected_text:
            # commonprefix compares characters, not path components.
            start = len(os.path.commonprefix([expected_text, actual_text]))
            end = start + _EXCERPT_CHARACTERS
            mismatches.append(
                f"request {index}: output text differs from character "
                f"{start}: {reference_name} {expected_text[start:end]!r}, "
                f"{MULTISTAGE_INFERENCE} {actual_text[start:end]!r}")
        count = "generated_token_count"
        if (count in expected and count in actual
                and actual[count] != expected[count]):
            mismatches.append(
                f"request {index}: {actual[count]} generated tokens, "
                f"{reference_name} {expected[count]}")
    return mismatches
