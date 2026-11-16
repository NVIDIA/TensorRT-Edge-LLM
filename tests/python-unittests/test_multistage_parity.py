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
"""Contracts of the multistage_inference parity check of direct-builder cases."""

import collections
import copy
import json
import logging
import math
import os
import random
import struct
import wave

import pytest
from conftest import EnvironmentConfig
from defs import config as test_config
from defs import model_test_config
from defs import test_builder_pipeline as pipeline
from defs.utils import multistage_parity

_EXECUTABLES = {
    name: f"/build/bin/{name}"
    for name in ("llm_inference", "qwen3_tts_inference",
                 "multistage_inference")
}
_ROUGE = dict(accuracy=dict(required=["rouge1"], max_regression=0.30))
_WER = dict(accuracy=dict(required=["wer"], max_regression=0.30))
_Kind = collections.namedtuple("_Kind",
                               "model_id inputs outputs fields binary")
_KINDS = {
    "speech":
    _Kind("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", ["text"], ["audio"], {},
          "qwen3_tts_inference"),
    "text":
    _Kind("Qwen/Qwen3-0.6B", ["text"], ["text"], dict(checks=_ROUGE),
          "llm_inference"),
    "transcription":
    _Kind("Qwen/Qwen3-ASR-0.6B", ["text", "audio"], ["text"],
          dict(checks=_WER), "llm_inference"),
    "omni":
    _Kind("Qwen/Qwen3-Omni-30B-A3B-Instruct", ["text", "audio", "image"],
          ["text", "audio"], dict(checks=_ROUGE), "llm_inference"),
}
_ENGINE = dict(max_kv_cache_capacity=4096,
               max_input_len=4096,
               max_batch_size=1)
# RVQ codes of two requests: three and two frames of four codes.
_CODES = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
          [[13, 14, 15, 16], [17, 18, 19, 20]]]
_STRUCT_FORMATS = {"I32": "i", "U32": "I", "I64": "q"}


def _tone(length, period):
    return [
        round(8000 * math.sin(2 * math.pi * sample / period))
        for sample in range(length)
    ]


_WAVES = [_tone(480, 40), _tone(320, 64)]
# What the stand-in runtimes speak: codec frames of 1920 samples, a few more
# than one 25-frame vocoder window.
_SPOKEN_CODES = [[[request, frame, 0, 0] for frame in range(27 - request)]
                 for request in range(2)]
_SPOKEN_WAVES = [
    _tone(1920 * len(codes), period)
    for codes, period in zip(_SPOKEN_CODES, (40, 64))
]


def _document(kind, input_file, inference=None, **case):
    run = dict(name="run",
               scenario="generate",
               input_file=str(input_file),
               multistage_parity=True,
               **_KINDS[kind].fields)
    run.update(inference or {})
    engine = dict(_ENGINE)
    if "audio" in _KINDS[kind].inputs:
        engine.update(min_time_steps=100, max_time_steps=3000)
    if "image" in _KINDS[kind].inputs:
        engine.update(min_image_tokens=4, max_image_tokens=1024)
    direct = dict(name="direct",
                  mode="direct_builder",
                  platform="l0_e2e_a30",
                  quantized_model_id_or_dir="/models/checkpoint",
                  requirements=dict(min_sm=80, min_gpu_memory_gb=6),
                  engine=engine,
                  inference=[run])
    direct.update(case)
    # A case may only narrow the model's modalities, so an override is the model's too.
    return dict(schema_version=1,
                model_id=_KINDS[kind].model_id,
                inputs=case.get("inputs", _KINDS[kind].inputs),
                outputs=case.get("outputs", _KINDS[kind].outputs),
                cases=[direct])


def _direct_test(kind, input_file, inference=None):
    tests = model_test_config._parse_model_document(
        _document(kind, input_file, inference), "case")
    return next(test for test in tests
                if test.node == model_test_config._DIRECT_NODE)


def _environment(tmp_path):
    return EnvironmentConfig(llm_sdk_dir=".",
                             llm_models_dir=None,
                             edgellm_data_dir=None,
                             onnx_dir=None,
                             engine_dir=str(tmp_path / "engines"),
                             build_dir="/build",
                             test_log_dir=str(tmp_path / "logs"),
                             trt_package_dir=None)


def _write_rvq(path, frames, dtype="I32", shape=None):
    values = [code for frame in frames for code in frame]
    data = struct.pack(f"<{len(values)}{_STRUCT_FORMATS[dtype]}", *values)
    tensor = dict(dtype=dtype,
                  shape=shape or [len(frames), len(frames[0])],
                  data_offsets=[0, len(data)])
    header = json.dumps({"rvq_codes": tensor}).encode()
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(header)) + header + data)


def _write_wav(path, samples, channels=1, width=2, rate=24000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _input_file(tmp_path):
    path = tmp_path / "requests.json"
    requests = [
        dict(messages=[dict(role="assistant", content=text)])
        for text in ("Hello.", "Goodbye.")
    ]
    path.write_text(json.dumps({"requests": requests}))
    return path


class _Binaries:
    """Stand-ins for the build and both runtimes; each writes what its command names.

    *differ* ("codes", "text" or "audio") is what multistage_inference gets wrong
    for request 1 in every run. *windowed* ("tail" or "missing") is what its run
    without --wholeUtterance gets wrong for request 0: the last codec frame of
    the WAV, or the whole WAV. *talker* builds an Omni talker engine, so the
    runs speak.
    """

    def __init__(self,
                 differ=None,
                 skip_dump=False,
                 talker=False,
                 skip_wav=False,
                 windowed=None):
        self.differ = differ
        self.skip_dump = skip_dump
        self.talker = talker
        self.skip_wav = skip_wav
        self.windowed = windowed
        self.commands = []

    def __call__(self, command, name, timeout, env_config, test_logger):
        self.commands.append(command)
        binary = os.path.basename(command[0])
        if binary not in _EXECUTABLES:
            if self.talker:
                talker = os.path.join(
                    command[command.index("--engine-dir") + 1], "talker")
                os.makedirs(talker, exist_ok=True)
                open(os.path.join(talker, "llm.engine"), "wb").close()
            return {"success": True, "output": ""}
        flags = dict(
            argument.split("=", 1) for argument in command[1:]
            if "=" in argument)
        with open(flags["--inputFile"], encoding="utf-8") as stream:
            count = len(json.load(stream)["requests"])
        multistage = binary == "multistage_inference"
        windowed = multistage and "--wholeUtterance" not in command
        audio_dir = flags.get("--outputAudioDir")
        responses = []
        for index in range(count):
            frames = copy.deepcopy(_SPOKEN_CODES[index])
            text = f"answer {index}"
            if multistage and index == 1 and self.differ == "codes":
                frames[1][2] += 1
            if multistage and index == 1 and self.differ == "text":
                text += "!"
            if audio_dir:
                os.makedirs(audio_dir, exist_ok=True)
                samples = list(_SPOKEN_WAVES[index])
                if multistage and index == 1 and self.differ == "audio":
                    samples[7] += 2
                if windowed and index == 0 and self.windowed == "tail":
                    del samples[-1920:]
                # llm_inference names an Omni reply's WAV by its batch.
                wav = (f"audio_req{index}_batch0.wav" if binary
                       == "llm_inference" else f"audio_req{index}.wav")
                missing = ((multistage and self.skip_wav and index == 1)
                           or (windowed and self.windowed == "missing"
                               and index == 0))
                if not missing:
                    _write_wav(os.path.join(audio_dir, wav), samples)
                if not (multistage and self.skip_dump and index == 1):
                    _write_rvq(
                        os.path.join(audio_dir, f"rvq_req{index}.safetensors"),
                        frames)
            responses.append(
                dict(request_idx=index,
                     output_text=text,
                     generated_token_count=3,
                     audio_samples=1920 * len(frames),
                     ok=True,
                     stages=dict(talker=dict(exit_reason="eos"))))
        with open(flags["--outputFile"], "w", encoding="utf-8") as stream:
            json.dump({"responses": responses}, stream)
        return {"success": True, "output": ""}


def _run_case(tmp_path, monkeypatch, kind, binaries, inference=None):
    monkeypatch.setattr(pipeline, "_run", binaries)
    monkeypatch.setattr(pipeline, "_assert_component_engines",
                        lambda *args: None)
    monkeypatch.setattr(pipeline, "check_accuracy_with_dataset",
                        lambda *args: {})
    monkeypatch.setattr(pipeline, "check_declared_metrics", lambda *args: None)
    test_param = _direct_test(kind, _input_file(tmp_path), inference)
    pipeline._build_and_run(test_param,
                            _EXECUTABLES,
                            logging.getLogger(__name__),
                            _environment(tmp_path),
                            weightless_engine=False)


@pytest.mark.parametrize("kind, talker", [
    pytest.param("speech", False, id="speech"),
    pytest.param("text", False, id="text"),
    pytest.param("transcription", False, id="transcription"),
    pytest.param("omni", True, id="omni"),
    pytest.param("omni", False, id="omni-without-talker"),
])
def test_a_parity_case_runs_multistage_inference_on_its_engines(
        kind, talker, tmp_path, monkeypatch):
    """multistage_inference runs after the existing binary, on the case's engines.

    Where the existing run speaks, for TTS and for Omni when the build has a
    talker, it is also given an audio directory and vocodes each utterance
    whole, as the existing binaries do, then runs again into another audio
    directory with its default vocoder windows.
    """
    binaries = _Binaries(talker=talker)
    _run_case(tmp_path, monkeypatch, kind, binaries)
    build, existing, *runs = binaries.commands
    assert os.path.basename(existing[0]) == _KINDS[kind].binary
    engines = f"--config={build[build.index('--engine-dir') + 1]}"
    assert all(
        run[0] == _EXECUTABLES["multistage_inference"] and engines in run
        for run in runs)
    audio_dirs = [
        next((argument
              for argument in run if argument.startswith("--outputAudioDir=")),
             None) for run in runs
    ]
    whole = ["--wholeUtterance" in run for run in runs]
    if kind == "speech" or talker:
        assert whole == [True, False]
        assert None not in audio_dirs and audio_dirs[0] != audio_dirs[1]
    else:
        assert (whole, audio_dirs) == ([False], [None])


def test_a_case_without_parity_runs_only_the_existing_binary(
        tmp_path, monkeypatch):
    """A case that sets multistage_parity to false runs only the existing binary."""
    binaries = _Binaries()
    _run_case(tmp_path,
              monkeypatch,
              "text",
              binaries,
              inference={"multistage_parity": False})
    assert [os.path.basename(command[0])
            for command in binaries.commands[1:]] == ["llm_inference"]


@pytest.mark.parametrize("kind, differ, windowed", [
    ("speech", "codes", None),
    ("text", "text", None),
    ("omni", "audio", None),
    ("speech", None, "tail"),
    ("omni", None, "tail"),
    ("speech", "codes", "tail"),
])
def test_a_parity_case_fails_when_multistage_inference_differs(
        kind, differ, windowed, tmp_path, monkeypatch):
    """A request multistage_inference answers or vocodes in windows differently fails the case.

    One failure lists the requests of both comparisons, against the existing
    binary and against the whole-utterance run, and names the output files of
    every run.
    """
    binaries = _Binaries(differ=differ,
                         windowed=windowed,
                         talker=kind == "omni")
    with pytest.raises(pytest.fail.Exception) as failure:
        _run_case(tmp_path, monkeypatch, kind, binaries)
    message = str(failure.value)
    # differ breaks request 1, windowed request 0.
    assert ("\nrequest 1: " in message) == (differ is not None)
    assert ("\nrequest 0: " in message) == (windowed is not None)
    for command in binaries.commands[1:]:
        output = next(argument for argument in command
                      if argument.startswith("--outputFile="))
        assert output.split("=", 1)[1] in message


@pytest.mark.parametrize("kind, binaries, run, problem", [
    pytest.param(
        "speech",
        dict(skip_dump=True),
        "multistage",
        "request 1: multistage_inference wrote no rvq_req1.safetensors",
        id="dump"),
    pytest.param("omni",
                 dict(talker=True, skip_wav=True),
                 "multistage",
                 "request 1: multistage_inference wrote no audio_req1.wav",
                 id="omni-waveform"),
    pytest.param("speech",
                 dict(windowed="missing"),
                 "multistage-windowed",
                 "request 0: the windowed run wrote no audio_req0.wav",
                 id="windowed-waveform"),
])
def test_a_stale_file_cannot_stand_in_for_a_missing_one(
        kind, binaries, run, problem, tmp_path, monkeypatch):
    """A dump or WAV of an earlier run cannot stand in for one this run did not write."""
    test_param = _direct_test(kind, _input_file(tmp_path))
    stale = tmp_path / "logs" / f"{test_param.runtime_id}-{run}"
    stale.mkdir(parents=True)
    for index, frames in enumerate(_SPOKEN_CODES):
        _write_rvq(stale / f"rvq_req{index}.safetensors", frames)
        _write_wav(stale / f"audio_req{index}.wav", _SPOKEN_WAVES[index])
    with pytest.raises(pytest.fail.Exception, match=problem):
        _run_case(tmp_path, monkeypatch, kind, _Binaries(**binaries))


_MIRRORED = {
    "speech": ["--inputFile=/inputs/case.json", "--checkpointDir=/checkpoint"],
    "text": [
        "--inputFile=/inputs/case.json", "--checkpointDir=/checkpoint",
        "--batchSize=2", "--maxGenerateLength=64", "--warmup=1"
    ],
}
_MIRRORED["transcription"] = _MIRRORED["text"]
_MIRRORED["omni"] = _MIRRORED["text"]
# Flags the stage graph or the parity run supplies, or that only the existing run needs.
_NOT_MIRRORED = {
    "--engineDir", "--multimodalEngineDir", "--talkerEngineDir",
    "--code2wavEngineDir", "--tokenizerDir", "--outputFile",
    "--outputAudioDir", "--dumpProfile", "--enableAudioOutput"
}


@pytest.mark.parametrize("kind", ["speech", "text", "transcription", "omni"])
def test_the_parity_run_repeats_the_flags_of_the_existing_run(kind, tmp_path):
    """The parity run repeats the existing run's flags and adds none.

    Those are the input, checkpoint, batch size, generation cap and warmup flags.
    Every other flag of the existing run is one the stage graph or the parity run
    supplies, or one only the existing run needs, so a flag added to the existing
    run cannot be dropped unnoticed. A spoken run also vocodes whole utterances.
    """
    test_param = _direct_test(kind, _input_file(tmp_path))
    config = test_config.TestConfig.from_param_string(
        test_param,
        test_config.infer_checkpoint_export_model_type(test_param),
        test_config.TaskType.CHECKPOINT_BUILD,
        _environment(tmp_path),
        validate_environment=False)
    config.batch_size, config.output_seq_len, config.warmup = 2, 64, 1
    engines = tmp_path / "built"
    if kind == "omni":
        (engines / "talker").mkdir(parents=True)
        (engines / "talker" / "llm.engine").touch()
    existing = pipeline._runtime_command(config, "/checkpoint", str(engines),
                                         _EXECUTABLES, "/inputs/case.json")
    audio_dir = "/logs/case" if kind in ("speech", "omni") else None
    parity = multistage_parity.command(_EXECUTABLES["multistage_inference"],
                                       existing,
                                       str(engines),
                                       "/logs/case.json",
                                       audio_dir,
                                       whole_utterance=audio_dir is not None)
    expected = [f"--config={engines}", "--outputFile=/logs/case.json"]
    if audio_dir:
        expected += [f"--outputAudioDir={audio_dir}", "--wholeUtterance"]
    assert parity[0] == _EXECUTABLES["multistage_inference"]
    assert sorted(parity[1:]) == sorted(expected + _MIRRORED[kind])
    names = {argument.split("=", 1)[0] for argument in existing[1:]}
    mirrored = {flag.rstrip("=") for flag in multistage_parity._MIRRORED_FLAGS}
    assert names <= mirrored | _NOT_MIRRORED


@pytest.mark.parametrize("inference, case", [
    pytest.param({"multistage_parity": "yes"}, {}, id="not-a-boolean"),
    pytest.param({}, {"mode": "onnx"}, id="onnx"),
    pytest.param({"scenario": "few_layer"}, {}, id="few-layer"),
    pytest.param({}, {
        "outputs": ["text", "audio"],
        "engine": dict(_ENGINE, max_batch_size=2)
    },
                 id="batched-omni"),
    pytest.param({}, {"outputs": ["action"]}, id="vla"),
    pytest.param({}, {"spec_decode_config": "mtp"}, id="speculative"),
    pytest.param({}, {"engine": dict(_ENGINE, tp_size=2)},
                 id="tensor-parallel"),
])
def test_parity_is_refused_where_it_cannot_run(inference, case, tmp_path):
    """multistage_parity must be a boolean, on a case whose run it can repeat and compare."""
    document = _document("speech", tmp_path / "requests.json", inference,
                         **case)
    with pytest.raises(ValueError, match="multistage_parity"):
        model_test_config._parse_model_document(document, "case")


def _speech_run(directory, codes, dtype="I32"):
    os.makedirs(directory)
    for index, frames in enumerate(codes):
        if frames is not None:
            _write_rvq(os.path.join(directory, f"rvq_req{index}.safetensors"),
                       frames, dtype)
        _write_wav(os.path.join(directory, f"audio_req{index}.wav"),
                   _WAVES[index])
    return {
        "responses": [
            dict(request_idx=index,
                 audio_samples=1000 + index,
                 ok=True,
                 stages=dict(talker=dict(exit_reason="eos")))
            for index in range(len(codes))
        ]
    }


def _speech_mismatches(tmp_path,
                       candidate_codes=_CODES,
                       edit=None,
                       dtype="I32",
                       damage=None):
    """Mismatches of a candidate run, its responses passed to *edit* and its directory to *damage*."""
    reference = _speech_run(tmp_path / "reference", _CODES)
    candidate = _speech_run(tmp_path / "multistage", candidate_codes, dtype)
    if edit:
        edit(candidate["responses"])
    if damage:
        damage(tmp_path / "multistage")
    return multistage_parity.speech_mismatches(reference,
                                               str(tmp_path / "reference"),
                                               candidate,
                                               str(tmp_path / "multistage"),
                                               "qwen3_tts_inference")


def test_a_differing_code_names_its_request_frame_and_code(tmp_path):
    """One differing RVQ code fails its request, naming the frame, the code and both values."""
    codes = copy.deepcopy(_CODES)
    codes[1][1][2] = 99
    assert _speech_mismatches(tmp_path, codes) == [
        "request 1: RVQ codes differ at frame 1, code 2: "
        "qwen3_tts_inference 19, multistage_inference 99"
    ]


@pytest.mark.parametrize("frames, shape", [
    pytest.param(_CODES[0] + [[0, 0, 0, 0]], [4, 4], id="extra-frame"),
    pytest.param([frame[:3] for frame in _CODES[0]], [3, 3],
                 id="fewer-codes-per-frame"),
])
def test_codes_that_agree_as_far_as_they_go_still_need_the_same_shape(
        frames, shape, tmp_path):
    """A dump of another shape fails even when every code the two dumps share agrees."""
    assert _speech_mismatches(tmp_path, [frames, _CODES[1]]) == [
        f"request 0: RVQ codes of shape {shape}, qwen3_tts_inference [3, 4]"
    ]


@pytest.mark.parametrize("dtype", ["U32", "I64"])
def test_codes_of_another_type_are_a_mismatch(dtype, tmp_path):
    """RVQ codes stored as another type fail and name that type, also when their bytes are the same."""
    mismatches = _speech_mismatches(tmp_path, _CODES, dtype=dtype)
    assert [mismatch.split(": ")[0]
            for mismatch in mismatches] == ["request 0", "request 1"]
    assert all(
        mismatch.endswith(f"holds {dtype} rvq_codes, not I32")
        for mismatch in mismatches)


def _truncate(path):
    with open(path, "r+b") as stream:
        stream.truncate(os.path.getsize(path) - 4)


@pytest.mark.parametrize("corrupt", [
    pytest.param(_truncate, id="truncated"),
    pytest.param(lambda path: _write_rvq(path, _CODES[0], shape=[1, 3, 4]),
                 id="three-dimensional"),
])
def test_an_unreadable_dump_is_a_mismatch(corrupt, tmp_path):
    """An RVQ dump that is truncated or not 2-D fails its request instead of raising."""
    dump = tmp_path / "multistage" / "rvq_req0.safetensors"
    mismatches = _speech_mismatches(tmp_path,
                                    damage=lambda directory: corrupt(dump))
    assert len(mismatches) == 1
    assert mismatches[0].startswith(
        f"request 0: {dump} holds no readable 2-D rvq_codes tensor: ")


def test_a_missing_dump_is_a_mismatch(tmp_path):
    """A request without an RVQ dump fails instead of being skipped."""
    assert _speech_mismatches(tmp_path, [_CODES[0], None]) == [
        "request 1: multistage_inference wrote no rvq_req1.safetensors"
    ]


def test_a_dump_the_existing_run_did_not_write_is_a_mismatch(tmp_path):
    """A request whose existing run wrote no RVQ dump fails instead of raising."""
    reference = _speech_run(tmp_path / "reference", [_CODES[0], None])
    candidate = _speech_run(tmp_path / "multistage", _CODES)
    assert multistage_parity.speech_mismatches(
        reference, str(tmp_path / "reference"), candidate,
        str(tmp_path / "multistage"), "qwen3_tts_inference") == [
            "request 1: qwen3_tts_inference wrote no rvq_req1.safetensors"
        ]


def test_sample_counts_must_match(tmp_path):
    """A request whose waveform length differs fails even when its codes match."""

    def lengthen(responses):
        responses[0]["audio_samples"] += 480

    assert _speech_mismatches(tmp_path, _CODES, lengthen) == [
        "request 0: 1480 audio samples, qwen3_tts_inference 1000"
    ]


def test_a_tts_waveform_within_one_of_every_sample_passes(tmp_path):
    """A TTS waveform whose every sample is within 1 of the existing one's matches it."""
    generator = random.Random(0)
    jittered = [sample + generator.choice((-1, 0, 1)) for sample in _WAVES[0]]
    assert _speech_mismatches(
        tmp_path,
        damage=lambda directory: _write_wav(directory / "audio_req0.wav",
                                            jittered)) == []


def test_a_tts_sample_off_by_two_fails_its_request(tmp_path):
    """One sample 2 away from the existing one's fails its request, naming the sample and the difference."""
    samples = list(_WAVES[0])
    samples[9] += 2
    assert _speech_mismatches(
        tmp_path,
        damage=lambda directory: _write_wav(
            directory / "audio_req0.wav", samples)) == [
                "request 0: audio_req0.wav differs from qwen3_tts_inference's "
                "audio_req0.wav by 2 at sample 9, more than 1"
            ]


def test_sound_against_a_silent_existing_waveform_is_a_mismatch(tmp_path):
    """Sound where the existing run wrote silence fails at its first audible sample."""
    reference = _speech_run(tmp_path / "reference", _CODES)
    _write_wav(tmp_path / "reference" / "audio_req0.wav", [0] * len(_WAVES[0]))
    candidate = _speech_run(tmp_path / "multistage", _CODES)
    assert multistage_parity.speech_mismatches(
        reference, str(tmp_path / "reference"), candidate,
        str(tmp_path / "multistage"), "qwen3_tts_inference") == [
            "request 0: audio_req0.wav differs from qwen3_tts_inference's "
            f"audio_req0.wav by {_WAVES[0][1]} at sample 1, more than 1"
        ]


def test_silence_against_a_silent_existing_waveform_passes(tmp_path):
    """Silence where the existing run also wrote silence is not a mismatch."""
    reference = _speech_run(tmp_path / "reference", _CODES)
    candidate = _speech_run(tmp_path / "multistage", _CODES)
    for run in ("reference", "multistage"):
        _write_wav(tmp_path / run / "audio_req0.wav", [0] * len(_WAVES[0]))
    assert multistage_parity.speech_mismatches(reference,
                                               str(tmp_path / "reference"),
                                               candidate,
                                               str(tmp_path / "multistage"),
                                               "qwen3_tts_inference") == []


def _write_float_wav(path, samples):
    data = struct.pack(f"<{len(samples)}f", *samples)
    # IEEE float format, mono, 24 kHz, 32 bits per sample.
    fmt = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
    path.write_bytes(b"RIFF" + struct.pack("<I", 20 + len(fmt) + len(data)) +
                     b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt +
                     b"data" + struct.pack("<I", len(data)) + data)


@pytest.mark.parametrize("corrupt, problem", [
    pytest.param(lambda wav: wav.unlink(),
                 "multistage_inference wrote no audio_req0.wav",
                 id="missing"),
    pytest.param(lambda wav: wav.write_bytes(b""),
                 "{wav} ends inside its WAV header",
                 id="empty"),
    pytest.param(lambda wav: _write_float_wav(
        wav, [sample / 32768 for sample in _WAVES[0]]),
                 "{wav} is not a PCM WAV: unknown format: 3",
                 id="float-samples"),
    pytest.param(lambda wav: _write_wav(wav, _WAVES[0], channels=2),
                 "{wav} holds 2-channel 16-bit samples, not 16-bit mono",
                 id="stereo"),
    pytest.param(lambda wav: _write_wav(wav, _WAVES[0], width=1),
                 "{wav} holds 1-channel 8-bit samples, not 16-bit mono",
                 id="eight-bit"),
    pytest.param(lambda wav: _write_wav(wav, _WAVES[0][:240]),
                 "multistage_inference's audio_req0.wav holds 240 samples, "
                 "qwen3_tts_inference's audio_req0.wav 480",
                 id="shorter"),
    pytest.param(lambda wav: _write_wav(wav, _WAVES[0] + _WAVES[0][:240]),
                 "multistage_inference's audio_req0.wav holds 720 samples, "
                 "qwen3_tts_inference's audio_req0.wav 480",
                 id="longer"),
])
def test_a_missing_or_unreadable_waveform_is_a_mismatch(
        corrupt, problem, tmp_path):
    """A WAV that is missing, unreadable, not 16-bit mono or of another length fails its request instead of raising."""
    wav = tmp_path / "multistage" / "audio_req0.wav"
    mismatches = _speech_mismatches(tmp_path,
                                    damage=lambda directory: corrupt(wav))
    assert mismatches == [f"request 0: {problem.format(wav=wav)}"]


def test_a_talker_that_did_not_stop_at_eos_is_a_mismatch(tmp_path):
    """A talker stopped by the frame cap fails even when its codes match."""

    def run_on(responses):
        responses[1]["stages"]["talker"]["exit_reason"] = "max_length"

    assert _speech_mismatches(tmp_path, _CODES, run_on) == [
        "request 1: the talker stopped with 'max_length', not at EOS"
    ]


def _omni_run(directory, name):
    os.makedirs(directory)
    for index, samples in enumerate(_WAVES):
        _write_wav(os.path.join(directory, name.format(index)), samples)
    return {
        "responses": [
            dict(request_idx=index,
                 output_text=f"answer {index}",
                 generated_token_count=5,
                 ok=True,
                 stages=dict(talker=dict(exit_reason="eos")))
            for index in range(len(_WAVES))
        ]
    }


def _omni_mismatches(tmp_path, edit=None, damage=None):
    """Mismatches of a candidate Omni run, its responses passed to *edit* and *tmp_path* to *damage*."""
    reference = _omni_run(tmp_path / "reference", "audio_req{}_batch0.wav")
    candidate = _omni_run(tmp_path / "multistage", "audio_req{}.wav")
    if edit:
        edit(candidate["responses"])
    if damage:
        damage(tmp_path)
    return multistage_parity.omni_mismatches(reference,
                                             str(tmp_path / "reference"),
                                             candidate,
                                             str(tmp_path / "multistage"),
                                             "llm_inference")


def test_an_omni_reply_within_one_of_every_sample_passes(tmp_path):
    """An Omni reply whose every sample is within 1 of llm_inference's matches it."""
    generator = random.Random(0)

    def jitter(directory):
        _write_wav(
            directory / "multistage" / "audio_req0.wav",
            [sample + generator.choice((-1, 0, 1)) for sample in _WAVES[0]])

    assert _omni_mismatches(tmp_path, damage=jitter) == []


def test_an_omni_sample_off_by_two_fails_its_request(tmp_path):
    """One sample 2 away from llm_inference's fails its request, naming the sample and the difference."""
    samples = list(_WAVES[1])
    samples[5] -= 2
    mismatches = _omni_mismatches(
        tmp_path,
        damage=lambda directory: _write_wav(
            directory / "multistage" / "audio_req1.wav", samples))
    assert mismatches == [
        "request 1: audio_req1.wav differs from llm_inference's "
        "audio_req1_batch0.wav by -2 at sample 5, more than 1"
    ]


@pytest.mark.parametrize("corrupt, problem", [
    pytest.param(lambda directory:
                 (directory / "reference" / "audio_req0_batch0.wav").unlink(),
                 "llm_inference wrote no audio_req0_batch0.wav",
                 id="existing-missing"),
    pytest.param(lambda directory:
                 (directory / "multistage" / "audio_req0.wav").unlink(),
                 "multistage_inference wrote no audio_req0.wav",
                 id="missing"),
    pytest.param(
        lambda directory:
        (directory / "multistage" / "audio_req0.wav").write_bytes(b""),
        "{multistage}/audio_req0.wav ends inside its WAV header",
        id="empty"),
    pytest.param(lambda directory: _write_wav(
        directory / "multistage" / "audio_req0.wav", _WAVES[0][:240]),
                 "multistage_inference's audio_req0.wav holds 240 samples, "
                 "llm_inference's audio_req0_batch0.wav 480",
                 id="shorter"),
])
def test_an_omni_waveform_missing_unreadable_or_of_another_length_is_a_mismatch(
        corrupt, problem, tmp_path):
    """A reply WAV either run did not write, that cannot be read or that has another length fails its request instead of raising."""
    assert _omni_mismatches(tmp_path, damage=corrupt) == [
        f"request 0: {problem.format(multistage=tmp_path / 'multistage')}"
    ]


def test_an_omni_reply_is_compared_as_text_too(tmp_path):
    """An Omni reply whose audio matches still fails on its text or token count."""

    def reword(responses):
        responses[0]["output_text"] += "!"
        responses[1]["generated_token_count"] = 6

    assert _omni_mismatches(tmp_path, edit=reword) == [
        "request 0: output text differs from character 8: llm_inference '', "
        "multistage_inference '!'",
        "request 1: 6 generated tokens, llm_inference 5",
    ]


def test_an_omni_talker_that_did_not_stop_at_eos_is_a_mismatch(tmp_path):
    """An Omni talker stopped by the frame cap fails even when its audio matches."""

    def run_on(responses):
        responses[1]["stages"]["talker"]["exit_reason"] = "max_length"

    assert _omni_mismatches(tmp_path, edit=run_on) == [
        "request 1: the talker stopped with 'max_length', not at EOS"
    ]


# A WAV rate other than the binaries' 24 kHz, so 2 s is read from the file.
_RATE = 8000
_WINDOW = 2 * _RATE


def _square(length, half_period, amplitude=8000):
    """A square wave; two of different power-of-2 half periods do not correlate."""
    return [
        amplitude if sample // half_period % 2 == 0 else -amplitude
        for sample in range(length)
    ]


def _utterance():
    """3.5 windows of sound, each 2 s span uncorrelated with the others."""
    return [
        _square(_WINDOW, 1),
        _square(_WINDOW, 2),
        _square(_WINDOW, 4),
        _square(_WINDOW // 2, 8)
    ]


def _noise(span):
    """Noise at *span*'s level, of its sign at exactly half the samples, so uncorrelated."""
    signs = [1, -1] * (len(span) // 2)
    random.Random(0).shuffle(signs)
    return [sample * sign for sample, sign in zip(span, signs)]


def _distorted(span):
    """*span* at 0.86 of its level plus 0.54 of another: 56% RMS off, correlation 0.85."""
    return [
        round(0.86 * sample) + other
        for sample, other in zip(span, _square(len(span), 16, 4320))
    ]


def _windowed_mismatches(tmp_path, whole, windowed, damage=None):
    """windowed_mismatches of runs that speak request 0 as *whole* and *windowed*.

    Both speak request 1 the same, half a window long; *damage* gets the
    windowed run's directory and responses.
    """
    runs = []
    for name, spans in (("whole", whole), ("windowed", windowed)):
        (tmp_path / name).mkdir()
        waveforms = [[sample for span in spans for sample in span],
                     _square(_WINDOW // 2, 16)]
        for index, samples in enumerate(waveforms):
            _write_wav(tmp_path / name / f"audio_req{index}.wav",
                       samples,
                       rate=_RATE)
            _write_rvq(tmp_path / name / f"rvq_req{index}.safetensors",
                       _CODES[index])
        runs.append({
            "responses": [
                dict(request_idx=index,
                     ok=True,
                     stages=dict(talker=dict(exit_reason="eos")))
                for index in range(len(waveforms))
            ]
        })
    if damage:
        damage(tmp_path / "windowed", runs[1]["responses"])
    return multistage_parity.windowed_mismatches(runs[0],
                                                 str(tmp_path / "whole"),
                                                 runs[1],
                                                 str(tmp_path / "windowed"))


_AT = "request 0: the windowed run's audio_req0.wav at {} s "
_UNCORRELATED = _AT + "correlates {} with the whole-utterance run's, below 0.5"
_LEVEL = (_AT + "has {} times the RMS of the whole-utterance run's, "
          "outside [0.5, 2]")


@pytest.mark.parametrize("edit_whole, edit_windowed, expected", [
    pytest.param(None, None, [], id="identical"),
    pytest.param(None,
                 lambda spans: [_distorted(span) for span in spans], [],
                 id="off-by-56-percent-in-shape"),
    pytest.param(None,
                 lambda spans: spans[:3], [
                     "request 0: the windowed run's audio_req0.wav holds "
                     "48000 samples, the whole-utterance run's 56000"
                 ],
                 id="tail-dropped"),
    pytest.param(None,
                 lambda spans: spans[:2] + spans[1:], [
                     "request 0: the windowed run's audio_req0.wav holds "
                     "72000 samples, the whole-utterance run's 56000"
                 ],
                 id="window-repeated"),
    pytest.param(None,
                 lambda spans: [spans[0], [0] * _WINDOW] + spans[2:],
                 [_UNCORRELATED.format("2.00-4.00", "0.00")],
                 id="zeroed"),
    pytest.param(None,
                 lambda spans: [spans[0], spans[2], spans[1], spans[3]], [
                     _UNCORRELATED.format("2.00-4.00", "0.00"),
                     _UNCORRELATED.format("4.00-6.00", "0.00")
                 ],
                 id="swapped-with-the-next"),
    pytest.param(None,
                 lambda spans: spans[:3] + [_noise(spans[3])],
                 [_UNCORRELATED.format("6.00-7.00", "0.00")],
                 id="noise-in-the-tail"),
    pytest.param(None,
                 lambda spans: [[-sample for sample in spans[0]]] + spans[1:],
                 [_UNCORRELATED.format("0.00-2.00", "-1.00")],
                 id="negated"),
    pytest.param(lambda spans: [spans[0], _square(_WINDOW, 2, 90)] + spans[2:],
                 lambda spans: [spans[0], _square(_WINDOW, 4, 90)] + spans[2:],
                 [],
                 id="quiet-in-both-runs"),
    pytest.param(lambda spans: [spans[0], [0] * _WINDOW] + spans[2:],
                 None, [_UNCORRELATED.format("2.00-4.00", "0.00")],
                 id="silent-in-the-whole-run-only"),
    pytest.param(
        None,
        lambda spans: [spans[0], _square(_WINDOW, 2, 2400)] + spans[2:],
        [_LEVEL.format("2.00-4.00", "0.30")],
        id="quieter"),
    pytest.param(
        None,
        lambda spans: [spans[0], _square(_WINDOW, 2, 24000)] + spans[2:],
        [_LEVEL.format("2.00-4.00", "3.00")],
        id="louder"),
    pytest.param(
        lambda spans: spans[:1],
        lambda spans: spans[:1], [
            "no request is longer than one 2 s vocoder window, so no window "
            "seam was compared"
        ],
        id="one-window-long"),
])
def test_a_windowed_waveform_keeps_the_shape_and_level_of_the_whole_one(
        edit_whole, edit_windowed, expected, tmp_path):
    """A windowed waveform needs the whole one's length and, in every 2 s span either run makes audible, its shape and level.

    A failing span is named by its time and its correlation or RMS ratio. A
    span quiet in both runs is skipped, and a distortion that keeps the shape,
    56% of the RMS like the worst measured window, passes. Some request must
    be longer than one window.
    """
    whole = (edit_whole or list)(_utterance())
    windowed = (edit_windowed or list)(_utterance())
    assert _windowed_mismatches(tmp_path, whole, windowed) == expected


def _remove_request_1(directory, responses):
    for name in ("rvq_req1.safetensors", "audio_req1.wav"):
        (directory / name).unlink()


@pytest.mark.parametrize("damage, expected", [
    pytest.param(lambda directory, responses: _write_rvq(
        directory / "rvq_req1.safetensors", [_CODES[1][0], [17, 18, 99, 20]]),
                 [
                     "request 1: RVQ codes differ at frame 1, code 2: "
                     "the whole-utterance run 19, the windowed run 99"
                 ],
                 id="codes"),
    pytest.param(lambda directory, responses: responses[1].update(stages=dict(
        talker=dict(exit_reason="max_length"))), [
            "request 1: the windowed run's talker stopped with 'max_length', "
            "not at EOS"
        ],
                 id="talker-ran-on"),
    pytest.param(
        lambda directory, responses: responses[1].update(
            ok=False, error="stage 'vocoder': no memory"),
        ["request 1: the windowed run failed: stage 'vocoder': "
         "no memory"],
        id="failed"),
    pytest.param(lambda directory, responses: responses.pop(), [
        "the windowed run answered requests [0], the whole-utterance run "
        "[0, 1]"
    ],
                 id="unanswered"),
    pytest.param(_remove_request_1, [
        "request 1: the windowed run wrote no rvq_req1.safetensors",
        "request 1: the windowed run wrote no audio_req1.wav"
    ],
                 id="no-files"),
])
def test_a_windowed_run_needs_the_codes_and_eos_stop_of_the_whole_one(
        damage, expected, tmp_path):
    """The windowed run answers every request, with the whole run's RVQ codes, both files and a talker stopped at EOS."""
    assert _windowed_mismatches(tmp_path, _utterance(), _utterance(),
                                damage) == expected


def _text_run(texts):
    return {
        "responses": [
            dict(request_idx=index,
                 output_text=text,
                 generated_token_count=5,
                 ok=True) for index, text in enumerate(texts)
        ]
    }


def test_a_failed_request_reports_its_error_instead_of_a_comparison():
    """A request multistage_inference failed is reported with its error, not compared."""
    reference, candidate = _text_run(["a", "b"]), _text_run(["", "b"])
    candidate["responses"][0].update(ok=False, error="stage 'llm': no memory")
    mismatches = multistage_parity.text_mismatches(reference, candidate,
                                                   "llm_inference")
    assert mismatches == [
        "request 0: multistage_inference failed: stage 'llm': no memory"
    ]


def test_every_request_must_be_answered():
    """multistage_inference must answer exactly the requests the existing binary answered."""
    reference, candidate = _text_run(["a", "b", "c"]), _text_run(["a", "b"])
    mismatches = multistage_parity.text_mismatches(reference, candidate,
                                                   "llm_inference")
    assert mismatches == [
        "multistage_inference answered requests [0, 1], "
        "llm_inference [0, 1, 2]"
    ]


@pytest.mark.parametrize("actual, start", [
    pytest.param("The answer is 42. ", 17, id="trailing-space"),
    pytest.param("The answer is", 13, id="truncated"),
    pytest.param("The answer is 43.", 15, id="changed-character"),
])
def test_output_text_must_match_character_for_character(actual, start):
    """Text that differs anywhere, whitespace and truncation included, fails and names where."""
    expected = "The answer is 42."
    mismatches = multistage_parity.text_mismatches(_text_run([expected]),
                                                   _text_run([actual]),
                                                   "llm_inference")
    assert mismatches == [
        f"request 0: output text differs from character {start}: "
        f"llm_inference {expected[start:]!r}, "
        f"multistage_inference {actual[start:]!r}"
    ]


def test_token_counts_are_compared_where_both_binaries_report_them():
    """Token counts are compared only when both binaries report one."""
    reference, candidate = _text_run(["a", "b"]), _text_run(["a", "b"])
    candidate["responses"][0]["generated_token_count"] = 6
    del reference["responses"][1]["generated_token_count"]
    mismatches = multistage_parity.text_mismatches(reference, candidate,
                                                   "llm_inference")
    assert mismatches == ["request 0: 6 generated tokens, llm_inference 5"]
