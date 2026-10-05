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
LibriSpeech ASR dataset converter for TensorRT Edge-LLM.

Source: HuggingFace ``openslr/librispeech_asr`` (test-clean split, 2620
utterances). Each utterance is written as an encoded audio file so the C++
runtime can decode the container and run its model-specific audio frontend.

Per-request JSON fields:

* ``messages[0].content`` -> ``[{type:audio,audio:'<dir>/audio/<id>.flac'},
                                {type:text,text:'Please transcribe the following audio.'}]``
* ``reference`` -> transcript.
* ``id`` / ``speaker_id`` / ``chapter_id`` for downstream slice analysis.
"""

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_eval_utils import (audio_extension_from_payload,
                              save_audio_array_to_flac, write_audio_bytes)
from datasets import Audio, Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset

_DEFAULT_PROMPT = "Please transcribe the following audio."


def select_speaker_prefixes(requests, sample_count):
    """Select near-equal per-speaker prefixes while preserving source order."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(requests):
        raise ValueError(
            f"sample_count ({sample_count}) exceeds request count "
            f"({len(requests)})")

    speaker_rows = defaultdict(list)
    for source_index, request in enumerate(requests):
        if "speaker_id" not in request:
            raise ValueError(
                f"Request at source index {source_index} has no speaker_id")
        speaker_rows[request["speaker_id"]].append((source_index, request))

    speakers = sorted(speaker_rows, key=int)
    if sample_count < len(speakers):
        raise ValueError(f"sample_count ({sample_count}) must cover all "
                         f"{len(speakers)} speakers")

    per_speaker, remainder = divmod(sample_count, len(speakers))
    selected_rows = []
    for speaker_index, speaker in enumerate(speakers):
        quota = per_speaker + (1 if speaker_index < remainder else 0)
        available = speaker_rows[speaker]
        if len(available) < quota:
            raise ValueError(
                f"Speaker {speaker!r} has {len(available)} rows, fewer than "
                f"the required prefix quota of {quota}")
        selected_rows.extend(available[:quota])

    selected_rows.sort(key=lambda row: row[0])
    return [request for _, request in selected_rows]


def create_librispeech_lite_dataset(input_file, output_dir, sample_count=524):
    """Create deterministic Lite data from an existing LibriSpeech Full JSON.

    Default 524 is 1/5 of test-clean (2620). Only the request list is sliced;
    audio paths stay as in Full, typically ``librispeech_clean_test/audio/<id>.flac``.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    source = json.loads(input_file.read_text(encoding="utf-8"))
    requests = source.get("requests")
    if not isinstance(requests, list):
        raise ValueError("Input dataset must contain a requests list")

    selected = select_speaker_prefixes(requests, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "librispeech_clean_test.json"
    lite = dict(source)
    lite["requests"] = selected
    dataset_path.write_text(json.dumps(lite, indent=4, ensure_ascii=False) +
                            "\n",
                            encoding="utf-8")
    return dataset_path


class LibriSpeechDataset(EdgeLLMDataset):
    """LibriSpeech ASR (test-clean) dataset converter.

    Audio is saved under ``<output_dir>/audio/<id>.<ext>`` and the manifest
    stores the absolute audio path.
    """

    def __init__(self,
                 dataset: Dataset,
                 config: DatasetConfig,
                 dataset_subdir: str = "librispeech_clean_test",
                 prompt: Optional[str] = None,
                 **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.dataset_subdir = dataset_subdir
        self.audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.prompt = prompt or _DEFAULT_PROMPT

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        return self.prompt

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        return ""

    @staticmethod
    def _utterance_id(data: Dict[str, Any]) -> str:
        """Compose LibriSpeech-style id: ``<speaker>-<chapter>-<utt_id>``."""
        if "id" in data and isinstance(data["id"], str):
            return data["id"]
        speaker = data.get("speaker_id") or data.get("speaker")
        chapter = data.get("chapter_id") or data.get("chapter")
        utt = data.get("file") or data.get("utterance_id") or id(data)
        return f"{speaker}-{chapter}-{utt}"

    def _manifest_audio_path(self, filename: str) -> str:
        return os.path.join(self.audio_dir, filename)

    def save_audio(self, data: Dict[str, Any]) -> Optional[str]:
        audio = data.get("audio")
        if audio is None:
            return None

        utt_id = self._utterance_id(data)
        if isinstance(audio, dict):
            encoded = audio.get("bytes")
            source_path = audio.get("path")
            if encoded:
                ext = audio_extension_from_payload(source_path, encoded)
                filename = f"{utt_id}{ext}"
                out_path = os.path.join(self.audio_dir, filename)
                if not os.path.exists(out_path):
                    write_audio_bytes(encoded, out_path)
                return self._manifest_audio_path(filename)

            if source_path and os.path.exists(source_path):
                ext = audio_extension_from_payload(source_path)
                filename = f"{utt_id}{ext}"
                out_path = os.path.join(self.audio_dir, filename)
                if not os.path.exists(out_path):
                    shutil.copyfile(source_path, out_path)
                return self._manifest_audio_path(filename)

            if "array" in audio:
                arr = np.asarray(audio["array"], dtype=np.float32)
                sr = audio.get("sampling_rate", 16000)
            else:
                return None
        else:
            arr = np.asarray(audio, dtype=np.float32)
            sr = 16000

        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        filename = f"{utt_id}.flac"
        out_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(out_path):
            save_audio_array_to_flac(arr, out_path, sample_rate=sr)
        return self._manifest_audio_path(filename)

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        for key in ("text", "transcription", "transcript", "reference"):
            v = data.get(key)
            if v:
                return v if isinstance(v, str) else str(v)
        return None


def _override_answers_to_references(out_json_path: str):
    """Rename LibriSpeech transcripts to the key consumed by WER scoring."""
    with open(out_json_path) as f:
        data = json.load(f)
    changed = False
    for request in data.get("requests", []):
        answer = request.pop("answer", None)
        if answer is not None:
            changed = True
            if "reference" not in request:
                request["reference"] = answer
    if changed:
        with open(out_json_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def convert_librispeech_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "openslr/librispeech_asr",
        output_dir: Union[str, os.PathLike] = "librispeech_clean_test",
        split: str = "test.clean",
        config_name: str = "clean",
        max_samples: Optional[int] = None,
        dataset_subdir: Optional[str] = None,
        output_filename: Optional[str] = None,
        prompt: Optional[str] = None) -> str:
    """Convert LibriSpeech test-clean into the TensorRT Edge-LLM JSON layout."""
    if dataset_subdir is None:
        dataset_subdir = "librispeech_clean_test"
    if output_filename is None:
        output_filename = f"{dataset_subdir}.json"

    print(f"Loading LibriSpeech from {dataset_name_or_dir} "
          f"(config={config_name}, split={split})")
    try:
        dataset = load_dataset(dataset_name_or_dir,
                               config_name,
                               split=split,
                               trust_remote_code=True)
    except TypeError:
        dataset = load_dataset(dataset_name_or_dir, config_name, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to first {len(dataset)} samples")

    converter = LibriSpeechDataset(dataset=dataset,
                                   config=config,
                                   output_dir=output_dir,
                                   dataset_subdir=dataset_subdir,
                                   prompt=prompt)
    out_path = converter.process_and_save_dataset(
        output_filename=output_filename)
    _override_answers_to_references(out_path)
    print(f"LibriSpeech dataset written to {out_path}")
    return out_path
