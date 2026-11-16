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
Calculate Word Error Rate (WER) for ASR model output.

Input files
-----------
predictions_file (EdgeLLM inference output JSON):
    {
        "responses": [
            {"output_text": "<transcription>", "request_idx": <int>},
            ...
        ]
    }

dataset_file:
    {
        "requests": [
            {"reference": "<ground_truth_text>", "id": "<clip_id>", ...},
            ...
        ]
    }

Matching strategy
-----------------
If responses carry ``request_idx``, predictions and references are matched by
index so that missing or out-of-order responses are handled correctly.
Otherwise, positional (zip) matching is used.

Usage
-----
  python calculate_wer_score.py \\
      --predictions_file output/librispeech_predictions.json \\
      --dataset_file librispeech_clean_test/librispeech_clean_test.json
"""

import argparse
import json
import re
import string
import unicodedata

import jiwer

_ERROR_MESSAGE = "TensorRT Edge LLM cannot handle this request. Fails."

_SPLIT_WORDS = jiwer.ReduceToListOfListOfWords()

# Lead-in that chat models put before the transcript, e.g.
# ``The transcript of the audio is: "..."`` or ``Here is the transcription:``.
# Only these phrasings match, so ordinary speech is never cut.
_TRANSCRIPT_LEAD_IN_RE = re.compile(
    r"^(?:(?:here\s+is|here's|this\s+is)\s+)?(?:the\s+)?"
    r"(?:(?:correct|full|audio)\s+)?transcript(?:ion)?"
    r"(?:\s+of\s+(?:the\s+)?(?:(?:provided|given)\s+)?audio(?:\s+(?:clip|file))?)?"
    r"(?:\s+is)?\s*:\s*", re.IGNORECASE)


def normalize_text(text):
    """
    Normalize text for WER: strip reasoning blocks and special tokens,
    remove model-specific prefixes, lowercase, remove all punctuation,
    collapse whitespace.
    """
    # Drop ``<think>...</think>`` reasoning blocks emitted by reasoning models
    # (Qwen3-thinking, Nemotron-Reasoning, DeepSeek-R1, etc.). The reference
    # transcript never contains these, so leaving them inflates WER. ``re.DOTALL``
    # so the block can span newlines.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Drop chat / tokenizer special tokens (e.g. <|endoftext|>, <|im_end|>) so
    # WER scoring against a clean reference doesn't get inflated by sentinel tokens.
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"^language\s+\w+", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(c for c in text
                   if not unicodedata.category(c).startswith("P"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_wer(predictions, references):
    """
    Compute corpus-level WER across all prediction-reference pairs.

    Returns a dict with keys: wer (%), substitutions, deletions, insertions,
    hits, ref_words.
    """
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have the same length")

    refs_norm = [normalize_text(r) for r in references]
    # Strip the lead-in from predictions only; references are never wrapped.
    preds_norm = [
        normalize_text(_TRANSCRIPT_LEAD_IN_RE.sub("", p.strip(), count=1))
        for p in predictions
    ]

    output = jiwer.process_words(
        refs_norm,
        preds_norm,
        reference_transform=_SPLIT_WORDS,
        hypothesis_transform=_SPLIT_WORDS,
    )

    ref_words = output.hits + output.substitutions + output.deletions

    return {
        "wer": output.wer * 100,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "ref_words": ref_words,
    }


def _pair_by_request_idx(responses, requests):
    """
    Match responses to requests using the ``request_idx`` field.

    Returns (predictions, references, skipped_count, unmatched_count).
    """
    predictions, references = [], []
    skipped_count = unmatched_count = 0

    for resp in responses:
        output_text = resp.get("output_text", "")
        if output_text == _ERROR_MESSAGE:
            skipped_count += 1
            continue
        idx = resp.get("request_idx")
        if idx is None or idx < 0 or idx >= len(
                requests) or "reference" not in requests[idx]:
            unmatched_count += 1
            continue
        predictions.append(output_text)
        references.append(requests[idx]["reference"])

    return predictions, references, skipped_count, unmatched_count


def _pair_by_position(responses, requests):
    """
    Match responses to requests positionally (zip).

    Returns (predictions, references, skipped_count).
    """
    predictions, references = [], []
    skipped_count = 0

    for resp, req in zip(responses, requests):
        output_text = resp.get("output_text", "")
        if output_text == _ERROR_MESSAGE:
            skipped_count += 1
            continue
        predictions.append(output_text)
        references.append(req["reference"])

    return predictions, references, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER for ASR output")
    parser.add_argument("--predictions_file",
                        type=str,
                        required=True,
                        help="Path to EdgeLLM inference output JSON")
    parser.add_argument("--dataset_file",
                        type=str,
                        required=True,
                        help="Path to dataset JSON (requests[i].reference)")
    args = parser.parse_args()

    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        dataset_data = json.load(f)

    responses = predictions_data["responses"]
    requests = dataset_data["requests"]
    total_count = len(responses)

    has_request_idx = any("request_idx" in r for r in responses)
    unmatched_count = 0

    if has_request_idx:
        print("Matching predictions to references by 'request_idx' field.")
        predictions, references, skipped_count, unmatched_count = \
            _pair_by_request_idx(responses, requests)
        if unmatched_count > 0:
            print(f"Warning: {unmatched_count}/{total_count} responses had no "
                  "matching request_idx and were skipped.")
    else:
        print("No 'request_idx' fields found — using positional matching.")
        if len(responses) != len(requests):
            raise ValueError(
                f"Length mismatch: {len(responses)} responses vs "
                f"{len(requests)} requests. Provide 'request_idx' fields "
                "for reliable matching.")
        predictions, references, skipped_count = _pair_by_position(
            responses, requests)

    if skipped_count > 0:
        print(
            f"Skipped {skipped_count}/{total_count} entries with error messages."
        )

    valid_count = len(predictions)
    if valid_count == 0:
        print("No valid predictions to evaluate (all entries were errors).")
        return

    results = calculate_wer(predictions, references)

    print(f"\nWER: {results['wer']:.2f} %")
    print(f"Substitutions: {results['substitutions']}")
    print(f"Deletions:     {results['deletions']}")
    print(f"Insertions:    {results['insertions']}")
    print(f"Reference words: {results['ref_words']}")
    print(f"Evaluated:     {valid_count}/{total_count}")


if __name__ == "__main__":
    main()
