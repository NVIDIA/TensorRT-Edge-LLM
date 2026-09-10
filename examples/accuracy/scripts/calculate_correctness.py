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

import argparse
import json
import re
from collections import defaultdict

MULTI_CHOICE_LETTERS = "ABCDEFGHIJ"


def _apply_byte_fallback(text):
    """
    Apply SentencePiece ByteFallback decoding: convert <0xNN> sequences to actual bytes.

    SentencePiece byte_fallback tokenizers use <0xNN> notation for individual byte tokens.
    The ByteFallback decoder step converts these back to raw bytes. If the C++ tokenizer
    does not implement this step, the literal text "<0xE2><0x96><0x81>" appears in output
    instead of the UTF-8 character (U+2581).
    """

    def hex_to_byte(match):
        return bytes([int(match.group(1), 16)])

    # Only apply when consecutive <0xNN> tokens exist (SentencePiece ByteFallback
    # always emits multi-byte runs for non-ASCII). Single isolated <0xNN> in normal
    # text (e.g., hex discussions) is left unchanged to avoid false positives.
    byte_pattern = re.compile(r"<0x([0-9A-Fa-f]{2})>")
    if not re.search(r"<0x[0-9A-Fa-f]{2}><0x[0-9A-Fa-f]{2}>", text):
        return text

    # Process the string: convert <0xNN> runs to bytes, keep other text as-is
    parts = []
    last_end = 0
    byte_buffer = bytearray()

    for match in byte_pattern.finditer(text):
        start = match.start()
        if start > last_end:
            # Flush byte buffer before non-hex text
            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer = bytearray()
            parts.append(text[last_end:start])
        byte_buffer.append(int(match.group(1), 16))
        last_end = match.end()

    # Flush remaining
    if byte_buffer:
        parts.append(byte_buffer.decode("utf-8", errors="replace"))
    if last_end < len(text):
        parts.append(text[last_end:])

    return "".join(parts)


def clean_text(text):
    """
    Clean and normalize text by stripping whitespace and removing punctuation from ends.
    Args:
        text: Input text string.
    Returns:
        Cleaned text string.
    """

    # Apply SentencePiece ByteFallback decoding (<0xNN> -> bytes) only when
    # the text also contains the SentencePiece word-boundary marker '▁',
    # confirming a SentencePiece tokenizer produced this output.
    if "\u2581" in text:
        text = _apply_byte_fallback(text)
        text = text.replace("\u2581", " ")

    # Drop reasoning blocks emitted by reasoning models (Qwen3-thinking,
    # Nemotron-Reasoning, DeepSeek-R1, etc.) so an MCQ answer like
    # ``<think>...</think>\nC`` still scores against the clean reference.
    # The opening tag is optional because some chat templates inject it into
    # the prompt prefix, so the model output contains only ``</think>``.
    if "</think>" in text:
        text = re.sub(r"(?:<think>)?.*?</think>", "", text, flags=re.DOTALL)
    # Gemma4 thinking format: ``<|channel>thought\n...\n<|channel>response\n...``
    # Keep only the response portion after the last ``<|channel>response`` marker.
    if "<|channel>response" in text:
        text = text[text.rfind("<|channel>response") +
                    len("<|channel>response"):]
    elif "<|channel>thought" in text:
        # DiffusionGemma/Gemma4 can emit a thought channel followed directly by
        # the short answer without an explicit response marker, e.g.
        # ``<|channel>thought\n<channel|>B<turn|>``.
        if "<channel|>" in text:
            text = text[text.rfind("<channel|>") + len("<channel|>"):]
        else:
            # Truncated thinking (hit max-length): inspect only the tail for
            # the conservative answer parser below.
            text = text[-500:]
    # Drop chat / tokenizer special tokens (e.g. <|endoftext|>, <|im_end|>, <turn|>,
    # <end_of_turn>) so MCQ output like "C<|im_end|>" or "C<turn|>" still scores.
    # Replace with space (not empty) so "C<turn|>The..." → "C The..." preserves boundary.
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"<[a-z_]+\|>", " ", text)
    text = re.sub(r"<(?:eos|bos|pad|unk|mask|start_of_turn|end_of_turn)>", " ",
                  text)
    text = text.strip().strip("().,")
    return text


def _strip_markdown(text):
    """Strip markdown emphasis that gets in the way of letter matching."""
    return text.replace("**", "").replace("__", "").replace("`", "")


def parse_multi_choice_response(text):
    """
    Parse multiple choice answer from text that may be in various formats.
    Handles "A. xxx", "A", "(A)", "**Answer: A**", "the answer is A", or just
    returns the first letter if it's A-J.

    Args:
        text: Input text string potentially containing a multiple choice answer.
    Returns:
        Single letter (A-J) if found, otherwise returns the original cleaned text.
    """
    text = _strip_markdown(text.strip())

    # If text is already just a single choice letter, return it.
    if len(text) == 1 and text in MULTI_CHOICE_LETTERS:
        return text

    # A qualifier may sit between the answer noun and the copula: "the answer
    # *to the final question* is: C". Spelled out rather than a wildcard span,
    # which would also match "... (or there is a ...)" and score that "a".
    _QUALIFIER = (r"(?:\s+(?:to|for|of)\s+(?:the\s+)?"
                  r"(?:final|last|above|following)\s+question)?")

    # ``is`` and ``:`` are independently optional: ``(?:is|:)?`` makes them
    # mutually exclusive and drops "The correct answer is: B".
    explicit_answer_patterns = [
        r"\b(?:final\s+answer|answer|correct\s+(?:answer|option)|correct\s+choice|letter|matches?\s+option)"
        + _QUALIFIER + r"\s*(?:is)?\s*:?\s*\(?([A-J])\b",
        r"\boption\s*(?:is\s*:?|:)\s*\(?([A-J])\b",
        r"\boption\s+\(?([A-J])\)?(?:[\.\)]\s*|$)",
        r"\b(?:the\s+)?(?:answer|letter)" + _QUALIFIER +
        r"\s+is\s*:?\s*\(?([A-J])\b",
        r"\bresult\s*(?:is)?\s*:?\s*\(?([A-J])\b",
    ]
    for pattern in explicit_answer_patterns:
        matches = list(re.finditer(pattern, text,
                                   re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    # Try to match pattern like "A." or "A)" or "(A)" at the start.
    match = re.match(r'^[\(]?([A-J])[\.\):\s]', text)
    if match:
        return match.group(1)

    # Handle short responses that start with a valid letter (e.g. "A\n" or "B.").
    if len(text) <= 3 and text and text[0] in MULTI_CHOICE_LETTERS:
        return text[0]

    # Search the tail for an explicit final answer/option/letter marker. Some
    # thinking outputs finish with a single letter on the final line instead of
    # emitting a response-channel marker. Keep these patterns conservative: broad
    # connectors like "so X" can appear in rejected alternatives such as
    # "So B is wrong, leaving C."
    tail = text[-800:]
    tail_patterns = [
        r"(?:^|[\n\r])\s*([A-J])\s*(?:\.\s*)?$",
    ]
    for pattern in tail_patterns:
        matches = list(re.finditer(pattern, tail,
                                   re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    # If no match found, return the original text
    return text


def is_correct(pred, ref):
    """
    Determines correctness between a prediction and a reference.
    Handles text normalization and multiple-choice parsing.
    
    Args:
        pred: The predicted text from the model.
        ref: The reference/ground-truth answer.
    Returns:
        Boolean indicating if the prediction matches the reference.
    """
    pred_clean = clean_text(pred)
    ref_clean = clean_text(ref)

    if ref_clean in MULTI_CHOICE_LETTERS:
        pred_clean = parse_multi_choice_response(pred_clean)

    return pred_clean == ref_clean


def calculate_correctness(predictions, references):
    """
    Compute correctness score between predictions and references.
    Args:
        predictions: List of predictions.
        references: List of references.
    Returns:
        Tuple of (correct_count, total_count).
    """
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have the same length")

    correct_count = 0
    total_count = len(predictions)

    for pred, ref in zip(predictions, references):
        if is_correct(pred, ref):
            correct_count += 1

    return correct_count, total_count


def calculate_subject_accuracy(predictions, references, subjects):
    """
    Compute subject-specific accuracy scores.
    Args:
        predictions: List of predictions.
        references: List of references.
        subjects: List of subjects corresponding to each prediction/reference.
    Returns:
        Dictionary with subject-specific accuracy scores.
    """
    if len(predictions) != len(references) or len(predictions) != len(
            subjects):
        raise ValueError(
            "Predictions, references, and subjects must have the same length")

    subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred, ref, subject in zip(predictions, references, subjects):
        subject_stats[subject]['total'] += 1
        if is_correct(pred, ref):
            subject_stats[subject]['correct'] += 1

    # Calculate accuracy for each subject
    subject_accuracy = {}
    for subject, stats in subject_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats[
            'total'] > 0 else 0.0
        subject_accuracy[subject] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }

    return subject_accuracy


def main():
    """Main function to calculate correctness score from command line arguments."""
    parser = argparse.ArgumentParser(
        description=
        "Calculate correctness score between predictions and answers")
    parser.add_argument("--predictions_file",
                        type=str,
                        required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--answers_file",
                        type=str,
                        required=True,
                        help="Path to answers JSON file")

    args = parser.parse_args()

    # Load JSON files
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(args.answers_file, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)

    # Error message to skip
    error_message = "TensorRT Edge LLM cannot handle this request. Fails."

    # Extract predictions and answers, filtering out error messages
    predictions = []
    answers = []
    subjects = []
    skipped_count = 0
    total_count = 0

    for response, request in zip(predictions_data["responses"],
                                 answers_data["requests"]):
        total_count += 1
        output_text = response["output_text"]

        # Skip entries with error messages
        if output_text == error_message:
            skipped_count += 1
            continue

        predictions.append(output_text)
        answers.append(request["answer"])
        # Extract subject if available
        if "subject" in request:
            subjects.append(request["subject"])
        else:
            subjects.append(None)

    # Calculate overall correctness
    assert len(predictions) == len(
        answers), "Predictions and answers must have the same length"

    # Report skipped entries
    if skipped_count > 0:
        print(
            f"Skipped {skipped_count}/{total_count} entries with error messages"
        )

    if len(predictions) == 0:
        print("No valid predictions to evaluate (all entries were errors)")
        return {
            'overall_accuracy': 0.0,
            'subject_accuracy': {},
            'skipped_count': skipped_count,
            'total_count': total_count
        }

    correct_count, valid_count = calculate_correctness(predictions, answers)
    # Calculate correctness float
    overall_correctness = correct_count / valid_count if valid_count > 0 else 0.0

    print("Correctness Results:")
    print(
        f"Overall Accuracy: {overall_correctness:.4f} ({overall_correctness*100:.2f}%) - {correct_count}/{valid_count} correct"
    )

    # Calculate subject-specific accuracy if subjects are available
    valid_subjects = [s for s in subjects if s is not None]
    subject_accuracy = {}
    if valid_subjects:
        print("Subject-Specific Accuracy:")
        subject_accuracy = calculate_subject_accuracy(predictions, answers,
                                                      subjects)

        for subject, stats in sorted(subject_accuracy.items(),
                                     key=lambda x: x[1]['accuracy'],
                                     reverse=True):
            if subject is not None:
                print(
                    f"{subject}: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%) - {stats['correct']}/{stats['total']} correct"
                )
    else:
        print("No subject information found in the data.")

    return {
        'overall_accuracy': overall_correctness,
        'subject_accuracy': subject_accuracy,
        'skipped_count': skipped_count,
        'total_count': total_count,
        'valid_count': valid_count
    }


if __name__ == "__main__":
    main()
