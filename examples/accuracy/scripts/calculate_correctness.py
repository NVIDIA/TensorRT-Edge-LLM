# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def clean_text(text):
    """
    Clean and normalize text by stripping whitespace and removing punctuation from ends.
    Args:
        text: Input text string.
    Returns:
        Cleaned text string.
    """

    # Remove <|endoftext|> from text for phi4mm
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    text = text.strip().strip("().,")
    return text


def parse_multi_choice_response(text):
    """
    Parse multiple choice answer from text that may be in various formats.
    Handles formats like "A. xxx", "A", "(A)", or just returns first letter if it's A-H.
    
    Args:
        text: Input text string potentially containing a multiple choice answer.
    Returns:
        Single letter (A-H) if found, otherwise returns the original cleaned text.
    """
    text = text.strip()

    # If text is already just a single letter A-H, return it
    if len(text) == 1 and text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        return text

    # Try to match pattern like "A." or "A)" or "(A)" at the start
    match = re.match(r'^[\(]?([A-H])[\.\):\s]', text)
    if match:
        return match.group(1)

    # If no match found, return the original text
    return text


def calculate_correctness(predictions, references):
    """
    Compute correctness score between predictions and references.
    Args:
        predictions: List of predictions.
        references: List of references.
    Returns:
        Overall correctness score (float between 0 and 1).
    """
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have the same length")

    correct_count = 0
    total_count = len(predictions)

    for pred, ref in zip(predictions, references):
        # Clean and normalize text for comparison
        pred_clean = clean_text(pred)
        ref_clean = clean_text(ref)
        # Clean for multi-choice
        if ref_clean in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            pred_clean = parse_multi_choice_response(pred_clean)
        if pred_clean == ref_clean:
            correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


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
        # Clean and normalize text for comparison
        pred_clean = clean_text(pred)
        ref_clean = clean_text(ref)
        # Clean for multi-choice
        if ref_clean in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            pred_clean = parse_multi_choice_response(pred_clean)
        if pred_clean == ref_clean:
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

    overall_correctness = calculate_correctness(predictions, answers)

    print("Correctness Results:")
    valid_count = len(predictions)
    correct_count = int(overall_correctness * valid_count)
    print(
        f"Overall Accuracy: {overall_correctness:.4f} ({overall_correctness*100:.2f}%) - {correct_count}/{valid_count} correct"
    )

    # Calculate subject-specific accuracy if subjects are available
    valid_subjects = [s for s in subjects if s is not None]
    if valid_subjects:
        print("Subject-Specific Accuracy:")
        subject_accuracy = calculate_subject_accuracy(predictions, answers,
                                                      subjects)

        for subject, stats in subject_accuracy.items():
            if subject is not None:  # Skip None subjects
                print(
                    f"{subject}: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%) - {stats['correct']}/{stats['total']} correct"
                )
    else:
        print("No subject information found in the data.")

    return {
        'overall_accuracy': overall_correctness,
        'subject_accuracy': subject_accuracy if valid_subjects else {},
        'skipped_count': skipped_count,
        'total_count': total_count,
        'valid_count': valid_count
    }


if __name__ == "__main__":
    main()
