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

import evaluate


def calculate_rouge_score(predictions, references, rouge_dir=None):
    """
    Compute Rouge score between predictions and references.
    Args:
        predictions: List of predictions.
        references: List of references.
        rouge_dir: Optional path to local rouge metric directory.
    Returns:
        Rouge score. Format: {
            "rouge1": float,
            "rouge2": float,
            "rougeL": float,
            "rougeLsum": float
        }
    """
    if rouge_dir is not None:
        rouge_evaluator = evaluate.load(rouge_dir)
    else:
        rouge_evaluator = evaluate.load("rouge")
    return rouge_evaluator.compute(predictions=predictions,
                                   references=references)


def main():
    """Main function to calculate Rouge score from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate Rouge score between two JSON files")
    parser.add_argument("--predictions_file",
                        type=str,
                        required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--references_file",
                        type=str,
                        required=True,
                        help="Path to references JSON file")
    parser.add_argument("--rouge_dir",
                        type=str,
                        default=None,
                        help="Path to local rouge metric directory")

    args = parser.parse_args()

    # Load JSON files
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(args.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)

    # Error message to skip
    error_message = "TensorRT Edge LLM cannot handle this request. Fails."

    # Extract predictions and references, filtering out error messages
    predictions = []
    references = []
    skipped_count = 0
    total_count = 0

    for response, request in zip(predictions_data["responses"],
                                 references_data["requests"]):
        total_count += 1
        output_text = response["output_text"]

        # Skip entries with error messages
        if output_text == error_message:
            skipped_count += 1
            continue

        predictions.append(output_text)
        references.append(request["reference"])

    # Calculate and print Rouge score
    assert len(predictions) == len(
        references), "Predictions and references must have the same length"

    # Report skipped entries
    if skipped_count > 0:
        print(
            f"Skipped {skipped_count}/{total_count} entries with error messages"
        )

    if len(predictions) == 0:
        print("No valid predictions to evaluate (all entries were errors)")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'rougeLsum': 0.0,
            'skipped_count': skipped_count,
            'total_count': total_count,
            'valid_count': 0
        }

    rouge_score_result = calculate_rouge_score(predictions, references,
                                               args.rouge_dir)

    print("Rouge Score Results:")
    valid_count = len(predictions)
    print(f"Evaluated {valid_count} valid predictions")
    print(f"Rouge-1:  {rouge_score_result['rouge1']:.4f}")
    print(f"Rouge-2:  {rouge_score_result['rouge2']:.4f}")
    print(f"Rouge-L:  {rouge_score_result['rougeL']:.4f}")
    print(f"Rouge-Lsum: {rouge_score_result['rougeLsum']:.4f}")

    # Add metadata to result
    rouge_score_result['skipped_count'] = skipped_count
    rouge_score_result['total_count'] = total_count
    rouge_score_result['valid_count'] = valid_count

    return rouge_score_result


if __name__ == "__main__":
    main()
