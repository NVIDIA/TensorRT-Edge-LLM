#!/usr/bin/env python3
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
"""
Prepare MMMU output xlsx file in VLMEvalKit format from JSON output.

This script combines:
1. MMMU_DEV_VAL.tsv - VLMEvalKit's MMMU dataset metadata (questions, options, answers, etc.)
2. JSON output file - TensorRT Edge LLM inference predictions

To produce an xlsx file compatible with VLMEvalKit evaluation.

Download MMMU_DEV_VAL.tsv from:
    https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv

Usage:
    python prepare_mmmu_vlmevalkit.py \
        --tsv_file /path/to/MMMU_DEV_VAL.tsv \
        --json_file /path/to/outputs/mmmu_predictions.json \
        --output_file /path/to/outputs/Model_MMMU_DEV_VAL.xlsx
"""

import argparse
import json
import os

import pandas as pd


def load_json_predictions(json_file: str) -> list:
    """Load predictions from JSON output file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Handle both formats: list of responses or dict with 'responses' key
    if isinstance(data, dict) and 'responses' in data:
        responses = data['responses']
    elif isinstance(data, list):
        responses = data
    else:
        raise ValueError(
            f"Unsupported JSON format. Expected dict with 'responses' key or list."
        )

    # Sort by request_idx to ensure correct order
    responses_sorted = sorted(responses, key=lambda x: x.get('request_idx', 0))

    # Extract output_text as predictions
    predictions = [r.get('output_text', '') for r in responses_sorted]

    return predictions


def prepare_mmmu_vlmevalkit_output(
    tsv_file: str,
    json_file: str,
    output_file: str,
) -> None:
    """
    Prepare MMMU output xlsx in VLMEvalKit format.
    
    Args:
        tsv_file: Path to MMMU_DEV_VAL.tsv template file
        json_file: Path to JSON output file with predictions
        output_file: Path to output xlsx file
    """
    # Load TSV template
    print(f"Loading TSV template from: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"  Total rows: {len(df)}")

    # Filter by validation split
    df_filtered = df[df['split'] == 'validation'].copy()
    print(f"  Rows after filtering by split='validation': {len(df_filtered)}")

    # Remove 'image' column if it exists (VLMEvalKit format doesn't include it)
    if 'image' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['image'])
        print("  Removed 'image' column")

    # Load predictions from JSON
    print(f"\nLoading predictions from: {json_file}")
    predictions = load_json_predictions(json_file)
    print(f"  Number of predictions: {len(predictions)}")

    # Verify counts match
    if len(predictions) != len(df_filtered):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(df_filtered)} dataset rows. "
            "Ensure the JSON output corresponds to the same dataset split.")

    # Add predictions column
    df_filtered['prediction'] = predictions
    print("  Added 'prediction' column")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to xlsx
    print(f"\nSaving output to: {output_file}")
    df_filtered.to_excel(output_file, index=False, engine='openpyxl')
    print(
        f"  Saved {len(df_filtered)} rows with {len(df_filtered.columns)} columns"
    )
    print(f"  Columns: {df_filtered.columns.tolist()}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Prepare MMMU output xlsx in VLMEvalKit format from JSON output")
    parser.add_argument(
        '--tsv_file',
        type=str,
        required=True,
        help='Path to MMMU_DEV_VAL.tsv file. Download from: '
        'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv')
    parser.add_argument('--json_file',
                        type=str,
                        required=True,
                        help='Path to JSON output file with predictions')
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='Path to output xlsx file')

    args = parser.parse_args()

    prepare_mmmu_vlmevalkit_output(
        tsv_file=args.tsv_file,
        json_file=args.json_file,
        output_file=args.output_file,
    )


if __name__ == '__main__':
    main()
