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
Accuracy checking utilities for TensorRT Edge-LLM.

This module provides functions to check the accuracy of model predictions against reference data.
"""

import json
import os
import re

import pytest
from pytest_helpers import run_command


def check_accuracy_with_dataset(output_json_file,
                                reference_json_file,
                                test_case_name,
                                logger=None):
    """
    Calculate accuracy/rouge score based on dataset type using the accuracy scripts.
    
    Args:
        output_json_file: Path to the output JSON file.
        reference_json_file: Path to the reference JSON file.
        test_case_name: Name of the test case to determine which metric to use.
        logger: Optional logger for command execution.
        
    Returns:
        Dictionary with metric results and metadata.
    """

    # Datasets that use ROUGE score (with "reference" field)
    ROUGE_DATASETS = [
        "gsm8k", "gsm8k_10", "mmstar", "llm_basic", "llm_lora", "vlm_basic",
        "vlm_lora"
    ]

    # Datasets that use correctness/accuracy (with "answer" field)
    CORRECTNESS_DATASETS = [
        "mmlu_0", "mmlu_5", "mmlu_pro", "mmmu", "mmmu_pro_4", "mmmu_pro_10",
        "mmmu_pro_vision"
    ]

    # Other datasets to be handled later
    OTHER_DATASETS = ["mtbench", "coco", "aime", "humaneval", "math500"]

    # Dataset-specific thresholds for ROUGE scores (rouge1, rougeL). Current thresholds are set to 25% for ROUGE-1 and 20% for ROUGE-L.
    ROUGE_THRESHOLDS = {
        "gsm8k": (0.25, 0.20),
        "gsm8k_10": (0.25, 0.20),
        "mmstar": (0.25, 0.20),
        "llm_basic": (0.25, 0.20),
        "llm_lora": (0.25, 0.20),
        "vlm_basic": (0.25, 0.20),
        "vlm_lora": (0.25, 0.20),
    }

    # Dataset-specific thresholds for accuracy/correctness
    ACCURACY_THRESHOLDS = {
        "mmlu_0": 0.30,  # 0-shot MMLU
        "mmlu_5": 0.35,  # 5-shot MMLU - expect better with examples
        "mmlu_pro": 0.12,  # MMLU Pro - harder, lower threshold
        "mmmu": 0.30,  # Multimodal understanding
        "mmmu_pro_4": 0.30,  # MMMU Pro with 4 options
        "mmmu_pro_10": 0.12,  # MMMU Pro with 10 options - harder
        "mmmu_pro_vision": 0.12,  # MMMU Pro vision tasks
    }

    # Load the output JSON to get prediction count
    with open(output_json_file, 'r', encoding='utf-8') as f:
        output_json = json.load(f)

    num_predictions = len(output_json.get("responses", []))

    result = {
        'success': True,
        'test_case': test_case_name,
        'num_predictions': num_predictions
    }

    if test_case_name in ROUGE_DATASETS:
        # Use ROUGE score script for these datasets
        # Assume we're in the project root directory
        rouge_script = 'examples/accuracy/scripts/calculate_rouge_score.py'

        try:
            cmd = [
                'python3', rouge_script, '--predictions_file',
                output_json_file, '--references_file', reference_json_file
            ]

            # Add rouge_dir from environment variable if available
            edge_llm_cache_dir = os.environ.get('EDGE_LLM_CACHE_DIR')
            if edge_llm_cache_dir:
                rouge_dir = os.path.join(edge_llm_cache_dir, 'rouge')
                if os.path.exists(rouge_dir):
                    cmd.extend(['--rouge_dir', rouge_dir])
            cmd_result = run_command(cmd,
                                     remote_config=None,
                                     timeout=600,
                                     logger=logger)

            if not cmd_result['success']:
                raise RuntimeError(
                    f"ROUGE calculation failed: {cmd_result.get('error', 'Unknown error')}"
                )

            # Parse the output to extract ROUGE scores
            output = cmd_result['output']
            rouge_score = {}

            # Extract scores using regex
            rouge1_match = re.search(r'Rouge-1:\s+([\d.]+)', output)
            rouge2_match = re.search(r'Rouge-2:\s+([\d.]+)', output)
            rougeL_match = re.search(r'Rouge-L:\s+([\d.]+)', output)
            rougeLsum_match = re.search(r'Rouge-Lsum:\s+([\d.]+)', output)

            if rouge1_match:
                rouge_score['rouge1'] = float(rouge1_match.group(1))
            if rouge2_match:
                rouge_score['rouge2'] = float(rouge2_match.group(1))
            if rougeL_match:
                rouge_score['rougeL'] = float(rougeL_match.group(1))
            if rougeLsum_match:
                rouge_score['rougeLsum'] = float(rougeLsum_match.group(1))

            result['rouge_score'] = rouge_score
            result['metric_type'] = 'rouge'

            # Get dataset-specific thresholds (rouge1, rougeL)
            rouge1_threshold, rougeL_threshold = ROUGE_THRESHOLDS.get(
                test_case_name, (0.25, 0.2))

            # Check both ROUGE-1 and ROUGE-L thresholds
            rouge1_score = rouge_score.get("rouge1", 0)
            rougeL_score = rouge_score.get("rougeL", 0)

            if rouge1_score < rouge1_threshold or rougeL_score < rougeL_threshold:
                failure_details = [
                    f"ROUGE score below threshold for {test_case_name}",
                    f"Rouge1: {rouge1_score:.4f} (threshold: {rouge1_threshold})",
                    f"Rouge2: {rouge_score.get('rouge2', 0):.4f}",
                    f"RougeL: {rougeL_score:.4f} (threshold: {rougeL_threshold})",
                    f"RougeLsum: {rouge_score.get('rougeLsum', 0):.4f}",
                    f"Number of predictions: {num_predictions}"
                ]
                pytest.fail("\n".join(failure_details))

        except Exception as e:
            raise RuntimeError(f"Failed to run ROUGE script: {str(e)}")

    elif test_case_name in CORRECTNESS_DATASETS:
        # Use correctness/accuracy script for these datasets
        # Assume we're in the project root directory
        correctness_script = 'examples/accuracy/scripts/calculate_correctness.py'

        try:
            cmd = [
                'python3', correctness_script, '--predictions_file',
                output_json_file, '--answers_file', reference_json_file
            ]
            cmd_result = run_command(cmd,
                                     remote_config=None,
                                     timeout=600,
                                     logger=logger)

            if not cmd_result['success']:
                raise RuntimeError(
                    f"Correctness calculation failed: {cmd_result.get('error', 'Unknown error')}"
                )

            # Parse the output to extract accuracy
            output = cmd_result['output']

            # Extract accuracy using regex - looking for "Overall Accuracy: 0.XXXX (XX.XX%)"
            accuracy_match = re.search(
                r'Overall Accuracy:\s+([\d.]+)\s+\(([\d.]+)%\)\s+-\s+(\d+)/(\d+)',
                output)

            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                correct_count = int(accuracy_match.group(3))
                total_count = int(accuracy_match.group(4))

                result['accuracy'] = accuracy
                result['correct_count'] = correct_count
                result['total_count'] = total_count
                result['metric_type'] = 'accuracy'

                # Get dataset-specific threshold, default to 0.30 if not specified
                accuracy_threshold = ACCURACY_THRESHOLDS.get(
                    test_case_name, 0.30)
                if accuracy < accuracy_threshold:
                    failure_details = [
                        f"Accuracy below threshold for {test_case_name}",
                        f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) - {correct_count}/{total_count} correct",
                        f"Threshold: {accuracy_threshold:.4f} ({accuracy_threshold*100:.2f}%)",
                    ]
                    pytest.fail("\n".join(failure_details))
            else:
                raise RuntimeError(
                    f"Could not parse accuracy from output: {output}")

        except Exception as e:
            raise RuntimeError(f"Failed to run correctness script: {str(e)}")

    elif test_case_name in OTHER_DATASETS:
        # Skip validation for these datasets
        result['metric_type'] = 'skipped'
        result[
            'message'] = f"Post-processing for {test_case_name} will be implemented later"

    else:
        # Unknown dataset - skip validation but log warning
        result['metric_type'] = 'unknown'
        result[
            'message'] = f"Unknown dataset type: {test_case_name}. Skipping validation."

    return result
