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
"""Verify VLM accuracy before/after DART visual-token pruning.

Runs `llm_inference` on one or more prepared multiple-choice VLM datasets (e.g. MMStar or
MMMU, see prepare_dataset.py) once without DART and once per requested reduction ratio,
scores every run with the calculate_correctness logic, and prints a per-dataset markdown
table: score (with delta vs. baseline), average input sequence length (ISL), average
text/visual token counts, and the estimated prefill speedup.

The reported speedup is the *linear token-count estimate* `baseline_ISL / pruned_ISL` — a
deterministic arithmetic ratio, not a timing measurement. The sequential single-pass runs this
script performs are not a fair timing benchmark (no warmup, fixed run order); measure real
latency with `llm_inference --dumpProfile --warmup N` or `llm_bench`.

Example:
    # 1. Prepare datasets (once)
    python3 examples/accuracy/scripts/prepare_dataset.py --dataset MMStar --output_dir tmp/mmstar_output

    # 2. Compare baseline vs DART 25% / 50% pruning (run from repo root)
    python3 examples/accuracy/scripts/run_dart_accuracy.py \
        --engine_dir engines/qwen3-vl-4b/llm \
        --multimodal_engine_dir engines/qwen3-vl-4b/visual \
        --dataset_files tmp/mmstar_output/mmstar_dataset.json \
        --ratios 0.25 0.5 \
        --output_dir tmp/dart_accuracy

Note: for an out-of-tree CMake build directory, pass
--llm_inference_bin <builddir>/examples/llm/llm_inference. EDGELLM_PLUGIN_PATH is derived
from the binary location automatically when the variable is not already set.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculate_correctness import is_correct  # noqa: E402

ERROR_MESSAGE = "TensorRT Edge LLM cannot handle this request. Fails."


def score_per_question(predictions_file, answers_file):
    """Return a per-question list: True/False for answered questions, None for runtime errors.

    The response count must match the dataset exactly — a mismatch means the predictions file
    is stale (e.g. --skip_existing after the dataset changed) or the run was cut short, and
    silently zip-truncating would misalign every following answer.
    """
    with open(predictions_file, encoding="utf-8") as f:
        responses = json.load(f)["responses"]
    with open(answers_file, encoding="utf-8") as f:
        requests = json.load(f)["requests"]
    if len(responses) != len(requests):
        raise RuntimeError(
            f"{predictions_file} has {len(responses)} responses but the dataset has "
            f"{len(requests)} requests — stale or truncated predictions; delete the file and rerun"
        )

    return [
        None if response["output_text"] == ERROR_MESSAGE else is_correct(
            response["output_text"], request["answer"])
        for response, request in zip(responses, requests)
    ]


def read_token_counts(profile_file):
    """Return (avg_isl, avg_visual_tokens, pruned_tokens) from a profile JSON.

    All three are deterministic token counts. avg_isl is the average number of prompt tokens
    the engine actually computed per request (post-pruning for DART runs); pruned_tokens is
    the total number of tokens the visual-token pruner removed across the run.
    """
    try:
        with open(profile_file, encoding="utf-8") as f:
            profile = json.load(f)
        prefill = profile["prefill"]
        runs = max(1, prefill.get("total_runs", 1))
        avg_isl = prefill["computed_tokens"] / runs
        pruned = prefill.get("pruned_tokens", 0)
        multimodal = profile.get("multimodal", {})
        avg_visual = multimodal.get("total_multimodal_tokens", 0) / max(
            1, multimodal.get("total_runs", runs))
        return avg_isl, avg_visual, pruned
    except (OSError, KeyError, json.JSONDecodeError, ZeroDivisionError):
        return None, None, None


def make_inference_env(llm_inference_bin):
    """Return the subprocess environment, deriving EDGELLM_PLUGIN_PATH when unset.

    llm_inference dlopens `build/libNvInfer_edgellm_plugin.so` relative to the CWD by
    default, which fails for out-of-tree build directories. The plugin always sits at the
    build root, two levels above the binary (<build>/examples/llm/llm_inference).
    """
    env = os.environ.copy()
    if "EDGELLM_PLUGIN_PATH" not in env:
        build_root = os.path.dirname(
            os.path.dirname(os.path.dirname(
                os.path.abspath(llm_inference_bin))))
        candidate = os.path.join(build_root, "libNvInfer_edgellm_plugin.so")
        if os.path.exists(candidate):
            env["EDGELLM_PLUGIN_PATH"] = candidate
            print(f"EDGELLM_PLUGIN_PATH not set; using {candidate}")
    return env


def check_dataset_batch_size(dataset_file):
    """DART pruning is gated to batch size 1 in the runtime — reject datasets that would make
    every "DART" run silently identical to baseline."""
    with open(dataset_file, encoding="utf-8") as f:
        batch_size = json.load(f).get("batch_size", 1)
    if batch_size != 1:
        raise RuntimeError(
            f"{dataset_file} has batch_size={batch_size}, but visual-token pruning only runs at "
            "batch size 1 — the DART configurations would silently equal baseline. "
            "Regenerate the dataset with batch_size 1.")


def run_config(args, dataset_file, dataset_name, tag, dart_ratio):
    """Run llm_inference for one (dataset, configuration); return (scores, avg_isl, avg_visual)."""
    predictions_file = os.path.join(args.output_dir,
                                    f"predictions_{dataset_name}_{tag}.json")
    profile_file = os.path.join(args.output_dir,
                                f"profile_{dataset_name}_{tag}.json")

    if os.path.exists(predictions_file) and args.skip_existing:
        print(f"[{dataset_name}/{tag}] reusing existing {predictions_file}")
    else:
        cmd = [
            args.llm_inference_bin,
            f"--engineDir={args.engine_dir}",
            f"--inputFile={dataset_file}",
            f"--outputFile={predictions_file}",
            "--dumpProfile",  # required for profile output; read for token counts only, never timing
            f"--profileOutputFile={profile_file}",
        ]
        if args.multimodal_engine_dir:
            cmd.append(f"--multimodalEngineDir={args.multimodal_engine_dir}")
        if args.max_generate_length > 0:
            cmd.append(f"--maxGenerateLength={args.max_generate_length}")
        if dart_ratio is not None:
            cmd.extend([
                "--visualPrune",
                f"--dartReductionRatio={dart_ratio}",
                f"--dartPivotImageTokens={args.pivot_image_tokens}",
                f"--dartPivotTextTokens={args.pivot_text_tokens}",
            ])
        log_file = os.path.join(args.output_dir,
                                f"inference_{dataset_name}_{tag}.log")
        print(f"[{dataset_name}/{tag}] running: {' '.join(cmd)}")
        with open(log_file, "w", encoding="utf-8") as log:
            result = subprocess.run(cmd,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env=make_inference_env(
                                        args.llm_inference_bin))
        if result.returncode != 0:
            print(
                f"[{dataset_name}/{tag}] llm_inference FAILED (rc={result.returncode}), see {log_file}"
            )
            return None

    scores = score_per_question(predictions_file, dataset_file)
    avg_isl, avg_visual, pruned_tokens = read_token_counts(profile_file)
    # Guard against silent no-op "DART" runs: if the runtime never pruned anything, the
    # configuration is unsupported (wrong engine type, tiny images below minVisualTokens, ...)
    # and the comparison would be baseline vs. baseline.
    if dart_ratio is not None and pruned_tokens is not None and pruned_tokens == 0:
        raise RuntimeError(
            f"[{dataset_name}/{tag}] the profile reports pruned_tokens=0 — visual-token pruning "
            "never ran (check: mRoPE VLM engine, batch size 1, images large enough for "
            "minVisualTokens). The result would silently equal baseline.")
    return scores, avg_isl, avg_visual


def evaluate_dataset(args, dataset_file, dataset_name, configs):
    """Run all configurations on one dataset; return the per-dataset result row dict."""
    check_dataset_batch_size(dataset_file)
    scores = {}
    token_counts = {}
    for tag, ratio in configs:
        result = run_config(args, dataset_file, dataset_name, tag, ratio)
        if result is None:
            sys.exit(1)
        scores[tag] = result[0]
        token_counts[tag] = (result[1], result[2])

    # Score every configuration over the SAME question set: only questions that all
    # configurations answered (no runtime error in any config) enter the accuracy, so the
    # delta always compares like with like.
    total = len(next(iter(scores.values())))
    common = [
        i for i in range(total)
        if all(scores[tag][i] is not None for tag, _ in configs)
    ]
    errors = {tag: sum(s is None for s in scores[tag]) for tag, _ in configs}
    if not common:
        print(
            f"[{dataset_name}] No question was answered by every configuration; nothing to score."
        )
        sys.exit(1)
    has_errors = any(errors[tag] > 0 for tag, _ in configs)
    if has_errors:
        print(
            f"[{dataset_name}] Scoring over the {len(common)}/{total} questions answered by every "
            f"configuration (per-config errors: {errors}). Token counts / speedup estimates are "
            "suppressed: the profile aggregates cover a different request set per configuration."
        )

    return {
        "dataset": dataset_name,
        "total_questions": total,
        "scored_questions": len(common),
        "errors": errors,
        "accuracy": {
            tag: sum(scores[tag][i] for i in common) / len(common)
            for tag, _ in configs
        },
        # Accuracy is computed over the common answered set, but the profile token counts
        # aggregate over every request a config processed — with errors those sets differ per
        # config, so the ISL/speedup estimates would be inconsistent with the scores. Suppress.
        "avg_isl": {
            tag: None if has_errors else token_counts[tag][0]
            for tag, _ in configs
        },
        "avg_visual": {
            tag: None if has_errors else token_counts[tag][1]
            for tag, _ in configs
        },
    }


def fmt(value, pattern="{:.1f}", missing="n/a"):
    return pattern.format(value) if value is not None else missing


def print_markdown_table(rows, configs):
    ratio_tags = [(tag, ratio) for tag, ratio in configs if ratio is not None]
    header = "| dataset | avg T/V tokens | baseline score | baseline ISL |"
    align = "|---|---|---:|---:|"
    for tag, ratio in ratio_tags:
        header += f" DART {ratio} score (Δ) | ISL | est. speedup |"
        align += "---:|---:|---:|"
    print("\n" + header)
    print(align)

    for row in rows:
        acc = row["accuracy"]
        isl = row["avg_isl"]
        base_isl = isl["baseline"]
        base_visual = row["avg_visual"]["baseline"]
        text_tokens = base_isl - base_visual if base_isl is not None and base_visual is not None else None
        tv = (f"{fmt(text_tokens)} / {fmt(base_visual)}")
        line = (
            f"| {row['dataset']} | {tv} | {acc['baseline']:.4f} | {fmt(base_isl)} |"
        )
        for tag, unused_ratio in ratio_tags:
            delta = acc[tag] - acc["baseline"]
            speedup = (base_isl / isl[tag]) if base_isl and isl[tag] else None
            line += (
                f" {acc[tag]:.4f} ({delta:+.4f}) | {fmt(isl[tag])} | {fmt(speedup, '~{:.3f}x')} |"
            )
        print(line)
    print(
        "\nISL = average prompt tokens computed by the engine per request (post-pruning for DART runs)."
    )
    print(
        "Estimated speedup is the linear token-count ratio baseline_ISL / pruned_ISL, not a timing"
    )
    print(
        "measurement — measure real latency with llm_inference --dumpProfile --warmup N."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare VLM accuracy with DART visual-token pruning on/off "
        "(accuracy + token counts only — use llm_inference --dumpProfile --warmup N for real latency)"
    )
    parser.add_argument("--engine_dir",
                        required=True,
                        help="LLM engine directory")
    parser.add_argument(
        "--multimodal_engine_dir",
        default="",
        help="Visual engine directory (required for VLM datasets)")
    parser.add_argument(
        "--dataset_files",
        "--dataset_file",
        nargs="+",
        required=True,
        dest="dataset_files",
        help=
        "Prepared dataset JSONs with ground-truth answers (see prepare_dataset.py); "
        "one table row per dataset")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.5],
        help="DART reduction ratios to test (default: 0.25 0.5)")
    parser.add_argument("--pivot_image_tokens", type=int, default=4)
    parser.add_argument("--pivot_text_tokens", type=int, default=4)
    parser.add_argument("--llm_inference_bin",
                        default="./build/examples/llm/llm_inference",
                        help="Path to the llm_inference binary")
    parser.add_argument("--output_dir",
                        default="tmp/dart_accuracy",
                        help="Directory for predictions/profiles/logs")
    parser.add_argument(
        "--max_generate_length",
        type=int,
        default=0,
        help="Override max generate length (0 = use dataset value)")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing prediction files instead of re-running inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = [("baseline", None)]
    configs += [(f"dart_r{int(round(r * 100)):03d}", r) for r in args.ratios]

    rows = []
    for dataset_file in args.dataset_files:
        dataset_name = os.path.splitext(
            os.path.basename(dataset_file))[0].removesuffix("_dataset")
        rows.append(evaluate_dataset(args, dataset_file, dataset_name,
                                     configs))

    print_markdown_table(rows, configs)

    summary_file = os.path.join(args.output_dir, "dart_accuracy_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary written to {summary_file}")


if __name__ == "__main__":
    main()
