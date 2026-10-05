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

"""Calibrate the BLASST skip-softmax deployment threshold via NVIDIA ModelOpt.

Skip-softmax (arXiv:2512.12087) skips a KV tile when exp(scale * (m_tile - m_running))
falls below a threshold lambda. A fixed lambda yields wildly different sparsity across
context lengths, so the threshold follows lambda = scale_factor / L with a
model-specific scale factor calibrated per target sparsity.

This tool is a thin wrapper around the OFFICIAL calibration implementation —
``modelopt.torch.sparsity.attention_sparsity`` (``DynamicThresholdCalibrator``, the
same machinery TensorRT-LLM's threshold_scale_factor comes from):

  1. Load the HF checkpoint with ``attn_implementation="eager"`` (the pytorch
     backend patches softmax, which only eager attention calls).
  2. ``mtsa.sparsify(model, config)`` with a calibration config: ModelOpt
     auto-generates the RULER calibration set (default 64 samples across
     descending power-of-2 length bins), runs ONE forward pass evaluating all
     20 built-in threshold trials simultaneously, and fits
     ``scale_factor = a * exp(b * sparsity)`` over every individual
     (sample, threshold) point with scipy curve_fit (sparsity filtered to
     [0.10, 0.90]).
  3. Read back (a, b) and print the deployable scale factor per requested
     target sparsity: ``S = a * exp(b * s_target)``.

WARNING — the fit does NOT extrapolate beyond its calibration max_seqlen:
a "30% target" S fitted at 4k actually skips ~80% of tiles at 16k and breaks
accuracy. For contexts beyond the calibration range pick S empirically:
S = lambda_target * kvCacheCapacity with lambda_target in [0.002, 0.005]
(validated on Qwen3-1.7B via RULER/MMLU/NIAH), or recalibrate with long samples.

Deploy S by re-exporting the model with
``python -m tensorrt_edgellm.scripts.export ... --skip-softmax-scale-factor S``
(it becomes an AttentionPlugin attribute) and rebuilding the engine. At
inference the kernel derives ``lambda_i = S / seqlen_kv_i`` PER SEQUENCE
(no clamp, no capacity floor) — the kernel artifacts are
lambda-free and built once.

DEPLOYMENT RULE — skip-softmax is a LONG-CONTEXT feature. Sequences shorter
than S run at lambda > 1, outside the calibrated domain, and degrade sharply
(measured: MMLU -10.5pts on edgellm 0.8B and -16.3pts on official TRT-LLM
1.2.1 at lambda ~ 10, while 13k-context retrieval at lambda 0.6 stays intact).
If the deployment traffic contains requests with L < S, serve them with a
dense engine (S = 0), or choose S no larger than the shortest expected
request length. Validation gates are therefore long-context only (RULER +
LongBench); MMLU is reported as an advisory short-seq column.

Note on sparsity semantics: ModelOpt measures simulated block sparsity POOLED
ACROSS ALL LAYERS with a single running-max chain over causal-valid blocks;
the deployed kernel's per-layer skip ratio at the same lambda can differ
substantially (two interleaved softmax-stage max chains, first tile never
skips, and per-layer sparsity varies widely around the pooled mean). Treat the
calibrated lambda as the TRT-LLM-ecosystem-consistent starting point and let
end-to-end evals arbitrate; the kernel's verify-build tile counters report the
deployed per-layer sparsity if measurement is needed.

Two subcommands:

  calibrate (default) — the ModelOpt flow above; venv needs torch,
      transformers, nvidia-modelopt, scipy, wonderwords.
  evaluate — the paper's accuracy validation: score a DEPLOYED engine on real
      RULER samples (HF dataset simonjegou/ruler, exact-answer matching per
      task), optionally plus an MMLU subset (--mmlu-samples). Run it once per
      engine (dense baseline vs each deployed S) and compare; pair with the
      prefill benchmark (llm_bench) for TTFT.

Examples:

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
        --model-dir /path/to/Qwen3-1.7B --max-seqlen 4096 \
        --target-sparsity 0.3 0.5 --max-context 4096

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
        --model-dir /path/to/Qwen3-1.7B \
        --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
        --max-context 4096 --per-task 20
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_RULER_HF_REPO = "simonjegou/ruler"


def _infer_env(llm_inference):
    """Env for llm_inference subprocesses: pin EDGELLM_PLUGIN_PATH to the
    plugin NEXT TO the binary's build tree unless the caller already set it.

    Without this, llm_inference dlopens the RELATIVE default
    ``build/libNvInfer_edgellm_plugin.so`` — i.e. the MAIN checkout's plugin
    when CWD is the repo root, even when --llm-inference points at a worktree
    build. Main's plugin lacks d256/d512 skip kernels and silently dispatches
    DENSE → every eval verdict becomes a vacuous dense-vs-dense PASS
    (2026-08-03 incident: all d256 ladder + LongBench verdicts invalidated;
    engagement probes passed because they exported the env var explicitly
    while this subprocess did not — probe and eval MUST share one env)."""
    env = os.environ.copy()
    if not env.get("EDGELLM_PLUGIN_PATH"):
        plug = Path(llm_inference).resolve().parents[2] \
            / "libNvInfer_edgellm_plugin.so"
        if plug.exists():
            env["EDGELLM_PLUGIN_PATH"] = str(plug)
    return env


def _tty() -> bool:
    return sys.__stdout__ is not None and sys.__stdout__.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def stage(cmd: str, idx: int, total: int, text: str) -> None:
    """Slim, labeled stage header: distinguishes calibrate/evaluate stages."""
    head = f"[{cmd} {idx}/{total}] {text}"
    print(f"\n{_c('1;36', head)}\n{_c('36', '-' * min(len(head), 72))}", flush=True)


def result_box(lines: list[str], color: str = "1;37") -> None:
    """Heavy box around the lines that matter."""
    width = max(len(line) for line in lines) + 2
    print()
    print(_c(color, "┏" + "━" * width + "┓"))
    for line in lines:
        print(_c(color, "┃ " + line.ljust(width - 2) + " ┃"))
    print(_c(color, "┗" + "━" * width + "┛"), flush=True)


def program_banner(title: str, subtitle: str = "") -> None:
    """Top-level separator: one per subcommand invocation."""
    bar = "━" * 72
    print(f"\n{_c('1;35', bar)}")
    print(_c("1;35", f"  {title}"))
    if subtitle:
        print(_c("35", f"  {subtitle}"))
    print(_c("1;35", bar))
    print(_c("2", "  (dim '│'-indented lines = library output; "
                  "normal lines = this tool)"), flush=True)


class _IndentDim:
    """Contain a library's stdout: reprint it dim + indented under our stage."""

    def __init__(self):
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            sys.__stdout__.write(_c("2", f"  │ {line}") + "\n")

    def flush(self) -> None:
        if self._buf:                       # drain a partial (no-newline) line
            sys.__stdout__.write(_c("2", f"  \u2502 {self._buf}") + "\n")
            self._buf = ""
        sys.__stdout__.flush()

    def isatty(self) -> bool:      # some libraries probe the stream
        return False


def _repo_root() -> Path:
    """kernelSrcs/fmha_cutedsl_blackwell/ -> repo root."""
    return Path(__file__).resolve().parents[2]


def _run_mmlu(args, role: str, n_stages: int) -> tuple:
    """Score the engine on an MMLU subset via the repo's accuracy tooling.

    Reuses examples/accuracy end to end: prepare_dataset.py generates the
    prompt set (cached, it is seed-independent), the subset is sampled with
    args.seed, llm_inference runs it, and calculate_correctness.py scores it
    (same letter-extraction rules as CI). Returns (accuracy, n_scored).
    """
    import importlib.util
    import random
    import time

    scripts_dir = _repo_root() / "examples" / "accuracy" / "scripts"

    stage(f"evaluate·{role}", 4, n_stages,
          f"MMLU subset ({args.mmlu_samples} of 14042, "
          f"{args.mmlu_num_shot}-shot, seed {args.seed})")
    t0 = time.time()
    ds_json = args.mmlu_dataset_json or (
        Path(tempfile.gettempdir()) /
        f"mmlu_{args.mmlu_num_shot}shot_dataset.json")
    if not ds_json.exists():
        print(f"  generating {ds_json} (first run only)")
        with tempfile.TemporaryDirectory() as gen_dir:
            cmd = [sys.executable, str(scripts_dir / "prepare_dataset.py"),
                   "--dataset", "MMLU", "--output_dir", gen_dir,
                   "--num_shot", str(args.mmlu_num_shot)]
            print(_c("2", "  running: " + " ".join(cmd)))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            gen_file = Path(gen_dir) / "mmlu_dataset.json"
            if not gen_file.exists():
                print(proc.stdout[-2000:])
                print(proc.stderr[-2000:])
                print("ERROR: MMLU dataset generation failed")
                return None, 0
            ds_json.parent.mkdir(parents=True, exist_ok=True)
            gen_file.rename(ds_json)
    payload = json.load(open(ds_json))
    rng = random.Random(args.seed)
    subset = rng.sample(payload["requests"],
                        min(args.mmlu_samples, len(payload["requests"])))
    payload["requests"] = subset
    print(f"  MMLU: {len(subset)} questions sampled in {time.time() - t0:.0f}s")

    stage(f"evaluate·{role}", 5, n_stages, "run engine + score (CI letter rules)")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "mmlu_input.json"
        out_path = Path(tmp) / "mmlu_output.json"
        json.dump(payload, open(in_path, "w"))
        cmd = [str(args.llm_inference), "--engineDir", str(args.engine_dir),
               "--inputFile", str(in_path), "--outputFile", str(out_path)]
        print(_c("2", "  running: " + " ".join(cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=_infer_env(args.llm_inference))
        if not out_path.exists():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print("ERROR: MMLU inference produced no output file")
            return None, 0
        responses = json.load(open(out_path))["responses"]

    spec = importlib.util.spec_from_file_location(
        "calculate_correctness", scripts_dir / "calculate_correctness.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    error_message = "TensorRT Edge LLM cannot handle this request. Fails."
    predictions, answers = [], []
    n_failed = 0
    for r, req in zip(sorted(responses, key=lambda r: r["request_idx"]), subset):
        text = r.get("output_text", "")
        if text == error_message:      # same denominator rule as the CI scorer
            n_failed += 1
            continue
        predictions.append(text)
        answers.append(req["answer"])
    if n_failed:
        print(f"  WARNING: {n_failed}/{len(subset)} requests failed in the engine")
    if not predictions or n_failed > len(subset) // 10:
        print("ERROR: too many failed MMLU requests — engine/config problem, "
              "not an accuracy signal")
        return None, 0
    correct, valid = scorer.calculate_correctness(predictions, answers)
    acc = correct / valid if valid else 0.0
    print(f"  MMLU accuracy: {acc:.4f} ({correct}/{valid}) "
          f"in {time.time() - t0:.0f}s")
    return acc, valid


def evaluate(args) -> int:
    """Score a deployed engine on real RULER samples (exact-answer matching)."""
    import time

    from datasets import load_dataset
    from transformers import AutoTokenizer

    role = args.label or ("baseline" if args.save_results and not args.baseline
                          else "candidate")
    n_stages = 5 if args.mmlu_samples > 0 else 3
    program_banner(f"SKIP-SOFTMAX EVALUATE — {role}",
                   f"engine: {args.engine_dir}   RULER@{args.max_context}, "
                   f"{args.per_task}/task, seed {args.seed}"
                   + (f"   MMLU x{args.mmlu_samples}"
                      if args.mmlu_samples > 0 else ""))
    stage(f"evaluate·{role}", 1, n_stages, "build RULER sample set")
    t0 = time.time()
    cfgs = [4096, 8192, 16384]
    cfg = next((c for c in cfgs if c >= args.max_context), cfgs[-1])
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ds = load_dataset(_RULER_HF_REPO, str(cfg), split="test").shuffle(seed=args.seed)

    max_tok = args.max_context - 196     # headroom for the chat template
    picked: dict = defaultdict(list)
    requests, answers = [], []
    for row in ds:
        if len(picked[row["task"]]) >= args.per_task:
            continue
        prompt = row["context"] + "\n" + row["question"] + row["answer_prefix"]
        if len(tokenizer(prompt).input_ids) > max_tok:
            continue
        picked[row["task"]].append(1)
        requests.append({"messages": [{"role": "user", "content": prompt}]})
        answers.append({"task": row["task"], "answer": row["answer"]})
        if sum(len(v) for v in picked.values()) >= args.per_task * 13:
            break
    dropped = [k for k, v in picked.items() if not v]
    print(f"RULER({cfg}): {len(requests)} samples across "
          f"{sum(1 for v in picked.values() if v)} tasks"
          + (f" (all samples too long for: {dropped})" if dropped else ""))

    print(f"  sample set built in {time.time() - t0:.0f}s")

    stage(f"evaluate·{role}", 2, n_stages,
          "run engine (greedy, exact-answer scoring next)")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "ruler_input.json"
        out_path = Path(tmp) / "ruler_output.json"
        json.dump({"temperature": 1.0, "top_p": 1.0, "top_k": 1,   # greedy
                   "max_generate_length": args.max_generate_length,
                   "apply_chat_template": True, "requests": requests},
                  open(in_path, "w"))
        cmd = [str(args.llm_inference), "--engineDir", str(args.engine_dir),
               "--inputFile", str(in_path), "--outputFile", str(out_path)]
        print(_c("2", "  running: " + " ".join(cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=_infer_env(args.llm_inference))
        if not out_path.exists():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print("ERROR: inference produced no output file")
            return 1
        responses = json.load(open(out_path))["responses"]
    ok = sum(1 for r in responses if r.get("output_text"))
    print(f"  {len(responses)} responses ({ok} non-empty) in {time.time() - t0:.0f}s")

    stage(f"evaluate·{role}", 3, n_stages, "score (exact-answer match per task)")
    per_task, total = defaultdict(list), []
    for resp, ans in zip(sorted(responses, key=lambda r: r["request_idx"]), answers):
        raw = ans["answer"]
        if isinstance(raw, (list, tuple)):     # dataset stores answers as lists
            expected = list(raw)
        else:
            try:                               # stringified list fallback
                val = ast.literal_eval(raw)
                expected = list(val) if isinstance(val, (list, tuple)) else [val]
            except (ValueError, SyntaxError):  # plain answer string
                expected = [raw]
        text = resp.get("output_text", "")
        score = sum(1 for e in expected if str(e) in text) / max(len(expected), 1)
        per_task[ans["task"]].append(score)
        total.append(score)
    task_means = {task: sum(v) / len(v) for task, v in per_task.items()}
    overall = sum(total) / len(total)
    base = json.load(open(args.baseline)) if args.baseline else None
    print(f"\n{'task':<20}{'n':>4}{'score':>9}"
          + ("{:>10}{:>8}".format("baseline", "delta") if base else ""))
    for task in sorted(task_means):
        line = f"{task:<20}{len(per_task[task]):>4}{task_means[task]:>9.3f}"
        if base:
            base_t = base.get("per_task", {}).get(task)
            if base_t is not None:
                line += f"{base_t:>10.3f}{task_means[task] - base_t:>+8.3f}"
        print(line)
    line = f"{'OVERALL':<20}{len(total):>4}{overall:>9.4f}"
    if base:
        line += f"{base['overall']:>10.4f}{overall - base['overall']:>+8.4f}"
    print(line)

    payload = {"overall": overall, "per_task": task_means,
               "n": len(total), "seed": args.seed,
               "per_task_n": {k: len(v) for k, v in per_task.items()}}

    def _save() -> None:
        if args.save_results:
            json.dump(payload, open(args.save_results, "w"), indent=1)
            print(f"saved results -> {args.save_results}")

    # Persist the RULER scores BEFORE attempting MMLU: an engine failure in
    # the MMLU stage must not discard a completed (and expensive) RULER run.
    _save()
    mmlu_acc, mmlu_n = (None, 0)
    if args.mmlu_samples > 0:
        mmlu_acc, mmlu_n = _run_mmlu(args, role, n_stages)
        if mmlu_acc is None:
            return 1               # RULER results are already on disk
        payload["mmlu"] = {"accuracy": mmlu_acc, "n": mmlu_n,
                           "num_shot": args.mmlu_num_shot}
        _save()

    if base:
        if base.get("seed") != args.seed or base.get("n") != len(total):
            print(f"\nWARNING: baseline sampled differently "
                  f"(seed={base.get('seed')}, n={base.get('n')}) — comparison is unpaired")
        for task in sorted(task_means):
            base_t = base.get("per_task", {}).get(task)
            if base_t is not None and base_t - task_means[task] > 0.10:
                print(f"WARNING: {task} dropped {base_t - task_means[task]:+.3f} "
                      f"({base_t:.3f} -> {task_means[task]:.3f}) — small n, but inspect")
        drop = base["overall"] - overall
        ruler_pass = drop <= args.max_drop
        label = f" [{args.label}]" if args.label else ""

        # MMLU is ADVISORY ONLY — never gates. Rationale (2026-08-03, aligned
        # with TRT-LLM's contract): skip-softmax is a LONG-CONTEXT feature;
        # short prompts (L < S, lambda > 1) are outside the calibrated domain
        # and are expected to degrade (measured: MMLU -10.5pts on edgellm
        # 0.8B, -16.3pts on official TRT-LLM 1.2.1 at lambda~10). The
        # deployment rule lives in the docs: traffic with L < S must use a
        # dense engine. The verdict here = RULER (long-context); held-out
        # acceptance = LongBench.
        base_mmlu = (base.get("mmlu") or {}).get("accuracy")
        box = [f"RULER:  {'PASS' if ruler_pass else 'FAIL'}   "
               f"{base['overall']:.4f} -> {overall:.4f}   "
               f"drop {drop:+.4f}  (gate {args.max_drop})"]
        if mmlu_acc is not None and base_mmlu is not None:
            mmlu_drop = base_mmlu - mmlu_acc
            box.append(f"MMLU:   {base_mmlu:.4f} -> {mmlu_acc:.4f}   "
                       f"drop {mmlu_drop:+.4f}  (ADVISORY, short-seq domain "
                       "— not gated)")
        elif mmlu_acc is not None:
            box.append(f"MMLU:   {mmlu_acc:.4f}  (advisory only)")
        verdict = ruler_pass
        box.insert(0, f"VERDICT: {'PASS' if verdict else 'FAIL'}{label}"
                   "  (RULER long-context gate)")
        result_box(box, color="1;32" if verdict else "1;31")
        if verdict:
            print(f"RECOMMENDATION: this build{label} is validated for deployment. "
                  "On real text, prefill speedup GROWS with the skip ratio at "
                  "long context (measured up to -8% TTFT at 32k for lambda_eff "
                  "0.005) — prefer the LARGEST S that passes this gate. Below "
                  "~8k context the columns are within noise either way.")
        else:
            print(f"RECOMMENDATION: do NOT deploy this build{label}. Re-export "
                  "with a smaller scale factor (lower --target-sparsity in "
                  "`calibrate`) and re-run this evaluation.")
        return 0 if verdict else 1
    print("\n(no --baseline given: scores reported without a verdict; save this "
          "run with --save-results and pass it as --baseline on the skip build)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser(
        "evaluate", help="Score a deployed engine on real RULER samples")
    p_eval.add_argument("--model-dir", type=Path, required=True,
                        help="HF checkpoint directory (tokenizer for length filtering)")
    p_eval.add_argument("--engine-dir", type=Path, required=True,
                        help="Deployed engine directory (llm.engine + tokenizer)")
    p_eval.add_argument("--llm-inference", type=Path, required=True,
                        help="Path to the llm_inference binary")
    p_eval.add_argument("--max-context", type=int, default=4096,
                        help="Engine max input length (default: %(default)s)")
    p_eval.add_argument("--per-task", type=int, default=20,
                        help="Samples per RULER task (default: %(default)s)")
    p_eval.add_argument("--max-generate-length", type=int, default=96,
                        help="Generation budget per sample (default: %(default)s)")
    p_eval.add_argument("--seed", type=int, default=0,
                        help="Sample-selection seed (default: %(default)s)")
    p_eval.add_argument("--mmlu-samples", type=int, default=0,
                        help="Also score N MMLU questions and gate their delta "
                        "alongside RULER (0 = RULER only; default: %(default)s)")
    p_eval.add_argument("--mmlu-num-shot", type=int, default=5,
                        help="MMLU few-shot examples (default: %(default)s, "
                        "matches the CI mmlu_5 configuration)")
    p_eval.add_argument("--mmlu-dataset-json", type=Path, default=None,
                        help="Cache path for the generated MMLU prompt json "
                        "(default: $TMPDIR/mmlu_<shot>shot_dataset.json; "
                        "generated on first use via prepare_dataset.py)")
    p_eval.add_argument("--save-results", type=Path, default=None,
                        help="Write scores to this json (use on the dense "
                             "baseline run)")
    p_eval.add_argument("--baseline", type=Path, default=None,
                        help="Baseline scores json to compare against; prints "
                             "a PASS/FAIL verdict and sets the exit code")
    p_eval.add_argument("--max-drop", type=float, default=0.03,
                        help="Max tolerated OVERALL accuracy drop vs the "
                             "baseline (default: %(default)s)")
    p_eval.add_argument("--label", default=None,
                        help="Human-readable label of the build under test "
                             "(e.g. 'lambda=0.115'), echoed in the verdict")

    p_cal = sub.add_parser(
        "calibrate", help="Calibrate a/b via ModelOpt (default command)")
    p_cal.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache dir for ModelOpt's RULER generation data (default: "
             "ModelOpt's ~/.cache/modelopt/data — point this somewhere with room)")
    parser_target = p_cal
    parser_target.add_argument(
        "--model-dir", type=Path, required=True,
        help="HF checkpoint directory")
    parser_target.add_argument(
        "--samples", type=int, default=64,
        help="RULER calibration samples (default: %(default)s, aligned with "
             "ModelOpt's official SKIP_SOFTMAX_CALIB preset — the same "
             "parameters TRT-LLM's threshold_scale_factor ships with)")
    parser_target.add_argument(
        "--max-seqlen", type=int, default=16384,
        help="Longest calibration length; RULER length bins are descending "
             "powers of 2 from here, >= 1024 (default: %(default)s, the "
             "SKIP_SOFTMAX_CALIB preset value — the a/b fit does NOT "
             "extrapolate beyond this, so it must cover the deployment "
             "context; short values leave targets in the extrapolated zone)")
    parser_target.add_argument(
        "--chunk-size", type=int, default=4096,
        help="Chunked-prefill size for the calibration forwards (default: "
             "%(default)s per SKIP_SOFTMAX_CALIB; -1 disables chunking)")
    parser_target.add_argument(
        "--target-sparsity", type=float, nargs="+", default=[0.5],
        help="Target sparsity(ies) to print deployment thresholds for "
             "(default: %(default)s, the paper's near-lossless safe-zone bound). "
             "ModelOpt calibrates at the FIRST value; the rest are read off "
             "the fitted a*exp(b*s) curve, not independently calibrated")
    parser_target.add_argument(
        "--max-context", type=int, nargs="+", default=[4096],
        help="Engine max context length(s) for the deployment lambda")
    parser_target.add_argument(
        "--module-pattern", default="*self_attn*",
        help="Wildcard matching the attention modules to calibrate "
             "(default: %(default)s)")
    parser_target.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16"],
        help="Model dtype for the calibration forwards (default: %(default)s)")
    parser_target.add_argument(
        "--gpu", default="0", help="GPU id (default: %(default)s)")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("calibrate", "evaluate", "-h", "--help"):
        argv = ["calibrate"] + argv          # default subcommand
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return evaluate(args)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    from transformers import AutoModelForCausalLM

    import modelopt.torch.sparsity.attention_sparsity as mtsa

    program_banner(f"SKIP-SOFTMAX CALIBRATE — {Path(args.model_dir).name}",
                   f"ModelOpt official, RULER samples={args.samples}, "
                   f"max_seqlen={args.max_seqlen}, targets={args.target_sparsity}")
    stage("calibrate", 1, 3, f"load model (eager attention, {args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=getattr(torch, args.dtype),
        device_map="cuda",
        attn_implementation="eager",       # required: pytorch backend patches softmax
    )
    model.eval()

    config = {
        "sparse_cfg": {
            args.module_pattern: {
                "method": "flash_skip_softmax",
                "backend": "pytorch",
                "enable": True,
            },
            # NOTE: "calibration" is a top-level sparse_cfg key (sibling of the
            # module wildcards) — _extract_calibration_config reads it there.
            "calibration": {
                "target_sparse_ratio": {"prefill": args.target_sparsity[0],
                                        "decode": 0.0},
                "samples": args.samples,
                "max_seqlen": args.max_seqlen,
                "chunk_size": args.chunk_size,
                **({"cache_dir": str(args.cache_dir)} if args.cache_dir else {}),
            },
            "default": {"enable": False},
        },
    }

    stage("calibrate", 2, 3, "ModelOpt calibration "
          "(20 threshold trials, one forward pass — library output below)")
    import contextlib

    from modelopt.torch.sparsity.attention_sparsity.calibration.calibrate import (
        calibrate_sparse_attention,
    )

    print(_c("2", "  ┌─ modelopt library output " + "─" * 40))
    with contextlib.redirect_stdout(_IndentDim()):     # dim+indent modelopt prints
        # sparsify = convert + calibrate, but it discards the calibration
        # results dict (R^2, observed sparsity range, ...). Run the two steps
        # explicitly to keep it.
        model = mtsa.sparsify(model, {"sparse_cfg": {
            k: v for k, v in config["sparse_cfg"].items() if k != "calibration"}})
        results = calibrate_sparse_attention(model, config)
    print(_c("2", "  └─ end modelopt output " + "─" * 43))

    params = results.get("calibration_results", {}).get("prefill")
    if not params:
        print("ERROR: calibration produced no parameters — see modelopt output above")
        return 1

    a, b = float(params["a"]), float(params["b"])
    r2 = params.get("r_squared")
    n_pts = params.get("num_data_points")
    s_lo = params.get("min_observed_sparsity")
    s_hi = params.get("max_observed_sparsity")
    stage("calibrate", 3, 3, "fitted parameters and deployment scale factors")
    lines = ["fit:  scale_factor = a * exp(b * sparsity)",
             f"      a = {a:.6g}   b = {b:.6g}"
             + (f"   R^2 = {float(r2):.3f}" if r2 is not None else "")
             + (f"   ({n_pts} points)" if n_pts else ""),
             ]
    if s_lo is not None and s_hi is not None:
        lines.append(f"      observed sparsity range: [{s_lo:.1%}, {s_hi:.1%}]"
                     "  (targets beyond it are extrapolated)")
    # Primary deliverable: the scale factor S. It is baked into the engine as
    # an AttentionPlugin attribute; the runtime derives lambda = S / L from
    # the actual context length of every request.
    lines.append("")
    lines.append("deployable scale factor S (lambda = S / context_length at runtime):")
    for s_target in args.target_sparsity:
        sf = a * math.exp(b * s_target)
        extra = ("  <- EXTRAPOLATED" if s_hi is not None and s_target > s_hi
                 else "")
        lines.append(f"  target {s_target:>4.0%}  S = {sf:<10.6g}"
                     f"  (tensorrt-edgellm export --skip-softmax-scale-factor"
                     f" {sf:.6g})" + extra)
    # Illustration only: the lambda the engine will actually use at a given L.
    lines.append("")
    lines.append("illustration — lambda the runtime derives at context length L:")
    for s_target in args.target_sparsity:
        sf = a * math.exp(b * s_target)
        for max_ctx in args.max_context:
            lam = sf / max_ctx
            lines.append(f"  target {s_target:>4.0%}  L {max_ctx:<6}  "
                         f"lambda = {lam:<12.6g} (log2 {math.log2(lam):+.2f})")
    result_box(lines)
    print(_c("1;33",
             "NEXT: re-export the model with the chosen S\n"
             "      (python -m tensorrt_edgellm.scripts.export <ckpt> <out> "
             "--skip-softmax-scale-factor <S>),\n"
             "      rebuild the engine, then run `evaluate --baseline <dense "
             "scores>` — deploy the\n      LARGEST S that PASSES (real-text "
             "prefill gain grows with skip ratio at long context)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
