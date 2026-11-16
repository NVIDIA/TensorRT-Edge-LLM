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

The fit is recorded into the checkpoint's ``config.json`` as
``skip_softmax_prefill_calibration`` {a, b} plus
``skip_softmax_prefill_target_sparsity`` (the first target), so a plain
``tensorrt_edgellm.onnx.cli`` or ``tensorrt-edgellm-build`` of that checkpoint
deploys S = a * exp(b * target) with no further flags; ``--no-write-config``
keeps the checkpoint untouched. ``--phase decode`` writes the
``skip_softmax_decode_*`` keys instead; keys without a phase are not read.
With ``--bin-step N`` the tool additionally calibrates S per N-token length
bin and records it as ``skip_softmax_prefill_scale_table`` (rows
``[seqlen_bin, S]``),
which both deployment paths prefer over the single fit: one S drifts off the
target sparsity as the context length moves away from the calibration
anchor, the table holds it. Without ``--target-sparsity`` the whole default
ladder (50 / 60 / 75 / 80 %) is calibrated from the same pass, every tier's
table is kept under ``skip_softmax_prefill_scale_tables`` and 75 % deploys;
``export --prefill-target-sparsity R`` later switches to tier R's table
without recalibrating. The export flags are exactly four, two per phase:
``--skip-softmax-{prefill,decode}-scale-table <json|none>`` and
``--{prefill,decode}-target-sparsity R``; a table flag outranks every
checkpoint key of its phase and ``none`` makes that phase dense. At
inference the kernel derives ``lambda_i = S / seqlen_kv_i`` PER SEQUENCE
(no clamp, no capacity floor) — the kernel artifacts are
lambda-free and built once.

SHAPE ADVISOR — the same calibration pass also prices the deployed table:
ModelOpt counts the tiles a lambda skips (sparsity, predicts speed); this tool
hooks that call and adds the softmax MASS those tiles carry (predicts
accuracy). Read at the deployed lambda per request, ``mass share << sparsity``
means the skipped tiles are near-zero (A-shaped scores: advice ON);
``mass share ~ sparsity`` means every tile carries its count share (flat
scores, e.g. the few full-attention layers of a hybrid: advice OFF — points
traded for ~1.0x). The verdict, both numbers, per-bin rows and the flat layers
are printed and recorded as ``skip_softmax_prefill_shape_advice``; ``--mass-ratio``
moves the ON/OFF line (mass share / sparsity, default 0.3),
``--no-shape-advisor`` skips the measurement. A recommendation only —
`evaluate` / `price` still gate.

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

Three subcommands:

  calibrate (default) — the ModelOpt flow above; venv needs torch,
      transformers, nvidia-modelopt, scipy, wonderwords. Without
      --target-sparsity it also runs `price` for the calibrated ladder when
      the llm_build / llm_bench / llm_inference binaries are found.
  price — export + build one engine per tier (and a dense one), then measure
      TTFT (llm_bench prefill), RULER and LongBench at every --lengths
      entry; prints the tier x length price list and records it in the
      checkpoint config.json ("skip_softmax_price_list").
  evaluate — the paper's accuracy validation: score a DEPLOYED engine on real
      RULER samples (HF dataset simonjegou/ruler, exact-answer matching per
      task), optionally plus an MMLU subset (--mmlu-samples). Run it once per
      engine (dense baseline vs each deployed S) and compare; pair with the
      prefill benchmark (llm_bench) for TTFT.

Examples:

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
        --model-dir /path/to/Qwen3-8B --max-seqlen 40960 --bin-step 2048

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
        --model-dir /path/to/Qwen3-1.7B --max-seqlen 4096 \
        --target-sparsity 0.3 0.5 --max-context 4096

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py price \
        --model-dir /path/to/Qwen3-8B --lengths 8192 16384 32768 40960

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
        --model-dir /path/to/Qwen3-1.7B \
        --engine-dir engines/qwen3-1.7b --llm-inference build/bin/llm_inference \
        --max-context 4096 --per-task 20
"""

import argparse
import ast
import atexit
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_RULER_HF_REPO = "simonjegou/ruler"


#: Skip-softmax arms of one engine: the runtime binds the prefill table on
#: context steps and the decode table on decode steps, each only when its
#: table is present in config.json, so one skip engine measures all four
#: cells through a config copy with the other phase's table removed.
ARMS = ("both", "prefill", "decode", "dense")
_ARM_KEYS = {"prefill": ("skip_softmax_decode_scale_table",),
             "decode": ("skip_softmax_prefill_scale_table",),
             "dense": ("skip_softmax_prefill_scale_table", "skip_softmax_decode_scale_table")}


def _prefill_key(cfg: dict, suffix: str):
    return cfg.get("skip_softmax_prefill_" + suffix)
#: (engine dir, arm) -> TemporaryDirectory, removed by ``_cleanup_arm_dirs``.
_ARM_DIRS: dict = {}


def _cleanup_arm_dirs() -> None:
    for tmp in _ARM_DIRS.values():
        tmp.cleanup()
    _ARM_DIRS.clear()


atexit.register(_cleanup_arm_dirs)


def _arm_engine(engine_dir, arm: str) -> Path:
    """Engine dir to run for ``arm``: the engine itself for ``both``, else a
    cached temp dir that links every engine file and carries a config.json
    without the other phase's table (an absent table = that phase dense)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    engine_dir = Path(engine_dir)
    if arm == "both":
        return engine_dir
    key = (str(engine_dir.resolve()), arm)
    if key not in _ARM_DIRS:
        tmp = tempfile.TemporaryDirectory(prefix=f"{engine_dir.name}-{arm}-")
        d = Path(tmp.name)
        for f in engine_dir.iterdir():
            if f.name != "config.json":
                (d / f.name).symlink_to(f.resolve())
        cfg = json.loads((engine_dir / "config.json").read_text())
        for k in _ARM_KEYS[arm]:
            cfg.pop(k, None)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        _ARM_DIRS[key] = tmp
    return Path(_ARM_DIRS[key].name)


def _infer_env(llm_inference):
    """Env for llm_inference subprocesses: pin EDGELLM_PLUGIN_PATH to the
    plugin NEXT TO the binary's build tree unless the caller already set it.

    Without this, llm_inference dlopens the RELATIVE default
    ``build/lib/libNvInfer_edgellm_plugin.so`` — i.e. the MAIN checkout's plugin
    when CWD is the repo root, even when --llm-inference points at a worktree
    build. Main's plugin lacks d256/d512 skip kernels and silently dispatches
    DENSE → every eval verdict becomes a vacuous dense-vs-dense PASS
    (2026-08-03 incident: all d256 ladder + LongBench verdicts invalidated;
    engagement probes passed because they exported the env var explicitly
    while this subprocess did not — probe and eval MUST share one env)."""
    env = os.environ.copy()
    if not env.get("EDGELLM_PLUGIN_PATH"):
        plug = Path(llm_inference).resolve().parents[1] \
            / "lib" / "libNvInfer_edgellm_plugin.so"
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
        cmd = [str(args.llm_inference), "--engineDir",
               str(_arm_engine(args.engine_dir, getattr(args, "arm", "both"))),
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
        cmd = [str(args.llm_inference), "--engineDir",
               str(_arm_engine(args.engine_dir, getattr(args, "arm", "both"))),
               "--inputFile", str(in_path), "--outputFile", str(out_path)]
        print(_c("2", "  running: " + " ".join(cmd)
                 + (f"   [arm {args.arm}]" if getattr(args, "arm", "both") != "both" else "")))
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
                  "with a smaller scale factor (a lower tier via `export "
                  "--prefill-target-sparsity` / `--decode-target-sparsity`, "
                  "or a lower `calibrate --target-sparsity`) and re-run this "
                  "evaluation.")
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
    p_eval.add_argument("--arm", choices=ARMS, default="both",
                        help="Which skip-softmax phase(s) the engine runs with: "
                        "both (default), prefill only, decode only, or dense "
                        "(the other phase's table removed from a config copy)")

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
        "--target-sparsity", type=float, nargs="+", default=None,
        help="Target sparsity(ies). Omitted: the default ladder "
             f"{DEFAULT_TIERS} is calibrated in one pass, every tier gets its "
             f"own scale table (--bin-step) and {round(DEPLOY_TIER * 100)}%% "
             "deploys by "
             "default; the exporter picks another tier with "
             "--prefill-target-sparsity / --decode-target-sparsity. "
             "Given: the first value is what deploys; ModelOpt fits a/b at the "
             "first value, the rest are read off the fitted curve")
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
    parser_target.add_argument(
        "--no-write-config", dest="write_config", action="store_false",
        help="Do not record the fit (skip_softmax_<phase>_calibration {a, b}, "
             "skip_softmax_<phase>_target_sparsity = first target and, with "
             "--bin-step, skip_softmax_<phase>_scale_table) into the checkpoint's "
             "config.json; export then needs an explicit "
             "--skip-softmax-<phase>-scale-table")
    parser_target.add_argument(
        "--bin-step", type=int, default=0, metavar="N",
        help="Also calibrate a per-length scale table: RULER samples are "
             "generated at every N tokens from --bin-min to --max-seqlen and "
             "S is interpolated per bin for the first target, so the deployed "
             "sparsity stays on target across lengths instead of drifting "
             "with the single a/b fit (default: 0 = a/b fit only)")
    parser_target.add_argument(
        "--bin-min", type=int, default=4096, metavar="L",
        help="Shortest table bin; requests below it run dense "
             "(default: %(default)s)")
    parser_target.add_argument(
        "--samples-per-bin", type=int, default=32, metavar="N",
        help="RULER samples per table bin, the three longest bins get twice "
             "as many (default: %(default)s); overrides --samples with "
             "--bin-step")
    parser_target.add_argument(
        "--phase", choices=("prefill", "decode", "both"), default="prefill",
        help="Attention phase(s) to calibrate (default: %(default)s). decode "
             "runs ModelOpt's decode-phase calibrator (SDPA prefill, then "
             "--num-decode-tokens eager decode steps per sample with the "
             "softmax hook on 128-token KV blocks — the paged decode kernel's "
             "page) and records skip_softmax_decode_* keys; both does the two "
             "phases from one calibration set")
    parser_target.add_argument(
        "--num-decode-tokens", type=int, default=32, metavar="N",
        help="Decode steps generated per sample for the decode-phase "
             "statistics (default: %(default)s)")
    parser_target.add_argument(
        "--no-price", dest="price", action="store_false",
        help="Do not run `price` after calibrating the default ladder")
    parser_target.add_argument(
        "--mass-ratio", type=float, default=FLAT_MASS_RATIO, metavar="R",
        help="Shape advisor line: mass share / sparsity of the skipped tiles "
             "at the deployed lambda (request tail) at or above R reads as a "
             "flat distribution -> advice OFF, below -> ON (default: "
             "%(default)s)")
    parser_target.add_argument(
        "--no-shape-advisor", dest="shape_advisor", action="store_false",
        help="Skip the mass measurement (one extra row-tile softmax per "
             "calibration call) and the advice it produces")
    _add_price_args(p_cal)

    p_price = sub.add_parser(
        "price", help="Export/build every calibrated tier and measure TTFT, "
        "RULER and LongBench per length")
    p_price.add_argument("--model-dir", type=Path, required=True,
                         help="Calibrated HF checkpoint directory")
    p_price.add_argument("--tiers", type=float, nargs="+", default=None,
                         help="Tiers to price (default: every table under "
                         "skip_softmax_prefill_scale_tables, else the deploying one)")
    p_price.add_argument("--gpu", default="0", help="GPU id (default: %(default)s)")
    _add_price_args(p_price)

    p_report = sub.add_parser(
        "report", help="Render the HTML report (kernel and TTFT speedup vs length, "
        "accuracy table) from one or more price runs")
    p_report.add_argument("--price-json", type=Path, nargs="*", default=None,
                          help="price.json files written by `price` (one per model/platform)")
    p_report.add_argument("--model-dir", type=Path, nargs="*", default=None,
                          help="Priced checkpoints (skip_softmax_price_list in config.json)")
    p_report.add_argument("--out", type=Path, default=Path("skip_softmax_report.html"),
                          help="Output HTML path (default: %(default)s)")

    argv = sys.argv[1:]
    if argv and argv[0] not in ("calibrate", "evaluate", "price", "report", "-h", "--help"):
        argv = ["calibrate"] + argv          # default subcommand
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "report":
        return report(args)
    if args.command == "price":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
        return price(args)
    ladder = args.target_sparsity is None
    tiers = list(DEFAULT_TIERS) if ladder else args.target_sparsity
    deploy = DEPLOY_TIER if ladder else tiers[0]
    args.target_sparsity = tiers

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    from transformers import AutoModelForCausalLM

    import modelopt.torch.sparsity.attention_sparsity as mtsa

    bins = []
    if args.bin_step > 0:
        bins = list(range(args.bin_min, args.max_seqlen + 1, args.bin_step))
        if len(bins) < 2:
            print("ERROR: --bin-step needs at least two bins between --bin-min "
                  "and --max-seqlen")
            return 1
        # The longest bins only get their own samples (shorter bins also see
        # the long samples' chunk prefixes), so they are generated twice.
        gen_bins = bins + bins[-3:]
        args.samples = args.samples_per_bin * len(gen_bins)
        from modelopt.torch.sparsity.attention_sparsity.calibration import (
            ruler_dataset,
        )
        ruler_dataset._generate_target_lengths = (
            lambda max_seqlen, num_length_bins=4, min_seqlen=1024:
            sorted(gen_bins, reverse=True))

    phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
    program_banner(f"SKIP-SOFTMAX CALIBRATE — {Path(args.model_dir).name}",
                   f"ModelOpt official, RULER samples={args.samples}, "
                   f"max_seqlen={args.max_seqlen}, targets={args.target_sparsity}, "
                   f"phase={args.phase}"
                   + (f", table bins={bins[0]}..{bins[-1]} step {args.bin_step}"
                      if bins else ""))
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
                # A zero target switches that phase's calibrator off.
                "target_sparse_ratio": {"prefill": deploy if "prefill" in phases else 0.0,
                                        "decode": deploy if "decode" in phases else 0.0},
                "num_decode_tokens": args.num_decode_tokens,
                "samples": args.samples,
                "max_seqlen": args.max_seqlen,
                "chunk_size": args.chunk_size,
                **({"cache_dir": str(args.cache_dir)} if args.cache_dir else {}),
            },
            "default": {"enable": False},
        },
    }

    stage("calibrate", 2, 3, "ModelOpt calibration "
          "(20 threshold trials, one forward pass per phase — library output below)")
    import contextlib

    from modelopt.torch.sparsity.attention_sparsity.calibration.calibrate import (
        calibrate_sparse_attention,
    )

    # Per-sample (length, sparsity-per-threshold) statistics are internal to
    # the calibrator; tap them per phase for the per-bin tables.
    scatter: dict = {phase: [] for phase in phases}
    if bins:
        from modelopt.torch.sparsity.attention_sparsity.calibration.calibrator import (
            DynamicThresholdCalibrator,
        )
        _orig_extract = DynamicThresholdCalibrator._extract_calibration_stats

        def _tap(self, modules, phase=None):
            stats = _orig_extract(self, modules, phase=phase)
            if phase in scatter:
                scatter[phase].append({
                    "thresholds": [float(t) for t in self.threshold_trials],
                    "stats": [{"sample_length": s.get("sample_length"),
                               "sparsity": s.get("sparsity")} for s in stats],
                })
            return stats

        DynamicThresholdCalibrator._extract_calibration_stats = _tap

    mass_scatter: list[dict] = []
    if args.shape_advisor:
        install_mass_tap(mass_scatter)

    print(_c("2", "  ┌─ modelopt library output " + "─" * 40))
    with contextlib.redirect_stdout(_IndentDim()):     # dim+indent modelopt prints
        # sparsify = convert + calibrate, but it discards the calibration
        # results dict (R^2, observed sparsity range, ...). Run the two steps
        # explicitly to keep it.
        model = mtsa.sparsify(model, {"sparse_cfg": {
            k: v for k, v in config["sparse_cfg"].items() if k != "calibration"}})
        results = calibrate_sparse_attention(model, config)
    print(_c("2", "  └─ end modelopt output " + "─" * 43))

    fits = {}
    for phase in phases:
        params = results.get("calibration_results", {}).get(phase)
        if not params:
            print(f"ERROR: {phase} calibration produced no parameters — see "
                  "modelopt output above")
            return 1
        fits[phase] = params

    stage("calibrate", 3, 3, "fitted parameters and deployment scale factors")
    lines = []
    tables_by_phase: dict = {}
    for phase in phases:
        params = fits[phase]
        a, b = float(params["a"]), float(params["b"])
        r2 = params.get("r_squared")
        n_pts = params.get("num_data_points")
        s_lo = params.get("min_observed_sparsity")
        s_hi = params.get("max_observed_sparsity")
        if lines:
            lines.append("")
        lines += [f"[{phase}] fit:  scale_factor = a * exp(b * sparsity)",
                  f"      a = {a:.6g}   b = {b:.6g}"
                  + (f"   R^2 = {float(r2):.3f}" if r2 is not None else "")
                  + (f"   ({n_pts} points)" if n_pts else ""),
                  ]
        if s_lo is not None and s_hi is not None:
            lines.append(f"      observed sparsity range: [{s_lo:.1%}, {s_hi:.1%}]"
                         "  (targets beyond it are extrapolated)")
        # Primary deliverable: S, deployed as the one-row table [[0, S]] in the
        # engine config.json; the runtime derives lambda = S / L per request.
        lines.append("")
        lines.append(f"[{phase}] deployable scale factor S (lambda = S / "
                     "context_length at runtime):")
        for s_target in args.target_sparsity:
            sf = a * math.exp(b * s_target)
            extra = ("  <- EXTRAPOLATED" if s_hi is not None and s_target > s_hi
                     else "")
            lines.append(f"  target {s_target:>4.0%}  S = {sf:<10.6g}"
                         f"  (tensorrt-edgellm export --{phase}-target-sparsity "
                         f"{s_target:g})" + extra)
        # Illustration only: the lambda the engine will actually use at a given L.
        lines.append("")
        lines.append(f"[{phase}] illustration — lambda the runtime derives at "
                     "context length L:")
        for s_target in args.target_sparsity:
            sf = a * math.exp(b * s_target)
            for max_ctx in args.max_context:
                lam = sf / max_ctx
                lines.append(f"  target {s_target:>4.0%}  L {max_ctx:<6}  "
                             f"lambda = {lam:<12.6g} (log2 {math.log2(lam):+.2f})")
        tables = {}
        if bins:
            if not scatter[phase]:
                print(f"ERROR: no per-sample {phase} statistics captured for the table")
                return 1
            for tier in tiers:
                table, per_bin = build_scale_table(scatter[phase][-1], bins, tier)
                if table:
                    tables[tier_key(tier)] = table
                lines.append("")
                lines.append(f"[{phase}] per-length scale table (target {tier:.0%}, "
                             f"{args.bin_step}-token bins; S=1 = dense guard"
                             + (", deploys by default)" if tier == deploy else ")"))
                for L in bins:
                    n, lo, hi, s = per_bin[L]
                    lines.append(f"  L {L:<6} n={n:<3} sparsity[{lo:.2f}, {hi:.2f}]  "
                                 + (f"S = {s}" if s else "S = 1 (target out of range)"))
        tables_by_phase[phase] = tables
    # The advisor reads the deployed table; when the deploy tier has no table
    # (target out of range everywhere — the hybrid signature) it reads the
    # highest tier that has one, else the a/b fit, so the verdict still
    # explains WHY the ladder found no solution.
    advice = None
    if mass_scatter and "prefill" in phases:
        a, b = float(fits["prefill"]["a"]), float(fits["prefill"]["b"])
        tables = tables_by_phase["prefill"]
        advised_tier = deploy
        if bins and tier_key(deploy) not in tables:
            advised_tier = max((t for t in tiers if tier_key(t) in tables),
                               default=deploy)
        deploy_table = tables.get(tier_key(advised_tier)) or [
            [0, int(round(a * math.exp(b * advised_tier)))]]
        advice = advise_shape(mass_scatter, deploy_table, bins or None,
                              args.mass_ratio)
        if advice:
            advice["tier"] = tier_key(advised_tier)
        else:
            # No lambda < 1 reaches the tier anywhere: read each request at
            # its own most-sparse trial instead, so the mass still gets priced.
            advice = advise_shape(mass_scatter, None, bins or None,
                                  args.mass_ratio, target=advised_tier)
            if advice:
                advice["tier"] = f"{tier_key(advised_tier)} ask, no deployable S"
    if advice:
        lines.append("")
        lines.append(f"shape advisor (mass priced against sparsity at the "
                     f"{advice['tier']} lambda, {advice['requests']} requests):")
        lines.append(f"  sparsity {advice['sparsity']:.1%}   mass share tail "
                     f"{advice['mass_share_tail']:.1%} (mean "
                     f"{advice['mass_share_mean']:.1%})   ratio "
                     + (f"{advice['ratio']:.2f}" if advice["ratio"] is not None
                        else "n/a")
                     + f"   -> {advice['verdict'].upper()}")
        lines.append(f"  {advice['reason']}")
        for L, row in advice.get("per_bin", {}).items():
            lines.append(f"  L {L:<6} sparsity {row['sparsity']:.1%}  mass tail "
                         f"{row['mass_share_tail']:.1%}  -> {row['verdict']}")
        if advice["flat_layers"]:
            lines.append(f"  flat layers: {len(advice['flat_layers'])} / "
                         f"{advice['num_layers']} "
                         f"(indices {advice['flat_layers']})")
    result_box(lines)
    if advice and advice["verdict"] == "off":
        print(_c("1;31", "ADVICE: OFF — the skipped tiles carry as much mass as they "
                 "count for; the calibrated tables are recorded, but deploying "
                 "them trades points for little speed. Prefer the dense engine."))
    for phase in phases:
        if bins and tier_key(deploy) not in tables_by_phase[phase]:
            print(f"ERROR: the {deploy:.0%} {phase} target is out of range at "
                  "every bin")
            return 1
    if args.write_config:
        recorded = []
        for phase in phases:
            params = fits[phase]
            tables = tables_by_phase[phase]
            cfg_path = record_calibration(
                args.model_dir, float(params["a"]), float(params["b"]), deploy,
                tables.get(tier_key(deploy)), tables or None, phase=phase,
                shape_advice=advice if phase == "prefill" else None)
            recorded.append(f"skip_softmax_{phase}_calibration + target {deploy:.0%}"
                            + (f" + {len(tables)} scale table(s) ({', '.join(tables)})"
                               if tables else ""))
        print(_c("1;33",
                 "recorded " + "; ".join(recorded) + f" in {cfg_path}\n"
                 "NEXT: export (or tensorrt-edgellm-build) the checkpoint as "
                 "usual — S is derived from config.json;\n"
                 "      re-target later with `export --prefill-target-sparsity R` / "
                 "`--decode-target-sparsity R` (picks that tier's table, no "
                 "recalibration), then `evaluate --baseline <dense scores>`."))
        tables = tables_by_phase[phases[0]]
        if tables and args.price and ladder:
            args.lengths = price_lengths(args.lengths, args.max_seqlen)
            if not args.lengths:
                print(_c("1;33", "price skipped — no --lengths entry is <= "
                         f"--max-seqlen {args.max_seqlen}"))
                return 0
            del model
            torch.cuda.empty_cache()
            args.tiers = [t for t in tiers if tier_key(t) in tables]
            return price(args)
    else:
        print(_c("1;33",
                 "NEXT: re-export the model with the chosen S as a one-row table\n"
                 "      (python -m tensorrt_edgellm.onnx.cli <ckpt> <out> "
                 "--skip-softmax-<phase>-scale-table <json holding [[0, S]]>),\n"
                 "      rebuild the engine, then run `evaluate --baseline <dense "
                 "scores>` — deploy the\n      LARGEST S that PASSES (real-text "
                 "prefill gain grows with skip ratio at long context)."))
    return 0


#: Samples whose sparsity never exceeds this at any trial threshold are
#: chunk-boundary artifacts (a bin's last, nearly empty chunk), not evidence.
DEGENERATE_SPARSITY = 0.05

#: Tier ladder calibrated when --target-sparsity is omitted, and the tier that
#: deploys unless the user picks another one at export.
DEFAULT_TIERS = (0.5, 0.6, 0.75, 0.8)
DEPLOY_TIER = 0.75


def tier_key(target: float) -> str:
    """config.json key of a tier's table, e.g. 0.75 -> "75%"."""
    return f"{round(target * 100):d}%"


def build_scale_table(tap: dict, bins: list[int], target: float):
    """Interpolate S per length bin from the calibrator's per-sample stats.

    ``tap`` is one ``_extract_calibration_stats`` capture: ``thresholds``
    (the trial lambdas) and ``stats`` (per sample: ``sample_length`` and
    ``sparsity`` per threshold). Each sample joins its nearest bin; the bin's
    mean sparsity curve is inverted at ``target`` and scaled to S = lambda*L.
    Rows above the first calibratable bin that fall out of range get S=1, the
    runtime's dense guard; rows below it are omitted so the runtime treats
    those lengths as below the table.

    Returns ``(table_rows, per_bin)`` with ``per_bin[L] = (n, min_sp, max_sp,
    S_or_None)``.
    """
    thresholds = [float(t) for t in tap["thresholds"]]
    rows: dict[int, list[list[float]]] = {L: [] for L in bins}
    for s in tap["stats"]:
        L, sp = s.get("sample_length"), s.get("sparsity")
        if not L or not sp or max(sp) < DEGENERATE_SPARSITY:
            continue
        rows[min(bins, key=lambda x: abs(x - L))].append([float(v) for v in sp])

    per_bin = {}
    for L in bins:
        entries = rows[L]
        if not entries:
            per_bin[L] = (0, 0.0, 0.0, None)
            continue
        mean_sp = [sum(col) / len(col) for col in zip(*entries)]
        order = sorted(range(len(mean_sp)), key=mean_sp.__getitem__)
        sp_sorted = [mean_sp[i] for i in order]
        lam_sorted = [thresholds[i] for i in order]
        scale = None
        if sp_sorted[0] <= target <= sp_sorted[-1]:
            scale = int(round(_interp(target, sp_sorted, lam_sorted) * L))
        per_bin[L] = (len(entries), sp_sorted[0], sp_sorted[-1], scale)

    table, started = [], False
    for L in bins:
        scale = per_bin[L][3]
        if scale:
            table.append([L, scale])
            started = True
        elif started:
            table.append([L, 1])
    return table, per_bin


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            if xs[i] == xs[i - 1]:
                return ys[i]
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


# ---------------------------------------------------------------------------
# shape advisor: softmax MASS of the skipped tiles priced against their COUNT
# ---------------------------------------------------------------------------

#: mass share / sparsity at or above which the skipped tiles are NOT near-zero:
#: A-shaped scores measure ~0.02-0.06 (the tiles skipped carry almost nothing),
#: flat ones ~1 (each tile carries its count share). The line sits between.
FLAT_MASS_RATIO = 0.3


def tile_mass(attn_weights, br: int, bc: int):
    """Softmax-numerator mass sum(exp(x - x_rowmax)) of every (br x bc) tile,
    [batch, heads, n_row_tiles, n_col_tiles] float32. Rows are independent,
    so the exp runs one row tile at a time and never holds a second full-size
    copy of ``attn_weights``."""
    import torch
    batch, heads, seq_q, seq_k = attn_weights.shape
    n_rows, n_cols = -(-seq_q // br), -(-seq_k // bc)
    mass = torch.zeros(batch, heads, n_rows, n_cols, dtype=torch.float32,
                       device=attn_weights.device)
    for r in range(n_rows):
        rows = attn_weights[:, :, r * br:(r + 1) * br, :].float()
        p = torch.exp(rows - rows.amax(dim=-1, keepdim=True)).sum(dim=-2)  # [B, H, seq_k]
        if n_cols * bc != seq_k:
            p = torch.nn.functional.pad(p, (0, n_cols * bc - seq_k))
        mass[:, :, r] = p.view(batch, heads, n_cols, bc).sum(dim=-1)
    return mass


def tile_mass_share(attn_weights, log_thresholds: "list[float]", br: int,
                    bc: int) -> "list[float]":
    """Fraction of the total softmax mass carried by the tiles the skip rule
    drops, one value per ``log_thresholds`` entry (natural log of lambda).

    The tile decision is ModelOpt's ``flash_skip_softmax`` rule: a tile is
    kept iff some row has ``tile_max - running_max > ln(lambda)``, with the
    running max taken across column tiles (streaming semantics)."""
    import torch
    batch, heads, seq_q, seq_k = attn_weights.shape
    n_rows, n_cols = -(-seq_q // br), -(-seq_k // bc)
    mass = tile_mass(attn_weights, br, bc)
    padded = torch.nn.functional.pad(
        attn_weights, (0, n_cols * bc - seq_k, 0, n_rows * br - seq_q),
        value=torch.finfo(attn_weights.dtype).min)
    tile_max = padded.view(batch, heads, n_rows, br, n_cols, bc).max(dim=-1)[0]
    del padded
    gap = tile_max - tile_max.cummax(dim=-1)[0]                # [B, H, nr, br, nc]
    del tile_max
    total = mass.sum()
    shares = []
    for log_threshold in log_thresholds:
        kept = (gap > log_threshold).any(dim=-2)               # [B, H, nr, nc]
        shares.append(float((mass * ~kept).sum() / total))
    return shares


def install_mass_tap(scatter: list) -> None:
    """Add the mass measurement to ModelOpt's calibration pass without touching
    ModelOpt: wrap ``FlashSkipSoftmax.calc_correction_factor_and_p`` so every
    prefill call in calibration mode appends {layer, sample_length,
    thresholds, sparsity, mass_share} to ``scatter``. The calibrator's own
    stats path is unchanged (its stats manager only forwards sparsity)."""
    from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
        FlashSkipSoftmax,
    )
    original = FlashSkipSoftmax.calc_correction_factor_and_p

    def tapped(self, attn_weights, phase):
        mask, stats = original(self, attn_weights, phase)
        if phase == "prefill" and self._calibration_mode:
            thresholds = [float(t) for t in self.thresholds]
            scatter.append({
                "layer": id(self),
                "sample_length": int(stats["sample_length"]),
                "thresholds": thresholds,
                "sparsity": [float(s) for s in stats["sparsity"]],
                "mass_share": tile_mass_share(
                    attn_weights, [math.log(t) for t in thresholds], self.br, self.bc),
            })
        return mask, stats

    FlashSkipSoftmax.calc_correction_factor_and_p = tapped


def lookup_scale(table: "list[list[int]]", seq_len: int,
                 bins: "list[int] | None" = None) -> "int | None":
    """S the table deploys for a sequence of ``seq_len``, or None when it runs
    dense (no row, the S=1 guard row, or S >= L).

    Without ``bins`` this is the kernel's floor lookup (last row with
    bin <= seq_len). With ``bins`` a calibration sample is read at its NEAREST
    bin — the convention ``build_scale_table`` built the row from — so a
    sample a few tokens short of its bin is priced, not dropped."""
    if bins:
        nearest = min(bins, key=lambda b: abs(b - seq_len))
        scale = dict((int(b), int(s)) for b, s in table).get(nearest)
    else:
        scale = None
        for bin_len, s in table:
            if bin_len <= seq_len:
                scale = s
    return scale if scale and 1 < scale < seq_len else None


def shape_verdict(sparsity: float, mass_share: float,
                  ratio: float = FLAT_MASS_RATIO) -> "tuple[str, str]":
    """(verdict, reason) from the two advisor numbers at the deployed lambda."""
    if sparsity > 0 and mass_share / sparsity >= ratio:
        return ("off", "mass share ~ sparsity: flat score distribution, every "
                "skipped tile is paid at full price — keep the engine dense")
    return ("on", "mass share << sparsity: the skipped tiles carry next to "
            "nothing")


def advise_shape(scatter: "list[dict]", table: "list[list[int]] | None",
                 bins: "list[int] | None" = None,
                 ratio: float = FLAT_MASS_RATIO,
                 target: "float | None" = None) -> "dict | None":
    """Price the deployed table on the calibration traffic.

    Every tapped (layer, call) point is read at the lambda the kernel would
    use for its length (``lookup_scale`` / L, curves interpolated in ln
    lambda); points that run dense are dropped. With ``table=None`` a point
    is read instead at the lambda where ITS OWN sparsity curve reaches
    ``target`` (its most sparse trial when unreachable) — the "mass at the
    s ask" reading for models whose ladder found no deployable S. Points are
    pooled per request (the k-th prefill call of every layer is the same
    request; the request's share is the layer mean). Returns the pooled
    sparsity, the tail (max over requests) and mean mass share, their ratio,
    the verdict, the same per length bin, and the layers whose own ratio is
    flat."""
    layer_ids: dict[int, int] = {}
    calls: dict[int, int] = defaultdict(int)
    per_request: dict[int, list] = defaultdict(list)   # call idx -> [(layer, L, s, m)]
    for pt in scatter:
        layer = layer_ids.setdefault(pt["layer"], len(layer_ids))
        idx = calls[layer]
        calls[layer] += 1
        length = pt["sample_length"]
        xs = [math.log(t) for t in pt["thresholds"]]
        if table is not None:
            scale = lookup_scale(table, length, bins)
            if scale is None:
                continue
            x = math.log(scale / length)
        else:
            order = sorted(range(len(xs)), key=pt["sparsity"].__getitem__)
            if pt["sparsity"][order[-1]] < DEGENERATE_SPARSITY:
                continue
            x = _interp(target, [pt["sparsity"][i] for i in order],
                        [xs[i] for i in order])
        per_request[idx].append((layer, length, _interp(x, xs, pt["sparsity"]),
                                 _interp(x, xs, pt["mass_share"])))
    requests = [(max(L for _, L, _, _ in pts),
                 statistics.fmean(s for _, _, s, _ in pts),
                 statistics.fmean(m for _, _, _, m in pts))
                for pts in per_request.values() if pts]
    if not requests:
        return None

    def summarize(reqs):
        sparsity = statistics.fmean(s for _, s, _ in reqs)
        tail = max(m for _, _, m in reqs)
        verdict, reason = shape_verdict(sparsity, tail, ratio)
        return {
            "requests": len(reqs),
            "sparsity": round(sparsity, 4),
            "mass_share_tail": round(tail, 4),
            "mass_share_mean": round(statistics.fmean(m for _, _, m in reqs), 4),
            "ratio": round(tail / sparsity, 4) if sparsity > 0 else None,
            "verdict": verdict,
            "reason": reason,
        }

    advice = {"flat_ratio": ratio, **summarize(requests)}
    if bins:
        per_bin = defaultdict(list)
        for req in requests:
            per_bin[min(bins, key=lambda b: abs(b - req[0]))].append(req)
        advice["per_bin"] = {str(L): summarize(per_bin[L]) for L in bins if per_bin[L]}
    layer_stats = defaultdict(list)
    for pts in per_request.values():
        for layer, _, s, m in pts:
            layer_stats[layer].append((s, m))
    advice["num_layers"] = len(layer_ids)
    advice["flat_layers"] = sorted(
        layer for layer, sm in layer_stats.items()
        if shape_verdict(statistics.fmean(s for s, _ in sm),
                         statistics.fmean(m for _, m in sm), ratio)[0] == "off")
    return advice


def record_calibration(model_dir: Path, a: float, b: float,
                       target_sparsity: float,
                       scale_table: "list[list[int]] | None" = None,
                       scale_tables: "dict[str, list[list[int]]] | None" = None,
                       phase: str = "prefill",
                       shape_advice: "dict | None" = None) -> Path:
    """Merge the fit into the checkpoint config.json; other keys are kept.

    Written at the root (not text_config) — both the ONNX exporter and the
    ONNX-less builder read these keys from the root first. ``scale_table`` is
    what deploys; ``scale_tables`` keeps every calibrated tier by key ("75%")
    so a later export can switch tier without recalibrating. Tables take
    precedence over the a/b formula in both front-ends; stale tables from an
    earlier run are removed so the new fit is what deploys. Keys are
    ``skip_softmax_<phase>_*``; a prefill run also drops the unread pre-prefix
    ``skip_softmax_*`` spellings. ``shape_advice`` (``advise_shape`` output) is
    recorded as ``skip_softmax_<phase>_shape_advice``; the front-ends do not
    read it.
    """
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")
    prefix = f"skip_softmax_{phase}_"
    cfg_path = Path(model_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if phase == "prefill":
        for suffix in ("calibration", "target_sparsity", "scale_table",
                       "scale_tables"):
            cfg.pop("skip_softmax_" + suffix, None)
    cfg[prefix + "calibration"] = {
        "a": a, "b": b, "formula": "a * exp(b * target_sparsity)"}
    cfg[prefix + "target_sparsity"] = float(target_sparsity)
    if scale_table:
        cfg[prefix + "scale_table"] = [[int(L), int(s)] for L, s in scale_table]
    else:
        cfg.pop(prefix + "scale_table", None)
    if scale_tables:
        cfg[prefix + "scale_tables"] = {
            k: [[int(L), int(s)] for L, s in rows] for k, rows in scale_tables.items()}
    else:
        cfg.pop(prefix + "scale_tables", None)
    if shape_advice:
        cfg[prefix + "shape_advice"] = shape_advice
    else:
        cfg.pop(prefix + "shape_advice", None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# price: TTFT / RULER / LongBench per tier and prompt length
# ---------------------------------------------------------------------------

#: Prompt lengths priced per tier; entries above the calibrated --max-seqlen
#: are dropped because the tables carry no S for them.
DEFAULT_PRICE_LENGTHS = (8192, 16384, 32768, 40960)

#: Longest RULER dataset configuration; longer lengths reuse the same samples,
#: so RULER is scored only up to here.
# RULER samples come in fixed 4k / 8k / 16k configurations; a length between two of
# them fits none of the samples of the larger one, so RULER is scored only at these.
RULER_LENGTHS = (4096, 8192, 16384)

_INFER_TTFT_RE = re.compile(r"Average Time per Run:\s*([0-9.]+)\s*ms")
#: nsys kernel names of both prefill FMHA families (optimized Blackwell / FMHA-v2).
DEFAULT_ATTENTION_KERNEL_REGEX = "BlackwellFusedMultiHeadAttention|FMHAV2Ampere"
#: Real-text prompt set: cnn_dailymail articles concatenated per request, tokenized to
#: the target length; distinct article start per request defeats prefix caching.
_PROMPT_DATASET = ("abisee/cnn_dailymail", "3.0.0", "validation")
_PROMPT_REQUESTS = 4
_PROMPT_MESSAGE_BYTE_CAP = 120 * 1024


def _add_price_args(p) -> None:
    ex = _repo_root() / "build" / "bin"
    p.add_argument("--lengths", type=int, nargs="+",
                   default=list(DEFAULT_PRICE_LENGTHS), metavar="L",
                   help="Prompt lengths priced per tier (default: %(default)s)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where per-tier ONNX exports and engines go (default: "
                   "<model-dir>-skip-softmax-price next to the checkpoint); "
                   "existing exports/engines there are reused")
    p.add_argument("--llm-build", type=Path, default=ex / "llm_build",
                   help="Engine builder binary (default: %(default)s)")
    p.add_argument("--llm-bench", type=Path, default=ex / "llm_bench",
                   help="Prefill TTFT benchmark binary (default: %(default)s)")
    p.add_argument("--llm-inference", type=Path, default=ex / "llm_inference",
                   help="Inference binary for RULER/LongBench "
                   "(default: %(default)s)")
    p.add_argument("--bench-iterations", type=int, default=5,
                   help="llm_bench --iterations per TTFT run (default: %(default)s)")
    p.add_argument("--bench-repeats", type=int, default=3,
                   help="llm_bench invocations per (tier, length); the median "
                   "is reported (default: %(default)s)")
    p.add_argument("--ruler-per-task", type=int, default=20,
                   help="RULER samples per task (default: %(default)s)")
    p.add_argument("--longbench-per-task", type=int, default=30,
                   help="LongBench samples per task (default: %(default)s)")
    p.add_argument("--no-accuracy", dest="accuracy", action="store_false",
                   help="TTFT / TPOT only")
    p.add_argument("--no-perf", dest="perf", action="store_false",
                   help="RULER + LongBench only")
    p.add_argument("--arms", nargs="+", choices=("both", "prefill", "decode"),
                   default=None, metavar="ARM",
                   help="Skip-softmax arms scored per tier engine: `both` is the "
                   "deployment (TTFT, TPOT and accuracy); `prefill` / `decode` "
                   "switch the other phase off (its table removed from a config copy) "
                   "and add an accuracy-only ablation row (default: both plus "
                   "the single arms when the checkpoint carries decode tables)")
    p.add_argument("--tpot-osl", type=int, default=256, metavar="N",
                   help="Tokens generated per TPOT measurement on a real-text "
                   "prompt of each --lengths entry (default: %(default)s)")
    p.add_argument("--no-tpot", dest="tpot", action="store_false",
                   help="Skip the decode TPOT measurement")
    p.add_argument("--accuracy-lengths", type=int, nargs="+", default=None,
                   metavar="L", help="Subset of --lengths that gets RULER/LongBench "
                   "(default: all); TTFT still covers every length")
    p.add_argument("--ttft-source", choices=("real", "bench"), default="real",
                   help="real: llm_inference on cnn_dailymail prompts of each length "
                   "(the skip predicate needs real text); bench: llm_bench --mode "
                   "prefill on synthetic tokens (default: %(default)s)")
    p.add_argument("--no-kernel-profile", dest="kernel_profile", action="store_false",
                   help="Skip the nsys pass that measures the attention kernel's own "
                   "speedup and its share of GPU time (default: on when nsys is found)")
    p.add_argument("--attention-kernel-regex", default=DEFAULT_ATTENTION_KERNEL_REGEX,
                   help="nsys kernel-name regex summed as 'attention' "
                   "(default: %(default)s)")
    p.add_argument("--report", type=Path, default=None,
                   help="HTML report path (default: <work-dir>/skip_softmax_report.html)")
    p.add_argument("--export-only", action="store_true",
                   help="Only write the dense and per-tier ONNX exports into --work-dir "
                   "(needs torch); a later `price` on the target device reuses them, "
                   "so devices without an export stack only build and measure")


def price_lengths(requested: "list[int]", max_seqlen: int) -> "list[int]":
    """Requested lengths the calibrated tables cover, ascending (may be empty)."""
    return sorted({L for L in requested if L <= max_seqlen})


def parse_bench_ms(report: dict) -> float:
    """Read the production prefill-stage mean; this is not HTTP TTFT."""
    if report.get("schema_version") != 2 or report.get("mode") != "fixed_round":
        raise ValueError("Expected a schema-v2 fixed_round benchmark report")
    stage = report.get("statistics", {}).get("invocation_gpu_ms", {}).get("llm_prefill", {})
    value = stage.get("mean")
    if stage.get("count", 0) <= 0 or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Benchmark report has no valid prefill-stage samples")
    return float(value)


def config_tiers(cfg: dict) -> "list[float]":
    """Tiers a calibrated checkpoint carries tables for (deploying one first)."""
    tables = _prefill_key(cfg, "scale_tables") or {}
    tiers = sorted(int(k.rstrip("%")) / 100 for k in tables)
    deploy = _prefill_key(cfg, "target_sparsity")
    if deploy is not None and deploy not in tiers:
        tiers.insert(0, float(deploy))
    if not tiers:
        raise ValueError("config.json carries no skip-softmax calibration; run "
                         "`calibrate` first")
    return tiers


def _run(cmd: list, log: Path, env=None, cwd=None) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run([str(c) for c in cmd], stdout=fh,
                            stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode
    text = log.read_text(errors="replace")
    if rc != 0:
        tail = "\n".join(text.splitlines()[-20:])
        raise RuntimeError(f"{cmd[0]} failed (rc={rc}), see {log}\n{tail}")
    return text


def _export(args, tag: str, export_flags: list, env) -> Path:
    """ONNX export under work-dir/onnx/<tag>; reuses a finished one."""
    onnx = args.work_dir / "onnx" / tag
    if not (onnx / "llm" / "config.json").exists():
        _run([sys.executable, "-m", "tensorrt_edgellm.onnx.cli",
              args.model_dir, onnx, *export_flags],
             args.work_dir / "logs" / f"export_{tag}.log", env=env, cwd=_repo_root())
    return onnx


def _engine(args, tag: str, export_flags: list, max_len: int, env) -> Path:
    """Export + build one engine under work-dir/<tag>; reuses a finished one."""
    onnx = _export(args, tag, export_flags, env)
    eng = args.work_dir / "engines" / tag
    logs = args.work_dir / "logs"
    if not (eng / "llm.engine").exists():
        _run([args.llm_build, "--onnxDir", onnx / "llm", "--engineDir", eng,
              "--maxBatchSize=1", f"--maxInputLen={max_len}",
              f"--maxKVCacheCapacity={max_len + 512}"],
             logs / f"build_{tag}.log", env=env)
    return eng


def _ttft_ms(args, eng: Path, L: int, env) -> float:
    vals = []
    reports = args.work_dir / "bench_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for r in range(args.bench_repeats):
        run_dir = Path(tempfile.mkdtemp(prefix=f"{eng.name}_{L}_{r}_", dir=reports)) / "run"
        _run([args.llm_bench, "--engineDir", eng, "--checkpointDir", args.model_dir,
              "--mode", "fixed_round", "--inputLen", L, "--outputDir", run_dir,
              "--iterations", args.bench_iterations, "--warmup", 2, "--batchSize", 1],
             args.work_dir / "logs" / f"bench_{eng.name}_{L}_{r}.log", env=env)
        vals.append(parse_bench_ms(json.loads((run_dir / "report.json").read_text())))
    return statistics.median(vals)


def parse_inference_ttft_ms(text: str) -> "float | None":
    """TTFT from ``llm_inference --dumpProfile`` output, or None if absent."""
    hits = _INFER_TTFT_RE.findall(text)
    return float(hits[-1]) if hits else None


def build_real_text_requests(articles: "list[str]", tokenizer, length: int,
                             n_requests: int = _PROMPT_REQUESTS,
                             byte_cap: int = _PROMPT_MESSAGE_BYTE_CAP) -> dict:
    """llm_inference request file with ``n_requests`` real-text prompts of
    ``length`` tokens (chat-template headroom excluded) and one generated token.

    Request r starts at article ``37 * r`` so no two share a prefix; a message
    is split at ``byte_cap`` because the runtime caps one message at 128 KiB.
    """
    body_tokens = max(length - 64, 32)
    requests = []
    for r in range(n_requests):
        parts, i = [], 37 * r
        while True:
            parts.append(articles[i % len(articles)])
            ids = tokenizer("\n\n".join(parts), add_special_tokens=False).input_ids
            if len(ids) >= body_tokens:
                break
            i += 1
        text = tokenizer.decode(ids[:body_tokens]).encode()
        messages = []
        while text:
            piece, text = text[:byte_cap], text[byte_cap:]
            messages.append({"role": "user", "content": piece.decode(errors="ignore")})
        requests.append({"messages": messages})
    return {"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1,
            "max_generate_length": 1, "requests": requests}


def _real_prompts(args, L: int) -> Path:
    """Cached ``work-dir/prompts/L<L>.json`` real-text request file."""
    path = args.work_dir / "prompts" / f"L{L}.json"
    if path.exists():
        return path
    from datasets import load_dataset
    from transformers import AutoTokenizer
    name, config, split = _PROMPT_DATASET
    articles = [row["article"] for row in
                load_dataset(name, config, split=split).select(range(400))]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    payload = build_real_text_requests(articles, tokenizer, L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _ttft_real_ms(args, eng: Path, L: int, env) -> float:
    prompts = _real_prompts(args, L)
    vals = []
    for r in range(args.bench_repeats):
        text = _run([args.llm_inference, "--engineDir", eng, "--inputFile", prompts,
                     "--outputFile", args.work_dir / "logs" / "last_output.json",
                     "--dumpProfile", "--warmup", 1],
                    args.work_dir / "logs" / f"ttft_{eng.name}_{L}_{r}.log", env=env)
        ms = parse_inference_ttft_ms(text)
        if ms is None:
            raise RuntimeError(f"llm_inference printed no 'Average Time per Run' "
                               f"for {eng} @ {L}")
        vals.append(ms)
    return statistics.median(vals)


def parse_nsys_kernel_sum(csv_text: str, attention_regex: str) -> "tuple[float, float]":
    """(attention ms, all-kernel ms) from ``nsys stats -r cuda_gpu_kern_sum -f csv``."""
    import csv
    import io
    pattern = re.compile(attention_regex)
    attention = total = 0.0
    for row in csv.DictReader(io.StringIO(csv_text)):
        ns = float(row["Total Time (ns)"])
        total += ns
        if pattern.search(row["Name"]):
            attention += ns
    return attention / 1e6, total / 1e6


def _kernel_profile(args, eng: Path, L: int, env) -> "tuple[float, float]":
    """One profiled llm_inference run: (attention kernel ms, all-kernel ms)."""
    prompts = _real_prompts(args, L)
    rep = args.work_dir / "nsys" / f"{eng.name}_L{L}"
    rep.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".nsys-rep", ".sqlite"):
        rep.with_suffix(suffix).unlink(missing_ok=True)
    _run(["nsys", "profile", "-o", rep, "-t", "cuda", "--cuda-graph-trace=node",
          "-f", "true", args.llm_inference, "--engineDir", eng, "--inputFile", prompts,
          "--outputFile", args.work_dir / "logs" / "last_output.json", "--warmup", 0],
         args.work_dir / "logs" / f"nsys_{eng.name}_{L}.log", env=env)
    csv_text = _run(["nsys", "stats", "-q", "-r", "cuda_gpu_kern_sum", "-f", "csv",
                     rep.with_suffix(".nsys-rep")],
                    args.work_dir / "logs" / f"nsys_stats_{eng.name}_{L}.log", env=env)
    return parse_nsys_kernel_sum(csv_text, args.attention_kernel_regex)


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else "unknown GPU"
    except (OSError, subprocess.SubprocessError):
        return "unknown GPU"


def _real_prompt(args, L: int) -> Path:
    """Request file with one real-text prompt of ~L tokens (RULER contexts,
    concatenated and cut to L - 64 tokens) generating --tpot-osl tokens.
    Cached per length in the work dir: synthetic tokens never light the skip
    predicate, so TPOT is measured on real text."""
    path = args.work_dir / "prompts" / f"tpot_L{L}_osl{args.tpot_osl}.json"
    if path.exists():
        return path
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ds = load_dataset(_RULER_HF_REPO, "16384", split="test").shuffle(seed=0)
    budget = L - 64
    ids: list = []
    for row in ds:
        ids += tokenizer(row["context"] + "\n").input_ids
        if len(ids) >= budget:
            break
    text = tokenizer.decode(ids[:budget], skip_special_tokens=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"temperature": 1.0, "top_p": 1.0, "top_k": 1,
               "max_generate_length": args.tpot_osl, "ignore_eos": True,
               "apply_chat_template": False,
               "requests": [{"messages": [{"role": "user", "content": text}]}]},
              open(path, "w"))
    return path


def _tpot_ms(args, eng: Path, L: int, env, arm: str = "both") -> float:
    """Median decode time per token (llm_inference profile JSON) on the
    real-text prompt of length L, run to the full --tpot-osl (EOS ignored)."""
    prompt = _real_prompt(args, L)
    run_env = dict(env, EDGELLM_IGNORE_EOS="1")
    arm_eng = _arm_engine(eng, arm)
    vals = []
    for r in range(args.bench_repeats):
        prof = args.work_dir / "logs" / f"tpot_{eng.name}_{arm}_{L}_{r}.profile.json"
        _run([args.llm_inference, "--engineDir", arm_eng, "--inputFile", prompt,
              "--outputFile", args.work_dir / "logs" / f"tpot_{eng.name}_{arm}_{L}_{r}.out.json",
              "--dumpProfile", "--profileOutputFile", prof],
             args.work_dir / "logs" / f"tpot_{eng.name}_{arm}_{L}_{r}.log", env=run_env)
        gen = json.loads(prof.read_text()).get("generation") or {}
        ms = gen.get("average_time_per_token_ms")
        if ms is None:
            raise RuntimeError(f"llm_inference profile has no generation TPOT for {eng} @ {L}")
        if gen.get("generated_tokens", args.tpot_osl) < args.tpot_osl // 2:
            raise RuntimeError(f"TPOT run generated only {gen.get('generated_tokens')} tokens "
                               f"for {eng} @ {L}; the measurement needs --tpot-osl")
        vals.append(float(ms))
    return statistics.median(vals)


def _ruler(args, eng: Path, L: int, baseline: "Path | None", arm: str = "both") -> float:
    suffix = "" if arm == "both" else f"_{arm}"
    out = args.work_dir / "scores" / f"ruler_{eng.name}{suffix}_{L}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # finished score from an earlier run
        return float(json.loads(out.read_text())["overall"])
    evaluate(argparse.Namespace(
        model_dir=args.model_dir, engine_dir=eng, llm_inference=args.llm_inference,
        max_context=L, per_task=args.ruler_per_task, max_generate_length=96,
        seed=0, mmlu_samples=0, mmlu_num_shot=5, mmlu_dataset_json=None,
        save_results=out, baseline=baseline, max_drop=0.03,
        label=eng.name + suffix, arm=arm))
    return float(json.loads(out.read_text())["overall"])


def _longbench(args, eng: Path, L: int, baseline: "Path | None", env,
               arm: str = "both") -> float:
    suffix = "" if arm == "both" else f"_{arm}"
    out = args.work_dir / "scores" / f"longbench_{eng.name}{suffix}_{L}.json"
    if out.exists():  # finished score from an earlier run
        return float(json.loads(out.read_text())["overall"])
    cmd = [sys.executable, Path(__file__).with_name("longbench_eval.py"),
           "--model-dir", args.model_dir, "--engine-dir", _arm_engine(eng, arm),
           "--llm-inference", args.llm_inference, "--max-context", L,
           "--per-task", args.longbench_per_task, "--save-results", out,
           "--label", eng.name + suffix]
    if baseline:
        cmd += ["--baseline", baseline]
    try:
        _run(cmd, args.work_dir / "logs" / f"longbench_{eng.name}{suffix}_{L}.log",
             env=env)
    except RuntimeError:
        if not out.exists():        # rc=1 also means "FAIL verdict" — keep the score
            raise
    return float(json.loads(out.read_text())["overall"])


def price(args) -> int:
    """Price every tier: TTFT speedup and accuracy drop vs dense per length."""
    args.model_dir = Path(args.model_dir)
    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tiers = args.tiers or config_tiers(cfg)
    if args.work_dir is None:
        args.work_dir = args.model_dir.parent / (args.model_dir.name
                                                 + "-skip-softmax-price")
    lengths = sorted(set(args.lengths))
    max_len = lengths[-1]
    accuracy_lengths = set(args.accuracy_lengths or lengths)
    unknown = accuracy_lengths - set(lengths)
    if unknown:
        raise ValueError(f"--accuracy-lengths {sorted(unknown)} are not in --lengths")
    kernel_profile = args.perf and args.kernel_profile and shutil.which("nsys") is not None
    if args.perf and args.kernel_profile and not kernel_profile:
        print(_c("1;33", "kernel profile skipped — nsys not found on PATH"))
    if args.export_only:
        decode_tiers = cfg.get("skip_softmax_decode_scale_tables") or {}
        for tag, flags in [("dense", ["--skip-softmax-prefill-scale-table", "none",
                                      "--skip-softmax-decode-scale-table", "none"])] + [
                (f"skip{tier_key(t).rstrip('%')}", ["--prefill-target-sparsity", t]
                 + (["--decode-target-sparsity", t] if tier_key(t) in decode_tiers else []))
                for t in tiers]:
            _export(args, tag, flags, os.environ.copy())
        print(_c("1;33", f"exports written under {args.work_dir / 'onnx'}; run `price` "
                 "on the target device with the same --work-dir"))
        return 0
    needed = [args.llm_build, args.llm_inference]
    if args.perf and args.ttft_source == "bench":
        needed.append(args.llm_bench)
    missing = [b for b in needed if not Path(b).exists()]
    if missing:
        print(_c("1;33", "price skipped — binaries not found: "
                 + ", ".join(map(str, missing))
                 + "\n      build the repo (examples/cpp) or pass --llm-build / "
                 "--llm-bench / --llm-inference, then run `price`."))
        return 0
    env = _infer_env(args.llm_inference)
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    lb_cfg = _repo_root() / "_deps" / "longbench-config" / "dataset2prompt.json"
    longbench = args.accuracy and lb_cfg.exists()
    if args.accuracy and not longbench:
        print(_c("1;33", f"LongBench skipped — {lb_cfg} not found (see "
                 "longbench_eval.py for the vendored THUDM/LongBench files)"))

    # Which phases the checkpoint calibrated decides the export flags per tier
    # and the arms worth scoring: a single arm only says something when the
    # other phase exists to switch off.
    prefill_tables = _prefill_key(cfg, "scale_tables") or {}
    decode_tables = cfg.get("skip_softmax_decode_scale_tables") or {}
    has_decode = bool(decode_tables) or bool(cfg.get("skip_softmax_decode_scale_table"))
    has_prefill = bool(prefill_tables) or bool(_prefill_key(cfg, "scale_table")) \
        or _prefill_key(cfg, "target_sparsity") is not None
    arms = list(args.arms or (["both", "prefill", "decode"]
                              if has_decode and has_prefill else ["both"]))
    tpot = args.perf and getattr(args, "tpot", True) and has_decode
    ttft = _ttft_real_ms if args.ttft_source == "real" else _ttft_ms
    program_banner(f"SKIP-SOFTMAX PRICE — {args.model_dir.name}",
                   f"tiers {[tier_key(t) for t in tiers]}   lengths {lengths}   "
                   f"arms {arms}   TTFT {args.ttft_source}   kernel profile "
                   f"{'nsys' if kernel_profile else 'off'}   work dir {args.work_dir}")
    n_stages = 2 + len(tiers)
    stage("price", 1, n_stages, f"dense engine (max input {max_len})")
    dense = _engine(args, "dense", ["--skip-softmax-prefill-scale-table", "none",
                                    "--skip-softmax-decode-scale-table", "none"],
                    max_len, env)
    base: dict = {}
    for L in lengths:
        row: dict = {}
        if args.perf:
            row["ttft_ms"] = ttft(args, dense, L, env)
            if tpot:
                row["tpot_ms"] = _tpot_ms(args, dense, L, env)
            if kernel_profile:
                row["attn_ms"], row["gpu_ms"] = _kernel_profile(args, dense, L, env)
                row["attn_share"] = row["attn_ms"] / row["gpu_ms"] if row["gpu_ms"] else 0.0
        if args.accuracy and L in accuracy_lengths:
            if L in RULER_LENGTHS:
                row["ruler"] = _ruler(args, dense, L, None)
            if longbench:
                row["longbench"] = _longbench(args, dense, L, None, env)
        base[str(L)] = row
        print(f"  dense L {L:<6} " + _fmt_row(row))

    def _accuracy(eng: Path, L: int, arm: str) -> dict:
        row: dict = {}
        if L in RULER_LENGTHS:
            row["ruler"] = _ruler(args, eng, L,
                                  args.work_dir / "scores" / f"ruler_dense_{L}.json", arm)
            row["ruler_drop"] = base[str(L)]["ruler"] - row["ruler"]
        if longbench:
            row["longbench"] = _longbench(
                args, eng, L, args.work_dir / "scores" / f"longbench_dense_{L}.json",
                env, arm)
            row["longbench_drop"] = base[str(L)]["longbench"] - row["longbench"]
        return row

    priced: dict = {}
    for i, tier in enumerate(tiers):
        key = tier_key(tier)
        flags: list = []
        if key in prefill_tables or not has_decode:
            flags += ["--prefill-target-sparsity", tier]
        if key in decode_tables:
            flags += ["--decode-target-sparsity", tier]
        stage("price", 2 + i, n_stages, f"tier {key}: export {' '.join(map(str, flags))} "
              "-> build -> measure")
        eng = _engine(args, f"skip{key.rstrip('%')}", flags, max_len, env)
        priced[key] = {}
        for L in lengths:
            row: dict = {}
            if args.perf:
                # TTFT is a prefill quantity and TPOT a decode one: the `both`
                # arm measures each phase's own gain.
                row["ttft_ms"] = ttft(args, eng, L, env)
                row["speedup"] = base[str(L)]["ttft_ms"] / row["ttft_ms"]
                if tpot:
                    row["tpot_ms"] = _tpot_ms(args, eng, L, env)
                    row["tpot_speedup"] = base[str(L)]["tpot_ms"] / row["tpot_ms"]
                if kernel_profile:
                    row["attn_ms"], row["gpu_ms"] = _kernel_profile(args, eng, L, env)
                    row["kernel_speedup"] = (base[str(L)]["attn_ms"] / row["attn_ms"]
                                             if row["attn_ms"] else 0.0)
            if args.accuracy and L in accuracy_lengths:
                row.update(_accuracy(eng, L, "both"))
                ablation = {arm: _accuracy(eng, L, arm) for arm in arms if arm != "both"}
                if ablation:
                    row["arms"] = ablation
            priced[key][str(L)] = row
            print(f"  {key:<5} L {L:<6} " + _fmt_row(row))
            for arm, sub in (row.get("arms") or {}).items():
                print(f"  {'':<5} L {'':<6} {arm + ' only':<13}" + _fmt_row(sub))

    price_list = {"lengths": lengths, "dense": base, "tiers": priced, "arms": arms,
                  "meta": {"model": args.model_dir.name, "platform": _gpu_name(),
                           "ttft_source": args.ttft_source,
                           "kernel_profile": kernel_profile,
                           "deploy_tier": tier_key(cfg.get("skip_softmax_prefill_target_sparsity",
                                                           DEPLOY_TIER))}}
    result_box(format_price_report(price_list))
    cfg_path = record_price_list(args.model_dir, price_list)
    price_json = args.work_dir / "price.json"
    price_json.write_text(json.dumps(price_list, indent=2) + "\n", encoding="utf-8")
    report = args.report or args.work_dir / "skip_softmax_report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_price_report_html([price_list]), encoding="utf-8")
    print(_c("1;33", f"price list also in {price_json}; HTML report {report}"))
    print(_c("1;33", f"recorded skip_softmax_price_list in {cfg_path}\n"
             f"NEXT: pick the tier whose drop you accept at your lengths; "
             f"{tier_key(_prefill_key(cfg, 'target_sparsity') or DEPLOY_TIER)} "
             "deploys by default, `export --prefill-target-sparsity R` selects another."))
    return 0


def _fmt_row(row: dict) -> str:
    parts = []
    if "ttft_ms" in row:
        parts.append(f"TTFT {row['ttft_ms']:8.1f} ms"
                     + (f"  {row['speedup']:.3f}x" if "speedup" in row else ""))
    if "tpot_ms" in row:
        parts.append(f"TPOT {row['tpot_ms']:7.2f} ms"
                     + (f"  {row['tpot_speedup']:.3f}x" if "tpot_speedup" in row else ""))
    if "kernel_speedup" in row:
        parts.append(f"kernel {row['kernel_speedup']:.3f}x")
    elif "attn_share" in row:
        parts.append(f"attention {100 * row['attn_share']:.1f}% of GPU time")
    if "ruler" in row:
        parts.append(f"RULER {row['ruler']:.3f}"
                     + (f" ({-row['ruler_drop']:+.3f})" if "ruler_drop" in row else ""))
    if "longbench" in row:
        parts.append(f"LongBench {row['longbench']:.3f}"
                     + (f" ({-row['longbench_drop']:+.3f})"
                        if "longbench_drop" in row else ""))
    return "   ".join(parts)


def format_price_report(price_list: dict) -> "list[str]":
    """Tier x length table: speedup and accuracy delta vs dense."""
    lengths = price_list["lengths"]
    has_tpot = any("tpot_speedup" in r for rows in price_list["tiers"].values()
                   for r in rows.values())
    has_kernel = any("kernel_speedup" in r for rows in price_list["tiers"].values()
                     for r in rows.values())
    cw = 26 + (8 if has_tpot else 0) + (8 if has_kernel else 0)
    lines = ["skip-softmax price list (vs dense; e2e TTFT speedup"
             + ("/TPOT speedup" if has_tpot else "")
             + ("; k = attention-kernel speedup" if has_kernel else "")
             + "; accuracy delta in points, RULER @"
             + "/".join(f"{L // 1024}k" for L in RULER_LENGTHS) + ")", "",
             "  tier     " + "".join(f"{L:>{cw}}" for L in lengths)]

    def _acc(r: dict) -> str:
        acc = []
        if "ruler_drop" in r:
            acc.append(f"R{0.0 - r['ruler_drop']:+.3f}")
        if "longbench_drop" in r:
            acc.append(f"LB{0.0 - r['longbench_drop']:+.3f}")
        return " ".join(acc) or "-"

    for key, rows in price_list["tiers"].items():
        cells = []
        arm_cells: dict = {}
        for L in lengths:
            r = rows.get(str(L), {})
            sp = f"{r['speedup']:.2f}x" if "speedup" in r else "  -  "
            if has_tpot:
                sp += f"/{r['tpot_speedup']:.2f}x" if "tpot_speedup" in r else "/  -  "
            if has_kernel:
                sp += f" k{r['kernel_speedup']:.2f}x" if "kernel_speedup" in r else "       "
            cells.append(f"{sp} {_acc(r):>15}")
            for arm, sub in (r.get("arms") or {}).items():
                arm_cells.setdefault(arm, []).append(f"{'':>{len(sp)}} {_acc(sub):>15}")
        lines.append(f"  {key:<8} " + "".join(f"{c:>{cw}}" for c in cells))
        for arm, cs in arm_cells.items():
            lines.append(f"  {arm:<8} " + "".join(f"{c:>{cw}}" for c in cs))
    lines.append("")
    dense = price_list["dense"]
    lines.append("  dense    " + "".join(
        f"{dense.get(str(L), {}).get('ttft_ms', float('nan')):>{cw - 3}.1f} ms"
        for L in lengths))
    if has_tpot:
        lines.append("  tpot     " + "".join(
            f"{dense.get(str(L), {}).get('tpot_ms', float('nan')):>{cw - 3}.2f} ms"
            for L in lengths))
    return lines


# ---------------------------------------------------------------------------
# HTML report: per model one row — attention-kernel speedup (left) and end-to-end
# TTFT speedup (right) against prompt length on a log-2 axis, plus the accuracy table.
# ---------------------------------------------------------------------------
_REPORT_INK, _REPORT_MUTE, _REPORT_GRID = "#0b0b0b", "#8a8984", "#e6e4de"
_REPORT_COLORS = ("#0f7b32", "#1f5fbf", "#a05a2c", "#7a3e9d", "#c0392b")


def _svg_text(x, y, s, size=10, fill=_REPORT_INK, anchor="start", bold=False):
    import html
    weight = ' font-weight="600"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-family="Segoe UI,Arial"{weight}>'
            f'{html.escape(str(s))}</text>')


def _svg_panel(px, py, pw, ph, lengths, ylo, yhi, yticks, series):
    """Framed panel: log2-x over ``lengths``, y in [ylo, yhi], one polyline per
    (label, color, [(L, value)]) series; the 1.0x line is dashed."""
    lo, hi = math.log2(lengths[0]), math.log2(lengths[-1])

    def x_of(L):
        return px + (math.log2(L) - lo) / (hi - lo) * pw if hi > lo else px + pw / 2

    def y_of(v):
        return py + (yhi - max(ylo, min(yhi, v))) / (yhi - ylo) * ph

    out = (f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" '
           f'stroke="#c9c7c1" stroke-width="1"/>')
    for tick in yticks:
        yy = y_of(tick)
        out += (f'<line x1="{px}" y1="{yy:.1f}" x2="{px + pw}" y2="{yy:.1f}" '
                f'stroke="{_REPORT_GRID}" stroke-width="1"/>'
                + _svg_text(px - 5, yy + 3, f"{tick:.2f}", 8, _REPORT_MUTE, "end"))
    y1 = y_of(1.0)
    out += (f'<line x1="{px}" y1="{y1:.1f}" x2="{px + pw}" y2="{y1:.1f}" '
            f'stroke="{_REPORT_MUTE}" stroke-width="1.2" stroke-dasharray="5 3"/>')
    shown = lengths if len(lengths) <= 8 else [
        L for L in lengths if L & (L - 1) == 0 or L in (lengths[-1], 24576)]
    for L in shown:
        out += _svg_text(x_of(L), py + ph + 11, f"{L // 1024}k", 7.6, _REPORT_MUTE, "middle")
    for label, color, points in series:
        pts = [(x_of(L), y_of(v)) for L, v in points if v is not None]
        if not pts:
            continue
        out += (f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                f'fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        out += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}"/>'
                       for x, y in pts)
    return out


def _y_range(values, floor=0.9, ceil=1.2):
    vals = [v for v in values if v is not None]
    hi = max([ceil] + [v + 0.05 for v in vals])
    lo = min([floor] + [v - 0.05 for v in vals])
    step = 0.1 if hi - lo <= 0.8 else 0.2
    ticks = [round(t, 2) for t in
             [lo + i * step for i in range(int((hi - lo) / step) + 2)] if t <= hi + 1e-9]
    return lo, hi, ticks


def merge_price_lists(price_lists: "list[dict]") -> "list[dict]":
    """Fold runs of the same (model, platform) into one entry: lengths are unioned,
    tiers are unioned, and within a repeated tier (or dense) each length's row is
    merged field by field, the first run winning on a repeated field — so a
    `--no-perf` accuracy run and a `--no-accuracy` perf run of the same tier become
    one full row. Runs of different models or platforms stay separate rows."""
    merged: "dict[tuple, dict]" = {}

    def fill(rows: dict, extra: dict) -> None:
        for L, row in extra.items():
            for k, v in row.items():
                rows.setdefault(L, {}).setdefault(k, v)

    for pl in price_lists:
        meta = pl.get("meta", {})
        key = (meta.get("model"), meta.get("platform"))
        if key not in merged:
            merged[key] = {**pl, "lengths": list(pl["lengths"]), "tiers": {}, "dense": {}}
        target = merged[key]
        target["lengths"] = sorted(set(target["lengths"]) | set(pl["lengths"]))
        fill(target["dense"], pl["dense"])
        for tier, rows in pl["tiers"].items():
            fill(target["tiers"].setdefault(tier, {}), rows)
    return list(merged.values())


def render_price_report_svg(price_lists: "list[dict]") -> str:
    """One row per priced (model, platform): kernel speedup (left), TTFT speedup
    (right), one line per tier; the deploy tier's peak is annotated."""
    row_h, top = 164, 46
    left_x, left_w, right_x, right_w, legend_x = 118, 400, 600, 450, 1066
    body = _svg_text(14, 16, "skip-softmax: attention-kernel speedup (left) and "
                     "end-to-end TTFT speedup (right) vs prompt length", 12.5, bold=True)
    body += _svg_text(left_x + left_w / 2, 34, "attention-kernel speedup (nsys)", 10,
                      anchor="middle", bold=True)
    body += _svg_text(right_x + right_w / 2, 34, "end-to-end TTFT speedup", 10,
                      anchor="middle", bold=True)
    price_lists = merge_price_lists(price_lists)
    for ri, pl in enumerate(price_lists):
        lengths = [int(L) for L in pl["lengths"]]
        meta = pl.get("meta", {})
        ry = top + ri * row_h
        body += _svg_text(14, ry + 20, meta.get("model", f"model {ri + 1}"), 11.5, bold=True)
        body += _svg_text(14, ry + 34, meta.get("platform", ""), 8.6, _REPORT_MUTE)
        kernel_series, e2e_series = [], []
        for ti, (tier, rows) in enumerate(pl["tiers"].items()):
            color = _REPORT_COLORS[ti % len(_REPORT_COLORS)]
            kernel_series.append((tier, color, [
                (L, rows.get(str(L), {}).get("kernel_speedup")) for L in lengths]))
            e2e_series.append((tier, color, [
                (L, rows.get(str(L), {}).get("speedup")) for L in lengths]))
            ly = ry + 14 + 14 * ti
            body += (f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" '
                     f'stroke="{color}" stroke-width="2.2"/>'
                     + _svg_text(legend_x + 22, ly + 3, tier, 8.6, color, bold=True))
        panel_h = row_h - 56
        k_vals = [v for _, _, pts in kernel_series for _, v in pts]
        if any(v is not None for v in k_vals):
            lo, hi, ticks = _y_range(k_vals, 0.85, 1.5)
            body += _svg_panel(left_x, ry + 6, left_w, panel_h, lengths, lo, hi, ticks,
                               kernel_series)
        else:
            body += _svg_text(left_x + left_w / 2, ry + panel_h / 2 + 8,
                              "no nsys kernel profile in this run", 9, _REPORT_MUTE, "middle")
        e_vals = [v for _, _, pts in e2e_series for _, v in pts]
        lo, hi, ticks = _y_range(e_vals, 0.93, 1.2)
        body += _svg_panel(right_x, ry + 6, right_w, panel_h, lengths, lo, hi, ticks,
                           e2e_series)
        deploy = meta.get("deploy_tier")
        if deploy in pl["tiers"]:
            rows = pl["tiers"][deploy]
            peaks = [(rows[str(L)].get("kernel_speedup"), rows[str(L)].get("speedup"), L)
                     for L in lengths if str(L) in rows]
            best_e2e = max(((e, L) for _, e, L in peaks if e is not None), default=None)
            best_k = max(((k, L) for k, _, L in peaks if k is not None), default=None)
            if best_k:
                body += _svg_text(left_x + left_w, ry + 2, f"{deploy} peak {best_k[0]:.3f}x "
                                  f"@ {best_k[1] // 1024}k", 8.6, _REPORT_COLORS[0], "end", True)
            if best_e2e:
                body += _svg_text(right_x + right_w, ry + 2, f"{deploy} peak {best_e2e[0]:.3f}x "
                                  f"@ {best_e2e[1] // 1024}k", 8.6, _REPORT_COLORS[0], "end", True)
        shares = [pl["dense"].get(str(L), {}).get("attn_share") for L in lengths]
        if any(v is not None for v in shares):
            first = next(v for v in shares if v is not None)
            last = next(v for v in reversed(shares) if v is not None)
            body += _svg_text(left_x, ry + panel_h + 30, f"attention share of GPU time "
                              f"{100 * first:.0f}% -> {100 * last:.0f}% (dense)", 8.2, "#52514e")
    h = top + len(price_lists) * row_h + 14
    body += _svg_text(14, h - 4, "TTFT: real text, one generated token; kernel: nsys sum of "
                      "the attention kernels of the same run; lengths at or below the "
                      "table's first skippable length run the dense kernel by design.",
                      8.4, _REPORT_MUTE)
    return (f'<svg width="1170" height="{h}" viewBox="0 0 1170 {h}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def render_price_report_html(price_lists: "list[dict]") -> str:
    """Self-contained light-theme HTML: the figure plus one accuracy/TTFT table per model."""
    import html
    parts = ["<!doctype html><html><head><meta charset='utf-8'>"
             "<title>skip-softmax price report</title><style>"
             "body{font-family:Segoe UI,Arial;color:#0b0b0b;background:#fff;margin:24px}"
             "table{border-collapse:collapse;font-size:12.5px;margin:8px 0 24px}"
             "th,td{border:1px solid #d8d6d0;padding:3px 9px;text-align:right}"
             "th:first-child,td:first-child{text-align:left}"
             "h2{font-size:15px;margin:22px 0 4px}</style></head><body>",
             render_price_report_svg(price_lists)]
    for pl in merge_price_lists(price_lists):
        meta = pl.get("meta", {})
        lengths = [int(L) for L in pl["lengths"]]
        parts.append(f"<h2>{html.escape(meta.get('model', 'model'))} — "
                     f"{html.escape(meta.get('platform', ''))}</h2>")
        parts.append("<table><tr><th>tier</th><th>metric</th>"
                     + "".join(f"<th>{L // 1024}k</th>" for L in lengths) + "</tr>")
        dense = pl["dense"]
        parts.append("<tr><td>dense</td><td>TTFT ms</td>" + "".join(
            f"<td>{dense.get(str(L), {}).get('ttft_ms', float('nan')):.0f}</td>"
            for L in lengths) + "</tr>")
        if any("attn_share" in dense.get(str(L), {}) for L in lengths):
            parts.append("<tr><td>dense</td><td>attention share</td>" + "".join(
                f"<td>{100 * dense[str(L)]['attn_share']:.1f}%</td>"
                if "attn_share" in dense.get(str(L), {}) else "<td>-</td>"
                for L in lengths) + "</tr>")
        for metric, key, fmt in (("TTFT speedup", "speedup", "{:.3f}x"),
                                 ("kernel speedup", "kernel_speedup", "{:.3f}x"),
                                 ("RULER drop", "ruler_drop", "{:+.3f}"),
                                 ("LongBench drop", "longbench_drop", "{:+.3f}")):
            for tier, rows in pl["tiers"].items():
                if not any(key in rows.get(str(L), {}) for L in lengths):
                    continue
                cells = []
                for L in lengths:
                    r = rows.get(str(L), {})
                    v = r.get(key)
                    if v is None:
                        cells.append("<td>-</td>")
                    elif key.endswith("_drop"):
                        cells.append(f"<td>{fmt.format(-v)}</td>")
                    else:
                        cells.append(f"<td>{fmt.format(v)}</td>")
                parts.append(f"<tr><td>{html.escape(tier)}</td><td>{metric}</td>"
                             + "".join(cells) + "</tr>")
        parts.append("</table>")
    parts.append("<p style='color:#8a8984;font-size:11px'>accuracy drops are dense minus "
                 "tier, in score points (negative = the tier scored higher); RULER only at "
                 f"{'/'.join(str(L) for L in RULER_LENGTHS)} tokens.</p></body></html>")
    return "".join(parts)


def report(args) -> int:
    """Render the HTML report from price.json files or priced checkpoints."""
    price_lists = []
    for path in args.price_json or []:
        price_lists.append(json.loads(Path(path).read_text(encoding="utf-8")))
    for model_dir in args.model_dir or []:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        if "skip_softmax_price_list" not in cfg:
            raise ValueError(f"{model_dir}/config.json carries no skip_softmax_price_list; "
                             "run `price` first")
        pl = cfg["skip_softmax_price_list"]
        pl.setdefault("meta", {})["model"] = pl.get("meta", {}).get("model",
                                                                    Path(model_dir).name)
        price_lists.append(pl)
    if not price_lists:
        raise ValueError("report needs --price-json and/or --model-dir")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_price_report_html(price_lists), encoding="utf-8")
    print(f"report written to {args.out}")
    return 0


def record_price_list(model_dir: Path, price_list: dict) -> Path:
    """Merge the price list into config.json; every other key is kept."""
    cfg_path = Path(model_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["skip_softmax_price_list"] = price_list
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg_path


if __name__ == "__main__":
    sys.exit(main())
