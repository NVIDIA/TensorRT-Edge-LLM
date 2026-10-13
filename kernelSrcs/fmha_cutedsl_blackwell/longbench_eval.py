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

"""LongBench V1 acceptance evaluation on a deployed TensorRT Edge-LLM engine.

Held-out realistic-task acceptance baseline for skip-softmax deployments —
aligned with TensorRT-LLM's `trtllm-eval longbench_v1` workflow (tech blog 16).
RULER stays the calibration set; LongBench is deliberately NOT the calibration
distribution, so a pass here is free of train/test contamination.

Fidelity: uses the OFFICIAL LongBench prompt templates, per-task generation
lengths, and metrics (vendored from THUDM/LongBench under
_deps/longbench-config/). English V1 subset with dependency-light metrics:

  qa_f1:          narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa
  rouge:          gov_report qmsum multi_news samsum   (pip `rouge`)
  classification: trec
  retrieval:      passage_retrieval_en

Prompts longer than the engine input limit are middle-truncated (head+tail
halves), matching the official LongBench pred.py strategy. Chat template is
NOT applied (official LongBench evaluates base-style completion for most
models; keep parity with trtllm-eval defaults).

Usage:
  python longbench_eval.py --model-dir <hf> --engine-dir <engine> \
      --llm-inference <bin> --max-context 16384 --per-task 30 \
      [--save-results dense_lb.json | --baseline dense_lb.json --max-drop 0.03] \
      [--label "S=18337"]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

_CFG_DIR = Path(__file__).resolve().parents[2] / "_deps" / "longbench-config"
_HF_REPO = "THUDM/LongBench"

QA_F1 = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "triviaqa"]
ROUGE = ["gov_report", "qmsum", "multi_news", "samsum"]
CLS = ["trec"]
RETR = ["passage_retrieval_en"]
ALL_TASKS = QA_F1 + ROUGE + CLS + RETR


# --------------------------------------------------------------------------
# Official LongBench metrics (faithful ports of THUDM/LongBench metrics.py)
# --------------------------------------------------------------------------
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction: str, ground_truth: str) -> float:
    from rouge import Rouge
    if not prediction.strip():
        return 0.0
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except ValueError:
        return 0.0


def classification_score(prediction: str, ground_truth: str,
                         all_classes: list) -> float:
    # official: prediction counts if the gold class appears in it and no
    # longer class containing the gold as substring also appears
    em_match_list = [c for c in (all_classes or []) if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    return float(ground_truth in em_match_list)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    pattern = r"Paragraph (\d+)"
    ground = re.search(pattern, ground_truth)
    numbers = re.findall(r"\d+", prediction)
    return float(bool(ground) and ground.group(1) in numbers)


def score_of(task: str, pred: str, answers: list, all_classes: list) -> float:
    # official post-processing: first line only for some tasks
    if task in ("trec", "triviaqa", "samsum"):
        pred = pred.lstrip("\n").split("\n")[0]
    best = 0.0
    for gt in answers:
        if task in QA_F1:
            best = max(best, qa_f1_score(pred, gt))
        elif task in ROUGE:
            best = max(best, rouge_score(pred, gt))
        elif task in CLS:
            best = max(best, classification_score(pred, gt, all_classes))
        elif task in RETR:
            best = max(best, retrieval_score(pred, gt))
    return best


# --------------------------------------------------------------------------
def enforce_byte_cap(prompt: str, max_bytes: int = 126976) -> str:
    """llm_inference rejects messages > 128KiB (cpp/common/inputLimits.h).
    Token-based truncation can still exceed it on byte-heavy tasks
    (gov_report ~5.0 B/token at 28k ctx) — middle-trim by characters as a
    final guard. Applied identically to baseline and candidate runs."""
    raw = prompt.encode("utf-8")
    if len(raw) <= max_bytes:
        return prompt
    # 按字符中段裁,留 2% 余量防多字节字符边界膨胀
    keep = int(len(prompt) * (max_bytes / len(raw)) * 0.98)
    half = keep // 2
    return prompt[:half] + " ... " + prompt[-half:]


def truncate_middle(prompt: str, tokenizer, max_tok: int) -> str:
    """Official LongBench strategy: keep head+tail halves, drop the middle."""
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) <= max_tok:
        return prompt
    half = max_tok // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))


# --------------------------------------------------------------------------
# LongBench-v2 (THUDM/LongBench-v2): single multiple-choice set, 503 samples.
# Official 0-shot template (prompts/0shot.txt) + official extract_answer regex;
# metric = accuracy, reported overall and by difficulty/length buckets.
# --------------------------------------------------------------------------
_V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".')


def v2_extract_answer(response: str):
    response = response.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", response)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", response)
    return m.group(1) if m else None


def run_v2(args, tokenizer, role) -> int:
    import random
    data_p = _CFG_DIR.parent / "longbench-v2" / "data.json"
    rows = json.load(open(data_p, encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    n = min(args.per_task, len(rows)) if args.per_task > 0 else len(rows)
    rows = rows[:n]
    max_gen = 128
    max_tok = args.max_context - max_gen - 64
    print(f"== LongBench V2 [{role}] engine={args.engine_dir} "
          f"ctx={args.max_context} n={n}")
    requests = []
    for row in rows:
        prompt = _V2_TEMPLATE.format(context=row["context"], question=row["question"],
                                     choice_A=row["choice_A"], choice_B=row["choice_B"],
                                     choice_C=row["choice_C"], choice_D=row["choice_D"])
        requests.append({"messages": [{"role": "user",
                                       "content": enforce_byte_cap(truncate_middle(prompt, tokenizer, max_tok))}]})
    with tempfile.TemporaryDirectory() as tmp:
        in_p, out_p = Path(tmp) / "in.json", Path(tmp) / "out.json"
        json.dump({"temperature": 1.0, "top_p": 1.0, "top_k": 1,
                   "max_generate_length": max_gen,
                   "apply_chat_template": False,
                   "requests": requests}, open(in_p, "w"))
        env = dict(os.environ)
        if not env.get("EDGELLM_PLUGIN_PATH"):
            _plug = Path(args.llm_inference).resolve().parents[2] \
                / "libNvInfer_edgellm_plugin.so"
            if _plug.exists():
                env["EDGELLM_PLUGIN_PATH"] = str(_plug)
        proc = subprocess.run(
            [str(args.llm_inference), "--engineDir", str(args.engine_dir),
             "--inputFile", str(in_p), "--outputFile", str(out_p)],
            capture_output=True, text=True, env=env)
        if not out_p.exists():
            print(proc.stdout[-1500:], proc.stderr[-800:], sep="\n")
            print("ERROR: no output for LongBench-v2 batch")
            return 1
        responses = sorted(json.load(open(out_p))["responses"],
                           key=lambda r: r["request_idx"])
    hits, buckets = [], defaultdict(list)
    for row, resp in zip(rows, responses):
        ok = float(v2_extract_answer(resp.get("output_text", "")) == row["answer"])
        hits.append(ok)
        buckets["difficulty=" + row["difficulty"]].append(ok)
        buckets["length=" + row["length"]].append(ok)
    overall = sum(hits) / max(len(hits), 1)
    for k in sorted(buckets):
        v = buckets[k]
        print(f"  {k:20s} n={len(v):3d}  acc={sum(v)/len(v):.4f}")
    print(f"OVERALL[{role}]: {overall:.4f} (accuracy, n={len(hits)})")
    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"overall": overall, "n": len(hits), "ctx": args.max_context,
                   "suite": "v2",
                   "buckets": {k: sum(v)/len(v) for k, v in buckets.items()}},
                  open(args.save_results, "w"), indent=1)
        print(f"saved -> {args.save_results}")
    if args.baseline:
        base = json.load(open(args.baseline))
        drop = base["overall"] - overall
        verdict = drop <= args.max_drop
        print(f"LONGBENCH VERDICT: {'PASS' if verdict else 'FAIL'} [{role}]  "
              f"{base['overall']:.4f} -> {overall:.4f}  drop {drop:+.4f} "
              f"(gate {args.max_drop})")
        return 0 if verdict else 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--llm-inference", required=True)
    ap.add_argument("--suite", choices=["v1", "v2"], default="v1",
                    help="v1: 13-task macro-avg suite; v2: LongBench-v2 "
                    "multiple-choice (THUDM/LongBench-v2 data.json, "
                    "official 0-shot template, accuracy metric)")
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--per-task", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                    help=f"subset of: {' '.join(ALL_TASKS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--save-results", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--max-drop", type=float, default=0.03)
    args = ap.parse_args()

    import random

    from transformers import AutoTokenizer

    def load_task(task: str) -> list:
        """Official data.zip jsonl (datasets>=5 dropped LongBench's hub script)."""
        p = _CFG_DIR.parent / "longbench-data" / "data" / f"{task}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        random.Random(args.seed).shuffle(rows)
        return rows

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    role = args.label or ("baseline" if args.save_results else "candidate")

    if args.suite == "v2":
        return run_v2(args, tokenizer, role)

    d2p = json.load(open(_CFG_DIR / "dataset2prompt.json"))
    d2m = json.load(open(_CFG_DIR / "dataset2maxlen.json"))
    print(f"== LongBench V1 [{role}] engine={args.engine_dir} "
          f"ctx={args.max_context} per_task={args.per_task} tasks={len(args.tasks)}")

    # per-task batches: generation length differs per task -> one inference
    # invocation per task (llm_inference takes a single max_generate_length)
    per_task_scores: dict = {}
    for task in args.tasks:
        ds = load_task(task)
        n = min(args.per_task, len(ds))
        max_gen = int(d2m[task])
        max_tok = args.max_context - max_gen - 64
        requests, meta = [], []
        for row in ds[:n]:
            prompt = d2p[task].format(context=row["context"],
                                      input=row.get("input", ""))
            prompt = enforce_byte_cap(truncate_middle(prompt, tokenizer, max_tok))
            requests.append({"messages": [{"role": "user", "content": prompt}]})
            meta.append({"answers": row["answers"],
                         "all_classes": row.get("all_classes") or []})
        with tempfile.TemporaryDirectory() as tmp:
            in_p, out_p = Path(tmp) / "in.json", Path(tmp) / "out.json"
            json.dump({"temperature": 1.0, "top_p": 1.0, "top_k": 1,  # greedy
                       "max_generate_length": max_gen,
                       "apply_chat_template": False,
                       "requests": requests}, open(in_p, "w"))
            env = dict(os.environ)
            if not env.get("EDGELLM_PLUGIN_PATH"):
                # Pin the plugin to the binary's own build tree — the relative
                # default resolves to MAIN's plugin (no d256 skip → silent
                # dense → vacuous verdicts; 2026-08-03 incident).
                _plug = Path(args.llm_inference).resolve().parents[2] \
                    / "libNvInfer_edgellm_plugin.so"
                if _plug.exists():
                    env["EDGELLM_PLUGIN_PATH"] = str(_plug)
            proc = subprocess.run(
                [str(args.llm_inference), "--engineDir", str(args.engine_dir),
                 "--inputFile", str(in_p), "--outputFile", str(out_p)],
                capture_output=True, text=True, env=env)
            if not out_p.exists():
                print(proc.stdout[-1500:], proc.stderr[-800:], sep="\n")
                print(f"ERROR: no output for task {task}")
                return 1
            responses = sorted(json.load(open(out_p))["responses"],
                               key=lambda r: r["request_idx"])
        scores = [score_of(task, r.get("output_text", ""), m["answers"],
                           m["all_classes"])
                  for r, m in zip(responses, meta)]
        per_task_scores[task] = sum(scores) / max(len(scores), 1)
        print(f"  {task:22s} n={len(scores):3d}  score={per_task_scores[task]:.4f}")

    overall = sum(per_task_scores.values()) / len(per_task_scores)
    print(f"OVERALL[{role}]: {overall:.4f} (macro avg over {len(per_task_scores)} tasks)")

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"overall": overall, "per_task": per_task_scores,
                   "per_task_n": args.per_task, "ctx": args.max_context},
                  open(args.save_results, "w"), indent=1)
        print(f"saved -> {args.save_results}")
    if args.baseline:
        base = json.load(open(args.baseline))
        drop = base["overall"] - overall
        verdict = drop <= args.max_drop
        print(f"LONGBENCH VERDICT: {'PASS' if verdict else 'FAIL'} [{role}]  "
              f"{base['overall']:.4f} -> {overall:.4f}  drop {drop:+.4f} "
              f"(gate {args.max_drop})")
        return 0 if verdict else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
