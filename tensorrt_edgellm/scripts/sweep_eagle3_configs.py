#!/usr/bin/env python3
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
"""Benchmark Eagle3 speculative-decoding hyperparameters.

Focus
-----
This script does one thing: measure TPS and accept length across a grid of
``(topK, draftStep, verifyTreeSize)`` combinations against a pre-built Eagle3
engine + a JSONL dataset, then rank the configs best-first.

Caller-supplied environment (not bootstrapped by this script)
  LD_LIBRARY_PATH must include libcudart and libnvinfer
  TRT_PACKAGE_DIR  (optional) used by the plugin loader
  EDGELLM_PLUGIN_PATH (optional) overrides plugin .so discovery

Sweep grid (defaults; overridable via --top_ks / --draft_steps)
  topK        = 6, 8, 10, 12
  draftStep   = 4, 6, 8
  verifyTreeSize (vts) per (topK, draftStep) pair:
    - ds × tk           (full product)
    - ⌊0.67 × ds × tk⌋  (67 % of full product)
    - max_ts            (engine's maxVerifyTreeSize from base_config.json)
  Duplicates are removed; values > max_ts are clamped to max_ts.

Reported metrics (per run)
  tps              output tokens / elapsed wall-clock seconds
  accept_length    avg accepted draft tokens per step  (bonus token excluded).
                   With --batch_size > 1 this is a per-batch aggregate, not
                   a per-request value, because the runtime metrics counter
                   is global.
  accept_rate_%    accept_length as % of draftStep

Usage
-----
  python3 sweep_eagle3_configs.py  \\
      --engine_dir /path/to/engines_dir   \\
      --dataset   /path/to/dataset.jsonl  \\
      --batch_size N                      \\
      [--subset_entries N]                \\
      [--top_ks 6,8,10,12]                \\
      [--draft_steps 4,6,8]               \\
      [--output_dir /path/to/out]         # optional; writes results_table.txt + results_plot.png

Dataset JSONL schema (one record per line)
  {"prompt": <str>, "output_tokens": <int>}
  ("num_output_tokens" is also accepted; defaults to 128 if neither is present.)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# Default sweep parameter grid (overridable via --top_ks / --draft_steps CLI flags)
_DEFAULT_TOP_KS = [6, 8, 10, 12]
_DEFAULT_DRAFT_STEPS = [4, 6, 8]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_dataset(path: str, subset_entries: Optional[int]) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: malformed JSON: {exc}") from exc
            if "prompt" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: record is missing required 'prompt' field"
                )
            rows.append({
                "prompt":
                obj["prompt"],
                "output_tokens":
                int(obj.get("output_tokens", obj.get("num_output_tokens",
                                                     128))),
            })
    if not rows:
        raise ValueError(f"{path}: dataset is empty (no non-blank lines)")
    if subset_entries is not None and subset_entries >= len(rows):
        subset_entries = None  # use all
    if subset_entries is not None and subset_entries > 0:
        rows = rows[:subset_entries]
    return rows


# ---------------------------------------------------------------------------
# Engine config reader
# ---------------------------------------------------------------------------


def _read_engine_limits(engines_dir: Path) -> tuple[int, int]:
    """Return (max_verify_tree_size, max_batch_size) from base_config.json."""
    cfg_path = engines_dir / "base_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"base_config.json not found in {engines_dir}")
    cfg = json.loads(cfg_path.read_text())
    bc = cfg.get("builder_config", {})
    ts = bc.get("max_verify_tree_size")
    if ts is None:
        raise KeyError(f"max_verify_tree_size not found in {cfg_path}")
    bs = bc.get("max_batch_size")
    if bs is None:
        raise KeyError(f"max_batch_size not found in {cfg_path}")
    return int(ts), int(bs)


# ---------------------------------------------------------------------------
# Combo generation
# ---------------------------------------------------------------------------


def _generate_combos(max_ts: int, top_ks: list[int],
                     draft_steps: list[int]) -> list[tuple[int, int, int]]:
    """Return list of (topK, draftStep, vts) tuples, deduplicated and clamped."""
    seen: set[tuple[int, int, int]] = set()
    combos: list[tuple[int, int, int]] = []
    for tk in top_ks:
        for ds in draft_steps:
            full = ds * tk
            partial = max(1, int(ds * tk * 0.67))
            candidates = sorted({
                min(full, max_ts),
                min(partial, max_ts),
                max_ts,
            })
            for vts in candidates:
                key = (tk, ds, vts)
                if key not in seen:
                    seen.add(key)
                    combos.append(key)
    return combos


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def _run_one(
    rt,
    engines_dir: Path,
    requests: list[dict],
    topk: int,
    draft_step: int,
    vts: int,
    batch_size: int,
) -> dict:
    """
    Create a fresh LLMRuntime for (topk, draft_step, vts), run all requests
    in batches of batch_size, collect TPS and acceptance length.

    Accept length is computed as (delta_generated / delta_iters) - 1
    to exclude the bonus token. The delta is measured across each batch, so
    with batch_size > 1 the reported accept_length is a per-batch aggregate,
    not a per-request value.
    """
    runtime = rt.LLMRuntime(str(engines_dir), "", {}, topk, draft_step, vts)
    # Mirror production path: capture decoding CUDA graphs so the sweep
    # measures the same kernel-launch overhead profile users see in practice.
    # Without this, all configs are penalized equally by graph-less launches
    # and rankings may not match production behavior.
    runtime.capture_decoding_cuda_graph()

    snap0 = runtime.get_spec_decode_generation_metrics()
    prev_gen = float(getattr(snap0, "total_generated_tokens", 0))
    prev_iters = float(getattr(snap0, "total_iterations", 0))

    output_token_counts: list[int] = []
    accept_lens: list[float] = []

    # Sort by target output_tokens so each batch contains requests with similar
    # lengths. The pybind API takes a single max_generate_length per batch, so
    # heterogeneous batches would let shorter-targeted requests run up to the
    # batch max — inflating total_out (and TPS) with unintended tokens.
    ordered_requests = sorted(requests, key=lambda r: r["output_tokens"])

    t0 = time.perf_counter()
    for batch_start in range(0, len(ordered_requests), batch_size):
        batch = ordered_requests[batch_start:batch_start + batch_size]
        all_msgs = [[rt.create_text_message("user", r["prompt"])]
                    for r in batch]
        max_out = max(r["output_tokens"] for r in batch)
        greq = rt.create_generation_request(
            all_msgs,
            temperature=1.0,
            top_p=1.0,
            top_k=1,  # greedy — reproducible across runs
            max_generate_length=max_out,
            apply_chat_template=True,
            add_generation_prompt=True,
            enable_thinking=False,
            lora_weights_name="",
            save_system_prompt_kv_cache=False,
            disable_spec_decode=False,
        )
        resp = runtime.handle_request(greq)

        # Accept length is derived from the runtime's global counter delta
        # across the batch, so with batch_size > 1 it is an aggregate over
        # all requests in the batch, not per-request. `accept_lens` therefore
        # holds one entry per batch, and downstream mean() weights each
        # batch equally regardless of request count — acceptable because
        # sorting above keeps batch composition consistent across combos.
        snap = runtime.get_spec_decode_generation_metrics()
        delta_gen = float(getattr(snap, "total_generated_tokens",
                                  0)) - prev_gen
        delta_iters = max(
            float(getattr(snap, "total_iterations", 0)) - prev_iters, 1.0)
        # Subtract 1 to remove the bonus token from accept length
        al = max((delta_gen / delta_iters) - 1.0, 0.0)
        accept_lens.append(al)
        prev_gen = float(getattr(snap, "total_generated_tokens", 0))
        prev_iters = float(getattr(snap, "total_iterations", 0))

        for k in range(len(batch)):
            out_ids = resp.output_ids[k] if resp.output_ids and k < len(
                resp.output_ids) else []
            output_token_counts.append(len(out_ids))

    elapsed = max(time.perf_counter() - t0, 1e-9)
    total_out = sum(output_token_counts)
    tps = total_out / elapsed
    avg_al = statistics.mean(accept_lens) if accept_lens else 0.0
    accept_rate = 100.0 * avg_al / max(draft_step, 1)

    del runtime  # release GPU memory (engines + KV cache) before next combo

    return {
        "topk": topk,
        "draft_step": draft_step,
        "vts": vts,
        "batch_size": batch_size,
        "tps": tps,
        "accept_length": avg_al,
        "accept_rate_%": accept_rate,
        "total_output_tokens": total_out,
        "elapsed_s": elapsed,
        "num_requests": len(requests),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_HEADER = (f"{'Rank':>4}  {'topK':>4}  {'draftStep':>9}  {'vts':>5}  "
           f"{'TPS':>10}  {'acceptLen':>13}  {'acceptRate%':>13}")
_SEP = "-" * len(_HEADER)


def _row_str(rank: int, r: dict, best_tps: float) -> str:
    marker = " *" if abs(r["tps"] - best_tps) < 1e-9 else "  "
    return (f"{rank:>4}{marker}"
            f"  {r['topk']:>4}"
            f"  {r['draft_step']:>9}"
            f"  {r['vts']:>5}"
            f"  {r['tps']:>10.2f}"
            f"  {r['accept_length']:>13.3f}"
            f"  {r['accept_rate_%']:>12.1f}%")


def _print_table(results: list[dict]) -> None:
    best_tps = results[0]["tps"] if results else 0.0
    print()
    print("Eagle3 Config Sweep Results  (sorted by TPS, best first)  * = best")
    print(_SEP)
    print(_HEADER)
    print(_SEP)
    for i, r in enumerate(results, 1):
        print(_row_str(i, r, best_tps))
    print(_SEP)
    if results:
        b = results[0]
        print(
            f"\nBest: topK={b['topk']}  draftStep={b['draft_step']}  vts={b['vts']}"
            f"  →  TPS={b['tps']:.2f}  acceptLen={b['accept_length']:.3f}")
    print()


def _save_table(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_tps = results[0]["tps"] if results else 0.0
    lines = [
        "Eagle3 Config Sweep Results  (sorted by TPS, best first)  * = best",
        _SEP,
        _HEADER,
        _SEP,
    ]
    for i, r in enumerate(results, 1):
        lines.append(_row_str(i, r, best_tps))
    lines.append(_SEP)
    if results:
        b = results[0]
        lines.append(
            f"\nBest: topK={b['topk']}  draftStep={b['draft_step']}  vts={b['vts']}"
            f"  →  TPS={b['tps']:.2f}  acceptLen={b['accept_length']:.3f}")
    out_path = output_dir / "results_table.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[output] Table saved → {out_path}")


def _save_plot(results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[output] matplotlib not available — skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    draft_steps = sorted({r["draft_step"] for r in results})
    topks = sorted({r["topk"] for r in results})

    # Color = draftStep, marker shape = topK. Use tab10 + a marker pool large
    # enough to handle arbitrary --top_ks / --draft_steps lengths; cycle if
    # the user passes more values than the pool size.
    _tab10 = plt.get_cmap("tab10")
    step_colors = {ds: _tab10(i % 10) for i, ds in enumerate(draft_steps)}
    _marker_pool = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
    topk_markers = {
        tk: _marker_pool[i % len(_marker_pool)]
        for i, tk in enumerate(topks)
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    best_tps = results[0]["tps"] if results else 0.0

    for r in results:
        is_best = abs(r["tps"] - best_tps) < 1e-9
        ax.scatter(
            r["accept_length"],
            r["tps"],
            color=step_colors[r["draft_step"]],
            marker=topk_markers[r["topk"]],
            s=120 if not is_best else 260,
            zorder=3 if not is_best else 5,
            edgecolors="gold" if is_best else "none",
            linewidths=2.5 if is_best else 0,
        )
        ax.annotate(
            f"vts={r['vts']}",
            xy=(r["accept_length"], r["tps"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#444444",
        )

    # Legend: draftStep (color)
    for ds in draft_steps:
        ax.scatter([], [],
                   color=step_colors[ds],
                   label=f"draftStep={ds}",
                   s=80)
    # Legend: topK (marker)
    for tk in topks:
        ax.scatter([], [],
                   marker=topk_markers[tk],
                   color="gray",
                   label=f"topK={tk}",
                   s=80)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_xlabel("Accept Length (excl. bonus token)", fontsize=11)
    ax.set_ylabel("TPS (output tokens / s)", fontsize=11)
    ax.set_title(
        "Eagle3 Config Sweep: TPS vs Accept Length\n(★ = best config)",
        fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Mark best point explicitly
    if results:
        b = results[0]
        ax.annotate(
            f"★ BEST\ntopK={b['topk']}, ds={b['draft_step']}, vts={b['vts']}\nTPS={b['tps']:.1f}",
            xy=(b["accept_length"], b["tps"]),
            xytext=(15, -30),
            textcoords="offset points",
            fontsize=8,
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        )

    fig.tight_layout()
    out_path = output_dir / "results_plot.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[output] Plot saved  → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    """Parse a comma-separated list of positive ints (e.g. '4,6,8')."""
    values = [int(x) for x in s.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated list of positive integers, got {s!r}")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=
        "Sweep Eagle3 speculative-decoding configurations to find the fastest TPS setting."
    )
    p.add_argument(
        "--engine_dir",
        required=True,
        metavar="DIR",
        help=
        "Path to the Eagle3 engine directory (must contain base_config.json).",
    )
    p.add_argument(
        "--dataset",
        required=True,
        metavar="FILE",
        help="Path to JSONL dataset file. Each line: {\"prompt\": str, "
        "\"output_tokens\": int} (alias \"num_output_tokens\" accepted; "
        "defaults to 128 if neither is present).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        metavar="N",
        help="Number of requests to process concurrently. Must not exceed the "
        "max_batch_size baked into the engine at build time.",
    )
    p.add_argument(
        "--subset_entries",
        type=int,
        default=None,
        metavar="N",
        help=
        "Use at most N entries from the dataset. If N >= dataset size (or omitted), the whole set is used.",
    )
    p.add_argument(
        "--top_ks",
        type=_parse_int_list,
        default=_DEFAULT_TOP_KS,
        metavar="LIST",
        help=f"Comma-separated topK values to sweep (default: "
        f"{','.join(str(v) for v in _DEFAULT_TOP_KS)}).",
    )
    p.add_argument(
        "--draft_steps",
        type=_parse_int_list,
        default=_DEFAULT_DRAFT_STEPS,
        metavar="LIST",
        help=f"Comma-separated draftStep values to sweep (default: "
        f"{','.join(str(v) for v in _DEFAULT_DRAFT_STEPS)}). "
        f"draftStep dominates drafter latency — narrow this for a tighter sweep.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        metavar="DIR",
        help="Optional. If set, writes two files into DIR: "
        "results_table.txt (the ranked table also printed to stdout) and "
        "results_plot.png (TPS vs accept-length scatter). "
        "If omitted, nothing is written to disk — results only go to stdout.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    engines_dir = Path(args.engine_dir).resolve()

    if not engines_dir.exists():
        print(f"ERROR: engines_dir not found: {engines_dir}", file=sys.stderr)
        return 1
    if not Path(args.dataset).exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    # Read engine capacity
    try:
        max_ts, max_batch_size = _read_engine_limits(engines_dir)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[config] engines_dir      = {engines_dir}")
    print(f"[config] max_verify_tree_size = {max_ts}")
    print(f"[config] max_batch_size   = {max_batch_size}")

    if args.batch_size > max_batch_size:
        print(
            f"ERROR: --batch_size {args.batch_size} exceeds the engine's "
            f"max_batch_size={max_batch_size}. Rebuild the engine with "
            f"--maxBatchSize >= {args.batch_size}.",
            file=sys.stderr,
        )
        return 1

    # Load dataset
    try:
        requests = _load_dataset(args.dataset, args.subset_entries)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[config] dataset entries  = {len(requests)}")
    print(f"[config] batch_size       = {args.batch_size}")

    # Generate sweep combinations
    combos = _generate_combos(max_ts, args.top_ks, args.draft_steps)
    print(f"[config] top_ks           = {args.top_ks}")
    print(f"[config] draft_steps      = {args.draft_steps}")
    print(f"[config] total combos     = {len(combos)}")
    print()

    # Make experimental/ importable regardless of cwd. Everything else
    # (LD_LIBRARY_PATH, TRT_PACKAGE_DIR, etc.) is the caller's responsibility.
    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root / "experimental"))

    from server.runtime.engine import _import_runtime  # noqa: E402
    rt = _import_runtime()
    rt.set_profiling_enabled(True)

    # Run sweep
    results: list[dict] = []
    total = len(combos)
    sweep_t0 = time.perf_counter()

    for idx, (topk, ds, vts) in enumerate(combos, 1):
        elapsed_so_far = time.perf_counter() - sweep_t0
        eta = (elapsed_so_far / (idx - 1) *
               (total - idx + 1)) if idx > 1 else float("nan")
        eta_str = f"{eta:.0f}s" if math.isfinite(eta) else "?"
        print(
            f"[{idx:>2}/{total}] topK={topk:>2}  draftStep={ds}  vts={vts:>3}"
            f"  (ETA ≈ {eta_str})",
            flush=True,
        )

        try:
            r = _run_one(rt, engines_dir, requests, topk, ds, vts,
                         args.batch_size)
            results.append(r)
            print(
                f"         → TPS={r['tps']:.2f}  "
                f"acceptLen={r['accept_length']:.3f}  "
                f"acceptRate={r['accept_rate_%']:.1f}%",
                flush=True,
            )
        except Exception as exc:
            print(f"         → FAILED: {exc}", file=sys.stderr, flush=True)

    if not results:
        print("ERROR: all configurations failed.", file=sys.stderr)
        return 1

    # Sort by TPS descending
    results.sort(key=lambda r: r["tps"], reverse=True)

    _print_table(results)

    if args.output_dir:
        out = Path(args.output_dir)
        _save_table(results, out)
        _save_plot(results, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
