#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Compare the PyTorch golden and EdgeLLM 4-layer dumps.

Loads two safetensors files:
- golden  : produced by ``golden_layer_dump.py`` (left padding).
- edgellm : produced by the C++ runtime when ``EDGELLM_DUMP_LOGITS_KVCACHE_*`` is set
            (right-aligned).

Compares logits and KV cache per round (round 0 = prefill, 1..N = decode), per sequence and
per layer, and checks that both sides' greedy token choices agree each round.

== Pairing rounds: by committed length, not by round index ==
Each (engine round, sequence) is paired with the golden round holding the same committed length,
derived from the dumped ``context_lengths``. Under vanilla decoding every row advances by one token
per round and this is just ``g == r``; under speculative decoding a round commits a different
number of tokens per sequence, so one engine round can line up with a different golden round in
each row -- and with none at all in a row that committed nothing. Rows the golden never reached are
reported as ``skipped``.

== Format differences and how they are reconciled ==
- **logits**: golden ``round_{r}.logits`` is ``[B, 1, vocab]``; edgellm is ``[B, vocab]``. Both are
  the last real token's logits, so they compare directly.
- **KV layout**: golden stores separate ``round_{r}.layer_{i}.key`` / ``.value`` of shape
  ``[B, kv_heads, seq, head_dim]``; edgellm stores a combined ``round_{r}.layer_{i}.kv`` of shape
  ``[B, 2, kv_heads, maxSeqLen, head_dim]`` (dim 1: 0=key, 1=value), dumped full-length (the engine
  does no truncation). The combined tensor is split, and each side is sliced to ``valid`` below.
- **padding side**: golden is left-padded (real tokens occupy each row's right segment); edgellm is
  right-aligned (real tokens occupy ``[0, valid)``). So within the valid length ``valid`` for
  sequence b at round r:
    golden takes the last ``valid`` columns: ``g[..., -valid:, ...]``
    edgellm takes the first ``valid`` columns: ``e[..., :valid, ...]``
  where ``valid = edgellm round_{r}.context_lengths[b]`` (= prefill real length + r).
- **dtype**: both sides are upcast to float32 before comparing (golden KV/logits are usually fp16,
  edgellm logits are fp32).

== What gates the run ==
Two independent checks, because either one alone has a blind spot:
- **cosine** (``--cos``): direction agreement. It is scale-invariant, so a dropped scale
  factor -- both sides computing the same thing up to a constant -- passes it at 1.0.
- **norm ratio** (``--scale-tol``): ``||edgellm|| / ||golden||``. Both sides carry
  same-strength quantization noise, which largely cancels in a ratio, so this stays within
  1e-4 (fp16) to 1e-2 (NVFP4 KV state) even where cosine has already fallen to 0.97. A
  dropped factor is common-mode and does not cancel, so it shows up here immediately.
``allclose`` (``--atol`` / ``--rtol``) is reported per round but does **not** gate: absolute
error is tied to each tensor's dynamic range, and that spans orders of magnitude across
layers (a Qwen3 K cache carries outlier channels ~100x the bulk), so no single tolerance
fits every tensor.

== Input-alignment check ==
At round 0, verify that the golden's real prefill lengths (per-row attention_mask sums) equal the
edgellm round_0.context_lengths. If they differ, the two sides consumed different input_ids (most
commonly: one applied a chat template while the other used raw tokenization), the numeric comparison
would be meaningless, and the tool reports this structural error immediately.

== Exit code ==
0 = everything passes; 1 = any round/tensor fails or the structure is misaligned.
"""

from __future__ import annotations

import argparse
import re
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

_ROUND_RE = re.compile(r"^round_(\d+)\.")


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load an entire safetensors file into a {name: tensor} dict."""
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _order_dumps(paths: list[str]) -> list[str]:
    """Sort engine dumps by request index (``edgellm_dump.safetensors`` = 0, ``_N`` = N).

    A run that issues several requests writes one file per request; plain lexicographic order
    would put ``_10`` before ``_2``, so key on the parsed index.
    """

    def key(path: str) -> tuple:
        m = re.search(r"edgellm_dump(?:_(\d+))?\.safetensors$", path)
        return (0, int(m.group(1) or 0), "") if m else (1, 0, path)

    return sorted(paths, key=key)


def _infer_rounds(d: dict[str, torch.Tensor]) -> list[int]:
    """Infer the available rounds from the keys (looking at round_{r}.logits)."""
    rs = set()
    for k in d:
        m = _ROUND_RE.match(k)
        if m and k.endswith(".logits"):
            rs.add(int(m.group(1)))
    return sorted(rs)


def _infer_layers(d: dict[str, torch.Tensor], round_idx: int) -> list[int]:
    """Infer a round's layers from the keys.

    Attention layers: golden ``.key``/``.value``, edgellm ``.kv``. Recurrent layers (Mamba /
    Gated DeltaNet): both sides ``.recurrent_state`` (+ ``.conv_state``).
    """
    ls = set()
    for k in d:
        m = re.match(
            rf"^round_{round_idx}\.layer_(\d+)\.(key|kv|recurrent_state)$", k)
        if m:
            ls.add(int(m.group(1)))
    return sorted(ls)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Whole-tensor cosine similarity (both operands flattened to one vector).

    Delegates to ``torch.nn.functional.cosine_similarity`` for the usual case.
    The explicit zero-norm guard is kept on purpose: ``F.cosine_similarity``
    clamps the denominator with ``eps`` and would return a near-zero value for
    an all-zero tensor, but here an all-zero side is a meaningful degenerate
    case (e.g. a near-zero gate activation), so we report an exact ``1.0`` when
    both sides are identically zero and ``0.0`` when only one side is zero.
    """
    a = a.flatten()
    b = b.flatten()
    if a.norm() == 0 or b.norm() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(F.cosine_similarity(a, b, dim=0))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def norm_ratio(golden: torch.Tensor, edgellm: torch.Tensor) -> float:
    """``||edgellm|| / ||golden||`` -- the magnitude agreement cosine cannot see.

    Same degenerate-case convention as ``cosine_similarity``: an identically zero
    pair is an exact match (1.0), one-sided zero is maximally wrong (inf).
    """
    gn = float(golden.flatten().norm())
    en = float(edgellm.flatten().norm())
    if gn == 0.0:
        return 1.0 if en == 0.0 else float("inf")
    return en / gn


class Stat:
    """Result of comparing a single tensor."""

    def __init__(self, name: str, cos: float, max_abs: float, allclose: bool,
                 scale: float):
        self.name = name
        self.cos = cos
        self.max_abs = max_abs
        self.allclose = allclose
        self.scale = scale

    @property
    def scale_dev(self) -> float:
        """Distance of the norm ratio from 1; what ``--scale-tol`` bounds."""
        return abs(self.scale - 1.0)


def _compare_tensor(name: str, golden: torch.Tensor, edgellm: torch.Tensor,
                    atol: float, rtol: float) -> Stat:
    golden = golden.to(torch.float32)
    edgellm = edgellm.to(torch.float32)
    if golden.shape != edgellm.shape:
        # A shape mismatch is recorded as the worst possible result (usually means the reconcile
        # slicing is wrong or the lengths don't line up).
        return Stat(
            name +
            f" SHAPE-MISMATCH golden{tuple(golden.shape)} edgellm{tuple(edgellm.shape)}",
            0.0, float("inf"), False, float("inf"))
    return Stat(name, cosine_similarity(golden, edgellm),
                _max_abs(golden, edgellm),
                bool(torch.allclose(golden, edgellm, atol=atol, rtol=rtol)),
                norm_ratio(golden, edgellm))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden",
                   required=True,
                   help="golden safetensors (PyTorch side)")
    p.add_argument(
        "--edgellm",
        required=True,
        nargs="+",
        help=
        "edgellm safetensors (C++ runtime side). Pass several when the run issued "
        "several requests -- context-reuse validation does, one per request; they are "
        "ordered by the request index in the file name and consume the golden's rows in "
        "that order")
    p.add_argument("--cos",
                   type=float,
                   default=0.99,
                   help="minimum cosine similarity (default 0.99)")
    p.add_argument(
        "--scale-tol",
        type=float,
        default=0.25,
        help="max |norm(edgellm)/norm(golden) - 1| per tensor (default 0.25); "
        "catches a dropped scale factor, which cosine is blind to")
    p.add_argument("--atol",
                   type=float,
                   default=2e-2,
                   help="atol for allclose (default 2e-2)")
    p.add_argument("--rtol",
                   type=float,
                   default=2e-2,
                   help="rtol for allclose (default 2e-2)")
    p.add_argument("--verbose",
                   action="store_true",
                   help="print every tensor's comparison result")
    p.add_argument(
        "--teacher-forced",
        action="store_true",
        help=
        "the run was teacher-forced (EdgeLLM was fed the golden's tokens), so treat per-round "
        "token disagreement as informational rather than a failure (cosine is the gate)"
    )
    args = p.parse_args()

    golden = _load_safetensors(args.golden)
    dumps = [_load_safetensors(path) for path in _order_dumps(args.edgellm)]

    golden_rounds = _infer_rounds(golden)
    print(
        f"[compare] golden rounds={golden_rounds}; {len(dumps)} engine dump(s): "
        + ", ".join(f"{_infer_rounds(d)}" for d in dumps))
    if not golden_rounds or any(not _infer_rounds(d) for d in dumps):
        print("[compare] FAIL: one side dumped no rounds.")
        return 1

    bs = int(golden["input_ids"].shape[0])

    # Each dump consumes as many golden rows as it has sequences, in request order: the golden
    # runs every request as one padded batch, so its row r is the request the r-th dumped
    # sequence came from.
    row_base: list[int] = []
    total_rows = 0
    for d in dumps:
        row_base.append(total_rows)
        total_rows += int(d["round_0.context_lengths"].shape[0])
    if total_rows != bs:
        print(
            f"[compare] FAIL: the engine dumped {total_rows} sequence(s) but the golden "
            f"holds {bs}.")
        return 1

    # ---- Input-alignment check (using the round-0 lengths) ----
    # Golden real prefill length = per-row attention_mask sum; edgellm = round_0.context_lengths.
    golden_real_len = golden["attention_mask"].sum(-1).to(torch.int64)  # [B]
    edgellm_ctx0 = torch.cat(
        [d["round_0.context_lengths"].to(torch.int64) for d in dumps])
    if not torch.equal(golden_real_len, edgellm_ctx0):
        print(
            "[compare] FAIL: input misalignment — golden real prefill lengths "
            f"{golden_real_len.tolist()} != edgellm round_0 context_lengths "
            f"{edgellm_ctx0.tolist()}.")
        print(
            "           The two sides consumed different input_ids (commonly: one "
            "applied a chat template, the other used raw tokenization), or a reused "
            "context-cache prefix was not added back to the dumped length. Skipping the "
            "numeric comparison.")
        return 1
    print(
        f"[compare] input alignment OK: real prefill lengths {golden_real_len.tolist()}"
    )

    overall_ok = True
    worst = None  # (cos, name)
    worst_scale = None  # (scale, name) of the largest deviation from 1

    golden_round_set = set(golden_rounds)
    n_generated = int(golden["generated_ids"].shape[1])
    compared: set = set()  # (row, golden round) pairs already compared

    for d_idx, edgellm in enumerate(dumps):
        base = row_base[d_idx]
        req = f"q{d_idx}." if len(dumps) > 1 else ""
        for r in _infer_rounds(edgellm):
            ctx = edgellm[f"round_{r}.context_lengths"].to(torch.int64)  # [B]
            layers = _infer_layers(edgellm, r)
            round_stats: list[Stat] = []
            edgellm_tok = edgellm.get(f"round_{r}.generated_token_ids")

            tok_agree = 0
            tok_total = 0
            tok_detail: list[str] = []
            unmatched: list[str] = []

            for b in range(int(ctx.shape[0])):
                valid = int(ctx[b])
                # Pair by committed length, not by round index: a speculative round commits a
                # different number of tokens per sequence, so one engine round can line up with a
                # different golden round in each row (and with none at all when it commits nothing).
                gb = base + b  # this dump's row b is the golden's row gb
                g = valid - int(golden_real_len[gb])
                if g not in golden_round_set:
                    unmatched.append(f"b{b}@{valid}")
                    continue
                if (gb, g) in compared:
                    continue  # this row did not advance this round
                compared.add((gb, g))
                tag = (f"{req}r{r}.b{b}"
                       if g == r else f"{req}r{r}.b{b}(g{g})")

                # Token agreement: the engine's own choice for the token this round leaves pending
                # vs the golden's. Under teacher forcing the runtime is fed the golden's tokens, so a
                # mismatch only means the runtime's own greedy would have differed (a near-tie) --
                # informational, not a failure; cosine is the gate.
                if edgellm_tok is not None and g < n_generated:
                    et = int(edgellm_tok[b])
                    if et >= 0:  # a speculative round that committed nothing records -1
                        gt = int(golden["generated_ids"][gb, g])
                        tok_total += 1
                        if gt == et:
                            tok_agree += 1
                        else:
                            tok_detail.append(f"b{b} golden={gt} edgellm={et}")

                # ---- logits ----
                golden_logits = golden[f"round_{g}.logits"][
                    gb, -1, :]  # [vocab] (last position of [B,1,vocab])
                edgellm_logits = edgellm[f"round_{r}.logits"][b, :]  # [vocab]
                round_stats.append(
                    _compare_tensor(f"{tag}.logits", golden_logits,
                                    edgellm_logits, args.atol, args.rtol))
                # ---- per-layer state: attention KV, or recurrent (Mamba / Gated DeltaNet) ----
                for i in layers:
                    g_rec = f"round_{g}.layer_{i}.recurrent_state"
                    e_rec = f"round_{r}.layer_{i}.recurrent_state"
                    if g_rec in golden and e_rec in edgellm:
                        # Recurrent layer: fixed-size state, no sequence dim -> compare row b directly
                        # (no left/right reconcile needed). conv_state likewise when present.
                        round_stats.append(
                            _compare_tensor(f"{tag}.layer_{i}.recurrent_state",
                                            golden[g_rec][gb],
                                            edgellm[e_rec][b], args.atol,
                                            args.rtol))
                        g_conv = f"round_{g}.layer_{i}.conv_state"
                        e_conv = f"round_{r}.layer_{i}.conv_state"
                        if g_conv in golden and e_conv in edgellm:
                            round_stats.append(
                                _compare_tensor(f"{tag}.layer_{i}.conv_state",
                                                golden[g_conv][gb],
                                                edgellm[e_conv][b], args.atol,
                                                args.rtol))
                        continue
                    # Attention layer.
                    # golden: separate key/value, [B, kv_heads, seq, head_dim], take the last valid cols.
                    golden_key = golden[f"round_{g}.layer_{i}.key"][gb, :,
                                                                    -valid:, :]
                    golden_value = golden[f"round_{g}.layer_{i}.value"][
                        gb, :, -valid:, :]
                    # edgellm: combined [B, 2, kv_heads, L, head_dim], split 0/1 and take the first valid cols.
                    edgellm_kv = edgellm[f"round_{r}.layer_{i}.kv"]
                    edgellm_key = edgellm_kv[b, 0, :, :valid, :]
                    edgellm_value = edgellm_kv[b, 1, :, :valid, :]
                    round_stats.append(
                        _compare_tensor(f"{tag}.layer_{i}.key", golden_key,
                                        edgellm_key, args.atol, args.rtol))
                    round_stats.append(
                        _compare_tensor(f"{tag}.layer_{i}.value", golden_value,
                                        edgellm_value, args.atol, args.rtol))

            tag_note = f"{req}prefill" if r == 0 else f"{req}decode {r}"
            if not round_stats:
                print(
                    f"  round {r} [{tag_note}]: nothing to compare "
                    f"(no row reached a new golden round; lengths {ctx.tolist()})"
                )
                continue

            tok_note = "tokens n/a"
            token_mismatch = False
            if tok_total:
                tok_note = f"tokens {tok_agree}/{tok_total}"
                if tok_agree != tok_total:
                    token_mismatch = True
                    tok_note += "  " + "; ".join(tok_detail)
                    if args.teacher_forced:
                        tok_note += "  (teacher-forced: runtime's own greedy differs — informational)"
                    elif r == 0:
                        tok_note += "  (round-0 mismatch — check that the inputs really align)"

            # Round summary
            min_cos_stat = min(round_stats, key=lambda s: s.cos)
            max_abs_stat = max(round_stats, key=lambda s: s.max_abs)
            scale_stat = max(round_stats, key=lambda s: s.scale_dev)
            all_close = all(s.allclose for s in round_stats)
            cos_ok = min_cos_stat.cos >= args.cos
            scale_ok = scale_stat.scale_dev <= args.scale_tol
            # A token mismatch fails the round only when NOT teacher-forced; otherwise the two
            # numeric checks are the gate.
            token_fail = token_mismatch and not args.teacher_forced
            round_ok = cos_ok and scale_ok and not token_fail
            if not round_ok:
                overall_ok = False
            if worst is None or min_cos_stat.cos < worst[0]:
                worst = (min_cos_stat.cos, min_cos_stat.name)
            if worst_scale is None or scale_stat.scale_dev > abs(
                    worst_scale[0] - 1.0):
                worst_scale = (scale_stat.scale, scale_stat.name)

            skip_note = f" skipped={','.join(unmatched)}" if unmatched else ""
            print(f"  round {r} [{tag_note}]: {tok_note:<24s} "
                  f"min_cos={min_cos_stat.cos:.5f} ({min_cos_stat.name}) "
                  f"worst_scale={scale_stat.scale:.5f} ({scale_stat.name}) "
                  f"max_abs={max_abs_stat.max_abs:.4g} "
                  f"allclose={'Y' if all_close else 'N'}{skip_note} "
                  f"-> {'PASS' if round_ok else 'FAIL'}")
            if args.verbose:
                for s in sorted(round_stats, key=lambda s: s.name):
                    print(
                        f"      {s.name:34s} cos={s.cos:.5f} scale={s.scale:.5f} "
                        f"max_abs={s.max_abs:.4g} "
                        f"allclose={'Y' if s.allclose else 'N'}")

    if worst is None:
        print(
            "[compare] FAIL: no (round, sequence) pair could be matched to a golden round."
        )
        return 1

    print(f"[compare] {'PASS' if overall_ok else 'FAIL'}: "
          f"worst cosine {worst[0]:.5f} @ {worst[1]}, "
          f"worst scale {worst_scale[0]:.5f} @ {worst_scale[1]} "
          f"(thresholds: cos {args.cos}, scale-tol {args.scale_tol}; "
          f"atol {args.atol}, rtol {args.rtol})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
