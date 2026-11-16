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
"""Skip-softmax (BLASST) deployment shared by the ONNX exporter and the
checkpoint-direct builder: which per-length scale table (if any) each phase
deploys, resolved once from the checkpoint config and the export flags.

The runtime reads S from the tables written to the engine config.json; the
plugin attributes ``skip_softmax_prefill_input`` / ``skip_softmax_decode_input``
only say which table inputs a layer carries.

Naming: identifiers of prefill-only things carry ``prefill``, decode-only
things carry ``decode``; names with neither are phase-agnostic. Checkpoint
keys without a phase (``skip_softmax_scale_table`` etc.) are not read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

ScaleTable = List[List[int]]

#: Maximum element count of the ``skip_softmax_prefill_scale`` and
#: ``skip_softmax_decode_scale`` engine inputs (128 ``(seqlen_bin, S)`` pairs).
#: Must match ``rt::kSKIP_SOFTMAX_SCALE_TABLE_MAX_LEN`` (cpp/common/skipSoftmaxScaleTable.h).
SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN = 256

PHASES = ("prefill", "decode")


class SkipSoftmaxConfigError(ValueError):
    """Malformed or contradictory skip-softmax flags / checkpoint keys."""


@dataclass(frozen=True)
class SkipSoftmaxDeployment:
    """What one engine deploys. ``None`` table = that phase runs dense."""
    prefill_table: Optional[ScaleTable] = None
    decode_table: Optional[ScaleTable] = None
    #: Prefill ``{a, b}`` fit and the target that selected ``prefill_table``,
    #: recorded in the engine config only when a prefill table is deployed.
    prefill_calibration: Optional[Dict[str, float]] = None
    prefill_target_sparsity: Optional[float] = None

    @property
    def prefill_enabled(self) -> bool:
        return self.prefill_table is not None

    @property
    def decode_enabled(self) -> bool:
        return self.decode_table is not None

    @property
    def enabled(self) -> bool:
        return self.prefill_enabled or self.decode_enabled


def tier_key(target: float) -> str:
    """config.json key of a tier's table, e.g. 0.75 -> "75%"."""
    return f"{round(float(target) * 100):d}%"


def phase_prefix(phase: str) -> str:
    """Checkpoint / engine-config key prefix of ``phase``:
    ``skip_softmax_prefill_`` or ``skip_softmax_decode_``."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    return f"skip_softmax_{phase}_"


def validate_table(
        rows: Any,
        name: str = "skip_softmax_prefill_scale_table") -> ScaleTable:
    """Return ``rows`` as a bin-sorted ``[[seqlen_bin, S], ...]`` table.

    Mirrors the runtime parser (llmEngineConfig.cpp): non-empty, integral
    non-negative entries, at most ``SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN / 2``
    rows, bins unique (the kernel floor-looks-up S by bin).
    """
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(r, (list, tuple)) or len(r) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or int(v) != v or v < 0 for v in r) for r in rows)):
        raise SkipSoftmaxConfigError(
            f"{name} must be a non-empty list of non-negative integer "
            "[seqlen_bin, S] rows")
    if 2 * len(rows) > SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN:
        raise SkipSoftmaxConfigError(
            f"{name} has more than {SKIP_SOFTMAX_SCALE_TABLE_MAX_LEN // 2} "
            "rows")
    table = sorted([int(b), int(s)] for b, s in rows)
    bins = [b for b, _ in table]
    if len(set(bins)) != len(bins):
        dup = sorted({b for b in bins if bins.count(b) > 1})
        raise SkipSoftmaxConfigError(f"{name} repeats seqlen_bin {dup}")
    return table


def _check_sparsity(value: Any, name: str) -> float:
    try:
        sparsity = float(value)
    except (TypeError, ValueError):
        raise SkipSoftmaxConfigError(f"{name} must be a number in (0, 1), "
                                     f"got {value!r}") from None
    if not 0.0 < sparsity < 1.0:
        raise SkipSoftmaxConfigError(
            f"{name} must be in (0, 1), got {sparsity!r}")
    return sparsity


def _check_calibration(value: Any, name: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise SkipSoftmaxConfigError(f"{name} must be a JSON object with "
                                     "numeric keys a and b")
    fit: Dict[str, float] = {}
    for key in ("a", "b"):
        try:
            fit[key] = float(value[key])
        except (KeyError, TypeError, ValueError):
            raise SkipSoftmaxConfigError(
                f"{name} must carry numeric keys a and b "
                f"(got {dict(value)!r})") from None
        if not math.isfinite(fit[key]):
            raise SkipSoftmaxConfigError(f"{name}[{key!r}] is not finite")
    return fit


def _formula_table(fit: Dict[str, float], target: float) -> ScaleTable:
    return [[0, int(round(fit["a"] * math.exp(fit["b"] * target)))]]


def _is_dense_flag(table: Any) -> bool:
    """``[]`` is the explicit "this phase runs dense" spelling; anything else
    non-None is a candidate table and goes through ``validate_table``."""
    return isinstance(table, list) and not table


def resolve(src_root: Mapping[str, Any],
            src_llm: Mapping[str, Any],
            *,
            prefill_target_sparsity: Optional[float] = None,
            prefill_table: Any = None,
            decode_target_sparsity: Optional[float] = None,
            decode_table: Any = None) -> SkipSoftmaxDeployment:
    """Resolve the skip-softmax deployment of one engine.

    ``src_llm`` is the (promoted) text config, ``src_root`` the checkpoint
    root; every checkpoint key is looked up in that order. Prefill keys are
    read under ``skip_softmax_prefill_*``, decode keys under
    ``skip_softmax_decode_*``; nothing without a phase is read.

    The keyword arguments are the four export flags. A table flag is ``None``
    (flag absent), ``[]`` (``none``: this phase runs dense whatever the
    checkpoint says) or already-loaded ``[[seqlen_bin, S], ...]`` rows. The
    two phases are selected independently; a prefill flag never changes the
    decode selection and vice versa.

    Prefill, first match wins:
      1. ``prefill_table`` flag: ``[]`` -> dense, else that table.
      2. ``prefill_target_sparsity`` flag R with a ``tier_key(R)`` table under
         the checkpoint ``skip_softmax_prefill_scale_tables``.
      3. No prefill flag: checkpoint tier
         (``skip_softmax_prefill_target_sparsity`` in
         ``skip_softmax_prefill_scale_tables``), then
         ``skip_softmax_prefill_scale_table``.
      4. Target R (flag, or checkpoint ``skip_softmax_prefill_target_sparsity``
         when no prefill flag) through the checkpoint fit
         ``skip_softmax_prefill_calibration`` {a, b} ->
         ``[[0, round(a * exp(b * R))]]``; a target without a fit is an error.
      5. Dense.

    Decode: the same five steps over ``decode_table``,
    ``decode_target_sparsity`` and the ``skip_softmax_decode_*`` keys.

    """

    def ckpt(key: str) -> Any:
        for cfg in (src_llm, src_root):
            value = cfg.get(key)
            if value is not None:
                return value
        return None

    def canonical(phase: str, suffix: str) -> str:
        return phase_prefix(phase) + suffix

    def phase_value(phase: str, suffix: str) -> Any:
        return ckpt(canonical(phase, suffix))

    def resolve_phase(phase: str, table_flag: Any,
                      target_flag: Optional[float]):
        """(table, target that selected it, checkpoint fit) of one phase."""
        if target_flag is not None:
            target_flag = _check_sparsity(target_flag,
                                          f"--{phase}-target-sparsity")
        tables = phase_value(phase, "scale_tables")
        tables = tables if isinstance(tables, Mapping) else {}
        tables_key = canonical(phase, "scale_tables")
        fit = phase_value(phase, "calibration")
        if fit is not None:
            fit = _check_calibration(fit, canonical(phase, "calibration"))

        table: Optional[ScaleTable] = None
        target: Optional[float] = None
        if table_flag is not None:
            if not _is_dense_flag(table_flag):
                table = validate_table(table_flag,
                                       f"--skip-softmax-{phase}-scale-table")
        elif target_flag is not None:
            target = target_flag
            if tier_key(target) in tables:
                table = validate_table(tables[tier_key(target)],
                                       f"{tables_key}[{tier_key(target)!r}]")
        else:
            ckpt_target = phase_value(phase, "target_sparsity")
            if ckpt_target is not None:
                ckpt_target = _check_sparsity(
                    ckpt_target, canonical(phase, "target_sparsity"))
            ckpt_table = phase_value(phase, "scale_table")
            if ckpt_target is not None and tier_key(ckpt_target) in tables:
                target = ckpt_target
                table = validate_table(tables[tier_key(target)],
                                       f"{tables_key}[{tier_key(target)!r}]")
            elif ckpt_table is not None:
                table = validate_table(ckpt_table,
                                       canonical(phase, "scale_table"))
            elif ckpt_target is not None:
                target = ckpt_target

        if table is None and target is not None:
            if fit is None:
                raise SkipSoftmaxConfigError(
                    f"{phase} target sparsity {target}: no "
                    f"'{tier_key(target)}' table under {tables_key} (have "
                    f"{sorted(tables) or 'none'}) and no "
                    f"{canonical(phase, 'calibration')} {{a, b}} to derive S "
                    "from; run calibrate_skip_softmax.py")
            table = _formula_table(fit, target)
        return table, target, fit

    resolved_prefill, prefill_target, prefill_fit = resolve_phase(
        "prefill", prefill_table, prefill_target_sparsity)
    resolved_decode, _, _ = resolve_phase("decode", decode_table,
                                          decode_target_sparsity)

    return SkipSoftmaxDeployment(
        prefill_table=resolved_prefill,
        decode_table=resolved_decode,
        prefill_calibration=(prefill_fit
                             if resolved_prefill is not None else None),
        prefill_target_sparsity=(prefill_target
                                 if resolved_prefill is not None else None),
    )
