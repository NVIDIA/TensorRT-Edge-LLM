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
"""Validate canonical benchmark workloads."""

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Return a stable hash for a token sequence."""
    encoded = json.dumps(list(token_ids),
                         ensure_ascii=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def positive_number(value: Any, name: str) -> float:
    """Validate and return a positive numeric value."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite and positive") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def normalize_online(workload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an online workload and return its canonical form."""
    run_id = workload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Online workload requires a run_id")
    concurrency = workload.get("concurrency")
    if (not isinstance(concurrency, int) or isinstance(concurrency, bool)
            or concurrency < 1):
        raise ValueError("Online workload concurrency must be positive")

    generation = workload.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Online workload requires generation settings")
    temperature = generation.get("temperature")
    if (not isinstance(temperature, (int, float))
            or isinstance(temperature, bool) or not 0 <= temperature <= 2):
        raise ValueError("generation.temperature must be between 0 and 2")
    top_p = generation.get("top_p")
    if (not isinstance(top_p, (int, float)) or isinstance(top_p, bool)
            or not 0 < top_p <= 1):
        raise ValueError(
            "generation.top_p must be greater than 0 and at most 1")
    top_k = generation.get("top_k")
    if (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
        raise ValueError("generation.top_k must be a non-negative integer")
    if not isinstance(generation.get("ignore_eos"), bool):
        raise ValueError("generation.ignore_eos must be a boolean")

    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Online workload requires requests")
    normalized_requests = []
    request_ids = set()
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Online workload requests must be mappings")
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Each online request requires an id")
        if request_id in request_ids:
            raise ValueError(f"Duplicate online request id: {request_id}")
        request_ids.add(request_id)
        token_ids = request.get("token_ids")
        if (not isinstance(token_ids, list) or not token_ids
                or any(not isinstance(token, int) or isinstance(token, bool)
                       or token < 0 or token > 2**31 - 1
                       for token in token_ids)):
            raise ValueError(
                "Each online request requires non-negative token_ids")
        output_tokens = request.get("output_tokens")
        if (not isinstance(output_tokens, int)
                or isinstance(output_tokens, bool) or output_tokens < 1):
            raise ValueError(
                "Each online request requires positive output_tokens")
        token_hash = token_ids_sha256(token_ids)
        supplied_hash = request.get("input_token_ids_sha256")
        if supplied_hash is not None and supplied_hash != token_hash:
            raise ValueError(
                f"Input token hash does not match request {request_id}")
        normalized_requests.append({
            "id": request_id,
            "token_ids": list(token_ids),
            "input_token_ids_sha256": token_hash,
            "output_tokens": output_tokens,
        })

    load = workload.get("load", {})
    if not isinstance(load, dict):
        raise ValueError("Online workload load settings must be a mapping")
    arrival_rate = load.get("arrival_rate_rps")
    if arrival_rate is not None:
        arrival_rate = positive_number(arrival_rate, "load.arrival_rate_rps")

    warmup = workload.get("warmup", {})
    if not isinstance(warmup, dict):
        raise ValueError("Online workload warmup must be a mapping")
    warmup_iterations = warmup.get("iterations", 0)
    if (not isinstance(warmup_iterations, int)
            or isinstance(warmup_iterations, bool) or warmup_iterations < 0):
        raise ValueError("warmup.iterations must be non-negative")
    warmup_scope = warmup.get("scope", "none")
    if warmup_scope not in ("none", "canonical_batch", "full_dataset"):
        raise ValueError(f"Unsupported warmup scope: {warmup_scope}")
    if warmup_scope == "none" and warmup_iterations:
        raise ValueError("Warmup iterations require a warmup scope")

    return {
        "run_id": run_id,
        "concurrency": concurrency,
        "generation": dict(generation),
        "requests": normalized_requests,
        "arrival_rate_rps": arrival_rate,
        "warmup_iterations": warmup_iterations,
        "warmup_scope": warmup_scope,
    }
