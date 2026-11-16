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
"""Run a NeMo Evaluator benchmark against an Edge-LLM engine."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1/chat/completions"


def _resolve_executable(executable: str) -> str:
    path = shutil.which(executable)
    if path:
        return path
    if os.path.exists(executable):
        return executable
    raise FileNotFoundError(f"Executable not found: {executable}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited before becoming healthy, code={proc.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for server health at {url}: {last_error}")


def _terminate_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _tail_file(path: Path, nb_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-nb_lines:])


def _iter_json_scalars(node: Any,
                       path: str = "") -> Iterable[Tuple[str, float]]:
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else key
            yield from _iter_json_scalars(value, next_path)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _iter_json_scalars(value, f"{path}[{idx}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


SUMMARY_METRIC_KEYS = frozenset({
    "accuracy",
    "exact_match",
    "f1",
    "inference_time_seconds",
    "pass_at_1",
    "peak_memory_bytes",
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "runtime_seconds",
    "score",
    "score_macro",
    "scoring_time_seconds",
    "successful_responses",
    "total_responses",
    "wer",
})


def _iter_json_files(output_dir: Path) -> Iterable[Path]:
    report_path = output_dir / "report.json"
    seen = set()
    if report_path.is_file():
        seen.add(str(report_path))
        yield report_path
    for path in sorted(output_dir.rglob("*.json")):
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        yield path


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_report_metric_value(metric_path: str,
                            score_key: Optional[str] = None) -> bool:
    parts = metric_path.split(".")
    if len(parts) != 8:
        return False
    if (parts[0] != "results" or parts[1] not in ("groups", "tasks")
            or parts[3] != "metrics" or parts[5] != "scores"
            or parts[7] != "value"):
        return False
    return score_key is None or parts[4] == score_key


def _summary_metric_label(metric_path: str) -> Optional[str]:
    if _is_report_metric_value(metric_path):
        parts = metric_path.split(".")
        return f"{parts[1]}.{parts[2]}.{parts[4]}.{parts[6]}"
    if metric_path.split(".")[-1] in SUMMARY_METRIC_KEYS:
        return metric_path
    if metric_path.endswith(".value") and ".metrics." in metric_path:
        return metric_path[:-len(".value")]
    return None


def _format_metric_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _summary_metric_key(label: str) -> str:
    parts = label.split(".")
    if len(parts) >= 4 and parts[0] in ("groups", "tasks"):
        return parts[-2]
    return parts[-1]


def _collect_metric_summary(
    output_dir: Path,
    max_metrics: int = 16,
    excluded_keys: Iterable[str] = ()
) -> List[Tuple[str, float]]:
    metrics: Dict[str, float] = {}
    for path in _iter_json_files(output_dir):
        data = _load_json(path)
        if data is None:
            continue
        for metric_path, value in _iter_json_scalars(data):
            label = _summary_metric_label(metric_path)
            if label is None or label in metrics:
                continue
            metrics[label] = value

    excluded = set(excluded_keys)
    summary = []
    for label, value in metrics.items():
        if _summary_metric_key(label) in excluded:
            continue
        if "." not in label and any(
                other_label.endswith(f".{label}") and other_value == value
                for other_label, other_value in metrics.items()):
            continue
        summary.append((label, value))
        if len(summary) >= max_metrics:
            break
    return summary


def _load_score(output_dir: Path, score_key: str) -> Optional[float]:
    direct_matches: List[float] = []
    structured_micro_matches: List[float] = []
    structured_matches: List[float] = []
    for path in _iter_json_files(output_dir):
        data = _load_json(path)
        if data is None:
            continue
        for metric_path, value in _iter_json_scalars(data):
            if metric_path == score_key or metric_path.split(
                    ".")[-1] == score_key:
                direct_matches.append(value)
            elif _is_report_metric_value(metric_path, score_key):
                score_type = metric_path.split(".")[-2]
                if score_type == "micro":
                    structured_micro_matches.append(value)
                else:
                    structured_matches.append(value)
    if direct_matches:
        return direct_matches[0]
    if structured_micro_matches:
        return structured_micro_matches[0]
    if structured_matches:
        return structured_matches[0]
    return None


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in ("all", "full", "none"):
        return None
    return int(value)


def _override_value_to_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _build_overrides(args: argparse.Namespace) -> str:
    override_map = {
        "config.params.limit_samples": args.limit_samples,
        "config.params.parallelism": args.parallelism,
        "config.params.max_new_tokens": args.max_new_tokens,
        "config.params.request_timeout": args.request_timeout,
        "config.params.temperature": args.temperature,
        "config.params.top_p": args.top_p,
    }

    overrides = []
    for key, value in override_map.items():
        value_str = _override_value_to_string(value)
        if value_str is not None:
            overrides.append(f"{key}={value_str}")
    if args.extra_overrides.strip():
        overrides.append(args.extra_overrides.strip())
    return ",".join(overrides)


def _default_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    if args.model:
        return Path(args.model).name
    raise ValueError("model ID is unavailable")


def _discover_model_id(model_url: str) -> str:
    parsed = urllib.parse.urlsplit(model_url)
    path = parsed.path.rstrip("/")
    marker = "/v1/"
    if marker not in path:
        raise ValueError(
            f"--model-url {model_url!r} must point to an OpenAI-compatible "
            "/v1 endpoint")
    models_path = path.split(marker, maxsplit=1)[0] + "/v1/models"
    models_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, models_path, "", ""))
    try:
        with urllib.request.urlopen(models_url, timeout=30) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not discover the served model from {models_url}: {exc}") \
            from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if (not isinstance(models, list) or not models
            or not isinstance(models[0], dict)
            or not isinstance(models[0].get("id"), str)
            or not models[0]["id"]):
        raise RuntimeError(f"No served model found at {models_url}")
    return models[0]["id"]


def _make_server_cmd(args: argparse.Namespace, port: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "tensorrt_edgellm.server",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--served-model-name",
        _default_model_id(args),
    ]
    if args.speculative_config:
        cmd.extend(["--speculative-config", args.speculative_config])
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    cmd.extend([
        "--max-input-len",
        str(args.max_input_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-kv-cache-capacity",
        str(args.max_kv_cache_capacity),
    ])
    return cmd


def _make_eval_cmd(args: argparse.Namespace, model_url: str) -> List[str]:
    return [
        args.nemo_evaluator_bin,
        "run_eval",
        "--eval_type",
        args.eval_type,
        "--model_id",
        _default_model_id(args),
        "--model_url",
        model_url,
        "--model_type",
        "chat",
        "--output_dir",
        args.output_dir,
        "--overrides",
        _build_overrides(args),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeMo Evaluator against TensorRT Edge-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--model-url",
        default="",
        help="OpenAI-compatible chat completions URL for an existing server",
    )
    source_group.add_argument(
        "--model",
        default="",
        help="Hugging Face model ID or local checkpoint to serve locally",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--speculative-config",
        default="",
        help="JSON speculative-decoding configuration passed to the server",
    )
    parser.add_argument("--nemo-evaluator-bin", default="nemo-evaluator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=8192,
    )
    parser.add_argument("--eval-type", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--output-dir", default="nemo-results")
    parser.add_argument("--limit-samples",
                        type=_parse_optional_int,
                        default=None)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--request-timeout", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--extra-overrides", default="")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--score-key", default="score")
    args = parser.parse_args()
    if not args.model_url and not args.model:
        args.model_url = _DEFAULT_MODEL_URL
    return args


def main() -> int:
    args = parse_args()
    args.nemo_evaluator_bin = _resolve_executable(args.nemo_evaluator_bin)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_url = args.model_url
    server_log: Optional[Path] = None
    if args.model:
        port = args.port if args.port else _free_port()
        model_url = f"http://{args.host}:{port}/v1/chat/completions"
        server_log = output_dir / "edgellm_server.log"
        server_cmd = _make_server_cmd(args, port)
        print("Starting Edge-LLM server:")
        print(" ".join(server_cmd))
        with server_log.open("w") as log_file:
            server_proc = subprocess.Popen(server_cmd,
                                           stdout=log_file,
                                           stderr=subprocess.STDOUT,
                                           text=True)
            try:
                _wait_for_health(f"http://{args.host}:{port}/health",
                                 server_proc, args.startup_timeout)
                eval_cmd = _make_eval_cmd(args, model_url)
                print("Running NeMo Evaluator:")
                print(" ".join(eval_cmd))
                result = subprocess.run(eval_cmd, check=False)
            except (RuntimeError, TimeoutError):
                log_file.flush()
                print("Edge-LLM server log tail:")
                print(_tail_file(server_log))
                raise
            finally:
                _terminate_server(server_proc)
    else:
        print(f"Using existing Edge-LLM server: {model_url}")
        if not args.model_id:
            args.model_id = _discover_model_id(model_url)
            print(f"Discovered served model: {args.model_id}")
        eval_cmd = _make_eval_cmd(args, model_url)
        print("Running NeMo Evaluator:")
        print(" ".join(eval_cmd))
        result = subprocess.run(eval_cmd, check=False)

    if result.returncode != 0:
        if server_log is not None:
            print("Edge-LLM server log tail:")
            print(_tail_file(server_log))
        return result.returncode

    score = _load_score(output_dir, args.score_key)
    if score is None:
        print(f"Could not find score key {args.score_key!r} in {output_dir}")
        if args.min_score is not None:
            return 1
    else:
        print(
            f"NeMo Evaluator {args.score_key}: {_format_metric_value(score)}")

    metric_summary = _collect_metric_summary(output_dir,
                                             excluded_keys=(args.score_key, ))
    if metric_summary:
        print("NeMo Evaluator metrics:")
        for metric_name, metric_value in metric_summary:
            print(f"  {metric_name}: {_format_metric_value(metric_value)}")

    if args.min_score is not None and score < args.min_score:
        print(f"Score below threshold: {score} < {args.min_score}")
        return 1
    print(f"NeMo Evaluator output written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
