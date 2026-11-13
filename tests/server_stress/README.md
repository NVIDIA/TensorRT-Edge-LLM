<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Server stress testing

`tests/server_stress/` drives a **dedicated** Edge-LLM HTTP server with sustained
mixed streaming/non-streaming traffic, randomized disconnects, overload and
recovery, a soak, and SIGINT/SIGTERM during startup and active inference.
The tests are opt-in and run serially. Each failure produces pytest diagnostics
and preserves the server log and partial measurements. Use the same suite on
x86 and DGX Spark; launch pytest on the server host for process lifecycle tests.

Install `httpx`, `psutil`, `pytest`, and the
[testing dependencies](../README.md#2-install-dependencies). The server additionally
requires a built runtime and the `[build,runtime]` extras. Run the commands below
from the repository root.
The example uses the smallest L0 model above 0.6B, `Qwen/Qwen3.5-0.8B`, covered
by the L0 server case in `tests/model_configs/l0/qwen3_5/qwen3_5_0_8b.yml`.
The checked-in [complete test setup](../test_cases/llm_server_stress.json)
contains the server launch command, engine limits, request prompts, load profiles,
timeouts, and pass/fail limits. With the runtime built and its Python environment
activated, run it directly from the repository root:

```bash
LLM_SDK_DIR="$PWD" python -m pytest tests/server_stress \
  --server-stress-config=tests/test_cases/llm_server_stress.json \
  --server-stress-output=logs/server-stress -v
```

The setup launches `python` from the active `PATH`, serves Qwen/Qwen3.5-0.8B on
`127.0.0.1:18080`, and stores the checkpoint/engine cache under
`logs/server-stress-engine-cache`. The first startup downloads the model and
builds its engine as needed. Choose an available GPU through `CUDA_VISIBLE_DEVICES`
and provide the runtime library environment required by your build.

For another workload or machine, copy the setup to a local JSON file and pass
that path to `--server-stress-config`. Adjust `command` and `cwd` for the runtime,
`concurrency`, `duration_s`, and `soak_s` for the load, and `prompts` and
`max_tokens` for the requests. The model argument also accepts an absolute local
checkpoint path. Keep `url` aligned with `--host` and `--port`, and `model`
aligned with `--served-model-name`.

Timeout fields use seconds. `max_p99_s` and `max_rss_growth_mb` are optional
pass/fail limits; `null` records the measurements only. Omitted fields use the
defaults in `Settings` in [support.py](support.py). This setup runs through the
server stress pytest suite; `llm_inference --input-file` consumes the separate
runtime request format used by `llm_basic.json`.

Remove `--enable-in-flight-batching` from a second configuration to exercise the
blocking runtime. Run configurations sequentially. `command` is an argv array
executed directly; environment variables such as `CUDA_VISIBLE_DEVICES`,
`EDGELLM_PYBIND_DIR`, and `LD_LIBRARY_PATH` are inherited. Credentials use
`SERVER_BENCH_API_KEY` and stay out of report metadata. For an existing dedicated
server, omit `command`; loading/exit cases skip explicitly, and HTTP cases run.

Select `-k concurrency`, `-k cancellation`, `-k loading_interrupt`, or
`-k server_benchmark` for focused runs. A first server startup prepares the
checkpoint-direct engine cache. Loading-interrupt cases first establish a
working cached startup, then wait for `load_pattern` (default:
`Loading runtime bundle from`) before sending a signal. They fail if readiness
has already been reached: use a larger model or earlier `load_delays_s` to
exercise the desired loading window. The log-based loading window is best-effort. These cases measure signal-to-
process-group-exit for the configured command; `loaded_at_signal`, `loaded_before_exit`, and the
return code show whether loading completed or a signal terminated the process.
Cooperative tensor-load cancellation is covered by the CLI implementation
and its dedicated tests. A fast SIGTERM exit can be the default OS signal action.

Busy shutdown checks lifespan completion, process-group exit, and restart while
inference is active. Client outcomes are recorded separately; successful shutdown
alone establishes no guarantee that every in-flight request completed. Its separate `busy_exit_timeout_s` budget defaults to 30 seconds
and should reflect the admitted workload on the target hardware; loading keeps
the separate 2-second `exit_timeout_s` budget. `shutdown_requests.json` records
completed responses, disconnected responses, and requests stopped by test cleanup.
Keep info-level lifecycle logging enabled for the shutdown-completion check.

Reports live under a unique timestamped directory. Each case writes
`requests.jsonl`, `health.jsonl`, and `summary.json` where applicable; process
cases write `server.log` and `exit.json`. The top-level summary includes every case, its pytest status, benchmark
metrics, and lifecycle measurements. Measured metrics include successful requests/s, output
tokens/s, P50/P95/P99/max latency, streaming TTFT, HTTP status counts, separate
success/error/rejection/cancellation counts, health failures, cohort drain time,
and managed-process RSS samples. Server-reported usage supplies token counts;
SSE chunks are not treated as individual tokens. Throughput includes the tail
of in-flight requests after load generation ends. Request deadlines include
connection acquisition and the entire response, including trickling streams.

The default error allowance is zero, measured among successful and failed
requests; cancellations and 429 refusals have separate counters. Each pressure
case requires at least `min_successes` (default 4). `cancel_max_tokens` (default
1024) sets a longer cancellation workload. The focused cancel-drain test requires
an exact native cancellation increment under IFB; first-token churn additionally
requires at least 90% of client disconnects to reach native cancellation, allowing
a small completion/disconnect race. Timed disconnects can occur before native
submission, so that mode requires observed native cancellation plus full drain.
All IFB pressure cases require zero native failures. HTTP admission 429 counts
are distinct from native engine refusals. For single-request cancellation, the blocking path first
verifies that natural completion exceeds twice the drain budget. Use a shorter
`drain_timeout_s` or a longer `cancel_max_tokens` when calibrating a fast model.
An insufficient natural tail produces an explicit inconclusive skip, accompanied
by `calibration.json`; that result establishes no cancellation coverage.
On the blocking runtime, timed-disconnect churn can exhaust the bounded
admission queue and produce 429 responses; the cancellation-churn test retains
that failure as evidence of unavailable capacity.
`max_p99_s` and `max_rss_growth_mb` provide
optional environment-specific gates; record a trusted baseline before setting
those limits. A soak with the default null RSS limit checks sustained service
and reports memory observations; it makes no memory-regression assertion.
`rss_growth_mb` compares post-drain RSS with the post-warmup baseline, and
`rss_limit_mb` records the configured gate (null means observation only). RSS includes the owned process and its descendants; on Spark it
is process RSS, not total unified GPU memory. A stable allocation plateau can
be expected from KV-cache pools. Cancellation-churn drain is a cohort metric;
it does not establish a per-request cancellation latency. A run must observe
actual cancellations or overload rejections to pass those scenarios. Record
shortened runs as smoke validation; use `soak_s: 3600` or longer for endurance.
These transport/lifecycle tests complement the model quality baselines in the
model-owned deployment cases.

## Online server benchmark CLI

The existing shared online driver is callable from pytest's
`test_server_benchmark` and directly from the command line:

```bash
python -m scripts.benchmark_runner \
  --url http://127.0.0.1:18080 --model stress-model \
  --workload /path/to/workload.json --output logs/online-bench/run-001
```

A minimal workload using the existing canonical online format is:

```json
{
  "run_id": "server-baseline",
  "concurrency": 2,
  "generation": {"temperature": 0, "top_p": 1, "top_k": 0, "ignore_eos": true},
  "warmup": {"iterations": 1, "scope": "canonical_batch"},
  "requests": [
    {"id": "a", "token_ids": [1000, 1001, 1002], "output_tokens": 32},
    {"id": "b", "token_ids": [1003, 1004, 1005], "output_tokens": 32}
  ]
}
```

The pytest benchmark uses identical token-ID prompts as a CLI smoke check;
its numbers provide no performance baseline, particularly with prefix caching.
Use tokenizer-produced IDs for a representative workload and tokens valid for
the selected model. The benchmark targets main's `/v1/completions` API, writes
`events.jsonl` and `summary.json`, excludes warmup from measured statistics, and
returns nonzero on a failed run. The existing C++ `build/bin/llm_bench` measures
the native runtime directly; this CLI measures HTTP serving overhead and load.
