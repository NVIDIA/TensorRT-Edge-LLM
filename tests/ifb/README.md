<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# IFB trace tools

Trace-driven end-to-end checks for the in-flight-batching `RequestEngine` (nanobind
`_edgellm_runtime.RequestEngine`). The traces live in `tests/test_cases/ifb_*.json`. Arrivals are
defined by token counts ("submit when request X has generated k tokens"), which keeps the request
mix reproducible across runs and builds.

| File | Purpose |
|---|---|
| `run_ifb_trace.py` | Drives one trace on one engine and writes a result JSON (modes `run`, `reference`) |
| `check_ifb_correctness.py` | Compares a run with the batch-1 reference; exit code 1 on failure |
| `ifb_test.sh` | `quick` / `full` tiers: reference + run + check for each engine and trace |

## Modes

| Mode | Engine | Requests |
|---|---|---|
| `run` | Trace `engine` options, plus the `--scheduling-mode` override; one fresh engine per variant | The trace as given |
| `reference` | `max_batch_size=1`, `ifb_scheduling_mode=singleCohort` | Each request without a cancel rule, alone and in trace order |

Every request decodes greedily (`temperature=1`, `top_k=1`, `top_p=1`) with `num_logprobs` top-k
logprobs per token (default and maximum 20, minimum 2). `num_logprobs` is a cohort field in
`BatchCompatibility`: requests that differ in it wait for an empty session. `max_generate_length`
is per request.

The reference uses `singleCohort` on the build under test because `prefillFirst` and
`max_batch_size=1` still run compact steps on a deployment that supports them, while `singleCohort`
runs single-cohort steps on any deployment and reproduces a build without compact execution bit for
bit.

## Trace schema

```json
{
  "name": "example_trace",
  "description": "free text",
  "num_logprobs": 20,
  "expect_policy": "mixed",
  "engine": {"max_batch_size": 4, "ifb_scheduling_mode": "auto", "max_chunk_tokens": 512,
             "scheduling_policy": "fcfs", "preemption": "none", "priority_reserved_slots": 0,
             "max_pause_steps": 0},
  "variants": [{"name": "pause", "engine": {"preemption": "pause"}, "expect_policy": "mixed",
                "requests_exclude": ["p3"]}],
  "request_defaults": {"max_generate_length": 64, "enable_thinking": false},
  "snippets": {"rivers": "A paragraph ..."},
  "requests": [
    {"id": "r0", "messages": [{"role": "user", "content": "Question ..."}], "arrival": "at_start"},
    {"id": "r1", "messages": [{"role": "user", "content": [{"snippet": "rivers"}, "Summarize."]}],
     "arrival": {"after": {"request": "r0", "tokens": 4}},
     "expect": {"prompt_tokens": {"min": 513, "max": 1000, "not_multiple_of": 128}}},
    {"id": "r2", "prompt": "Raw text without a chat template", "priority": -1,
     "arrival": {"after": {"request": "r0", "event": "finish"}},
     "cancel_after": {"request": "self", "tokens": 0}}
  ]
}
```

- `engine`: `RequestEngine` keyword arguments. `max_batch_size` is clamped to the engine's build-time
  value. `context_cache_config` is an object whose fields are set on a `ContextCacheConfig`, in every
  mode including the reference (see [Context-cache pools](#context-cache-pools)).
- `variants` (run mode): each variant overrides `engine` and optionally `expect_policy`, and may drop
  requests with `requests_exclude`; a trigger on a dropped request fires immediately.
  `expect_metrics` (`{metric: {"min": a, "max": b}}`, also allowed at trace level) bounds the run's
  `EngineMetrics` delta, for example `{"paused_resident_steps": {"min": 1}}`.
  `request_overrides` (`{id: {field: value}}`) replaces request fields for that variant and merges an
  `expect` mapping into the request's.
- Request fields: `id`; exactly one of `messages` (chat template applied) or `prompt` (raw text);
  `max_generate_length`, `priority`, `ignore_eos`, `enable_thinking`, `apply_chat_template`,
  `add_generation_prompt`, `skip_special_tokens` (any of them may come from `request_defaults`).
- `content` / `prompt`: a string or a list of segments; a segment is a string or
  `{"snippet": name, "repeat": n}`; segments are joined with a blank line. In `messages`, a segment
  may also be `{"image": "path"}` (relative to the repository root); the engine then needs
  `--multimodal-engine-dir`.
- `arrival`: `"at_start"`, `{"after": {"request": id, "tokens": k}}`, or
  `{"after": {"request": id, "event": "finish"}}`. A token trigger also fires when the source request
  terminates with fewer than k tokens.
- `cancel_after`: `{"request": "self" | id, "tokens": k}`; `tokens: 0` on `self` cancels right
  after submission (queued or in prefill). Requests with a cancel rule are excluded from the
  reference and must end `CANCELLED`.
- `expect`:
  - `status`: `finished` or `cancelled` (default derived from `cancel_after`).
  - `prompt_tokens`: hints (`min`, `max`, `not_multiple_of`) that confirm the trace still exercises
    the intended chunk boundaries on a given tokenizer; violations are warnings.
  - `reuse_tokens`: `min` / `max` prompt tokens the context cache must serve at a finished request's
    admission (`RequestHandle.stats().reused_prompt_tokens`). `hybrid` holds bounds that replace these
    on an engine with recurrent (linear-attention) layers, whose cache hits only at published snapshot
    lengths. `if_prompt_cancelled: id` applies the bounds only when request `id` was cancelled before
    its prompt completed (its stats show no first token); `ifb_context_cache` uses it with `max: 0`
    to state that a prompt cancelled before its final chunk publishes nothing. Violations are the
    run's `cache_hint_issues`.
  - `joins_running`: requests the request must start beside: its first token must arrive before each
    of them finishes (the run's `admission_issues`); a listed request a variant dropped is skipped.
  - `stats` (`{stat: {"min": a, "max": b}}`) bounds the request's `RequestHandle.stats()`, and
    `admitted_before` lists requests the engine must admit it strictly before (admission time =
    submission time + `queue_time_us`; a listed request never admitted counts as later). Violations of
    these and of `expect_metrics` are the run's `stats_issues`, evaluated only when the resolved
    policy matches `expect_policy`.
- `expect_policy`: compared with `RequestEngine.resolved_policy.mode` (`mixed` / `prefillFirst` /
  `singleCohort`). `--expect-policy` overrides it. A mismatch fails the run when the mode was
  requested, and is a warning when `auto` chose it (`resolved_policy.mode_source`), since `auto`
  resolves per deployment.

Token-count triggers fire as soon as the driver observes the count; the engine keeps stepping while
the driver polls, so the exact step a submission lands in is best effort. Each request records its
trigger lag (`trigger_lag_tokens`: the source's tokens past the trigger's count when it fired). When
the largest lag exceeds `--max-trigger-lag` (default 8), the run is `trigger_lag.degraded`: the
timing-dependent expectations (`joins_running`, `expect_metrics`, `stats`, `admitted_before`) are not
evaluated, and the checker reports their criteria as `SKIP` with the lag.

Each run also records the `EngineMetrics` delta and prints its step mix (`decode_steps`,
`prompt_steps`, `mixed_steps`, `decode_subset_steps`), the CUDA-graph hit rate of full decode steps,
average step sizes, host step time, and the per-request engine stats.

## Correctness criteria

For every request present in the reference:

1. Greedy tokens identical, or the first divergence is at a position where the reference top-2
   logprob gap is below 0.1 (logprob gap equals logit gap) and the run's token is one of the
   reference's top two there. Comparison stops at the first divergence. A request that stopped at a
   variant's shorter `max_generate_length` is compared over the tokens it generated.
2. First-token centered cosine of at least 0.999. Full logits are not exposed by the runtime, so the
   checker computes the cosine over the top-k logprob vectors of the first generated token, aligned
   on the union of token ids with missing entries filled by that vector's minimum logprob, and
   mean-removed (centered top-k logits). This is an approximation. Centering matters: the uncentered
   logprobs share a log-sum-exp offset that makes their cosine about ten times less sensitive to a
   perturbed logit (1 - cosine is Delta^2 / |v|^2 rather than Delta^2 / sum((v - mean)^2)); it is
   printed for information only, with the largest absolute first-token logprob difference.
3. The run's drift (diverging requests, earliest divergence, and per request the largest |logprob
   difference| up to its first divergence) is printed as "drift (informational, see README)"; it does
   not decide `RESULT`.
4. With CUDA graph capture enabled (`capture_cuda_graph`, on by default), pure-decode steps over
   every resident must replay a graph at least as often as in the reference, measured by
   `decode_graph_steps / decode_steps` of the `EngineMetrics` delta. The reference runs one request
   at a time, so its rate is a lower bound rather than a value to match. The criterion is skipped
   when capture is disabled or the run or the reference has no pure-decode full step.
5. Every finished request's context-cache reuse lies within its `expect.reuse_tokens` bounds
   (`cache_hint_issues`).
6. Every request with `expect.joins_running` produced its first token before each listed request
   finished (`admission_issues`).
7. The engine's step metrics and each request's engine stats meet the trace's `expect_metrics`,
   `expect.stats` and `expect.admitted_before` (`stats_issues`). `ifb_priority` uses them to require,
   under `pause`, that residents are paused while the urgent request prefills and that the urgent
   request is never paused and runs its prompt in one step, and, with the reserved slot, that it is
   admitted before the low-priority request that arrived first. Wall-clock times are not gated.

Criteria 6 and 7 are `SKIP` when the driver's trigger lag degraded the run.

The checker also fails on requests that did not reach their expected status, on a run or reference
engine that did not start, on a run that compared no request with the reference, and on a requested
mode that resolved to another one. After the trace, the driver's run mode submits one prompt longer
than the engine's `max_input_len` (`failure_kind_probe`); the checker fails unless `get()` raised
`RequestFailedError` with `kind` `RequestFailureKind.OTHER`, which covers the binding that attaches
the request's failure kind. `ifb_test.sh` also fails a trace whose reference run did not complete. It prints each criterion as
`[PASS]`, `[FAIL]` or `[SKIP]` with the numbers behind it and stores the same under `criteria` in
`--json-out`; `RESULT` fails when any gating criterion fails.
`tests/python-unittests/test_ifb_correctness.py` covers these decisions on synthetic results.

Criterion 3 is informational because compact steps are not bit-exact with single-cohort execution:
the decode rows of a mixed step run on the engine's prefill optimization profile, where TensorRT
picks other kernels for the GEMMs outside the plugins than on the decode profile (the plugins'
decode rows are bit-exact with a decode-only step), and the packed-varlen and padded GDN prefill kernels accumulate in different orders. Both
are measured to have equal accuracy against an FP32 reference, FP8 models amplify them through the
downstream quantization, and rebuilding the same ONNX alone moves the logprobs by 0.08–0.09.
Criterion 1 still requires every divergence to be a near tie between the reference's top two tokens.

## Traces

| Trace | Coverage |
|---|---|
| `ifb_smoke.json` | One-step prefill with nothing decoding; a 725-token prompt arriving during decode (512-token chunk plus a non-page-aligned final chunk under `mixed`); cancel right after submission; cancel during decode; 8 requests over time on B = 4 (slot reuse); pure-decode tail |
| `ifb_priority.json` | Low-priority requests decoding and prefilling when a `priority: -1` request arrives; variants `pause` (free slot), `free_slot_other_length` (preemption `none`, free slot) and `reserved_other_length` (`priority_reserved_slots: 1` with every non-reserved slot busy); in the last two the urgent request asks for another `max_generate_length` and must start beside the running requests. `pause` requires paused residents and an urgent prompt that runs unpaused in one step; `reserved_other_length` requires the urgent request to be admitted before the earlier low-priority prompt |
| `ifb_context_cache.json` | Context cache enabled: a shared system prompt (partial hit while the first request decodes, full-page hit when it repeats), a chat-template conversation, a raw-ChatML conversation whose history exceeds `max_input_len` (only the uncached tail runs, chunked under `mixed`), a prompt cancelled between chunks that must not publish, and its repeat |
| `ifb_multimodal.json` | Image and text requests beside text-only ones: an image after about 450 text tokens so a 512-token chunk boundary falls inside it under `mixed`, a two-image request, and repeated images that hit the encoder embedding cache (VLM engines only) |

Prompts target engines with `max_input_len = 1024` and `max_kv_cache_capacity = 4096`. Each run
records every prompt's length and warns when it breaks the request's `prompt_tokens` hints
(`prompt_hint_issues`), which confirms the chunk boundaries on a new tokenizer.

### Context-cache pools

On a hybrid model, a context-cache hit resumes from a recurrent snapshot published at the cached
prompt length. The cache keeps `max_snapshots` of them and evicts the oldest first.
`ifb_context_cache.json` keeps 25. With 6, the later conversation turns find no snapshot, recompute
their whole history and fail with `EDGELLM_INPUT_TOO_LONG` in the reference and the run alike.

## Running

CI runs `ifb_smoke`, `ifb_priority` and `ifb_context_cache` in `mixed` mode, each as an `ifb_trace`
scenario of the Qwen3-0.6B and Qwen3.5-0.8B FP16 cases under `tests/model_configs/l0/`, on
`l0_e2e_b100` (see `tests/model_configs/README.md`). Run the other modes, `ifb_multimodal` and the
full tier by hand when changing the IFB runtime or scheduler. The scripts need a build with the
nanobind module (`_edgellm_runtime*.so`) and the plugin library, and a GPU that can run the engine:

```bash
export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib:<build dir>/lib:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_LIB=<build dir>/lib/libNvInfer_edgellm_plugin.so
export EDGELLM_PLUGIN_PATH=<build dir>/lib/libNvInfer_edgellm_plugin.so  # read by the runtime's plugin loader
export LLM_SDK_DIR=<repository root>

# Quick tier: ifb_smoke.json, reference + run + check
bash tests/ifb/ifb_test.sh --tier quick --mode mixed \
  --engine <engine dir> [--engine <engine dir> ...] --out <out dir>
# Full tier: smoke, priority, context cache, and multimodal on an engine directory with llm/ and visual/
bash tests/ifb/ifb_test.sh --tier full --mode prefillFirst \
  --engine <engine dir> [--engine <engine dir> ...] --out <out dir>
```

`ifb_test.sh` takes `--nanobind-dir` from `$EDGELLM_NANOBIND_DIR`, else the directory of
`EDGELLM_PLUGIN_LIB`, else `<repository root>/build/lib`. Single steps:

```bash
python3 tests/ifb/run_ifb_trace.py --trace tests/test_cases/ifb_smoke.json --engine-dir <engine dir> \
  --mode reference --nanobind-dir <build dir>/lib --out ref.json
python3 tests/ifb/run_ifb_trace.py --trace tests/test_cases/ifb_smoke.json --engine-dir <engine dir> \
  --mode run --scheduling-mode mixed --nanobind-dir <build dir>/lib --out run.json
python3 tests/ifb/check_ifb_correctness.py --run run.json --reference ref.json
```
