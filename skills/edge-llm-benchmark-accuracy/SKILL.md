---
name: edge-llm-benchmark-accuracy
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, benchmark, accuracy, nemo-evaluator, mmlu]
description: Measure dataset accuracy (MMLU and other NeMo Evaluator benchmarks) of a TensorRT Edge-LLM deployment through its OpenAI-compatible chat endpoint with NeMo Evaluator. Use for "benchmark accuracy", "run MMLU/GSM8K/IFEval on this checkpoint", comparing a quantization, speculative-decoding, or engine-profile change against a baseline score, or gating a change on a minimum score. Do not use to launch a server only (edge-llm-serve), to measure latency or throughput (edge-llm-benchmark-performance), or to locate which layer diverges after a mismatch (edge-llm-few-layer-test, edge-llm-accuracy-debugging).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark dataset accuracy with NeMo Evaluator

NeMo Evaluator scores a deployment through `POST /v1/chat/completions`, so the
number measures the whole served path: checkpoint, quantization, engine
profile, chat template, sampling, and server. It is a regression signal, not a
proof of numerical equivalence.

## Instructions

Take the benchmark, sample budget, and baseline from the user; then run Phases 1 to 5: prepare the evaluator, serve with a sized profile, run the evaluation, validate the run, and compare with the baseline. Report the score, sample count, both commands, and the results path.

## Read first

- `docs/source/user_guide/examples/integration/nemo-evaluator.md`
- `scripts/run_nemo_eval.py` (`--help` and the parser at its end)
- `requirements-nemo-evaluator.txt` (pinned evaluator packages)
- `tests/model_configs/l0/qwen3_5/qwen3_5_0_8b.yml`, case `nemo-evaluator`: the
  CI-verified recipe and its `score.min` gate

## Inputs

| Input | Required | Notes |
|---|---|---|
| Checkpoint | yes | Exact Hugging Face ID and revision, or absolute local path |
| Benchmark | yes | `--eval-type` registered by the installed evaluator packages; default to `mmlu` only if the user has no preference |
| Sample budget | yes | `limit_samples` for a smoke gate, omitted for a full run; state which |
| Baseline | for comparisons | Score from the reference deployment on the same benchmark, samples, and sampling |
| Gate | optional | `--min-score` from an agreed baseline; never invent one |

Stop and ask when the benchmark, sample budget, or baseline for a comparison is
missing. Do not substitute another benchmark.

## Phase 1: environment

1. Run `scripts/run_nemo_eval.py` with the Edge-LLM runtime interpreter (from
   `edge-llm-install`): with `--model` it starts the server with
   the current interpreter, which must be able to import `tensorrt_edgellm`; an
   evaluator-only venv cannot. Install `requirements-nemo-evaluator.txt` into that
   environment (the documented default), or keep the evaluator in a separate venv
   and pass `--nemo-evaluator-bin <venv>/bin/nemo-evaluator`; do not activate the
   evaluator venv for the helper.
2. If a run fails on an unknown option, check `nemo-evaluator run_eval --help`
   against the installed version and report the versions.
3. The helper has no API-key support. For a server started with `--api-key`, call
   `nemo-evaluator run_eval --api_key_name <ENV_VAR>` directly.
4. Confirm the benchmark is registered (`nemo-evaluator ls | grep -i <name>`;
   the full list spans several harnesses) and its dataset
   is reachable from this host (network or a pre-populated cache). Datasets
   load through Hugging Face `datasets`; a Python `CERTIFICATE_VERIFY_FAILED` or
   "Couldn't reach ... on the Hub" while `curl` works means the venv lacks the
   site CA bundle: export `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to the
   system bundle (for example `/etc/pki/tls/certs/ca-bundle.crt`); set them
   before the first run on such a host. An offline
   host needs the user's dataset location; do not substitute another dataset.
   The first run downloads the dataset (MMLU loads 57 subject sets), so allow
   minutes before the first request reaches the server.
5. NeMo Evaluator prints telemetry notices; set
   `NEMO_EVALUATOR_TELEMETRY_LEVEL=0` when the user's policy requires it, and
   say so in the report.

## Phase 2: serve

Launch with `edge-llm-serve`, sized for the benchmark rather than the default:

- `--max-input-len` must cover the longest prompt, including few-shot examples.
- `--max-kv-cache-capacity` must cover prompt plus `max_new_tokens`.
- MMLU answers step by step, so use a generous `max_new_tokens` (the CI recipe
  uses 2048) or answers are cut off before the final choice and scored wrong.
- Set the evaluator request timeout above the worst-case generation time (CI uses
  300 s); a timeout is retried and inflates the server's request count.
- Use `--parallelism 1` for a deterministic baseline. In-flight batching only
  works with a server you launch yourself (the helper has no passthrough for it).
- Add `--speculative-config`, a quantized checkpoint, or any profile change only
  when that is the variable under test.

`run_nemo_eval.py --model <checkpoint>` can start and stop the server itself
(see its `--help`; its profile defaults differ from the server's, and the served
name is the checkpoint's basename). Use it for a one-shot run; start the server
yourself (`edge-llm-serve`) to reuse one engine for several benchmarks.

## Phase 3: run

Run from the repository root with Python 3.10 or newer (the helper is a
repository script). A first MMLU run takes minutes (dataset download plus
scoring) and prints many progress lines, so run it in the background or with a
long timeout and redirect output to a file. Dataset caches follow `HF_HOME`.

Against a server that is already ready:

```bash
python scripts/run_nemo_eval.py \
  --model-url http://127.0.0.1:<port>/v1/chat/completions \
  --eval-type <benchmark> \
  --limit-samples <n> --parallelism 1 --max-new-tokens <n> \
  --temperature 0.0 --top-p 1.0 \
  --output-dir <task-dir>/nemo-results [--min-score <baseline-gate>]
```

The helper discovers the served model ID from `/v1/models`. The direct
equivalent is `nemo-evaluator run_eval --eval_type ... --model_id ...
--model_url ... --model_type chat --output_dir ... --overrides
"config.params.limit_samples=...,config.params.parallelism=1"`.

Use greedy sampling for regression comparisons. Keep benchmark, sample count,
seed, sampling, `max_new_tokens`, and chat template identical between runs
being compared; otherwise the difference is not attributable.

## Phase 4: validate the run

A score alone is not evidence. Before reporting:

1. The command exited 0 and printed `NeMo Evaluator score: <value>`; "Could not
   find score key" means the metric name differs. Inspect the JSON under
   `--output-dir` and pass the right `--score-key`; do not report a guess.
2. The server received one chat request per sample (count them in the server
   log, or the `/health` counters under in-flight batching). Fewer than
   `limit_samples` means failed or skipped requests. The printed summary has
   no sample count; the requested count is `limit_samples` in `results.yml`
   (the generated `--first_n`). If you cannot observe the server, say this
   check was not possible instead of inferring it.
3. No request errors, timeouts, or 4xx/5xx in the server log. A 413 means the
   prompt exceeded `--max-input-len`; rerun with a larger profile.
4. Inspect a few raw responses: the "Examples" section of
   `<output-dir>/<eval-type>/<eval-type>.html` lists prompts and model
   answers; `report.json` and `results.yml` hold the metrics and the exact
   command. Empty or truncated answers mean `max_new_tokens` is too small or a
   reasoning parser is cutting output.

A tiny `max_new_tokens` (for example 16 in a smoke run) truncates MMLU's
step-by-step answers and scores near zero by design; such a run proves the
plumbing only, so label it a smoke run and do not report the score as accuracy.

Small sample budgets are noisy. For a pass-rate score the standard error is
`sqrt(p * (1 - p) / n)`: about 3 points at `p = 0.6` and `n = 250`. Compute it
for the actual score and sample count, and do not call a difference within two
standard errors a regression; rerun with more samples or the full benchmark.

## Phase 5: compare

For a baseline comparison run both deployments back to back on the same device
with identical settings and report both scores, sample counts, and the delta
with its sampling noise. A drop larger than noise goes to
`edge-llm-few-layer-test` and then `edge-llm-accuracy-debugging`; preserve the
results directory and server log.

## Output format

Report the Edge-LLM revision, evaluator versions, checkpoint, the redacted server
command and the evaluator command, benchmark, samples evaluated, sampling and
`max_new_tokens`, the score with its sampling error, the baseline and delta when
compared, the results and server log paths, and whether the server was stopped.

## Limitations

- `run_nemo_eval.py` drives only the `chat` model type. Other evaluator model
  types (`completions`, `completions_logprob`) are not exercised by this
  workflow; do not assume log-likelihood benchmarks work.
- Scores from the quantized or speculative path differ from the reference
  precision by design; judge them against an agreed baseline.
- Do not compare scores across evaluator package versions.

## Troubleshooting

| Symptom | Cause and action |
|---|---|
| Server exits before health | Read `<output-dir>/edgellm_server.log` (`--model` mode) or your own log; use `edge-llm-serve` failure routing |
| `Score below threshold` | Real regression or noise; follow Phase 5 before reporting |
| Scores near zero | `max_new_tokens` too small or a timeout; raise it with the KV capacity |
| HTTP 413 in the server log | Prompt exceeds `--max-input-len`; relaunch with a larger profile |
| Dataset download fails | If `curl` reaches the Hub, set the CA variables (Phase 1 step 4); otherwise no network or cache, so ask the user for a dataset location |

## Examples

- "Run MMLU on this checkpoint and fail below 0.50": serve the checkpoint, run `scripts/run_nemo_eval.py` with `--eval-type mmlu` and `--min-score 0.50`, validate samples and responses, report the score with its sampling error.
- "Did the quantized checkpoint lose accuracy against FP16?": run both deployments back to back with identical settings and report both scores, the delta, and the noise.
