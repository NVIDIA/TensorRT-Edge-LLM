# Evaluate with NeMo Evaluator

NeMo Evaluator can measure the accuracy of a TensorRT Edge-LLM engine through the experimental OpenAI-compatible
server. Export and build the engine before starting this workflow.

## Install Dependencies

From the TensorRT Edge-LLM repository:

```bash
python -m pip install -e ".[server]"
pip install -r requirements-nemo-evaluator.txt
```

Make sure the built Python bindings, TensorRT libraries, and Edge-LLM plugin are available in the active environment.
See the [experimental server guide](experimental-server.md) for setup details.

## Evaluate a Running Server

Start the server in one terminal:

```bash
python -m experimental.server \
  --model /path/to/llm_engine \
  --port 8000
```

In another terminal, run NeMo Evaluator directly:

```bash
nemo-evaluator run_eval \
  --eval_type mmlu \
  --model_id my-model \
  --model_url http://127.0.0.1:8000/v1/chat/completions \
  --model_type chat \
  --output_dir nemo-results \
  --overrides "config.params.limit_samples=250,config.params.parallelism=1,config.params.max_new_tokens=6144"
```

The helper script provides the same workflow with shorter options and prints a concise metric summary. Its default
URL is `http://127.0.0.1:8000/v1/chat/completions`, so the common local command is:

```bash
python scripts/run_nemo_eval.py \
  --eval-type mmlu \
  --limit-samples 250 \
  --parallelism 1 \
  --max-new-tokens 6144 \
  --output-dir nemo-results
```

Use `--model-url` when the server listens elsewhere.

## Start the Server Automatically

For a one-command local run, pass a pre-built engine instead of a URL:

```bash
python scripts/run_nemo_eval.py \
  --engine-dir /path/to/llm_engine \
  --eval-type mmlu \
  --limit-samples 250 \
  --output-dir nemo-results
```

The helper starts the server with the current Python environment, waits for `/health`, runs the evaluation, and stops
the server. Server output is written to `nemo-results/edgellm_server.log`.

Use the server options only with `--engine-dir`. For example, concurrent evaluator requests can be micro-batched:

```bash
python scripts/run_nemo_eval.py \
  --engine-dir /path/to/llm_engine \
  --max-batch-size 2 \
  --enable-batching \
  --max-queue-batch-size 2 \
  --parallelism 2
```

Visual and speculative-decoding engines can be supplied with `--visual-engine-dir` and
`--spec-decode-engine-dir`, respectively.

## Optional Named Cases

Command-line options are sufficient for local use. YAML cases are optional and provide reusable model settings for
CI or repeated runs. The repository cases are in `tests/nemo_eval/cases.yml` and use model-based names:

```bash
python scripts/run_nemo_eval.py \
  --case Qwen2.5-0.5B-Instruct \
  --engine-dir /path/to/llm_engine
```

Pass `--config /path/to/cases.yml` with `--case` to use a different case file. Values from the selected case override
the command-line defaults.

## Scores and Overrides

Results are written under `--output-dir`. Use a threshold to fail the command on a clear accuracy regression:

```bash
python scripts/run_nemo_eval.py \
  --min-score 0.50 \
  --score-key score
```

Additional NeMo Evaluator settings can be supplied as comma-separated overrides:

```bash
python scripts/run_nemo_eval.py \
  --extra-overrides "config.params.limit_samples=100,config.params.temperature=0.0"
```

The helper currently supports NeMo Evaluator's `chat` model type because the experimental server exposes
`/v1/chat/completions`.
