# Evaluate with NeMo Evaluator

NeMo Evaluator can measure a supported checkpoint through the experimental OpenAI-compatible server. The server
builds or reuses its complete runtime bundle before evaluation.

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
tensorrt-edgellm-serve \
  Qwen/Qwen2.5-0.5B-Instruct \
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

For a one-command local run, pass a Hugging Face model ID or local checkpoint instead of a URL:

```bash
python scripts/run_nemo_eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --cache-dir /data/edgellm-cache \
  --eval-type mmlu \
  --limit-samples 250 \
  --output-dir nemo-results
```

The helper starts the server with the current Python environment, waits for `/health`, runs the evaluation, and stops
the server. Server output is written to `nemo-results/edgellm_server.log`.

Use the server build-profile options only with `--model`:

```bash
python scripts/run_nemo_eval.py \
  --model /path/to/checkpoint \
  --max-batch-size 2 \
  --max-input-len 2048 \
  --max-kv-cache-capacity 8192
```

Multimodal components are built from the checkpoint as part of the same bundle. Pass the same JSON accepted by the
server's `--speculative-config` option to evaluate a speculative-decoding configuration.

## Optional Named Cases

Command-line options are sufficient for local use. YAML cases are optional and provide reusable model settings for
CI or repeated runs. The repository cases are in `tests/nemo_eval/cases.yml` and use model-based names:

```bash
python scripts/run_nemo_eval.py \
  --case Qwen2.5-0.5B-Instruct \
  --model Qwen/Qwen2.5-0.5B-Instruct
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
