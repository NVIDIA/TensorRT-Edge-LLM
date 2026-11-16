# Evaluate with NeMo Evaluator

NeMo Evaluator can measure a supported checkpoint through the OpenAI-compatible server. The server
builds or reuses its complete runtime bundle before evaluation.

## Install Dependencies

Set up the [runtime](../server.md#install) from a published wheel or
source build first. Then install evaluator dependencies from the repository
without replacing the installed Edge-LLM package:

```bash
python -m pip install -r /path/to/TensorRT-Edge-LLM/requirements-nemo-evaluator.txt
```

## Evaluate a Running Server

Start the server in one terminal:

```bash
tensorrt-edgellm-serve \
  Qwen/Qwen3.5-0.8B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --port 8000
```

In another terminal, run NeMo Evaluator directly:

```bash
nemo-evaluator run_eval \
  --eval_type mmlu \
  --model_id Qwen/Qwen3.5-0.8B \
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

Use `--model-url` when the server listens elsewhere. When `--model-id` is not
set, the helper reads the served model ID from the server's `/v1/models`
endpoint.

## Start the Server Automatically

For a one-command local run, pass a Hugging Face model ID or local checkpoint instead of a URL:

```bash
python scripts/run_nemo_eval.py \
  --model Qwen/Qwen3.5-0.8B \
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
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960
```

Multimodal components are built from the checkpoint as part of the same bundle. Pass the same JSON accepted by the
server's `--speculative-config` option to evaluate a speculative-decoding configuration. `--eval-type` accepts any
evaluation registered by the installed NeMo Evaluator packages that supports a chat endpoint.

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

The helper currently supports NeMo Evaluator's `chat` model type because the server exposes
`/v1/chat/completions`.
