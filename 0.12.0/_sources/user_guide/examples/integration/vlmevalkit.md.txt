<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluate with VLMEvalKit

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) can score the
OpenAI-compatible server on any of its image, video, and text benchmarks. In
API mode it downloads each dataset, sends every sample to
`/v1/chat/completions`, and scores the answers. No conversion scripts are
needed.

## Serve the Model

Use the [recommended sequence length](index.md). The image-token flags are the
recommended image budget for this model. Raise a limit when a prompt or image
does not fit:

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --max-image-tokens 3920 \
  --max-image-tokens-per-image 3920 \
  --served-model-name qwen35-local \
  --port 8000
```

## Install VLMEvalKit

API mode needs VLMEvalKit main from April 2026 or later
([open-compass/VLMEvalKit#1499](https://github.com/open-compass/VLMEvalKit/pull/1499)).
This guide was tested at commit
[`54a063c`](https://github.com/open-compass/VLMEvalKit/commit/54a063c).

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -r requirements.txt rouge-score
pip install --no-deps -e .
```

## Run Benchmarks

Pass one or more dataset names to `--data`:

```bash
python run.py --api-mode \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen35-local \
  --key local \
  --data MMStar HallusionBench OCRBench \
  --judge qwen35-local \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --judge-key local \
  --api-nproc 4 \
  --max-tokens 1024 \
  --temperature 0 \
  --work-dir outputs
```

- `--model` must match `--served-model-name`. `--key` is sent as the bearer
  token. Any value works unless the server sets `--api-key`.
- The `--judge*` options reuse the same server to read the chosen option out
  of free-form answers. Any OpenAI-compatible model works as the judge.
- Keep `--max-tokens` below the server's KV-cache capacity. VLMEvalKit's
  default is 32768.
- Keep `--api-nproc` at or below the number of requests the server can handle
  at once.
- Do not pass `presence_penalty` or `frequency_penalty` through
  `--extra-body`. The server accepts only `0`.

Each dataset is downloaded on first use to `$LMUData` (default `~/LMUData`),
with an MD5 check. Video datasets are downloaded to the Hugging Face cache.
Scores are written to `outputs/qwen35-local/<run id>/`. Rerun with `--reuse`
to resume an interrupted run, or with `--mode eval --reuse` to rescore
existing predictions.

## Choose Datasets

List every registered dataset name with:

```bash
vlmutil dlist all
```

The dataset type sets how answers are scored:

| Type | Examples | Scoring |
|---|---|---|
| Multiple choice | `MMStar`, `MMBench_DEV_EN`, `AI2D_TEST`, `SEEDBench_IMG`, `RealWorldQA`, `MMMU_DEV_VAL` | Rule match first, then the judge |
| Yes/no | `HallusionBench`, `POPE`, `MME` | Rule match first, then the judge |
| Rule-scored VQA | `OCRBench`, `TextVQA_VAL`, `DocVQA_VAL`, `ChartQA_TEST` | Built-in rules; no judge |
| Judge-scored | `MathVista_MINI`, `MMVet`, `LLaVABench`, most video open-ended sets | LLM judge |

Without `--judge`, multiple-choice and yes/no datasets use `gpt-4o-mini`. If
that is unreachable, they fall back to rule matching only, which is the same
as `--judge exact_matching`. Rule matching counts an answer as wrong when it
cannot find a single option in the reply. This undercounts models that
answer in prose.

Video datasets use frame-count or frame-rate variants, such as
`Video-MME_16frame` and `MVBench_1fps`. VLMEvalKit sends the sampled frames as
images, so size `--max-image-tokens` for all frames in one request.

To run your own data, save it as `~/LMUData/<name>.tsv` with `index`,
`question`, and `image` (base64) columns, then pass `--data <name>`. Files with
`A`, `B`, ... and `answer` columns are scored as multiple choice. Other files
produce predictions only.

## Score MMMU Offline

Without a server, VLMEvalKit can rescore MMMU predictions from `llm_inference`.
Run the dataset commands from `examples/accuracy`.

Qwen3-VL's VLMEvalKit setup uses 980–3920 tokens per image and up to 5 images
per MMMU input, so build with `maxImageTokens` = 3920 × 5 = 19600:

```bash
python3 scripts/prepare_dataset.py \
  --dataset MMMU_VLMEvalkit \
  --output_dir /path/to/datasets/mmmu_vlmevalkit \
  --top_p 0.8 --top_k 20 --temperature 0.7

./build/bin/visual_build \
  --onnxDir /path/to/visual/onnx/model/ \
  --engineDir /path/to/visual/engine/ \
  --minImageTokens 980 \
  --maxImageTokens 19600 \
  --maxImageTokensPerImage 3920

./build/bin/llm_build \
  --onnxDir /path/to/text/onnx/model/ \
  --engineDir /path/to/text/engine/ \
  --maxInputLen 21000 \
  --maxKVCacheCapacity 30000 \
  --maxBatchSize 1 \
  --vlm \
  --minImageTokens 980 \
  --maxImageTokens 19600

./build/bin/llm_inference \
  --engineDir /path/to/text/engine/ \
  --multimodalEngineDir /path/to/visual/engine/ \
  --inputFile /path/to/datasets/mmmu_vlmevalkit/mmmu_dataset.json \
  --outputFile /path/to/outputs/mmmu_predictions.json \
  --batchSize 1 \
  --maxGenerateLength 8192
```

Convert the predictions to VLMEvalKit's format and score them:

```bash
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv

python3 scripts/prepare_mmmu_vlmevalkit.py \
  --tsv_file MMMU_DEV_VAL.tsv \
  --json_file /path/to/outputs/mmmu_predictions.json \
  --output_file /path/to/outputs/${Model}_MMMU_DEV_VAL.xlsx

cd /path/to/VLMEvalKit
python3 run.py --data MMMU_DEV_VAL --model ${Model} \
  --work-dir /path/to/outputs --mode eval --reuse
```
