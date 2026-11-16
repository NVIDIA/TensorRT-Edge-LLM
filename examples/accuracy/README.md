# Accuracy Benchmark for TensorRT Edge LLM

This directory contains tools for running accuracy benchmarks on TensorRT Edge LLM models.

Supported benchmark types:
- **Accuracy Tests** — Multiple choice correctness (MMLU, MMMU, MMBench, MMStar, OmniBench)
- **ROUGE Similarity Tests** — Text generation quality (GSM8K, HumanEval, MTBench)
- **Audio Score Tests** — Speech generation WER + speaker similarity (Seed-TTS-eval, MiniMax)

## Prerequisites

Install only the profile for the work being run:

```bash
# Score existing inference outputs
pip install -r requirements-runtime.txt

# Download or convert benchmark datasets
pip install -r requirements-datasets.txt

# Prepare PhysicalAI-AV requests for Alpamayo-R1
pip install -r requirements-physical-ai-av.txt

# Add multilingual TTS scoring support
pip install -r requirements-runtime.txt -r requirements-tts.txt
```

The profiles are intentionally scoped:

- `requirements-runtime.txt` provides inference metrics without PyTorch or TTS models.
- `requirements-datasets.txt` provides dataset download and conversion tools.
- `requirements-physical-ai-av.txt` provides the PhysicalAI-AV dataset kit for Alpamayo-R1 requests.
- `requirements-tts.txt` is the optional TTS add-on. English speaker scoring
  also uses the PyTorch and TorchAudio versions supplied by the chosen NVIDIA
  framework container.

For ROUGE tests, also install vLLM:
```bash
pip install vllm
```

## Building TensorRT Engines

Build engines under the tensorrt_edgellm root directory. Use large sequence lengths (8192-10240) as accuracy datasets can be long.

### Text-only Models:
```bash
./build/bin/llm_build \
  --onnx-dir /path/to/text/onnx/model/ \
  --engine-dir /path/to/text/engine/ \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240 \
  --max-batch-size 1
```

### Multimodal Models (MMMU, MMMU_Pro):
Build both visual encoder and text engines:

**Visual Encoder:**
```bash
./build/bin/visual_build \
  --onnx-dir /path/to/visual/onnx/model/ \
  --engine-dir /path/to/visual/engine/ \
  --min-image-tokens 256 \
  --max-image-tokens 8192
```

**Text Engine:**
```bash
./build/bin/llm_build \
  --onnx-dir /path/to/text/onnx/model/ \
  --engine-dir /path/to/text/engine/ \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240 \
  --max-batch-size 1
```

## Benchmark Types

### 1. Accuracy Tests

Evaluate model predictions against ground truth answers. Supported datasets: **MMLU**, **MMLU_Pro**, **MMMU**, **MMMU_Pro**.

**Workflow:**
```bash
# 1. Convert dataset
python3 scripts/prepare_dataset.py \
  --dataset MMLU \
  --output_dir /path/to/datasets/mmlu_output

# 2. Run inference (under tensorrt_edgellm root dir)
./build/bin/llm_inference \
  --engine-dir /path/to/engines/text_engine/ \
  --multimodal-engine-dir /path/to/multimodal/engine/ \  # For multimodal models
  --input-file /path/to/datasets/mmlu_output/mmlu_dataset.json \
  --output-file /path/to/outputs/mmlu_predictions.json

# 3. Calculate accuracy
python3 scripts/calculate_correctness.py \
  --predictions_file /path/to/outputs/mmlu_predictions.json \
  --answers_file /path/to/datasets/mmlu_output/mmlu_dataset.json
```

**DART visual-token pruning comparison** (`run_dart_accuracy.py`): [visual-token pruning](../../docs/source/user_guide/features/visual-token-pruning.md)
removes redundant visual tokens before VLM prefill (`llm_inference --visual-prune`), trading a
small, ratio-dependent accuracy cost for prefill speedup. This script quantifies the accuracy
side of that trade-off: it runs a prepared multiple-choice VLM dataset (MMStar / MMMU) through
`llm_inference` once without pruning and once per `--ratios` value, scores every run, and prints
a per-dataset markdown table — score (Δ vs baseline), average text/visual token counts, ISL, and
the *estimated* prefill speedup (the linear token-count ratio `baseline_ISL / pruned_ISL`).
Pass several `--dataset_files` for one table row per dataset. It intentionally reports no timing
— the sequential single-pass runs are not a fair benchmark; measure real latency with
`llm_inference --dump-profile --warmup N`:

```bash
python3 scripts/prepare_dataset.py --dataset MMStar --output_dir tmp/mmstar_output

python3 scripts/run_dart_accuracy.py \
  --engine_dir /path/to/engines/llm \
  --multimodal_engine_dir /path/to/engines/visual \
  --dataset_file tmp/mmstar_output/mmstar_dataset.json \
  --ratios 0.25 0.5 \
  --output_dir tmp/dart_accuracy
```

### 2. ROUGE Similarity Tests

Evaluate text generation quality using ROUGE metrics against reference responses.

**Workflow:**
```bash
# 1. Generate references with vLLM
python3 scripts/generate_reference.py \
  --model <model_name_or_path> \
  --input_file /path/to/dataset.json \
  --output_file /path/to/references.json

# 2. Run inference (under tensorrt_edgellm root dir)
./build/bin/llm_inference \
  --engine-dir /path/to/engine/ \
  --input-file /path/to/dataset.json \
  --output-file /path/to/predictions.json

# 3. Calculate ROUGE scores
python scripts/calculate_rouge_score.py \
  --predictions_file /path/to/predictions.json \
  --references_file /path/to/references.json
```

## Available Datasets

Located in `example_datasets/` directory. Use `scripts/prepare_dataset.py` to convert datasets to Edge LLM format.

### Text-Only Datasets

**Multiple Choice & Knowledge:**
- **MMLU** (`mmlu.py`): Massive Multitask Language Understanding
- **MMLU_Pro** (`mmlu_pro.py`): Enhanced MMLU with more challenging questions

**Math & Reasoning:**
- **GSM8K** (`gsm8k.py`): Grade School Math 8K
- **AIME** (`aime.py`): American Invitational Mathematics Examination 2024
- **MATH-500** (`math500.py`): Mathematical reasoning problems
- **HumanEval** (`humaneval.py`): Code completion benchmark

**Conversational:**
- **MTBench** (`mtbench.py`): Multi-Turn Benchmark

**Performance:**
- **SPEED-Bench** (`speed_bench.py`): Speculative-decoding throughput prompts
  in nominal 1K, 2K, 8K, 16K, and 32K input-length buckets. Download and
  convert every bucket with:

```bash
python3 scripts/prepare_dataset.py \
  --dataset SPEEDBench \
  --subset all \
  --output_dir /path/to/datasets/speed_bench
```

Rows masked by the dataset provider are skipped.

### Multimodal Datasets

**Driving (Alpamayo-R1):**
- **PhysicalAIAV** (`alpamayo_driving.py`): 4-camera, 4-timestep requests with
  egomotion history from [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
  Requires the dataset license and `hf auth login`. Only the selected clips
  are streamed; `--max_samples` sets the clip count (default 1):

```bash
python3 scripts/prepare_dataset.py \
  --dataset PhysicalAIAV \
  --max_samples 1 \
  --output_dir /path/to/datasets/physical_ai_av
```

**Vision + Language:**
- **MMBench** (`mmbench.py`): Multimodal benchmark, EN dev split (letter-match accuracy; the
  official metric adds circular evaluation + GPT answer matching, so expect small differences)
- **MMMU** (`mmmu.py`): Massive Multi-discipline Multimodal Understanding
- **MMMU_Pro** (`mmmu.py`): Enhanced version of MMMU
- **MMStar** (`mmstar.py`): Multimodal benchmark with visual reasoning

**Audio TTS:**
- **SeedTTSEval** (`tts_eval.py`): [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) benchmark (test-zh, test-en, test-hard)
- **MiniMaxMultilingual** (`tts_eval.py`): [MiniMax TTS Multilingual Test Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set) (24 languages)
- **OmniBench** (`omnibench.py`): [OmniBench](https://huggingface.co/datasets/m-a-p/OmniBench) audio+image+text multimodal understanding (1142 samples)

**Audio ASR:**
- **LibriSpeech** (`librispeech.py`): Full test-clean split (2620 utterances)
- **LibriSpeech_Lite** (`librispeech.py`): Deterministic per-speaker JSON subset of an existing Full dataset (524 utterances by default); audio files stay in Full

### Framework
- **EdgeLLM Dataset** (`edgellm_dataset.py`): Base class for custom dataset implementations

## Scripts

- **`prepare_dataset.py`**: Convert datasets to Edge LLM format
- **`calculate_correctness.py`**: Calculate accuracy scores for multiple choice datasets
- **`generate_reference.py`**: Generate reference responses using vLLM
- **`calculate_rouge_score.py`**: Calculate ROUGE similarity scores
- **`calculate_tts_score.py`**: Calculate WER and speaker similarity (Seed-TTS-eval methodology)

## Output Metrics

- **Accuracy Tests**: Overall accuracy percentage, subject-specific accuracy, detailed statistics
- **ROUGE Tests**: Rouge-1, Rouge-2, Rouge-L, Rouge-Lsum scores
- **Audio WER**: English via Whisper-large-v3, Chinese via Paraformer-zh (Seed-TTS-eval standard)
- **Audio SIM**: Speaker similarity via WavLM-large embeddings, optionally loading a finetuned checkpoint with `--wavlm_checkpoint`
- **OmniBench**: Multimodal accuracy across audio+image+text understanding

## Important Notes

- **Engine Building**: Use `llm_build` for text engines, `visual_build` for visual encoders
- **Sequence Lengths**: Use large values (8192-10240) for accuracy datasets
- **Multimodal Models**: Build both LLM and visual engines, use `--multimodal-engine-dir` parameter for inference
- **Dataset Format**: All inputs must be in Edge LLM JSON format
- **Binary Location**: Run inference binaries from tensorrt_edgellm root directory

## Appendix: Reproducing HuggingFace MMMU Benchmark with VLMEvalKit

Follow [Score MMMU Offline](../../docs/source/user_guide/examples/integration/vlmevalkit.md#score-mmmu-offline).
