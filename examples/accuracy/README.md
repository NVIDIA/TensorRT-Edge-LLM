# Accuracy Benchmark for TensorRT Edge LLM

This directory contains tools for running accuracy benchmarks on TensorRT Edge LLM models. Two types of benchmarks are supported: **Accuracy Tests** and **ROUGE Similarity Tests**.

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

For ROUGE tests, also install vLLM:
```bash
pip install vllm
```

## Building TensorRT Engines

Build engines under the tensorrt_edgellm root directory. Use large sequence lengths (8192-10240) as accuracy datasets can be long.

### Text-only Models:
```bash
./build/examples/llm/llm_build \
  --onnxDir /path/to/text/onnx/model/ \
  --engineDir /path/to/text/engine/ \
  --maxInputLen 8192 \
  --maxKVCacheCapacity 10240 \
  --maxBatchSize 1
```

### Multimodal Models (MMMU, MMMU_Pro):
Build both visual encoder and text engines:

**Visual Encoder:**
```bash
./build/examples/multimodal/visual_build \
  --onnxDir /path/to/visual/onnx/model/ \
  --engineDir /path/to/visual/engine/ \
  --minImageTokens 256 \
  --maxImageTokens 8192
```

**Text Engine:**
```bash
./build/examples/llm/llm_build \
  --onnxDir /path/to/text/onnx/model/ \
  --engineDir /path/to/text/engine/ \
  --maxInputLen 8192 \
  --maxKVCacheCapacity 10240 \
  --maxBatchSize 1 \
  --vlm \
  --minImageTokens 256 \
  --maxImageTokens 8192
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
./build/examples/llm/llm_inference \
  --engineDir /path/to/engines/text_engine/ \
  --multimodalEngineDir /path/to/multimodal/engine/ \  # For multimodal models
  --inputFile /path/to/datasets/mmlu_output/mmlu_dataset.json \
  --outputFile /path/to/outputs/mmlu_predictions.json

# 3. Calculate accuracy
python3 scripts/calculate_correctness.py \
  --predictions_file /path/to/outputs/mmlu_predictions.json \
  --answers_file /path/to/datasets/mmlu_output/mmlu_dataset.json
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
./build/examples/llm/llm_inference \
  --engineDir /path/to/engine/ \
  --inputFile /path/to/dataset.json \
  --outputFile /path/to/predictions.json

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

### Multimodal Datasets

**Vision + Language:**
- **MMMU** (`mmmu.py`): Massive Multi-discipline Multimodal Understanding
- **MMMU_Pro** (`mmmu.py`): Enhanced version of MMMU
- **MMStar** (`mmstar.py`): Multimodal benchmark with visual reasoning

### Framework
- **EdgeLLM Dataset** (`edgellm_dataset.py`): Base class for custom dataset implementations

## Scripts

- **`prepare_dataset.py`**: Convert datasets to Edge LLM format
- **`calculate_correctness.py`**: Calculate accuracy scores for multiple choice datasets
- **`generate_reference.py`**: Generate reference responses using vLLM
- **`calculate_rouge_score.py`**: Calculate ROUGE similarity scores

## Output Metrics

- **Accuracy Tests**: Overall accuracy percentage, subject-specific accuracy, detailed statistics
- **ROUGE Tests**: Rouge-1, Rouge-2, Rouge-L, Rouge-Lsum scores

## Important Notes

- **Engine Building**: Use `llm_build` for text engines, `visual_build` for visual encoders
- **Sequence Lengths**: Use large values (8192-10240) for accuracy datasets
- **Multimodal Models**: Build both engines, use `--vlm` flag and `--multimodalEngineDir` parameter
- **Dataset Format**: All inputs must be in Edge LLM JSON format
- **Binary Location**: Run inference binaries from tensorrt_edgellm root directory

## Appendix: Reproducing HuggingFace MMMU Benchmark with VLMEvalKit

The default **MMMU** dataset (`mmmu.py`) uses the [MMMU-Benchmark](https://github.com/MMMU-Benchmark/MMMU) prompt format, expecting short outputs with option letters directly (e.g., "A", "B", "C", "D").

Most HuggingFace VLM models report official MMMU scores using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). This framework uses long-form generation with reasoning and evaluates via OpenAI API (e.g., GPT-3.5). It achieves higher scores but requires longer inference and evaluation time.

### 1. Check VLMEvalKit Inference Configurations

Model repositories may provide benchmark configurations. For example, see [Qwen3-VL/evaluation/mmmu](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu).

For default configurations, refer to VLMEvalKit's [`config.py`](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py):
```python
"Qwen3-VL-4B-Instruct": partial(
    Qwen3VLChat,
    model_path="Qwen/Qwen3-VL-4B-Instruct",
    use_custom_prompt=False,
    use_vllm=True,
    temperature=0.7, 
    max_new_tokens=16384,
    repetition_penalty=1.0,
    presence_penalty=1.5,
    top_p=0.8,
    top_k=20
),
```

### 2. Prepare MMMU Dataset (VLMEvalKit Format)

```bash
python3 scripts/prepare_dataset.py \
  --dataset MMMU_VLMEvalkit \
  --output_dir /path/to/datasets/mmmu_vlmevalkit \
  --top_p 0.8 --top_k 20 --temperature 0.7
```

### 3. Building TensorRT Engines

VLMEvalKit requires larger image sizes, input lengths, and output lengths than default settings. For example, [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/issues/1617#issuecomment-3421734724) uses MIN_PIXELS=1,003,520 and MAX_PIXELS=4,014,080, translating to 980–3920 tokens per image. With up to 5 images per MMMU input: `maxImageTokens` = 3920 × 5 = 19600.

**Visual Encoder:**
```bash
./build/examples/multimodal/visual_build \
  --onnxDir /path/to/visual/onnx/model/ \
  --engineDir /path/to/visual/engine/ \
  --minImageTokens 980 \
  --maxImageTokens 19600 \
  --maxImageTokensPerImage 3920
```

**Text Engine:**
```bash
./build/examples/llm/llm_build \
  --onnxDir /path/to/text/onnx/model/ \
  --engineDir /path/to/text/engine/ \
  --maxInputLen 21000 \
  --maxKVCacheCapacity 30000 \
  --maxBatchSize 1 \
  --vlm \
  --minImageTokens 980 \
  --maxImageTokens 19600
```

### 4. Run Inference

**Inference:**
```bash
./build/examples/llm/llm_inference \
  --engineDir /path/to/text/engine/ \
  --multimodalEngineDir /path/to/visual/engine/ \
  --inputFile /path/to/datasets/mmmu_vlmevalkit/mmmu_dataset.json \
  --outputFile /path/to/outputs/mmmu_predictions.json \
  --batchSize 1 \
  --maxGenerateLength 8192
```

**Prepare Output:**

Download VLMEvalKit MMMU metadata file:
```bash
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```

Combine the TSV metadata with JSON predictions to create the xlsx file for VLMEvalKit:
```bash
python3 scripts/prepare_mmmu_vlmevalkit.py \
  --tsv_file /path/to/MMMU_DEV_VAL.tsv \
  --json_file /path/to/outputs/mmmu_predictions.json \
  --output_file /path/to/outputs/${Model}_MMMU_DEV_VAL.xlsx 
```

### 5. Evaluate with VLMEvalKit

Download and install [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Refer to the official [documentation](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) for setup instructions and usage.

**Evaluate TensorRT Edge LLM output:**

Run in eval-only mode. VLMEvalKit reads `${Model}_MMMU_DEV_VAL.xlsx` from `--work-dir` and outputs accuracy scores (`_acc.xlsx`) and evaluation results (`_results.xlsx`):
```bash
cd /path/to/VLMEvalKit

python3 run.py \
  --data MMMU_DEV_VAL \
  --model ${Model} \
  --work-dir /path/to/outputs \
  --mode eval \
  --verbose \
  --reuse
```

**(Optional) Verify PyTorch Baseline:**

Compare TensorRT Edge LLM results against PyTorch/vLLM baseline using VLMEvalKit with the same configuration.

```bash
python3 run.py \
  --data MMMU_DEV_VAL \
  --model ${Model} \
  --work-dir /path/to/outputs \
  --mode all \
  --verbose
```