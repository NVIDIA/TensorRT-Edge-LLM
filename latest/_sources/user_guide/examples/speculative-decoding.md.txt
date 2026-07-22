# Speculative Decoding

## EAGLE3

EAGLE3 [https://arxiv.org/abs/2503.01840] uses a smaller draft model to accelerate generation. This guide covers EAGLE for both text-only LLMs and vision-language models.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

### Supported Draft Models

| Base Model | Draft Model |
|------------|-------------|
| Llama-3.1-8B-Instruct | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) |
| Qwen3-1.7B | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |
| Qwen3-4B | [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) |
| Qwen3-8B | [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) |
| Qwen2.5-VL-7B-Instruct | [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) |

Any EAGLE3-compatible draft model on HuggingFace can be tried however TensorRT Edge-LLM team does not test the accuracy or acceptance rate. Search for [eagle3 models](https://huggingface.co/models?search=eagle3) to find additional options.

Draft model quantization is supported via `tensorrt-edgellm-quantize` with
`fp8`, `int4_awq`, `nvfp4`, `mxfp8`, and `int8_sq` backbone quantization. For
example:

```bash
tensorrt-edgellm-quantize draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-draft
```

Note that quantizing the draft model will cause a drop in acceptance rate compared to running it in FP16, which may reduce the overall speedup.

---

### LLM EAGLE

**Example model:** Llama-3.1-8B-Instruct with [EAGLE draft model](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
cd EAGLE3-LLaMA3.1-Instruct-8B && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize llm \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base model with EAGLE flag
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-base \
  $MODEL_NAME/onnx/base_export \
  --eagle-base

# Quantize draft model
tensorrt-edgellm-quantize draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-draft \
  $MODEL_NAME/onnx/draft_export

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/base $MODEL_NAME/onnx/draft
cp -a $MODEL_NAME/onnx/base_export/llm/. $MODEL_NAME/onnx/base/
cp -a $MODEL_NAME/onnx/draft_export/llm/. $MODEL_NAME/onnx/draft/
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd /path/to/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --specBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 60 \
  --specDraft
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode
```

---

### VLM EAGLE

EAGLE for vision-language models combines accelerated text generation with image understanding.

**Example model:** Qwen2.5-VL-7B-Instruct with [EAGLE3 draft model](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl
cd qwen2.5-vl-7b-eagle3-sgl && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base LLM and FP16 visual encoder
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-base \
  $MODEL_NAME/onnx/base_export \
  --eagle-base

# Quantize draft model
tensorrt-edgellm-quantize draft \
  --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --draft_model_dir qwen2.5-vl-7b-eagle3-sgl \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-draft \
  $MODEL_NAME/onnx/draft_export

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/base $MODEL_NAME/onnx/draft $MODEL_NAME/onnx/visual
cp -a $MODEL_NAME/onnx/base_export/llm/. $MODEL_NAME/onnx/base/
cp -a $MODEL_NAME/onnx/base_export/visual/. $MODEL_NAME/onnx/visual/
cp -a $MODEL_NAME/onnx/draft_export/llm/. $MODEL_NAME/onnx/draft/
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd /path/to/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --specBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --specDraft

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode
```
---

## MTP (Multi-Token Prediction)

MTP is a speculative decoding method that proposes a linear chain of draft tokens and verifies them with the base model. The runtime reuses the same speculative decoding pipeline as EAGLE with `topK=1`.

TensorRT Edge-LLM supports two MTP checkpoint layouts:

- **Single-checkpoint MTP:** Qwen3.5 dense checkpoints with `num_draft_layers > 0` carry the MTP draft weights inside the base checkpoint. Export with `--mtp`.
- **Paired-assistant MTP:** Gemma4 uses a base checkpoint plus a matched assistant checkpoint. Export with `--mtp --mtp-draft-dir <assistant_checkpoint>`. Do not mix assistant checkpoints across Gemma4 model sizes or families.

---

### Single-Checkpoint MTP Example

**Example model:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

#### Step 1: Export (x86 Host)

MTP export produces both the base model and draft model ONNX from a single checkpoint using `tensorrt_edgellm`:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Export
tensorrt-edgellm-export \
  Qwen/Qwen3.5-4B \
  $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --mtp
```

This produces:
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/llm/` — MTP base model (hybrid attention + GDN layers, with tree-attention and intermediate state outputs)
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/mtp_draft/` — MTP draft head (single attention layer with Add-based hidden state fusion)

For Gemma4 paired-assistant MTP, download or stage both the base checkpoint and the matched assistant checkpoint, then pass the assistant with `--mtp-draft-dir`:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=gemma-4-E2B-it
export ASSISTANT_MODEL_NAME=gemma-4-E2B-it-assistant
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

tensorrt-edgellm-export \
  $WORKSPACE_DIR/$MODEL_NAME \
  $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --mtp \
  --mtp-draft-dir $WORKSPACE_DIR/$ASSISTANT_MODEL_NAME
```

This produces the same `llm/` and `mtp_draft/` layout used by the build and inference steps below.


#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $WORKSPACE_DIR/$MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```


#### Step 3: Build Engines

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
cd /path/to/TensorRT-Edge-LLM

# Build MTP base engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 7 \
  --specBase

# Build MTP draft engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/mtp_draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 7 \
  --specDraft
```


#### Step 4: Run Inference

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 3 \
  --specVerifySize 4
```

**Key differences from EAGLE:**
- `--specDraftTopK 1`: MTP uses a linear chain (no branching), so topK=1
- `--specDraftStep 3`: Number of MTP draft tokens (typically 3-7, matching the model's MTP head count)
- `--specVerifySize 4`: Equals `draftStep + 1` for the linear chain

---

## DFlash

DFlash is a speculative decoding method that uses a dedicated external draft model. The draft model is separately trained, is not built into the base model checkpoint, and must be paired with the specific base model it was trained for. DFlash draft models are published by z-lab in the [DFlash HuggingFace collection](https://huggingface.co/collections/z-lab/dflash). Unlike EAGLE3, DFlash draft architecture is model-family-specific: the draft consumes concatenated target hidden states from selected base-model layers, updates target K/V in its draft KV cache, and proposes either a linear block of draft tokens (`--specDraftTopK 1`) or a branching DDTree proposal (`--specDraftTopK > 1`).

So far DFlash support in TensorRT Edge-LLM is validated for Qwen3 and Qwen3.5 only. See [DFlash Draft Models](../getting_started/supported-models.md#dflash-draft-models) for the supported base/draft pairs. Other DFlash draft models in the z-lab collection are not tested for TensorRT Edge-LLM accuracy, acceptance rate, or runtime compatibility.

For best acceptance rate and throughput, use the thinking-mode setting that matches the paired z-lab draft model and HuggingFace generation behavior: enable thinking for Qwen3.5 DFlash models and disable thinking for Qwen3 DFlash models.

DFlash supports FP16 and quantized base models. Quantize a base checkpoint with `tensorrt-edgellm-quantize llm` before export, or start from a supported pre-quantized checkpoint. DFlash draft quantization is supported through `tensorrt-edgellm-quantize draft`; the command auto-detects DFlash from the draft checkpoint's `dflash_config`. NVFP4 draft quantization is validated, including optional NVFP4 LM-head quantization.

For large-vocabulary base models (e.g. Qwen3 family at 151936), the DFlash draft LM-head argmax can be reduced via vocabulary reduction on the draft only. Any reduced size can be passed to `tensorrt-edgellm-reduce-vocab --reduced_vocab_size` — there is no hardcoded set of supported sizes. Of the sweep we ran on Qwen3-8B DFlash, **64K** gave the best measured throughput (19.3 vs 18.5 tok/sec, ~4% speedup) with no acceptance loss; 32K regressed because acceptance dropped enough to undo the drafter-latency win. Measure on your own model and workload before committing to a setting. See [DFlash Speculative Decoding Support](../features/reduce-vocab.md#dflash-speculative-decoding-support) for the export workflow and the full results table.

The following example quantizes the paired [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) draft for [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Download the draft checkpoint first and pass its local directory as `--draft_model_dir`:

```bash
git clone https://huggingface.co/z-lab/Qwen3.5-4B-DFlash
cd Qwen3.5-4B-DFlash && git lfs pull && cd ..

tensorrt-edgellm-quantize draft \
  --base_model_dir Qwen/Qwen3.5-4B \
  --draft_model_dir Qwen3.5-4B-DFlash \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-draft
```

---

### Example

**Example model:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) with [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)

#### Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download DFlash draft model to workspace
git clone https://huggingface.co/z-lab/Qwen3.5-4B-DFlash
cd Qwen3.5-4B-DFlash && git lfs pull && cd ..

# Export DFlash base model for linear DFlash (`--specDraftTopK 1`)
tensorrt-edgellm-export \
  Qwen/Qwen3.5-4B \
  $MODEL_NAME/onnx/base_export \
  --dflash-base \
  --dflash-draft-dir Qwen3.5-4B-DFlash

# For Qwen3.5 hybrid DDTree (`--specDraftTopK > 1`), export the base with
# tree metadata inputs instead. Do not use this tree-base engine with
# `--specDraftTopK 1`.
tensorrt-edgellm-export \
  Qwen/Qwen3.5-4B \
  $MODEL_NAME/onnx/tree_base_export \
  --dflash-tree-base \
  --dflash-draft-dir Qwen3.5-4B-DFlash

# Export DFlash draft model
tensorrt-edgellm-export \
  Qwen/Qwen3.5-4B \
  $MODEL_NAME/onnx/draft_export \
  --dflash-draft \
  --dflash-draft-dir Qwen3.5-4B-DFlash

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/base $MODEL_NAME/onnx/tree_base $MODEL_NAME/onnx/draft
cp -a $MODEL_NAME/onnx/base_export/llm/. $MODEL_NAME/onnx/base/
cp -a $MODEL_NAME/onnx/tree_base_export/llm/. $MODEL_NAME/onnx/tree_base/
cp -a $MODEL_NAME/onnx/draft_export/dflash_draft/. $MODEL_NAME/onnx/draft/
```

This produces:
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/base/` - DFlash base model with target hidden-state outputs for linear DFlash
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/tree_base_export/llm/` - Qwen3.5 hybrid DDTree base model with `tree_parent_ids` and `tree_depths` inputs
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/draft/` - DFlash draft model

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $WORKSPACE_DIR/$MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
cd /path/to/TensorRT-Edge-LLM

# Build DFlash base engine.
# Use $MODEL_NAME/onnx/base for `--specDraftTopK 1`.
# Use $MODEL_NAME/onnx/tree_base for Qwen3.5 hybrid DDTree with `--specDraftTopK > 1`.
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 16 \
  --specBase

# Build DFlash draft engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 16 \
  --specDraft
```

#### Step 4: Run Inference (Thor Device)

For this Qwen3.5 example, set `enable_thinking` to `true` in the input JSON. For Qwen3 DFlash models, set `enable_thinking` to `false`. These settings match the paired HuggingFace generation behavior used to validate DFlash acceptance rate and throughput.

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 1 \
  --specVerifySize 16
```

For Qwen3.5 hybrid DDTree, build the base engine from the `--dflash-tree-base` export and run with a candidate top-K greater than 1:

```bash
./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output_ddtree.json \
  --specDecode \
  --specDraftTopK 8 \
  --specDraftStep 1 \
  --specVerifySize 64
```

**Key differences from EAGLE3:**
- `--dflash-base` exports the base model with DFlash target hidden-state outputs for linear DFlash; pass `--dflash-draft-dir` so export can read the draft's target-layer and block-size configuration
- `--dflash-tree-base` exports a Qwen3.5 hybrid DDTree base model with additional `tree_parent_ids` and `tree_depths` inputs; use it only with `--specDraftTopK > 1`
- `--dflash-draft` exports the dedicated draft model into `dflash_draft/`
- `--specDraftTopK 1`: DFlash proposes a linear block, so topK=1
- `--specDraftTopK > 1`: DFlash runs branching DDTree verification; Qwen3.5 hybrid DDTree requires a base engine exported with `--dflash-tree-base`
- `--specDraftStep 1`: One DFlash draft forward proposes the whole block
- `--specVerifySize 16`: Base verification checks the DFlash proposal block
- Thinking-mode settings are model-family-specific and should match the paired HuggingFace behavior: Qwen3.5 DFlash uses thinking mode enabled, while Qwen3 DFlash uses thinking mode disabled
- DFlash does not use EAGLE3 draft-to-target vocabulary mapping files; the draft checkpoint is a paired z-lab model for the base checkpoint
