# Speculative Decoding

TensorRT Edge-LLM supports MTP, EAGLE3, DFlash, DSpark, and JetSpec. Each method
uses a different draft architecture and checkpoint contract. Use only a base/draft
pair listed under [Speculative Draft Checkpoints](../getting_started/supported-models.md#speculative-draft-checkpoints).

The examples below follow the supported ONNX workflow:

1. Download the named checkpoint or checkpoint pair.
2. Export the base and draft components on an x86 host.
3. Build both engines into the same engine directory on the target.
4. Run `llm_inference` with the method-specific proposal settings.

The [ONNX-less alternative](#onnx-less-alternative) builds both engines in one
command. Choose one workflow; do not run both for the same deployment.

Complete [Installation](../getting_started/installation.md) first. Run the
build and inference commands from the repository root:

```bash
export REPO_DIR=/path/to/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export INPUT_FILE=$REPO_DIR/tests/test_cases/llm_basic.json
mkdir -p "$WORKSPACE_DIR"
cd "$REPO_DIR"
```

| Method | Base checkpoint | Draft checkpoint | Proposal configuration |
|---|---|---|---|
| MTP | `Qwen/Qwen3.5-4B` | Embedded in the base checkpoint | 3 draft tokens, 4 verification positions |
| EAGLE3 | `Qwen/Qwen3-1.7B` | `AngelSlim/Qwen3-1.7B_eagle3` | 6 draft steps, top-10 tree, 60 verification positions |
| DFlash | `Qwen/Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` | One block-16 draft pass |
| DSpark | `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` | Seven proposed tokens, eight verification positions |
| JetSpec | `Qwen/Qwen3-8B` | `JetSpec/jetspec-qwen3-8b` | Block-16 draft, top-7 branching tree verification |

## MTP

[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) contains its MTP
draft layer. One export command produces `llm/` for the base and `mtp_draft/`
for the draft.

### Export

```bash
export MODEL_ID=Qwen/Qwen3.5-4B
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3.5-4B-mtp
export MODEL_DIR=$MODEL_ROOT/checkpoints/base

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
tensorrt-edgellm-export "$MODEL_DIR" "$MODEL_ROOT/onnx" --mtp
```

If export and engine build run on different machines, copy
`$MODEL_ROOT/onnx` to the target before continuing.

### Build

```bash
./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/onnx/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 4 \
  --specBase

./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/onnx/mtp_draft" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 4 \
  --specDraft
```

### Run

```bash
./build/examples/llm/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 3 \
  --specVerifySize 4
```

MTP is linear: `--specDraftTopK 1` selects one token at each of three draft
steps, and the base verifies those tokens plus the current token.

Gemma4 uses a matched assistant checkpoint instead of embedded draft layers.
For Gemma4 12B, export the matched base and assistant checkpoints together:

```bash
export GEMMA_MODEL_ID=google/gemma-4-12B-it
export GEMMA_ASSISTANT_ID=google/gemma-4-12B-it-assistant
export GEMMA_ROOT=$WORKSPACE_DIR/gemma-4-12B-it-mtp

tensorrt-edgellm-export \
  "$GEMMA_MODEL_ID" \
  "$GEMMA_ROOT/onnx" \
  --mtp \
  --mtp-draft-dir "$GEMMA_ASSISTANT_ID"
```

Build and run `$GEMMA_ROOT/onnx/llm` and `$GEMMA_ROOT/onnx/mtp_draft` with the
same MTP commands above, substituting `MODEL_ROOT=$GEMMA_ROOT`. Assistant IDs
for other Gemma4 sizes are listed in
[Supported Models](../getting_started/supported-models.md#text-generation).

## EAGLE3

This example pairs [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
with [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3).
The base export enables EAGLE verification inputs; the draft checkpoint exports
as a regular `llm/` component.

### Export

```bash
export MODEL_ID=Qwen/Qwen3-1.7B
export DRAFT_ID=AngelSlim/Qwen3-1.7B_eagle3
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-1.7B-eagle3
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"
tensorrt-edgellm-export "$MODEL_DIR" "$MODEL_ROOT/base-export" --eagle-base
tensorrt-edgellm-export "$DRAFT_DIR" "$MODEL_ROOT/draft-export"
```

Copy both export directories to the target when using separate machines.

### Build

```bash
./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/base-export/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxVerifyTreeSize 60 \
  --specBase

./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/draft-export/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxDraftTreeSize 60 \
  --specDraft
```

### Run

```bash
./build/examples/llm/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 10 \
  --specDraftStep 6 \
  --specVerifySize 60
```

EAGLE3 expands a tree: each of six draft steps retains ten candidates, and the
base engine verifies up to 60 positions. Quantizing the draft is supported but
can reduce its acceptance rate.

## DFlash

This example pairs [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
with [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash).
DFlash reads selected target hidden states and proposes one block of tokens in
a single draft forward pass.

### Export

```bash
export MODEL_ID=Qwen/Qwen3.5-4B
export DRAFT_ID=z-lab/Qwen3.5-4B-DFlash
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3.5-4B-dflash
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"

tensorrt-edgellm-export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dflash-base --dflash-draft-dir "$DRAFT_DIR"

tensorrt-edgellm-export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dflash-draft --dflash-draft-dir "$DRAFT_DIR"
```

The draft export writes `dflash_draft/`, not `llm/`.

### Build

```bash
./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/base-export/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxVerifyTreeSize 16 \
  --specBase

./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/draft-export/dflash_draft" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxDraftTreeSize 16 \
  --specDraft
```

### Run

```bash
./build/examples/llm/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 1 \
  --specVerifySize 16
```

Provider-parity validation for Qwen3.5 DFlash uses `"enable_thinking": true`
at the top level of the input JSON. Qwen3 DFlash uses `false`.

The public Nemotron 3.5 pair uses the same workflow. Set `MODEL_ID` to
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
and `DRAFT_ID` to
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash).

A branching Qwen3.5 DDTree requires `--dflash-tree-base` during export and a
runtime `--specDraftTopK` greater than 1. Linear and DDTree base engines are
not interchangeable. See
[Reduce Vocabulary](../features/reduce-vocab.md#dflash-speculative-decoding-support)
for optional DFlash draft vocabulary reduction.

## DSpark

TensorRT Edge-LLM supports the three public block-7 pairs listed under
[DSpark Draft Models](../getting_started/supported-models.md#dspark-draft-models).
This example pairs [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with
[deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7).
The public draft proposes seven tokens, so the base verifies eight positions.

### Export

```bash
export MODEL_ID=Qwen/Qwen3-4B
export DRAFT_ID=deepseek-ai/dspark_qwen3_4b_block7
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-4B-dspark
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"

tensorrt-edgellm-export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dspark-base --dspark-draft-dir "$DRAFT_DIR"

tensorrt-edgellm-export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dspark-draft --dspark-draft-dir "$DRAFT_DIR"
```

The draft export writes `dspark_draft/` and its DSpark sidecar files.

### Build

```bash
./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/base-export/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxVerifyTreeSize 8 \
  --specBase

./build/examples/llm/llm_build \
  --onnxDir "$MODEL_ROOT/draft-export/dspark_draft" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxDraftTreeSize 7 \
  --specDraft
```

### Run

```bash
./build/examples/llm/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 1 \
  --specVerifySize 8
```

The default chain mode reads the full seven-token proposal length from the
draft artifacts and supports non-greedy sampling. `--dsparkScheduler threshold`
and `--dsparkScheduler sps` enable adaptive chain lengths; use
`--dsparkMinProposalLen` and `--dsparkMaxProposalLen` to bound them.

The ONNX export/build path also supports greedy DSpark DDTree. Build the base
engine with a larger verification profile, such as `--maxVerifyTreeSize 16`,
use a greedy input (`"temperature": 0.0`, `"top_k": 1`), and run with
`--specDraftTopK 4 --specVerifySize 16`. Tree mode accepts
`--dsparkScheduler off` or `threshold`; `sps` applies only to chain mode.

## JetSpec

JetSpec is a paired-draft speculative decoding method that uses a dedicated
external draft checkpoint and branching tree verification. TensorRT Edge-LLM
validates JetSpec with [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
and the [JetSpec/jetspec-qwen3-8b](https://huggingface.co/JetSpec/jetspec-qwen3-8b)
draft checkpoint.

JetSpec shares the cached-draft runtime contract with DFlash, but the draft uses
causal proposal attention. Use `--jetspec-tree-base` for the base export,
`--jetspec-draft` for the draft export, and run with `--specDraftTopK > 1`.
`--jetspecBlockSize` is an alias of `--dflashBlockSize`; both flags configure
the proposal block size read from the same runtime field.

Set `"enable_thinking": false` in the input JSON for Qwen3-8B JetSpec
validation. This matches the Qwen3 chat-template behavior used by the validated
greedy accuracy and throughput runs.

### Example

**Example model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with
[JetSpec/jetspec-qwen3-8b](https://huggingface.co/JetSpec/jetspec-qwen3-8b)

#### Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-8B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download JetSpec draft model to workspace
git clone https://huggingface.co/JetSpec/jetspec-qwen3-8b
cd jetspec-qwen3-8b && git lfs pull && cd ..

# Export JetSpec tree base model
tensorrt-edgellm-export \
  Qwen/Qwen3-8B \
  $MODEL_NAME/onnx/tree_base_export \
  --jetspec-tree-base \
  --jetspec-draft-dir jetspec-qwen3-8b

# Export JetSpec draft model
tensorrt-edgellm-export \
  Qwen/Qwen3-8B \
  $MODEL_NAME/onnx/draft_export \
  --jetspec-draft \
  --jetspec-draft-dir jetspec-qwen3-8b

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/tree_base $MODEL_NAME/onnx/draft
cp -a $MODEL_NAME/onnx/tree_base_export/llm/. $MODEL_NAME/onnx/tree_base/
cp -a $MODEL_NAME/onnx/draft_export/jetspec_draft/. $MODEL_NAME/onnx/draft/
```

This produces:
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/tree_base/` - JetSpec base model with target hidden-state outputs and tree-attention inputs
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/draft/` - JetSpec draft model

#### Step 2: Transfer to Device

```bash
scp -r $WORKSPACE_DIR/$MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-8B
cd /path/to/TensorRT-Edge-LLM

# Build JetSpec base engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/tree_base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 128 \
  --specBase

# Build JetSpec draft engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 128 \
  --specDraft
```

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output_jetspec.json \
  --specDecode \
  --specDraftTopK 7 \
  --specDraftStep 1 \
  --specVerifySize 128 \
  --jetspecBlockSize 16
```

**Key differences from EAGLE3 and DFlash:**
- `--jetspec-tree-base` exports the base model for JetSpec tree verification
- `--jetspec-draft` exports the dedicated JetSpec draft model into `jetspec_draft/`
- `--specDraftTopK 7` matches the validated JetSpec Qwen3-8B tree configuration
- `--specVerifySize 128` is the validated tree verification budget including the root token
- `--jetspecBlockSize 16` controls the draft proposal horizon; `--dflashBlockSize 16` is equivalent
- Qwen3-8B JetSpec validation uses thinking mode disabled

## ONNX-less Alternative

The experimental `tensorrt-edgellm-build` frontend reads local checkpoints and
builds the base and draft engines in one command. It does not export or parse
ONNX. Complete the
[direct-builder prerequisites](../getting_started/direct-engine-builder.md#prerequisites)
before using this path.

For embedded Qwen3.5 MTP:

```bash
tensorrt-edgellm-build \
  --model-dir "$WORKSPACE_DIR/Qwen3.5-4B-mtp/checkpoints/base" \
  --engine-dir "$WORKSPACE_DIR/Qwen3.5-4B-mtp/direct-engines" \
  --spec-type mtp \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1 \
  --max-verify-tree-size 4 \
  --max-draft-tree-size 4
```

Paired methods add `--draft-model-dir` and select their method:

```bash
# EAGLE3
tensorrt-edgellm-build \
  --model-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/checkpoints/base" \
  --draft-model-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/checkpoints/draft" \
  --engine-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/direct-engines" \
  --spec-type eagle3 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-batch-size 1 \
  --max-verify-tree-size 60 \
  --max-draft-tree-size 60
```

For DFlash or DSpark, use that section's checkpoint paths, tree sizes, and
`--spec-type`; direct-built DSpark engines currently use chain mode. Run
direct-built engines with the same method-specific inference settings shown
above. Change `--engineDir` to the direct engine directory and add
`--checkpointDir <base_checkpoint>`. Paired methods also require
`--draftCheckpointDir <draft_checkpoint>`.
