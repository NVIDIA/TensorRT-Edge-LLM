# Speculative Decoding

TensorRT Edge-LLM supports MTP, EAGLE3, EAGLE-4, DFlash, DFlash2, LiLiCorr, DSpark, JetSpec, and Domino. Each method
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
| MTP | `Qwen/Qwen3-8B` | `AngelSlim/Qwen3-8B-MTP` (dense head merged into a base checkpoint copy) | 3 draft tokens, 4 verification positions |
| EAGLE3 | `Qwen/Qwen3-1.7B` | `AngelSlim/Qwen3-1.7B_eagle3` | 6 draft steps, top-10 tree, 60 verification positions |
| EAGLE-4 | `RadixArk/Qwen3.8-27B-NVFP4` | `EAGLE-Corp/Qwen3.8-27B-EAGLE4` | 12 draft steps, top-4 tree, 16 verification positions |
| DFlash | `Qwen/Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` | One block-16 draft pass |
| DFlash | `nvidia/Nemotron-3.5-Bolt-E4B-BF16` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-DFlash` | One block-8 draft pass |
| DFlash2 | `Qwen/Qwen3.8-27B` | `z-lab/Qwen3.8-27B-DFlash2` | One selector-guided block-8 draft pass |
| DFlash2 | `RadixArk/Muse-Glimmer-NVFP4` | `incoai/Muse-Glimmer-30B-DFlash2` | One selector-guided block-16 draft pass |
| LiLiCorr | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` | Seven proposed tokens from a top-8 candidate lattice, eight verification positions |
| DSpark | `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` | Seven proposed tokens, eight verification positions |
| JetSpec | `Qwen/Qwen3-8B` | `JetSpec/jetspec-qwen3-8b` | Block-16 draft, top-7 branching tree verification |
| Domino | `Qwen/Qwen3-8B` | `Huang2020/Qwen3-8B-Domino-b16` | Block-16 draft with a GRU correction rollout, linear verification |
| Domino | `Qwen/Qwen3.6-27B` | `Huang2020/qwen3.6-27B-domino` | Block-16 draft with a GRU correction rollout, linear verification |

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
trtedge export "$MODEL_DIR" "$MODEL_ROOT/onnx" --mtp
```

If export and engine build run on different machines, copy
`$MODEL_ROOT/onnx` to the target before continuing.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/onnx/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-spec-verify-tokens 4 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/onnx/mtp_draft" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-spec-draft-tokens 4 \
  --spec-draft
```

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 1 \
  --spec-draft-step 3 \
  --spec-verify-tokens 4
```

MTP is linear: `--spec-draft-top-k 1` selects one token at each of three draft
steps, and the base verifies those tokens plus the current token. For
[jump-forward](../features/guided-decoding.md#jump-forward) with tree drafting
(`--spec-draft-top-k` above 1), build the base with `--max-spec-verify-tokens`
above the runtime `--spec-verify-tokens`; chain drafting needs no extra verify
positions, and both need `--max-spec-draft-tokens` above `--spec-draft-step`,
as in the build above.

Dense Qwen3 checkpoints with a trained MTP head follow the same commands. The
head must live in the base checkpoint directory under the `mtp.*` prefix with
`num_nextn_predict_layers: 1` in `config.json`, as shipped by
[AngelSlim/Qwen3-8B-MTP](https://huggingface.co/AngelSlim/Qwen3-8B-MTP) for
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). `trtedge quantize llm`
quantizes the head together with the base (`--lm_head_quantization` also
applies to the draft's shared lm_head), and `trtedge export --mtp`
on the quantized checkpoint produces `llm/` and `mtp_draft/`. The
checkpoint-direct builder and `trtedge serve --speculative-config
'{"method":"mtp"}'` build the same pair from the checkpoint directory.

Gemma4 uses a matched assistant checkpoint instead of embedded draft layers.
For Gemma4 12B, export the matched base and assistant checkpoints together:

```bash
export GEMMA_MODEL_ID=google/gemma-4-12B-it
export GEMMA_ASSISTANT_ID=google/gemma-4-12B-it-assistant
export GEMMA_ROOT=$WORKSPACE_DIR/gemma-4-12B-it-mtp

trtedge export \
  "$GEMMA_MODEL_ID" \
  "$GEMMA_ROOT/onnx" \
  --mtp \
  --mtp-draft-dir "$GEMMA_ASSISTANT_ID"
```

Build and run `$GEMMA_ROOT/onnx/llm` and `$GEMMA_ROOT/onnx/mtp_draft` with the
same MTP commands above, substituting `MODEL_ROOT=$GEMMA_ROOT`. Assistant IDs
for other Gemma4 sizes are listed in
[Supported Models](../getting_started/supported-models.md#text-generation).
For [jump-forward](../features/guided-decoding.md#jump-forward), Gemma4 chain
drafting needs neither extra verify positions nor `--max-spec-draft-tokens`
above `--spec-draft-step`, and jumps with at most 15 draft steps.

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
trtedge export "$MODEL_DIR" "$MODEL_ROOT/base-export" --eagle-base
trtedge export "$DRAFT_DIR" "$MODEL_ROOT/draft-export"
```

Copy both export directories to the target when using separate machines.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-verify-tokens 60 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-draft-tokens 60 \
  --spec-draft
```

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 10 \
  --spec-draft-step 6 \
  --spec-verify-tokens 60
```

EAGLE3 expands a tree: each of six draft steps retains ten candidates, and the
base engine verifies up to 60 positions. Quantizing the draft is supported but
can reduce its acceptance rate. For
[jump-forward](../features/guided-decoding.md#jump-forward), build with
`--max-spec-verify-tokens` above the runtime `--spec-verify-tokens`.

## EAGLE-4

This example pairs
[RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
with [EAGLE-Corp/Qwen3.8-27B-EAGLE4](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4).
The draft weights have their own license; see
[EAGLE-4 Draft Models](../getting_started/supported-models.md#eagle-4-draft-models).
The NVFP4 base requires an SM100 or newer GPU.

EAGLE-4 drafters use the EAGLE3 mode; there is no separate method name, and
exported configurations record `spec_decode_type: "eagle3"`.

The draft engine contains a low-rank candidate head. It projects the draft
state to a low rank (128 for this drafter), scores the full vocabulary from
that projection, rescores the best `num_candidates` tokens (128 for this
drafter) with its full `lm_head`, and outputs their token IDs and
log-probabilities instead of full-vocabulary logits. The draft needs no
`d2t.safetensors`.

### Export

The draft configuration names the base layers whose hidden states it reads. A
Qwen3.5-family base therefore requires `--eagle-draft-dir` with
`--eagle-base`; the base export records the resolved layers as
`eagle_hidden_state_layers` (`[4, 32, 60]` for this pair).

```bash
export MODEL_ID=RadixArk/Qwen3.8-27B-NVFP4
export DRAFT_ID=EAGLE-Corp/Qwen3.8-27B-EAGLE4
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3.8-27B-eagle4
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"

tensorrt-edgellm-export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --eagle-base --eagle-draft-dir "$DRAFT_DIR"

tensorrt-edgellm-export "$DRAFT_DIR" "$MODEL_ROOT/draft-export"
```

Two options change the draft export:

- `--draft-dense-head` exports the full-vocabulary `lm_head` instead of the
  candidate head. The draft then outputs `logits` and an identity
  `d2t.safetensors`, which is useful for comparing the two heads.
- `--externalize-weights lm_head` keeps the 2.5 GB FP16 `lm_head` out of the
  draft engine. The export writes it to `external_lm_head_weight.safetensors`,
  `llm_build` copies it into the engine directory as
  `draft_external_lm_head_weight.safetensors`, and the runtime binds it when it
  loads the draft. The candidate head still runs inside the draft engine.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-verify-tokens 16 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-draft-tokens 48 \
  --spec-draft
```

`--max-spec-draft-tokens` must cover `--spec-draft-step` times `--spec-draft-top-k`
at run time.

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 4 \
  --spec-draft-step 12 \
  --spec-verify-tokens 16
```

Each of 12 draft steps keeps four candidates, and the base engine verifies
the 16 highest-scoring tree nodes, including the root. As with EAGLE3, only
greedy requests use speculative decoding; sampled requests are decoded without
a draft.

As for EAGLE3, `--spec-verify-tokens` is at least `--spec-draft-step` + 1. With
a candidate-head draft or a hybrid (Qwen3.5-family) base, the runtime also
rejects trees that exceed these limits:

- `--spec-draft-step` is at most 15.
- `--spec-draft-step` times `--spec-draft-top-k` is at most 64.
- `--spec-verify-tokens` is at most 64 and at most `1 + K + (D - 1) * K * K`,
  where `K` is `--spec-draft-top-k` and `D` is `--spec-draft-step`.
- With a candidate head, `--spec-draft-top-k` is at most the draft's
  `num_candidates`.

### ONNX-less Build

The experimental [ONNX-less builder](#onnx-less-alternative) builds both
engines from the checkpoints in one command. `--spec-type eagle3` selects the
EAGLE-4 draft graph because the draft configuration sets `fc_norm`,
`norm_output`, or a low-rank candidate head:

```bash
trtedge build \
  --model-dir "$MODEL_DIR" \
  --draft-model-dir "$DRAFT_DIR" \
  --spec-type eagle3 \
  --components llm \
  --engine-dir "$MODEL_ROOT/direct-engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-verify-tokens 16 \
  --max-spec-draft-tokens 48
```

Add `--draft-dense-head` to build the full-vocabulary `lm_head` instead of the
candidate head, as the export option of the same name does. The engines read
their weights from both checkpoints at run time:

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/direct-engines" \
  --checkpoint-dir "$MODEL_DIR" \
  --draft-checkpoint-dir "$DRAFT_DIR" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/direct-output.json" \
  --spec-decode \
  --spec-draft-top-k 4 \
  --spec-draft-step 12 \
  --spec-verify-tokens 16
```

### Quantization

`EAGLE-Corp/Qwen3.8-27B-EAGLE4` ships in BF16. To quantize its decoder layer,
calibrate it against the BF16 base model and export the quantized checkpoint
instead of the original drafter:

```bash
export BF16_BASE_ID=Qwen/Qwen3.8-27B
export BF16_BASE_DIR=$MODEL_ROOT/checkpoints/base-bf16

hf download "$BF16_BASE_ID" --local-dir "$BF16_BASE_DIR"

tensorrt-edgellm-quantize draft \
  --base-model-dir "$BF16_BASE_DIR" \
  --draft-model-dir "$DRAFT_DIR" \
  --output-dir "$MODEL_ROOT/checkpoints/draft-nvfp4" \
  --quantization nvfp4

tensorrt-edgellm-export \
  "$MODEL_ROOT/checkpoints/draft-nvfp4" "$MODEL_ROOT/draft-export"
```

Pass `--quantization` explicitly: for EAGLE drafts the command defaults to
`fp8`. The quantizer uses the same ModelOpt configuration, calibration text,
and sample count as the DFlash-family drafts. Only the decoder-layer linears
are quantized: `fc`, `lm_head`, the low-rank candidate head, and the draft KV
cache stay FP16. To run the decoder layer as NVFP4-A16 (weight-only), export the
same checkpoint with `--no-quantize-activations`. Build and run the draft engine
with the `llm_build` and `llm_inference` commands above.

### Limitations

- The ONNX-less builder pairs EAGLE-4 drafts with the Qwen3.5 base above;
  other bases are not validated. It rejects Qwen3.5 MoE, Muse-Glimmer,
  Nemotron and Gemma4 bases before it builds an engine. It also rejects EAGLE-4
  drafts whose configuration does not name the base layers they read,
  drafts with an FP8 KV cache, drafts in the speculators `eagle3` checkpoint
  layout, and a candidate-head draft whose `lm_head` is quantized; build such a
  draft with `--draft-dense-head`. The OpenAI-compatible server builds with the
  ONNX-less builder, so these limits also apply to a `--speculative-config`
  with `"method": "eagle3"`.
- For EAGLE-4 drafters, `tensorrt-edgellm-quantize draft` rejects
  `--kv-cache-quantization`, `--quantization nvfp4_a16_gptq`, and
  `--lm-head-quantization` with a candidate head.
  The export rejects a candidate-head checkpoint that quantizes `lm_head` or the
  KV cache; export a draft with a quantized `lm_head` with `--draft-dense-head`.
- `--reduced-vocab-dir` is not supported for any EAGLE draft.

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

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dflash-base --dflash-draft-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dflash-draft --dflash-draft-dir "$DRAFT_DIR"
```

The draft export writes `dflash_draft/`, not `llm/`.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-verify-tokens 16 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/dflash_draft" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-draft-tokens 16 \
  --spec-draft
```

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 1 \
  --spec-proposal-tokens 15 \
  --spec-verify-tokens 16
```

Provider-parity validation for Qwen3.5 DFlash uses `"enable_thinking": true`
at the top level of the input JSON. Qwen3 DFlash uses `false`.

The public Nemotron 3.5 pair uses the same workflow. Set `MODEL_ID` to
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
and `DRAFT_ID` to
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash).

Muse-Glimmer uses the same block-16 workflow. Pair
[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)
with either [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
or [`RadixArk/Muse-Glimmer-NVFP4`](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4).

[`nvidia/Nemotron-3.5-Bolt-E4B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Bolt-E4B-BF16)
pairs with the 8-token-block
[`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-DFlash`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-DFlash)
draft, which was trained with thinking enabled; set `"enable_thinking": true`
in the input JSON. Use the [ONNX-less builder](#onnx-less-alternative) with
`--spec-type dflash --max-spec-verify-tokens 16 --max-spec-draft-tokens 16`.
Run the same engine pair with `--spec-draft-top-k 1 --spec-verify-tokens 8` for
a chain or `--spec-draft-top-k 4 --spec-verify-tokens 16` for DDTree.

Export Qwen3.5 with `--dflash-base` for both chain and DDTree verification.
The same target artifact supports runtime `--spec-draft-top-k 1` for a chain
and greater values for greedy branching, within its built capacities. For
[jump-forward](../features/guided-decoding.md#jump-forward), DDTree
verification needs the base built with `--max-spec-verify-tokens` above the
runtime `--spec-verify-tokens`; chain verification needs nothing extra. See
[Reduce Vocabulary](../features/reduce-vocab.md#dflash-speculative-decoding-support)
for optional DFlash draft vocabulary reduction. Chain-MTP draft reduction is
documented in [MTP Speculative Decoding Support](../features/reduce-vocab.md#mtp-speculative-decoding-support).

## DFlash V2

DFlash V2 is selected automatically from the draft checkpoint architecture and
uses the same public `dflash` mode as DFlash V1. It adds
dynamic grouped causal convolution around every draft attention and MLP
sublayer, then uses an in-engine Candidate Selector to construct a
predecessor-conditioned seven-token path. The runtime verifies this path with
greedy matching or lossless rejection sampling. With `--spec-draft-top-k`
greater than 1, the runtime verifies a parent-conditioned DDTree instead; see
[DFlash2 DDTree](#dflash2-ddtree).

The supported workflow exports ONNX before building the matched engines:

```bash
export MODEL_DIR=/path/to/Qwen3.8-27B
export DRAFT_DIR=/path/to/Qwen3.8-27B-DFlash2
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3.8-27B-dflash2

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dflash-base --dflash-draft-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dflash-draft --dflash-draft-dir "$DRAFT_DIR"

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-verify-tokens 16 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/dflash_draft" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-draft-tokens 16 \
  --spec-draft
```

The engine profile supports DFlash2 blocks from 2 through 16. With no runtime
override, each draft uses its checkpoint block size: 8 for the listed Qwen3.8
draft and 16 for Muse-Glimmer.

The experimental ONNX-less builder is an optional alternative. One command
builds the same matched base and draft engine layout:

```bash
export MODEL_DIR=/path/to/Qwen3.8-27B
export DRAFT_DIR=/path/to/Qwen3.8-27B-DFlash2
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3.8-27B-dflash2

.venv/bin/trtedge build \
  --model-dir "$MODEL_DIR" \
  --draft-model-dir "$DRAFT_DIR" \
  --spec-type dflash \
  --components llm \
  --engine-dir "$MODEL_ROOT/engines" \
  --plugin-path /path/to/build/lib/libNvInfer_edgellm_plugin.so \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-batch-size 8 \
  --externalize-weights all
```

Run the pair by providing both checkpoints:

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --checkpoint-dir "$MODEL_DIR" \
  --draft-checkpoint-dir "$DRAFT_DIR" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --profile-output-file "$MODEL_ROOT/profile.json" \
  --dump-profile \
  --spec-decode
```

With profiling enabled, every response row records `generated_token_count`,
`spec_verify_count`, and `spec_acceptance_length` (generated tokens per target
verification). The profile JSON also contains aggregate speculative throughput
and acceptance length. `--profile-output-file` without `--dump-profile` records
wall-clock and memory only.

The server downloads, builds, and loads the pair in one command:

```bash
trtedge serve Qwen/Qwen3.8-27B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2"}'
```

The public Qwen3.8 evaluation uses `temperature=1`, `top_p=0.95`,
`top_k=20`, and thinking enabled. Add `seed` to server requests or
`sampling_seed` to C++ input JSON for repeatable requests, and leave
`spec_proposal_sampling` at `auto`.

`auto` makes proposal sampling follow the target policy. `greedy` and
`probabilistic` can be forced for experiments; both remain distributionally
correct because the verifier consumes the realized proposal distribution.
DFlash2 always proposes one checkpoint block, so `--spec-proposal-tokens`
overrides are rejected.
[Jump-forward](../features/guided-decoding.md#jump-forward) needs nothing extra
at build time; DFlash2 grafts forced tokens for greedy requests only.

### DFlash2 DDTree

`--spec-draft-top-k K` with K greater than 1 turns the DFlash2 proposal into a
tree. The draft still runs one forward pass. For every proposal position and
every token of the previous position's selector support, the runtime scores the
current support with the selector, using that token as the predecessor. A tree
node's children are therefore scored with the node's own token, not with the
greedy chain. The DDTree builder keeps the K best children of each node and
grows the `--spec-verify-tokens N` most probable prefixes; the target verifies
all N nodes in one pass.

| Limit | Value |
|---|---|
| Children per node `K` | 2 to the checkpoint `selector_top_k` (16 for the listed drafts), and less than N |
| Verify nodes `N` | at most the base engine verify profile and 128; at most 64 for hybrid (GDN or Mamba) targets |
| Accepted tokens per round | at most the checkpoint block size |
| Sampling | greedy only; sampled requests run on the vanilla decoder |

Build the base with a verify profile of at least N. The draft engine is the
same as for the chain:

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 8 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-verify-tokens 32 \
  --spec-base
```

With the checkpoint-direct builder, add `--max-spec-verify-tokens 32` to the
`trtedge build` command above. Then select the tree at runtime:

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --checkpoint-dir "$MODEL_DIR" \
  --draft-checkpoint-dir "$DRAFT_DIR" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 4 \
  --spec-verify-tokens 32
```

A greedy request that sets `spec_proposal_sampling` to `probabilistic` fails in
tree mode; use `--spec-draft-top-k 1` for probabilistic proposals. The tree verifies
more target positions per round than the chain, so its verify cost grows with N
and the batch size; on hybrid (GDN) targets each verify node also needs
recurrent-state scratch. At larger batch sizes a smaller N (for example 16) is
usually faster than 32 or more.

The server accepts the same shape in `--speculative-config`. `verify_tokens`
also sizes the base engine when `--max-spec-verify-tokens` is not set:

```bash
trtedge serve Qwen/Qwen3.8-27B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","draft_top_k":4,"verify_tokens":32}'
```

FP16, NVFP4, and INT4 checkpoints use the same command. Precision is detected
from checkpoint quantization metadata. When the draft is distributed only in
FP16, create a matched quantized draft with `trtedge quantize draft`
before building the pair.

## LiLiCorr

LiLiCorr is a DFlash V2 draft whose `dflash_config` sets
`projector_type: "lilicorr"`. It keeps the DFlash V2 backbone but replaces the
Candidate Selector with a candidate-lattice head inside the draft engine. For
each proposal slot, the draft engine keeps the `lilicorr_candidate_topk`
highest-probability tokens and scores every transition from a candidate in one
slot to a candidate in the next. The runtime walks these transition scores to
build one token chain, then verifies it with greedy matching or lossless
rejection sampling, as in DFlash V2. The published draft uses a block size of 8
(seven proposal slots) and eight candidates per slot.

This example pairs
[nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
with
[nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr).
Build the engines with the ONNX workflow (export, then `llm_build`) or
directly from the checkpoints, as described in
[Build from checkpoints](#build-from-checkpoints).

### Export

```bash
export MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
export DRAFT_ID=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr
export MODEL_ROOT=$WORKSPACE_DIR/Nemotron-3.5-Bolt-lilicorr
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dflash-base --dflash-draft-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dflash-draft --dflash-draft-dir "$DRAFT_DIR"
```

The draft export writes `dflash_draft/` with `model.onnx`, its external weights,
and `config.json`. It writes no selector sidecar. The draft has no `lm_head` of
its own, so the draft export also reads the base checkpoint. The draft engine
also has no embedding table: at run time it reads candidate-token embeddings
from the runtime's FP16 base embedding table. That table is the base export's
`embedding.safetensors`, or the checkpoint embedding when you run with
`--checkpoint-dir`. LiLiCorr therefore requires the FP16 embedding table, and the
exporter rejects `--fp8-embedding` with a LiLiCorr draft directory.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-verify-tokens 8 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/dflash_draft" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --max-spec-draft-tokens 8 \
  --spec-draft
```

Build the draft with `--max-spec-draft-tokens 8`. The LiLiCorr transition-score
output has a fixed extent of seven proposal slots, so the draft engine supports
only the checkpoint block size of 8.

### Build from checkpoints

The [checkpoint-direct builder](../getting_started/direct-engine-builder.md)
builds both engines from the downloaded checkpoints without an ONNX export:

```bash
trtedge build \
  --model-dir "$MODEL_DIR" \
  --draft-model-dir "$DRAFT_DIR" \
  --spec-type dflash \
  --components llm \
  --engine-dir "$MODEL_ROOT/direct-engines" \
  --max-batch-size 1 \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192
```

The draft checkpoint's `dflash_config` selects LiLiCorr. Both verification and
draft capacities default to the checkpoint block size of 8, and the builder
rejects any other `--max-spec-draft-tokens`. It also rejects `--fp8-embedding`,
because the draft gathers candidate embeddings from the base's FP16 embedding
table. The engines expose the same runtime bindings and `dflash_config` as the
ONNX workflow's engines. Most checkpoint weights are bound at run time from
`--checkpoint-dir` and `--draft-checkpoint-dir`. The checkpoint-direct LiLiCorr
build is validated on SM120 only.

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 1 \
  --spec-proposal-tokens 7 \
  --spec-verify-tokens 8
```

DFlash V2 verifies exactly the checkpoint `block_size` (8) positions per round,
so runtime `--spec-verify-tokens` must be 8 and `--spec-proposal-tokens` 7.
LiLiCorr is a linear chain: use `--spec-draft-top-k 1`.

For engines from the checkpoint-direct builder, set `--engine-dir` to
`"$MODEL_ROOT/direct-engines"` and add
`--checkpoint-dir "$MODEL_DIR" --draft-checkpoint-dir "$DRAFT_DIR"`.

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

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --dspark-base --dspark-draft-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --dspark-draft --dspark-draft-dir "$DRAFT_DIR"
```

The draft export writes `dspark_draft/` and its DSpark sidecar files.

### Build

```bash
./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/base-export/llm" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-verify-tokens 8 \
  --spec-base

./build/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/draft-export/dspark_draft" \
  --engine-dir "$MODEL_ROOT/engines" \
  --max-batch-size 1 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-spec-draft-tokens 7 \
  --spec-draft
```

### Run

```bash
./build/bin/llm_inference \
  --engine-dir "$MODEL_ROOT/engines" \
  --input-file "$INPUT_FILE" \
  --output-file "$MODEL_ROOT/output.json" \
  --spec-decode \
  --spec-draft-top-k 1 \
  --spec-proposal-tokens 7 \
  --spec-verify-tokens 8
```

The default chain mode reads the full seven-token proposal length from the
draft artifacts and supports non-greedy sampling. `--dspark-scheduler threshold`
and `--dspark-scheduler sps` enable adaptive chain lengths; use
`--dspark-min-proposal-len` and `--dspark-max-proposal-len` to bound them.

The ONNX export/build path also supports greedy DSpark DDTree. Export a
base with `--dspark-base --dspark-draft-dir "$DRAFT_DIR"`,
then build that ONNX with the intended verification budget, such as
`--max-spec-verify-tokens 16`. Hybrid recurrent-state replay requires runtime
`--spec-verify-tokens` to equal this build-time budget. Use a greedy input
(`"temperature": 0.0`, `"top_k": 1`) and run with
`--spec-draft-top-k 4 --spec-verify-tokens 16`. Tree mode accepts
`--dspark-scheduler off` or `threshold`; `sps` applies only to chain mode.

[Jump-forward](../features/guided-decoding.md#jump-forward) grafts forced
tokens for greedy DSpark requests: chain drafting needs nothing extra, and
DDTree drafting needs a base without linear-attention layers, built with
`--max-spec-verify-tokens` above the runtime `--spec-verify-tokens`.

Selecting `--dsparkW2Backend=fp8` converts the Markov
correction weight (`markov_w2`) to FP8 E4M3 with per-row scales at load time,
reducing the greedy proposal cost on bandwidth-bound devices. It applies to
greedy chain drafting and, for rank 512, the parent-conditioned tree scorer.
Non-greedy sampling keeps FP16 weights. Chain mode requires a Markov rank of
`16 * 2^k` up to 512; unsupported ranks fail at startup rather than silently
falling back. Acceptance changes from FP8 quantization are model-dependent, so
validate both acceptance and end-to-end throughput on the deployment workload.

### Packed NVFP4 W2 with FP16 activations on Thor

Native NVFP4 Markov W2 weights automatically use the SM110 W4A16 backend for
unbiased greedy DSpark chain and tree drafting. Export preserves the original
packed W2 and scales in `dspark_w2_nvfp4.safetensors` instead of exporting a
dense FP16 W2. The engine builder copies this file alongside
`dspark_heads.safetensors`, which contains W1 and any confidence tensors.
Packed-only exports require a runtime with native NVFP4 W2 support; older
runtimes cannot load them. No runtime environment variable is required.
Selection depends on W2's
checkpoint representation, not the draft backbone precision.

The fast path requires Markov rank 512, vocabulary divisible by 128, and the
SM110 `nvfp4_a16_blackwell_gemm` CuTe DSL artifacts. Other configurations use
a dense fallback, materialized at initialization from the packed weights.
This fallback also supports logit bias and non-greedy requests without
allocating weights during graph execution. Existing FP16-only head sidecars
retain their default FP16 behavior.

Select the compute backend through `llm_inference --dsparkW2Backend=auto|fp16|fp8|w4a16`.
The default is `auto`; no W2 environment variable is required at export or inference.
C++ callers set `SpecDecodeDraftingConfig::dsparkW2Backend` before constructing the runtime.
The decoder reads this consolidated runtime configuration once during initialization.

| W2 storage | `auto` | `fp16` | `fp8` | `w4a16` |
|---|---|---|---|---|
| Native packed NVFP4 | W4A16 when supported, otherwise FP16 | Dequantize to FP16 | Dequantize to FP16, then quantize to FP8 | Require supported W4A16 backend |
| FP16 | FP16 | FP16 | Quantize to FP8 | Error: packed NVFP4 weights required |

Export records `w2_format` in `dspark_heads_info.json`; the builder copies this metadata
and the sidecars into the engine directory. Runtime validates the declared format against
the available tensors. Older sidecars without the field retain FP16 behavior when dense
W2 is present; packed-only legacy artifacts can be inferred from their contents.
A format mismatch fails at initialization. Selecting a backend does not rewrite engine
files and requires neither re-export nor engine rebuild when the required weights exist.
An old FP16-only export of an NVFP4 checkpoint needs re-export to recover packed weights.
FP16 weights are **not** quantized to NVFP4 by this option.

The former runtime overrides `EDGELLM_DSPARK_W2_FP8` and `EDGELLM_DSPARK_W2_NVFP4`
are no longer read. Use `--dsparkW2Backend=fp8`, `fp16`, or `w4a16` for explicit
comparisons. An explicit unsupported W4A16 selection fails instead of silently falling
back. On SM121, `auto` uses FP16; the SM110 W4A16 kernel is not dispatched there.
These backend selections apply to supported unbiased greedy requests. Non-greedy and
logit-bias requests retain the dense FP16 path. Initialization logs the stored weight
format, requested backend, and selected backend.

The kernel dequantizes packed weights and applies their scales in FP16 before
tensor-core multiplication with FP16 activations. It accumulates in FP32 and
stores an FP16 correction before adding the backbone logits. These rounding
and reduction semantics differ from the dense kernel; this is not a
bitwise-equivalent mode. Validate model accuracy, acceptance, and end-to-end
throughput with the intended chain or tree configuration. Packed weights are
repacked and scratch buffers are allocated at initialization, before CUDA
graph capture. Retaining the dense fallback weights means this option does
not reduce the overall resident W2 allocation. For vocabulary 131072, rank 512,
and maximum batch 4, the opaque packed layout and scratch add approximately
37 MiB beyond the dense fallback. The temporary source layout adds another
36 MiB during initialization and is released after repack completes.

## JetSpec

JetSpec is a paired-draft speculative decoding method that uses a dedicated
external draft checkpoint and branching tree verification. TensorRT Edge-LLM
validates JetSpec with [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
and the [JetSpec/jetspec-qwen3-8b](https://huggingface.co/JetSpec/jetspec-qwen3-8b)
draft checkpoint.

JetSpec shares the cached-draft runtime contract with DFlash, but the draft uses
causal proposal attention. Use `--jetspec-base` for the base export,
`--jetspec-draft` for the draft export, and run with `--spec-draft-top-k > 1`.
`--spec-proposal-tokens` configures the usable proposal depth independently of
the root-inclusive target verification budget.

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
trtedge export \
  Qwen/Qwen3-8B \
  $MODEL_NAME/onnx/tree_base_export \
  --jetspec-base \
  --jetspec-draft-dir jetspec-qwen3-8b

# Export JetSpec draft model
trtedge export \
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
./build/bin/llm_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/tree_base \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --max-batch-size 1 \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-spec-verify-tokens 128 \
  --spec-base

# Build JetSpec draft engine
./build/bin/llm_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --max-batch-size 1 \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-spec-draft-tokens 16 \
  --spec-draft
```

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/bin/llm_inference \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --input-file $WORKSPACE_DIR/input.json \
  --output-file $WORKSPACE_DIR/output_jetspec.json \
  --spec-decode \
  --spec-draft-top-k 7 \
  --spec-proposal-tokens 15 \
  --spec-verify-tokens 128
```

**Key differences from EAGLE3 and DFlash:**
- `--jetspec-base` exports the base model for JetSpec tree verification
- `--jetspec-draft` exports the dedicated JetSpec draft model into `jetspec_draft/`
- `--spec-draft-top-k 7` matches the validated JetSpec Qwen3-8B tree configuration
- `--spec-verify-tokens 128` is the validated tree verification budget including the root token;
  128 is also the largest tree any speculative mode verifies, on every platform
- `--spec-proposal-tokens 15` selects the usable proposal depth from the block-16 draft
- Qwen3-8B JetSpec validation uses thinking mode disabled

[Jump-forward](../features/guided-decoding.md#jump-forward) needs a runtime
`--spec-verify-tokens` below the built `--max-spec-verify-tokens`; with both at
128, as above, the runtime ignores the flag.

## Domino

Domino pairs [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with the
[Huang2020/Qwen3-8B-Domino-b16](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16)
draft checkpoint; the Qwen3-4B and Qwen3.6-27B drafts from the same collection
follow the identical workflow. The draft transformer, base engine, and verification path are
the DFlash block-16 draft; Domino adds a causal correction head. The draft
engine emits a shared top-2048 candidate pool with its base logits at every
proposal position, the correction codebook rows, and the projected draft hidden
states. The runtime runs a per-position GRU rollout over the pool
(`domino_heads.safetensors`, exported next to the draft ONNX), records each
position's corrected pool scores, and builds the verify tree from them with the
same DDTree builder DFlash uses: `--specDraftTopK 1` is the greedy chain and
`--specDraftTopK > 1` fans out around it.

Domino checkpoints are plain DFlash checkpoints with
`dflash_config.projector_type = "domino"`; `--dflash-draft` detects them
automatically, and `--domino-base` / `--domino-draft` are the explicit
spellings. Domino supports greedy decoding only; chain (`--specDraftTopK 1`)
and tree (`--specDraftTopK` up to 8) run on the same engine pair.

### Export

```bash
export MODEL_ID=Qwen/Qwen3-8B
export DRAFT_ID=Huang2020/Qwen3-8B-Domino-b16
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-8B-domino
export MODEL_DIR=$MODEL_ROOT/checkpoints/base
export DRAFT_DIR=$MODEL_ROOT/checkpoints/draft

hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
hf download "$DRAFT_ID" --local-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/base-export" \
  --domino-base --domino-draft-dir "$DRAFT_DIR"

trtedge export \
  "$MODEL_DIR" "$MODEL_ROOT/draft-export" \
  --domino-draft --domino-draft-dir "$DRAFT_DIR"
```

The base export lands in `base-export/llm/`. The draft export lands in
`draft-export/dflash_draft/` together with `domino_heads.safetensors`, which
`llm_build` copies into the engine directory. The sidecar holds the GRU input
projection precomputed over the vocabulary (about 0.9 GB for Qwen3-8B), so the
rollout reads one row per token instead of the full GRU input weight.

### Build

```bash
./build/bin/llm_build \
  --onnxDir "$MODEL_ROOT/base-export/llm" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxSpecVerifyTokens 60 \
  --specBase

./build/bin/llm_build \
  --onnxDir "$MODEL_ROOT/draft-export/dflash_draft" \
  --engineDir "$MODEL_ROOT/engines" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 2048 \
  --maxSpecDraftTokens 60 \
  --specDraft
```

### Run

```bash
# Greedy chain
./build/bin/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 1 \
  --specProposalTokens 15 \
  --specVerifyTokens 16

# DDTree verification on the same engines
./build/bin/llm_inference \
  --engineDir "$MODEL_ROOT/engines" \
  --inputFile "$INPUT_FILE" \
  --outputFile "$MODEL_ROOT/output.json" \
  --specDecode \
  --specDraftTopK 4 \
  --specVerifyTokens 60
```

- `--specProposalTokens 15 --specVerifyTokens 16` with `--specDraftTopK 1` verifies the full block-16 chain
  including the root token
- `--specVerifyTokens` up to `--maxSpecVerifyTokens` (at most 128) sets the tree node budget; `--specDraftTopK`
  (at most 8) is the per-depth fanout. On Qwen3-8B, `--specDraftTopK 4 --specVerifyTokens 60` adds about 14% acceptance length on
  GSM8K and 30% on MT-Bench over the chain
- [Jump-forward](../features/guided-decoding.md#jump-forward) needs nothing extra on the chain. DDTree verification needs
  `--specVerifyTokens` below `--maxSpecVerifyTokens`; the tree command above verifies all 60 positions and does not jump

### Quantization

The base and the draft are quantized independently and any FP8, NVFP4, or
INT4-AWQ combination builds with the commands above. Quantize the base with
`trtedge quantize llm` (add `--kv_cache_quantization fp8` for an FP8
KV cache) and point the base export at the quantized checkpoint. The public
Domino drafts ship in BF16; create a quantized draft with
`trtedge quantize draft`, then pass its output directory as
`--domino-draft-dir` to both exports:

```bash
trtedge quantize draft \
  --base_model_dir "$MODEL_DIR" \
  --draft_model_dir "$DRAFT_DIR" \
  --output_dir "$MODEL_ROOT/checkpoints/draft-nvfp4" \
  --quantization nvfp4
```

Precision is detected from the checkpoint quantization metadata. The
target-hidden projector, the candidate-pool projection, and the GRU correction
head stay FP16 in every draft precision; only the draft transformer layers are
quantized.

## ONNX-less Alternative

The experimental `trtedge build` frontend reads local checkpoints and
builds the base and draft engines in one command. It does not export or parse
ONNX. Complete the
[direct-builder prerequisites](../getting_started/direct-engine-builder.md#prerequisites)
before using this path.

For embedded Qwen3.5 MTP:

```bash
trtedge build \
  --model-dir "$WORKSPACE_DIR/Qwen3.5-4B-mtp/checkpoints/base" \
  --engine-dir "$WORKSPACE_DIR/Qwen3.5-4B-mtp/direct-engines" \
  --spec-type mtp \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1 \
  --max-spec-verify-tokens 4 \
  --max-spec-draft-tokens 4
```

Paired methods add `--draft-model-dir` and select their method:

```bash
# EAGLE3
trtedge build \
  --model-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/checkpoints/base" \
  --draft-model-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/checkpoints/draft" \
  --engine-dir "$WORKSPACE_DIR/Qwen3-1.7B-eagle3/direct-engines" \
  --spec-type eagle3 \
  --max-input-len 1024 \
  --max-kv-cache-capacity 2048 \
  --max-batch-size 1 \
  --max-spec-verify-tokens 60 \
  --max-spec-draft-tokens 60
```

For DFlash, LiLiCorr, Domino, or DSpark, use that section's checkpoint paths,
tree sizes, and `--spec-type`. Direct-built DSpark and Domino support both chain
mode and greedy tree mode using the same target artifact; `--spec-type domino`
also writes `domino_heads.safetensors` next to the draft engine. Run
direct-built engines with the same method-specific inference settings shown
above. Change `--engine-dir` to the direct engine directory and add
`--checkpoint-dir <base_checkpoint>`. Paired methods also require
`--draft-checkpoint-dir <draft_checkpoint>`.
