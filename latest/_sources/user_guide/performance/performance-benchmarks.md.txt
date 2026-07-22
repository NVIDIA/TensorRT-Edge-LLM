# Performance Benchmarks

> **Platforms:** NVIDIA Jetson AGX Thor Developer Kit, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, and Jetson Orin Nano 8GB. Older release sections may include a narrower platform set.

## Definitions

| Term | Description |
|------|-------------|
| **Prefill time** | Average wall-clock time (ms) to process the input prompt |
| **Prefill throughput** | Prompt tokens processed per second during prefill (tok/s) |
| **Generation throughput** | Tokens generated per second during decoding (tok/s) |
| **Batch size** | Number of concurrent sequences (BS=1 = single-user latency, BS=8 = multi-user throughput) |
| **Acceptance rate** | Average tokens accepted per speculative decoding verify step (higher is better) |
| **Speedup** | Speculative decoding generation throughput / vanilla generation throughput (same model, precision, batch size) |
| **ViT time** | Total visual encoder processing time per inference run (ms) |
| **ViT throughput** | Image tokens processed per second by the visual encoder (tok/s) |
| **GPU memory** | Peak GPU memory usage during inference (MB) |
| **MTP** | Multi-token prediction speculative decoding |
| **DFlash** | z-lab paired-draft speculative decoding with a dedicated external draft checkpoint |

### Precision Key

| Precision | Description | Platform Requirement |
|-----------|-------------|---------------------|
| FP16 | Half-precision float | All platforms |
| FP8 | 8-bit float | SM89+ (Ada Lovelace and newer) |
| INT4 AWQ | 4-bit integer (AWQ quantization) | All platforms |
| INT4 GPTQ | 4-bit integer (GPTQ quantization) | All platforms |
| NVFP4 | NVIDIA 4-bit float | SM100+ (Blackwell and newer) |

## Reproducing Benchmark Runs

Use the public TensorRT Edge-LLM user guides for model export, engine build, and
basic inference:

- [Quick Start Guide](../getting_started/quick-start-guide.md) for text-only
  LLM export, `llm_build`, `llm_inference`, and `llm_bench`.
- [VLM Inference](../examples/vlm.md) for visual encoder export/build and
  `--multimodalEngineDir`.
- [Speculative Decoding](../examples/speculative-decoding.md) for EAGLE3, MTP,
  and DFlash export/build layouts.
- [Input Format Guide](../format/input-format.md) for the Edge-LLM JSON request
  format and chat-template behavior.

This section records the benchmark-specific dataset choices, build-time limits,
runtime flags, and `llm_bench` shapes used to reproduce the tables.

### 1. Prepare Datasets

Generate datasets with the scripts under `examples/accuracy/scripts`. MTBench
and MMLU are text-only LLM datasets. COCO and MMMU are VLM datasets.

```bash
python3 -m pip install -r examples/accuracy/requirements.txt

DATASET_DIR=/path/to/datasets
mkdir -p "${DATASET_DIR}"

# LLM generation benchmark dataset.
python3 examples/accuracy/scripts/prepare_dataset.py \
  --dataset MTBench \
  --output_dir "${DATASET_DIR}/mtbench" \
  --batch_size 1

# LLM multiple-choice accuracy-side dataset.
python3 examples/accuracy/scripts/prepare_dataset.py \
  --dataset MMLU \
  --output_dir "${DATASET_DIR}/mmlu" \
  --batch_size 1

# VLM generation benchmark dataset.
python3 examples/accuracy/scripts/prepare_dataset.py \
  --dataset COCO \
  --output_dir "${DATASET_DIR}/coco" \
  --batch_size 1

# VLM multiple-choice accuracy-side dataset.
python3 examples/accuracy/scripts/prepare_dataset.py \
  --dataset MMMU \
  --output_dir "${DATASET_DIR}/mmmu" \
  --batch_size 1
```

### 2. Export and Build Specs

Follow the linked export/build docs for the selected model family, precision,
and decoding mode. Use these benchmark-specific build parameters:

| Engine | `llm_build` / `visual_build` parameters |
|--------|-----------------------------------------|
| Vanilla LLM | `--maxBatchSize <batch>` `--maxInputLen 2048` `--maxKVCacheCapacity 2200` |
| Vanilla VLM LLM engine | `--maxBatchSize <batch>` `--maxInputLen 2048` `--maxKVCacheCapacity 2200` |
| VLM visual engine | `--minImageTokens 8` `--maxImageTokens 16384` `--maxImageTokensPerImage 2048` |
| Orin NX / Orin Nano VLM visual engine | `--minImageTokens 8` `--maxImageTokens 2048` `--maxImageTokensPerImage 2048` |
| EAGLE3 base engine | Vanilla LLM parameters plus `--specBase --maxVerifyTreeSize 60` |
| EAGLE3 draft engine | Vanilla LLM parameters plus `--specDraft --maxDraftTreeSize 60` |
| MTP base engine | Vanilla LLM parameters plus `--specBase --maxVerifyTreeSize 4` |
| MTP draft engine | Vanilla LLM parameters plus `--specDraft --maxDraftTreeSize 4` |
| DFlash base engine | Vanilla LLM parameters plus `--specBase --maxVerifyTreeSize 16` |
| DFlash draft engine | Vanilla LLM parameters plus `--specDraft --maxDraftTreeSize 16` |

Use the batch size shown in the benchmark row. Thor and Jetson AGX Orin rows may
use batch `1` or `8`; Jetson Orin NX and Orin Nano rows are generally batch `1`.
For INT4 runs on Orin, follow the export docs but use externalized INT4 weights:
`--externalize-weights int4_ffn` for dense checkpoints and
`--externalize-weights int4_ffn int4_moe` for MoE checkpoints.

For speculative decoding, follow the exact export layouts in
[Speculative Decoding](../examples/speculative-decoding.md): EAGLE3 uses a base
and draft export, MTP uses the MTP base and `mtp_draft` export, and DFlash uses
the paired DFlash base and draft export. For DFlash, use the linear DFlash base
export for `--specDraftTopK 1`; use the tree-base export only for DDTree runs.

### 3. Runtime Benchmark Specs

Use `llm_inference` with the generated datasets and `--dumpProfile` to collect
runtime prefill, generation, visual, memory, and speculative-decoding metrics in
the profile JSON. Use these common runtime settings:

| Workload | `llm_inference` settings |
|----------|--------------------------|
| LLM runtime benchmark | `--inputFile ${DATASET_DIR}/mtbench/mtbench_dataset.json` |
| VLM runtime benchmark | `--inputFile ${DATASET_DIR}/coco/dataset.json --multimodalEngineDir <visual_engine_dir>` |
| All runtime benchmarks | `--batchSize <batch>` `--warmup 10` `--dumpProfile --profileOutputFile <profile.json>` |
| EAGLE3 runtime | Common settings plus `--specDecode --specVerifySize 60` |
| MTP runtime | Common settings plus `--specDecode --specDraftTopK 1 --specDraftStep 3 --specVerifySize 4` |
| DFlash linear runtime | Common settings plus `--specDecode --specDraftTopK 1 --specDraftStep 1 --specVerifySize 16` |
| DFlash DDTree runtime | Follow the DFlash guide; use `--specDraftTopK > 1` with the tree-base export |

For Qwen3.5 DFlash inputs, set `"enable_thinking": true`; for Qwen3 DFlash
inputs, set `"enable_thinking": false`. These settings match the paired
HuggingFace generation behavior used for DFlash validation.

For synthetic component timing, run `llm_bench` on the same engines:

| Component | `llm_bench` settings |
|-----------|----------------------|
| LLM prefill | `--mode prefill --batchSize <batch> --inputLen 2048 --warmup 3 --iterations 10 --profile` |
| LLM decode | `--mode decode --batchSize <batch> --pastKVLen 2048 --warmup 3 --iterations 10 --profile` |
| Visual encoder | `--mode visual --imageSize 1024x2048 --warmup 3 --iterations 10 --profile` |
| Spec draft prefill | `--mode spec_draft_prefill --batchSize <batch> --inputLen 2048 --warmup 3 --iterations 10 --profile` |
| Spec draft proposal | `--mode spec_draft_proposal --batchSize <batch> --draftTreeSize <draft_tree_size> --pastKVLen 2048 --warmup 3 --iterations 10 --profile` |
| Spec verify | `--mode spec_verify --batchSize <batch> --verifyTreeSize <verify_tree_size> --pastKVLen 2048 --warmup 3 --iterations 10 --profile` |

Use `draftTreeSize` / `verifyTreeSize` values of `60` for EAGLE3, `4` for MTP,
and `16` for linear DFlash.

## v0.9.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.9.0 &nbsp;|&nbsp; **JetPack:** 7.2 &nbsp;|&nbsp; **Devices:** Jetson AGX Thor, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, Jetson Orin Nano 8GB

> **Decode throughput:** Runtime `Decode (tok/s)` reports generated tokens per second for vanilla decoding and overall accepted-token throughput for speculative decoding. `llm_bench` BS=8 decode throughput is reported as aggregate batch throughput.

### `llm_bench` Component Performance

These rows report synthetic `llm_bench` prefill and decode measurements at the batch sizes shown.

| Platform | Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill E2E (ms) | Prefill (tok/s) | Decode Past KV Len | Decode (tok/s) |
|----------|-------|------|------|-----------|:-----:|----------------:|-----------------:|----------------:|-------------------:|---------------:|
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 167.3 | 12,238.7 | 2,048 | 76.0 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,037 | 1,974.9 | 2,048 | 272.0 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 1,150.8 | 1,779.6 | 2,048 | 67.3 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 5,352.8 | 382.6 | 2,048 | 433.6 |
| Jetson AGX Thor | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 163.6 | 12,516.2 | 2,048 | 71.2 |
| Jetson AGX Thor | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 992.9 | 2,062.6 | 2,048 | 266.4 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 20.3 | 101,070.8 | 2,048 | 245.4 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 234.6 | 8,730.5 | 2,048 | 715.2 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 29.9 | 68,527.5 | 2,048 | 135.5 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 341.7 | 5,993.2 | 2,048 | 519.2 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 137.0 | 14,952.3 | 2,048 | 84.8 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 970.8 | 2,109.6 | 2,048 | 249.6 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 2,048 | 64.7 | 31,645.4 | 2,048 | 73.7 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 2,048 | 810.0 | 2,528.5 | 2,048 | 349.6 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 108.4 | 18,896.8 | 2,048 | 44.4 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,214.5 | 1,686.2 | 2,048 | 252.0 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 30.5 | 67,255.9 | 2,048 | 135.7 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 343.1 | 5,969.3 | 2,048 | 561.6 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 64.3 | 31,859.4 | 2,048 | 73.4 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 813.9 | 2,516.2 | 2,048 | 350.4 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 108.4 | 18,896.6 | 2,048 | 44.8 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,219.6 | 1,679.2 | 2,048 | 250.4 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 36.0 | 56,959.2 | 2,048 | 229.1 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 348.3 | 5,880.3 | 2,048 | 1,205.6 |
| Jetson AGX Thor | Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 36.2 | 56,525.9 | 2,048 | 228.3 |
| Jetson AGX Thor | Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 347.9 | 5,887.2 | 2,048 | 1,206.4 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 463.3 | 4,420.5 | 2,048 | 14.6 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 4,249.2 | 482.0 | 2,048 | 94.4 |
| Jetson AGX Thor | Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 461.9 | 4,433.4 | 2,048 | 14.5 |
| Jetson AGX Thor | Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 4,268.3 | 479.8 | 2,048 | 95.2 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 45.3 | 45,230 | 2,048 | 122.4 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 434.8 | 4,710.4 | 2,048 | 756.0 |
| Jetson AGX Thor | Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 45.7 | 44,801 | 2,048 | 122.6 |
| Jetson AGX Thor | Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 436.3 | 4,694.2 | 2,048 | 757.6 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 101.1 | 20,248.4 | 2,048 | 68.0 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 996.5 | 2,055.2 | 2,048 | 395.2 |
| Jetson AGX Thor | Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 101.9 | 20,106.1 | 2,048 | 67.7 |
| Jetson AGX Thor | Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 992.0 | 2,064.5 | 2,048 | 394.4 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 139.0 | 14,738.3 | 2,048 | 40.1 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,406 | 1,456.6 | 2,048 | 261.6 |
| Jetson AGX Thor | Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 138.4 | 14,800 | 2,048 | 39.7 |
| Jetson AGX Thor | Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,414.5 | 1,447.8 | 2,048 | 260.0 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 195.0 | 10,501.2 | 2,048 | 84.1 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,212.8 | 1,688.7 | 2,048 | 257.6 |

### Runtime Performance Dashboard

Runtime rows are split by device and include batch size, prefill sequence length/time, visual encoder timing for VLMs, decode throughput, speculative acceptance rate, and peak GPU memory.

#### Jetson AGX Thor

| Model | Kind | Mode | Precision | Dataset | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|------------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | - | 1 | 66 | 73.9 | 891.0 | - | - | - | 77.3 | - | 18,754 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | - | 8 | 470 | 157.1 | 2,992.8 | - | - | - | 225.9 | - | 18,721 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | - | 1 | 66 | 35.4 | 1,858.9 | - | - | - | 67.9 | - | 3,546 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | - | 8 | 470 | 147.3 | 3,192.7 | - | - | - | 359.5 | - | 3,556 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 1,699 | 159.4 | 10,658 | 127.3 | 1,664 | 13,070.9 | 72.6 | - | 18,928 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 12,136 | 806.8 | 15,042.2 | 932.8 | 11,886 | 12,742.1 | 183.5 | - | 18,979 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 376 | 177.4 | 2,121.1 | 32.8 | 349 | 10,661.3 | 84.6 | 4.91 | 3,320 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,688 | 1,333.9 | 2,014.8 | 260.5 | 2,495 | 9,575.1 | 126.7 | 4.84 | 3,332 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | COCO | 1 | 376 | 23.2 | 16,193.8 | 32.7 | 349 | 10,685 | 188.4 | 4.77 | 4,260 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,688 | 180.0 | 14,931.8 | 259.5 | 2,495 | 9,613.4 | 531.1 | 4.73 | 4,288 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | - | 1 | 61 | 8.0 | 7,672.9 | - | - | - | 303.1 | - | 966 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | - | 8 | 437 | 19.6 | 22,337.1 | - | - | - | 1,569.5 | - | 961 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | - | 1 | 61 | 12.0 | 5,110 | - | - | - | 154.2 | - | 1,771 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | - | 8 | 437 | 29.6 | 14,790.2 | - | - | - | 828.7 | - | 1,806 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | - | 1 | 61 | 10.0 | 6,143.8 | - | - | - | 351.6 | 3.04 | 1,355 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | - | 8 | 437 | 27.7 | 15,768.7 | - | - | - | 950.4 | 3.06 | 1,365 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | - | 1 | 61 | 64.1 | 955.4 | - | - | - | 85.8 | - | 14,305 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | - | 8 | 437 | 263.7 | 1,657.9 | - | - | - | 232.1 | - | 14,314 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | - | 1 | 61 | 57.1 | 1,071.8 | - | - | - | 90.4 | - | 17,112 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | - | 8 | 437 | 107.1 | 4,081.1 | - | - | - | 252.8 | - | 17,092 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | - | 1 | 57 | 33.5 | 1,710.1 | - | - | - | 79.6 | - | 1,819 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | - | 8 | 409 | 229.8 | 1,778 | - | - | - | 328.6 | - | 1,808 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | - | 1 | 57 | 20.5 | 2,790 | - | - | - | 80.5 | - | 3,138 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | - | 8 | 409 | 53.9 | 7,586.7 | - | - | - | 494.9 | - | 3,151 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | - | 1 | 57 | 20.6 | 2,772.8 | - | - | - | 102.3 | 2.26 | 3,141 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | - | 8 | 409 | 53.8 | 7,597 | - | - | - | 462.1 | 2.24 | 3,148 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 56.0 | 1,093.1 | - | - | - | 49.3 | - | 3,157 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | - | 8 | 437 | 426.5 | 1,025.1 | - | - | - | 191.8 | - | 3,166 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | - | 1 | 61 | 30.4 | 2,011.8 | - | - | - | 46.8 | - | 5,361 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | - | 8 | 437 | 74.1 | 5,901.8 | - | - | - | 283.1 | - | 5,374 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 55.8 | 1,097.7 | - | - | - | 77.1 | 4.22 | 3,146 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 8 | 437 | 426.2 | 1,025.7 | - | - | - | 107.7 | 4.20 | 3,153 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | - | 1 | 61 | 26.7 | 2,291.6 | - | - | - | 160.5 | 4.24 | 4,513 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | - | 8 | 437 | 71.1 | 6,149.4 | - | - | - | 488.5 | 4.12 | 4,523 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | - | 1 | 61 | 29.5 | 2,073.7 | - | - | - | 87.1 | 3.24 | 5,363 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | - | 8 | 437 | 74.8 | 5,848.2 | - | - | - | 378.5 | 3.13 | 5,379 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 10.7 | 27,311.9 | 11.7 | 266 | 22,678.4 | 151.9 | - | 1,845 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 53.2 | 39,302.3 | 75.6 | 1,896 | 25,068 | 698.8 | - | 1,853 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 85.2 | 3,433.5 | 11.7 | 266 | 22,731.2 | 79.0 | - | 1,847 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 613.0 | 3,408 | 76.0 | 1,896 | 24,964.3 | 273.1 | - | 1,850 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 20.4 | 14,327 | 12.2 | 265 | 21,831.3 | 79.4 | - | 3,182 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 107.0 | 19,531 | 76.0 | 1,896 | 24,951.1 | 377.9 | - | 3,193 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 85.2 | 3,431.1 | 11.7 | 266 | 22,640.1 | 143.0 | 4.95 | 1,847 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,089 | 613.3 | 3,406.2 | 76.2 | 1,896 | 24,895.9 | 212.9 | 4.91 | 1,846 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | COCO | 1 | 292 | 18.3 | 15,978.8 | 11.7 | 265 | 22,742.1 | 258.3 | 4.48 | 2,660 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,089 | 105.0 | 19,894.8 | 75.5 | 1,896 | 25,132.1 | 676.7 | 4.57 | 2,680 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 156.5 | 1,869.1 | 15.8 | 265 | 16,778.3 | 48.9 | - | 3,204 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 1,113.2 | 1,876.6 | 108.1 | 1,896 | 17,546.9 | 184.5 | - | 3,191 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 30.2 | 9,681.1 | 15.8 | 265 | 16,854.4 | 47.1 | - | 5,403 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 175.5 | 11,904.6 | 108.5 | 1,896 | 17,474.1 | 246.9 | - | 5,424 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 156.5 | 1,868.5 | 15.7 | 265 | 16,899 | 70.8 | 3.93 | 3,199 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,089 | 1,114.1 | 1,875.1 | 108.1 | 1,896 | 17,549.2 | 99.5 | 3.93 | 3,193 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 9.4 | 31,415 | 3.9 | 265 | 68,378.8 | 234.6 | - | 1,239 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 51.6 | 41,042.8 | 26.9 | 1,896 | 70,417.5 | 976.7 | - | 1,223 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 8.1 | 36,604.8 | 3.9 | 266 | 68,283.8 | 377.4 | 2.09 | 1,016 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 49.9 | 42,408 | 26.9 | 1,896 | 70,395.1 | 1,118.1 | 2.09 | 1,135 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | - | 1 | 62 | 9.2 | 6,695.1 | - | - | - | 235.1 | - | 1,184 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | - | 8 | 441 | 26.2 | 16,871.5 | - | - | - | 1,230.9 | - | 1,153 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 567.0 | 522.8 | 14.6 | 266 | 18,137.7 | 15.8 | - | 8,975 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 4,498 | 470.8 | 102.5 | 1,896 | 18,502 | 56.9 | - | 8,982 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 104.0 | 2,851.2 | 14.5 | 265 | 18,265.8 | 14.8 | - | 16,040 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 654.1 | 3,237.4 | 102.6 | 1,896 | 18,486.3 | 76.0 | - | 16,081 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 567.0 | 522.9 | 14.6 | 265 | 18,134 | 27.9 | 2.85 | 8,972 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 4,498.2 | 470.8 | 102.3 | 1,896 | 18,538.5 | 92.3 | 2.84 | 9,006 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 94.8 | 3,126.6 | 14.7 | 265 | 18,122.6 | 37.6 | 2.84 | 14,309 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 645.5 | 3,280.4 | 102.8 | 1,896 | 18,448.6 | 152.9 | 2.85 | 14,353 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 103.6 | 2,862.1 | 14.6 | 265 | 18,239.9 | 21.7 | 2.59 | 16,063 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 656.0 | 3,227.8 | 103.0 | 1,896 | 18,416.8 | 55.6 | 2.51 | 16,077 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 172.1 | 358.9 | - | - | - | 15.8 | - | 8,932 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 1,717 | 256.9 | - | - | - | 60.2 | - | 8,938 |
| Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | - | 1 | 62 | 90.5 | 682.2 | - | - | - | 14.7 | - | 15,983 |
| Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | - | 8 | 441 | 276.6 | 1,594.7 | - | - | - | 88.5 | - | 16,018 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 13.8 | 21,423.1 | 11.2 | 265 | 23,721.7 | 125.3 | - | 2,186 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 69.4 | 30,536.6 | 72.8 | 1,896 | 26,044.4 | 475.1 | - | 2,188 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 42.0 | 7,067.5 | 11.1 | 265 | 23,820.9 | 131.0 | 2.43 | 1,579 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 347.2 | 6,099.6 | 73.3 | 1,896 | 25,874.3 | 483.5 | 2.41 | 1,581 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 11.4 | 26,036.1 | 11.2 | 265 | 23,798.3 | 254.5 | 2.40 | 1,475 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 65.4 | 32,395.7 | 72.5 | 1,896 | 26,172.8 | 730.4 | 2.33 | 1,502 |
| Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | - | 1 | 62 | 13.9 | 4,453.7 | - | - | - | 123.5 | - | 2,134 |
| Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | - | 8 | 441 | 34.5 | 12,782.4 | - | - | - | 729.5 | - | 2,161 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 1 | 296 | 109.8 | 2,700.1 | 14.5 | 265 | 18,333.6 | 47.9 | - | 15,868 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 8 | 2,118 | 458.2 | 4,621.5 | 101.6 | 1,896 | 18,655.7 | 196.3 | - | 15,872 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | - | 1 | 62 | 66.6 | 928.0 | - | - | - | 48.2 | - | 15,829 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | - | 8 | 441 | 212.2 | 2,079.4 | - | - | - | 203.1 | - | 15,823 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 98.6 | 3,007 | 11.0 | 265 | 24,210.2 | 67.5 | - | 1,731 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 740.8 | 2,858.4 | 72.7 | 1,896 | 26,080.2 | 243.1 | - | 1,748 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 26.1 | 11,347 | 10.9 | 265 | 24,398 | 69.4 | - | 3,635 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 142.1 | 14,906.9 | 72.9 | 1,896 | 26,020.4 | 271.3 | - | 3,636 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 98.6 | 3,005.8 | 10.9 | 265 | 24,308.6 | 83.1 | 2.55 | 1,838 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 840.9 | 2,518.2 | 73.1 | 1,896 | 25,935.5 | 278.7 | 2.55 | 1,835 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 22.6 | 13,107.3 | 11.0 | 266 | 24,251.3 | 144.1 | 2.55 | 2,749 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 139.6 | 15,171.4 | 73.2 | 1,896 | 25,895.5 | 439.8 | 2.50 | 2,774 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 26.3 | 11,277.7 | 11.2 | 265 | 23,704.3 | 73.8 | 2.27 | 3,631 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 143.6 | 14,750.3 | 73.4 | 1,896 | 25,828.5 | 169.0 | 2.29 | 3,627 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 37.4 | 1,651.5 | - | - | - | 68.3 | - | 1,691 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 287.5 | 1,534.3 | - | - | - | 274.0 | - | 1,674 |
| Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | - | 1 | 62 | 24.2 | 2,548.5 | - | - | - | 69.1 | - | 3,571 |
| Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | - | 8 | 441 | 69.1 | 6,382.2 | - | - | - | 386.2 | - | 3,561 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 173.8 | 1,705.9 | 14.6 | 265 | 18,164.8 | 42.5 | - | 2,896 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 1,354.8 | 1,563 | 101.6 | 1,896 | 18,664.1 | 161.7 | - | 2,922 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 37.6 | 7,895.3 | 14.5 | 265 | 18,291.2 | 40.8 | - | 6,137 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 211.4 | 10,016 | 101.1 | 1,896 | 18,757.2 | 187.2 | - | 6,150 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 173.8 | 1,705.4 | 14.6 | 265 | 18,240.9 | 59.2 | 2.77 | 2,901 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 1,355.5 | 1,562.2 | 100.9 | 1,896 | 18,798.2 | 225.0 | 2.78 | 2,920 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 31.9 | 9,294 | 14.7 | 265 | 18,097.9 | 96.7 | 2.70 | 4,754 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 204.8 | 10,338.5 | 101.9 | 1,896 | 18,616.8 | 396.5 | 2.73 | 4,768 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 37.6 | 7,885.8 | 14.5 | 265 | 18,263.8 | 47.1 | 2.42 | 6,145 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 212.6 | 9,961.9 | 101.7 | 1,896 | 18,653.1 | 132.4 | 2.34 | 6,161 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 60.7 | 1,017.2 | - | - | - | 42.3 | - | 2,833 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 521.4 | 846.1 | - | - | - | 167.2 | - | 2,865 |
| Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | - | 1 | 62 | 34.2 | 1,803.9 | - | - | - | 40.8 | - | 6,074 |
| Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | - | 8 | 441 | 92.4 | 4,774.8 | - | - | - | 242.0 | - | 6,104 |
| Qwen3.6-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 567.0 | 522.9 | 14.7 | 265 | 18,107.5 | 27.3 | 2.78 | 8,969 |
| Qwen3.6-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 4,498.4 | 470.7 | 102.1 | 1,896 | 18,573.5 | 90.8 | 2.81 | 8,976 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 96.8 | 3,062.1 | 14.7 | 265 | 18,093.5 | 37.1 | 2.82 | 14,328 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 645.8 | 3,279 | 102.2 | 1,896 | 18,559.2 | 143.3 | 2.81 | 14,335 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 88.8 | 3,339.5 | 14.5 | 265 | 18,334.9 | 87.6 | - | 19,472 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 229.3 | 9,236.3 | 101.4 | 1,896 | 18,696.7 | 251.1 | - | 19,468 |
| Qwen3.6-35B-A3B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 1 | 296 | 89.7 | 3,304.7 | 14.5 | 265 | 18,318.7 | 42.2 | 1.62 | 19,389 |
| Qwen3.6-35B-A3B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | COCO | 8 | 2,118 | 228.8 | 9,253.9 | 100.5 | 1,896 | 18,862 | 86.9 | 1.59 | 19,474 |

#### Jetson AGX Orin (64GB)

| Model | Kind | Mode | Precision | Dataset | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|------------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 376 | 260.9 | 1,442.1 | 89.2 | 349 | 3,915.4 | 64.4 | 4.81 | 12,141 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,688 | 1,936 | 1,388.2 | 635.0 | 2,495 | 3,928.8 | 85.4 | 4.78 | 14,078 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 14.8 | 4,124.1 | - | - | - | 186.4 | - | 1,931 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | - | 8 | 437 | 50.9 | 8,588.1 | - | - | - | 738.2 | - | 4,043 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 20.4 | 2,995.1 | - | - | - | 98.6 | - | 3,302 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | - | 8 | 437 | 134.2 | 3,257.9 | - | - | - | 465.5 | - | 5,602 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 20.5 | 2,992.9 | - | - | - | 128.2 | 3.18 | 3,395 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 8 | 437 | 134.8 | 3,243.1 | - | - | - | 215.3 | 3.18 | 6,714 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | - | 1 | 61 | 91.0 | 672.4 | - | - | - | 56.7 | - | 26,077 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | - | 8 | 437 | 389.2 | 1,123.2 | - | - | - | 163.3 | - | 25,304 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | - | 1 | 57 | 42.6 | 1,342.4 | - | - | - | 53.1 | - | 4,911 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | - | 8 | 409 | 323.6 | 1,262.8 | - | - | - | 225.8 | - | 8,119 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 65.7 | 931.2 | - | - | - | 32.5 | - | 8,221 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | - | 8 | 437 | 592.8 | 737.4 | - | - | - | 151.5 | - | 11,134 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 65.6 | 933.3 | - | - | - | 59.3 | 4.22 | 8,344 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 8 | 437 | 593.4 | 736.7 | - | - | - | 72.6 | 4.21 | 12,914 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 50.0 | 5,853 | 39.0 | 265 | 6,815.9 | 98.4 | - | 8,140 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 355.8 | 5,870.7 | 256.8 | 1,896 | 7,383.3 | 372.3 | - | 10,246 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 123.7 | 2,364 | 39.3 | 265 | 6,749.4 | 52.4 | - | 9,516 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 886.2 | 2,357.2 | 257.6 | 1,896 | 7,360.4 | 188.7 | - | 12,436 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 123.9 | 2,360.7 | 39.2 | 265 | 6,765.4 | 96.2 | 4.95 | 10,231 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,089 | 889.5 | 2,348.6 | 258.2 | 1,896 | 7,342.6 | 141.4 | 4.88 | 13,573 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 225.1 | 1,299.1 | 57.2 | 265 | 4,644.4 | 32.3 | - | 13,030 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 1,598.1 | 1,307.2 | 373.2 | 1,896 | 5,080.7 | 144.5 | - | 15,904 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 225.8 | 1,295.3 | 56.9 | 265 | 4,668.9 | 54.8 | 3.94 | 13,585 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,089 | 1,603.8 | 1,302.6 | 372.5 | 1,896 | 5,090.6 | 67.0 | 3.92 | 17,381 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 96.8 | 3,062.6 | 13.2 | 266 | 20,176.9 | 146.0 | - | 4,488 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 440.5 | 4,806.9 | 84.2 | 1,896 | 22,521 | 574.3 | - | 4,992 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 97.3 | 3,047.3 | 13.2 | 265 | 20,168.1 | 130.7 | 2.06 | 5,108 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 440.5 | 4,807.6 | 83.9 | 1,896 | 22,592 | 467.1 | 2.06 | 6,323 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 24.8 | 2,491.8 | - | - | - | 146.3 | - | 2,293 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 154.4 | 2,856.9 | - | - | - | 663.7 | - | 3,212 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 1,072.4 | 276.4 | 53.8 | 265 | 4,938.9 | 10.5 | - | 24,017 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 7,796.9 | 271.6 | 358.3 | 1,896 | 5,292.3 | 42.1 | - | 26,327 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 1,073 | 276.3 | 54.0 | 265 | 4,919 | 18.0 | 2.84 | 25,610 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 7,808.6 | 271.2 | 358.4 | 1,896 | 5,291 | 63.6 | 2.85 | 33,173 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 267.5 | 230.9 | - | - | - | 10.5 | - | 24,005 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 2,785.1 | 158.4 | - | - | - | 45.0 | - | 26,257 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 126.3 | 2,347.5 | 37.0 | 265 | 7,184.5 | 82.0 | - | 6,818 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 672.3 | 3,149.7 | 246.9 | 1,896 | 7,679.3 | 353.7 | - | 7,376 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 124.9 | 2,373.7 | 36.9 | 265 | 7,189.4 | 86.9 | 2.42 | 7,946 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 672.5 | 3,148.6 | 248.1 | 1,896 | 7,644.3 | 358.1 | 2.42 | 9,311 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 34.0 | 1,814.8 | - | - | - | 82.0 | - | 4,178 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 245.1 | 1,800.2 | - | - | - | 406.7 | - | 4,934 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 1 | 296 | 299.9 | 988.6 | 53.4 | 265 | 4,968.9 | 30.1 | - | 28,865 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 8 | 2,118 | 1,401.6 | 1,510.8 | 358.8 | 1,896 | 5,284.3 | 128.9 | - | 26,705 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | - | 1 | 62 | 116.8 | 528.8 | - | - | - | 31.1 | - | 35,722 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | - | 8 | 441 | 523.8 | 842.2 | - | - | - | 134.1 | - | 36,655 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 243.5 | 1,217.6 | 37.0 | 265 | 7,170.8 | 45.7 | - | 8,450 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 1,605 | 1,319.4 | 248.7 | 1,896 | 7,624.3 | 170.8 | - | 9,683 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 243.4 | 1,218.2 | 37.0 | 265 | 7,178.7 | 54.8 | 2.53 | 10,083 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 1,606.5 | 1,318.2 | 248.5 | 1,896 | 7,630.2 | 194.6 | 2.56 | 12,896 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 64.6 | 956.5 | - | - | - | 45.7 | - | 6,106 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 570.6 | 773.2 | - | - | - | 192.0 | - | 7,555 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 353.5 | 838.6 | 53.8 | 265 | 4,938.9 | 28.0 | - | 12,106 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 2,355.5 | 899.0 | 358.3 | 1,896 | 5,291.6 | 124.4 | - | 13,409 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 352.2 | 841.8 | 53.6 | 265 | 4,949.3 | 39.8 | 2.79 | 14,526 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 2,365.1 | 895.4 | 355.0 | 1,896 | 5,341.5 | 165.9 | 2.79 | 17,457 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 90.1 | 685.7 | - | - | - | 28.0 | - | 9,959 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | - | 8 | 441 | 852.5 | 517.5 | - | - | - | 130.4 | - | 11,237 |
| Qwen3.6-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 1,072.9 | 276.3 | 53.7 | 265 | 4,944.9 | 17.7 | 2.79 | 25,554 |
| Qwen3.6-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 8 | 2,118 | 7,793.2 | 271.7 | 358.2 | 1,896 | 5,293.5 | 65.2 | 2.79 | 33,225 |

#### Jetson Orin NX (16GB)

| Model | Kind | Mode | Precision | Dataset | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|------------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 376 | 535.5 | 702.6 | 208.5 | 349 | 1,675.2 | 33.0 | 4.76 | 9,161 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 18.9 | 3,237.4 | - | - | - | 117.4 | - | 1,937 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 35.4 | 1,729.3 | - | - | - | 60.4 | - | 3,250 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 35.4 | 1,728.4 | - | - | - | 73.5 | 3.19 | 3,315 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | - | 1 | 57 | 74.9 | 763.8 | - | - | - | 31.6 | - | 4,921 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 127.2 | 481.2 | - | - | - | 19.0 | - | 7,880 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 129.1 | 474.2 | - | - | - | 30.7 | 4.18 | 7,999 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 103.1 | 2,836.1 | 82.1 | 265 | 3,234.1 | 60.1 | - | 4,344 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 244.6 | 1,195.9 | 83.7 | 265 | 3,172.1 | 31.2 | - | 5,903 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 243.0 | 1,203.3 | 83.1 | 265 | 3,195.6 | 54.2 | 4.92 | 6,362 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 462.1 | 632.9 | 120.0 | 265 | 2,211.6 | 18.9 | - | 9,065 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 292 | 498.2 | 587.0 | 121.3 | 265 | 2,188 | 28.5 | 3.94 | 9,518 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 140.5 | 2,110.8 | 26.8 | 265 | 9,891.9 | 88.7 | - | 2,576 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 140.2 | 2,114.4 | 26.9 | 265 | 9,861.5 | 74.5 | 2.07 | 3,208 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 37.3 | 1,655.6 | - | - | - | 88.9 | - | 2,257 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 199.6 | 1,484.9 | 78.8 | 265 | 3,367.7 | 48.9 | - | 4,609 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 199.3 | 1,487.2 | 77.6 | 265 | 3,420.9 | 47.7 | 2.41 | 5,698 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 55.8 | 1,107.6 | - | - | - | 48.5 | - | 4,145 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 424.5 | 698.3 | 79.1 | 265 | 3,355.9 | 26.4 | - | 6,303 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 422.6 | 701.5 | 78.8 | 265 | 3,370 | 29.5 | 2.55 | 7,782 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 112.5 | 549.2 | - | - | - | 26.5 | - | 6,113 |

#### Jetson Orin Nano (8GB)

| Model | Kind | Mode | Precision | Dataset | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|------------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 28.7 | 2,133 | - | - | - | 72.8 | - | 1,889 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | - | 1 | 61 | 63.1 | 970.5 | - | - | - | 37.4 | - | 3,234 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | - | 1 | 61 | 63.2 | 967.8 | - | - | - | 41.2 | 3.19 | 3,328 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 190.9 | 1,532.2 | 151.1 | 265 | 1,756.4 | 36.9 | - | 4,380 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 251.9 | 1,176.8 | 49.2 | 265 | 5,400.2 | 55.3 | - | 2,603 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | COCO | 1 | 296 | 251.8 | 1,177.4 | 49.1 | 265 | 5,404.1 | 45.0 | 2.07 | 3,202 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 65.6 | 941.7 | - | - | - | 55.4 | - | 2,316 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 360.4 | 822.6 | 143.2 | 265 | 1,853.8 | 29.9 | - | 4,621 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | - | 1 | 62 | 99.5 | 621.0 | - | - | - | 30.0 | - | 4,176 |

### v0.9.0 Collection Method

- Engines were built from exported v0.9.0 ONNX artifacts using the build limits in [Export and Build Specs](#2-export-and-build-specs).
- Runtime throughput was collected with `llm_inference --warmup 10 --dumpProfile --profileOutputFile <profile.json>` using benchmark JSON inputs for each model family.
- Synthetic component timing was collected with `llm_bench --warmup 3 --iterations 10`; prefill uses `--inputLen 2048` and decode uses `--pastKVLen 2048`.
- Jetson AGX Thor runs include NVFP4 and INT4 entries. Jetson AGX Orin, Orin NX, and Orin Nano run the externalized INT4 entries supported by each memory target.

---

## v0.8.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.8.0 &nbsp;|&nbsp; **JetPack:** 7.2 &nbsp;|&nbsp; **Source:** v0.8.0 release benchmark outputs &nbsp;|&nbsp; **Devices:** Jetson AGX Thor, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, Jetson Orin Nano 8GB

> **Limitation:** v0.8.0 has a uniform performance regression across the benchmarked release devices. This regression is fixed in v0.9.0, so use v0.9.0 or later for current performance expectations.

### `llm_bench` Commands

The v0.8.0 release benchmarks use `llm_bench` for synthetic component timing. The release run used `--warmup=2`, `--iterations=10`, and `--profile`; replace paths and lengths with the engine directory and shape listed in the table.

```bash
# Prefill throughput
./build/examples/llm/llm_bench \
    --engineDir <llm_engine_dir> \
    --mode prefill \
    --batchSize <batch_size> \
    --inputLen <input_len> \
    --warmup 2 \
    --iterations 10 \
    --profile

# Decode throughput
./build/examples/llm/llm_bench \
    --engineDir <llm_engine_dir> \
    --mode decode \
    --batchSize <batch_size> \
    --pastKVLen <past_kv_len> \
    --warmup 2 \
    --iterations 10 \
    --profile

# Speculative decoding component timing
./build/examples/llm/llm_bench --engineDir <engine_dir> --mode spec_draft_prefill --batchSize <batch_size> --inputLen <input_len> --warmup 2 --iterations 10 --profile
./build/examples/llm/llm_bench --engineDir <engine_dir> --mode spec_draft_proposal --batchSize <batch_size> --draftTreeSize <draft_tree_size> --pastKVLen <past_kv_len> --warmup 2 --iterations 10 --profile
./build/examples/llm/llm_bench --engineDir <engine_dir> --mode spec_verify --batchSize <batch_size> --verifyTreeSize <verify_tree_size> --pastKVLen <past_kv_len> --warmup 2 --iterations 10 --profile

# Visual encoder timing
./build/examples/llm/llm_bench --engineDir <visual_engine_dir> --mode visual --imageSize <height>x<width> --warmup 2 --iterations 10 --profile
```

### `llm_bench` Prefill Performance (Jetson AGX Thor Only)

These are the parsed synthetic `llm_bench_prefill_*` results in the v0.8.0 release benchmark outputs. Only the Jetson AGX Thor data contains parsed `llm_bench_prefill_e2e_time_ms` and `llm_bench_prefill_tokens_per_sec` values. The AGX Orin, Orin NX, and Orin Nano data includes runtime prefill metrics in the dashboard below, but does not include those synthetic `llm_bench` prefill e2e/tok/s fields.

| Platform | Model | Kind | Mode | Precision | Batch | Input Len | Prefill E2E (ms) | Prefill (tok/s) |
|----------|-------|------|------|-----------|:-----:|----------:|-----------------:|----------------:|
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 228.1 | 8,978.9 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,385.7 | 1,478.0 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 1,152.7 | 1,776.7 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 5,347.0 | 383.0 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 22.4 | 91,469.3 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 248.5 | 8,241.0 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 32.3 | 63,441.6 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 356.8 | 5,739.6 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 2,048 | 67.1 | 30,539.5 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 2,048 | 823.8 | 2,486.0 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 111.0 | 18,444.1 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,245.9 | 1,643.8 |
| Jetson AGX Thor | Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 36.6 | 55,904.4 |
| Jetson AGX Thor | Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 349.0 | 5,868.1 |
| Jetson AGX Thor | Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 463.3 | 4,420.5 |
| Jetson AGX Thor | Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 4,268.6 | 479.8 |
| Jetson AGX Thor | Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 46.1 | 44,458.0 |
| Jetson AGX Thor | Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 435.6 | 4,701.8 |
| Jetson AGX Thor | Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 101.2 | 20,237.3 |
| Jetson AGX Thor | Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 997.1 | 2,053.9 |
| Jetson AGX Thor | Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 1 | 2,048 | 138.4 | 14,795.8 |
| Jetson AGX Thor | Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,411.1 | 1,451.3 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 32.6 | 62,763.4 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 358.0 | 5,721.4 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 68.5 | 29,908.3 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 833.3 | 2,457.8 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 112.2 | 18,254.6 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,239.3 | 1,652.6 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 36.5 | 56,042.9 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 349.2 | 5,864.1 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 459.2 | 4,460.3 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 4,201.7 | 487.4 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 46.0 | 44,568.0 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 438.0 | 4,676.0 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 101.2 | 20,235.5 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 996.5 | 2,055.2 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 138.7 | 14,766.6 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,399.9 | 1,462.9 |
| Jetson AGX Thor | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 223.6 | 9,157.3 |
| Jetson AGX Thor | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,355.9 | 1,510.4 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 2,048 | 30.6 | 66,929.4 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 2,048 | 356.9 | 5,738.5 |
| Jetson AGX Thor | Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 2,048 | 109.2 | 18,754.4 |
| Jetson AGX Thor | Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 2,048 | 1,241.7 | 1,649.4 |
| Jetson AGX Thor | Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 81.8 | 25,038.0 |
| Jetson AGX Thor | Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 984.0 | 2,081.3 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 64.6 | 31,680.9 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 833.8 | 2,456.2 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 34.9 | 58,600.2 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 348.8 | 5,872.0 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 454.5 | 4,505.6 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 4,209.1 | 486.6 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 43.0 | 47,649.3 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 433.9 | 4,719.7 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 99.2 | 20,648.3 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 994.7 | 2,058.9 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 2,048 | 134.2 | 15,265.9 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,048 | 1,399.6 | 1,463.3 |

### Runtime Performance Dashboard

All v0.8.0 runtime entries below were benchmarked under JetPack 7.2 and are split by device.

#### Jetson AGX Thor

| Model | Kind | Mode | Precision | Batch | Runtime Prefill (ms) | Runtime Prefill Tok/Run | Runtime Prefill (tok/s) | ViT (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|---------------------:|------------------------:|------------------------:|---------:|------------:|------------:|-------------------:|------------:|-------------:|
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 104.8 | 383 | 3,653.0 | - | - | - | 72.5 | - | 19,987 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 508.9 | 3,062 | 6,016.5 | - | - | - | 180.0 | - | 19,975 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 120.8 | 383 | 3,169.5 | - | - | - | 66.4 | - | 3,538 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 933.8 | 3,062 | 3,279.1 | - | - | - | 302.9 | - | 3,548 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 22.7 | 370 | 16,296.0 | - | - | - | 194.0 | - | 773 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 238.9 | 2,959 | 12,383.9 | - | - | - | 329.6 | - | 875 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 13.4 | 370 | 27,533.2 | - | - | - | 192.5 | - | 957 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 100.5 | 2,959 | 29,449.6 | - | - | - | 359.8 | - | 998 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 49.8 | 370 | 7,423.1 | - | - | - | 119.7 | - | 1,067 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 589.9 | 2,959 | 5,015.4 | - | - | - | 276.6 | - | 1,066 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 18.4 | 370 | 20,067.5 | - | - | - | 118.9 | - | 1,884 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 142.9 | 2,959 | 20,704.1 | - | - | - | 302.4 | - | 1,782 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 140.2 | 370 | 2,638.4 | - | - | - | 75.1 | - | 14,305 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 1,461.4 | 2,959 | 2,024.6 | - | - | - | 161.7 | - | 14,313 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 111.9 | 364 | 3,251.6 | - | - | - | 66.2 | - | 1,846 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 1,413.0 | 2,911 | 2,060.0 | - | - | - | 173.7 | - | 1,813 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 31.5 | 364 | 11,568.8 | - | - | - | 67.1 | - | 3,168 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 283.9 | 2,911 | 10,251.5 | - | - | - | 216.3 | - | 3,148 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 200.3 | 370 | 1,846.7 | - | - | - | 43.7 | - | 3,195 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 2,546.9 | 2,959 | 1,161.7 | - | - | - | 126.8 | - | 3,260 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 42.8 | 370 | 8,640.5 | - | - | - | 42.3 | - | 5,356 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 426.8 | 2,959 | 6,933.1 | - | - | - | 157.4 | - | 5,376 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 30.7 | 377 | 12,271.7 | - | - | - | 214.5 | - | 951 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 370.8 | 3,013 | 8,126.8 | - | - | - | 717.8 | - | 962 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 1 | 13.8 | 377 | 27,269.9 | - | - | - | 216.2 | - | 1,176 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | NVFP4 | 8 | 114.6 | 3,013 | 26,298.4 | - | - | - | 926.5 | - | 1,172 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 713.9 | 377 | 527.6 | - | - | - | 15.4 | - | 8,927 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 10,102.2 | 3,013 | 298.3 | - | - | - | 54.6 | - | 8,940 |
| Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 1 | 130.5 | 377 | 2,887.1 | - | - | - | 14.3 | - | 15,990 |
| Qwen3.5-27B-LLM | LLM | Vanilla | NVFP4 | 8 | 1,356.3 | 3,013 | 2,221.6 | - | - | - | 72.9 | - | 16,044 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 56.6 | 377 | 6,650.8 | - | - | - | 118.8 | - | 1,443 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 690.0 | 3,013 | 4,366.8 | - | - | - | 459.4 | - | 1,446 |
| Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 1 | 18.8 | 377 | 19,993.5 | - | - | - | 118.8 | - | 2,131 |
| Qwen3.5-2B-LLM | LLM | Vanilla | NVFP4 | 8 | 148.7 | 3,013 | 20,261.3 | - | - | - | 601.0 | - | 2,138 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | 1 | 118.7 | 377 | 3,173.7 | - | - | - | 46.5 | - | 15,847 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | 8 | 973.7 | 3,013 | 3,094.7 | - | - | - | 175.0 | - | 15,840 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 128.5 | 377 | 2,931.7 | - | - | - | 64.9 | - | 1,707 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 1,660.6 | 3,013 | 1,814.5 | - | - | - | 221.4 | - | 1,702 |
| Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 1 | 34.4 | 377 | 10,939.3 | - | - | - | 64.8 | - | 3,567 |
| Qwen3.5-4B-LLM | LLM | Vanilla | NVFP4 | 8 | 305.0 | 3,013 | 9,880.3 | - | - | - | 303.0 | - | 3,604 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 219.5 | 377 | 1,715.8 | - | - | - | 41.2 | - | 2,863 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 3,031.4 | 3,013 | 994.0 | - | - | - | 144.6 | - | 2,866 |
| Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 1 | 46.4 | 377 | 8,112.3 | - | - | - | 39.4 | - | 6,110 |
| Qwen3.5-9B-LLM | LLM | Vanilla | NVFP4 | 8 | 449.7 | 3,013 | 6,700.4 | - | - | - | 181.2 | - | 6,108 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 38.2 | 283 | 7,399.3 | 12.6 | 263 | 20,790.0 | 119.0 | - | 1,422 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 321.0 | 2,196 | 6,840.7 | 85.9 | 2,036 | 23,696.7 | 273.6 | - | 1,409 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 14.8 | 283 | 19,152.2 | 12.6 | 262 | 20,790.0 | 117.8 | - | 1,867 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 92.1 | 2,196 | 23,828.6 | 85.6 | 2,039 | 23,809.5 | 309.2 | - | 1,910 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 89.2 | 283 | 3,169.7 | 13.0 | 262 | 20,161.3 | 67.0 | - | 1,899 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 775.3 | 2,196 | 2,832.1 | 85.8 | 2,039 | 23,753.0 | 164.2 | - | 1,914 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 28.4 | 283 | 9,963.3 | 12.7 | 262 | 20,746.9 | 66.6 | - | 3,228 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 186.2 | 2,196 | 11,791.3 | 85.5 | 2,036 | 23,809.5 | 181.1 | - | 3,242 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 155.8 | 283 | 1,813.9 | 17.3 | 263 | 15,174.5 | 44.1 | - | 3,283 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 1,354.5 | 2,196 | 1,621.0 | 119.6 | 2,037 | 17,035.8 | 127.3 | - | 3,243 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 38.4 | 283 | 7,354.1 | 17.3 | 262 | 15,151.5 | 41.9 | - | 5,484 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 260.7 | 2,196 | 8,422.7 | 119.5 | 2,037 | 17,035.8 | 150.5 | - | 5,482 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 21.4 | 287 | 13,382.7 | 4.3 | 262 | 60,975.6 | 209.8 | - | 1,045 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 191.0 | 2,227 | 11,660.9 | 30.6 | 2,041 | 66,666.7 | 593.7 | - | 1,085 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 10.0 | 287 | 28,729.9 | 4.3 | 262 | 60,975.6 | 210.9 | - | 1,260 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 63.6 | 2,227 | 34,989.3 | 30.6 | 2,042 | 66,666.7 | 758.0 | - | 1,268 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 558.3 | 287 | 513.4 | 16.2 | 263 | 16,181.2 | 15.5 | - | 9,128 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 5,186.5 | 2,227 | 429.4 | 114.6 | 2,039 | 17,793.6 | 54.7 | - | 9,011 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 103.6 | 287 | 2,765.5 | 16.1 | 262 | 16,286.6 | 14.3 | - | 16,087 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 744.7 | 2,227 | 2,990.2 | 114.4 | 2,040 | 17,825.3 | 74.3 | - | 16,119 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 41.8 | 287 | 6,853.9 | 11.7 | 262 | 22,321.4 | 115.9 | - | 1,562 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 354.2 | 2,227 | 6,286.8 | 83.3 | 2,037 | 24,449.9 | 419.2 | - | 1,639 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 14.3 | 287 | 20,005.7 | 11.7 | 262 | 22,471.9 | 117.7 | - | 2,213 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 79.1 | 2,227 | 28,157.1 | 82.1 | 2,037 | 24,813.9 | 450.7 | - | 2,228 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 109.8 | 287 | 2,609.4 | 16.0 | 262 | 16,366.6 | 46.2 | - | 15,942 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 522.2 | 2,227 | 4,264.3 | 113.4 | 2,040 | 17,985.6 | 181.6 | - | 15,944 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 97.2 | 287 | 2,948.2 | 12.0 | 262 | 21,881.8 | 65.0 | - | 1,788 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 856.1 | 2,227 | 2,601.2 | 82.7 | 2,036 | 24,630.5 | 218.1 | - | 1,787 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 26.6 | 287 | 10,781.8 | 11.7 | 262 | 22,371.4 | 64.4 | - | 3,678 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 166.8 | 2,227 | 13,348.1 | 82.0 | 2,040 | 24,875.6 | 287.2 | - | 3,669 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 170.7 | 287 | 1,678.6 | 16.1 | 262 | 16,286.6 | 41.1 | - | 2,953 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 1,560.5 | 2,227 | 1,427.0 | 113.2 | 2,037 | 17,985.6 | 145.7 | - | 2,953 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 38.2 | 287 | 7,512.9 | 16.0 | 262 | 16,366.6 | 39.4 | - | 6,178 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 243.6 | 2,227 | 9,140.3 | 113.2 | 2,040 | 18,018.0 | 183.2 | - | 6,207 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 144.8 | 287 | 1,979.8 | 16.1 | 262 | 16,313.2 | 68.5 | - | 21,607 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 350.8 | 2,227 | 6,347.9 | 113.2 | 2,039 | 18,018.0 | 199.3 | - | 21,569 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 1 | 209.7 | 1,663 | 7,932.3 | 126.0 | 1,634 | 12,970.2 | 67.5 | - | 20,259 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 8 | 1,182.8 | 12,922 | 10,925.0 | 982.5 | 12,694 | 12,919.9 | 106.9 | - | 20,257 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 49.5 | 370 | 7,464.8 | - | - | - | 182.9 | 3.89 | 1,067 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 8 | 590.6 | 2,959 | 5,010.0 | - | - | - | 296.5 | 3.87 | 1,105 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 16.0 | 370 | 23,065.1 | - | - | - | 284.9 | 3.84 | 1,345 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 141.6 | 2,959 | 20,900.3 | - | - | - | 618.0 | 3.81 | 1,371 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 199.9 | 370 | 1,849.8 | - | - | - | 70.0 | 4.15 | 3,156 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 8 | 2,550.1 | 2,959 | 1,160.3 | - | - | - | 95.7 | 4.11 | 3,174 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 39.3 | 370 | 9,421.5 | - | - | - | 135.2 | 4.06 | 4,499 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 428.2 | 2,959 | 6,910.2 | - | - | - | 341.0 | 4.05 | 4,550 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 180.9 | 376 | 2,076.4 | 52.5 | 344 | 6,561.7 | 84.4 | 5.15 | 3,406 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,613.3 | 2,919 | 1,809.0 | 429.4 | 2,675 | 6,230.5 | 116.7 | 5.07 | 3,371 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 28.6 | 376 | 13,119.4 | 52.5 | 344 | 6,557.4 | 189.6 | 5.09 | 4,308 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 258.9 | 2,919 | 11,271.9 | 431.0 | 2,675 | 6,207.3 | 383.8 | 4.90 | 4,319 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 89.0 | 283 | 3,174.8 | 12.9 | 262 | 20,242.9 | 126.5 | 5.02 | 1,900 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 774.9 | 2,196 | 2,833.6 | 85.8 | 2,038 | 23,753.0 | 187.9 | 4.96 | 1,894 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 26.2 | 283 | 10,789.2 | 12.6 | 262 | 20,833.3 | 199.7 | 4.86 | 2,685 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 186.4 | 2,196 | 11,778.3 | 85.4 | 2,039 | 23,866.3 | 418.8 | 4.87 | 2,700 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 156.2 | 283 | 1,809.5 | 17.3 | 262 | 15,151.5 | 48.2 | 2.85 | 3,252 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,354.6 | 2,196 | 1,621.0 | 120.4 | 2,037 | 16,920.5 | 65.7 | 2.82 | 3,251 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 21.2 | 287 | 13,492.0 | 4.3 | 262 | 60,606.1 | 200.3 | 2.18 | 1,179 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 234.5 | 2,227 | 9,496.0 | 30.7 | 2,032 | 66,225.2 | 517.8 | 2.17 | 1,139 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 8.4 | 287 | 34,076.3 | 4.3 | 261 | 60,241.0 | 365.6 | 2.19 | 1,069 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 59.2 | 2,227 | 37,589.6 | 30.6 | 2,042 | 66,666.7 | 979.9 | 2.15 | 1,191 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 558.0 | 287 | 513.6 | 16.2 | 262 | 16,233.8 | 27.9 | 2.89 | 9,022 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 5,188.0 | 2,227 | 429.2 | 114.3 | 2,037 | 17,825.3 | 92.6 | 2.89 | 9,062 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 96.5 | 287 | 2,970.2 | 16.1 | 263 | 16,286.6 | 36.1 | 2.82 | 14,342 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 736.8 | 2,227 | 3,022.2 | 114.6 | 2,040 | 17,793.6 | 140.6 | 2.83 | 14,387 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 41.6 | 287 | 6,886.3 | 12.0 | 262 | 21,929.8 | 123.6 | 2.39 | 1,623 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 412.4 | 2,227 | 5,399.3 | 82.0 | 2,036 | 24,813.9 | 384.0 | 2.40 | 1,629 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 11.5 | 287 | 24,978.3 | 11.8 | 262 | 22,222.2 | 246.6 | 2.44 | 1,559 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 76.8 | 2,227 | 28,982.7 | 82.6 | 2,040 | 24,691.4 | 718.4 | 2.37 | 1,564 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 97.3 | 287 | 2,945.2 | 12.1 | 263 | 21,786.5 | 79.9 | 2.53 | 1,914 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 970.9 | 2,227 | 2,293.5 | 82.9 | 2,037 | 24,570.0 | 257.7 | 2.52 | 1,890 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 23.4 | 287 | 12,229.4 | 11.7 | 262 | 22,471.9 | 135.7 | 2.54 | 2,797 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 163.0 | 2,227 | 13,665.8 | 82.9 | 2,037 | 24,570.0 | 408.9 | 2.54 | 2,816 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 171.0 | 287 | 1,676.1 | 16.1 | 262 | 16,286.6 | 57.5 | 2.78 | 2,945 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,560.3 | 2,227 | 1,427.2 | 113.7 | 2,037 | 17,921.1 | 202.6 | 2.80 | 2,952 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 32.7 | 287 | 8,760.5 | 16.1 | 262 | 16,286.6 | 93.0 | 2.71 | 4,783 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 235.4 | 2,227 | 9,458.1 | 113.6 | 2,039 | 17,953.3 | 329.5 | 2.72 | 4,842 |

#### Jetson AGX Orin 64GB

| Model | Kind | Mode | Precision | Batch | Runtime Prefill (ms) | Runtime Prefill Tok/Run | Runtime Prefill (tok/s) | ViT (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|---------------------:|------------------------:|------------------------:|---------:|------------:|------------:|-------------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 27.3 | 370 | 13,557.9 | - | - | - | 177.2 | - | 1,968 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 275.9 | 2,959 | 10,723.5 | - | - | - | 584.0 | - | 4,188 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 63.5 | 370 | 5,822.7 | - | - | - | 95.5 | - | 3,292 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 787.5 | 2,959 | 3,757.3 | - | - | - | 406.7 | - | 5,715 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 206.4 | 370 | 1,792.0 | - | - | - | 55.2 | - | 30,025 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 2,094.1 | 2,959 | 1,412.9 | - | - | - | 154.2 | - | 31,613 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 150.6 | 364 | 2,416.5 | - | - | - | 52.0 | - | 4,933 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 1,952.8 | 2,911 | 1,490.6 | - | - | - | 211.4 | - | 8,194 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 273.9 | 370 | 1,350.5 | - | - | - | 32.1 | - | 8,237 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 3,561.2 | 2,959 | 830.8 | - | - | - | 139.0 | - | 11,229 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 124.2 | 377 | 3,033.3 | - | - | - | 145.7 | - | 2,329 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 939.4 | 3,013 | 3,207.4 | - | - | - | 633.3 | - | 3,333 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 1,335.8 | 377 | 282.0 | - | - | - | 10.5 | - | 24,136 |
| Qwen3.5-27B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 16,963.3 | 3,013 | 177.6 | - | - | - | 44.5 | - | 26,293 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 161.9 | 377 | 2,326.5 | - | - | - | 81.6 | - | 4,197 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 1,447.5 | 3,013 | 2,081.7 | - | - | - | 393.6 | - | 5,117 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | 1 | 340.7 | 377 | 1,105.4 | - | - | - | 31.0 | - | 35,742 |
| Qwen3.5-35B-A3B-LLM | LLM | Vanilla | INT4 GPTQ | 8 | 2,807.2 | 3,013 | 1,073.4 | - | - | - | 128.5 | - | 36,654 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 307.0 | 377 | 1,226.9 | - | - | - | 45.4 | - | 6,147 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 3,428.1 | 3,013 | 879.0 | - | - | - | 187.7 | - | 7,659 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 437.5 | 377 | 860.9 | - | - | - | 27.9 | - | 9,983 |
| Qwen3.5-9B-LLM | LLM | Vanilla | INT4 AWQ | 8 | 5,122.9 | 3,013 | 588.2 | - | - | - | 125.8 | - | 11,278 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 48.0 | 283 | 5,883.5 | 38.7 | 262 | 6,775.1 | 95.7 | - | 8,235 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 408.3 | 2,196 | 5,378.2 | 276.1 | 2,039 | 7,385.5 | 388.8 | - | 10,304 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 118.6 | 283 | 2,384.0 | 39.0 | 262 | 6,724.9 | 52.2 | - | 9,863 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 1,024.0 | 2,196 | 2,144.3 | 277.4 | 2,038 | 7,347.5 | 176.4 | - | 12,511 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 213.2 | 283 | 1,325.8 | 56.5 | 262 | 4,646.8 | 32.2 | - | 13,133 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 1,842.2 | 2,196 | 1,191.9 | 398.8 | 2,039 | 5,112.5 | 141.0 | - | 16,014 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 94.0 | 287 | 3,050.3 | 13.0 | 262 | 20,161.3 | 141.4 | - | 4,549 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 501.1 | 2,227 | 4,444.2 | 89.9 | 2,038 | 22,675.7 | 451.8 | - | 5,058 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 1,037.4 | 287 | 276.3 | 53.5 | 262 | 4,906.8 | 10.5 | - | 24,048 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 8,777.6 | 2,227 | 253.7 | 382.1 | 2,038 | 5,333.3 | 44.2 | - | 26,372 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 120.8 | 287 | 2,372.0 | 36.6 | 262 | 7,168.5 | 81.0 | - | 6,886 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 766.8 | 2,227 | 2,903.9 | 264.7 | 2,038 | 7,698.2 | 303.9 | - | 7,456 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 297.1 | 287 | 964.7 | 52.8 | 262 | 4,970.2 | 30.8 | - | 35,810 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 1,537.3 | 2,227 | 1,448.6 | 379.6 | 2,038 | 5,367.7 | 131.4 | - | 36,733 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 234.9 | 287 | 1,220.0 | 36.6 | 262 | 7,158.2 | 45.3 | - | 8,638 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 1,815.4 | 2,227 | 1,226.7 | 265.5 | 2,038 | 7,674.6 | 174.8 | - | 9,798 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 341.5 | 287 | 839.2 | 53.2 | 262 | 4,931.0 | 27.9 | - | 12,247 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,672.0 | 2,227 | 833.4 | 380.9 | 2,038 | 5,350.5 | 121.7 | - | 13,410 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 63.8 | 370 | 5,797.2 | - | - | - | 152.4 | 3.87 | 3,362 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 8 | 784.4 | 2,959 | 3,772.0 | - | - | - | 250.7 | 3.85 | 6,949 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 274.3 | 370 | 1,348.6 | - | - | - | 57.8 | 4.18 | 8,349 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 8 | 3,567.7 | 2,959 | 829.3 | - | - | - | 70.4 | 4.16 | 12,969 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 255.2 | 376 | 1,472.3 | 86.1 | 344 | 3,998.4 | 67.2 | 5.05 | 12,218 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 2,279.8 | 2,919 | 1,280.2 | 671.2 | 2,675 | 3,985.7 | 82.5 | 4.97 | 14,145 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 118.0 | 283 | 2,396.0 | 39.1 | 262 | 6,711.4 | 98.3 | 5.05 | 10,310 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,025.0 | 2,196 | 2,142.2 | 276.8 | 2,038 | 7,363.8 | 147.9 | 5.07 | 13,669 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 213.6 | 283 | 1,323.0 | 56.6 | 262 | 4,633.9 | 39.1 | 2.81 | 13,932 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,843.2 | 2,196 | 1,191.3 | 399.6 | 2,038 | 5,099.4 | 48.1 | 2.83 | 17,413 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 92.7 | 287 | 3,092.1 | 13.0 | 262 | 20,242.9 | 138.9 | 2.19 | 5,183 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 493.8 | 2,227 | 4,509.3 | 90.1 | 2,038 | 22,624.4 | 413.7 | 2.20 | 6,396 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 1,038.2 | 287 | 276.1 | 53.2 | 262 | 4,931.0 | 18.3 | 2.89 | 25,664 |
| Qwen3.5-27B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 8,790.5 | 2,227 | 253.3 | 383.0 | 2,038 | 5,322.0 | 69.5 | 2.87 | 33,332 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 120.4 | 287 | 2,381.6 | 36.6 | 262 | 7,173.6 | 86.4 | 2.41 | 8,027 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 762.3 | 2,227 | 2,921.4 | 265.6 | 2,038 | 7,674.6 | 293.7 | 2.39 | 9,382 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 234.7 | 287 | 1,221.4 | 36.7 | 262 | 7,142.9 | 54.6 | 2.53 | 10,181 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 1,811.5 | 2,227 | 1,229.3 | 265.9 | 2,037 | 7,662.8 | 204.4 | 2.50 | 12,976 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 341.6 | 287 | 839.1 | 53.0 | 262 | 4,952.9 | 40.4 | 2.82 | 14,598 |
| Qwen3.5-9B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 8 | 2,670.7 | 2,227 | 833.8 | 379.2 | 2,038 | 5,373.5 | 159.9 | 2.78 | 17,548 |

#### Jetson Orin NX 16GB

| Model | Kind | Mode | Precision | Batch | Runtime Prefill (ms) | Runtime Prefill Tok/Run | Runtime Prefill (tok/s) | ViT (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|---------------------:|------------------------:|------------------------:|---------:|------------:|------------:|-------------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 52.4 | 370 | 7,064.6 | - | - | - | 110.8 | - | 2,027 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 127.4 | 370 | 2,903.3 | - | - | - | 58.8 | - | 3,282 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 306.8 | 364 | 1,186.0 | - | - | - | 30.4 | - | 4,914 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 615.1 | 370 | 601.3 | - | - | - | 18.6 | - | 8,210 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 174.8 | 377 | 2,154.5 | - | - | - | 88.2 | - | 2,286 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 249.4 | 377 | 1,510.3 | - | - | - | 48.3 | - | 4,235 |
| Qwen3.5-4B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 535.8 | 377 | 703.0 | - | - | - | 26.2 | - | 6,093 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 98.6 | 283 | 2,866.0 | 81.6 | 262 | 3,214.4 | 58.9 | - | 4,406 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 238.5 | 283 | 1,184.8 | 83.3 | 262 | 3,148.6 | 30.8 | - | 5,990 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 455.2 | 283 | 620.9 | 119.6 | 262 | 2,193.0 | 18.8 | - | 9,092 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 136.6 | 287 | 2,099.0 | 26.7 | 262 | 9,832.8 | 87.6 | - | 2,704 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 193.6 | 287 | 1,480.4 | 77.5 | 262 | 3,386.4 | 48.0 | - | 4,704 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 414.6 | 287 | 691.3 | 79.0 | 262 | 3,321.2 | 25.9 | - | 6,372 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 129.0 | 370 | 2,867.6 | - | - | - | 86.5 | 3.87 | 3,394 |
| Qwen3-8B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 651.4 | 370 | 567.8 | - | - | - | 29.7 | 4.14 | 8,358 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 505.7 | 376 | 742.9 | 200.7 | 344 | 1,715.6 | 34.7 | 5.05 | 9,208 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 237.8 | 283 | 1,188.7 | 83.1 | 262 | 3,156.6 | 56.2 | 5.11 | 6,427 |
| Qwen3-VL-8B-Instruct | VLM | EAGLE3 | INT4 AWQ / INT4 AWQ / FP16 | 1 | 488.9 | 283 | 578.0 | 121.0 | 262 | 2,167.8 | 20.4 | 2.84 | 9,565 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 135.8 | 287 | 2,111.0 | 26.7 | 262 | 9,813.5 | 77.7 | 2.16 | 3,296 |
| Qwen3.5-2B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 193.3 | 287 | 1,482.8 | 77.8 | 262 | 3,372.7 | 46.8 | 2.38 | 5,758 |
| Qwen3.5-4B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 410.7 | 287 | 697.8 | 78.0 | 262 | 3,363.6 | 28.9 | 2.52 | 7,838 |

#### Jetson Orin Nano 8GB

| Model | Kind | Mode | Precision | Batch | Runtime Prefill (ms) | Runtime Prefill Tok/Run | Runtime Prefill (tok/s) | ViT (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|---------------------:|------------------------:|------------------------:|---------:|------------:|------------:|-------------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 94.7 | 370 | 3,904.2 | - | - | - | 69.4 | - | 1,999 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 231.5 | 370 | 1,597.9 | - | - | - | 36.3 | - | 3,286 |
| Qwen3.5-0.8B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 316.3 | 377 | 1,190.9 | - | - | - | 55.0 | - | 2,294 |
| Qwen3.5-2B-LLM | LLM | Vanilla | INT4 AWQ | 1 | 445.0 | 377 | 846.5 | - | - | - | 29.8 | - | 4,145 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 181.7 | 283 | 1,555.1 | 150.6 | 262 | 1,741.6 | 36.3 | - | 4,444 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 244.0 | 287 | 1,174.8 | 48.8 | 262 | 5,376.3 | 54.4 | - | 2,656 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 349.8 | 287 | 819.3 | 142.4 | 262 | 1,842.6 | 29.4 | - | 4,649 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ / INT4 AWQ | 1 | 232.2 | 370 | 1,593.0 | - | - | - | 48.7 | 3.87 | 3,398 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1 | 244.1 | 287 | 1,174.1 | 48.7 | 262 | 5,387.9 | 46.9 | 2.16 | 3,278 |

---

## v0.7.1 Results

> **SDK Version:** TensorRT Edge-LLM 0.7.1 &nbsp;|&nbsp; **TensorRT:** 10.13.3.9

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3-1.7B | NVFP4 | 1 | 11.2 | 370 | 32,990 | 173.2 | 1,531 |
| Qwen3-1.7B | NVFP4 | 8 | 132.5 | 2,959 | 22,324 | 935.8 | 1,475 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 1 | 133.2 | 370 | 2,777 | 72.6 | 15,916 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 8 | 1,344.5 | 2,959 | 2,201 | 215.9 | 15,894 |
| Nemotron-3-Nano-4B | NVFP4 | 1 | 127.4 | 383 | 3,004 | 64.7 | 3,568 |
| Nemotron-3-Nano-4B | NVFP4 | 8 | 986.6 | 3,062 | 3,104 | 312.9 | 3,592 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|----------|----------|:-----:|-------------:|:--------------:|----------------:|--------------:|------------:|------------:|-------------------:|-------------:|
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP16 | 1 | 31.0 | 376 | 12,124 | 27.8 | 344 | 12,392 | 49.5 | 5,224 |
| Qwen3.5-0.8B | NVFP4 | FP16 | 1 | 9.4 | 287 | 30,410 | 4.3 | 262 | 60,606 | 287.0 | 1,192 |
| Qwen3.5-2B | NVFP4 | FP16 | 1 | 12.7 | 287 | 22,550 | 10.5 | 262 | 25,000 | 164.4 | 1,694 |
| Qwen3.5-27B | NVFP4 | FP16 | 1 | 103.3 | 287 | 2,775 | 15.0 | 262 | 17,483 | 16.1 | 14,725 |
| Nemotron-3-Nano-Omni-30B-A3B | NVFP4 | FP16 | 1 | 226.0 | 1,663 | 7,358 | 121.3 | 1,635 | 13,477 | 31.3 | 20,327 |

### LLM — EAGLE3 Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen3-1.7B | Qwen3-1.7B_eagle3 | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |

> **Note:** Both base and draft models are quantized to NVFP4.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup | GPU Mem (MB) |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|-------------:|
| Qwen3-1.7B | NVFP4 | NVFP4 | 1 | 12.2 | 370 | 339.0 | 3.7 | 1.96x | 1,534 |
| Qwen3-1.7B | NVFP4 | NVFP4 | 8 | 132.3 | 2,959 | 984.9 | 3.7 | 1.05x | 1,466 |

### Vision Language Model — MTP Speculative Decoding

> **Note:** MTP uses the model's built-in draft heads; no external draft checkpoint is required.
> **Highlight:** MTP is the main v0.7.1 performance improvement, increasing Qwen3.5 VLM BS=1 generation throughput by 1.21x to 2.12x over vanilla decoding.

| Model | Base Prec | Draft Prec | ViT Prec | Batch | Prefill (ms) | Prefill Tokens | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|-----------|------------|----------|:-----:|-------------:|:--------------:|--------------:|------------:|------------:|-------------------:|:-----------:|-------------:|
| Qwen3.5-0.8B | NVFP4 | NVFP4 | FP16 | 1 | 9.2 | 287 | 4.2 | 263 | 62,500 | 348.5 | 2.1 | 1,210 |
| Qwen3.5-0.8B | NVFP4 | NVFP4 | FP16 | 8 | 69.0 | 2,227 | 27.6 | 2,042 | 74,074 | 1,056.7 | 2.2 | 1,375 |
| Qwen3.5-2B | NVFP4 | NVFP4 | FP16 | 1 | 13.8 | 287 | 10.9 | 262 | 24,096 | 236.9 | 2.4 | 1,662 |
| Qwen3.5-2B | NVFP4 | NVFP4 | FP16 | 8 | 89.7 | 2,227 | 75.1 | 2,040 | 27,174 | 787.2 | 2.4 | 1,647 |
| Qwen3.5-27B | NVFP4 | NVFP4 | FP16 | 1 | 111.1 | 287 | 14.4 | 262 | 18,215 | 34.2 | 2.8 | 14,680 |
| Qwen3.5-27B | NVFP4 | NVFP4 | FP16 | 8 | 811.0 | 2,227 | 108.2 | 2,038 | 18,832 | 146.7 | 2.8 | 14,705 |

---

## v0.7.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.7.0 &nbsp;|&nbsp; **TensorRT:** 10.13

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3-1.7B | NVFP4 | 1 | 13.9 | 370 | 26,683 | 170.4 | 1,453 |
| Qwen3-1.7B | NVFP4 | 8 | 150.5 | 2,959 | 19,663 | 798.8 | 1,491 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 1 | 125.3 | 370 | 2,951 | 81.3 | 15,938 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 8 | 1,342.2 | 2,959 | 2,204 | 223.2 | 15,961 |
| Nemotron-3-Nano-4B | NVFP4 | 1 | 126.8 | 383 | 3,018 | 65.4 | 3,647 |
| Nemotron-3-Nano-4B | NVFP4 | 8 | 1,017.6 | 3,062 | 3,009 | 315.4 | 3,684 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|----------|----------|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3.5-0.8B | NVFP4 | FP16 | 7.0 | 753 | 107,571 | 232.2 | 1,052 |
| Qwen3.5-2B | NVFP4 | FP16 | 13.8 | 753 | 54,565 | 111.0 | 1,671 |
| Qwen3.5-27B | NVFP4 | FP16 | 122.6 | 753 | 6,143 | 10.5 | 14,985 |
| Nemotron-3-Nano-Omni-30B-A3B | NVFP4 | FP16 | 846.7 | 1,663 | 1,964 | 24.5 | 20,267 |

### LLM — EAGLE3 Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen3-1.7B | Qwen3-1.7B_eagle3 | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |

> **Note:** Both base and draft models are quantized to NVFP4.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Qwen3-1.7B | NVFP4 | NVFP4 | 1 | 14.5 | 370 | 312.4 | 3.75 | 1.83x |
| Qwen3-1.7B | NVFP4 | NVFP4 | 8 | 153.5 | 2,959 | 828.8 | 3.73 | 1.04x |

---

## v0.4.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.4.0 &nbsp;|&nbsp; **TensorRT:** 10.13

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|
| Llama-3.1-8B-Instruct | INT4 AWQ | 1 | 215.5 | 383 | 1,777 | 50.8 |
| Llama-3.1-8B-Instruct | INT4 AWQ | 8 | 2737.4 | 3064 | 1,119 | 135.3 |
| Llama-3.1-8B-Instruct | NVFP4 | 1 | 31.0 | 383 | 12,355 | 54.9 |
| Llama-3.1-8B-Instruct | NVFP4 | 8 | 387.6 | 3064 | 7,905 | 308.7 |
| Qwen3-0.6B | INT4 AWQ | 1 | 21.0 | 366 | 17,429 | 270.2 |
| Qwen3-0.6B | INT4 AWQ | 8 | 241.8 | 2927 | 12,104 | 828.0 |
| Qwen3-0.6B | NVFP4 | 1 | 8.8 | 366 | 41,591 | 318.6 |
| Qwen3-0.6B | NVFP4 | 8 | 95.4 | 2927 | 30,681 | 1562.4 |
| Qwen3-4B-Instruct-2507 | INT4 AWQ | 1 | 116.2 | 364 | 3,133 | 76.4 |
| Qwen3-4B-Instruct-2507 | INT4 AWQ | 8 | 1502.3 | 2911 | 1,938 | 240.3 |
| Qwen3-4B-Instruct-2507 | NVFP4 | 1 | 22.9 | 364 | 15,895 | 90.2 |
| Qwen3-4B-Instruct-2507 | NVFP4 | 8 | 301.9 | 2911 | 9,642 | 507.4 |
| Qwen3-8B | INT4 AWQ | 1 | 212.0 | 366 | 1,726 | 47.7 |
| Qwen3-8B | INT4 AWQ | 8 | 2719.1 | 2927 | 1,076 | 162.3 |
| Qwen3-8B | NVFP4 | 1 | 32.8 | 366 | 11,159 | 53.7 |
| Qwen3-8B | NVFP4 | 8 | 425.8 | 2927 | 6,874 | 372.2 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) |
|-------|----------|----------|-------------:|:--------------:|----------------:|---------------:|:-----------:|------------:|-------------------:|
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | FP16 | 195.1 | 376 | 1,927 | 51.1 | 344 | 6,732 | 53.1 |
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | FP8 | 195.1 | 376 | 1,927 | 42.7 | 344 | 8,056 | 53.1 |
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP16 | 25.7 | 376 | 14,631 | 51.0 | 344 | 6,745 | 57.7 |
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP8 | 25.7 | 376 | 14,631 | 42.6 | 344 | 8,075 | 57.6 |
| Qwen3-VL-2B-Instruct | INT4 AWQ | FP16 | 39.4 | 283 | 7,183 | 19.0 | 262 | 13,789 | 144.4 |
| Qwen3-VL-2B-Instruct | INT4 AWQ | FP8 | 39.4 | 283 | 7,183 | 15.4 | 262 | 17,013 | 144.7 |
| Qwen3-VL-2B-Instruct | NVFP4 | FP16 | 10.1 | 283 | 28,020 | 19.0 | 262 | 13,789 | 180.8 |
| Qwen3-VL-2B-Instruct | NVFP4 | FP8 | 10.1 | 283 | 28,020 | 15.5 | 262 | 16,903 | 181.0 |

> **Note:** ViT time = per-token ViT latency x image tokens per run. FP8 ViT reduces visual encoder time by ~17% compared to FP16 with negligible impact on generation throughput.

### LLM — EAGLE3 Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-Instruct-8B | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) |
| Qwen3-8B | qwen3_8b_eagle3 | [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) |

> **Note:** Both base and draft models are quantized to the same precision (INT4 AWQ or NVFP4) as listed in the table below.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Llama-3.1-8B-Instruct | INT4 AWQ | INT4 AWQ | 1 | 215.2 | 382 | 81.0 | 5.25 | 1.59x |
| Llama-3.1-8B-Instruct | INT4 AWQ | INT4 AWQ | 8 | 2735.5 | 3056 | 118.0 | 5.21 | 0.87x |
| Llama-3.1-8B-Instruct | NVFP4 | NVFP4 | 1 | 30.8 | 382 | 189.2 | 5.21 | 3.45x |
| Llama-3.1-8B-Instruct | NVFP4 | NVFP4 | 8 | 413.1 | 3056 | 484.7 | 5.15 | 1.57x |
| Qwen3-8B | INT4 AWQ | INT4 AWQ | 1 | 212.2 | 366 | 66.1 | 4.36 | 1.39x |
| Qwen3-8B | INT4 AWQ | INT4 AWQ | 8 | 2719.1 | 2927 | 99.1 | 4.31 | 0.61x |
| Qwen3-8B | NVFP4 | NVFP4 | 1 | 33.1 | 366 | 151.7 | 4.26 | 2.82x |
| Qwen3-8B | NVFP4 | NVFP4 | 8 | 429.1 | 2927 | 457.7 | 4.25 | 1.23x |

> **Note:** EAGLE3 speculative decoding provides the greatest speedup at BS=1 (latency-bound). At BS=8, base model compute is already well-utilized, limiting speculative acceleration. See [Speculative Decoding](../examples/speculative-decoding.md) for setup instructions.

### Vision Language Model — EAGLE3 Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen2.5-VL-7B-Instruct | qwen2.5-vl-7b-eagle3-sgl | [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) |

> **Note:** Both base and draft models are quantized to the same precision as listed in the table below.

| Model | Base Prec | Draft Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|----------|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | INT4 AWQ | FP16 | 195.1 | 376 | 57.3 | 3.66 | 1.08x |
| Qwen2.5-VL-7B-Instruct | NVFP4 | NVFP4 | FP16 | 25.8 | 376 | 149.6 | 3.82 | 2.59x |
| Qwen2.5-VL-7B-Instruct | NVFP4 | NVFP4 | FP8 | 32.8 | 376 | 117.3 | 3.76 | 2.04x |

---

## Key Observations

### v0.9.0

- **Current release baseline:** v0.9.0 is the current performance baseline and supersedes the v0.8.0 regression note. Runtime tables include batch size, dataset, prefill sequence length/time, decode throughput, speculative acceptance rate, and peak GPU memory.
- **Speculative decoding coverage:** EAGLE3, MTP, and DFlash rows are included for supported v0.9.0 engines. Orin platforms use the externalized INT4 subset supported by each memory target.

### v0.8.0

- **All release devices are covered:** v0.8.0 adds Jetson AGX Thor, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, and Jetson Orin Nano 8GB results from the release benchmark outputs, benchmarked under JetPack 7.2.
- **First `llm_bench` prefill metrics:** AGX Thor includes parsed `llm_bench` prefill measurements at `inputLen=2048`. Qwen3-0.6B NVFP4 reaches 91,469.3 tok/s at BS=1; Qwen3-1.7B EAGLE3 NVFP4 reaches 66,929.4 tok/s at BS=1.
- **Speculative decode remains platform-dependent:** EAGLE3 and MTP report strong acceptance rates, but generation throughput depends heavily on model size, precision, and platform memory bandwidth.

### v0.7.1

- **MTP speculative decoding:** This is the v0.7.1 performance highlight. Qwen3.5 MTP improves BS=1 generation throughput by 1.21x for 0.8B, 1.44x for 2B, and 2.12x for 27B over vanilla decoding, with BS=8 throughput up to 1,056.7 tok/s for Qwen3.5-0.8B.

### v0.7.0

- **MoE support:** Qwen3-30B-A3B-GPTQ-Int4 (MoE, 3B active params out of 30B) achieves 81.3 tok/s at BS=1 and 223.2 tok/s at BS=8 with INT4 GPTQ, demonstrating efficient sparse model inference on edge.
- **Small model throughput:** Qwen3-1.7B with NVFP4 delivers 170.4 tok/s at BS=1 and 798.8 tok/s at BS=8, suitable for latency-sensitive edge applications.
- **Qwen3.5 VLM family:** Ranges from 232.2 tok/s (0.8B) to 10.5 tok/s (27B), providing a scalable VLM option across memory and throughput budgets.
- **Nemotron-3-Nano-Omni-30B-A3B:** The first audio+video multimodal model benchmarked, achieving 24.5 tok/s generation at 20 GB GPU memory.

### v0.4.0

- **NVFP4 delivers highest throughput**: NVFP4 achieves 1.1–2.3x higher generation throughput than INT4 AWQ, with substantially faster prefill (e.g., 31 ms vs 216 ms for Llama-3.1-8B at BS=1).
- **EAGLE3 at BS=1 provides meaningful speedup**: 1.4–3.5x for LLMs, best for Llama-3.1-8B NVFP4 (3.45x). The draft model acceptance rate is high for Llama (~5.2 tokens/step) and moderate for Qwen3-8B (~4.3 tokens/step).
- **EAGLE3 at BS=8 has limited benefit**: At high batch sizes, base model compute is already well-utilized. Speedup drops to <1x for INT4 AWQ and 1.2–1.6x for NVFP4.
- **Qwen3-0.6B achieves the highest throughput**: 1562 tok/s at BS=8 with NVFP4 — a lightweight model well-suited for latency-sensitive edge applications.

### General

- Benchmarks use default TensorRT Edge-LLM inference settings on the listed device. Production performance may vary with system-level tuning (power mode, memory configuration, thermal management).
