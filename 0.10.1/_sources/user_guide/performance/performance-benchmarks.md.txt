# Performance Benchmarks

> **Platforms:** NVIDIA Jetson AGX Thor Developer Kit, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, Jetson Orin Nano 8GB, and NVIDIA DGX Spark (GB10). Older release sections may include a narrower platform set.

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
| **JetSpec** | JetSpec paired-draft speculative decoding with causal proposal attention and branching tree verification |

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

- [Quick Start Guide](../getting_started/quick-start-guide.md) for LLM and VLM
  export, component engine builds, `llm_inference`, and `llm_bench`.
- [Speculative Decoding](../examples/speculative-decoding.md) for EAGLE3, MTP,
  DFlash, and JetSpec export/build layouts.
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
| JetSpec tree base engine | Vanilla LLM parameters plus `--specBase --maxVerifyTreeSize 128` |
| JetSpec draft engine | Vanilla LLM parameters plus `--specDraft --maxDraftTreeSize 128` |

Use the batch size shown in the benchmark row. Thor and Jetson AGX Orin rows may
use batch `1` or `8`; Jetson Orin NX and Orin Nano rows are generally batch `1`.
For INT4 runs on Orin, follow the export docs but use externalized INT4 weights:
`--externalize-weights int4_ffn` for dense checkpoints and
`--externalize-weights int4_ffn int4_moe` for MoE checkpoints.

For speculative decoding, follow the exact export layouts in
[Speculative Decoding](../examples/speculative-decoding.md): EAGLE3 uses a base
and draft export, MTP uses the MTP base and `mtp_draft` export, and DFlash uses
the paired DFlash base and draft export. JetSpec uses the paired JetSpec
tree-base and draft export. For DFlash, use the linear DFlash base export for
`--specDraftTopK 1`; use the tree-base export only for DDTree runs. For JetSpec,
use the tree-base export with `--specDraftTopK > 1`.

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
| JetSpec tree runtime | Common settings plus `--specDecode --specDraftTopK 7 --specDraftStep 1 --specVerifySize 128 --jetspecBlockSize 16` |

For Qwen3.5 DFlash inputs, set `"enable_thinking": true`; for Qwen3 DFlash and
Qwen3 JetSpec inputs, set `"enable_thinking": false`. These settings match the
paired HuggingFace generation behavior used for speculative decoding
validation.

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
`16` for linear DFlash, and `128` for JetSpec tree runs.

## v0.10.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.10.0 &nbsp;|&nbsp; **Jetson:** JetPack 7.2, CUDA 13.2, TensorRT 10.16 &nbsp;|&nbsp; **DGX Spark:** CUDA 13.0, TensorRT 10.16

> **Decode throughput:** Runtime `Decode (tok/s)` reports generated tokens per second for vanilla decoding and overall accepted-token throughput for speculative decoding. `llm_bench` BS=8 decode throughput is reported as aggregate batch throughput.

### v0.10.0 `llm_bench` Component Performance

These rows report fixed-length `llm_bench` prefill and decode measurements at the batch sizes shown. Batch-8 decode is shown as aggregate throughput (`per-sequence tok/s x batch size`).

| Platform | Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill E2E (ms) | Prefill (tok/s) | Decode Past KV Len | Decode (tok/s) |
|----------|-------|------|------|-----------|:-----:|----------------:|-----------------:|----------------:|-------------------:|---------------:|
| Jetson AGX Thor | gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 1 | 2,048 | 622.8 | 3,288.6 | 2,048 | 10.1 |
| Jetson AGX Thor | gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,048 | 7,291.6 | 280.9 | 2,048 | 74.4 |
| Jetson AGX Thor | gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 1 | 2,048 | 1,036.4 | 1,976.1 | 2,048 | 7.4 |
| Jetson AGX Thor | gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,048 | 9,171.5 | 223.3 | 2,048 | 48.0 |
| Jetson AGX Thor | gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 2,899.5 | 706.3 | 2,048 | 12.8 |
| Jetson AGX Thor | gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 36,592.2 | 56.0 | 2,048 | 74.4 |
| Jetson AGX Thor | nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 187.9 | 10,901.7 | 2,048 | 41.8 |
| Jetson AGX Thor | nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,718.1 | 1,192.0 | 2,048 | 193.6 |
| Jetson AGX Thor | nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 786.9 | 2,602.5 | 2,048 | 7.7 |
| Jetson AGX Thor | nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 10,663.3 | 192.1 | 2,048 | 54.4 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 168.7 | 12,138.6 | 2,048 | 76.2 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,033.3 | 1,982.0 | 2,048 | 275.2 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 1,228.4 | 1,667.3 | 2,048 | 66.8 |
| Jetson AGX Thor | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 5,724.6 | 357.8 | 2,048 | 423.2 |
| Jetson AGX Thor | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 376.5 | 5,439.0 | 2,048 | 80.7 |
| Jetson AGX Thor | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,885.1 | 709.9 | 2,048 | 282.4 |
| Jetson AGX Thor | Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 83.1 | 24,637.4 | 2,048 | 59.2 |
| Jetson AGX Thor | Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 961.9 | 2,129.2 | 2,048 | 382.4 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 54.8 | 37,350.2 | 2,048 | 253.1 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 658.4 | 3,110.6 | 2,048 | 718.4 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 17.9 | 114,242.2 | 2,048 | 244.3 |
| Jetson AGX Thor | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 209.3 | 9,785.5 | 2,048 | 740.0 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 145.4 | 14,082.4 | 2,048 | 135.1 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 1,777.8 | 1,152.0 | 2,048 | 547.2 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 27.7 | 73,858.3 | 2,048 | 135.5 |
| Jetson AGX Thor | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 314.4 | 6,514.6 | 2,048 | 532.8 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 2,048 | 490.6 | 4,174.4 | 2,048 | 76.7 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 2,048 | 4,264.1 | 480.3 | 2,048 | 233.6 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 137.7 | 14,869.9 | 2,048 | 84.1 |
| Jetson AGX Thor | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 964.5 | 2,123.4 | 2,048 | 251.2 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 364.6 | 5,617.3 | 2,048 | 76.5 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 4,671.1 | 438.4 | 2,048 | 350.4 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 2,048 | 62.0 | 33,029.8 | 2,048 | 73.4 |
| Jetson AGX Thor | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 2,048 | 746.3 | 2,744.4 | 2,048 | 356.0 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 663.4 | 3,087.2 | 2,048 | 46.4 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 8,266.6 | 247.7 | 2,048 | 250.4 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 104.9 | 19,516.9 | 2,048 | 44.6 |
| Jetson AGX Thor | Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,144.7 | 1,789.1 | 2,048 | 254.4 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 145.3 | 14,091.7 | 2,048 | 135.3 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 1,846.1 | 1,109.4 | 2,048 | 541.6 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 28.0 | 73,082.5 | 2,048 | 136.0 |
| Jetson AGX Thor | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 317.7 | 6,446.4 | 2,048 | 564.0 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 364.4 | 5,620.4 | 2,048 | 76.3 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 4,658.7 | 439.6 | 2,048 | 353.6 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 62.2 | 32,915.8 | 2,048 | 72.9 |
| Jetson AGX Thor | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 750.5 | 2,728.8 | 2,048 | 358.4 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 662.2 | 3,092.9 | 2,048 | 46.2 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 7,897.8 | 259.3 | 2,048 | 249.6 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 104.5 | 19,604.5 | 2,048 | 44.4 |
| Jetson AGX Thor | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,149.5 | 1,781.7 | 2,048 | 254.4 |
| Jetson AGX Thor | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 72.5 | 28,237.7 | 2,048 | 232.5 |
| Jetson AGX Thor | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 821.0 | 2,494.5 | 2,048 | 1,183.2 |
| Jetson AGX Thor | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 32.7 | 62,644.7 | 2,048 | 229.6 |
| Jetson AGX Thor | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 327.0 | 6,263.0 | 2,048 | 1,212.8 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 72.4 | 28,286.9 | 2,048 | 232.6 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 800.9 | 2,557.2 | 2,048 | 1,172.8 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 32.7 | 62,606.0 | 2,048 | 229.6 |
| Jetson AGX Thor | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 327.1 | 6,261.2 | 2,048 | 1,217.6 |
| Jetson AGX Thor | Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 2,353.1 | 870.3 | 2,048 | 15.2 |
| Jetson AGX Thor | Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 30,171.7 | 67.9 | 2,048 | 100.8 |
| Jetson AGX Thor | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 448.7 | 4,563.9 | 2,048 | 14.7 |
| Jetson AGX Thor | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 4,234.7 | 483.6 | 2,048 | 95.2 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 2,311.9 | 885.9 | 2,048 | 15.0 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 29,907.2 | 68.5 | 2,048 | 101.6 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 447.3 | 4,578.5 | 2,048 | 14.5 |
| Jetson AGX Thor | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 4,213.9 | 486.0 | 2,048 | 95.2 |
| Jetson AGX Thor | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 152.4 | 13,440.9 | 2,048 | 122.5 |
| Jetson AGX Thor | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 1,858.5 | 1,101.9 | 2,048 | 738.4 |
| Jetson AGX Thor | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 42.9 | 47,716.2 | 2,048 | 122.4 |
| Jetson AGX Thor | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 423.5 | 4,835.8 | 2,048 | 752.8 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 151.7 | 13,499.3 | 2,048 | 122.8 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 1,869.2 | 1,095.7 | 2,048 | 736.8 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 43.3 | 47,327.5 | 2,048 | 122.5 |
| Jetson AGX Thor | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 419.4 | 4,883.7 | 2,048 | 772.8 |
| Jetson AGX Thor | Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 2,048 | 358.0 | 5,721.4 | 2,048 | 47.3 |
| Jetson AGX Thor | Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 2,048 | 2,898.8 | 706.5 | 2,048 | 212.0 |
| Jetson AGX Thor | Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 2,048 | 360.0 | 5,688.9 | 2,048 | 46.7 |
| Jetson AGX Thor | Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 2,048 | 2,891.7 | 708.2 | 2,048 | 212.8 |
| Jetson AGX Thor | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 370.2 | 5,531.9 | 2,048 | 69.8 |
| Jetson AGX Thor | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 4,771.8 | 429.2 | 2,048 | 384.8 |
| Jetson AGX Thor | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 96.1 | 21,316.3 | 2,048 | 67.9 |
| Jetson AGX Thor | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 943.0 | 2,171.8 | 2,048 | 397.6 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 371.4 | 5,515.0 | 2,048 | 69.5 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 4,693.0 | 436.4 | 2,048 | 388.0 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 95.2 | 21,507.3 | 2,048 | 67.9 |
| Jetson AGX Thor | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 941.3 | 2,175.7 | 2,048 | 392.0 |
| Jetson AGX Thor | Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 677.9 | 3,020.9 | 2,048 | 41.3 |
| Jetson AGX Thor | Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 7,979.9 | 256.6 | 2,048 | 260.0 |
| Jetson AGX Thor | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 134.7 | 15,205.2 | 2,048 | 40.1 |
| Jetson AGX Thor | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,390.4 | 1,473.0 | 2,048 | 259.2 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 690.5 | 2,966.2 | 2,048 | 41.3 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 8,068.1 | 253.8 | 2,048 | 256.8 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 135.3 | 15,139.5 | 2,048 | 40.1 |
| Jetson AGX Thor | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,365.3 | 1,500.1 | 2,048 | 259.2 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 187.0 | 10,950.3 | 2,048 | 84.5 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,194.2 | 1,715.0 | 2,048 | 264.8 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 188.5 | 10,865.8 | 2,048 | 85.0 |
| Jetson AGX Thor | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,200.8 | 1,705.5 | 2,048 | 264.0 |
| Jetson AGX Orin (64GB) | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 118.5 | 17,277.3 | 2,048 | 162.5 |
| Jetson AGX Orin (64GB) | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 308.3 | 6,643.9 | 2,048 | 89.5 |
| Jetson AGX Orin (64GB) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 761.9 | 2,687.9 | 2,048 | 50.7 |
| Jetson AGX Orin (64GB) | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,384.1 | 1,479.7 | 2,048 | 30.6 |
| Jetson AGX Orin (64GB) | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 298.4 | 6,863.4 | 2,048 | 88.7 |
| Jetson AGX Orin (64GB) | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 757.7 | 2,702.9 | 2,048 | 50.8 |
| Jetson AGX Orin (64GB) | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,388.1 | 1,475.5 | 2,048 | 30.6 |
| Jetson AGX Orin (64GB) | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 559.8 | 3,658.6 | 2,048 | 153.7 |
| Jetson AGX Orin (64GB) | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 559.9 | 3,657.9 | 2,048 | 153.3 |
| Jetson AGX Orin (64GB) | Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 6,352.6 | 322.4 | 2,048 | 10.5 |
| Jetson AGX Orin (64GB) | Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 6,319.5 | 324.1 | 2,048 | 10.5 |
| Jetson AGX Orin (64GB) | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 730.9 | 2,802.2 | 2,048 | 81.8 |
| Jetson AGX Orin (64GB) | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 721.4 | 2,839.0 | 2,048 | 81.7 |
| Jetson AGX Orin (64GB) | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,374.4 | 1,490.1 | 2,048 | 46.2 |
| Jetson AGX Orin (64GB) | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,369.8 | 1,495.1 | 2,048 | 46.1 |
| Jetson AGX Orin (64GB) | Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 2,016.5 | 1,015.6 | 2,048 | 27.6 |
| Jetson AGX Orin (64GB) | Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 2,140.5 | 956.8 | 2,048 | 27.5 |
| Jetson Orin NX (16GB) | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 268.8 | 7,618.8 | 2,048 | 98.9 |
| Jetson Orin NX (16GB) | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 674.0 | 3,038.8 | 2,048 | 52.2 |
| Jetson Orin NX (16GB) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,880.5 | 1,089.1 | 2,048 | 29.2 |
| Jetson Orin NX (16GB) | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 2,944.1 | 695.6 | 2,048 | 17.5 |
| Jetson Orin NX (16GB) | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 760.3 | 2,693.8 | 2,048 | 52.6 |
| Jetson Orin NX (16GB) | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,740.8 | 1,176.4 | 2,048 | 29.4 |
| Jetson Orin NX (16GB) | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 3,106.2 | 659.3 | 2,048 | 17.4 |
| Jetson Orin NX (16GB) | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 840.8 | 2,435.8 | 2,048 | 92.8 |
| Jetson Orin NX (16GB) | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 787.2 | 2,601.6 | 2,048 | 94.1 |
| Jetson Orin NX (16GB) | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,160.8 | 1,764.2 | 2,048 | 47.0 |
| Jetson Orin NX (16GB) | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,173.4 | 1,745.3 | 2,048 | 47.3 |
| Jetson Orin NX (16GB) | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 2,557.7 | 800.7 | 2,048 | 26.6 |
| Jetson Orin NX (16GB) | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 2,775.8 | 737.8 | 2,048 | 26.8 |
| Jetson Orin Nano (8GB) | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 445.2 | 4,600.5 | 2,048 | 61.0 |
| Jetson Orin Nano (8GB) | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,021.0 | 2,005.9 | 2,048 | 32.3 |
| Jetson Orin Nano (8GB) | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,017.0 | 2,013.7 | 2,048 | 32.3 |
| Jetson Orin Nano (8GB) | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,380.8 | 1,483.2 | 2,048 | 57.5 |
| Jetson Orin Nano (8GB) | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,381.5 | 1,482.4 | 2,048 | 57.4 |
| Jetson Orin Nano (8GB) | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,885.2 | 1,086.4 | 2,048 | 29.0 |
| Jetson Orin Nano (8GB) | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,885.6 | 1,086.1 | 2,048 | 29.1 |
| DGX Spark (GB10) | gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 1 | 2,048 | 835.5 | 2,451.2 | 2,048 | 9.1 |
| DGX Spark (GB10) | gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,048 | 5,711.0 | 358.6 | 2,048 | 68.8 |
| DGX Spark (GB10) | gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 1 | 2,048 | 1,001.9 | 2,044.0 | 2,048 | 6.2 |
| DGX Spark (GB10) | gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,048 | 8,025.5 | 255.2 | 2,048 | 40.0 |
| DGX Spark (GB10) | gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,683.7 | 1,216.4 | 2,048 | 11.8 |
| DGX Spark (GB10) | gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 41,137.9 | 49.8 | 2,048 | 68.8 |
| DGX Spark (GB10) | nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 360.4 | 5,682.1 | 2,048 | 37.1 |
| DGX Spark (GB10) | nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 2,536.5 | 807.4 | 2,048 | 164.0 |
| DGX Spark (GB10) | nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 1,146.3 | 1,786.7 | 2,048 | 6.5 |
| DGX Spark (GB10) | nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 8,194.9 | 249.9 | 2,048 | 51.2 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 313.3 | 6,536.3 | 2,048 | 71.6 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,174.9 | 941.7 | 2,048 | 258.4 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 666.6 | 3,072.5 | 2,048 | 61.4 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,894.6 | 707.5 | 2,048 | 400.0 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 280.0 | 7,314.8 | 2,048 | 74.3 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,977.7 | 1,035.6 | 2,048 | 268.0 |
| DGX Spark (GB10) | Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 167.8 | 12,202.9 | 2,048 | 51.9 |
| DGX Spark (GB10) | Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,271.3 | 1,610.9 | 2,048 | 350.4 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 42.2 | 48,474.3 | 2,048 | 251.4 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 747.5 | 2,739.9 | 2,048 | 703.2 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 29.9 | 68,558.9 | 2,048 | 214.0 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 265.5 | 7,714.8 | 2,048 | 674.4 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 98.7 | 20,750.1 | 2,048 | 133.9 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 2,069.0 | 989.9 | 2,048 | 525.6 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 55.5 | 36,909.8 | 2,048 | 116.6 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 454.5 | 4,505.9 | 2,048 | 520.8 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 2,048 | 285.0 | 7,187.2 | 2,048 | 81.1 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 2,048 | 2,949.8 | 694.3 | 2,048 | 235.2 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 282.4 | 7,251.0 | 2,048 | 74.7 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,039.9 | 1,004.0 | 2,048 | 224.0 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 224.4 | 9,128.4 | 2,048 | 73.9 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 5,254.9 | 389.7 | 2,048 | 343.2 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 2,048 | 124.2 | 16,494.0 | 2,048 | 63.1 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 2,048 | 989.4 | 2,069.9 | 2,048 | 313.6 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 386.4 | 5,300.6 | 2,048 | 43.5 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 10,819.0 | 189.3 | 2,048 | 240.8 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 182.2 | 11,238.1 | 2,048 | 38.4 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,381.9 | 1,482.0 | 2,048 | 227.2 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 98.0 | 20,899.1 | 2,048 | 133.9 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 2,113.6 | 969.0 | 2,048 | 522.4 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 54.8 | 37,386.8 | 2,048 | 119.9 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 475.5 | 4,307.3 | 2,048 | 512.8 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 226.1 | 9,056.7 | 2,048 | 73.6 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 5,434.3 | 376.9 | 2,048 | 340.0 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 122.2 | 16,753.2 | 2,048 | 61.3 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 956.6 | 2,141.0 | 2,048 | 321.6 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 380.9 | 5,377.4 | 2,048 | 42.4 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 9,981.0 | 205.2 | 2,048 | 252.8 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 185.1 | 11,067.2 | 2,048 | 39.5 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,393.8 | 1,469.4 | 2,048 | 239.2 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 54.5 | 37,597.5 | 2,048 | 228.0 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 902.6 | 2,269.1 | 2,048 | 1,173.6 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 38.0 | 53,842.3 | 2,048 | 207.4 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 364.4 | 5,620.4 | 2,048 | 1,063.2 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 53.7 | 38,104.1 | 2,048 | 237.7 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 883.5 | 2,318.2 | 2,048 | 1,166.4 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 39.1 | 52,318.4 | 2,048 | 204.6 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 352.5 | 5,809.5 | 2,048 | 1,078.4 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 1,329.5 | 1,540.4 | 2,048 | 13.9 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 32,867.1 | 62.3 | 2,048 | 94.4 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 594.0 | 3,447.7 | 2,048 | 12.7 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 4,572.0 | 447.9 | 2,048 | 83.2 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 1,326.1 | 1,544.3 | 2,048 | 14.3 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 32,850.6 | 62.3 | 2,048 | 94.4 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 606.9 | 3,374.3 | 2,048 | 13.1 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 4,550.8 | 450.0 | 2,048 | 84.8 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 103.8 | 19,730.0 | 2,048 | 118.4 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 2,032.1 | 1,007.8 | 2,048 | 716.8 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 61.0 | 33,579.6 | 2,048 | 109.5 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 529.3 | 3,869.2 | 2,048 | 674.4 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 105.5 | 19,407.4 | 2,048 | 115.7 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 2,152.4 | 951.5 | 2,048 | 711.2 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 61.1 | 33,491.5 | 2,048 | 108.9 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 519.4 | 3,943.0 | 2,048 | 690.4 |
| DGX Spark (GB10) | Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 2,048 | 277.1 | 7,390.7 | 2,048 | 47.3 |
| DGX Spark (GB10) | Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 2,048 | 1,918.1 | 1,067.7 | 2,048 | 198.4 |
| DGX Spark (GB10) | Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 2,048 | 277.6 | 7,376.6 | 2,048 | 42.7 |
| DGX Spark (GB10) | Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 2,048 | 1,914.0 | 1,070.0 | 2,048 | 201.6 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 238.4 | 8,590.6 | 2,048 | 65.9 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 5,263.1 | 389.1 | 2,048 | 367.2 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 130.9 | 15,641.8 | 2,048 | 60.7 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,050.9 | 1,948.7 | 2,048 | 345.6 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 236.7 | 8,651.6 | 2,048 | 66.3 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 5,407.9 | 378.7 | 2,048 | 368.0 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 130.1 | 15,743.5 | 2,048 | 61.8 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,045.0 | 1,959.8 | 2,048 | 341.6 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 2,048 | 402.4 | 5,089.9 | 2,048 | 38.0 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 8 | 2,048 | 10,605.7 | 193.1 | 2,048 | 245.6 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 195.6 | 10,468.2 | 2,048 | 35.4 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,505.4 | 1,360.4 | 2,048 | 228.0 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 2,048 | 402.8 | 5,084.7 | 2,048 | 35.5 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,048 | 12,123.8 | 168.9 | 2,048 | 247.2 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 197.9 | 10,347.2 | 2,048 | 34.0 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,506.5 | 1,359.4 | 2,048 | 232.0 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 273.9 | 7,477.0 | 2,048 | 67.1 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,852.3 | 1,105.7 | 2,048 | 220.0 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 270.5 | 7,570.2 | 2,048 | 66.5 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,840.3 | 1,112.8 | 2,048 | 220.8 |

### v0.10.0 Runtime Performance Dashboard

Every row contains end-to-end decode throughput: generated-token throughput for vanilla decoding and accepted-token throughput for speculative decoding.

#### v0.10.0 Jetson AGX Thor

| Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| gemma-4-12B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 133.0 | 2,198.7 | 1.1 | 263 | 239,342.2 | 25.4 | 2.80 | 2,229 |
| gemma-4-12B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 617.8 | 3,380.8 | 6.8 | 1,882 | 277,517.9 | 131.3 | 2.79 | 2,242 |
| gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 146.8 | 1,992.3 | 1.1 | 264 | 240,083.8 | 10.2 | - | 2,240 |
| gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 654.2 | 3,192.9 | 6.8 | 1,882 | 277,109.2 | 37.3 | - | 2,254 |
| gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 220.5 | 1,326.4 | 83.8 | 263 | 3,143.1 | 7.6 | - | 3,002 |
| gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 1,300.8 | 1,605.7 | 595.9 | 1,882 | 3,157.7 | 33.0 | - | 3,028 |
| gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 504.1 | 580.1 | 83.2 | 263 | 3,166.2 | 13.3 | - | 2,997 |
| gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 3,368.7 | 620.0 | 598.6 | 1,882 | 3,143.0 | 60.0 | - | 3,038 |
| gemma-4-E2B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 36.5 | 8,020.2 | 16.7 | 263 | 15,768.6 | 86.7 | 1.99 | 5,032 |
| gemma-4-E2B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 124.8 | 16,735.6 | 146.2 | 1,882 | 12,872.4 | 476.9 | 1.98 | 5,008 |
| gemma-4-E2B-it | VLM | MTP | FP8 / FP8 / FP16 | 1 | 292 | 28.3 | 10,335.0 | 16.7 | 263 | 15,756.1 | 121.7 | 1.95 | 5,061 |
| gemma-4-E2B-it | VLM | MTP | FP8 / FP8 / FP16 | 8 | 2,089 | 102.2 | 20,433.9 | 146.3 | 1,882 | 12,861.7 | 613.1 | 1.97 | 5,029 |
| gemma-4-E2B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 25.3 | 11,544.6 | 16.8 | 263 | 15,679.2 | 142.7 | 1.89 | 5,032 |
| gemma-4-E2B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 93.8 | 22,269.3 | 146.2 | 1,882 | 12,865.4 | 681.5 | 1.89 | 5,032 |
| gemma-4-E2B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 36.6 | 7,988.9 | 16.6 | 263 | 15,823.3 | 50.3 | - | 4,988 |
| gemma-4-E2B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 122.5 | 17,049.5 | 146.0 | 1,882 | 12,886.3 | 327.7 | - | 4,974 |
| gemma-4-E2B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 28.6 | 10,217.3 | 16.8 | 263 | 15,685.9 | 79.0 | - | 5,024 |
| gemma-4-E2B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 97.9 | 21,332.0 | 145.5 | 1,882 | 12,934.4 | 485.1 | - | 5,043 |
| gemma-4-E2B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 47.7 | 6,136.0 | 16.7 | 263 | 15,775.4 | 110.9 | - | 4,996 |
| gemma-4-E2B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 226.0 | 9,240.5 | 145.8 | 1,882 | 12,904.6 | 636.0 | - | 4,987 |
| gemma-4-E2B-it | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 24.8 | 11,770.5 | 16.7 | 263 | 15,816.3 | 101.4 | - | 5,052 |
| gemma-4-E2B-it | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 94.3 | 22,157.1 | 146.3 | 1,882 | 12,862.9 | 551.5 | - | 5,011 |
| gemma-4-E4B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 59.8 | 4,887.9 | 16.9 | 263 | 15,604.5 | 45.9 | 1.95 | 5,894 |
| gemma-4-E4B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 226.9 | 9,203.9 | 145.6 | 1,882 | 12,926.6 | 242.3 | 1.98 | 5,932 |
| gemma-4-E4B-it | VLM | MTP | FP8 / FP8 / FP16 | 1 | 292 | 44.4 | 6,589.3 | 16.9 | 263 | 15,589.2 | 72.9 | 1.97 | 5,938 |
| gemma-4-E4B-it | VLM | MTP | FP8 / FP8 / FP16 | 8 | 2,089 | 156.1 | 13,377.5 | 145.8 | 1,882 | 12,901.9 | 376.8 | 1.97 | 5,954 |
| gemma-4-E4B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 36.1 | 8,111.1 | 16.9 | 263 | 15,573.9 | 96.4 | 1.99 | 5,926 |
| gemma-4-E4B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 140.2 | 14,894.4 | 146.1 | 1,882 | 12,875.0 | 469.1 | 2.03 | 5,926 |
| gemma-4-E4B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 61.0 | 4,790.4 | 16.7 | 263 | 15,797.1 | 25.4 | - | 5,877 |
| gemma-4-E4B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 225.6 | 9,259.9 | 145.3 | 1,882 | 12,951.3 | 147.5 | - | 5,891 |
| gemma-4-E4B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 45.1 | 6,487.7 | 16.7 | 263 | 15,805.1 | 42.7 | - | 5,939 |
| gemma-4-E4B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 157.9 | 13,228.0 | 145.5 | 1,882 | 12,931.6 | 245.5 | - | 5,960 |
| gemma-4-E4B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 84.8 | 3,447.3 | 16.8 | 263 | 15,692.8 | 61.9 | - | 5,916 |
| gemma-4-E4B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 456.1 | 4,579.2 | 145.2 | 1,882 | 12,960.3 | 363.6 | - | 5,919 |
| gemma-4-E4B-it | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 36.1 | 8,109.9 | 16.6 | 263 | 15,858.9 | 59.3 | - | 5,940 |
| gemma-4-E4B-it | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 138.9 | 15,042.1 | 145.4 | 1,882 | 12,941.4 | 350.0 | - | 5,972 |
| nvidia-Gemma-4-26B-A4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 74.2 | 3,941.9 | 83.3 | 263 | 3,161.5 | 72.4 | 2.80 | 1,730 |
| nvidia-Gemma-4-26B-A4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 245.7 | 8,502.8 | 586.2 | 1,882 | 3,209.9 | 224.9 | 2.81 | 1,720 |
| nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 74.2 | 3,940.4 | 80.9 | 263 | 3,254.9 | 43.2 | - | 1,725 |
| nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 245.8 | 8,496.8 | 580.4 | 1,882 | 3,241.7 | 142.5 | - | 1,805 |
| nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 173.5 | 1,685.9 | 81.6 | 263 | 3,228.4 | 8.0 | - | 2,995 |
| nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 979.2 | 2,133.1 | 585.4 | 1,882 | 3,214.1 | 31.7 | - | 3,021 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 66 | 75.6 | 870.8 | - | - | - | 77.1 | - | 966 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 470 | 162.5 | 2,894.9 | - | - | - | 225.1 | - | 932 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 66 | 39.0 | 1,686.9 | - | - | - | 67.1 | - | 1,071 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 470 | 165.2 | 2,846.2 | - | - | - | 371.4 | - | 1,061 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | DFlash | NVFP4 / NVFP4 | 1 | 66 | 75.0 | 878.4 | - | - | - | 58.7 | 2.22 | 945 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | DFlash | NVFP4 / NVFP4 | 8 | 470 | 237.9 | 1,977.1 | - | - | - | 146.9 | 2.22 | 966 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | MTP | NVFP4 / NVFP4 | 1 | 66 | 75.3 | 874.4 | - | - | - | 95.3 | 2.69 | 1,031 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | MTP | NVFP4 / NVFP4 | 8 | 470 | 236.5 | 1,988.5 | - | - | - | 229.4 | 2.69 | 1,028 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 66 | 74.7 | 881.9 | - | - | - | 81.6 | - | 984 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 470 | 237.2 | 1,982.4 | - | - | - | 236.0 | - | 935 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 376 | 25.2 | 14,909.4 | 23.3 | 349 | 14,965.7 | 177.3 | 4.72 | 1,350 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,688 | 180.3 | 14,907.0 | 178.0 | 2,495 | 14,015.8 | 525.2 | 4.65 | 1,349 |
| Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 376 | 25.0 | 15,033.9 | 23.4 | 349 | 14,928.8 | 61.0 | - | 1,422 |
| Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,688 | 179.4 | 14,984.3 | 177.4 | 2,495 | 14,062.5 | 307.9 | - | 1,446 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 8.2 | 7,425.0 | - | - | - | 319.7 | - | 614 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 27.6 | 15,839.0 | - | - | - | 1,624.8 | - | 680 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 61 | 6.6 | 9,278.1 | - | - | - | 306.4 | - | 620 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 437 | 14.9 | 29,434.0 | - | - | - | 1,680.5 | - | 682 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 61 | 8.7 | 7,046.1 | - | - | - | 336.9 | 3.02 | 870 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 437 | 22.1 | 19,786.5 | - | - | - | 917.7 | 3.01 | 855 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 14.6 | 4,205.4 | - | - | - | 154.1 | - | 867 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 70.1 | 6,240.1 | - | - | - | 903.2 | - | 903 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 61 | 10.3 | 5,962.7 | - | - | - | 153.6 | - | 902 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 437 | 24.0 | 18,187.9 | - | - | - | 862.2 | - | 867 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 61 | 72.7 | 841.4 | - | - | - | 84.4 | - | 14,115 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 437 | 244.7 | 1,786.7 | - | - | - | 261.4 | - | 14,119 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 61 | 56.0 | 1,093.7 | - | - | - | 89.8 | - | 865 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 437 | 97.1 | 4,503.7 | - | - | - | 252.3 | - | 880 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | 1 | 57 | 18.7 | 3,056.7 | - | - | - | 104.6 | 2.26 | 1,008 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | 8 | 409 | 45.0 | 9,084.2 | - | - | - | 468.7 | 2.24 | 1,003 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 57 | 30.8 | 1,857.8 | - | - | - | 83.3 | - | 1,006 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 409 | 165.4 | 2,470.2 | - | - | - | 515.0 | - | 1,034 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 57 | 18.6 | 3,070.6 | - | - | - | 79.8 | - | 1,012 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 409 | 45.1 | 9,049.2 | - | - | - | 513.1 | - | 1,038 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | 1 | 61 | 28.0 | 2,188.4 | - | - | - | 90.0 | 3.26 | 1,466 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | 8 | 437 | 65.2 | 6,702.0 | - | - | - | 409.8 | 3.16 | 1,549 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 61 | 24.1 | 2,539.4 | - | - | - | 153.7 | 4.16 | 1,485 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 437 | 60.6 | 7,214.6 | - | - | - | 474.5 | 4.08 | 1,459 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 52.2 | 1,173.1 | - | - | - | 49.0 | - | 1,521 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 313.7 | 1,393.5 | - | - | - | 307.9 | - | 1,452 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 61 | 27.7 | 2,208.9 | - | - | - | 47.3 | - | 1,469 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 437 | 65.0 | 6,724.9 | - | - | - | 300.3 | - | 1,468 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 29.3 | 9,982.1 | 11.7 | 265 | 22,706.7 | 151.1 | - | 916 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 176.4 | 11,841.7 | 74.8 | 1,896 | 25,331.1 | 831.8 | - | 920 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 12.7 | 23,095.3 | 11.4 | 265 | 23,282.7 | 151.9 | - | 939 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 50.5 | 41,358.4 | 75.0 | 1,896 | 25,290.1 | 779.0 | - | 916 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 292 | 19.8 | 14,796.4 | 11.8 | 265 | 22,506.9 | 254.0 | 4.56 | 1,069 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 96.3 | 21,688.9 | 75.7 | 1,896 | 25,063.7 | 643.7 | 4.49 | 1,065 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 64.5 | 4,535.6 | 11.8 | 265 | 22,592.3 | 82.8 | - | 1,069 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 448.4 | 4,658.5 | 75.8 | 1,896 | 25,013.2 | 492.2 | - | 1,087 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 22.3 | 13,112.6 | 11.6 | 265 | 22,946.2 | 79.2 | - | 1,059 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 98.4 | 21,226.0 | 75.5 | 1,896 | 25,100.7 | 374.4 | - | 1,070 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 115.1 | 2,541.4 | 15.7 | 266 | 16,956.4 | 48.5 | - | 1,550 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 874.5 | 2,388.7 | 109.1 | 1,896 | 17,378.5 | 297.0 | - | 1,507 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 32.1 | 9,109.7 | 15.7 | 266 | 16,880.5 | 47.1 | - | 1,510 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 168.3 | 12,410.5 | 108.5 | 1,896 | 17,472.3 | 244.2 | - | 1,495 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 11.4 | 5,414.9 | - | - | - | 240.0 | - | 755 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 40.8 | 10,825.3 | - | - | - | 1,087.8 | - | 765 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 1 | 62 | 10.5 | 5,867.1 | - | - | - | 236.0 | - | 751 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 8 | 441 | 27.6 | 15,971.3 | - | - | - | 1,219.9 | - | 833 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 12.5 | 23,747.3 | 3.8 | 266 | 69,946.2 | 325.6 | 2.06 | 812 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 53.5 | 39,546.7 | 26.5 | 1,896 | 71,618.2 | 1,047.8 | 2.07 | 863 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 18.7 | 15,830.9 | 3.8 | 266 | 70,242.4 | 239.8 | - | 868 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 95.4 | 22,193.2 | 26.4 | 1,896 | 71,874.2 | 996.3 | - | 802 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 13.2 | 22,394.0 | 3.8 | 265 | 69,231.2 | 235.5 | - | 840 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 54.9 | 38,557.7 | 26.5 | 1,896 | 71,514.0 | 1,063.7 | - | 802 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 165.7 | 372.6 | - | - | - | 15.4 | - | 2,692 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 1,125.9 | 391.8 | - | - | - | 95.1 | - | 2,705 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | 1 | 62 | 93.2 | 662.3 | - | - | - | 14.8 | - | 2,714 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | 8 | 441 | 284.6 | 1,550.2 | - | - | - | 88.0 | - | 2,701 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 118.0 | 2,511.5 | 14.5 | 265 | 18,280.1 | 21.6 | 2.56 | 2,806 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 675.8 | 3,133.3 | 103.7 | 1,896 | 18,281.9 | 60.4 | 2.49 | 2,759 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 110.4 | 2,684.1 | 14.4 | 265 | 18,417.1 | 35.0 | 2.82 | 2,754 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 664.8 | 3,185.4 | 103.2 | 1,896 | 18,364.6 | 145.9 | 2.78 | 2,823 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 394.2 | 752.0 | 14.5 | 266 | 18,323.0 | 15.4 | - | 2,758 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 2,991.2 | 707.9 | 103.1 | 1,896 | 18,384.7 | 90.6 | - | 2,739 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 117.5 | 2,523.5 | 14.4 | 265 | 18,382.9 | 14.8 | - | 2,747 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 670.0 | 3,160.5 | 103.4 | 1,896 | 18,345.5 | 74.9 | - | 2,760 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 18.7 | 3,308.4 | - | - | - | 123.6 | - | 1,237 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 79.5 | 5,546.5 | - | - | - | 635.6 | - | 1,241 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | 1 | 62 | 15.1 | 4,103.1 | - | - | - | 124.9 | - | 1,279 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | 8 | 441 | 36.1 | 12,210.7 | - | - | - | 736.4 | - | 1,239 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 15.8 | 18,767.1 | 10.9 | 265 | 24,286.8 | 216.0 | 2.40 | 1,284 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 71.7 | 29,522.3 | 72.4 | 1,896 | 26,174.8 | 620.2 | 2.40 | 1,300 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 33.8 | 8,759.0 | 10.9 | 265 | 24,447.4 | 124.5 | - | 1,287 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 193.3 | 10,955.7 | 72.6 | 1,896 | 26,111.0 | 608.6 | - | 1,279 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 18.2 | 16,324.2 | 10.9 | 265 | 24,307.3 | 125.4 | - | 1,305 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 73.4 | 28,839.3 | 72.2 | 1,896 | 26,249.4 | 444.2 | - | 1,282 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 62 | 66.5 | 929.3 | - | - | - | 48.2 | - | 15,666 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 441 | 217.5 | 2,028.0 | - | - | - | 199.0 | - | 15,652 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 296 | 116.4 | 2,547.0 | 14.3 | 265 | 18,510.3 | 47.5 | - | 15,683 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 2,118 | 473.0 | 4,476.8 | 101.8 | 1,896 | 18,627.2 | 189.7 | - | 15,720 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 36.9 | 1,675.4 | - | - | - | 70.9 | - | 1,488 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 193.7 | 2,277.3 | - | - | - | 377.0 | - | 1,522 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | 1 | 62 | 25.2 | 2,448.5 | - | - | - | 69.4 | - | 1,480 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | 8 | 441 | 72.6 | 6,078.8 | - | - | - | 388.6 | - | 1,536 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 32.1 | 9,246.2 | 10.9 | 266 | 24,269.1 | 74.2 | 2.27 | 1,530 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 151.8 | 13,951.8 | 73.1 | 1,896 | 25,936.5 | 179.1 | 2.24 | 1,595 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 28.5 | 10,400.6 | 10.9 | 265 | 24,257.5 | 126.6 | 2.46 | 1,583 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 148.3 | 14,279.1 | 72.8 | 1,896 | 26,061.8 | 412.1 | 2.51 | 1,527 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 72.6 | 4,085.9 | 11.0 | 265 | 24,210.6 | 71.2 | - | 1,554 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 488.2 | 4,337.7 | 72.7 | 1,896 | 26,065.9 | 319.7 | - | 1,529 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 31.6 | 9,379.1 | 10.9 | 265 | 24,348.3 | 69.5 | - | 1,538 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 151.2 | 14,005.8 | 72.8 | 1,896 | 26,042.9 | 311.2 | - | 1,523 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 58.8 | 1,050.1 | - | - | - | 41.4 | - | 2,208 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 344.0 | 1,282.5 | - | - | - | 242.3 | - | 2,259 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | 1 | 62 | 36.8 | 1,677.9 | - | - | - | 40.7 | - | 2,273 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | 8 | 441 | 96.1 | 4,589.4 | - | - | - | 240.4 | - | 2,231 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 44.6 | 6,653.0 | 14.4 | 265 | 18,462.4 | 47.0 | 2.40 | 2,312 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 221.3 | 9,569.1 | 101.6 | 1,896 | 18,666.0 | 139.9 | 2.35 | 2,267 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 39.3 | 7,541.7 | 14.4 | 265 | 18,404.3 | 88.0 | 2.69 | 2,310 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 214.1 | 9,889.8 | 102.0 | 1,896 | 18,586.0 | 346.7 | 2.70 | 2,296 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 124.4 | 2,383.0 | 14.4 | 265 | 18,424.2 | 41.9 | - | 2,260 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 902.2 | 2,347.2 | 101.6 | 1,896 | 18,670.4 | 215.4 | - | 2,277 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 44.0 | 6,743.0 | 14.5 | 266 | 18,348.3 | 40.9 | - | 2,270 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 219.4 | 9,651.0 | 101.3 | 1,896 | 18,722.3 | 191.1 | - | 2,267 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 109.6 | 2,703.8 | 14.6 | 266 | 18,212.4 | 34.4 | 2.75 | 2,764 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 662.3 | 3,197.3 | 103.2 | 1,896 | 18,381.6 | 135.7 | 2.76 | 2,782 |
| Qwen3.6-35B-A3B | LLM | MTP | NVFP4 / NVFP4 | 1 | 62 | 55.7 | 1,109.7 | - | - | - | 109.3 | 2.93 | 1,327 |
| Qwen3.6-35B-A3B | LLM | MTP | NVFP4 / NVFP4 | 8 | 441 | 136.8 | 3,224.0 | - | - | - | 293.6 | 2.94 | 1,292 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 1 | 62 | 55.7 | 1,108.0 | - | - | - | 86.6 | - | 1,286 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 8 | 441 | 136.9 | 3,221.8 | - | - | - | 252.7 | - | 1,232 |
| Qwen3.6-35B-A3B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 96.7 | 3,066.9 | 14.4 | 265 | 18,450.1 | 60.9 | 2.37 | 1,287 |
| Qwen3.6-35B-A3B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 241.9 | 8,754.6 | 101.8 | 1,896 | 18,625.9 | 129.5 | 2.32 | 1,336 |
| Qwen3.6-35B-A3B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 96.0 | 3,089.3 | 14.4 | 266 | 18,439.1 | 102.5 | 2.73 | 1,293 |
| Qwen3.6-35B-A3B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 244.3 | 8,666.6 | 101.2 | 1,896 | 18,728.7 | 290.3 | 2.76 | 1,331 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 96.7 | 3,067.2 | 14.4 | 265 | 18,462.2 | 86.6 | - | 1,292 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 242.4 | 8,736.4 | 101.7 | 1,896 | 18,653.9 | 268.3 | - | 1,312 |

#### v0.10.0 Jetson AGX Orin (64GB)

| Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 12.8 | 4,766.7 | - | - | - | 200.5 | - | 1,943 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 24.3 | 2,523.2 | - | - | - | 99.8 | - | 2,978 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 57 | 49.9 | 1,146.6 | - | - | - | 55.3 | - | 4,619 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 74.7 | 819.8 | - | - | - | 32.2 | - | 7,199 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 45.7 | 6,393.8 | 36.2 | 265 | 7,333.7 | 98.9 | - | 4,554 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 105.7 | 2,767.5 | 36.5 | 265 | 7,270.5 | 54.7 | - | 6,076 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 178.3 | 1,640.7 | 53.0 | 265 | 5,009.5 | 32.1 | - | 9,004 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 23.6 | 2,617.2 | - | - | - | 158.3 | - | 2,226 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 83.5 | 3,552.4 | 12.2 | 266 | 21,709.2 | 157.8 | - | 2,783 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 291.9 | 211.6 | - | - | - | 10.6 | - | 18,652 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 843.8 | 351.3 | 49.7 | 265 | 5,344.4 | 10.6 | - | 19,899 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 33.4 | 1,846.8 | - | - | - | 83.3 | - | 3,669 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 106.7 | 2,777.6 | 34.3 | 265 | 7,735.8 | 83.2 | - | 4,751 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 67.6 | 913.6 | - | - | - | 47.2 | - | 5,447 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 197.5 | 1,500.8 | 34.3 | 265 | 7,735.8 | 47.3 | - | 6,458 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 93.4 | 661.0 | - | - | - | 27.9 | - | 8,614 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 275.3 | 1,076.9 | 49.7 | 265 | 5,345.8 | 27.9 | - | 9,849 |

#### v0.10.0 Jetson Orin NX (16GB)

| Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 17.6 | 3,478.1 | - | - | - | 124.9 | - | 1,914 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 39.5 | 1,551.2 | - | - | - | 59.0 | - | 2,999 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 57 | 78.4 | 729.5 | - | - | - | 32.2 | - | 4,605 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 134.5 | 455.1 | - | - | - | 18.4 | - | 7,167 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 93.5 | 3,129.2 | 77.1 | 265 | 3,444.3 | 58.5 | - | 4,533 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 217.2 | 1,346.2 | 77.4 | 265 | 3,428.0 | 31.7 | - | 6,023 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 400.5 | 730.3 | 110.5 | 265 | 2,402.1 | 18.4 | - | 8,929 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 35.5 | 1,740.9 | - | - | - | 95.8 | - | 2,204 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 121.0 | 2,450.0 | 25.1 | 265 | 10,597.9 | 95.3 | - | 2,744 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 56.2 | 1,099.5 | - | - | - | 47.9 | - | 3,657 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 170.6 | 1,737.9 | 72.6 | 265 | 3,658.7 | 48.3 | - | 4,718 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 120.1 | 514.2 | - | - | - | 27.0 | - | 5,418 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 371.1 | 798.8 | 73.3 | 265 | 3,623.6 | 27.0 | - | 6,433 |

#### v0.10.0 Jetson Orin Nano (8GB)

| Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 30.7 | 1,992.7 | - | - | - | 77.0 | - | 1,917 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 69.8 | 876.9 | - | - | - | 36.5 | - | 2,992 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 167.2 | 1,749.6 | 141.3 | 265 | 1,879.2 | 36.1 | - | 4,486 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 63.2 | 977.2 | - | - | - | 59.1 | - | 2,127 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 215.6 | 1,375.0 | 45.4 | 265 | 5,851.3 | 59.0 | - | 2,760 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 101.4 | 609.0 | - | - | - | 29.6 | - | 3,642 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 306.8 | 966.4 | 133.4 | 265 | 1,989.9 | 29.6 | - | 4,692 |

#### v0.10.0 DGX Spark (GB10)

| Model | Kind | Mode | Precision | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| gemma-4-12B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 160.9 | 1,816.9 | 1.3 | 263 | 205,732.6 | 23.6 | 2.80 | 2,375 |
| gemma-4-12B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 677.0 | 3,085.3 | 6.3 | 1,882 | 297,851.7 | 118.7 | 2.80 | 2,364 |
| gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 183.5 | 1,593.4 | 1.2 | 263 | 226,656.3 | 9.4 | - | 2,380 |
| gemma-4-12B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 704.9 | 2,963.3 | 6.2 | 1,880 | 300,845.1 | 35.9 | - | 2,382 |
| gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 241.3 | 1,211.7 | 101.8 | 263 | 2,588.2 | 6.6 | - | 3,146 |
| gemma-4-31B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 997.8 | 2,093.2 | 772.2 | 1,882 | 2,436.5 | 26.6 | - | 3,147 |
| gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 277.6 | 1,053.5 | 105.8 | 263 | 2,489.0 | 12.7 | - | 3,143 |
| gemma-4-31B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 1,783.8 | 1,170.9 | 787.9 | 1,882 | 2,388.2 | 51.0 | - | 3,134 |
| gemma-4-E2B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 38.8 | 7,535.7 | 20.6 | 263 | 12,787.0 | 87.8 | 1.98 | 5,191 |
| gemma-4-E2B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 138.1 | 15,124.0 | 161.9 | 1,882 | 11,619.6 | 488.8 | 1.98 | 5,213 |
| gemma-4-E2B-it | VLM | MTP | FP8 / FP8 / FP16 | 1 | 292 | 25.4 | 11,495.6 | 20.6 | 263 | 12,757.7 | 117.6 | 1.95 | 5,134 |
| gemma-4-E2B-it | VLM | MTP | FP8 / FP8 / FP16 | 8 | 2,089 | 95.8 | 21,805.2 | 162.2 | 1,882 | 11,603.7 | 615.8 | 1.93 | 5,133 |
| gemma-4-E2B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 23.6 | 12,408.1 | 20.8 | 263 | 12,666.0 | 136.9 | 1.91 | 5,170 |
| gemma-4-E2B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 95.4 | 21,895.3 | 161.5 | 1,882 | 11,652.3 | 711.6 | 1.88 | 5,175 |
| gemma-4-E2B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 38.9 | 7,518.7 | 19.4 | 263 | 13,601.5 | 50.9 | - | 5,220 |
| gemma-4-E2B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 143.0 | 14,609.1 | 161.9 | 1,882 | 11,623.1 | 322.1 | - | 5,159 |
| gemma-4-E2B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 25.5 | 11,484.7 | 20.7 | 263 | 12,738.5 | 74.8 | - | 5,108 |
| gemma-4-E2B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 96.5 | 21,638.7 | 161.5 | 1,882 | 11,650.2 | 468.2 | - | 5,131 |
| gemma-4-E2B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 27.8 | 10,528.3 | 20.8 | 263 | 12,665.6 | 111.4 | - | 5,114 |
| gemma-4-E2B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 161.2 | 12,954.3 | 160.6 | 1,882 | 11,716.0 | 661.1 | - | 5,118 |
| gemma-4-E2B-it | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 22.5 | 13,007.5 | 20.8 | 263 | 12,650.3 | 93.7 | - | 5,161 |
| gemma-4-E2B-it | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 94.3 | 22,143.0 | 162.1 | 1,882 | 11,605.5 | 538.0 | - | 5,107 |
| gemma-4-E4B-it | VLM | MTP | FP16 / FP16 / FP16 | 1 | 292 | 69.9 | 4,184.3 | 19.6 | 263 | 13,468.1 | 42.8 | 1.98 | 6,147 |
| gemma-4-E4B-it | VLM | MTP | FP16 / FP16 / FP16 | 8 | 2,089 | 265.4 | 7,870.8 | 159.8 | 1,882 | 11,774.5 | 232.5 | 1.96 | 6,059 |
| gemma-4-E4B-it | VLM | MTP | FP8 / FP8 / FP16 | 1 | 292 | 43.8 | 6,674.8 | 21.2 | 263 | 12,418.3 | 67.9 | 1.99 | 6,047 |
| gemma-4-E4B-it | VLM | MTP | FP8 / FP8 / FP16 | 8 | 2,089 | 169.9 | 12,296.2 | 160.0 | 1,882 | 11,759.2 | 344.0 | 1.97 | 6,031 |
| gemma-4-E4B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 37.9 | 7,709.2 | 21.1 | 263 | 12,460.3 | 89.5 | 1.98 | 5,999 |
| gemma-4-E4B-it | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 157.9 | 13,229.9 | 158.4 | 1,882 | 11,879.8 | 499.8 | 2.00 | 6,091 |
| gemma-4-E4B-it | VLM | Vanilla | FP16 / FP16 | 1 | 292 | 71.2 | 4,108.4 | 21.2 | 263 | 12,448.3 | 24.0 | - | 6,134 |
| gemma-4-E4B-it | VLM | Vanilla | FP16 / FP16 | 8 | 2,089 | 264.0 | 7,912.7 | 162.0 | 1,882 | 11,617.7 | 146.4 | - | 6,070 |
| gemma-4-E4B-it | VLM | Vanilla | FP8 / FP16 | 1 | 292 | 44.6 | 6,558.8 | 20.9 | 263 | 12,578.3 | 37.7 | - | 6,026 |
| gemma-4-E4B-it | VLM | Vanilla | FP8 / FP16 | 8 | 2,089 | 169.7 | 12,311.2 | 162.3 | 1,882 | 11,594.4 | 215.0 | - | 6,041 |
| gemma-4-E4B-it | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 47.8 | 6,121.7 | 21.1 | 263 | 12,490.3 | 60.7 | - | 6,033 |
| gemma-4-E4B-it | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 337.0 | 6,197.8 | 162.5 | 1,882 | 11,581.5 | 359.0 | - | 6,040 |
| gemma-4-E4B-it | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 36.1 | 8,100.1 | 19.6 | 263 | 13,458.1 | 54.0 | - | 6,042 |
| gemma-4-E4B-it | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 152.8 | 13,669.1 | 163.7 | 1,882 | 11,493.8 | 319.6 | - | 6,091 |
| nvidia-Gemma-4-26B-A4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 292 | 107.6 | 2,717.0 | 110.3 | 263 | 2,387.9 | 66.6 | 2.91 | 1,849 |
| nvidia-Gemma-4-26B-A4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 367.7 | 5,679.8 | 762.1 | 1,882 | 2,468.9 | 194.6 | 2.83 | 1,850 |
| nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 101.7 | 2,874.5 | 105.6 | 263 | 2,493.8 | 38.9 | - | 1,854 |
| nvidia-Gemma-4-26B-A4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 368.1 | 5,674.1 | 777.8 | 1,882 | 2,419.1 | 105.0 | - | 1,854 |
| nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 239.8 | 1,219.2 | 108.7 | 263 | 2,423.5 | 6.9 | - | 3,149 |
| nvidia-Gemma-4-31B-IT | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 1,087.2 | 1,921.1 | 767.6 | 1,882 | 2,451.1 | 31.6 | - | 3,118 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 66 | 75.9 | 867.8 | - | - | - | 73.9 | - | 1,061 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 470 | 186.3 | 2,523.9 | - | - | - | 216.4 | - | 1,063 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 66 | 29.3 | 2,248.9 | - | - | - | 63.5 | - | 1,185 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 470 | 99.7 | 4,717.6 | - | - | - | 342.9 | - | 1,170 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | DFlash | NVFP4 / NVFP4 | 1 | 66 | 68.6 | 960.3 | - | - | - | 61.5 | 2.23 | 1,052 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | DFlash | NVFP4 / NVFP4 | 8 | 470 | 164.4 | 2,860.2 | - | - | - | 149.2 | 1.66 | 1,051 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | MTP | NVFP4 / NVFP4 | 1 | 66 | 68.6 | 959.7 | - | - | - | 93.2 | 2.70 | 1,068 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | MTP | NVFP4 / NVFP4 | 8 | 470 | 162.0 | 2,903.6 | - | - | - | 185.0 | 1.82 | 1,078 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 66 | 67.5 | 976.0 | - | - | - | 75.6 | - | 1,084 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 470 | 162.6 | 2,892.4 | - | - | - | 244.9 | - | 1,072 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 376 | 37.5 | 10,025.8 | 33.2 | 349 | 10,522.7 | 154.9 | 4.72 | 1,495 |
| Qwen2.5-VL-7B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,688 | 262.8 | 10,227.5 | 243.5 | 2,495 | 10,246.7 | 452.5 | 4.62 | 1,496 |
| Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 376 | 37.7 | 9,970.3 | 32.9 | 349 | 10,604.7 | 55.6 | - | 1,482 |
| Qwen2.5-VL-7B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,688 | 262.0 | 10,258.2 | 243.9 | 2,495 | 10,229.1 | 294.7 | - | 1,483 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 6.7 | 9,198.4 | - | - | - | 319.3 | - | 722 |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 17.7 | 24,694.6 | - | - | - | 1,788.2 | - | 760 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 61 | 5.8 | 10,564.7 | - | - | - | 283.2 | - | 756 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 437 | 13.8 | 31,670.5 | - | - | - | 1,725.8 | - | 805 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 61 | 8.7 | 7,037.3 | - | - | - | 291.3 | 2.96 | 1,005 |
| Qwen3-1.7B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 437 | 23.3 | 18,723.3 | - | - | - | 883.3 | 3.01 | 1,002 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 10.7 | 5,710.5 | - | - | - | 152.1 | - | 1,005 |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 45.1 | 9,690.8 | - | - | - | 939.3 | - | 1,007 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 61 | 10.3 | 5,958.8 | - | - | - | 136.5 | - | 1,007 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 437 | 24.8 | 17,597.0 | - | - | - | 902.2 | - | 975 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 61 | 63.1 | 970.4 | - | - | - | 85.2 | - | 14,236 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 437 | 147.7 | 2,960.0 | - | - | - | 268.8 | - | 13,715 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 61 | 62.9 | 973.6 | - | - | - | 82.1 | - | 983 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 437 | 148.5 | 2,943.4 | - | - | - | 226.5 | - | 987 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | 1 | 57 | 18.8 | 3,043.6 | - | - | - | 98.6 | 2.28 | 1,140 |
| Qwen3-4B-Instruct-2507 | LLM | DFlash | NVFP4 / NVFP4 | 8 | 409 | 52.5 | 7,779.0 | - | - | - | 421.9 | 2.19 | 1,150 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 1 | 57 | 20.1 | 2,850.3 | - | - | - | 80.2 | - | 1,144 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | 8 | 409 | 110.9 | 3,684.3 | - | - | - | 528.8 | - | 1,154 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 57 | 19.1 | 3,000.5 | - | - | - | 70.5 | - | 1,140 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 409 | 51.6 | 7,916.5 | - | - | - | 455.1 | - | 1,130 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | 1 | 61 | 31.7 | 1,928.3 | - | - | - | 77.2 | 3.19 | 1,598 |
| Qwen3-8B | LLM | DFlash | NVFP4 / NVFP4 | 8 | 437 | 81.9 | 5,335.8 | - | - | - | 347.5 | 3.05 | 1,592 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 1 | 61 | 27.6 | 2,220.3 | - | - | - | 127.5 | 3.99 | 1,599 |
| Qwen3-8B | LLM | EAGLE3 | NVFP4 / NVFP4 | 8 | 437 | 77.4 | 5,647.2 | - | - | - | 417.9 | 4.11 | 1,600 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 1 | 61 | 31.7 | 1,931.3 | - | - | - | 45.6 | - | 1,595 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | 8 | 437 | 185.9 | 2,351.8 | - | - | - | 309.3 | - | 1,589 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 61 | 29.8 | 2,056.4 | - | - | - | 41.2 | - | 1,598 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 437 | 81.3 | 5,380.0 | - | - | - | 265.2 | - | 1,594 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 18.5 | 15,836.3 | 14.3 | 265 | 18,612.3 | 150.0 | - | 1,052 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 125.2 | 16,688.8 | 91.8 | 1,896 | 20,650.7 | 847.1 | - | 1,043 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 13.2 | 22,167.6 | 14.2 | 266 | 18,659.2 | 137.1 | - | 1,054 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 65.6 | 31,839.5 | 94.6 | 1,896 | 20,052.6 | 654.2 | - | 1,053 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 1 | 292 | 26.0 | 11,256.5 | 14.0 | 266 | 18,994.3 | 227.2 | 4.54 | 1,201 |
| Qwen3-VL-4B-Instruct | VLM | EAGLE3 | NVFP4 / NVFP4 / FP16 | 8 | 2,089 | 134.8 | 15,502.5 | 90.0 | 1,896 | 21,057.2 | 628.2 | 4.45 | 1,202 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 39.4 | 7,422.9 | 14.1 | 265 | 18,882.4 | 80.6 | - | 1,186 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 329.8 | 6,334.9 | 90.1 | 1,896 | 21,050.5 | 493.3 | - | 1,202 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 26.7 | 10,967.5 | 14.6 | 266 | 18,235.9 | 67.4 | - | 1,204 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 143.1 | 14,600.6 | 91.9 | 1,896 | 20,639.7 | 340.1 | - | 1,201 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 292 | 66.0 | 4,429.9 | 21.5 | 266 | 12,367.1 | 44.0 | - | 1,649 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,089 | 506.4 | 4,124.9 | 138.8 | 1,896 | 13,659.0 | 303.4 | - | 1,633 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 292 | 40.8 | 7,174.8 | 21.3 | 265 | 12,469.9 | 42.2 | - | 1,626 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,089 | 234.6 | 8,905.5 | 138.3 | 1,896 | 13,712.2 | 238.3 | - | 1,632 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 8.7 | 7,099.0 | - | - | - | 235.1 | - | 899 |
| Qwen3.5-0.8B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 25.2 | 17,534.7 | - | - | - | 1,150.8 | - | 897 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 1 | 62 | 8.2 | 7,519.7 | - | - | - | 224.6 | - | 885 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 8 | 441 | 21.8 | 20,207.1 | - | - | - | 1,130.8 | - | 890 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 9.7 | 30,463.3 | 4.9 | 265 | 54,464.5 | 299.4 | 2.06 | 929 |
| Qwen3.5-0.8B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 51.9 | 40,806.1 | 29.9 | 1,896 | 63,461.6 | 1,006.8 | 2.08 | 1,028 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 12.6 | 23,528.1 | 4.8 | 266 | 54,754.3 | 233.4 | - | 945 |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 66.6 | 31,806.4 | 29.9 | 1,896 | 63,331.4 | 884.1 | - | 947 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 11.1 | 26,762.0 | 4.8 | 266 | 55,010.8 | 224.1 | - | 947 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 54.1 | 39,161.6 | 30.0 | 1,896 | 63,138.6 | 912.0 | - | 947 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 106.6 | 579.6 | - | - | - | 14.2 | - | 2,839 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 628.2 | 702.3 | - | - | - | 88.6 | - | 2,836 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | 1 | 62 | 99.7 | 619.2 | - | - | - | 12.6 | - | 2,836 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | 8 | 441 | 304.3 | 1,449.6 | - | - | - | 78.1 | - | 2,836 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 133.2 | 2,224.9 | 20.2 | 265 | 13,133.9 | 21.0 | 2.56 | 2,880 |
| Qwen3.5-27B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 886.7 | 2,388.1 | 137.9 | 1,896 | 13,750.6 | 65.2 | 2.52 | 2,877 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 126.0 | 2,353.6 | 20.3 | 265 | 13,091.4 | 32.2 | 2.82 | 2,880 |
| Qwen3.5-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 833.4 | 2,541.0 | 131.0 | 1,896 | 14,473.9 | 124.3 | 2.81 | 2,885 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 224.9 | 1,318.0 | 20.3 | 265 | 13,064.7 | 14.6 | - | 2,886 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 1,657.2 | 1,277.8 | 131.3 | 1,896 | 14,445.2 | 82.7 | - | 2,875 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 131.6 | 2,252.0 | 20.3 | 266 | 13,079.8 | 13.6 | - | 2,860 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 842.1 | 2,514.7 | 131.7 | 1,896 | 14,398.7 | 68.4 | - | 2,875 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 13.0 | 4,751.4 | - | - | - | 119.5 | - | 1,384 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 50.0 | 8,817.0 | - | - | - | 611.5 | - | 1,367 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | 1 | 62 | 12.6 | 4,901.5 | - | - | - | 117.6 | - | 1,383 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | 8 | 441 | 34.5 | 12,780.3 | - | - | - | 683.7 | - | 1,383 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 14.9 | 19,911.1 | 13.4 | 265 | 19,762.1 | 205.1 | 2.45 | 1,416 |
| Qwen3.5-2B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 78.0 | 27,159.0 | 90.1 | 1,896 | 21,054.9 | 613.7 | 2.38 | 1,420 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 21.4 | 13,829.8 | 13.3 | 265 | 19,967.5 | 119.0 | - | 1,432 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 132.8 | 15,940.4 | 90.7 | 1,896 | 20,903.7 | 538.8 | - | 1,427 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 16.3 | 18,177.3 | 13.7 | 265 | 19,342.8 | 115.8 | - | 1,430 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 79.6 | 26,603.6 | 88.8 | 1,896 | 21,347.5 | 411.1 | - | 1,420 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 1 | 62 | 61.0 | 1,012.4 | - | - | - | 47.9 | - | 15,740 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | 8 | 441 | 160.1 | 2,755.9 | - | - | - | 191.2 | - | 15,773 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 1 | 296 | 103.7 | 2,859.1 | 17.9 | 266 | 14,866.2 | 43.0 | - | 15,810 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | 8 | 2,118 | 329.4 | 6,429.1 | 128.7 | 1,896 | 14,729.6 | 182.0 | - | 15,067 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 23.5 | 2,626.5 | - | - | - | 67.2 | - | 1,623 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 127.5 | 3,459.4 | - | - | - | 352.0 | - | 1,626 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | 1 | 62 | 23.0 | 2,680.9 | - | - | - | 63.4 | - | 1,626 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | 8 | 441 | 72.3 | 6,104.1 | - | - | - | 356.1 | - | 1,626 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 34.3 | 8,637.5 | 13.3 | 265 | 20,000.9 | 73.7 | 2.25 | 1,675 |
| Qwen3.5-4B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 172.0 | 12,309.1 | 85.8 | 1,896 | 22,112.5 | 196.3 | 2.26 | 1,672 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 30.4 | 9,763.1 | 13.4 | 265 | 19,854.0 | 121.2 | 2.55 | 1,674 |
| Qwen3.5-4B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 169.5 | 12,491.8 | 87.2 | 1,896 | 21,740.1 | 385.5 | 2.49 | 1,659 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 43.7 | 6,784.0 | 13.3 | 266 | 20,026.3 | 68.0 | - | 1,673 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 374.2 | 5,658.9 | 86.2 | 1,896 | 22,010.9 | 313.9 | - | 1,675 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 33.2 | 8,929.7 | 13.2 | 265 | 20,076.2 | 65.1 | - | 1,652 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 171.8 | 12,327.6 | 86.1 | 1,896 | 22,023.7 | 263.6 | - | 1,670 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 1 | 62 | 36.6 | 1,687.9 | - | - | - | 38.3 | - | 2,340 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | 8 | 441 | 199.8 | 2,207.4 | - | - | - | 235.9 | - | 2,354 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | 1 | 62 | 36.6 | 1,689.0 | - | - | - | 36.5 | - | 2,351 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | 8 | 441 | 99.9 | 4,416.2 | - | - | - | 221.7 | - | 2,353 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 1 | 296 | 48.9 | 6,063.5 | 20.5 | 265 | 12,972.6 | 43.5 | 2.36 | 2,402 |
| Qwen3.5-9B | VLM | DFlash | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 270.7 | 7,821.7 | 131.1 | 1,896 | 14,462.1 | 148.5 | 2.39 | 2,400 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 44.3 | 6,692.9 | 20.1 | 265 | 13,204.3 | 79.3 | 2.69 | 2,403 |
| Qwen3.5-9B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 258.9 | 8,179.9 | 130.5 | 1,896 | 14,525.9 | 334.5 | 2.74 | 2,384 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 1 | 296 | 72.5 | 4,086.9 | 17.7 | 265 | 15,010.6 | 35.1 | - | 2,400 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | 8 | 2,118 | 533.0 | 3,972.6 | 130.4 | 1,896 | 14,539.7 | 213.5 | - | 2,401 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 49.0 | 6,054.0 | 19.9 | 265 | 13,304.5 | 35.0 | - | 2,402 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 266.5 | 7,947.1 | 132.0 | 1,896 | 14,369.2 | 180.0 | - | 2,390 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 126.6 | 2,342.4 | 20.6 | 265 | 12,898.2 | 31.4 | 2.74 | 2,885 |
| Qwen3.6-27B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 823.3 | 2,572.1 | 131.3 | 1,896 | 14,443.7 | 126.3 | 2.76 | 2,869 |
| Qwen3.6-35B-A3B | LLM | MTP | NVFP4 / NVFP4 | 1 | 62 | 67.6 | 913.9 | - | - | - | 102.3 | 2.94 | 1,361 |
| Qwen3.6-35B-A3B | LLM | MTP | NVFP4 / NVFP4 | 8 | 441 | 166.3 | 2,652.0 | - | - | - | 239.0 | 2.89 | 1,359 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 1 | 62 | 66.9 | 923.8 | - | - | - | 70.2 | - | 1,359 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 8 | 441 | 164.4 | 2,683.9 | - | - | - | 216.4 | - | 1,356 |
| Qwen3.6-35B-A3B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 1 | 296 | 120.1 | 2,469.1 | 20.7 | 265 | 12,810.5 | 96.7 | 2.74 | 1,408 |
| Qwen3.6-35B-A3B | VLM | MTP | NVFP4 / NVFP4 / FP16 | 8 | 2,118 | 330.2 | 6,413.3 | 129.4 | 1,896 | 14,647.5 | 249.2 | 2.72 | 1,409 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 296 | 119.0 | 2,491.3 | 17.5 | 265 | 15,176.9 | 69.1 | - | 1,412 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,118 | 332.3 | 6,372.0 | 128.2 | 1,896 | 14,786.4 | 226.2 | - | 1,408 |

### v0.10.0 Collection Method

- Engines were built from exported v0.10.0 ONNX artifacts using the build limits in [Export and Build Specs](#2-export-and-build-specs).
- Runtime throughput was collected with `llm_inference --warmup 10 --dumpProfile --profileOutputFile <profile.json>` using benchmark JSON inputs for each model family.
- Synthetic component timing was collected with `llm_bench --warmup 3 --iterations 10`; prefill uses `--inputLen 2048` and decode uses `--pastKVLen 2048`.
- Jetson AGX Thor and DGX Spark runs include the supported NVFP4 and INT4 entries. Jetson AGX Orin, Orin NX, and Orin Nano run the externalized INT4 entries supported by each memory target.

---
## v0.9.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.9.0 &nbsp;|&nbsp; **JetPack:** 7.2 &nbsp;|&nbsp; **DGX Spark:** CUDA 13.0, TensorRT 10.16.1.11 &nbsp;|&nbsp; **Devices:** Jetson AGX Thor, Jetson AGX Orin 64GB, Jetson Orin NX 16GB, Jetson Orin Nano 8GB, DGX Spark (GB10)

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
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 312.3 | 6,557.2 | 2,048 | 69.4 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,162.2 | 947.2 | 2,048 | 248.0 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 671.8 | 3,048.4 | 2,048 | 56.8 |
| DGX Spark (GB10) | NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 3,103.6 | 659.9 | 2,048 | 392.8 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 34.5 | 59,320.0 | 2,048 | 202.6 |
| DGX Spark (GB10) | Qwen3-0.6B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 310.5 | 6,595.2 | 2,048 | 620.8 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 60.5 | 33,860.2 | 2,048 | 113.8 |
| DGX Spark (GB10) | Qwen3-1.7B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 505.9 | 4,047.9 | 2,048 | 479.2 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 292.8 | 6,994.7 | 2,048 | 73.3 |
| DGX Spark (GB10) | Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,133.4 | 960.0 | 2,048 | 214.4 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 1 | 2,048 | 128.4 | 15,948.1 | 2,048 | 62.6 |
| DGX Spark (GB10) | Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,084.9 | 1,887.8 | 2,048 | 298.4 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 192.4 | 10,646.5 | 2,048 | 37.4 |
| DGX Spark (GB10) | Qwen3-8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 1,509.9 | 1,356.4 | 2,048 | 216.8 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 58.9 | 34,797.5 | 2,048 | 116.0 |
| DGX Spark (GB10) | Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 530.0 | 3,864.2 | 2,048 | 476.0 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 136.6 | 14,993.1 | 2,048 | 61.3 |
| DGX Spark (GB10) | Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,094.1 | 1,871.8 | 2,048 | 299.2 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 198.9 | 10,297.3 | 2,048 | 37.4 |
| DGX Spark (GB10) | Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 1,519.6 | 1,347.7 | 2,048 | 220.0 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 302.7 | 6,764.7 | 2,048 | 189.5 |
| DGX Spark (GB10) | Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 759.6 | 2,696.2 | 2,048 | 1,073.6 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 299.3 | 6,842.6 | 2,048 | 197.7 |
| DGX Spark (GB10) | Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 763.3 | 2,683.0 | 2,048 | 1,050.4 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 1,291.1 | 1,586.2 | 2,048 | 12.5 |
| DGX Spark (GB10) | Qwen3.5-27B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 7,144.6 | 286.6 | 2,048 | 84.8 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 1,306.0 | 1,568.2 | 2,048 | 12.4 |
| DGX Spark (GB10) | Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 7,211.7 | 284.0 | 2,048 | 83.2 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 327.0 | 6,263.6 | 2,048 | 106.9 |
| DGX Spark (GB10) | Qwen3.5-2B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 923.6 | 2,217.4 | 2,048 | 666.4 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 325.9 | 6,283.7 | 2,048 | 105.5 |
| DGX Spark (GB10) | Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 907.1 | 2,257.8 | 2,048 | 666.4 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 498.5 | 4,108.7 | 2,048 | 56.7 |
| DGX Spark (GB10) | Qwen3.5-4B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,099.3 | 975.6 | 2,048 | 340.8 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 498.4 | 4,109.3 | 2,048 | 58.2 |
| DGX Spark (GB10) | Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 2,130.9 | 961.1 | 2,048 | 344.0 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 556.6 | 3,679.8 | 2,048 | 34.7 |
| DGX Spark (GB10) | Qwen3.5-9B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 2,511.7 | 815.4 | 2,048 | 228.0 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 555.0 | 3,690.1 | 2,048 | 34.1 |
| DGX Spark (GB10) | Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 2,513.4 | 814.8 | 2,048 | 231.2 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 1 | 2,048 | 728.8 | 2,809.9 | 2,048 | 67.0 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | 8 | 2,048 | 3,069.1 | 667.3 | 2,048 | 214.4 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 728.0 | 2,813.1 | 2,048 | 67.1 |
| DGX Spark (GB10) | Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 3,075.0 | 666.0 | 2,048 | 212.0 |
| DGX Spark (GB10) | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 1 | 2,048 | 311.7 | 6,570.7 | 2,048 | 63.8 |
| DGX Spark (GB10) | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | 8 | 2,048 | 2,141.4 | 956.4 | 2,048 | 245.6 |

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

#### DGX Spark (GB10)

| Model | Kind | Mode | Precision | Dataset | Batch | Prefill Seq Len | Prefill Time (ms) | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Decode (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|------|------|-----------|---------|:-----:|----------------:|------------------:|----------------:|--------------:|------------:|------------:|---------------:|------------:|-------------:|
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | MTBench | 1 | 66 | 77.6 | 848.7 | - | - | - | 70.9 | - | 18,537 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | LLM | Vanilla | NVFP4 | MTBench | 8 | 470 | 186.2 | 2,525.9 | - | - | - | 207.9 | - | 18,713 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | MTBench | 1 | 66 | 30.2 | 2,180.7 | - | - | - | 59.4 | - | 3,611 |
| NVIDIA-Nemotron-3-Nano-4B | LLM | Vanilla | NVFP4 | MTBench | 8 | 470 | 107.6 | 4,370.8 | - | - | - | 330.4 | - | 3,610 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | MTBench | 1 | 61 | 6.9 | 8,896.0 | - | - | - | 273.7 | - | 1,027 |
| Qwen3-0.6B | LLM | Vanilla | NVFP4 | MTBench | 8 | 437 | 15.7 | 27,878.9 | - | - | - | 1,531.3 | - | 972 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | MTBench | 1 | 61 | 11.3 | 5,422.5 | - | - | - | 137.4 | - | 1,787 |
| Qwen3-1.7B | LLM | Vanilla | NVFP4 | MTBench | 8 | 437 | 27.6 | 15,851.2 | - | - | - | 824.5 | - | 1,845 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | MTBench | 1 | 61 | 54.7 | 1,118.6 | - | - | - | 84.9 | - | 14,052 |
| Qwen3-30B-A3B | LLM | Vanilla | INT4 GPTQ | MTBench | 8 | 437 | 147.3 | 2,966.8 | - | - | - | 232.2 | - | 13,883 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | MTBench | 1 | 61 | 65.5 | 934.3 | - | - | - | 80.0 | - | 17,193 |
| Qwen3-30B-A3B | LLM | Vanilla | NVFP4 | MTBench | 8 | 437 | 154.7 | 2,825.8 | - | - | - | 220.3 | - | 15,344 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | MTBench | 1 | 57 | 22.2 | 2,576.2 | - | - | - | 82.8 | - | 1,854 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | INT4 AWQ | MTBench | 8 | 409 | 114.4 | 3,571.0 | - | - | - | 389.3 | - | 1,857 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | MTBench | 1 | 57 | 20.4 | 2,805.8 | - | - | - | 71.8 | - | 3,152 |
| Qwen3-4B-Instruct-2507 | LLM | Vanilla | NVFP4 | MTBench | 8 | 409 | 55.6 | 7,345.9 | - | - | - | 435.6 | - | 3,207 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | MTBench | 1 | 61 | 34.7 | 1,765.8 | - | - | - | 47.4 | - | 3,210 |
| Qwen3-8B | LLM | Vanilla | INT4 AWQ | MTBench | 8 | 437 | 239.6 | 1,824.6 | - | - | - | 252.2 | - | 3,205 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | MTBench | 1 | 61 | 32.3 | 1,896.8 | - | - | - | 40.6 | - | 5,371 |
| Qwen3-8B | LLM | Vanilla | NVFP4 | MTBench | 8 | 437 | 87.6 | 4,987.6 | - | - | - | 258.3 | - | 5,367 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 12.2 | 24,061.0 | 15.3 | 265 | 17,391.3 | 136.8 | - | 1,891 |
| Qwen3-VL-2B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 67.6 | 30,902.3 | 93.0 | 1,896 | 20,366.6 | 618.5 | - | 1,893 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 48.3 | 6,048.9 | 14.9 | 265 | 17,825.3 | 79.1 | - | 1,846 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 311.4 | 6,708.7 | 97.3 | 1,896 | 19,493.2 | 332.4 | - | 1,902 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 27.8 | 10,510.7 | 14.7 | 265 | 18,050.5 | 68.5 | - | 3,195 |
| Qwen3-VL-4B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 149.2 | 13,998.4 | 92.8 | 1,896 | 20,408.2 | 310.2 | - | 3,192 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 292 | 87.8 | 3,330.2 | 23.0 | 265 | 11,520.7 | 46.5 | - | 3,252 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,089 | 618.5 | 3,377.4 | 143.9 | 1,896 | 13,175.2 | 239.5 | - | 3,193 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 292 | 40.6 | 7,201.1 | 22.7 | 265 | 11,723.3 | 40.1 | - | 5,473 |
| Qwen3-VL-8B-Instruct | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,089 | 245.9 | 8,495.5 | 143.9 | 1,896 | 13,175.2 | 223.0 | - | 5,281 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 13.7 | 4,523.5 | - | - | - | 212.6 | - | 1,245 |
| Qwen3.5-0.8B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 36.7 | 12,035.4 | - | - | - | 1,160.9 | - | 1,252 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 37.4 | 7,921.4 | 4.9 | 265 | 54,644.8 | 220.7 | - | 1,293 |
| Qwen3.5-0.8B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 106.9 | 19,815.2 | 30.4 | 1,896 | 62,500.0 | 902.1 | - | 1,296 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | MTBench | 1 | 62 | 134.1 | 460.7 | - | - | - | 14.6 | - | 8,681 |
| Qwen3.5-27B | LLM | Vanilla | INT4 AWQ | MTBench | 8 | 441 | 1,183.8 | 372.6 | - | - | - | 46.1 | - | 8,871 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 513.3 | 577.6 | 22.8 | 265 | 11,614.4 | 6.6 | - | 8,820 |
| Qwen3.5-27B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 3,088.9 | 685.5 | 141.8 | 1,896 | 13,369.0 | 69.3 | - | 8,930 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 110.8 | 557.2 | - | - | - | 12.9 | - | 16,079 |
| Qwen3.5-27B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 381.3 | 1,156.9 | - | - | - | 78.4 | - | 16,017 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 201.2 | 1,473.8 | 20.8 | 265 | 12,755.1 | 12.8 | - | 16,124 |
| Qwen3.5-27B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 1,191.1 | 1,777.8 | 138.9 | 1,896 | 13,661.2 | 66.2 | - | 16,121 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 18.4 | 3,364.8 | - | - | - | 115.0 | - | 2,207 |
| Qwen3.5-2B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 48.3 | 9,124.0 | - | - | - | 656.7 | - | 2,204 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 44.1 | 6,730.0 | 13.7 | 265 | 19,455.3 | 112.1 | - | 2,251 |
| Qwen3.5-2B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 131.4 | 16,114.6 | 86.9 | 1,896 | 21,834.1 | 456.9 | - | 2,254 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | MTBench | 1 | 62 | 69.8 | 884.9 | - | - | - | 47.3 | - | 15,349 |
| Qwen3.5-35B-A3B | LLM | Vanilla | INT4 GPTQ | MTBench | 8 | 441 | 203.5 | 2,167.4 | - | - | - | 191.7 | - | 15,341 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 1 | 296 | 147.3 | 2,013.2 | 20.9 | 265 | 12,706.5 | 45.6 | - | 15,666 |
| Qwen3.5-35B-A3B | VLM | Vanilla | INT4 GPTQ / FP16 | COCO | 8 | 2,118 | 494.0 | 4,286.6 | 137.3 | 1,896 | 13,812.2 | 174.7 | - | 15,470 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | MTBench | 1 | 62 | 35.3 | 1,749.5 | - | - | - | 67.9 | - | 1,694 |
| Qwen3.5-4B | LLM | Vanilla | INT4 AWQ | MTBench | 8 | 441 | 192.4 | 2,292.5 | - | - | - | 311.6 | - | 1,755 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 100.5 | 2,949.1 | 14.7 | 265 | 18,083.2 | 67.5 | - | 1,737 |
| Qwen3.5-4B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 554.0 | 3,822.1 | 115.5 | 1,896 | 16,420.4 | 258.9 | - | 1,742 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 31.2 | 1,980.0 | - | - | - | 59.2 | - | 3,587 |
| Qwen3.5-4B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 101.3 | 4,356.8 | - | - | - | 341.5 | - | 3,647 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 67.9 | 4,368.0 | 13.3 | 265 | 19,960.1 | 63.0 | - | 3,692 |
| Qwen3.5-4B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 295.1 | 7,176.8 | 92.6 | 1,896 | 20,491.8 | 240.4 | - | 3,691 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | MTBench | 1 | 62 | 49.9 | 1,238.6 | - | - | - | 40.1 | - | 2,855 |
| Qwen3.5-9B | LLM | Vanilla | INT4 AWQ | MTBench | 8 | 441 | 341.7 | 1,291.0 | - | - | - | 206.5 | - | 2,859 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 1 | 296 | 145.7 | 2,035.3 | 24.0 | 265 | 11,086.5 | 34.8 | - | 2,902 |
| Qwen3.5-9B | VLM | Vanilla | INT4 AWQ / FP16 | COCO | 8 | 2,118 | 914.5 | 2,315.6 | 138.2 | 1,896 | 13,717.4 | 197.8 | - | 2,961 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 44.6 | 1,384.6 | - | - | - | 35.8 | - | 6,112 |
| Qwen3.5-9B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 131.6 | 3,352.0 | - | - | - | 214.5 | - | 6,111 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 82.7 | 3,585.7 | 20.1 | 265 | 13,227.5 | 35.3 | - | 6,159 |
| Qwen3.5-9B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 396.1 | 5,345.9 | 139.8 | 1,896 | 13,568.5 | 180.5 | - | 6,157 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | MTBench | 1 | 62 | 77.3 | 799.3 | - | - | - | 72.1 | - | 19,464 |
| Qwen3.6-35B-A3B | LLM | Vanilla | NVFP4 | MTBench | 8 | 441 | 204.8 | 2,153.7 | - | - | - | 211.4 | - | 18,060 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 296 | 165.0 | 1,797.1 | 18.5 | 265 | 14,326.6 | 71.3 | - | 17,896 |
| Qwen3.6-35B-A3B | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 2,118 | 491.3 | 4,310.2 | 134.6 | 1,896 | 14,084.5 | 223.0 | - | 18,567 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | COCO | 1 | 1,699 | 275.6 | 6,163.7 | 157.0 | 1,664 | 10,593.2 | 64.9 | - | 19,790 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | VLM | Vanilla | NVFP4 / FP16 | COCO | 8 | 12,136 | 1,718.3 | 7,062.6 | 1,120.1 | 11,886 | 10,615.7 | 155.0 | - | 18,435 |

### v0.9.0 Collection Method

- Engines were built from exported v0.9.0 ONNX artifacts using the build limits in [Export and Build Specs](#2-export-and-build-specs).
- Runtime throughput was collected with `llm_inference --warmup 10 --dumpProfile --profileOutputFile <profile.json>` using benchmark JSON inputs for each model family.
- Synthetic component timing was collected with `llm_bench --warmup 3 --iterations 10`; prefill uses `--inputLen 2048` and decode uses `--pastKVLen 2048`.
- Jetson AGX Thor and DGX Spark runs include the supported NVFP4 and INT4 entries. Jetson AGX Orin, Orin NX, and Orin Nano run the externalized INT4 entries supported by each memory target.

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
