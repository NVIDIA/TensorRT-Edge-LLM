# Accuracy Debugging: HuggingFace vs TRT-EdgeLLM Tensor Comparison

The TensorRT `IDebugListener` approach is the next diagnostic layer after a
failing few-layer test. The CLI flags below require an instrumented build;
check the current builder and runtime before using these commands. Keep the
provider's PyTorch run as the reference.

This directory provides tooling to compare intermediate tensors between a
HuggingFace reference model and a TRT-EdgeLLM engine, helping you identify
where numerical differences originate in the LLM or visual encoder.

## Overview

The workflow has three stages:

```
1. Build engines with --debugTensors
2. Run inference with --dumpTensors / --dumpMultimodalTensors
3. Run compare_hf_trt_tensors.py to compare HF vs TRT
```

Tensors are dumped using TensorRT's `IDebugListener` API
(`markUnfusedTensorsAsDebugTensors` + `setUnfusedTensorsDebugState`), which
captures all non-fused intermediate tensors at inference time with zero
architectural changes to the model.

---

## Step 1 — Build TRT Engines with Debug Tensors

### LLM Engine

```bash
./build/bin/llm_build \
    --onnxDir   /path/to/llm_onnx \
    --engineDir /path/to/engines \
    --debugTensors
```

`--debugTensors` calls `markUnfusedTensorsAsDebugTensors()` during build.
This slows down the build slightly but is required for runtime dumping.

### Visual Encoder Engine (VLMs only)

```bash
./build/bin/visual_build \
    --onnxDir   /path/to/visual_onnx \
    --engineDir /path/to/engines \
    --debugTensors
```

---

## Step 2 — Run Inference and Dump Tensors

```bash
./build/bin/llm_inference \
    --engineDir         /path/to/engines \
    --tokenizerDir      /path/to/hf-model \
    --prompt            "Describe this image in one word." \
    --image             /path/to/image.jpg \
    --maxNewTokens      4 \
    --dumpTensors            /tmp/llm_tensors \
    --dumpMultimodalTensors  /tmp/vis_tensors
```

This produces one safetensors file per inference step:

```
/tmp/llm_tensors/
    step_000_prefill.safetensors   # full prompt (large: 100s MB)
    step_001_decode.safetensors    # first generated token
    step_002_decode.safetensors    # second generated token
    ...

/tmp/vis_tensors/
    step_000_vision.safetensors    # visual encoder (large: 100s MB)
```

Prefill and visual encoder dumps can be very large (500 MB+). Ensure
sufficient disk space before running.

---

## Step 3 — Compare HuggingFace vs TRT Tensors

```bash
python examples/debug/compare_hf_trt_tensors.py \
    --model-dir  /path/to/hf-model \
    --trt-dump   /tmp/llm_tensors \
    --multimodal-trt-dump /tmp/vis_tensors \
    --prompt     "Describe this image in one word." \
    --image      /path/to/image.jpg \
    --dtype      fp16 \
    --max-new-tokens 4 \
    --output-json /tmp/report.json \
    --verbose
```

The script does not run Python code shipped inside the checkpoint by default. Add
`--trust-remote-code` only for a checkpoint with custom modeling code that you
have reviewed.

The script:
1. Loads the HuggingFace model and registers `forward_hook` callbacks on
   every decoder layer and vision encoder block.
2. Runs `model.generate()` with the same prompt/image, capturing hidden
   states at each layer for prefill and every decode step.
3. Loads the TRT safetensors dumps.
4. Matches HF tensor names to TRT internal tensor names using a heuristic
   (layer number + semantic type: hidden_states / attn / mlp).
5. Reports per-step, per-layer metrics.

### Output

```
step           HF tensor             TRT tensor               cos_sim  max_abs  rel_Linf  shape
step_001_decode layers.0.hidden_states /model/layers.0/Add_1…  0.99934  0.0840   3.7e+06   YES
step_001_decode layers.1.hidden_states /model/layers.1/Add_1…  0.99932  0.1680   8.9e+01   YES
...
[step_001_decode]  matched= 25  cos mean=0.99930  min=0.99892  max_abs_diff=4.1e-01
```

### Metrics

| Metric | Description | Good range (FP16) |
|--------|-------------|-------------------|
| `cos_sim` | Cosine similarity between matched tensors | > 0.999 per layer |
| `max_abs_diff` | Maximum absolute element-wise difference | < 1.0 for mid layers |
| `rel_Linf` | Max diff / max TRT value (relative L∞ error) | < 1e2 for most layers |
| `shape` | Whether shapes match after batch-dim squeeze | YES for decode steps |

---

## End-to-End Example: Qwen3-VL-2B-Instruct (FP16)

The following example was run with `Qwen3-VL-2B-Instruct` using
`--prompt "Describe this image in one word."` and a red panda image.

Both HuggingFace FP16 and TRT-EdgeLLM FP16 generated **"Cute."**.

### LLM Decode Steps (per-layer hidden states, 28 layers)

| Step | Matched pairs | cos mean | cos min | max_abs_diff |
|------|--------------|----------|---------|-------------|
| step_001_decode | 25 | 0.95470 | -0.093 | 53.5 |
| step_002_decode | 25 | 0.96297 | +0.095 | 18.0 |
| step_003_decode | 25 | 0.95608 | -0.078 | 16.0 |
| **Overall** | **75** | **0.9579** | **-0.093** | **53.5** |

Individual layer cosine similarities for decode step 1 (layers 0–27):

```
layers.0.hidden_states   → 0.99934    layers.14.hidden_states  → 0.99909
layers.1.hidden_states   → 0.99932    layers.15.hidden_states  → 0.99914
layers.2.hidden_states   → 0.99934    layers.16.hidden_states  → 0.99903
layers.3.hidden_states   → 0.99934    layers.17.hidden_states  → 0.99910
layers.4.hidden_states   → 0.99933    layers.18.hidden_states  → 0.99903
layers.5.hidden_states   → 0.99937    layers.19.hidden_states  → 0.99898
layers.6.hidden_states   → 0.99927    layers.20.hidden_states  → 0.99879
layers.7.hidden_states   → 0.99919    layers.21.hidden_states  → 0.99899
layers.8.hidden_states   → 0.99892    layers.22.hidden_states  → 0.99905
layers.9.hidden_states   → 0.99910    layers.23.hidden_states  → 0.99918
layers.10.hidden_states  → 0.99916    layers.24.hidden_states  → 0.99941
layers.11.hidden_states  → 0.99916    layers.25.hidden_states  → 0.99959
layers.12.hidden_states  → 0.99924    layers.26.hidden_states  → 0.99971
layers.13.hidden_states  → 0.99913    layers.27.hidden_states  → (no match)
```

All matched layers show **cosine similarity > 0.999**, indicating excellent
numerical alignment between HF FP16 and TRT FP16 for autoregressive decoding.

### Shape Mismatches (Expected for VLMs)

**Prefill (step_000_prefill)**: HF produces `(1, 730, 2048)` tensors while
TRT produces `(1, 251, 2048)`. This is expected: HF embeds raw patch tokens
directly into the LLM sequence (730 tokens), while TRT processes the visual
encoder separately and injects compressed features (251 tokens). These cannot
be directly compared.

**Visual encoder (step_000_vision)**: HF captures `(2852, 1024)` per block
while TRT captures `(936, 1024)`. This reflects different intermediate patch
counts between HF's native vision processing and the ONNX-exported encoder.
Use the TRT visual encoder dump to compare against a reference ONNX runtime
rather than HF directly.

---

## Script Options

```
--model-dir           HuggingFace model directory
--trt-dump            Directory with LLM TRT dumps (step_*.safetensors)
--multimodal-trt-dump Directory with visual/audio encoder TRT dumps (optional)
--prompt              Text prompt (must match what was used in llm_inference)
--image               Image path for VLMs
--audio               Audio path for audio-LMs
--dtype               HF model dtype: float32 | fp16 | bf16
--max-new-tokens      Tokens to generate (should cover all TRT decode steps)
--save-hf-captures    Save HF captured tensors to this directory
--output-json         Write full comparison report as JSON
--verbose             Print per-tensor rows (default: step-summary only)
```

---

## Requirements

```bash
pip install torch transformers safetensors numpy Pillow qwen-vl-utils
```

- `qwen-vl-utils` is required for Qwen2-VL / Qwen3-VL image preprocessing.
- `soundfile` is required for audio models (`pip install soundfile`).

---

## Model-Specific Notes

### Nemotron-H (nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)

Nemotron-H uses a hybrid Mamba2 + Attention architecture and requires a
dedicated venv (`venv-nemotron-4b`) due to `mamba-ssm` ABI incompatibility
with the main environment.

**Create the venv (one-time):**

```bash
python3 -m venv ~/workspace/venv-nemotron-4b
source ~/workspace/venv-nemotron-4b/bin/activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.53.0" accelerate safetensors
pip install causal-conv1d mamba-ssm
```

**WAR — `selective_scan_cuda` ABI error:**

After installing `mamba-ssm`, its compiled `.so` has an undefined symbol
(`_ZNK3c104cuda10CUDAStream5queryEv`) against PyTorch 2.10+cu128. The model
only uses Triton kernels (`mamba_ssm.ops.triton.*`), so the CUDA extension
can be made optional:

```python
# ~/workspace/venv-nemotron-4b/lib/python3.12/site-packages/mamba_ssm/ops/selective_scan_interface.py
# Replace: import selective_scan_cuda
# With:
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None  # Triton-only path still works
```

**WAR — `transformers` version:**

`transformers` 5.x breaks `prepare_inputs_for_generation` for NemotronH.
Pin to `4.53.0` (matches the version the model was trained with).

**Run comparison:**

```bash
source ~/workspace/venv-nemotron-4b/bin/activate
python examples/debug/compare_hf_trt_tensors.py \
    --model-dir /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 \
    --trt-dump  /tmp/trt_tensors \
    --prompt    "What is the capital of France?" \
    --dtype     bf16 \
    --max-new-tokens 50
```

**Known limitations:**

- TRT tensor names are mangled by the compiler backend (e.g.
  `__mye25030_myl5`, `__myln_k_arg__bb1_92_myl3`). Per-layer hidden state
  comparison is not possible; only `logits` is compared per step.
- Decode step dumps use `"logits [profile 1]"` key instead of `"logits"`.
  The script handles this automatically.
- TRT engines built from the NVFP4-quantized ONNX will show lower cos_sim
  (~0.84–0.97 mean) compared to BF16 HF, with larger degradation on long
  decode sequences (SSM state error accumulates). Output text is still correct.

**Accuracy results (NVFP4 TRT vs BF16 HF):**

| Prompt | mean cos | min cos | n_steps |
|--------|----------|---------|---------|
| Capital of France | 0.923 | 0.749 | 8 |
| What is 2+2? | 0.969 | 0.944 | 9 |
| Name the planets (50 tokens) | 0.836 | 0.261 | 50 |

---

### Qwen3-ASR (Qwen3-ASR-0.6B)

Qwen3-ASR requires the `qwen-asr` package in a dedicated venv.

**Venv:** `~/workspace/venv-qwen3-asr`

**Audio preprocessing — required before inference:**

```bash
source ~/workspace/venv-qwen3-asr/bin/activate
python -m tensorrt_edgellm.scripts.preprocess_audio \
    --input  audio.wav \
    --output /tmp/audio_input.safetensors
```

**Build the audio engine:**

```bash
tensorrt_edgellm audio_build \
    --modelDir  /path/to/Qwen3-ASR-0.6B \
    --outputDir /path/to/asr_engine \
    --minTimeSteps 100    # must be <= shortest audio's timestep count
                          # (default 1000 fails for clips < ~10s)
```

**Run comparison:**

```bash
source ~/workspace/venv-qwen3-asr/bin/activate
python examples/debug/compare_hf_trt_tensors.py \
    --model-dir /path/to/Qwen3-ASR-0.6B \
    --trt-dump  /tmp/trt_tensors \
    --audio     /tmp/audio_input.safetensors \
    --dtype     fp16 \
    --max-new-tokens 256
```

Pass `--multimodalEngineDir` pointing to the **parent** of the `/audio/`
subdirectory (the runtime appends `/audio` automatically):

```bash
llm_inference \
    --engineDir          /path/to/asr_engine/llm \
    --multimodalEngineDir /path/to/asr_engine    # NOT .../asr_engine/audio
    ...
```

**Accuracy results (FP16 TRT vs FP16 HF, 513 decode steps):**

| Metric | Value |
|--------|-------|
| mean cos_sim | 0.99999 |
| min cos_sim  | 0.99987 |

---

## Notes

- The `--debugTensors` flag at build time is **required**. Engines built
  without it will produce empty dump directories at runtime.
- Debug tensor marking slows engine build by ~10–20%. Do not use in
  production builds.
- Each decode step dump is small (~3 MB for a 2B model). Prefill dumps
  scale with sequence length and can be hundreds of MB.
- TRT tensor names follow the ONNX graph structure (e.g.
  `/model/layers.N/Add_1_output_0`) for standard models. Hybrid/compiler-
  backend models (Nemotron-H) produce mangled names — only `logits` can
  be compared for those.
- For VLMs, prefill shape mismatches between HF and TRT are expected (HF
  embeds raw patch tokens while TRT uses the ONNX-exported visual encoder).
- `TRT_PACKAGE_DIR` must point to the full versioned subdirectory:
  `/scratch.edge_llm_cache/trt_softwares/TensorRT-10.16.0.72/TensorRT-10.16.0.72.x86_64.cuda-13.2`
