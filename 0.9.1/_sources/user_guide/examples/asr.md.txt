# ASR (Automatic Speech Recognition)

Complete workflow for speech recognition with audio understanding capabilities.

**Example model:** Qwen3-ASR-0.6B

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Export language model and audio encoder
tensorrt-edgellm-export \
  Qwen/Qwen3-ASR-0.6B \
  $MODEL_NAME/onnx
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
# Set up workspace directory on device
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
cd /path/to/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096

# Build audio encoder engine
./build/examples/multimodal/audio_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/audio \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --minTimeSteps 1000 \
  --maxTimeSteps 3000
```

## Step 4: Run Inference (Thor Device)

Audio decode + mel-spectrogram extraction run in the C++ runtime via
vendored `miniaudio` + an in-tree mel extractor; no offline Python
preprocessing step is required. Supported containers: `.wav`, `.mp3`,
`.flac`. The mel pipeline (whisper / parakeet) is auto-derived from the
engine's `audio/config.json`, pinned by the model — mirroring HF / vLLM
where the FE choice ships with the model checkpoint.

Create an input file `$WORKSPACE_DIR/input_asr.json` (replace `/path/to/audio.wav` with the actual audio file path):

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": "/path/to/audio.wav"}]
                }
            ]
        }
    ]
}
```

Run inference:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --inputFile $WORKSPACE_DIR/input_asr.json \
  --outputFile $WORKSPACE_DIR/output_asr.json
```

Check `output_asr.json` for the speech recognition transcription.

---

## Quantization (Optional)

For deployments where engine size matters more than peak accuracy, the
new pipeline supports independent precision selection for the LLM
backbone and the audio encoder. Numbers below are measured on
Qwen3-ASR-0.6B; 1.7B is noted where it diverges.

### Supported precision combinations

| LLM backbone | Audio encoder | 0.6B status | 1.7B status |
|---|---|:---:|:---:|
| FP16 | FP16 | ✅ | ✅ |
| FP8 | FP16 | ✅ | ✅ |
| FP8 | FP8 | ✅ | ✅ |
| NVFP4 | FP16 | ✅ | ✅ |
| NVFP4 | FP8 | ❌ Empty output (combined quant noise exceeds the first-token EOS-vs-correct logit margin on 0.6B) | ✅ (1.7B tolerates the combined noise) |

> **Mixed-precision rules** (`tensorrt_edgellm/scripts/quantize.py`):
> - `--quantization` quantizes the LLM backbone only.
> - `--audio_quantization fp8` opts the audio tower in. **Omit it to
>   keep the audio tower at FP16** regardless of `--quantization` --
>   audio encoders are quantization-sensitive, so the default is
>   conservative (mirrors the `--visual_quantization` behaviour for
>   VLMs).
> - `--lm_head_quantization` is similarly opt-in.

### Three featured recipes

Insert a quantization step before [Step 1: Export](#step-1-export-x86-host),
then point the export at the quantized output instead of the Hugging
Face checkpoint. Joint multimodal calibration streams audio-transcript pairs
through the model using the default `librispeech` audio dataset. Pick another
registered audio dataset with `--audio_dataset <name>`; to add your own, see
[Calibration Dataset Customization](../../developer_guide/customization/calibration-datasets.md).

**Recipe A: 0.6B NVFP4 LLM + FP16 audio**

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-nvfp4-lh.nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Then in Step 1, replace ``Qwen/Qwen3-ASR-0.6B`` with the quantized dir:
tensorrt-edgellm-export \
  $WORKSPACE_DIR/$MODEL_NAME-nvfp4-lh.nvfp4 \
  $MODEL_NAME/onnx
```

**Recipe B: 0.6B FP8 LLM + FP8 audio**

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-fp8-a.fp8 \
  --quantization fp8 \
  --audio_quantization fp8

# Then in Step 1, replace ``Qwen/Qwen3-ASR-0.6B`` with the quantized dir:
tensorrt-edgellm-export \
  $WORKSPACE_DIR/$MODEL_NAME-fp8-a.fp8 \
  $MODEL_NAME/onnx
```

**Recipe C: 1.7B NVFP4 LLM + FP8 audio**

```bash
export MODEL_NAME=Qwen3-ASR-1.7B

tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-ASR-1.7B \
  --output_dir $WORKSPACE_DIR/$MODEL_NAME-nvfp4-a.fp8 \
  --quantization nvfp4 \
  --audio_quantization fp8

# Then in Step 1, replace ``Qwen/Qwen3-ASR-1.7B`` with the quantized dir:
tensorrt-edgellm-export \
  $WORKSPACE_DIR/$MODEL_NAME-nvfp4-a.fp8 \
  $MODEL_NAME/onnx
```
