# Omni (Audio + Vision + Speech I/O)

End-to-end workflow for **Qwen3-Omni** — a unified multimodal
model that ingests text + audio + image and emits text + speech. Covers
quantization, ONNX export, engine build, and inference.

## Supported models

| Model | Backbone | HF checkpoint | Recipes |
|-------|----------|---------------|---------|
| Qwen3-Omni-30B-A3B-Instruct | MoE (128 experts) | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | NVFP4 Thinker + Talker |

Omni uses a six-engine layout:

> **Thinker** (text decoder, generates assistant tokens) +
> **Talker** (text decoder, generates codec tokens) +
> **CodePredictor** (small head over codec tokens) +
> **Audio encoder** (Whisper-style mel) +
> **Visual encoder** (Qwen3-VL ViT) +
> **Code2Wav** (codec → 24 kHz PCM).

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Part 1: Quantize (NVFP4 Thinker + Talker)

First set the shell variables used throughout the remaining parts:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
mkdir -p $WORKSPACE_DIR

export OMNI_MODEL=Qwen3-Omni-30B-A3B-Instruct

export HF_ROOT=Qwen/$OMNI_MODEL                           # or a local snapshot dir
export ONNX=$WORKSPACE_DIR/$OMNI_MODEL/onnx
export ENG=$WORKSPACE_DIR/$OMNI_MODEL/engines
```

Thinker and Talker text-MoE backbones are jointly post-training quantized
to NVFP4 in a single ModelOpt pass. Calibration uses a multimodal dataset
(LibriSpeech audio + MMMU images + cnn_dailymail text) chained through the
FP16 Thinker so the Talker sees realistic `hidden_projection` /
`text_projection` inputs. The visual encoder, audio encoder, code2wav
vocoder, talker projections, and code-predictor head stay in FP16.

```bash
export QUANT_ROOT=$WORKSPACE_DIR/$OMNI_MODEL/nvfp4

tensorrt-edgellm-quantize qwen3-omni \
    --model_dir   $HF_ROOT \
    --output_dir  $QUANT_ROOT \
    --talker_num_audio 150 \
    --talker_num_image 150 \
    --talker_num_text  200
```

**Output:**

```
$QUANT_ROOT/
├── thinker/        # standalone HF NVFP4 ckpt, model_type=qwen3_omni_moe_text
└── talker/         # standalone HF NVFP4 ckpt, model_type=qwen3_omni_moe_talker
```

---

## Part 2: Export ONNX

`tensorrt-edgellm-export` exports every component the input checkpoint
supports. By default it runs every stage; pass `--components` to restrict
the run to a subset (useful when iterating on a single component).

### CLI reference (Omni-relevant flags)

| Flag | Description |
|------|-------------|
| `--components LIST` | Comma-separated allow-list (`thinker,talker,code_predictor,visual,audio,code2wav,action`). Default: export every component the checkpoint supports. |
| `--talker-sidecar-from PATH` | Path to a full HF root checkpoint. When set, the Talker stage extracts `hidden_projection.safetensors` and `text_projection.safetensors` from that root after exporting the LLM. Required because the standalone NVFP4 Talker checkpoint ships only the LLM backbone + codec embedding. |
| `--fp8-embedding` | Write `embedding.safetensors` in FP8 E4M3 (Thinker only). |
| `--reduced-vocab-dir DIR` | Use `vocab_map.safetensors` to shrink the LM head vocabulary. |
| `--skip-llm`, `--skip-visual`, `--skip-audio`, `--skip-code2wav` | Negative filters: skip these components entirely (applied on top of `--components`). |

### Export commands (three exports)

The NVFP4 quantizer writes Thinker and Talker as standalone checkpoints
(separate `config.json` per submodel). Visual / audio / code2wav /
code_predictor remain in the original HF root.

```bash
mkdir -p $ONNX

# Thinker: NVFP4 standalone Thinker checkpoint -> ONNX
tensorrt-edgellm-export $QUANT_ROOT/thinker $ONNX/thinker

# Talker: NVFP4 standalone Talker checkpoint -> ONNX + projection weights
tensorrt-edgellm-export \
    --talker-sidecar-from $HF_ROOT \
    $QUANT_ROOT/talker $ONNX/talker

# Visual + audio + code2wav + code_predictor: all from the HF root
tensorrt-edgellm-export \
    --components visual,audio,code2wav,code_predictor \
    $HF_ROOT $ONNX/multimodal
```

### Expected layout

```
$ONNX/
├── thinker/llm/                          # Thinker MoE ONNX
│   ├── model.onnx + model.onnx.data
│   ├── config.json                       # model: qwen3_omni_moe_text
│   ├── embedding.safetensors
│   ├── processed_chat_template.json
│   └── tokenizer files
├── talker/llm/                           # Talker MoE ONNX + sidecars
│   ├── model.onnx + model.onnx.data
│   ├── config.json                       # model: qwen3_omni_moe_talker
│   ├── embedding.safetensors             # codec embedding
│   ├── hidden_projection.safetensors     # extracted from HF root
│   └── text_projection.safetensors       # extracted from HF root
└── multimodal/
    ├── audio/                            # audio encoder
    ├── visual/                           # visual ViT
    ├── code2wav/                         # codec → PCM
    └── code_predictor/                   # codec head
```

---

## Part 3: Build Engines

Six engines total; `llm_build` covers Thinker / Talker / CodePredictor,
`audio_build` covers the audio encoder and Code2Wav, `visual_build` covers
the visual encoder.

```bash
export THINKER_ONNX=$ONNX/thinker/llm
export TALKER_ONNX=$ONNX/talker/llm
export CP_ONNX=$ONNX/multimodal/code_predictor
export AUDIO_ONNX=$ONNX/multimodal/audio
export CODE2WAV_ONNX=$ONNX/multimodal/code2wav
export VISUAL_ONNX=$ONNX/multimodal/visual
```

### Build commands

```bash
# 1. Thinker
./build/examples/llm/llm_build \
    --onnxDir $THINKER_ONNX \
    --engineDir $ENG/thinker \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 2048

# 2. Talker (sidecars live next to the engine; symlink or copy)
./build/examples/llm/llm_build \
    --onnxDir $TALKER_ONNX \
    --engineDir $ENG/talker \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 2048
ln -sf $TALKER_ONNX/hidden_projection.safetensors $ENG/talker/
ln -sf $TALKER_ONNX/text_projection.safetensors   $ENG/talker/

# 3. CodePredictor — must sit next to the Talker engine (the runtime
#    derives its path from --talkerEngineDir as <parent>/code_predictor)
./build/examples/llm/llm_build \
    --onnxDir $CP_ONNX \
    --engineDir $ENG/code_predictor \
    --maxBatchSize 1 \
    --maxInputLen 256 \
    --maxKVCacheCapacity 256

# 4. Code2Wav
./build/examples/multimodal/audio_build \
    --onnxDir $CODE2WAV_ONNX \
    --engineDir $ENG/code2wav \
    --maxCodeLen 1000

# 5. Audio encoder
./build/examples/multimodal/audio_build \
    --onnxDir $AUDIO_ONNX \
    --engineDir $ENG/multimodal/audio

# 6. Visual encoder
./build/examples/multimodal/visual_build \
    --onnxDir $VISUAL_ONNX \
    --engineDir $ENG/multimodal/visual
```

`--maxBatchSize` can be raised to the concurrency budget needed at
runtime. Place the CodePredictor engine in the same parent directory as
the Talker engine — the Qwen3-Omni runtime derives its path from
`--talkerEngineDir` (`<parent>/code_predictor`).

---

## Part 4: Preprocess Audio Input

Audio files must be converted to mel-spectrogram safetensors before
inference (same as ASR):

```bash
tensorrt-edgellm-preprocess-audio \
    --input  /path/to/audio.wav \
    --output $WORKSPACE_DIR/audio_input.safetensors
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.

---

## Part 5: Run Inference

`llm_inference` drives all four scenarios. Audio output is opt-in via
`--enableAudioOutput`; without that flag only the Thinker generates text.
The runtime detects model_type from the engine config.

### Scenario A: Text input → text output

```json
{
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user",   "content": [{"type": "text", "text": "What is the capital of France?"}]}
            ]
        }
    ]
}
```

```bash
./build/examples/llm/llm_inference \
    --engineDir           $ENG/thinker \
    --multimodalEngineDir $ENG/multimodal \
    --inputFile           input_text.json \
    --outputFile          output_text.json
```

### Scenario B: Audio input → text output

`messages[*].content` carries `{"type": "audio", "audio": "<preprocessed.safetensors>"}`.

### Scenario C: Image input → text output

`messages[*].content` carries `{"type": "image", "image": "<path/to/image.jpg>"}`.

### Scenario D: Text/audio/image input → text + speech output

```bash
./build/examples/llm/llm_inference \
    --engineDir              $ENG/thinker \
    --multimodalEngineDir    $ENG/multimodal \
    --talkerEngineDir        $ENG/talker \
    --code2wavEngineDir      $ENG/code2wav \
    --inputFile              input.json \
    --outputFile             output.json \
    --outputAudioDir         ./audio_output \
    --enableAudioOutput \
    --enableThinkerTalkerStreaming
```

`--enableThinkerTalkerStreaming` interleaves Talker prefill / decode with
Thinker token generation so the first audio chunk is emitted before the
Thinker text is fully complete, significantly reducing time-to-first-audio.

### Input file extra parameters (audio output)

| Parameter            | Default        | Description |
|----------------------|----------------|-------------|
| `talker_temperature` | 0.9            | Sampling temperature for codec tokens |
| `talker_top_k`       | 10             | Top-K (top_k=10 empirically best for English TTS) |
| `talker_top_p`       | 1.0            | Top-P (kept at 1.0 with low top_k) |
| `repetition_penalty` | 1.0            | Penalize repeated codec tokens |
| `max_audio_length`   | 4096           | Max codec frames per request |
| `speaker`            | config default | Speaker name: `chelsie`, `ethan`, or `aiden` |

### Output JSON Example

```json
{
  "responses": [
    {
      "request_idx": 0,
      "output_text": "The capital of France is Paris.",
      "audio_file": "./audio_output/audio_req0_batch0.wav",
      "audio_samples": 88830,
      "audio_sample_rate": 24000,
      "audio_duration_ms": 3700
    }
  ]
}
```

---

## Notes

- **Calibration quality:** the default `--talker_num_{audio,image,text}`
  values were tuned for English TTS. Reducing `--talker_num_text` below 100
  or removing modalities (audio/image) can regress both OmniBench and TTS
  WER.
- **Talker is the critical path:** the Talker MoE is heavier per step than
  the Thinker MoE on the same model (Talker 128 experts × routed top-6 +
  shared expert, vs Thinker 128 experts × top-8). In streaming mode the
  Talker drives the audio-out latency.
- **Talker `text_projection` MLP:** the Talker consumes the Thinker hidden
  state through a `text_projection` MLP sidecar. Build C++ with
  `-DENABLE_CUTE_DSL=gemm` — without it this MLP silently returns zeros on
  Ampere/Blackwell, producing garbled or empty Talker audio.
