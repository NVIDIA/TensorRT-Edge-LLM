# TTS (Text-to-Speech)

This guide covers the full pipeline for running Qwen3-TTS: export on x86 host, engine build on device, and inference.

**Supported models:** [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) and [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).

> **Note:** Unlike Qwen3-Omni, Qwen3-TTS has no Thinker or visual encoder. The text embedding is self-contained in the Talker and exported as `text_embedding.safetensors`.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Part 1: Export on x86 Host

Qwen3-TTS has three components: Talker, CodePredictor, and Code2Wav. Export all of them with `tensorrt-edgellm-export`.

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice
export ONNX_OUTPUT_DIR=$WORKSPACE_DIR/$TTS_MODEL/onnx

tensorrt-edgellm-export \
    Qwen/$TTS_MODEL \
    $ONNX_OUTPUT_DIR
```

### Expected Export Output

```
$ONNX_OUTPUT_DIR/
├── llm/
│   ├── model.onnx + model.onnx.data       # Talker ONNX
│   ├── config.json                        # model_type: qwen3_tts_talker
│   ├── embedding.safetensors              # codec embedding
│   ├── text_embedding.safetensors         # TTS-only (no Thinker)
│   ├── text_projection.safetensors
│   ├── tokenizer_config.json
│   ├── processed_chat_template.json
│   └── tokenizer files
├── code_predictor/
│   ├── model.onnx + model.onnx.data       # CodePredictor ONNX
│   ├── config.json
│   ├── codec_embeddings.safetensors
│   ├── lm_heads.safetensors
│   └── small_to_mtp_projection.safetensors  # if not Identity
└── code2wav/
    ├── model.onnx + model.onnx.data       # Code2Wav vocoder
    └── config.json
```

### Transfer to Device

```bash
scp -r $ONNX_OUTPUT_DIR <user>@<device>:~/tensorrt-edgellm-workspace/$TTS_MODEL/
```

---

## Part 2: Build Engines

Three engine builds are required. Run these on the edge device.

```bash
cd /path/to/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice
export ONNX=$WORKSPACE_DIR/$TTS_MODEL/onnx
export ENG=$WORKSPACE_DIR/$TTS_MODEL/engines

# 1. Build Talker LLM engine
./build/examples/llm/llm_build \
    --onnxDir $ONNX/llm \
    --engineDir $ENG/talker \
    --maxInputLen 4096 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1

# 2. Build CodePredictor LLM engine
./build/examples/llm/llm_build \
    --onnxDir $ONNX/code_predictor \
    --engineDir $ENG/code_predictor \
    --maxInputLen 4096 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1

# 3. Build Code2Wav engine
./build/examples/multimodal/audio_build \
    --onnxDir $ONNX/code2wav \
    --engineDir $ENG
```

`audio_build` writes the Code2Wav engine to `$ENG/code2wav`. Use `--engineDir $ENG`; passing `$ENG/code2wav` would create an extra nested directory.

> **Note:** Use `--maxBatchSize 1` for the current Qwen3-TTS runtime.

Build time: < 5 minutes

---

## Part 3: Run Inference

### Input File Format

Each request specifies a `messages` array and an optional per-request `speaker`. If omitted, the top-level `speaker` default is used.

```json
{
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "repetition_penalty": 1.05,
    "speaker": "ryan",
    "requests": [
        {
            "messages": [{"role": "assistant", "content": "Hello, how can I help you today?"}]
        },
        {
            "speaker": "serena",
            "messages": [{"role": "assistant", "content": "The weather is sunny and warm."}]
        }
    ]
}
```

**Available speakers:** `ryan`, `serena`, `aiden`, `vivian`, `dylan`, `eric`, `uncle_fu`, `ono_anna`, `sohee`

**Sampling parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `talker_temperature` | 0.9 | Sampling temperature |
| `talker_top_k` | 50 | Top-K sampling |
| `talker_top_p` | 1.0 | Top-P sampling |
| `repetition_penalty` | 1.05 | Penalize repeated codec tokens |
| `max_audio_length` | 4096 | Max codec frames per request |
| `speaker` | config default | Top-level speaker fallback |

### Run

```bash
cd /path/to/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice
export ENG=$WORKSPACE_DIR/$TTS_MODEL/engines

./build/examples/omni/qwen3_tts_inference \
    --talkerEngineDir   $ENG/talker \
    --code2wavEngineDir $ENG/code2wav \
    --tokenizerDir      $ENG/talker \
    --inputFile         input.json \
    --outputFile        output.json \
    --outputAudioDir    ./audio_output
```

Generated `.wav` files are named `audio_req{N}.wav` (one per request). The output JSON records per-request metadata: audio file path, sample count, duration, and RVQ code file path.

### Output JSON Example

```json
{
  "responses": [
    {
      "request_idx": 0,
      "messages": [{"role": "assistant", "content": "Hello, how can I help you today?"}],
      "audio_file": "./audio_output/audio_req0.wav",
      "audio_samples": 120960,
      "audio_sample_rate": 24000,
      "audio_duration_ms": 5040,
      "rvq_file": "./audio_output/rvq_req0.safetensors"
    }
  ]
}
```

### Streaming Mode

Pass `--streaming --chunkFrames=<N>` to vocode RVQ codes inline as Talker generates
them, rather than waiting for the full sequence. Each request receives its own
`onChunkReady` callback inside the runtime; the CLI synchronously vocodes the chunk
via Code2Wav and appends the PCM to a per-request WAV buffer. The final WAV file is
written when the request finishes, identical to the non-streaming path.

```bash
./build/examples/omni/qwen3_tts_inference \
    --talkerEngineDir   $ENG/talker \
    --code2wavEngineDir $ENG/code2wav \
    --tokenizerDir      $ENG/talker \
    --inputFile         input.json \
    --outputAudioDir    ./audio_output \
    --streaming --chunkFrames=10
```

**RVQ codes are bit-exact** vs the non-streaming path for the same input and seed —
streaming is purely an emission-layer change. Chunk-boundary PCM samples lose a small
amount of Code2Wav context (chunks are vocoded independently with no left-context
overlap), so the WAV is **not** byte-identical across chunk sizes; perceptual quality
and WER are unaffected for typical chunk sizes (≥ 10 frames).

**Choosing `chunkFrames`:** trades off TTFPA (time-to-first-playable-audio) vs vocoder
overhead. Smaller chunks ship audio sooner but call Code2Wav more often.

| `chunkFrames` | TTFPA (TTS 1.7B, single prompt) | Vocoder calls per 200 frames |
|---------------|--------------------------------|------------------------------|
| 1             | ~80 ms                         | 200 |
| 10            | ~190 ms                        | 20  |
| 50            | ~695 ms                        | 4   |

`--streaming` enables per-batch independent callbacks when the input JSON bundles
multiple requests into a Talker group (`batch_size > 1`); each request finishes and
emits its final WAV independently.

**Latency reporting:** with `--streaming`, the CLI logs `TTFC` (time-to-first-codec
token) and `TTFPA` per request using CUDA events:

```
[stream req 0] chunks=22 frames=215 samples=412800 TTFC=19.3ms TTFPA=189.9ms
```

For `batch_size == 1` runs, `timeToFirstAudioCodeMs` and `timeToFirstPlayableAudioMs`
are also populated in the profile JSON when `--profileOutputFile` is set.

---

## Notes

- `--code2wavEngineDir` is optional: auto-detected as `parent(talkerEngineDir)/code2wav` if not set.
- RVQ code files (`.safetensors`) are saved alongside audio when `--outputAudioDir` is set and can be used to re-synthesize audio without re-running the TTS model.
- `--streaming` requires an explicit `--chunkFrames=N` (no default — pick based on your latency/throughput target).
