# TTS (Text-to-Speech)

This guide covers the full pipeline for running Qwen3-TTS: export on x86 host, engine build on device, and inference.

The [supported model list](../getting_started/supported-models.md#speech-generation)
contains the validated CustomVoice, VoiceDesign, and Base checkpoints. They
share the Talker, CodePredictor, and Code2Wav pipeline; the runtime selects the
prompt contract from `tts_model_type` in the engine configuration.

## Precision & Quantization

| Component | Precision | Notes |
|---|---|---|
| Talker | FP16 | Quantized Talker checkpoints are not supported for Qwen3-TTS yet |
| CodePredictor | FP16, **FP8** | Quantize with `tensorrt-edgellm-quantize ... --cp_quantization fp8`; `down_proj`, LM heads, and KV-cache BMM remain FP16 |
| Code2Wav | FP16 | |
| Clone encoders (Base) | FP16 build from FP32 ONNX | x-vector cosine 1.0 / codes 100% vs reference at FP16 |

> **Note:** Unlike Qwen3-Omni, Qwen3-TTS has no Thinker or visual encoder. The text embedding is self-contained in the Talker and exported as `text_embedding.safetensors`.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Part 1: Export on x86 Host

Qwen3-TTS has three components: Talker, CodePredictor, and Code2Wav. Export all of them with `tensorrt-edgellm-export`.

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice

tensorrt-edgellm-export \
    "$MODEL_ID" \
    "$MODEL_ROOT/onnx"
```

### Expected Export Output

```
$MODEL_ROOT/onnx/
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
scp -r "$MODEL_ROOT/onnx" \
    <user>@<device>:~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-1.7B-CustomVoice/
```

---

## Part 2: Build Engines

Three engine builds are required. Run these on the edge device.

```bash
cd /path/to/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice
export ONNX=$MODEL_ROOT/onnx
export ENG=$MODEL_ROOT/engines

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
| `language` | `"auto"` | Top-level language fallback (see below) |

### Language Conditioning (CustomVoice)

CustomVoice checkpoints support explicit language conditioning: pass a `language` key
(top-level default and/or per-request override, same convention as `speaker`) and the
Talker prefill switches to the language-conditioned layout used by the PyTorch reference.

```json
{
    "speaker": "vivian",
    "language": "chinese",
    "requests": [
        {"messages": [{"role": "assistant", "content": "今天天气真不错。"}]},
        {"language": "english", "messages": [{"role": "assistant", "content": "The weather is nice today."}]}
    ]
}
```

**Supported language names** (from the checkpoint's `codec_language_id` map): `chinese`,
`english`, `german`, `italian`, `portuguese`, `spanish`, `japanese`, `korean`, `french`,
`russian`, `beijing_dialect`, `sichuan_dialect`. Names are matched case-insensitively.

Behavior notes:

- Omitting `language` (or setting `"auto"`) keeps the historical no-language prefill —
  output is byte-identical to engines/runtimes without language support.
- Unknown language names log a warning and fall back to `auto`.
- Dialect speakers (`eric` → `sichuan_dialect`, `dylan` → `beijing_dialect`) automatically
  activate their dialect conditioning when `language` is `auto` or `chinese`, matching the
  PyTorch reference behavior.
- Engines exported before language support (config.json without `codec_language_id`) ignore
  the field with a warning. Re-export with a CustomVoice checkpoint to pick up the map.

### Instruction Control

CustomVoice and VoiceDesign checkpoints accept a natural-language style instruction via the
`instruct` key (top-level default and/or per-request override):

```json
{
    "speaker": "ryan",
    "requests": [
        {"instruct": "Speak in a whisper, very softly",
         "messages": [{"role": "assistant", "content": "The weather is sunny and warm."}]}
    ]
}
```

The instruction is wrapped as a user turn, projected through `text_projection`, and prepended
to the Talker prefill, matching the PyTorch reference. Omitting `instruct` keeps the exact
historical prefill.

### VoiceDesign

VoiceDesign checkpoints (`tts_model_type: voice_design`) design the entire voice from the
instruction — there are no preset speakers and the prefill carries no speaker row. Export and
build work the same as CustomVoice; requests use `instruct` (+ optional `language`) and any
`speaker` field is ignored.

### Voice Clone (Base checkpoints)

Base checkpoints (`tts_model_type: base`) clone a voice from reference audio. The reference
encoders (ECAPA speaker encoder and the Mimi speech-tokenizer encoder) run **on device as
TensorRT engines** — export emits them automatically for Base checkpoints under
`clone_encoders/`:

```
onnx/clone_encoders/
├── speaker_encoder.onnx           # 24kHz wav (dynamic) -> x-vector; mel front-end folded in
└── speech_tokenizer_encoder.onnx  # 24kHz wav (static 40s bucket) -> RVQ codes [T, 16]
```

Build them alongside the other engines:

```bash
trtexec --onnx=$ONNX/clone_encoders/speaker_encoder.onnx --fp16 \
    --minShapes=wav:1x24000 --optShapes=wav:1x240000 --maxShapes=wav:1x960000 \
    --saveEngine=$ENG/clone_encoders/speaker_encoder.engine
trtexec --onnx=$ONNX/clone_encoders/speech_tokenizer_encoder.onnx --fp16 \
    --saveEngine=$ENG/clone_encoders/speech_tokenizer_encoder.engine
```

Pass `--cloneEncoderDir=$ENG/clone_encoders` to `qwen3_tts_inference` and reference audio
directly in requests (`ref_audio` / `ref_text`, top-level default and/or per-request):

```json
{
    "requests": [
        {"ref_audio": "/path/to/reference.wav",
         "messages": [{"role": "assistant", "content": "Cloned timbre only (x-vector mode)."}]},
        {"ref_audio": "/path/to/reference.wav",
         "ref_text": "exact transcript of the reference audio",
         "messages": [{"role": "assistant", "content": "Cloned timbre and prosody (ICL mode)."}]}
    ]
}
```

Any wav/mp3/flac sample rate is accepted (decoded and resampled to 24kHz internally).
Omitting `ref_text` clones timbre alone from the x-vector; providing it additionally
conditions in-context on the reference (transcript, codec codes) pair for closer prosody
matching. References longer than 40 s are truncated for the codec encoder.

### Sub-talker Sampling

The CodePredictor (sub-talker) sampling is independent from the Talker and can be tuned via
`subtalker_temperature` / `subtalker_top_k` / `subtalker_top_p`. Unset values fall back to
the reference implementation's hardcoded `code_predictor.generate` defaults
(temperature 1.0 / top-k 50 / top-p 0.8); they do not inherit the `talker_*` values.

### Run

```bash
cd /path/to/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_ROOT=$WORKSPACE_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice
export ENG=$MODEL_ROOT/engines

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

### Streaming Mode (audio output)

Pass `--streaming --chunkFrames=<N>` to vocode RVQ codes inline as the Talker generates
them, rather than waiting for the full sequence. Each request receives its own
`onChunkReady` callback inside the runtime (`bs >= 1` supported, per-batch independent);
the CLI synchronously vocodes each chunk via Code2Wav and appends the PCM to a
per-request WAV buffer.

Streaming behavior:

- **RVQ codes are bit-exact** vs the non-streaming path for every chunk size — streaming
  is purely an emission-layer change; WER is unchanged.
- **Waveforms may differ slightly at chunk boundaries**: each chunk is vocoded
  independently and Code2Wav resets its context per call, so a small amount of boundary
  context is lost. Acceptable for streaming playback; use non-streaming for offline
  comparison against reference audio.
- Time-to-first-codec-token is invariant to `chunkFrames`; time-to-first-playable-audio
  grows with it (one chunk must accumulate before the first Code2Wav call, ≈ `N / 12.5` s
  of audio content plus vocode time).

```bash
./build/examples/omni/qwen3_tts_inference \
    --talkerEngineDir   $ENG/talker \
    --code2wavEngineDir $ENG/code2wav \
    --tokenizerDir      $ENG/talker \
    --inputFile         input.json \
    --outputAudioDir    out/ \
    --streaming --chunkFrames=25
```

Works with every checkpoint family and voice-control feature above (speaker / language /
instruct / VoiceDesign / clone). Text input is consumed whole per request; streaming
*text* input (feeding a request's text incrementally) is not supported yet.

> **Not the same as Qwen3-Omni streaming.** Both paths share the Talker/CodePredictor
> engine framework and chunk accumulator, but they
> expose different streaming concepts with different configuration surfaces:
>
> | | Qwen3-TTS (`qwen3_tts_inference`) | Qwen3-Omni (`llm_inference`) |
> |---|---|---|
> | What streams | audio output only (chunked vocoding of a fixed text) | the full Thinker→Talker pipeline (speech synthesis starts while the Thinker is still generating text) |
> | Enabled via | CLI: `--streaming --chunkFrames=<N>` | input JSON: `"streaming": {"enable": true, "codec_chunk_frames": <N>, "talker_prefill_threshold": <M>}` |
> | Chunk knob | `--chunkFrames` | `codec_chunk_frames` |
