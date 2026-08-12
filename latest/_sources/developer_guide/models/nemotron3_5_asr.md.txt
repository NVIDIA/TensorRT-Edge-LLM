# Nemotron-3.5-ASR

Nemotron-3.5-ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is a speech-transcription model. Unlike every other multimodal path
in Edge-LLM, it has **no LLM backbone**: it is an RNN-T (transducer) — a
FastConformer encoder plus an LSTM prediction network and a joint network.
Decoding is a greedy transducer loop, not autoregressive LLM generation, so it
does not reuse the LLM inference runtime.

Its Python export stays in the core package (`tensorrt_edgellm.models.nemotron3_5_asr`);
the C++ inference runtime lives under `experimental_models/nemotron3_5_asr/` and
builds only with `-DBUILD_EXPERIMENTAL_MODELS=ON`. The engine build itself uses
the shared `audio_build` tool — the encoder is a standard audio encoder — so
there is no dedicated builder here.

## Architecture

The model exports to **two** engines, driven by the runtime in a greedy loop:

- **Encoder** (`audio_encoder.engine`, dynamic length):
  `(input_features [1, T_mel, 128] fp16, prompt_ids [1] i64) -> encoder_frames [1, T, 640] fp16`.
  `T_mel` is the mel-frame count; three causal stride-2 subsampling stages make
  `T ≈ T_mel / 8`. Batch is fixed at 1 (the depthwise conv is a local operator;
  cross-clip padding would corrupt short-clip boundaries). `prompt_ids` is a
  language-prompt index fused into the encoder frames — `default_prompt_id`
  (101) means automatic language detection and the model emits an `<xx-XX>`
  tag; a specific index (e.g. `en-US` = 0) conditions on a known language.

- **RNN-T step** (`rnnt_step.engine`, fully static):
  `(decoder_input_ids [1, 1] i64, hidden_state/cell_state [L, 1, H] fp16, encoder_frame [1, H] fp16)`
  `-> (logits [1, V] fp16, present_hidden_state, present_cell_state)`.
  A single decode step has no sequence axis (one token, one frame, fixed
  LSTM/vocab dims), so the engine is static-shape and needs **no optimization
  profile**.

**Greedy loop** (must match HF greedy RNN-T exactly): the step engine always
runs; on a **blank** prediction the encoder-frame cursor advances and the present
LSTM states are discarded (blank never updates the prediction network); on a
**non-blank** prediction the token is emitted, the present states are adopted,
and the cursor stays on the frame. A forced advance after
`max_symbols_per_step` consecutive non-blanks bounds the loop; decoding stops
when the cursor passes the last encoder frame.

The mel front-end is the CPU `MelExtractor` (`nemotron_asr` config: nFFT 512,
hop 160, win 400, 128 mel bins, natural-log mel, no normalization), auto-selected
from the engine's `config.json`.

## 1. Export (x86 host, CPU-only)

```bash
tensorrt-edgellm-export "nvidia/nemotron-3.5-asr-streaming-0.6b" "$ONNX_DIR"
```

Produces `$ONNX_DIR/audio/` (encoder ONNX + `config.json` + tokenizer) and
`$ONNX_DIR/rnnt_decoder/` (step ONNX + a `config.json` carrying the
`rnnt_decoder_config` build marker).

## 2. Build engines

`audio_build` auto-detects the build type from each `config.json`
(`encoder_config` -> audio encoder; `rnnt_decoder_config` -> static RNN-T step):

```bash
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
"$BUILD_DIR/examples/multimodal/audio_build" \
    --onnxDir "$ONNX_DIR/audio"        --engineDir "$ENGINE_DIR" --maxTimeSteps 8192
"$BUILD_DIR/examples/multimodal/audio_build" \
    --onnxDir "$ONNX_DIR/rnnt_decoder" --engineDir "$ENGINE_DIR"
```

Then assemble one self-contained runtime dir:

```bash
mkdir -p "$RUN_DIR"
cp "$ENGINE_DIR/audio/audio_encoder.engine" "$ENGINE_DIR/rnnt_decoder/rnnt_step.engine" "$RUN_DIR/"
cp "$ONNX_DIR/audio/config.json" "$ONNX_DIR/audio/tokenizer.json" "$ONNX_DIR/audio/tokenizer_config.json" "$RUN_DIR/"
```

## 3. Run

```bash
"$BUILD_DIR/experimental_models/nemotron3_5_asr/examples/nemotron_asr_inference" \
    --engineDir "$RUN_DIR" --audioFile "$AUDIO"     # wav / mp3 / flac, mono
# --promptId <N> overrides the language prompt (default = automatic detection).
# --benchmark reports the mel / encoder / decode phase breakdown and RTF.
```

Output is the transcript with the emitted `<xx-XX>` language tag, plus frame /
step / token counts.

## Runtime design notes

- **Per-frame encoder slicing.** The encoder writes all `T` frames once; the
  greedy loop copies the current frame (device-to-device) into the step
  engine's fixed input slot each iteration and advances the cursor by the
  blank/emit rule — mirroring the HF reference, which gathers one frame per step.
- **LSTM state ping-pong.** Two state buffers (current / present); the present
  states are copied back over the current only on emit, so blank steps are free.
- **CUDA graph.** The whole step (engine `enqueueV3` + row-wise argmax) is
  captured once into a CUDA graph with fixed bindings and replayed per step.
- **Pinned host staging.** The mel H2D source and the per-step token H2D/D2H
  scalars are pinned (`cudaMallocHost`) so the async copies are truly
  asynchronous rather than falling back to synchronous pageable staging.
- **Batch 1.** Both the encoder (conv-boundary correctness) and the greedy step
  run at batch 1. The HF reference supports batched decode via per-stream frame
  cursors; a batched runtime is future work.
