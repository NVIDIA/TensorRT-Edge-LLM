# Nemotron-3.5-ASR (experimental runtime)

Nemotron-3.5-ASR has both checkpoint-direct model definitions under
`experimental/builder/models/nemotron3_5_asr` and the maintained ONNX exporter
under `tensorrt_edgellm.models.nemotron3_5_asr`. This directory owns its C++
RNN-T transcription runtime and inference CLI. Its restricted Python server is
packaged from `experimental/nemotron3_5_asr`.

The model is an offline (batch 1) RNN-T transducer — a FastConformer encoder plus an LSTM prediction network and joint network — with **no LLM backbone**. Decoding is a greedy transducer loop over two engines (a dynamic-length encoder and a static-shape single-step engine), not autoregressive LLM generation, which is why the runtime does not reuse the LLM inference path.

The supported input is the HF checkpoint `nvidia/nemotron-3.5-asr-streaming-0.6b`.

## Layout

```text
nemotron3_5_asr/
  cpp/        # experimental C++ RNN-T transcription runtime
  examples/   # nemotron_asr_inference CLI
```

The experimental checkpoint-direct builder creates the encoder and RNN-T step
engines in one command. The maintained ONNX exporter and shared `audio_build`
executable produce the same runtime contract.

## Quickstart

The `nemotron_asr_inference` binary is produced by the standard Edge-LLM
experimental-model build (configure with `-DBUILD_EXPERIMENTAL_MODELS=ON`).
`$BUILD_DIR` below points at that build tree.

```bash
# 1. Build both engines directly from the provider checkpoint. The output is
#    immediately consumable by the runtime.
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
    --local-dir "$CHECKPOINT_DIR"
tensorrt-edgellm-build \
    --model-dir "$CHECKPOINT_DIR" \
    --engine-dir "$ENGINE_DIR" \
    --max-time-steps 8192

# 2. Transcribe. --promptId selects the language prompt (default = config
#    default_prompt_id = automatic language detection; the model emits an
#    <xx-XX> language tag).
"$BUILD_DIR/experimental_models/nemotron3_5_asr/examples/nemotron_asr_inference" \
    --engineDir "$ENGINE_DIR" --audioFile "$AUDIO"     # wav / mp3 / flac

# 3. Or serve the complete model bundle through the isolated experimental API.
#    This source module is not installed as a console command. Server startup
#    only loads the bundle; it never builds engines.
PYTHONPATH="$BUILD_DIR/pybind${PYTHONPATH:+:$PYTHONPATH}" \
python -m experimental.nemotron3_5_asr.server "$ENGINE_DIR" \
    --served-model-name nemotron-3.5-asr-streaming-0.6b
```

The maintained ONNX route exports both components, builds them with
`audio_build`, and assembles a flat runtime directory. Serve it with:

```bash
PYTHONPATH="$BUILD_DIR/pybind${PYTHONPATH:+:$PYTHONPATH}" \
python -m experimental.nemotron3_5_asr.server "$RUN_DIR" \
    --served-model-name nemotron-3.5-asr-streaming-0.6b
```

See `docs/source/developer_guide/models/nemotron3_5_asr.md` for exact ONNX
export/build/assembly commands, the HTTP request contract, and runtime design.
