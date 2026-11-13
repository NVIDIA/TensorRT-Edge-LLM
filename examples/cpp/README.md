# C++ Applications

This directory contains the TensorRT Edge-LLM C++ applications. Every
application includes Edge-LLM headers through `tensorrt_edgellm/` and links to
the exported `TensorRTEdgeLLM` CMake targets.

The applications are built with the main project and installed under `bin`.
They can also be compiled against an installed SDK:

```bash
cmake -S examples/cpp -B build/cpp-examples \
    -DCMAKE_PREFIX_PATH=/path/to/tensorrt-edge-llm \
    -DTRT_PACKAGE_DIR=/path/to/TensorRT
cmake --build build/cpp-examples --parallel
```

The builder applications are `llm_build`, `visual_build`, `audio_build`, and
`action_build`. Runtime applications are `llm_inference`, `llm_stream`,
`llm_bench`, `action_inference`, `qwen3_tts_inference`, and
`multistage_inference`.

`multistage_inference --config <dir>` runs the stage graph that
`tensorrt-edgellm-build` writes into an engine directory as
`multistage_config.json`. For Qwen3-Omni and Qwen3-TTS, it runs the same
engines as `llm_inference --enable-audio-output` and `qwen3_tts_inference` and
reads their request files; for example,
`multistage_inference --config <dir> --prompt "Hello" --out-audio hello.wav`
saves the speech to `hello.wav`. See the
[multi-stage graph guide](../../docs/source/developer_guide/software-design/multi-stage-graph.md)
for its usage and how it differs from those binaries.
