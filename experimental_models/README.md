# Experimental Models

This tree hosts experimental model integrations that are **not** first-class models in the core
`tensorrt_edgellm` (Python export) / `cpp` (C++ runtime) packages. Code here **links against and reuses** the
core libraries (KV cache, builders, kernels, ONNX export utilities, runner patterns) rather than duplicating
them, and follows the same runtime design conventions as `cpp/` (a runner owns its config, device buffers and
TensorRT context, and drives its loop in-process on the GPU).

It is built only when `-DBUILD_EXPERIMENTAL_MODELS=ON` is passed to CMake.

## Models

| Model | Path | Status |
|-------|------|--------|
| Cosmos3-Omni | `cosmos3/` | autoregressive text scaffolding + diffusion policy / action generation |
| Nemotron-3.5-ASR | `nemotron3_5_asr/` | offline (batch 1) RNN-T speech transcription (FastConformer encoder + LSTM prediction network; no LLM backbone) |
| InternVLA-N1 | `internvla_n1/` | dual-system vision-language navigation (Qwen2.5-VL planner served by the core runtime + flow-matching trajectory expert; asynchronous in-process handoff) |

See each model's `README.md` (e.g. `cosmos3/README.md`, `nemotron3_5_asr/README.md`, `internvla_n1/README.md`) for details.
