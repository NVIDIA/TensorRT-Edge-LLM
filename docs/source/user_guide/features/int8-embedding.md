# INT8 Embedding Sidecars

`--int8-embedding` writes the runtime token-embedding table,
`embedding.safetensors`, as symmetric INT8 with one FP32 scale per vocabulary
row. For Gemma 4 E-models it also writes the per-layer embedding table,
`ple_embedding.safetensors`, as symmetric INT8 with one FP32 scale per
vocabulary row and layer. The runtime gathers only the requested rows and
dequantizes them into the FP16 inputs; neither table is expanded to FP16 in
memory.

Unlike `--fp8-embedding`, which stores FP8 values with block scales for the
token embedding only, the INT8 format uses one scale per row and also covers
the Gemma 4 per-layer table, which has no FP8 option. The two options are
mutually exclusive. FP16 and FP8 token-embedding sidecars and FP16/BF16 PLE
sidecars load unchanged.

## Export

```bash
tensorrt-edgellm-export \
  /path/to/checkpoint \
  /path/to/onnx \
  --int8-embedding
```

The option applies to the runtime sidecars only. It does not change the ONNX
graph, the engine, or the checkpoint's linear-weight quantization.

## Sidecar Contract

The runtime identifies the format by the dtype of the table tensor. The
safetensors metadata records the format, version `1`, and the tensor role.

| File | Tensor | dtype | Shape | Metadata format |
|---|---|---|---|---|
| `embedding.safetensors` | `embedding` | INT8 | `[vocab, hidden]` | `int8_symmetric_per_row` |
| `embedding.safetensors` | `embedding_scale` | FP32 | `[vocab]` | |
| `ple_embedding.safetensors` | `weight` | INT8 | `[vocab, layers * ple_hidden]` | `int8_symmetric_column_groups_per_row` |
| `ple_embedding.safetensors` | `weight_scale` | FP32 | `[vocab, layers]` | |

Each row, or each row's per-layer group for the PLE table, is scaled by its own
maximum absolute value so that the payload uses the full `[-127, 127]` range;
an all-zero row or group uses scale `1.0`. The PLE table gets one scale per
layer because its row is the concatenation of every layer's input: a single
row scale would let an outlier in one layer cost precision in all the others.
The runtime reconstructs each element as `int8_value * scale` in FP16. In the
token-embedding lookup, image and audio placeholder positions receive their
FP16 encoder features exactly as with an FP16 table, and token IDs outside the
vocabulary or placeholders without a mapped feature row are zero-filled. The
PLE gather writes zeros for every placeholder position and out-of-vocabulary
ID, as the FP16/BF16 PLE gather does.

## Memory Example

For Gemma 4 E4B (vocabulary 262,144):

| Table | FP16 bytes | INT8 + FP32 scales bytes |
|---|---:|---:|
| `[262144, 2560]` token embedding | 1,342,177,280 | 672,137,216 |
| `[262144, 10752]` PLE | 5,637,144,576 | 2,862,612,480 |
| Combined | 6,979,321,856 | 3,534,749,696 |

The safetensors headers add a small amount beyond the tensor payloads.

## Limitations

- The experimental direct checkpoint builder has no INT8 embedding option; it
  writes FP16 tables (or FP8 with its own `--fp8-embedding`).
- A checkpoint-backed PLE table (direct checkpoint builder) cannot be INT8; the
  runtime rejects it at load.
- Qwen3-Omni models are not supported: the Talker runtime shares the Thinker
  embedding table and reads it as dense FP16.
- A DFlash or DSpark draft that carries its own mask-token embedding cannot be
  folded into a quantized `embedding.safetensors`; export such a base without
  `--int8-embedding`.
