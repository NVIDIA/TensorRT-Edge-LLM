# INT8 Embedding Sidecar

`--int8-embedding` writes the runtime token-embedding table,
`embedding.safetensors`, as symmetric INT8 with one FP32 scale per vocabulary
row. The runtime gathers only the requested rows and dequantizes them into the
FP16 hidden states; the table is never expanded to FP16 in memory.

Unlike `--fp8-embedding`, which stores FP8 values with block scales, the INT8
format uses one FP32 scale per vocabulary row. The two options are mutually
exclusive. FP16 and FP8 sidecars load unchanged.

## Export

```bash
tensorrt-edgellm-export \
  /path/to/checkpoint \
  /path/to/onnx \
  --int8-embedding
```

The option applies to the runtime sidecar only. It does not change the ONNX
graph, the engine, or the checkpoint's linear-weight quantization.

## Sidecar Contract

The runtime identifies the format by the dtype of the `embedding` tensor. The
safetensors metadata records the format `int8_symmetric_per_row`, version `1`,
and tensor role `embedding`.

| Tensor | dtype | Shape |
|---|---|---|
| `embedding` | INT8 | `[vocab, hidden]` |
| `embedding_scale` | FP32 | `[vocab]` |

Each row is scaled by its own maximum absolute value so that the payload uses
the full `[-127, 127]` range; an all-zero row uses scale `1.0`. The runtime
reconstructs each element as `int8_value * row_scale` in FP16. Image and audio
placeholder positions receive their FP16 encoder features exactly as with an
FP16 table; token IDs outside the vocabulary, and placeholders without a mapped
feature row, are zero-filled.

## Memory Example

For a `[262144, 2560]` token-embedding table:

| Format | Bytes |
|---|---:|
| FP16 | 1,342,177,280 |
| INT8 + FP32 row scales | 672,137,216 |

The safetensors header adds a small amount beyond the tensor payloads.

## Limitations

- The experimental direct checkpoint builder has no INT8 embedding option; it
  writes FP16 tables (or FP8 with its own `--fp8-embedding`).
- Qwen3-Omni models are not supported: the Talker runtime shares the Thinker
  embedding table and reads it as dense FP16.
- A DFlash or DSpark draft that carries its own mask-token embedding cannot be
  folded into a quantized `embedding.safetensors`; export such a base without
  `--int8-embedding`.
