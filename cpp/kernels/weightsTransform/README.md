# Runtime Weight Transforms

These kernels convert provider checkpoint tensors directly into the immutable
layouts bound as TensorRT network inputs. The directory hierarchy names both
the checkpoint precision and the consuming operation contract.

## Design Contract

- The experimental builder records a final engine-input shape, dtype, source
  checkpoint keys, and a transform recipe. It does not encode a generic
  intermediate weight layout.
- Runtime maps only the file-backed page ranges for the keys named by those
  recipes. `mmap` reserves a virtual range; it does not copy the checkpoint.
  CUDA registration is likewise limited to the merged pages containing the
  selected tensors.
- One persistent GPU arena owns every final engine-input weight. Transform
  kernels write directly into non-owning `rt::Tensor` views of that arena; they
  allocate no temporary device buffers and perform no H2D or D2H copies.
- Recipes may consume another prepared engine input. The runtime materializes
  prerequisite bindings first (currently GPTQ activation permutations), then
  transforms the dependent weights into the same arena.
- A checkpoint source deliberately uses a dual-address view rather than
  `rt::Tensor`: validation reads a const host alias while kernels consume the
  CUDA alias of the same mapped pages. `rt::Tensor` represents one mutable
  address on one device and cannot preserve that contract.
- Kernels that assemble per-expert tensors accept eight source pointers per
  launch to keep parameters out of thread-local memory. This is only a launch
  batch size. Host orchestration processes any number of experts in consecutive
  batches.
- Source mappings stay alive until the startup stream is synchronized. They are
  released before the first inference enqueue; only the final arena remains.

| Path | Destination contract | Checkpoint formats |
|---|---|---|
| `fp16/linear` | TensorRT native `MatrixMultiply` weights | FP16, BF16, FP32 |
| `fp16/moe` | `Fp16MoePlugin` | Per-expert FP16, BF16, FP32 |
| `int4/groupwiseGemm/v1` | `Int4GroupwiseGemmPlugin` | ModelOpt AWQ, AWQ, GPTQ |
| `int4/groupwiseGemm/v2` | `Int4GroupwiseGemmPluginV2` | ModelOpt AWQ, AWQ, GPTQ |
| `int4/moe/gptqMarlin` | `Int4MoePlugin` Marlin weights and scales | GPTQ |
| `iq2_xxs` | `W2A16GemmPlugin` and `W2A16MoePlugin` payload and FP16 bases | GGUF IQ2_XXS blocks (byte reorder only) |
| `nvfp4/moe/sm110` | `Nvfp4MoePlugin` | ModelOpt NVFP4, 64-row-interleaved FC1 |
| `nvfp4/moe/sm120` | `NvFP4MoEPluginGeforce` | ModelOpt NVFP4, concatenated FC1 |

INT4 groupwise GEMM V2 is the default. V1 is built only when the user
explicitly selects `--int4-gemm-plugin-version 1`. The two layouts are not
interchangeable. V2 requires a positive output width and a 64-aligned input
width; its final output tile is predicated. Qwen3-MoE fuses GPTQ Q/K/V into one
operation when each source projection is also 128-aligned, so concatenating
their independently transformed fragment blocks is layout-preserving.

NVFP4 MoE has architecture-specific adapters over kernels in
`nvfp4/moe/common`. SM110 uses the 64-row-interleaved FC1 contract consumed by
`Nvfp4MoePlugin`; SM120 uses the concatenated FC1 contract consumed by
`NvFP4MoEPluginGeforce`. These outputs are not interchangeable. Dense NVFP4
Q/DQ weights remain TensorRT constants and do not use this runtime transform
tree.

The top-level `common` directory contains only checkpoint-source pointer
batching shared by multiple precisions. `int4/groupwiseGemm/common` contains
source-format accessors and Q/K/V scale concatenation shared by V1 and V2; it
does not define a weight destination layout. Model and plugin selection
remains in the builder/runtime binding contract; transform kernels do not
infer a destination layout.
