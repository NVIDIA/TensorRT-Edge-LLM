# `nvfp4_a16_blackwell_moe` — Thor (SM110) grouped W4A16 MoE GEMM

Prefill kernels of `Nvfp4A16BlackwellMoePlugin`: a tcgen05
`kind::f16` mixed-input grouped GEMM for NVFP4 routed-expert weights with
FP16 activations, sharing the SM110 mainloop of `nvfp4_a16_blackwell_gemm`
(weights are the A/M operand, K-major, dequantized straight into TMEM; tokens
are the B/N operand).

## Weight layout (`BLACKWELL_MOE_N128_K64_V1`)

One buffer per projection serves this kernel **and** the NVRTC-compiled CUDA-core
decode kernels in `kernelSrcs/nvfp4A16BlackwellMoe/`:

```
qweight      int8 [E, N_pad/128, K/64, 128, 32]   64 E2M1 codes per row tile, low nibble = even k
block_scales int8 [E, N_pad/128, K/64, 128, 4]    raw E4M3, one per 16 k
global_scale fp32 [E]                             verbatim ModelOpt weight_scale_2
```

`tensorrt_edgellm/checkpoint/repacking.py::nvfp4_a16_blackwell_moe_offsets` is the
executable specification; `repack_nvfp4_a16_blackwell_moe_experts` produces it as a
pure byte permutation of the checkpoint. For Nemotron 3.5 Lightning: FC1
`[E,15,42,128,32]` (I=1856 padded to 1920), FC2 `[E,21,29,128,32]` (K never padded).

## Variants

| Variant | Fusion | Token tile |
|---|---|---|
| `nvfp4_a16_blackwell_moe_fc1_relu2_fp16_tm128_tn{8,16,32,64,128}_tk64` | `relu(alpha*acc)^2`, TMA store to the permuted `[R_pad, N]` intermediate | 8 / 16 / 32 / 64 / 128 |
| `nvfp4_a16_blackwell_moe_fc2_scatter_fp16_tm128_tn{8,16,32,64,128}_tk64` | `alpha * topk_weight * acc`, `red.global.v4.f16x2.add` scatter into `[T, N]` | 8 / 16 / 32 / 64 / 128 |

Only FP16 is baked (the plugin rejects BF16). E, N, K, the padded row count,
the token count, `top_k`, the SM count (`max_active_clusters`) and `enable_pdl`
are runtime arguments. The token tile is also the per-expert padding
granularity of the permuted activation buffer; the runner selects it by token
count (`nvfp4A16BlackwellMoeDispatchPolicy.h`).

Programmatic Dependent Launch: the kernel runs its whole dependency-free
prologue (TMA descriptor prefetch, shared-memory carve-out, pipeline barrier
init, TMEM allocation) first, then issues `griddepcontrol.wait` immediately
followed by `griddepcontrol.launch_dependents`, right before the first read
of `num_valid_tiles` / `tile_group_idx` (layout kernel output) and the first
TMA of the B operand (gather or FC1 output). The early trigger is intended:
the dependent grid is scheduled once every persistent CTA has started, its
own wait orders the data, and its prologue overlaps this kernel's tail.
`enable_pdl != 0` adds the programmatic-stream-serialization attribute to the
launch; the wait/trigger are no-ops without it.

Weight bytes: every 32-byte code row inside a `[128, 32]` tile stores the TMA
SWIZZLE_32B image (rows 4-7 of every 8 swap their 16-byte halves), so the
producer streams each 4 KB row tile as one TMA box of 2 x 2 KB uint64 rows
straight into the K_SW32 shared-memory image. On Thor, 32-byte TMA box rows top
out at about 230 GB/s, while 2 KB rows or 4 KB bulk copies reach 258-270 GB/s.
Block scales stay a 512-byte TMA transfer per tile.

Grouping: `tile_group_idx[n_tile]` selects the expert as the L coordinate of the
weight **and** block-scale TMA descriptors (one base pointer, no tensormap
updates). Every scheduler-owning warp skips token tiles `>= num_valid_tiles[0]`,
a device value written by the layout builder, so the host launches a
CUDA-graph-stable conservative grid. `tile_group_idx` is staged into shared
memory once per CTA (at most `MAX_TOKEN_TILES` = 1024 tiles per launch, enforced
by the runner) so the TMA producer never waits on a global load between tiles.

Occupancy: one persistent CTA per SM (384 threads x 168 registers fill the
register file, and the pipeline takes the whole SMEM/TMEM budget). The
`ctas_per_sm=2` knob of `Nvfp4A16BlackwellMoeGemmLaunch` halves the SMEM/TMEM
budgets, but on Thor ncu still reported an occupancy limit of 1 block
(registers and the 1 KB SMEM reserve), so the doubled persistent grid only
queued CTAs and ran 2-13% slower; a real two-CTA variant would also need
smaller `setmaxnreg` budgets per warp role. Left as an experiment knob. The
host passes the SM count as `max_active_clusters`; the wrapper scales the
persistent grid.

Token tiles: tn8/16/32/64/128. The tile is also the per-expert padding
granularity, so the runner picks small tiles when experts hold few rows
(tn8 up to 16 tokens, tn16 up to 32, tn32 up to 256, tn64 up to 2048, tn128
above). Extra N tiles of a hot expert re-read its weights through L2 (the
tiles are adjacent in the persistent schedule), so DRAM bytes stay one pass
per expert, but the re-reads cost L2 request bandwidth, which is why the
thresholds are the best worst case over uniform and skewed routing.

## Standalone oracle / micro-benchmark (on the board)

```bash
python kernelSrcs/nvfp4_a16_blackwell_moe/moe_gemm_oracle.py --tokens 128 --token_tile 32
python kernelSrcs/nvfp4_a16_blackwell_moe/moe_gemm_oracle.py --tokens 2048 --token_tile 64 --bench --iters 30
python kernelSrcs/nvfp4_a16_blackwell_moe/moe_gemm_oracle.py --tokens 128 --token_tile 32 --pdl 1
```

Needs `numpy`, `cupy` and `nvidia-cutlass-dsl` (no torch). The oracle builds the
routing, the permuted tile-padded activations and an fp32 NumPy reference from the
dequantized weights and checks both fusions (`ORACLE PASS`).

## Export

```bash
python kernelSrcs/build_cutedsl.py --kernels nvfp4_a16_blackwell_moe --gpu_arch sm_110 --arch aarch64
```

`cmake/CuteDsl.cmake` fails the configure when any of the ten variants is
missing from the artifact; the runner is compiled only when
`CUTE_DSL_NVFP4_A16_BLACKWELL_MOE_ENABLED` is set.
