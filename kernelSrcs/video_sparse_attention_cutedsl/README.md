# Video sparse-attention kernel source

AOT source of the BF16 block-sparse attention kernel behind `SparseAttentionPlugin`:
one packed sequence with 56 heads of 128 channels and 64-row routing blocks, a runtime
sequence length, and a runtime dense prefix. SM100, SM101, SM103, and SM110 use the
tcgen05 path; SM120 and SM121 use the warp-MMA/TMA path. It is built with the other
CuTe DSL kernels by `kernelSrcs/build_cutedsl.py` and needs CUDA 13.

`sparse_attn/` is a minimal copy of the forward kernels from the NVlabs/Sana revisions
recorded in `UPSTREAM_SNAPSHOT.json`. The routing preprocessing that feeds the kernel is
the CUDA source in `cpp/kernels/sparseAttention`. See `UPSTREAM_SNAPSHOT.json`,
`THIRD_PARTY_NOTICES.md`, and `sparse_attn/sm100/LICENSE.flash-attention` for
provenance, modifications, and licenses.
