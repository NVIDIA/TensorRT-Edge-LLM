# Third-party notices

The sparse-attention source in `sparse_attn/` is copied from
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/8e0db4fa562d727ea28b8d63015c196db7d97cae/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn),
branch `sol-engine`, commit
`8e0db4fa562d727ea28b8d63015c196db7d97cae`. The H3 source was introduced in
`2936c47637380842aaa4a4488fac5006cc542b70`; its release snapshot records
`cee25847afdd34bc656abcca126262200b088dc8` as the upstream integration contract.
The Sana project documents this code as Apache-2.0.
The provenance record keeps each upstream path, revision, and SHA-256 alongside
the repository-relative path and SHA-256 of its checked-in counterpart. A
relationship field distinguishes verbatim copies, modified copies, extracted
derivatives, and the CUDA preprocessing port. Copied identifiers are
renamed to sparse-attention terminology.
The SM121 warp-MMA/TMA closure is additionally copied from commit
`ca26dbd2b7034cc90a64c093d715d99b0bfa5b7f`; its upstream and vendored paths
and hashes are recorded in `UPSTREAM_SNAPSHOT.json`.

The files under `sparse_attn/_vendor/flash_attn/cute/`, the extracted
`sparse_attn/common/alignment.py` helper, and portions of the SM100 design derive
from FlashAttention. The BSD-3-Clause license copied from the upstream
snapshot is at `sparse_attn/sm100/LICENSE.flash-attention`; original copyright
notices are retained in the copied files.

The exported kernel depends on NVIDIA CUTLASS / CuTe DSL and cuda-python.
Those build dependencies are not redistributed here and remain subject to
their respective licenses.

Edge-LLM changes include dynamic-sequence export entry points for the
SM100-family and SM120-family Blackwell backends, runtime conditioning-prefix
boundaries for exact query routing, identifier normalization, removal of unused selector helpers and the unused
log-sum-exp output path, a small AOT-only alignment helper and corresponding
`sm100/mainloop.py` import, and a CUDA port of the fixed diagonal-routing
preprocessing formulas. The
CUTLASS DSL 4.7-compatible FlashAttention utility is copied from the generic
upstream sparse backend at the recorded Sana revisions. The PyTorch/Triton
interface and unrelated architecture backends are not copied. Upstream and
vendored paths and hashes are recorded in `UPSTREAM_SNAPSHOT.json`; the
extracted alignment helper and CUDA preprocessing port are marked as
derivatives rather than byte-for-byte copies.
