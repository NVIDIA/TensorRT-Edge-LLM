# Continuous batching: P1 native feasibility probe

This optional diagnostic does **not** enable a scheduler or alter HTTP admission.
It uses one vanilla, two-slot hybrid runtime with eager execution and the existing
serialized engine. The only runtime-header hook is a friend declaration; no
production method, data layout or thread-safety contract changes.

Build with `-DBUILD_CONTINUOUS_BATCHING_PROBE=ON`, with `TRT_PACKAGE_DIR` set and
submodules initialized. Run `continuous_batching_probe ENGINE_DIR CHECKPOINT_DIR`
with the normal `LD_LIBRARY_PATH` and `EDGELLM_PLUGIN_PATH`. Do not run it alongside
a resident model on memory-constrained devices. Use an external 300-second total
deadline and preserve/restore service and watchdog state during maintenance.

The operator script in the OpenClaw project can link this source against the
exact pinned `e8b29522938901f6df19ebeedd4b69bc8edbcd97` archive using a header
overlay and the reference CUDA device-link object. That saves a complete rebuild;
it must not be used against another revision or after layout/method changes.

## Checks and numerical policy

- Synthetic token fixtures have lengths 1, 3, 4, 63, 64, 65, 127, 128 and 129.
  Slot 0 executes full prefill; slot 1 uses chunks of at most 64 tokens. A
  teacher-forced decode step follows each path.
- Identical singleton execution shapes must produce exactly identical finite
  logits. Chunked and batch-two comparisons require identical greedy IDs,
  maximum absolute logit error at most 0.1 and relative L2 error at most 0.005.
  These are declared-before-run preliminary FP16 screens, not evidence of task
  quality or sufficient production acceptance. Every measured error is printed;
  failures are retained and investigated, not fixed by loosening tolerances.
- Staggered execution checks A decoding while B prefills, profile switches,
  B's one-token tail, both physical slots independently and a two-row decode.
- Inactive state comparison is byte-for-byte: all recurrent and convolution
  state, materialized attention KV and the physical page-table row. Unwritten
  KV capacity is not semantically live state and is not compared. Snapshots are
  host-only diagnostic allocations, copied through 1 MiB pinned staging; they
  are not proposed serving-time state copies.
- Execution lengths are packed into the existing cache-manager scratch tensor;
  canonical endpoints stay in a slot-indexed host array. Both input and output
  recurrent/convolution bindings alias the selected physical rows. KV pools stay
  fixed while only page-table rows are selected. These test-only bindings do not
  implement the production ownership/lifetime API planned for P2.
- A resumed one-token prompt chunk uses the decode profile and absolute
  `context_lengths = committed + 1`. The attention plugin chooses vanilla decode
  from nonempty start indices and sequence length one, even under the prefill
  profile. Supplying chunk length one instead corrupts addressing. The probe
  discards the decoder's sample and advances prompt accounting by one; P3 must
  expose a sampling-free forward boundary with this same execution contract.

Exit 0 means these fixtures passed. Any error, nonfinite output, state mutation,
greedy mismatch or threshold failure is nonzero. Full active-state numerical
comparisons, larger prompts, diverse natural-language fixtures, long-term memory
stability, graph execution and server concurrency still require later coverage.
