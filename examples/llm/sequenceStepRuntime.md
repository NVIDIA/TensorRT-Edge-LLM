# P2: persistent sequence ownership and forward-only execution

`runtime/state/sequenceSlots.{h,cpp}` holds request-local state without CUDA.
`runtime/sequenceStepRuntime.{h,cpp}` binds that state to one existing rank runtime.
This is an internal execution mechanism, not an HTTP scheduler.

## Lifecycle

Construct `SequenceStepRuntime` before legacy inference reshapes state, and keep
the parent `LLMRankRuntime` and the explicit CUDA stream alive until the session
is destroyed. One externally serialized owner uses the session. It leases the
legacy overlap gate for its entire lifetime; a second session or legacy
`handleRequest` cannot enter. Other parent APIs must not be invoked under the
lease. A successful teardown drains the stream and releases the gate. Any engine
or CUDA failure poisons the session and keeps the parent gate closed: recreate
the parent runtime before continuing. Full injected-fault coverage belongs to P5.

1. `acquire(id, preparedTokens, options)` assigns a free physical slot and resets
   only its recurrent/convolution rows. Tokenization/templating is the caller's
   responsibility. Admission validates prompt plus requested output capacity per
   sequence; it does not clamp all rows to a shared shortest headroom.
2. `beginPrefill(handle, count)` enqueues a valid prompt span. `beginDecode` takes
   one slot or two distinct slots in physical order, each with exactly one
   pending sampled output token. Both return borrowed GPU logits without sampling.
3. `completeStep()` waits on the forward-completion event and then publishes
   committed endpoints. Another forward, release or acceptance is prohibited
   while a step is pending. The event covers forward execution, not any additional
   sampling/D2H operations a caller subsequently enqueues on the stream.
4. `acceptToken` records an externally selected token and its random-draw count.
   A sampled token is not committed to KV until the next decode forward. Prompt
   chunks never create output tokens. The configured output cap ends the logical
   sequence; EOS, stop strings and other generation policy are left to P5.
5. `finish` ends logical work early; `release` invalidates the handle. Another
   request may reuse that slot while its partner stays live. Handles include
   pool ownership and allocation generation, preventing stale or foreign use.

State references are borrowed owner-thread views, not objects to publish to
other threads. Logits remain valid only until the next forward; consume/sample
them before reusing the shared output buffer.

## Physical storage and supported scope

The three execution views are `{0}`, `{1}`, and `{0,1}`. They are prebuilt and
bind aliased recurrent/convolution rows, selected page-table and text-RoPE rows,
and the unchanged attention pools. Logical endpoints are slot-owned; one
preallocated eight-byte device array stages selected start indices. Existing
embedding and pipeline buffers are reused. There are no serving-time state
snapshots or cache compaction copies. The extra pinned metadata payload is
24 bytes, excluding allocator granularity and event/view bookkeeping.

The current guard accepts only single-rank, two-slot vanilla text-only
`qwen3_5_text`, without context reuse, LoRA, speculative decode, context-dependent
or dual RoPE, PLE or deepstack. It intentionally does not advertise support for
other engines, graphs, multimodal inputs or tensor parallelism.

A resumed physical sequence length of one selects the attention decode kernel;
the forward adapter therefore uses absolute context lengths and the decode
profile while retaining **logical prompt** accounting. This exactly preserves
the P1 execution recipe without invoking its sampler. A new fixture exposed
full-versus-chunked numerical divergence in that existing recipe. P2's exact
execution-preservation gate must not be confused with passing that separate
numerical/quality gate. An experimental padded two-position tail also missed
the unchanged threshold and was not adopted.

## Verification

`unittests/cpp/runtime/state/sequenceSlotsTest.cpp` is picked up by the existing
runtime-state unit-test target. It can also be compiled with the state source
and bundled GoogleTest without CUDA. Nine CPU tests passed normally and with
AddressSanitizer/UndefinedBehaviorSanitizer, including 100 reuse cycles.

The optional `continuous_batching_probe ENGINE_DIR CHECKPOINT_DIR --steps`
tests this API against independent P1 and legacy execution. The OpenClaw
`run-continuous-batching-p2.sh` operator wrapper preserves the production runtime,
builds the added sources beside the pinned archive, bounds inference and restores
the model/watchdog. P1's earlier standalone source/binary remain preserved in
their separate evidence directory.

The `P2_STATE_GATE` line reports mechanism correctness. The retained
`P1_TAIL_BASELINE quality_passed=0` is a **failing** numerical diagnostic, even
when P2 succeeds. P3 must resolve or rigorously qualify it before production
chunking can be considered complete. No threshold was relaxed.

## Bounded prompt continuation

P3 adds `beginPrefillChunk(handle)` for a fixed 128-token cap with numerically
qualified tail partitioning. Use it from the first prompt step; low-level manual
partitions remain diagnostic. See [chunked prefill](chunkedPrefill.md) for the
shape restriction, final-sample accounting and retained failing controls.
