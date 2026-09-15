# P4: continuous native scheduler

`ContinuousScheduler` owns one worker and a `SchedulerBackend`. The TensorRT
adapter `GreedySchedulerBackend` owns one exclusive `SequenceStepRuntime` lease
on an existing parent runtime and explicit stream. Construct it before legacy
inference; keep the parent and stream alive through scheduler destruction.
Do not invoke any parent API during this lease. `close()` joins the worker;
the lease remains held until the scheduler is destroyed.

## Submission and ownership

`submit(preparedTokenIds, maxOutputTokens)` returns a ticket immediately after
bounded CPU admission. The scheduler copies the once-formatted/tokenized prompt
only after checking the queue bounds. Defaults are eight queued requests and
256 KiB of queued token payload, in addition to at most two resident requests.
The count cap also bounds metadata; each prompt is at most 6144 tokens and
prompt plus output at most 8192. Results contain at most the requested output
cap. Consumers own completed futures; retaining arbitrary completed tickets is
caller-owned memory, not an internal result backlog.

FIFO admission, step execution and release belong exclusively to the worker.
Each round executes one decode for every ready row, followed by at most one
qualified prefill chunk. Prefills alternate when both slots need prompt work,
so each gets one chunk per two rounds. Every chunk uses `beginPrefillChunk`
from the very first step; the P3 cap and short-tail restrictions are unchanged.
There is no mixed prefill/decode engine invocation and no execution overlap.

Safe boundaries before/after forwards reclaim finished or cancelled slots and
admit queued work immediately. Cancelled queue heads are skipped in a bounded
scan. Physical rows never move; two-row decode handles are sorted into physical
order. Idle workers sleep on a condition variable, woken by submit or close.

`ticket.cancel()` sets that submission's atomic flag, so a stale ticket cannot
cancel a reused slot. Active cancellation is observed at safe boundaries;
queued cancellation is settled on admission or shutdown, and does not remove a
queued item immediately while both slots remain occupied. This is P4 basic
plumbing, not P5's complete cancellation/deadline contract. `close()` rejects
new work, cancels outstanding requests and joins the worker. Concurrent calls
to close are serialized. A completion racing cancellation may finish normally.

Any worker exception fails all outstanding tickets and rejects further
submission. The TensorRT adapter also poisons the parent lease, including errors
in D2H/sampling after forward completion. Recovery requires destroying and
recreating the parent runtime. This deliberately conservative behavior is not
fine-grained P5 fault recovery. Never call close or destroy the scheduler from
its worker/observer callback.

## Deliberately restricted generation

The P4 adapter accepts only token IDs and an output limit. It selects finite
FP32 logits greedily and stops at the length cap. It does not honor EOS,
temperature/top-p/top-k, RNG, stop strings, thinking, logprobs or streaming.
There is no public options argument that silently ignores those settings.
The temporary sampler copies logits to one preallocated pinned two-row buffer
and scans on the CPU; efficient GPU sampling and tuning belong to later phases.
No state-sized copies, per-token heap allocation, graphs or second model are
introduced by the scheduler. Diagnostic observers may allocate; production
callers should omit them. The backend validToken method must read immutable
metadata safely from submission threads; all other backend methods are owned
by the worker after construction.

## Verification

`continuous_batching_probe ENGINE_DIR CHECKPOINT_DIR --scheduler` compares
serial greedy reference outputs with automatic staggered A/B/C execution:
A=65 prompt/6 output tokens, B=1025/12, C=129/12. B and C are submitted from the
diagnostic observer after A's first decode. The observer only submits work;
all admission, selection, forwarding and slot release are automatic.
`P4_EVENT` records steady-clock microseconds, request, slot, generation, partner
and prompt cursor (kAdmit=0, kPrefill=1, kDecode=2, kRelease=3, kIdle=4).
`P4_SCHEDULER_GATE` requires exact serial token equality, later B admission,
C reuse after A release, continued B prefill after reuse and paired B/C decode.
These are mechanism/greedy-output checks, not broad numerical or performance
qualification. P3's state/logit matrix remains the chunk-policy evidence.

The CPU tests in `unittests/cpp/runtime/continuousSchedulerTest.cpp` exercise
queue count/byte bounds, token limits, FIFO reuse, two-prefill fairness,
cancellation, failure settlement, shutdown, idle sleep and concurrent
producers with repeated reuse. See the OpenClaw P4 results for exact runs,
failed fixture coverage, sanitizer checks, source identity and restoration.
HTTP still admits one sequence; independent request policy is P5, HTTP/SSE is
P6, performance/graphs are P7 and qualified deployment is P8.
