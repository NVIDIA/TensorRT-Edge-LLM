# In-Flight Batching (V0)

## Scope

In-flight batching (IFB) lets the `RequestEngine` (`cpp/scheduler/`) admit a queued
request into a batch that is already generating, instead of waiting for that batch to
drain. It is opt-in (`--enable-in-flight-batching` on the experimental server) and
early-stage: this page states exactly what the V0 control plane serves, what it refuses,
and where the rest of the runtime's feature set runs in the meantime. Remaining gaps are
listed below in implementation order.

The engine ships **one** control plane. Every request it accepts runs through the stepped
loop: the actor owns the loop and drives the runtime's typed stepper one `prefill` or
`decode` tick at a time (`RuntimeStepper`, `SteppedExecution` in `cpp/runtime/`). A request
the stepped plane cannot serve is refused at `submit()` with `std::invalid_argument` and a
message naming the alternative -- the same permanent-refusal semantics as per-request
LoRA. There is deliberately no fallback plane: a workaround that keeps a feature "working"
removes the pressure that gets its real implementation scheduled.

## Support matrix

Legend: **served** -- runs under IFB, may share a batch; **founder-only** -- runs under
IFB but only as the founder of its own batch (the head waits, it is never rejected);
**blocking path** -- the experimental server keeps the deployment on
`LLMInferenceRuntime::handleRequest`, IFB is never engaged, and starting such a deployment
*with* `--enable-in-flight-batching` fails with a message naming the reason (one gate,
`ifb_unsupported_reason`, decided once at load time -- never a silent fallback);
**text-only** -- the deployment starts under IFB and serves text; its speech endpoints
refuse each request with a message naming the flag;
**rejected** -- `submit()` throws `std::invalid_argument`.

### Deployment level

| Deployment | V0 | Notes / target |
|---|---|---|
| Text LLM, single rank | served | |
| Text LLM, tensor parallelism (thread- or MPI-launched) | blocking path | The stepped plane runs on one rank in this release; `RequestEngine` refuses a multi-rank runtime at construction and the server keeps the deployment on `handleRequest`. Supporting this requires broadcasting stepped commands to every rank. |
| Vision / audio **input** (multimodal) | served | Encoders run at admission; see the visual-token-pruner row below. |
| Speculative decoding (draft engine, MTP, EAGLE, DFlash) | blocking path | The draft-side per-slot state has no seating swap yet; the engine refuses `maxBatchSize > 1` for these deployments. |
| Diffusion backbone | blocking path | Cannot reseat mid-request; same construction-time refusal. |
| Hybrid Mamba (Nemotron-H) | served, **unverified** | Since the ragged runtime landed, joiners prefill in place and no per-slot state is relocated, so the engine admits hybrid deployments. Greedy outputs under batching are not yet bit-exact with single-request runs on GB100; verify before advertising. |
| Speech **output**, Qwen3-Omni bundle | **text-only** | The Thinker is a text runtime the engine serves, so the deployment starts under IFB with the Talker/code2wav engines unloaded; `/v1/audio/speech` and audio output on chat are refused per request with a message naming the flag, and the capabilities report no speech. The Talker pipeline calls the runtime beyond `handleRequest`, which the actor does not mediate. |
| Speech **output**, standalone Qwen3-TTS model | blocking path | No request engine exists for a TTS-only runtime; the flag is refused at startup. |
| Context cache (prefix reuse) | served | Joiners lease their pages at admission; cache visibility commits with the step that produced it. |

### Request level (what `submit()` does with a request)

| Request option | V0 | Notes / target |
|---|---|---|
| Plain text generation, streaming, logprobs, stop strings | served | |
| Sampling parameters that differ from the resident batch | founder-only | Sampling is batch-wide; an incompatible head waits for its own batch (`BatchCompatibility`). |
| Guided decoding (grammar / JSON) | founder-only | The `GuidedDecoder`'s matchers are numbered by slot and the seating swaps do not cover them. |
| Media under an active visual-token pruner | founder-only | Pruning runs only in the founding prefill; a seated prefill would silently skip it. |
| `generateAudio` (speech output) | **rejected** | Response state is assembled only by the drain-time path. Use the blocking path. |
| `saveSystemPromptKVCache` (legacy capture) | **rejected** | The capture runs only in the founding prefill path. Warm the prompt up through the blocking path before handing the runtime to the engine, or use the context cache, which covers the shared-prefix use case under IFB. |
| `pastTrajectory` (trajectory execution) | **rejected** | Trajectory execution founds its own batch and carries drain-time response state. Use the blocking path. |
| Per-request LoRA (`loraWeightsName`) | **rejected** | Adapter weights are engine-wide today. |
| More than one sequence per request | **rejected** | The engine batches across requests, not within one. |

## Where the boundaries are enforced

- Construction (`RequestEngine::RequestEngine`): the runtime must run the stepped plane
  (`supportsSteppedExecution`), `maxBatchSize > 1` requires seated admission
  (`supportsSeatedAdmission`), and `maxBatchSize` may not exceed the engine's built batch
  dimension (`maxBatchSize()`).
- Submission (`RequestEngine::submit`): the **rejected** rows above.
- Admission (`RequestEngine::headMayJoin`): the **founder-only** rows above. These wait
  rather than reject, so FCFS order is preserved and the head founds the next batch.
- Runtime (`GenerationSession::buildAdmissionIntent` / `reserveAdmission`): a joiner the
  batch cannot take is refused as transient (`kNoCapacity`) when nothing was modified, or
  rejected with a reason when the intent itself is invalid.

## Follow-ups

The remaining work is ordered by runtime dependency on the stepped plane:

1. Speech output: assemble the Talker/TTS response inside the stepped `finish()` so
   `generateAudio` requests can found a batch.
2. Trajectory execution as stepped founder state.
3. Legacy system-prompt capture retires into pre-engine warmup and the context cache.
4. Speculative decoding under IFB: draft-side per-slot state joins the seating protocol.
5. Guided decoding joiners: matchers keyed by resident handle.
6. Per-sequence sampling parameters, so requests with different temperature / top-p /
   top-k / max-length share a step (what Stage 2 batching needs to pay off).
7. Persistent runtime session: residents and KV outlive requests, the scheduler emits
   `ScheduledStep` directly, and the founder model retires after the first scheduling stage.
8. Cancellation as a request-owned token rather than a `StreamChannel` flag, polled at
   chunk boundaries too.
9. Tensor parallelism under IFB: broadcast each stepped command to thread-launched
   ranks, then carry the same stream over MPI.
10. Retire the runtime's `GenerationBoundaryHook`, which nothing in the engine calls
    anymore.
