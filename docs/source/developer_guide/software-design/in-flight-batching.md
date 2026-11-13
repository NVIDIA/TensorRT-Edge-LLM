# In-Flight Batching

## Scope

In-flight batching (IFB) lets the `RequestEngine` (`cpp/scheduler/`) admit a queued
request into a session that is already generating, instead of waiting for the running
requests to finish. It is opt-in (`--enable-in-flight-batching` on the server) and early-stage: this page states exactly what the control plane serves, what it
refuses, and where the rest of the runtime's feature set runs in the meantime. Remaining
gaps are listed below in implementation order.

The engine ships **one** control plane, over **one** runtime session that lives as long as
the engine (`RuntimeSession` in `cpp/runtime/`, opened once by
`LLMInferenceRuntime::openSession`). The actor owns the loop: at every step boundary it
folds in its command queue, admits queued requests into the session, then schedules one
step and commits the result. What a step may contain depends on the execution the
deployment supports (see [Scheduling](#scheduling)): under single-cohort execution the
actor admits at most one request per boundary and a step is either the context step of
the request it just admitted or one decode step over every resident; under compact
execution it admits every queue head that can join and a step combines prompt chunks of
several residents with decode rows. A request enters through the same
admission whether the session is empty or full; no request opens the session and no
teardown between requests, so an empty session simply idles until the next submit. A
request the session cannot serve is refused at `submit()` with `std::invalid_argument`
and a message naming the alternative -- the same permanent-refusal semantics as
per-request LoRA. There is deliberately no fallback plane: a workaround that keeps a
feature "working" removes the pressure that gets its real implementation scheduled.

## Scheduling

### Execution and strategy

Phase 3 and phase 8 below are `rt::ExecutionPhase` values, `kAutoregressiveDecode` and
`kMixedPrefillDecode`: the extents of the execution-phase marker that tells the plugins which kind
of step runs (see the phase table in
[TensorRT Plugins](../customization/tensorrt-plugins.md#attentionplugin)).

| Term | Meaning |
|---|---|
| Single-cohort execution | A step is one admission's whole-prompt context step or one decode step over every resident (phase 3). Every deployment supports it. |
| Compact execution | A step is any subset of residents: prompt chunks first, then decode rows, packed without padding (phase 8, `RankRuntimeSession::executeCompact`). A step that decodes every resident still runs as phase 3. |
| `mixed` | Decode first: every decoding resident gets one token per step, and prompt chunks fill the rest of the step budget. Needs compact execution. |
| `prefillFirst` | While any resident has an unfinished prompt the step runs only prompt work and the decodes wait; otherwise every resident decodes. On a compact deployment the waiting prompts of several residents share one phase-8 step; otherwise it is single-cohort execution. |
| `auto` (default) | `mixed` when the deployment supports compact execution and a mixed step fits (`maxNumTokens >= B + P`: an explicit `maxNumTokens` always does, the default does when `roundDown_P(T_max) >= B + P`), else `prefillFirst`. |
| `singleCohort` | Single-cohort execution on any deployment. `RequestEngine` sets `SessionOptions::allowCompactExecution` (default false) for every other mode, so the sessions of a `singleCohort` engine do not resolve the compact capability, reserve no media memory and no dedicated context, and report compact execution as unavailable (`compact execution disabled for this session`). Outputs match a build without compact execution bit for bit, so the IFB trace tools use it as their reference; it also serves for debugging and as a fallback on a deployment where compact steps misbehave. |

Whether a deployment supports compact execution is decided once per runtime, when the first
session that allows compact execution opens (`LLMRankRuntime::resolveCompactCapability`). Compact
execution first requires these static preconditions (`LLMRankRuntime::compactPreconditionFailure`):

- not a speculative (draft engine, MTP, EAGLE, DFlash) or diffusion deployment;
- not a TensorRT-RTX engine;
- the engine's `maxBatchSize` is at least 2;
- no Gemma4 vision-bidirectional attention (image tokens attend bidirectionally within an image,
  which a chunk boundary cannot split);
- the vision and audio encoders' outputs, if any, are FP16 `[rows, hidden size]`;
- the vision encoder, if any, produces as many deepstack feature layers as the decoder engine takes
  (`num_deepstack_features`); a mispaired bundle would otherwise refuse every image request when its
  features are staged;
- no n-gram embedding preprocessor (its row map assumes entry-padded token rows; Gemma4 per-layer
  embeddings are supported);
- under cached Gemma4 per-layer embedding storage (`--ple-residency cached`), the cache maps at
  least `min(PLE table rows, B x S_in)` distinct rows, that is
  `cache_slots >= min(PLE table rows, B x S_in) + 1` (slot 0 is reserved). One compact step
  gathers the rows of all its tokens at once, up to `B x S_in` of them when `prefillFirst` packs
  several waiting prompts, and a step whose distinct rows exceed the cache would fail the session.
  A single-cohort step needs only one prompt's rows or one row per resident. Resident storage
  always qualifies;
- no bounded sliding-window KV storage;
- no layer debugger;
- the engine's prefill and decode profiles both admit execution phase 8 (steps with prompt rows run
  on the prefill profile, decode rows of only some residents on the decode profile).

Then the runtime decides from the engine's `config.json` and the GPU's SM (`compactExecutionGap`,
`cpp/runtime/config/compactCapability.cpp`). It sorts the engine's layers into kinds: attention
(`kv_layer_configs`, `num_attention_heads`, `kv_cache_dtype`, `builder_config.tp_size`), QSA
attention (`indexer_head_dim` greater than 0 makes every attention layer QSA),
`gated_delta_net` or Mamba2 SSD (the recurrent-state fields; only Mamba2 exports
`recurrent_state_num_groups`), `causal_conv1d` (`conv_dim`, `conv_kernel`) and the n-gram conv. It
checks each kind with the predicate that kind's plugin applies when it reserves the phase-8
workspace and when it accepts or refuses a phase-8 enqueue (`cpp/kernels/compactPhase/`):

| Layer kind | Predicate | Phase 8 needs |
|---|---|---|
| Attention | `attentionCompactSupport` | no context mask selector, tree, vision-block or sink; CTM decode at TP 1 on SM100/101/103/110, CTM or XQA decode elsewhere; Blackwell or FMHA-v2 ragged paged prefill |
| QSA attention | none | refused on every SM: `QsaAttentionPlugin` runs entry-padded phases only |
| `gated_delta_net` | `gatedDeltaNetCompactSupport` | the packed-varlen GDN kernel, in an artifact built for the SM |
| Mamba2 SSD | `mamba2SsdCompactSupport` | the packed SSD kernel; on SM120/121 the state-pool kernel |
| `causal_conv1d` | `causalConv1dCompactSupport` | a depthwise layer and the packed-varlen conv kernel at dilation 1, in an artifact built for the SM |
| n-gram conv | `causalConv1dCompactSupport` | the same, at a dilation of the n-gram size |

The predicates are compiled into both the runtime and the plugin library, under the same CUDA
version and CuTe DSL kernel-group macros, and read `EDGELLM_DISABLE_CTM_DECODE` and the device's SM
count in the calling process. Within one build the runtime therefore decides as the plugins do,
including CUDA 11 builds, builds without a CuTe DSL kernel group (`ENABLE_CUTE_DSL=OFF`) and the GQA
ratios XQA and CTM cover. In a build with every kernel group and the kernels for the target SM,
attention runs on SM80 to SM121 for the shapes its kernels cover, `gated_delta_net` (K = V = 128) on
SM80 to SM90 and on SM100, 101, 103, 110, 120 and 121, `causal_conv1d` (width 2 to 8) on SM80 to
SM90 and on SM100, 101, 103, 110, 120 and 121, Mamba2 SSD ((64, 64), (64, 128), (80, 128)) on
SM100, 101, 103 and 110 and (head dim 64, 80 or 128, state size 64 or 128) on SM120 and 121, and the
n-gram conv and context-mask-selector attention nowhere. A CuTe DSL artifact carries the kernels of
the SMs it was built for, its `metadata.json` `gpu_arch` list. The kernel-group macros
(`CUTE_DSL_CAUSAL_CONV1D_ENABLED` and the GDN group) are defined per artifact, not per SM, so
`cmake/CuteDsl.cmake` also passes that list (`CUTE_DSL_ARTIFACT_SMS`) and the `causal_conv1d` and
`gated_delta_net` predicates accept only an SM the artifact carries (`cuteDslArtifactCarriesSm`, an
exact match, as the multi-SM dispatcher of a combined artifact dispatches). An SM100 build run on an
SM86 or SM120 GPU therefore refuses those kinds, with `the linked CuTe DSL artifact has no kernels
for this SM`, instead of aborting the process at the first phase-8 enqueue (`failUnsupportedSm` in
the multi-SM dispatcher that `kernelSrcs/build_cutedsl.py` generates).
A single-SM artifact of an earlier minor revision of the same major, whose kernels the newer SM
could load, is refused too. An unknown SM is refused.
The reason names every blocking kind once, with its shape, why and the SMs that support that shape
in this build, for example `no compact-execution kernels on SM87 for Mamba2 SSD layers
(head dim 64, state size 128: no packed SSD prefill kernel for this SM, head dim and state size in
this build; supported on SM100/SM101/SM103/SM110)`. No forward runs to decide.
`CompactCapabilityTest.EachKindFollowsItsPluginPredicateOnEverySm` checks that every kind is checked
with its predicate on every SM.

What the engine config does not record is taken from what the export paths produce: no attention
layer is a tree, vision-block or attention-sink layer (the preconditions above refuse speculative
and vision-block deployments), and a KV-shared bounded-SWA-capable layer receives the donor's current
K/V.

The predicates say what this build's plugins can run, not what an engine reserved: a plugin sizes
its phase-8 workspace in `getWorkspaceSize()` when the engine is built, and TensorRT serializes that
size with the engine. Both builders (`LLMBuilder::build` and the checkpoint-direct
`build_with_compact_report`) therefore open a compact build report of the plugin library around the
TensorRT build (`edgellmBeginCompactBuildReport` / `edgellmEndCompactBuildReport` in
`cpp/plugins/utils/pluginUtils.h`). Each attention, `gated_delta_net`, `causal_conv1d` and
`update_ssm_state` layer reports from `getWorkspaceSize()` whether it reserved the compact workspace;
a layer counts as reserved only if every query of it did. When at least one such layer was queried,
every one reserved it and no other build overlapped, the builder writes
`builder_config.compact_execution: true`; otherwise it omits the field, and the library logs why
as a WARNING, naming (by kind and layer name) the layers that did not reserve it. Only the LLM
engine's build opens a report; visual, audio and other component engines run no compact steps.
Under tensor parallelism only rank 0 writes `config.json`, so its build's report decides, and every
rank builds the same plugin layers with the same per-rank shapes. After the predicates pass, the
runtime requires that field to be `true`; any other engine, for example one built before the field
existed or one with a layer that has no compact path in the build that built it, runs without
compact steps with the reason `the engine was not built for compact execution`, which points to
that builder log. A change that enlarges a plugin's compact workspace must, in the
same change, make engines built before it stop running compact steps, for example by renaming this
`builder_config` field. Each plugin keeps its enqueue-time check, described next.

A plugin that refuses a phase-8 enqueue logs an ERROR that identifies the layer by kind, name and
shape (`AttentionPlugin '<layer>' (16 query heads, 8 KV heads, head dim 128): compact execution is not
supported for this layer on this GPU.`) and returns before it launches any work. TensorRT passes an
empty name when it deserializes an engine, so the plugins serialize the build-time layer name (for
an ONNX-path engine, the ONNX node name, which TensorRT's own enqueue error also prints).
`LLMRankRuntime::runCompactForward` tells this refusal apart from the other failures of a
compact forward: a failed `prepare()` (input shapes, bindings) and an enqueue that leaves the stream
in error or a CUDA error pending fail the step and leave the capability as it is. A refused step
fails too (`StepResult::ok` false), with `StepResult::compactRefused` set and `StepResult::failure`
stating that compact execution is now disabled and that the request can be retried. The session's
state is no longer trusted, because the layers before the refusing one already advanced recurrent
and convolution state in place. The runtime then records compact execution as unavailable for its life
(`a plugin refused compact execution at runtime (see the plugin's ERROR log, which
identifies the layer by kind and shape)`), so every session opened afterwards reports `compact`
false and runs single-cohort steps. The runtime never falls back within a step, and the caller
closes the failed session, as for any step that is not `ok`. The runtime cannot see why a plugin
returned failure, so any enqueue failure that leaves no device error counts as a refusal: a shape
the predicate refuses, a kernel module that fails to load, or an error the plugin catches. Outside
an explicit `mixed`, such a failure therefore disables compact execution for the runtime's life and
the engine keeps serving; it shows as `compact_refusals`, the plugin's ERROR log and
`compact_unavailable_reason` in `/health`, not as a stopped engine. How `RequestEngine` continues is
described under [Failure handling](#failure-handling).

`RuntimeSession::capabilities()` carries the result (`compact`, `reason`) and the engine's B, S_in
and P.

`RequestEngine` resolves the configured strategy against these capabilities on its actor
thread (`resolveSchedulingPolicy`) when its first session opens, and again against the reopened
session after a refused compact step, and logs each outcome:

```text
RequestEngine scheduling: mode=mixed (auto: compact execution supported), execution=compact, schedulingPolicy=fcfs,
preemption=none, maxNumTokens=640, maxChunkTokens=512, priorityReservedSlots=0, maxPauseSteps=0; engine B=4,
S_in=1024, P=128; compact execution available.
```

`ResolvedSchedulingPolicy::modeSource` (nanobind `mode_source`) records how the mode was chosen:
`kRequested` keeps the configured mode, `kAuto` is `auto`'s choice, and only then `reason` says why;
the line prints `requested` or `auto: <reason>`. `resolveSchedulingPolicy` runs as named steps:
`validateSchedulingOptions` refuses an option that could never apply, `computeStepBudgets` range-checks the
budgets and derives the defaults, `resolveMode` picks the mode, and `collectWarnings` lists what the
options asked for that the resolved policy does not apply as given.
`execution` is `compact` when the session runs compact steps and `single-cohort` otherwise
(`executionName`, nanobind `ResolvedSchedulingPolicy.execution`): `prefillFirst` names a step
composition, not a backend, and runs either way. After a refused compact step the line is logged as
a WARNING that starts `RequestEngine scheduling changed after a plugin refused a compact step:`.
`RequestEngine::resolvedPolicy()` (nanobind `resolved_policy`) returns the policy in effect by
value, because that fallback can replace it once.

Construction fails with `std::invalid_argument` (Python `ValueError`, server startup error
`invalid server configuration: ...` with the options spelled as command-line flags) when `mixed`
is requested on a deployment without compact execution (the message gives the reason, which
names each blocking layer kind and the SMs that support it, and suggests `prefillFirst`), when `preemption=pause` is set
without `schedulingPolicy=priority` or with an explicit `prefillFirst` or `singleCohort`, when
`priorityReservedSlots` is set without `schedulingPolicy=priority`, when `maxPauseSteps` is set without
`preemption=pause`, or when a budget below
violates its constraint. Any other failure while the first session is set up is a
`std::runtime_error`, not a configuration error. What the engine accepts but does not apply as given is reported in
`ResolvedSchedulingPolicy::warnings` (nanobind `warnings`), each logged once at startup as a
`RequestEngine configuration:` warning:

- `preemption=pause` under `auto` that resolved to `prefillFirst`: pause is dropped
  (`preemption=none`, `maxPauseSteps=0`), since a pending prompt already runs while decoding
  residents wait. The same
  flags therefore start on deployments with and without compact execution.
- An explicit `maxNumTokens` or `maxChunkTokens` when the mode resolved to `prefillFirst` or
  `singleCohort`, which do not use them. The range checks still apply.
- An explicit `maxChunkTokens` above S_in under `mixed`: one prompt chunk holds at most S_in
  tokens, so the budget resolves to `max(P, roundDown_P(S_in))`, the value the startup line shows.
- `schedulingPolicy=priority` with `priorityReservedSlots=0` and `maxBatchSize >= 2`: an urgent
  request waits for a resident to finish while every slot is busy. The engine does not reserve a
  slot on its own; the recipe is an engine built for `B + k` and
  `maxBatchSize = B + k, priorityReservedSlots = k`.

### Configuration

The same options exist on `SchedulingOptions` (`EngineConfig::scheduling`), the nanobind
`RequestEngine` and the server (the server flags require `--enable-in-flight-batching`). B is the engine's built
batch size, S_in its maximum input length, T_max = B x S_in, P the KV page size (128).

| `SchedulingOptions` / server flag | Default | Constraint and meaning |
|---|---|---|
| `ifbSchedulingMode` / `--ifb-scheduling-mode` | `auto` | `auto`, `mixed`, `prefillFirst` or `singleCohort`, as above. |
| `maxNumTokens` / `--max-num-tokens` | `min(roundUp_P(512 + B), roundDown_P(T_max))` | In [B + P, T_max]. Token budget of a `mixed` step that has decode rows. |
| `maxChunkTokens` / `--max-chunk-tokens` | `min(512, roundDown_P(maxNumTokens - B), max(P, roundDown_P(S_in)))` | A multiple of P in [P, maxNumTokens - B], then capped at `max(P, roundDown_P(S_in))` (with a warning for an explicit value). Per-request prompt tokens in a `mixed` step that has decode rows. |
| `schedulingPolicy` / `--scheduling-policy` | `fcfs` | `fcfs` or `priority`. Under `priority` the queue and the step budget are ordered by `(priority, arrival)`; a lower value is more urgent (as in vLLM and the OpenAI-compatible API). Under `fcfs` every request counts as priority 0, and the engine logs one warning, for the first request with a non-zero priority. |
| `preemption` / `--preemption` | `none` | `none` or `pause`; `pause` requires `priority` (`fcfs` ignores request priorities, so nothing would pause) and acts under `mixed`: it is refused with an explicit `prefillFirst` or `singleCohort`, and dropped with a warning when `auto` resolves to `prefillFirst`. |
| `priorityReservedSlots` / `--priority-reserved-slots` | 0 | In [0, maxBatchSize - 1] and requires `priority`. A request with priority >= 0 is admitted only while more than this many slots are free. 0 under `priority` with `maxBatchSize >= 2` is reported as a warning. |
| `maxPauseSteps` / `--max-pause-steps` | 0 | Non-negative; a positive value requires `pause`. A resident paused for this many steps in a row is not paused in the next step, which guarantees a decode row only to a resident that decodes: one with an unfinished prompt competes for the prompt budget in priority order and may get no chunk. 0 sets no limit. |

The server checks the options that need no engine fact before the engine is built
(`validate_ifb_scheduling_options`, also run by `LLM.__init__` before `prepare_model`): scheduling
flags without `--enable-in-flight-batching` and the option types, then the C++
`validateSchedulingOptions` through the binding `validate_scheduling_options`, for the batch size
the engine is built with. That binding parses the names with the scheduler's own name tables and
refuses with the scheduler's own messages, so the server keeps no copy of either. The
step budgets depend on the engine's page size and are checked once, by the engine when it starts;
the CLI reports its refusal as `invalid server configuration: ...` with the flag names. Under
`--scheduling-policy priority` its HTTP admission gate
(`_IFBAdmissionGate`) lets a request with priority < 0 hold up to `--max-batch-size` leases past
the bound at which other requests get 429, so an urgent request reaches the engine's queue under
overload. `--min-request-priority` (`ApiConfig.min_request_priority`, default 0) is the most
urgent priority the server accepts: `EngineClient._prepare_native_request` answers a request below
it with HTTP 400 (`param: "priority"`) before it takes an admission lease or builds a native
request. The check is a server option, not an `EngineConfig` field, and applies with and without
`--enable-in-flight-batching`. The minimum is at most 0 (`ApiConfig` refuses a positive one), so a
request without a priority, which includes every `/v1/responses` and `/v1/messages` request, is
always accepted. With the default, a client cannot use the urgent headroom above or the reserved
slots; `EngineClient` logs a startup WARNING when `priority` with `--priority-reserved-slots` > 0
meets a non-negative minimum. To start an urgent HTTP request at once while every slot is busy, an
operator sets all of `--scheduling-policy priority`, `--priority-reserved-slots k` on an engine built
for k extra slots, and `--min-request-priority` below 0; `--preemption pause` in addition keeps less
urgent decode rows out of its prompt's steps. With the defaults a negative priority gets HTTP 400,
and with a full batch and no reserved slot an urgent request waits for a running request to
finish: no running request is evicted or swapped out.

The request field is `LLMGenerationRequest::priority` (nanobind `priority`, OpenAI request
`priority`).

### Step planning

`StepPlanner::plan` (`cpp/scheduler/stepPlanner.cpp`) composes each compact step from the
resident table's ledger (prompt length, committed positions, whether the final chunk ran):

1. `mixed`: every decoding resident gets one decode row; with any decode row the step budget
   is `maxNumTokens` and the per-request cap `maxChunkTokens`, otherwise T_max and S_in.
   `prefillFirst`: with any unfinished prompt only prompt chunks run, with budget T_max and
   cap S_in; under `schedulingPolicy=priority` only the unfinished prompts of the best (lowest)
   priority value run, and the others wait for a later prompt-only step.
2. Prompt chunks are handed out in order (arrival, or `(priority, arrival)`):
   `n = min(remaining, cap, S_in, budget)`. A chunk that does not finish its prompt is cut
   back to end on a page boundary; a request left with `n <= 0` is skipped so a shorter one
   later in order may still fit. The planner accepts only the budgets the resolver produces, so
   the first prompt in order always gets a chunk: a plan over residents is never empty and never
   consists of the decode rows of only some residents.

Exclusive residents (a guided request, media under a visual-token pruner, media whose token
layout only its encoder determines) keep single-cohort steps inside a compact session, whose
context step accepts only the whole uncommitted prompt. The planner meets that contract without
knowing about them: an exclusive resident is alone in the session, so a step has no decode row,
its cap is S_in and its budget T_max, and admission already bounds the uncommitted prompt by S_in,
so the whole prompt is one chunk. A planner change that chunks prompts when no request is
decoding (for example to bound step latency) must leave exclusive residents out, or the session
fails the step. The `guided` row of
`RequestEngineCompactTests.AdmissionAtAStepBoundaryFollowsTheModeThePolicyAndTheSlots` pins this.

Admission (`RequestEngine::tryAdmitHead`): the queue head in policy order is admitted when a
slot is free (and the reserved-slot rule allows it), it is compatible with the residents and
not exclusive, and its prompt fits; otherwise it waits and nothing behind it overtakes it. KV is reserved per slot (`B x` the per-sequence capacity),
so an admitted request always runs to completion: there is no eviction or recomputation.
Under `schedulingPolicy=priority`, a head that outranks every resident but waits on batch
compatibility is logged once, with the first field that differs
(`BatchCompatibility::firstDifference`).

Each resident carries its own generate budget: `maxGenerateLength` is not a batch
compatibility field. The session bounds a request's budget by the KV capacity behind its own
prompt at admission (`RankRuntimeSession::generateBudgetFor`), stores it on the slot
(`DecodingInferenceContext::maxGenerateLengths`, moved with the row by eviction and the
seated prefill's row exchange), and every length stop check reads the row's budget
(`DecodingInferenceContext::generateBudget`). A batch built at request start gives every slot
the batch's budget, so the blocking path stops where it always did.

### Chunked prefill

| Aspect | Rule |
|---|---|
| Boundaries | The session accepts a chunk of any positive length up to the remaining prompt and S_in; it does not require page alignment. The step planner ends every non-final chunk on a multiple of P (see Step planning); the final chunk has any length. |
| Attention | The chunk attends to all KV the sequence already has (causal mask aligned bottom-right) and writes its own K/V. |
| Recurrent state (GDN, conv1d, SSD) | Continues from the slot's state and writes the end state back. |
| Output | A non-final chunk only advances the committed length: it is not sampled and emits nothing. The final chunk samples the first token. The ledger marks the prompt complete only at the final chunk. |
| Cancellation | Takes effect at the next step boundary; a request cancelled between chunks is retired without completing its prompt and publishes nothing to the context cache. |
| Numerics | Chunking changes the accumulation segments of attention softmax and the recurrences, and decode rows inside a mixed step run the prefill profile's GEMMs: outputs may differ from single-cohort execution by rounding (greedy flips only at near-ties). On SM120/121 the prompt rows of a compact step run Mamba2 SSD through the packed state-pool kernel, while single-cohort prefill runs the SM80 chunk-scan kernel, so Mamba2 hybrids differ more there: Nemotron-3-Nano-4B FP16 can flip greedy tokens where the reference top-2 logprob gap is up to about 0.16, against up to 0.1 for the other families and GPUs. Neither path is the outlier: measured against B100 single-cohort output, SM120 single-cohort and SM120 compact steps differ by a similar amount. |
| Skip-softmax (BLASST) | Compact prompt chunks of an engine exported with a prefill skip-softmax table run the exact (dense) attention kernel by design: the ragged phase-8 kernels take no skip table. Such a deployment therefore gets no skip-softmax prefill speedup under `mixed` or under `prefillFirst` with compact execution, and its outputs differ from single-cohort execution of the same engine, which applies the table in phases 1/2. Decode rows of a compact step also run dense, while phase-3 decode steps apply the decode table (`skip_softmax_decode_scale_table`) when the engine has one. `singleCohort` keeps skip-softmax prefill and decode. |

### Priority and pause

With `schedulingPolicy=priority` and `preemption=pause`, each step takes h, the first resident in
`(priority, arrival)` order with an unfinished prompt, and leaves out every resident whose priority
value is larger than h's (its KV and recurrent state stay in its slot, untouched). A paused resident
therefore resumes in the first step in which no resident with a smaller priority value has an
unfinished prompt; a second urgent prompt admitted before h's final chunk extends the pause. If no
decode row remains in the step, h's budget widens to its whole prompt (budget T_max, cap S_in), so its
time to first token approaches that of a request running alone. That holds only when no resident of
the same or a smaller priority value is decoding (and none is exempt under `maxPauseSteps`): such
residents keep their decode rows, and h's prompt is chunked beside them under `maxNumTokens` and
`maxChunkTokens`. Pause never
frees KV: requests that need a slot when all are busy use `priorityReservedSlots`. With
`maxPauseSteps` = n > 0, a resident paused n steps in a row (`RequestRecord::consecutivePausedSteps`)
is not paused in the next step, then may be paused again: a decoding resident gets its decode row,
while one with an unfinished prompt gets a chunk only if the budget left after the more urgent
prompts holds one (strict priority order would starve it without pause too). With 0, a steady
stream of more urgent prompts keeps a resident paused. Under
`auto` resolved to `prefillFirst`, pause is dropped with a warning (see above); there, and with
an explicit `prefillFirst`, a resident's decode already waits while a prompt runs, and under the
priority policy a compact prompt-only step carries only the most urgent prompts. Under
single-cohort execution the ordering applies and each admission's prompt runs alone.

An urgent request therefore starts at the next boundary when a slot (free, or reserved for
priority < 0) is available, its batch-wide parameters match the residents' (the fields
`BatchCompatibility` compares, such as the LoRA adapter and thinking mode; sampling parameters,
seed, generate budget and logprobs width are each sequence's own), and neither it nor a resident
is exclusive. Without a slot it waits for a resident to finish; with
differing batch-wide parameters or beside an exclusive resident it waits for every resident to
finish. The ordering is strict, without aging. `tests/test_cases/ifb_priority.json` covers the free-slot and
reserved-slot cases, with the urgent request's budget equal to and different from the
residents'.

### CUDA graphs

| Step | Profile | Execution |
|---|---|---|
| Every resident decodes (phase 3) | decode | A CUDA graph captured by the application (`captureDecodingCUDAGraph`) is replayed when one matches the bindings, exactly as under single-cohort execution. |
| At least one prompt chunk (phase 8) | prefill | Eager. |
| Decode rows of only some residents (phase 8, `Np = 0`) | decode | Eager. The planner never plans one, because a step with a pending prompt or a paused resident always carries a chunk; it runs only when every prompt chunk of the step is cancelled before the step runs while another prompt is still pending or residents are paused. |

Under `mixed`, whenever a prompt is unfinished every step carries a chunk of it
(`maxNumTokens >= B + P`), so a step without a chunk decodes every unpaused resident and
stays on the graph. When the residents whose chunks a step names are cancelled before it runs
and its decode rows cover every resident left, the cancelled residents are evicted first and the
step runs as phase 3 (counted in `decodeSteps`, on the graph when one matches). When paused
residents are left out or another prompt is still pending, the remaining decode rows run as the
`Np = 0` phase-8 step above.

A compact session runs its steps with prompt chunks (phase 8 on the prefill profile) on a second
TensorRT execution context that stays on the prefill profile (`EngineExecutor::dedicateCompactContext`).
TensorRT updates layer resources on the first enqueue after a profile switch: on B100 with
Qwen3-0.6B, the first compact step after a decode step spent 4.1 to 5.9 ms in `enqueueV3`, against
2.1 ms for the next step of the same shape. Every other step keeps the original context: decode
steps and their graphs, `Np = 0` steps on the decode profile, the blocking path, and every step of a
single-cohort session, which creates no second context. Captured graphs are keyed by context,
profile and bindings, and a step looks one up only when its context and profile own a captured
graph. When TensorRT cannot create the second context or select the prompt profile on it
(`EngineExecutor::dedicateCompactContext`), the executor logs a warning and compact execution stays
available: prompt steps run on the original context and pay the profile switch. When the second
context's auxiliary streams cannot be created, the setup throws and compact execution is recorded as
unavailable, as for a failed media reservation (see the next section).

The dedicated context has its own auxiliary streams but shares the executor's device memory with
the original context and with the vision and audio encoders that run at admission. Sharing is sound
only because every enqueue of the session is ordered on the session stream; a change that runs an
encoder or a step on another stream must give it memory of its own.

What the dedicated context adds is TensorRT's per-context state and the private buffers of its plugin
clones (TensorRT clones every plugin per execution context). Per attention layer, a clone holds
`128 x B x Hkv x 8` bytes of CTM decode semaphores when the layer decodes with CTM, and 256 bytes of
XQA counters when it can decode with XQA; recurrent and convolution plugins allocate nothing per
clone. The cost therefore grows linearly with the number of attention layers, B and Hkv: when every
layer decodes with CTM, as on B100, the semaphores of Qwen3-0.6B at B = 4 (28 attention layers,
Hkv = 8) take 0.9 MiB. Measured as the process's peak device memory with and without the dedicated
context, at B = 4 under a workload of prompts arriving while others decode, it added 2 to 6 MiB on B100 (nvidia-smi, 2 MiB granularity)
and 2 to 4 MiB on Thor, for Qwen3-0.6B and Qwen3.5-0.8B. Larger models and B = 8 have not been
measured.

A compact step synchronizes the host with the session stream once in the steady state, after
sampling (`COMPACT_SAMPLE`), to read the sampled tokens. A step in which a sequence finishes
synchronizes once more in its eviction (`LLMRankRuntime::performBatchEvict`, or
`ContextCacheCoordinator` batch compaction under a context cache), as a single-cohort step does. Under
a context cache, a hybrid deployment's step that publishes decode-end recurrent snapshots synchronizes
once more (`completeDecodeStep`). The host-staging and page-table uploads wait on events of the
previous step, which have normally completed by then. Admission work (the deferred media preprocess,
image content digests, encoders) synchronizes between steps, not inside one.

### Context cache and multimodal input

- Context cache: a request's reused prefix is its starting committed length; only the
  uncached part is chunked. Cache pages and hybrid snapshots are published when the final
  chunk commits, never for a prompt that was cancelled or failed before it. The first chunk's
  `prepareSequenceAdmission` restores the sequence's hybrid snapshot or zeroes its recurrent
  state; without a context cache, `reserveAdmission` clears the slot's recurrent and convolution
  state when the sequence is admitted.
- Multimodal input: media prompts run in compact steps like text prompts, and an image may span
  chunks. A media request's vision encoder runs at admission only after its slot (and, under a
  context cache, its page lease) is reserved, and only when the encoder embedding cache does not
  hold all of its images. Audio is split differently, because an audio runner's `preprocess`
  (which lays out the prompt's audio tokens) has no text-only mode: it runs at intent, before the
  reservation. For the Qwen3-Omni / Qwen3-ASR, Gemma4 and Nemotron-Omni audio runners that
  preprocess already runs the audio encoder, so their encoder runs before the reservation, again
  at each boundary that retries a request refused for capacity, and also on an encoder embedding
  cache hit (the hit only replaces the rows it produced). The Gemma4 unified audio runner extracts
  features in `preprocess` and runs its encoder after the reservation. The rows
  of its media positions from the prefill start on, together with the deepstack features, are
  copied into the slot's region of a per-slot reservation (`CompactMediaReservation`,
  `cpp/runtime/state/`), which later admissions cannot overwrite while the request still has
  prompt chunks to run; positions inside a reused prefix are not embedded again. Prompt chunks
  gather their media rows from the reservation by global row index (64-bit offsets in the embedding kernels). M-RoPE positions
  are written once per admission into the slot's resident row, when the media is staged for a
  media prompt and at the first chunk for a text prompt, and are not rewritten by later steps.
  Encoders run inside `RuntimeSession::admit` on the session stream, so the next step waits for
  them.
- Reservation memory: allocated once, when the first session that allows compact execution opens on
  a deployment that supports it and has a vision or audio encoder (`prepareCompactExecution`).
  `RequestEngine` allows compact execution in every mode except `singleCohort`, so `auto` and
  `prefillFirst` allocate it too, whether or not their steps then mix prompts with decode rows. It
  takes FP16 buffers of
  `B x (E_image x (1 + deepstack layers) + E_audio) x hidden size` elements, where
  `E_image = min(S_in, rows of the vision encoder's output buffer)` (0 without a vision
  encoder) and `E_audio = S_in` (0 without an audio encoder, and reserved whenever the deployment
  has one). The runtime logs the layout and size at INFO when it allocates the reservation
  (`Compact multimodal reservation: <B> slots x (<E_image> image rows x (1 + <D> deepstack layers)
  + <E_audio> audio rows) x <H> hidden FP16 = <size> MiB.`). Validated vision-language and ASR
  deployments at B = 4 and S_in 1024 or 2048 log 8 to 64 MiB. The size grows with B, S_in and the
  hidden size: `E_image = S_in = 4096`, hidden size 4096 and 3 deepstack layers reserve 512 MiB at
  B = 4. The reservation stays allocated for the runtime's life, also after a refused step disables compact
  execution.
- Failed compact setup: when setting up compact resources throws with the stream healthy
  (`LLMRankRuntime::prepareCompactExecution`: the media reservation cannot be allocated, or the
  dedicated context's auxiliary streams cannot be created), the reservation is released, compact
  execution is recorded as unavailable for the runtime's life with a reason naming the failure
  (for the reservation, its layout and size), a warning is logged, and the session serves
  single-cohort steps: `auto` resolves to `prefillFirst` without compact execution and `mixed`
  fails at startup with that reason. Every later session of the runtime keeps that result. A setup that leaves
  the stream in error makes `RankRuntimeSession::open` throw instead.
- Media under a visual-token pruner and grammar-constrained (guided) requests are exclusive
  residents that keep single-cohort steps on a compact deployment: the pruner gates on an empty
  base KV and runs a whole-prompt prefill, and the guided decoder's matchers follow execution
  rows. A request with a token callback is also exclusive (see the request-level matrix).
- Media whose token layout only its encoder determines (`MultimodalRunner::tokenLayoutNeedsEncoder`)
  is an exclusive resident with single-cohort steps on a compact deployment too, its encoder run
  at admission as on a deployment without compact execution. A Nemotron-Omni video is such media:
  its tubelets' token counts come from EVS pruning, which needs the encoder output, so an encoder
  embedding cache hit cannot lay it out either and the encoder runs again.

### Observability

- Startup: the resolved strategy, its reason, the execution, the budgets, the engine limits
  and the compact capability with the reason it is unavailable, naming any rejecting layer (log
  line above), then one warning per entry of `ResolvedSchedulingPolicy::warnings`.
- NVTX ranges: `IFB_PLAN`, `IFB_EXECUTE`, `IFB_COMMIT` per step on the actor;
  `COMPACT_BUILD`, `COMPACT_ENQUEUE`, `COMPACT_SAMPLE`, `COMPACT_COMMIT` inside a compact
  step (built with `ENABLE_NVTX_PROFILING`).
- `RequestEngine::metrics()` (`EngineMetrics`, nanobind `metrics()`, server `/health`
  `scheduling`): besides queue latency, admission stalls and outcomes, counted over steps
  that ran: steps by kind (`decodeSteps` and the `decodeGraphSteps` among them that replayed
  a CUDA graph, `promptSteps` and the `mixedSteps` among them, `decodeSubsetSteps`), running
  sums of N, Np (`stepPromptChunks`, `partialPromptChunks` of them non-final), Nd, T and L
  (`stepSequences`, `stepDecodeRows`, `stepTokens`, `stepSampledRows`), actor host step time
  (`stepTimeTotalUs`, `stepTimeMaxUs`, plan through commit), `pausedResidentSteps`, and
  `compactRefusals`, the compact steps a plugin refused.
- `RequestEngine::status()` (`EngineStatus`, nanobind `status()`): `serving`, false once the engine
  opens no session again (see [Failure handling](#failure-handling)), `unavailableReason`, and
  `compactUnavailableReason`, the current session's capability reason or the refusal that stopped
  the engine. The server's `/health` adds it, with the policy in effect (mode, execution, reason,
  scheduling policy, applied preemption, budgets, reserved slots, pause limit and warnings), as the
  `engine` block (`LLM.get_engine_status()`), and answers HTTP 503 with `status: "unhealthy"` and a
  top-level `reason` while `serving` is false; `/health/ready` answers 503 with the reason too, and
  `EngineClient._prepare_native_request` refuses every new request with 503 `engine_unavailable`.
- `RequestHandle::stats()` (nanobind `stats()`), available once the request is ready:
  admission, queue time, time to first token measured at the end of the step that sampled
  it, prompt and reused prompt tokens, prompt chunks, and the steps and host time the request
  was paused.

## Support matrix

Legend: **served** -- runs under IFB, may share a step; **exclusive** -- runs under IFB
but alone: it is admitted only into an empty session and nothing joins it while it is
resident (the head waits, it is never rejected);
**blocking path** -- the server keeps the deployment on
`LLMInferenceRuntime::handleRequest`, IFB is never engaged, and starting such a deployment
*with* `--enable-in-flight-batching` fails with a message naming the reason (one gate,
`ifb_unsupported_reason`, decided once at load time -- never a silent fallback);
**text-only** -- the deployment starts under IFB and serves text; its speech endpoints
refuse each request with a message naming the flag;
**rejected** -- `submit()` throws `std::invalid_argument`.

### Deployment level

| Deployment | Status | Notes / target |
|---|---|---|
| Text LLM, single rank | served | |
| Text LLM, tensor parallelism (thread- or MPI-launched) | blocking path | The session runs on one rank in this release; `RequestEngine` refuses a multi-rank runtime at construction and the server keeps the deployment on `handleRequest`. Supporting this requires broadcasting each step to every rank. |
| Vision / audio **input** (multimodal) | served | Encoders run at admission; see the visual-token-pruner row below. |
| Speculative decoding (EAGLE, MTP, Gemma4 MTP, DFlash, JetSpec, Domino, DSpark) | served | Single-cohort steps only (see [Speculative decoding](#speculative-decoding)), with or without a context cache. |
| Diffusion backbone | blocking path | Cannot be seated mid-request; same construction-time refusal. |
| Bounded sliding-window KV storage | blocking path | Its request backend has no live-slot admission; same construction-time refusal. Full KV storage (`swaKVCacheMode=full`) is served. |
| Hybrid Mamba (Nemotron-H) | served | Admitted sequences prefill in place and no per-slot state is relocated. Validated in `mixed` on B100 and SM120 against a `singleCohort` run of the same engine with the IFB trace checker's tolerance (top-2 near-ties, centered logit cosine of at least 0.999): batched SSD and conv kernels are not bit-exact with single-request runs. |
| Speech **output**, Qwen3-Omni bundle | **text-only** | The Thinker is a text runtime the engine serves, so the deployment starts under IFB with the Talker/code2wav engines unloaded; `/v1/audio/speech` and audio output on chat are refused per request with a message naming the flag, and the capabilities report no speech. The Talker pipeline calls the runtime beyond `handleRequest`, which the actor does not mediate. |
| Speech **output**, standalone Qwen3-TTS model | blocking path | No request engine exists for a TTS-only runtime; the flag is refused at startup. |
| Context cache (prefix reuse) | served | The session opens one cache request for its life; every sequence, full-attention or attention-plus-recurrent hybrid, leases its pages at admission and returns them when it retires. A hybrid sequence restores only its resident recurrent slot, and cache visibility commits with the step that produced it. Under compact execution several sequences may be in admission at once and publish at their final chunk. |
| Compact execution (`mixed`, multi-request `prefillFirst`) | served where the static preconditions hold and every layer kind has phase-8 kernels on this GPU | Text and multimodal deployments whose plugins run phase 8 on this GPU (for example Qwen3, Qwen3.5 and Gemma4 E2B with full sliding-window KV storage on SM100). TensorRT-RTX engines, engines with `maxBatchSize < 2` or without a phase-8 prefill profile, n-gram-embedding and Gemma4 vision-bidirectional deployments, cached Gemma4 per-layer embedding storage with fewer than `min(PLE table rows, B x S_in) + 1` slots, vision or audio encoders whose output is not FP16 `[rows, hidden size]`, runs with the layer debugger, and engines with a layer kind or shape the compact capability table refuses on this GPU use `prefillFirst` with single-cohort execution; requesting `mixed` there fails at construction. Speculative, diffusion and bounded sliding-window deployments stay on the blocking path (rows above). |

### Request level (what `submit()` does with a request)

| Request option | Status | Notes / target |
|---|---|---|
| Plain text generation, streaming, logprobs, stop strings | served | |
| Sampling parameters, seed, `max_tokens` or logprobs width that differ from the residents' | served | Each sequence carries its own settings (`DecodingInferenceContext` per-slot vectors). Sampling runs the top-k kernels once over every row that uses top-k (greedy rows as top-k 1) and the top-p kernels once over the rest, each row reading its own parameters (`sampleRows` in `cpp/sampler/sampling.h`), so the launch count does not depend on how parameters interleave; every row draws its randomness from its own seed, so a sequence's tokens do not depend on who shares its batch. Only `enableThinking` and `proposalSampling` remain batch-wide. |
| Temperature, top-p or top-k that differ from the residents' on a speculative deployment | waits | A speculative decoder takes one set of sampling parameters per batch, and the sampling parameters select the decoder (see [Speculative decoding](#speculative-decoding)). The head waits until the session is empty; seeds, budgets and logprobs widths remain per sequence. |
| `disableSpecDecode` on a speculative deployment | waits | The sequence runs on the default decoder with the blocking path's sampling and never shares a step with speculative sequences: the head waits until the session is empty, and opt-out sequences batch with each other. A speculative head behind them waits in turn. |
| Guided decoding (grammar / JSON) | exclusive | The `GuidedDecoder`'s matchers are numbered by execution row and the seated prefill's row swap does not cover them; a constrained sequence runs as the session's only row, with single-cohort steps even on a compact deployment. |
| `priority` | served | Ordering under `schedulingPolicy=priority`; ignored with one warning under `fcfs`. Not a batch-compatibility field. |
| Media under an active visual-token pruner | exclusive | Pruning gates on an empty base KV; a sequence seated beside residents would silently skip it. It runs a whole-prompt prefill with single-cohort steps even on a compact deployment. |
| Nemotron-Omni video, compact deployment | exclusive | Its token layout comes from the encoder (EVS), so it cannot be laid out before its slot is reserved; it runs its encoder at admission and a whole-prompt prefill with single-cohort steps. Served like other media without compact execution. |
| `onTokenGenerated` (token callback) | exclusive | The context holds one callback for the whole batch, so a callback-bearing request runs as the session's only resident; its first token reaches the callback from the context step like every later one. |
| `generateAudio` (speech output) | **rejected** | Response state is assembled only by the blocking path. Use the blocking path. |
| `pastTrajectory` (trajectory execution) | **rejected** | Trajectory execution carries blocking-path response state. Use the blocking path. |
| Per-request LoRA (`loraWeightsName`) | **rejected** | Adapter weights are engine-wide today. |
| More than one sequence per request | **rejected** | The engine batches across requests, not within one. |

The legacy `saveSystemPromptKVCache` request field is retired: it is superseded by
the context cache / pre-engine warmup. A shared prefix is warmed through the blocking
path before the runtime is handed to the engine, or routed to the context cache (prefix
reuse), which already covers the shared-prefix use case under IFB. No caller needs the
legacy capture on the engine path.

## Where the boundaries are enforced

- Construction (`RequestEngine::RequestEngine`): the runtime must run on a single rank
  (`supportsSteppedExecution`), must be able to seat sequences into a running session
  (`supportsSeatedAdmission`), and `maxBatchSize` may not exceed the engine's built batch
  dimension (`maxBatchSize()`). The session is opened here, on the actor's thread; a
  runtime that refuses to open one fails construction.
- Submission (`RequestEngine::submit`): the **rejected** rows above.
- Admission (`RequestEngine::mayAdmit`): the **waits** and **exclusive** rows above. These
  wait rather than reject, so FCFS order is preserved: the head is admitted once the
  session is empty, and while an exclusive sequence is resident nothing is admitted beside
  it. `RuntimeSession::requiresExclusiveAdmission` tells the engine which requests are
  exclusive.
- Runtime (`RankRuntimeSession::admit`): a request the session cannot take is refused as
  transient (`kNoCapacity`) when nothing was modified, or rejected with a reason when the
  intent itself is invalid. A request whose batch-wide parameters disagree with the
  residents' fails loudly there -- the gate above is what keeps them equal. A failure
  inside the reservation itself (the context-cache lease), media staging that fails on the
  device, or an admission preprocess that leaves the session's stream in error is not a
  refusal: it is reported as `SessionLostError`, and the engine fails the head and the
  session together, as it does for a failed step. An encoder that refuses its input is a refusal.
- Steps (`RuntimeSession::execute`): the engine's plan (one `ScheduledSequenceDescriptor`
  per participant, built from the resident table's committed lengths) is checked against
  the session's ledger before it runs; a plan that disagrees is a contract violation and
  fails the step. The result's execution record says what ran -- how many positions each
  sequence gained, which a visual-token pruner makes fewer than the prompt range a context
  step was asked to cover -- and the resident table advances from that record, never from
  its own plan. After every step the table and the session must name the same residents;
  a divergence fails the session rather than the step that would eventually trip over it.
- Failures: see the next section.

## Failure handling

A failure is answered at the narrowest scope whose state is still known to be sound.

- **The request's own.** Validation, templating, tokenization, the encoder preprocess and,
  under a context cache, the media-placeholder count run before anything is reserved
  (`GenerationSession::buildAdmissionIntent`); a throw there refuses the head and modifies
  nothing, unless it leaves the session's stream in error. The encoders run on that stream, so
  the next step would fail whichever request was blamed: `RankRuntimeSession::admit` then
  throws `SessionLostError` and the failure is the session's. A seated prefill that fails --
  the engine reports failure, or throws -- closes its admission (`failAdmission`: the row is
  terminal with `kError`, its lease aborted) and the context step's eviction files it in that
  same step, while the residents beside it decode on. That eviction synchronizes the stream,
  so it is also the check that the device survived: a sticky device error surfaces there and
  fails the step. A response that cannot be materialized at commit fails that one sequence.
- **The session's.** A throw inside the reservation (the context-cache lease, device work) or
  an admission preprocess that leaves the session's stream in error leaves the session's
  device state unknown. A decode step that is not `ok` or throws, a plan the session refuses,
  or a divergence between the resident table and the session leaves a step half-applied
  that every resident took part in. In either case the engine retires every resident with the
  reason, closes the session (`RequestEngine::failSession`) and opens a fresh one. Closing is a full reset:
  the context is new, the session's one context-cache request is released
  (`ContextCacheCoordinator::abandon` synchronizes its pending device work, or quarantines
  it and poisons the coordinator when that fails), and the next session opens with a
  cleared recurrent/convolution state pool and asserts that the coordinator holds no
  request. Hybrid checkpoints the failed step had staged are dropped with the request; the
  ones it had already published were backed by device work that `abandon` synchronized,
  and when that synchronization fails the coordinator is poisoned and no later session can
  look them up. There is no in-place recovery of a failed decode step: none of its
  residents has state known to be good, so evicting them in place reaches the same empty
  session at the cost of a compaction over a half-applied step. vLLM's `EngineCore` and
  TensorRT-LLM's executor loop likewise fail every request on a step exception.
- **A refused compact step.** A step whose `StepResult` has `compactRefused` set (see
  [Execution and strategy](#execution-and-strategy)) is a session failure: the session closes and
  `EngineMetrics::compactRefusals` counts it. No resident continues in place, because the layers
  before the refusing one already advanced their recurrent and convolution state. What follows
  depends on the configured mode:
  - `auto` or an explicit `prefillFirst`: a resident whose prompt's final chunk has not committed
    has sampled no token, so its caller has seen nothing and it can run again from scratch on the
    next session. `RequestEngine::requeuePromptIncomplete` takes these off the ledger
    (`RequestCoordinator::releasePromptIncomplete`), has the session release each one's stream
    channel unfinished and unattached (`RuntimeSession::releaseForReadmission`, so the stream
    finalizer skips it when the session closes; the context-cache lease returns with the session
    as for any resident), and queues the request again at its `(priority, arrival)` place, ahead of
    later arrivals of the same or a less urgent priority, with its handle, stream, cancel token and backlog reservation. Its stats restart
    and its queue latency is not counted twice. A cancelled one retires as cancelled, including one
    the session retired at the top of the refused step, which has nothing to release. The other
    residents, which have streamed tokens, fail with `StepResult::failure`, which says that the
    request can be retried. Their outcome carries `FailureKind::kCompactStepRefused`
    (`RequestHandle::get()` throws `RequestFailedError`, nanobind `RequestFailedError.kind`), which
    the server maps to HTTP 503 `compact_step_refused` with `Retry-After`. The engine reopens the session without counting the failure
    toward `maxConsecutiveSessionFailures`, since the runtime reports no compact execution for its
    remaining life and the refusal cannot repeat. It resolves the policy again against the new
    session (`RequestEngine::adoptPolicy`): both become `prefillFirst` over single-cohort steps, the
    planner is removed, and `preemption=pause` under `auto` is dropped with its startup warning.
    Queued requests are served by the new session. The residents' failure is published after the
    reopen: when no session opens or the policy cannot be resolved again, the engine stops serving
    and they fail with `FailureKind::kEngineStopped` and the reason, not as retryable.
  - An explicit `mixed`: it asks for compact steps, which the runtime no longer runs, so the engine
    does not reopen. Every resident fails, none is queued again, with a reason that names the
    refusal and states that the engine stops serving because `ifbSchedulingMode=mixed` requires
    compact execution. Every later
    request fails at admission with `the runtime session is unavailable: ifbSchedulingMode=mixed
    requires compact execution, and a plugin refused ...`, which suggests `auto`. All of them fail with `FailureKind::kEngineStopped`, which the
    server maps to HTTP 503 `engine_unavailable`. `resolvedPolicy()` keeps the mixed policy, and
    `status().serving` is false.
- **The engine's.** Opening a session hardly touches the device, so a reopen succeeds even
  when the failure's cause is sticky. After `EngineConfig::maxConsecutiveSessionFailures`
  session failures in a row with no step completed between them (default 2; a refused compact
  step does not count), or when no session can be opened at all (a poisoned coordinator refuses
  every request), the engine stops reopening: every later request fails at admission with the
  reason (`FailureKind::kEngineStopped`) rather than parking its caller, and the engine keeps running
  until it is recreated.
  `RequestEngine::status()` then reports `serving` false with that reason, and the server's
  `/health` and `/health/ready` answer HTTP 503, as does every new request (`engine_unavailable`).

## Cancellation

A request's cancellation is one bit the request owns (`RequestRecord::cancelToken`): set
on the caller's thread by `RequestHandle::cancel()` or `RequestEngine::cancel(id)`, read by
the actor before every admission and at every drain, and polled by the runtime at the top
of every step the sequence takes part in (it travels into the slot at admission). The
token channel keeps token transport and finish; its own cancel flag remains for consumers
that cancel the channel directly, and counts as a cancel too. Under compact execution the
top of every step checks every resident, so a request cancelled between prompt chunks is
retired before its next chunk runs.

## Speculative decoding

A speculative deployment runs one speculative round per decode step over every resident. The
decoder is a batch-wide parameter: the first sequence into an empty session selects it as the
blocking path does (`DecoderRegistry::select`), from `disableSpecDecode` and the sampling
parameters. A speculative decoder also takes one set of sampling parameters per batch
(`DecodingInferenceContext::uniformSamplingParams`). So beside the `BatchCompatibility` gate, which
compares `disableSpecDecode`, `RequestEngine::mayAdmit` holds a head when it or the residents run
on such a decoder (`RuntimeSession::requiresUniformSampling`) and its temperature, top-p or top-k
differ from theirs (`BatchCompatibility::samplingAgrees`). Requests that sample alike select the
same decoder, so a head that would select a different decoder waits too, counted in
`stallsIncompatible`.

**Seating.** A sequence's draft state is built in its own context step. Inside the batch-one
projection, after the base prefill, `DecodingStrategy::seatRow` runs the draft prefill over every
prompt position but the last and leaves the row as if its last verification had accepted one
token: the first sampled token, with the target hidden state at the last prompt position. The
next decode step treats the row like every resident. Gemma4 MTP has no draft KV and seats the
seed alone.

**Carried state.** Each decoder keeps the target hidden states of every row's accepted span
itself, since a seated prefill overwrites the pipeline's `baseHiddenStates`. Draft KV lengths
persist across rounds, so the projection swaps every cache manager's row (base and draft) to row
zero and back.

**Ledger.** A speculative round reports each sequence's accepted length as the step record's
`queryLength`; the caller's ledger advances by that.

**Context cache.** The cache batch is opened for the cohort's decoder (a speculative one leases
paired base and draft working sets) and reopened when an empty session latches the other kind. A
seated draft holds the prompt but its last position, which the next decode step materializes.
- An EAGLE-flavor endpoint publishes at the row's first decode completion, capped at the common
  state, or at its eviction if it leaves before decoding.
- A Hybrid+spec sequence runs the blocking path's endpoint-reuse prefill as a batch of one
  (`runHybridSpecPrefill`): the boundary fold on a hit, the volatile-tail split, and the checkpoint
  publication after the seat. DFlash and DSpark write the last prompt position for the checkpoint
  and roll the draft length back by one. Such a request must use `PREFILL_STATE_ONLY` with
  endpoint-only capture, as on the blocking path.

**Not yet:** compact steps (a verify segment beside a prompt chunk), per-sequence sampling and
per-step draft length. DSpark's probabilistic acceptance draws its uniforms by execution row and
decode round, so a sampled DSpark request is not reproducible across batch compositions.

**Batch bound.** The session admits at most the smaller of the base and draft engines' built
batch (`LLMRankRuntime::maxBatchSize`).

## Follow-ups

The remaining work is ordered by runtime dependency on the session:

1. Encoder batching and encoder/decoder overlap for media admissions.
2. Compact execution where kernels are missing: Mamba2 SSD outside Blackwell GPUs, and vision-
   block bidirectional and attention-sink ragged FMHA. Also per-SM kernel groups in the artifact
   metadata, so that a combined multi-SM artifact defines `CUTE_DSL_CAUSAL_CONV1D_ENABLED` and
   the GDN group only for the SMs that carry them.
3. CUDA graphs for steps with prompt chunks (piecewise capture), if the measured eager host
   overhead of compact steps justifies it.
4. Speech output: assemble the Talker/TTS response inside the session so `generateAudio`
   requests can be admitted.
5. Trajectory execution as session state.
6. Speculative decoding under IFB, beyond single-cohort steps: speculative rows in compact
   steps, chunked draft prefill, per-sequence sampling with rejection sampling, and a per-step
   draft length chosen by batch size.
7. Guided decoding beside other residents: matchers keyed by resident handle.
8. Tensor parallelism under IFB: broadcast each step to thread-launched ranks, then carry the
   same stream over MPI.
9. Recovery by recomputation after other step failures: a refused compact step already requeues
   its residents that sampled no token; a failed step of any other kind still fails every
   resident, and the ones that streamed tokens could be recomputed in a fresh session too.
