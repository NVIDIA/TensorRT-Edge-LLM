# HTTP Inference Server Architecture

## Scope

The experimental inference server is an adapter over the TensorRT Edge-LLM
high-level runtime. It owns HTTP protocol behavior and request lifecycle, while
model execution and device memory remain owned by the C++ runtime. The public
wire contracts are the OpenAI-compatible Chat, Audio, and Models APIs and the
Anthropic Messages API; those names identify protocols, not implementation
ownership.

The protocol frontend is deliberately separated from model preparation and the
native runtime. HTTP code does not own engines, weights, or generation state.

```text
CLI/config -> model cache/build -> LLM load -> EngineClient -> serving -> routes
                                                            -> parsers
```

## Layers

| Layer | Files | Responsibility |
|---|---|---|
| Model preparation | `runtime/engine_build.py`, `runtime/engine_layout.py` | Resolve checkpoints, select a profile-specific cache entry, build every model-owned component on a miss, and validate the complete bundle. |
| Configuration | `config.py`, `cli.py` | Parse and validate model, cache, profile, and HTTP options before loading. |
| Runtime adapter | `runtime/engine.py`, `runtime/engine_client.py` | Isolate native execution, expose capabilities, provide admission control, and own cancellation cleanup. Owns the choice between the one-at-a-time runtime and the in-flight-batching request engine. |
| Protocol and HTTP | `api/` | Define typed OpenAI-compatible data, adapt Anthropic Messages requests, validate model/request compatibility, and own routes and application assembly. |
| Media | `media/` | Resolve and preprocess audio, image, and video inputs independently of protocol routing. |
| Parsing | `parsing/` | Register reasoning and tool-output parsers and apply tool-aware chat templates. |

The package root contains only public exports, configuration, the CLI, and
pybind setup. The former flat module paths are not compatibility aliases; each
implementation has one owning package.

The CLI and `LLM.serve()` both construct the same `EngineClient` and FastAPI
application. There is no second compatibility serving path.

## Model Loading

The public source is always a supported local checkpoint or Hugging Face model
ID. `cache_dir` is the only artifact-location option. Engine directories are
internal cache entries and cannot be served directly. ONNX is not part of this
path.

A cache key covers base checkpoint metadata, optional draft checkpoint
metadata, and every build-profile value. A miss invokes the experimental
builder once with all model-owned components and supported external weight
kinds. The build occurs in a staging directory under an exclusive cache lock;
only a validated complete bundle is published. Generated configs record the
checkpoint identity, which the C++ loader validates before binding runtime
weights.

## Request Lifecycle

On the default path the runtime supports one in-flight generation. `EngineClient`
provides a bounded async queue around that slot:

1. Validate the request and loaded-model capability.
2. Reserve the runtime slot or return 429 when the queue is full or times out.
3. Construct the native generation request before streaming response headers.
4. Run blocking runtime calls outside the event loop.
5. Read prompt and completion token counts from the native generation result.
6. On completion, failure, or disconnect, cancel the native channels, join the
   generation worker, and release the slot exactly once.
7. During application shutdown, reject new reservations, drain the active
   request, and destroy the native runtime once.

Normal generation never invokes the tokenizer merely to report usage. The
tokenizer count operation exists only for the explicit token-count endpoint.

Context reuse is a runtime-construction option, not an engine-build profile.
When enabled, the native coordinator owns retained pages, records, request
policies, and metrics. `EngineClient` continues to serialize requests so the
coordinator's single-writer contract is preserved. Protocol adapters must
follow the user-facing [cache-domain requirements](../../user_guide/features/kv-cache-reuse.md#cache-domain).

The admission lease and native lock have separate roles. `EngineClient` owns
the asynchronous queue and request-scoped media buffers. `LLM._infer_lock`
serializes entry into the stateful chat/Omni C++ runtime; a TTS-only runtime has
its own single admission gate. A disconnected stream retains its admission
lease until its native worker has returned, so no later request can overlap
teardown. HTTP connections do not own the model: engines remain resident
between requests and are released by the FastAPI lifespan on server shutdown.

### In-flight batching path

With `--enable-in-flight-batching`, `LLM._load_runtime` decides once, at load
time, whether the deployment can honour the flag (`ifb_unsupported_reason`) and
then builds a `RequestEngine` instead of an `LLMRuntime`. The engine owns the
runtime and the one thread that drives it, so the lifecycle above changes in
four places:

1. Admission is `_IFBAdmissionGate`, sized to `max_batch_size` plus
   `max_queued_requests`. It never parks a caller: a full gate is an immediate
   429, mirroring the engine's own `SubmitError` back-pressure.
2. Generation is `RequestEngine.submit()` plus `RequestHandle.get()`; there is
   no `_infer_lock` and no per-stream worker thread. Streams read their
   `StreamChannel` on a pool sized for the gate, and every stream the client did
   not cancel collects its outcome through `get()`, so an execution error behind
   a terminal chunk still reaches the client.
3. A disconnect cancels the request in the engine through `RequestHandle.cancel()`
   (and, for streams, the channel's cancel flag, which the actor checks before
   every admission), so the request leaves its batch seat instead of decoding to
   `max_tokens`.
4. Direct Python API callers take the per-`LLM` semaphore, which under the
   engine is widened to the same batch-plus-queue bound; the HTTP layer bypasses
   it with prebuilt requests as before.

Which deployments refuse the flag at startup, which requests the engine rejects
or makes wait, and the follow-up work are recorded in
[In-Flight Batching (V0)](in-flight-batching.md).

## Extension Rules

- Add protocol fields to typed models and either implement or reject them. Do
  not accept ignored fields.
- Keep model-output parsing in a named parser registry. Routes must not contain
  model-specific token logic.
- Read runtime capabilities from generated metadata. Launch flags cannot
  mutate a cached engine.
- Keep context-cache deployment validation and page ownership in the native
  runtime. Python may validate scalar configuration but must not duplicate
  model-specific reuse rules.
- Reject unsupported launch options during configuration validation.
- Build stream requests before sending status 200 so input and admission errors
  remain normal JSON errors.
- Preserve each route's streaming schema. OpenAI-compatible streams include
  stable IDs, model name, created timestamp, nullable finish reason, optional
  final usage, and `[DONE]`; Anthropic streams preserve Messages event order.
- Do not add HTTP concurrency above one until the runtime provides independent
  request state and KV-cache ownership.
