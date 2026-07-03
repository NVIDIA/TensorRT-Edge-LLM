# LLM Streaming — Design

Streaming output on top of `LLMInferenceRuntime`. Per-slot chunked delivery, works for vanilla + Eagle spec-decode, text + multimodal, with cancellation and per-slot throttling.

## Goals

- **Per-batch-item streaming** — one channel per slot, attached to `LLMGenerationRequest`.
- **Typed deltas** — `StreamChunk { tokenIds, text, finished, reason }` with always-well-formed UTF-8 on the wire.
- **Burst-atomic emits** — one chunk per iteration per slot, regardless of whether vanilla decode appended 1 token or spec-decode accepted N.
- **Batch-compaction safety** — per-slot state compacts in lockstep with `tokenIds` in `performBatchEvict`.
- **Cancellation** — fire-and-forget from any thread; KV released on the next eviction boundary.
- **Terminal-chunk guarantee** — RAII finalizer ensures every attached channel unblocks on every exit path.
- **Zero impact on non-streaming callers** — empty `streamChannels` means the response path is byte-identical to before.

## Key decisions

| Concern | Decision |
|---|---|
| Sync primitive | `std::mutex` + `std::condition_variable` + `std::deque<StreamChunk>` per slot |
| Detokenization | Strategy A: per-token piece lookup + UTF-8 sanitizer. Works because `Tokenizer` stores pre-ByteLevel-decoded bytes, so `decode([A]) ++ decode([B]) == decode([A,B])` exactly. |
| UTF-8 safety | `sanitizeUtf8Streaming` holds trailing incomplete codepoints across iterations; `sanitizeUtf8Flush` converts held bytes to U+FFFD on the final chunk. Invalid bytes mid-stream become U+FFFD inline. |
| Ownership | `std::shared_ptr<StreamChannel>` attached to each slot; factory-enforced to prevent stack allocation. |
| Producer API | Private on `StreamChannel`; reachable only by named friend free functions + `StreamChannelFinalizer` — consumers can't skip the final-chunk consume or hold the lock. |
| Cancellation | `std::atomic<bool> cancelled` on the channel; runtime observes at iteration top, latches `terminalReason=kCancelled`, proceeds through `performBatchEvict`. |
| Overflow policy | Phase-1 unbounded `std::deque`. Add coalescing if measurements demand. |
| Submission model | Synchronous `handleRequest` on caller's thread. Consumer runs on a separate thread. Worker-thread submit/run split deferred to Phase 2. |

## Data types (`cpp/runtime/streaming.h`)

```cpp
enum class FinishReason : uint8_t { kNotFinished = 0, kEndId = 1, kLength = 2, kCancelled = 3, kError = 4, kStopWords = 5 };

struct StreamChunk {
    std::vector<int32_t> tokenIds;   // delta tokens since last chunk
    std::string text;                // delta text, always well-formed UTF-8
    bool finished{false};            // true for the terminal chunk
    FinishReason reason{FinishReason::kNotFinished};
};

class StreamChannel {
public:
    static std::shared_ptr<StreamChannel> create();

    // Consumer API
    template <typename Handler>
    void consume(Handler&& handler, std::chrono::milliseconds poll = 100ms);
    std::optional<StreamChunk> tryPop();
    std::optional<StreamChunk> waitPop(std::chrono::milliseconds timeout);
    void cancel() noexcept;
    void setStreamInterval(int32_t n) noexcept;

    // Status queries (atomic reads, any thread)
    bool isFinished() const noexcept;
    FinishReason getReason() const noexcept;
    int32_t getOriginalBatchIdx() const noexcept;
    bool isCancelled() const noexcept;
    int32_t getStreamInterval() const noexcept;

private:
    // Producer API is private; friends below are the only callers.
    void push(StreamChunk);
    void finish(FinishReason);
    void setOriginalBatchIdx(int32_t) noexcept;

    friend class StreamChannelFinalizer;
    friend void attachStreamChannel(std::shared_ptr<StreamChannel> const&, int32_t);
    friend bool validateStreamingSubmission(LLMGenerationRequest const&);
    friend void applyCancellationToFinishStates(SpecDecodeInferenceContext&);
    friend void emitChunks(SpecDecodeInferenceContext&, tokenizer::Tokenizer const&);
    // ... private state: mutex, cv, deque, atomics
};

// Per-slot detokenization state, compacted in lockstep with context.tokenIds.
struct SlotStreamState {
    std::shared_ptr<StreamChannel> channel;
    size_t sentTokenCount{0};
    size_t lastEmittedTokenCount{0};
    std::string pendingBytes;
    FinishReason terminalReason{FinishReason::kNotFinished};
};
```

## Runtime integration — `handleRequest` flow

Five insertion points plus one RAII guard:

1. **M5 submission validation** — `validateStreamingSubmission(request)` rejects mismatched sizes, already-finished channels, or channels concurrently attached to another request.
2. **Setup (after `setUpForPrefillExecution`)** — loop over slots, call `attachStreamChannel(ch, originalIdx)`, record the channel in `context.slotStreams[i]`, seed `sentTokenCount = context.tokenIds[i].size()` so streaming emits only generated tokens. Construct `StreamChannelFinalizer`.
3. **Post-prefill** — `applyCancellationToFinishStates(context)` (cancel wins over natural finish) → `updateFinishStates` (latches `kEndId`/`kLength` atomically with the `finishedStates[i]` flip) → `emitChunks(context, *mTokenizer)`.
4. **Per iteration** — same ordering at the top of every main-loop iteration, before `performBatchEvict`.
5. **In `performBatchEvict`** — `rt::compactVector(batchMapping, context.slotStreams)` runs alongside the existing `compactVector` calls.

The `StreamChannelFinalizer` destructor runs on every exit path from `handleRequest` (normal return, error return, exception). For every channel not already finalized, it tries to push a terminal chunk with any un-emitted tokens/text (sanitized + flushed) and `reason=kError`, then calls `finish(kError)`. `finish` is idempotent and no-throw; chunk assembly can swallow OOM.

## Terminal-reason latching (race-free by construction)

`terminalReason` is written exactly once per slot, at the instruction that flips `finishedStates[i]` from 0 to 1. Two writers, guarded by `!finishedStates[i]`:

- `applyCancellationToFinishStates` — runs at iteration top. Writes `kCancelled`.
- `updateFinishStates` — runs after the engine step. Writes `kEndId` or `kLength`.

First writer wins; `emitChunks` reads the latched reason on the same (runtime) thread that wrote it — no synchronization needed. Consumer-side reads use `StreamChannel::getReason()` which pairs acquire/release with `finish()`.

## Detokenization algorithm (Strategy A)

Per iteration, per slot with a channel:

1. `newlyAvailable = context.tokenIds[i].size() - sentTokenCount`.
2. If not final and `newlyAvailable < streamInterval`, skip.
3. Concatenate piece bytes: `raw = concat(idToPiece(t) for t in tokenIds[i][sentTokenCount..end])`. Advance `sentTokenCount`.
4. `delta = sanitizeUtf8Streaming(raw, pendingBytes)` — replaces invalid bytes with U+FFFD; trailing incompletes go to `pendingBytes`.
5. On the final chunk: `delta += sanitizeUtf8Flush(pendingBytes)`.
6. If `delta` non-empty OR final, push `StreamChunk { tokenIds[lastEmitted..sent), delta, finished, reason }` and update `lastEmittedTokenCount`.

Key property: output is always well-formed UTF-8, whether the input contained valid multi-token codepoints, adversarial isolated continuation bytes, or both.

## Non-Streaming Path

When `request.streamChannels.empty()`, `slotStreams[i].channel` is null, all five insertion points short-circuit. Zero overhead on the non-streaming path. One intentional content-level change: `Tokenizer::decode` routes output through `sanitizeUtf8Streaming + Flush` so invalid-byte adversarial outputs surface as U+FFFD in `response.outputTexts` — a latent-bug fix, bytes are identical for all valid outputs.

## Non-goals (deferred)

- HTTP/SSE/gRPC server surface.
- Worker-thread submit/run split (Phase 2 — `submitRequest(...) → future<StreamChannel>`).
- Logprobs, `n > 1` parallel samples, tool-calling deltas.
- Stop-string prefix hold-back (reserved field in `SlotStreamState`).

## Code layout

| File | Role |
|---|---|
| `cpp/runtime/streaming.{h,cpp}` | `StreamChunk`, `StreamChannel`, `SlotStreamState`, `FinishReason`, `StreamChannelFinalizer`, streaming free functions |
| `cpp/runtime/llmRuntimeUtils.{h,cpp}` | `LLMGenerationRequest::streamChannels` field; forward-declares `StreamChannel` |
| `cpp/runtime/llmInferenceRuntime.{h,cpp}` | Wires streaming hooks into `handleRequest` |
| `cpp/common/utf8.{h,cpp}` | `sanitizeUtf8Streaming`, `sanitizeUtf8Flush`, + shared byte-level primitives |
| `cpp/tokenizer/tokenizer.{h,cpp}` | `idToPiece`, `emitDelta`, `emitDeltaFlush`; `decode` now sanitizes |
| `examples/llm/llm_stream.cpp` | Interactive demo (JSON input, live chunk printing, hotkey control) |
| `unittests/streamingTests.cpp` | 51 CPU-only invariant tests |
| `unittests/streamingRuntimeTests.cpp` | 4 engine-backed scenario tests |

## User guide

See `docs/source/user_guide/features/streaming.md` for consumer-side usage and examples.
