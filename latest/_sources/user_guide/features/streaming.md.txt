# Streaming Output

Stream generated tokens to your application one chunk at a time instead of waiting for the full response. Works for text and VLM (image) prompts, on both vanilla and supported speculative decoding flows.

## Quick start

Attach a `StreamChannel` per request, submit on a worker thread, consume chunks on your main thread.

```cpp
#include "runtime/llmInferenceRuntime.h"
#include "runtime/streaming.h"

using namespace trt_edgellm;

// 1. Build your request as usual.
rt::LLMGenerationRequest req;
req.requests.push_back(makeUserRequest("Tell me a short story."));
req.maxGenerateLength = 128;

// 2. Attach a channel per slot.
auto ch = rt::StreamChannel::create();
req.streamChannels.push_back(ch);

// 3. Run the request on a worker thread.
rt::LLMGenerationResponse resp;
std::thread worker([&] {
    runtime.handleRequest(req, resp, stream);
});

// 4. Consume chunks on this thread. Blocks until the stream finishes
//    or is cancelled; invokes the handler once per delivered chunk.
ch->consume([](rt::StreamChunk&& chunk) {
    std::fwrite(chunk.text.data(), 1, chunk.text.size(), stdout);
    std::fflush(stdout);
    if (chunk.finished) {
        std::printf("\n[finished: %d]\n", static_cast<int>(chunk.reason));
    }
});

worker.join();
```

After `worker.join()` returns, `resp.outputIds` and `resp.outputTexts` are populated exactly as in the non-streaming path — streaming is additive.

## The `StreamChunk` payload

```cpp
struct StreamChunk {
    std::vector<int32_t> tokenIds;  // delta tokens since last chunk
    std::string text;               // delta text, always well-formed UTF-8
    bool finished;                  // true only for the terminal chunk
    FinishReason reason;            // meaningful when finished == true
};

enum class FinishReason : uint8_t {
    kNotFinished = 0,   // (non-terminal chunk)
    kEndId       = 1,   // model emitted end-of-sequence
    kLength      = 2,   // hit maxGenerateLength
    kCancelled   = 3,   // consumer called channel->cancel()
    kError       = 4,   // runtime aborted (rare — OOM, engine error)
    kStopWords   = 5,   // a per-request stop string was produced
};
```

`chunk.text` is **always well-formed UTF-8** on the wire. Invalid byte sequences the model might produce (isolated continuation bytes, overlongs) are replaced with U+FFFD before the chunk is pushed. You can safely write `chunk.text` into JSON, SSE, or any UTF-8 transport without revalidating.

## Batching with per-slot opt-in / opt-out

One channel per slot, or `nullptr` to opt a slot out of streaming. The `streamChannels` vector must be empty (streaming disabled globally) or the same length as `requests`.

```cpp
rt::LLMGenerationRequest req;
req.requests = {r0, r1, r2, r3};

req.streamChannels = {
    rt::StreamChannel::create(),   // slot 0 streams
    nullptr,                       // slot 1 does NOT stream
    rt::StreamChannel::create(),   // slot 2 streams
    nullptr,                       // slot 3 does NOT stream
};

// Spawn one consumer per streaming channel.
std::vector<std::thread> consumers;
for (size_t i = 0; i < req.streamChannels.size(); ++i) {
    if (!req.streamChannels[i]) continue;
    auto ch = req.streamChannels[i];
    consumers.emplace_back([ch, i] {
        ch->consume([i](rt::StreamChunk&& c) {
            // ... render chunk for slot i ...
        });
    });
}

std::thread worker([&]{ runtime.handleRequest(req, resp, stream); });
for (auto& t : consumers) t.join();
worker.join();
```

Non-streaming slots still populate their `response.outputTexts[i]` / `response.outputIds[i]` as usual — they just don't emit chunks along the way.

## Throttling with `streamInterval`

By default, chunks arrive every iteration (one token per chunk under vanilla decode). For UIs that don't need token-by-token updates, throttle:

```cpp
auto ch = rt::StreamChannel::create();
ch->setStreamInterval(4);  // batch up 4 tokens per chunk
req.streamChannels.push_back(ch);
```

Rules:
- Non-final chunks carry **≥ `streamInterval` tokens** (exactly `streamInterval` under vanilla decode; can be more under spec-decode bursts).
- The final chunk always fires regardless of interval, so the tail tokens aren't lost.
- Must be set before `handleRequest` is called.

## Cancellation

Call `cancel()` from any thread. The runtime observes it at the next iteration boundary, emits any final chunk, and releases KV cache.

```cpp
// From a timer, signal handler, or UI "stop" button:
ch->cancel();
```

Behavior:
- The consumer's `consume()` (or `waitPop()`) wakes immediately — cancel() wakes blocked waiters even if the runtime hasn't produced the terminal chunk yet.
- `response.outputTexts[i]` contains whatever was generated up to the cancel point.
- The terminal chunk (if delivered before `consume()` exits) carries `reason = kCancelled`.
- Calling `cancel()` twice is a no-op.
- Calling `cancel()` after natural finish is a no-op — `reason` stays at whatever terminated first (kEndId/kLength).

Note: cancel takes effect at the next iteration boundary, so a vanilla slot may produce up to 1 extra token, a spec-decode slot up to `maxAcceptDepth` extra tokens, before `performBatchEvict` runs.

## Alternative delivery patterns

`consume(handler)` is the common case. For finer control:

```cpp
// Non-blocking single-shot — returns std::nullopt if the queue is empty.
while (auto chunk = ch->tryPop()) {
    handle(*chunk);
}

// Blocking single-shot with timeout — returns std::nullopt on timeout,
// cancel+empty, or finish+empty. Combine with isFinished()/isCancelled()
// to distinguish.
while (!ch->isFinished() && !ch->isCancelled()) {
    if (auto chunk = ch->waitPop(std::chrono::milliseconds{200})) {
        handle(*std::move(chunk));
    }
}
```

Use `tryPop` when you're multiplexing with other polling work, `waitPop` when you want a per-call deadline.

## Channel lifetime rules

- **Create a new channel per request.** Channels are single-use — once finished, they can't be reattached.
- **Hold the `shared_ptr` until you've finished consuming.** The runtime keeps its own reference; the channel object stays valid even if the runtime outlives your copy.
- **Always drain or cancel.** Dropping a `shared_ptr` without calling `consume()`, `tryPop()` until empty, or `cancel()` leaks the channel and any runtime-held reference to it.

## End-to-end demo: `llm_stream`

`examples/llm/llm_stream` is an interactive CLI that reads an `llm_inference`-compatible JSON input file and streams each response live, with hotkey control.

```bash
# Build
cmake .. \
    -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR \
    -DENABLE_CUTE_DSL=ALL
make -j$(nproc) llm_stream

# Run
export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH=$(pwd)/libNvInfer_edgellm_plugin.so

./examples/llm/llm_stream \
    --engineDir /path/to/Qwen3-4B-Engine-bs4 \
    --inputFile /path/to/input.json \
    --maxGenerateLength 128

# VLM mode
./examples/llm/llm_stream \
    --engineDir /path/to/Qwen3-VL-4B-Engine-bs4 \
    --multimodalEngineDir /path/to/Qwen3-VL-4B-Visual-Engine \
    --inputFile /path/to/vlm_input.json

# Eagle spec-decode
./examples/llm/llm_stream \
    --engineDir /path/to/Qwen3-4B-Eagle-Engine-bs4 \
    --inputFile /path/to/input.json --specDecode
```

Hotkeys while streaming:

| Key | Effect |
|---|---|
| `s` | Skip the current request (calls `cancel()`, moves to next) |
| `q` | Quit (cancel current, stop iterating) |
| Ctrl-C | Same as `q` |

Each request's footer reports tokens generated, TTFT, total time, tok/s, and finish reason.

## Common pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Calling `consume()` on the same thread that runs `handleRequest` | Deadlock — the runtime produces chunks but nothing consumes them, and the runtime waits on the producer-to-consumer handoff | Always run `handleRequest` on a separate thread |
| Reusing a finished channel for a new request | `handleRequest` returns `false` with "StreamChannel already finished" | Create a new `StreamChannel` per request |
| Sharing one channel across multiple requests concurrently | `handleRequest` rejects with "concurrently attached" | One channel per in-flight request |
| `streamChannels.size() != requests.size()` (when non-empty) | `handleRequest` returns `false` with a size-mismatch error | Either empty (disable streaming) or same length, with `nullptr` to opt out |
| Forgetting to call `setStreamInterval` before submission | The setter is a no-op after attach; your value is ignored | Set it before `req.streamChannels.push_back(ch)` |
