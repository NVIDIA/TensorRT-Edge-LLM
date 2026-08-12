# KV Cache Reuse

KV cache reuse is a process-local, content-addressed cache for repeated input
prefixes. It can reuse prefill state from documents, prior turns, generated
continuations, and repeated image prefixes. Entries are isolated by LoRA
adapter and live for the lifetime of one runtime instance.

See the [support matrix](../getting_started/support-matrix.md#kv-cache-reuse-support)
before enabling this feature. Reuse requires FP16 KV cache.

## Build Capacity

`--maxKVPoolPages` controls the total page pool. Reserve pages beyond the
active-request minimum when retained contexts must remain resident:

```bash
./build/examples/llm/llm_build \
  --onnxDir /path/to/onnx/llm \
  --engineDir /path/to/engine \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxKVPoolPages 64
```

## Enable Reuse

```bash
./build/examples/llm/llm_inference \
  --engineDir /path/to/engine \
  --inputFile input.json \
  --outputFile output.json \
  --enableContextReuse
```

For a VLM, pass the visual engine and a request set containing repeated image
prefixes:

```bash
./build/examples/llm/llm_inference \
  --engineDir /path/to/engine \
  --multimodalEngineDir /path/to/visual/engine \
  --inputFile tests/test_cases/vlm_context_reuse.json \
  --outputFile output.json \
  --enableContextReuse \
  --profileOutputFile profile.json
```

Image content and its position in the prompt are part of the cache identity.
Repeating the same image prefix can hit; changing an image or changing image
order cannot reuse the affected prefix. Reuse remains page-aligned, and the
runtime recomputes a media span when a page boundary would split it.

The cache retains up to 1024 records by default. Use
`--contextCacheMaxRecords` only when the deployment needs a different limit.

Pure recurrent and hybrid attention/recurrent models also need a recurrent
snapshot pool. Hybrid models additionally need a partial-KV snapshot pool:

```bash
  --contextCacheRecurrentSnapshotPoolBytes 67108864 \
  --contextCachePartialKVSnapshotPoolBytes 67108864
```

The values above are example 64 MiB budgets. Required capacity depends on the
model state dimensions and number of retained contexts.

## Request Policies

The top-level request fields select lookup and publication behavior for the
entire invocation:

```json
{
  "context_cache_lookup_policy": "use_cache",
  "context_cache_commit_policy": "prefill_state_only",
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "A long reusable context followed by a question"}
      ]
    }
  ]
}
```

- `context_cache_lookup_policy`: `use_cache` (default) or `bypass`.
- `context_cache_commit_policy`: `including_generated_tokens` (default) or
  `prefill_state_only`.

Audio generation and hidden-state-output requests bypass lookup. Action runners
are rejected, and audio-input reuse is outside the release-tested boundary.
MTP, DFlash, DSpark, Gemma4 MTP, hybrid EAGLE, FP8 KV cache, and block diffusion
are also unsupported.

Use runtime profile output to verify reuse. `prefill.reused_tokens` must be
positive for a request that reused cached state. For image requests,
`context_cache.media_aware_sequences` reports how many admitted sequences used
media-aware cache identities.
