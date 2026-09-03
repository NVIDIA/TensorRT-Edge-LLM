# KV Cache Reuse

KV cache reuse is a process-local, content-addressed cache for repeated input
prefixes. It can reuse prefill state from documents, prior turns, generated
continuations, and repeated image prefixes. Entries are isolated by LoRA
adapter and live for the lifetime of one runtime instance.

## Support Matrix

This is the authoritative support matrix for generalized context reuse. It does
not describe the legacy exact system-prompt cache or the separate encoder
embedding cache.

Every supported attention pool, including an independently owned draft pool,
must use FP16 or FP8 KV storage and valid page geometry. For speculative
deployments, the base role, draft role, method, and execution geometry must
match.

| Deployment | Status | Contract |
|---|---|---|
| Vanilla, attention-only text model | Supported | No snapshot pool is required. |
| Vanilla, pure recurrent text model | Supported | The recurrent snapshot budget must provide at least one slot. |
| Vanilla, hybrid attention/recurrent text model | Supported | Recurrent and partial-KV snapshot budgets must each provide at least one slot. |
| Vanilla image input on a supported base topology | Supported in the C++ runtime | Uses [media identity](#media-identity). |
| Vanilla audio input on a supported base topology | Implemented, not release-qualified | Uses [media identity](#media-identity), but release validation does not cover audio-input reuse. |
| Attention-only EAGLE3 | Supported | The draft owns an independent KV pool, and the final matched page is replayed for the one-token future dependency. Linear and tree drafting are accepted. |
| Attention-only DFlash, JetSpec, or DSpark | Supported | The draft owns an independent KV pool. These methods have no future-token replay dependency. |
| Gemma4 MTP | Supported | The read-only assistant shares target KV, owns no draft page pool, and has no future-token replay dependency. |
| MTP with a hybrid attention/recurrent base | Supported | The attention-only draft owns independent KV. Reuse requires a text-only batch of one, endpoint-only capture, `prefill_state_only` publication, and both snapshot pools. The successor boundary is reconstructed when a checkpoint is restored. |
| Standard MTP with an attention-only base | Rejected | Only hybrid MTP and the separate Gemma4 MTP contract are supported. |
| Pure-recurrent speculative decoding, or hybrid EAGLE3/DFlash/JetSpec/DSpark | Rejected | The only supported recurrent speculative deployment is hybrid MTP. |
| Speculative request containing image or audio input | Bypassed | The request runs cold on coordinator-managed pages and neither looks up nor publishes reusable state. |
| Speech generation or thinker-hidden-state output | Bypassed | The request runs without lookup or publication. |
| Vision-bidirectional attention or block diffusion | Rejected | The runtime cannot initialize generalized context reuse for these deployments. |
| Action runner | Rejected | The runtime cannot initialize an action runner and generalized context reuse together. |
| Tensor parallel execution | Not supported | The public inference workflows reject context reuse when tensor-parallel size is greater than one. |

The experimental Python API and server expose the supported text-only
contracts through the same native runtime.

### Media Identity

For vanilla media requests, decoded image bytes or PCM audio samples and their
placeholder positions are part of the cache identity. Audio without decoded
PCM uses its mel-spectrogram path. Changing media content or order invalidates
the affected prefix. Speculative media requests follow the bypass behavior in
the matrix.

### Cache Domain

A cache-enabled runtime is one trusted domain. Use separate runtime instances
when requests require tenant isolation; there is no request-level tenant or
salt key.

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

Reuse remains page-aligned, and the runtime recomputes a media span when a page
boundary would split it.

The cache retains up to 1024 records by default. Use
`--contextCacheMaxRecords` only when the deployment needs a different limit.

For matrix rows that require snapshot storage, configure the corresponding
budgets:

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

`bypass` uses coordinator-managed private pages but does not look up or publish
reusable state. Do not combine generalized context reuse with the legacy
`save_system_prompt_kv_cache` request field.

## Configuring the Encoder Embedding Cache

The encoder embedding cache is separate from KV cache reuse. A runtime with a
ViT or audio runner allows cache entries to use up to 256 MiB of device memory
by default even when `--enableContextReuse` is not set. It hashes decoded image
bytes or PCM audio samples and retains the corresponding encoder output on the
GPU. A request for identical media can then skip ViT or audio encoder
execution. The cache evicts least-recently-used entries when its budget is
full; an individual entry larger than the budget runs uncached.

Set the C++ runtime budget in bytes with `--encoderCacheBudgetBytes`; zero
disables the cache:

```bash
./build/examples/llm/llm_inference \
  --engineDir /path/to/engine \
  --multimodalEngineDir /path/to/encoder/engine \
  --inputFile input.json \
  --outputFile output.json \
  --encoderCacheBudgetBytes 134217728
```

The experimental Python API and server use the same default but do not expose a
separate encoder-cache budget in this release. An INFO log reports
when all media items hit and encoder execution is skipped. Encoder-cache
hit/miss counters are not included in the profile JSON; the context-cache
profile fields below describe KV/recurrent-state reuse only.

Use runtime profile output to verify reuse. `prefill.reused_tokens` must be
positive for a request that reused cached state. For image requests,
`context_cache.media_aware_sequences` reports how many admitted sequences used
media-aware cache identities.
