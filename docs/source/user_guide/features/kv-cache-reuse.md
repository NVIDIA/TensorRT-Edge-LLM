# KV Cache Reuse

KV cache reuse is a process-local, content-addressed cache for repeated input
prefixes. It can reuse prefill state from documents, prior turns, generated
continuations, and repeated image prefixes. Entries are isolated by LoRA
adapter and live for the lifetime of one runtime instance.

## Support Matrix

This is the authoritative support matrix for generalized context reuse. It does
not describe the separate encoder embedding cache.

Every supported attention pool, including an independently owned draft pool,
must use FP16 or FP8 KV storage and valid page geometry. For speculative
deployments, the base role, draft role, method, and execution geometry must
match.

| Deployment | Status | Contract |
|---|---|---|
| Vanilla, attention-only text model | Supported | No snapshot pool is required. |
| Vanilla, pure recurrent text model | Supported | Keeps [snapshots](#snapshots) of the recurrent state. |
| Vanilla, hybrid attention/recurrent text model | Supported | Keeps [snapshots](#snapshots) of the recurrent state and the last KV page. |
| Vanilla image input on a supported base topology | Supported in the C++ runtime | Uses [media identity](#media-identity). |
| Vanilla audio input on a supported base topology | Implemented, not release-qualified | Uses [media identity](#media-identity), but release validation does not cover audio-input reuse. |
| Attention-only EAGLE3 | Supported | The draft owns an independent KV pool, and the final matched page is replayed for the one-token future dependency. Linear and tree drafting are accepted. |
| Attention-only DFlash, JetSpec, DSpark, or Domino | Supported | The draft owns an independent KV pool. These methods have no future-token replay dependency. |
| Gemma4 MTP | Supported | The read-only assistant shares target KV, owns no draft page pool, and has no future-token replay dependency. |
| MTP with a hybrid attention/recurrent base | Supported | The attention-only draft owns independent KV. Reuse requires both snapshot pools. The successor boundary is reconstructed when a checkpoint is restored. |
| EAGLE3 with a hybrid attention/recurrent base | Rejected | EAGLE3 concatenates 3 layers of hidden states into the boundary snapshot; sizing and restore have not been validated with a real hybrid EAGLE3 model. |
| DFlash, JetSpec, DSpark, or Domino with a hybrid attention/recurrent base | Supported | The attention-only draft owns independent KV. These methods have no future-token replay dependency. Both snapshot pools are required. |
| Standard MTP with an attention-only base | Rejected | Only hybrid MTP and the separate Gemma4 MTP contract are supported. |
| Pure-recurrent speculative decoding (no attention layers) | Rejected | Hybrid speculative reuse requires at least one attention layer in the base model. |
| Speculative request containing image or audio input | Supported | Suffix embedding indices are offset by the cached prefix media count so that image and audio lookups correctly skip media already served from cache. |
| Speech generation or thinker-hidden-state output | Bypassed | The request runs without lookup or publication. |
| Vision-bidirectional attention or block diffusion | Rejected | The runtime cannot initialize generalized context reuse for these deployments. |
| Alpamayo paged action, TP=1 | Supported in the C++ runtime | Captured prefix leases retain read-only FP16 or FP8 LLM pages throughout action inference. Action K/V remains private and is not published for reuse. |
| Action runner, TP>1 | Rejected | The existing dense action path does not enable generalized context reuse. |
| Tensor parallel execution | Not supported | The public inference workflows reject context reuse when tensor-parallel size is greater than one. |

The Python API and server expose the supported text-only
contracts through the same native runtime.

### Media Identity

For media requests, decoded image bytes or PCM audio samples and their
placeholder positions are part of the cache identity. Audio without decoded
PCM uses its mel-spectrogram path. Changing media content or order invalidates
the affected prefix. When a cached prefix contains media tokens, the runtime
offsets suffix embedding indices to account for media tokens already served
from cache.

### Cache Domain

A cache-enabled runtime is one trusted domain. Use separate runtime instances
when requests require tenant isolation; there is no request-level tenant or
salt key.

## Build Capacity

`--max-kv-pool-pages` controls the total page pool. Reserve pages beyond the
active-request minimum when retained contexts must remain resident:

```bash
./build/bin/llm_build \
  --onnx-dir /path/to/onnx/llm \
  --engine-dir /path/to/engine \
  --max-input-len 1024 \
  --max-kv-cache-capacity 4096 \
  --max-kv-pool-pages 64
```

## Enable Reuse

```bash
./build/bin/llm_inference \
  --engine-dir /path/to/engine \
  --input-file input.json \
  --output-file output.json \
  --enable-context-reuse
```

For a VLM, pass the visual engine and a request set containing repeated image
prefixes:

```bash
./build/bin/llm_inference \
  --engine-dir /path/to/engine \
  --multimodal-engine-dir /path/to/visual/engine \
  --input-file tests/test_cases/vlm_context_reuse.json \
  --output-file output.json \
  --enable-context-reuse \
  --profile-output-file profile.json
```

Image content and its position in the prompt are part of the cache identity.
Repeating the same image prefix can hit; changing an image or changing image
order cannot reuse the affected prefix. Reuse remains page-aligned, and the
runtime offsets suffix embedding indices to account for media tokens in the
cached prefix.

The cache retains up to 1024 records by default. Use
`--context-cache-max-records` only when the deployment needs a different limit.

### Snapshots

Attention-only models, such as Llama or Qwen3, do not use snapshots.

Recurrent layers, as in Qwen3.5 or Nemotron-H, keep a running state instead of
a KV cache. To reuse a cached prefix, the runtime saves the model state at the
end of that prefix as a snapshot. Each cached prefix, such as one conversation
turn, takes one snapshot.

The runtime keeps up to `--context-cache-max-snapshots` snapshots in total
(`max_snapshots` in the Python API). When it needs room, it evicts the least
recently used snapshot, whichever conversation it belongs to. Every request
that uses the context cache saves a snapshot, including short side requests
such as title generation or sub-agent calls.

- The default is four times the maximum batch size.
- Change it to keep more or fewer conversations, but keep at least twice the
  maximum batch size. With fewer, a full batch can skip saving some snapshots.
- Raise it for agent workloads that run several conversations or send side
  requests. Otherwise a conversation's snapshot can be evicted before its next
  turn, which then recomputes its whole prompt and fails if that is longer than
  `--max-input-len`.
- The runtime logs each skipped and each evicted snapshot.

Each snapshot takes device memory:

| Model | Memory per snapshot (FP16 KV) |
|---|---|
| Qwen3.5-0.8B | About 20 MiB |
| Qwen3.8-27B | About 156 MiB |

For example, Qwen3.8-27B with `--max-batch-size 4` keeps 16 snapshots by
default, about 2.4 GiB.

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
reusable state.

## Configuring the Encoder Embedding Cache

The encoder embedding cache is separate from KV cache reuse. A runtime with a
ViT or audio runner allows cache entries to use up to 256 MiB of device memory
by default even when `--enable-context-reuse` is not set. It hashes decoded image
bytes or PCM audio samples and retains the corresponding encoder output on the
GPU. A request for identical media can then skip ViT or audio encoder
execution. The cache evicts least-recently-used entries when its budget is
full; an individual entry larger than the budget runs uncached.

Set the C++ runtime budget in bytes with `--encoder-cache-budget-bytes`; zero
disables the cache:

```bash
./build/bin/llm_inference \
  --engine-dir /path/to/engine \
  --multimodal-engine-dir /path/to/encoder/engine \
  --input-file input.json \
  --output-file output.json \
  --encoder-cache-budget-bytes 134217728
```

The Python API and server use the same default but do not expose a
separate encoder-cache budget in this release. An INFO log reports
when all media items hit and encoder execution is skipped. Encoder-cache
hit/miss counters are not included in the profile JSON; the context-cache
profile fields below describe KV/recurrent-state reuse only.

Use runtime profile output to verify reuse. `prefill.reused_tokens` must be
positive for a request that reused cached state. For image requests,
`context_cache.media_aware_sequences` reports how many admitted sequences used
media-aware cache identities.
