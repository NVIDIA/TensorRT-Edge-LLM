# Visual-Token Pruning (DART)

## Overview

Visual-token pruning shortens VLM prefill by removing a configurable fraction of the visual
tokens before the LLM engine runs. VLM inference on edge platforms is prefill-bound — a single
image can contribute hundreds to thousands of tokens, and every one of them is re-computed on
each request. Pruning the redundant ones cuts prefill latency roughly in proportion to the
removed fraction, and slightly speeds up decode as well (shorter KV cache).

The default selection algorithm is **DART** (Duplication-Aware Reduction of Tokens; paper:
[Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters
More](https://arxiv.org/abs/2502.11494)): instead of ranking tokens by importance, it removes
*duplicated* visual tokens, keeping a diverse subset. It is training-free, requires no attention
scores, and needs **no engine or export changes** — pruning operates purely on runtime buffers,
so existing engines work as-is.

**Key Points**:

- Prefill-only, one-shot: assembled input embeddings (and DeepStack planes / mRoPE rows) are
  compacted before the engine executes; the decode path and CUDA graphs are untouched
- Text tokens are never pruned; kept tokens retain their original absolute RoPE positions and
  generation continues as if nothing was removed
- Disabled by default; enable per run with `--visualPrune`
- Measured on Qwen3-VL-4B (FP16, single image, 1995-token prefill):

| Reduction ratio | Accuracy impact | Prefill speedup (Jetson Thor) | Prefill speedup (x86 Blackwell) |
|---|---|---|---|
| 0.25 (default) | ~lossless (MME/MMBench/POPE/GQA/VQAv2) | 1.32× | 1.18× |
| 0.5 | −1…2 pts on sensitive benchmarks (OCR/POPE) | 1.95× | 1.69× |

---

## Usage

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-vl-4b/llm \
  --multimodalEngineDir engines/qwen3-vl-4b/visual \
  --inputFile tests/test_cases/vlm_basic.json \
  --outputFile output.json \
  --visualPrune --dartReductionRatio 0.25
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--visualPrune` | off | Enable visual-token pruning |
| `--visualPruneAlgo` | `dart` | Selection algorithm (custom algorithms register via `rt::registerVisualPruner`) |
| `--dartReductionRatio` | 0.25 | Fraction of visual tokens to remove, in (0, 1) |
| `--dartPivotImageTokens` | 4 | DART: number of image pivot tokens |
| `--dartPivotTextTokens` | 4 | DART: number of text pivot tokens |

### C++ API

```cpp
rt::VisualPrunerConfig config;
config.enabled = true;
config.algorithm = "dart";       // resolved via the pruner registry
config.reductionRatio = 0.25F;
runtime.setVisualPrunerConfig(config);   // before the first request
```

### Profiling

With `--dumpProfile`, pruned tokens are reported separately from KV-cache reuse:

```
=== LLM Prefill ===
Reused Tokens: 0
Computed Tokens: 1005
Pruned Tokens (visual-token pruning): 990
```

(`pruned_tokens` in the profile JSON.)

### Verifying accuracy

`examples/accuracy/scripts/run_dart_accuracy.py` compares accuracy with pruning off vs. one run
per reduction ratio on a multiple-choice VLM dataset (MMStar / MMMU). It reports accuracy only;
for latency numbers use `llm_inference --dumpProfile --warmup N` (warmup matters — cold-start
and GPU-clock effects bias unwarmed single-pass timings):

```bash
python3 examples/accuracy/scripts/prepare_dataset.py --dataset MMStar --output_dir tmp/mmstar_output
python3 examples/accuracy/scripts/run_dart_accuracy.py \
  --engine_dir engines/qwen3-vl-4b/llm \
  --multimodal_engine_dir engines/qwen3-vl-4b/visual \
  --dataset_file tmp/mmstar_output/mmstar_dataset.json \
  --ratios 0.25 0.5
```

---

## How DART selects tokens

Selection runs **independently per image**: each contiguous visual span gets its own retention
quota `ceil(spanTokens × (1 − reductionRatio))` (at least 1), so in multi-image or video
requests no image can be starved by the others. Within each span:

1. **Pivots**: pick the `pivotImageTokens` span tokens with the highest L1 norm of their input
   embeddings, plus `pivotTextTokens` text tokens (shared across spans).
2. **Anti-duplication growth**: per pivot, retain the candidate span tokens with the *most
   negative* cosine similarity to the pivot (i.e. the least duplicated information), removing
   them from the pool, until the span's quota is retained.

Two small CUDA kernels compute the norms and pivot similarities; an exact greedy loop runs on
the host. Total selector overhead is ~1–2 ms per prefill and is included in all numbers above.

> **Note**: the cited paper runs DART after decoder layer 2, scoring pivots with that layer's
> projected key states. Inside a monolithic TensorRT engine those activations are not
> observable, so this implementation is the *embedding-level* variant (validated to be
> near-lossless at ratio 0.25 against layer-2 DART on Qwen3-VL-4B). See the note in
> `cpp/runtime/preprocess/dartPruner.h`.

## Limitations

Pruning is applied only when all of the following hold (it silently no-ops otherwise):

- mRoPE VLM engine (Qwen2.5-VL / Qwen3-VL family) with visual input present
- Batch size 1, fresh KV cache (not combined with system-prompt cache reuse)
- Not combined with speculative decoding, the Omni Talker, or the VLA action runner
- At least `minVisualTokens` (default 16) visual tokens in the request

Quality is dataset-dependent: ratio 0.25 is a safe default; re-validate on your own workload
before raising it (OCR-heavy tasks degrade first).

## Adding a custom pruning algorithm

Subclass `rt::VisualTokenPruner` (`cpp/runtime/preprocess/visualTokenPruner.h`) and register a
factory before constructing the runtime:

```cpp
class MyPruner final : public rt::VisualTokenPruner
{
public:
    using VisualTokenPruner::VisualTokenPruner;
    char const* name() const noexcept override { return "my-algo"; }

protected:
    int32_t prune(rt::PruneRequest const& req, rt::PipelineIO& io, cudaStream_t stream) override
    {
        std::vector<int32_t> retained = /* choose a subset of *req.imagePositions */;
        return compactToKeepList(io, retained, req, stream);  // shared, invariant-safe compaction
    }
};

rt::registerVisualPruner("my-algo",
    [](rt::VisualPrunerConfig const& cfg, rt::LLMEngineConfig const& engineCfg)
    { return std::make_unique<MyPruner>(cfg, engineCfg); });
```

The base class owns the guards and the buffer compaction; a selection-style algorithm only
chooses which visual tokens survive. Select the algorithm at runtime with
`--visualPruneAlgo my-algo` or `VisualPrunerConfig::algorithm`.
