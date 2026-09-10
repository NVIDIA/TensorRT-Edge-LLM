## Test Cases

Small JSON request sets for `llm_inference` smoke and runtime-sanity runs.

### LLM

- `llm_basic.json`
  Basic text-only smoke coverage.
  Recommended engine: vanilla or EAGLE LLM, `maxBatchSize >= 1`, `maxInputLen >= 1024`, `maxKVCacheCapacity >= 4096`.
- `llm_lora.json`
  Text-only LoRA coverage.
  Recommended engine: LLM engine built with the matching LoRA support, `maxBatchSize >= 1`, `maxInputLen >= 1024`.
- `llm_context_reuse.json`
  Three sequential prompts: cold `P+Q`, shorter prefix `P`, then `P+Q` again. The final request must reuse `P` and remain token-identical to the cold request.
  Recommended engine: any context-reuse-supported text LLM, `maxBatchSize >= 1`, `maxInputLen >= 1024`,
  and extra retained base-KV pages for cross-request reuse.
  Hybrid engines additionally require at least one recurrent snapshot slot and, when they contain attention, at least one partial-KV snapshot slot.
  EAGLE engines additionally require extra retained draft-KV pages and greedy sampling; this fixture sets `top_k: 1`.
- `llm_spec_prefill_evict.json`
  Batch-2 speculative regression where slot 0 stops on its first generated token and slot 1 survives managed row
  compaction. The survivor is repeated with an explicit cache-bypass cold control and must remain token-identical.
  Recommended engine: a DFlash/JetSpec engine with `maxBatchSize >= 2` and context reuse enabled.
- `llm_runtime_sanity_check.json`
  Text-runtime sanity coverage for chat templating, system-prompt KV cache, `disable_spec_decode`, and same-process reuse.
  Recommended engine: vanilla or EAGLE LLM, `maxBatchSize >= 1`, `maxInputLen >= 1024`, `maxKVCacheCapacity >= 4096`.
- `llm_diffusion_gemma.json`
  Text-only DiffusionGemma block-diffusion smoke coverage, including the
  request-level `diffusion_config.max_denoising_steps` override. It sets 16
  denoise steps to mirror the default runtime budget used when the request omits
  the override.
  Recommended engine: DiffusionGemma `dllm.engine` built with
  `decoding_strategy: block_diffusion`, `diffusion_unified_conditioning: true`,
  `maxBatchSize >= 1`, and KV capacity large enough for the prompt plus
  `max_generate_length`.

### VLM

- `vlm_basic.json`
  Basic multimodal smoke coverage.
  Recommended engine: VLM LLM + visual engines, `maxBatchSize >= 1`, `maxInputLen >= 2048`, visual token capacity sized for the sample images.
- `vlm_lora.json`
  Multimodal LoRA coverage.
  Recommended engine: VLM LLM + visual engines built with the matching LoRA support, `maxBatchSize >= 1`.
- `vlm_runtime_sanity_check.json`
  VLM runtime sanity coverage for text-only-on-VLM, single-image, multi-image, system-prompt KV cache, `disable_spec_decode`, and request-mode switching.
  Recommended engine: VLM vanilla or EAGLE, `maxBatchSize >= 1`, `maxInputLen >= 8192`, `maxKVCacheCapacity >= 8192`, visual token capacity sized for all referenced images.
  **Requires a visual engine built with a larger image-token shape range than the defaults** — the fixture contains multi-image requests whose combined ViT `cuSeqlens` exceeds the default `--maxImageTokens`. Rebuild the visual engine with a larger `--maxImageTokens` / `--maxImageTokensPerImage` before running this fixture.

### Notes

- These are runtime sanity tests, not semantic accuracy benchmarks.
- The sanity suites check request handling, state isolation, and output-shape sanity.
- The current sanity files use `batch_size: 1`.
