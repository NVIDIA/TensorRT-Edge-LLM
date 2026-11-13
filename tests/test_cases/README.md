## Test Cases

Small JSON request sets for `llm_inference` smoke and runtime-sanity runs, server stress testing, and the
in-flight-batching traces listed under [IFB traces](#ifb-traces).

### LLM

- `llm_basic.json`
  Basic text-only smoke coverage. Greedy (`top_k: 1`), so accuracy gates are deterministic.
- `llm_basic_sampled.json`
  The `llm_basic.json` request with `top_k: 50`, for DSpark's lossless sampling verifier.
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
- `llm_sliding_window.json`
  Deterministic text request whose prompt exceeds Gemma4's 1024-token sliding window.
  Recommended engine: Gemma4 FP16 with context reuse disabled, `maxInputLen >= 1536`, and
  `maxKVCacheCapacity >= 4096` so the bounded `2M+1` reservation is smaller than full storage.
- `llm_spec_prefill_evict.json`
  Batch-2 speculative regression where slot 0 stops on its first generated token and slot 1 survives managed row
  compaction. The survivor is repeated with an explicit cache-bypass cold control and must remain token-identical.
  Recommended engine: a DFlash/JetSpec engine with `maxBatchSize >= 2` and context reuse enabled.
- `ragged_batch.json`
  Four greedy requests whose prompts differ in length, issued as one batch. The default
  `--input-file` for `scripts/few-layer-validation.sh`: a batch-1 fixture cannot catch a
  per-slot indexing fault in the engine's layer dump.
  Recommended engine: any text LLM with `maxBatchSize >= 4`.
- `llm_few_layer.json`
  Single short greedy request for few-layer numeric validation on checkpoints too large to
  run four sequences against a PyTorch golden.
  Recommended engine: any text LLM, `maxBatchSize >= 1`.
- `llm_runtime_sanity_check.json`
  Text-runtime sanity coverage for chat templating, `disable_spec_decode`, and same-process reuse.
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
  Basic multimodal smoke coverage. Greedy (`top_k: 1`), so accuracy gates are deterministic.
  Recommended engine: VLM LLM + visual engines, `maxBatchSize >= 1`, `maxInputLen >= 2048`, visual token capacity sized for the sample images.
- `vlm_lora.json`
  Multimodal LoRA coverage.
  Recommended engine: VLM LLM + visual engines built with the matching LoRA support, `maxBatchSize >= 1`.
- `vlm_runtime_sanity_check.json`
  VLM runtime sanity coverage for text-only-on-VLM, single-image, multi-image, `disable_spec_decode`, and request-mode switching.
  Recommended engine: VLM vanilla or EAGLE, `maxBatchSize >= 1`, `maxInputLen >= 8192`, `maxKVCacheCapacity >= 8192`, visual token capacity sized for all referenced images.
  **Requires a visual engine built with a larger image-token shape range than the defaults** — the fixture contains multi-image requests whose combined ViT `cuSeqlens` exceeds the default `--maxImageTokens`. Rebuild the visual engine with a larger `--maxImageTokens` / `--maxImageTokensPerImage` before running this fixture.

### Server stress setup

- [llm_server_stress.json](llm_server_stress.json)
  Complete pytest setup for Qwen/Qwen3.5-0.8B: launch command, engine limits,
  prompts, concurrency 1/4/16, overload concurrency 64, a 600-second soak,
  cancellation, and startup/busy shutdown checks. Uses the active Python
  environment and a repository-relative engine cache.
  Run from the repository root after building the runtime and installing dependencies:

  ```bash
  LLM_SDK_DIR="$PWD" python -m pytest tests/server_stress \
    --server-stress-config=tests/test_cases/llm_server_stress.json -v
  ```

  This fixture uses the stress harness settings schema. See the
  [stress testing guide](../server_stress/README.md) for environment setup,
  customization, and result interpretation. Execution is opt-in; L0 selection
  stays unchanged.

### Runtime request notes

- These are runtime sanity tests, not semantic accuracy benchmarks.
- The sanity suites check request handling, state isolation, and output-shape sanity.
- The current sanity files use `batch_size: 1`.

### IFB traces

The `ifb_*.json` files are request traces for `tests/ifb/run_ifb_trace.py` (schema in
`tests/ifb/README.md`), not `llm_inference` inputs: they drive the nanobind `RequestEngine`
with token-count-keyed arrivals and cancels, and `tests/ifb/ifb_test.sh` compares each run with a
batch-1 reference. Prompts are sized for an engine with `maxBatchSize >= 4`, `maxInputLen = 1024` and
`maxKVCacheCapacity >= 4096`.

- `ifb_smoke.json`: quick tier; one-step prefill, a chunked prompt arriving during decode, cancels,
  slot reuse.
- `ifb_priority.json`: `priority` scheduling with `pause`/`none` preemption and a reserved slot, and an
  urgent request with its own `max_generate_length` that must start beside the running requests.
- `ifb_context_cache.json`: context cache enabled; prefix hits during decode, multi-turn reuse, a
  prompt cancelled between chunks.
- `ifb_multimodal.json`: image and text requests beside text-only ones; needs a VLM engine (an
  engine directory with `llm/` and `visual/`).

In CI, `ifb_smoke.json`, `ifb_priority.json` and `ifb_context_cache.json` run in `mixed` mode as `ifb_trace`
scenarios of the Qwen3-0.6B and Qwen3.5-0.8B FP16 model cases on `l0_e2e_b100` (see
`tests/model_configs/README.md`). `ifb_multimodal.json` and the other modes run by hand through
`tests/ifb/ifb_test.sh --tier full`, whose reference runs use `ifb_scheduling_mode=singleCohort`;
see `tests/ifb/README.md` for the coverage table and the commands.
