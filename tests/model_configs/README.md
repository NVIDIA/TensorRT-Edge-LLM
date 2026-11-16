# Model-owned L0 tests

Use this directory to add or change a model deployment tested by L0. Each base
model has one file at `l0/<model-family>/<base-model>.yml`; each named case
selects a checkpoint, target job, engine settings, and inference scenarios.

## Add or modify a case

Before adding a case, identify the behavior it must prove and check whether
an existing case already covers it. Choose a workflow that exercises the
changed code and outputs or metrics that demonstrate success. Follow the
[testing skill](../../skills/edge-llm-e2e-test/SKILL.md) for test selection
and execution evidence.

Start from a checked-in case under `l0/` that uses the same model type and
target hardware. Add to the model's existing file, or create a file for a new
base model. Give each case and each inference entry a unique name within its
list.

This example builds a text model directly from its checkpoint and checks
accuracy and performance:

```yaml
schema_version: 1
model_id: Qwen/Qwen3-0.6B
inputs: [text]
outputs: [text]
cases:
  - name: fp16-direct
    model_name: Qwen3-0.6B
    mode: direct_builder
    platform: l0_e2e_a30
    quantized_model_id_or_dir: Qwen/Qwen3-0.6B
    requirements:
      min_sm: 80
      min_gpu_memory_gb: 5
    engine:
      max_kv_cache_capacity: 2048
      max_input_len: 1024
      max_batch_size: 1
    inference:
      - name: basic
        scenario: generate
        input_file: tests/test_cases/llm_basic.json
        checks:
          accuracy:
            required: [rouge1, rouge_l]
            max_regression: 0.30
          performance:
            required: [prefill_tokens_per_second]
            max_regression: 0.30
```

- `model_id` identifies the provider's base model. `quantized_model_id_or_dir` selects the checkpoint to test: a full Hugging Face ID, absolute path, or recognized [shared-root reference](../defs/shared_roots.py), such as `${LLM_MODELS_DIR}/Qwen/Qwen3-0.6B`.
- Checkpoint-name suffixes such as `INT4-AWQ` and `NVFP4` determine precision; a name without a precision suffix means FP16.
- `inputs` and `outputs` declare model capabilities. A case can narrow them, but cannot add capabilities.
- `requirements` declares minimum GPU capability (`min_sm`) and memory in GiB. Choose these values from the model's resource needs and target hardware; successful collection does not verify that the model fits.
- Put build settings under `engine`, ONNX exporter settings under `export`, and at least one scenario under `inference`. Reuse input files from [test_cases](../test_cases/). For generation, declare the metrics that establish success under `checks`; choose tolerances appropriate to the behavior and measured variability. The example's `0.30` is not a universal recommendation. Guided-decoding fixtures (`llm_guided_decoding*.json`) are checked for grammar conformance, and the jump-forward fixture for a jump, so their scenarios may omit `checks`.
- `engine.spec_verify_tokens` is the runtime verify count of a speculative case (`--specVerifyTokens`); it defaults to `max_spec_verify_tokens`, and EAGLE jump-forward needs it below that capacity.

## Choose what runs and where

| `mode` | Execution | Available scenarios |
|---|---|---|
| `onnx` | Export, then engine build and inference | `generate`, `runtime_bench`, `compact_engine_tests`, `ifb_trace`, `media_ipc_server` |
| `direct_builder` | Build and run from a checkpoint | `generate`, `few_layer` |
| `server` | Start a server and test requests or evaluation | `openai` and/or `ifb_server`, or a separate `nemo_evaluator` case |

`few_layer` uses its own driver: it defaults to ONNX even in a `direct_builder`
case. Set `arguments: [--direct-builder]` to choose checkpoint-direct building.
Few-layer tests check numerical agreement; NeMo evaluation checks a declared
score threshold. Runtime benchmarks check report completeness, and compact-engine
tests require the selected engine tests to run and pass. Those checks do not
replace generation accuracy/performance checks.

`ifb_trace` checks mixed in-flight batching on an ONNX text-model engine;
`test_case` selects `ifb_smoke` (the default), `ifb_priority`, or
`ifb_context_cache`. `ifb_server` checks server scheduling and requires
`engine.in_flight_batching: true`. Both require a text-model workflow without
speculative decoding. Use the existing cases and the [IFB guide](../ifb/README.md)
for engine sizes, hardware requirements, and correctness criteria.

`media_ipc_server` runs the [media IPC server example](../../examples/media_ipc_server/README.md)
on an ONNX image, audio, or omni engine without speculative decoding. The job
must set `BUILD_MEDIA_IPC_SERVER=ON`, and the inputs come from
`${EDGELLM_DATA_DIR}/test_media/ipc_example`.

`pipeline: cosmos3_policy` routes an ONNX case to the Cosmos3 policy tests.
Export, engine build, and inference run in the case's own job, so `platform`
must be `l0_e2e_a30`, and `engine` may set only `max_batch_size`. Each
`inference` entry selects `test_case: cosmos3_droid` or `cosmos3_droid_lite`,
which compares every sample's actions with the dataset's PyTorch reference.
These tests apply their own pass criteria, so omit `checks`.

`multistage_parity: true` on a `direct_builder` `generate` entry of an LLM,
VLM, ASR, TTS, or Omni case also runs `multistage_inference` on the graph the
build writes, with the same input file, and requires the existing binary's
results: the same text, and for speech, which both binaries then vocode in one
pass, a talker stopped at EOS and every 16-bit sample within 1 of the existing
waveform, plus the same RVQ codes for TTS. A speech case then reruns
`multistage_inference` with its default 2-second vocoder windows and compares
that run with its whole-utterance one: the same RVQ codes, sample count and EOS
stop, every 2-second span either run makes audible correlating at least 0.5 at
0.5 to 2 times the whole-utterance RMS, and at least one request longer than 2
seconds. Give a parity input one request per batch, since the two binaries
batch differently; an Omni case must set `max_batch_size: 1`. Speculative
decoding and tensor parallelism are rejected.

`platform` selects an existing CI priority, such as `l0_e2e_a30`. Copy the
priority from a case targeting the same hardware. The supported names are in
[`Platform`](../defs/model_config_schema/document.py); ONNX export jobs are
assigned separately by the [loader](../defs/model_test_config.py).

## Check collection and run

From the repository root, follow the [testing setup](../README.md#quick-start).
Collect your target priority to validate YAML and inspect the generated tests:

```bash
python3 -m pytest tests --priority=l0_e2e_a30 --collect-only -q
```

Collection does not run the model. Check that your case appears in the expected
job. For ONNX cases, also collect the export priority assigned by the loader.
Validation errors identify the source file or custom MR block and the invalid
field or combination.

Run the example case by name. Include `build_project` on the first run or after
changing C++, CUDA, or the build system:

```bash
python3 -m pytest tests --priority=l0_e2e_a30 \
  -k "build_project or fp16-direct" -v
```

For an existing build, use `-k "fp16-direct"`. ONNX cases need their export
phase completed before target build and inference. Check that the expected
phases ran and were not skipped; logs are written under `logs/`.
Direct-generation cases also collect a weightless variant, which skips unless
`ENABLE_EXPERIMENTAL_WEIGHTLESS=ON`. It requires compatible refit libraries
and uses the ordinary variant's baseline identity.

### Establish a generation baseline

Generation cases with declared metric checks compare results with
`BASELINE_CSV`, defaulting to `logs/baseline.csv`. Required metrics must be
produced. Missing baseline files, entries, or reference metrics skip the
corresponding comparison and record current values in the test log as
`BASELINE_COMPARISON_DATA`. Available references still detect regressions.
For a new case, enable seeding explicitly:

```bash
BASELINE_CSV=logs/baseline.csv BASELINE_AUTOSAVE=1 \
  python3 -m pytest tests --priority=l0_e2e_a30 \
  -k "build_project or fp16-direct" -v
```

Seeding fills absent entries and blank cells after successful checks, preserving
existing values. Concurrent writers serialize updates. Write errors are logged;
the original measurements remain in the test log. Review the saved reference,
then rerun with automatic updates disabled:

```bash
BASELINE_CSV=logs/baseline.csv BASELINE_AUTOSAVE=0 \
  python3 -m pytest tests --priority=l0_e2e_a30 -k "fp16-direct" -v
```

The first run creates a reference; it does not prove a regression comparison.
The second must run and pass the selected case. Do not widen tolerances to
hide a known regression.

## Custom MR L0 run

At the beginning of the merge request's `## Summary`, place this marker followed
by one fenced YAML block containing a complete model document, such as the
example above. Replace the comment below with that document:

````markdown
<!-- EDGE_LLM_L0_TESTS -->
```yaml
# Paste the complete model document here.
```
````

Choose distinct case names: custom cases are added to checked-in cases, and
regular L0 coverage still runs. Each case's `platform` selects its existing job.
The block accepts model configurations, not commands, images, or runner settings.
Keep it near the start of the Summary so CI receives the complete block.
Custom MR cases do not require a recorded metric baseline; if an entry exists,
it is still compared. Required metrics must still be produced.

## Field definitions

The Pydantic models define allowed fields, types, and combinations:

- [Document and case fields](../defs/model_config_schema/document.py)
- [Engine, export, and hardware requirements](../defs/model_config_schema/deployment.py)
- [Inference scenarios and metric checks](../defs/model_config_schema/inference.py)

When adding a new field, update its owning model and check that the selected
executor consumes it. The [loader](../defs/model_test_config.py) resolves paths
and turns validated cases into executable tests.
