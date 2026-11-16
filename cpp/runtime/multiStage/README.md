# multiStage

`MultiStageConfig` lists a deployment's stages and which output feeds which input;
`MultiStageRuntime` runs any such config and knows no model family. A stage is a class behind
`IStage`, built by name from a runner table, so a new model is a config plus, at most, a new runner. See
the [multi-stage graph guide](../../../docs/source/developer_guide/software-design/multi-stage-graph.md)
for usage and limits.

## A minimal config

```json
{"schema_version": 1,
 "stages": [
   {"id": "encoder", "runner": "demo.encoder", "kind": "engine",
    "inputs": [{"name": "wav", "from": "request:audio"}],
    "emits": [{"id": "embeds", "kind": "embeds"}]},
   {"id": "decoder", "runner": "demo.decoder", "kind": "autoregressive",
    "inputs": [{"name": "embeds", "from": "encoder:embeds"}],
    "emits": [{"id": "text", "kind": "text", "transport": "chunk_stream", "terminal": true}]}],
 "terminals": ["decoder:text"]}
```

`MultiStageConfig::load(root)` reads `<root>/multistage_config.json`, and every `artifact` is a
directory under `root`. `terminals` lists port ids of kind `text`, `audio`, or `image`, exactly the
ports flagged `terminal`.

## Adding a stage

1. **Write the runner.** Implement `IStage`. Allocate every device buffer in `preallocate()`; the
   `step()` and `stepCohort()` path must not allocate device or pinned memory. Check
   `ctx.slot->abandon` at each step, and emit the terminating chunk from `finish()`, which runs on
   success, failure, and cancellation. A throw from `prepare()` fails only that request, so
   `prepare()` must not write state that another request reads. To serve more than one request,
   key per-request state by `ctx.slot->batchSlot`, override `stepCohort()`, and refuse in
   `admit()` a request that cannot share the cohort's settings.
2. **Declare it.** Next to its factory, return its `RunnerDecl`: the runner name, the
   `StageKind`, the ports it can emit, the inputs it accepts, its required config keys, `batches`
   when it overrides `stepCohort()`, and, when it reads the `sampling` or `style` input, the keys
   a request file may set for it. Add the declaration and the factory as one row of
   `builtinRunners()` in `stageFactories.cpp`; that reference is also what keeps the runner's
   object file in a static link.
3. **Connect it.** Emit the stage from the family's `multistage_topology` hook, with each
   `inputs[].from` naming `<stageId>:<portId>` or `request:<field>`, and add the runner to
   `RUNNER_KINDS`, and to `NEEDS_SEQUENCE_BUDGET` if it needs `max_seq`, in
   `tensorrt_edgellm/builder/core/artifacts/multistage.py`. `MultiStageRuntime::load()` checks the
   structure with `validate()`, then the entry against the runner's declaration and the edges
   against the lanes with `checkLoadable()`, which `multistage_inference --validate-config` also
   runs.

## `on_missing`

| Value | When the input's source is missing for a request |
|---|---|
| `fail` (default) | The stage leaves the route, its consumers apply their own `on_missing`, and the request fails if a requested terminal needs the stage. |
| `skip` | The stage runs without the input: `ctx.in->find(name)` returns `nullptr`, not an empty tensor. |
| `prune` | Same as `fail`; the name marks a stage that has nothing to compute without the source. |

Keep every stage on `stream_lane` 0: `MultiStageRuntime::load()` rejects edges between lanes.
