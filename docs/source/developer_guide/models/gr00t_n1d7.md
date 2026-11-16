# GR00T N1.7

GR00T N1.7 is NVIDIA Isaac's flow-matching Vision-Language-Action model. Edge-LLM runs it as three
TensorRT components driven by an experimental C++ policy runtime: camera views, an instruction and
a robot state in, an action chunk in robot units out. See the
[GR00T N1.7 example](../../user_guide/examples/vla/gr00t_n1d7.md) for the build, run and verify
walkthrough; this page records the contracts.

Two build routes write the same bundle. The ONNX-less direct builder
(`tensorrt_edgellm/builder/models/gr00t_n1d7`) compiles the checkpoint on the target;
`tensorrt-edgellm-export` (`tensorrt_edgellm/models/gr00t_n1d7`) writes ONNX on CPU and
`gr00t_policy_build` compiles it. The checkpoint facts, request geometry, component contracts and
constant-table math are one implementation under `tensorrt_edgellm/models/gr00t_n1d7`, imported by
both, so the runtime reads one `config.json` shape and one `policy.json` whichever route produced
them. The ONNX route lowers attention to `trt::attention_onnx` / `trt::rope_onnx`, the path pi0.5's
prefix takes; the cross-attention masks stay bool engine inputs and become additive fp16 inside the
graph.

## Checkpoint

`nvidia/GR00T-N1.7-LIBERO`, the `libero_10/` subdirectory. The checkpoint carries `model_type:
Gr00t` and nests everything under two roots: the backbone under `backbone.model.model.` and the
policy head under `action_head.`. It ships no tokenizer; upstream loads the backbone's
(`Qwen/Qwen3-VL-2B-Instruct`, byte-identical to Cosmos-Reason2-2B's) from the Hub, and the builders
expect the same files under `tokenizer/` in the checkpoint directory.

Its Cosmos-Reason2-2B backbone config lives in a separate, gated repository, so the backbone geometry
is declared in `tensorrt_edgellm/models/gr00t_n1d7/configuration.py` and checked against the
checkpoint's own tensor shapes. A
checkpoint that is not this architecture fails before a network is created rather than building a
silently wrong engine. Cosmos-Reason2-2B is post-trained from Qwen3-VL-2B-Instruct and keeps its
architecture, so the vision tower is the shared Qwen3-VL encoder and needs no new modeling code.

The released checkpoint already carries only `select_layer` (16) of the backbone's 28 text layers;
the reference pops the rest at load time.

## Who runs what

| Stage | Owner | Notes |
|---|---|---|
| checkpoint -> engines, constant tables, policy contract | `tensorrt-edgellm-build`, or `tensorrt-edgellm-export` + `gr00t_policy_build` | all static resolution happens here: embodiment row, ada_norm schedule, normalization bounds |
| observation -> action chunk in robot units | `Gr00tRuntime::infer` | owns the three below and binds them in order; one call per observation |
| camera views + instruction -> VL context | `Gr00tVisualRunner` | eval transform, chat template, tokenization, embedding gather, image and DeepStack scatter |
| packed patches -> image features + DeepStack | `visual` engine, driven by the shared `Qwen3VLViTRunner` | reuses the shared Qwen3-VL tower and `ViTAttentionPlugin` unchanged |
| VL context -> cross-attention K/V | `vl_backbone` engine, `Gr00tPolicyRunner::prefill` | runs once; its outputs stay bound for the whole denoise loop |
| one denoise step -> velocity | `action` engine, `Gr00tPolicyRunner::generate` | 4 enqueues plus a device Euler kernel, captured as one CUDA graph |
| robot state <-> normalized space | `Gr00tPolicy` | pure CPU; no CUDA, no engine |

The runner owns engine execution and buffer lifetime; the policy owns robot
semantics and never touches a stream. Nothing in either resolves checkpoint
structure at run time -- that is the builder's job, and a mismatch fails at load.

## Components

### visual

The shared Qwen3-VL vision tower (SigLIP2-Large, 24 blocks, hidden 1024) plus its patch merger and
three DeepStack mergers. The contract is the existing Qwen3-VL packed-patch contract, unchanged.

| | Name | Type | Shape |
|---|---|---|---|
| in | `input` | fp16 | `[patches, 1536]` |
| in | `rotary_pos_emb`, `cu_seqlens`, `max_seqlen_carrier`, `fast_pos_embed_idx`, `fast_pos_embed_weight` | | Qwen3-VL contract |
| out | `output` | fp16 | `[patches / 4, 2048]` |
| out | `deepstack_features_0..2` | fp16 | `[patches / 4, 2048]` |

### vl_backbone

One prefill of the truncated Cosmos-Reason2-2B decoder (causal, GQA 16/8, head_dim 128, per-head
q/k RMSNorm), its VL LayerNorm and the 4-layer VL self-attention refinement, followed by the DiT's
cross-attention K/V projections.

| | Name | Type | Shape |
|---|---|---|---|
| in | `inputs_embeds` | fp16 | `[B, S, 2048]` |
| in | `rope_rotary_cos_sin` | fp32 | `[B, S, 128]`, packed `(cos|sin)` halves |
| in | `attention_pos_id` | int32 | `[B, S]` |
| in | `deepstack_embeds_0..2` | fp16 | `[B, S, 2048]` |
| out | `cross_k_layerNN` / `cross_v_layerNN` | fp16 | `[B, S, 32, 48]`, even blocks only |

Two invariants the graph relies on and cannot check:

1. **DeepStack inputs are zero outside image-token positions.** Qwen3-VL scatter-adds its DeepStack
   features at image tokens; the graph adds the whole tensor, which is the same thing only if the
   caller zero-fills the rest. This matches the shared `DecoderModel` contract.
2. **`rope_rotary_cos_sin` is in token order and `attention_pos_id` indexes it.** mRoPE is a host-side
   property of the VL layout, so the engine only ever sees per-token cos/sin. At `B > 1` the position
   ids must be offset by `b * S`, because the cache is flattened before the gather.

### action

One flow-matching denoise step. The graph emits the velocity; the runtime owns the Euler update, so
the step count stays a runtime setting and the loop captures as one CUDA graph.

| | Name | Type | Shape |
|---|---|---|---|
| in | `noise_trajectory` | fp32 | `[B, 40, 132]` |
| in | `state` | fp32 | `[B, 1, 132]` |
| in | `adanorm_modulation` | fp16 | `[33, 3072]`, one row per site for this step |
| in | `action_time_bias` | fp16 | `[1536]` |
| in | `text_attention_mask` / `image_attention_mask` | bool | `[B, 1, 1, S]` |
| in | `cross_k_layerNN` / `cross_v_layerNN` | fp16 | `[B, S, 32, 48]` |
| out | `velocity` | fp32 | `[B, 40, 132]` |

The 32 DiT blocks alternate: odd blocks self-attend over the 41 state+action tokens, even blocks
cross-attend into the VL context. `attend_text_every_n_blocks` is 2, so a cross block reads the text
mask when `idx % 4 == 0` and the image mask otherwise. The masks are arbitrary booleans over the VL
context, not contiguous spans, so they are engine inputs rather than a slicing convention.

## Build-time constants

`num_inference_timesteps` is fixed by the checkpoint, so everything that depends only on the timestep
is evaluated by the builder and bound per step instead of being rebuilt by the action graph.

| artifact | shape | replaces |
|---|---|---|
| `adanorm_modulation.safetensors` | `[steps, 33, 3072]` fp16 | 33 `[1536, 3072]` AdaLayerNorm projections (311 MB fp16) re-read every step |
| `action_time_bias.safetensors` | `[steps, 1536]` fp16 | the timestep half of the action encoder's `W2`, plus the concatenation |

Row `i` is block `i`'s `norm1.linear` output and the last row is `proj_out_1`. The two split their
output in opposite orders — `(scale, shift)` for the blocks, `(shift, scale)` for `proj_out_1` — so
the raw projection is stored and the graph owns the split.

The embodiment id is also fixed at build time, so `CategorySpecificLinear`'s
`W [32, in, out]` gather is resolved in the builder. It is read from the checkpoint's
`embodiment_id.json` by matching its tags against the checkpoint directory name, which selects
`libero_sim` (id 2) for `libero_10`; `--gr00t-embodiment-id` overrides it and an unmatched
directory falls back to `unknown` with a warning. That drops 31/32 of the weights on the state
encoder, action encoder and action decoder; the action encoder alone goes from ~466 MB stacked to
~15 MB, and it runs every denoise step.

An embodiment is a slot inside the checkpoint rather than a checkpoint of its own: the backbone is
shared and only those three modules carry per-embodiment rows. `libero_10` names 52 tags over 17
ids, of which 9 carry both statistics and a modality config; eight of those nine declare
`rep: RELATIVE, type: EEF` actions, so `libero_sim` is the only one whose action representation this
runtime implements. An engine bundle is therefore built for one embodiment, and another needs
another build.

## Policy contract

The builder resolves the embodiment's statistics and joint-group order into
`action/policy.json`, so the runtime needs two flat vectors and one slice width
instead of the checkpoint's nested processor config.

| field | LIBERO value |
|---|---|
| `embodiment_tag` / `embodiment_id` | `libero_sim` / 2 |
| `state.keys` / `dims` | `x y z roll pitch yaw gripper` / `1 1 1 1 1 1 2` (8 wide, padded to 132) |
| `action.keys` / `dims` | same keys / all 1 (7 wide) |
| `action.horizon` | 16 -- `len(delta_indices)`, shorter than the model's 40 |
| bounds | `q01` / `q99`, since `use_percentiles` is true |
| `export_id` / `frontend` / `checkpoint_fingerprint` | the build's identity, also stamped on every component `config.json` |

`export_id` hashes the checkpoint's weight fingerprint with the options that change the engines
(embodiment id, context and image lengths, route), so rebuilding one component from the same
inputs reproduces it. Both builders refuse to write beside a component carrying another id, and
the runtime refuses a bundle whose components disagree on id or route, or carry none. They also
refuse a checkpoint without `tokenizer/`, its `tokenizer.json` and a chat template before any
engine is written, since the runtime loads all three.

`Gr00tPolicy` applies the reference's own formulas: normalization is
`2 * (x - q01) / (q99 - q01) - 1` clipped to `[-1, 1]`, with a degenerate joint
group mapping to zero rather than dividing by its range; unnormalization is
`(clip(x, -1, 1) + 1) / 2 * (q99 - q01) + q01` and has no such guard.

Only absolute action representations are supported. The builder rejects a
checkpoint whose `action_configs` ask for relative-to-absolute conversion, which
would need end-effector pose math the runtime does not implement. LIBERO is
absolute throughout.

## Precision

**FP16 only.** The graphs are built from a bf16 checkpoint with no quantization metadata, so
`Net.linear_from_weights` takes its FP16 path throughout.

Measured on Thor SM110, FP16, batch 1, 4 denoise steps, against a float32 PyTorch reference
assembled from Isaac-GR00T's own `dit.py` and HuggingFace `Qwen3VLForConditionalGeneration`. The
policy head is driven with identical embeddings, DeepStack features, rotary tables, masks, state and
initial noise; the vision front end is scored on two LIBERO views and a task instruction, against the
same checkpoint's tower behind HuggingFace's Qwen3-VL processor.

| | |
|---|---|
| token ids | identical, 148 tokens |
| `inputs_embeds` cosine | 0.99985 (image rows), 1.0 (text rows) |
| DeepStack cosine | 0.99987 - 0.99999 |
| backbone cross K/V vs fp32, golden inputs | cosine >= 0.99997, all builds |
| per-step velocity cosine, real observation | >= 0.999998, all builds |
| per-step velocity cosine, golden seeds 0 and 1 | >= 0.99986 |
| end to end, real observation and reference `x_0`, vs fp32 chunk | cosine 0.9999999, max abs 3.3e-3 - 6.8e-3 |
| of which the vision front end alone, fp32 downstream | 4.2e-3 |
| `visual` / `vl_backbone` / `action` | 11.1 / 11.5 / 31.3 ms (GPU event spans) |
| end to end, one real observation | 54.9 ms direct / 53.9 ms ONNX-route bundle; 53.0 - 55.2 across all builds |

The engine figures cover several builds of both routes; two direct builds differ
by 1.5 ms end to end from TensorRT's tactic choices, since there is no timing cache. Synthetic golden
seed 2 is left out of the table: its step 3 amplifies any fp16-level backbone deviation a hundred to a
thousand times, so builds land anywhere between 0.9913 and 0.9996 there and white noise at the same
rms scores 0.9992 to 0.9999; it probes sensitivity, not accuracy. The two routes sit equally close to
the reference and their action engines are interchangeable. The `visual`
stage covers the whole front end,
not just the tower: on the same two views the tower alone is 6.1 ms, and the rest is the eval
transform, the chat template and tokenization, all on the host, so the engines alone run the same
context in 48.8 ms. The per-step velocity cosine is the accuracy figure that matters: the action
chunk is dominated by its initial noise, so its cosine stays near 1 even when the velocity is badly
wrong.
