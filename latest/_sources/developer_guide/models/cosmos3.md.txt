# Cosmos3-Edge Design

[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) supports two
tasks with separate component and runtime contracts:

| Task | Input to output | Components | Runtime |
|---|---|---|---|
| Policy | observation and instruction to action chunk | UND prefill, generation policy, VAE encoder | `cosmos3_policy_inference` |
| Reasoning | image and prompt to text | visual encoder, LLM | `llm_inference` |

See the [Cosmos3-Edge example](../../user_guide/examples/vla/cosmos3.md) for
export, engine build, and inference commands.

## Policy Components

Policy export writes:

```text
onnx/
  und_prefill/ model.onnx config.json embed_tokens.safetensors
  gen/         model.onnx config.json
  vae_encoder/ model.onnx config.json
  text_tokenizer/ tokenizer.json processed_chat_template.json ...
```

Each `config.json` is the component contract (optimization profile, tensor
shapes, and runtime constants) consumed by the component builder and runtime.
The runtime stages the tokenizer and embeddings once, then executes VAE encode,
UND prefill, and the generation denoise loop. The JSON action shape is
`[batch, chunk, action_dimension]`.

The UND graph has no attention-mask input, so prompts in a batch must tokenize
to the same length. The image observation is resized to `736x544` and shared
across the batch.

Cosmos3-Edge sets `use_und_k_norm_for_gen=True`. UND K is RMSNorm-normalized
per head before RoPE for generation cross-attention. Text self-attention keeps
the raw K because `qk_norm_for_text=False`.

## Reasoner Components

The reasoner is a standard Edge-LLM VLM with a SigLIP2 vision encoder,
PatchMerger and an autoregressive text decoder. The only model-specific pieces
are on the export side: the decoder swaps SwiGLU for the Nemotron-H
squared-ReLU MLP (`Cosmos3ReasonerCausalLM`), and the exporter maps the
checkpoint's native flat schema onto the shared decoder names.

Image and video synthesis are not exposed.

## RoboLab Integration

`experimental_models/cosmos3/robolab/policy_server.py` exposes the policy JSON
action contract over HTTP by wrapping `cosmos3_policy_inference`;
`cosmos3_client.py` turns simulator observations into action chunks. This
integration is optional and lives outside the core Python package.

```bash
python -m experimental_models.cosmos3.robolab.policy_server \
    --binary ./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
    --engine-dir "$ENGINE_DIR" \
    --host 0.0.0.0 --port 8080
```
