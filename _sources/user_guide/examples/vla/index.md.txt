# Vision-Language-Action

Vision-language-action models use model-specific action contracts rather than
the standard text-generation output. Choose the guide for the checkpoint and
runtime:

| Model | Input | Output | Executable |
|---|---|---|---|
| [Alpamayo-R1](alpamayo.md) | camera frames, instruction, past trajectory | future acceleration/curvature trajectory | `action_inference` |
| [Cosmos3-Edge policy](cosmos3.md) | observation image or frame list, instruction | robot action chunk | `cosmos3_policy_inference` |

Both workflows export on CPU, build all required TensorRT engines on the target,
and invoke one end-to-end runtime executable.

```{toctree}
:maxdepth: 1
:hidden:

alpamayo.md
cosmos3.md
```
