# Vision-Language-Action

Vision-language-action models use model-specific action contracts rather than
the standard text-generation output. Choose the guide for the checkpoint and
runtime:

| Model | Input | Output | Executable |
|---|---|---|---|
| [Alpamayo-R1](alpamayo.md) | camera frames, instruction, past trajectory | future acceleration/curvature trajectory | `action_inference` |
| [Cosmos3-Edge policy](cosmos3.md) | observation image or frame list, instruction | robot action chunk | `cosmos3_policy_inference` |
| [pi0.5](pi05.md) | camera views, instruction | robot action chunk | `pi05_policy_inference` |

Each workflow exports on CPU, builds all required TensorRT engines on the target, and invokes one
end-to-end runtime executable.

```{toctree}
:maxdepth: 1
:hidden:

alpamayo.md
cosmos3.md
pi05.md
```
