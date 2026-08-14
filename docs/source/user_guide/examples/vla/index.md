# Vision-Language-Action

Vision-language-action models use model-specific action contracts rather than
the standard text-generation output. Choose the guide for the checkpoint and
runtime:

| Model | Input | Output | Executable |
|---|---|---|---|
| [Alpamayo-R1](alpamayo.md) | camera frames, instruction, past trajectory | future acceleration/curvature trajectory | `action_inference` |
| [Cosmos3-Edge policy](cosmos3.md) | observation image or frame list, instruction | robot action chunk | `cosmos3_policy_inference` |
| [InternVLA-N1-DualVLN](internvla_n1.md) | navigation frames, instruction | future waypoint trajectory | `internvla_n1_system1_inference` |

These workflows export on CPU, build all required TensorRT engines on the target,
and invoke one end-to-end runtime executable. InternVLA-N1 is the exception in one
respect: its two systems run at different rates, so the planner and the trajectory
head are driven separately rather than by a single executable.

```{toctree}
:maxdepth: 1
:hidden:

alpamayo.md
cosmos3.md
internvla_n1.md
```
