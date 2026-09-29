# Supported Models

Supported checkpoint IDs are listed below. See the
[support matrix](support-matrix.md) for platform coverage and the
[examples](../examples/index.md) for workflows.

## Support Policy

TensorRT Edge-LLM supports the checkpoint IDs listed below. Dense LLM families include official dense checkpoints below 30B parameters. Larger dense checkpoints and non-dense variants require case-by-case validation. MoE, multimodal, audio, TTS, omni, EAGLE3, DFlash, and JetSpec support is limited to the listed rows.

The model coverage list is not comprehensive, and not every listed checkpoint has been fully verified on every supported platform and precision. If a listed model does not export, build, or run correctly, please report an issue with the checkpoint ID, precision, platform, and command line used.

The model class names were checked against the upstream [Transformers model source tree](https://github.com/huggingface/transformers/tree/main/src/transformers/models). Checkpoint IDs are linked to their Hugging Face pages and grouped into original checkpoints and quantized checkpoints. All links below are public or gated; run `hf auth login` after accepting the provider's terms for a gated checkpoint.

## Precision Notes

- Dense precision set: FP16/BF16 checkpoints, ModelOpt FP8/MXFP8/FP4/NVFP4/INT4 AWQ/INT8 SmoothQuant checkpoints, and INT4 GPTQ checkpoints. INT8 GPTQ is not supported.
- Jetson Orin supports FP16, INT8, and INT4 runtime precision in the supported JetPack configurations. Do not select FP8, MXFP8, FP4, or NVFP4 checkpoints for Orin.
- For INT4 engine builds on Jetson Orin devices with less system memory, such as Jetson Orin Nano, pass `--externalize-weights int4_ffn` for dense checkpoints or `--externalize-weights int4_ffn int4_moe` for MoE checkpoints to reduce engine build memory.
- For FP16/BF16 source checkpoints, use the [Quantization](../features/quantization.md) script to create a unified quantized checkpoint for `tensorrt_edgellm`, then export the generated checkpoint.
- FP8 KV cache is detected automatically from checkpoint metadata by `tensorrt_edgellm`.
- `tensorrt-edgellm-export` exports visual encoders. Use `tensorrt-edgellm-quantize llm --visual_quantization fp8` before export when FP8 visual weights are required.
- MXFP8 and FP4/NVFP4 require Blackwell-class hardware for runtime execution.
- For platform-specific NVFP4 MoE layouts, follow [Export NVFP4 MoE for SM12x](../features/quantization.md#export-nvfp4-moe-for-sm12x) before building the engine.

## Text Generation

<details>
<summary><b>Llama 3.x</b></summary>

`model_type: llama`; text input to text output.

**Original:**

- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

**Pre-quantized:**

- [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- [nvidia/Llama-3.1-8B-Instruct-NVFP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4)

</details>

<details>
<summary><b>Qwen2 / Qwen2.5</b></summary>

`model_type: qwen2`; text input to text output.

**Original:**

- [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B), [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B), [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B), [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Qwen/Qwen2-Math-1.5B](https://huggingface.co/Qwen/Qwen2-Math-1.5B), [Qwen/Qwen2-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-1.5B-Instruct)
- [Qwen/Qwen2-Math-7B](https://huggingface.co/Qwen/Qwen2-Math-7B), [Qwen/Qwen2-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct)
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B), [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B), [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B), [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B), [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B), [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Qwen/Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B), [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
- [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B), [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

**Pre-quantized:**

- [Qwen/Qwen2-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-AWQ), [Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-AWQ), [Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-7B-Instruct-AWQ), [Qwen/Qwen2-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ), [Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ), [Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ), [Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ), [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ), [Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ), [Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ), [Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-AWQ), [Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ), [Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ), [Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4)

</details>

<details>
<summary><b>Qwen3</b></summary>

`model_type: qwen3` or `qwen3_moe`; text input to text output.

**Original:**

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)

**Pre-quantized:**

- [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)
- [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ), [nvidia/Qwen3-8B-FP8](https://huggingface.co/nvidia/Qwen3-8B-FP8), [nvidia/Qwen3-8B-NVFP4](https://huggingface.co/nvidia/Qwen3-8B-NVFP4)
- [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ), [nvidia/Qwen3-14B-FP8](https://huggingface.co/nvidia/Qwen3-14B-FP8), [nvidia/Qwen3-14B-NVFP4](https://huggingface.co/nvidia/Qwen3-14B-NVFP4)
- [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4), [nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)

</details>

<details>
<summary><b>HunYuan V1 Dense</b></summary>

`model_type: hunyuan_v1_dense`; text input to text output. These checkpoints
apply per-head QK RMSNorm after RoPE and use the
DynamicNTKAlpha RoPE variant (`rope_scaling.type == "dynamic"` with `alpha`);
both are handled automatically by export and runtime.

**Original:**

- [tencent/HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B)
- [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B), [tencent/Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)

</details>

<details>
<summary><b>Qwen vision-language families</b></summary>

`model_type: qwen2_5_vl` or `qwen3_vl`; text and image/video input to text
output.

**Qwen2.5-VL:**

- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2.5-VL-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ), [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)
- [nvidia/Qwen2.5-VL-7B-Instruct-FP8](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP8), [nvidia/Qwen2.5-VL-7B-Instruct-NVFP4](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4)

**Qwen3-VL and compatible checkpoints:**

- [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [Qwen/Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking)
- [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen/Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B), [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

</details>

<details>
<summary><b>Qwen3.5 / Qwen3.6 / Qwen3.8</b></summary>

`model_type: qwen3_5` or `qwen3_5_moe`; text-only checkpoints accept text,
while multimodal checkpoints accept text and image/video. Both produce text.

- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
- [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
- [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B), [nvidia/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4)

</details>

<details>
<summary><b>InternVL and Phi-4</b></summary>

InternVL uses `model_type: internvl_chat` or `internvl`; Phi-4 Multimodal uses
`phi4mm` or `phi4_multimodal`. Both accept text and image input and produce
text; Phi-4 also accepts audio.

**InternVL:**

- [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf), [OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf)
- [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- [OpenGVLab/InternVL3-14B-hf](https://huggingface.co/OpenGVLab/InternVL3-14B-hf)
- [OpenGVLab/InternVL3_5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF), [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF), [OpenGVLab/InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF)
- [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF), [OpenGVLab/InternVL3_5-14B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF)
- [OpenGVLab/InternVL3-1B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-1B-AWQ), [OpenGVLab/InternVL3-2B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ), [OpenGVLab/InternVL3-8B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-8B-AWQ), [OpenGVLab/InternVL3-14B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-14B-AWQ)

**Phi-4 Multimodal:**

- [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

</details>

<details>
<summary><b>Nemotron-3 / 3.5</b></summary>

Nemotron-H language checkpoints use `model_type: nemotron_h` and accept text
input. Nemotron Omni uses `NemotronH_Nano_Omni_Reasoning_V3` and accepts text,
image/video, and audio. Both produce text.

**Nemotron-H:**

- [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8)
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8), [nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4)
- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)

**Nemotron Omni:**

- [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)

</details>

<details>
<summary><b>Gemma4</b></summary>

`model_type: gemma4` or `gemma4_unified`; accepted modalities depend on the
checkpoint and all variants produce text. Paired `gemma4_assistant` or
`gemma4_unified_assistant` checkpoints provide MTP.

**Text, image, and audio input:**

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)

**Text and image input:**

- [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized)
- [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)

**MTP assistants:**

- [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant), [google/gemma-4-E4B-it-assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant)
- [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant), [google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant)
- [google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)

</details>

<details>
<summary><b>DiffusionGemma</b></summary>

`model_type: diffusion_gemma` or `diffusiongemma`; text and image input to text
output through block-diffusion decoding.

- [nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)

</details>

<details>
<summary><b>Muse-Glimmer</b></summary>

`model_type: muse_glimmer`; text, image, or video input to text output.

The FP16 checkpoint supports text, image, and video
input; the NVFP4 checkpoint is text-only. Compatible DFlash and DFlash2 drafts
are listed under [Speculative Draft Checkpoints](#speculative-draft-checkpoints).

- [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4)

</details>

## Speech Recognition

- **Qwen3-ASR** (`qwen3_asr`): audio to transcript text.
  [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B),
  [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **Nemotron-3.5-ASR** (`nemotron3_5_asr`): streaming audio to transcript
  text through its model-specific RNN-T runtime.
  [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

## Speech Generation

- **Qwen3-TTS** (`qwen3_tts`): text, language, style, or reference-speech
  conditioning to speech.
  [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
  [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
  [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign),
  [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base),
  [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **Qwen3-Omni** (`qwen3_omni_moe`): text, image/video, and audio to text
  and optional speech.
  [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)

## Action and Multimodal Reasoning

- **Alpamayo** (`alpamayo_r1`): text and image/video to reasoning and a
  trajectory. [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- **Cosmos3-Edge** (`cosmos3_edge`): text and image/video to reasoning.
  [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)
- **pi0.5** (model-specific exporter/runtime): camera observations and an
  instruction to a robot action chunk.
  [lerobot/pi05_libero_base](https://huggingface.co/lerobot/pi05_libero_base),
  [lerobot/pi05_droid](https://huggingface.co/lerobot/pi05_droid),
  [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) (served under the
  `pi05_aloha` contract)

## Speculative Draft Checkpoints

### EAGLE3 Draft Models

| Draft checkpoint | Base model |
|---|---|
| [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) or [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| [deepseek-ai/eagle3_gemma4_12b_ttt7](https://huggingface.co/deepseek-ai/eagle3_gemma4_12b_ttt7) | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |

### DFlash Draft Models

| Draft checkpoint | Base model |
|---|---|
| [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) (quantized checkpoint: `Qwen3.5-4B-DFlash-NVFP4`) | Qwen3.5-4B-NVFP4 |
| [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) | [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) |
| [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) or [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) |
| [deepseek-ai/dflash_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dflash_gemma4_12b_block7) or [z-lab/gemma4-12B-it-DFlash](https://huggingface.co/z-lab/gemma4-12B-it-DFlash) | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash) | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) |
| [meta-models/Muse-Glimmer-30B-assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) or [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4) |

### DFlash2 Draft Models

DFlash2 uses the public DFlash engine/runtime mode. The draft checkpoint
selects the versioned contract, while the runtime may choose any block size
covered by the engine profile. DFlash2 does not support DDTree.

| Draft checkpoint | Base model |
|---|---|
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), including matched NVFP4 or INT4 quantized checkpoints |
| [incoai/Muse-Glimmer-30B-DFlash2](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2) | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) or [RadixArk/Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4) |

### DSpark Draft Models

| Draft checkpoint | Base model |
|---|---|
| [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark) | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) |
| [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) |

The Nemotron-3.5-Lightning DSpark draft differs from the DeepSpec block7 drafts
above: it uses a block size of 8, sliding-window attention (1024) with a learned
per-head attention sink, and causal proposal attention. The sliding window is
baked into the draft engine as the contiguous-query XQA variant. The Nemotron
draft supports both chain decoding and greedy DSpark DDTree; tree mode
reconstructs path-dependent recurrent state. The published checkpoint uses
`sample_from_anchor=false`: slot 0 is anchor-only and proposals start at slot 1.
The runtime also supports the older anchor-sampled layout. Tree mode uses
`--specDraftTopK > 1` with a tree-capable base.

### JetSpec Draft Models

JetSpec draft checkpoints are detected by `jetspec_config` in `config.json` and
exported with `DFlashDraftModel` using causal proposal attention. The validated
runtime path is branching tree verification: export the base with
`--jetspec-tree-base --jetspec-draft-dir <draft_checkpoint>`, export the draft
with `--jetspec-draft --jetspec-draft-dir <draft_checkpoint>`, and run with
`--specDraftTopK > 1`. `--jetspecBlockSize` and `--dflashBlockSize` configure
the same cached-draft proposal block size; the JetSpec spelling is provided for
clarity in JetSpec command lines.

For the listed Qwen3 pair, disable thinking mode in the input JSON when
evaluating accuracy, acceptance rate, or throughput.

| Draft checkpoint | Base model |
|---|---|
| [JetSpec/jetspec-qwen3-8b](https://huggingface.co/JetSpec/jetspec-qwen3-8b) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
