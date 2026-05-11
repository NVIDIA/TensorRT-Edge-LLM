# ViT FMHA Cubins

This folder contains the reproducible recipe for generating ViT-oriented FMHA
cubins from TensorRT-LLM.

The goal is to generate non-causal FMHA v2 kernels for ViT attention:

- FP16 first
- `PACKED_QKV` input layout
- `PADDING` mask type, which is the non-causal/segment-boundary mode
- variable sequence length kernels
- head sizes `64`, `80`, and `128`
- SM targets `80`, `86`, `87`, `89`, `100`, `101`, and `120`
- Edge-LLM style embedded `.cubin.cpp` packaging

## Generation Flow

Start from the same TensorRT-LLM checkout used by `kernelSrcs/fmha_v2`:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 636c622bb8685b9db7422b3fa064a173cf1ff2a8
```

Apply the existing Edge-LLM FMHA patch first, then apply this ViT patch:

```bash
git apply ../TensorRT-Edge-LLM/kernelSrcs/fmha_v2/gen_fmha_cubin.patch
git apply ../TensorRT-Edge-LLM/kernelSrcs/vit_fmha/gen_vit_fmha_cubin.patch
```

Generate the ViT FMHA source files:

```bash
cd cpp/kernels/fmha_v2
export GENERATE_EDGE_LLM_VIT=1 GENERATE_CUBIN=1 ENABLE_SM100=1
python3 setup.py
```

For Blackwell/Thor `sm120`, generate the SM12x batch separately:

```bash
export GENERATE_EDGE_LLM_VIT=1 GENERATE_CUBIN=1 ENABLE_SM12X=1
python3 setup.py
```

Then build cubins and convert them into Edge-LLM's checked-in `.cubin.cpp`
format.

```bash
make cubin_demobert -j$(nproc)
```

The expected first-pass output is 24 `.cubin.cpp` files:

- 6 SM targets
- 4 kernel variants per SM: `qkv_64`, `qkv_80`, `qkv_128`, and tiled `qkv_128`

The SM12x batch adds 4 more `.cubin.cpp` files:

- 1 SM target: `sm120`
- 4 kernel variants per SM: `qkv_64`, `qkv_80`, `qkv_128`, and tiled `qkv_128`

Copy the generated files into `cpp/kernels/vitAttentionKernels/cubin/` with a
`vit_` filename prefix, and prefix the generated cubin symbols with
`vit_cubin_` so they do not collide with the context-attention FMHA cubins.
