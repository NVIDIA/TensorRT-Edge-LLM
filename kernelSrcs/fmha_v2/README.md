# FMHA_v2

This document outlines the process for generating the pre-compiled FMHA_V2 CUDA kernel binaries (`.cubin` files) and how to run the associated unit tests.

## 1. Generating Kernel Binaries (CUBINs)

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 636c622bb8685b9db7422b3fa064a173cf1ff2a8
git apply gen_fmha_cubin.patch
cd cpp/kernels/fmha_v2
```

### 1.1. Generate cubins for SM 80, 86, 87, 89, 100, 101

These architectures can be compiled with a **CUDA 12.8** Toolkit.

**Steps:**
```bash
# 1) Generate the arch–specific .cu sources & headers
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 ENABLE_SM100=1
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda128
```
---

### 1.2. Generate cubins for SM 120, 121

These newer architectures require a more recent **CUDA 12.9** or higher Toolkit.

**Steps:**

1.  **Generate SM 12x CUBINs:**

```bash
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 ENABLE_SM12X=1 

# 1) Generate Blackwell-only .cu sources & headers
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda129
```

2.  **Merge the new CUBIN files:**

Merge the `generated_cuda128` and `generated_cuda129` dirs into a single cubin dir, located at `cpp/kernels/contextAttentionKernels/cubin`.


## Kernel Unit Test
```bash
ln -s generated_cuda128 generated

make bin/fmha.exe -j$(nproc)

bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 1024 -d 128  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 128 -d 64  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -force-non-tiled
```