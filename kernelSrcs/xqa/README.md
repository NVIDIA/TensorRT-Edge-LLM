# XQA - A set of optimized kernels for generation-phase MQA/GQA

This document outlines the process for generating the pre-compiled XQA CUDA kernel binaries (`.cubin` files) and how to run the associated unit tests.

The xqa source is from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) commit [`a4b4ed45`](https://github.com/NVIDIA/TensorRT-LLM/commit/a4b4ed45359167eb6cf3c2100d5d0dcd326bc588).

## 1. Generating Kernel Binaries (CUBINs)

### 1.1. Generate cubins for SM 80, 86, 87, 89, 100, 101

These architectures can be compiled with a **CUDA 12.8** Toolkit.

**Steps:**
```bash
# Navigate to the kernel source directory
cd kernelSrcs/xqa/

# Run the generation script
python3 gen_cubins.py
```
This will generate the `.cubin.cpp` files and place them in the `cpp/kernels/decodeAttentionKernels/cubin/` directory.

### 1.2. Generate cubins for SM 120, 121

These newer architectures require a more recent **CUDA 12.9** or higher Toolkit.

**Steps:**

1.  **Generate SM 12x CUBINs:**

    ```bash
    # Ensure you are in the xqa directory
    cd kernelSrcs/xqa

    python3 gen_cubins.py sm12x
    ```

2.  **Merge the new CUBIN files:**

    Copy the newly generated `.cubin.cpp` files into the final `cpp/kernels/decodeAttentionKernels/cubin/` directory.
    ```bash
    cd cpp/kernels/decodeAttentionKernels
    cp cubin_sm12x/xqa_kernel_dt_* cubin/
    ```

3.  **Update the Kernel Header:**

    You need to manually merge the contents of `cubin_sm12x/xqa_kernel_cubin.h` into the existing `cubin/xqa_kernel_cubin.h`.

4.  **Clean up:**

    Remove the temporary directory once the merge is complete.
    ```bash
    rm -r cubin_sm12x
    ```

---

## 2. Kernel Unit Tests

The project includes a suite of unit tests to verify the correctness of the attention kernels.

The test executable will be located in your build directory (e.g., `build/`). You can use `gtest_filter` to run specific tests.

**To run all primary attention and tree-attention decoding tests:**
```bash
./build/unitTest --gtest_filter=XQAAttentionDecodingTest.*:XQATreeAttentionDecodingTest.*
```

**To list all available tests:**
```bash
./build/unitTest --gtest_list_tests
```

## 3. Generating New Cubins

If you encounter a scenario that requires a kernel with parameters not covered by the pre-compiled cubins, you can generate a new one.

To do this, modify the compile configurations in `kernelSrcs/xqa/gen_cubins.py`. Locate the `edgellm_config_list` or `edgellm_config_list_spec_dec` and modify the `CompileMacroOption` entries to match your required parameters.

For example:
```python
edgellm_config_list = [[
    CompileMacroOption('DTYPE', 'dt', ['__half']),
    CompileMacroOption('HEAD_ELEMS', 'd', [128, 64, 32]),
    CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
    CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0]),
    CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                       [0]),  # 0 denotes contiguous kv cache.
    CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [1, 2, 3, 4, 5, 6, 7, 8]),
    CompileMacroOption('M_TILESIZE', 'm', [8]),
    CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
]]
```
After modifying the script, run `gen_cubins.py` to generate the new cubin.

**NOTE：** The adapt_source.patch file in this directory records the adaptations made for EdgeLLM to the source XQA code. However, you do not need to apply it, as these changes have already been included in the source.