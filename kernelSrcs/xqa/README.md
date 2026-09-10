# XQA - optimized kernels for generation-phase MQA/GQA

The XQA source is from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
commit [`a4b4ed45`](https://github.com/NVIDIA/TensorRT-LLM/commit/a4b4ed45359167eb6cf3c2100d5d0dcd326bc588).

EdgeLLM compiles XQA kernels with NVRTC during TensorRT engine build. The
AttentionPlugin serializes the generated module bytes into the engine, and
runtime deserialization loads those bytes with the CUDA driver API. Runtime
inference does not invoke NVRTC on the normal path.

Do not check generated XQA cubin blobs into the source tree.

## Source Embedding

`kernelSrcs/jit_utils/gen_cpp_header.py` bakes the XQA sources, the two project headers they include
(`common/cudaMacros.h`, `kernels/decodeAttentionKernels/xqaKernelTypes.h`) and
the CUDA toolkit headers they reach into a single generated
`pluginJitEmbeddedSources.cpp`. Every file is handed to `nvrtcCreateProgram` as a
virtual header, so NVRTC compilation never touches the filesystem.

CMake drives the script through the `generatePluginJitEmbeddedSources` target --
do not invoke it by hand and do not check the generated `.cpp` in. Its module
docstring calls out the two parts that are load-bearing.

If a new XQA configuration reaches a CUDA header that is not yet embedded, NVRTC
fails with `catastrophic error: could not open source file "<name>"`. Add the
header to `REQUIRED_CUDA_HEADERS`, or to `OPTIONAL_CUDA_HEADERS` if it only
exists on newer toolkits.

## Kernel Unit Tests

The project includes unit tests to verify the correctness of the attention
kernels. The test executable is located in the build directory.

To run all primary attention and tree-attention decoding tests:

```bash
./build/unittests/unitTestKernelsAttention --gtest_filter=XQAAttentionDecodingTest.*:XQATreeAttentionDecodingTest.*
```

To list all available tests:

```bash
./build/unittests/unitTestKernelsAttention --gtest_list_tests
```

## Adding XQA Configurations

The NVRTC option mapping is implemented in
`cpp/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp`. Add new
supported shapes there, then run the unit tests and an engine-build smoke test
to verify that the generated module is serialized and deserialized correctly.

**NOTE:** `adapt_source.patch` records the source-level adaptations made for
EdgeLLM. You do not need to apply it; those changes are already included in the
source.
