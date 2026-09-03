# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier:
# LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -fPIC"
)

# Define additional CUDA flags for kernel compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0 -fPIC")
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)

set(CMAKE_CUDA_FLAGS_DEBUG "-g")

# Hide internal C++ symbols; only EDGELLM_PLUGIN_EXPORT entry points stay
# visible. Prevents duplicate-symbol collisions when our libraries are loaded
# together.
function(edgellm_set_hidden_visibility target)
  set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden
                                             VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# ------------------------------------------------------------------------------
# FMHA Cubin Conditional Compilation Exclude SM architectures that are not in
# CMAKE_CUDA_ARCHITECTURES to reduce binary size. Only cubins for the target
# architectures will be compiled.
# ------------------------------------------------------------------------------
set(FMHA_ALL_SM_VERSIONS
    80
    86
    87
    89
    100
    101
    120
    121)
set(FMHA_EXCLUDE_DEFINITIONS "")

# Build a list of required FMHA SM versions based on target architectures. Some
# SM versions reuse cubins from other SM versions at runtime. Strip the
# accelerated/virtual/real suffixes (e.g. 100a, 101a, 110a, 90-real) so the
# IN_LIST checks below compare against plain numeric SM versions.
set(REQUIRED_FMHA_SM_VERSIONS "")
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
  string(REGEX REPLACE "[^0-9].*$" "" arch_num "${arch}")
  list(APPEND REQUIRED_FMHA_SM_VERSIONS ${arch_num})
endforeach()
list(REMOVE_DUPLICATES REQUIRED_FMHA_SM_VERSIONS)

# SM 110 uses SM 101 FMHA cubins at runtime
if(110 IN_LIST REQUIRED_FMHA_SM_VERSIONS AND NOT 101 IN_LIST
                                             REQUIRED_FMHA_SM_VERSIONS)
  list(APPEND REQUIRED_FMHA_SM_VERSIONS 101)
endif()

# Exclude SM versions that are not required
foreach(SM_VER ${FMHA_ALL_SM_VERSIONS})
  if(NOT SM_VER IN_LIST REQUIRED_FMHA_SM_VERSIONS)
    list(APPEND FMHA_EXCLUDE_DEFINITIONS "EXCLUDE_SM_${SM_VER}")
  endif()
endforeach()

if(FMHA_EXCLUDE_DEFINITIONS)
  message(
    STATUS
      "FMHA Kernels: Excluding SM architectures: ${FMHA_EXCLUDE_DEFINITIONS}")
else()
  message(STATUS "FMHA Kernels: Compiling cubins for all SM architectures")
endif()

# Generate plugin JIT sources (kernel sources + CUDA headers baked into a .cpp).
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT
    "${CMAKE_BINARY_DIR}/generated/pluginJitEmbeddedSources.cpp")
set(PLUGIN_JIT_EMBED_SCRIPT
    "${CMAKE_SOURCE_DIR}/kernelSrcs/jit_utils/gen_cpp_header.py")
file(GLOB XQA_SOURCE_FILES "${CMAKE_SOURCE_DIR}/kernelSrcs/xqa/*.cu"
     "${CMAKE_SOURCE_DIR}/kernelSrcs/xqa/*.cuh"
     "${CMAKE_SOURCE_DIR}/kernelSrcs/xqa/*.h")
set(NVFP4_A16_BLACKWELL_GEMV_JIT_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/kernelSrcs/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemv.cu"
)

# The generated TU embeds CUDA toolkit headers verbatim, so it is stale after a
# toolkit change even when no repo file moved. Depend on the headers themselves
# and stamp the toolkit version into the rule.
set(PLUGIN_JIT_EMBEDDED_CUDA_HEADER_DEPS "")
foreach(
  _plugin_jit_cuda_header
  cuda_bf16.h
  cuda_bf16.hpp
  cuda_fp16.h
  cuda_fp16.hpp
  cuda_fp8.h
  cuda_fp8.hpp
  cuda_fp4.h
  cuda_fp4.hpp
  cuda_fp6.h
  cuda_fp6.hpp)
  if(EXISTS "${CUDA_RUNTIME_API_INCLUDE_DIR}/${_plugin_jit_cuda_header}")
    list(APPEND PLUGIN_JIT_EMBEDDED_CUDA_HEADER_DEPS
         "${CUDA_RUNTIME_API_INCLUDE_DIR}/${_plugin_jit_cuda_header}")
  endif()
endforeach()

add_custom_command(
  OUTPUT ${PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E echo
          "Plugin JIT embedded sources for CUDA ${CMAKE_CUDA_COMPILER_VERSION}"
  COMMAND
    ${Python3_EXECUTABLE} ${PLUGIN_JIT_EMBED_SCRIPT} -o
    ${PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT} --xqa_dir
    ${CMAKE_SOURCE_DIR}/kernelSrcs/xqa --gemv_dir
    ${CMAKE_SOURCE_DIR}/kernelSrcs/nvfp4A16BlackwellGemv --cpp_dir
    ${CMAKE_SOURCE_DIR}/cpp --cuda_include_dir ${CUDA_RUNTIME_API_INCLUDE_DIR}
  DEPENDS
    ${XQA_SOURCE_FILES}
    ${NVFP4_A16_BLACKWELL_GEMV_JIT_SOURCE_FILES}
    ${PLUGIN_JIT_EMBED_SCRIPT}
    ${PLUGIN_JIT_EMBEDDED_CUDA_HEADER_DEPS}
    "${CMAKE_SOURCE_DIR}/cpp/common/cudaMacros.h"
    "${CMAKE_SOURCE_DIR}/cpp/kernels/decodeAttentionKernels/xqaKernelTypes.h"
  COMMENT "Generating embedded plugin JIT sources for NVRTC"
  VERBATIM)

# Keep the generator single-owner even when multiple final targets consume the
# static JIT archive.
add_custom_target(generatePluginJitEmbeddedSources
                  DEPENDS ${PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT})
set_source_files_properties(${PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT}
                            PROPERTIES GENERATED TRUE)

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")
# Exclude MoE Marlin ops/instantiations - compiled only via include from
# moeMarlin.cu.
list(
  FILTER
  KERNELS_CU_SRCS
  EXCLUDE
  REGEX
  "marlin_moe_wna16/(ops|sm80_kernel_(float16_u4_float16|float16_fe2m1_float16|bfloat16_fe2m1_bfloat16))\\.cu$"
)
# JIT compilation and module loading belong to the plugin, not the core kernel
# archives. Unit tests link this target explicitly for direct compiler coverage.
set(PLUGIN_JIT_CPP_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/PluginJitKernels/pluginJitCompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitCompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/nvfp4A16BlackwellGemv/nvfp4A16BlackwellGemvJitRunner.cpp"
)
list(REMOVE_ITEM KERNELS_CPP_SRCS ${PLUGIN_JIT_CPP_SRCS})

file(GLOB_RECURSE SAMPLER_CPP_SRCS "sampler/*.cpp")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE MULTIMODAL_CPP_SRCS "multimodal/*.cpp")
file(GLOB_RECURSE ACTION_CPP_SRCS "action/*.cpp")
file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")
file(GLOB_RECURSE PROFILER_CPP_SRCS "profiling/*.cpp")

# Enable relocatable device code for MoE Marlin kernels
file(GLOB_RECURSE MOE_MARLIN_CU_SRCS "kernels/moe/moe_marlin/*.cu")
set_source_files_properties(${MOE_MARLIN_CU_SRCS} PROPERTIES COMPILE_FLAGS
                                                             "-rdc=true")

# Build-time NVRTC compilers and runtime CUDA Driver module loaders.
add_library(edgellmPluginJit STATIC ${PLUGIN_JIT_CPP_SRCS}
                                    ${PLUGIN_JIT_EMBEDDED_SOURCES_OUTPUT})
target_include_directories(edgellmPluginJit PRIVATE ${COMMON_INCLUDE_DIRS}
                                                    ${NVRTC_INCLUDE_DIR})
target_link_libraries(
  edgellmPluginJit PRIVATE ${NVRTC_LIB} ${CUDA_DRIVER_LINK_LIB} ${CUDART_LIB})
add_dependencies(edgellmPluginJit generatePluginJitEmbeddedSources)
edgellm_apply_qnx_warning_suppressions(edgellmPluginJit)
edgellm_set_hidden_visibility(edgellmPluginJit)

add_library(edgellmKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS}
                                  ${COMMON_CPP_SRCS})
target_include_directories(edgellmKernels PRIVATE ${COMMON_INCLUDE_DIRS})
if(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
  target_compile_definitions(edgellmKernels
                             PRIVATE ENABLE_CUTEDSL_MODULE_TEST_HOOK)
endif()
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmKernels PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
edgellm_apply_qnx_warning_suppressions(edgellmKernels)
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmKernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
edgellm_set_hidden_visibility(edgellmKernels)

file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")

add_library(
  edgellmCore STATIC
  ${COMMON_CPP_SRCS}
  ${COMMON_CU_SRCS}
  ${KERNELS_CPP_SRCS}
  ${KERNELS_CU_SRCS}
  ${SAMPLER_CPP_SRCS}
  ${SAMPLER_CU_SRCS}
  ${MULTIMODAL_CPP_SRCS}
  ${ACTION_CPP_SRCS}
  ${RUNTIME_CPP_SRCS}
  ${PROFILER_CPP_SRCS}
  ${TOKENIZER_CPP_SRCS})
target_include_directories(edgellmCore PRIVATE ${COMMON_INCLUDE_DIRS})
if(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
  target_compile_definitions(edgellmCore
                             PRIVATE ENABLE_CUTEDSL_MODULE_TEST_HOOK)
endif()
target_link_libraries(edgellmCore PRIVATE ${CMAKE_DL_LIBS})
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmCore PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
edgellm_apply_qnx_warning_suppressions(edgellmCore)
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
edgellm_set_hidden_visibility(edgellmCore)

# Generate builder C++ static library
file(GLOB_RECURSE BUILDER_CPP_SRCS "builder/*.cpp")
add_library(edgellmBuilder STATIC ${BUILDER_CPP_SRCS} ${COMMON_CPP_SRCS})
target_include_directories(edgellmBuilder PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                  ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmBuilder PRIVATE ${NV_ONNX_PARSER_LIB}
                                             ${CMAKE_DL_LIBS})
edgellm_apply_qnx_warning_suppressions(edgellmBuilder)
edgellm_set_hidden_visibility(edgellmBuilder)

# Generate the TensorRT plugin shared libraries
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(
  NvInfer_edgellm_plugin
  PRIVATE commonLibraryExt edgellmKernels edgellmPluginJit
          ${CUDA_DRIVER_LINK_LIB} ${CUDART_LIB})
add_cross_build_link_options(NvInfer_edgellm_plugin)
edgellm_apply_qnx_warning_suppressions(NvInfer_edgellm_plugin)
edgellm_set_hidden_visibility(NvInfer_edgellm_plugin)
# Also drop symbols pulled in from static archives (edgellmKernels, CuTe DSL).
target_link_options(NvInfer_edgellm_plugin PRIVATE "-Wl,--exclude-libs,ALL")

# cmake-format: off
# CuTe DSL unified kernels: link locally generated static library. Artifacts are
# generated by: python kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN>. Set
# ENABLE_CUTE_DSL to enable one of:
#   fmha             -- Context/ViT FMHA, plus the optimized Blackwell
#                       variants when the artifact has them
#   gdn              -- gated delta net (Mamba2-style hybrid SSM)
#   f16_moe          -- target-specific FP16 grouped MoE for Ampere,
#                       Blackwell DC, and Blackwell GeForce
#   nvfp4_moe        -- SM110 (Thor) split FC1/FC2 NVFP4 MoE; consumed by the
#                       SM110-only ``Nvfp4MoePlugin``
#   nvfp4_fused_moe  -- SM12x (consumer Blackwell / Spark GB10) fused NVFP4
#                       MoE; consumed by ``NvFP4MoEPluginGeforce``
#   rmsnorm          -- FP16/BF16 RMSNorm for supported production hidden sizes
# or a semicolon-separated list (e.g. fmha;gdn), or ALL. fmha is always linked.
# See the top-level CMakeLists.txt for the canonical option list.
# cmake-format: on
include(${CMAKE_SOURCE_DIR}/cmake/CuteDsl.cmake)
cute_dsl_setup(LINK_TARGETS edgellmKernels edgellmCore NvInfer_edgellm_plugin)
