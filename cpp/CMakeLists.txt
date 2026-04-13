# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

set(CMAKE_CUDA_FLAGS_DEBUG "")

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
# SM versions reuse cubins from other SM versions at runtime.
set(REQUIRED_FMHA_SM_VERSIONS ${CMAKE_CUDA_ARCHITECTURES})

# SM 110 uses SM 101 FMHA cubins at runtime
if(110 IN_LIST CMAKE_CUDA_ARCHITECTURES AND NOT 101 IN_LIST
                                            CMAKE_CUDA_ARCHITECTURES)
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

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")
# Exclude MoE Marlin ops/sm80 - compiled only via include from moeMarlin.cu
list(FILTER KERNELS_CU_SRCS EXCLUDE REGEX
     "marlin_moe_wna16/(ops|sm80_kernel_float16_u4_float16)\\.cu$")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE MULTIMODAL_CPP_SRCS "multimodal/*.cpp")
file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")
file(GLOB_RECURSE PROFILER_CPP_SRCS "profiling/*.cpp")

# Enable relocatable device code for MoE Marlin kernels
file(GLOB_RECURSE MOE_MARLIN_CU_SRCS "kernels/moe/moe_marlin/*.cu")
set_source_files_properties(${MOE_MARLIN_CU_SRCS} PROPERTIES COMPILE_FLAGS
                                                             "-rdc=true")

add_library(edgellmKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS}
                                  ${COMMON_CPP_SRCS})
target_include_directories(edgellmKernels PRIVATE ${COMMON_INCLUDE_DIRS})
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmKernels PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmKernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(
  edgellmCore STATIC
  ${COMMON_CPP_SRCS}
  ${COMMON_CU_SRCS}
  ${KERNELS_CPP_SRCS}
  ${KERNELS_CU_SRCS}
  ${SAMPLER_CU_SRCS}
  ${MULTIMODAL_CPP_SRCS}
  ${RUNTIME_CPP_SRCS}
  ${PROFILER_CPP_SRCS})
target_include_directories(edgellmCore PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmCore PRIVATE dl curand)
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmCore PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Generate tokenizer C++ static library
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(edgellmTokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(edgellmTokenizer PRIVATE ${COMMON_INCLUDE_DIRS})

# Generate builder C++ static library
file(GLOB_RECURSE BUILDER_CPP_SRCS "builder/*.cpp")
add_library(edgellmBuilder STATIC ${BUILDER_CPP_SRCS} ${COMMON_CPP_SRCS})
target_include_directories(edgellmBuilder PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                  ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmBuilder PRIVATE ${NV_ONNX_PARSER_LIB} dl)

# Generate the TensorRT plugin shared libraries
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
# Apply FMHA SM exclusion definitions
target_compile_definitions(NvInfer_edgellm_plugin
                           PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
target_link_libraries(NvInfer_edgellm_plugin commonLibraryExt edgellmKernels
                      ${CUDA_DRIVER_LIB} ${CUDART_LIB})
add_cross_build_link_options(NvInfer_edgellm_plugin)

# CuTe DSL FMHA (Blackwell SM100+): build-time kernel compilation
if(ENABLE_CUTE_DSL_FMHA)
  include(${CMAKE_SOURCE_DIR}/cmake/CuteDslFMHA.cmake)
  cute_dsl_fmha_setup(TARGETS edgellmKernels edgellmCore PLUGIN_TARGET
                      NvInfer_edgellm_plugin)
endif()
