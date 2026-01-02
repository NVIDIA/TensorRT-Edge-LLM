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

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE MULTIMODAL_CPP_SRCS "multimodal/*.cpp")
file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")
file(GLOB_RECURSE PROFILER_CPP_SRCS "profiling/*.cpp")

add_library(edgellmKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS}
                                  ${COMMON_CPP_SRCS})
target_include_directories(edgellmKernels PRIVATE ${COMMON_INCLUDE_DIRS})

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

# Generate tokenizer C++ static library
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(edgellmTokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(edgellmTokenizer PRIVATE ${COMMON_INCLUDE_DIRS})

# Generate builder C++ static library
file(GLOB_RECURSE BUILDER_CPP_SRCS "builder/*.cpp")
add_library(edgellmBuilder STATIC ${BUILDER_CPP_SRCS} ${COMMON_CPP_SRCS})
target_include_directories(edgellmBuilder PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                  ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmBuilder PRIVATE ${NV_ONNX_PARSER_LIB})

# Generate the TensorRT plugin shared libraries
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(NvInfer_edgellm_plugin commonLibraryExt edgellmKernels
                      ${CUDA_DRIVER_LIB})
add_cross_build_link_options(NvInfer_edgellm_plugin)
