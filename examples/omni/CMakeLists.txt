# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Qwen3-Omni Inference Example Note: cuBLAS is loaded at runtime via
# dlopen/dlsym in TTSRuntime, no compile-time linking needed.
add_executable(qwen3_tts_inference qwen3_tts_inference.cpp)

target_include_directories(
  qwen3_tts_inference
  PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/examples/utils
          $<$<PLATFORM_ID:Windows>:${CMAKE_SOURCE_DIR}/examples/utils/windows>)

target_link_libraries(qwen3_tts_inference PRIVATE edgellmCore exampleUtils
                                                  commonLibraryExt)

add_cross_build_link_options(qwen3_tts_inference)

# Set output directory
set_target_properties(
  qwen3_tts_inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                 ${CMAKE_BINARY_DIR}/examples/omni)
