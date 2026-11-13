# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# QNX Safety admission requires application-level VM lifecycle coordination.
if(QNX_BUILD AND QNX_SAFE_CUDA)
  return()
endif()

add_executable(llm_inference_safe llm_inference_safe.cpp)
target_include_directories(llm_inference_safe PRIVATE "${CMAKE_SOURCE_DIR}/cpp")
target_include_directories(
  llm_inference_safe SYSTEM PRIVATE ${CUDA_INCLUDE_DIR}
                                    "${CMAKE_SOURCE_DIR}/3rdParty/stb")
target_link_libraries(
  llm_inference_safe PRIVATE edgellmSafetyRuntime TensorRT::TensorRTSafe
                             ${CUDART_LIB} ${EDGELLM_SAFETY_CUDA_DRIVER})
set_target_properties(llm_inference_safe PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_BINARY_DIR}/bin")
add_cross_build_link_options(llm_inference_safe)
edgellm_apply_qnx_warning_suppressions(llm_inference_safe)
