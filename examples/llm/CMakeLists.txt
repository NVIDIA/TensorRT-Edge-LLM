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

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_build llm_build.cpp)
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt
                                        edgellmBuilder)
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_build)

add_executable(llm_inference llm_inference.cpp)
target_link_libraries(llm_inference PRIVATE edgellmCore exampleUtils
                                            commonLibraryExt)
target_include_directories(
  llm_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_inference)

add_executable(llm_stream llm_stream.cpp)
target_link_libraries(llm_stream PRIVATE edgellmCore exampleUtils
                                         commonLibraryExt)
target_include_directories(
  llm_stream PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_stream)

add_executable(llm_bench llm_bench.cpp)
target_link_libraries(llm_bench PRIVATE commonLibraryExt edgellmCore
                                        exampleUtils)
target_include_directories(llm_bench PRIVATE ${COMMON_INCLUDE_DIRS}
                                             ${CMAKE_SOURCE_DIR}/examples/utils)
target_compile_options(llm_bench PRIVATE -Wno-psabi)
add_cross_build_link_options(llm_bench)

# All-reduce communication is registered through symbols resolved from the
# dlopen'd plugin, instead of linking the plugin into this executable.
# Build-time linking causes duplicate CUDA device-code modules between the
# plugin and executable, which corrupts kernel execution on local multi-device
# systems.

# Optional multi-device support. Runtime collective code loads NCCL dynamically.
# MPI launch support is optional for llm_inference so local single-process
# threaded launch can be built without an MPI dependency.
option(ENABLE_MULTI_DEVICE
       "Enable multi-device inference using runtime-loaded NCCL" OFF)
option(ENABLE_MULTI_DEVICE_MPI
       "Enable MPI rank-launch support for multi-device inference" OFF)

if(ENABLE_MULTI_DEVICE)
  target_compile_definitions(llm_inference PRIVATE EDGELLM_ENABLE_MULTI_DEVICE)

  if(ENABLE_MULTI_DEVICE_MPI)
    find_package(MPI REQUIRED)

    target_compile_definitions(llm_inference
                               PRIVATE EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    target_link_libraries(llm_inference PRIVATE MPI::MPI_CXX)
    target_include_directories(llm_inference PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    message(STATUS "Multi-device support ENABLED (MPI + runtime-loaded NCCL)")
  else()
    message(
      STATUS
        "Multi-device support ENABLED (threaded/default + runtime-loaded NCCL; MPI disabled)"
    )
  endif()
endif()
