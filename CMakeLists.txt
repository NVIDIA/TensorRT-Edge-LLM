# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(
  tensorrt_edgellm_sdk
  VERSION 0.6.0
  LANGUAGES CXX CUDA)

# Specify the C++ standard and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter"
)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations -Xcompiler=-Werror,-Wno-error=unused-parameter"
)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Code coverage with gcov (for SonarQube analysis)
option(ENABLE_COVERAGE "Enable gcov code coverage instrumentation" OFF)
include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)

macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

macro(add_cross_build_link_options target)
  if(DEFINED AARCH64_BUILD)
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib")
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib/stubs")
  endif()
endmacro(add_cross_build_link_options)

# CUDA flags and dependencies.
if(DEFINED CUDA_VERSION)
  message(
    FATAL_ERROR
      "CUDA_VERSION can cause ambiguity with CUDA Macros. Please use -DCUDA_CTK_VERSION to specify the CUDA Toolkit version."
  )
endif()
set_ifndef(CUDA_CTK_VERSION 12.8)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_CTK_VERSION})

if(NOT DEFINED AARCH64_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
  if(CUDA_CTK_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
  endif()
endif()

find_path(
  CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES include)

find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64)

find_library(
  CUDA_DRIVER_LIB cuda
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs)

find_path(
  CURAND_KERNEL_INCLUDE_DIR
  NAMES curand_kernel.h
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES include)

set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                     ${CURAND_KERNEL_INCLUDE_DIR})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# NVTX profiling support option (default: OFF)
option(ENABLE_NVTX_PROFILING "Enable NVTX profiling support" OFF)

if(ENABLE_NVTX_PROFILING)
  # Use bundled NVTX3 headers (v3.4.0-c-cpp) for consistency across all
  # platforms This ensures profiling support works on all targets
  # (GPU/Thor/Orin) with CUDA 11/12/13
  set(NVTX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdParty/NVTX/include/nvtx3")

  if(EXISTS "${NVTX_INCLUDE_DIR}/nvtx3.hpp")
    message(STATUS "NVTX profiling ENABLED: ${NVTX_INCLUDE_DIR}")
    # Define preprocessor macro for conditional compilation
    add_compile_definitions(ENABLE_NVTX_PROFILING)
  else()
    message(
      FATAL_ERROR
        "NVTX3 headers not found at ${NVTX_INCLUDE_DIR}. "
        "Please ensure 3rdParty/NVTX is initialized: git submodule update --init 3rdParty/NVTX"
    )
  endif()
else()
  message(
    STATUS "NVTX profiling DISABLED (use -DENABLE_NVTX_PROFILING=ON to enable)")
  set(NVTX_INCLUDE_DIR "")
endif()

# Set TensorRT header includes and library dependencies.
if(NOT DEFINED TRT_PACKAGE_DIR)
  message(
    FATAL_ERROR "Please specify the -DTRT_PACKAGE_DIR when invoking CMake.")
endif()

find_path(
  TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_path(ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h HINTS ${TRT_INCLUDE_DIR})
if(NOT ONNX_PARSER_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "NvOnnxParser.h not found in TensorRT headers, please specify the -DONNX_PARSER_INCLUDE_DIR when invoking CMake"
  )
endif()

find_library(
  NVINFER_LIB nvinfer
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_library(
  NV_ONNX_PARSER_LIB nvonnxparser
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 x86_64-linux-gnu)

# TRT and CUDA headers are treated as system includes so that the compiler
# suppresses warnings originating from them (e.g. -Wunused-parameter in
# NvInferRuntime.h virtual stubs).
include_directories(SYSTEM ${CUDA_INCLUDE_DIR} ${TRT_INCLUDE_DIR})

set(COMMON_INCLUDE_DIRS
    ${NVTX_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/examples/multimodal
    ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include
    ${CMAKE_SOURCE_DIR}/3rdParty/stb
    ${CMAKE_SOURCE_DIR}/3rdParty/cpp-httplib)

add_library(commonLibraryExt INTERFACE)
target_link_libraries(commonLibraryExt INTERFACE ${NVINFER_LIB} ${CUDART_LIB})
# Workaround for aarch64 cross compilation where libvninfer.so links to
# libnvdla_compilers. The libnvinfer dependencies shall be resolved at execution
# time on an embedded board.
target_link_options(commonLibraryExt INTERFACE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")

# Build Core libraries of the project. attentionPlugin shared library and
# edgellmCore/edgellmTokenizer/edgellmBuilder static library will be generated.
add_subdirectory(cpp)

# Build C++ examples to showcase the usage
add_subdirectory(examples)

option(BUILD_UNIT_TESTS "Enable building unit tests" OFF)

if(BUILD_UNIT_TESTS)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Disable Google Mock")
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/googletest)
  file(GLOB_RECURSE UNIT_TESTS_SRCS ${CMAKE_SOURCE_DIR}/unittests/*.cpp
       ${CMAKE_SOURCE_DIR}/unittests/*.cu)
  add_executable(unitTest ${UNIT_TESTS_SRCS})
  target_include_directories(unitTest PRIVATE ${COMMON_INCLUDE_DIRS}
                                              ${CMAKE_SOURCE_DIR})
  # Define project root path for tests to access resources
  target_compile_definitions(unitTest
                             PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(unitTest PRIVATE gtest gtest_main edgellmCore
                                         ${CUDART_LIB} ${CUDA_DRIVER_LIB})
  add_cross_build_link_options(unitTest)
endif()
