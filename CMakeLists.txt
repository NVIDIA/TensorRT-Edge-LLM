# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

option(EDGELLM_PYTHON_ONLY_WHEEL
       "Configure only the Python sources for the wheel fan-in base" OFF)
if(EDGELLM_PYTHON_ONLY_WHEEL)
  set(_EDGELLM_PROJECT_LANGUAGES NONE)
else()
  set(_EDGELLM_PROJECT_LANGUAGES C CXX CUDA)
endif()

project(
  tensorrt_edgellm_sdk
  VERSION 0.10.1
  LANGUAGES ${_EDGELLM_PROJECT_LANGUAGES})

if(EDGELLM_PYTHON_ONLY_WHEEL)
  return()
endif()

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

function(edgellm_apply_qnx_warning_suppressions target)
  if(QNX_BUILD)
    # Suppress QNX 8 system-header false positives on affected project targets.
    set(_qnx_warning_suppressions
        -Wno-array-bounds
        -Wno-maybe-uninitialized
        -Wno-dangling-pointer
        -Wno-free-nonheap-object
        -Wno-stringop-overflow
        -Wno-return-type
        -Wno-stringop-overread)
    target_compile_options(
      ${target}
      PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${_qnx_warning_suppressions}>)

    set(_qnx_cuda_warning_suppressions ${_qnx_warning_suppressions})
    list(TRANSFORM _qnx_cuda_warning_suppressions PREPEND "-Xcompiler=")
    target_compile_options(
      ${target}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${_qnx_cuda_warning_suppressions}>)
  endif()
endfunction()

# Code coverage with gcov (for SonarQube analysis)
option(ENABLE_COVERAGE "Enable gcov code coverage instrumentation" OFF)
include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)

# Declare unit-test options before cpp/ so its library targets can enable
# test-only instrumentation when requested.
option(BUILD_UNIT_TESTS "Enable building unit tests" OFF)
option(ENABLE_CUTEDSL_MODULE_TEST_HOOK
       "Enable deterministic CuTe DSL module-load failure injection" OFF)
if(ENABLE_CUTEDSL_MODULE_TEST_HOOK AND NOT BUILD_UNIT_TESTS)
  message(
    FATAL_ERROR "ENABLE_CUTEDSL_MODULE_TEST_HOOK requires BUILD_UNIT_TESTS=ON")
endif()

macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

macro(add_cross_build_link_options target)
  if(AARCH64_BUILD)
    get_target_property(_cross_link_options_added ${target}
                        TRT_EDGELLM_CROSS_LINK_OPTIONS_ADDED)
    if(NOT _cross_link_options_added)
      get_target_property(_cross_target_type ${target} TYPE)
      set(_cross_link_scope PRIVATE)
      if(_cross_target_type STREQUAL "STATIC_LIBRARY")
        set(_cross_link_scope INTERFACE)
      endif()
      target_link_options(${target} ${_cross_link_scope} "-L${CUDA_DIR}/lib"
                          "-L${CUDA_DIR}/lib/stubs")
      set_property(TARGET ${target}
                   PROPERTY TRT_EDGELLM_CROSS_LINK_OPTIONS_ADDED TRUE)
    endif()
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

# Wheel payloads are single-SM builds; preserve the caller's architecture.
if(NOT DEFINED EDGELLM_WHEEL_PAYLOAD_DIR AND NOT AARCH64_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)
  if(CUDA_CTK_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100a 120)
  endif()
endif()

# Orin/JetPack cuda.h may lack cudaLibrary_t; GDN runner uses compat typedef
# when this is set.
if(DEFINED EMBEDDED_TARGET AND EMBEDDED_TARGET STREQUAL "jetson-orin")
  add_compile_definitions(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
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
set(CUDA_DRIVER_LINK_LIB "${CUDA_DRIVER_LIB}")
if(AARCH64_BUILD)
  # Cross builds link against the CUDA driver stub via -L.../lib/stubs -lcuda.
  # Do not pass the absolute stub .so to target_link_libraries(), or CMake will
  # bake the stub directory into build RUNPATH and target execution can load it.
  set(CUDA_DRIVER_LINK_LIB cuda)
endif()

find_path(
  CURAND_KERNEL_INCLUDE_DIR
  NAMES curand_kernel.h
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES include)

set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                     ${CURAND_KERNEL_INCLUDE_DIR} ${CUDA_PLATFORM_INCLUDE_DIR})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include(${CMAKE_SOURCE_DIR}/cmake/FindNVRTC.cmake)

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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED COMPONENTS OnnxParser)

# TRT and CUDA headers are treated as system includes so that the compiler
# suppresses warnings originating from them (e.g. -Wunused-parameter in
# NvInferRuntime.h virtual stubs).
include_directories(SYSTEM ${CUDA_INCLUDE_DIR} ${TensorRT_INCLUDE_DIR})

set(COMMON_INCLUDE_DIRS
    ${NVTX_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/examples/multimodal
    ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include
    ${CMAKE_SOURCE_DIR}/3rdParty/stb
    ${CMAKE_SOURCE_DIR}/3rdParty/miniaudio)

add_library(commonLibraryExt INTERFACE)
target_link_libraries(
  commonLibraryExt INTERFACE TensorRT::TensorRT TensorRT::OnnxParser
                             ${CUDART_LIB})
# Workaround for aarch64 cross compilation where libvninfer.so links to
# libnvdla_compilers. The libnvinfer dependencies shall be resolved at execution
# time on an embedded board.
target_link_options(commonLibraryExt INTERFACE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")

# CuTe DSL unified kernel option — declare here so it is always defined before
# any subdirectory or include references it.
# cmake-format: off
# Choices:
#   fmha             -- FP16 Context/ViT FMHA, plus the optimized Blackwell
#                       variants when the artifact has them
#   ALL              -- enable every kernel group below
#   gdn              -- gated delta net (Mamba2-style hybrid SSM)
#   f16_moe          -- target-specific FP16 grouped MoE for Ampere,
#                       Blackwell DC, and Blackwell GeForce
#   nvfp4_moe        -- SM110 (Thor) split FC1/FC2 NVFP4 MoE; consumed by the
#                       SM110-only ``Nvfp4MoePlugin``
#   nvfp4_fused_moe  -- SM12x (consumer Blackwell / Spark GB10) fused NVFP4
#                       MoE; consumed by ``NvFP4MoEPluginGeforce``
#   rmsnorm          -- FP16/BF16 RMSNorm for supported production hidden sizes
#   semicolon-separated group list (e.g. -DENABLE_CUTE_DSL=fmha;gdn)
# cmake-format: on
set(ENABLE_CUTE_DSL
    "fmha"
    CACHE
      STRING
      "CuTe DSL kernels: fmha (default), ALL, or semicolon-separated group list (fmha;gdn;f16_moe;rmsnorm)"
)

# Build Core libraries of the project. attentionPlugin shared library and
# edgellmCore/edgellmBuilder static library will be generated.
add_subdirectory(cpp)

# Build C++ examples to showcase the usage
add_subdirectory(examples)

# Experimental models (opt-in). These are not first-class models; they live in a
# separate tree and link against the core edgellm libraries. Cosmos3 is the
# first.
option(BUILD_EXPERIMENTAL_MODELS
       "Enable building experimental models (e.g. cosmos3)" OFF)
if(BUILD_EXPERIMENTAL_MODELS)
  add_subdirectory(experimental_models)
endif()

option(BUILD_PYTHON_BINDINGS "Enable building Python pybind11 bindings" OFF)

if(BUILD_UNIT_TESTS)
  # FORCE, because a build tree configured before gmock was enabled already
  # caches OFF, and a plain `set(... CACHE ...)` would leave it there — the
  # failure is then a missing gmock/gmock.h, not a stale-cache message.
  set(BUILD_GMOCK
      ON
      CACHE BOOL "Enable Google Mock" FORCE)
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/googletest)
  include(${CMAKE_SOURCE_DIR}/cmake/CuteDsl.cmake)
  enable_testing()
  # Targets, grouping and per-target wiring all live in
  # unittests/CMakeLists.txt.
  add_subdirectory(unittests)
endif()

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(experimental/pybind)
  if(DEFINED EDGELLM_WHEEL_PAYLOAD_DIR AND NOT EDGELLM_WHEEL_PAYLOAD_DIR
                                           STREQUAL "")
    include(${CMAKE_SOURCE_DIR}/packaging/cmake/InstallWheel.cmake)
  endif()
endif()
