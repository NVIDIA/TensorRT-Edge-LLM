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

option(ENABLE_SAFETY_RUNTIME "Build the Edge-LLM Safety Runtime" OFF)
option(ENABLE_EXPERIMENTAL_WEIGHTLESS
       "Enable experimental internal TensorRT/Myelin weightless engine refit"
       OFF)
option(EDGELLM_SAFETY_ONLY_PACKAGE
       "Build only the Safety Runtime against a safety-only dependency package"
       OFF)
option(EDGELLM_PYTHON_ONLY_WHEEL
       "Configure only the Python sources for the wheel fan-in base" OFF)

if(ENABLE_EXPERIMENTAL_WEIGHTLESS AND ENABLE_SAFETY_RUNTIME)
  message(
    FATAL_ERROR
      "Experimental weightless engines require enterprise TensorRT, without Safety Runtime"
  )
endif()

if(EDGELLM_PYTHON_ONLY_WHEEL AND EDGELLM_SAFETY_ONLY_PACKAGE)
  message(
    FATAL_ERROR
      "EDGELLM_PYTHON_ONLY_WHEEL and EDGELLM_SAFETY_ONLY_PACKAGE are mutually exclusive"
  )
endif()

include(cmake/EdgeLLMPlatform.cmake)
if(NOT EDGELLM_PYTHON_ONLY_WHEEL)
  edgellm_prepare_toolchain()
endif()

if(EDGELLM_PYTHON_ONLY_WHEEL)
  set(_edgellm_languages NONE)
else()
  set(_edgellm_languages C CXX CUDA)
endif()

project(
  tensorrt_edgellm_sdk
  VERSION 0.12.0
  LANGUAGES ${_edgellm_languages})

if(EDGELLM_PYTHON_ONLY_WHEEL)
  return()
endif()
if(EDGELLM_SAFETY_ONLY_PACKAGE AND NOT ENABLE_SAFETY_RUNTIME)
  message(
    FATAL_ERROR "EDGELLM_SAFETY_ONLY_PACKAGE requires ENABLE_SAFETY_RUNTIME=ON")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/EdgeLLMBuild.cmake)
edgellm_configure_platform()

if(EDGELLM_SAFETY_ONLY_PACKAGE)
  find_path(
    CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES include)
  find_library(
    CUDART_LIB cudart
    HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
    PATH_SUFFIXES lib lib64)
  set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                       ${CUDA_PLATFORM_INCLUDE_DIR})
  include(cmake/SafetyRuntimeBuild.cmake)
  return()
endif()

option(ENABLE_COVERAGE "Enable gcov code coverage instrumentation" OFF)
option(ENABLE_NVTX_PROFILING "Enable NVTX profiling support" OFF)
option(BUILD_UNIT_TESTS "Enable building unit tests" OFF)
option(ENABLE_CUTEDSL_MODULE_TEST_HOOK
       "Enable deterministic CuTe DSL module-load failure injection" OFF)
option(BUILD_EXPERIMENTAL_MODELS "Enable building experimental models" OFF)
option(BUILD_PYTHON_BINDINGS "Enable building Python nanobind bindings" OFF)

if(ENABLE_CUTEDSL_MODULE_TEST_HOOK AND NOT BUILD_UNIT_TESTS)
  message(
    FATAL_ERROR "ENABLE_CUTEDSL_MODULE_TEST_HOOK requires BUILD_UNIT_TESTS=ON")
endif()

set(ENABLE_CUTE_DSL
    "ALL"
    CACHE STRING "CuTe DSL kernel groups; production builds use ALL")
if(QNX_BUILD)
  set(ENABLE_CUTE_DSL
      "OFF"
      CACHE STRING "CuTe DSL is unavailable on QNX" FORCE)
elseif(ENABLE_CUTE_DSL STREQUAL "OFF" AND NOT ENABLE_SAFETY_RUNTIME)
  message(FATAL_ERROR "CuTe DSL is required. Use ENABLE_CUTE_DSL=ALL.")
endif()

include(cmake/CodeCoverage.cmake)

if(ENABLE_NVTX_PROFILING)
  set(NVTX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdParty/NVTX/include/nvtx3")
  if(NOT EXISTS "${NVTX_INCLUDE_DIR}/nvtx3.hpp")
    message(FATAL_ERROR "NVTX headers are missing. Initialize git submodules.")
  endif()
  add_compile_definitions(ENABLE_NVTX_PROFILING)
else()
  set(NVTX_INCLUDE_DIR "")
endif()

include(cmake/EdgeLLMDependencies.cmake)

if(EMBEDDED_TARGET STREQUAL "jetson-orin")
  add_compile_definitions(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
endif()
if(ENABLE_EXPERIMENTAL_WEIGHTLESS AND IS_TRT_RTX)
  message(
    FATAL_ERROR
      "Experimental weightless engines require enterprise TensorRT, without Safety Runtime"
  )
endif()

include(cmake/GuidedDecoding.cmake)
add_subdirectory(cpp)
add_subdirectory(examples)

if(BUILD_EXPERIMENTAL_MODELS)
  add_subdirectory(experimental_models)
endif()

if(BUILD_UNIT_TESTS)
  set(BUILD_GMOCK
      ON
      CACHE BOOL "Enable Google Mock" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "Do not install test-only dependencies" FORCE)
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/googletest)
  enable_testing()
  add_subdirectory(unittests)
endif()

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(python/nanobind)
  if(EDGELLM_WHEEL_PAYLOAD_DIR)
    include(packaging/cmake/InstallWheel.cmake)
  endif()
endif()

if(NOT QNX_BUILD
   AND NOT ENABLE_SAFETY_RUNTIME
   AND NOT EDGELLM_WHEEL_PAYLOAD_DIR)
  include(cmake/EdgeLLMInstall.cmake)
  if(BUILD_UNIT_TESTS)
    add_dependencies(unitTests ${_edgellm_export_targets})
    # Not every program is built on every platform (llm_stream is not built on
    # Windows).
    foreach(_edgellm_program IN LISTS _edgellm_programs)
      if(TARGET ${_edgellm_program})
        add_dependencies(unitTests ${_edgellm_program})
      endif()
    endforeach()
  endif()
endif()
