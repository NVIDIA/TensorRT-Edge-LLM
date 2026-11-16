# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

cmake_minimum_required(VERSION 3.20)
project(safety_plugin_tests LANGUAGES CXX CUDA)
option(EDGELLM_TEST_QNX_STANDARD "Run functional tests with standard QNX CUDA"
       OFF)
option(EDGELLM_TEST_QNX_SAFE
       "Build externally coordinated QNX Safety kernel smoke" OFF)
if(EDGELLM_TEST_QNX_SAFE)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX"
     OR EDGELLM_TEST_QNX_STANDARD
     OR NOT EDGELLM_SAFETY_ONLY_PACKAGE
     OR NOT QNX_SAFE_PDK_ROOT)
    message(
      FATAL_ERROR
        "Safe tests require QNX, Safety-only toolchain settings and the matching Safety PDK"
    )
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  if(NOT EDGELLM_TEST_QNX_STANDARD
     OR EDGELLM_SAFETY_ONLY_PACKAGE
     OR QNX_SAFE_PDK_ROOT
     OR QNX_CUDA_TARGET_ROOT MATCHES "safe")
    message(FATAL_ERROR "Select EDGELLM_TEST_QNX_SAFE for QNX Safety")
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(Threads REQUIRED)
get_filename_component(EDGELLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                       ABSOLUTE)
set(SAFETY_SOURCE_DIR "${EDGELLM_ROOT}/cpp/safety_inference")
find_path(
  SAFETY_HEADERS NvInferSafePlugin.h
  HINTS "${TRT_PACKAGE_DIR}"
  PATH_SUFFIXES include REQUIRED)
if(EDGELLM_TEST_QNX_SAFE)
  set(CUDAToolkit_INCLUDE_DIRS "${CUDA_DIR}/include")
  find_library(
    _test_cudart cudart
    PATHS "${CUDA_DIR}/lib"
    NO_DEFAULT_PATH REQUIRED)
  find_library(
    _test_driver cuda
    PATHS "${CUDA_DIR}/lib/stubs"
    NO_DEFAULT_PATH REQUIRED)
  set(_test_cuda_libraries "${_test_cudart}" "${_test_driver}")
else()
  find_package(CUDAToolkit REQUIRED)
  set(_test_cuda_libraries CUDA::cudart CUDA::cuda_driver)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "QNX"
   AND NOT EDGELLM_TEST_QNX_SAFE
   AND CUDAToolkit_INCLUDE_DIRS MATCHES "safe")
  message(FATAL_ERROR "These tests require standard QNX CUDA, not QNX Safety")
endif()
add_library(
  safetyPluginUnderTest SHARED
  ${SAFETY_SOURCE_DIR}/plugins/safetyAttentionContract.cpp
  ${SAFETY_SOURCE_DIR}/plugins/safetyAttentionPlugin.cpp
  ${SAFETY_SOURCE_DIR}/plugins/safetyPluginRegistry.cpp
  ${SAFETY_SOURCE_DIR}/kernels/safetyAttentionKernels.cu
  ${SAFETY_SOURCE_DIR}/kernels/safetyAttentionModules.cpp)
target_include_directories(safetyPluginUnderTest PUBLIC "${EDGELLM_ROOT}/cpp")
target_include_directories(
  safetyPluginUnderTest SYSTEM PUBLIC "${SAFETY_HEADERS}"
                                      ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(safetyPluginUnderTest PUBLIC ${_test_cuda_libraries})
if(NOT EDGELLM_TEST_QNX_SAFE)
  target_compile_definitions(safetyPluginUnderTest
                             PRIVATE EDGELLM_SAFETY_PLUGIN_BUILDER=1)
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
  target_compile_options(safetyPluginUnderTest
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-legacy-launch-seq>)
endif()
target_compile_options(
  safetyPluginUnderTest
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-Wno-unused-parameter>
)
include("${EDGELLM_ROOT}/cmake/SafetyAttentionArtifacts.cmake")
edgellm_admit_safety_fmha(safetyPluginUnderTest "${CMAKE_CUDA_ARCHITECTURES}")
add_executable(safetyKernelSmoke safetyKernelSmoke.cpp)
target_link_libraries(safetyKernelSmoke PRIVATE safetyPluginUnderTest)
add_executable(
  internvl3ProcessingKernelSmoke
  internvl3ProcessingKernelSmoke.cu
  ${SAFETY_SOURCE_DIR}/embedding/fp16EmbeddingKernels.cu
  ${SAFETY_SOURCE_DIR}/image_preprocess/rgbResizeNormalizeKernels.cu
  ${SAFETY_SOURCE_DIR}/image_preprocess/models/internvl3/internvl3PreprocessKernels.cu
  ${SAFETY_SOURCE_DIR}/sampler/greedyArgmaxKernel.cu)
target_include_directories(internvl3ProcessingKernelSmoke
                           PRIVATE "${EDGELLM_ROOT}/cpp")
target_include_directories(internvl3ProcessingKernelSmoke SYSTEM
                           PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(internvl3ProcessingKernelSmoke
                      PRIVATE ${_test_cuda_libraries})
if(EDGELLM_TEST_QNX_SAFE)
  target_include_directories(safetyKernelSmoke SYSTEM
                             PRIVATE "${QNX_SAFE_PDK_ROOT}/include")
  target_link_libraries(safetyKernelSmoke PRIVATE nvdvms_client)
  target_include_directories(internvl3ProcessingKernelSmoke SYSTEM
                             PRIVATE "${QNX_SAFE_PDK_ROOT}/include")
  target_link_libraries(internvl3ProcessingKernelSmoke PRIVATE nvdvms_client)
elseif(EDGELLM_TEST_QNX_STANDARD)
  target_compile_definitions(safetyKernelSmoke
                             PRIVATE EDGELLM_TEST_QNX_STANDARD=1)
  target_compile_definitions(internvl3ProcessingKernelSmoke
                             PRIVATE EDGELLM_TEST_QNX_STANDARD=1)
else()
  add_test(NAME safetyKernelSmoke COMMAND safetyKernelSmoke kernel-smoke.json)
  add_test(NAME internvl3ProcessingKernelSmoke
           COMMAND internvl3ProcessingKernelSmoke)
endif()
install(
  TARGETS safetyKernelSmoke internvl3ProcessingKernelSmoke safetyPluginUnderTest
  RUNTIME DESTINATION . COMPONENT safety-plugin-tests
  LIBRARY DESTINATION . COMPONENT safety-plugin-tests)
# Native tests intentionally exercise terminal failures and per-test RAII
# teardown.
if(EDGELLM_TEST_QNX_SAFE)
  return()
endif()
add_subdirectory("${EDGELLM_ROOT}/3rdParty/googletest" googletest
                 EXCLUDE_FROM_ALL)
set(_safety_plugin_test_sources safetyPluginTests.cpp safetyRopeTests.cu
                                safetyPluginTestMain.cpp)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND _safety_plugin_test_sources safetyErrorRecorderTests.cpp)
endif()
add_executable(
  safetyPluginTests
  ${_safety_plugin_test_sources}
  ${SAFETY_SOURCE_DIR}/runtime/error_handler/safetyErrorRecorder.cpp
  ${SAFETY_SOURCE_DIR}/runtime/safetyStatusUtils.cpp
  ${SAFETY_SOURCE_DIR}/trt_runner/safetyGraphInitialization.cpp)
target_link_libraries(safetyPluginTests PRIVATE safetyPluginUnderTest gtest
                                                Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  target_compile_definitions(safetyPluginTests
                             PRIVATE EDGELLM_TEST_QNX_STANDARD=1)
  target_link_libraries(safetyPluginTests PRIVATE regex)
endif()
install(
  TARGETS safetyPluginTests
  RUNTIME DESTINATION . COMPONENT safety-plugin-tests
  LIBRARY DESTINATION . COMPONENT safety-plugin-tests)
add_test(NAME safetyPluginContracts COMMAND safetyPluginTests
                                            --gtest_filter=-SafetyGpu*)
set_tests_properties(
  safetyPluginContracts
  PROPERTIES ENVIRONMENT
             "GTEST_OUTPUT=xml:${CMAKE_CURRENT_BINARY_DIR}/contracts.xml"
             TIMEOUT 60)
add_test(NAME safetyPluginGpu COMMAND safetyPluginTests
                                      --gtest_filter=SafetyGpu*)
add_executable(
  safetyModuleLifecycleTests
  safetyModuleLifecycleTests.cpp
  ${SAFETY_SOURCE_DIR}/kernels/safetyAttentionModules.cpp)
# Driver doubles exercise Blackwell host lifecycle on the SM80 CI host too.
target_compile_definitions(safetyModuleLifecycleTests
                           PRIVATE EDGELLM_SAFETY_BLACKWELL=1)
target_include_directories(safetyModuleLifecycleTests
                           PRIVATE "${EDGELLM_ROOT}/cpp")
target_include_directories(safetyModuleLifecycleTests SYSTEM
                           PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
edgellm_admit_safety_fmha(safetyModuleLifecycleTests
                          "${CMAKE_CUDA_ARCHITECTURES}")
foreach(
  mode
  success
  load_failure
  prepare_failure
  attribute_failure
  wrong_order
  repeated_load
  launch_failure
  teardown_failure
  bindings
  map_failure
  device_query_failure
  owner_isolation
  loaded_teardown)
  add_test(NAME safetyModule_${mode} COMMAND safetyModuleLifecycleTests ${mode})
endforeach()
