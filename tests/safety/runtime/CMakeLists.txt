# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(safety_runtime_host_tests LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(EDGELLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
                       ABSOLUTE)
set(SAFETY_SOURCE_DIR
    "${EDGELLM_ROOT}/cpp/safety_inference"
    CACHE PATH "Safety implementation under test")
find_path(
  SAFETY_HEADERS NvInferSafeRuntime.h
  HINTS "${TRT_PACKAGE_DIR}"
  PATH_SUFFIXES include REQUIRED)
find_path(
  SAFETY_BASE_HEADERS NvInferRuntimeBase.h
  HINTS "${TRT_SAFE_BASE_INCLUDE_DIR}" "${SAFETY_HEADERS}"
  NO_DEFAULT_PATH REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory("${EDGELLM_ROOT}/3rdParty/googletest" googletest
                 EXCLUDE_FROM_ALL)
add_executable(
  safetyRuntimeHostTests
  safetyRuntimeHostTests.cpp
  requestRunnerIntegrationTests.cpp
  internvl3ProcessingCapabilityTests.cpp
  safetyPublicApiTests.cpp
  safetyRuntimeTestDoubles.cpp
  ${EDGELLM_ROOT}/unittests/cpp/safety/declaredResourceTests.cpp
  ${SAFETY_SOURCE_DIR}/runtime/error_handler/safetyErrorRecorder.cpp
  ${SAFETY_SOURCE_DIR}/runtime/safetyStatusUtils.cpp
  ${SAFETY_SOURCE_DIR}/runtime/safetyRuntime.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/llm/llmModelLocalData.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/modelInference.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/modelInferenceCatalog.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/internvl3/internvl3VanillaInference.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/llm/llmRequestContext.cpp
  ${SAFETY_SOURCE_DIR}/runtime/model_inference/multimodal/vlmRequestContext.cpp
  ${SAFETY_SOURCE_DIR}/runtime/processing/processingCapabilityCoordinator.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/declaredResourceManager.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/runtimeResource.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/containerStorage.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/estd/estdAllocator.cpp
  ${SAFETY_SOURCE_DIR}/trt_runner/safetyTensorView.cpp
  ${SAFETY_SOURCE_DIR}/trt_runner/runnerResourceResolver.cpp
  ${SAFETY_SOURCE_DIR}/trt_runner/trtSafetyGraphExecutor.cpp
  ${SAFETY_SOURCE_DIR}/trt_runner/trtSafetyRunner.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/trtResourceManager.cpp)
target_sources(
  safetyRuntimeHostTests
  PRIVATE
    ${SAFETY_SOURCE_DIR}/runtime/processing/singleRequestTextResponse.cpp
    ${SAFETY_SOURCE_DIR}/tokenizer/byteLevelBpeTokenizer.cpp
    ${SAFETY_SOURCE_DIR}/tokenizer/models/internvl3/internvl3TokenizerCapability.cpp
)
get_filename_component(SAFETY_SOURCE_PARENT "${SAFETY_SOURCE_DIR}" DIRECTORY)
target_include_directories(
  safetyRuntimeHostTests PRIVATE "${SAFETY_SOURCE_PARENT}"
                                 "${EDGELLM_ROOT}/cpp")
target_include_directories(
  safetyRuntimeHostTests SYSTEM
  PRIVATE "${SAFETY_HEADERS}" "${SAFETY_BASE_HEADERS}"
          "${EDGELLM_ROOT}/3rdParty/nlohmannJson/single_include"
          ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(safetyRuntimeHostTests PRIVATE -Wall -Wextra -Werror
                                                      -Wno-unused-parameter)
target_link_libraries(safetyRuntimeHostTests PRIVATE gtest_main CUDA::cudart)
add_test(NAME safetyRuntimeHostContracts COMMAND safetyRuntimeHostTests)

add_executable(
  internvl3ProcessingCapabilityTests
  internvl3ProcessingCapabilityTests.cpp
  ${EDGELLM_ROOT}/unittests/cpp/safety/declaredResourceTests.cpp
  ${SAFETY_SOURCE_DIR}/runtime/safetyStatusUtils.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/declaredResourceManager.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/runtimeResource.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/containerStorage.cpp
  ${SAFETY_SOURCE_DIR}/runtime/resources/estd/estdAllocator.cpp
  ${SAFETY_SOURCE_DIR}/runtime/processing/singleRequestTextResponse.cpp
  ${SAFETY_SOURCE_DIR}/tokenizer/byteLevelBpeTokenizer.cpp
  ${SAFETY_SOURCE_DIR}/tokenizer/models/internvl3/internvl3TokenizerCapability.cpp
)
target_include_directories(
  internvl3ProcessingCapabilityTests PRIVATE "${SAFETY_SOURCE_PARENT}"
                                             "${EDGELLM_ROOT}/cpp")
target_include_directories(
  internvl3ProcessingCapabilityTests SYSTEM
  PRIVATE "${SAFETY_HEADERS}" "${SAFETY_BASE_HEADERS}"
          "${EDGELLM_ROOT}/3rdParty/nlohmannJson/single_include"
          ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(internvl3ProcessingCapabilityTests
                       PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)
target_link_libraries(internvl3ProcessingCapabilityTests PRIVATE gtest_main
                                                                 CUDA::cudart)
add_test(NAME internvl3ProcessingCapabilityContracts
         COMMAND internvl3ProcessingCapabilityTests)
