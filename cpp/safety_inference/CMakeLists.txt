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

# Keep every production source explicit. Subsequent tasks add reviewed sources
# to the owning area instead of sweeping Standard sources in with a glob.
set(SAFETY_RUNTIME_CONTROL_SRCS runtime/safetyRuntime.cpp)
set(SAFETY_MODEL_INFERENCE_SRCS
    runtime/model_inference/modelInference.cpp
    runtime/model_inference/modelInferenceCatalog.cpp
    runtime/model_inference/internvl3/internvl3VanillaInference.cpp
    runtime/model_inference/llm/llmRequestContext.cpp
    runtime/model_inference/llm/llmModelLocalData.cpp
    runtime/model_inference/multimodal/vlmRequestContext.cpp)
set(SAFETY_RESOURCE_SRCS
    runtime/resources/runtimeResource.cpp
    runtime/resources/declaredResourceManager.cpp
    runtime/resources/containerStorage.cpp
    runtime/resources/estd/estdAllocator.cpp
    runtime/resources/trtResourceManager.cpp)
set(SAFETY_ERROR_HANDLER_SRCS runtime/safetyStatusUtils.cpp
                              runtime/error_handler/safetyErrorRecorder.cpp)
set(SAFETY_TRT_EXECUTION_SRCS
    trt_runner/safetyGraphInitialization.cpp trt_runner/safetyTensorView.cpp
    trt_runner/runnerResourceResolver.cpp trt_runner/trtSafetyGraphExecutor.cpp
    trt_runner/trtSafetyRunner.cpp)
set(SAFETY_RUNTIME_CAPABILITY_SRCS
    runtime/safetyRequestProcessing.cpp
    runtime/processing/processingCapabilityCoordinator.cpp
    runtime/processing/singleRequestTextResponse.cpp
    runtime/processing/models/internvl3/internvl3RequestProcessing.cpp
    tokenizer/byteLevelBpeTokenizer.cpp
    tokenizer/models/internvl3/internvl3TokenizerCapability.cpp
    image_preprocess/models/internvl3/internvl3MultimodalPreprocessorCapability.cpp
    image_preprocess/models/internvl3/internvl3PreprocessKernels.cu
    image_preprocess/rgbResizeNormalizeKernels.cu
    embedding/models/internvl3/internvl3EmbeddingCapability.cpp
    embedding/fp16EmbeddingKernels.cu
    embedding/ropeCache.cpp
    sampler/greedyArgmaxKernel.cu
    sampler/greedySamplerCapability.cu)

if(NOT TARGET TensorRT::TensorRTSafe)
  message(FATAL_ERROR "edgellmSafetyRuntime requires TensorRT::TensorRTSafe")
endif()
if(NOT CUDART_LIB)
  message(FATAL_ERROR "edgellmSafetyRuntime requires the CUDA runtime library")
endif()
find_package(Threads REQUIRED)

add_library(
  edgellmSafetyRuntime STATIC
  ${SAFETY_RUNTIME_CONTROL_SRCS} ${SAFETY_MODEL_INFERENCE_SRCS}
  ${SAFETY_RESOURCE_SRCS} ${SAFETY_ERROR_HANDLER_SRCS}
  ${SAFETY_TRT_EXECUTION_SRCS} ${SAFETY_RUNTIME_CAPABILITY_SRCS})
target_include_directories(edgellmSafetyRuntime PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
target_include_directories(
  edgellmSafetyRuntime SYSTEM
  PUBLIC ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/single_include)
target_include_directories(edgellmSafetyRuntime SYSTEM
                           PRIVATE ${CUDA_INCLUDE_DIR})
set_target_properties(
  edgellmSafetyRuntime
  PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
             CUDA_RUNTIME_LIBRARY None
             CUDA_VISIBILITY_PRESET hidden)
target_compile_definitions(edgellmSafetyRuntime PRIVATE ENABLE_SAFETY_RUNTIME=1)
if(QNX_BUILD AND QNX_SAFE_CUDA)
  target_compile_definitions(edgellmSafetyRuntime
                             PRIVATE EDGELLM_USE_SAFE_CUDA=1)
endif()
target_link_libraries(
  edgellmSafetyRuntime PRIVATE TensorRT::TensorRTSafe ${CUDART_LIB}
                               NvInfer_edgellm_plugin_safe Threads::Threads)
add_cross_build_link_options(edgellmSafetyRuntime)
edgellm_apply_qnx_warning_suppressions(edgellmSafetyRuntime)
edgellm_set_hidden_visibility(edgellmSafetyRuntime)

set(EDGELLM_SAFETY_CUDA_ARCHITECTURES
    "${CMAKE_CUDA_ARCHITECTURES}"
    CACHE STRING "Admitted CUDA architectures for the Safety plugins")
add_library(
  NvInfer_edgellm_plugin_safe SHARED
  plugins/safetyAttentionContract.cpp plugins/safetyAttentionPlugin.cpp
  plugins/safetyPluginRegistry.cpp kernels/safetyAttentionKernels.cu
  kernels/safetyAttentionModules.cpp)
target_include_directories(NvInfer_edgellm_plugin_safe SYSTEM
                           PRIVATE ${CUDA_INCLUDE_DIR})
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3
   AND (NOT QNX_BUILD OR NOT QNX_SAFE_CUDA))
  target_compile_options(NvInfer_edgellm_plugin_safe
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-legacy-launch-seq>)
endif()
find_library(
  EDGELLM_SAFETY_CUDA_DRIVER cuda
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs REQUIRED)
set(_safety_driver "${EDGELLM_SAFETY_CUDA_DRIVER}")
if(AARCH64_BUILD)
  set(_safety_driver cuda)
endif()
target_link_libraries(
  NvInfer_edgellm_plugin_safe PRIVATE TensorRT::TensorRTSafe ${CUDART_LIB}
                                      ${_safety_driver})
include(${CMAKE_SOURCE_DIR}/cmake/SafetyAttentionArtifacts.cmake)
edgellm_admit_safety_fmha(NvInfer_edgellm_plugin_safe
                          "${EDGELLM_SAFETY_CUDA_ARCHITECTURES}")
set_target_properties(
  NvInfer_edgellm_plugin_safe
  PROPERTIES CUDA_ARCHITECTURES "${EDGELLM_SAFETY_CUDA_ARCHITECTURES}"
             CUDA_RUNTIME_LIBRARY None
             CUDA_VISIBILITY_PRESET hidden
             VERSION 1.0
             SOVERSION 1)
if(NOT EDGELLM_SAFETY_ONLY_PACKAGE)
  target_compile_definitions(NvInfer_edgellm_plugin_safe
                             PRIVATE EDGELLM_SAFETY_PLUGIN_BUILDER=1)
endif()
add_cross_build_link_options(NvInfer_edgellm_plugin_safe)
edgellm_apply_qnx_warning_suppressions(NvInfer_edgellm_plugin_safe)
edgellm_set_hidden_visibility(NvInfer_edgellm_plugin_safe)

# A static archive alone does not establish final-link dependency closure.
add_executable(safetyPluginLinkCheck
               ${CMAKE_SOURCE_DIR}/tests/safety/safetyPluginLinkCheck.cpp)
target_include_directories(safetyPluginLinkCheck SYSTEM
                           PRIVATE ${CUDA_INCLUDE_DIR})
target_include_directories(safetyPluginLinkCheck
                           PRIVATE "${CMAKE_SOURCE_DIR}/cpp")
target_link_libraries(
  safetyPluginLinkCheck PRIVATE edgellmSafetyRuntime TensorRT::TensorRTSafe
                                ${_safety_driver})
add_cross_build_link_options(safetyPluginLinkCheck)

option(EDGELLM_BUILD_SAFETY_ENGINE_TESTS
       "Build real Safety engine construction and execution probes" OFF)
if(EDGELLM_BUILD_SAFETY_ENGINE_TESTS)
  add_executable(
    safetyRuntimeErrorHandlerTests
    ${CMAKE_SOURCE_DIR}/tests/safety/safetyErrorHandlerTests.cpp
    ${CMAKE_SOURCE_DIR}/tests/safety/safetyErrorRecorderTestUtils.cpp
    ${CMAKE_SOURCE_DIR}/tests/safety/safetyStatusTestUtils.cpp)
  target_include_directories(safetyRuntimeErrorHandlerTests SYSTEM
                             PRIVATE ${CUDA_INCLUDE_DIR})
  target_include_directories(safetyRuntimeErrorHandlerTests
                             PRIVATE "${CMAKE_SOURCE_DIR}/cpp")
  target_link_libraries(
    safetyRuntimeErrorHandlerTests
    PRIVATE edgellmSafetyRuntime NvInfer_edgellm_plugin_safe
            TensorRT::TensorRTSafe ${_safety_driver})
  if(QNX_BUILD)
    target_include_directories(safetyRuntimeErrorHandlerTests SYSTEM
                               PRIVATE "${QNX_SAFE_PDK_ROOT}/include")
    target_link_libraries(safetyRuntimeErrorHandlerTests PRIVATE nvdvms_client)
  endif()
  add_cross_build_link_options(safetyRuntimeErrorHandlerTests)
  edgellm_apply_qnx_warning_suppressions(safetyRuntimeErrorHandlerTests)

  foreach(_probe safetyEngineSmoke safetyRunnerSmoke)
    add_executable(${_probe} ${CMAKE_SOURCE_DIR}/tests/safety/${_probe}.cpp)
    target_include_directories(${_probe} SYSTEM PRIVATE ${CUDA_INCLUDE_DIR})
    target_include_directories(${_probe} PRIVATE "${CMAKE_SOURCE_DIR}/cpp")
    target_link_libraries(
      ${_probe} PRIVATE edgellmSafetyRuntime NvInfer_edgellm_plugin_safe
                        TensorRT::TensorRTSafe ${CUDART_LIB} ${_safety_driver})
    if(QNX_BUILD AND QNX_SAFE_CUDA)
      target_include_directories(${_probe} SYSTEM
                                 PRIVATE "${QNX_SAFE_PDK_ROOT}/include")
      target_link_libraries(${_probe} PRIVATE nvdvms_client)
    elseif(QNX_BUILD)
      target_compile_definitions(${_probe} PRIVATE EDGELLM_TEST_QNX_STANDARD=1)
    endif()
    add_cross_build_link_options(${_probe})
    edgellm_apply_qnx_warning_suppressions(${_probe})
  endforeach()

  if(NOT EDGELLM_SAFETY_ONLY_PACKAGE)
    add_executable(safetyEngineBuild
                   ${CMAKE_SOURCE_DIR}/tests/safety/safetyEngineBuild.cpp)
    target_include_directories(safetyEngineBuild SYSTEM
                               PRIVATE ${CUDA_INCLUDE_DIR})
    target_include_directories(safetyEngineBuild
                               PRIVATE "${CMAKE_SOURCE_DIR}/cpp")
    target_link_libraries(
      safetyEngineBuild
      PRIVATE NvInfer_edgellm_plugin_safe TensorRT::TensorRT
              TensorRT::TensorRTSafe ${CUDART_LIB} ${_safety_driver})
    add_cross_build_link_options(safetyEngineBuild)
    edgellm_apply_qnx_warning_suppressions(safetyEngineBuild)
  endif()
endif()

if(AARCH64_BUILD)
  # Deployment supplies its admitted library path, not build-container paths.
  foreach(
    _target
    NvInfer_edgellm_plugin_safe safetyPluginLinkCheck
    safetyRuntimeErrorHandlerTests safetyEngineSmoke safetyRunnerSmoke
    safetyEngineBuild)
    if(TARGET ${_target})
      set_target_properties(${_target} PROPERTIES SKIP_BUILD_RPATH TRUE)
    endif()
  endforeach()
endif()
