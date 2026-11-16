# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(tensorrt_edgellm_cpp_examples LANGUAGES CXX CUDA)
  find_package(TensorRTEdgeLLM CONFIG REQUIRED)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

add_library(
  edgellmApplicationSupport STATIC
  common/audioWriter.cpp
  common/benchLogger.cpp
  common/memoryMonitor.cpp
  common/profileFormatter.cpp
  common/requestFileParser.cpp
  common/requestVocabulary.cpp
  common/runtimeArgs.cpp)
target_include_directories(
  edgellmApplicationSupport
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/common"
         $<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/common/windows>)
target_link_libraries(edgellmApplicationSupport PUBLIC TensorRTEdgeLLM::Core)
target_link_libraries(edgellmApplicationSupport PRIVATE ${CMAKE_DL_LIBS})
if(COMMAND edgellm_apply_qnx_warning_suppressions)
  edgellm_apply_qnx_warning_suppressions(edgellmApplicationSupport)
endif()

# Windows caps file paths at 260 characters unless the executable opts in to
# long paths in its manifest (the LongPathsEnabled system setting alone is not
# enough). Output and profile files named after long test cases exceed that
# limit.
if(WIN32)
  set(EDGELLM_LONG_PATH_MANIFEST
      "${CMAKE_CURRENT_BINARY_DIR}/edgellmLongPathAware.manifest")
  file(
    WRITE "${EDGELLM_LONG_PATH_MANIFEST}"
    [=[<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings xmlns:ws2="http://schemas.microsoft.com/SMI/2016/WindowsSettings">
      <ws2:longPathAware>true</ws2:longPathAware>
    </windowsSettings>
  </application>
</assembly>
]=])
endif()

function(edgellm_add_cpp_application target)
  add_executable(${target} ${target}.cpp)
  if(WIN32)
    target_sources(${target} PRIVATE "${EDGELLM_LONG_PATH_MANIFEST}")
  endif()
  target_link_libraries(${target} PRIVATE ${ARGN})
  target_include_directories(
    ${target}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common"
            $<$<PLATFORM_ID:Windows>:${CMAKE_CURRENT_SOURCE_DIR}/common/windows>
  )
  if(COMMAND add_cross_build_link_options)
    add_cross_build_link_options(${target})
  endif()
endfunction()

edgellm_add_cpp_application(llm_build TensorRTEdgeLLM::Builder)
edgellm_add_cpp_application(visual_build TensorRTEdgeLLM::Builder)
if(ENABLE_SAFETY_RUNTIME)
  foreach(target llm_build visual_build)
    target_compile_definitions(${target} PRIVATE ENABLE_SAFETY_RUNTIME=1)
    set_target_properties(
      ${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                           "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
  endforeach()
endif()
edgellm_add_cpp_application(audio_build TensorRTEdgeLLM::Builder)
edgellm_add_cpp_application(action_build TensorRTEdgeLLM::Builder)

edgellm_add_cpp_application(llm_inference TensorRTEdgeLLM::Core
                            edgellmApplicationSupport)
edgellm_add_cpp_application(llm_bench TensorRTEdgeLLM::Core
                            edgellmApplicationSupport)
add_custom_target(
  benchSourceRevision
  COMMAND
    ${CMAKE_COMMAND} "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DOUTPUT_HEADER=${CMAKE_CURRENT_BINARY_DIR}/benchSourceRevision.h" -P
    ${CMAKE_CURRENT_SOURCE_DIR}/benchSourceRevision.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/benchSourceRevision.h
  VERBATIM)
add_dependencies(llm_bench benchSourceRevision)
target_include_directories(llm_bench PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
edgellm_add_cpp_application(action_inference TensorRTEdgeLLM::Core
                            edgellmApplicationSupport)
edgellm_add_cpp_application(qwen3_tts_inference TensorRTEdgeLLM::Core
                            edgellmApplicationSupport)
edgellm_add_cpp_application(multistage_inference TensorRTEdgeLLM::Core
                            edgellmApplicationSupport)

if(NOT WIN32)
  edgellm_add_cpp_application(llm_stream TensorRTEdgeLLM::Core
                              edgellmApplicationSupport)
endif()

set(_edgellm_runtime_applications llm_inference llm_bench action_inference
                                  qwen3_tts_inference multistage_inference)
if(TARGET llm_stream)
  list(APPEND _edgellm_runtime_applications llm_stream)
endif()
set_target_properties(${_edgellm_runtime_applications}
                      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(NOT MSVC)
  target_compile_options(llm_bench PRIVATE -Wno-psabi)
endif()

option(ENABLE_MULTI_DEVICE
       "Enable multi-device inference using runtime-loaded NCCL" OFF)
option(ENABLE_MULTI_DEVICE_MPI
       "Enable MPI rank-launch support for multi-device inference" OFF)

if(WIN32 AND ENABLE_MULTI_DEVICE)
  message(FATAL_ERROR "Multi-device inference is not supported on Windows.")
endif()

if(ENABLE_MULTI_DEVICE)
  target_compile_definitions(llm_inference PRIVATE EDGELLM_ENABLE_MULTI_DEVICE)
  target_compile_definitions(action_inference
                             PRIVATE EDGELLM_ENABLE_MULTI_DEVICE)

  if(ENABLE_MULTI_DEVICE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(llm_inference
                               PRIVATE EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    target_compile_definitions(action_inference
                               PRIVATE EDGELLM_ENABLE_MULTI_DEVICE_MPI)
    target_link_libraries(llm_inference PRIVATE MPI::MPI_CXX)
    target_link_libraries(action_inference PRIVATE MPI::MPI_CXX)
    message(STATUS "Multi-device support ENABLED (MPI)")
  else()
    message(STATUS "Multi-device support ENABLED (threaded)")
  endif()
endif()
