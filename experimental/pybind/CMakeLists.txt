# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. SPDX-License-Identifier: Apache-2.0
#
# Python bindings (_edgellm_runtime) for TensorRT Edge-LLM.
#
# Two build modes: A) Subdirectory of top-level project
# (BUILD_PYTHON_BINDINGS=ON): cmake .. -DTRT_PACKAGE_DIR=...
# -DBUILD_PYTHON_BINDINGS=ON && make -j Core libraries are cmake targets — no
# find_library needed.
#
# B) Standalone (build after core): cd experimental/pybind && mkdir -p build &&
# cd build cmake .. -DTRT_PACKAGE_DIR=... -DEDGELLM_BUILD_DIR=<core_build> &&
# make -j

# Detect whether we are included from the top-level project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_PYBIND_STANDALONE TRUE)
else()
  set(_PYBIND_STANDALONE FALSE)
endif()

if(_PYBIND_STANDALONE)
  cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
  project(edgellm_pybind LANGUAGES CXX CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)

  macro(set_ifndef var val)
    if(NOT DEFINED ${var})
      set(${var} "${val}")
    endif()
  endmacro()
endif()

# ── Project root (two levels up from experimental/pybind) ─────────────────
get_filename_component(EDGELLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                       ABSOLUTE)

# ── Required: TRT_PACKAGE_DIR ─────────────────────────────────────────────
if(NOT DEFINED TRT_PACKAGE_DIR)
  message(FATAL_ERROR "Set -DTRT_PACKAGE_DIR to your TensorRT installation.")
endif()

if(_PYBIND_STANDALONE)
  # ── Standalone: resolve CUDA, TRT, and core libraries manually ──────────
  set_ifndef(EDGELLM_BUILD_DIR "${EDGELLM_ROOT}/build")
  set_ifndef(CUDA_CTK_VERSION 12.8)
  set_ifndef(CUDA_DIR "/usr/local/cuda-${CUDA_CTK_VERSION}")

  find_path(
    CUDA_INCLUDE cuda_runtime_api.h
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES include)
  find_library(
    CUDART_LIB cudart
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib lib64)
  find_library(
    CUDA_DRIVER_LIB cuda
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs)
  set(CUDA_DRIVER_LINK_LIB "${CUDA_DRIVER_LIB}")
  if(AARCH64_BUILD)
    set(CUDA_DRIVER_LINK_LIB cuda)
  endif()
  find_path(
    TRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_PACKAGE_DIR}
    PATH_SUFFIXES include)
  find_library(
    NVINFER_LIB nvinfer
    HINTS ${TRT_PACKAGE_DIR}
    PATH_SUFFIXES lib lib64)
  find_library(
    NV_ONNX_PARSER_LIB nvonnxparser
    HINTS ${TRT_PACKAGE_DIR}
    PATH_SUFFIXES lib lib64)

  set(_CORE_LIB_DIR "${EDGELLM_BUILD_DIR}/cpp")
  find_library(
    EDGELLM_CORE_LIB edgellmCore
    HINTS ${_CORE_LIB_DIR}
    NO_DEFAULT_PATH)
  find_library(
    EDGELLM_BUILDER_LIB edgellmBuilder
    HINTS ${_CORE_LIB_DIR}
    NO_DEFAULT_PATH)

  foreach(_lib EDGELLM_CORE_LIB EDGELLM_BUILDER_LIB)
    if(NOT ${_lib})
      message(
        FATAL_ERROR
          "${_lib} not found in ${_CORE_LIB_DIR}.\n"
          "Build the core libraries first:\n"
          "  cd ${EDGELLM_ROOT} && mkdir -p build && cd build\n"
          "  cmake .. -DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR} && make -j")
    endif()
    message(STATUS "Found ${_lib}: ${${_lib}}")
  endforeach()
else()
  # ── Subdirectory: use cmake targets from the parent build ───────────────
  # edgellmCore and edgellmBuilder are already cmake targets. Tokenizer sources
  # are compiled into edgellmCore. CUDA and TRT targets/variables are inherited
  # from the parent scope (FindTensorRT.cmake creates TensorRT::TensorRT and
  # TensorRT::OnnxParser; CUDA variables come from the parent find_library
  # calls).
  set(CUDA_INCLUDE ${CUDA_INCLUDE_DIR})
  set(TRT_INCLUDE_DIR ${TensorRT_INCLUDE_DIR})
endif()

# ── pybind11 ──────────────────────────────────────────────────────────────
find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ── Header include paths (from source tree) ───────────────────────────────
set(NEMOTRON_ASR_DIR ${EDGELLM_ROOT}/experimental_models/nemotron3_5_asr/cpp)
set(EDGELLM_NEMOTRON_ASR_RUNTIME)
if(_PYBIND_STANDALONE)
  find_library(
    EDGELLM_NEMOTRON_ASR_RUNTIME
    NAMES nemotronAsrExperimental
    HINTS ${EDGELLM_BUILD_DIR}/experimental_models/nemotron3_5_asr/cpp
    NO_DEFAULT_PATH)
elseif(TARGET nemotronAsrExperimental)
  set(EDGELLM_NEMOTRON_ASR_RUNTIME nemotronAsrExperimental)
endif()

set(EDGELLM_INCLUDE_DIRS
    ${EDGELLM_ROOT}/cpp ${EDGELLM_ROOT}/examples/multimodal
    ${EDGELLM_ROOT}/3rdParty/nlohmannJson/include ${EDGELLM_ROOT}/3rdParty/stb)
if(EDGELLM_NEMOTRON_ASR_RUNTIME)
  list(APPEND EDGELLM_INCLUDE_DIRS ${NEMOTRON_ASR_DIR})
endif()

# ── Build the pybind shared library ───────────────────────────────────────
# device_link_stub.cu is an empty CUDA file that forces CMake to run the
# device-link step so __cudaRegisterLinkedBinary_* symbols from edgellmCore are
# resolved.
pybind11_add_module(
  _edgellm_runtime MODULE NO_EXTRAS
  ${CMAKE_CURRENT_SOURCE_DIR}/edgellm_pybind.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_link_stub.cu)
set_target_properties(_edgellm_runtime PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)

set_target_properties(
  _edgellm_runtime
  PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF
             PREFIX ""
             OUTPUT_NAME "_edgellm_runtime")

# Place the .so next to the core libraries in subdirectory mode, unless the
# caller (e.g. setup_pybind.py) already pinned CMAKE_LIBRARY_OUTPUT_DIRECTORY.
if(NOT _PYBIND_STANDALONE AND NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(
    _edgellm_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                "${CMAKE_BINARY_DIR}/pybind")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_edgellm_runtime PRIVATE -fno-lto)
  target_link_options(_edgellm_runtime PRIVATE -fno-lto)
endif()

target_include_directories(_edgellm_runtime PRIVATE ${EDGELLM_INCLUDE_DIRS})
target_include_directories(_edgellm_runtime SYSTEM PRIVATE ${CUDA_INCLUDE}
                                                           ${TRT_INCLUDE_DIR})
if(EDGELLM_NEMOTRON_ASR_RUNTIME)
  target_compile_definitions(_edgellm_runtime
                             PRIVATE EDGELLM_ENABLE_NEMOTRON_ASR=1)
  target_link_libraries(_edgellm_runtime
                        PRIVATE ${EDGELLM_NEMOTRON_ASR_RUNTIME})
endif()

if(TARGET Python::Module)
  target_link_libraries(_edgellm_runtime PRIVATE Python::Module)
else()
  target_link_libraries(_edgellm_runtime PRIVATE Python::Python)
endif()

# Wheel payloads retain every AOT kernel declared by the selected artifact,
# including variants that current runtime sources do not reference.
set(_EDGELLM_WHEEL_CUTE_DSL_ARCHIVE)
if(NOT _PYBIND_STANDALONE
   AND DEFINED EDGELLM_WHEEL_PAYLOAD_DIR
   AND NOT EDGELLM_WHEEL_PAYLOAD_DIR STREQUAL "")
  get_property(_EDGELLM_WHEEL_CUTE_DSL_ARCHIVE GLOBAL
               PROPERTY EDGELLM_CUTE_DSL_STATIC_LIBRARY)
endif()

# edgellmCore contains CUDA device code; --whole-archive ensures fatbin data and
# all device-code objects are included in the final .so.
if(_PYBIND_STANDALONE)
  target_link_libraries(
    _edgellm_runtime
    PRIVATE -Wl,--whole-archive
            ${EDGELLM_CORE_LIB}
            ${_EDGELLM_WHEEL_CUTE_DSL_ARCHIVE}
            -Wl,--no-whole-archive
            ${EDGELLM_BUILDER_LIB}
            ${NVINFER_LIB}
            ${NV_ONNX_PARSER_LIB}
            ${CUDA_DRIVER_LINK_LIB}
            ${CUDART_LIB})
else()
  # Subdirectory mode: TRT cmake targets are provided by FindTensorRT.cmake
  # (find_package(TensorRT REQUIRED COMPONENTS OnnxParser) in the root
  # CMakeLists.txt). Use them directly instead of the now-gone NVINFER_LIB /
  # NV_ONNX_PARSER_LIB variables.
  target_link_libraries(
    _edgellm_runtime
    PRIVATE -Wl,--whole-archive
            edgellmCore
            ${_EDGELLM_WHEEL_CUTE_DSL_ARCHIVE}
            -Wl,--no-whole-archive
            edgellmBuilder
            TensorRT::TensorRT
            TensorRT::OnnxParser
            ${CUDA_DRIVER_LINK_LIB}
            ${CUDART_LIB})
endif()

# On aarch64 (Orin/Thor), add CUDA library search paths so the device-link step
# can find cudadevrt/cudart_static and -lcuda can resolve the driver stub.
if(COMMAND add_cross_build_link_options)
  add_cross_build_link_options(_edgellm_runtime)
elseif(AARCH64_BUILD)
  target_link_options(_edgellm_runtime PRIVATE "-L${CUDA_DIR}/lib")
  target_link_options(_edgellm_runtime PRIVATE "-L${CUDA_DIR}/lib/stubs")
endif()

# CPython API symbols are resolved by the running interpreter. All other
# dependencies are linked explicitly and audited during payload verification.
