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
  COMPONENTS Interpreter Development
  REQUIRED)
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE _pybind11_cmakedir_result
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_pybind11_cmakedir_result EQUAL 0 AND IS_DIRECTORY "${_pybind11_cmakedir}")
    list(PREPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

# ── Header include paths (from source tree) ───────────────────────────────
set(EDGELLM_INCLUDE_DIRS
    ${EDGELLM_ROOT}/cpp ${EDGELLM_ROOT}/examples/multimodal
    ${EDGELLM_ROOT}/3rdParty/nlohmannJson/include ${EDGELLM_ROOT}/3rdParty/stb)

# ── Build the pybind shared library ───────────────────────────────────────
# device_link_stub.cu is an empty CUDA file that forces CMake to run the
# device-link step so __cudaRegisterLinkedBinary_* symbols from edgellmCore are
# resolved.
pybind11_add_module(
  _edgellm_runtime SHARED NO_EXTRAS
  ${CMAKE_CURRENT_SOURCE_DIR}/edgellm_pybind.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_link_stub.cu)
set_target_properties(_edgellm_runtime PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)

set_target_properties(
  _edgellm_runtime
  PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF
             PREFIX ""
             OUTPUT_NAME "_edgellm_runtime")

# Place the .so next to the core libraries in subdirectory mode.
if(NOT _PYBIND_STANDALONE)
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

if(TARGET Python::Module)
  target_link_libraries(_edgellm_runtime PRIVATE Python::Module)
else()
  target_link_libraries(_edgellm_runtime PRIVATE Python::Python)
endif()

# edgellmCore contains CUDA device code; --whole-archive ensures fatbin data and
# all device-code objects are included in the final .so.
if(_PYBIND_STANDALONE)
  # Standalone builds must also link the prebuilt CuTe DSL kernel archive that
  # edgellmCore references (the subdirectory build gets this via CuteDsl.cmake).
  if(NOT DEFINED EDGELLM_CUTEDSL_LIB)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(_cutedsl_arch "aarch64")
    else()
      set(_cutedsl_arch "x86_64")
    endif()
    file(
      GLOB
      _cutedsl_libs
      "${EDGELLM_ROOT}/cpp/kernels/cuteDSLArtifact/${_cutedsl_arch}/*/libcutedsl_${_cutedsl_arch}.a"
    )
    list(LENGTH _cutedsl_libs _cutedsl_lib_count)
    if(_cutedsl_lib_count EQUAL 0)
      message(
        FATAL_ERROR
          "No CuTe DSL archive found for ${_cutedsl_arch}. Set -DEDGELLM_CUTEDSL_LIB=/path/to/libcutedsl_${_cutedsl_arch}.a."
      )
    elseif(_cutedsl_lib_count GREATER 1)
      message(
        FATAL_ERROR
          "Multiple CuTe DSL archives found for ${_cutedsl_arch}: ${_cutedsl_libs}. Set -DEDGELLM_CUTEDSL_LIB explicitly."
      )
    endif()
    list(GET _cutedsl_libs 0 EDGELLM_CUTEDSL_LIB)
  endif()
  if(NOT EXISTS "${EDGELLM_CUTEDSL_LIB}")
    message(
      FATAL_ERROR "CuTe DSL archive does not exist: ${EDGELLM_CUTEDSL_LIB}")
  endif()
  message(STATUS "CuTe DSL archive for pybind: ${EDGELLM_CUTEDSL_LIB}")
  target_link_libraries(
    _edgellm_runtime
    PRIVATE -Wl,--whole-archive
            ${EDGELLM_CORE_LIB}
            ${EDGELLM_CUTEDSL_LIB}
            -Wl,--no-whole-archive
            ${EDGELLM_BUILDER_LIB}
            ${NVINFER_LIB}
            ${NV_ONNX_PARSER_LIB}
            ${CUDA_DRIVER_LIB}
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
            -Wl,--no-whole-archive
            edgellmBuilder
            TensorRT::TensorRT
            TensorRT::OnnxParser
            ${CUDA_DRIVER_LIB}
            ${CUDART_LIB})
endif()

# On aarch64 (Orin/Thor), add CUDA library search paths so the device-link step
# can find cudadevrt and cudart_static.
if(NOT _PYBIND_STANDALONE)
  add_cross_build_link_options(_edgellm_runtime)
endif()

# Allow unresolved symbols that will be satisfied at load time (e.g. libnvdla).
target_link_options(_edgellm_runtime PRIVATE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")
