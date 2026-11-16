# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(QNX_BUILD)
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
  set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                       ${CUDA_PLATFORM_INCLUDE_DIR})
  set(CUDA_DRIVER_LINK_LIB cuda)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
else()
  find_package(CUDAToolkit REQUIRED)

  if(DEFINED CUDA_VERSION)
    message(
      FATAL_ERROR "CUDA_VERSION is reserved by CUDA headers; do not set it.")
  endif()
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" _detected_cuda_series
               "${CUDAToolkit_VERSION}")
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" _requested_cuda_series
               "${CUDA_CTK_VERSION}")
  if(DEFINED CUDA_CTK_VERSION AND NOT _requested_cuda_series STREQUAL
                                  _detected_cuda_series)
    message(
      FATAL_ERROR
        "CUDA_CTK_VERSION=${CUDA_CTK_VERSION} does not match the detected CUDA Toolkit ${CUDAToolkit_VERSION}. Remove the override or select the matching toolkit with CUDAToolkit_ROOT."
    )
  endif()
  set(CUDA_CTK_VERSION
      "${CUDAToolkit_VERSION}"
      CACHE INTERNAL "Detected CUDA Toolkit version" FORCE)

  get_filename_component(_cuda_compiler_realpath "${CMAKE_CUDA_COMPILER}"
                         REALPATH)
  get_filename_component(_cuda_bin_dir "${_cuda_compiler_realpath}" DIRECTORY)
  get_filename_component(_cuda_toolkit_root "${_cuda_bin_dir}" DIRECTORY)

  set(CUDA_TARGET_DIRS)
  if(AARCH64_BUILD)
    if(EMBEDDED_TARGET STREQUAL "auto-thor"
       OR (EMBEDDED_TARGET STREQUAL "jetson-orin" AND CUDAToolkit_VERSION
                                                      VERSION_LESS 13.0))
      set(_cuda_platform aarch64-linux)
    else()
      set(_cuda_platform sbsa-linux)
    endif()
    if(NOT CUDA_DIR)
      set(CUDA_DIR "${_cuda_toolkit_root}/targets/${_cuda_platform}")
    endif()
    if(NOT IS_DIRECTORY "${CUDA_DIR}")
      message(
        FATAL_ERROR
          "CUDA_DIR=${CUDA_DIR} is not an extracted CUDA target directory for ${EMBEDDED_TARGET}."
      )
    endif()
    list(APPEND CUDA_TARGET_DIRS "${CUDA_DIR}")
    if(NOT CUDA_TARGET_DIR)
      if(EMBEDDED_TARGET STREQUAL "auto-thor")
        if(IS_DIRECTORY "${_cuda_toolkit_root}/thor/targets/aarch64-linux")
          set(CUDA_TARGET_DIR
              "${_cuda_toolkit_root}/thor/targets/aarch64-linux")
        endif()
      elseif(EMBEDDED_TARGET STREQUAL "gb10")
        if(IS_DIRECTORY "${_cuda_toolkit_root}/n1/targets/sbsa-linux")
          set(CUDA_TARGET_DIR "${_cuda_toolkit_root}/n1/targets/sbsa-linux")
        endif()
      endif()
    endif()
    if(CUDA_TARGET_DIR)
      if(NOT IS_DIRECTORY "${CUDA_TARGET_DIR}")
        message(
          FATAL_ERROR "CUDA_TARGET_DIR=${CUDA_TARGET_DIR} does not exist.")
      endif()
      list(APPEND CUDA_TARGET_DIRS "${CUDA_TARGET_DIR}")
    endif()
  else()
    set(CUDA_DIR "${_cuda_toolkit_root}")
  endif()

  set(CUDA_INCLUDE_DIR
      ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      ${CUDA_PLATFORM_INCLUDE_DIR})
  foreach(_cuda_target IN LISTS CUDA_TARGET_DIRS)
    if(IS_DIRECTORY "${_cuda_target}/include")
      list(APPEND CUDA_INCLUDE_DIR "${_cuda_target}/include")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES CUDA_INCLUDE_DIR)
  list(GET CUDA_INCLUDE_DIR 0 CUDA_RUNTIME_API_INCLUDE_DIR)

  if(EDGELLM_CROSS_AARCH64)
    if(NOT CUDART_LIB)
      find_library(
        CUDART_LIB cudart
        HINTS ${CUDA_TARGET_DIRS}
        PATH_SUFFIXES lib lib64 REQUIRED
        NO_DEFAULT_PATH)
    endif()
    if(NOT CUDA_DRIVER_LIB)
      find_library(
        CUDA_DRIVER_LIB cuda
        HINTS ${CUDA_TARGET_DIRS}
        PATH_SUFFIXES lib/stubs REQUIRED
        NO_DEFAULT_PATH)
    endif()
    set(NVRTC_ROOT_DIR ${CUDA_TARGET_DIRS})
  elseif(NOT CUDART_LIB)
    set(CUDART_LIB CUDA::cudart)
  endif()
  if(NOT CUDA_DRIVER_LIB)
    set(CUDA_DRIVER_LIB CUDA::cuda_driver)
  endif()
  if(EDGELLM_CROSS_AARCH64)
    set(CUDA_DRIVER_LINK_LIB cuda)
  else()
    set(CUDA_DRIVER_LINK_LIB "${CUDA_DRIVER_LIB}")
  endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/FindNVRTC.cmake)
find_package(TensorRT REQUIRED COMPONENTS OnnxParser)
if(ENABLE_SAFETY_RUNTIME)
  find_package(TensorRTSafe REQUIRED)
endif()

include_directories(SYSTEM ${CUDA_INCLUDE_DIR} ${TensorRT_INCLUDE_DIR})

set(COMMON_INCLUDE_DIRS
    ${NVTX_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include
    ${CMAKE_SOURCE_DIR}/3rdParty/stb ${CMAKE_SOURCE_DIR}/3rdParty/miniaudio)

add_library(commonLibraryExt INTERFACE)
set_target_properties(commonLibraryExt PROPERTIES EXPORT_NAME Dependencies)
if(QNX_BUILD)
  target_link_libraries(
    commonLibraryExt INTERFACE TensorRT::TensorRT ${CUDART_LIB}
                               ${CUDA_DRIVER_LINK_LIB})
else()
  target_link_libraries(
    commonLibraryExt
    INTERFACE TensorRT::TensorRT CUDA::cudart
              $<BUILD_INTERFACE:${CUDA_DRIVER_LINK_LIB}>
              $<INSTALL_INTERFACE:CUDA::cuda_driver>)
endif()
target_include_directories(
  commonLibraryExt SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/3rdParty/inja/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(ENABLE_NVTX_PROFILING)
  target_compile_definitions(commonLibraryExt INTERFACE ENABLE_NVTX_PROFILING)
  target_include_directories(commonLibraryExt SYSTEM
                             INTERFACE $<BUILD_INTERFACE:${NVTX_INCLUDE_DIR}>)
endif()

if(NOT QNX_BUILD)
  message(
    STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
endif()
