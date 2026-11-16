# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL
                                        EDGELLM_AUTO_CUDA_ARCHITECTURES)
  set(EDGELLM_CUDA_ARCHITECTURES_EXPLICIT TRUE)
else()
  set(EDGELLM_CUDA_ARCHITECTURES_EXPLICIT FALSE)
endif()

function(edgellm_prepare_toolchain)
  if(CMAKE_TOOLCHAIN_FILE)
    return()
  endif()

  cmake_host_system_information(RESULT _host_arch QUERY OS_PLATFORM)
  if(NOT EMBEDDED_TARGET
     AND _host_arch MATCHES "aarch64|arm64"
     AND (EXISTS "/proc/device-tree/model" OR EXISTS
                                              "/sys/class/dmi/id/product_name"))
    if(EXISTS "/proc/device-tree/model")
      file(READ "/proc/device-tree/model" _device_model)
    else()
      file(READ "/sys/class/dmi/id/product_name" _device_model)
    endif()
    string(TOLOWER "${_device_model}" _device_model)
    if(_device_model MATCHES "gb300")
      set(EMBEDDED_TARGET
          gb300
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "gb200")
      set(EMBEDDED_TARGET
          gb200
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "orin")
      set(EMBEDDED_TARGET
          jetson-orin
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "dgx spark|gb10")
      set(EMBEDDED_TARGET
          gb10
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "(drive|auto).*thor|thor.*(drive|auto)")
      set(EMBEDDED_TARGET
          auto-thor
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "igx.*thor")
      set(EMBEDDED_TARGET
          igx-thor
          CACHE STRING "Detected embedded target" FORCE)
    elseif(_device_model MATCHES "thor")
      set(EMBEDDED_TARGET
          jetson-thor
          CACHE STRING "Detected embedded target" FORCE)
    endif()
  endif()

  if(NOT CMAKE_CUDA_COMPILER)
    if(CUDAToolkit_ROOT)
      set(_edgellm_cuda_roots "${CUDAToolkit_ROOT}")
    else()
      file(
        GLOB _edgellm_cuda_roots
        LIST_DIRECTORIES TRUE
        "/usr/local/cuda-[0-9]*")
      list(
        SORT _edgellm_cuda_roots
        COMPARE NATURAL
        ORDER DESCENDING)
      list(APPEND _edgellm_cuda_roots "/usr/local/cuda")
    endif()
    foreach(_edgellm_cuda_root IN LISTS _edgellm_cuda_roots)
      if(EXISTS "${_edgellm_cuda_root}/bin/nvcc")
        set(_edgellm_nvcc "${_edgellm_cuda_root}/bin/nvcc")
        break()
      endif()
    endforeach()
    if(NOT _edgellm_nvcc AND NOT CUDAToolkit_ROOT)
      find_program(_edgellm_nvcc nvcc)
    endif()
    if(_edgellm_nvcc)
      set(CMAKE_CUDA_COMPILER
          "${_edgellm_nvcc}"
          PARENT_SCOPE)
    elseif(CUDAToolkit_ROOT)
      message(
        FATAL_ERROR "No nvcc found under CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
    endif()
  endif()

  if(EMBEDDED_TARGET AND NOT _host_arch MATCHES "aarch64|arm64")
    find_program(_aarch64_c_compiler aarch64-linux-gnu-gcc HINTS /usr/bin
                                                                 REQUIRED)
    find_program(_aarch64_cxx_compiler aarch64-linux-gnu-g++ HINTS /usr/bin
                                                                   REQUIRED)
    set(CMAKE_SYSTEM_NAME
        Linux
        PARENT_SCOPE)
    set(CMAKE_SYSTEM_PROCESSOR
        aarch64
        PARENT_SCOPE)
    set(CMAKE_C_COMPILER
        "${_aarch64_c_compiler}"
        PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER
        "${_aarch64_cxx_compiler}"
        PARENT_SCOPE)
    set(CMAKE_CUDA_HOST_COMPILER
        "${_aarch64_cxx_compiler}"
        PARENT_SCOPE)
    if(CMAKE_CUDA_COMPILER)
      set(_edgellm_nvcc "${CMAKE_CUDA_COMPILER}")
    endif()
    execute_process(
      COMMAND "${_edgellm_nvcc}" --version
      OUTPUT_VARIABLE _nvcc_version
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND "${_aarch64_cxx_compiler}" -dumpversion
      OUTPUT_VARIABLE _host_gcc_version
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_nvcc_version MATCHES "release 11\\.4," AND _host_gcc_version
                                                   VERSION_GREATER_EQUAL 9)
      set(CMAKE_CUDA_FLAGS_INIT
          "${CMAKE_CUDA_FLAGS_INIT} --allow-unsupported-compiler"
          PARENT_SCOPE)
    endif()
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
        NEVER
        PARENT_SCOPE)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
        ONLY
        PARENT_SCOPE)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
        ONLY
        PARENT_SCOPE)
  endif()
endfunction()

function(edgellm_configure_platform)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_arch aarch64)
    set(AARCH64_BUILD
        TRUE
        CACHE INTERNAL "Build targets AArch64" FORCE)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(_arch x86_64)
    set(AARCH64_BUILD
        FALSE
        CACHE INTERNAL "Build targets AArch64" FORCE)
  else()
    message(
      FATAL_ERROR "Unsupported target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(QNX_BUILD
        TRUE
        CACHE INTERNAL "Build targets QNX" FORCE)
  else()
    set(QNX_BUILD
        FALSE
        CACHE INTERNAL "Build targets QNX" FORCE)
  endif()

  if(AARCH64_BUILD AND NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(EDGELLM_CROSS_AARCH64
        TRUE
        CACHE INTERNAL "Cross-compiling for AArch64" FORCE)
  else()
    set(EDGELLM_CROSS_AARCH64
        FALSE
        CACHE INTERNAL "Cross-compiling for AArch64" FORCE)
  endif()

  if(NOT EDGELLM_CUDA_ARCHITECTURES_EXPLICIT)
    set(_cuda_architectures)
    if(EMBEDDED_TARGET STREQUAL "jetson-orin")
      set(_cuda_architectures 87)
    elseif(EMBEDDED_TARGET STREQUAL "auto-thor")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
        set(_cuda_architectures 101a)
      else()
        set(_cuda_architectures 110a)
      endif()
    elseif(EMBEDDED_TARGET STREQUAL "jetson-thor")
      set(_cuda_architectures 110a)
    elseif(EMBEDDED_TARGET STREQUAL "igx-thor")
      set(_cuda_architectures 110a 120)
    elseif(EMBEDDED_TARGET STREQUAL "gb10")
      set(_cuda_architectures 121a)
    elseif(EMBEDDED_TARGET STREQUAL "gb300")
      set(_cuda_architectures 103a)
    elseif(EMBEDDED_TARGET STREQUAL "gb200")
      set(_cuda_architectures 100a)
    elseif(NOT EDGELLM_CROSS_AARCH64)
      find_program(_nvidia_smi nvidia-smi)
      if(_nvidia_smi)
        execute_process(
          COMMAND "${_nvidia_smi}" --query-gpu=compute_cap
                  --format=csv,noheader,nounits
          OUTPUT_VARIABLE _compute_capabilities
          OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REPLACE "\n" ";" _compute_capabilities
                       "${_compute_capabilities}")
        foreach(_compute_capability IN LISTS _compute_capabilities)
          string(STRIP "${_compute_capability}" _compute_capability)
          string(REPLACE "." "" _cuda_arch "${_compute_capability}")
          if(_cuda_arch MATCHES "^[0-9]+$")
            if(_cuda_arch MATCHES "^(100|101|103|110|121)$")
              string(APPEND _cuda_arch "a")
            endif()
            list(APPEND _cuda_architectures "${_cuda_arch}")
          endif()
        endforeach()
        list(REMOVE_DUPLICATES _cuda_architectures)
        if(_cuda_architectures)
          list(GET _cuda_architectures 0 _first_cuda_arch)
          set(_cuda_architectures "${_first_cuda_arch}")
        endif()
      endif()
    endif()

    if(_cuda_architectures)
      set(CMAKE_CUDA_ARCHITECTURES
          "${_cuda_architectures}"
          CACHE STRING "CUDA architectures detected by TensorRT Edge-LLM" FORCE)
      set(EDGELLM_AUTO_CUDA_ARCHITECTURES
          "${_cuda_architectures}"
          CACHE INTERNAL "Last automatically detected CUDA architectures" FORCE)
    endif()
  endif()

  message(STATUS "TensorRT Edge-LLM target: ${CMAKE_SYSTEM_NAME}/${_arch}")
  message(STATUS "Embedded target: ${EMBEDDED_TARGET}")
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()
