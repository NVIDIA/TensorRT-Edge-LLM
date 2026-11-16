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

# Cross-compilation toolchain for QNX on AArch64.
set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED ENV{QNX_HOST} OR NOT DEFINED ENV{QNX_TARGET})
  message(
    FATAL_ERROR
      "QNX_HOST and QNX_TARGET must both be set for a QNX cross build.")
endif()

set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")
set(CMAKE_SYSROOT "${QNX_TARGET}")

set(QNX_VERSION
    "8.0.0"
    CACHE STRING "QNX SDP version")
set(QNX_GCC_VERSION
    "12.2.0"
    CACHE STRING "QNX GCC toolchain version")
set(QNX_TRIPLE
    "aarch64-unknown-nto-qnx${QNX_VERSION}"
    CACHE STRING "QNX AArch64 compiler triple")
set(QNX_COMPILER_TARGET
    "${QNX_GCC_VERSION},gcc_ntoaarch64le"
    CACHE STRING "QNX compiler target")
set(QNX_TOOLCHAIN_BIN_DIR
    "${QNX_HOST}/usr/bin"
    CACHE PATH "QNX host toolchain binary directory")

set(_qnx_c_driver "${QNX_TRIPLE}-gcc")
set(_qnx_cxx_driver "${QNX_TRIPLE}-g++")
if(NOT DEFINED ENV{QNX_USE_QCC})
  message(
    WARNING
      "QNX_USE_QCC is not set; defaulting to GCC/G++ for this QNX build. Set QNX_USE_QCC=1 to use qcc/q++."
  )
elseif("$ENV{QNX_USE_QCC}" STREQUAL "1")
  set(_qnx_c_driver qcc)
  set(_qnx_cxx_driver q++)
  set(CMAKE_C_COMPILER_TARGET "${QNX_COMPILER_TARGET}")
  set(CMAKE_CXX_COMPILER_TARGET "${QNX_COMPILER_TARGET}")
elseif(NOT "$ENV{QNX_USE_QCC}" STREQUAL "0")
  message(FATAL_ERROR "QNX_USE_QCC must be 0 or 1")
endif()
set(CMAKE_C_COMPILER
    "${QNX_TOOLCHAIN_BIN_DIR}/${_qnx_c_driver}"
    CACHE FILEPATH "QNX C compiler")
set(CMAKE_CXX_COMPILER
    "${QNX_TOOLCHAIN_BIN_DIR}/${_qnx_cxx_driver}"
    CACHE FILEPATH "QNX C++ compiler")
set(CMAKE_LINKER
    "${QNX_TOOLCHAIN_BIN_DIR}/${QNX_TRIPLE}-ld"
    CACHE FILEPATH "QNX linker")
set(CMAKE_AR
    "${QNX_TOOLCHAIN_BIN_DIR}/${QNX_TRIPLE}-ar"
    CACHE FILEPATH "QNX archiver")
set(CMAKE_RANLIB
    "${QNX_TOOLCHAIN_BIN_DIR}/${QNX_TRIPLE}-ranlib"
    CACHE FILEPATH "QNX archive indexer")

# NVCC must use the underlying QNX cross-compiler rather than the q++ driver.
set(CMAKE_CUDA_HOST_COMPILER
    "${QNX_TOOLCHAIN_BIN_DIR}/${QNX_TRIPLE}-g++"
    CACHE FILEPATH "QNX host compiler used by NVCC")

find_program(
  QNX_OBJCOPY
  NAMES "${QNX_TRIPLE}-objcopy"
  PATHS "${QNX_TOOLCHAIN_BIN_DIR}"
  NO_DEFAULT_PATH REQUIRED)
set(CMAKE_OBJCOPY "${QNX_OBJCOPY}")

find_program(
  QNX_STRIP
  NAMES "${QNX_TRIPLE}-strip"
  PATHS "${QNX_TOOLCHAIN_BIN_DIR}"
  NO_DEFAULT_PATH REQUIRED)
set(CMAKE_STRIP "${QNX_STRIP}")

# QNX provides pthread support through the system library.
set(CMAKE_HAVE_LIBC_PTHREAD TRUE)

if(ENABLE_CUTE_DSL)
  message(FATAL_ERROR "CuTe DSL kernels do not support QNX cross-compilation.")
endif()
set(ENABLE_CUTE_DSL
    "OFF"
    CACHE STRING "CuTe DSL kernels are unavailable for QNX")

set(CUDA_CTK_VERSION
    ""
    CACHE STRING "CUDA Toolkit version")
if(NOT CUDA_CTK_VERSION)
  message(FATAL_ERROR "CUDA_CTK_VERSION must specify the CUDA Toolkit version.")
endif()

option(QNX_SAFE_CUDA "Use Safe CUDA and the DRIVE QNX Safety lifecycle"
       "${EDGELLM_SAFETY_ONLY_PACKAGE}")
if(QNX_SAFE_CUDA)
  set(_qnx_cuda_default_root "/usr/local/cuda-safe-${CUDA_CTK_VERSION}")
else()
  set(_qnx_cuda_default_root "/usr/local/cuda-${CUDA_CTK_VERSION}")
  if(QNX_SAFE_PDK_ROOT)
    message(FATAL_ERROR "Standard QNX CUDA must not use QNX_SAFE_PDK_ROOT")
  endif()
endif()
set(QNX_CUDA_TARGET_ROOT
    "${_qnx_cuda_default_root}"
    CACHE PATH "QNX CUDA target toolkit root")
unset(_qnx_cuda_default_root)
set(CUDA_TOOLKIT_ROOT
    "/usr/local/cuda-${CUDA_CTK_VERSION}"
    CACHE PATH "CUDA compiler toolkit root")
set(CUDA_DIR
    "${QNX_CUDA_TARGET_ROOT}/targets/aarch64-qnx"
    CACHE PATH "QNX CUDA target directory")
set(CUDA_TARGET_DIR
    "${QNX_CUDA_TARGET_ROOT}/thor/targets/aarch64-qnx"
    CACHE PATH "Additional QNX CUDA target directory")
set(CUDA_PLATFORM_INCLUDE_DIR
    ""
    CACHE PATH "Additional QNX CUDA include directory")

if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER
      "${CUDA_TOOLKIT_ROOT}/bin/nvcc"
      CACHE FILEPATH "Cross-capable CUDA compiler")
endif()
if(NOT EXISTS "${CUDA_DIR}")
  message(
    FATAL_ERROR
      "CUDA_DIR must point to an existing QNX CUDA target directory: ${CUDA_DIR}"
  )
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(CUDA_CTK_VERSION VERSION_GREATER_EQUAL "13.0")
    set(QNX_CUDA_ARCHITECTURES 110a)
  elseif(CUDA_CTK_VERSION VERSION_GREATER_EQUAL "12.7")
    set(QNX_CUDA_ARCHITECTURES 101a)
  else()
    message(
      FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES must be set for CUDA Toolkit versions before 12.7."
    )
  endif()
  set(CMAKE_CUDA_ARCHITECTURES
      "${QNX_CUDA_ARCHITECTURES}"
      CACHE STRING "QNX CUDA architectures")
endif()

if(CUDA_TARGET_DIR AND NOT CUDA_PLATFORM_INCLUDE_DIR)
  set(CUDA_PLATFORM_INCLUDE_DIR "${CUDA_TARGET_DIR}/include")
endif()
foreach(_qnx_cuda_include "${CUDA_DIR}/include" "${CUDA_TARGET_DIR}/include"
                          "${CUDA_PLATFORM_INCLUDE_DIR}")
  if(EXISTS "${_qnx_cuda_include}/cuda_runtime_api.h")
    if(QNX_SAFE_CUDA AND NOT EXISTS
                         "${_qnx_cuda_include}/cuda_runtime_api_safe_ex.h")
      message(FATAL_ERROR "QNX_SAFE_CUDA requires Safe CUDA target headers")
    elseif(NOT QNX_SAFE_CUDA
           AND EXISTS "${_qnx_cuda_include}/cuda_runtime_api_safe_ex.h")
      message(
        FATAL_ERROR "Standard QNX CUDA must not use Safe CUDA target headers")
    endif()
  endif()
endforeach()
unset(_qnx_cuda_include)

set(QNX_COMPILE_DEFINITIONS
    "-D_XOPEN_SOURCE=700 -D_POSIX_C_SOURCE=2 -D_QNX_SOURCE -DQNX=1 -D__aarch64__"
)
set(CMAKE_C_FLAGS_INIT "${QNX_COMPILE_DEFINITIONS}")
set(CMAKE_CXX_FLAGS_INIT "${QNX_COMPILE_DEFINITIONS}")
set(CMAKE_CUDA_FLAGS_INIT "${QNX_COMPILE_DEFINITIONS}")

set(CMAKE_FIND_ROOT_PATH "${QNX_TARGET}" "${CUDA_DIR}")
if(CUDA_TARGET_DIR)
  list(APPEND CMAKE_FIND_ROOT_PATH "${CUDA_TARGET_DIR}")
endif()
if(DEFINED TRT_PACKAGE_DIR)
  list(APPEND CMAKE_FIND_ROOT_PATH "${TRT_PACKAGE_DIR}")
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(AARCH64_BUILD TRUE)
set(QNX_BUILD TRUE)

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    QNX_VERSION
    QNX_GCC_VERSION
    QNX_TRIPLE
    QNX_COMPILER_TARGET
    QNX_TOOLCHAIN_BIN_DIR
    CUDA_CTK_VERSION
    CUDA_TOOLKIT_ROOT
    QNX_CUDA_TARGET_ROOT
    CUDA_DIR
    CUDA_TARGET_DIR
    CUDA_PLATFORM_INCLUDE_DIR
    CMAKE_CUDA_COMPILER
    CMAKE_CUDA_ARCHITECTURES
    QNX_SAFE_PDK_ROOT
    QNX_SAFE_CUDA
    EDGELLM_SAFETY_ONLY_PACKAGE
    AARCH64_BUILD
    QNX_BUILD)

if(QNX_SAFE_CUDA)
  set(QNX_SAFE_PDK_ROOT
      ""
      CACHE PATH "DRIVE QNX Safety PDK root containing lib-target")
  if(NOT IS_DIRECTORY "${QNX_SAFE_PDK_ROOT}/lib-target")
    message(
      FATAL_ERROR "Set QNX_SAFE_PDK_ROOT to the matching DRIVE QNX Safety PDK")
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  # SafeCUDA requires legacy launches and the DRIVE lifecycle-check libraries.
  set(_qnx_safe_link_flags
      "-L${QNX_SAFE_PDK_ROOT}/lib-target -lslog2 -lnvdvms_client -lnvos_s3_safety"
  )
  string(
    APPEND
    CMAKE_CUDA_FLAGS_INIT
    " --cudart=shared --cudadevrt=none -legacy-launch-seq ${_qnx_safe_link_flags}"
  )
  string(
    APPEND CMAKE_EXE_LINKER_FLAGS_INIT
    " ${_qnx_safe_link_flags} -Wl,-rpath-link,${QNX_SAFE_PDK_ROOT}/lib-target")
  string(
    APPEND CMAKE_SHARED_LINKER_FLAGS_INIT
    " ${_qnx_safe_link_flags} -Wl,-rpath-link,${QNX_SAFE_PDK_ROOT}/lib-target")
endif()
