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

# Cross-compilation toolchain for QNX Standard on AArch64.
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

set(CMAKE_C_COMPILER
    "${QNX_TOOLCHAIN_BIN_DIR}/qcc"
    CACHE FILEPATH "QNX C compiler")
set(CMAKE_CXX_COMPILER
    "${QNX_TOOLCHAIN_BIN_DIR}/q++"
    CACHE FILEPATH "QNX C++ compiler")
set(CMAKE_C_COMPILER_TARGET "${QNX_COMPILER_TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${QNX_COMPILER_TARGET}")
set(CMAKE_LINKER
    "${QNX_TOOLCHAIN_BIN_DIR}/${QNX_TRIPLE}-ld"
    CACHE FILEPATH "QNX linker")

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

# qcc/q++ provide pthread support through the QNX system library.
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

set(QNX_CUDA_TARGET_ROOT
    "/usr/local/cuda-safe-${CUDA_CTK_VERSION}"
    CACHE PATH "QNX CUDA target toolkit root")
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
    AARCH64_BUILD
    QNX_BUILD)
