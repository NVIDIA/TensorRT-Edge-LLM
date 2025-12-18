# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

# aarch64_toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER
    ${CMAKE_CXX_COMPILER}
    CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_FLAGS
    " -Xcompiler=\"-fPIC \""
    CACHE STRING "" FORCE)

# Specify the architecture for CUDA
macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

if("${EMBEDDED_TARGET}" STREQUAL "auto-thor")
  set_ifndef(CUDA_VERSION 13.0)
  if(CUDA_VERSION VERSION_LESS 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 101)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 110)
  endif()
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
  set(CUDA_TARGET_DIR
      /usr/local/cuda/thor/targets/aarch64-linux
      CACHE STRING "CUDA toolkit target dir")
  message(STATUS "Using CUDA toolkit dir: ${CUDA_DIR}")
elseif("${EMBEDDED_TARGET}" STREQUAL "jetson-thor")
  set_ifndef(CUDA_VERSION 13.0)
  set(CMAKE_CUDA_ARCHITECTURES 110)
  set(CUDA_DIR
      /usr/local/cuda/targets/sbsa-linux
      CACHE STRING "CUDA toolkit dir")
elseif("${EMBEDDED_TARGET}" STREQUAL "jetson-orin")
  set_ifndef(CUDA_VERSION 12.6)
  set(CMAKE_CUDA_ARCHITECTURES 87)
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
elseif("${EMBEDDED_TARGET}" STREQUAL "gb10")
  set_ifndef(CUDA_VERSION 13.0)
  set(CMAKE_CUDA_ARCHITECTURES 121)
  set(CUDA_DIR
      /usr/local/cuda/targets/sbsa-linux
      CACHE STRING "CUDA toolkit dir")
endif()

# Tell CMake how to search for the libraries and programs
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Set variable to indicate CMake is running aarch64 build
set(AARCH64_BUILD TRUE)
