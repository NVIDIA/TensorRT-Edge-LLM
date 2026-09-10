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

set(NVRTC_ROOT_HINTS)
foreach(nvrtc_root NVRTC_ROOT_DIR CUDA_TARGET_DIR CUDA_DIR)
  if(DEFINED ${nvrtc_root})
    list(APPEND NVRTC_ROOT_HINTS "${${nvrtc_root}}")
  endif()
endforeach()
if(NOT DEFINED AARCH64_BUILD AND DEFINED CMAKE_CUDA_COMPILER_TOOLKIT_ROOT)
  list(APPEND NVRTC_ROOT_HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
endif()
list(REMOVE_DUPLICATES NVRTC_ROOT_HINTS)

set(NVRTC_INCLUDE_HINTS)
set(NVRTC_LIBRARY_HINTS)
foreach(nvrtc_root ${NVRTC_ROOT_HINTS})
  list(APPEND NVRTC_INCLUDE_HINTS "${nvrtc_root}/include")
  list(APPEND NVRTC_LIBRARY_HINTS "${nvrtc_root}/lib" "${nvrtc_root}/lib64")
endforeach()
if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
  list(APPEND NVRTC_INCLUDE_HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
list(REMOVE_DUPLICATES NVRTC_INCLUDE_HINTS)
list(REMOVE_DUPLICATES NVRTC_LIBRARY_HINTS)

find_path(NVRTC_INCLUDE_DIR nvrtc.h HINTS ${NVRTC_INCLUDE_HINTS})

set(NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
find_library(NVRTC_DYNAMIC_LIB nvrtc HINTS ${NVRTC_LIBRARY_HINTS})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES)

# Some cross CUDA packages, such as D6L CUDA 11.4, ship only versioned
# libnvrtc.so.* files in the target sysroot. Keep the fallback scoped to the
# toolchain-provided target library hints so cross builds do not pick a host
# x86_64 NVRTC library.
if(NOT NVRTC_DYNAMIC_LIB)
  foreach(nvrtc_library_hint ${NVRTC_LIBRARY_HINTS})
    file(GLOB NVRTC_VERSIONED_LIBS "${nvrtc_library_hint}/libnvrtc.so.*")
    if(NVRTC_VERSIONED_LIBS)
      list(
        SORT NVRTC_VERSIONED_LIBS
        COMPARE NATURAL
        ORDER DESCENDING)
      list(GET NVRTC_VERSIONED_LIBS 0 NVRTC_VERSIONED_LIB)
      set(NVRTC_DYNAMIC_LIB
          "${NVRTC_VERSIONED_LIB}"
          CACHE FILEPATH "Path to the dynamic NVRTC library" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NVRTC_DYNAMIC_LIB)
  set(NVRTC_LIB
      "${NVRTC_DYNAMIC_LIB}"
      CACHE STRING "Path(s) to the NVRTC libraries" FORCE)
  set(NVRTC_LIB_TYPE "dynamic")
else()
  set(NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  find_library(NVRTC_STATIC_LIB nvrtc_static HINTS ${NVRTC_LIBRARY_HINTS})
  find_library(
    NVRTC_BUILTINS_STATIC_LIB
    NAMES nvrtc-builtins_static
    HINTS ${NVRTC_LIBRARY_HINTS})
  find_library(
    NVPTXCOMPILER_STATIC_LIB
    NAMES nvptxcompiler_static
    HINTS ${NVRTC_LIBRARY_HINTS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(NVRTC_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES)

  if(NVRTC_STATIC_LIB)
    if(NOT NVRTC_BUILTINS_STATIC_LIB)
      message(
        FATAL_ERROR
          "Static NVRTC library found at ${NVRTC_STATIC_LIB}, but "
          "libnvrtc-builtins_static.a was not found. "
          "Set CUDA_DIR/CUDA_TARGET_DIR to a CUDA Toolkit with complete static NVRTC libraries."
      )
    endif()
    if(NOT NVPTXCOMPILER_STATIC_LIB)
      message(
        FATAL_ERROR
          "Static NVRTC library found at ${NVRTC_STATIC_LIB}, but "
          "libnvptxcompiler_static.a was not found. "
          "Set CUDA_DIR/CUDA_TARGET_DIR to a CUDA Toolkit with complete static NVRTC libraries."
      )
    endif()
    set(NVRTC_LIB
        "${NVRTC_STATIC_LIB};${NVRTC_BUILTINS_STATIC_LIB};${NVPTXCOMPILER_STATIC_LIB}"
        CACHE STRING "Path(s) to the NVRTC libraries" FORCE)
    set(NVRTC_LIB_TYPE "static")
  endif()
endif()

if(NOT NVRTC_INCLUDE_DIR OR NOT NVRTC_LIB)
  message(
    FATAL_ERROR
      "Plugin build-time NVRTC JIT requires nvrtc.h and libnvrtc.so "
      "or libnvrtc_static.a. "
      "Set NVRTC_ROOT_DIR or CUDA_DIR/CUDA_TARGET_DIR to a CUDA Toolkit with NVRTC."
  )
endif()

message(STATUS "Plugin NVRTC JIT ENABLED (${NVRTC_LIB_TYPE}): ${NVRTC_LIB}")
