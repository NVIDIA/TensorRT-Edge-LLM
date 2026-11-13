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

set(_trt_safe_hints)
if(TENSORRT_SAFE_ROOT)
  list(APPEND _trt_safe_hints ${TENSORRT_SAFE_ROOT})
endif()
if(TRT_PACKAGE_DIR)
  list(APPEND _trt_safe_hints ${TRT_PACKAGE_DIR})
endif()
if(TENSORRT_ROOT)
  list(APPEND _trt_safe_hints ${TENSORRT_ROOT})
endif()
list(APPEND _trt_safe_hints /usr /opt/tensorrt)
list(REMOVE_DUPLICATES _trt_safe_hints)
set(_trt_safe_lib_suffixes lib lib64 lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set(_trt_safe_inc_suffixes include include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND _trt_safe_lib_suffixes lib/aarch64-qnx)
  list(APPEND _trt_safe_inc_suffixes include/aarch64-qnx)
endif()

find_path(
  TensorRTSafe_INCLUDE_DIR
  NAMES NvInferSafeRuntime.h
  HINTS ${_trt_safe_hints}
  PATH_SUFFIXES ${_trt_safe_inc_suffixes})
find_library(
  TensorRTSafe_LIBRARY
  NAMES nvinfer_safe
  HINTS ${_trt_safe_hints}
  PATH_SUFFIXES ${_trt_safe_lib_suffixes})

unset(_trt_safe_hints)
unset(_trt_safe_lib_suffixes)
unset(_trt_safe_inc_suffixes)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRTSafe
  REQUIRED_VARS
    TensorRTSafe_INCLUDE_DIR
    TensorRTSafe_LIBRARY
    REASON_FAILURE_MESSAGE
    "TensorRT Safety Runtime not found. Specify -DTENSORRT_SAFE_ROOT=/path/to/TRT-safety or -DTRT_PACKAGE_DIR=/path/to/TRT."
)

mark_as_advanced(TensorRTSafe_INCLUDE_DIR TensorRTSafe_LIBRARY)

if(TensorRTSafe_FOUND)
  message(STATUS "TensorRT Safety Runtime: ${TensorRTSafe_LIBRARY}")
  if(NOT TARGET TensorRT::TensorRTSafe)
    add_library(TensorRT::TensorRTSafe UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::TensorRTSafe
      PROPERTIES IMPORTED_LOCATION "${TensorRTSafe_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${TensorRTSafe_INCLUDE_DIR}")
  endif()
endif()
