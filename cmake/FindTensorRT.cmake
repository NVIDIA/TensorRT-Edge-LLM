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

# Honor TRT_PACKAGE_DIR as the primary root hint (facilitates multi-version x86
# systems). Falls back to standard system paths (e.g. JetPack).
if(DEFINED TRT_PACKAGE_DIR AND NOT DEFINED TENSORRT_ROOT)
  set(TENSORRT_ROOT ${TRT_PACKAGE_DIR})
endif()
set(_trt_hints ${TENSORRT_ROOT} /usr /opt/tensorrt)
set(_trt_lib_suffixes lib lib64 lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
set(_trt_inc_suffixes include include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  list(APPEND _trt_lib_suffixes lib/aarch64-qnx)
  list(APPEND _trt_inc_suffixes include/aarch64-qnx)
endif()

find_path(
  TensorRT_INCLUDE_DIR NvInfer.h
  HINTS ${_trt_hints}
  PATH_SUFFIXES ${_trt_inc_suffixes})
# Find either one of tensorrt_rtx or nvinfer from the provided path, so only one
# lib should be presented in the lib folder.
find_library(
  TensorRT_LIBRARY
  NAMES tensorrt_rtx nvinfer
  HINTS ${_trt_hints}
  PATH_SUFFIXES ${_trt_lib_suffixes})

find_path(TensorRT_OnnxParser_INCLUDE_DIR NvOnnxParser.h
          HINTS ${TensorRT_INCLUDE_DIR})
# TRT-RTX ships the parser as tensorrt_onnxparser_rtx; Enterprise as
# nvonnxparser.
find_library(
  TensorRT_OnnxParser_LIBRARY
  NAMES nvonnxparser tensorrt_onnxparser_rtx
  HINTS ${_trt_hints}
  PATH_SUFFIXES ${_trt_lib_suffixes})

unset(_trt_hints)
unset(_trt_lib_suffixes)
unset(_trt_inc_suffixes)

if(TensorRT_OnnxParser_INCLUDE_DIR AND TensorRT_OnnxParser_LIBRARY)
  set(TensorRT_OnnxParser_FOUND TRUE)
else()
  set(TensorRT_OnnxParser_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
  HANDLE_COMPONENTS REASON_FAILURE_MESSAGE
  "TensorRT not found. Please specify -DTRT_PACKAGE_DIR=/path/to/TRT.")

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY
                 TensorRT_OnnxParser_INCLUDE_DIR TensorRT_OnnxParser_LIBRARY)

if(TensorRT_FOUND)
  # Make the resolved runtime unmistakable in the configure log.
  get_filename_component(_trt_lib_name "${TensorRT_LIBRARY}" NAME)
  if(_trt_lib_name MATCHES "tensorrt_rtx")
    message(
      STATUS
        "TensorRT: linking TRT-RTX (${_trt_lib_name}) -> ${TensorRT_LIBRARY}")
  else()
    message(
      STATUS
        "TensorRT: linking TensorRT (${_trt_lib_name}) -> ${TensorRT_LIBRARY}")
  endif()
  unset(_trt_lib_name)

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::TensorRT
      PROPERTIES IMPORTED_LOCATION "${TensorRT_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()

  if(TensorRT_OnnxParser_FOUND AND NOT TARGET TensorRT::OnnxParser)
    add_library(TensorRT::OnnxParser UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::OnnxParser
      PROPERTIES IMPORTED_LOCATION "${TensorRT_OnnxParser_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES
                 "${TensorRT_OnnxParser_INCLUDE_DIR}"
                 INTERFACE_LINK_LIBRARIES TensorRT::TensorRT)
  endif()
endif()

# Compatibility variables used by the current builder and example targets.
set(ONNX_PARSER_INCLUDE_DIR "${TensorRT_OnnxParser_INCLUDE_DIR}")
set(NV_ONNX_PARSER_LIB TensorRT::OnnxParser)
