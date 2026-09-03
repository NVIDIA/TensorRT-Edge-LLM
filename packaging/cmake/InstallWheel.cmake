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

if(NOT TARGET _edgellm_runtime OR NOT TARGET NvInfer_edgellm_plugin)
  message(
    FATAL_ERROR
      "Wheel install rules require the existing pybind and plugin targets.")
endif()

if(IS_ABSOLUTE "${EDGELLM_WHEEL_PAYLOAD_DIR}" OR EDGELLM_WHEEL_PAYLOAD_DIR
                                                 MATCHES "(^|/)\\.\\.(/|$)")
  message(FATAL_ERROR "EDGELLM_WHEEL_PAYLOAD_DIR must be a safe relative path.")
endif()

if(NOT DEFINED EDGELLM_WHEEL_EXTENSION_NAME
   OR NOT EDGELLM_WHEEL_EXTENSION_NAME MATCHES
      "^_edgellm_runtime\\.cpython-[0-9]+-(x86_64|aarch64)-linux-gnu\\.so$")
  message(
    FATAL_ERROR
      "EDGELLM_WHEEL_EXTENSION_NAME must be an explicit target CPython filename."
  )
endif()

set_target_properties(
  _edgellm_runtime
  PROPERTIES BUILD_RPATH_USE_ORIGIN ON
             BUILD_WITH_INSTALL_RPATH ON
             INSTALL_RPATH "$ORIGIN/lib")
set_target_properties(
  NvInfer_edgellm_plugin
  PROPERTIES BUILD_RPATH_USE_ORIGIN ON
             BUILD_WITH_INSTALL_RPATH ON
             INSTALL_RPATH "$ORIGIN")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(NvInfer_edgellm_plugin PRIVATE "-Wl,--no-undefined")
endif()

set(_EDGELLM_WHEEL_LINK_MAP_DIR "${CMAKE_BINARY_DIR}/wheel-link-maps")
file(MAKE_DIRECTORY "${_EDGELLM_WHEEL_LINK_MAP_DIR}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(_edgellm_runtime PRIVATE
                      "-Wl,-Map,${_EDGELLM_WHEEL_LINK_MAP_DIR}/extension.map")
  target_link_options(NvInfer_edgellm_plugin PRIVATE
                      "-Wl,-Map,${_EDGELLM_WHEEL_LINK_MAP_DIR}/plugin.map")
endif()

# FILES with TARGET_FILE copies the resolved versioned plugin DSO as the stable
# filename consumed by the unchanged runtime, without shipping wheel symlinks.
install(
  FILES "$<TARGET_FILE:_edgellm_runtime>"
  DESTINATION "${EDGELLM_WHEEL_PAYLOAD_DIR}"
  RENAME "${EDGELLM_WHEEL_EXTENSION_NAME}")
install(
  FILES "$<TARGET_FILE:NvInfer_edgellm_plugin>"
  DESTINATION "${EDGELLM_WHEEL_PAYLOAD_DIR}/lib"
  RENAME "libNvInfer_edgellm_plugin.so")
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
        DESTINATION "${EDGELLM_WHEEL_PAYLOAD_DIR}/licenses")
