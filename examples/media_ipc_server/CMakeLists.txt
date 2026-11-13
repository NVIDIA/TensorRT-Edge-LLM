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

# NvSci is required; NvBufSurface (Jetson Multimedia API) is optional.
# NVSCI_ROOT overrides where to look.
if(NOT DEFINED NVSCI_ROOT AND DEFINED ENV{NVSCI_ROOT})
  set(NVSCI_ROOT $ENV{NVSCI_ROOT})
endif()
find_path(
  NVSCI_INCLUDE_DIR nvscibuf.h
  HINTS "${NVSCI_ROOT}/include" "${NVSCI_ROOT}"
  PATHS /usr/src/jetson_sipl_api/sipl/include/nvsci /usr/include/nvsci
        /usr/local/driveos/include /usr/include)
find_path(
  NVBUFSURFACE_INCLUDE_DIR nvbufsurface.h
  HINTS "${NVSCI_ROOT}/include" "${NVSCI_ROOT}"
  PATHS /usr/src/jetson_multimedia_api/include
        /usr/src/jetson_sipl_api/sipl/include /usr/include)
foreach(_nvsci_lib nvscibuf nvsciipc nvbufsurface)
  find_library(
    ${_nvsci_lib}_LIB ${_nvsci_lib}
    HINTS "${NVSCI_ROOT}/lib" "${NVSCI_ROOT}/lib64"
    PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra
          /usr/lib/aarch64-linux-gnu /usr/lib)
endforeach()
if(NOT NVSCI_INCLUDE_DIR
   OR NOT nvscibuf_LIB
   OR NOT nvsciipc_LIB)
  message(FATAL_ERROR "NvSci (nvscibuf.h, libnvscibuf, libnvsciipc) not found. "
                      "Set NVSCI_ROOT to its install location.")
endif()

add_library(edgellm_ipc_message INTERFACE)
target_include_directories(
  edgellm_ipc_message INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/message
                                ${NVSCI_INCLUDE_DIR})
target_link_libraries(edgellm_ipc_message INTERFACE ${nvsciipc_LIB}
                                                    ${nvscibuf_LIB})
if(NVBUFSURFACE_INCLUDE_DIR AND nvbufsurface_LIB)
  set(EDGELLM_IPC_HAS_NVBUFSURFACE 1)
  target_include_directories(edgellm_ipc_message
                             INTERFACE ${NVBUFSURFACE_INCLUDE_DIR})
  target_link_libraries(edgellm_ipc_message INTERFACE ${nvbufsurface_LIB})
else()
  message(STATUS "NvBufSurface: not found, frames can arrive only as NvSciBuf")
  set(EDGELLM_IPC_HAS_NVBUFSURFACE 0)
endif()
target_compile_definitions(
  edgellm_ipc_message
  INTERFACE EDGELLM_IPC_HAS_NVBUFSURFACE=${EDGELLM_IPC_HAS_NVBUFSURFACE})
target_compile_options(edgellm_ipc_message INTERFACE -Wall -Wextra -Wpedantic)

add_executable(media_ipc_server server/inferenceServer.cpp)
target_include_directories(media_ipc_server PRIVATE server)
target_link_libraries(media_ipc_server PRIVATE edgellm_ipc_message
                                               TensorRTEdgeLLM::Core)
set_target_properties(media_ipc_server PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)
add_dependencies(media_ipc_server NvInfer_edgellm_plugin)
add_cross_build_link_options(media_ipc_server)

add_executable(media_ipc_client client/exampleClient.cpp)
target_include_directories(media_ipc_client PRIVATE client)
target_link_libraries(media_ipc_client PRIVATE edgellm_ipc_message)
