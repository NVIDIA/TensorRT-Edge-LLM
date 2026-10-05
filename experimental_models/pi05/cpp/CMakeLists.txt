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

# pi0.5 experimental runtime: links the core edgellm libraries and only adds
# pi0.5-specific code (component builder, the policy runtime and the visual and
# action runners it drives, the Euler-update kernel, policy
# pre/post-processing).

set(PI05_CPP_SRCS
    builder/pi05Builder.cpp
    common/pi05Common.cpp
    kernels/pi05Kernels.cu
    multimodal/pi05VisualRunner.cpp
    action/pi05ActionRunner.cpp
    runtime/pi05Runtime.cpp
    runtime/pi05Policy.cpp)

add_library(pi05Experimental STATIC ${PI05_CPP_SRCS})

# A static library only needs the core headers to compile; the executables link
# the core libraries. Avoids a build-order dependency on a core rebuild.
target_include_directories(
  pi05Experimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR})
