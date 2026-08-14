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

set(INTERNVLA_N1_CPP_SRCS action/internvlaN1Scheduler.cpp
                          action/internvlaN1Kernels.cu
                          action/internvlaN1System1Runner.cpp
                          runtime/internvlaN1DualSystem.cpp)

add_library(internvlaN1Experimental STATIC ${INTERNVLA_N1_CPP_SRCS})

# A static library only needs the core headers to compile; the executables do
# the linking. This keeps it independent of a core rebuild.
target_include_directories(
  internvlaN1Experimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR})
set_property(TARGET internvlaN1Experimental PROPERTY POSITION_INDEPENDENT_CODE ON)
