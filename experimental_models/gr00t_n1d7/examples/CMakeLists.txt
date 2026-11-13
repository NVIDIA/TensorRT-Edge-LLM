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

add_executable(gr00t_policy_build gr00t_policy_build.cpp)
target_link_libraries(
  gr00t_policy_build PRIVATE gr00tExperimental edgellmBuilder edgellmCore
                             commonLibraryExt ${NV_ONNX_PARSER_LIB})
target_include_directories(
  gr00t_policy_build PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${ONNX_PARSER_INCLUDE_DIR}
                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(gr00t_policy_build)

add_executable(gr00t_policy_inference gr00t_policy_inference.cpp)
target_link_libraries(gr00t_policy_inference
                      PRIVATE gr00tExperimental edgellmCore commonLibraryExt)
target_include_directories(
  gr00t_policy_inference PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(gr00t_policy_inference)
