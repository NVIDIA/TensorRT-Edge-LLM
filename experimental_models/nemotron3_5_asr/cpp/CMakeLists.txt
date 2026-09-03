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

# Nemotron-3.5-ASR experimental runtime: only the model-specific inference
# runtime (greedy RNN-T transducer loop over the encoder + step engines). Shared
# infrastructure it depends on (mel front-end, tokenizer, row-wise argmax
# kernel) stays in the core libraries and is provided at link time by the
# executables.

set(NEMOTRON_ASR_CPP_SRCS runtime/nemotronAsrRuntime.cpp)

add_library(nemotronAsrExperimental STATIC ${NEMOTRON_ASR_CPP_SRCS})
set_target_properties(nemotronAsrExperimental
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

# A static library only needs the core headers to compile; it does not link the
# core libraries (the executables do). Avoid a build-order dependency on the
# core targets so this library compiles independently of a core rebuild.
target_include_directories(
  nemotronAsrExperimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS})
