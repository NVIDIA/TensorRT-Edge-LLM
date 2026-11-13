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

message(STATUS "Configuring experimental model: nemotron3_5_asr")

# Standalone RNN-T transcription runtime (FastConformer encoder + LSTM
# prediction-network greedy decode) and its example CLI. They link the core
# libraries: edgellmCore, edgellmApplicationSupport, commonLibraryExt. The
# Python export and the shared audio engine builder (audio_build) stay in the
# main tree; only the model-specific inference runtime lives here.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt)
  add_subdirectory(cpp)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
  add_subdirectory(examples)
endif()
