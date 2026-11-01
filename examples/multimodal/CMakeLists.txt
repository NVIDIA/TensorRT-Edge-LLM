# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

add_executable(visual_build visual_build.cpp)
target_link_libraries(visual_build PRIVATE ${NV_ONNX_PARSER_LIB}
                                           commonLibraryExt edgellmBuilder)
target_include_directories(
  visual_build
  PRIVATE ${ONNX_PARSER_INCLUDE_DIR} ${COMMON_INCLUDE_DIRS}
          $<$<PLATFORM_ID:Windows>:${CMAKE_SOURCE_DIR}/examples/utils/windows>)
add_cross_build_link_options(visual_build)

add_executable(audio_build audio_build.cpp)
target_link_libraries(audio_build PRIVATE ${NV_ONNX_PARSER_LIB}
                                          commonLibraryExt edgellmBuilder)
target_include_directories(
  audio_build
  PRIVATE ${ONNX_PARSER_INCLUDE_DIR} ${COMMON_INCLUDE_DIRS}
          $<$<PLATFORM_ID:Windows>:${CMAKE_SOURCE_DIR}/examples/utils/windows>)
add_cross_build_link_options(audio_build)

add_executable(action_build action_build.cpp)
target_link_libraries(action_build PRIVATE ${NV_ONNX_PARSER_LIB}
                                           commonLibraryExt edgellmBuilder)
target_include_directories(
  action_build
  PRIVATE ${ONNX_PARSER_INCLUDE_DIR} ${COMMON_INCLUDE_DIRS}
          $<$<PLATFORM_ID:Windows>:${CMAKE_SOURCE_DIR}/examples/utils/windows>)
add_cross_build_link_options(action_build)

add_executable(action_inference action_inference.cpp)
target_link_libraries(action_inference PRIVATE edgellmCore exampleUtils
                                               commonLibraryExt)
target_include_directories(
  action_inference
  PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/examples/utils
          $<$<PLATFORM_ID:Windows>:${CMAKE_SOURCE_DIR}/examples/utils/windows>)
add_cross_build_link_options(action_inference)
