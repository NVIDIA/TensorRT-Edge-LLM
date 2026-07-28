# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier:
# LicenseRef-NvidiaProprietary

add_executable(wfm_inference wfm_inference.cpp)
target_link_libraries(wfm_inference PRIVATE edgellmCore exampleUtils
                                            commonLibraryExt)
target_include_directories(
  wfm_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(wfm_inference)
