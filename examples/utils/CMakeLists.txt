# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

file(GLOB_RECURSE EXAMPLE_UTILS_CPP_SOURCES "*.cpp")
file(GLOB_RECURSE EXAMPLE_UTILS_CU_SOURCES "*.cu")

add_library(exampleUtils STATIC ${EXAMPLE_UTILS_CPP_SOURCES}
                                ${EXAMPLE_UTILS_CU_SOURCES})

target_include_directories(exampleUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                               ${COMMON_INCLUDE_DIRS})

target_link_libraries(exampleUtils PRIVATE edgellmCore commonLibraryExt)
