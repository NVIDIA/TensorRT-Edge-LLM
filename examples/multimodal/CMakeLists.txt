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
target_include_directories(visual_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(visual_build)
