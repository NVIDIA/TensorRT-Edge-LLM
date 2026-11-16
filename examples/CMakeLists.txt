# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier:
# LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
add_subdirectory(cpp)
if(TARGET edgellmSafetyRuntime)
  add_subdirectory(safety)
endif()

option(BUILD_MEDIA_IPC_SERVER
       "Build the media_ipc_server zero-copy inference server example" OFF)
if(BUILD_MEDIA_IPC_SERVER)
  add_subdirectory(media_ipc_server)
endif()
