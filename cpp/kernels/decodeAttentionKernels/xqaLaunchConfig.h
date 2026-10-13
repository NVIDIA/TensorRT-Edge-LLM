/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/cudaMacros.h"

#if SUPPORTS_CLUSTER_LAUNCH || SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
#include <cuda.h>

#include <array>
#endif

#include <cstdint>

namespace trt_edgellm
{

constexpr int32_t kXQA_MIN_PDL_SM_VERSION{90};

struct XQALaunchAttributes
{
#if SUPPORTS_CLUSTER_LAUNCH || SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    static constexpr uint32_t kMAX_ATTRIBUTES{2U};
    std::array<CUlaunchAttribute, kMAX_ATTRIBUTES> attrs{};
#endif
    uint32_t count{0U};
};

inline XQALaunchAttributes makeXQALaunchAttributes(
    bool const useClusterLaunch, uint32_t const clusterDimX, bool const enablePdl, int32_t const smVersion) noexcept
{
    XQALaunchAttributes launchAttributes{};

#if SUPPORTS_CLUSTER_LAUNCH
    if (useClusterLaunch)
    {
        CUlaunchAttribute& clusterAttribute = launchAttributes.attrs[launchAttributes.count++];
        clusterAttribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        clusterAttribute.value.clusterDim.x = clusterDimX;
        clusterAttribute.value.clusterDim.y = 1U;
        clusterAttribute.value.clusterDim.z = 1U;
    }
#else
    (void) useClusterLaunch;
    (void) clusterDimX;
#endif // SUPPORTS_CLUSTER_LAUNCH

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    if (enablePdl && smVersion >= kXQA_MIN_PDL_SM_VERSION)
    {
        CUlaunchAttribute& pdlAttribute = launchAttributes.attrs[launchAttributes.count++];
        pdlAttribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        pdlAttribute.value.programmaticStreamSerializationAllowed = 1;
    }
#else
    (void) enablePdl;
    (void) smVersion;
#endif // SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH

    return launchAttributes;
}

} // namespace trt_edgellm
