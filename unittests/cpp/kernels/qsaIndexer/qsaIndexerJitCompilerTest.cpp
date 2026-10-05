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

#include "kernels/qsaIndexer/qsaIndexerJitCompiler.h"
#include "common/cudaUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace
{

bool hasCudaDevice()
{
    int32_t deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

bool isNativeCubin(std::vector<uint8_t> const& cubin)
{
    return cubin.size() > 4 && cubin[0] == 0x7FU && cubin[1] == 0x45U && cubin[2] == 0x4CU && cubin[3] == 0x46U;
}

} // namespace

TEST(QsaIndexerJitCompiler, KeyValidation)
{
    QsaIndexerJitKey key;
    key.sm = 110;
    EXPECT_TRUE(canCompileQsaIndexerJitKernel(key));
    key.dataType = QsaIndexerJitDataType::kBF16;
    EXPECT_TRUE(canCompileQsaIndexerJitKernel(key));

    QsaIndexerJitKey badSm = key;
    badSm.sm = 0;
    EXPECT_FALSE(canCompileQsaIndexerJitKernel(badSm));

    QsaIndexerJitKey badDataType = key;
    badDataType.dataType = static_cast<QsaIndexerJitDataType>(2);
    EXPECT_FALSE(canCompileQsaIndexerJitKernel(badDataType));
}

//! The kernels are arch-generic; NVRTC-compile both dtype variants for whatever device is
//! present and require a native (non-empty ELF) cubin, mirroring the production contract.
TEST(QsaIndexerJitCompiler, CompilesBothDataTypesForCurrentDevice)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    for (QsaIndexerJitDataType const dataType : {QsaIndexerJitDataType::kHALF, QsaIndexerJitDataType::kBF16})
    {
        QsaIndexerJitKey key;
        key.sm = getSMVersion();
        key.dataType = dataType;
        QsaIndexerJitKernel const kernel = compileQsaIndexerJitKernel(key);
        EXPECT_TRUE(kernel.key == key);
        EXPECT_TRUE(isNativeCubin(kernel.cubin)) << "dtype " << static_cast<uint32_t>(dataType);
    }
}

} // namespace trt_edgellm
