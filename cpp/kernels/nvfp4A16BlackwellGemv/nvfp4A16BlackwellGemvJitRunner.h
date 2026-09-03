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

#include "nvfp4A16BlackwellGemvJitCompiler.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace trt_edgellm
{

struct Nvfp4A16BlackwellGemvLoadedModule;

bool isNvfp4A16BlackwellGemvJitSupported(Nvfp4A16BlackwellGemvJitKey const& key, int32_t m, int32_t splitK) noexcept;

size_t getNvfp4A16BlackwellGemvJitWorkspaceSize(
    Nvfp4A16BlackwellGemvJitKey const& key, int32_t m, int32_t splitK) noexcept;

class Nvfp4A16BlackwellGemvJitRunner
{
public:
    void load(Nvfp4A16BlackwellGemvJitKernel const& kernel);

    void launch(void const* activation, uint8_t const* qweights, uint8_t const* blockScales, float const* globalScale,
        void* output, void* workspace, size_t workspaceSize, int32_t m, int32_t splitK, cudaStream_t stream) const;

    bool isLoaded() const noexcept
    {
        return mLoadedModule != nullptr;
    }

    Nvfp4A16BlackwellGemvJitKey const& getKey() const noexcept
    {
        return mKey;
    }

private:
    Nvfp4A16BlackwellGemvJitKey mKey{};
    Nvfp4A16BlackwellGemvJitDigest mDigest{};
    std::shared_ptr<Nvfp4A16BlackwellGemvLoadedModule> mLoadedModule;
};

} // namespace trt_edgellm
