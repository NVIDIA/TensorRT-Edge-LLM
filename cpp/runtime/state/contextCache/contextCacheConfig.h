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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Request-level lookup policy for a runtime whose context cache is enabled.
enum class ContextCacheLookupPolicy : uint8_t
{
    kUseCache,
    //! Use coordinator-managed private pages without consulting or publishing reusable state.
    kBypass,
};

//! Request-level publication policy for ready context-cache endpoints.
enum class ContextCacheCommitPolicy : uint8_t
{
    kIncludingGeneratedTokens,
    //! Publish only prefill state; every kDecodeEnd publication is suppressed. Hybrid+MTP reuse depends on this: its
    //! draft KV lags the base cache by the last accepted span (see MTPDecoder::decodeStep), so a decode-end
    //! publication would record a short draft cache. Widening this policy must flush that pending accept first.
    kPrefillStateOnly,
};

//! Runtime configuration for one deployment-scoped context cache.
//!
//! One enabled runtime is one trusted cache domain. Callers that require tenant isolation must use separate runtime
//! instances; request-level cache-domain partitioning is not part of this interface.
struct ContextCacheConfig
{
    bool enabled{false};      //!< Leave the existing identity-page runtime path unchanged when false.
    int32_t maxRecords{1024}; //!< Maximum retained publication endpoints; zero keeps no reusable records.
    //! Device-memory budget for preallocated recurrent/conv checkpoints.
    int64_t recurrentSnapshotPoolBytes{};
    //! Device-memory budget for preallocated partial attention-KV checkpoints.
    int64_t partialKvSnapshotPoolBytes{};
    //! Device-memory budget for encoder embedding cache (ViT/audio). Zero disables.
    int64_t encoderEmbeddingCacheBudgetBytes{256LL * 1024 * 1024};
};

} // namespace rt
} // namespace trt_edgellm
