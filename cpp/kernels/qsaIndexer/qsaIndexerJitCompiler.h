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
#include <tuple>
#include <vector>

namespace trt_edgellm
{

enum class QsaIndexerJitDataType : uint32_t
{
    kHALF = 0,
    kBF16 = 1,
};

//! \brief Key that identifies one NVRTC-compiled QSA indexer module variant.
//!
//! One compilation produces a single cubin holding all ten extern "C" entry points
//! (qsa_indexer_q_prep, qsa_indexer_k_compress, qsa_indexer_k_compress_paged,
//! qsa_indexer_scores, qsa_indexer_ids_fill, qsa_indexer_expand,
//! qsa_indexer_raw_k_tail_write_prefill, qsa_indexer_pre_decode,
//! qsa_indexer_scores_decode, qsa_indexer_topk_expand_decode) specialized for
//! (sm, dataType).
struct QsaIndexerJitKey
{
    int32_t sm{};
    QsaIndexerJitDataType dataType{QsaIndexerJitDataType::kHALF};

    auto asTuple() const noexcept
    {
        return std::tie(sm, dataType);
    }

    bool operator==(QsaIndexerJitKey const& other) const noexcept
    {
        return asTuple() == other.asTuple();
    }
};

struct QsaIndexerJitKernel
{
    QsaIndexerJitKey key;
    std::vector<uint8_t> cubin;
};

bool canCompileQsaIndexerJitKernel(QsaIndexerJitKey const& key) noexcept;

//! \brief Compile the QSA indexer kernels for one (sm, dataType) with NVRTC.
//!
//! Results are shared process-wide through a compile cache keyed by the full key, so
//! repeated calls (including concurrent ones) pay for at most one NVRTC compilation.
//! \throws std::invalid_argument on an unsupported key, std::runtime_error on NVRTC failure.
QsaIndexerJitKernel compileQsaIndexerJitKernel(QsaIndexerJitKey const& key);

} // namespace trt_edgellm
