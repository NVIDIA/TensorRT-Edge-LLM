/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ssdVarlenMetadata.h"

#include <cstdio>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace mamba
{

namespace
{

/// Generate seq_idx tensor [B, T]. seq_idx[b, t] = b if t < cl[b] else -1.
/// When d_context_lengths is null, treat all positions as valid (cl[b] = seq_len).
__global__ void buildSeqIdxKernel(int32_t* d_seq_idx, int32_t const* d_context_lengths, int32_t batch, int32_t seq_len)
{
    int32_t const t = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const b = blockIdx.y;
    if (b >= batch || t >= seq_len)
        return;
    int32_t const cl = d_context_lengths ? d_context_lengths[b] : seq_len;
    d_seq_idx[b * seq_len + t] = (t < cl) ? b : -1;
}

/// Fused metadata kernel: prefix-sums chunks_per_seq into seq_chunk_cumsum, then fills
/// chunk_indices/chunk_offsets for the full upper-bound region. Real slots map each
/// sequence's [b * seq_len, b * seq_len + context_lengths[b]) interval into the flat
/// physical 128-token chunk grid. Trailing slack gets sentinel -1 so the SSD kernel's
/// chunk_indices[physical_chunk+1] safety read can't false-trigger on garbage.
/// Single block -- batch is small (<= 256), inter-block sync would be wasted.
__global__ void buildChunkMetadataKernel(int32_t* d_seq_chunk_cumsum, // [batch+1] int32
    int32_t* d_chunk_indices,                                         // [num_logical_chunks_upper] int32
    int32_t* d_chunk_offsets,                                         // [num_logical_chunks_upper] int32
    int32_t const* d_context_lengths,                                 // [batch] int32 or nullptr
    int32_t batch, int32_t seq_len, int32_t chunk_size, int32_t total_slots)
{
    int32_t const tid = threadIdx.x;
    extern __shared__ int32_t smem[];

    if (tid < batch)
    {
        int32_t const cl = d_context_lengths ? d_context_lengths[tid] : seq_len;
        int32_t const seq_start = tid * seq_len;
        int32_t const start_offset = seq_start - (seq_start / chunk_size) * chunk_size;
        smem[tid] = cl > 0 ? (start_offset + cl + chunk_size - 1) / chunk_size : 0;
    }
    __syncthreads();

    // Sequential scan/fill in thread 0 -- batch is small enough that parallel scan setup costs more.
    if (tid == 0)
    {
        int32_t cum = 0;
        d_seq_chunk_cumsum[0] = 0;
        for (int32_t b = 0; b < batch; ++b)
        {
            int32_t const cl = d_context_lengths ? d_context_lengths[b] : seq_len;
            int32_t flat = b * seq_len;
            int32_t remaining = cl;
            for (int32_t c = 0; c < smem[b]; ++c)
            {
                int32_t const offset = flat - (flat / chunk_size) * chunk_size;
                d_chunk_indices[cum + c] = flat / chunk_size;
                d_chunk_offsets[cum + c] = offset;
                int32_t const room = chunk_size - offset;
                int32_t const chunk_tokens = room < remaining ? room : remaining;
                flat += chunk_tokens;
                remaining -= chunk_tokens;
            }
            cum += smem[b];
            d_seq_chunk_cumsum[b + 1] = cum;
        }
        smem[batch] = cum;
    }
    __syncthreads();

    int32_t const real_end = smem[batch];
    for (int32_t k = real_end + tid; k < total_slots; k += blockDim.x)
    {
        d_chunk_indices[k] = -1;
        d_chunk_offsets[k] = 0;
    }
}

/// Token range of packed sequence b, clamped to the token axis so malformed offsets stay in bounds.
__device__ __forceinline__ void packedSequenceRange(
    int32_t const* d_cu_seqlens, int32_t b, int32_t total_tokens, int32_t& start, int32_t& length)
{
    start = min(max(d_cu_seqlens[b], 0), total_tokens);
    length = min(max(d_cu_seqlens[b + 1], start), total_tokens) - start;
}

/// seq_idx[t] = owning sequence of token t: the last b with cu_seqlens[b] <= t.
__global__ void buildPackedSeqIdxKernel(
    int32_t* d_seq_idx, int32_t const* d_cu_seqlens, int32_t num_seqs, int32_t total_tokens)
{
    int32_t const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_tokens)
        return;
    int32_t lo = 0;
    int32_t hi = num_seqs;
    while (hi - lo > 1)
    {
        int32_t const mid = (lo + hi) / 2;
        if (d_cu_seqlens[mid] <= t)
            lo = mid;
        else
            hi = mid;
    }
    int32_t start{};
    int32_t length{};
    packedSequenceRange(d_cu_seqlens, lo, total_tokens, start, length);
    d_seq_idx[t] = (t >= start && t < start + length) ? lo : -1;
}

/// Packed counterpart of buildChunkMetadataKernel: logical chunks are the (sequence, physical chunk) pairs of each
/// sequence's token range.
__global__ void buildPackedChunkMetadataKernel(int32_t* d_seq_chunk_cumsum, int32_t* d_chunk_indices,
    int32_t* d_chunk_offsets, int32_t const* d_cu_seqlens, int32_t num_seqs, int32_t total_tokens, int32_t chunk_size,
    int32_t total_slots)
{
    int32_t const tid = threadIdx.x;
    extern __shared__ int32_t smem[];

    if (tid < num_seqs)
    {
        int32_t start{};
        int32_t length{};
        packedSequenceRange(d_cu_seqlens, tid, total_tokens, start, length);
        smem[tid] = length > 0 ? (start % chunk_size + length + chunk_size - 1) / chunk_size : 0;
    }
    __syncthreads();

    if (tid == 0)
    {
        int32_t cum = 0;
        d_seq_chunk_cumsum[0] = 0;
        for (int32_t b = 0; b < num_seqs; ++b)
        {
            int32_t start{};
            int32_t length{};
            packedSequenceRange(d_cu_seqlens, b, total_tokens, start, length);
            int32_t const count = min(smem[b], total_slots - cum);
            int32_t flat = start;
            int32_t remaining = length;
            for (int32_t c = 0; c < count; ++c)
            {
                int32_t const offset = flat % chunk_size;
                d_chunk_indices[cum + c] = flat / chunk_size;
                d_chunk_offsets[cum + c] = offset;
                int32_t const chunk_tokens = min(chunk_size - offset, remaining);
                flat += chunk_tokens;
                remaining -= chunk_tokens;
            }
            cum += count;
            d_seq_chunk_cumsum[b + 1] = cum;
        }
        smem[num_seqs] = cum;
    }
    __syncthreads();

    for (int32_t k = smem[num_seqs] + tid; k < total_slots; k += blockDim.x)
    {
        d_chunk_indices[k] = -1;
        d_chunk_offsets[k] = 0;
    }
}

__global__ void fillValidLensFromCuSeqlensKernel(
    int32_t* d_valid_lens, int32_t const* d_cu_seqlens, int32_t num_seqs, int32_t total_tokens)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_seqs)
    {
        int32_t start{};
        int32_t length{};
        packedSequenceRange(d_cu_seqlens, idx, total_tokens, start, length);
        d_valid_lens[idx] = length;
    }
}

/// Fill valid_lens[0..batch) with uniform seq_len.
__global__ void fillUniformValidLensKernel(int32_t* d_valid_lens, int32_t batch, int32_t seq_len)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch)
    {
        d_valid_lens[idx] = seq_len;
    }
}

} // anonymous namespace

void fillUniformValidLens(int32_t* d_valid_lens, int32_t batch, int32_t seq_len, cudaStream_t stream)
{
    int32_t const block = 64;
    int32_t const grid = (batch + block - 1) / block;
    fillUniformValidLensKernel<<<grid, block, 0, stream>>>(d_valid_lens, batch, seq_len);
}

void buildSSDPackedVarlenMetadata(int32_t* d_seq_idx, int32_t* d_chunk_indices, int32_t* d_chunk_offsets,
    int32_t* d_seq_chunk_cumsum, int32_t const* d_cu_seqlens, int32_t num_seqs, int32_t total_tokens,
    int32_t chunk_size, int32_t total_slots, cudaStream_t stream)
{
    int32_t const threads = (num_seqs <= 256) ? 256 : ((num_seqs + 31) / 32) * 32;
    size_t const smem_bytes = static_cast<size_t>(num_seqs + 1) * sizeof(int32_t);
    buildPackedChunkMetadataKernel<<<1, threads, smem_bytes, stream>>>(d_seq_chunk_cumsum, d_chunk_indices,
        d_chunk_offsets, d_cu_seqlens, num_seqs, total_tokens, chunk_size, total_slots);
    if (total_tokens > 0)
    {
        int32_t const threadsX = 128;
        buildPackedSeqIdxKernel<<<(total_tokens + threadsX - 1) / threadsX, threadsX, 0, stream>>>(
            d_seq_idx, d_cu_seqlens, num_seqs, total_tokens);
    }
}

void fillValidLensFromCuSeqlens(
    int32_t* d_valid_lens, int32_t const* d_cu_seqlens, int32_t num_seqs, int32_t total_tokens, cudaStream_t stream)
{
    int32_t const block = 64;
    fillValidLensFromCuSeqlensKernel<<<(num_seqs + block - 1) / block, block, 0, stream>>>(
        d_valid_lens, d_cu_seqlens, num_seqs, total_tokens);
}

void buildSSDVarlenMetadata(int32_t* d_seq_idx, int32_t* d_chunk_indices, int32_t* d_chunk_offsets,
    int32_t* d_seq_chunk_cumsum, int32_t const* d_context_lengths, int32_t batch, int32_t seq_len, int32_t chunk_size,
    cudaStream_t stream)
{
    int32_t const nchunks_per_seq = (seq_len + chunk_size - 1) / chunk_size;
    int32_t const chunks_per_seq_upper = nchunks_per_seq + ((seq_len % chunk_size) == 0 ? 0 : 1);
    int32_t const total_slots = batch * chunks_per_seq_upper;

    // Threads >= max(batch, 256); smem fits chunks_per_seq[batch] + cum total at [batch].
    int32_t const threads = (batch <= 256) ? 256 : ((batch + 31) / 32) * 32;
    size_t const smem_bytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    buildChunkMetadataKernel<<<1, threads, smem_bytes, stream>>>(d_seq_chunk_cumsum, d_chunk_indices, d_chunk_offsets,
        d_context_lengths, batch, seq_len, chunk_size, total_slots);

    int32_t const threadsX = 128;
    dim3 const block(threadsX, 1, 1);
    dim3 const grid((seq_len + threadsX - 1) / threadsX, batch, 1);
    buildSeqIdxKernel<<<grid, block, 0, stream>>>(d_seq_idx, d_context_lengths, batch, seq_len);
}

} // namespace mamba
} // namespace trt_edgellm
