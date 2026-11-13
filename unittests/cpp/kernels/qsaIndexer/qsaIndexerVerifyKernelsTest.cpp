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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/qsaAttention/cuteDslQsaSparseRunner.h"
#include "kernels/qsaIndexer/qsaIndexerRunner.h"
#include "qsaIndexerReference.h"
#include "testUtils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kQHeads = 24;
constexpr int32_t kKVHeads = 2;
constexpr int32_t kHeadDim = 256;
constexpr int32_t kPoolDim = 384;
constexpr float kEps = 1.e-6F;

using Path = std::vector<int32_t>;
using Tree = std::vector<Path>;

template <typename T>
constexpr trt_edgellm::DataType dataType()
{
    if constexpr (std::is_same_v<T, half>)
    {
        return trt_edgellm::DataType::kHALF;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return trt_edgellm::DataType::kBF16;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return trt_edgellm::DataType::kFLOAT;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return trt_edgellm::DataType::kINT32;
    }
    else
    {
        return trt_edgellm::DataType::kINT8;
    }
}

template <typename T>
rt::Tensor deviceVector(size_t count)
{
    return rt::Tensor({static_cast<int64_t>(count)}, rt::DeviceType::kGPU, dataType<T>());
}

template <typename T>
rt::Tensor upload(std::vector<T> const& values)
{
    auto result = deviceVector<T>(values.size());
    copyHostToDevice(result, values);
    return result;
}

template <typename T>
std::vector<T> readSlice(void const* base, size_t offset, size_t count)
{
    std::vector<T> result(count);
    CUDA_CHECK(
        cudaMemcpy(result.data(), static_cast<char const*>(base) + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

template <typename T>
void expectBits(T const* actual, T const* expected, size_t count, char const* message)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (std::memcmp(actual + i, expected + i, sizeof(T)) != 0)
        {
            uint32_t a = 0, e = 0;
            std::memcpy(&a, actual + i, sizeof(T));
            std::memcpy(&e, expected + i, sizeof(T));
            EXPECT_EQ(a, e) << message << ", element " << i;
            return;
        }
    }
}

Tree pathsFromChildren(Path const& parents, int32_t count)
{
    Tree children(count), paths(count);
    for (int32_t node = 1; node < count; ++node)
    {
        children[parents[node]].push_back(node);
    }
    Path pending;
    if (count != 0)
    {
        pending.push_back(0);
        paths[0] = {0};
    }
    while (!pending.empty())
    {
        int32_t const node = pending.back();
        pending.pop_back();
        for (int32_t child : children[node])
        {
            paths[child] = paths[node];
            paths[child].push_back(child);
            pending.push_back(child);
        }
    }
    return paths;
}

std::vector<float> ropeRows(Path const& positions)
{
    std::vector<float> result(positions.size() * 64);
    for (size_t t = 0; t < positions.size(); ++t)
    {
        for (int32_t d = 0; d < 32; ++d)
        {
            float const angle = positions[t] * std::pow(10000.F, -static_cast<float>(d) / 32);
            result[t * 64 + d] = std::cos(angle);
            result[t * 64 + 32 + d] = std::sin(angle);
        }
    }
    return result;
}

template <typename T>
struct DenseResult
{
    std::vector<T> q, k;
    std::vector<float> scores;
    Path indices;
};

template <typename T>
struct VerifyCase
{
    int32_t batch, seq, capacity, maxPages, numPages, maxBlocks;
    std::vector<Tree> paths;
    Path pasts, counts, parents, depths, pages;
    std::vector<T> x, pool, keys, values, queries;
    std::vector<float> rope;
    std::vector<std::vector<T>> prefixX, prefixKbar;
    std::vector<std::vector<float>> prefixRope;
    rt::Tensor dX, dRope, dParents, dDepths, dCounts, dPasts, dPool, dPages, dQ;
    rt::Tensor indices, workspace;

    size_t poolOffset(int32_t b, int32_t plane, int32_t token, int32_t head = 0, int32_t col = 0) const
    {
        int32_t const page = pages[(b * 2 + plane) * maxPages + token / 128];
        return ((static_cast<size_t>(page) * 128 + token % 128) * kKVHeads + head) * kPoolDim + col;
    }
};

template <typename T>
class QsaIndexerVerifyTest : public ::testing::Test
{
protected:
    cudaStream_t mStream{};
    rt::Tensor mWQ, mWK;

    void SetUp() override
    {
        int32_t devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        {
            GTEST_SKIP() << "CUDA device required";
        }
        int32_t device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (!CuteDslQsaSparsePrefillRunner::canImplement(
                kQHeads, kKVHeads, kHeadDim, kPoolDim, prop.major * 10 + prop.minor, dataType<T>()))
        {
            GTEST_SKIP() << "QSA sparse attention AOT kernels required on a supported SM";
        }
        // Blocking, so kernels wait for the pageable uploads, which go through the legacy stream.
        CUDA_CHECK(cudaStreamCreate(&mStream));
        std::mt19937 rng(20261007);
        mWQ = upload(randomValues(128, rng, .2F));
        mWK = upload(randomValues(128, rng, .2F));
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
    }

    static std::vector<T> randomValues(size_t count, std::mt19937& rng, float scale)
    {
        std::uniform_real_distribution<float> distribution(-scale, scale);
        std::vector<T> result(count);
        for (auto& value : result)
        {
            value = static_cast<T>(distribution(rng));
        }
        return result;
    }

    DenseResult<T> reference(std::vector<T> const& x, std::vector<float> const& rope)
    {
        int32_t const length = x.size() / 640, blocks = qsa_ref::ceilDiv(length, 4);
        auto dx = upload(x), dr = upload(rope), ctx = upload(Path{length});
        auto q = deviceVector<T>(length * 512), k = deviceVector<T>(blocks * 128);
        auto scores = deviceVector<float>(blocks), idx = deviceVector<int32_t>(qsa_ref::kWidth);
        auto counts = deviceVector<int32_t>(1);
        kernel::launchQsaIndexQPrep(q.template dataPointer<T>(), dx.template dataPointer<T>(),
            dr.template dataPointer<float>(), ctx.template dataPointer<int32_t>(), mWQ.dataPointer<T>(), kEps, 1,
            length, mStream);
        kernel::launchQsaIndexKCompress(k.template dataPointer<T>(), dx.template dataPointer<T>(),
            dr.template dataPointer<float>(), ctx.template dataPointer<int32_t>(), mWK.dataPointer<T>(), kEps, 1,
            length, blocks, mStream);
        kernel::launchQsaIndexScores(scores.template dataPointer<float>(), counts.template dataPointer<int32_t>(),
            q.template dataPointer<T>(), k.template dataPointer<T>(), ctx.template dataPointer<int32_t>(), 1, length,
            blocks, length - 1, 1, mStream);
        kernel::launchQsaTopKExpandDecode(idx.template dataPointer<int32_t>(), scores.template dataPointer<float>(),
            ctx.template dataPointer<int32_t>(), nullptr, kKVHeads, 1, blocks, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return {readSlice<T>(q.rawPointer(), (length - 1) * 512 * sizeof(T), 512), copyDeviceToHost<T>(k),
            copyDeviceToHost<float>(scores), copyDeviceToHost<int32_t>(idx)};
    }

    void updateTree(VerifyCase<T>& c, Path const& parents)
    {
        c.parents = parents;
        c.depths.assign(c.batch * c.seq, -1);
        Path positions(c.batch * c.seq);
        c.paths.clear();
        for (int32_t b = 0; b < c.batch; ++b)
        {
            c.paths.push_back(
                pathsFromChildren(Path(parents.begin() + b * c.seq, parents.begin() + (b + 1) * c.seq), c.counts[b]));
            for (int32_t node = 0; node < c.counts[b]; ++node)
            {
                int32_t const row = b * c.seq + node;
                c.depths[row] = c.paths[b][node].size() - 1;
                positions[row] = c.pasts[b] + c.depths[row];
            }
        }
        c.rope = ropeRows(positions);
        for (int32_t b = 0; b < c.batch; ++b)
        {
            std::fill(c.rope.begin() + (b * c.seq + c.counts[b]) * 64, c.rope.begin() + (b + 1) * c.seq * 64,
                std::numeric_limits<float>::quiet_NaN());
        }
    }

    VerifyCase<T> makeCase(Path pasts, Path counts, Path parents, bool ties = false)
    {
        VerifyCase<T> c;
        c.batch = pasts.size();
        c.seq = parents.size() / c.batch;
        c.pasts = std::move(pasts);
        c.counts = std::move(counts);
        c.capacity = *std::max_element(c.pasts.begin(), c.pasts.end()) + c.seq;
        c.maxPages = qsa_ref::ceilDiv(c.capacity, 128);
        c.numPages = c.batch * c.maxPages + 1;
        c.maxBlocks = qsa_ref::ceilDiv(c.capacity, 4);
        updateTree(c, parents);
        std::mt19937 rng(20261007 + std::accumulate(c.pasts.begin(), c.pasts.end(), 0));
        T const nan = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        c.pool.assign(static_cast<size_t>(2) * c.numPages * 128 * kKVHeads * kPoolDim, nan);
        c.pages = qsa_ref::makePermutedPageTable(c.batch, c.maxPages, c.maxPages, c.numPages, rng());
        c.x = randomValues(c.batch * c.seq * 640, rng, 1.F);
        c.keys = randomValues(c.batch * c.capacity * kKVHeads * kHeadDim, rng, .7F);
        c.values = randomValues(c.keys.size(), rng, .7F);
        c.queries = randomValues(c.batch * c.seq * kQHeads * kHeadDim, rng, .7F);
        for (int32_t b = 0; b < c.batch; ++b)
        {
            int32_t const past = c.pasts[b];
            for (int32_t node = 0; node < c.counts[b]; ++node)
            {
                if (ties)
                {
                    std::fill_n(c.x.data() + (b * c.seq + node) * 640, 512, static_cast<T>(0.F));
                }
            }
            std::fill(c.x.begin() + (b * c.seq + c.counts[b]) * 640, c.x.begin() + (b + 1) * c.seq * 640, nan);
            std::fill(c.queries.begin() + (b * c.seq + c.counts[b]) * kQHeads * kHeadDim,
                c.queries.begin() + (b + 1) * c.seq * kQHeads * kHeadDim, nan);
            for (int32_t token = 0; token < past + c.counts[b]; ++token)
            {
                for (int32_t head = 0; head < kKVHeads; ++head)
                {
                    size_t const offset = ((b * c.capacity + token) * kKVHeads + head) * kHeadDim;
                    std::copy_n(c.keys.data() + offset, kHeadDim, c.pool.data() + c.poolOffset(b, 0, token, head));
                    std::copy_n(c.values.data() + offset, kHeadDim, c.pool.data() + c.poolOffset(b, 1, token, head));
                }
            }
            c.prefixX.push_back(randomValues(past * 640, rng, 1.F));
            Path positions(past);
            std::iota(positions.begin(), positions.end(), 0);
            c.prefixRope.push_back(ropeRows(positions));
            c.prefixKbar.emplace_back();
            if (past != 0)
            {
                c.prefixKbar.back() = reference(c.prefixX.back(), c.prefixRope.back()).k;
                // The newest kQSA_SEAL_BLOCKS blocks arrive unsealed, as a commit may leave them.
                writeTails(c, c.pool, b, c.prefixX.back(), c.prefixRope.back(), c.prefixKbar.back(), past,
                    std::max(0, past / 4 - kernel::kQSA_SEAL_BLOCKS));
            }
        }
        c.dX = upload(c.x);
        c.dRope = upload(c.rope);
        c.dParents = upload(c.parents);
        c.dDepths = upload(c.depths);
        c.dCounts = upload(c.counts);
        c.dPasts = upload(c.pasts);
        c.dPool = upload(c.pool);
        c.dPages = upload(c.pages);
        c.dQ = upload(c.queries);
        size_t const rows = c.batch * c.seq;
        c.indices = deviceVector<int32_t>(rows * qsa_ref::kWidth);
        c.workspace = deviceVector<int8_t>(kernel::getQsaIndexerVerifyWorkspaceSize(c.batch, c.seq, c.maxBlocks));
        CUDA_CHECK(cudaMemsetAsync(c.workspace.rawPointer(), 0xA5, c.workspace.getShape().volume(), mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return c;
    }

    void run(VerifyCase<T>& c, size_t bytes, cudaStream_t stream)
    {
        kernel::QsaIndexerPoolState pool{c.dPool.rawPointer(), c.dPages.template dataPointer<int32_t>(), c.maxPages,
            c.numPages, kKVHeads, kPoolDim, kHeadDim};
        kernel::runQsaIndexerVerify(c.indices.template dataPointer<int32_t>(), c.dX.template dataPointer<T>(),
            c.dRope.template dataPointer<float>(), c.dParents.template dataPointer<int32_t>(),
            c.dDepths.template dataPointer<int32_t>(), c.dCounts.template dataPointer<int32_t>(),
            c.dPasts.template dataPointer<int32_t>(), mWQ.dataPointer<T>(), mWK.dataPointer<T>(), kEps, pool,
            c.workspace.rawPointer(), bytes, c.batch, c.seq, c.maxBlocks, stream);
    }

    void run(VerifyCase<T>& c)
    {
        run(c, c.workspace.getShape().volume(), mStream);
    }

    //! The tails an ordinary prefill of history x leaves for tokens [0, end): raw K, block rope rows, and kbar of the
    //! blocks [0, sealedBlocks).
    static void writeTails(VerifyCase<T> const& c, std::vector<T>& pool, int32_t b, std::vector<T> const& x,
        std::vector<float> const& rope, std::vector<T> const& kbar, int32_t end, int32_t sealedBlocks)
    {
        for (int32_t t = 0; t < end; ++t)
        {
            std::copy_n(x.data() + t * 640 + 512, 128, pool.data() + c.poolOffset(b, 0, t, 0, 256));
        }
        for (int32_t first = 0; first < end; first += 4)
        {
            std::memcpy(static_cast<void*>(pool.data() + c.poolOffset(b, 1, first, 0, 256)), rope.data() + first * 64,
                64 * sizeof(float));
        }
        for (int32_t g = 0; g < sealedBlocks; ++g)
        {
            std::copy_n(kbar.data() + g * 128, 128, pool.data() + c.poolOffset(b, 1, 4 * g + 1, 0, 256));
        }
    }

    //! The pool verify leaves: the prefix sealed, and each valid node's raw index-K in the K-tail of its slot past +
    //! node, plus its rope row in the V-tail when past + depth opens a block. Rejected rows write nothing.
    std::vector<T> verifiedPool(VerifyCase<T> const& c, Path const& rejectedRows = {}) const
    {
        auto expected = c.pool;
        for (int32_t b = 0; b < c.batch; ++b)
        {
            int32_t const past = c.pasts[b];
            writeTails(c, expected, b, c.prefixX[b], c.prefixRope[b], c.prefixKbar[b], past, past / 4);
            for (int32_t node = 0; node < c.counts[b]; ++node)
            {
                int32_t const row = b * c.seq + node, slot = past + node;
                if (std::find(rejectedRows.begin(), rejectedRows.end(), row) != rejectedRows.end())
                {
                    continue;
                }
                std::copy_n(c.x.data() + row * 640 + 512, 128, expected.data() + c.poolOffset(b, 0, slot, 0, 256));
                if ((past + c.depths[row]) % 4 == 0)
                {
                    std::memcpy(static_cast<void*>(expected.data() + c.poolOffset(b, 1, slot, 0, 256)),
                        c.rope.data() + row * 64, 64 * sizeof(float));
                }
            }
        }
        return expected;
    }

    void expectVerifiedPool(VerifyCase<T> const& c, Path const& rejectedRows = {})
    {
        auto const actual = copyDeviceToHost<T>(c.dPool);
        auto const expected = verifiedPool(c, rejectedRows);
        expectBits(actual.data(), expected.data(), expected.size(), "verify pool differs");
    }

    std::vector<T> attention(VerifyCase<T> const& c, rt::Tensor const& indices)
    {
        auto out = deviceVector<T>(c.queries.size());
        QsaSparsePrefillParams params{c.dQ.rawPointer(), c.dPool.rawPointer(), c.dPages.template dataPointer<int32_t>(),
            indices.dataPointer<int32_t>(), c.dCounts.template dataPointer<int32_t>(), out.rawPointer(), c.batch, c.seq,
            kQHeads, kKVHeads, kHeadDim, kPoolDim, 2 * c.numPages, c.maxPages, qsa_ref::kWidth, 1.F / 16, mStream};
        ELLM_CHECK(CuteDslQsaSparsePrefillRunner::run(dataType<T>(), params), "QSA attention launch failed");
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return copyDeviceToHost<T>(out);
    }

    void verifyAttention(VerifyCase<T> const& c, Path const& expected)
    {
        auto const output = attention(c, c.indices);
        auto const expectedDevice = upload(expected);
        auto const denseOutput = attention(c, expectedDevice);
        expectBits(output.data(), denseOutput.data(), output.size(), "sparse attention path mismatch");
        std::vector<float> queries(c.queries.begin(), c.queries.end());
        std::vector<float> keys(c.keys.begin(), c.keys.end()), values(c.values.begin(), c.values.end());
        double const tol = std::is_same_v<T, half> ? .002 : .02;
        for (int32_t b = 0; b < c.batch; ++b)
        {
            for (int32_t node = 0; node < c.seq; ++node)
            {
                size_t const row = b * c.seq + node;
                if (node >= c.counts[b])
                {
                    std::vector<T> const zero(kQHeads * kHeadDim, static_cast<T>(0.F));
                    expectBits(
                        output.data() + row * kQHeads * kHeadDim, zero.data(), zero.size(), "padded attention row");
                    continue;
                }
                Path ids;
                for (int32_t j = 0; j < qsa_ref::kWidth && expected[row * qsa_ref::kWidth + j] >= 0; ++j)
                {
                    ids.push_back(expected[row * qsa_ref::kWidth + j]);
                }
                for (int32_t h = 0; h < kQHeads; ++h)
                {
                    size_t const qOffset = (row * kQHeads + h) * kHeadDim;
                    std::vector<double> scores(ids.size());
                    for (size_t j = 0; j < ids.size(); ++j)
                    {
                        size_t const kvOffset = ((b * c.capacity + ids[j]) * kKVHeads + h / 12) * kHeadDim;
                        for (int32_t d = 0; d < kHeadDim; ++d)
                        {
                            scores[j] += static_cast<double>(queries[qOffset + d]) * keys[kvOffset + d] / 16;
                        }
                    }
                    double const max = *std::max_element(scores.begin(), scores.end());
                    double sum = 0;
                    for (double& value : scores)
                    {
                        value = std::exp(value - max);
                        sum += value;
                    }
                    std::vector<double> golden(kHeadDim, 0.);
                    for (size_t j = 0; j < ids.size(); ++j)
                    {
                        size_t const kvOffset = ((b * c.capacity + ids[j]) * kKVHeads + h / 12) * kHeadDim;
                        for (int32_t d = 0; d < kHeadDim; ++d)
                        {
                            golden[d] += scores[j] / sum * values[kvOffset + d];
                        }
                    }
                    for (int32_t d = 0; d < kHeadDim; ++d)
                    {
                        EXPECT_NEAR(static_cast<float>(output[qOffset + d]), golden[d], tol + tol * std::abs(golden[d]))
                            << "attention differs from FP64, row=" << row << " head=" << h << " dim=" << d;
                    }
                }
            }
        }
    }

    void verifyReference(VerifyCase<T>& c, bool checkAttention = true)
    {
        run(c);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        expectVerifiedPool(c);
        auto align = [](size_t bytes) { return (bytes + 127) / 128 * 128; };
        size_t const rows = c.batch * c.seq;
        auto const q = readSlice<T>(c.workspace.rawPointer(), 0, rows * 512);
        auto const k = readSlice<T>(c.workspace.rawPointer(), align(rows * 512 * sizeof(T)), rows * 128);
        size_t const scoreOffset = align(rows * 512 * sizeof(T)) + align(rows * 128 * sizeof(T))
            + align(rows * c.seq * sizeof(int32_t)) + align(rows * sizeof(int32_t));
        auto const scores = readSlice<float>(c.workspace.rawPointer(), scoreOffset, rows * c.maxBlocks);
        Path expected(rows * qsa_ref::kWidth, -1);
        for (int32_t b = 0; b < c.batch; ++b)
        {
            for (int32_t node = 0; node < c.seq; ++node)
            {
                SCOPED_TRACE(::testing::Message() << "past=" << c.pasts[b] << " batch=" << b << " node=" << node);
                size_t const row = b * c.seq + node;
                if (node >= c.counts[b])
                {
                    continue;
                }
                auto x = c.prefixX[b];
                auto rope = c.prefixRope[b];
                Path const& path = c.paths[b][node];
                for (int32_t ancestor : path)
                {
                    size_t const r = b * c.seq + ancestor;
                    x.insert(x.end(), c.x.begin() + r * 640, c.x.begin() + (r + 1) * 640);
                    rope.insert(rope.end(), c.rope.begin() + r * 64, c.rope.begin() + (r + 1) * 64);
                }
                auto const ref = reference(x, rope);
                int32_t const length = c.pasts[b] + path.size(), visible = length / 4;
                expectBits(q.data() + row * 512, ref.q.data(), 512, "query prep differs");
                if (length % 4 == 0)
                {
                    expectBits(k.data() + row * 128, ref.k.data() + (visible - 1) * 128, 128, "branch kbar differs");
                }
                expectBits(scores.data() + row * c.maxBlocks, ref.scores.data(), visible, "block scores differ");
                Path selected(visible);
                std::iota(selected.begin(), selected.end(), 0);
                std::stable_sort(selected.begin(), selected.end(),
                    [&](int32_t a, int32_t b) { return ref.scores[a] > ref.scores[b]; });
                selected.resize(std::min(visible, 512));
                std::sort(selected.begin(), selected.end());
                Path logical(qsa_ref::kWidth, -1);
                int32_t j = 0;
                for (int32_t group : selected)
                {
                    for (int32_t d = 0; d < 4; ++d)
                    {
                        logical[j++] = 4 * group + d;
                    }
                }
                for (int32_t token = 4 * visible; token < length; ++token)
                {
                    logical[j++] = token;
                }
                expectBits(ref.indices.data(), logical.data(), logical.size(), "top-k differs from stable CPU sort");
                for (int32_t i = 0; i < qsa_ref::kWidth; ++i)
                {
                    int32_t const id = ref.indices[i];
                    expected[row * qsa_ref::kWidth + i] = id < c.pasts[b] ? id : c.pasts[b] + path.at(id - c.pasts[b]);
                }
            }
        }
        auto const actual = copyDeviceToHost<int32_t>(c.indices);
        expectBits(actual.data(), expected.data(), expected.size(), "tree indices differ from independent paths");
        if (checkAttention && !::testing::Test::HasFailure())
        {
            verifyAttention(c, expected);
        }
    }

    void seal(VerifyCase<T>& c, rt::Tensor const& lengths)
    {
        kernel::QsaIndexerPoolState pool{c.dPool.rawPointer(), c.dPages.template dataPointer<int32_t>(), c.maxPages,
            c.numPages, kKVHeads, kPoolDim, kHeadDim};
        kernel::launchQsaIndexerSeal(
            lengths.dataPointer<int32_t>(), mWK.dataPointer<T>(), kEps, pool, c.batch, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }

    //! The verified pool once each accepted node's whole rows (both heads, tails included) sit at slot past + i, as the
    //! runtime's KV commit leaves them.
    std::vector<T> acceptedPool(VerifyCase<T> const& c, std::vector<T> const& verified, Tree const& accepted) const
    {
        auto moved = verified;
        for (int32_t b = 0; b < c.batch; ++b)
        {
            for (size_t i = 0; i < accepted[b].size(); ++i)
            {
                for (int32_t plane = 0; plane < 2; ++plane)
                {
                    std::copy_n(verified.data() + c.poolOffset(b, plane, c.pasts[b] + accepted[b][i]),
                        kKVHeads * kPoolDim, moved.data() + c.poolOffset(b, plane, c.pasts[b] + i));
                }
            }
        }
        return moved;
    }

    //! `before` with the tails of [0, past + n) an ordinary prefill of the prefix plus the accepted path leaves.
    std::vector<T> prefilledPool(VerifyCase<T> const& c, std::vector<T> const& before, Tree const& accepted)
    {
        auto expected = before;
        for (int32_t b = 0; b < c.batch; ++b)
        {
            auto x = c.prefixX[b];
            auto rope = c.prefixRope[b];
            for (int32_t node : accepted[b])
            {
                size_t const row = b * c.seq + node;
                x.insert(x.end(), c.x.begin() + row * 640, c.x.begin() + (row + 1) * 640);
                rope.insert(rope.end(), c.rope.begin() + row * 64, c.rope.begin() + (row + 1) * 64);
            }
            if (x.empty())
            {
                continue;
            }
            int32_t const end = x.size() / 640;
            writeTails(c, expected, b, x, rope, reference(x, rope).k, end, end / 4);
        }
        return expected;
    }

    void expectPool(VerifyCase<T> const& c, std::vector<T> const& expectedPool, char const* message)
    {
        auto const actual = copyDeviceToHost<T>(c.dPool);
        expectBits(actual.data(), expectedPool.data(), expectedPool.size(), message);
    }
};

using VerifyTypes = ::testing::Types<half, __nv_bfloat16>;
TYPED_TEST_SUITE(QsaIndexerVerifyTest, VerifyTypes);

TYPED_TEST(QsaIndexerVerifyTest, PathEquivalence)
{
    // Given boundary, ragged and sparse trees over a prefix whose newest blocks are unsealed, when verified, then
    // independent paths agree, the prefix is sealed and each node's tails land in its own slot.
    Path const fork{-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, -1, -1};
    for (int32_t past : {0, 1, 2, 3, 126, 127, 128, 129, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2307})
    {
        auto c = this->makeCase({past}, {10}, fork);
        this->verifyReference(c);
        if (::testing::Test::HasFailure())
        {
            return;
        }
    }
    Path ragged;
    for (int32_t b = 0; b < 4; ++b)
    {
        ragged.insert(ragged.end(), fork.begin(), fork.end());
    }
    auto c = this->makeCase({2051, 3, 127, 0}, {10, 8, 0, 1}, ragged);
    this->verifyReference(c);
    c = this->makeCase({2307}, {10}, fork, true);
    this->verifyReference(c);
    Path chain(128);
    std::iota(chain.begin(), chain.end(), -1);
    c = this->makeCase({2051}, {128}, chain);
    this->verifyReference(c, false);
}

TYPED_TEST(QsaIndexerVerifyTest, ReplayAndSiblingIsolation)
{
    // Given sibling changes and graph replays, when verified, then other branches stay identical and updated trees take
    // effect.
    Path const fork{-1, 0, 1, 2, 3, 0, 5, 6, 7, 8};
    auto c = this->makeCase({2051}, {10}, fork);
    this->verifyReference(c);
    auto const expected = copyDeviceToHost<int32_t>(c.indices);
    auto const expectedOut = this->attention(c, c.indices);
    for (float poison : {99.F, std::numeric_limits<float>::quiet_NaN()})
    {
        std::fill(c.x.begin() + 5 * 640, c.x.end(), static_cast<TypeParam>(poison));
        for (int32_t node = 5; node < 10; ++node)
        {
            for (int32_t plane = 0; plane < 2; ++plane)
            {
                for (int32_t head = 0; head < kKVHeads; ++head)
                {
                    std::fill_n(c.pool.data() + c.poolOffset(0, plane, 2051 + node, head), kHeadDim,
                        static_cast<TypeParam>(poison));
                }
            }
        }
        copyHostToDevice(c.dX, c.x);
        copyHostToDevice(c.dPool, c.pool);
        this->run(c);
        CUDA_CHECK(cudaStreamSynchronize(this->mStream));
        auto const actual = copyDeviceToHost<int32_t>(c.indices);
        auto const out = this->attention(c, c.indices);
        expectBits(actual.data(), expected.data(), 5 * qsa_ref::kWidth, "sibling changed indices");
        expectBits(out.data(), expectedOut.data(), 5 * kQHeads * kHeadDim, "sibling changed attention");
        this->expectVerifiedPool(c);
    }
    c = this->makeCase({2049}, {10}, fork);
    this->run(c);
    CUDA_CHECK(cudaStreamSynchronize(this->mStream));
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    CUDA_CHECK(cudaStreamBeginCapture(this->mStream, cudaStreamCaptureModeThreadLocal));
    this->run(c);
    CUDA_CHECK(cudaStreamEndCapture(this->mStream, &graph));
    Defer cleanupGraph([&]() { cudaGraphDestroy(graph); });
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    Defer cleanupExecutable([&]() { cudaGraphExecDestroy(executable); });
    for (int32_t replay = 0; replay < 3; ++replay)
    {
        Path parents = fork;
        if (replay % 2 != 0)
        {
            std::iota(parents.begin(), parents.end(), -1);
        }
        this->updateTree(c, parents);
        copyHostToDevice(c.dParents, c.parents);
        copyHostToDevice(c.dDepths, c.depths);
        copyHostToDevice(c.dRope, c.rope);
        // Another tree writes rope rows at other nodes, so each replay starts from the unverified pool.
        copyHostToDevice(c.dPool, c.pool);
        CUDA_CHECK(cudaGraphLaunch(executable, this->mStream));
        CUDA_CHECK(cudaStreamSynchronize(this->mStream));
        auto const captured = copyDeviceToHost<int32_t>(c.indices);
        this->verifyReference(c, false);
        auto const actual = copyDeviceToHost<int32_t>(c.indices);
        expectBits(captured.data(), actual.data(), actual.size(), "graph replay ignored updated tree");
    }
}

TYPED_TEST(QsaIndexerVerifyTest, InvalidInputs)
{
    // Given invalid metadata or workspace, when verified, then rejected nodes expose no indices and write nothing.
    for (int32_t error = 0; error < 6; ++error)
    {
        SCOPED_TRACE(error);
        auto c = this->makeCase({3}, {5}, {-1, 0, 1, 0, 3});
        switch (error)
        {
        case 0:
            c.parents[3] = 4;
            copyHostToDevice(c.dParents, c.parents);
            break;
        case 1:
            c.depths[3] = 100;
            copyHostToDevice(c.dDepths, c.depths);
            break;
        case 2: copyHostToDevice(c.dCounts, Path{6}); break;
        case 3: copyHostToDevice(c.dPasts, Path{-1}); break;
        case 4:
        {
            auto pages = c.pages;
            pages[c.maxPages] = -1;
            copyHostToDevice(c.dPages, pages);
            break;
        }
        case 5: copyHostToDevice(c.dPasts, Path{127}); break;
        }
        this->run(c);
        CUDA_CHECK(cudaStreamSynchronize(this->mStream));
        // A broken parent or depth rejects node 3 and its child; bad lengths, pages or capacity reject every node.
        Path const rejected = error < 2 ? Path{3, 4} : Path{0, 1, 2, 3, 4};
        auto const idx = copyDeviceToHost<int32_t>(c.indices);
        Path const empty(qsa_ref::kWidth, -1);
        for (int32_t node : rejected)
        {
            expectBits(idx.data() + node * qsa_ref::kWidth, empty.data(), empty.size(), "failed indices");
        }
        this->expectVerifiedPool(c, rejected);
    }
    auto c = this->makeCase({3}, {5}, {-1, 0, 1, 0, 3});
    EXPECT_THROW(this->run(c, 1, this->mStream), std::runtime_error);
    EXPECT_THROW(this->run(c, c.workspace.getShape().volume(), nullptr), std::runtime_error);
    EXPECT_THROW(kernel::getQsaIndexerVerifyWorkspaceSize(1, 129, 100), std::runtime_error);
}

TYPED_TEST(QsaIndexerVerifyTest, AcceptedPathSealsLikePrefill)
{
    // Given a verified tree whose accepted path's rows are moved to positions past + i, when the next call seals, then
    // the tails of [0, past + n) match an ordinary prefill of the same tokens and nothing else changes.
    Path const fork{-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, -1, -1};
    for (int32_t past : {0, 1, 2, 3, 125, 126, 127, 128, 129, 2047, 2048, 2051})
    {
        auto c = this->makeCase({past}, {10}, fork);
        this->run(c);
        CUDA_CHECK(cudaStreamSynchronize(this->mStream));
        auto const verified = copyDeviceToHost<TypeParam>(c.dPool);
        for (int32_t leaf : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        {
            SCOPED_TRACE(::testing::Message() << "past=" << past << " leaf=" << leaf);
            Path const& path = c.paths[0][leaf];
            auto const before = this->acceptedPool(c, verified, {path});
            copyHostToDevice(c.dPool, before);
            this->seal(c, upload(Path{past + static_cast<int32_t>(path.size())}));
            this->expectPool(c, this->prefilledPool(c, before, {path}), "sealed pool differs from prefill");
            if (::testing::Test::HasFailure())
            {
                return;
            }
        }
    }
    // The longest commit: 128 accepted tokens complete kQSA_SEAL_BLOCKS blocks.
    Path chain(128);
    std::iota(chain.begin(), chain.end(), -1);
    auto c = this->makeCase({127}, {128}, chain);
    this->run(c);
    CUDA_CHECK(cudaStreamSynchronize(this->mStream));
    auto before = this->acceptedPool(c, copyDeviceToHost<TypeParam>(c.dPool), {c.paths[0].back()});
    copyHostToDevice(c.dPool, before);
    this->seal(c, upload(Path{255}));
    this->expectPool(c, this->prefilledPool(c, before, {c.paths[0].back()}), "longest commit differs from prefill");

    // A batch with ragged accepted lengths, zero included; sealing twice changes nothing.
    Path const tree{-1, 0, 1, 2, 3, 0, 5, 6, 7, 8};
    Path parents;
    for (int32_t b = 0; b < 4; ++b)
    {
        parents.insert(parents.end(), tree.begin(), tree.end());
    }
    c = this->makeCase({127, 2, 0, 129}, {10, 8, 0, 10}, parents);
    this->run(c);
    CUDA_CHECK(cudaStreamSynchronize(this->mStream));
    Tree const paths{c.paths[0][9], c.paths[1][3], {}, c.paths[3][0]};
    before = this->acceptedPool(c, copyDeviceToHost<TypeParam>(c.dPool), paths);
    auto const expected = this->prefilledPool(c, before, paths);
    copyHostToDevice(c.dPool, before);
    auto const lengths = upload(Path{132, 6, 0, 130});
    this->seal(c, lengths);
    this->expectPool(c, expected, "batched seal differs from prefill");
    this->seal(c, lengths);
    this->expectPool(c, expected, "second seal changed the pool");
}

TYPED_TEST(QsaIndexerVerifyTest, SealSkipsInvalidLengthsAndPages)
{
    // Given one batch with a negative length or an unusable page, when sealed, then that batch's pool is unchanged and
    // the other batch seals.
    Path const fork{-1, 0, 1, 0, 3};
    Path parents = fork;
    parents.insert(parents.end(), fork.begin(), fork.end());
    for (int32_t error = 0; error < 4; ++error)
    {
        SCOPED_TRACE(error);
        auto c = this->makeCase({127, 2}, {5, 5}, parents);
        this->run(c);
        CUDA_CHECK(cudaStreamSynchronize(this->mStream));
        Tree const paths{{0, 3, 4}, {0, 1, 2}};
        auto const before = this->acceptedPool(c, copyDeviceToHost<TypeParam>(c.dPool), paths);
        auto const expected = this->prefilledPool(c, before, {{}, paths[1]});
        copyHostToDevice(c.dPool, before);
        Path lengths{130, 5};
        // Every block batch 0 seals lives in its first page.
        switch (error)
        {
        case 0: lengths[0] = -1; break;
        case 1: c.pages[0] = -1; break;
        case 2: c.pages[c.maxPages] = -1; break;
        case 3: c.pages[c.maxPages] = 2 * c.numPages; break;
        }
        copyHostToDevice(c.dPages, c.pages);
        this->seal(c, upload(lengths));
        this->expectPool(c, expected, "invalid batch was sealed");
    }
}

} // namespace
