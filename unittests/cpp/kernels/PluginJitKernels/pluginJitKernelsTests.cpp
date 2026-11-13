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

//! Shared plugin JIT plumbing: key hashing, blob layout and rejection, digest known answers and the
//! context-keyed module registry. The registry cases compile the arch-generic QSA indexer program
//! for whatever device is present, so they run on every GPU lane, not only on Thor.
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/PluginJitKernels/pluginJitBlob.h"
#include "kernels/PluginJitKernels/pluginJitCubin.h"
#include "kernels/PluginJitKernels/pluginJitDigest.h"
#include "kernels/PluginJitKernels/pluginJitKey.h"
#include "kernels/PluginJitKernels/pluginJitModuleRegistry.h"
#include "kernels/qsaIndexer/qsaIndexerJitCompiler.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace
{

//! Minimal key following the pluginJitKey.h contract.
struct TestKey
{
    int32_t shape{};
    uint32_t abi{};
    bool flag{};

    static constexpr size_t kNumWords{3};

    std::array<uint32_t, kNumWords> keyWords() const noexcept
    {
        return {static_cast<uint32_t>(shape), abi, flag ? 1U : 0U};
    }

    static TestKey fromKeyWords(std::array<uint32_t, kNumWords> const& words) noexcept
    {
        return {static_cast<int32_t>(words[0]), words[1], words[2] != 0U};
    }

    bool operator==(TestKey const& other) const noexcept
    {
        return keyWords() == other.keyWords();
    }
};

using TestKernel = PluginJitKernel<TestKey>;

constexpr PluginJitBlobFormat kFORMAT{"Plugin JIT test", 0x54534554U /* "TEST" */};
constexpr size_t kHEADER_BYTES{sizeof(uint32_t)};
constexpr size_t kRECORD_PREFIX_BYTES{TestKey::kNumWords * sizeof(uint32_t) + sizeof(uint32_t) + 2 * sizeof(uint64_t)};

//! The ELF magic (the blob refuses anything else) followed by seeded bytes.
std::vector<uint8_t> fakeCubin(uint8_t const seed, size_t const size)
{
    std::vector<uint8_t> cubin{0x7FU, 0x45U, 0x4CU, 0x46U};
    while (cubin.size() < size)
    {
        cubin.push_back(static_cast<uint8_t>(seed + cubin.size()));
    }
    return cubin;
}

TestKernel makeKernel(TestKey const& key, std::vector<uint8_t> cubin)
{
    TestKernel kernel;
    kernel.key = key;
    kernel.cubin = std::move(cubin);
    kernel.digest = computePluginJitDigest(key, kernel.cubin.data(), kernel.cubin.size());
    return kernel;
}

void appendLittleEndian(std::vector<uint8_t>& out, uint64_t const value, int32_t const bytes)
{
    for (int32_t index = 0; index < bytes; ++index)
    {
        out.push_back(static_cast<uint8_t>(value >> (8 * index)));
    }
}

//! Overwrite the digest of the first record so a hand-patched blob fails only the check under test.
void resignFirstRecord(std::vector<uint8_t>& blob, size_t const recordOffset, PluginJitDigest const& digest)
{
    std::vector<uint8_t> bytes;
    appendLittleEndian(bytes, digest.lo, 8);
    appendLittleEndian(bytes, digest.hi, 8);
    std::copy(bytes.begin(), bytes.end(),
        blob.begin()
            + static_cast<std::ptrdiff_t>(recordOffset + TestKey::kNumWords * sizeof(uint32_t) + sizeof(uint32_t)));
}

TestKernel deserialize(std::vector<uint8_t> const& blob)
{
    return deserializePluginJitKernel<TestKey>(kFORMAT, blob.data(), blob.size());
}

TEST(PluginJitKeyTest, WordsRoundTripAndDriveTheHasher)
{
    TestKey const key{-5, 0xDEADBEEFU, true};
    EXPECT_TRUE(TestKey::fromKeyWords(key.keyWords()) == key);

    PluginJitKeyHasher<TestKey> const hasher;
    EXPECT_EQ(hasher(key), hashPluginJitWords(key.keyWords()));
    EXPECT_EQ(hasher(key), hasher(TestKey::fromKeyWords(key.keyWords())));
    TestKey other = key;
    other.flag = false;
    EXPECT_NE(hasher(key), hasher(other));
    other = key;
    other.shape = -4;
    EXPECT_NE(hasher(key), hasher(other));
}

TEST(PluginJitDigestTest, KnownAnswers)
{
    // The two lanes are part of the serialized blob contract; these values must never change.
    EXPECT_EQ(PluginJitDigestBuilder{}.get(), (PluginJitDigest{0xCBF29CE484222325ULL, 0x6C62272E07BB0142ULL}));

    PluginJitDigestBuilder builder;
    builder.appendU32(0x01020304U);
    std::array<uint8_t, 6> const bytes{0x7FU, 0x45U, 0x4CU, 0x46U, 0xAAU, 0x55U};
    builder.appendBytes(bytes.data(), bytes.size());
    EXPECT_EQ(builder.get(), (PluginJitDigest{0x8FB84E076004EF3AULL, 0x8E5BCAD30BB9BEE5ULL}));

    // computePluginJitDigest feeds the key words, then the cubin bytes.
    TestKey const key{0x01020304, 0xDEADBEEFU, false};
    std::array<uint8_t, 6> const cubin{0x7FU, 0x45U, 0x4CU, 0x46U, 0x10U, 0x20U};
    PluginJitDigestBuilder expected;
    for (uint32_t const word : key.keyWords())
    {
        expected.appendU32(word);
    }
    expected.appendBytes(cubin.data(), cubin.size());
    EXPECT_EQ(computePluginJitDigest(key, cubin.data(), cubin.size()), expected.get());
}

TEST(PluginJitDigestTest, BindsKeyAndCubin)
{
    std::vector<uint8_t> const cubin = fakeCubin(0x10, 32);
    TestKey const key{7, 1U, false};
    PluginJitDigest const digest = computePluginJitDigest(key, cubin.data(), cubin.size());
    EXPECT_EQ(computePluginJitDigest(key, cubin.data(), cubin.size()), digest);

    TestKey otherKey = key;
    otherKey.shape = 8;
    EXPECT_NE(computePluginJitDigest(otherKey, cubin.data(), cubin.size()), digest);
    std::vector<uint8_t> otherCubin = cubin;
    otherCubin.back() ^= 1U;
    EXPECT_NE(computePluginJitDigest(key, otherCubin.data(), otherCubin.size()), digest);
}

TEST(PluginJitCubinTest, RecognizesTheElfMagic)
{
    std::vector<uint8_t> const cubin = fakeCubin(0, 8);
    EXPECT_TRUE(isNativeCubin(cubin.data(), cubin.size()));
    EXPECT_TRUE(isNativeCubin(cubin.data(), 4));
    EXPECT_FALSE(isNativeCubin(cubin.data(), 3));
    EXPECT_FALSE(isNativeCubin(nullptr, 0));
    std::string const ptx{"//\n.version 8.0\n"};
    EXPECT_FALSE(isNativeCubin(ptx.data(), ptx.size()));
}

TEST(PluginJitBlobTest, LayoutIsLittleEndianAndPaddingFree)
{
    TestKernel const kernel = makeKernel(TestKey{0x00010203, 0xA0B0C0D0U, true}, fakeCubin(0x40, 9));

    std::vector<uint8_t> expected;
    appendLittleEndian(expected, kFORMAT.magic, 4);
    for (uint32_t const word : kernel.key.keyWords())
    {
        appendLittleEndian(expected, word, 4);
    }
    appendLittleEndian(expected, kernel.cubin.size(), 4);
    appendLittleEndian(expected, kernel.digest.lo, 8);
    appendLittleEndian(expected, kernel.digest.hi, 8);
    expected.insert(expected.end(), kernel.cubin.begin(), kernel.cubin.end());

    std::vector<uint8_t> const blob = serializePluginJitKernel(kFORMAT, kernel);
    EXPECT_EQ(blob, expected);
    EXPECT_EQ(std::string(blob.begin(), blob.begin() + 4), "TEST");
    EXPECT_EQ(blob.size(), kHEADER_BYTES + kRECORD_PREFIX_BYTES + kernel.cubin.size());

    // The list layout inserts the count between the header and the records.
    std::vector<uint8_t> expectedList(expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(kHEADER_BYTES));
    appendLittleEndian(expectedList, 1U, 4);
    expectedList.insert(
        expectedList.end(), expected.begin() + static_cast<std::ptrdiff_t>(kHEADER_BYTES), expected.end());
    EXPECT_EQ(serializePluginJitKernels(kFORMAT, std::vector<TestKernel>{kernel}), expectedList);
}

TEST(PluginJitBlobTest, RoundTripsOneKernelAndLists)
{
    TestKernel const first = makeKernel(TestKey{1, 2U, false}, fakeCubin(0x11, 257));
    TestKernel const second = makeKernel(TestKey{1, 2U, true}, fakeCubin(0x22, 64));

    TestKernel const restored = deserialize(serializePluginJitKernel(kFORMAT, first));
    EXPECT_TRUE(restored.key == first.key);
    EXPECT_EQ(restored.cubin, first.cubin);
    EXPECT_EQ(restored.digest, first.digest);

    std::vector<uint8_t> const listBlob = serializePluginJitKernels(kFORMAT, std::vector<TestKernel>{first, second});
    std::vector<TestKernel> const restoredList
        = deserializePluginJitKernels<TestKey>(kFORMAT, listBlob.data(), listBlob.size());
    ASSERT_EQ(restoredList.size(), 2U);
    EXPECT_TRUE(restoredList[0].key == first.key);
    EXPECT_EQ(restoredList[0].cubin, first.cubin);
    EXPECT_TRUE(restoredList[1].key == second.key);
    EXPECT_EQ(restoredList[1].cubin, second.cubin);
    EXPECT_EQ(restoredList[1].digest, second.digest);

    std::vector<uint8_t> const emptyBlob = serializePluginJitKernels(kFORMAT, std::vector<TestKernel>{});
    EXPECT_TRUE(deserializePluginJitKernels<TestKey>(kFORMAT, emptyBlob.data(), emptyBlob.size()).empty());
    // A single-kernel blob is not a list and vice versa.
    EXPECT_THROW(deserialize(listBlob), std::runtime_error);
}

TEST(PluginJitBlobTest, RejectsMalformedBlobs)
{
    TestKernel const kernel = makeKernel(TestKey{1, 2U, false}, fakeCubin(0x20, 40));
    std::vector<uint8_t> const blob = serializePluginJitKernel(kFORMAT, kernel);

    EXPECT_THROW(deserializePluginJitKernel<TestKey>(kFORMAT, nullptr, 0), std::invalid_argument);
    for (size_t length = 1; length < blob.size(); ++length)
    {
        std::vector<uint8_t> const prefix(blob.begin(), blob.begin() + static_cast<std::ptrdiff_t>(length));
        EXPECT_THROW(deserialize(prefix), std::runtime_error) << "prefix of " << length << " bytes";
    }

    std::vector<uint8_t> tampered = blob;
    tampered.push_back(0U); // trailing byte
    EXPECT_THROW(deserialize(tampered), std::runtime_error);
    tampered = blob;
    tampered[0] ^= 0xFFU; // magic
    EXPECT_THROW(deserialize(tampered), std::runtime_error);
    tampered = blob;
    tampered.back() ^= 1U; // cubin byte: digest mismatch
    EXPECT_THROW(deserialize(tampered), std::runtime_error);
    tampered = blob;
    tampered[kHEADER_BYTES] ^= 1U; // key word: digest mismatch
    EXPECT_THROW(deserialize(tampered), std::runtime_error);

    // A re-signed record whose cubin lost the ELF magic fails the ELF check alone.
    TestKernel notElf = kernel;
    notElf.cubin[0] = 0x00U;
    notElf.digest = computePluginJitDigest(notElf.key, notElf.cubin.data(), notElf.cubin.size());
    tampered = blob;
    tampered[kHEADER_BYTES + kRECORD_PREFIX_BYTES] = 0x00U;
    resignFirstRecord(tampered, kHEADER_BYTES, notElf.digest);
    EXPECT_THROW(deserialize(tampered), std::runtime_error);

    // An empty cubin is refused before its (absent) payload is read.
    tampered.assign(blob.begin(),
        blob.begin() + static_cast<std::ptrdiff_t>(kHEADER_BYTES + TestKey::kNumWords * sizeof(uint32_t)));
    appendLittleEndian(tampered, 0U, 4);
    appendLittleEndian(tampered, kernel.digest.lo, 8);
    appendLittleEndian(tampered, kernel.digest.hi, 8);
    EXPECT_THROW(deserialize(tampered), std::runtime_error);
}

TEST(PluginJitBlobTest, SerializeRefusesInconsistentKernels)
{
    TestKernel const kernel = makeKernel(TestKey{1, 2U, false}, fakeCubin(0x20, 16));

    TestKernel stale = kernel;
    stale.cubin.back() ^= 1U; // digest no longer matches the payload
    EXPECT_THROW(serializePluginJitKernel(kFORMAT, stale), std::invalid_argument);
    EXPECT_THROW(serializePluginJitKernels(kFORMAT, std::vector<TestKernel>{kernel, stale}), std::invalid_argument);

    TestKernel notElf = kernel;
    notElf.cubin[1] = 0x00U;
    notElf.digest = computePluginJitDigest(notElf.key, notElf.cubin.data(), notElf.cubin.size());
    EXPECT_THROW(serializePluginJitKernel(kFORMAT, notElf), std::invalid_argument);

    TestKernel empty = kernel;
    empty.cubin.clear();
    empty.digest = computePluginJitDigest(empty.key, empty.cubin.data(), empty.cubin.size());
    EXPECT_THROW(serializePluginJitKernel(kFORMAT, empty), std::invalid_argument);
}

struct TestLoadedModule : PluginJitLoadedModule<QsaIndexerJitKey>
{
    CUfunction qPrep{};
};

using TestRegistry = PluginJitModuleRegistry<QsaIndexerJitKey, TestLoadedModule>;

bool hasCudaDevice()
{
    int32_t deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

QsaIndexerJitKernel compileForCurrentDevice(QsaIndexerJitDataType const dataType)
{
    QsaIndexerJitKey key;
    key.sm = getSMVersion();
    key.dataType = dataType;
    return compileQsaIndexerJitKernel(key);
}

TEST(PluginJitModuleRegistryTest, SharesModulesPerContextKeyAndDigest)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    QsaIndexerJitKernel const kernel = compileForCurrentDevice(QsaIndexerJitDataType::kHALF);

    TestRegistry registry{"Plugin JIT test"};
    int32_t binds{0};
    auto bind = [&binds](TestLoadedModule& loaded) {
        ++binds;
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded.qPrep, loaded.module, "qsa_indexer_q_prep"));
    };

    std::shared_ptr<TestLoadedModule> first = registry.load(kernel, bind);
    ASSERT_NE(first, nullptr);
    EXPECT_NE(first->module, nullptr);
    EXPECT_NE(first->qPrep, nullptr);
    EXPECT_TRUE(first->key == kernel.key);
    EXPECT_EQ(first->digest, kernel.digest);
    CUcontext context{};
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
    EXPECT_EQ(first->context, context);
    EXPECT_EQ(binds, 1);

    // The same kernel in the same context shares the entry.
    std::shared_ptr<TestLoadedModule> second = registry.load(kernel, bind);
    EXPECT_EQ(second, first);
    EXPECT_EQ(binds, 1);

    // A different key (and digest) is a different module.
    std::shared_ptr<TestLoadedModule> const other
        = registry.load(compileForCurrentDevice(QsaIndexerJitDataType::kBF16), bind);
    EXPECT_NE(other, first);
    EXPECT_EQ(binds, 2);

    // Entries are weak: once every holder is gone the next load binds a fresh module.
    std::weak_ptr<TestLoadedModule> const weak = first;
    first.reset();
    second.reset();
    EXPECT_TRUE(weak.expired());
    std::shared_ptr<TestLoadedModule> const reloaded = registry.load(kernel, bind);
    EXPECT_NE(reloaded, nullptr);
    EXPECT_EQ(binds, 3);
}

TEST(PluginJitModuleRegistryTest, RefusesTamperedKernelsAndUnwindsFailedBinds)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    QsaIndexerJitKernel const kernel = compileForCurrentDevice(QsaIndexerJitDataType::kHALF);

    TestRegistry registry{"Plugin JIT test"};
    int32_t binds{0};
    auto bind = [&binds](TestLoadedModule&) { ++binds; };

    QsaIndexerJitKernel tampered = kernel;
    tampered.cubin.back() ^= 0x5AU;
    EXPECT_THROW(registry.load(tampered, bind), std::invalid_argument);
    QsaIndexerJitKernel notElf = kernel;
    notElf.cubin[0] ^= 1U;
    notElf.digest = computePluginJitDigest(notElf.key, notElf.cubin.data(), notElf.cubin.size());
    EXPECT_THROW(registry.load(notElf, bind), std::invalid_argument);
    QsaIndexerJitKernel empty = kernel;
    empty.cubin.clear();
    empty.digest = computePluginJitDigest(empty.key, empty.cubin.data(), empty.cubin.size());
    EXPECT_THROW(registry.load(empty, bind), std::invalid_argument);
    EXPECT_EQ(binds, 0);

    // A throwing bindFunctions registers nothing; the next load starts over.
    EXPECT_THROW(
        registry.load(kernel, [](TestLoadedModule&) { throw std::logic_error("bind failed"); }), std::logic_error);
    std::shared_ptr<TestLoadedModule> const loaded = registry.load(kernel, bind);
    EXPECT_NE(loaded, nullptr);
    EXPECT_EQ(binds, 1);
}

} // namespace
} // namespace trt_edgellm
