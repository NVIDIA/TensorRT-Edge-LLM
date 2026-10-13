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

// CheckpointReader is the boundary between a HuggingFace checkpoint on disk and
// every weight the runtime assembles from it. Its failure modes are quiet: an
// offset that is wrong by a header length yields plausible bytes at the wrong
// place, and a device alias that points outside its registration yields whatever
// the neighbouring page holds. Both produce a model that loads and runs.
//
// These tests build real safetensors shards, then assert on what the reader
// reports about them and on the bytes it hands back through each of its two
// address spaces.

#include "runtime/weight/checkpointReader.h"

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "scratchDir.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <numeric>
#include <unistd.h>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;
using View = CheckpointReader::View;
using TensorLocation = CheckpointReader::TensorLocation;

namespace
{

//! Shard file names are chosen so lexicographic order differs from the order the
//! tests care about, keeping directory-scan order from standing in for the index.
constexpr char const* kShardA{"model-00001-of-00002.safetensors"};
constexpr char const* kShardB{"model-00002-of-00002.safetensors"};

class CheckpointReaderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        // Per-process so parallel ctest groups cannot collide on one directory.
        mDir = makeScratchDir("checkpointReaderTests");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! Write one safetensors shard holding FP32 tensors filled with a distinct
    //! ascending ramp per tensor, so a view that lands on the wrong tensor or at
    //! the wrong offset reads recognizably wrong values rather than zeros.
    void writeShard(std::string const& fileName, std::vector<std::pair<std::string, int64_t>> const& tensors)
    {
        std::vector<Tensor> toSave;
        float base = 1.0F;
        for (auto const& [name, count] : tensors)
        {
            Tensor tensor({count}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, name);
            auto* data = static_cast<float*>(tensor.rawPointer());
            std::iota(data, data + count, base);
            base += 1000.0F;
            toSave.push_back(std::move(tensor));
        }
        ASSERT_TRUE(safetensors::saveSafetensors(mDir / fileName, toSave, mStream));
    }

    void writeIndex(std::string const& fileName, std::map<std::string, std::string> const& weightMap)
    {
        Json index;
        index["weight_map"] = weightMap;
        std::ofstream out(mDir / fileName);
        out << index.dump();
    }

    //! The ramp writeShard() laid down for the tensor at `tensorIndex` in a shard.
    static std::vector<float> expectedRamp(int32_t tensorIndex, int64_t count)
    {
        std::vector<float> values(static_cast<size_t>(count));
        std::iota(values.begin(), values.end(), 1.0F + 1000.0F * static_cast<float>(tensorIndex));
        return values;
    }

    static size_t pageSize()
    {
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }

    static std::vector<float> hostFloats(View const& view)
    {
        std::vector<float> values(view.bytes / sizeof(float));
        std::memcpy(values.data(), view.data, view.bytes);
        return values;
    }

    //! Read the tensor back through its CUDA alias rather than its host pointer.
    //! This is what the transform kernels see, and it is the only way to catch a
    //! device pointer that is offset relative to its registration.
    std::vector<float> deviceFloats(View const& view)
    {
        std::vector<float> values(view.bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(values.data(), view.deviceData, view.bytes, cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        return values;
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
};

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

// The unindexed shape: a directory of shards and nothing else. Every tensor in
// every shard has to be reachable, and the metadata the reader reports has to
// match what was written, including the payload itself.
TEST_F(CheckpointReaderTest, ScansEveryShardWhenNoIndexIsPresent)
{
    writeShard(kShardA, {{"layer.0.weight", 8}});
    writeShard(kShardB, {{"layer.1.weight", 4}});

    CheckpointReader reader(mDir);

    View view{};
    ASSERT_TRUE(reader.findHost("layer.0.weight", view));
    EXPECT_EQ(view.dtype, nvinfer1::DataType::kFLOAT);
    EXPECT_EQ(view.shape, Coords({8}));
    EXPECT_EQ(view.bytes, 8U * sizeof(float));
    EXPECT_EQ(hostFloats(view), expectedRamp(0, 8));

    ASSERT_TRUE(reader.findHost("layer.1.weight", view));
    EXPECT_EQ(view.shape, Coords({4}));
    EXPECT_EQ(hostFloats(view), expectedRamp(0, 4));
}

// Two tensors in one shard must resolve to different byte ranges. A reader that
// ignored the per-tensor offset would return the first tensor's data for both,
// which the ramp makes visible.
TEST_F(CheckpointReaderTest, ResolvesEachTensorToItsOwnRangeWithinAShard)
{
    writeShard(kShardA, {{"first", 6}, {"second", 6}});

    CheckpointReader reader(mDir);

    View first{};
    View second{};
    ASSERT_TRUE(reader.findHost("first", first));
    ASSERT_TRUE(reader.findHost("second", second));

    EXPECT_EQ(hostFloats(first), expectedRamp(0, 6));
    EXPECT_EQ(hostFloats(second), expectedRamp(1, 6));
    EXPECT_NE(first.data, second.data);
}

// When an index exists it is authoritative: the reader indexes names from
// weight_map, not from scanning shard contents. A tensor physically present in a
// shard but absent from the map must not resolve, otherwise a stale shard left
// in the directory could shadow the intended weights.
//
// Given an index that maps a tensor to a shard other than the one physically holding it
// When the reader resolves that tensor
// Then it follows the index
TEST_F(CheckpointReaderTest, IndexIsAuthoritativeOverShardContents)
{
    writeShard(kShardA, {{"mapped", 4}, {"unmapped", 4}});
    writeIndex("model.safetensors.index.json", {{"mapped", kShardA}});

    CheckpointReader reader(mDir);

    View view{};
    EXPECT_TRUE(reader.findHost("mapped", view));
    EXPECT_FALSE(reader.findHost("unmapped", view));
}

// An export can leave more than one index behind. `model.safetensors.index.json`
// is the canonical name and is selected regardless of where it sorts, so the
// competing index here is named to sort first.
//
// Given a directory carrying both the canonical index and another *.index.json
// When the reader discovers the checkpoint
// Then it reads the canonical one
TEST_F(CheckpointReaderTest, PrefersTheCanonicalIndexOverOtherIndexFiles)
{
    writeShard(kShardA, {{"canonical.weight", 4}});
    writeShard(kShardB, {{"legacy.weight", 4}});
    writeIndex("consolidated.safetensors.index.json", {{"legacy.weight", kShardB}});
    writeIndex("model.safetensors.index.json", {{"canonical.weight", kShardA}});

    CheckpointReader reader(mDir);

    View view{};
    EXPECT_TRUE(reader.findHost("canonical.weight", view));
    EXPECT_FALSE(reader.findHost("legacy.weight", view));
}

// A name nobody exported is a normal negative answer for callers that probe for
// optional weights, so it reports absence instead of throwing.
TEST_F(CheckpointReaderTest, MissingTensorIsReportedAsAbsentRatherThanThrowing)
{
    writeShard(kShardA, {{"present", 4}});

    CheckpointReader reader(mDir);

    View view{};
    EXPECT_FALSE(reader.findHost("absent", view));
}

// The three ways a checkpoint directory can be unusable. Each has to fail at
// construction: the alternative is a runtime that reports a missing weight much
// later, with the directory no longer in the message.
TEST_F(CheckpointReaderTest, RejectsUnusableCheckpointDirectories)
{
    EXPECT_THROW(CheckpointReader{mDir / "nonexistent"}, std::runtime_error);

    // Directory exists but holds no checkpoint at all.
    EXPECT_THROW(CheckpointReader{mDir}, std::runtime_error);

    writeShard(kShardA, {{"weight", 4}});
    std::ofstream(mDir / "model.safetensors.index.json") << Json({{"metadata", Json::object()}}).dump();
    EXPECT_THROW(CheckpointReader{mDir}, std::runtime_error);
}

// ---------------------------------------------------------------------------
// CUDA registration
// ---------------------------------------------------------------------------

// find() is the accessor that promises a usable device pointer. Before
// registration there is none, and returning the view anyway would hand a kernel
// a null pointer that faults far from the cause.
TEST_F(CheckpointReaderTest, FindRefusesToReturnADeviceAliasBeforeRegistration)
{
    writeShard(kShardA, {{"weight", 8}});
    CheckpointReader reader(mDir);

    View view{};
    EXPECT_THROW(static_cast<void>(reader.find("weight", view)), std::runtime_error);
    // The host view is still available without registration; only the device
    // alias requires it.
    EXPECT_TRUE(reader.findHost("weight", view));
    EXPECT_EQ(view.deviceData, nullptr);
}

// The device alias has to address the same bytes as the host pointer. It is
// computed by offsetting the registration base, so an error in that arithmetic
// silently feeds the transform kernels a shifted window.
//
// Given a shard registered for device access
// When a tensor is read through its CUDA alias and through its host pointer
// Then both return the same bytes
TEST_F(CheckpointReaderTest, RegisteredDeviceAliasAddressesTheSameBytesAsTheHostView)
{
    writeShard(kShardA, {{"first", 6}, {"second", 6}});
    CheckpointReader reader(mDir);
    reader.registerTensors({"first", "second"});

    View first{};
    View second{};
    ASSERT_TRUE(reader.find("first", first));
    ASSERT_TRUE(reader.find("second", second));

    ASSERT_NE(first.deviceData, nullptr);
    ASSERT_NE(second.deviceData, nullptr);
    EXPECT_EQ(deviceFloats(first), expectedRamp(0, 6));
    EXPECT_EQ(deviceFloats(second), expectedRamp(1, 6));
}

// Tensors that share a page produce overlapping page-aligned ranges. Merging
// them is not an optimization: cudaHostRegister rejects a range that overlaps an
// existing registration, so an unmerged second range fails the whole load. Small
// adjacent tensors are the normal case for biases and scale factors.
//
// Given two tensors whose byte ranges fall inside one host page
// When both are registered
// Then they are pinned as a single range
TEST_F(CheckpointReaderTest, RegistersTensorsSharingAPageAsOneRange)
{
    writeShard(kShardA, {{"a", 2}, {"b", 2}, {"c", 2}});
    CheckpointReader reader(mDir);

    // Succeeding at all is the primary evidence: an implementation that skipped
    // the merge would hand cudaHostRegister three overlapping page ranges, and
    // the second call would fail.
    ASSERT_NO_THROW(reader.registerTensors({"a", "b", "c"}));

    // All three live in the first page, so one range covers them.
    EXPECT_LE(reader.registeredBytes(), pageSize());

    View view{};
    ASSERT_TRUE(reader.find("c", view));
    EXPECT_EQ(deviceFloats(view), expectedRamp(2, 2));
}

// The other half of merging: ranges that do not touch stay separate. Collapsing
// them into their enclosing span would pin every page between two requested
// tensors, which on a real multi-gigabyte checkpoint is the whole shard.
//
// Given two tensors separated by pages nothing asked for
// When both are registered
// Then the gap is left unpinned
TEST_F(CheckpointReaderTest, DoesNotPinThePagesBetweenDistantTensors)
{
    int64_t const fillerFloats = 4 * static_cast<int64_t>(pageSize()) / static_cast<int64_t>(sizeof(float));
    writeShard(kShardA, {{"head", 2}, {"filler", fillerFloats}, {"tail", 2}});
    CheckpointReader reader(mDir);

    reader.registerTensors({"head", "tail"});

    // One page per requested tensor at most; the four pages of filler between
    // them are not mapped.
    EXPECT_LE(reader.registeredBytes(), 2U * pageSize());

    // Both are nonetheless reachable through their device aliases.
    View head{};
    View tail{};
    ASSERT_TRUE(reader.find("head", head));
    ASSERT_TRUE(reader.find("tail", tail));
    EXPECT_EQ(deviceFloats(head), expectedRamp(0, 2));
    EXPECT_EQ(deviceFloats(tail), expectedRamp(2, 2));
}

// Registration is per shard and exclusive. Registering twice without releasing
// would leak the first mapping and is refused rather than silently accumulating.
TEST_F(CheckpointReaderTest, RefusesToRegisterAShardTwice)
{
    writeShard(kShardA, {{"weight", 8}});
    CheckpointReader reader(mDir);
    reader.registerTensors({"weight"});

    EXPECT_THROW(reader.registerTensors({"weight"}), std::runtime_error);
}

// Runtime initialization registers and releases once per output binding, so the
// cycle has to be repeatable and has to return the accounting to zero. A
// register that survived release would grow pinned memory across the whole load.
//
// Given a shard that has been registered
// When it is released
// Then nothing stays pinned and the same shard can be registered again
TEST_F(CheckpointReaderTest, ReleaseReturnsAccountingToZeroAndAllowsReRegistration)
{
    writeShard(kShardA, {{"weight", 8}});
    CheckpointReader reader(mDir);

    reader.registerTensors({"weight"});
    size_t const registered = reader.registeredBytes();
    ASSERT_GT(registered, 0U);

    reader.unregisterTensors();
    EXPECT_EQ(reader.registeredBytes(), 0U);

    EXPECT_NO_THROW(reader.registerTensors({"weight"}));
    EXPECT_EQ(reader.registeredBytes(), registered);

    // The peak is a high-water mark: it reports the most that was ever mapped at
    // once, so releasing must not reset it. It is what the load reports as its
    // pinned-memory cost.
    EXPECT_EQ(reader.peakRegisteredBytes(), registered);
}

// Names the checkpoint does not contain are skipped rather than fatal, because
// callers register a superset covering optional weights.
TEST_F(CheckpointReaderTest, RegisteringUnknownNamesIsANoOp)
{
    writeShard(kShardA, {{"weight", 8}});
    CheckpointReader reader(mDir);

    EXPECT_NO_THROW(reader.registerTensors({"absent"}));
    EXPECT_EQ(reader.registeredBytes(), 0U);
}

// ---------------------------------------------------------------------------
// Bounded windows
// ---------------------------------------------------------------------------

// registerTensorRange maps one window inside a tensor so a large weight can be
// converted in slices. The returned view must point at exactly the requested
// window: it is derived by subtracting a page-aligned base, which is where an
// off-by-one page lands the caller on the wrong slice.
//
// Given a request for a window inside one tensor
// When it is registered and read
// Then the view starts at that window and is exactly that long
TEST_F(CheckpointReaderTest, BoundedRangeViewAddressesExactlyTheRequestedWindow)
{
    constexpr int64_t kCount{64};
    writeShard(kShardA, {{"weight", kCount}});
    CheckpointReader reader(mDir);

    constexpr size_t kSkipped{16};
    constexpr size_t kTaken{8};
    View const view = reader.registerTensorRange("weight", kSkipped * sizeof(float), kTaken * sizeof(float));

    EXPECT_EQ(view.bytes, kTaken * sizeof(float));
    // Shape and dtype describe the whole tensor; only the byte window is narrowed.
    EXPECT_EQ(view.shape, Coords({kCount}));

    auto const full = expectedRamp(0, kCount);
    std::vector<float> const expected(full.begin() + kSkipped, full.begin() + kSkipped + kTaken);
    EXPECT_EQ(hostFloats(view), expected);
    EXPECT_EQ(deviceFloats(view), expected);

    reader.unregisterTensors();
}

// The header states that no other range may be registered while a bounded view
// is in use, because both paths map overlapping page ranges of the same file.
//
// Given a tensor window already registered
// When an overlapping window is registered
// Then it is refused
TEST_F(CheckpointReaderTest, BoundedRangeRefusesToOverlapAnExistingRegistration)
{
    writeShard(kShardA, {{"weight", 64}});
    CheckpointReader reader(mDir);
    reader.registerTensors({"weight"});

    EXPECT_THROW(static_cast<void>(reader.registerTensorRange("weight", 0, sizeof(float))), std::runtime_error);

    reader.unregisterTensors();
    EXPECT_NO_THROW(static_cast<void>(reader.registerTensorRange("weight", 0, sizeof(float))));
    reader.unregisterTensors();
}

// A window that runs past the tensor would map pages belonging to the next
// tensor, or past the file. An empty window has no valid page range at all.
TEST_F(CheckpointReaderTest, BoundedRangeRejectsWindowsOutsideTheTensor)
{
    constexpr size_t kBytes{64 * sizeof(float)};
    writeShard(kShardA, {{"weight", 64}});
    CheckpointReader reader(mDir);

    EXPECT_THROW(static_cast<void>(reader.registerTensorRange("weight", 0, kBytes + 1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(reader.registerTensorRange("weight", kBytes, sizeof(float))), std::runtime_error);
    EXPECT_THROW(static_cast<void>(reader.registerTensorRange("weight", 0, 0)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(reader.registerTensorRange("absent", 0, sizeof(float))), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Explicit tensor locations (indexed PyTorch ZIP ranges)
// ---------------------------------------------------------------------------

//! An explicit location names a byte range inside a file the reader does not
//! parse, which is how tensors inside a PyTorch ZIP are addressed.
TensorLocation makeLocation(std::string name, std::string file, size_t offset, Coords shape, size_t bytes)
{
    TensorLocation location;
    location.name = std::move(name);
    location.file = std::move(file);
    location.dtype = "F32";
    location.shape = std::move(shape);
    location.offset = offset;
    location.bytes = bytes;
    return location;
}

// The raw-file path: shape and dtype come from the caller, not from a parsed
// header, and the bytes are taken verbatim from the declared offset.
//
// Given a tensor described by a file, an offset and a length instead of by a safetensors header
// When it is read
// Then exactly those bytes come back
TEST_F(CheckpointReaderTest, ExplicitLocationReadsARawByteRangeVerbatim)
{
    constexpr size_t kLeadingBytes{32};
    std::vector<float> const payload{5.0F, 6.0F, 7.0F, 8.0F};
    {
        std::ofstream raw(mDir / "archive.bin", std::ios::binary);
        std::vector<char> const filler(kLeadingBytes, '\0');
        raw.write(filler.data(), static_cast<std::streamsize>(filler.size()));
        raw.write(reinterpret_cast<char const*>(payload.data()),
            static_cast<std::streamsize>(payload.size() * sizeof(float)));
    }

    CheckpointReader reader(
        mDir, {makeLocation("zip.weight", "archive.bin", kLeadingBytes, Coords({4}), payload.size() * sizeof(float))});

    View view{};
    ASSERT_TRUE(reader.findHost("zip.weight", view));
    EXPECT_EQ(view.dtype, nvinfer1::DataType::kFLOAT);
    EXPECT_EQ(view.shape, Coords({4}));
    EXPECT_EQ(hostFloats(view), payload);
}

// The file field is attacker-adjacent only in the sense that it comes from a
// config the runtime did not write. It is constrained to stay inside the
// checkpoint directory, so absolute paths and parent traversal are refused
// rather than resolved.
//
// Given an explicit location whose path escapes the checkpoint directory
// When it is registered
// Then it is refused, while the same file reached from inside the directory is accepted
TEST_F(CheckpointReaderTest, ExplicitLocationMustStayInsideTheCheckpointDirectory)
{
    // The traversal target has to exist outside the checkpoint directory, or the
    // "file does not exist" check would reject it first and this test would pass
    // without the traversal guard ever running.
    std::ofstream(mDir / "outside.bin", std::ios::binary) << std::string(64, '\0');

    auto const inner = mDir / "inner";
    std::filesystem::create_directories(inner);
    std::filesystem::copy_file(mDir / "outside.bin", inner / "inside.bin");
    {
        std::vector<Tensor> shard;
        shard.emplace_back(Coords({4}), DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "weight");
        ASSERT_TRUE(safetensors::saveSafetensors(inner / kShardA, shard, mStream));
    }

    // Resolvable, and rejected anyway: a config may not reach outside the
    // directory it was loaded from.
    EXPECT_THROW(
        CheckpointReader(inner, {makeLocation("escape", "../outside.bin", 0, Coords({4}), 16)}), std::runtime_error);
    EXPECT_THROW(
        CheckpointReader(inner, {makeLocation("absolute", "/etc/hostname", 0, Coords({4}), 16)}), std::runtime_error);
    EXPECT_THROW(
        CheckpointReader(inner, {makeLocation("nofile", "missing.bin", 0, Coords({4}), 16)}), std::runtime_error);

    // The control: the same file reached without traversal is accepted, so the
    // three rejections above are attributable to the path, not to the location
    // being malformed in some other way.
    EXPECT_NO_THROW(CheckpointReader(inner, {makeLocation("ok", "inside.bin", 0, Coords({4}), 16)}));
}

// A declared range that runs past the file would map pages that do not exist.
// The check uses the real file size rather than trusting the declaration.
TEST_F(CheckpointReaderTest, ExplicitLocationRejectsRangesPastTheEndOfTheFile)
{
    writeShard(kShardA, {{"weight", 4}});
    constexpr size_t kFileBytes{64};
    std::ofstream(mDir / "archive.bin", std::ios::binary) << std::string(kFileBytes, '\0');

    EXPECT_THROW(CheckpointReader(mDir, {makeLocation("past.end", "archive.bin", 0, Coords({4}), kFileBytes + 1)}),
        std::runtime_error);
    EXPECT_THROW(CheckpointReader(mDir, {makeLocation("past.offset", "archive.bin", kFileBytes, Coords({4}), 16)}),
        std::runtime_error);
    EXPECT_THROW(CheckpointReader(mDir, {makeLocation("empty", "archive.bin", 0, Coords({4}), 0)}), std::runtime_error);
}

// Two locations naming one tensor differently cannot both be right, and picking
// either silently would make weight assembly depend on vector order.
//
// Given one tensor name declared twice with different byte ranges
// When the second definition arrives
// Then it is refused and the first definition stands
TEST_F(CheckpointReaderTest, ExplicitLocationRejectsConflictingDefinitionsOfOneTensor)
{
    writeShard(kShardA, {{"weight", 4}});
    std::ofstream(mDir / "archive.bin", std::ios::binary) << std::string(128, '\0');

    EXPECT_THROW(CheckpointReader(mDir,
                     {makeLocation("dup", "archive.bin", 0, Coords({4}), 16),
                         makeLocation("dup", "archive.bin", 32, Coords({4}), 16)}),
        std::runtime_error);

    // Restating the identical location is not a conflict: callers may pass a
    // list that mentions a shared tensor once per consumer.
    EXPECT_NO_THROW(CheckpointReader(mDir,
        {makeLocation("dup", "archive.bin", 0, Coords({4}), 16),
            makeLocation("dup", "archive.bin", 0, Coords({4}), 16)}));
}

} // namespace
