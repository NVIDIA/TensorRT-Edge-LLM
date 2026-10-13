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

// loadCheckpointWeight() is the recipe interpreter that turns checkpoint bytes
// into the layout an engine binding expects. Everything below it is an anonymous
// helper, so these tests drive the public entry point and assert on the bytes it
// produces, which is also the level a recipe's contract is stated at.
//
// The scope here is the three recipes every model uses regardless of
// quantization -- identity, cast_to_fp32, fill -- plus the dispatch guards and
// the two phase predicates the caller sequences work with. The quantized MoE and
// GPTQ recipes need checkpoint tensors whose layout is itself the thing under
// test; they are left to the kernel-level suites that already cover those
// transforms.

#include "runtime/weight/checkpointWeightAssemble.h"

#include "common/tensor.h"
#include "runtime/weight/checkpointReader.h"
#include "scratchDir.h"
#include "testUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;
using ::testing::HasSubstr;

namespace
{

constexpr char const* kShard{"model.safetensors"};

//! Every recipe name loadCheckpointWeight() dispatches on, other than the one
//! prerequisite. Listed so the phase and registration predicates are asserted
//! against the whole set rather than a convenient sample.
constexpr char const* kOrdinaryRecipes[]{"identity", "cast_to_fp32", "fill", "gptq_ffn_qweight", "gptq_qkv_qweight",
    "gptq_qkv_scales", "awq_ffn_qweight", "modelopt_awq_ffn_qweight", "nvfp4_gated_fc1_qweight",
    "nvfp4_gated_fc2_scale", "nvfp4_expert_qweight", "nvfp4_fc1_alpha", "int4_moe_gate_up", "int4_moe_down_scales",
    "fp16_moe_fc1", "fp16_moe_fc2"};

class CheckpointWeightAssembleTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("checkpointWeightAssembleTests");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! One checkpoint tensor described by the values it holds; the dtype decides
    //! how those values are narrowed on the way to disk.
    struct SourceTensor
    {
        std::string name;
        Coords shape;
        nvinfer1::DataType dtype;
        std::vector<float> values;
    };

    //! Serialize a safetensors shard with the data section 8-byte aligned, the
    //! way the reference exporter pads it.
    //!
    //! The recipes hand mapped source pointers straight to typed kernels, so a
    //! shard whose data section landed on an odd offset would fault inside the
    //! kernel instead of exercising the recipe. Building the file here rather
    //! than through saveSafetensors, which does not pad, keeps the fixture
    //! shaped like the checkpoints this code actually reads.
    void writeCheckpoint(std::vector<SourceTensor> const& sources)
    {
        constexpr size_t kAlignment{8};
        auto const alignUp = [](size_t value) { return (value + kAlignment - 1) & ~(kAlignment - 1); };

        Json header;
        std::vector<char> payload;
        for (SourceTensor const& source : sources)
        {
            ASSERT_EQ(source.shape.volume(), static_cast<int64_t>(source.values.size()));
            payload.resize(alignUp(payload.size()));
            size_t const begin = payload.size();
            for (float const value : source.values)
            {
                appendElement(payload, value, source.dtype);
            }

            std::vector<int64_t> shape;
            for (int32_t dim = 0; dim < source.shape.getNumDims(); ++dim)
            {
                shape.push_back(source.shape[dim]);
            }
            header[source.name] = Json::object({{"dtype", dtypeName(source.dtype)}, {"shape", shape},
                {"data_offsets", Json::array({begin, payload.size()})}});
        }

        std::string headerText = header.dump();
        headerText.resize(alignUp(sizeof(uint64_t) + headerText.size()) - sizeof(uint64_t), ' ');
        uint64_t const headerBytes = headerText.size();

        std::ofstream file(mDir / kShard, std::ios::binary);
        file.write(reinterpret_cast<char const*>(&headerBytes), sizeof(headerBytes));
        file.write(headerText.data(), static_cast<std::streamsize>(headerText.size()));
        file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    static char const* dtypeName(nvinfer1::DataType dtype)
    {
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT: return "F32";
        case nvinfer1::DataType::kHALF: return "F16";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kINT32: return "I32";
        default: return "";
        }
    }

    static void appendElement(std::vector<char>& payload, float value, nvinfer1::DataType dtype)
    {
        auto append = [&payload](void const* bytes, size_t count) {
            auto const* first = static_cast<char const*>(bytes);
            payload.insert(payload.end(), first, first + count);
        };
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT: append(&value, sizeof(value)); break;
        case nvinfer1::DataType::kHALF:
        {
            __half const half = __float2half(value);
            append(&half, sizeof(half));
            break;
        }
        case nvinfer1::DataType::kBF16:
        {
            __nv_bfloat16 const bf16 = __float2bfloat16(value);
            append(&bf16, sizeof(bf16));
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            auto const integer = static_cast<int32_t>(value);
            append(&integer, sizeof(integer));
            break;
        }
        default: ADD_FAILURE() << "unsupported source dtype in test fixture";
        }
    }

    Tensor makeOutput(std::string const& engineName, Coords const& shape, nvinfer1::DataType dtype)
    {
        return Tensor(shape, DeviceType::kGPU, dtype, engineName);
    }

    std::vector<float> readOutput(Tensor const& output)
    {
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<float> values;
        if (output.getDataType() == nvinfer1::DataType::kFLOAT)
        {
            return copyDeviceToHost<float>(output);
        }
        for (__half const value : copyDeviceToHost<__half>(output))
        {
            values.push_back(__half2float(value));
        }
        return values;
    }

    //! Sources that a recipe reads through find() must be CUDA-mapped by the
    //! caller. Identity recipes map their own windows and must not be pre-mapped;
    //! that split is exactly what checkpointWeightManagesSourceRegistration
    //! reports, and it is asserted directly in its own test.
    void loadWithCallerRegistration(
        CheckpointReader& reader, Json const& binding, std::vector<std::string> const& sourceKeys, Tensor& output)
    {
        reader.registerTensors(sourceKeys);
        try
        {
            loadCheckpointWeight(reader, binding, {}, output, mStream);
        }
        catch (...)
        {
            reader.unregisterTensors();
            throw;
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        reader.unregisterTensors();
    }

    template <typename Fn>
    static std::string messageFrom(Fn&& fn)
    {
        try
        {
            fn();
        }
        catch (std::exception const& error)
        {
            return error.what();
        }
        return {};
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
};

Json makeBinding(std::string const& engineName, std::string const& assemble, std::vector<std::string> const& keys)
{
    Json binding;
    binding["engine_name"] = engineName;
    if (!assemble.empty())
    {
        binding["assemble"] = assemble;
    }
    if (!keys.empty())
    {
        binding["checkpoint_keys"] = keys;
    }
    return binding;
}

// ---------------------------------------------------------------------------
// Phase and registration predicates
// ---------------------------------------------------------------------------

// The caller materializes prerequisites before ordinary weights. Only the GPTQ
// activation permutation is consumed by other recipes, so only it is a
// prerequisite. Promoting an ordinary recipe would reorder the load; demoting
// this one would run the GPTQ qweight recipes before the permutation they read.
//
// Given every recipe the loader dispatches on
// When each is asked which phase it belongs to
// Then only the activation permutation runs early
TEST_F(CheckpointWeightAssembleTest, OnlyTheActivationPermutationIsAPrerequisitePhase)
{
    EXPECT_EQ(checkpointWeightPhase(makeBinding("w", "gptq_activation_permutation", {})),
        CheckpointWeightPhase::kPrerequisite);

    for (char const* recipe : kOrdinaryRecipes)
    {
        EXPECT_EQ(checkpointWeightPhase(makeBinding("w", recipe, {})), CheckpointWeightPhase::kWeight) << recipe;
    }
    // An absent `assemble` is the identity recipe, which is an ordinary weight.
    EXPECT_EQ(checkpointWeightPhase(makeBinding("w", "", {})), CheckpointWeightPhase::kWeight);
}

// Identity maps and releases bounded source windows itself so a multi-gigabyte
// weight never has its whole source pinned at once. Every other recipe reads
// through find(), which requires the caller to have registered first. A wrong
// answer here fails the load either way: pre-registering an identity source
// collides with its own mapping, and not registering another recipe's sources
// leaves find() with no device alias to return.
//
// Given every recipe the loader dispatches on
// When each is asked whether it registers its own sources
// Then only identity does
TEST_F(CheckpointWeightAssembleTest, OnlyIdentityRecipesManageTheirOwnSourceRegistration)
{
    EXPECT_TRUE(checkpointWeightManagesSourceRegistration(makeBinding("w", "", {})));
    EXPECT_TRUE(checkpointWeightManagesSourceRegistration(makeBinding("w", "identity", {})));

    for (char const* recipe : kOrdinaryRecipes)
    {
        if (std::string(recipe) == "identity")
        {
            continue;
        }
        EXPECT_FALSE(checkpointWeightManagesSourceRegistration(makeBinding("w", recipe, {}))) << recipe;
    }
}

// ---------------------------------------------------------------------------
// Dispatch guards
// ---------------------------------------------------------------------------

// An unrecognized recipe is a build/runtime version mismatch: the exporter
// emitted a transform this runtime does not implement. Silently falling back to
// identity would copy quantized bytes into a binding expecting a repacked
// layout, so it fails and names the op.
TEST_F(CheckpointWeightAssembleTest, RejectsAnUnknownAssembleOp)
{
    writeCheckpoint({{"src", Coords({4}), nvinfer1::DataType::kFLOAT, {1.0F, 2.0F, 3.0F, 4.0F}}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("w", Coords({4}), nvinfer1::DataType::kFLOAT);
    Json const binding = makeBinding("w", "nvfp4_future_recipe", {"src"});

    EXPECT_THAT(messageFrom([&] { loadCheckpointWeight(reader, binding, {}, output, mStream); }),
        HasSubstr("nvfp4_future_recipe"));
}

// The binding and the output tensor are looked up independently by the caller.
// Checking that they agree is what keeps a recipe from writing one weight's
// bytes into another weight's buffer, which no later validation would catch.
TEST_F(CheckpointWeightAssembleTest, RejectsABindingThatDoesNotNameTheOutputTensor)
{
    writeCheckpoint({{"src", Coords({4}), nvinfer1::DataType::kFLOAT, {1.0F, 2.0F, 3.0F, 4.0F}}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("expected.weight", Coords({4}), nvinfer1::DataType::kFLOAT);

    EXPECT_THROW(loadCheckpointWeight(reader, makeBinding("other.weight", "", {"src"}), {}, output, mStream),
        std::runtime_error);

    Json noName;
    noName["checkpoint_keys"] = Json::array({"src"});
    EXPECT_THROW(loadCheckpointWeight(reader, noName, {}, output, mStream), std::runtime_error);

    EXPECT_THROW(loadCheckpointWeight(reader, Json("not an object"), {}, output, mStream), std::runtime_error);
}

// ---------------------------------------------------------------------------
// identity
// ---------------------------------------------------------------------------

// The recipe behind nearly every weight in a plain FP16 model: matching dtype
// and shape, copied through. Values are chosen exactly representable in FP16 so
// a mismatch means the bytes moved wrong, not that they were rounded.
TEST_F(CheckpointWeightAssembleTest, IdentityCopiesAMatchingTensorVerbatim)
{
    std::vector<float> const values{1.0F, -2.0F, 3.5F, 4.0F, -5.25F, 6.0F};
    writeCheckpoint({{"layer.weight", Coords({2, 3}), nvinfer1::DataType::kHALF, values}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("engine.weight", Coords({2, 3}), nvinfer1::DataType::kHALF);

    loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"layer.weight"}), {}, output, mStream);

    EXPECT_EQ(readOutput(output), values);
}

// Checkpoints are exported in the trained dtype; an FP16 engine binding gets the
// narrowing for free. The conversion is inferred from the dtype pair rather than
// declared in the recipe, so both supported source dtypes are exercised.
//
// Given a checkpoint tensor stored wider than the binding it feeds
// When identity copies it
// Then the values arrive narrowed
TEST_F(CheckpointWeightAssembleTest, IdentityNarrowsFp32AndBf16SourcesToAnFp16Binding)
{
    std::vector<float> const values{1.0F, -2.0F, 3.5F, 4.0F};
    writeCheckpoint({{"fp32.weight", Coords({2, 2}), nvinfer1::DataType::kFLOAT, values},
        {"bf16.weight", Coords({2, 2}), nvinfer1::DataType::kBF16, values}});
    CheckpointReader reader(mDir);

    Tensor fromFp32 = makeOutput("engine.weight", Coords({2, 2}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"fp32.weight"}), {}, fromFp32, mStream);
    EXPECT_EQ(readOutput(fromFp32), values);

    Tensor fromBf16 = makeOutput("engine.weight", Coords({2, 2}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"bf16.weight"}), {}, fromBf16, mStream);
    EXPECT_EQ(readOutput(fromBf16), values);
}

// A binding whose shape is the source's reverse is a transposed weight, and the
// recipe infers that from the shapes alone. Asserting the transposed values (not
// just the shape) is what separates a real transpose from a reinterpreting copy,
// which would produce identically shaped garbage.
//
// Given a binding whose shape is the source's two dimensions swapped
// When identity copies it
// Then the data is transposed, not just reshaped over the same bytes
TEST_F(CheckpointWeightAssembleTest, IdentityTransposesWhenTheBindingSwapsTheSourceDimensions)
{
    // Row-major [2,3]: rows {1,2,3} and {4,5,6}.
    writeCheckpoint({{"src", Coords({2, 3}), nvinfer1::DataType::kHALF, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("engine.weight", Coords({3, 2}), nvinfer1::DataType::kHALF);

    loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"src"}), {}, output, mStream);

    // Row-major [3,2]: rows {1,4}, {2,5}, {3,6}.
    EXPECT_EQ(readOutput(output), std::vector<float>({1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));
}

// A shape that is neither equal nor a 2-D transpose means the checkpoint and the
// engine disagree about the weight. Proceeding would read or write out of
// bounds, so it is rejected with both shapes in the message.
TEST_F(CheckpointWeightAssembleTest, IdentityRejectsAShapeThatIsNotATranspose)
{
    writeCheckpoint({{"src", Coords({2, 3}), nvinfer1::DataType::kHALF, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}}});
    CheckpointReader reader(mDir);

    Tensor wrongExtent = makeOutput("engine.weight", Coords({4, 5}), nvinfer1::DataType::kHALF);
    EXPECT_THROW(
        loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"src"}), {}, wrongExtent, mStream),
        std::runtime_error);

    // Same element count, different rank: a flattening is not a transpose.
    Tensor flattened = makeOutput("engine.weight", Coords({6}), nvinfer1::DataType::kHALF);
    EXPECT_THROW(
        loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"src"}), {}, flattened, mStream),
        std::runtime_error);
}

// Narrowing to FP16 is the only conversion identity performs. Any other dtype
// pair is a binding the exporter should have given a real recipe, and copying
// the bytes through would reinterpret them.
TEST_F(CheckpointWeightAssembleTest, IdentityRejectsAConversionItDoesNotImplement)
{
    writeCheckpoint({{"src", Coords({4}), nvinfer1::DataType::kHALF, {1.0F, 2.0F, 3.0F, 4.0F}}});
    CheckpointReader reader(mDir);

    // Widening FP16 -> FP32 is cast_to_fp32's job, not identity's.
    Tensor widened = makeOutput("engine.weight", Coords({4}), nvinfer1::DataType::kFLOAT);
    EXPECT_THROW(loadCheckpointWeight(reader, makeBinding("engine.weight", "identity", {"src"}), {}, widened, mStream),
        std::runtime_error);
}

// checkpoint_keys is a fallback list: an engine binding names every checkpoint
// spelling it accepts, and the first one present wins. Order is the contract, so
// two keys that both exist must resolve to the first.
//
// Given a binding naming several candidate checkpoint keys, only a later one of which exists
// When identity resolves it
// Then it takes that one
TEST_F(CheckpointWeightAssembleTest, IdentityTakesTheFirstCheckpointKeyThatExists)
{
    writeCheckpoint({{"preferred", Coords({2}), nvinfer1::DataType::kHALF, {1.0F, 2.0F}},
        {"fallback", Coords({2}), nvinfer1::DataType::kHALF, {30.0F, 40.0F}}});
    CheckpointReader reader(mDir);

    Tensor bothPresent = makeOutput("engine.weight", Coords({2}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(
        reader, makeBinding("engine.weight", "identity", {"preferred", "fallback"}), {}, bothPresent, mStream);
    EXPECT_EQ(readOutput(bothPresent), std::vector<float>({1.0F, 2.0F}));

    Tensor firstAbsent = makeOutput("engine.weight", Coords({2}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(
        reader, makeBinding("engine.weight", "identity", {"absent", "fallback"}), {}, firstAbsent, mStream);
    EXPECT_EQ(readOutput(firstAbsent), std::vector<float>({30.0F, 40.0F}));
}

// When no spelling matches, the message has to name the engine binding rather
// than the last key tried: the binding is what the reader of the log can look up.
//
// Given a binding none of whose candidate keys exist
// When identity resolves it
// Then it fails, naming the binding
TEST_F(CheckpointWeightAssembleTest, IdentityFailsNamingTheBindingWhenNoKeyExists)
{
    writeCheckpoint({{"present", Coords({2}), nvinfer1::DataType::kHALF, {1.0F, 2.0F}}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("engine.weight", Coords({2}), nvinfer1::DataType::kHALF);

    EXPECT_THAT(messageFrom([&] {
        loadCheckpointWeight(
            reader, makeBinding("engine.weight", "identity", {"absent.a", "absent.b"}), {}, output, mStream);
    }),
        HasSubstr("engine.weight"));
}

// Some models fold a constant embedding scale into the table at load time rather
// than at inference. It applies after the copy, so the assertion is that the
// stored values are scaled, and it is FP16-only.
//
// Given an embedding binding declaring a scale
// When identity copies it
// Then the destination holds the scaled values
TEST_F(CheckpointWeightAssembleTest, IdentityAppliesTheEmbeddingScaleAfterCopying)
{
    writeCheckpoint({{"embed", Coords({4}), nvinfer1::DataType::kHALF, {1.0F, 2.0F, 3.0F, 4.0F}}});
    CheckpointReader reader(mDir);

    Json scaled = makeBinding("engine.embed", "identity", {"embed"});
    scaled["embedding_scale"] = 2.0F;
    Tensor output = makeOutput("engine.embed", Coords({4}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(reader, scaled, {}, output, mStream);
    EXPECT_EQ(readOutput(output), std::vector<float>({2.0F, 4.0F, 6.0F, 8.0F}));

    // A scale of exactly 1 must leave the copy untouched rather than run the
    // kernel, which is what lets non-FP16 bindings use this recipe at all.
    Json unscaled = makeBinding("engine.embed", "identity", {"embed"});
    unscaled["embedding_scale"] = 1.0F;
    Tensor untouched = makeOutput("engine.embed", Coords({4}), nvinfer1::DataType::kHALF);
    loadCheckpointWeight(reader, unscaled, {}, untouched, mStream);
    EXPECT_EQ(readOutput(untouched), std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}));
}

// ---------------------------------------------------------------------------
// cast_to_fp32
// ---------------------------------------------------------------------------

// Bindings that must be FP32 regardless of the checkpoint dtype -- layer norms
// and scale factors -- go through this recipe. All three accepted source dtypes
// have to land on the same values.
//
// Given a source in each dtype the recipe accepts
// When cast_to_fp32 runs
// Then all of them arrive as the same fp32 values
TEST_F(CheckpointWeightAssembleTest, CastToFp32WidensEverySupportedSourceDtype)
{
    std::vector<float> const values{1.0F, -2.0F, 0.5F, 4.0F};
    writeCheckpoint({{"as.fp32", Coords({4}), nvinfer1::DataType::kFLOAT, values},
        {"as.fp16", Coords({4}), nvinfer1::DataType::kHALF, values},
        {"as.bf16", Coords({4}), nvinfer1::DataType::kBF16, values}});
    CheckpointReader reader(mDir);

    for (char const* key : {"as.fp32", "as.fp16", "as.bf16"})
    {
        Tensor output = makeOutput("engine.norm", Coords({4}), nvinfer1::DataType::kFLOAT);
        loadWithCallerRegistration(reader, makeBinding("engine.norm", "cast_to_fp32", {key}), {key}, output);
        EXPECT_EQ(readOutput(output), values) << key;
    }
}

// The recipe's whole purpose is producing FP32, and it takes exactly one source.
// Both guards exist because the alternative is writing FP32 bytes into a
// narrower binding, overrunning it.
TEST_F(CheckpointWeightAssembleTest, CastToFp32RejectsANonFp32BindingOrMultipleSources)
{
    std::vector<float> const values{1.0F, 2.0F, 3.0F, 4.0F};
    writeCheckpoint(
        {{"a", Coords({4}), nvinfer1::DataType::kHALF, values}, {"b", Coords({4}), nvinfer1::DataType::kHALF, values}});
    CheckpointReader reader(mDir);

    Tensor halfBinding = makeOutput("engine.norm", Coords({4}), nvinfer1::DataType::kHALF);
    EXPECT_THROW(
        loadWithCallerRegistration(reader, makeBinding("engine.norm", "cast_to_fp32", {"a"}), {"a"}, halfBinding),
        std::runtime_error);

    Tensor twoSources = makeOutput("engine.norm", Coords({4}), nvinfer1::DataType::kFLOAT);
    EXPECT_THROW(loadWithCallerRegistration(
                     reader, makeBinding("engine.norm", "cast_to_fp32", {"a", "b"}), {"a", "b"}, twoSources),
        std::runtime_error);

    // A shape disagreement is the same class of error and is caught before any
    // copy is enqueued.
    Tensor wrongShape = makeOutput("engine.norm", Coords({8}), nvinfer1::DataType::kFLOAT);
    EXPECT_THROW(
        loadWithCallerRegistration(reader, makeBinding("engine.norm", "cast_to_fp32", {"a"}), {"a"}, wrongShape),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// fill
// ---------------------------------------------------------------------------

// Some engine inputs are generated rather than loaded. The recipe reads no
// checkpoint tensor at all, so the only contract is that every element gets the
// configured value -- and that an absent fill_value means zero rather than
// leaving the allocation as it was.
TEST_F(CheckpointWeightAssembleTest, FillWritesTheConfiguredValueAcrossTheWholeBinding)
{
    writeCheckpoint({{"unused", Coords({2}), nvinfer1::DataType::kFLOAT, {1.0F, 2.0F}}});
    CheckpointReader reader(mDir);

    Json withValue = makeBinding("engine.bias", "fill", {});
    withValue["fill_value"] = 0.5F;
    Tensor filled = makeOutput("engine.bias", Coords({2, 3}), nvinfer1::DataType::kFLOAT);
    loadCheckpointWeight(reader, withValue, {}, filled, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_THAT(readOutput(filled), ::testing::Each(0.5F));

    Tensor defaulted = makeOutput("engine.bias", Coords({2, 3}), nvinfer1::DataType::kFLOAT);
    loadCheckpointWeight(reader, makeBinding("engine.bias", "fill", {}), {}, defaulted, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_THAT(readOutput(defaulted), ::testing::Each(0.0F));
}

// The fill kernel writes FP32 words. A narrower binding would be overrun by
// exactly the ratio of the element sizes.
TEST_F(CheckpointWeightAssembleTest, FillRejectsANonFp32Binding)
{
    writeCheckpoint({{"unused", Coords({2}), nvinfer1::DataType::kFLOAT, {1.0F, 2.0F}}});
    CheckpointReader reader(mDir);
    Tensor output = makeOutput("engine.bias", Coords({4}), nvinfer1::DataType::kHALF);

    EXPECT_THROW(
        loadCheckpointWeight(reader, makeBinding("engine.bias", "fill", {}), {}, output, mStream), std::runtime_error);
}

} // namespace
