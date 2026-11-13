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

#include "runtime/commitPolicy/gdnConvCommitPolicy.h"
#include "runtime/commitPolicy/mamba2CommitPolicy.h"
#include "runtime/commitPolicy/recurrentCommitPolicyFactory.h"

#include "common/bindingNames.h"
#include "common/cudaMacros.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/exec/tensorRegistry.h"
#include "runtime/mambaCacheManager.h"

#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using trt_edgellm::DataType;

namespace
{

using Config = RecurrentCommitPolicy::Config;

//! A GDN spec-verify geometry whose recurrent and convolution extents are all distinct, so a reshape
//! that swapped an axis or reused one layer's extent for another is visible in the reported shape.
Config makeGdnConfig()
{
    Config cfg{};
    cfg.numRecurrentLayers = 1;
    cfg.maxBatchSize = 2;
    cfg.recurrentStateNumHeads = 2;
    cfg.recurrentStateHeadDim = 4;
    cfg.recurrentStateSize = 3;
    cfg.convDim = 5;
    cfg.convKernel = 2;
    cfg.maxIntermediateSeqLen = 8;
    cfg.recurrentStateType = DataType::kHALF;
    cfg.convStateType = DataType::kHALF;
    return cfg;
}

//! A Mamba2 spec-verify geometry. Its B/C group count is distinct from the head count so the replay-B
//! stash cannot be confused with the dA/dt stashes that key off the head count.
Config makeMambaConfig()
{
    Config cfg{};
    cfg.numRecurrentLayers = 1;
    cfg.maxBatchSize = 2;
    cfg.recurrentStateNumHeads = 2;
    cfg.recurrentStateHeadDim = 4;
    cfg.recurrentStateSize = 3;
    cfg.convDim = 5;
    cfg.convKernel = 2;
    cfg.maxIntermediateSeqLen = 8;
    cfg.recurrentStateType = DataType::kHALF;
    cfg.convStateType = DataType::kHALF;
    cfg.recurrentStateNumGroups = 1;
    return cfg;
}

//! `count` placeholder slabs. The policy only reads their device pointers when it packs the commit
//! layer-info bundle, so the row shape is immaterial to the behaviours under test.
std::vector<Tensor> makeSlabs(int32_t count)
{
    std::vector<Tensor> slabs;
    slabs.reserve(count);
    for (int32_t i = 0; i < count; ++i)
    {
        slabs.emplace_back(Tensor({1}, DeviceType::kGPU, DataType::kFLOAT, "slab_" + std::to_string(i)));
    }
    return slabs;
}

TensorSpec const* findSpec(TensorRegistry const& reg, std::string const& name)
{
    for (auto const& spec : reg.allExpandedSpecs())
    {
        if (spec.name == name)
        {
            return &spec;
        }
    }
    return nullptr;
}

//! Drive one commit and surface the validation message it raised, or empty when it did not throw.
std::string commitTreeThrowMessage(GdnConvCommitPolicy& policy, Tensor const& acceptedNodeIds,
    Tensor const& acceptLengths, Tensor const& stateIndices, int32_t verifySize, cudaStream_t stream)
{
    try
    {
        policy.commitTree(acceptedNodeIds, acceptLengths, stateIndices, verifySize, stream);
    }
    catch (std::exception const& e)
    {
        return e.what();
    }
    return {};
}

} // namespace

// Given a mamba2 mixer, when the factory builds its commit policy, then the policy reconstructs the
// accepted state by replay and reports both the replay recurrent stash and the conv snapshot as
// allocated. The mixer kind alone fixes the commit path -- there is no runtime mode flag -- so the
// factory must hand back the replay machinery for mamba2 without any further configuration.
TEST(RecurrentCommitPolicyTests, FactoryBuildsReplayPolicyForMamba2)
{
    cudaStream_t stream{};
    Config const cfg = makeMambaConfig();
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);

    auto policy = makeRecurrentCommitPolicy(StateKind::kMamba2, cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(policy->usesReplay()) << "mamba2 commits by replaying the SSD scan, not by scatter";
    EXPECT_TRUE(policy->hasIntermediateRecurrentStates());
    EXPECT_TRUE(policy->hasIntermediateConvStates());
    EXPECT_EQ(policy->numLayers(), cfg.numRecurrentLayers);
}

// Given a gdn mixer, when the factory builds its commit policy, then the policy commits the accepted
// endpoint by scatter rather than replay and reports both the recurrent snapshot and the conv
// snapshot as allocated. gdn and mamba2 share the factory entry point, so the `usesReplay` flag is
// the only thing that separates the two commit paths and must come back false here.
TEST(RecurrentCommitPolicyTests, FactoryBuildsScatterPolicyForGdn)
{
    cudaStream_t stream{};
    Config const cfg = makeGdnConfig();
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);

    auto policy = makeRecurrentCommitPolicy(StateKind::kGdn, cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_FALSE(policy->usesReplay()) << "gdn commits the accepted endpoint by scatter, not replay";
    EXPECT_TRUE(policy->hasIntermediateRecurrentStates());
    EXPECT_TRUE(policy->hasIntermediateConvStates());
    EXPECT_EQ(policy->numLayers(), cfg.numRecurrentLayers);
}

// Given a kind that is a pure convolution lineage and carries no recurrent commit, when the factory
// is asked for a policy, then it throws instead of fabricating one. kConv and kNgramConv name states
// a layer may hold, but only gdn and mamba2 evolve a recurrent state the verify pass commits, so the
// factory must reject the other kinds rather than return a policy that commits nothing.
TEST(RecurrentCommitPolicyTests, FactoryRejectsNonRecurrentKinds)
{
    cudaStream_t stream{};
    Config const cfg = makeGdnConfig();
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);

    EXPECT_THROW(makeRecurrentCommitPolicy(StateKind::kConv, cfg, liveRecurrent, liveConv, stream), std::runtime_error);
    EXPECT_THROW(
        makeRecurrentCommitPolicy(StateKind::kNgramConv, cfg, liveRecurrent, liveConv, stream), std::runtime_error);
}

// Given a mamba2 policy allocated for the full [maxBatch, maxIntermediateSeqLen] verify window, when
// it is reshaped to a smaller active window, then every replay stash and the conv snapshot report the
// active [batch, seqLen] with their own trailing extents intact. Commit kernels read these side
// buffers by stride, so a reshape that left one buffer at the allocation shape, or applied the wrong
// trailing extent, would address the wrong token and silently corrupt the committed state.
TEST(RecurrentCommitPolicyTests, Mamba2ReshapesReplayAndConvSideBuffersToActiveWindow)
{
    cudaStream_t stream{};
    Config const cfg = makeMambaConfig();
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);
    Mamba2CommitPolicy policy(cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t const activeBatch = 1;
    int32_t const seqLen = 4;
    policy.reshapeIntermediate(activeBatch, seqLen);

    auto const da = policy.replayDaState(0).getShape();
    ASSERT_EQ(da.getNumDims(), 3);
    EXPECT_EQ(da[0], activeBatch);
    EXPECT_EQ(da[1], seqLen);
    EXPECT_EQ(da[2], cfg.recurrentStateNumHeads);

    auto const u = policy.replayUState(0).getShape();
    ASSERT_EQ(u.getNumDims(), 4);
    EXPECT_EQ(u[0], activeBatch);
    EXPECT_EQ(u[1], seqLen);
    EXPECT_EQ(u[2], cfg.recurrentStateNumHeads);
    EXPECT_EQ(u[3], cfg.recurrentStateHeadDim);

    auto const b = policy.replayBState(0).getShape();
    ASSERT_EQ(b.getNumDims(), 4);
    EXPECT_EQ(b[0], activeBatch);
    EXPECT_EQ(b[1], seqLen);
    EXPECT_EQ(b[2], cfg.recurrentStateNumGroups);
    EXPECT_EQ(b[3], cfg.recurrentStateSize);

    auto const dt = policy.replayDtState(0).getShape();
    ASSERT_EQ(dt.getNumDims(), 3);
    EXPECT_EQ(dt[1], seqLen);
    EXPECT_EQ(dt[2], cfg.recurrentStateNumHeads);

    auto const conv = policy.intermediateConvState(0).getShape();
    ASSERT_EQ(conv.getNumDims(), 4);
    EXPECT_EQ(conv[0], activeBatch);
    EXPECT_EQ(conv[1], seqLen);
    EXPECT_EQ(conv[2], cfg.convDim);
    EXPECT_EQ(conv[3], cfg.convKernel);
}

// Given a gdn policy allocated for the full verify window, when it is reshaped to a smaller active
// window, then the full-state recurrent snapshot and the conv snapshot report the active
// [batch, seqLen] with their head/dim/state extents intact. The scatter commit indexes these
// snapshots by the active stride, so a reshape that missed the recurrent snapshot would scatter from
// the wrong token row.
TEST(RecurrentCommitPolicyTests, GdnReshapesRecurrentAndConvSideBuffersToActiveWindow)
{
    cudaStream_t stream{};
    Config const cfg = makeGdnConfig();
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);
    GdnConvCommitPolicy policy(cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t const activeBatch = 1;
    int32_t const seqLen = 4;
    policy.reshapeIntermediate(activeBatch, seqLen);

    auto const rec = policy.intermediateRecurrentState(0).getShape();
    ASSERT_EQ(rec.getNumDims(), 5);
    EXPECT_EQ(rec[0], activeBatch);
    EXPECT_EQ(rec[1], seqLen);
    EXPECT_EQ(rec[2], cfg.recurrentStateNumHeads);
    EXPECT_EQ(rec[3], cfg.recurrentStateHeadDim);
    EXPECT_EQ(rec[4], cfg.recurrentStateSize);

    auto const conv = policy.intermediateConvState(0).getShape();
    ASSERT_EQ(conv.getNumDims(), 4);
    EXPECT_EQ(conv[0], activeBatch);
    EXPECT_EQ(conv[1], seqLen);
    EXPECT_EQ(conv[2], cfg.convDim);
    EXPECT_EQ(conv[3], cfg.convKernel);
}

// Given a verify-token capacity of zero, when a policy is built, then it allocates no side buffers
// and both commit entry points are no-ops. A zero capacity is the signal that speculative decoding is
// off for this deployment; the policy still exists and still reports its layer count, but every
// commit must return without touching the persistent state.
TEST(RecurrentCommitPolicyTests, ZeroVerifyCapacityAllocatesNoSideBuffersAndCommitsAreNoOps)
{
    cudaStream_t stream{};
    Config cfg = makeGdnConfig();
    cfg.maxIntermediateSeqLen = 0;
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);

    auto policy = makeRecurrentCommitPolicy(StateKind::kGdn, cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_FALSE(policy->hasIntermediateRecurrentStates());
    EXPECT_FALSE(policy->hasIntermediateConvStates());
    EXPECT_EQ(policy->numLayers(), cfg.numRecurrentLayers) << "numLayers reports the configured layers even when off";

    Tensor acceptedNodeIds({1, 1}, DeviceType::kGPU, DataType::kINT32, "acceptedNodeIds");
    Tensor acceptLengths({1}, DeviceType::kGPU, DataType::kINT32, "acceptLengths");
    Tensor stateIndices({1}, DeviceType::kGPU, DataType::kINT32, "stateIndices");
    EXPECT_NO_THROW(policy->commitLinear(acceptLengths, stateIndices, stream));
    EXPECT_NO_THROW(policy->commitTree(acceptedNodeIds, acceptLengths, stateIndices, /*verifySize=*/8, stream));
}

// Given a gdn policy whose recurrent head dim fails the replay guard (dk != 128), when commitTree runs
// with the verify tree size at the former chunk-form bound and just past it, then both sizes raise the
// same replay-path guard. gdn tree-commit is a single replay path -- it advances the persistent state
// along the accepted path from the per-node stash regardless of tree size -- so there is no size-keyed
// branch that could route one tree to a scatter kernel and another to replay. The verifySize argument no
// longer selects a path; the proof is that two sizes straddling the old bound take the identical path and
// fail on the identical guard.
TEST(RecurrentCommitPolicyTests, GdnCommitTreeAlwaysReplaysRegardlessOfVerifySize)
{
    cudaStream_t stream{};
    Config cfg = makeGdnConfig();
    cfg.recurrentStateType = DataType::kFLOAT; // clear the FP32 gate that precedes the replay path
    cfg.maxBatchSize = 1;
    cfg.maxIntermediateSeqLen = 8;
    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);
    GdnConvCommitPolicy policy(cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    policy.reshapeIntermediate(/*activeBatch=*/1, /*seqLen=*/8);

    Tensor acceptedNodeIds({1, 2}, DeviceType::kGPU, DataType::kINT32, "acceptedNodeIds");
    Tensor acceptLengths({1}, DeviceType::kGPU, DataType::kINT32, "acceptLengths");
    Tensor stateIndices({1}, DeviceType::kGPU, DataType::kINT32, "stateIndices");

    std::string const atBound
        = commitTreeThrowMessage(policy, acceptedNodeIds, acceptLengths, stateIndices, /*verifySize=*/64, stream);
    EXPECT_NE(atBound.find("dk == dv == 128"), std::string::npos)
        << "a tree of 64 nodes must take the replay path and fail its dk/dv guard; got: " << atBound;

    std::string const pastBound
        = commitTreeThrowMessage(policy, acceptedNodeIds, acceptLengths, stateIndices, /*verifySize=*/65, stream);
    EXPECT_NE(pastBound.find("dk == dv == 128"), std::string::npos)
        << "a tree of 65 nodes must take the same replay path, not a size-keyed scatter branch; got: " << pastBound;
}

// Given a mamba2 mixer carrying a convolution, when its spec-verify output bindings are declared into
// a registry under a local recurrent-layer index, then exactly the four FP32 replay stashes and the
// conv snapshot are registered as outputs keyed by that same index, each shaped [seqLen, ...] with
// seqLen left symbolic. The builder profiles and runtime tensor map key these names by the local
// recurrent-layer index, so the declaration must use the index it is handed verbatim -- in an
// interleaved hybrid the absolute layer index would name tensors the runtime never binds. The engine
// learns the spec-verify output set from the mixer kind alone, so the kind must declare the replay
// stashes -- and only those -- so the commit policy of the same kind finds the buffers it reads.
TEST(RecurrentCommitPolicyTests, DeclaresMamba2SpecVerifyOutputSetAndShapes)
{
    Config const cfg = makeMambaConfig();
    int32_t const recurrentLayerIdx = 5;
    TensorRegistry reg;
    declareRecurrentSpecVerifyBindings(StateKind::kMamba2, cfg, recurrentLayerIdx, reg);

    auto const* u = findSpec(reg, binding_names::formatReplayUStateName(recurrentLayerIdx));
    ASSERT_NE(u, nullptr) << "the replay-u stash must be declared for mamba2";
    EXPECT_EQ(u->io, TensorIO::kOutput);
    EXPECT_EQ(u->dtype, DataType::kFLOAT);
    ASSERT_EQ(u->shape.size(), 3U);
    EXPECT_TRUE(u->shape[0].isSymbolic());
    EXPECT_EQ(u->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(u->shape[1].value, cfg.recurrentStateNumHeads);
    EXPECT_EQ(u->shape[2].value, cfg.recurrentStateHeadDim);

    auto const* b = findSpec(reg, binding_names::formatReplayBStateName(recurrentLayerIdx));
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->shape.size(), 3U);
    EXPECT_EQ(b->shape[1].value, cfg.recurrentStateNumGroups)
        << "the replay-B stash is keyed by the B/C group count, not the head count";
    EXPECT_EQ(b->shape[2].value, cfg.recurrentStateSize);

    EXPECT_TRUE(reg.contains(binding_names::formatReplayDaStateName(recurrentLayerIdx)));
    EXPECT_TRUE(reg.contains(binding_names::formatReplayDtStateName(recurrentLayerIdx)));
    EXPECT_TRUE(reg.contains(binding_names::formatIntermediateConvStateName(recurrentLayerIdx)));
    EXPECT_FALSE(reg.contains(binding_names::formatIntermediateRecurrentStateName(recurrentLayerIdx)))
        << "mamba2 reconstructs the recurrent state by replay and declares no full-state snapshot";
}

// Given a gdn mixer carrying a convolution, when its spec-verify output bindings are declared, then a
// single full-state recurrent snapshot typed by the recurrent state type and the conv snapshot are
// registered, and none of the mamba2 replay stashes appear. gdn snapshots the whole recurrent state
// per token and scatters the accepted endpoint, so its declared output set must be the snapshot pair
// and nothing from the replay machinery.
TEST(RecurrentCommitPolicyTests, DeclaresGdnSpecVerifyOutputSetAndShapes)
{
    Config cfg = makeGdnConfig();
    cfg.recurrentStateType = DataType::kFLOAT;
    cfg.convStateType = DataType::kHALF;
    int32_t const recurrentLayerIdx = 5;
    TensorRegistry reg;
    declareRecurrentSpecVerifyBindings(StateKind::kGdn, cfg, recurrentLayerIdx, reg);

    auto const* rec = findSpec(reg, binding_names::formatIntermediateRecurrentStateName(recurrentLayerIdx));
    ASSERT_NE(rec, nullptr) << "gdn must declare the full-state recurrent snapshot";
    EXPECT_EQ(rec->io, TensorIO::kOutput);
    EXPECT_EQ(rec->dtype, cfg.recurrentStateType) << "the snapshot is typed by the recurrent state type";
    ASSERT_EQ(rec->shape.size(), 4U);
    EXPECT_TRUE(rec->shape[0].isSymbolic());
    EXPECT_EQ(rec->shape[0].symbol, &InferenceDims::seqLen);
    EXPECT_EQ(rec->shape[1].value, cfg.recurrentStateNumHeads);
    EXPECT_EQ(rec->shape[2].value, cfg.recurrentStateHeadDim);
    EXPECT_EQ(rec->shape[3].value, cfg.recurrentStateSize);

    auto const* conv = findSpec(reg, binding_names::formatIntermediateConvStateName(recurrentLayerIdx));
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->dtype, cfg.convStateType);

    EXPECT_FALSE(reg.contains(binding_names::formatReplayUStateName(recurrentLayerIdx)))
        << "gdn declares no replay stash";
    EXPECT_FALSE(reg.contains(binding_names::formatReplayBStateName(recurrentLayerIdx)));
}

// Given a kind that is a pure convolution lineage, when its spec-verify bindings are declared, then
// the call throws. Only gdn and mamba2 produce spec-verify recurrent outputs, so the declaration hook
// must reject the other kinds in lockstep with the factory rather than register an empty set that the
// engine would then expect the model to emit.
TEST(RecurrentCommitPolicyTests, DeclareRejectsNonRecurrentKinds)
{
    Config const cfg = makeGdnConfig();
    TensorRegistry reg;
    EXPECT_THROW(
        declareRecurrentSpecVerifyBindings(StateKind::kConv, cfg, /*recurrentLayerIdx=*/0, reg), std::runtime_error);
    EXPECT_THROW(declareRecurrentSpecVerifyBindings(StateKind::kNgramConv, cfg, /*recurrentLayerIdx=*/0, reg),
        std::runtime_error);
}

// Given a gdn mixer whose layer carries no convolution, when its spec-verify bindings are declared,
// then only the recurrent snapshot is registered and no conv snapshot appears. The conv snapshot is
// declared off the presence of a convolution, not off the mixer kind, so a layer with no conv width
// must not declare a conv output the engine has nothing to fill.
TEST(RecurrentCommitPolicyTests, DeclareOmitsConvSnapshotWhenLayerCarriesNoConvolution)
{
    Config cfg = makeGdnConfig();
    cfg.convDim = 0;
    cfg.convKernel = 0;
    int32_t const recurrentLayerIdx = 3;
    TensorRegistry reg;
    declareRecurrentSpecVerifyBindings(StateKind::kGdn, cfg, recurrentLayerIdx, reg);

    EXPECT_TRUE(reg.contains(binding_names::formatIntermediateRecurrentStateName(recurrentLayerIdx)));
    EXPECT_FALSE(reg.contains(binding_names::formatIntermediateConvStateName(recurrentLayerIdx)))
        << "a layer with no convolution width must declare no conv snapshot";
}

namespace
{

//! The deployed Qwen3.8-27B GDN geometry (16 key heads, 48 value heads, 128 x 128 FP32 state) at the largest verify
//! tree the chunk-form verify takes.
Config makeChunkVerifyGdnConfig()
{
    Config cfg{};
    cfg.numRecurrentLayers = 1;
    cfg.maxBatchSize = 8;
    cfg.recurrentStateNumHeads = 48;
    cfg.recurrentStateHeadDim = 128;
    cfg.recurrentStateSize = 128;
    cfg.convDim = (2 * 16 + 48) * 128;
    cfg.convKernel = 4;
    cfg.maxIntermediateSeqLen = kernel::kGDN_TREE_CHUNK_MAX_NODES;
    cfg.recurrentStateType = DataType::kFLOAT;
    cfg.convStateType = DataType::kHALF;
    return cfg;
}

//! The verify stash must hold maxBatch rows of the largest tree and only round each verify node up to whole FP32
//! words; the active shape's rows must fit after a reshape.
void expectStashSizedToVerifyRows(Tensor& stash, Config const& cfg, std::function<void(int32_t, int32_t)> reshape)
{
    size_t const usedBytes = static_cast<size_t>(cfg.maxBatchSize)
        * kernel::gdnTreeVerifyRowBytes(16, cfg.recurrentStateNumHeads, cfg.maxIntermediateSeqLen);
    size_t const roundingBytes = static_cast<size_t>(cfg.maxBatchSize) * cfg.maxIntermediateSeqLen * sizeof(float);
    auto const capacity = static_cast<size_t>(stash.getMemoryCapacity());
    EXPECT_GE(capacity, usedBytes);
    EXPECT_LT(capacity, usedBytes + roundingBytes);
    int32_t const activeBatch = 3;
    int32_t const verifyNodes = 16;
    reshape(activeBatch, verifyNodes);
    EXPECT_EQ(stash.getShape()[0], activeBatch);
    EXPECT_EQ(stash.getShape()[1], verifyNodes);
    EXPECT_GE(capacity,
        static_cast<size_t>(activeBatch) * kernel::gdnTreeVerifyRowBytes(16, cfg.recurrentStateNumHeads, verifyNodes));
}

} // namespace

// Given the deployed GDN geometry at the largest verify tree and batch
// When the cache manager and the GDN commit policy allocate their chunk-verify stash
// Then each layer's buffer holds exactly maxBatch rows of gdnTreeVerifyRowBytes plus per-node word rounding, instead
// of a full recurrent state per verify node, and still fits the rows of a smaller active shape after a reshape
TEST(RecurrentCommitPolicyTests, GdnChunkVerifyStashIsSizedToTheVerifyRows)
{
    cudaStream_t stream{};
    Config const cfg = makeChunkVerifyGdnConfig();

    MambaCacheManager::Config mgrCfg{};
    mgrCfg.numRecurrentLayers = cfg.numRecurrentLayers;
    mgrCfg.maxBatchSize = cfg.maxBatchSize;
    mgrCfg.recurrentStateNumHeads = cfg.recurrentStateNumHeads;
    mgrCfg.recurrentStateHeadDim = cfg.recurrentStateHeadDim;
    mgrCfg.recurrentStateSize = cfg.recurrentStateSize;
    mgrCfg.convDim = cfg.convDim;
    mgrCfg.convKernel = cfg.convKernel;
    mgrCfg.maxIntermediateSeqLen = cfg.maxIntermediateSeqLen;
    mgrCfg.recurrentStateType = cfg.recurrentStateType;
    mgrCfg.convStateType = cfg.convStateType;
    MambaCacheManager mgr(mgrCfg, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectStashSizedToVerifyRows(mgr.getIntermediateRecurrentState(0), cfg,
        [&](int32_t batch, int32_t nodes) { mgr.reshapeIntermediateStates(batch, nodes); });

    std::vector<Tensor> liveRecurrent = makeSlabs(cfg.numRecurrentLayers);
    std::vector<Tensor> liveConv = makeSlabs(cfg.numRecurrentLayers);
    GdnConvCommitPolicy policy(cfg, liveRecurrent, liveConv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectStashSizedToVerifyRows(policy.intermediateRecurrentState(0), cfg,
        [&](int32_t batch, int32_t nodes) { policy.reshapeIntermediate(batch, nodes); });
}
