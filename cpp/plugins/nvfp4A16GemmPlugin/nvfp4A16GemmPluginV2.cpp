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

#include "nvfp4A16GemmPluginV2.h"
#include "nvfp4A16BlackwellDispatchPolicy.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.h"
#include "kernels/nvfp4A16BlackwellSupport.h"
#include "profiling/nvtx_wrapper.h"

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{

constexpr char const* kPluginName{"Nvfp4A16GemmPlugin"};
constexpr char const* kPluginVersion{"2"};

constexpr int32_t kInActivation{0};
constexpr int32_t kInQWeights{1};
constexpr int32_t kInBlockScales{2};
constexpr int32_t kInGlobalScale{3};
constexpr int32_t kOutOutput{4};
constexpr int32_t kNbInputs{4};

constexpr int32_t kLayoutBlackwellN128K64V1{1};
constexpr int32_t kBackendDefault{0};
constexpr int32_t kBackendGemv{1};
constexpr int32_t kBackendTcgen05{2};
constexpr char const* kGemvJitBundleField{"gemv_jit_bundle"};

static_assert(kBackendDefault == static_cast<int32_t>(Nvfp4A16BlackwellDispatchBackend::kDefault));
static_assert(kBackendGemv == static_cast<int32_t>(Nvfp4A16BlackwellDispatchBackend::kGemv));
static_assert(kBackendTcgen05 == static_cast<int32_t>(Nvfp4A16BlackwellDispatchBackend::kTcgen05));

constexpr int32_t kFieldGemmN{0};
constexpr int32_t kFieldGemmK{1};
constexpr int32_t kFieldMaxM{2};
constexpr int32_t kFieldLayout{3};
constexpr int32_t kFieldBackend{4};
constexpr int32_t kNbFields{5};

bool isIoType(DataType type) noexcept
{
    return type == DataType::kHALF || type == DataType::kBF16;
}

int32_t getTokenCount(Dims const& dims) noexcept
{
    if (dims.nbDims != 3 || dims.d[0] <= 0 || dims.d[1] <= 0
        || dims.d[0] > std::numeric_limits<int32_t>::max() / dims.d[1])
    {
        return 0;
    }
    int64_t const tokens = static_cast<int64_t>(dims.d[0]) * dims.d[1];
    return tokens <= std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(tokens) : 0;
}

Nvfp4A16BlackwellGemvDataType toGemvType(DataType type)
{
    if (type == DataType::kHALF)
    {
        return Nvfp4A16BlackwellGemvDataType::kHALF;
    }
    if (type == DataType::kBF16)
    {
        return Nvfp4A16BlackwellGemvDataType::kBF16;
    }
    throw std::invalid_argument("Nvfp4A16GemmPluginV2: activation must be FP16 or BF16");
}

Nvfp4A16BlackwellGemvJitKey makeGemvJitKey(int32_t smVersion, int32_t n, int32_t k, DataType type)
{
    return {
        smVersion, kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI, n, k, toGemvType(type), kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI};
}

kernels::Nvfp4A16BlackwellDtype toTcgenType(DataType type)
{
    if (type == DataType::kHALF)
    {
        return kernels::Nvfp4A16BlackwellDtype::kFp16;
    }
    if (type == DataType::kBF16)
    {
        return kernels::Nvfp4A16BlackwellDtype::kBf16;
    }
    throw std::invalid_argument("Nvfp4A16GemmPluginV2: activation must be FP16 or BF16");
}

Nvfp4A16BlackwellDispatchDtype toDispatchType(DataType type) noexcept
{
    if (type == DataType::kHALF)
    {
        return Nvfp4A16BlackwellDispatchDtype::kFp16;
    }
    if (type == DataType::kBF16)
    {
        return Nvfp4A16BlackwellDispatchDtype::kBf16;
    }
    return static_cast<Nvfp4A16BlackwellDispatchDtype>(-1);
}

Nvfp4A16BlackwellDispatch getDispatch(int32_t backend, int32_t m, int32_t n, int32_t k, DataType type) noexcept
{
    return getNvfp4A16BlackwellDispatch(
        static_cast<Nvfp4A16BlackwellDispatchBackend>(backend), m, n, k, toDispatchType(type));
}

bool useGemv(int32_t backend, int32_t m, int32_t n, int32_t k, DataType type) noexcept
{
    auto const dispatch = getDispatch(backend, m, n, k, type);
    if (dispatch.kernel != Nvfp4A16BlackwellKernel::kGemv)
    {
        return false;
    }
    return isNvfp4A16BlackwellGemvJitSupported(
        makeGemvJitKey(nvfp4_a16_blackwell::kTargetSm, n, k, type), m, dispatch.splitK);
}

bool isAligned(void const* pointer, size_t alignment) noexcept
{
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

} // namespace

PluginFieldCollection Nvfp4A16GemmPluginV2Creator::mFieldCollection{};
std::vector<PluginField> Nvfp4A16GemmPluginV2Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4A16GemmPluginV2Creator);

Nvfp4A16GemmPluginV2::Nvfp4A16GemmPluginV2(
    std::string const& name, int32_t gemmN, int32_t gemmK, int32_t maxM, int32_t layout, int32_t backend)
    : mLayerName(name)
    , mGemmN(gemmN)
    , mGemmK(gemmK)
    , mMaxM(maxM)
    , mProfileDerivedMaxM(maxM == 0)
    , mLayout(layout)
    , mBackend(backend)
{
    validateAttributes();
}

Nvfp4A16GemmPluginV2::Nvfp4A16GemmPluginV2(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: plugin field collection must not be empty");
    }

    std::array<bool, kNbFields> fieldsSeen{};
    bool gemvJitBundleSeen{false};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t index) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4A16GemmPluginV2: attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[index])
        {
            throw std::invalid_argument("Nvfp4A16GemmPluginV2: duplicate plugin attribute");
        }
        fieldsSeen[index] = true;
        return *static_cast<int32_t const*>(field.data);
    };

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("Nvfp4A16GemmPluginV2: plugin attribute name must not be null");
        }
        std::string const fieldName(field.name);
        if (fieldName == "gemm_n")
        {
            mGemmN = readIntField(field, kFieldGemmN);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = readIntField(field, kFieldGemmK);
        }
        else if (fieldName == "max_m")
        {
            mMaxM = readIntField(field, kFieldMaxM);
        }
        else if (fieldName == "layout")
        {
            mLayout = readIntField(field, kFieldLayout);
        }
        else if (fieldName == "backend")
        {
            mBackend = readIntField(field, kFieldBackend);
        }
        else if (fieldName == kGemvJitBundleField)
        {
            if (gemvJitBundleSeen)
            {
                throw std::invalid_argument("Nvfp4A16GemmPluginV2: duplicate gemv_jit_bundle attribute");
            }
            if (field.data == nullptr || field.type != PluginFieldType::kCHAR || field.length <= 0)
            {
                throw std::invalid_argument("Nvfp4A16GemmPluginV2: gemv_jit_bundle must be a nonempty CHAR field");
            }
            gemvJitBundleSeen = true;
            auto const* bytes = static_cast<uint8_t const*>(field.data);
            mGemvJitBundle.assign(bytes, bytes + field.length);
        }
        else
        {
            throw std::invalid_argument("Nvfp4A16GemmPluginV2: unknown plugin attribute " + fieldName);
        }
    }

    if (!fieldsSeen[kFieldGemmN] || !fieldsSeen[kFieldGemmK] || !fieldsSeen[kFieldLayout] || !fieldsSeen[kFieldBackend])
    {
        throw std::invalid_argument(
            "Nvfp4A16GemmPluginV2: gemm_n, gemm_k, layout, and backend attributes are required");
    }
    mProfileDerivedMaxM = mMaxM == 0;
    validateAttributes();
    if (!mGemvJitBundle.empty())
    {
        mGemvJitKernel = deserializeNvfp4A16BlackwellGemvJitKernel(mGemvJitBundle.data(), mGemvJitBundle.size());
    }
}

void Nvfp4A16GemmPluginV2::validateAttributes() const
{
    if (mLayout != kLayoutBlackwellN128K64V1)
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: layout must be BLACKWELL_N128_K64_V1 (1)");
    }
    if (!nvfp4_a16_blackwell::isSupportedProblemShape(mGemmN, mGemmK))
    {
        throw std::invalid_argument(
            "Nvfp4A16GemmPluginV2: N must be a positive int32 multiple of 128 and K must "
            "be a positive int32 multiple of 64");
    }
    if (mMaxM < 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: max_m must be nonnegative (0 means profile-derived)");
    }
    if (mMaxM > 0 && !nvfp4_a16_blackwell::isTmaRepresentableProblem(mMaxM, mGemmN, mGemmK))
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: max_m/N/K cannot be represented by the TMA layouts");
    }
    if (mBackend < kBackendDefault || mBackend > kBackendTcgen05)
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: backend must be auto (0), gemv (1), or tcgen05 (2)");
    }
}

Nvfp4A16BlackwellGemvJitKey Nvfp4A16GemmPluginV2::getGemvJitKey(DataType type) const
{
    // configurePlugin() and runtime deserialization validate the actual GPU.
    // Keep enqueue/onShapeChange free of CUDA runtime device queries so graph
    // capture only observes the preloaded module and kernel launches.
    return makeGemvJitKey(nvfp4_a16_blackwell::kTargetSm, mGemmN, mGemmK, type);
}

void Nvfp4A16GemmPluginV2::compileGemvJitBundle(DataType type)
{
    Nvfp4A16BlackwellGemvJitKey const key = getGemvJitKey(type);
    if (!canCompileNvfp4A16BlackwellGemvJitKernel(key))
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: JIT GEMV does not support the configured SM/shape/dtype");
    }
    if (!mGemvJitBundle.empty())
    {
        if (!(mGemvJitKernel.key == key))
        {
            throw std::invalid_argument(
                "Nvfp4A16GemmPluginV2: one plugin instance cannot use multiple GEMV JIT semantic keys");
        }
        return;
    }
    mGemvJitKernel = compileNvfp4A16BlackwellGemvJitKernel(key);
    mGemvJitBundle = serializeNvfp4A16BlackwellGemvJitKernel(mGemvJitKernel);
}

bool Nvfp4A16GemmPluginV2::hasSerializedGemvJitBundle() const noexcept
{
    return !mGemvJitBundle.empty();
}

void Nvfp4A16GemmPluginV2::loadSerializedGemvJitBundle()
{
    if (mGemvJitBundle.empty() || mGemvJitKernel.cubin.empty())
    {
        throw std::invalid_argument("Nvfp4A16GemmPluginV2: runtime GEMV requires a serialized JIT bundle");
    }
    Nvfp4A16BlackwellGemvJitKey const& key = mGemvJitKernel.key;
    if (key.sm != getSMVersion() || key.layout != static_cast<uint32_t>(mLayout) || key.n != mGemmN || key.k != mGemmK
        || key.sourceAbi != kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI)
    {
        throw std::invalid_argument(
            "Nvfp4A16GemmPluginV2: serialized GEMV JIT key does not match the runtime SM/layout/shape");
    }
    mGemvJitRunner.load(mGemvJitKernel);
}

IPluginCapability* Nvfp4A16GemmPluginV2::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* Nvfp4A16GemmPluginV2::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<Nvfp4A16GemmPluginV2>(mLayerName, mGemmN, mGemmK, mMaxM, mLayout, mBackend);
        plugin->mProfileDerivedMaxM = mProfileDerivedMaxM;
        plugin->mConfiguredNeedsGemv = mConfiguredNeedsGemv;
        plugin->mGemvJitKernel = mGemvJitKernel;
        plugin->mGemvJitBundle = mGemvJitBundle;
        if (!plugin->mGemvJitBundle.empty())
        {
            // configurePlugin() compiles and serializes the GEMV bundle but
            // deliberately does not load it. attachToContext() may therefore
            // clone either a build-phase plugin with an unloaded runner or a
            // runtime plugin whose runner is already loaded. In both cases,
            // load through the destination context-keyed module registry
            // instead of keying this decision on the source runner state.
            plugin->loadSerializedGemvJitBundle();
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to clone Nvfp4A16GemmPluginV2: %s", error.what());
        return nullptr;
    }
}

char const* Nvfp4A16GemmPluginV2::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16GemmPluginV2::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Nvfp4A16GemmPluginV2::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16GemmPluginV2::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t Nvfp4A16GemmPluginV2::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4A16GemmPluginV2::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNbInputs
        || !isIoType(inputTypes[kInActivation]))
    {
        return -1;
    }
    outputTypes[0] = inputTypes[kInActivation];
    return 0;
}

int32_t Nvfp4A16GemmPluginV2::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
    {
        return -1;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[kInActivation].d[0];
    outputs[0].d[1] = inputs[kInActivation].d[1];
    outputs[0].d[2] = exprBuilder.constant(mGemmN);
    return 0;
}

bool Nvfp4A16GemmPluginV2::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    switch (pos)
    {
    case kInActivation: return isIoType(desc.type) && desc.dims.nbDims == 3 && desc.dims.d[2] == mGemmK;
    case kInQWeights:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 4 && desc.dims.d[0] == mGemmN / 128
            && desc.dims.d[1] == mGemmK / 64 && desc.dims.d[2] == 128 && desc.dims.d[3] == 32;
    case kInBlockScales:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 4 && desc.dims.d[0] == mGemmN / 128
            && desc.dims.d[1] == mGemmK / 64 && desc.dims.d[2] == 128 && desc.dims.d[3] == 4;
    case kInGlobalScale: return desc.type == DataType::kFLOAT && desc.dims.nbDims == 1 && desc.dims.d[0] == 1;
    case kOutOutput: return isIoType(desc.type) && desc.dims.nbDims == 3 && desc.dims.d[2] == mGemmN;
    default: return false;
    }
}

bool Nvfp4A16GemmPluginV2::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNbInputs || nbOutputs != 1 || pos < 0 || pos > kOutOutput
        || !validateTensorDesc(pos, inOut[pos].desc))
    {
        return false;
    }
    return pos != kOutOutput || inOut[pos].desc.type == inOut[kInActivation].desc.type;
}

int32_t Nvfp4A16GemmPluginV2::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
        {
            return -1;
        }
        for (int32_t pos = 0; pos < kNbInputs; ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("Nvfp4A16GemmPluginV2: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, out[0].desc) || out[0].desc.type != in[kInActivation].desc.type)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: output shape and dtype must follow the activation");
            return -1;
        }

        int32_t const smVersion = getSMVersion();
        if (smVersion != nvfp4_a16_blackwell::kTargetSm)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: BLACKWELL_N128_K64_V1 requires SM110, got SM%d", smVersion);
            return -1;
        }

        DataType const type = in[kInActivation].desc.type;
        std::array<Dims, 3> const profileDims{in[kInActivation].min, in[kInActivation].opt, in[kInActivation].max};
        if (mBackend == kBackendGemv
            && (getTokenCount(profileDims[0]) != getTokenCount(profileDims[1])
                || getTokenCount(profileDims[1]) != getTokenCount(profileDims[2])))
        {
            LOG_ERROR(
                "Nvfp4A16GemmPluginV2: forced GEMV requires a static M profile because only M={1,2,4,8,16} "
                "specializations exist");
            return -1;
        }
        for (Dims const& dims : profileDims)
        {
            int32_t const m = getTokenCount(dims);
            if (m <= 0 || dims.d[2] != mGemmK || !nvfp4_a16_blackwell::isTmaRepresentableProblem(m, mGemmN, mGemmK))
            {
                LOG_ERROR(
                    "Nvfp4A16GemmPluginV2: profile M/N/K must be positive, match gemm_k, and be TMA "
                    "representable");
                return -1;
            }
        }

        int32_t const profileMinM = getTokenCount(in[kInActivation].min);
        int32_t const profileOptM = getTokenCount(in[kInActivation].opt);
        int32_t const profileMaxM = getTokenCount(in[kInActivation].max);
        if (profileMinM > profileOptM || profileOptM > profileMaxM)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: profile M bounds must satisfy min <= opt <= max");
            return -1;
        }
        if (!mProfileDerivedMaxM && profileMaxM > mMaxM)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: profile maximum M=%d exceeds max_m=%d", profileMaxM, mMaxM);
            return -1;
        }

        // Backend 0 is the serialized compatibility spelling for the fixed
        // production policy: M=1 uses GEMV and every M>1 uses TCGen05.
        bool const profileNeedsGemv = mBackend == kBackendGemv || (mBackend == kBackendDefault && profileMinM == 1);
        if (profileNeedsGemv)
        {
            int32_t const gemvM = mBackend == kBackendGemv ? profileMinM : 1;
            if (!useGemv(mBackend, gemvM, mGemmN, mGemmK, type))
            {
                LOG_ERROR("Nvfp4A16GemmPluginV2: GEMV has no M=%d N=%d K=%d dtype=%d variant", gemvM, mGemmN, mGemmK,
                    static_cast<int32_t>(type));
                return -1;
            }
        }

        // Validate at most one representative of each reachable TCGen token
        // tile. This is constant work even when max_m is close to INT32_MAX.
        int32_t const tcgenMinM = std::max(profileMinM, mBackend == kBackendDefault ? 2 : 1);
        constexpr std::array<int32_t, 6> kTileLowerBounds{1, 9, 17, 33, 65, 129};
        constexpr std::array<int32_t, 6> kTileUpperBounds{8, 16, 32, 64, 128, std::numeric_limits<int32_t>::max()};
        for (size_t tileClass = 0; mBackend != kBackendGemv && tileClass < kTileLowerBounds.size(); ++tileClass)
        {
            int32_t const m = std::max(tcgenMinM, kTileLowerBounds[tileClass]);
            if (m > profileMaxM || m > kTileUpperBounds[tileClass])
            {
                continue;
            }
            if (!kernels::Nvfp4A16BlackwellGemmRunner::isSupported(smVersion, toTcgenType(type), m, mGemmN, mGemmK))
            {
                auto const tile = kernels::Nvfp4A16BlackwellGemmRunner::selectTokenTile(m);
                LOG_ERROR(
                    "Nvfp4A16GemmPluginV2: no TCGen05 AOT variant for token tile %d (M=%d) N=%d K=%d "
                    "dtype=%d",
                    static_cast<int32_t>(tile), m, mGemmN, mGemmK, static_cast<int32_t>(type));
                return -1;
            }
        }

        if (profileNeedsGemv)
        {
            compileGemvJitBundle(type);
            mConfiguredNeedsGemv = true;
        }

        if (mProfileDerivedMaxM)
        {
            // configurePlugin may be called once per optimization profile.
            // Preserve the original max_m=0 intent while accumulating a
            // concrete cap that can be serialized for runtime contexts.
            mMaxM = std::max(mMaxM, profileMaxM);
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Nvfp4A16GemmPluginV2 configurePlugin failed: %s", error.what());
        return -1;
    }
}

size_t Nvfp4A16GemmPluginV2::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) outputs;
    if (inputs == nullptr || nbInputs != kNbInputs || nbOutputs != 1 || !isIoType(inputs[kInActivation].desc.type))
    {
        return 0;
    }

    int32_t const profileMaxM = getTokenCount(inputs[kInActivation].max);
    int32_t const workspaceMaxM = std::min(profileMaxM, 16);
    Nvfp4A16BlackwellGemvJitKey const gemvKey
        = makeGemvJitKey(nvfp4_a16_blackwell::kTargetSm, mGemmN, mGemmK, inputs[kInActivation].desc.type);
    size_t maxBytes = 0;
    for (int32_t const m : {1, 2, 4, 8, 16})
    {
        if (m > workspaceMaxM || !useGemv(mBackend, m, mGemmN, mGemmK, inputs[kInActivation].desc.type))
        {
            continue;
        }
        int32_t const splitK = getDispatch(mBackend, m, mGemmN, mGemmK, inputs[kInActivation].desc.type).splitK;
        maxBytes = std::max(maxBytes, getNvfp4A16BlackwellGemvJitWorkspaceSize(gemvKey, m, splitK));
    }

    kernels::Nvfp4A16BlackwellGemmParams const tcgenParams{nullptr, nullptr, nullptr, nullptr, nullptr, profileMaxM,
        mGemmN, mGemmK, toTcgenType(inputs[kInActivation].desc.type)};
    maxBytes = std::max(maxBytes, kernels::Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(tcgenParams));
    return maxBytes;
}

int32_t Nvfp4A16GemmPluginV2::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || outputs[0] == nullptr)
        {
            return -1;
        }
        for (int32_t pos = 0; pos < kNbInputs; ++pos)
        {
            if (inputs[pos] == nullptr || !validateTensorDesc(pos, inputDesc[pos]))
            {
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, outputDesc[0]) || outputDesc[0].type != inputDesc[kInActivation].type)
        {
            return -1;
        }
        if (!isAligned(inputs[kInQWeights], 16) || !isAligned(inputs[kInBlockScales], 16)
            || !isAligned(inputs[kInGlobalScale], alignof(float)))
        {
            LOG_ERROR(
                "Nvfp4A16GemmPluginV2: packed weights/scales do not satisfy the Blackwell kernel alignment contract");
            return -1;
        }

        Dims const& activationDims = inputDesc[kInActivation].dims;
        int32_t const m = getTokenCount(activationDims);
        if (m <= 0 || m > mMaxM || !nvfp4_a16_blackwell::isTmaRepresentableProblem(m, mGemmN, mGemmK)
            || outputDesc[0].dims.d[0] != activationDims.d[0] || outputDesc[0].dims.d[1] != activationDims.d[1])
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: invalid runtime M or output shape");
            return -1;
        }

        DataType const type = inputDesc[kInActivation].type;
        if (useGemv(mBackend, m, mGemmN, mGemmK, type))
        {
            NVTX_SCOPED_RANGE(nvtx_gemv, "Nvfp4A16GemmPluginV2::gemv", nvtx_colors::GREEN);
            int32_t const splitK = getDispatch(mBackend, m, mGemmN, mGemmK, type).splitK;
            Nvfp4A16BlackwellGemvJitKey const expectedKey = getGemvJitKey(type);
            if (!mGemvJitRunner.isLoaded() || !(mGemvJitRunner.getKey() == expectedKey))
            {
                LOG_ERROR("Nvfp4A16GemmPluginV2: selected GEMV JIT module was not loaded before enqueue");
                return -1;
            }
            size_t const workspaceSize = getNvfp4A16BlackwellGemvJitWorkspaceSize(expectedKey, m, splitK);
            if (workspaceSize > 0 && workspace == nullptr)
            {
                LOG_ERROR("Nvfp4A16GemmPluginV2: GEMV split-K workspace is null");
                return -1;
            }
            mGemvJitRunner.launch(inputs[kInActivation], static_cast<uint8_t const*>(inputs[kInQWeights]),
                static_cast<uint8_t const*>(inputs[kInBlockScales]), static_cast<float const*>(inputs[kInGlobalScale]),
                outputs[0], workspace, workspaceSize, m, splitK, stream);
            return 0;
        }
        if (mBackend == kBackendGemv)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: forced GEMV does not support runtime M=%d", m);
            return -1;
        }

        NVTX_SCOPED_RANGE(nvtx_tcgen, "Nvfp4A16GemmPluginV2::tcgen05", nvtx_colors::BLUE);
        kernels::Nvfp4A16BlackwellGemmParams const params{inputs[kInActivation], inputs[kInQWeights],
            inputs[kInBlockScales], static_cast<float const*>(inputs[kInGlobalScale]), outputs[0], m, mGemmN, mGemmK,
            toTcgenType(type)};
        size_t const workspaceSize = kernels::Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(params);
        cudaError_t const runError
            = kernels::Nvfp4A16BlackwellGemmRunner::run(params, workspace, workspaceSize, stream);
        if (runError != cudaSuccess)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: TCGen05 launch failed: %s", cudaGetErrorString(runError));
            return -1;
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Nvfp4A16GemmPluginV2 enqueue failed: %s", error.what());
        return -1;
    }
}

int32_t Nvfp4A16GemmPluginV2::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < kNbInputs; ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            return -1;
        }
    }
    int32_t const m = getTokenCount(in[kInActivation].dims);
    if (!validateTensorDesc(kOutOutput, out[0]) || out[0].type != in[kInActivation].type || m <= 0 || m > mMaxM
        || !nvfp4_a16_blackwell::isTmaRepresentableProblem(m, mGemmN, mGemmK))
    {
        return -1;
    }
    bool const gemv = useGemv(mBackend, m, mGemmN, mGemmK, in[kInActivation].type);
    if (mBackend == kBackendGemv && !gemv)
    {
        LOG_ERROR("Nvfp4A16GemmPluginV2: forced GEMV does not support runtime M=%d", m);
        return -1;
    }
    if (gemv)
    {
        try
        {
            Nvfp4A16BlackwellGemvJitKey const expectedKey = getGemvJitKey(in[kInActivation].type);
            if (!mGemvJitRunner.isLoaded() || !(mGemvJitRunner.getKey() == expectedKey))
            {
                LOG_ERROR("Nvfp4A16GemmPluginV2: selected GEMV JIT module was not preloaded");
                return -1;
            }
        }
        catch (std::exception const& error)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: GEMV JIT validation failed: %s", error.what());
            return -1;
        }
    }
    else
    {
        cudaError_t const error = kernels::Nvfp4A16BlackwellGemmRunner::prepare(
            toTcgenType(in[kInActivation].type), m, mGemmN, mGemmK, nullptr);
        if (error != cudaSuccess)
        {
            LOG_ERROR("Nvfp4A16GemmPluginV2: selected TCGen05 module initialization failed for M=%d: %s", m,
                cudaGetErrorString(error));
            return -1;
        }
    }
    return 0;
}

IPluginV3* Nvfp4A16GemmPluginV2::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Nvfp4A16GemmPluginV2::getFieldsToSerialize() noexcept
{
    if ((mConfiguredNeedsGemv && mGemvJitBundle.empty())
        || mGemvJitBundle.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    {
        LOG_ERROR("Nvfp4A16GemmPluginV2: required GEMV JIT bundle is missing or oversized");
        return nullptr;
    }
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("max_m", &mMaxM, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("layout", &mLayout, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("backend", &mBackend, PluginFieldType::kINT32, 1);
    if (!mGemvJitBundle.empty())
    {
        mDataToSerialize.emplace_back(kGemvJitBundleField, mGemvJitBundle.data(), PluginFieldType::kCHAR,
            static_cast<int32_t>(mGemvJitBundle.size()));
    }
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Nvfp4A16GemmPluginV2Creator::Nvfp4A16GemmPluginV2Creator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("gemm_n", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("gemm_k", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("max_m", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("layout", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("backend", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(kGemvJitBundleField, nullptr, PluginFieldType::kCHAR, 0);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4A16GemmPluginV2Creator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16GemmPluginV2Creator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Nvfp4A16GemmPluginV2Creator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Nvfp4A16GemmPluginV2Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16GemmPluginV2Creator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* Nvfp4A16GemmPluginV2Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        auto plugin = std::make_unique<Nvfp4A16GemmPluginV2>(name == nullptr ? kPluginName : name, fc);
        if (phase == TensorRTPhase::kBUILD)
        {
            if (plugin->hasSerializedGemvJitBundle())
            {
                throw std::invalid_argument("Nvfp4A16GemmPluginV2: BUILD phase must not receive gemv_jit_bundle");
            }
        }
        else if (plugin->hasSerializedGemvJitBundle())
        {
            plugin->loadSerializedGemvJitBundle();
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to create Nvfp4A16GemmPluginV2: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
