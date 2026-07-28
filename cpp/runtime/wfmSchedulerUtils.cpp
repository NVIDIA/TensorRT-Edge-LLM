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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/wfmRuntimeUtils.h"

#include "common/cudaUtils.h"
#include "common/logger.h"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{

std::vector<float> tensorToHostFloat(rt::Tensor const& tensor, cudaStream_t stream)
{
    auto const elements = static_cast<std::size_t>(tensor.getShape().volume());
    std::vector<float> host(elements);
    if (tensor.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(), tensor.rawPointer(), elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    else
    {
        std::vector<half> hostHalf(elements);
        CUDA_CHECK(cudaMemcpyAsync(
            hostHalf.data(), tensor.rawPointer(), elements * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < elements; ++i)
        {
            host[i] = __half2float(hostHalf[static_cast<std::size_t>(i)]);
        }
        return host;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

void hostFloatToTensor(std::vector<float> const& host, rt::Tensor& tensor, cudaStream_t stream)
{
    auto const elements = static_cast<std::size_t>(tensor.getShape().volume());
    if (host.size() != elements)
    {
        throw std::runtime_error("CosmosVisionScheduler: host/tensor element count mismatch.");
    }

    if (tensor.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            tensor.rawPointer(), host.data(), elements * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    else
    {
        std::vector<half> hostHalf(elements);
        for (std::size_t i = 0; i < elements; ++i)
        {
            hostHalf[i] = __float2half(host[i]);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            tensor.rawPointer(), hostHalf.data(), elements * sizeof(half), cudaMemcpyHostToDevice, stream));
    }
}

float expm1fStable(float x)
{
    return std::expm1(x);
}

} // namespace

CosmosVisionScheduler::CosmosVisionScheduler(CosmosEngineConfig const& config)
    : mConfig(config)
{
}

void CosmosVisionScheduler::sigmaToAlphaSigma(float sigma, float& alpha, float& sigmaOut) noexcept
{
    alpha = 1.F - sigma;
    sigmaOut = sigma;
}

void CosmosVisionScheduler::resetState()
{
    mStepIndex = 0;
    mLowerOrderNums = 0;
    mThisOrder = 1;
    mModelOutputs.assign(static_cast<std::size_t>(mConfig.schedulerSolverOrder), {});
    mTimestepList.assign(static_cast<std::size_t>(mConfig.schedulerSolverOrder), 0.F);
    mLastSample.clear();
}

void CosmosVisionScheduler::setTimesteps(int32_t numInferenceSteps)
{
    if (numInferenceSteps <= 0)
    {
        throw std::runtime_error("CosmosVisionScheduler: numInferenceSteps must be positive.");
    }

    mNumInferenceSteps = numInferenceSteps;
    mTimesteps.resize(static_cast<std::size_t>(numInferenceSteps));
    mSigmas.resize(static_cast<std::size_t>(numInferenceSteps) + 1U);

    std::vector<float> raw(static_cast<std::size_t>(numInferenceSteps) + 1U);
    float const end = 1.F / static_cast<float>(mConfig.numTrainTimesteps);
    for (int32_t i = 0; i <= numInferenceSteps; ++i)
    {
        float const frac = static_cast<float>(i) / static_cast<float>(numInferenceSteps);
        raw[static_cast<std::size_t>(i)] = 1.F + frac * (end - 1.F);
    }

    for (int32_t i = 0; i < numInferenceSteps; ++i)
    {
        float sigma = raw[static_cast<std::size_t>(i)];
        sigma = mConfig.flowShift * sigma / (1.F + (mConfig.flowShift - 1.F) * sigma);
        if (i == 0 && std::fabs(sigma - 1.F) < 1e-6F)
        {
            sigma -= 1e-6F;
        }
        mSigmas[static_cast<std::size_t>(i)] = sigma;
        mTimesteps[static_cast<std::size_t>(i)] = sigma * static_cast<float>(mConfig.numTrainTimesteps);
    }
    mSigmas[static_cast<std::size_t>(numInferenceSteps)] = 0.F;
    resetState();
}

float CosmosVisionScheduler::getTimestepValue() const
{
    if (mStepIndex < 0 || mStepIndex >= mNumInferenceSteps)
    {
        throw std::runtime_error("CosmosVisionScheduler: timestep requested outside inference range.");
    }
    return mTimesteps[static_cast<std::size_t>(mStepIndex)];
}

std::vector<float> CosmosVisionScheduler::convertModelOutput(
    std::vector<float> const& sample, std::vector<float> const& modelOutput) const
{
    float sigmaT = 0.F;
    float alphaUnused = 0.F;
    sigmaToAlphaSigma(mSigmas[static_cast<std::size_t>(mStepIndex)], alphaUnused, sigmaT);
    std::vector<float> converted(sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i)
    {
        converted[i] = sample[i] - sigmaT * modelOutput[i];
    }
    return converted;
}

std::vector<float> CosmosVisionScheduler::multistepUniPBhUpdate(
    std::vector<float> const& sample, std::vector<float> const& convertedOutput, int32_t order) const
{
    float sigmaS0 = mSigmas[static_cast<std::size_t>(mStepIndex)];
    float sigmaT = mSigmas[static_cast<std::size_t>(mStepIndex + 1)];
    float alphaS0 = 0.F;
    float alphaT = 0.F;
    float sigmaS0T = 0.F;
    float sigmaTVal = 0.F;
    sigmaToAlphaSigma(sigmaS0, alphaS0, sigmaS0T);
    sigmaToAlphaSigma(sigmaT, alphaT, sigmaTVal);

    float const lambdaT = std::log(alphaT) - std::log(sigmaTVal);
    float const lambdaS0 = std::log(alphaS0) - std::log(sigmaS0T);
    float const h = lambdaT - lambdaS0;
    float const hh = -h;
    float const hPhi1 = expm1fStable(hh);
    float const B_h = expm1fStable(hh);

    std::vector<float> d1s;
    std::vector<float> rhosP;
    if (order > 1 && mStepIndex > 0 && !mModelOutputs[static_cast<std::size_t>(mModelOutputs.size() - 2)].empty())
    {
        auto const& m0 = convertedOutput;
        auto const& mi = mModelOutputs[static_cast<std::size_t>(mModelOutputs.size() - 2)];
        float sigmaSi = mSigmas[static_cast<std::size_t>(mStepIndex - 1)];
        float alphaSi = 0.F;
        float sigmaSiVal = 0.F;
        sigmaToAlphaSigma(sigmaSi, alphaSi, sigmaSiVal);
        float const lambdaSi = std::log(alphaSi) - std::log(sigmaSiVal);
        float const rk = (lambdaSi - lambdaS0) / h;
        d1s.resize(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i)
        {
            d1s[i] = (mi[i] - m0[i]) / rk;
        }
        rhosP = {0.5F};
    }

    std::vector<float> prev(sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i)
    {
        float predRes = 0.F;
        if (!d1s.empty())
        {
            predRes = rhosP[0] * d1s[i];
        }
        float const xTilde = (sigmaTVal / sigmaS0T) * sample[i] - alphaT * hPhi1 * convertedOutput[i];
        prev[i] = xTilde - alphaT * B_h * predRes;
    }
    return prev;
}

std::vector<float> CosmosVisionScheduler::multistepUniCBhUpdate(std::vector<float> const& convertedOutput,
    std::vector<float> const& lastSample, std::vector<float> const& sample, int32_t order) const
{
    if (mStepIndex <= 0)
    {
        return sample;
    }

    float sigmaS0 = mSigmas[static_cast<std::size_t>(mStepIndex - 1)];
    float sigmaT = mSigmas[static_cast<std::size_t>(mStepIndex)];
    float alphaS0 = 0.F;
    float alphaT = 0.F;
    float sigmaS0T = 0.F;
    float sigmaTVal = 0.F;
    sigmaToAlphaSigma(sigmaS0, alphaS0, sigmaS0T);
    sigmaToAlphaSigma(sigmaT, alphaT, sigmaTVal);

    float const lambdaT = std::log(alphaT) - std::log(sigmaTVal);
    float const lambdaS0 = std::log(alphaS0) - std::log(sigmaS0T);
    float const h = lambdaT - lambdaS0;
    float const hh = -h;
    float const hPhi1 = expm1fStable(hh);
    float const B_h = expm1fStable(hh);

    auto const& m0 = mModelOutputs.back();
    std::vector<float> corrected(sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i)
    {
        float const xTilde = (sigmaTVal / sigmaS0T) * lastSample[i] - alphaT * hPhi1 * m0[i];
        corrected[i] = xTilde - alphaT * B_h * (convertedOutput[i] - m0[i]);
    }
    (void) order;
    return corrected;
}

bool CosmosVisionScheduler::stepLatents(rt::Tensor& latents, rt::Tensor const& modelOutput, cudaStream_t stream)
{
    if (mStepIndex >= mNumInferenceSteps)
    {
        LOG_ERROR("CosmosVisionScheduler: step called after all inference steps completed.");
        return false;
    }
    if (latents.getShape() != modelOutput.getShape())
    {
        LOG_ERROR("CosmosVisionScheduler: latent/model_output shape mismatch.");
        return false;
    }

    try
    {
        auto sample = tensorToHostFloat(latents, stream);
        auto modelHost = tensorToHostFloat(modelOutput, stream);
        auto converted = convertModelOutput(sample, modelHost);

        bool const useCorrector = mStepIndex > 0 && !mLastSample.empty();
        if (useCorrector)
        {
            sample = multistepUniCBhUpdate(converted, mLastSample, sample, mThisOrder);
        }

        for (std::size_t i = 0; i + 1U < mModelOutputs.size(); ++i)
        {
            mModelOutputs[i] = std::move(mModelOutputs[i + 1]);
            mTimestepList[i] = mTimestepList[i + 1];
        }
        mModelOutputs.back() = converted;
        mTimestepList.back() = getTimestepValue();

        int32_t const orderCap = mConfig.schedulerSolverOrder;
        int32_t thisOrder = orderCap;
        if (mConfig.numInferenceSteps > 0)
        {
            thisOrder = std::min(orderCap, mNumInferenceSteps - mStepIndex);
        }
        mThisOrder = std::min(thisOrder, mLowerOrderNums + 1);
        if (mThisOrder <= 0)
        {
            mThisOrder = 1;
        }

        mLastSample = sample;
        sample = multistepUniPBhUpdate(sample, converted, mThisOrder);

        if (mLowerOrderNums < mConfig.schedulerSolverOrder)
        {
            ++mLowerOrderNums;
        }
        ++mStepIndex;

        hostFloatToTensor(sample, latents, stream);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("CosmosVisionScheduler::stepLatents failed: %s", e.what());
        return false;
    }
}

} // namespace rt
} // namespace trt_edgellm
