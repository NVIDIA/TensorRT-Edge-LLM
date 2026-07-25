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

#include "trtUtils.h"
#include "checkMacros.h"
#include "cudaUtils.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace trt_edgellm
{
namespace
{

#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
constexpr std::size_t kEngineDeviceTransferChunkSize = 4U * 1024U * 1024U;
constexpr std::size_t kEngineDeviceTransferBufferCount = 2U;

bool isDeviceAccessiblePointer(void* ptr) noexcept
{
    cudaPointerAttributes attributes{};
    cudaError_t const status = cudaPointerGetAttributes(&attributes, ptr);
    if (status != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

class PinnedHostBuffer
{
public:
    explicit PinnedHostBuffer(std::size_t size)
        : mSize(size)
    {
        void* data{};
        cudaError_t status = cudaMallocHost(&data, mSize);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to allocate pinned host buffer for engine deserialization: ")
                + cudaGetErrorString(status));
        }

        cudaEvent_t event{};
        status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (status != cudaSuccess)
        {
            cudaFreeHost(data);
            throw std::runtime_error(
                std::string("Failed to create CUDA event for engine deserialization: ") + cudaGetErrorString(status));
        }

        mData = static_cast<std::byte*>(data);
        mEvent = event;
    }

    PinnedHostBuffer(PinnedHostBuffer const&) = delete;
    PinnedHostBuffer& operator=(PinnedHostBuffer const&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : mData(std::exchange(other.mData, nullptr))
        , mSize(std::exchange(other.mSize, 0U))
        , mEvent(std::exchange(other.mEvent, nullptr))
        , mCopyPending(std::exchange(other.mCopyPending, false))
    {
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) = delete;

    ~PinnedHostBuffer()
    {
        if (mCopyPending && mEvent != nullptr)
        {
            static_cast<void>(cudaEventSynchronize(mEvent));
        }
        if (mEvent != nullptr)
        {
            static_cast<void>(cudaEventDestroy(mEvent));
        }
        if (mData != nullptr)
        {
            static_cast<void>(cudaFreeHost(mData));
        }
    }

    bool waitForPendingCopy() noexcept
    {
        if (!mCopyPending)
        {
            return true;
        }
        cudaError_t const status = cudaEventSynchronize(mEvent);
        if (status != cudaSuccess)
        {
            return false;
        }
        mCopyPending = false;
        return true;
    }

    bool recordPendingCopy(cudaStream_t stream) noexcept
    {
        cudaError_t const status = cudaEventRecord(mEvent, stream);
        if (status != cudaSuccess)
        {
            return false;
        }
        mCopyPending = true;
        return true;
    }

    std::byte* data() noexcept
    {
        return mData;
    }

    std::size_t size() const noexcept
    {
        return mSize;
    }

private:
    std::byte* mData{nullptr};
    std::size_t mSize{0};
    cudaEvent_t mEvent{nullptr};
    bool mCopyPending{false};
};
#endif

#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
class FileStreamReader : public nvinfer1::IStreamReaderV2
#else
class FileStreamReader : public nvinfer1::IStreamReader
#endif
{
public:
    explicit FileStreamReader(std::filesystem::path const& filePath)
        : mPath(filePath.string())
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
        , mDeviceTransferBuffers{PinnedHostBuffer{kEngineDeviceTransferChunkSize}, PinnedHostBuffer {
                                     kEngineDeviceTransferChunkSize
                                 }}
#endif
    {
        mFd = open(mPath.c_str(), O_RDONLY);
        if (mFd < 0)
        {
            throw std::runtime_error("Failed to open engine file: " + mPath + ": " + std::strerror(errno));
        }

        struct stat status;
        if (fstat(mFd, &status) != 0)
        {
            close(mFd);
            mFd = -1;
            throw std::runtime_error("Failed to stat engine file: " + mPath + ": " + std::strerror(errno));
        }
        if (status.st_size <= 0)
        {
            close(mFd);
            mFd = -1;
            throw std::runtime_error("Engine file is empty: " + mPath);
        }
        mSize = static_cast<int64_t>(status.st_size);
    }

    FileStreamReader(FileStreamReader const&) = delete;
    FileStreamReader& operator=(FileStreamReader const&) = delete;

    ~FileStreamReader() override
    {
        if (mFd >= 0)
        {
            close(mFd);
        }
    }

#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
    int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept override
#else
    int64_t read(void* destination, int64_t nbBytes) override
#endif
    {
        if (destination == nullptr || nbBytes < 0)
        {
            return -1;
        }
        if (nbBytes == 0 || mOffset >= mSize)
        {
            return 0;
        }

        int64_t const bytesToRead = std::min(nbBytes, mSize - mOffset);
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
        if (isDeviceAccessiblePointer(destination))
        {
            return readToDevice(destination, bytesToRead, stream);
        }
#endif
        return readToHost(destination, bytesToRead);
    }

#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override
    {
        int64_t base = 0;
        switch (where)
        {
        case nvinfer1::SeekPosition::kSET: base = 0; break;
        case nvinfer1::SeekPosition::kCUR: base = mOffset; break;
        case nvinfer1::SeekPosition::kEND: base = mSize; break;
        default: return false;
        }

        int64_t const nextOffset = base + offset;
        if (nextOffset < 0 || nextOffset > mSize)
        {
            return false;
        }
        mOffset = nextOffset;
        return true;
    }
#endif

private:
    int64_t readToHost(void* destination, int64_t bytesToRead) noexcept
    {
        auto* output = static_cast<std::byte*>(destination);
        int64_t totalRead = 0;
        while (totalRead < bytesToRead)
        {
            ssize_t const chunk = pread(mFd, output + totalRead, static_cast<size_t>(bytesToRead - totalRead),
                static_cast<off_t>(mOffset + totalRead));
            if (chunk < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return -1;
            }
            if (chunk == 0)
            {
                break;
            }
            totalRead += chunk;
        }
        mOffset += totalRead;
        return totalRead;
    }

#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
    int64_t readToDevice(void* destination, int64_t bytesToRead, cudaStream_t stream) noexcept
    {
        auto* output = static_cast<std::byte*>(destination);
        int64_t totalRead = 0;
        std::size_t bufferIndex = 0;
        while (totalRead < bytesToRead)
        {
            auto& transferBuffer = mDeviceTransferBuffers[bufferIndex];
            if (!transferBuffer.waitForPendingCopy())
            {
                return -1;
            }

            size_t const chunkSize = std::min(static_cast<size_t>(bytesToRead - totalRead), transferBuffer.size());
            int64_t const hostRead = readToHost(transferBuffer.data(), static_cast<int64_t>(chunkSize));
            if (hostRead <= 0)
            {
                return totalRead > 0 ? totalRead : hostRead;
            }

            cudaError_t status = cudaMemcpyAsync(output + totalRead, transferBuffer.data(),
                static_cast<size_t>(hostRead), cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess)
            {
                return -1;
            }
            if (!transferBuffer.recordPendingCopy(stream))
            {
                static_cast<void>(cudaStreamSynchronize(stream));
                return -1;
            }

            totalRead += hostRead;
            bufferIndex = (bufferIndex + 1U) % mDeviceTransferBuffers.size();
        }
        return totalRead;
    }
#endif

    std::string mPath;
    int mFd{-1};
    int64_t mSize{0};
    int64_t mOffset{0};
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
    std::array<PinnedHostBuffer, kEngineDeviceTransferBufferCount> mDeviceTransferBuffers;
#endif
};

} // namespace

std::optional<std::pair<cudaGraph_t, cudaGraphExec_t>> captureTRTCudaGraph(
    nvinfer1::IExecutionContext* context, cudaStream_t stream)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool executeStatus{true};
    try
    {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        executeStatus &= context->enqueueV3(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("Failed to capture CUDA graph: %s", e.what());
        // Clean up any CUDA error if the context is not graph-capturable.
        // We do not want to check return value here, so add static_cast<void> to suppress coverity error.
        static_cast<void>(cudaGetLastError());
        // Stop the capture mode and clear CUDA error status if the stream is still in capturing mode.
        cudaStreamCaptureStatus streamStatus;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &streamStatus));
        if (streamStatus != cudaStreamCaptureStatusNone)
        {
            static_cast<void>(cudaStreamEndCapture(stream, &graph));
            static_cast<void>(cudaGetLastError());
        }
        // At this point, there should be no more cuda errors.
        CUDA_CHECK(cudaGetLastError());
        return std::nullopt;
    }

    if (!executeStatus)
    {
        return std::nullopt;
    }

    return std::make_pair(graph, graphExec);
}

void setNonBlockingAuxStreams(
    nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine const* engine, AuxStreamSet& out)
{
    int32_t const nbAuxStreams = engine->getNbAuxStreams();
    if (nbAuxStreams <= 0)
    {
        return;
    }

    // `out` may already hold streams for another context, so record the offset to
    // hand setAuxStreams only the streams created here.
    size_t const start = out.size();
    for (int32_t i = 0; i < nbAuxStreams; ++i)
    {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        out.add(stream);
    }

    context->setAuxStreams(out.data() + start, nbAuxStreams);
    LOG_INFO("configured %d non-blocking auxiliary stream(s) for execution context", nbAuxStreams);
}

std::string dimsToString(nvinfer1::Dims const& dims) noexcept
{
    std::ostringstream oss;
    oss << "[";
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

bool hasDynamicDims(nvinfer1::Dims const& dims) noexcept
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return true;
        }
    }
    return false;
}

bool dimsEqual(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < lhs.nbDims; ++i)
    {
        if (lhs.d[i] != rhs.d[i])
        {
            return false;
        }
    }
    return true;
}

bool isEngineInput(nvinfer1::ICudaEngine const& engine, std::string const& tensorName) noexcept
{
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
    {
        char const* const bindingName = engine.getIOTensorName(i);
        if (std::string_view{bindingName} == tensorName)
        {
            return engine.getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT;
        }
    }
    return false;
}

std::string printEngineInfo(nvinfer1::ICudaEngine const* engine, int32_t profileIndex) noexcept
{
    std::stringstream ss;
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const* const bindingName = engine->getIOTensorName(i);
        nvinfer1::Dims const maxDims
            = engine->getProfileShape(bindingName, profileIndex, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims const minDims
            = engine->getProfileShape(bindingName, profileIndex, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims const optDims
            = engine->getProfileShape(bindingName, profileIndex, nvinfer1::OptProfileSelector::kOPT);
        ss << "  " << bindingName << ": MIN=" << dimsToString(minDims) << ", OPT=" << dimsToString(optDims)
           << ", MAX=" << dimsToString(maxDims) << "\n";
    }
    return ss.str();
}

bool engineHasOutputTensor(nvinfer1::ICudaEngine const* engine, char const* tensorName) noexcept
{
    int32_t const nbIO = engine->getNbIOTensors();
    for (int32_t i = 0; i < nbIO; ++i)
    {
        char const* const name = engine->getIOTensorName(i);
        if (std::string_view{name} == tensorName && engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            return true;
        }
    }
    return false;
}

std::unique_ptr<nvinfer1::ICudaEngine> deserializeCudaEngineFromFile(
    nvinfer1::IRuntime& runtime, std::filesystem::path const& enginePath)
{
    FileStreamReader streamReader(enginePath);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime.deserializeCudaEngine(streamReader));
    if (!engine)
    {
        LOG_ERROR("Failed to deserialize TensorRT engine from file: %s", enginePath.string().c_str());
        throw std::runtime_error("Failed to deserialize TensorRT engine from file: " + enginePath.string());
    }
    return engine;
}

} // namespace trt_edgellm
