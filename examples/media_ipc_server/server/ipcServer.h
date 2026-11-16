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

#include "message.h"
#include "nvBufToCuda.h"

#include "runtime/audioUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>

namespace edgellm::ipc
{

inline constexpr char const* kNvSciIpcEndpointPrefix = "nvscistream_";
inline constexpr size_t kNvSciIpcMaxPairs = 16; //!< Pair i is endpoints prefix(2i) / prefix(2i+1).
inline constexpr int kCudaDevice = 0;           //!< Must be the device the runtime uses.

inline std::string nvSciIpcEndpointName(size_t index)
{
    return kNvSciIpcEndpointPrefix + std::to_string(index);
}

//! PCM region from the client's memfd; unmapped once the last request reading it lets go.
struct MappedRegion
{
    std::shared_ptr<void const> data;
    size_t bytes{0};
};

//! Process-wide. NvSciIpc first: every endpoint is opened after it and closed before it.
struct NvSciContext
{
    NvSciContext()
    {
        checkNvSci(NvSciIpcInit(), "NvSciIpcInit");
        checkNvSci(NvSciBufModuleOpen(&bufModule), "NvSciBufModuleOpen");
        checkNvSci(NvSciBufAttrListCreate(bufModule, &imageNeeds), "NvSciBufAttrListCreate");

        cudaDeviceProp properties = {};
        checkCuda(cudaGetDeviceProperties(&properties, kCudaDevice), "cudaGetDeviceProperties");
        NvSciRmGpuId gpuId = {};
        static_assert(sizeof(gpuId.bytes) == sizeof(properties.uuid.bytes), "GPU UUID width changed");
        std::memcpy(gpuId.bytes, properties.uuid.bytes, sizeof(gpuId.bytes));
        NvSciBufType const bufferType = NvSciBufType_Image;
        NvSciBufAttrValAccessPerm const permissions = NvSciBufAccessPerm_ReadWrite;
        bool const needCpuAccess = false;
        bool const enableCpuCache = false;
        NvSciBufAttrValImageScanType const scanType = NvSciBufScan_ProgressiveType;
        // No Layout key: the client chooses it, and the import reads it back.
        NvSciBufAttrKeyValuePair pairs[] = {
            {NvSciBufGeneralAttrKey_Types, &bufferType, sizeof(bufferType)},
            {NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId)},
            {NvSciBufGeneralAttrKey_RequiredPerm, &permissions, sizeof(permissions)},
            {NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess)},
            {NvSciBufGeneralAttrKey_EnableCpuCache, &enableCpuCache, sizeof(enableCpuCache)},
            {NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType)},
        };
        checkNvSci(
            NvSciBufAttrListSetAttrs(imageNeeds, pairs, sizeof(pairs) / sizeof(pairs[0])), "NvSciBufAttrListSetAttrs");
    }

    ~NvSciContext()
    {
        NvSciBufAttrListFree(imageNeeds);
        NvSciBufModuleClose(bufModule);
        NvSciIpcDeinit();
    }

    NvSciBufModule bufModule{nullptr};
    NvSciBufAttrList imageNeeds{nullptr}; //!< Attributes every NvSciBuf frame must satisfy.
};

//! An imported frame: the ImageData the runtime reads; subclasses own the native handle behind it.
struct ImportedImage
{
    virtual ~ImportedImage() = default;
    rt::imageUtils::ImageData frame;
};

struct NvSciBufImage : ImportedImage
{
    ~NvSciBufImage() override
    {
        unmapNvSciBufFromCuda(memory, mipmaps, frame);
        if (obj != nullptr)
        {
            NvSciBufObjFree(obj);
        }
    }

    NvSciBufObj obj{nullptr};
    cudaExternalMemory_t memory{nullptr};
    cudaMipmappedArray_t mipmaps[kMaxImagePlanes]{};
};

#if EDGELLM_IPC_HAS_NVBUFSURFACE
struct NvBufSurfaceImage : ImportedImage
{
    ~NvBufSurfaceImage() override
    {
        unmapNvBufSurfaceFromCuda(surface, frame); // Unmap, destroy the surface, then the dmabuf member closes.
        if (surface != nullptr)
        {
            NvBufSurfaceDestroy(surface);
        }
    }

    UniqueFd dmabuf;
    NvBufSurface* surface{nullptr};
};
#endif

//! NvSciIpc has no listener, so each connection leases one endpoint pair; a pair is free once its endpoint is closed.
class NvSciIpcEndpointRegistry
{
public:
    NvSciIpcEndpointRegistry()
    {
        size_t available = 0;
        for (size_t i = 0; i < kNvSciIpcMaxPairs; ++i)
        {
            NvSciIpcEndpoint endpoint = 0;
            if (NvSciIpcOpenEndpoint(nvSciIpcEndpointName(2U * i).c_str(), &endpoint) == NvSciError_Success)
            {
                NvSciIpcCloseEndpointSafe(endpoint, false);
                ++available;
            }
        }
        std::cout << "NvSciIpc: " << available << " " << kNvSciIpcEndpointPrefix
                  << "* endpoint pair(s) ready to lease\n";
    }

    //! Opens the service side of the first free pair; @return its index, or none if every pair is taken or missing.
    std::optional<size_t> lease(NvSciIpcChannel& channel)
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        for (size_t i = 0; i < kNvSciIpcMaxPairs; ++i)
        {
            if (!mLeased[i] && tryOpenNvSciIpcChannel(channel, nvSciIpcEndpointName(2U * i)))
            {
                mLeased[i] = true;
                return i;
            }
        }
        return std::nullopt;
    }

    void release(size_t pairIndex)
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        mLeased[pairIndex] = false;
    }

private:
    std::mutex mMutex;
    bool mLeased[kNvSciIpcMaxPairs]{};
};

//! A connection's NvSciIpc channel, and the frame registered for its next kRequest, which takes it over.
class RegisteredImages
{
public:
    explicit RegisteredImages(std::shared_ptr<NvSciContext> nvSci)
        : mNvSci(std::move(nvSci))
    {
    }

    ~RegisteredImages()
    {
        // Images before the channel they came over; the pair goes back only once its endpoint is closed.
        frame.reset();
        closeNvSciIpcChannel(channel);
        if (mRegistry != nullptr)
        {
            mRegistry->release(mLeasedPairIndex);
        }
    }

    //! @return The endpoint name for the client to open, or empty when no @c nvscistream_* pair could be opened.
    std::string leaseNvSciBufChannel(std::shared_ptr<NvSciIpcEndpointRegistry> registry)
    {
        std::optional<size_t> const pairIndex = registry->lease(channel);
        if (!pairIndex)
        {
            return {};
        }
        mLeasedPairIndex = *pairIndex;
        mRegistry = std::move(registry);
        return nvSciIpcEndpointName(2U * mLeasedPairIndex + 1U);
    }

    //! Announce the service's attributes once, then import whatever the client sent.
    void handleNvSciBufChannelMessages()
    {
        exchangeNvSciIpcBlobs(channel);
        if (!channel.established)
        {
            return;
        }
        if (!mNeedsAnnounced)
        {
            void* descriptor = nullptr;
            size_t descriptorBytes = 0;
            NvSciBufAttrList const lists[] = {mNvSci->imageNeeds};
            checkNvSci(NvSciBufAttrListIpcExportUnreconciled(lists, 1, channel.endpoint, &descriptor, &descriptorBytes),
                "NvSciBufAttrListIpcExportUnreconciled");
            reply(NvSciBufChannelMessageType::kConsumerAttributes, descriptor, descriptorBytes);
            NvSciBufAttrListFreeDesc(descriptor);
            mNeedsAnnounced = true;
        }
        // Receive each kImageExport descriptor the client sent over NvSciIpc.
        while (auto blob = receiveNvSciIpcBlob(channel))
        {
            size_t constexpr headerBytes = sizeof(NvSciBufChannelMessageHeader);
            if (blob->size() <= headerBytes)
            {
                continue;
            }
            try
            {
                // Import the descriptor as an NvSciBufObj.
                auto image = std::make_shared<NvSciBufImage>();
                NvSciBufAttrList const needs[] = {mNvSci->imageNeeds};
                checkNvSci(NvSciBufIpcImportAttrListAndObj(mNvSci->bufModule, channel.endpoint,
                               blob->data() + headerBytes, blob->size() - headerBytes, needs, 1,
                               NvSciBufAccessPerm_ReadWrite, kNvSciBufImportTimeoutUs, &image->obj),
                    "NvSciBufIpcImportAttrListAndObj");
                // ---NvBufToInfer-1 map (NvSciBuf): NvSciBufObj -> CUDA ImageData (cudaImportExternalMemory)---
                image->frame = mapNvSciBufToCuda(image->obj, image->memory, image->mipmaps);
                frame = std::move(image);
            }
            catch (std::exception const& error)
            {
                reply(NvSciBufChannelMessageType::kImageRejected, error.what(), std::strlen(error.what()));
                continue;
            }
            reply(NvSciBufChannelMessageType::kImageAccepted, nullptr, 0);
        }
        // Exchange again: the replies above were only queued.
        exchangeNvSciIpcBlobs(channel);
    }

    NvSciIpcChannel channel;                    //!< Closed until the client asks for one.
    std::shared_ptr<ImportedImage const> frame; //!< Registered for the next kRequest, which takes it.

private:
    void reply(NvSciBufChannelMessageType type, void const* body, size_t bodyBytes)
    {
        NvSciBufChannelMessageHeader const header{static_cast<uint32_t>(type), static_cast<uint32_t>(bodyBytes)};
        sendNvSciIpcBlob(channel, packDesc(header, body, bodyBytes));
    }

    std::shared_ptr<NvSciContext> mNvSci;
    std::shared_ptr<NvSciIpcEndpointRegistry> mRegistry;
    size_t mLeasedPairIndex{0};
    bool mNeedsAnnounced{false};
};

// --- replies: posted from any thread, written by the loop ------------------------------------

//! Make a poll on eventfd @p fd return; safe from any thread or a signal handler.
inline void wake(int fd) noexcept
{
    uint64_t const one = 1;
    [[maybe_unused]] ssize_t const written = ::write(fd, &one, sizeof(one));
}

inline std::vector<uint8_t> errorPayload(ErrorCode code, std::string_view detail)
{
    ErrorDesc error{};
    error.code = static_cast<uint16_t>(code);
    std::memcpy(error.detail, detail.data(), std::min(detail.size(), sizeof(error.detail) - 1U));
    return packDesc(error);
}

struct OutgoingReply
{
    uint64_t connectionId; //!< Routes the reply; not the fd, which the kernel reuses after a close.
    uint32_t requestId;
    MessageType type;
    std::vector<uint8_t> payload;
};

//! Only the loop writes sockets, so other threads post here and the loop wakes on wakeEvent.
class ReplyOutbox
{
public:
    void post(OutgoingReply&& reply)
    {
        {
            std::lock_guard<std::mutex> const guard(mMutex);
            mQueue.push_back(std::move(reply));
        }
        wake(wakeEvent.get());
    }

    std::vector<OutgoingReply> drain()
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        return std::exchange(mQueue, {});
    }

    UniqueFd const wakeEvent{::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)};

private:
    std::mutex mMutex;
    std::vector<OutgoingReply> mQueue;
};

// --- what a connection shares with its requests, and what one request is ---------------------

//! Shared with the requests, so a disconnect cannot pull memory one of them reads.
struct ConnectionMedia
{
    ConnectionMedia(uint64_t id, std::shared_ptr<NvSciContext> nvSci)
        : connectionId(id)
        , images(std::move(nvSci))
    {
    }

    uint64_t const connectionId;       //!< Monotonic, unlike the fd, so a stale reply cannot reach a new client.
    std::string clientId;              //!< Set by kIdentify, which must be the connection's first message.
    std::atomic<bool> connected{true}; //!< False once the client is gone, the cue to stop working for it.
    MappedRegion audioRegion;
    RegisteredImages images;
};

//! Holds the region, so replacing it cannot pull the samples out from under the request.
struct AudioView
{
    std::shared_ptr<void const> region;
    float const* samples;
    uint64_t numSamples;
};

//! One admitted kRequest: copied prompts, the client's image and PCM in place, and its reply path; any thread.
class InflightRequest
{
public:
    InflightRequest(std::shared_ptr<ConnectionMedia> connectionMedia, std::shared_ptr<ReplyOutbox> outbox, uint32_t id)
        : media(std::move(connectionMedia))
        , requestId(id)
        , mOutbox(std::move(outbox))
    {
    }

    //! @p finished marks the chunk that ends the request.
    void sendChunk(std::string_view text, bool finished, uint16_t finishReason = 0, uint32_t serverInferenceMs = 0,
        uint32_t inputTokens = 0, uint32_t outputTokens = 0)
    {
        StreamChunkDesc const chunk{static_cast<uint32_t>(text.size()), static_cast<uint16_t>(finished ? 1U : 0U),
            finishReason, finished ? serverInferenceMs : 0U, finished ? inputTokens : 0U, finished ? outputTokens : 0U};
        post(MessageType::kResponse, packDesc(chunk, text.data(), text.size()));
    }

    //! A whole answer, in as many chunks as the datagram limit needs, never split inside a UTF-8 sequence.
    void sendAnswer(std::string_view text, uint16_t finishReason, uint32_t serverInferenceMs, uint32_t inputTokens,
        uint32_t outputTokens)
    {
        size_t constexpr kMaxChunkTextBytes = kMaxMessageBytes - sizeof(MessageHeader) - sizeof(StreamChunkDesc);
        size_t sent = 0;
        do
        {
            std::string_view const left = text.substr(sent);
            size_t take = std::min(left.size(), kMaxChunkTextBytes);
            while (take < left.size() && (static_cast<unsigned char>(left[take]) & 0xC0U) == 0x80U)
            {
                --take;
            }
            sent += take;
            bool const last = sent >= text.size();
            sendChunk(left.substr(0, take), last, finishReason, serverInferenceMs, inputTokens, outputTokens);
        } while (sent < text.size());
    }

    //! The terminal alternative to sendChunk.
    void sendError(ErrorCode code, std::string_view detail)
    {
        post(MessageType::kError, errorPayload(code, detail));
    }

    std::shared_ptr<ConnectionMedia> const media;
    uint32_t const requestId;
    SamplingParams params{};
    std::string systemPrompt;
    std::string prompt;
    std::shared_ptr<ImportedImage const> image;
    std::optional<AudioView> audio;

private:
    void post(MessageType type, std::vector<uint8_t> payload)
    {
        mOutbox->post(OutgoingReply{media->connectionId, requestId, type, std::move(payload)});
    }

    std::shared_ptr<ReplyOutbox> mOutbox;
};

// --- TensorRT Edge-LLM request -------------------------------------------------------------
inline rt::LLMGenerationRequest buildLLMGenerationRequest(SamplingParams const& params,
    rt::imageUtils::ImageData const* image, AudioView const* audio, std::string_view systemPrompt,
    std::string_view userPrompt)
{
    // TensorRT Edge-LLM: request-level fields on LLMGenerationRequest (llmRuntimeUtils.h)
    rt::LLMGenerationRequest llmRequest;
    llmRequest.temperature = params.temperature;
    llmRequest.topP = params.topP;
    llmRequest.topK = params.topK;
    llmRequest.maxGenerateLength = params.maxGenerateLength;
    llmRequest.diffusionMaxDenoisingSteps = params.diffusionMaxDenoisingSteps;
    llmRequest.numLogprobs = params.numLogprobs;
    llmRequest.enableThinking = (params.flags & kEnableThinking) != 0U;
    llmRequest.disableSpecDecode = (params.flags & kDisableSpecDecode) != 0U;
    llmRequest.applyChatTemplate = (params.flags & kNoChatTemplate) == 0U;
    llmRequest.addGenerationPrompt = (params.flags & kNoGenerationPrompt) == 0U;

    // TensorRT Edge-LLM: the single LLMGenerationRequest::Request, batch size is always 1 (llmRuntimeUtils.h)
    rt::LLMGenerationRequest::Request& slot = llmRequest.requests.emplace_back();
    if (!systemPrompt.empty())
    {
        rt::Message system;
        system.role = "system";
        system.contents.push_back({"text", std::string(systemPrompt)});
        slot.messages.push_back(std::move(system));
    }
    rt::Message user;
    user.role = "user";
    if (image != nullptr)
    {
        user.contents.push_back({"image", "<frame>"});
        // TensorRT Edge-LLM: multimodal vision input (CUDA-mapped NV12 from nvBufToCuda.h)
        slot.imageBuffers.push_back(*image);
    }
    if (audio != nullptr && audio->numSamples != 0U)
    {
        user.contents.push_back({"audio", "<clip>"});
        auto samples = std::make_shared<rt::Tensor>(const_cast<float*>(audio->samples),
            rt::Coords{static_cast<int64_t>(audio->numSamples)}, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT,
            "pcmChunk");
        // TensorRT Edge-LLM: wrap host PCM for the audio encoder path (audioUtils.h)
        slot.audioBuffers.push_back(
            rt::audioUtils::wrapPcm(std::move(samples), static_cast<int32_t>(kAudioSampleRate)));
    }
    if (!userPrompt.empty())
    {
        user.contents.push_back({"text", std::string(userPrompt)});
    }
    slot.messages.push_back(std::move(user));
    return llmRequest;
}

// --- the server: the loop thread moves messages, the worker thread runs inference ------------

class Server
{
public:
    //! Listens on @ref kSocketPath and starts the worker thread that runs @p runtime on @p stream.
    Server(rt::LLMInferenceRuntime& runtime, cudaStream_t stream)
        : mRuntime(runtime)
        , mStream(stream)
        , mListener(listenOnSocket(kSocketPath))
    {
        mWorker = std::thread(&Server::serveInference, this);
    }

    ~Server()
    {
        {
            std::lock_guard<std::mutex> const guard(mPendingMutex);
            mStopping = true;
        }
        mPendingReady.notify_all();
        mWorker.join();
        ::unlink(kSocketPath);
    }

    //! Until requestStop. Whatever one connection throws drops only that connection.
    void run()
    {
        std::vector<pollfd> watch;
        // Per watch past the fixed three: its connection, and whether NvSciIpc.
        std::vector<std::pair<uint64_t, bool>> owners;
        while (true)
        {
            // watch[0] stop event, [1] outbox wake, [2] listener
            watch.assign(
                {{mStopEvent.get(), POLLIN, 0}, {mOutbox->wakeEvent.get(), POLLIN, 0}, {mListener.get(), POLLIN, 0}});
            owners.clear();
            // Add every connected client's fds to the watch list, recording which connection each belongs to.
            for (auto const& [id, connection] : mConnections)
            {
                // POLLOUT only while replies are held, or every writable socket would wake the loop.
                watch.push_back({connection->socket.get(),
                    static_cast<short>(connection->waiting.empty() ? POLLIN : POLLIN | POLLOUT), 0});
                owners.emplace_back(id, false);
                // This client has opened an NvSciIpc channel: also watch its event fd for image buffer messages.
                if (connection->media->images.channel.endpoint != 0)
                {
                    watch.push_back({connection->media->images.channel.eventFd, POLLIN, 0});
                    owners.emplace_back(id, true);
                }
            }

            // Block until any watched fd is ready; a failure here is fatal unless a signal interrupted it.
            if (::poll(watch.data(), watch.size(), -1) < 0)
            {
                // Interrupted by a signal, not a real error: rebuild the watch list and poll again.
                if (errno == EINTR)
                {
                    continue;
                }
                throwErrno("poll");
            }
            // stop() signalled the stop event: leave the loop.
            if ((watch[0].revents & POLLIN) != 0)
            {
                return;
            }
            // The response thread posted output to the outbox: write each to its client's socket.
            if ((watch[1].revents & POLLIN) != 0)
            {
                drainOutbox();
            }
            // The listening socket is readable when a new client is connecting: accept it, otherwise -1.
            int const accepted = (watch[2].revents & POLLIN) != 0
                ? ::accept4(mListener.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK)
                : -1;
            // A client was accepted: register it as a new connection so the next poll watches it.
            if (accepted >= 0)
            {
                uint64_t const id = mNextConnectionId++;
                mConnections.emplace(id, std::make_unique<Connection>(accepted, id, mNvSci));
            }

            // watch[3] client A socket, [4] A's NvSciIpc event fd (only if A opened a channel), [5] client B, ...
            for (size_t i = 3; i < watch.size(); ++i)
            {
                auto const [id, nvSciIpc] = owners[i - 3];
                auto const found = mConnections.find(id);
                if (watch[i].revents != 0 && found != mConnections.end()
                    && !handleClientEvent(*found->second, watch[i].revents, nvSciIpc))
                {
                    drop(id);
                }
            }
        }
    }

    //! Wake the loop and let run() return. Safe from another thread or a signal handler.
    void requestStop() noexcept
    {
        wake(mStopEvent.get());
    }

private:
    struct Connection
    {
        Connection(int fd, uint64_t id, std::shared_ptr<NvSciContext> nvSci)
            : socket(fd)
            , media(std::make_shared<ConnectionMedia>(id, std::move(nvSci)))
        {
        }

        ~Connection()
        {
            media->connected.store(false, std::memory_order_release);
        }

        UniqueFd socket;
        std::shared_ptr<ConnectionMedia> media;
        std::vector<uint8_t> buffer;
        std::deque<OutgoingReply> waiting; //!< Replies the socket had no room for, oldest first.
    };

    //! Requests still running keep the media they read; their replies are dropped.
    void drop(uint64_t id)
    {
        std::cerr << "client " << mConnections.at(id)->media->clientId << " disconnected (conn "
                  << mConnections.at(id)->media->connectionId << ")\n";
        mConnections.erase(id);
    }

    //! Handle one wakeup on @p connection's socket or NvSciIpc channel. Returns false to drop the connection.
    bool handleClientEvent(Connection& connection, short revents, bool nvSciIpc)
    {
        try
        {
            // NvSciIpc event fd fired: receive the image buffers the client sent over its channel.
            if (nvSciIpc)
            {
                connection.media->images.handleNvSciBufChannelMessages();
                return true;
            }
            // The socket hung up, errored, or is invalid: the client is gone, so drop the connection.
            if ((revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            {
                return false;
            }
            // Flush held replies before reading new messages, so they go out ahead of later requests' replies.
            if ((revents & POLLOUT) != 0)
            {
                flushTo(connection);
            }
            return (revents & POLLIN) == 0 || handleClientMessage(connection);
        }
        catch (std::exception const& error)
        {
            std::cerr << "client " << connection.media->clientId << ": " << error.what() << '\n';
            return false;
        }
    }

    //! Handle one socket message. A refused message gets a kError and the connection stays open.
    bool handleClientMessage(Connection& connection)
    {
        ReceivedMessage const message = receiveMessage(connection.socket.get(), connection.buffer);
        UniqueFd passed(message.fd); // Taken by an NvBufSurface registration, closed on return otherwise.
        if (message.peerClosed)
        {
            return false;
        }
        ConnectionMedia& media = *connection.media;
        auto const type = static_cast<MessageType>(message.header.type);
        if (media.clientId.empty() && type != MessageType::kIdentify)
        {
            sendError(connection, ErrorCode::kBadRequest, "first message must identify the client");
            return false;
        }

        try
        {
            switch (type)
            {
            // Client identification: records the client ID; required as the first message on each connection.
            case MessageType::kIdentify:
            {
                auto const desc = readDesc<ClientIdentityDesc>(message.payload, message.payloadBytes);
                media.clientId.assign(desc.id, ::strnlen(desc.id, sizeof(desc.id)));
                std::cerr << "client " << media.clientId << " connected (conn " << media.connectionId << ")\n";
                sendTo(connection, MessageType::kAck, {});
                return true;
            }
            // Client passes a memfd holding its audio: map it read-only for later requests.
            case MessageType::kRegisterAudioRegion:
            {
                auto const regionBytes = readDesc<uint64_t>(message.payload, message.payloadBytes);
                int const seals = ::fcntl(passed.get(), F_GET_SEALS);
                if (seals < 0 || (seals & F_SEAL_SHRINK) == 0)
                {
                    throw std::runtime_error("the audio region must be a memfd sealed with F_SEAL_SHRINK");
                }
                struct stat status = {};
                if (::fstat(passed.get(), &status) != 0)
                {
                    throwErrno("fstat on the audio region");
                }
                // Map no more than the memfd actually holds.
                size_t const bytes = std::min<size_t>(regionBytes, static_cast<size_t>(status.st_size));
                if (bytes < sizeof(float)
                    || bytes / sizeof(float) > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
                {
                    throw std::runtime_error("the audio region must hold 1 to INT32_MAX float samples");
                }
                void* const mapped = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, passed.get(), 0);
                if (mapped == MAP_FAILED)
                {
                    throwErrno("mmap of the audio region");
                }
                media.audioRegion = {std::shared_ptr<void const>(mapped,
                                         [bytes](void const* data) { ::munmap(const_cast<void*>(data), bytes); }),
                    bytes};
                sendTo(connection, MessageType::kAck, {});
                return true;
            }
            // Client asks for an NvSciIpc channel for images: lease an endpoint pair and reply with its name.
            case MessageType::kOpenNvSciBufChannel:
            {
                if (media.images.channel.endpoint != 0)
                {
                    throw std::runtime_error("this connection already has an NvSciBuf channel");
                }
                std::string const clientEndpoint = media.images.leaseNvSciBufChannel(mNvSciIpcRegistry);
                if (clientEndpoint.empty())
                {
                    sendError(connection, ErrorCode::kBusy,
                        "no NvSciIpc endpoint pair is available; every nvscistream_* pair is in use or missing from "
                        "nvsciipc.cfg");
                    return true;
                }
                NvSciBufChannelDesc desc{};
                std::snprintf(desc.endpointName, sizeof(desc.endpointName), "%s", clientEndpoint.c_str());
                sendTo(connection, MessageType::kNvSciBufChannel, packDesc(desc));
                return true;
            }
#if EDGELLM_IPC_HAS_NVBUFSURFACE
            // Client passes an NvBufSurface image (dmabuf fd + map params): import it for the next request.
            case MessageType::kRegisterNvBufSurface:
            {
                // Read the map params; the dmabuf fd came with this message (SCM_RIGHTS).
                auto const paramsBytes = readDesc<uint32_t>(message.payload, message.payloadBytes);
                if (paramsBytes != sizeof(NvBufSurfaceMapParams))
                {
                    throw std::runtime_error("NvBufSurfaceMapParams size differs between client and service builds");
                }
                auto importParams = readDesc<NvBufSurfaceMapParams>(
                    message.payload + sizeof(paramsBytes), message.payloadBytes - sizeof(paramsBytes));
                // Import the dmabuf and map params as an NvBufSurface.
                auto image = std::make_shared<NvBufSurfaceImage>();
                image->dmabuf = std::move(passed);
                // The client's fd number and pointers are only valid in its process: use our dmabuf fd and clear the
                // pointers.
                importParams.fd = static_cast<uint64_t>(image->dmabuf.get());
                importParams.extendedMapParams = nullptr;
                for (auto*& reserved : importParams._reserved)
                {
                    reserved = nullptr;
                }
                NvBufSurfaceImport(&image->surface, &importParams);
                if (image->surface == nullptr)
                {
                    throw std::runtime_error("NvBufSurfaceImport refused this surface's parameters");
                }
                image->surface->numFilled = 1;
                // ---NvBufToInfer-1 map (NvBufSurface): NvBufSurface -> CUDA ImageData (NvBufSurfaceMapCudaBuffer)
                image->frame = mapNvBufSurfaceToCuda(image->surface);
                media.images.frame = std::move(image);
                sendTo(connection, MessageType::kAck, {});
                return true;
            }
#endif
            // Client sends an inference request: validate it and hand it to the worker.
            case MessageType::kRequest: return handleRequest(connection, message);
            // Any other type is not something a client may send.
            default: throw std::runtime_error("unsupported message type " + std::to_string(message.header.type));
            }
        }
        catch (std::runtime_error const& error)
        {
            std::cerr << "client " << media.clientId << ": refused: " << error.what() << '\n';
            sendError(connection, ErrorCode::kBadRequest, error.what(), message.header.requestId);
            return true;
        }
    }

    //! Validate one kRequest and hand it to the worker; a refused one gets a kError.
    bool handleRequest(Connection& connection, ReceivedMessage const& message)
    {
        uint32_t const requestId = message.header.requestId;
        // Each section is bounds-checked against the received payload, not trusted from the flags.
        size_t offset = 0;
        auto const left = [&] { return message.payloadBytes > offset ? message.payloadBytes - offset : 0U; };
        auto const take = [&](auto section) {
            section = readDesc<decltype(section)>(message.payload + offset, left());
            offset += sizeof(section);
            return section;
        };
        auto const reject = [&](ErrorCode code, char const* reason) {
            sendError(connection, code, reason, requestId);
            return true;
        };

        auto const desc = take(RequestDesc{});
        if ((desc.params.flags & kGenerateAudio) != 0U)
        {
            return reject(ErrorCode::kUnsupportedFormat, "a spoken response is not supported yet");
        }

        auto request = std::make_shared<InflightRequest>(connection.media, mOutbox, requestId);
        request->params = desc.params;

        if ((desc.payloadFlags & kHasImage) != 0U)
        {
            request->image = std::move(connection.media->images.frame);
            if (request->image == nullptr)
            {
                return reject(
                    ErrorCode::kBadRequest, "the request includes an image but none was registered before it");
            }
        }
        if ((desc.payloadFlags & kHasAudio) != 0U)
        {
            MappedRegion const& region = connection.media->audioRegion;
            if (region.data == nullptr)
            {
                return reject(ErrorCode::kBadRequest, "the request includes audio but no PCM region is registered");
            }
            request->audio
                = AudioView{region.data, static_cast<float const*>(region.data.get()), region.bytes / sizeof(float)};
        }
        if ((desc.payloadFlags & (kHasPrompt | kHasSystemPrompt)) != 0U)
        {
            auto const prompt = take(PromptDesc{});
            auto const* const text = reinterpret_cast<char const*>(message.payload);
            size_t const systemPromptBytes = std::min<size_t>(prompt.systemPromptBytes, left());
            request->systemPrompt.assign(text + offset, systemPromptBytes);
            offset += systemPromptBytes;
            request->prompt.assign(text + offset, std::min<size_t>(prompt.promptBytes, left()));
        }

        // Queue for the worker thread, which runs one request at a time in arrival order.
        {
            std::lock_guard<std::mutex> const guard(mPendingMutex);
            mPending.push_back(std::move(request));
        }
        mPendingReady.notify_one();
        return true;
    }

    void drainOutbox()
    {
        uint64_t counter = 0;
        // Cleared before the queue is taken, so a reply posted in between costs a wakeup, not a stall.
        [[maybe_unused]] ssize_t const consumed = ::read(mOutbox->wakeEvent.get(), &counter, sizeof(counter));
        for (OutgoingReply& reply : mOutbox->drain())
        {
            auto const found = mConnections.find(reply.connectionId);
            if (found == mConnections.end())
            {
                continue;
            }
            Connection& connection = *found->second;
            try
            {
                sendTo(connection, reply.type, std::move(reply.payload), reply.requestId);
            }
            catch (std::exception const& error)
            {
                std::cerr << "client " << connection.media->clientId << ": " << error.what() << '\n';
                drop(found->first);
            }
        }
    }

    //! Queued behind whatever is already held, so replies keep their order.
    static void sendTo(Connection& connection, MessageType type, std::vector<uint8_t> payload, uint32_t requestId = 0)
    {
        connection.waiting.push_back(
            OutgoingReply{connection.media->connectionId, requestId, type, std::move(payload)});
        flushTo(connection);
    }

    static void sendError(Connection& connection, ErrorCode code, std::string_view detail, uint32_t requestId = 0)
    {
        sendTo(connection, MessageType::kError, errorPayload(code, detail), requestId);
    }

    //! Send held replies until the socket is full; the rest waits for the next POLLOUT.
    static void flushTo(Connection& connection)
    {
        while (!connection.waiting.empty())
        {
            OutgoingReply const& held = connection.waiting.front();
            if (!trySendMessage(connection.socket.get(), held.type,
                    {ByteSpan{held.payload.data(), held.payload.size()}}, -1, held.requestId))
            {
                return;
            }
            connection.waiting.pop_front();
        }
    }

    //! Worker thread: takes @ref mPending one request at a time and runs it to its reply.
    void serveInference()
    {
        while (true)
        {
            std::shared_ptr<InflightRequest> request;
            {
                std::unique_lock<std::mutex> lock(mPendingMutex);
                mPendingReady.wait(lock, [this] { return mStopping || !mPending.empty(); });
                if (mStopping)
                {
                    return;
                }
                request = std::move(mPending.front());
                mPending.pop_front();
            }
            if (!request->media->connected.load(std::memory_order_acquire))
            {
                continue;
            }

            // ---NvBufToInfer-2 build request: rt::LLMGenerationRequest from ImageData, PCM and prompts---
            rt::LLMGenerationRequest llmRequest;
            try
            {
                llmRequest
                    = buildLLMGenerationRequest(request->params, request->image ? &request->image->frame : nullptr,
                        request->audio ? &*request->audio : nullptr, request->systemPrompt, request->prompt);
            }
            catch (std::exception const& thrown)
            {
                request->sendError(ErrorCode::kBadRequest, std::string("cannot build the request: ") + thrown.what());
                continue;
            }

            // ----------NvBufToInfer-3 stream relay: set up before handleRequest blocks----------
            // A pump thread forwards StreamChannel text to the client during inference.
            bool const stream = (request->params.flags & kStream) != 0U;
            std::thread relay;
            if (stream)
            {
                // TensorRT Edge-LLM: streaming sink for the single request (streaming.h)
                std::shared_ptr<rt::StreamChannel> channel = rt::StreamChannel::create();
                llmRequest.streamChannels.push_back(channel);
                relay = std::thread([channel, request] { relayStreamChannel(*channel, *request); });
            }

            // ----------NvBufToInfer-4 handleRequest: LLMInferenceRuntime::handleRequest----------
            rt::LLMGenerationResponse response;
            std::string error;
            auto const started = std::chrono::steady_clock::now();
            try
            {
                // TensorRT Edge-LLM: prefill + decode (llmInferenceRuntime.h)
                if (!mRuntime.handleRequest(llmRequest, response, mStream))
                {
                    error = "the runtime refused the request";
                }
            }
            catch (std::exception const& thrown)
            {
                error = std::string("the runtime threw: ") + thrown.what();
            }
            double const inferenceMs
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            if (relay.joinable())
            {
                llmRequest.streamChannels.front()->cancel();
                relay.join();
            }
            // ---NvBufToInfer-5 reply: terminal stream chunk, full answer, or error, posted to the outbox---
            if (!error.empty())
            {
                std::cerr << "perf: client=" << request->media->clientId << " requestId=" << request->requestId
                          << " server_inference_ms=" << inferenceMs << " failed=" << error << std::endl;
                request->sendError(ErrorCode::kInternal, error);
                continue;
            }
            // TensorRT Edge-LLM: ISL from inputTokenCounts, OSL from outputIds (llmRuntimeUtils.h)
            auto const inputTokens = static_cast<uint32_t>(response.inputTokenCounts.front());
            auto const outputTokens = static_cast<uint32_t>(response.outputIds.front().size());
            std::cerr << "perf: client=" << request->media->clientId << " requestId=" << request->requestId
                      << " server_inference_ms=" << inferenceMs << " isl=" << inputTokens << " osl=" << outputTokens
                      << std::endl;
            auto const finishReason = static_cast<uint16_t>(response.finishReasons.front());
            auto const roundedMs = static_cast<uint32_t>(inferenceMs + 0.5);
            if (stream)
            {
                request->sendChunk("", true, finishReason, roundedMs, inputTokens, outputTokens);
            }
            else
            {
                // TensorRT Edge-LLM: non-streaming text from LLMGenerationResponse::outputTexts
                request->sendAnswer(response.outputTexts.front(), finishReason, roundedMs, inputTokens, outputTokens);
            }
        }
    }

    //! Forward decoded text until the stream ends or the client leaves; the finished chunk is sent by serveInference().
    static void relayStreamChannel(rt::StreamChannel& channel, InflightRequest& request)
    {
        while (request.media->connected.load(std::memory_order_acquire))
        {
            bool const done = channel.isFinished() || channel.isCancelled();
            bool idle = true;
            // TensorRT Edge-LLM: decode tokens queued on StreamChannel during handleRequest (streaming.h)
            for (auto chunk = channel.tryPop(); chunk; chunk = channel.tryPop())
            {
                idle = false;
                request.sendChunk(chunk->text, false);
            }
            if (done)
            {
                return;
            }
            if (idle)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        channel.cancel();
    }

    rt::LLMInferenceRuntime& mRuntime;
    cudaStream_t mStream;
    UniqueFd mListener;
    UniqueFd mStopEvent{::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)};
    std::shared_ptr<NvSciContext> mNvSci{std::make_shared<NvSciContext>()};
    std::shared_ptr<NvSciIpcEndpointRegistry> mNvSciIpcRegistry{std::make_shared<NvSciIpcEndpointRegistry>()};
    std::shared_ptr<ReplyOutbox> mOutbox{std::make_shared<ReplyOutbox>()};
    std::unordered_map<uint64_t, std::unique_ptr<Connection>> mConnections; //!< By connectionId, which replies carry.
    uint64_t mNextConnectionId{1};
    std::mutex mPendingMutex;
    std::condition_variable mPendingReady;
    std::deque<std::shared_ptr<InflightRequest>> mPending; //!< Requests waiting for the worker, oldest first.
    bool mStopping{false};
    std::thread mWorker; //!< Last: started once everything it reads is constructed.
};

} // namespace edgellm::ipc
