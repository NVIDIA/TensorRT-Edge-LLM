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

//! Reference client for the protocol in message.h; use it as is rather than re-implementing the wire format.

#pragma once

#include "message.h"

#include <nvscibuf.h>
#if EDGELLM_IPC_HAS_NVBUFSURFACE
#include <nvbufsurface.h>
#endif

#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <optional>
#include <poll.h>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>

namespace edgellm::ipc
{

//! One piece of a response; text is valid until the next call on the same Client.
struct ResponseChunk
{
    uint32_t requestId;
    std::string_view text;
    bool finished;
    uint16_t finishReason;
    int64_t latencyNs;
    uint32_t serverInferenceMs; //!< Set on the finished chunk only.
    uint32_t inputTokens;       //!< ISL; set on the finished chunk only.
    uint32_t outputTokens;      //!< OSL; set on the finished chunk only.
};

//! What an NvSciBuf frame has to be allocated from and reconciled against.
struct NvSciBufNeeds
{
    NvSciBufModule module;         //!< Every list involved has to come from this one.
    NvSciBufAttrList serviceNeeds; //!< The service's, without which the buffer cannot travel.
};

//! One connection to the service; one request in flight at a time.
class Client
{
public:
    //! Connect to @ref kSocketPath, identify as pid:fd, and register a PCM region of @p audioSamples float32 if
    //! non-zero.
    explicit Client(size_t audioSamples = 0)
        : mSocket(connectToSocket(kSocketPath))
    {
        ClientIdentityDesc desc{};
        std::snprintf(desc.id, sizeof(desc.id), "%d:%d", static_cast<int>(::getpid()), mSocket.get());
        send(MessageType::kIdentify, {{&desc, sizeof(desc)}});
        awaitMessage();
        std::fprintf(stderr, "edgellm-ipc: grep server log for \"client %s\"\n", desc.id);
        if (audioSamples == 0)
        {
            return;
        }

        // memfd: no name to agree on, and nothing left behind after a crash. Sealed against resizing, never
        // F_SEAL_WRITE.
        size_t const bytes = audioSamples * sizeof(float);
        UniqueFd const fd(::memfd_create("edgellm-pcm", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (fd.get() < 0 || ::ftruncate(fd.get(), static_cast<off_t>(bytes)) != 0
            || ::fcntl(fd.get(), F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0)
        {
            throwErrno("creating the PCM region");
        }
        void* const mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
        if (mapped == MAP_FAILED)
        {
            throwErrno("mmap of the PCM region");
        }
        mAudioSamples = static_cast<float*>(mapped);
        mAudioBytes = bytes;
        uint64_t const regionBytes = bytes;
        try
        {
            send(MessageType::kRegisterAudioRegion, {{&regionBytes, sizeof(regionBytes)}}, fd.get());
            awaitMessage();
        }
        catch (...)
        {
            ::munmap(mAudioSamples, mAudioBytes);
            throw;
        }
    }

    ~Client()
    {
        // Reverse of setup: the service's attribute list, the module, then the channel and NvSciIpc.
        if (mServiceNeeds != nullptr)
        {
            NvSciBufAttrListFree(mServiceNeeds);
        }
        if (mNvSciBufModule != nullptr)
        {
            NvSciBufModuleClose(mNvSciBufModule);
        }
        if (mNvSciIpcInitialised)
        {
            closeNvSciIpcChannel(mChannel);
            NvSciIpcDeinit();
        }
        if (mAudioSamples != nullptr)
        {
            ::munmap(mAudioSamples, mAudioBytes);
        }
    }

    //! The first call leases the NvSciBuf channel from the service and takes the service's attributes off it.
    NvSciBufNeeds nvSciBufNeeds()
    {
        if (mChannel.endpoint != 0)
        {
            return NvSciBufNeeds{mNvSciBufModule, mServiceNeeds};
        }
        checkNvSci(NvSciBufModuleOpen(&mNvSciBufModule), "NvSciBufModuleOpen");
        checkNvSci(NvSciIpcInit(), "NvSciIpcInit");
        mNvSciIpcInitialised = true;

        // The service leases a free endpoint pair, opens one end, and tells us the name of the other.
        send(MessageType::kOpenNvSciBufChannel, {});
        ReceivedMessage const lease = awaitMessage();
        auto endpoint = readDesc<NvSciBufChannelDesc>(lease.payload, lease.payloadBytes);
        endpoint.endpointName[sizeof(endpoint.endpointName) - 1U] = '\0';
        openNvSciIpcChannel(mChannel, endpoint.endpointName);
        waitFor("the service to open its end of the NvSciBuf channel", [this] { return mChannel.established; });

        std::vector<uint8_t> const blob = awaitBlob("the service's buffer attributes");
        checkNvSci(NvSciBufAttrListIpcImportUnreconciled(mNvSciBufModule, mChannel.endpoint,
                       blob.data() + sizeof(NvSciBufChannelMessageHeader),
                       blob.size() - sizeof(NvSciBufChannelMessageHeader), &mServiceNeeds),
            "NvSciBufAttrListIpcImportUnreconciled");
        return NvSciBufNeeds{mNvSciBufModule, mServiceNeeds};
    }

    //! The registered PCM region, written in place before every request.
    float* audioBuffer() const noexcept
    {
        return mAudioSamples;
    }

    //! Prompts plus, when one is registered, the PCM region.
    uint32_t request(SamplingParams const& params, std::string_view systemPrompt = {}, std::string_view prompt = {})
    {
        requireNoRequestInFlight();
        return sendRequest(params, false, systemPrompt, prompt);
    }

    //! Exports @p frame over the NvSciBuf channel, then sends the kRequest that uses it; leave @p frame untouched until
    //! the finished chunk. A null @p frame sends the request without an image.
    uint32_t request(SamplingParams const& params, NvSciBufObj frame, std::string_view systemPrompt = {},
        std::string_view prompt = {})
    {
        if (frame == nullptr)
        {
            return request(params, systemPrompt, prompt);
        }
        requireNoRequestInFlight();
        nvSciBufNeeds();
        void* descriptor = nullptr;
        size_t descriptorBytes = 0;
        checkNvSci(NvSciBufIpcExportAttrListAndObj(
                       frame, NvSciBufAccessPerm_ReadWrite, mChannel.endpoint, &descriptor, &descriptorBytes),
            "NvSciBufIpcExportAttrListAndObj (NotPermitted: the buffer was not reconciled against nvSciBufNeeds())");
        NvSciBufChannelMessageHeader const header{
            static_cast<uint32_t>(NvSciBufChannelMessageType::kImageExport), static_cast<uint32_t>(descriptorBytes)};
        sendNvSciIpcBlob(mChannel, packDesc(header, descriptor, descriptorBytes));
        NvSciBufAttrListFreeDesc(descriptor);

        std::vector<uint8_t> const verdict = awaitBlob("the service's verdict on an image");
        auto const reply = readDesc<NvSciBufChannelMessageHeader>(verdict.data(), verdict.size());
        if (static_cast<NvSciBufChannelMessageType>(reply.type) == NvSciBufChannelMessageType::kImageRejected)
        {
            throw std::runtime_error(
                "the service refused the frame: " + std::string(verdict.begin() + sizeof(reply), verdict.end()));
        }
        return sendRequest(params, true, systemPrompt, prompt);
    }

#if EDGELLM_IPC_HAS_NVBUFSURFACE
    //! Sends @p frame's dmabuf and map params on the socket, then the kRequest that uses it; leave @p frame untouched
    //! until the finished chunk. A null @p frame sends the request without an image.
    uint32_t request(SamplingParams const& params, NvBufSurface* frame, std::string_view systemPrompt = {},
        std::string_view prompt = {})
    {
        if (frame == nullptr)
        {
            return request(params, systemPrompt, prompt);
        }
        requireNoRequestInFlight();
        NvBufSurfaceMapParams mapParams{};
        if (NvBufSurfaceGetMapParams(frame, 0, &mapParams) != 0)
        {
            throw std::runtime_error("NvBufSurfaceGetMapParams refused this surface's first buffer");
        }
        mapParams.extendedMapParams = nullptr;
        uint32_t const paramsBytes = static_cast<uint32_t>(sizeof(mapParams));
        send(MessageType::kRegisterNvBufSurface, {{&paramsBytes, sizeof(paramsBytes)}, {&mapParams, sizeof(mapParams)}},
            static_cast<int>(mapParams.fd));
        awaitMessage();
        return sendRequest(params, true, systemPrompt, prompt);
    }
#endif

    ResponseChunk awaitResponses()
    {
        ReceivedMessage const message = awaitMessage();
        auto const arrived = std::chrono::steady_clock::now();
        if (static_cast<MessageType>(message.header.type) != MessageType::kResponse)
        {
            throw std::runtime_error("expected kResponse, got message type " + std::to_string(message.header.type));
        }
        auto const chunk = readDesc<StreamChunkDesc>(message.payload, message.payloadBytes);
        // Clamp to the bytes actually received, so the view never reads past the message.
        size_t const textBytes
            = std::min<size_t>(chunk.textBytes, message.payloadBytes - std::min(sizeof(chunk), message.payloadBytes));

        ResponseChunk const result{message.header.requestId,
            std::string_view(reinterpret_cast<char const*>(message.payload) + sizeof(chunk), textBytes),
            chunk.finished != 0U, chunk.finishReason,
            std::chrono::duration_cast<std::chrono::nanoseconds>(arrived - mSentAt.value_or(arrived)).count(),
            chunk.serverInferenceMs, chunk.inputTokens, chunk.outputTokens};
        if (result.finished)
        {
            mSentAt.reset();
        }
        return result;
    }

private:
    void requireNoRequestInFlight() const
    {
        if (mSentAt)
        {
            throw std::runtime_error("one request at a time: wait for the previous request's finished chunk first");
        }
    }

    uint32_t sendRequest(
        SamplingParams const& params, bool wantsImage, std::string_view systemPrompt, std::string_view prompt)
    {
        RequestDesc const desc{params,
            (wantsImage ? kHasImage : 0U) | (mAudioSamples != nullptr ? kHasAudio : 0U)
                | (systemPrompt.empty() ? 0U : kHasSystemPrompt) | (prompt.empty() ? 0U : kHasPrompt)};
        PromptDesc const promptDesc{static_cast<uint32_t>(systemPrompt.size()), static_cast<uint32_t>(prompt.size())};

        std::vector<ByteSpan> sections{{&desc, sizeof(desc)}};
        if (!systemPrompt.empty() || !prompt.empty())
        {
            sections.insert(sections.end(),
                {{&promptDesc, sizeof(promptDesc)}, {systemPrompt.data(), systemPrompt.size()},
                    {prompt.data(), prompt.size()}});
        }

        uint32_t const requestId = mNextRequestId++;
        // Timestamp before sending, so time spent in a blocking send counts toward latencyNs.
        mSentAt = std::chrono::steady_clock::now();
        send(MessageType::kRequest, sections, -1, requestId);
        return requestId;
    }

    void send(MessageType type, std::vector<ByteSpan> const& parts, int fdToPass = -1, uint32_t requestId = 0)
    {
        if (!trySendMessage(mSocket.get(), type, parts, fdToPass, requestId))
        {
            throw std::runtime_error("sendmsg: the service is not reading and its buffer is full");
        }
    }

    //! Blocks for the next message and throws on kError, so a normal return also serves as an ack.
    ReceivedMessage awaitMessage()
    {
        ReceivedMessage message = receiveMessage(mSocket.get(), mBuffer);
        if (message.peerClosed)
        {
            throw std::runtime_error("service closed the connection");
        }
        if (static_cast<MessageType>(message.header.type) == MessageType::kError)
        {
            auto error = readDesc<ErrorDesc>(message.payload, message.payloadBytes);
            error.detail[sizeof(error.detail) - 1U] = '\0';
            mSentAt.reset();
            throw std::runtime_error("service error " + std::to_string(error.code) + " on request "
                + std::to_string(message.header.requestId) + ": " + error.detail);
        }
        return message;
    }

    //! Exchange blobs on the channel until @p ready, polling so a lost peer is noticed; five seconds at most.
    template <typename Ready>
    void waitFor(char const* what, Ready ready)
    {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (true)
        {
            exchangeNvSciIpcBlobs(mChannel);
            if (ready())
            {
                return;
            }
            auto const left
                = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
                      .count();
            if (mChannel.peerGone || left <= 0)
            {
                throw std::runtime_error(std::string("the NvSciBuf channel closed or timed out waiting for ") + what);
            }
            pollfd watch{mChannel.eventFd, POLLIN, 0};
            ::poll(&watch, 1, static_cast<int>(left));
        }
    }

    std::vector<uint8_t> awaitBlob(char const* what)
    {
        std::optional<std::vector<uint8_t>> blob;
        waitFor(what, [&] {
            blob = receiveNvSciIpcBlob(mChannel);
            return blob && blob->size() >= sizeof(NvSciBufChannelMessageHeader);
        });
        return std::move(*blob);
    }

    UniqueFd mSocket;
    std::vector<uint8_t> mBuffer;
    float* mAudioSamples{nullptr};
    size_t mAudioBytes{0};
    uint32_t mNextRequestId{1}; //!< Starts at 1; 0 means "no request" (acks, connection-level errors).
    std::optional<std::chrono::steady_clock::time_point> mSentAt; //!< Set while a request is in flight, for latencyNs.
    NvSciBufModule mNvSciBufModule{nullptr};
    bool mNvSciIpcInitialised{false}; //!< NvSciIpcInit is per process, so it is deinitialised last.
    NvSciIpcChannel mChannel;         //!< Closed until the first NvSciBuf frame.
    NvSciBufAttrList mServiceNeeds{nullptr};
};

} // namespace edgellm::ipc
