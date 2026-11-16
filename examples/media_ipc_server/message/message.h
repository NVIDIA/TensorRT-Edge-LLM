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

#include <nvscierror.h>
#include <nvsciipc.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

//! The transport protocol for multimodal requests: the structs that cross, and the calls that carry one.
//!   kIdentify              [MessageHeader 16][ClientIdentityDesc 64]
//!   kAck                   [MessageHeader 16]
//!   kRegisterAudioRegion   [MessageHeader 16][uint64 regionBytes 8] + the region's memfd
//!   kOpenNvSciBufChannel   [MessageHeader 16]
//!   kNvSciBufChannel       [MessageHeader 16][NvSciBufChannelDesc 64]
//!   kRegisterNvBufSurface  [MessageHeader 16][uint32 paramsBytes 4][paramsBytes] + the dmabuf
//!   kRequest               [MessageHeader 16][RequestDesc 48][PromptDesc 8]?
//!                          [UTF-8 systemPromptBytes][UTF-8 promptBytes]
//!   kResponse              [MessageHeader 16][StreamChunkDesc 20][UTF-8 textBytes]
//!   kError                 [MessageHeader 16][ErrorDesc 130]
namespace edgellm::ipc
{

inline constexpr uint32_t kTransportMagic = 0x454C4D50U; //!< 'ELMP'.

inline constexpr uint16_t kTransportVersion = 1U;

inline constexpr char const* kSocketPath = "/tmp/edgellm-ipc.sock";

enum class MessageType : uint16_t
{
    kIdentify = 1,
    kAck,

    kRegisterAudioRegion,
    kOpenNvSciBufChannel,
    kNvSciBufChannel,
    kRegisterNvBufSurface,

    kRequest,
    kResponse,

    kError,
};

struct MessageHeader
{
    uint32_t magic;   //!< kTransportMagic.
    uint16_t version; //!< kTransportVersion.
    uint16_t type;    //!< MessageType.
    uint32_t payloadBytes;
    uint32_t requestId;
};

inline constexpr size_t kMaxMessageBytes = 128U * 1024U;

struct ClientIdentityDesc
{
    char id[64];
};

struct SamplingParams
{
    float temperature{1.0F};
    float topP{0.8F};
    int64_t topK{50};
    int64_t maxGenerateLength{256};
    int32_t numLogprobs{};
    int32_t diffusionMaxDenoisingSteps{};
    uint32_t flags{}; //!< GenerationFlag
};

enum GenerationFlag : uint32_t
{
    kStream = 1U << 0U,
    kNoChatTemplate = 1U << 1U,
    kNoGenerationPrompt = 1U << 2U,
    kEnableThinking = 1U << 3U,
    kDisableSpecDecode = 1U << 5U,
    kGenerateAudio = 1U << 6U,
};

// --- audio ------------------------------------------------------------------------------------

inline constexpr uint32_t kAudioSampleRate = 16000U;

// --- the request -------------------------------------------------------------------------------

enum PayloadFlag : uint32_t
{
    kHasImage = 1U << 0U,
    kHasAudio = 1U << 1U,
    kHasPrompt = 1U << 2U,
    kHasSystemPrompt = 1U << 3U,
};

struct RequestDesc
{
    SamplingParams params;
    uint32_t payloadFlags; //!< PayloadFlag. A request states at least one of them.
};

//! The last section: this desc, then the system prompt bytes, then the user prompt bytes; an unset prompt has length 0.
struct PromptDesc
{
    uint32_t systemPromptBytes;
    uint32_t promptBytes;
};

// --- the reply ---------------------------------------------------------------------------------

struct StreamChunkDesc
{
    uint32_t textBytes;
    uint16_t finished;
    uint16_t finishReason;      //!< The runtime's FinishReason, opaque here.
    uint32_t serverInferenceMs; //!< LLMInferenceRuntime::handleRequest duration; set on the terminal chunk only.
    uint32_t inputTokens;       //!< ISL: prompt tokens after chat template and media expansion; terminal chunk only.
    uint32_t outputTokens;      //!< OSL: generated tokens; terminal chunk only.
};

enum class ErrorCode : uint16_t
{
    kBadRequest = 0,
    kBusy,
    kUnsupportedFormat,
    kInternal
};

struct ErrorDesc
{
    uint16_t code;
    char detail[128];
};

struct NvSciBufChannelDesc
{
    char endpointName[64];
};

enum class NvSciBufChannelMessageType : uint32_t
{
    kConsumerAttributes = 1,
    kImageExport,
    kImageAccepted,
    kImageRejected,
};

struct NvSciBufChannelMessageHeader
{
    uint32_t type; //!< NvSciBufChannelMessageType.
    uint32_t bodyBytes;
};

// --- what both ends build messages and handle faults with --------------------------------------

//! @p what names the call and whatever identifies its subject.
inline void checkNvSci(NvSciError status, std::string const& what)
{
    if (status != NvSciError_Success)
    {
        throw std::runtime_error(what + " failed with NvSciError " + std::to_string(static_cast<uint32_t>(status)));
    }
}

[[noreturn]] inline void throwErrno(std::string const& what)
{
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

//! Owns one descriptor. The calls below take plain ints.
class UniqueFd
{
public:
    UniqueFd() = default;

    explicit UniqueFd(int fd) noexcept
        : mFd(fd)
    {
    }

    ~UniqueFd()
    {
        reset();
    }

    UniqueFd(UniqueFd&& other) noexcept
        : mFd(other.release())
    {
    }

    UniqueFd& operator=(UniqueFd&& other) noexcept
    {
        reset(other.release());
        return *this;
    }

    int get() const noexcept
    {
        return mFd;
    }

    int release() noexcept
    {
        return std::exchange(mFd, -1);
    }

    void reset(int fd = -1) noexcept
    {
        if (mFd >= 0)
        {
            ::close(mFd);
        }
        mFd = fd;
    }

private:
    int mFd{-1};
};

//! @p head, then @p bodyBytes of @p body: one payload or one blob.
template <typename T>
std::vector<uint8_t> packDesc(T const& head, void const* body = nullptr, size_t bodyBytes = 0)
{
    std::vector<uint8_t> bytes(sizeof(T) + bodyBytes);
    std::memcpy(bytes.data(), &head, sizeof(T));
    if (bodyBytes != 0)
    {
        std::memcpy(bytes.data() + sizeof(T), body, bodyBytes);
    }
    return bytes;
}

//! Copies only the bytes that arrived; a short payload leaves the rest zeroed.
template <typename T>
T readDesc(void const* payload, size_t payloadBytes)
{
    T value{};
    std::memcpy(&value, payload, std::min(sizeof(T), payloadBytes));
    return value;
}

// --- carrying a message: the AF_UNIX connection -----------------------------------------------

namespace detail
{

//! SOCK_SEQPACKET, so one datagram is one message; @p address is filled in for bind or connect.
inline UniqueFd openSeqpacketSocket(std::string const& path, sockaddr_un& address)
{
    if (path.empty() || path.size() >= sizeof(address.sun_path))
    {
        throw std::runtime_error("socket path is empty or longer than sun_path allows: " + path);
    }
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size());
    UniqueFd fd(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (fd.get() < 0)
    {
        throwErrno("socket(AF_UNIX, SOCK_SEQPACKET)");
    }
    int const wanted = static_cast<int>(kMaxMessageBytes);
    ::setsockopt(fd.get(), SOL_SOCKET, SO_SNDBUF, &wanted, sizeof(wanted));
    ::setsockopt(fd.get(), SOL_SOCKET, SO_RCVBUF, &wanted, sizeof(wanted));
    return fd;
}

} // namespace detail

//! The service's end, replacing whatever was left at @p path.
inline UniqueFd listenOnSocket(std::string const& path)
{
    sockaddr_un address{};
    UniqueFd fd = detail::openSeqpacketSocket(path, address);
    ::unlink(path.c_str());
    if (::bind(fd.get(), reinterpret_cast<sockaddr const*>(&address), sizeof(address)) != 0
        || ::listen(fd.get(), 16) != 0)
    {
        throwErrno("bind or listen on " + path);
    }
    return fd;
}

//! A client's end, blocking.
inline UniqueFd connectToSocket(std::string const& path)
{
    sockaddr_un address{};
    UniqueFd fd = detail::openSeqpacketSocket(path, address);
    if (::connect(fd.get(), reinterpret_cast<sockaddr const*>(&address), sizeof(address)) != 0)
    {
        throwErrno("connect(" + path + ")");
    }
    return fd;
}

//! One section of an outgoing payload, gathered into the datagram rather than copied into it.
struct ByteSpan
{
    void const* data;
    size_t bytes;
};

//! One received datagram, whose payload points into the buffer handed to receiveMessage.
struct ReceivedMessage
{
    MessageHeader header{};
    uint8_t const* payload{nullptr};
    size_t payloadBytes{0};
    int fd{-1}; //!< What the peer passed over SCM_RIGHTS, the receiver's to close; -1 when none.
    bool peerClosed{false};
};

//! One datagram: @p parts behind a MessageHeader, @p fdToPass as SCM_RIGHTS. False if the peer's buffer is full.
inline bool trySendMessage(
    int fd, MessageType type, std::vector<ByteSpan> const& parts, int fdToPass = -1, uint32_t requestId = 0)
{
    MessageHeader header{kTransportMagic, kTransportVersion, static_cast<uint16_t>(type), 0, requestId};
    std::vector<iovec> vectors{iovec{&header, sizeof(header)}};
    for (ByteSpan const& part : parts)
    {
        header.payloadBytes += static_cast<uint32_t>(part.bytes);
        vectors.push_back(iovec{const_cast<void*>(part.data), part.bytes});
    }

    msghdr message{};
    message.msg_iov = vectors.data();
    message.msg_iovlen = vectors.size();
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    if (fdToPass >= 0)
    {
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        cmsghdr* const ancillary = CMSG_FIRSTHDR(&message);
        ancillary->cmsg_level = SOL_SOCKET;
        ancillary->cmsg_type = SCM_RIGHTS;
        ancillary->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(ancillary), &fdToPass, sizeof(int));
    }

    while (::sendmsg(fd, &message, MSG_NOSIGNAL) < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return false;
        }
        if (errno != EINTR)
        {
            throwErrno("sendmsg");
        }
    }
    return true;
}

//! Receive one datagram into @p buffer.
inline ReceivedMessage receiveMessage(int fd, std::vector<uint8_t>& buffer)
{
    buffer.resize(kMaxMessageBytes);
    iovec vector{buffer.data(), buffer.size()};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);

    ssize_t received = 0;
    while ((received = ::recvmsg(fd, &message, MSG_CMSG_CLOEXEC)) < 0)
    {
        if (errno != EINTR)
        {
            throwErrno("recvmsg");
        }
    }

    ReceivedMessage result{};
    cmsghdr* const ancillary = CMSG_FIRSTHDR(&message);
    if (ancillary != nullptr && ancillary->cmsg_level == SOL_SOCKET && ancillary->cmsg_type == SCM_RIGHTS)
    {
        std::memcpy(&result.fd, CMSG_DATA(ancillary), sizeof(int));
    }
    result.peerClosed = received == 0;
    if (!result.peerClosed)
    {
        bool const complete = static_cast<size_t>(received) >= sizeof(MessageHeader)
            && (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) == 0;
        if (complete)
        {
            std::memcpy(&result.header, buffer.data(), sizeof(MessageHeader));
        }
        if (!complete || result.header.magic != kTransportMagic || result.header.version != kTransportVersion)
        {
            if (result.fd >= 0)
            {
                ::close(result.fd);
            }
            throw std::runtime_error("received an incomplete message or one from an incompatible protocol version");
        }
    }
    // Use the received size, not the header's claim, so a payload never reaches past the datagram.
    result.payload = buffer.data() + sizeof(MessageHeader);
    result.payloadBytes = static_cast<size_t>(received) > sizeof(MessageHeader)
        ? static_cast<size_t>(received) - sizeof(MessageHeader)
        : 0;
    return result;
}

// --- carrying a blob: an NvSciIpc channel, each blob sent as [uint32 blobBytes][blob] ----------

//! One end of an NvSciIpc channel; zero-initialised is closed, openNvSciIpcChannel fills it.
struct NvSciIpcChannel
{
    NvSciIpcEndpoint endpoint{0};
    std::string name;
    int eventFd{-1}; //!< Readable while events are pending; owned by NvSciIpc.
    bool established{false};
    bool peerGone{false};
    std::vector<uint8_t> frame; //!< One NvSciIpc frame (frame_size from /etc/nvsciipc.cfg), not an image frame.
    std::deque<std::vector<uint8_t>> outgoing;
    std::deque<std::vector<uint8_t>> incoming;
};

inline void closeNvSciIpcChannel(NvSciIpcChannel& channel) noexcept
{
    if (channel.endpoint != 0)
    {
        NvSciIpcCloseEndpointSafe(channel.endpoint, false);
    }
    channel = NvSciIpcChannel{};
}

//! @return Whether @p endpointName opened and reset; leaves @p channel closed on failure.
inline bool tryOpenNvSciIpcChannel(NvSciIpcChannel& channel, std::string const& endpointName)
{
    closeNvSciIpcChannel(channel);
    channel.name = endpointName;
    if (NvSciIpcOpenEndpoint(channel.name.c_str(), &channel.endpoint) != NvSciError_Success)
    {
        closeNvSciIpcChannel(channel);
        return false;
    }
    if (NvSciIpcGetLinuxEventFd(channel.endpoint, &channel.eventFd) != NvSciError_Success)
    {
        closeNvSciIpcChannel(channel);
        return false;
    }
    NvSciIpcEndpointInfo info{};
    if (NvSciIpcGetEndpointInfo(channel.endpoint, &info) != NvSciError_Success)
    {
        closeNvSciIpcChannel(channel);
        return false;
    }
    channel.frame.resize(info.frame_size);
    if (NvSciIpcResetEndpointSafe(channel.endpoint) != NvSciError_Success)
    {
        closeNvSciIpcChannel(channel);
        return false;
    }
    return true;
}

//! Open @p endpointName into @p channel and reset it, which is what a peer waits to see.
inline void openNvSciIpcChannel(NvSciIpcChannel& channel, std::string const& endpointName)
{
    if (!tryOpenNvSciIpcChannel(channel, endpointName))
    {
        throw std::runtime_error("NvSciIpcOpenEndpoint " + endpointName);
    }
}

//! Non-blocking: track connect / reset, read arrived blobs into @c incoming, write queued @c outgoing.
inline void exchangeNvSciIpcBlobs(NvSciIpcChannel& channel)
{
    uint32_t events = 0;
    checkNvSci(NvSciIpcGetEventSafe(channel.endpoint, &events), "NvSciIpcGetEventSafe " + channel.name);
    if ((events & NV_SCI_IPC_EVENT_CONN_RESET) != 0U && channel.established)
    {
        channel.peerGone = true;
        channel.established = false;
        return;
    }
    channel.established = channel.established || (events & NV_SCI_IPC_EVENT_CONN_EST_ALL) != 0U;
    if (!channel.established)
    {
        return;
    }

    uint32_t bytes = 0;
    NvSciError status = NvSciError_Success;
    auto const frameBytes = static_cast<uint32_t>(channel.frame.size());
    while (
        (status = NvSciIpcReadSafe(channel.endpoint, channel.frame.data(), frameBytes, &bytes)) == NvSciError_Success)
    {
        // Clamped to the frame, so a bad length cannot reach past it.
        auto const blobBytes
            = std::min<size_t>(readDesc<uint32_t>(channel.frame.data(), bytes), bytes > 4U ? bytes - 4U : 0U);
        channel.incoming.emplace_back(channel.frame.begin() + 4, channel.frame.begin() + 4 + blobBytes);
    }
    if (status != NvSciError_InsufficientMemory)
    {
        checkNvSci(status, "NvSciIpcReadSafe " + channel.name);
    }

    while (!channel.outgoing.empty())
    {
        std::vector<uint8_t> const& blob = channel.outgoing.front();
        auto const blobBytes = static_cast<uint32_t>(blob.size());
        std::memcpy(channel.frame.data(), &blobBytes, 4U);
        std::memcpy(channel.frame.data() + 4, blob.data(), blob.size());
        status = NvSciIpcWriteSafe(channel.endpoint, channel.frame.data(), blobBytes + 4U, &bytes);
        if (status == NvSciError_InsufficientMemory)
        {
            return;
        }
        checkNvSci(status, "NvSciIpcWriteSafe " + channel.name);
        channel.outgoing.pop_front();
    }
}

//! Queue one blob; it is sent by this or a later exchangeNvSciIpcBlobs call.
inline void sendNvSciIpcBlob(NvSciIpcChannel& channel, std::vector<uint8_t> blob)
{
    if (blob.size() + 4U > channel.frame.size())
    {
        throw std::runtime_error("blob of " + std::to_string(blob.size()) + " bytes does not fit one "
            + std::to_string(channel.frame.size()) + "-byte frame of NvSciIpc endpoint " + channel.name);
    }
    channel.outgoing.push_back(std::move(blob));
}

//! @return The next blob to have arrived, or nothing when none has.
inline std::optional<std::vector<uint8_t>> receiveNvSciIpcBlob(NvSciIpcChannel& channel)
{
    if (channel.incoming.empty())
    {
        return std::nullopt;
    }
    std::vector<uint8_t> blob = std::move(channel.incoming.front());
    channel.incoming.pop_front();
    return blob;
}

} // namespace edgellm::ipc
