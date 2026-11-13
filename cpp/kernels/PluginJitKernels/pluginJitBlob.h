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

#include "common/stringUtils.h"
#include "pluginJitCubin.h"
#include "pluginJitKey.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{

//! Identity of one plugin's serialized JIT blob. `name` prefixes every error message. `magic` names
//! both the plugin and its record layout: it opens every blob and is checked before anything else is
//! read, so a blob from another plugin or from a library with a different layout is refused here
//! instead of misread. Change it together with the record layout; the digest over key words and
//! cubin catches whatever a stale magic would let through.
struct PluginJitBlobFormat
{
    char const* name;
    uint32_t magic;
};

//! Little-endian blob writer: the layout never depends on host byte order or struct padding, which
//! keeps engine bytes reproducible.
class PluginJitBlobWriter
{
public:
    explicit PluginJitBlobWriter(PluginJitBlobFormat const& format) noexcept
        : mFormat(format)
    {
    }

    PluginJitBlobFormat const& format() const noexcept
    {
        return mFormat;
    }

    void appendHeader()
    {
        appendU32(mFormat.magic);
    }

    void appendU32(uint32_t const value)
    {
        for (int32_t shift = 0; shift < 32; shift += 8)
        {
            mBlob.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
        }
    }

    void appendU64(uint64_t const value)
    {
        for (int32_t shift = 0; shift < 64; shift += 8)
        {
            mBlob.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
        }
    }

    void appendBytes(std::vector<uint8_t> const& bytes)
    {
        mBlob.insert(mBlob.end(), bytes.begin(), bytes.end());
    }

    std::vector<uint8_t> release() noexcept
    {
        return std::move(mBlob);
    }

private:
    PluginJitBlobFormat mFormat;
    std::vector<uint8_t> mBlob;
};

//! Bounds-checked little-endian reader over a caller-owned byte range.
//! \throws std::invalid_argument for a null blob, std::runtime_error for a malformed one.
class PluginJitBlobReader
{
public:
    PluginJitBlobReader(PluginJitBlobFormat const& format, void const* const data, size_t const size)
        : mFormat(format)
        , mCursor(static_cast<uint8_t const*>(data))
        , mRemaining(size)
    {
        if (data == nullptr)
        {
            throw std::invalid_argument(std::string(mFormat.name) + " blob must not be null");
        }
    }

    PluginJitBlobFormat const& format() const noexcept
    {
        return mFormat;
    }

    void readHeader()
    {
        uint32_t const magic = readU32();
        if (magic != mFormat.magic)
        {
            throw std::runtime_error(format::fmtstr(
                "%s blob does not start with the expected magic word (found 0x%08X); the engine was built by an "
                "incompatible plugin library",
                mFormat.name, magic));
        }
    }

    uint32_t readU32()
    {
        require(sizeof(uint32_t));
        uint32_t value{0};
        for (int32_t shift = 0; shift < 32; shift += 8)
        {
            value |= static_cast<uint32_t>(*mCursor++) << shift;
        }
        mRemaining -= sizeof(uint32_t);
        return value;
    }

    uint64_t readU64()
    {
        require(sizeof(uint64_t));
        uint64_t value{0};
        for (int32_t shift = 0; shift < 64; shift += 8)
        {
            value |= static_cast<uint64_t>(*mCursor++) << shift;
        }
        mRemaining -= sizeof(uint64_t);
        return value;
    }

    std::vector<uint8_t> readBytes(size_t const size)
    {
        require(size);
        std::vector<uint8_t> bytes(mCursor, mCursor + size);
        mCursor += size;
        mRemaining -= size;
        return bytes;
    }

    size_t remaining() const noexcept
    {
        return mRemaining;
    }

    //! Every byte must belong to the format; trailing bytes mean the blob is not what it claims.
    void finish() const
    {
        if (mRemaining != 0)
        {
            throw std::runtime_error(format::fmtstr("%s blob has %zu trailing byte(s)", mFormat.name, mRemaining));
        }
    }

private:
    void require(size_t const bytes) const
    {
        if (mRemaining < bytes)
        {
            throw std::runtime_error(std::string("Truncated ") + mFormat.name + " blob");
        }
    }

    PluginJitBlobFormat mFormat;
    uint8_t const* mCursor;
    size_t mRemaining;
};

//! One kernel record: key words, cubin size, digest, cubin bytes. The digest is verified against the
//! payload before anything is written, so a kernel whose cubin changed after compilation cannot be
//! serialized.
//! \throws std::invalid_argument on a non-ELF or oversized cubin or a stale digest
template <typename Key>
void appendPluginJitKernelRecord(PluginJitBlobWriter& writer, PluginJitKernel<Key> const& kernel)
{
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size())
        || kernel.cubin.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(
            std::string(writer.format().name) + " payload must be a native ELF cubin with 32-bit size");
    }
    if (computePluginJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size()) != kernel.digest)
    {
        throw std::invalid_argument(std::string(writer.format().name) + " kernel digest does not match its payload");
    }
    for (uint32_t const word : kernel.key.keyWords())
    {
        writer.appendU32(word);
    }
    writer.appendU32(static_cast<uint32_t>(kernel.cubin.size()));
    writer.appendU64(kernel.digest.lo);
    writer.appendU64(kernel.digest.hi);
    writer.appendBytes(kernel.cubin);
}

//! Inverse of appendPluginJitKernelRecord. Structural only: verifyPluginJitKernelRecord checks the
//! payload once the whole blob has been consumed, so truncation is reported as such rather than as a
//! digest mismatch.
template <typename Key>
PluginJitKernel<Key> readPluginJitKernelRecord(PluginJitBlobReader& reader)
{
    std::array<uint32_t, Key::kNumWords> words{};
    for (uint32_t& word : words)
    {
        word = reader.readU32();
    }
    PluginJitKernel<Key> kernel;
    kernel.key = Key::fromKeyWords(words);
    uint32_t const cubinSize = reader.readU32();
    kernel.digest.lo = reader.readU64();
    kernel.digest.hi = reader.readU64();
    if (cubinSize == 0)
    {
        throw std::runtime_error(std::string(reader.format().name) + " blob carries an empty cubin");
    }
    kernel.cubin = reader.readBytes(cubinSize);
    return kernel;
}

//! \throws std::runtime_error when the cubin is not a native ELF image or the digest does not match
template <typename Key>
void verifyPluginJitKernelRecord(PluginJitBlobFormat const& format, PluginJitKernel<Key> const& kernel)
{
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size()))
    {
        throw std::runtime_error(std::string(format.name) + " blob does not contain a native ELF cubin");
    }
    if (computePluginJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size()) != kernel.digest)
    {
        throw std::runtime_error(std::string(format.name) + " blob digest mismatch");
    }
}

//! Blob holding exactly one kernel: header, record.
template <typename Key>
std::vector<uint8_t> serializePluginJitKernel(PluginJitBlobFormat const& format, PluginJitKernel<Key> const& kernel)
{
    PluginJitBlobWriter writer(format);
    writer.appendHeader();
    appendPluginJitKernelRecord(writer, kernel);
    return writer.release();
}

//! Inverse of serializePluginJitKernel. The caller validates the key it gets back.
template <typename Key>
PluginJitKernel<Key> deserializePluginJitKernel(
    PluginJitBlobFormat const& format, void const* const data, size_t const size)
{
    PluginJitBlobReader reader(format, data, size);
    reader.readHeader();
    PluginJitKernel<Key> kernel = readPluginJitKernelRecord<Key>(reader);
    reader.finish();
    verifyPluginJitKernelRecord(format, kernel);
    return kernel;
}

//! Blob holding a list of kernels: header, count, records.
template <typename Key>
std::vector<uint8_t> serializePluginJitKernels(
    PluginJitBlobFormat const& format, std::vector<PluginJitKernel<Key>> const& kernels)
{
    if (kernels.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::string(format.name) + " blob cannot hold that many kernels");
    }
    PluginJitBlobWriter writer(format);
    writer.appendHeader();
    writer.appendU32(static_cast<uint32_t>(kernels.size()));
    for (PluginJitKernel<Key> const& kernel : kernels)
    {
        appendPluginJitKernelRecord(writer, kernel);
    }
    return writer.release();
}

//! Inverse of serializePluginJitKernels. The caller validates the keys it gets back.
template <typename Key>
std::vector<PluginJitKernel<Key>> deserializePluginJitKernels(
    PluginJitBlobFormat const& format, void const* const data, size_t const size)
{
    PluginJitBlobReader reader(format, data, size);
    reader.readHeader();
    uint32_t const count = reader.readU32();
    std::vector<PluginJitKernel<Key>> kernels;
    for (uint32_t index = 0; index < count; ++index)
    {
        kernels.push_back(readPluginJitKernelRecord<Key>(reader));
    }
    reader.finish();
    for (PluginJitKernel<Key> const& kernel : kernels)
    {
        verifyPluginJitKernelRecord(format, kernel);
    }
    return kernels;
}

} // namespace trt_edgellm
