/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace trt_edgellm
{
namespace file_io
{

/*!
 * @brief Memory-mapped file reader
 *
 * Provides efficient file reading using memory mapping (mmap).
 * The file contents are mapped directly into memory without copying.
 */
class MmapReader
{
public:
    //! @brief Default constructor
    MmapReader();

    /*!
     * @brief Construct and load file
     * @param fp Path to file to load
     */
    explicit MmapReader(std::filesystem::path const& fp);

    //! @brief Deleted copy constructor
    MmapReader(MmapReader const&) = delete;

    //! @brief Deleted copy assignment operator
    MmapReader& operator=(MmapReader const&) = delete;

    //! @brief Destructor
    ~MmapReader();

    //! @brief Release mapped memory
    void release();

    /*!
     * @brief Load and memory-map a file
     * @param fp Path to file to load
     * @return True on success, false on failure
     */
    bool loadFile(std::filesystem::path const& fp);

    //! @brief Get mapped data as byte array
    //! @return Const pointer to byte data
    int8_t const* getByteData() const noexcept;

    //! @brief Get mapped data as void pointer
    //! @return Const pointer to data
    void const* getData() const noexcept;

    //! @brief Get size of mapped data
    //! @return Size in bytes
    size_t getSize() const noexcept;

private:
    void* mData;   //!< Pointer to mapped memory
    size_t mBytes; //!< Size of mapped data in bytes
};

} // namespace file_io
} // namespace trt_edgellm
