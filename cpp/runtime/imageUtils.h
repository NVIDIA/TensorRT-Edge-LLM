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

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

/*!
 * @brief Image data container
 *
 * Wraps image tensor loaded with stbi. Expected shape: [height, width, channels].
 * Channels is typically 3 for RGB images.
 */
class ImageData
{
public:
    std::shared_ptr<rt::Tensor> buffer; //!< Image data buffer
    int64_t width;                      //!< Image width
    int64_t height;                     //!< Image height
    int64_t channels;                   //!< Number of channels (e.g., 3 for RGB)

    /*!
     * @brief Default constructor (creates uninitialized ImageData)
     */
    ImageData() = default;

    /*!
     * @brief Construct image data
     * @param data Image tensor
     */
    ImageData(rt::Tensor&& data);

    //! @brief Get raw image data pointer
    //! @return Pointer to image data
    unsigned char* data() const;
};

/*!
 * @brief Load image from file
 * @param path Path to image file
 * @return Loaded image data
 */
ImageData loadImageFromFile(std::string const& path);

/*!
 * @brief Load image from memory
 * @param data Pointer to image data in memory
 * @param size Size of image data in bytes
 * @return Loaded image data
 */
ImageData loadImageFromMemory(unsigned char const* data, size_t size);

/*!
 * @brief Resize image into pre-allocated buffer
 * @param image Source image
 * @param resizedImage Output buffer (will be reshaped to target dimensions)
 * @param newWidth Target width
 * @param newHeight Target height
 */
void resizeImage(ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm