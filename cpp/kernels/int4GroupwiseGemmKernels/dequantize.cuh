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

/*
 * Copyright (c) 2023 MIT HAN Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * reference: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization_new/dequantize.cuh
 */

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Dequantize signed 4-bit integers to FP16x2 format
//!
//! Converts packed 4-bit signed integers to FP16 (half precision) values using optimized PTX instructions.
//! This function extracts 8 4-bit values from the source and produces 8 FP16 values in the result.
//! The dequantization maps the 4-bit values to the range [-8, 7] in FP16 representation.
//!
//! \param[in] source Input packed 4-bit signed integers as half2
//! \param[out] result Output FP16 values stored in uint4 (8 half values)
__inline__ __device__ void dequantize_s4_to_fp16x2(half2 const& source, uint4* result)
{

    uint32_t* h = reinterpret_cast<uint32_t*>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    //! First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    //! Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    //! format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    //! In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    //! elt_67 to fp16 without having to shift them to the bottom bits before hand.

    //! Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    //! immediately before required.
    uint32_t const top_i4s = i4s >> 8;
    //! Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    //! Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    //! Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2])
        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    //! Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    //! I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    //! half2 ctor. In this case, I chose performance reliability over code readability.

    //! This is the half2 {1032, 1032} represented as an integer. We need to map value to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    //! Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    //! static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    //! This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    //! This is the half2 {-72, -72} represented as an integer.
    //! Use NEG_72 mapping the quantized weights to [-8,7]
    static constexpr uint32_t NEG_72 = 0xd480d480;
    //! static constexpr uint32_t NEG_64 = 0xd400d400;

    //! Finally, we construct the output numbers.
    //! Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    //! Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    //! Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    //! Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
}

} // namespace kernel
} // namespace trt_edgellm