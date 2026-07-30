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

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

//! Helpers for populating the tensor descriptors emitted by the CuTe DSL C exporter.
//!
//! Every AOT variant exports its own nominally distinct but layout-identical descriptor structs
//! (@c fmha_d64_Tensor_q_tensor_t vs @c fmha_d128_Tensor_q_tensor_t) plus a
//! @c cute_dsl_<variant>_wrapper entry point. There is no umbrella C type, so descriptor types are
//! recovered here from the signature of the wrapper that consumes them: a call site names only the
//! wrapper and the kernel module, and pairing a descriptor with the wrong variant is not
//! expressible.
//!
//! The exporter (@c cutlass/cute/export/c_header_generator.py) always names the members @c data,
//! @c dynamic_shapes and @c dynamic_strides, but emits each array only when its dynamic mask is
//! non-empty. A rank-1 descriptor therefore has no @c dynamic_strides member at all and needs
//! makeCuSeqLenTensor() rather than the strided builders below.
namespace trt_edgellm
{
namespace cutedsl
{

//! Type of the I-th parameter of a generated @c cute_dsl_<variant>_wrapper, with the pointer
//! stripped. Index 0 is the kernel module; the tensor descriptors follow in declaration order.
template <std::size_t I, class Wrapper>
struct WrapperArg;

// Compile-time type computation. The specialization pattern-matches the wrapper signature to
// capture its parameter list, and std::tuple is only a vehicle for indexing that pack -- no tuple
// is ever created, and nothing here survives into the generated code.
template <std::size_t I, class Ret, class... Args>
struct WrapperArg<I, Ret (*)(Args...)>
{
    using type = std::remove_pointer_t<std::tuple_element_t<I, std::tuple<Args...>>>;
};

template <std::size_t I, class Wrapper>
using WrapperArgT = typename WrapperArg<I, Wrapper>::type;

//! Parameter count of a generated wrapper, used to reject a wrapper from the wrong kernel family
//! before its descriptors get bound positionally.
template <class Wrapper>
struct WrapperArity;

template <class Ret, class... Args>
struct WrapperArity<Ret (*)(Args...)> : std::integral_constant<std::size_t, sizeof...(Args)>
{
};

//! Number of dynamic extents a generated descriptor carries.
template <class TensorT>
inline constexpr std::size_t kShapeRank = std::extent_v<decltype(TensorT::dynamic_shapes)>;

//! Populate a descriptor from explicit extents and strides. Use this whenever the strides are not
//! the packed product of the extents, e.g. when the physical layout is a permutation of the logical
//! shape the kernel expects.
//!
//! The descriptors only ever expose a non-const @c void* @c data, and the AOT kernels treat the
//! input pointers as read-only, so constness is cast away exactly once, here.
//! @p StrideRank is deduced from the stride list rather than spelled @c Rank-1, because a computed
//! bound is a non-deduced context: a caller passing too few strides would otherwise bind to the
//! larger array and have the missing entries silently value-initialised to zero.
template <class TensorT, std::size_t Rank, std::size_t StrideRank>
constexpr TensorT makeStridedTensor(
    void const* data, int32_t const (&shape)[Rank], int64_t const (&strides)[StrideRank])
{
    static_assert(Rank >= 2, "Rank-1 descriptors carry no dynamic_strides member; use makeCuSeqLenTensor().");
    static_assert(kShapeRank<TensorT> == Rank, "Extent list length does not match the generated descriptor rank.");
    static_assert(StrideRank == Rank - 1, "Expected exactly one stride fewer than extents.");
    static_assert(std::extent_v<decltype(TensorT::dynamic_strides)> == Rank - 1,
        "Stride list length does not match the generated descriptor rank.");
    static_assert(std::is_same_v<std::remove_extent_t<decltype(TensorT::dynamic_strides)>, int64_t>,
        "CuTe DSL now exports 32-bit strides (see _use_32bit_stride in c_header_generator.py); update this builder "
        "before the int64 stride arithmetic below silently truncates.");

    TensorT tensor{};
    tensor.data = const_cast<void*>(data);
    for (std::size_t i = 0; i < Rank; ++i)
    {
        tensor.dynamic_shapes[i] = shape[i];
    }
    for (std::size_t i = 0; i + 1 < Rank; ++i)
    {
        tensor.dynamic_strides[i] = strides[i];
    }
    return tensor;
}

//! Populate a row-major packed descriptor, i.e. one whose strides are the running product of the
//! trailing extents. The product is accumulated in int64 so a large batch cannot overflow.
template <class TensorT, std::size_t Rank>
constexpr TensorT makePackedTensor(void const* data, int32_t const (&shape)[Rank])
{
    static_assert(Rank >= 2, "Rank-1 descriptors carry no dynamic_strides member; use makeCuSeqLenTensor().");

    int64_t strides[Rank - 1]{};
    int64_t running = 1;
    for (std::size_t i = Rank - 1; i > 0; --i)
    {
        running *= shape[i];
        strides[i - 1] = running;
    }
    return makeStridedTensor<TensorT>(data, shape, strides);
}

//! Populate a rank-1 cumulative-sequence-length descriptor. The exporter emits no
//! @c dynamic_strides member for these, so they cannot go through the strided builders.
template <class TensorT>
constexpr TensorT makeCuSeqLenTensor(int32_t const* data, int32_t length)
{
    static_assert(kShapeRank<TensorT> == 1, "Cumulative sequence length descriptors must be rank 1.");

    TensorT tensor{};
    tensor.data = const_cast<int32_t*>(data);
    tensor.dynamic_shapes[0] = length;
    return tensor;
}

} // namespace cutedsl
} // namespace trt_edgellm
