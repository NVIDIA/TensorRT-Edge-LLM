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

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace trt_edgellm
{
namespace math
{

namespace detail
{

/*!
 * @brief Type trait to check if T is a standard integer type (excluding bool and character types)
 *
 * This matches the C++20 definition of integer types for std::cmp_less, which includes
 * signed/unsigned variants of char, short, int, long, and long long, but excludes
 * bool, char, wchar_t, char16_t, and char32_t.
 */
template <typename T>
struct IsStandardInteger
    : std::bool_constant<std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>
          && !std::is_same_v<std::remove_cv_t<T>, char> && !std::is_same_v<std::remove_cv_t<T>, wchar_t>
          && !std::is_same_v<std::remove_cv_t<T>, char16_t> && !std::is_same_v<std::remove_cv_t<T>, char32_t>>
{
};

template <typename T>
inline constexpr bool kIsStandardInteger = IsStandardInteger<T>::value;

/*!
 * @brief Implementation for comparing two signed integers
 */
template <typename S, typename T>
constexpr bool lessImpl(S s, T t, std::true_type /* sSigned */, std::true_type /* tSigned */) noexcept
{
    return s < t;
}

/*!
 * @brief Implementation for comparing two unsigned integers
 */
template <typename S, typename T>
constexpr bool lessImpl(S s, T t, std::false_type /* sSigned */, std::false_type /* tSigned */) noexcept
{
    return s < t;
}

/*!
 * @brief Implementation for comparing signed S with unsigned T
 *
 * If s is negative, it is always less than any unsigned value.
 * Otherwise, we safely cast s to its unsigned counterpart and compare.
 * The comparison between potentially different-width unsigned types
 * uses standard integral promotion rules (smaller type widens to larger).
 */
template <typename S, typename T>
constexpr bool lessImpl(S s, T t, std::true_type /* sSigned */, std::false_type /* tSigned */) noexcept
{
    using UnsignedS = std::make_unsigned_t<S>;
    return s < 0 || static_cast<UnsignedS>(s) < t;
}

/*!
 * @brief Implementation for comparing unsigned S with signed T
 *
 * If t is negative or zero, then unsigned s cannot be less than t
 * (since s >= 0 by definition of unsigned). If t is positive,
 * we safely cast t to its unsigned counterpart and compare.
 */
template <typename S, typename T>
constexpr bool lessImpl(S s, T t, std::false_type /* sSigned */, std::true_type /* tSigned */) noexcept
{
    using UnsignedT = std::make_unsigned_t<T>;
    return t > 0 && s < static_cast<UnsignedT>(t);
}

} // namespace detail

/*!
 * @brief Safe integer comparison that handles signed/unsigned comparisons correctly
 *
 * This function provides the same semantics as std::cmp_less from C++20, but
 * implemented using only C++17 features. It correctly compares integers of
 * different signedness without triggering undefined behavior or compiler warnings.
 *
 * Unlike the built-in < operator, this function:
 * - Returns true if s is mathematically less than t
 * - Handles signed/unsigned comparisons without sign conversion issues
 * - Avoids undefined behavior from signed overflow
 * - Does not trigger -Wsign-compare, -Wtype-limits, or -Wnarrowing warnings
 *
 * @tparam S First integer type (must be a standard integer type)
 * @tparam T Second integer type (must be a standard integer type)
 * @param s First value to compare
 * @param t Second value to compare
 * @return true if s is mathematically less than t, false otherwise
 *
 * @note This function is constexpr and noexcept
 *
 * Example usage:
 * @code
 *   int signedVal = -1;
 *   unsigned int unsignedVal = 1;
 *
 *   // Built-in comparison gives unexpected result due to signed-to-unsigned conversion:
 *   // signedVal < unsignedVal  =>  false (because -1 becomes UINT_MAX)
 *
 *   // Safe comparison gives mathematically correct result:
 *   // less(signedVal, unsignedVal)  =>  true (because -1 < 1)
 * @endcode
 */
template <typename S, typename T,
    std::enable_if_t<detail::kIsStandardInteger<S> && detail::kIsStandardInteger<T>, int> = 0>
constexpr bool less(S s, T t) noexcept
{
    return detail::lessImpl(s, t, std::is_signed<S>{}, std::is_signed<T>{});
}

/*!
 * @brief Exception thrown when an integer cast would result in overflow
 *
 * This exception is thrown by the cast() function when the source value
 * exceeds the maximum representable value of the target type.
 */
class OverflowError : public std::runtime_error
{
public:
    explicit OverflowError(char const* message)
        : std::runtime_error(message)
    {
    }

    explicit OverflowError(std::string const& message)
        : std::runtime_error(message)
    {
    }
};

/*!
 * @brief Exception thrown when an integer cast would result in underflow
 *
 * This exception is thrown by the cast() function when the source value
 * is less than the minimum representable value of the target type.
 */
class UnderflowError : public std::runtime_error
{
public:
    explicit UnderflowError(char const* message)
        : std::runtime_error(message)
    {
    }

    explicit UnderflowError(std::string const& message)
        : std::runtime_error(message)
    {
    }
};

namespace detail
{

/*!
 * @brief Check if value s can be represented by type T without underflow
 *
 * Returns true if s >= std::numeric_limits<T>::min()
 */
template <typename T, typename S>
constexpr bool isInLowerBound(S s) noexcept
{
    // s >= T_MIN is equivalent to !(s < T_MIN)
    return !less(s, std::numeric_limits<T>::min());
}

/*!
 * @brief Check if value s can be represented by type T without overflow
 *
 * Returns true if s <= std::numeric_limits<T>::max()
 */
template <typename T, typename S>
constexpr bool isInUpperBound(S s) noexcept
{
    // s <= T_MAX is equivalent to !(T_MAX < s)
    return !less(std::numeric_limits<T>::max(), s);
}

} // namespace detail

/*!
 * @brief Safe integer cast with overflow and underflow checking
 *
 * Converts a value of type S to type T, throwing an exception if the
 * conversion would result in overflow or underflow. This function
 * correctly handles signed/unsigned conversions and different integer widths.
 *
 * @tparam T Target integer type (must be explicitly specified)
 * @tparam S Source integer type (deduced from argument)
 * @param s Value to convert
 * @return The value converted to type T
 *
 * @throws UnderflowError if s < std::numeric_limits<T>::min()
 * @throws OverflowError if s > std::numeric_limits<T>::max()
 *
 * Example usage:
 * @code
 *   int64_t bigValue = 1000;
 *   int8_t smallValue = cast<int8_t>(bigValue);  // throws OverflowError
 *
 *   int signedValue = -1;
 *   unsigned int unsignedValue = cast<unsigned int>(signedValue);  // throws UnderflowError
 *
 *   int safeValue = 42;
 *   int8_t result = cast<int8_t>(safeValue);  // OK, result = 42
 * @endcode
 */
template <typename T, typename S,
    std::enable_if_t<detail::kIsStandardInteger<T> && detail::kIsStandardInteger<S>, int> = 0>
constexpr T cast(S s)
{
    if (!detail::isInLowerBound<T>(s))
    {
        throw UnderflowError("integer cast underflow: value below target type minimum");
    }
    if (!detail::isInUpperBound<T>(s))
    {
        throw OverflowError("integer cast overflow: value exceeds target type maximum");
    }
    return static_cast<T>(s);
}

//! @brief Smallest power of two that is >= @p value.
//!
//! Ring buffers size themselves this way so an index can be reduced with a mask instead of a
//! modulo. C++20 would spell this std::bit_ceil; this project builds as C++17.
//!
//! @throws OverflowError if the result is not representable in int32_t.
inline int32_t ceilToPowerOfTwo(int32_t value)
{
    if (value <= 1)
    {
        return 1;
    }
    constexpr int32_t kMaxPowerOfTwo = 1 << 30;
    if (value > kMaxPowerOfTwo)
    {
        throw OverflowError("ceilToPowerOfTwo: no power of two of this size fits in int32_t");
    }
    int32_t rounded = 1;
    while (rounded < value)
    {
        rounded <<= 1;
    }
    return rounded;
}

} // namespace math
} // namespace trt_edgellm
