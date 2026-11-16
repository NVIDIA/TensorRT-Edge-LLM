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

#include "common/tensor.h"
#include "runtime/config/inferenceDims.h"
#include "runtime/exec/tensorMap.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Engine tensor direction — input (runtime sets address + shape) or output (runtime sets address only).
enum class TensorIO : uint8_t
{
    kInput,
    kOutput,
};

/*!
 * @brief Declarative specification for a single tensor binding.
 *
 * Per-layer tensors use a printf-style @c %d placeholder in the name and
 * a positive @c perLayer count; the registry expands these into concrete
 * specs at query time.
 */
struct TensorSpec
{
    std::string name;                                          //!< Tensor name (may contain "%d" for per-layer)
    TensorIO io{TensorIO::kInput};                             //!< Binding direction
    trt_edgellm::DataType dtype{trt_edgellm::DataType::kHALF}; //!< Data type
    std::vector<ShapeDim> shape;                               //!< Shape template
    int32_t perLayer{0};                                       //!< >0: expand "%d" for 0..perLayer-1
};

/*!
 * @brief Central registry of all tensor specs needed by a TRT engine.
 *
 * TensorRegistry holds tensor specs with symbolic shape templates.
 * @c bindAll() resolves all shapes from InferenceDims and calls
 * @c setTensorAddress + @c setInputShape in a single pass, replacing many
 * manual binding calls.  Per-layer tensors (e.g. KV cache) are auto-expanded
 * from @c %d templates.
 */
class TensorRegistry
{
public:
    /*!
     * @brief Add a tensor specification to the registry.
     * @param spec The tensor specification to add
     */
    void addTensor(TensorSpec spec);

    /*!
     * @brief Bind all registered tensors to a TRT execution context.
     *
     * Resolves symbolic shapes from @p inferenceDims, then calls
     * @c setTensorAddress and (for inputs) @c setInputShape on each tensor.
     *
     * @param ctx TRT execution context
     * @param tensorMap Name-to-tensor mapping
     * @param inferenceDims Current symbolic dimension values
     * @return True if all tensors were bound successfully
     */
    bool bindAll(
        nvinfer1::IExecutionContext* ctx, TensorMap const& tensorMap, InferenceDims const& inferenceDims) const;

    /*!
     * @brief Return the set of `InferenceDims` members referenced by any symbolic
     * `ShapeDim` in the registry.
     *
     * Used by `EngineExecutor::prepare` to restrict its field-completeness assertion
     * to dims the engine actually consumes. Computed once (lazily) when the
     * expanded spec set is materialized; cached across calls.
     *
     * @return Vector of `int64_t InferenceDims::*` — deduplicated
     */
    std::vector<int64_t InferenceDims::*> const& referencedMembers() const;

    /*!
     * @brief Return all tensor names after per-layer expansion.
     * @return Vector of concrete tensor names
     */
    std::vector<std::string> allTensorNames() const;

    /*!
     * @brief Test whether a tensor name is registered (post-expansion).
     *
     * Used by `EngineExecutor::prepare` to skip the registry-managed bindings in its
     * fall-back loop: those are already covered by `bindAll`, so re-binding
     * them via the tensor map's actual TRTDims would override the
     * registry-computed shape.
     *
     * @param name Concrete tensor name (after `%d` expansion if applicable)
     * @return True if the registry has a spec for @p name
     */
    bool contains(std::string_view name) const;

    /*!
     * @brief Return all tensor specs after per-layer expansion.
     * @return Vector of expanded TensorSpec objects
     */
    std::vector<TensorSpec> const& allExpandedSpecs() const;

    /*!
     * @brief Resolve a shape template into concrete TRT Dims.
     *
     * Fixed dimensions are used as-is. Symbolic dimensions are looked up
     * from @p inferenceDims.
     *
     * @param shape Shape template with fixed and/or symbolic dims
     * @param inferenceDims Current symbolic dimension values
     * @return Resolved TRT Dims
     */
    nvinfer1::Dims resolveShape(std::vector<ShapeDim> const& shape, InferenceDims const& inferenceDims) const;

private:
    std::vector<TensorSpec> mSpecs; //!< Raw specs (may contain per-layer templates)

    mutable std::vector<TensorSpec> mExpandedSpecs;                   //!< Cached result of allExpandedSpecs()
    mutable std::vector<int64_t InferenceDims::*> mReferencedMembers; //!< Cached result of referencedMembers()
    mutable bool mDirty{true};                                        //!< True when caches must be rebuilt

    /*!
     * @brief Expand a single spec into one-or-more concrete specs.
     *
     * If @c spec.perLayer is positive, returns @c perLayer copies with
     * the @c %d placeholder replaced by @c 0..perLayer-1.  Otherwise
     * returns the spec unchanged.
     *
     * @param spec The spec to expand
     * @return Vector of expanded specs
     */
    std::vector<TensorSpec> expandPerLayer(TensorSpec const& spec) const;
};

} // namespace rt
} // namespace trt_edgellm
