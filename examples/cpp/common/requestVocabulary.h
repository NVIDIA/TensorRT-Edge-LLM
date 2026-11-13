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

#include <nlohmann/json.hpp>
#include <string>
#include <tensorrt_edgellm/runtime/multiStage/stageRegistry.h>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace exampleUtils
{

/*!
 * @brief What a request file may set for the runners of a table, and how other front ends spell it.
 *
 * The union over every runner in the table, not only those one graph uses, so a request file reads the
 * same whichever graph runs it.
 */
struct RequestVocabulary
{
    std::vector<std::string> samplingKeys;
    std::vector<std::string> styleFields;
    std::vector<std::pair<std::string, std::string>> aliases; //!< {spelling, sampling key or input name}
};

RequestVocabulary vocabularyOf(rt::multiStage::StageRegistry const& runners);

/*!
 * @brief Fold the document-level sampling keys onto the per-request sampling object.
 *
 * llm_inference and qwen3_tts_inference input files put sampling at the document's top level
 * and apply it to every request. Layers, weakest first: @p cliSampling, the document's top level,
 * the request's own `sampling` object. An alias is read as its key, layer by layer, so a stronger
 * layer's alias beats a weaker layer's key, and within a layer the key wins. A key no runner reads
 * is reported and dropped.
 *
 * @throws std::runtime_error when @p cliSampling or the request's `sampling` is not a JSON object,
 *         naming which.
 */
nlohmann::json foldSampling(RequestVocabulary const& vocabulary, std::string const& cliSampling,
    nlohmann::json const& document, nlohmann::json const& entry);

//! The value @p entry gives @p name or an alias of it, else the one @p document gives; nullptr when neither does.
nlohmann::json const* findRequestField(RequestVocabulary const& vocabulary, std::string const& name,
    nlohmann::json const& document, nlohmann::json const& entry);

//! The request's own `style` object, with every style field it lacks taken, of any JSON type, from the request's or the
//! document's top level. An unparsable `style` counts as empty.
nlohmann::json foldStyle(
    RequestVocabulary const& vocabulary, nlohmann::json const& document, nlohmann::json const& entry);

} // namespace exampleUtils
} // namespace trt_edgellm
