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

#include "modelInference.h"

#include "internvl3/internvl3VanillaInference.h"
#include "modelInferenceCatalog.h"

namespace trt_edgellm::safety
{

SafetyStatus createModelInference(ModelInferenceDefinition const& definition, llm::ModelLocalData const& data,
    std::vector<LMStageContext> const& contexts, llm::RequestOperations& llmOperations,
    vlm::RequestOperations& visualOperations, std::unique_ptr<ModelInference>& output) noexcept
{
    switch (definition.id)
    {
    case ModelInferenceId::kInternVL3_1B_Vlm_Vanilla:
        return InternVL3VanillaInference::create(definition, data, contexts, llmOperations, visualOperations, output);
    }
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, "Unsupported model inference recipe");
}

} // namespace trt_edgellm::safety
