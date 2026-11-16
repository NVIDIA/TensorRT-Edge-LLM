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

#include "runtime/multiStage/stageRegistry.h"

#include "multimodal/minimax_h3/minimaxH3Stages.h"
#include "multimodal/qwen3_omni/code2WavStage.h"
#include "multimodal/qwen3_omni/talkerRunner.h"
#include "runtime/llmRunner.h"

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

/*!
 * @brief Every runner this library ships, its declaration next to its factory.
 *
 * The only translation unit that names a concrete runner. Its references are also what keep each
 * runner's object file in a static link.
 */
StageRegistry const& builtinRunners()
{
    static StageRegistry const sRunners{
        {llmRunnerDeclaration(), &createLlmRunner},
        {qwen3_omni::talkerRunnerDeclaration(), &qwen3_omni::createTalkerRunner},
        {qwen3_omni::code2WavDeclaration(), &qwen3_omni::createCode2WavStage},
        {minimax_h3::textEncoderDeclaration(), &minimax_h3::createTextEncoderStage},
        {minimax_h3::denoiserDeclaration(), &minimax_h3::createDenoiserStage},
        {minimax_h3::audioDecoderDeclaration(), &minimax_h3::createAudioDecoderStage},
        {minimax_h3::videoDecoderDeclaration(), &minimax_h3::createVideoDecoderStage},
    };
    return sRunners;
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
