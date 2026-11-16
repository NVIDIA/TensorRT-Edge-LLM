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

#include "runtime/multiStage/multiStageRuntime.h"

#include "common/checkMacros.h"
#include "runtime/multiStage/stageConfig.h"

#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

std::vector<std::string> Sinks::attachedTerminals(MultiStageConfig const& config) const
{
    std::vector<std::string> attached;
    attached.reserve(config.terminals.size());

    for (auto const& terminal : config.terminals)
    {
        std::string const where = "terminal '" + terminal + "'";
        SourceRef const ref = parseSourceRef(terminal);

        // validate() has already accepted these; the lookups are here because the port
        // kind decides which channel a caller may attach.
        auto const* producer = config.findStage(ref.stageId);
        ELLM_CHECK(producer != nullptr, where + ": unknown stage '" + ref.stageId + "'");
        auto const* port = producer->findPort(ref.portId);
        ELLM_CHECK(port != nullptr, where + ": stage '" + ref.stageId + "' does not emit port '" + ref.portId + "'");

        bool haveChannel = false;
        switch (port->kind)
        {
        case PortKind::kText: haveChannel = text != nullptr; break;
        case PortKind::kAudio: haveChannel = audio != nullptr; break;
        case PortKind::kImage: haveChannel = video != nullptr; break;
        case PortKind::kCodec:
        case PortKind::kEmbeds:
        case PortKind::kHidden:
        case PortKind::kRequest:
            // Intermediate representations no caller can collect; validate() rejects such a
            // terminal at load.
            ELLM_CHECK(false,
                where + ": kind " + toString(port->kind) + " cannot carry a sink; only text, audio, and image can");
        }

        if (haveChannel)
        {
            attached.push_back(terminal);
        }
    }
    return attached;
}

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
