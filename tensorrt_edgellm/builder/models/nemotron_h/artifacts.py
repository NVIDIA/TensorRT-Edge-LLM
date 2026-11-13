# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nemotron-H runtime artifact writing."""

from ...core import contracts
from ...core.artifacts.embeddings import write_cached_draft_embedding
from ...core.artifacts.runtime_artifacts import write_runtime_artifacts
from . import weights


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    if config is None:
        raise ValueError("Nemotron-H artifacts require an LLM configuration")
    write_runtime_artifacts(config,
                            args,
                            engine_dir,
                            weight_conversion=weights)
    if (args.resolved_spec_role == contracts.SpecRole.BASE
            and args.spec_type in ("dflash", "dspark")):
        mask_token_id = (config.dflash_mask_token_id if args.spec_type
                         == "dflash" else config.dspark_mask_token_id)
        output_dir = contracts.component_spec(
            args.resolved_component).output_dir(engine_dir)
        write_cached_draft_embedding(output_dir, args.draft_model_dir,
                                     mask_token_id, config.embedding_scale)
