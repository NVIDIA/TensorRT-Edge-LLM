# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""EAGLE-4 draft runtime configuration."""


def update_llm_config(runtime_config, root, cfg, args) -> None:
    """Record the draft's ``eagle3_config`` as the ONNX exporter does."""
    del root, args
    eagle3_config = {
        "target_layer_ids": list(cfg.eagle3_target_layer_ids),
        "num_target_layers": len(cfg.eagle3_target_layer_ids),
    }
    if cfg.eagle_fc_norm:
        eagle3_config["fc_norm"] = True
    if cfg.eagle_norm_output:
        eagle3_config["norm_output"] = True
    if cfg.draft_candidate_count > 0:
        eagle3_config["candidate_head"] = {
            "type": "low_rank",
            "rank": cfg.draft_candidate_rank,
            "num_candidates": cfg.draft_candidate_count,
        }
    runtime_config["eagle3_config"] = eagle3_config
