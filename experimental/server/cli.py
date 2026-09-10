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
"""Command-line entry point for the Edge-LLM OpenAI server."""

import logging

from .api.app import run_http_server
from .config import ServerConfigError, parse_server_config
from .runtime.engine import load_model
from .runtime.engine_client import EngineClient


def main() -> None:
    try:
        config = parse_server_config()
    except ServerConfigError as exc:
        raise SystemExit(f"invalid server configuration: {exc}") from exc

    logging.basicConfig(
        level=getattr(logging, config.api.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    llm = load_model(**config.model.llm_kwargs())
    client = EngineClient(llm, config.api)
    logging.getLogger("edgellm.server").info(
        "Loaded model=%s max_model_len=%s kv_cache_dtype=%s "
        "speculative_decoding=%s context_reuse=%s",
        client.model_name,
        client.capabilities.max_model_len,
        client.capabilities.kv_cache_dtype,
        client.capabilities.speculative_decoding,
        client.capabilities.context_reuse,
    )
    run_http_server(client, config.api)
