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
"""RoboLab policy-server wrapper for the Edge-LLM Cosmos3 policy.

Exposes the observation -> action-chunk contract over HTTP + JSON and over the
OpenPI WebSocket protocol used by RoboLab's official Cosmos3 client. See
``policy_server.py`` for the backend, ``openpi_server.py`` for the parity
transport, and ``selftest.py`` for a local mock self-test without Isaac Sim.
"""
