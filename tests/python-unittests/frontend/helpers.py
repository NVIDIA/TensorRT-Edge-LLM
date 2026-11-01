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
"""Shared settings and a serial fake backend for frontend tests."""

import signal

from experimental import server
from experimental.frontend import config
from experimental.server.runtime import engine


def settings(tmp_path, *options):
    return config.resolve(config.parser().parse_args([str(tmp_path),
                                                      *options]))


class FakeBackend:

    context_capacity = 8192

    def __init__(self, *_args):
        self.closed = False
        self.cancelled = False
        self.drained = False
        self.interrupt = False
        self.text = "answer"
        self.error = None

    def validate(self, _messages):
        if self.error:
            raise self.error
        return 3

    def stream(self, _messages, on_metrics):
        self.last_messages = _messages

        def iterate():
            try:
                if self.text:
                    yield engine.StreamDelta(text=self.text, token_ids=[1])
                if self.interrupt:
                    signal.raise_signal(signal.SIGINT)
            finally:
                self.drained = True
                on_metrics(
                    engine.GenerationMetrics(prompt_tokens=3,
                                             generated_tokens=1,
                                             time_to_first_token=0.1,
                                             total_time=0.2))

        def cancel():
            self.cancelled = True

        return server.CancellableIterator(iterate(), cancel)

    def close(self):
        self.closed = True
