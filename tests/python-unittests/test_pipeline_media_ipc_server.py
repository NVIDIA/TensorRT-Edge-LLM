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
"""Answer checks of the ``media_ipc_server`` model-case scenario."""

import pytest
from defs import test_media_ipc_server as media_ipc

# Client stdout and stderr merged, as the driver captures them.
_ANSWERED = """\
edgellm-ipc: grep server log for "client 4242:5"
The image shows a giant panda eating bamboo.

perf: requestId=1 client_round_trip_ms=812.40 server_inference_ms=812 isl=301 osl=42

A black and white Panda sits among green plants.

perf: requestId=2 client_round_trip_ms=790.10 server_inference_ms=789 isl=301 osl=40

2 requests sent and answered
"""

# yapf: disable
_UNANSWERED = {
    "request_missing": (_ANSWERED.split("A black")[0], "1 of 2 requests finished"),
    "no_tokens": (_ANSWERED.replace("osl=42", "osl=0"), "request 1 generated no tokens"),
    "second_answer_off_topic": (_ANSWERED.replace("Panda", "bear"), "request 2 answer lacks ['panda']"),
}
# yapf: enable


@pytest.mark.parametrize("output, message",
                         _UNANSWERED.values(),
                         ids=_UNANSWERED.keys())
def test_media_ipc_answer_check_needs_every_request_answered(output, message):
    """The phase fails unless each of the requests finished with tokens that name the media."""
    assert media_ipc.media_ipc_failures(_ANSWERED, ("panda", ), 2) == []

    failures = media_ipc.media_ipc_failures(output, ("panda", ), 2)
    assert message in failures, failures
