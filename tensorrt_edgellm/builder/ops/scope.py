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
"""Active symbolic build scope.

Model and op signatures carry tensors and ordinary attributes only. The
scope binds those calls to the component currently being compiled, analogous
to the trace owned by a Python compiler frontend.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .module import BuildContext

_CURRENT_CONTEXT: ContextVar["BuildContext | None"] = ContextVar(
    "edgellm_ops_context", default=None)


@contextmanager
def build_scope(context: "BuildContext") -> Iterator[None]:
    """Make ``context`` available to nested symbolic operations."""
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_context() -> "BuildContext":
    """Return the active component context or report a misplaced op call."""
    context = _CURRENT_CONTEXT.get()
    if context is None:
        raise RuntimeError("operation called outside a Module call")
    return context


def current_net():
    """Return the internal TensorRT graph builder for the active scope."""
    return current_context().net
