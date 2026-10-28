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
"""Keep Python and native diagnostics off the response stream."""

import contextlib
import ctypes
import logging
import os
import sys


@contextlib.contextmanager
def diagnostics(verbose=False):
    """Route process stdout to stderr while yielding a response-only stream.

    The CLI owns the process. Duplicating stdout before redirecting fd 1 also
    captures native code and child processes that bypass Python's sys.stdout.
    """
    flush_native = ctypes.CDLL(None).fflush
    flush_native.argtypes = [ctypes.c_void_p]
    flush_native.restype = ctypes.c_int
    level = "DEBUG" if verbose else "WARNING"
    previous_level = os.environ.get("EDGELLM_LOG_LEVEL")
    os.environ["EDGELLM_LOG_LEVEL"] = level
    root = logging.getLogger()
    previous_handlers, previous_python_level = root.handlers[:], root.level
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.handlers = [handler]
    root.setLevel(level)
    sys.stdout.flush()
    flush_native(None)
    saved_stdout = os.dup(1)
    try:
        with os.fdopen(os.dup(saved_stdout),
                       "w",
                       encoding="utf-8",
                       buffering=1) as responses:
            os.dup2(2, 1)
            with contextlib.redirect_stdout(sys.stderr):
                yield responses
    finally:
        # Native stdio buffers must still see the redirected file descriptor.
        flush_native(None)
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        root.handlers, root.level = previous_handlers, previous_python_level
        handler.close()
        if previous_level is None:
            os.environ.pop("EDGELLM_LOG_LEVEL", None)
        else:
            os.environ["EDGELLM_LOG_LEVEL"] = previous_level
