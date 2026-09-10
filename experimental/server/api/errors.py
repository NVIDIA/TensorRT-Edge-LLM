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
"""Structured errors shared by the HTTP and engine-client layers."""

from http import HTTPStatus
from typing import Any, Dict, Optional


class ServerError(Exception):
    """An error that can be returned through the OpenAI error schema."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_type = error_type
        self.param = param

    def payload(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.status_code,
            }
        }


class InvalidRequestError(ServerError):
    pass


class ModelNotFoundError(ServerError):

    def __init__(self,
                 message: str,
                 *,
                 param: Optional[str] = "model") -> None:
        super().__init__(message,
                         status_code=HTTPStatus.NOT_FOUND,
                         error_type="not_found_error",
                         param=param)


class UnsupportedFeatureError(ServerError):

    def __init__(self, message: str, *, param: Optional[str] = None) -> None:
        super().__init__(message,
                         status_code=HTTPStatus.BAD_REQUEST,
                         error_type="unsupported_feature",
                         param=param)


class ServerOverloadedError(ServerError):

    def __init__(self, message: str = "server request queue is full") -> None:
        super().__init__(message,
                         status_code=HTTPStatus.TOO_MANY_REQUESTS,
                         error_type="server_overloaded")


class ServerUnavailableError(ServerError):

    def __init__(self, message: str = "server is shutting down") -> None:
        super().__init__(message,
                         status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                         error_type="server_unavailable")


class EngineError(ServerError):

    def __init__(self, message: str) -> None:
        if "EDGELLM_INPUT_TOO_LONG" in message:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            error_type = "invalid_request_error"
        elif "EDGELLM_BAD_MEDIA_COUNT" in message:
            status = HTTPStatus.BAD_REQUEST
            error_type = "invalid_request_error"
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            error_type = "engine_error"
        super().__init__(message, status_code=status, error_type=error_type)


class PayloadTooLargeError(ServerError):

    def __init__(self, message: str, *, param: Optional[str] = None) -> None:
        super().__init__(message,
                         status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                         error_type="invalid_request_error",
                         param=param)
