<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Safety Coding Guidelines

Based on [AUTOSAR C++14, AP Release 19-03](https://www.autosar.org/fileadmin/standards/R22-11/AP/AUTOSAR_RS_CPP14Guidelines.pdf).

## Review Scope

- Apply only to changes under `cpp/safety_inference/**`.

## Friend Declarations — A11-3-1

- Do not use `friend` declarations for classes, functions, or templates.
- AUTOSAR permits comparison operators as friend functions (A13-5-5). Project policy requires maintainer approval before using this exception.

## Abrupt Termination — A15-5-2

- Do not abruptly terminate the program through explicit or implicit calls to `std::abort`, `std::quick_exit`, `std::_Exit`, or `std::terminate`.
