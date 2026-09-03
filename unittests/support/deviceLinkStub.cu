/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Empty CUDA translation unit that gives each test executable a device-link
// step. edgellmCore is compiled with separable compilation, so an executable
// that pulls one of its .cu members through a host-side call also inherits that
// member's `__cudaRegisterLinkedBinary_*` reference, which only a device link
// resolves. A target built purely from .cpp sources gets no such step.
//
// The same stub exists for the Python bindings; see
// experimental/pybind/device_link_stub.cu.
