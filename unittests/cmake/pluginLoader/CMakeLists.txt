# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(edgellm_plugin_loader_smoke LANGUAGES CXX)

find_package(TensorRTEdgeLLM CONFIG REQUIRED)

add_executable(plugin_loader_smoke main.cc)
target_compile_features(plugin_loader_smoke PRIVATE cxx_std_17)
target_link_libraries(plugin_loader_smoke PRIVATE TensorRTEdgeLLM::Core)
