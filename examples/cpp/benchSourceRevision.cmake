# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${SOURCE_DIR}"
  OUTPUT_VARIABLE SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
set(HEADER_CONTENT "#pragma once\n")
if(SOURCE_REVISION)
  execute_process(
    COMMAND git diff --quiet HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE SOURCE_DIRTY
    ERROR_QUIET)
  if(SOURCE_DIRTY EQUAL 0)
    set(SOURCE_DIRTY 0)
  else()
    set(SOURCE_DIRTY 1)
  endif()
  string(APPEND HEADER_CONTENT
         "#define EDGELLM_BENCH_SOURCE_REVISION \"${SOURCE_REVISION}\"\n"
         "#define EDGELLM_BENCH_SOURCE_DIRTY ${SOURCE_DIRTY}\n")
endif()
file(WRITE "${OUTPUT_HEADER}.tmp" "${HEADER_CONTENT}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                        "${OUTPUT_HEADER}.tmp" "${OUTPUT_HEADER}")
file(REMOVE "${OUTPUT_HEADER}.tmp")
