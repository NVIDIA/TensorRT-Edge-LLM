# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Guided decoding: builds the XGrammar backend from source.
#
# We compile XGrammar's C++ sources directly into our own static library rather
# than vendoring a prebuilt archive, so cross-compilation targets stay in step
# with the rest of the tree. Verified on v0.2.1: aarch64-linux-gnu-g++ with
# C++17 builds and links all of cpp/*.cc with no patches.

set(XGRAMMAR_ROOT "${CMAKE_SOURCE_DIR}/3rdParty/xgrammar")

# picojson is vendored into XGrammar's tree, but dlpack is a nested submodule
# and matcher.h includes <dlpack/dlpack.h>, so a non-recursive checkout is not
# enough.
set(_xgr_probes
    "include/xgrammar/xgrammar.h" "cpp/grammar_matcher.cc"
    "3rdparty/picojson/picojson.h" "3rdparty/dlpack/include/dlpack/dlpack.h")

foreach(_probe IN LISTS _xgr_probes)
  if(NOT EXISTS "${XGRAMMAR_ROOT}/${_probe}")
    message(FATAL_ERROR "Missing ${XGRAMMAR_ROOT}/${_probe}. Run:\n"
                        "  git submodule update --init --recursive")
  endif()
endforeach()

# Only cpp/*.cc. cpp/tvm_ffi/* is the Python binding layer (v0.2.1 replaced the
# old cpp/nanobind/* with it); excluding it drops the Python dependency
# entirely. There are no .cu files under cpp/ -- the apply-bitmask kernel is
# ours (see cpp/sampler/sampling.cu).
file(GLOB_RECURSE XGRAMMAR_SRCS "${XGRAMMAR_ROOT}/cpp/*.cc")
list(FILTER XGRAMMAR_SRCS EXCLUDE REGEX "/cpp/tvm_ffi/")
list(LENGTH XGRAMMAR_SRCS _xgr_src_count)
if(_xgr_src_count EQUAL 0)
  message(FATAL_ERROR "No XGrammar sources found under ${XGRAMMAR_ROOT}/cpp")
endif()

set(XGRAMMAR_INCLUDE_DIRS
    "${XGRAMMAR_ROOT}/include" "${XGRAMMAR_ROOT}/cpp"
    "${XGRAMMAR_ROOT}/3rdparty/picojson"
    "${XGRAMMAR_ROOT}/3rdparty/dlpack/include")

add_library(xgrammarCore STATIC ${XGRAMMAR_SRCS})
# PUBLIC so consumers (edgellmCore) pick the headers up; SYSTEM so third-party
# headers do not trip our -Wall -Werror.
target_include_directories(xgrammarCore SYSTEM PUBLIC ${XGRAMMAR_INCLUDE_DIRS})
# Third-party sources are not held to our warning policy.
target_compile_options(xgrammarCore PRIVATE -w)
set_target_properties(
  xgrammarCore
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON)
find_package(Threads REQUIRED)
target_link_libraries(xgrammarCore PUBLIC Threads::Threads)

message(STATUS "Guided decoding: XGrammar, ${_xgr_src_count} sources")
