# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Test sources mirror the tree they exercise: unittests/cpp/... covers cpp/...,
# unittests/kernelSrcs/... covers kernelSrcs/..., and so on.
#
# They are grouped into several executables rather than one. A change to a
# single test then relinks only its own group, and ctest can run the groups
# concurrently. The grouping is deliberately coarser than the directory layout:
# every executable links the whole static edgellmCore, so each one costs about
# 70 MB and a full link regardless of how few tests it holds.

# ---------------------------------------------------------------------------
# Shared helpers (CPU reference implementations, tensor assertions). Consumers
# include them unqualified, e.g. #include "references.h".
# ---------------------------------------------------------------------------
file(GLOB UNIT_TEST_SUPPORT_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/support/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/support/*.cu")
add_library(unitTestSupport STATIC ${UNIT_TEST_SUPPORT_SRCS})
target_include_directories(
  unitTestSupport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/support"
                         ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
# edgellmCore is deliberately not propagated from here. Each executable links it
# itself, wrapped in --whole-archive, and a transitive copy arriving ahead of
# that marker would be pulled normally and make the marker a no-op.
target_link_libraries(unitTestSupport PUBLIC commonLibraryExt)
add_cross_build_link_options(unitTestSupport)

# ---------------------------------------------------------------------------
# edgellm_add_unit_test(NAME <target> DIRS <dir> [<dir>...])
#
# Each DIR is globbed non-recursively, so a directory that has its own group
# (for example cpp/runtime/state under cpp/runtime) is not swept up by its
# parent. Every target gets the same wiring the single unitTest binary used to
# get, so adding a group is one call.
# ---------------------------------------------------------------------------
function(edgellm_add_unit_test)
  cmake_parse_arguments(ARG "" "NAME" "DIRS" ${ARGN})

  set(_srcs)
  foreach(_dir IN LISTS ARG_DIRS)
    file(GLOB _dir_srcs CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/*.cu")
    list(APPEND _srcs ${_dir_srcs})
  endforeach()

  if(NOT _srcs)
    message(FATAL_ERROR "edgellm_add_unit_test(${ARG_NAME}): no sources found "
                        "under ${ARG_DIRS}")
  endif()

  # Recorded so the sweep below can tell a directory nobody registered from one
  # that is deliberately empty.
  set_property(GLOBAL APPEND PROPERTY EDGELLM_REGISTERED_TEST_DIRS ${ARG_DIRS})

  # deviceLinkStub.cu gives the executable a device-link step, and
  # CUDA_RESOLVE_DEVICE_SYMBOLS makes CMake run it. edgellmCore is built with
  # separable compilation, so a test that calls into one of its .cu members
  # through a host-side wrapper also inherits that member's
  # `__cudaRegisterLinkedBinary_*` reference, which nothing else resolves.
  #
  # Fixing this on the consumer rather than on edgellmCore is deliberate: making
  # the archive device-link itself would change how every other consumer links
  # it, including the Python bindings, which whole-archive it and device-link it
  # again. experimental/pybind/CMakeLists.txt solves the same problem the same
  # way.
  add_executable(
    ${ARG_NAME} ${_srcs}
                "${CMAKE_CURRENT_SOURCE_DIR}/support/deviceLinkStub.cu")
  set_target_properties(${ARG_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(${ARG_NAME} PRIVATE ${COMMON_INCLUDE_DIRS}
                                                 ${CMAKE_SOURCE_DIR})
  # Tests that read fixtures resolve them against the source tree.
  target_compile_definitions(${ARG_NAME}
                             PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
  if(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
    target_compile_definitions(${ARG_NAME}
                               PRIVATE ENABLE_CUTEDSL_MODULE_TEST_HOOK)
  endif()
  # Resource-registration tests resolve the plugin's exported lifecycle ABI. The
  # plugin is a shared library built into the build tree's root, so a test that
  # dlopen()s it cannot derive the path from its own location down here.
  add_dependencies(${ARG_NAME} NvInfer_edgellm_plugin)
  target_compile_definitions(
    ${ARG_NAME}
    PRIVATE EDGELLM_PLUGIN_PATH="$<TARGET_FILE:NvInfer_edgellm_plugin>")
  # Plugin JIT support is not part of edgellmCore. Tests link it explicitly for
  # direct XQA and NVFP4-A16 GEMV compiler coverage. edgellmCore whole-archive:
  # the device link only pulls an archive member that something references from
  # device code, while the host link pulls any member whose host wrapper a test
  # calls. Those two sets differ per group, and the members only the host link
  # pulled are exactly the ones left with an unresolved
  # `__cudaRegisterLinkedBinary_*`. Forcing every member in makes the two sets
  # the same.
  target_link_libraries(
    ${ARG_NAME}
    PRIVATE gtest
            gtest_main
            gmock
            unitTestSupport
            -Wl,--whole-archive
            edgellmCore
            -Wl,--no-whole-archive
            edgellmPluginJit
            commonLibraryExt
            exampleUtils
            ${CUDA_DRIVER_LINK_LIB})
  cute_dsl_setup(LINK_TARGETS ${ARG_NAME})
  add_cross_build_link_options(${ARG_NAME})

  # No RESOURCE_LOCK: the groups are safe to run concurrently. Each writes its
  # scratch files under a name unique to its own suite, and the device memory
  # they hold at once is small. Running them serialized costs more wall clock
  # than the single binary did, because each pays its own CUDA context setup.
  add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
  add_dependencies(unitTests ${ARG_NAME})
endfunction()

# Convenience target: `ninja unitTests` builds every group.
add_custom_target(unitTests)

edgellm_add_unit_test(
  NAME
  unitTestCommon
  DIRS
  cpp/common
  cpp/profiling
  cpp/builder
  cpp/sampler
  cpp/multimodal)

edgellm_add_unit_test(
  NAME
  unitTestRuntime
  DIRS
  cpp/runtime
  cpp/runtime/config
  cpp/runtime/exec
  cpp/runtime/decoding
  cpp/runtime/multiDevice)

edgellm_add_unit_test(NAME unitTestRuntimeState DIRS cpp/runtime/state)

edgellm_add_unit_test(NAME unitTestContextCache DIRS
                      cpp/runtime/state/contextCache)

edgellm_add_unit_test(
  NAME unitTestKernelsAttention DIRS cpp/kernels/contextAttentionKernels
  cpp/kernels/decodeAttentionKernels cpp/kernels/audioAttentionKernels)

edgellm_add_unit_test(NAME unitTestKernelsMoe DIRS cpp/kernels/moe)

edgellm_add_unit_test(NAME unitTestKernelsSpeculative DIRS
                      cpp/kernels/speculative)

edgellm_add_unit_test(
  NAME
  unitTestKernelsMisc
  DIRS
  cpp/kernels
  cpp/kernels/mamba
  cpp/kernels/posEncoding
  cpp/kernels/embeddingKernels
  cpp/kernels/preprocessKernels
  cpp/kernels/talkerMLPKernels
  cpp/kernels/gdnKernels
  cpp/kernels/int4GroupwiseGemmKernels
  cpp/kernels/kvCacheUtilKernels
  cpp/kernels/multiDeviceKernels
  cpp/kernels/nvfp4A16BlackwellGemm
  cpp/kernels/nvfp4A16BlackwellGemv
  cpp/kernels/dart)

edgellm_add_unit_test(NAME unitTestPlugins DIRS cpp/plugins/nvfp4A16GemmPlugin)

edgellm_add_unit_test(NAME unitTestCuteDslKernels DIRS
                      kernelSrcs/int4_fp16_gemm_cutedsl)

edgellm_add_unit_test(NAME unitTestExamples DIRS examples/utils)
target_include_directories(unitTestExamples
                           PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)

# ---------------------------------------------------------------------------
# Every test source must belong to a group.
#
# DIRS is an allowlist, so adding a test in a directory nobody registered builds
# nothing and reports nothing: the suite still passes, with the new tests simply
# absent. That failure is invisible in a green run, which is why it is checked
# here rather than left to be noticed.
#
# CONFIGURE_DEPENDS re-runs this at build time when the directory contents
# change, so the check fires on the commit that adds the file.
# ---------------------------------------------------------------------------
get_property(_registered_dirs GLOBAL PROPERTY EDGELLM_REGISTERED_TEST_DIRS)
list(APPEND _registered_dirs support)

file(GLOB_RECURSE _all_test_srcs CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

set(_orphans)
foreach(_src IN LISTS _all_test_srcs)
  file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_src}")
  get_filename_component(_dir "${_rel}" DIRECTORY)
  if(NOT _dir IN_LIST _registered_dirs)
    list(APPEND _orphans "${_rel}")
  endif()
endforeach()

if(_orphans)
  string(REPLACE ";" "\n  " _orphan_lines "${_orphans}")
  message(
    FATAL_ERROR
      "Unit test sources are not in any group, so they would not be built:\n"
      "  ${_orphan_lines}\n"
      "Add the containing directory to an edgellm_add_unit_test(DIRS ...) call "
      "in unittests/CMakeLists.txt.")
endif()
