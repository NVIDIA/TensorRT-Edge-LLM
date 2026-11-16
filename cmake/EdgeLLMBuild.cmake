# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ENABLE_SAFETY_RUNTIME)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

# Generate forwarding headers for the namespace used by the installed SDK.
# Forwarding to the source file preserves pragma-once identity when an in-tree
# target includes both the legacy and installed-style paths.
set(EDGELLM_PUBLIC_BUILD_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
file(
  GLOB_RECURSE _edgellm_public_headers CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_SOURCE_DIR}/cpp"
  "${CMAKE_SOURCE_DIR}/cpp/*.h" "${CMAKE_SOURCE_DIR}/cpp/*.hpp"
  "${CMAKE_SOURCE_DIR}/cpp/*.cuh")
foreach(_header IN LISTS _edgellm_public_headers)
  get_filename_component(_header_dir "${_header}" DIRECTORY)
  file(MAKE_DIRECTORY
       "${EDGELLM_PUBLIC_BUILD_INCLUDE_DIR}/tensorrt_edgellm/${_header_dir}")
  file(CONFIGURE OUTPUT
       "${EDGELLM_PUBLIC_BUILD_INCLUDE_DIR}/tensorrt_edgellm/${_header}"
       CONTENT "#include \"${CMAKE_SOURCE_DIR}/cpp/${_header}\"\n" @ONLY)
endforeach()

if(MSVC)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror,-Wno-error=unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--no-undefined")
endif()

function(edgellm_apply_qnx_warning_suppressions target)
  if(QNX_BUILD)
    set(_suppressions
        -Wno-array-bounds
        -Wno-maybe-uninitialized
        -Wno-dangling-pointer
        -Wno-free-nonheap-object
        -Wno-stringop-overflow
        -Wno-return-type
        -Wno-stringop-overread)
    target_compile_options(
      ${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${_suppressions}>)
    list(TRANSFORM _suppressions PREPEND "-Xcompiler=")
    target_compile_options(${target}
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${_suppressions}>)
  endif()
endfunction()

function(edgellm_set_hidden_visibility target)
  set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden
                                             VISIBILITY_INLINES_HIDDEN ON)
endfunction()

function(add_cross_build_link_options target)
  if(EDGELLM_CROSS_AARCH64)
    get_target_property(_already_added ${target}
                        TRT_EDGELLM_CROSS_LINK_OPTIONS_ADDED)
    if(NOT _already_added)
      get_target_property(_target_type ${target} TYPE)
      set(_scope PRIVATE)
      if(_target_type STREQUAL "STATIC_LIBRARY")
        set(_scope INTERFACE)
      endif()
      if(_target_type STREQUAL "EXECUTABLE" OR _target_type STREQUAL
                                               "SHARED_LIBRARY")
        target_link_options(${target} PRIVATE
                            -Wl,--unresolved-symbols=ignore-in-shared-libs)
      endif()
      foreach(_cuda_root IN LISTS CUDA_TARGET_DIRS)
        if(EXISTS "${_cuda_root}/lib")
          target_link_options(${target} ${_scope} "-L${_cuda_root}/lib")
        endif()
        if(EXISTS "${_cuda_root}/lib/stubs")
          target_link_options(${target} ${_scope} "-L${_cuda_root}/lib/stubs")
        endif()
      endforeach()
      foreach(_cuda_root CUDA_DIR CUDA_TARGET_DIR)
        if(DEFINED ${_cuda_root} AND EXISTS "${${_cuda_root}}/lib")
          target_link_options(${target} ${_scope} "-L${${_cuda_root}}/lib")
        endif()
        if(DEFINED ${_cuda_root} AND EXISTS "${${_cuda_root}}/lib/stubs")
          target_link_options(${target} ${_scope}
                              "-L${${_cuda_root}}/lib/stubs")
        endif()
      endforeach()
      set_property(TARGET ${target}
                   PROPERTY TRT_EDGELLM_CROSS_LINK_OPTIONS_ADDED TRUE)
    endif()
  endif()
endfunction()
