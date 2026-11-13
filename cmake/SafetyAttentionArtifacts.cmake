# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(EDGELLM_SAFETY_FMHA_ARTIFACT_DIR
    ""
    CACHE PATH "Pinned Safety FMHA cubins, header and manifest")

function(edgellm_admit_safety_fmha TARGET ARCHITECTURE)
  get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  set(_sources "${_root}/cpp/safety_inference/kernels/fmha")
  set(_manifest "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}/manifest.json")
  if(NOT EXISTS "${_manifest}")
    message(
      FATAL_ERROR
        "Set EDGELLM_SAFETY_FMHA_ARTIFACT_DIR to artifacts from kernels/fmha/generate.py"
    )
  endif()
  file(READ "${_manifest}" _json)
  string(JSON _artifact_contract_version GET "${_json}"
         artifact_contract_version)
  string(JSON _cutlass GET "${_json}" cutlass_version)
  string(JSON _arch GET "${_json}" architecture)
  string(JSON _commit GET "${_json}" source_commit)
  if(NOT _artifact_contract_version EQUAL 1
     OR NOT _cutlass STREQUAL "4.7.0"
     OR NOT _arch STREQUAL "sm_${ARCHITECTURE}")
    message(
      FATAL_ERROR
        "Safety FMHA artifact contract/source/compiler/architecture mismatch: ${_manifest}"
    )
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "QNX" AND QNX_SAFE_CUDA)
    string(JSON _sdk_compiler GET "${_json}" sdk_compiler)
    if(NOT _sdk_compiler)
      message(
        FATAL_ERROR
          "QNX Safety FMHA requires generate.py --cuda-compiler from the target SDK"
      )
    endif()
  endif()
  string(JSON _backend GET "${_json}" backend)
  if(ARCHITECTURE STREQUAL "110a")
    if(NOT _backend STREQUAL "blackwell"
       OR NOT _commit STREQUAL "52dfc7e00295d8c3d6875aa4acd1962c6003dbb7")
      message(FATAL_ERROR "Thor Safety requires Blackwell FMHA artifacts")
    endif()
    target_compile_definitions(${TARGET} PRIVATE EDGELLM_SAFETY_BLACKWELL=1)
    set(_files safetyBlackwell.py generate.py safetyFmhaLaunch.h
               safetyBlackwellLaunch.h fmha_config.py safety_fmha_helpers.py)
  else()
    if(NOT _backend STREQUAL "ampere"
       OR NOT _commit STREQUAL "52dfc7e00295d8c3d6875aa4acd1962c6003dbb7")
      message(FATAL_ERROR "SM80 CI requires Ampere FMHA artifacts")
    endif()
    set(_files safetyFmha.py generate.py safetyFmhaLaunch.h fmha_config.py)
  endif()
  foreach(_file IN LISTS _files)
    set(_path "${_sources}/${_file}")
    string(JSON _expected GET "${_json}" source_sha256 "${_file}")
    file(SHA256 "${_path}" _actual)
    if(NOT _actual STREQUAL _expected)
      message(
        FATAL_ERROR
          "Stale Safety FMHA artifact source: ${_file}; regenerate the artifact"
      )
    endif()
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_path}")
  endforeach()
  string(JSON _expected GET "${_json}" header_sha256)
  file(SHA256 "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}/safetyFmhaArtifacts.h"
       _actual)
  if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR "Safety FMHA generated header digest mismatch")
  endif()
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}/safetyFmhaArtifacts.h")
  string(JSON _count LENGTH "${_json}" variants)
  if(NOT _count EQUAL 4)
    message(FATAL_ERROR "Safety FMHA requires exactly four admitted variants")
  endif()
  set(_variants)
  foreach(_i RANGE 0 3)
    string(
      JSON
      _name
      GET
      "${_json}"
      variants
      ${_i}
      name)
    if(NOT _name MATCHES "^(contiguous|visual)D(64|128)$")
      message(FATAL_ERROR "Unadmitted Safety FMHA variant: ${_name}")
    endif()
    list(APPEND _variants "${_name}")
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}/${_name}.fatbin")
    string(
      JSON
      _expected
      GET
      "${_json}"
      variants
      ${_i}
      binary_sha256)
    file(SHA256 "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}/${_name}.fatbin" _actual)
    if(NOT _actual STREQUAL _expected)
      message(FATAL_ERROR "Safety FMHA cubin digest mismatch: ${_name}")
    endif()
  endforeach()
  list(SORT _variants)
  if(NOT _variants STREQUAL "contiguousD128;contiguousD64;visualD128;visualD64")
    message(FATAL_ERROR "Missing or duplicate Safety FMHA variants")
  endif()
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest}")
  target_include_directories(
    ${TARGET} PRIVATE "${_sources}" "${EDGELLM_SAFETY_FMHA_ARTIFACT_DIR}")
endfunction()
