# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# cmake-format: off
# ---------------------------------------------------------------------------
# CuTe DSL unified kernel library
#
# Artifacts are looked up from:
#   cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/
#
# If the directory does not exist, CMake auto-extracts a matching prebuilt
# tarball from kernelSrcs/cuteDSLPrebuilt/ (e.g. cutedsl_aarch64_sm_110_cuda13.tar.gz).
# Users can also generate artifacts locally:
#   python kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN>
#
# ENABLE_CUTE_DSL cache variable controls which kernel groups are linked:
#   OFF      — disable entirely (default)
#   ALL      — enable all groups found in metadata.json
#   fmha     — enable only the Blackwell FMHA group
#   fmha_v2  — enable only the portable FP16 FMHA-v2 group
#   ffpa     — enable only the Ampere FFPA FMHA group
#   gdn      — enable only the GDN group
#   f16_moe  — enable the target-specific homogeneous-FP16 MoE group
#   fmha;gdn — semicolon-separated list of groups (CMake list syntax)
#
# Usage:
#   include(cmake/CuteDsl.cmake)
#   cute_dsl_setup(
#     TARGETS      target1 target2 ...   # compile definitions + include path only
#     LINK_TARGETS target3 target4 ...   # compile definitions + include path + link
#   )
#
# Per-group compile definitions set on each target:
#   CUTE_DSL_FMHA_ENABLED  — set when the fmha group is active
#   CUTE_DSL_FMHA_V2_ENABLED — set when the fmha_v2 group is active
#   CUTE_DSL_FFPA_ENABLED  — set when the ffpa group is active
#   CUTE_DSL_GDN_ENABLED   — set when the gdn group is active
#   CUTE_DSL_F16_MOE_ENABLED — set when the f16_moe group is active
#   CUTE_DSL_SSD_ENABLED   — set when the ssd group is active
#   CUTE_DSL_GEMM_ENABLED  — set when any gemm variant is active
# ---------------------------------------------------------------------------
# cmake-format: on

set(ENABLE_CUTE_DSL
    "OFF"
    CACHE
      STRING
      "CuTe DSL kernels: OFF, ALL, or semicolon-separated group list (fmha;gdn)"
)

set(CUTE_DSL_ARTIFACT_TAG
    ""
    CACHE
      STRING
      "CuTe DSL artifact tag under cuteDSLArtifact/<arch>/ (e.g. sm_80, sm_110, sm_121). Leave empty to auto-select when unambiguous."
)

# Include guard — safe to include from multiple CMakeLists.txt directories.
if(DEFINED _CUTE_DSL_CMAKE_INCLUDED)
  return()
endif()
set(_CUTE_DSL_CMAKE_INCLUDED TRUE)

function(_cute_dsl_normalize_artifact_tag OUT_VAR INPUT_TAG)
  string(STRIP "${INPUT_TAG}" _tag)
  if(_tag STREQUAL "")
    set(${OUT_VAR}
        ""
        PARENT_SCOPE)
    return()
  endif()

  string(TOLOWER "${_tag}" _tag)
  string(REPLACE "-" "_" _tag "${_tag}")

  if(_tag MATCHES "^([0-9]+)$")
    set(_tag "sm_${CMAKE_MATCH_1}")
  elseif(_tag MATCHES "^sm([0-9]+)$")
    set(_tag "sm_${CMAKE_MATCH_1}")
  endif()

  if(NOT _tag MATCHES "^[a-z0-9_]+$")
    message(
      FATAL_ERROR "Invalid CUTE_DSL_ARTIFACT_TAG='${INPUT_TAG}'. "
                  "Use a simple directory tag such as sm_80, sm_110, or sm_121."
    )
  endif()

  set(${OUT_VAR}
      "${_tag}"
      PARENT_SCOPE)
endfunction()

function(_cute_dsl_infer_artifact_tag OUT_VAR ARCH)
  _cute_dsl_normalize_artifact_tag(_explicit_tag "${CUTE_DSL_ARTIFACT_TAG}")
  if(NOT _explicit_tag STREQUAL "")
    set(${OUT_VAR}
        "${_explicit_tag}"
        PARENT_SCOPE)
    return()
  endif()

  set(_default_tag "")
  if("${ARCH}" STREQUAL "aarch64"
     AND DEFINED EMBEDDED_TARGET
     AND NOT EMBEDDED_TARGET STREQUAL "")
    string(TOLOWER "${EMBEDDED_TARGET}" _embedded_target)
    string(REPLACE "-" "_" _embedded_target "${_embedded_target}")
    if(_embedded_target STREQUAL "gb10")
      set(_default_tag "sm_121")
    elseif(_embedded_target STREQUAL "auto_thor" OR _embedded_target STREQUAL
                                                    "jetson_thor")
      set(_default_tag "sm_110")
    elseif(_embedded_target STREQUAL "jetson_orin")
      set(_default_tag "sm_87")
    endif()
  endif()

  if(NOT _default_tag STREQUAL "")
    set(${OUT_VAR}
        "${_default_tag}"
        PARENT_SCOPE)
    return()
  endif()

  set(_artifact_root "${CMAKE_SOURCE_DIR}/cpp/kernels/cuteDSLArtifact/${ARCH}")
  set(_candidate_tags)
  if(EXISTS "${_artifact_root}")
    file(
      GLOB _artifact_children
      RELATIVE "${_artifact_root}"
      LIST_DIRECTORIES true
      "${_artifact_root}/*")
    foreach(_child ${_artifact_children})
      if(IS_DIRECTORY "${_artifact_root}/${_child}"
         AND EXISTS "${_artifact_root}/${_child}/metadata.json")
        list(APPEND _candidate_tags "${_child}")
      endif()
    endforeach()
  endif()

  list(LENGTH _candidate_tags _num_candidates)
  if(_num_candidates EQUAL 1)
    list(GET _candidate_tags 0 _only_tag)
    message(
      STATUS
        "CuTe DSL: inferred artifact tag '${_only_tag}' from ${_artifact_root}")
    set(${OUT_VAR}
        "${_only_tag}"
        PARENT_SCOPE)
    return()
  endif()

  if(_num_candidates GREATER 1)
    message(
      FATAL_ERROR
        "CuTe DSL artifact selection is ambiguous for arch=${ARCH}.\n"
        "Found multiple artifact tags under ${_artifact_root}: ${_candidate_tags}\n"
        "Set -DCUTE_DSL_ARTIFACT_TAG=<tag> explicitly.")
  endif()

  set(${OUT_VAR}
      ""
      PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# cute_dsl_setup()
#
# TARGETS      — targets that need compile definitions + include path
# LINK_TARGETS — targets that additionally link libcutedsl_<arch>.a
# ---------------------------------------------------------------------------
function(cute_dsl_setup)
  cmake_parse_arguments(ARG "" "" "TARGETS;LINK_TARGETS" ${ARGN})

  string(TOUPPER "${ENABLE_CUTE_DSL}" _cute_dsl_norm)

  if(_cute_dsl_norm STREQUAL "OFF")
    return()
  endif()

  # Guard against accidental empty-string assignment (e.g. -DENABLE_CUTE_DSL=).
  if(_cute_dsl_norm STREQUAL "")
    message(
      FATAL_ERROR
        "ENABLE_CUTE_DSL is set to an empty string.\n"
        "Set it to OFF, ALL, or a semicolon-separated group list (e.g. fmha;gdn)."
    )
  endif()

  # Detect host/target CPU architecture.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_arch "aarch64")
  else()
    set(_arch "x86_64")
  endif()

  _cute_dsl_infer_artifact_tag(_artifact_tag "${_arch}")

  # Tagged artifact directory (preferred) with compatibility fallback to the
  # previous flat layout at cuteDSLArtifact/<arch>/.
  set(_artifact_root "${CMAKE_SOURCE_DIR}/cpp/kernels/cuteDSLArtifact/${_arch}")
  set(_artifact_dir "${_artifact_root}")
  if(NOT _artifact_tag STREQUAL "")
    set(_tagged_artifact_dir "${_artifact_root}/${_artifact_tag}")
    if(EXISTS "${_tagged_artifact_dir}/metadata.json")
      set(_artifact_dir "${_tagged_artifact_dir}")
    elseif(EXISTS "${_artifact_root}/metadata.json")
      message(
        WARNING
          "CuTe DSL: using legacy flat artifact layout at ${_artifact_root}. "
          "Regenerate artifacts into ${_tagged_artifact_dir} to avoid cross-target overwrites."
      )
    else()
      set(_artifact_dir "${_tagged_artifact_dir}")
    endif()
  endif()

  if(_artifact_tag STREQUAL ""
     AND "${_arch}" STREQUAL "aarch64"
     AND DEFINED EMBEDDED_TARGET
     AND EMBEDDED_TARGET STREQUAL "thor-all")
    message(
      FATAL_ERROR
        "CuTe DSL artifact selection is ambiguous for EMBEDDED_TARGET=thor-all.\n"
        "Set -DCUTE_DSL_ARTIFACT_TAG=sm_110 or -DCUTE_DSL_ARTIFACT_TAG=sm_121 explicitly."
    )
  endif()

  set(_static_lib "${_artifact_dir}/libcutedsl_${_arch}.a")
  set(_runtime_lib "${_artifact_dir}/libcute_dsl_runtime.so")
  set(_inc_dir "${_artifact_dir}/include")
  set(_metadata "${_artifact_dir}/metadata.json")

  # Auto-extract prebuilt tarball if artifacts are not present. Tarballs live in
  # kernelSrcs/cuteDSLPrebuilt/ and are named:
  # cutedsl_{arch}_{artifact_tag}_cuda{VER}.tar.gz
  if(NOT EXISTS "${_metadata}" AND NOT _artifact_tag STREQUAL "")
    set(_prebuilt_dir "${CMAKE_SOURCE_DIR}/kernelSrcs/cuteDSLPrebuilt")
    file(GLOB _prebuilt_tarballs
         "${_prebuilt_dir}/cutedsl_${_arch}_${_artifact_tag}_cuda*.tar.gz")
    if(_prebuilt_tarballs)
      list(GET _prebuilt_tarballs 0 _tarball)
      message(STATUS "CuTe DSL: extracting prebuilt from ${_tarball}")
      file(MAKE_DIRECTORY "${_artifact_root}")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tarball}"
        WORKING_DIRECTORY "${_artifact_root}"
        RESULT_VARIABLE _tar_rc)
      if(NOT _tar_rc EQUAL 0)
        message(FATAL_ERROR "CuTe DSL: failed to extract ${_tarball}")
      endif()
    endif()
  endif()

  # Validate artifacts exist.
  if(NOT EXISTS "${_static_lib}")
    message(
      FATAL_ERROR
        "Prebuilt CuTe DSL library not found:\n"
        "  ${_static_lib}\n"
        "Generate it with:\n"
        "  python kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN> --arch ${_arch}\n"
        "Artifacts are generated locally under:\n"
        "  cpp/kernels/cuteDSLArtifact/<arch>/<artifact_tag>/")
  endif()

  if(NOT EXISTS "${_metadata}")
    message(FATAL_ERROR "metadata.json not found in ${_artifact_dir}/\n"
                        "Re-run build_cutedsl.py to regenerate artifacts.")
  endif()

  if(NOT EXISTS "${_inc_dir}/cutedsl_all.h")
    message(
      FATAL_ERROR "Umbrella header cutedsl_all.h not found in ${_inc_dir}/\n"
                  "Re-run build_cutedsl.py to regenerate artifacts.")
  endif()

  # Parse the "groups" array from metadata.json. metadata.json example: {
  # "groups": ["gdn", "fmha"], "variants": [...] } Requires CMake >= 3.19 for
  # string(JSON ...).
  file(READ "${_metadata}" _meta_json)
  string(
    JSON
    _meta_gpu_arch
    ERROR_VARIABLE
    _meta_gpu_arch_err
    GET
    "${_meta_json}"
    "gpu_arch")
  if(NOT _meta_gpu_arch_err
     AND NOT _artifact_tag STREQUAL ""
     AND NOT _meta_gpu_arch STREQUAL "${_artifact_tag}")
    message(
      FATAL_ERROR
        "CuTe DSL artifact tag mismatch: selected '${_artifact_tag}' but "
        "metadata.json in ${_artifact_dir} reports gpu_arch='${_meta_gpu_arch}'."
    )
  endif()
  string(JSON _n_groups LENGTH "${_meta_json}" "groups")

  if(_n_groups EQUAL 0)
    message(
      WARNING
        "CuTe DSL: metadata.json has empty 'groups' array in ${_artifact_dir}/. "
        "Re-run build_cutedsl.py to regenerate artifacts.")
    return()
  endif()

  math(EXPR _last_idx "${_n_groups} - 1")

  # Determine which groups to activate based on ENABLE_CUTE_DSL.
  set(_active_groups)
  foreach(_i RANGE ${_last_idx})
    string(JSON _g GET "${_meta_json}" "groups" ${_i})
    if(_cute_dsl_norm STREQUAL "ALL")
      list(APPEND _active_groups "${_g}")
    else()
      # _cute_dsl_norm is a semicolon-separated list (CMake list), e.g.
      # "FMHA;GDN".
      string(TOUPPER "${_g}" _g_upper)
      if("${_g_upper}" IN_LIST _cute_dsl_norm)
        list(APPEND _active_groups "${_g}")
      endif()
    endif()
  endforeach()

  if(NOT _active_groups)
    message(
      WARNING
        "CuTe DSL: ENABLE_CUTE_DSL='${ENABLE_CUTE_DSL}' matched no groups in "
        "${_metadata} (available: ${_meta_json}). Nothing will be linked.")
    return()
  endif()

  # Shim / --wrap branches follow the toolkit version the project uses:
  # CUDA_CTK_VERSION (see root CMakeLists.txt). Fall back to
  # CMAKE_CUDA_COMPILER_VERSION only if CTK is unset.
  if(DEFINED CUDA_CTK_VERSION AND NOT CUDA_CTK_VERSION STREQUAL "")
    set(_cute_dsl_cuda_ver "${CUDA_CTK_VERSION}")
  elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION
         AND NOT CMAKE_CUDA_COMPILER_VERSION STREQUAL "")
    set(_cute_dsl_cuda_ver "${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    set(_cute_dsl_cuda_ver "")
  endif()

  # CUDA 11.4 is allowed only for special CuTe DSL package deliveries.
  if(NOT _cute_dsl_cuda_ver STREQUAL ""
     AND NOT _cute_dsl_cuda_ver VERSION_EQUAL 11.4
     AND _cute_dsl_cuda_ver VERSION_LESS 12.6)
    message(
      FATAL_ERROR
        "CuTe DSL requires CUDA Toolkit 12.6+ (detected ${_cute_dsl_cuda_ver}). "
        "Use -DENABLE_CUTE_DSL=OFF or set -DCUDA_CTK_VERSION to a supported toolkit."
    )
  endif()

  # Shim: cudaLibrary* → cu* when libcudart omits exports (e.g. some 12.0–12.6
  # embedded). From CUDA 12.8 onward, cuda_runtime_api.h declares these APIs
  # with runtime types; compiling the weak shim conflicts with those
  # declarations. Use INTERFACE only (no .c) for 12.8+.
  set(_cutedsl_cudart_shim_src
      "${CMAKE_SOURCE_DIR}/cpp/kernels/gdnKernels/cutedsl_cuda_runtime_library_shim.c"
  )
  if(NOT TARGET trt_edgellm_cutedsl_cudart_shim)
    if(_cute_dsl_cuda_ver STREQUAL ""
       OR _cute_dsl_cuda_ver VERSION_LESS 12.0
       OR _cute_dsl_cuda_ver VERSION_GREATER_EQUAL 12.8)
      add_library(trt_edgellm_cutedsl_cudart_shim INTERFACE)
    else()
      if(NOT EXISTS "${_cutedsl_cudart_shim_src}")
        message(
          FATAL_ERROR
            "CuTe DSL libcudart shim source missing:\n  ${_cutedsl_cudart_shim_src}\n"
            "It must be committed with the repository (not generated).")
      endif()
      add_library(trt_edgellm_cutedsl_cudart_shim STATIC
                  "${_cutedsl_cudart_shim_src}")
      target_include_directories(trt_edgellm_cutedsl_cudart_shim
                                 PRIVATE ${CUDA_INCLUDE_DIR})
      # 12.0–12.6: AOT uses cudaKernel_t; shim needs
      # CUTEDSL_WRAP_LAUNCH_KERNEL_EX.
      if(NOT _cute_dsl_cuda_ver STREQUAL ""
         AND _cute_dsl_cuda_ver VERSION_GREATER_EQUAL 12.0
         AND _cute_dsl_cuda_ver VERSION_LESS 12.8)
        target_compile_definitions(trt_edgellm_cutedsl_cudart_shim
                                   PRIVATE CUTEDSL_WRAP_LAUNCH_KERNEL_EX)
      endif()
    endif()
  endif()

  # Parse the "variants" array from metadata.json to determine which kernel
  # variants are present (used for fine-grained per-variant compile defines).
  set(_variants)
  string(JSON _n_variants ERROR_VARIABLE _json_err LENGTH "${_meta_json}"
                                                          "variants")
  if(NOT _json_err AND NOT _n_variants EQUAL 0)
    math(EXPR _last_var_idx "${_n_variants} - 1")
    foreach(_vi RANGE ${_last_var_idx})
      string(
        JSON
        _vname
        ERROR_VARIABLE
        _verr
        GET
        "${_meta_json}"
        "variants"
        ${_vi})
      if(NOT _verr AND _vname)
        list(APPEND _variants "${_vname}")
      endif()
    endforeach()
  endif()

  # The FP16 MoE runner links one exact-SM artifact. Parse the artifact SM for
  # its compile-time exact-SM guard.
  if("f16_moe" IN_LIST _active_groups)
    if(NOT _meta_gpu_arch_err AND _meta_gpu_arch MATCHES "^sm_([0-9]+)$")
      set(_meta_sm "${CMAKE_MATCH_1}")
    else()
      message(
        FATAL_ERROR
          "CuTe DSL f16_moe metadata gpu_arch must have form sm_<NN>, got '${_meta_gpu_arch}' in ${_metadata}."
      )
    endif()
  endif()

  # Apply compile definitions and include path to all targets.
  foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
    target_include_directories(${_tgt} SYSTEM PRIVATE "${_inc_dir}")
    foreach(_g ${_active_groups})
      string(TOUPPER "${_g}" _gu)
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_${_gu}_ENABLED")
    endforeach()
  endforeach()

  # Per-variant defines for FFPA GQA kernels (native grouped-query attention
  # without K/V head expansion). Each GQA group size is a separate AOT cubin.
  list(FIND _variants "ffpa_d512_causal_gqa4" _ffpa_gqa4_idx)
  if(NOT ${_ffpa_gqa4_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_FFPA_GQA4_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: ffpa_d512_causal_gqa4 variant found — CUTE_DSL_FFPA_GQA4_ENABLED set"
    )
  endif()

  list(FIND _variants "ffpa_d512_causal_gqa8" _ffpa_gqa8_idx)
  if(NOT ${_ffpa_gqa8_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_FFPA_GQA8_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: ffpa_d512_causal_gqa8 variant found — CUTE_DSL_FFPA_GQA8_ENABLED set"
    )
  endif()

  list(FIND _variants "ffpa_d512_causal_gqa16" _ffpa_gqa16_idx)
  if(NOT ${_ffpa_gqa16_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_FFPA_GQA16_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: ffpa_d512_causal_gqa16 variant found - CUTE_DSL_FFPA_GQA16_ENABLED set"
    )
  endif()

  # FFPA vision-block overlay variant.
  list(FIND _variants "ffpa_d512_causal_visionblock" _ffpa_visionblock_idx)
  if(NOT ${_ffpa_visionblock_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_FFPA_VISIONBLOCK_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: ffpa_d512_causal_visionblock variant found — CUTE_DSL_FFPA_VISIONBLOCK_ENABLED set"
    )
  endif()

  # Check for Blackwell GDN variant specifically and set a clean define.
  list(FIND _variants "gdn_prefill_blackwell" _bw_idx)
  if(NOT ${_bw_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_GDN_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: Blackwell GDN prefill variant found — CUTE_DSL_GDN_BLACKWELL_ENABLED set"
    )
  endif()

  # Check for Blackwell SSD variants and set a clean define.
  set(_ssd_bw_found FALSE)
  foreach(_ssd_bw_name "ssd_prefill_blackwell_d64_n128"
                       "ssd_prefill_blackwell_d64_n64")
    list(FIND _variants "${_ssd_bw_name}" _ssd_bw_idx)
    if(NOT ${_ssd_bw_idx} EQUAL -1)
      set(_ssd_bw_found TRUE)
    endif()
  endforeach()
  if(_ssd_bw_found)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_SSD_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: Blackwell SSD prefill variant(s) found — CUTE_DSL_SSD_BLACKWELL_ENABLED set"
    )
  endif()

  # Per-family definitions for the FP16 MoE grouped-GEMM modules. Each
  # target-specific artifact pack contains exactly one of these variants.
  if("f16_moe" IN_LIST _active_groups)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_F16_MOE_ARTIFACT_SM=${_meta_sm}")
    endforeach()
  endif()

  list(FIND _variants "f16_moe_ampere_grouped_fp16" _f16_moe_ampere_idx)
  if(NOT ${_f16_moe_ampere_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_F16_MOE_AMPERE_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: f16_moe Ampere grouped GEMM found — CUTE_DSL_F16_MOE_AMPERE_ENABLED set"
    )
  endif()

  list(FIND _variants "f16_moe_blackwell_grouped_fp16" _f16_moe_blackwell_idx)
  if(NOT ${_f16_moe_blackwell_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_F16_MOE_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: f16_moe Blackwell grouped GEMM found — CUTE_DSL_F16_MOE_BLACKWELL_ENABLED set"
    )
  endif()

  list(FIND _variants "f16_moe_blackwell_geforce_grouped_fp16"
       _f16_moe_blackwell_geforce_idx)
  if(NOT ${_f16_moe_blackwell_geforce_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: f16_moe Blackwell GeForce grouped GEMM found — CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED set"
    )
  endif()

  # Per-variant defines for GEMM kernels (Talker MLP cuBLAS replacement). The
  # runner expects clean architecture-level names rather than raw variant IDs.
  list(FIND _variants "gemm_ampere_decode_fp16" _gemm_ampere_decode_idx)
  if(NOT ${_gemm_ampere_decode_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_decode_fp16 variant found — CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_small_prefill_fp16"
       _gemm_ampere_small_prefill_idx)
  if(NOT ${_gemm_ampere_small_prefill_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_small_prefill_fp16 variant found — CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_medium_prefill_fp16"
       _gemm_ampere_medium_prefill_idx)
  if(NOT ${_gemm_ampere_medium_prefill_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_medium_prefill_fp16 variant found — CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_large_prefill_fp16"
       _gemm_ampere_large_prefill_idx)
  if(NOT ${_gemm_ampere_large_prefill_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_large_prefill_fp16 variant found — CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_splitk4_fp16" _gemm_ampere_splitk4_idx)
  if(NOT ${_gemm_ampere_splitk4_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_SPLITK4_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_splitk4_fp16 — CUTE_DSL_GEMM_AMPERE_SPLITK4_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_splitk2_fp16" _gemm_ampere_splitk2_idx)
  if(NOT ${_gemm_ampere_splitk2_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_SPLITK2_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_splitk2_fp16 — CUTE_DSL_GEMM_AMPERE_SPLITK2_ENABLED set"
    )
  endif()

  # Fused epilogue variants (Plan C: bias+SiLU for FC1, bias-only for FC2)
  list(FIND _variants "gemm_ampere_medium_bias_silu_fp16"
       _gemm_ampere_medium_bias_silu_idx)
  if(NOT ${_gemm_ampere_medium_bias_silu_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_medium_bias_silu_fp16 — CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_ampere_medium_bias_fp16"
       _gemm_ampere_medium_bias_idx)
  if(NOT ${_gemm_ampere_medium_bias_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_AMPERE_ENABLED"
                        "CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_ampere_medium_bias_fp16 — CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_bw_geforce_small_fp16" _gemm_bw_geforce_small_idx)
  if(NOT ${_gemm_bw_geforce_small_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_bw_geforce_small_fp16 variant found — CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_fp16" _gemm_blackwell_idx)
  if(NOT ${_gemm_blackwell_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_GEMM_BLACKWELL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_fp16 variant found — CUTE_DSL_GEMM_BLACKWELL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_bw_geforce_fp16" _gemm_bw_geforce_idx)
  if(NOT ${_gemm_bw_geforce_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_bw_geforce_fp16 variant found — CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED set"
    )
  endif()

  # Blackwell DC fused epilogue variants
  list(FIND _variants "gemm_blackwell_bias_silu_fp16"
       _gemm_blackwell_bias_silu_idx)
  if(NOT ${_gemm_blackwell_bias_silu_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_bias_silu_fp16 — CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_bias_fp16" _gemm_blackwell_bias_idx)
  if(NOT ${_gemm_blackwell_bias_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_bias_fp16 — CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED set"
    )
  endif()

  # Blackwell DC small-tile variants (tile=64x128 cluster=(1,2)) for M <= 512.
  list(FIND _variants "gemm_blackwell_small_fp16" _gemm_blackwell_small_idx)
  if(NOT ${_gemm_blackwell_small_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_small_fp16 — CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_small_bias_silu_fp16"
       _gemm_blackwell_small_bias_silu_idx)
  if(NOT ${_gemm_blackwell_small_bias_silu_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_small_bias_silu_fp16 — CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_small_bias_fp16"
       _gemm_blackwell_small_bias_idx)
  if(NOT ${_gemm_blackwell_small_bias_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_small_bias_fp16 — CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED set"
    )
  endif()

  # Blackwell DC small-tile FP16-in/FP32-out variant (tile=64x128, FP32 C
  # epilogue) for the parakeet mel GEMM (mel power spans ~17 orders of magnitude
  # and must not flush to zero in FP16 before the natural-log stats).
  list(FIND _variants "gemm_blackwell_small_fp16in_fp32out"
       _gemm_blackwell_small_fp16in_fp32out_idx)
  if(NOT ${_gemm_blackwell_small_fp16in_fp32out_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_small_fp16in_fp32out — CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED set"
    )
  endif()

  # Blackwell DC 2-CTA variants (tile=256x256 cluster=(2,1) use_2cta=True) for
  # low-SM-count GPUs (Thor) at M >= 256.
  list(FIND _variants "gemm_blackwell_2cta_fp16" _gemm_blackwell_2cta_idx)
  if(NOT ${_gemm_blackwell_2cta_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt}
                                 PRIVATE "CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_2cta_fp16 — CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_2cta_bias_silu_fp16"
       _gemm_blackwell_2cta_bias_silu_idx)
  if(NOT ${_gemm_blackwell_2cta_bias_silu_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_2cta_bias_silu_fp16 — CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_2cta_bias_fp16"
       _gemm_blackwell_2cta_bias_idx)
  if(NOT ${_gemm_blackwell_2cta_bias_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_2cta_bias_fp16 — CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED set"
    )
  endif()

  # BW GeForce fused epilogue variants
  list(FIND _variants "gemm_bw_geforce_bias_silu_fp16"
       _gemm_bw_geforce_bias_silu_idx)
  if(NOT ${_gemm_bw_geforce_bias_silu_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_bw_geforce_bias_silu_fp16 — CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_bw_geforce_bias_fp16" _gemm_bw_geforce_bias_idx)
  if(NOT ${_gemm_bw_geforce_bias_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_bw_geforce_bias_fp16 — CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED set"
    )
  endif()

  # NVFP4 GEMM variants. The umbrella CUTE_DSL_GEMM_NVFP4_ENABLED comes from the
  # active gemm_nvfp4 group; these per-variant defines let callers compile only
  # the AOT entry points present in the selected artifact.
  list(FIND _variants "gemm_blackwell_nvfp4_fp16_tn64"
       _gemm_bw_nvfp4_fp16_tn64_idx)
  if(NOT ${_gemm_bw_nvfp4_fp16_tn64_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_fp16_tn64 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN64_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_fp16_tn128"
       _gemm_bw_nvfp4_fp16_tn128_idx)
  if(NOT ${_gemm_bw_nvfp4_fp16_tn128_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_fp16_tn128 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP16_TN128_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_fp8_tn64"
       _gemm_bw_nvfp4_fp8_tn64_idx)
  if(NOT ${_gemm_bw_nvfp4_fp8_tn64_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_fp8_tn64 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN64_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_fp8_tn128"
       _gemm_bw_nvfp4_fp8_tn128_idx)
  if(NOT ${_gemm_bw_nvfp4_fp8_tn128_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_fp8_tn128 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_FP8_TN128_ENABLED set"
    )
  endif()

  # Warp-specialised NVFP4 GEMM variants. Per-variant defines follow the
  # CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_<OUT_DTYPE>_TN<N>_ENABLED pattern so
  # callers can compile the FP16-output and FP8-output paths independently.
  list(FIND _variants "gemm_blackwell_nvfp4_ws_fp16_tn64"
       _gemm_bw_nvfp4_ws_fp16_tn64_idx)
  if(NOT ${_gemm_bw_nvfp4_ws_fp16_tn64_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_ws_fp16_tn64 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN64_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_ws_fp16_tn128"
       _gemm_bw_nvfp4_ws_fp16_tn128_idx)
  if(NOT ${_gemm_bw_nvfp4_ws_fp16_tn128_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_ws_fp16_tn128 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN128_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_ws_fp16_tn256"
       _gemm_bw_nvfp4_ws_fp16_tn256_idx)
  if(NOT ${_gemm_bw_nvfp4_ws_fp16_tn256_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN256_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_ws_fp16_tn256 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP16_TN256_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_ws_fp8_tn64"
       _gemm_bw_nvfp4_ws_fp8_tn64_idx)
  if(NOT ${_gemm_bw_nvfp4_ws_fp8_tn64_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_ws_fp8_tn64 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN64_ENABLED set"
    )
  endif()

  list(FIND _variants "gemm_blackwell_nvfp4_ws_fp8_tn128"
       _gemm_bw_nvfp4_ws_fp8_tn128_idx)
  if(NOT ${_gemm_bw_nvfp4_ws_fp8_tn128_idx} EQUAL -1)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(
        ${_tgt} PRIVATE "CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED")
    endforeach()
    message(
      STATUS
        "CuTe DSL: gemm_blackwell_nvfp4_ws_fp8_tn128 — CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_FP8_TN128_ENABLED set"
    )
  endif()

  # Umbrella CUTE_DSL_GEMM_ENABLED — set if ANY gemm variant was found. Source
  # files guard the entire GEMM path with this define.
  set(_any_gemm FALSE)
  foreach(
    _gv
    _gemm_ampere_decode_idx
    _gemm_ampere_small_prefill_idx
    _gemm_ampere_medium_prefill_idx
    _gemm_ampere_large_prefill_idx
    _gemm_ampere_splitk4_idx
    _gemm_ampere_splitk2_idx
    _gemm_ampere_medium_bias_silu_idx
    _gemm_ampere_medium_bias_idx
    _gemm_blackwell_idx
    _gemm_blackwell_bias_silu_idx
    _gemm_blackwell_bias_idx
    _gemm_bw_nvfp4_fp16_tn64_idx
    _gemm_bw_nvfp4_fp16_tn128_idx
    _gemm_bw_nvfp4_fp8_tn64_idx
    _gemm_bw_nvfp4_fp8_tn128_idx
    _gemm_bw_nvfp4_ws_fp16_tn64_idx
    _gemm_bw_nvfp4_ws_fp16_tn128_idx
    _gemm_bw_nvfp4_ws_fp16_tn256_idx
    _gemm_bw_nvfp4_ws_fp8_tn64_idx
    _gemm_bw_nvfp4_ws_fp8_tn128_idx
    _gemm_bw_geforce_idx
    _gemm_bw_geforce_small_idx
    _gemm_bw_geforce_bias_silu_idx
    _gemm_bw_geforce_bias_idx)
    if(DEFINED ${_gv} AND NOT ${${_gv}} EQUAL -1)
      set(_any_gemm TRUE)
    endif()
  endforeach()
  if(_any_gemm)
    foreach(_tgt ${ARG_TARGETS} ${ARG_LINK_TARGETS})
      target_compile_definitions(${_tgt} PRIVATE "CUTE_DSL_GEMM_ENABLED")
    endforeach()
    message(STATUS "CuTe DSL: CUTE_DSL_GEMM_ENABLED set (GEMM variants found)")
  endif()

  # Link the static archive + shim + driver lib into LINK_TARGETS.
  #
  # For STATIC libraries, use PUBLIC so executables that link edgellmCore /
  # edgellmKernels also inherit the CuTe DSL archive. Otherwise unresolved AOT
  # wrapper symbols only show up at the final executable link step.
  set(_link_libs "${_static_lib}")
  if(EXISTS "${_runtime_lib}")
    list(APPEND _link_libs "${_runtime_lib}")
    message(STATUS "CuTe DSL: runtime library found: ${_runtime_lib}")
  endif()
  foreach(_tgt ${ARG_LINK_TARGETS})
    get_target_property(_tgt_type ${_tgt} TYPE)
    if(_tgt_type STREQUAL "STATIC_LIBRARY")
      target_link_libraries(${_tgt} PUBLIC ${_link_libs})
    else()
      target_link_libraries(${_tgt} PRIVATE ${_link_libs}
                                            trt_edgellm_cutedsl_cudart_shim)
    endif()
    if(CUDA_DRIVER_LIB AND NOT CUDA_DRIVER_LIB MATCHES "-NOTFOUND$")
      target_link_libraries(${_tgt} PRIVATE "${CUDA_DRIVER_LIB}")
    endif()
    # CUDA 12.0–12.6: wrap _cudaLaunchKernelEx (cudaKernel_t → CUfunction, e.g.
    # JetPack 6). CUDA 11.x artifacts use CUmodule ABI instead.
    if(NOT _cute_dsl_cuda_ver STREQUAL ""
       AND _cute_dsl_cuda_ver VERSION_GREATER_EQUAL 12.0
       AND _cute_dsl_cuda_ver VERSION_LESS 12.8)
      target_link_options(${_tgt} PRIVATE "-Wl,--wrap=_cudaLaunchKernelEx")
    endif()
  endforeach()

  message(
    STATUS
      "CuTe DSL: arch=${_arch}  artifact_tag=${_artifact_tag}  groups=[${_active_groups}]  lib=${_static_lib}"
  )
endfunction()
