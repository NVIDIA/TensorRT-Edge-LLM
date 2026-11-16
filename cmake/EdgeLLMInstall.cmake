# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

include(CMakePackageConfigHelpers)

set(_edgellm_export_targets
    commonLibraryExt
    edgellmChatTemplate
    xgrammarCore
    edgellmCore
    edgellmBuilder
    edgellmCuteDsl
    NvInfer_edgellm_plugin)
foreach(_optional_target trt_edgellm_cutedsl_dispatch
                         trt_edgellm_cutedsl_cudart_shim)
  if(TARGET ${_optional_target})
    get_target_property(_optional_target_type ${_optional_target} TYPE)
    if(NOT _optional_target_type STREQUAL "INTERFACE_LIBRARY")
      list(APPEND _edgellm_export_targets ${_optional_target})
    endif()
  endif()
endforeach()
install(
  TARGETS ${_edgellm_export_targets}
  EXPORT TensorRTEdgeLLMTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

get_property(_cutedsl_static_library GLOBAL
             PROPERTY EDGELLM_CUTE_DSL_STATIC_LIBRARY)
install(FILES ${_cutedsl_static_library}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensorrt_edgellm)

set(_edgellm_programs
    llm_build
    llm_inference
    llm_stream
    llm_bench
    visual_build
    audio_build
    action_build
    action_inference
    qwen3_tts_inference
    multistage_inference)
foreach(_edgellm_program IN LISTS _edgellm_programs)
  if(TARGET ${_edgellm_program})
    install(TARGETS ${_edgellm_program}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endforeach()

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/cpp/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorrt_edgellm
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.cuh"
  PATTERN ".greptile" EXCLUDE
  PATTERN "safety_inference" EXCLUDE)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdParty/inja/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(ENABLE_NVTX_PROFILING)
  install(
    DIRECTORY ${NVTX_INCLUDE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp")
endif()

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/TensorRTEdgeLLMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEdgeLLMConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorRTEdgeLLM)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEdgeLLMConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT TensorRTEdgeLLMTargets
  FILE TensorRTEdgeLLMTargets.cmake
  NAMESPACE TensorRTEdgeLLM::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorRTEdgeLLM)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEdgeLLMConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEdgeLLMConfigVersion.cmake
        ${CMAKE_SOURCE_DIR}/cmake/FindTensorRT.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorRTEdgeLLM)
install(FILES ${CMAKE_SOURCE_DIR}/LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/TensorRTEdgeLLM)
