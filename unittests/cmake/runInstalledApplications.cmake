# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

foreach(_required EDGELLM_BINARY_DIR EDGELLM_SOURCE_DIR EDGELLM_GENERATOR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

# MSVC developer environment recorded when the SDK was configured.
if(EDGELLM_TOOLCHAIN_ENV)
  include("${EDGELLM_TOOLCHAIN_ENV}")
endif()

set(_test_root "${EDGELLM_BINARY_DIR}/installedApplicationsTest")
set(_install_dir "${_test_root}/install")
set(_build_dir "${_test_root}/build")
set(_plugin_loader_build_dir "${_test_root}/pluginLoaderBuild")
file(REMOVE_RECURSE "${_test_root}")

set(_install_command "${CMAKE_COMMAND}" --install "${EDGELLM_BINARY_DIR}"
                     --prefix "${_install_dir}")
if(EDGELLM_CONFIG)
  list(APPEND _install_command --config "${EDGELLM_CONFIG}")
endif()
execute_process(COMMAND ${_install_command} RESULT_VARIABLE _result)
if(_result)
  message(FATAL_ERROR "TensorRT Edge-LLM installation failed: ${_result}")
endif()

set(_configure_command
    "${CMAKE_COMMAND}" -S "${EDGELLM_SOURCE_DIR}/examples/cpp" -B
    "${_build_dir}" -G "${EDGELLM_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${_install_dir}")
if(EDGELLM_CONFIG)
  # Single-config generators: build the consumer like the package (MSVC rejects
  # mixing Debug and Release CRTs).
  list(APPEND _configure_command "-DCMAKE_BUILD_TYPE=${EDGELLM_CONFIG}")
endif()
if(TRT_PACKAGE_DIR)
  list(APPEND _configure_command "-DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}")
endif()
if(CUDAToolkit_ROOT)
  list(APPEND _configure_command "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
endif()
if(EDGELLM_CUDA_COMPILER)
  list(APPEND _configure_command
       "-DCMAKE_CUDA_COMPILER=${EDGELLM_CUDA_COMPILER}")
endif()
if(EDGELLM_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND ${_configure_command}
            "-DCMAKE_CUDA_ARCHITECTURES=${EDGELLM_CUDA_ARCHITECTURES}"
    RESULT_VARIABLE _result)
else()
  execute_process(COMMAND ${_configure_command} RESULT_VARIABLE _result)
endif()
if(_result)
  message(FATAL_ERROR "Installed application configuration failed: ${_result}")
endif()

set(_build_command "${CMAKE_COMMAND}" --build "${_build_dir}")
if(EDGELLM_CONFIG)
  list(APPEND _build_command --config "${EDGELLM_CONFIG}")
endif()
execute_process(COMMAND ${_build_command} RESULT_VARIABLE _result)
if(_result)
  message(FATAL_ERROR "Installed application build failed: ${_result}")
endif()

if(WIN32)
  set(_executable_suffix ".exe")
else()
  set(_executable_suffix "")
endif()
set(_application_bin_dir "${_build_dir}/bin")
if(NOT EXISTS "${_application_bin_dir}/llm_inference${_executable_suffix}"
   AND EDGELLM_CONFIG)
  set(_application_bin_dir "${_build_dir}/bin/${EDGELLM_CONFIG}")
endif()
set(_application_names
    llm_build
    visual_build
    audio_build
    action_build
    llm_inference
    llm_bench
    action_inference
    qwen3_tts_inference)
if(NOT WIN32)
  list(APPEND _application_names llm_stream)
endif()
foreach(_application_name IN LISTS _application_names)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "LD_LIBRARY_PATH=${_install_dir}/lib:$ENV{LD_LIBRARY_PATH}"
      "${_application_bin_dir}/${_application_name}${_executable_suffix}" --help
    RESULT_VARIABLE _result
    OUTPUT_QUIET ERROR_QUIET)
  if(_result)
    message(
      FATAL_ERROR "Installed ${_application_name} --help failed: ${_result}")
  endif()
endforeach()

# Verify that an out-of-tree consumer can find the installed plugin without
# EDGELLM_PLUGIN_PATH or an LD_LIBRARY_PATH entry for the Edge-LLM SDK. This
# exercises both the imported Core target's RUNPATH and the loader's bare-name
# fallback; --help on the applications above does not load the plugin.
if(NOT WIN32)
  set(_plugin_loader_configure_command
      "${CMAKE_COMMAND}" -S
      "${EDGELLM_SOURCE_DIR}/unittests/cmake/pluginLoader" -B
      "${_plugin_loader_build_dir}" -G "${EDGELLM_GENERATOR}"
      "-DCMAKE_PREFIX_PATH=${_install_dir}")
  if(TRT_PACKAGE_DIR)
    list(APPEND _plugin_loader_configure_command
         "-DTRT_PACKAGE_DIR=${TRT_PACKAGE_DIR}")
  endif()
  if(CUDAToolkit_ROOT)
    list(APPEND _plugin_loader_configure_command
         "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  endif()
  execute_process(COMMAND ${_plugin_loader_configure_command}
                  RESULT_VARIABLE _result)
  if(_result)
    message(FATAL_ERROR "Plugin loader smoke configuration failed: ${_result}")
  endif()

  set(_plugin_loader_build_command "${CMAKE_COMMAND}" --build
                                   "${_plugin_loader_build_dir}")
  if(EDGELLM_CONFIG)
    list(APPEND _plugin_loader_build_command --config "${EDGELLM_CONFIG}")
  endif()
  execute_process(COMMAND ${_plugin_loader_build_command}
                  RESULT_VARIABLE _result)
  if(_result)
    message(FATAL_ERROR "Plugin loader smoke build failed: ${_result}")
  endif()

  set(_plugin_loader_bin_dir "${_plugin_loader_build_dir}")
  if(NOT EXISTS
     "${_plugin_loader_bin_dir}/plugin_loader_smoke${_executable_suffix}"
     AND EDGELLM_CONFIG)
    set(_plugin_loader_bin_dir "${_plugin_loader_build_dir}/${EDGELLM_CONFIG}")
  endif()
  set(_trt_runtime_paths)
  if(TRT_PACKAGE_DIR)
    list(APPEND _trt_runtime_paths "${TRT_PACKAGE_DIR}/lib"
         "${TRT_PACKAGE_DIR}/lib64")
  endif()
  list(JOIN _trt_runtime_paths ":" _trt_runtime_path)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env --unset=EDGELLM_PLUGIN_PATH
      "LD_LIBRARY_PATH=${_trt_runtime_path}"
      "${_plugin_loader_bin_dir}/plugin_loader_smoke${_executable_suffix}"
    RESULT_VARIABLE _result)
  if(_result)
    message(FATAL_ERROR "Installed plugin loader smoke failed: ${_result}")
  endif()
endif()
