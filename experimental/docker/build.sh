#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
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

set -euxo pipefail

: "${EDGELLM_HOME:=/opt/TensorRT-Edge-LLM}"
: "${TRT_PACKAGE_DIR:=/usr}"
: "${CUDA_CTK_VERSION:=13.0}"
: "${CMAKE_TOOLCHAIN_FILE:=cmake/aarch64_linux_toolchain.cmake}"
: "${EMBEDDED_TARGET:=jetson-thor}"
: "${ENABLE_CUTE_DSL:=ALL}"
: "${CUTE_DSL_ARTIFACT_TAG:=sm_110}"
: "${CUTE_DSL_PREBUILT_TARBALL:=}"
: "${CUDA_CUBIN_ARCH:=sm_110a}"

require_arg() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required build arg: ${name}" >&2
        echo "The Dockerfile defaults should configure this value." >&2
        exit 1
    fi
}

require_arg CUDA_CTK_VERSION
require_arg EMBEDDED_TARGET
require_arg ENABLE_CUTE_DSL

if [[ "${EMBEDDED_TARGET}" != "jetson-thor" ]]; then
    echo "The experimental Dockerfile currently supports only EMBEDDED_TARGET=jetson-thor." >&2
    exit 1
fi

if [[ "${ENABLE_CUTE_DSL^^}" == "OFF" ]]; then
    echo "TensorRT Edge-LLM Thor builds require CuTe DSL; use ENABLE_CUTE_DSL=ALL." >&2
    exit 1
fi

require_arg CUTE_DSL_ARTIFACT_TAG
require_arg CUDA_CUBIN_ARCH

python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "PyTorch is required in the base image. Use a PyTorch base image such "
        "as nvcr.io/nvidia/pytorch:26.04-py3."
    )
PY

cd "${EDGELLM_HOME}"

for required_file in \
    3rdParty/NVTX/include/nvtx3/nvtx3.hpp \
    3rdParty/googletest/CMakeLists.txt \
    3rdParty/nlohmannJson/include/nlohmann/json.hpp; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing submodule content: ${required_file}" >&2
        echo "Run 'git submodule update --init --recursive' before building the image." >&2
        exit 1
    fi
done

artifact_arch="aarch64"
artifact_root="cpp/kernels/cuteDSLArtifact/${artifact_arch}"
artifact_dir="${artifact_root}/${CUTE_DSL_ARTIFACT_TAG}"
cuda_major="${CUDA_CTK_VERSION%%.*}"
prebuilt_tarball="${CUTE_DSL_PREBUILT_TARBALL:-kernelSrcs/cuteDSLPrebuilt/cutedsl_${artifact_arch}_${CUTE_DSL_ARTIFACT_TAG}_cuda${cuda_major}.tar.gz}"

if [[ ! -f "${artifact_dir}/metadata.json" || ! -f "${artifact_dir}/libcutedsl_${artifact_arch}.a" ]]; then
    if [[ ! -f "${prebuilt_tarball}" ]]; then
        root_tarball="$(basename "${prebuilt_tarball}")"
        if [[ -f "${root_tarball}" ]]; then
            echo "Staging CuteDSL artifact ${root_tarball} into $(dirname "${prebuilt_tarball}")"
            mkdir -p "$(dirname "${prebuilt_tarball}")"
            cp -f "${root_tarball}" "${prebuilt_tarball}"
            if [[ -f "${root_tarball}.sha256" ]]; then
                cp -f "${root_tarball}.sha256" "${prebuilt_tarball}.sha256"
            fi
        fi
    fi

    if [[ ! -f "${prebuilt_tarball}" ]]; then
        echo "CuTe DSL is enabled, but the prebuilt artifact tarball was not found: ${prebuilt_tarball}" >&2
        echo "Available upstream prebuilt artifacts:" >&2
        find kernelSrcs/cuteDSLPrebuilt -maxdepth 1 -type f -name '*.tar.gz' -print >&2 || true
        exit 1
    fi

    if [[ -f "${prebuilt_tarball}.sha256" ]]; then
        (cd "$(dirname "${prebuilt_tarball}")" && sha256sum -c "$(basename "${prebuilt_tarball}").sha256")
    fi

    mkdir -p "${artifact_root}"
    tar -xzf "${prebuilt_tarball}" -C "${artifact_root}"
fi

test -f "${artifact_dir}/metadata.json"
test -f "${artifact_dir}/libcutedsl_${artifact_arch}.a"
test -f "${artifact_dir}/include/cutedsl_all.h"

if command -v uv >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PIP_INSTALL=(uv pip install)
else
    PIP_INSTALL=(python3 -m pip install --break-system-packages)
fi

"${PIP_INSTALL[@]}" --upgrade 'setuptools<82' wheel
"${PIP_INSTALL[@]}" -e ".[server,server-tools,native-build]"

pybind11_dir="$(python3 -m pybind11 --cmakedir)"
cmake_args=(
    -S .
    -B build
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_PYTHON_BINDINGS=ON
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}"
    -Dpybind11_DIR="${pybind11_dir}"
    -DCUDA_CTK_VERSION="${CUDA_CTK_VERSION}"
    -DENABLE_CUTE_DSL="${ENABLE_CUTE_DSL}"
    -DCUTE_DSL_ARTIFACT_TAG="${CUTE_DSL_ARTIFACT_TAG}"
)

if [[ -n "${CMAKE_TOOLCHAIN_FILE}" ]]; then
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
fi

if [[ -n "${EMBEDDED_TARGET}" ]]; then
    cmake_args+=("-DEMBEDDED_TARGET=${EMBEDDED_TARGET}")
fi

cmake "${cmake_args[@]}"
cmake --build build --target NvInfer_edgellm_plugin _edgellm_runtime --parallel "${MAX_JOBS:-$(nproc)}"

test -f build/libNvInfer_edgellm_plugin.so
test -n "$(find build/pybind -maxdepth 1 -name '*_edgellm_runtime*.so' -print -quit)"
