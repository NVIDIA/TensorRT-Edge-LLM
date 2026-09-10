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

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

: "${CUDA_CTK_VERSION:=13.0}"
: "${CUTE_DSL_ARTIFACT_TAG:=sm_110}"
: "${CUTE_DSL_ARCH:=aarch64}"
: "${CUTE_DSL_GPU_ARCH:=${CUTE_DSL_ARTIFACT_TAG}}"
: "${CUTE_DSL_KERNELS:=ALL}"
: "${EXPERIMENTAL_DOCKER_IMAGE:=tensorrt-edge-llm:experimental}"

cuda_major="${CUDA_CTK_VERSION%%.*}"
prebuilt_dir="${repo_root}/kernelSrcs/cuteDSLPrebuilt"
tarball_name="cutedsl_${CUTE_DSL_ARCH}_${CUTE_DSL_ARTIFACT_TAG}_cuda${cuda_major}.tar.gz"
tarball_path="${prebuilt_dir}/${tarball_name}"

case "${cuda_major}" in
    13)
        : "${CUTE_DSL_PACKAGE:=nvidia-cutlass-dsl[cu13]==4.7.0}"
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda13x==13.6.0}"
        ;;
    12)
        : "${CUTE_DSL_PACKAGE:=nvidia-cutlass-dsl[cu12]==4.7.0}"
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda12x==12.3.0}"
        ;;
    *)
        echo "Unsupported CUDA_CTK_VERSION=${CUDA_CTK_VERSION}; set CUTE_DSL_PACKAGE and CUTE_DSL_CUPY_PACKAGE explicitly." >&2
        exit 1
        ;;
esac

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "$1 is required to build the experimental image." >&2
        exit 1
    fi
}

build_cutedsl_tarball() {
    local output_dir
    output_dir="$(mktemp -d)"

    cleanup() {
        rm -rf "${output_dir}"
    }
    trap cleanup RETURN

    echo "No CuteDSL tarball found at ${tarball_path}"
    echo "Building CuteDSL artifact before docker build."

    CUTE_DSL_OUTPUT_DIR="${output_dir}" \
        CUTE_DSL_PREBUILT_DIR="${prebuilt_dir}" \
        CUTE_DSL_TARGETS="${CUTE_DSL_ARCH}:${CUTE_DSL_GPU_ARCH}" \
        kernelSrcs/build_cutedsl_tarballs.sh

    test -f "${tarball_path}"
}

cd "${repo_root}"

require_tool git
git submodule update --init --recursive

if [[ ! -f "${tarball_path}" ]]; then
    build_cutedsl_tarball
elif [[ -f "${tarball_path}.sha256" ]]; then
    (cd "${prebuilt_dir}" && sha256sum -c "${tarball_name}.sha256")
fi

if [[ "$#" -eq 0 ]]; then
    set -- \
        --network=host \
        --shm-size=8g \
        -f experimental/docker/Dockerfile \
        -t "${EXPERIMENTAL_DOCKER_IMAGE}" \
        .
fi

require_tool docker
docker build "$@"
