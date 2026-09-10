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
repo_root="$(cd "${script_dir}/.." && pwd)"

: "${CUDA_CTK_VERSION:=13.0}"
: "${CUTE_DSL_VERSION:=4.7.0}"
: "${CUTE_DSL_RUNTIME_LIBS_VERSION:=${CUTE_DSL_VERSION}}"
: "${CUTE_DSL_KERNELS:=ALL}"
: "${CUTE_DSL_JOBS:=$(nproc)}"
: "${CUTE_DSL_OUTPUT_DIR:=}"
: "${CUTE_DSL_PREBUILT_DIR:=${repo_root}/kernelSrcs/cuteDSLPrebuilt}"
: "${CUTE_DSL_SKIP_DEPENDENCY_INSTALL:=0}"
export CUTE_DSL_VERSION

host_cuda_major="${CUDA_CTK_VERSION%%.*}"
cache_dir="${HOME:-/tmp}/.cache/tensorrt-edge-llm"
: "${CUTE_DSL_VENV_ROOT:=${cache_dir}/cutedsl-venvs-host-cuda${host_cuda_major}}"

case "${host_cuda_major}" in
    13)
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda13x==13.6.0}"
        : "${CUTE_DSL_TARGETS:=x86_64:sm_80,x86_64:sm_90,x86_64:sm_100,x86_64:sm_120,aarch64:sm_87,aarch64:sm_90,aarch64:sm_110,aarch64:sm_121}"
        ;;
    12)
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda12x==12.3.0}"
        : "${CUTE_DSL_TARGETS:=x86_64:sm_80,x86_64:sm_90,x86_64:sm_100,x86_64:sm_120,aarch64:sm_110,aarch64:sm_121}"
        ;;
    *)
        if [[ -z "${CUTE_DSL_CUPY_PACKAGE:-}" ]] || [[ -z "${CUTE_DSL_TARGETS:-}" ]]; then
            echo "Unsupported host CUDA_CTK_VERSION=${CUDA_CTK_VERSION}; set " \
                "CUTE_DSL_CUPY_PACKAGE and CUTE_DSL_TARGETS explicitly." >&2
            exit 1
        fi
        ;;
esac

: "${CUTE_DSL_PACKAGE_CU12:=${CUTE_DSL_PACKAGE:-nvidia-cutlass-dsl[cu12]==${CUTE_DSL_VERSION}}}"
: "${CUTE_DSL_PACKAGE_CU13:=${CUTE_DSL_PACKAGE:-nvidia-cutlass-dsl[cu13]==${CUTE_DSL_VERSION}}}"

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "$1 is required to build CuTe DSL tarballs." >&2
        exit 1
    fi
}

normalize_sm_tag() {
    local tag="${1,,}"
    tag="${tag//-/_}"
    if [[ "${tag}" =~ ^[0-9]+$ ]]; then
        tag="sm_${tag}"
    elif [[ "${tag}" =~ ^sm([0-9]+)$ ]]; then
        tag="sm_${BASH_REMATCH[1]}"
    fi
    if [[ ! "${tag}" =~ ^sm_[0-9]+$ ]]; then
        echo "Invalid GPU arch '${1}'. Expected sm_87, sm_110, sm_121, etc." >&2
        exit 1
    fi
    printf '%s\n' "${tag}"
}

normalize_arch() {
    local arch="${1,,}"
    arch="${arch//-/_}"
    case "${arch}" in
        amd64 | x86_64)
            printf 'x86_64\n'
            ;;
        aarch64 | arm64)
            printf 'aarch64\n'
            ;;
        *)
            echo "Invalid target arch '${1}'. Expected x86_64 or aarch64." >&2
            exit 1
            ;;
    esac
}

is_true() {
    case "${1,,}" in
        1 | on | true | yes)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cutedsl_package_for_cuda() {
    case "$1" in
        12)
            printf '%s\n' "${CUTE_DSL_PACKAGE_CU12}"
            ;;
        13)
            printf '%s\n' "${CUTE_DSL_PACKAGE_CU13}"
            ;;
        *)
            echo "Unsupported artifact CUDA major '$1'. Expected 12 or 13." >&2
            exit 1
            ;;
    esac
}

declare -A prepared_venvs=()
ensured_venv_dir=""

ensure_venv() {
    local artifact_cuda_major="$1"
    local venv_dir="${CUTE_DSL_VENV_ROOT}/cu${artifact_cuda_major}"
    local cutedsl_package

    if [[ -n "${prepared_venvs[${artifact_cuda_major}]:-}" ]]; then
        ensured_venv_dir="${venv_dir}"
        return
    fi

    cutedsl_package="$(cutedsl_package_for_cuda "${artifact_cuda_major}")"
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        if is_true "${CUTE_DSL_SKIP_DEPENDENCY_INSTALL}"; then
            echo "Prebuilt CuTe DSL venv is missing: ${venv_dir}" >&2
            exit 1
        fi
        python3 -m venv "${venv_dir}"
    fi

    if ! is_true "${CUTE_DSL_SKIP_DEPENDENCY_INSTALL}"; then
        "${venv_dir}/bin/pip" install -q --upgrade pip wheel
        "${venv_dir}/bin/pip" install -q "${cutedsl_package}" "${CUTE_DSL_CUPY_PACKAGE}"
    fi

    "${venv_dir}/bin/python" - "${artifact_cuda_major}" "${host_cuda_major}" <<'PY'
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

artifact_cuda_major, host_cuda_major = sys.argv[1:]
cupy_package = f"cupy-cuda{host_cuda_major}x"
importlib.metadata.version("nvidia-cutlass-dsl")
importlib.metadata.version(cupy_package)

spec = importlib.util.find_spec("nvidia_cutlass_dsl")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("nvidia_cutlass_dsl package directory was not found")
package_dir = Path(next(iter(spec.submodule_search_locations)))
runtime_archive = (
    package_dir
    / f"cu{artifact_cuda_major}"
    / "lib"
    / "libcuda_dialect_runtime_static.a"
)
if not runtime_archive.is_file():
    raise SystemExit(f"CuTe DSL runtime archive was not found: {runtime_archive}")
PY

    prepared_venvs["${artifact_cuda_major}"]=1
    ensured_venv_dir="${venv_dir}"
}

gpu_checked=0

ensure_gpu_visible() {
    local python="$1"

    if [[ "${gpu_checked}" == "1" ]]; then
        return
    fi
    if ! "${python}" - <<'PY'
import cupy

if cupy.cuda.runtime.getDeviceCount() == 0:
    raise SystemExit("no CUDA device visible")
PY
    then
        echo "No CUDA GPU is visible to this process. Kernel AOT export requires" \
            "a GPU (device tensor allocation + local helper compiles)." >&2
        echo "In Docker, run with: docker run --gpus all ..." >&2
        exit 1
    fi
    gpu_checked=1
}

validate_artifact() {
    local python="$1"
    local artifact_dir="$2"
    local arch="$3"
    local artifact_tag="$4"
    local artifact_cuda_major="$5"
    local archive="${artifact_dir}/libcutedsl_${arch}.a"

    "${python}" - \
        "${artifact_dir}/metadata.json" \
        "${archive}" \
        "${arch}" \
        "${artifact_tag}" \
        "${artifact_cuda_major}" \
        "${host_cuda_major}" \
        "${CUTE_DSL_RUNTIME_LIBS_VERSION}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
arch, artifact_tag, artifact_cuda_major, host_cuda_major = sys.argv[3:7]
runtime_libs_version = sys.argv[7]

metadata = json.loads(metadata_path.read_text())
expected = {
    "arch": arch,
    "artifact_tag": artifact_tag,
    "gpu_arch": artifact_tag,
    "cuda_package_variant": f"cu{artifact_cuda_major}",
    "cupy_package": f"cupy-cuda{host_cuda_major}x",
    "runtime_libs_version": runtime_libs_version,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(
            f"{metadata_path}: {key}={metadata.get(key)!r}, expected {value!r}"
        )

if str(metadata.get("cuda_version", "")).split(".", 1)[0] != artifact_cuda_major:
    raise SystemExit(
        f"{metadata_path}: cuda_version={metadata.get('cuda_version')!r}, "
        f"expected CUDA major {artifact_cuda_major}"
    )
if (
    str(metadata.get("cutlass_dsl_cuda_version", "")).split(".", 1)[0]
    != artifact_cuda_major
):
    raise SystemExit(
        f"{metadata_path}: "
        f"cutlass_dsl_cuda_version={metadata.get('cutlass_dsl_cuda_version')!r}, "
        f"expected CUDA major {artifact_cuda_major}"
    )
if str(metadata.get("host_cuda_version", "")).split(".", 1)[0] != host_cuda_major:
    raise SystemExit(
        f"{metadata_path}: host_cuda_version={metadata.get('host_cuda_version')!r}, "
        f"expected CUDA major {host_cuda_major}"
    )
if not metadata.get("groups") or not metadata.get("variants"):
    raise SystemExit(f"{metadata_path}: groups and variants must both be non-empty")

sm = int(artifact_tag.removeprefix("sm_"))
suffix_sms = {100, 101, 103, 107, 109, 110, 120, 121}
expected_compile_arch = f"{artifact_tag}{'a' if sm in suffix_sms else ''}"
if metadata.get("compile_gpu_arch") != expected_compile_arch:
    raise SystemExit(
        f"{metadata_path}: compile_gpu_arch={metadata.get('compile_gpu_arch')!r}, "
        f"expected {expected_compile_arch!r}"
    )

machine_by_arch = {"x86_64": 0x3E, "aarch64": 0xB7}
members = subprocess.check_output(["ar", "t", archive_path], text=True).splitlines()
if not members:
    raise SystemExit(f"{archive_path}: archive has no members")
for member in members:
    data = subprocess.check_output(["ar", "p", archive_path, member])
    if len(data) < 20 or data[:4] != b"\x7fELF":
        raise SystemExit(f"{archive_path}({member}): member is not ELF")
    byteorder = "little" if data[5] == 1 else "big"
    machine = int.from_bytes(data[18:20], byteorder)
    if machine != machine_by_arch[arch]:
        raise SystemExit(
            f"{archive_path}({member}): e_machine=0x{machine:x}, "
            f"expected 0x{machine_by_arch[arch]:x}"
        )
PY
}

validate_tarball() {
    local python="$1"
    local tarball="$2"
    local artifact_tag="$3"
    local arch="$4"

    "${python}" - "${tarball}" "${artifact_tag}" "${arch}" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

tarball, artifact_tag, arch = sys.argv[1:]
required = {
    f"{artifact_tag}/metadata.json",
    f"{artifact_tag}/libcutedsl_{arch}.a",
    f"{artifact_tag}/include/cutedsl_all.h",
}
with tarfile.open(tarball, "r:gz") as archive:
    names = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"{tarball}: unsafe member path {member.name!r}")
        if not path.parts or path.parts[0] != artifact_tag:
            raise SystemExit(
                f"{tarball}: member {member.name!r} is outside {artifact_tag}/"
            )
        names.add(member.name.rstrip("/"))
missing = required - names
if missing:
    raise SystemExit(f"{tarball}: missing required members: {sorted(missing)}")
PY
}

declare -A seen_tarballs=()
expected_tarballs=()

build_target() {
    local target="$1"
    local arch
    local gpu_arch
    local artifact_cuda_version
    local artifact_cuda_major
    local artifact_tag
    local extra
    local target_output_dir
    local artifact_dir
    local tarball_name
    local tarball_path
    local venv_dir
    local python
    local -a build_command

    IFS=: read -r arch gpu_arch artifact_cuda_version extra <<< "${target}"
    if [[ -z "${arch:-}" || -z "${gpu_arch:-}" || -n "${extra:-}" ]]; then
        echo "Invalid target '${target}'. Use arch:sm_NN[:cuda_major], " \
            "for example aarch64:sm_110:13." >&2
        exit 1
    fi
    artifact_cuda_version="${artifact_cuda_version:-${CUTE_DSL_ARTIFACT_CUDA_VERSION:-${host_cuda_major}}}"
    artifact_cuda_major="${artifact_cuda_version%%.*}"
    cutedsl_package_for_cuda "${artifact_cuda_major}" >/dev/null

    arch="$(normalize_arch "${arch}")"
    artifact_tag="$(normalize_sm_tag "${gpu_arch}")"
    target_output_dir="${output_dir}/cuda${artifact_cuda_major}"
    artifact_dir="${target_output_dir}/${arch}/${artifact_tag}"
    tarball_name="cutedsl_${arch}_${artifact_tag}_cuda${artifact_cuda_major}.tar.gz"
    tarball_path="${CUTE_DSL_PREBUILT_DIR}/${tarball_name}"

    if [[ -n "${seen_tarballs[${tarball_name}]:-}" ]]; then
        echo "Duplicate CuTe DSL matrix entry produces ${tarball_name}." >&2
        exit 1
    fi
    seen_tarballs["${tarball_name}"]=1
    expected_tarballs+=("${tarball_name}")

    if [[ "${list_only}" == "1" ]]; then
        printf '%s\n' "${tarball_name}"
        return
    fi

    echo
    echo "Building CuTe DSL artifact: arch=${arch} gpu=${artifact_tag} " \
        "artifact_cuda=${artifact_cuda_major} host_cuda=${host_cuda_major}"
    ensure_venv "${artifact_cuda_major}"
    venv_dir="${ensured_venv_dir}"
    python="${venv_dir}/bin/python"
    ensure_gpu_visible "${python}"
    build_command=(
        "${python}"
        kernelSrcs/build_cutedsl.py
        --kernels "${CUTE_DSL_KERNELS}" \
        --gpu_arch "${artifact_tag}" \
        --arch "${arch}" \
        --cuda-version "${artifact_cuda_version}" \
        --runtime-libs-version "${CUTE_DSL_RUNTIME_LIBS_VERSION}" \
        --jobs "${CUTE_DSL_JOBS}" \
        --output_dir "${target_output_dir}" \
        --clean
    )
    "${build_command[@]}"

    test -f "${artifact_dir}/metadata.json"
    test -f "${artifact_dir}/libcutedsl_${arch}.a"
    test -f "${artifact_dir}/include/cutedsl_all.h"
    validate_artifact \
        "${python}" \
        "${artifact_dir}" \
        "${arch}" \
        "${artifact_tag}" \
        "${artifact_cuda_major}"

    mkdir -p "${CUTE_DSL_PREBUILT_DIR}"
    tar -C "${target_output_dir}/${arch}" \
        -czf "${tarball_path}" \
        "${artifact_tag}"
    validate_tarball "${python}" "${tarball_path}" "${artifact_tag}" "${arch}"
    (cd "${CUTE_DSL_PREBUILT_DIR}" && sha256sum "${tarball_name}" > "${tarball_name}.sha256")
    (cd "${CUTE_DSL_PREBUILT_DIR}" && sha256sum -c "${tarball_name}.sha256")
    echo "Wrote ${tarball_path}"
}

list_only=0
case "${1:-}" in
    "")
        ;;
    --list)
        list_only=1
        shift
        ;;
    *)
        echo "Usage: $0 [--list]" >&2
        exit 1
        ;;
esac
if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0 [--list]" >&2
    exit 1
fi

require_tool python3
if [[ "${list_only}" != "1" ]]; then
    require_tool ar
    require_tool sha256sum
    require_tool tar
fi

tmp_output_dir=""

cleanup() {
    if [[ -n "${tmp_output_dir}" ]]; then
        rm -rf "${tmp_output_dir}"
    fi
}
trap cleanup EXIT

if [[ -n "${CUTE_DSL_OUTPUT_DIR}" ]]; then
    output_dir="${CUTE_DSL_OUTPUT_DIR}"
else
    tmp_output_dir="$(mktemp -d)"
    output_dir="${tmp_output_dir}"
fi

cd "${repo_root}"
if [[ "${list_only}" != "1" ]]; then
    mkdir -p "${cache_dir}" "${CUTE_DSL_PREBUILT_DIR}" "${CUTE_DSL_VENV_ROOT}"
fi

targets=()
if [[ -n "${CUTE_DSL_MATRIX:-}" ]]; then
    IFS=, read -ra targets <<< "${CUTE_DSL_MATRIX}"
else
    IFS=, read -ra legacy_targets <<< "${CUTE_DSL_TARGETS}"
    for target in "${legacy_targets[@]}"; do
        targets+=("${target}:${CUTE_DSL_ARTIFACT_CUDA_VERSION:-${host_cuda_major}}")
    done
fi
if [[ "${#targets[@]}" -eq 0 ]]; then
    echo "No CuTe DSL targets were configured." >&2
    exit 1
fi

for target in "${targets[@]}"; do
    build_target "${target}"
done

if [[ "${list_only}" != "1" ]]; then
    echo
    echo "Generated ${#expected_tarballs[@]} CuTe DSL tarball(s):"
    for tarball_name in "${expected_tarballs[@]}"; do
        (cd "${CUTE_DSL_PREBUILT_DIR}" && sha256sum -c "${tarball_name}.sha256")
        ls -lh \
            "${CUTE_DSL_PREBUILT_DIR}/${tarball_name}" \
            "${CUTE_DSL_PREBUILT_DIR}/${tarball_name}.sha256"
    done
fi
