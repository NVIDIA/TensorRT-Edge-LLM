# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lightweight helpers in ``kernelSrcs/build_cutedsl.py``."""

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_CUTEDSL_PATH = _REPO_ROOT / "kernelSrcs" / "build_cutedsl.py"

_SPEC = importlib.util.spec_from_file_location("build_cutedsl",
                                               _BUILD_CUTEDSL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
build_cutedsl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = build_cutedsl
_SPEC.loader.exec_module(build_cutedsl)

_FMHA_V2_SUPPORTED_SMS = [80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121]
_FMHA_V2_DENSE_VARIANTS = {
    "fmha_v2_d64",
    "fmha_v2_d64_small",
    "fmha_v2_d128",
    "fmha_v2_d256",
    "fmha_v2_d64_sw",
    "fmha_v2_d128_sw",
    "fmha_v2_d256_sw",
    "fmha_v2_d512",
    "fmha_v2_d512_sw",
}
_FMHA_V2_PAGED_VARIANTS = {
    "fmha_v2_d64_paged",
    "fmha_v2_d64_small_paged",
    "fmha_v2_d128_paged",
    "fmha_v2_d256_paged",
    "fmha_v2_d512_paged",
    "fmha_v2_d64_sw_paged",
    "fmha_v2_d128_sw_paged",
    "fmha_v2_d256_sw_paged",
    "fmha_v2_d512_sw_paged",
}
_FMHA_V2_RAGGED_PAGED_VARIANTS = {
    "fmha_v2_d64_paged_ragged",
    "fmha_v2_d128_paged_ragged",
    "fmha_v2_d256_paged_ragged",
    "fmha_v2_d512_paged_ragged",
    "fmha_v2_d64_sw_paged_ragged",
    "fmha_v2_d128_sw_paged_ragged",
    "fmha_v2_d256_sw_paged_ragged",
    "fmha_v2_d512_sw_paged_ragged",
}
# Causal native-paged kernels that carry the skip-softmax path (table ABI).
_FMHA_V2_SKIP_SOFTMAX_PAGED_VARIANTS = {
    "fmha_v2_d64_paged",
    "fmha_v2_d128_paged",
    "fmha_v2_d256_paged",
    "fmha_v2_d512_paged",
}
_FMHA_V2_SPECIAL_VARIANTS = {
    "fmha_v2_d256_vision_paged_ragged",
    "fmha_v2_d512_vision_paged_ragged",
    "fmha_v2_d128_non_causal_paged_two_pool",
    "fmha_v2_d128_non_causal_paged_two_pool_fp8",
    "fmha_v2_d256_padding",
    "fmha_v2_vit_d64",
    "fmha_v2_vit_d72",
    "fmha_v2_vit_d80",
    "fmha_v2_vit_d96",
    "fmha_v2_vit_d128",
    "fmha_v2_d256_bidirectional",
    "fmha_v2_d512_bidirectional",
}
_FMHA_V2_VARIANTS = (_FMHA_V2_DENSE_VARIANTS | _FMHA_V2_PAGED_VARIANTS
                     | _FMHA_V2_RAGGED_PAGED_VARIANTS
                     | _FMHA_V2_SPECIAL_VARIANTS)
_BLACKWELL_PACKED_PAGED_VARIANTS = {
    f"fmha_d{head_dim}_packed{sliding}_paged{fp8}"
    for head_dim in (64, 128, 256, 512)
    for sliding in ("", "_sw")
    for fp8 in ("", "_fp8")
} | {"fmha_d512_packed_paged_bidirectional"}
_LAYERNORM_SUPPORTED_SMS = [80, 86, 87, 90, 100, 101, 103, 110, 120, 121]
_LAYERNORM_HIDDEN_SIZES = {4096, 4097, 5120, 7168, 8192}
_LAYERNORM_DTYPES = {"fp16", "bf16"}
_RMSNORM_SUPPORTED_SMS = [80, 86, 87, 90, 100, 101, 103, 110, 120, 121]
_RMSNORM_HIDDEN_SIZES = {4096, 5120, 7168, 8192}
_RMSNORM_DTYPES = {"fp16", "bf16"}
_RMSNORM_WEIGHT_BEFORE_CAST_MODES = {0, 1}


def _write_fake_elf(path: Path, machine: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # little-endian
    header[18:20] = machine.to_bytes(2, "little")
    path.write_bytes(header)


def _archive_machine(arch: str) -> int:
    machines = (build_cutedsl._COFF_MACHINE
                if build_cutedsl._is_windows() else build_cutedsl._ELF_MACHINE)
    return machines[arch]


def _write_fake_archive(path: Path, machine: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if build_cutedsl._is_windows():
        payload = machine.to_bytes(2, "little") + bytes(18)
        header = (b"runtime.obj/".ljust(16) + b"0".ljust(12) + b"0".ljust(6) +
                  b"0".ljust(6) + b"100644".ljust(8) +
                  str(len(payload)).encode("ascii").ljust(10) + b"`\n")
        path.write_bytes(b"!<arch>\n" + header + payload)
        return path

    obj = path.with_suffix(".o")
    _write_fake_elf(obj, machine)
    subprocess.run(["ar", "rcs", str(path), str(obj)], check=True)
    return path


@pytest.mark.parametrize(("module_spec", "expected"), [(object(), True),
                                                       (None, False)])
def test_cuda_driver_bindings_available(monkeypatch, module_spec, expected):

    def find_spec(module_name):
        assert module_name == "cuda.bindings.driver"
        return module_spec

    monkeypatch.setattr(build_cutedsl.importlib.util, "find_spec", find_spec)

    assert build_cutedsl._cuda_driver_bindings_available() is expected


def test_cuda_driver_bindings_available_handles_missing_parent(monkeypatch):

    def find_spec(_module_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(build_cutedsl.importlib.util, "find_spec", find_spec)

    assert not build_cutedsl._cuda_driver_bindings_available()


def test_cutlass_dsl_version_accepts_pinned_release_and_local_versions():
    pinned_version = build_cutedsl._CUTLASS_DSL_VERSION

    assert build_cutedsl._cutlass_dsl_version_matches(pinned_version)
    assert build_cutedsl._cutlass_dsl_version_matches(
        f"{pinned_version}+local")

    assert not build_cutedsl._cutlass_dsl_version_matches("4.5.2")
    assert not build_cutedsl._cutlass_dsl_version_matches("4.6.1")
    assert not build_cutedsl._cutlass_dsl_version_matches(
        f"{pinned_version}.dev1")
    assert not build_cutedsl._cutlass_dsl_version_matches(
        f"{pinned_version}.dev20260630+local")


def test_find_static_runtime_archive_prefers_cuda_variant_layout(tmp_path):
    pkg_dir = tmp_path / "nvidia_cutlass_dsl"
    runtime_name = build_cutedsl._runtime_static_archive_name()
    archive = _write_fake_archive(
        pkg_dir / "cu13" / "lib" / runtime_name,
        _archive_machine("aarch64"),
    )

    assert build_cutedsl._find_static_runtime_archive(pkg_dir,
                                                      "13.0") == archive


def test_validate_static_runtime_archive_rejects_arch_mismatch(tmp_path):
    archive = _write_fake_archive(
        tmp_path / build_cutedsl._runtime_static_archive_name(),
        _archive_machine("x86_64"),
    )

    with pytest.raises(RuntimeError, match="architecture mismatch"):
        build_cutedsl._validate_static_runtime_archive(archive, "aarch64")


def test_resolve_static_runtime_archive_uses_installed_matching_archive(
        tmp_path, monkeypatch):
    pkg_dir = tmp_path / "nvidia_cutlass_dsl"
    runtime_name = build_cutedsl._runtime_static_archive_name()
    archive = _write_fake_archive(
        pkg_dir / "cu13" / "lib" / runtime_name,
        _archive_machine("x86_64"),
    )

    def fail_download(*_args, **_kwargs):
        raise AssertionError(
            "download should not be used for matching installed archive")

    monkeypatch.setattr(build_cutedsl, "_download_runtime_libs_wheel",
                        fail_download)

    resolved = build_cutedsl.resolve_static_runtime_archive(
        "x86_64",
        "x86_64",
        pkg_dir,
        "13.0",
        "4.8.0",
        tmp_path / "staging",
    )
    assert resolved == archive


def test_resolve_static_runtime_archive_cross_downloads_target_wheel(
        tmp_path, monkeypatch):
    pkg_dir = tmp_path / "nvidia_cutlass_dsl"
    runtime_name = build_cutedsl._runtime_static_archive_name()
    _write_fake_archive(
        pkg_dir / "cu13" / "lib" / runtime_name,
        _archive_machine("x86_64"),
    )

    wheel = tmp_path / ("nvidia_cutlass_dsl_libs_cu13-4.8.0-cp312-cp312-"
                        "manylinux_2_28_aarch64.whl")
    archive_in_wheel = (tmp_path / "wheel_src" / "nvidia_cutlass_dsl" /
                        "cu13" / "lib" / runtime_name)
    _write_fake_archive(archive_in_wheel, _archive_machine("aarch64"))
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.write(
            archive_in_wheel,
            f"nvidia_cutlass_dsl/cu13/lib/{runtime_name}",
        )

    monkeypatch.setattr(build_cutedsl, "_download_runtime_libs_wheel",
                        lambda *_args: wheel)

    resolved = build_cutedsl.resolve_static_runtime_archive(
        "aarch64",
        "x86_64",
        pkg_dir,
        "13.0",
        "4.8.0",
        tmp_path / "staging",
    )
    assert resolved.name == runtime_name
    assert build_cutedsl._archive_first_member_machine(
        resolved) == _archive_machine("aarch64")


def test_download_runtime_libs_wheel_uses_configured_wheelhouse(
        tmp_path, monkeypatch):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    download_dir = tmp_path / "download"
    monkeypatch.setenv("CUTE_DSL_RUNTIME_WHEELHOUSE", str(wheelhouse))

    def fake_run(cmd, **_kwargs):
        assert "--no-index" in cmd
        assert cmd[cmd.index("--find-links") + 1] == str(wheelhouse)
        destination = Path(cmd[cmd.index("-d") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        wheel = destination / (
            "nvidia_cutlass_dsl_libs_cu12-4.8.0-cp312-cp312-"
            "manylinux_2_28_aarch64.whl")
        wheel.write_bytes(b"wheel")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(build_cutedsl.subprocess, "run", fake_run)

    wheel = build_cutedsl._download_runtime_libs_wheel("aarch64", "12",
                                                       "4.8.0", download_dir)
    assert wheel.name.startswith("nvidia_cutlass_dsl_libs_cu12-4.8.0")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="requires POSIX shell tools")
def test_tarball_builder_lists_mixed_cuda_matrix():
    script = _REPO_ROOT / "kernelSrcs" / "build_cutedsl_tarballs.sh"
    matrix = ("x86_64:sm_100:13,x86_64:sm_100:12,"
              "aarch64:sm_101:12,aarch64:sm_110+sm_120:13")
    env = os.environ.copy()
    env["CUTE_DSL_MATRIX"] = matrix

    result = subprocess.run([script, "--list"],
                            check=True,
                            capture_output=True,
                            text=True,
                            env=env)

    assert result.stdout.splitlines() == [
        "cutedsl_x86_64_sm_100_cuda13.tar.gz",
        "cutedsl_x86_64_sm_100_cuda12.tar.gz",
        "cutedsl_aarch64_sm_101_cuda12.tar.gz",
        "cutedsl_aarch64_sm_110_sm_120_cuda13.tar.gz",
    ]


@pytest.mark.skipif(sys.platform == "win32",
                    reason="requires POSIX shell tools")
@pytest.mark.parametrize("cuda_major", ["12", "13"])
def test_tarball_builder_default_matrix_includes_expected_targets(cuda_major):
    script = _REPO_ROOT / "kernelSrcs" / "build_cutedsl_tarballs.sh"
    env = os.environ.copy()
    env.pop("CUTE_DSL_MATRIX", None)
    env.pop("CUTE_DSL_TARGETS", None)
    env["CUDA_CTK_VERSION"] = cuda_major

    result = subprocess.run([script, "--list"],
                            check=True,
                            capture_output=True,
                            text=True,
                            env=env)

    tarballs = result.stdout.splitlines()
    assert f"cutedsl_x86_64_sm_80_cuda{cuda_major}.tar.gz" in tarballs
    if cuda_major == "13":
        assert "cutedsl_aarch64_sm_90_cuda13.tar.gz" in tarballs
        assert "cutedsl_aarch64_sm_110_sm_120_cuda13.tar.gz" in tarballs


def test_tarball_builder_docker_matrix_matches_ci_targets():
    dockerfile = (_REPO_ROOT / "kernelSrcs" /
                  "Dockerfile.cutedsl").read_text(encoding="utf-8")
    matrix = next(
        line.removeprefix("    CUTE_DSL_MATRIX=")
        for line in dockerfile.splitlines()
        if line.startswith("    CUTE_DSL_MATRIX="))

    assert matrix.split(",") == [
        "x86_64:sm_80:13",
        "x86_64:sm_86:13",
        "x86_64:sm_89:13",
        "x86_64:sm_90:13",
        "x86_64:sm_100:13",
        "x86_64:sm_120:13",
        "x86_64:sm_120:12",
        "aarch64:sm_87:13",
        "aarch64:sm_90:13",
        "aarch64:sm_100:13",
        "aarch64:sm_101:12",
        "aarch64:sm_103:13",
        "aarch64:sm_110:13",
        "aarch64:sm_110+sm_120:13",
        "aarch64:sm_121:12",
        "aarch64:sm_121:13",
    ]


@pytest.mark.parametrize(("sm", "expected"),
                         [(80, "sm_80"), (87, "sm_87"), (90, "sm_90"),
                          (100, "sm_100a"), (103, "sm_103a"), (110, "sm_110a"),
                          (120, "sm_120a"), (121, "sm_121a")])
def test_default_compile_gpu_arch_is_derived_from_target_sm(sm, expected):
    assert build_cutedsl.default_compile_gpu_arch(sm) == expected


def test_parse_sms_preserves_order_and_removes_duplicates():
    assert build_cutedsl._parse_sms("sm_110,120,sm_110") == [110, 120]
    assert build_cutedsl._multi_sm_artifact_tag([110, 120]) == "sm_110_sm_120"


def test_multi_sm_headers_generate_canonical_dispatch(tmp_path):
    variant = build_cutedsl.KernelVariant(name="example",
                                          group="fmha",
                                          supported_sms=[110, 120],
                                          script="unused")
    variants_by_sm = {110: [variant], 120: [variant]}
    staging_dirs = {}
    for sm in variants_by_sm:
        marker = build_cutedsl._multi_sm_marker(sm)
        staging = tmp_path / f"staging-{sm}"
        staging.mkdir()
        staging_dirs[(sm, variant.name)] = staging
        (staging / f"example{marker}.h").write_text(
            "#pragma once\n"
            f"void _mlir_example{marker}_cuda_init(void **);\n"
            f"void _mlir_example{marker}_call(void **args, int32_t num_args);\n"
        )

    output = tmp_path / "artifact"
    output.mkdir()
    dispatch = build_cutedsl._write_multi_sm_headers_and_dispatch(
        output, variants_by_sm, staging_dirs, [110, 120])

    canonical = (output / "include" / "example.h").read_text()
    dispatch_text = dispatch.read_text()
    assert "__arch" not in canonical
    assert '#include "example.h"' in (output / "include" /
                                      "cutedsl_all.h").read_text()
    assert "case 110:" in dispatch_text
    assert "case 120:" in dispatch_text
    assert "getProcessDeviceSm()" in dispatch_text
    assert "static int const sm" in dispatch_text
    assert "cudaGetDeviceCount" not in dispatch_text
    assert "_mlir_example__arch110_call(args, numArgs);" in dispatch_text
    assert "_mlir_example__arch120_call(args, numArgs);" in dispatch_text


def test_multi_sm_rejects_duplicate_global_definitions(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=("first.o:\n00000000 T unique_a\n00000010 T duplicate\n\n"
                "second.o:\n00000000 T unique_b\n00000010 T duplicate\n"),
        stderr="",
    )
    monkeypatch.setattr(build_cutedsl.subprocess, "run",
                        lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="Duplicate global definitions"):
        build_cutedsl._check_defined_symbol_collision(
            [Path("first.o"), Path("second.o")])


@pytest.mark.parametrize("sm", _FMHA_V2_SUPPORTED_SMS)
def test_fmha_v2_registry_is_complete_for_supported_sms(sm):
    variants = build_cutedsl.select_variants(sm, "fmha")
    fmha_v2_variants = [
        variant for variant in variants
        if variant.script == "fmha_v2_cutedsl/fmha.py"
    ]

    assert {variant.name for variant in fmha_v2_variants} == _FMHA_V2_VARIANTS
    assert all(variant.group == "fmha" for variant in fmha_v2_variants)
    assert all("--export_only" in variant.script_args
               for variant in fmha_v2_variants)

    dense_variants = [
        variant for variant in fmha_v2_variants
        if variant.name in _FMHA_V2_DENSE_VARIANTS
    ]
    assert {variant.name
            for variant in dense_variants} == _FMHA_V2_DENSE_VARIANTS
    assert all("--fmha_v2_context" in variant.script_args
               for variant in dense_variants)
    assert all("--paged_kv" not in variant.script_args
               for variant in dense_variants)

    paged_variants = [
        variant for variant in fmha_v2_variants
        if variant.name in _FMHA_V2_PAGED_VARIANTS
    ]
    assert {variant.name
            for variant in paged_variants} == _FMHA_V2_PAGED_VARIANTS
    assert all("--paged_kv" in variant.script_args
               for variant in paged_variants)
    assert all("--fmha_v2_context" not in variant.script_args
               for variant in paged_variants)
    sliding_paged_variants = {
        variant.name
        for variant in paged_variants
        if "--window_size_left" in variant.script_args
    }
    assert sliding_paged_variants == {
        "fmha_v2_d64_sw_paged",
        "fmha_v2_d128_sw_paged",
        "fmha_v2_d256_sw_paged",
        "fmha_v2_d512_sw_paged",
    }

    ragged_paged_variants = [
        variant for variant in fmha_v2_variants
        if variant.name in _FMHA_V2_RAGGED_PAGED_VARIANTS
    ]
    assert {variant.name
            for variant in ragged_paged_variants
            } == _FMHA_V2_RAGGED_PAGED_VARIANTS
    assert all("--paged_kv_ragged" in variant.script_args
               for variant in ragged_paged_variants)
    assert all("--paged_kv" not in variant.script_args
               for variant in ragged_paged_variants)
    sliding_ragged_paged_variants = [
        variant for variant in ragged_paged_variants
        if "--window_size_left" in variant.script_args
    ]
    assert {variant.name
            for variant in sliding_ragged_paged_variants} == {
                "fmha_v2_d64_sw_paged_ragged",
                "fmha_v2_d128_sw_paged_ragged",
                "fmha_v2_d256_sw_paged_ragged",
                "fmha_v2_d512_sw_paged_ragged",
            }
    assert all(
        variant.script_args[variant.script_args.index("--window_size_left") +
                            1] == "4096"
        for variant in sliding_ragged_paged_variants)

    skip_softmax_paged_variants = [
        variant for variant in fmha_v2_variants
        if variant.name in _FMHA_V2_SKIP_SOFTMAX_PAGED_VARIANTS
    ]
    assert {variant.name
            for variant in skip_softmax_paged_variants
            } == _FMHA_V2_SKIP_SOFTMAX_PAGED_VARIANTS
    assert all(
        "--paged_kv" in variant.script_args and "--skip_softmax" in
        variant.script_args and "--window_size_left" not in variant.script_args
        for variant in skip_softmax_paged_variants)
    # Only the causal paged kernels carry the path: sliding-window, ragged, small-D64 and dense variants take no
    # table.
    assert not any("--skip_softmax" in variant.script_args
                   for variant in fmha_v2_variants
                   if variant.name not in _FMHA_V2_SKIP_SOFTMAX_PAGED_VARIANTS)
    d64_paged = next(variant for variant in fmha_v2_variants
                     if variant.name == "fmha_v2_d64_paged")
    assert d64_paged.script_args[d64_paged.script_args.index("--m_block_size")
                                 + 1] == "64"
    assert d64_paged.script_args[d64_paged.script_args.index("--n_block_size")
                                 + 1] == "64"

    padding_variant = next(variant for variant in fmha_v2_variants
                           if variant.name == "fmha_v2_d256_padding")
    assert "--is_causal" not in padding_variant.script_args
    assert "--fmha_v2_context" not in padding_variant.script_args

    assert {
        variant.name
        for variant in fmha_v2_variants if "_fp8" in variant.name
    } == {"fmha_v2_d128_non_causal_paged_two_pool_fp8"}
    assert not any("--kv_dtype" in variant.script_args
                   for variant in fmha_v2_variants)

    optimized_variants = [
        variant for variant in variants
        if variant.script == "fmha_cutedsl_blackwell/fmha.py"
    ]
    assert bool(optimized_variants) == (sm in [100, 101, 103, 110])


@pytest.mark.parametrize("sm", [100, 101, 103, 110])
def test_fmha_registry_has_d512_bidirectional_variants(sm):
    """D512 vision-block kernels support padded and packed queries."""
    variants = build_cutedsl.select_variants(sm, "fmha")
    bidirectional_variants = [
        variant for variant in variants if variant.name.startswith("fmha_d512")
        and "bidirectional" in variant.name
    ]

    assert [variant.name for variant in bidirectional_variants] == [
        "fmha_d512_paged_bidirectional",
        "fmha_d512_packed_paged_bidirectional",
    ]
    assert all("--bidirectional" in variant.script_args
               for variant in bidirectional_variants)
    assert all("--window_size" in variant.script_args
               for variant in bidirectional_variants)
    assert all("visionblock" not in variant.name for variant in variants
               if variant.name.startswith("fmha_d512"))


@pytest.mark.parametrize("sm", [100, 101, 103, 110])
def test_blackwell_packed_paged_registry_is_complete(sm):
    """Packed paged kernels retain their query layout, mask, and dtype options."""
    variants = [
        variant for variant in build_cutedsl.select_variants(sm, "fmha")
        if variant.script == "fmha_cutedsl_blackwell/fmha.py"
        and "_packed" in variant.name
    ]

    assert {variant.name
            for variant in variants} == _BLACKWELL_PACKED_PAGED_VARIANTS
    assert all("--packed_q" in variant.script_args for variant in variants)
    assert all("--paged_kv" in variant.script_args for variant in variants)
    assert {
        variant.name
        for variant in variants if "--window_size" in variant.script_args
    } == {
        name
        for name in _BLACKWELL_PACKED_PAGED_VARIANTS
        if "_packed_sw_" in name or name.endswith("_bidirectional")
    }
    assert {
        variant.name
        for variant in variants if "--in_dtype" in variant.script_args
    } == {
        name
        for name in _BLACKWELL_PACKED_PAGED_VARIANTS if name.endswith("_fp8")
    }


@pytest.mark.parametrize("sm", [100, 101, 103, 110])
def test_lora_registry_uses_standalone_kernel_export_mode(sm):
    # Given a supported Blackwell SM, when the LoRA variants are selected, then
    # both GEMM specializations and the metadata kernel use their intended exporters.
    variants = build_cutedsl.select_variants(sm, "lora")
    variants_by_name = {variant.name: variant for variant in variants}

    assert len(variants) == len(variants_by_name) == 3
    assert set(variants_by_name) == {
        "lora_blackwell_grouped_fp16",
        "lora_blackwell_grouped_fp16_fused",
        "lora_blackwell_metadata",
    }
    assert variants_by_name[
        "lora_blackwell_grouped_fp16"].script == "lora_cutedsl/grouped_gemm_blackwell.py"
    assert variants_by_name["lora_blackwell_grouped_fp16"].script_args == [
        "--export_only"
    ]
    assert variants_by_name[
        "lora_blackwell_grouped_fp16_fused"].script == "lora_cutedsl/grouped_gemm_blackwell.py"
    assert variants_by_name[
        "lora_blackwell_grouped_fp16_fused"].script_args == [
            "--export_only", "--fuse_lora_epilogue"
        ]
    assert variants_by_name[
        "lora_blackwell_metadata"].script == "lora_cutedsl/metadata.py"
    assert variants_by_name["lora_blackwell_metadata"].script_args == []


def test_fmha_v2_d512_registry_uses_32x32_tiles_and_two_warps():
    d512_variants = [
        variant for variant in build_cutedsl.KERNEL_VARIANTS if
        variant.script == "fmha_v2_cutedsl/fmha.py" and "d512" in variant.name
    ]

    assert {variant.name
            for variant in d512_variants} == {
                "fmha_v2_d512",
                "fmha_v2_d512_sw",
                "fmha_v2_d512_bidirectional",
                "fmha_v2_d512_paged",
                "fmha_v2_d512_paged_ragged",
                "fmha_v2_d512_sw_paged_ragged",
                "fmha_v2_d512_sw_paged",
                "fmha_v2_d512_vision_paged_ragged",
            }
    assert all(variant.supported_sms == _FMHA_V2_SUPPORTED_SMS
               for variant in d512_variants)
    for variant in d512_variants:
        assert variant.script_args[variant.script_args.index("--head_dim") +
                                   1] == "512"
        assert variant.script_args[variant.script_args.index("--m_block_size")
                                   + 1] == "32"
        assert variant.script_args[variant.script_args.index("--n_block_size")
                                   + 1] == "32"
        assert variant.script_args[variant.script_args.index("--num_threads") +
                                   1] == "64"


@pytest.mark.parametrize("sm", [75])
def test_fmha_registry_rejects_unsupported_sms(sm):
    with pytest.raises(ValueError, match="No variants"):
        build_cutedsl.select_variants(sm, "fmha")


_NVFP4_A16_MOE_TOKEN_TILES = (8, 16, 32, 64, 128)
_NVFP4_A16_MOE_FUSIONS = {
    "fc1_relu2": "relu2_store",
    "fc2_scatter": "scatter_add"
}


def test_nvfp4_a16_blackwell_moe_registry_is_sm110_fp16_fixed_set():
    """The SM110 grouped W4A16 MoE group bakes exactly {FC1, FC2} x fp16 x tn{8,16,32,64,128}."""
    variants = build_cutedsl.select_variants(110, "nvfp4_a16_blackwell_moe")
    assert len(variants) == len(_NVFP4_A16_MOE_FUSIONS) * len(
        _NVFP4_A16_MOE_TOKEN_TILES)
    assert all(variant.group == "nvfp4_a16_blackwell_moe"
               for variant in variants)
    assert all(variant.supported_sms == [110] for variant in variants)
    assert all(variant.script == ("nvfp4_a16_blackwell_moe/"
                                  "nvfp4_a16_blackwell_moe_gemm.py")
               for variant in variants)
    assert all(not variant.wants_target_sm for variant in variants)

    configurations = set()
    for variant in variants:
        args = variant.script_args
        io_dtype = args[args.index("--io_dtype") + 1]
        token_tile = int(args[args.index("--token_tile") + 1])
        fusion_arg = args[args.index("--fusion") + 1]
        fusion_tag = {
            v: k
            for k, v in _NVFP4_A16_MOE_FUSIONS.items()
        }[fusion_arg]
        assert io_dtype == "fp16"
        assert "--export_only" in args
        assert variant.name == (
            f"nvfp4_a16_blackwell_moe_{fusion_tag}_{io_dtype}"
            f"_tm128_tn{token_tile}_tk64")
        configurations.add((fusion_tag, token_tile))
    assert configurations == {(fusion, tile)
                              for fusion in _NVFP4_A16_MOE_FUSIONS
                              for tile in _NVFP4_A16_MOE_TOKEN_TILES}


@pytest.mark.parametrize("sm", [80, 87, 90, 100, 101, 120, 121])
def test_nvfp4_a16_blackwell_moe_registry_rejects_other_sms(sm):
    with pytest.raises(ValueError, match="No variants"):
        build_cutedsl.select_variants(sm, "nvfp4_a16_blackwell_moe")


@pytest.mark.parametrize("sm", [120, 121])
def test_qsa_sm12x_registry_has_prefill_and_decode_for_both_dtypes(sm):
    """SM12x artifacts must supply all four wrappers used by the QSA runner.

    Contract: selecting QSA for either SM12x target exports FP16 and BF16
    prefill and decode entry points. Omitting a variant breaks runner linkage.
    """
    variants = build_cutedsl.select_variants(sm, "qsa")
    assert {variant.name
            for variant in variants} == {
                "qsa_sparse_chunk_prefill_d256_fp16",
                "qsa_sparse_chunk_prefill_d256_bf16",
                "qsa_sparse_decode_d256_fp16",
                "qsa_sparse_decode_d256_bf16",
            }


def test_qsa_decode_registry_matches_runner_split_k_constants():
    """Runner partial-workspace sizing must match the export-baked grid."""
    decode_variants = [
        variant for variant in build_cutedsl.KERNEL_VARIANTS
        if variant.group == "qsa"
        and variant.name.startswith("qsa_sparse_decode_")
    ]
    assert decode_variants

    # A literal typed back into the registry would bypass the constants.
    expected_max_splits = str(build_cutedsl._QSA_DECODE_MAX_SPLITS)
    expected_m_block = str(build_cutedsl._QSA_DECODE_M_BLOCK)
    expected_threads = str(build_cutedsl._QSA_DECODE_THREADS)
    expected_depth = str(build_cutedsl._QSA_DECODE_PIPE_DEPTH)
    for variant in decode_variants:
        args = variant.script_args
        assert args[args.index("--max_splits") + 1] == expected_max_splits
        assert args[args.index("--m_block_size") + 1] == expected_m_block
        assert args[args.index("--num_threads") + 1] == expected_threads
        assert args[args.index("--pipe_depth") + 1] == expected_depth

    # The kernel module's own decode defaults (what the JIT path and the
    # benchmark harness instantiate) must be the baked AOT geometry.
    kernel_src = (_REPO_ROOT / "kernelSrcs" / "qsa_cutedsl" /
                  "qsa_sparse_gqa.py").read_text(encoding="utf-8")
    threads = re.search(r"^QSA_DECODE_DEFAULT_THREADS = (\d+)$", kernel_src,
                        re.M)
    depth = re.search(r"^QSA_DECODE_DEFAULT_PIPE_DEPTH = (\d+)$", kernel_src,
                      re.M)
    assert threads is not None and depth is not None
    assert int(threads.group(1)) == build_cutedsl._QSA_DECODE_THREADS
    assert int(depth.group(1)) == build_cutedsl._QSA_DECODE_PIPE_DEPTH

    runner_header = (_REPO_ROOT / "cpp" / "kernels" / "qsaAttention" /
                     "cuteDslQsaSparseRunner.h").read_text(encoding="utf-8")
    max_splits = re.search(r"kMaxSplits\{(\d+)\}", runner_header)
    partial_rows = re.search(r"kPartialRows\{(\d+)\}", runner_header)
    assert max_splits is not None and partial_rows is not None
    assert int(max_splits.group(1)) == build_cutedsl._QSA_DECODE_MAX_SPLITS
    assert int(partial_rows.group(1)) == build_cutedsl._QSA_DECODE_M_BLOCK


@pytest.mark.parametrize("sm", _RMSNORM_SUPPORTED_SMS)
def test_rmsnorm_registry_has_all_compile_time_variants(sm):
    variants = build_cutedsl.select_variants(sm, "rmsnorm")

    assert len(variants) == (len(_RMSNORM_DTYPES) *
                             len(_RMSNORM_HIDDEN_SIZES) *
                             len(_RMSNORM_WEIGHT_BEFORE_CAST_MODES))
    assert all(variant.group == "rmsnorm" for variant in variants)
    assert all(variant.supported_sms == _RMSNORM_SUPPORTED_SMS
               for variant in variants)
    assert all(variant.script == "rmsnorm_cutedsl/rmsnorm.py"
               for variant in variants)

    configurations = set()
    for variant in variants:
        dtype = variant.script_args[variant.script_args.index("--dtype") + 1]
        hidden_size = int(
            variant.script_args[variant.script_args.index("--hidden_size") +
                                1])
        weight_before_cast = int(variant.script_args[
            variant.script_args.index("--weight_before_cast") + 1])
        configurations.add((dtype, hidden_size, weight_before_cast))
        assert variant.name == (f"rmsnorm_{dtype}_h{hidden_size}"
                                f"_wbc{weight_before_cast}")
        assert "--export_only" in variant.script_args

    assert configurations == {
        (dtype, hidden_size, weight_before_cast)
        for dtype in _RMSNORM_DTYPES
        for hidden_size in _RMSNORM_HIDDEN_SIZES
        for weight_before_cast in _RMSNORM_WEIGHT_BEFORE_CAST_MODES
    }


@pytest.mark.parametrize("sm", [89])
def test_rmsnorm_registry_rejects_unqualified_sms(sm):
    with pytest.raises(ValueError, match="No variants"):
        build_cutedsl.select_variants(sm, "rmsnorm")


@pytest.mark.parametrize("sm", _LAYERNORM_SUPPORTED_SMS)
def test_layernorm_registry_has_all_compile_time_variants(sm):
    variants = build_cutedsl.select_variants(sm, "layernorm")

    assert len(
        variants) == len(_LAYERNORM_DTYPES) * len(_LAYERNORM_HIDDEN_SIZES)
    assert all(variant.group == "layernorm" for variant in variants)
    assert all(variant.supported_sms == _LAYERNORM_SUPPORTED_SMS
               for variant in variants)
    assert all(variant.script == "layernorm_cutedsl/layernorm.py"
               for variant in variants)

    configurations = set()
    for variant in variants:
        dtype = variant.script_args[variant.script_args.index("--dtype") + 1]
        hidden_size = int(
            variant.script_args[variant.script_args.index("--hidden_size") +
                                1])
        configurations.add((dtype, hidden_size))
        assert variant.name == f"layernorm_{dtype}_h{hidden_size}"
        assert "--export_only" in variant.script_args
        assert "--enable_pdl" not in variant.script_args

    assert configurations == {(dtype, hidden_size)
                              for dtype in _LAYERNORM_DTYPES
                              for hidden_size in _LAYERNORM_HIDDEN_SIZES}


@pytest.mark.parametrize("sm", [89])
def test_layernorm_registry_rejects_unqualified_sms(sm):
    with pytest.raises(ValueError, match="No variants"):
        build_cutedsl.select_variants(sm, "layernorm")


@pytest.mark.parametrize("sm", _LAYERNORM_SUPPORTED_SMS)
def test_layernorm_compile_command_carries_the_target_sm(sm):
    """Every LayerNorm export is told which board it is being built for.

    CUTE_DSL_ARCH is unset for every target in build_cutedsl_tarballs.sh's
    matrix, so a device query inside the kernel script would resolve the build
    host's GPU instead of the target's and silently pick the wrong schedule.
    """
    for variant in build_cutedsl.select_variants(sm, "layernorm"):
        command = build_cutedsl._compile_command(variant, Path("/tmp/staging"),
                                                 f"sm_{sm}a", "linux-aarch64",
                                                 sm)

        assert "--target_sm" in command
        assert command[command.index("--target_sm") + 1] == str(sm)


def test_target_sm_requesters_read_the_flag():
    """A kernel script is only handed --target_sm when its source reads it;
    any other script's argparse would reject the flag.

    LayerNorm and RMSNorm specialize their compile-time launch geometry
    (schedule / cluster shape / async-copy budget) on the target SM and so
    read the flag; the CTM decode module is runtime-shaped and does not.
    """
    offenders = sorted(
        variant.name for variant in build_cutedsl.KERNEL_VARIANTS
        if variant.wants_target_sm and "--target_sm" not in (
            _REPO_ROOT / "kernelSrcs" / variant.script).read_text())

    assert offenders == []


@pytest.mark.parametrize("sm", [80, 86, 87, 89, 90])
def test_packed_causal_conv1d_registry_covers_ampere_and_hopper(sm):
    """The CI artifact matrix selects packed Conv1D variants on each target SM."""
    variants = build_cutedsl.select_variants(sm, "causal_conv1d")

    assert len(variants) == 14
    assert all(variant.group == "causal_conv1d" for variant in variants)


@pytest.mark.parametrize("sm", [120, 121])
def test_sm12x_nvfp4_registry_covers_prefill_and_raw_scale_decode(sm):
    # Given either supported SM12x target,
    # When the NVFP4 GEMM registry is selected,
    # Then all prefill and synchronized raw-scale decode artifacts are built.
    variants = [
        variant for variant in build_cutedsl.select_variants(sm, "gemm_nvfp4")
        if variant.script == "gemm_cutedsl/gemm_blackwell_geforce_nvfp4.py"
    ]
    expected_names = {
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn64",
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn128",
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn128_tk256",
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn64_raw_sfa_m1",
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn128_raw_sfa_m1",
        "gemm_blackwell_geforce_nvfp4_fp16_tm64_tn128_tk256_raw_sfa_m1",
    }

    assert {variant.name for variant in variants} == expected_names
    for variant in variants:
        is_raw_scale_decode = variant.name.endswith("raw_sfa_m1")
        assert ("--raw_sfa" in variant.script_args) is is_raw_scale_decode
        assert "--export_only" in variant.script_args


def test_nvfp4_host_drivers_keep_alpha_pointer_contracts():
    # Given the shared pointer allocator and the SM12x host driver,
    # When their AST contracts are inspected without a GPU,
    # Then alpha is allocated, returned, and passed to every launch.
    ws_source = (_REPO_ROOT / "kernelSrcs" / "gemm_cutedsl" /
                 "gemm_blackwell_nvfp4_ws.py").read_text()
    ws_module = ast.parse(ws_source)
    helper = next(node for node in ws_module.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_create_nvfp4_pointers")
    assigned_names = {
        target.id
        for node in ast.walk(helper) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    returned_pointer_dict = next(
        node for node in ast.walk(helper) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "dict")
    returned_keys = {keyword.arg for keyword in returned_pointer_dict.keywords}

    assert {"alpha_raw", "alpha_ptr"} <= assigned_names
    assert "alpha" in returned_keys

    geforce_source = (_REPO_ROOT / "kernelSrcs" / "gemm_cutedsl" /
                      "gemm_blackwell_geforce_nvfp4.py").read_text()
    geforce_module = ast.parse(geforce_source)
    run_function = next(
        node for node in geforce_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run")
    launch_args = next(
        node.value for node in ast.walk(run_function)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "launch_args"
            for target in node.targets))

    assert isinstance(launch_args, ast.Tuple)
    assert ast.unparse(launch_args.elts[5]) == "ptrs['alpha']"
    assert "backing[-1].fill(" in geforce_source
    assert 'ptrs["alpha"] =' not in geforce_source


def test_raw_sfa_path_uses_a_software_signaled_pipeline():
    # Given raw SFA is written to shared memory with ordinary stores,
    # When the producer and consumer synchronization contract is inspected,
    # Then a separate software-signaled pipeline orders those stores.
    source = (_REPO_ROOT / "kernelSrcs" / "nvfp4_fused_moe_cutedsl" /
              "dense_gemm_sm120.py").read_text()

    assert "raw_sfa_pipeline = pipeline.PipelineAsync.create" in source
    assert "raw_sfa_pipeline.producer_acquire(raw_sfa_producer_state)" in source
    assert "raw_sfa_pipeline.producer_commit(raw_sfa_producer_state)" in source
    assert "raw_sfa_pipeline.consumer_wait(raw_sfa_consumer_state)" in source
    assert "raw_sfa_pipeline.consumer_release(raw_sfa_consumer_state)" in source


def test_layernorm_source_pins_flashinfer_and_keeps_odd_h_masked():
    source = (_REPO_ROOT / "kernelSrcs" / "layernorm_cutedsl" /
              "layernorm.py").read_text()
    module = ast.parse(source)
    source_hidden_sizes = next(
        ast.literal_eval(node.value) for node in module.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "SUPPORTED_HIDDEN_SIZES"
            for target in node.targets))

    assert "d7f2c64647585b641590e309c75974519dea17db" in source
    assert set(source_hidden_sizes) == _LAYERNORM_HIDDEN_SIZES
    assert set(
        build_cutedsl._LAYERNORM_HIDDEN_SIZES) == _LAYERNORM_HIDDEN_SIZES
    assert "if column >= hidden_size:" in source
    assert "layernorm_kernel(output, x, gamma, beta, rows, eps, False, stream)" in source


def test_fmha_v2_per_variant_compile_definitions_are_absent():
    pattern = re.compile(r"\bCUTE_DSL_FMHA_V2_[A-Z0-9_]+\b")
    offenders = []
    candidate_paths = [_REPO_ROOT / "CMakeLists.txt"]
    for root in ("cmake", "cpp", "kernelSrcs"):
        for path in (_REPO_ROOT / root).rglob("*"):
            if path.suffix not in {
                    ".cmake", ".txt", ".cpp", ".cu", ".cuh", ".h", ".hpp",
                    ".py"
            }:
                continue
            candidate_paths.append(path)

    for path in candidate_paths:
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(path.relative_to(_REPO_ROOT).as_posix())

    assert offenders == []


def test_build_allows_f16_moe_for_foreign_target_sm(tmp_path, monkeypatch):
    args = argparse.Namespace(
        gpu_arch="sm_120",
        arch="x86_64",
        output_dir=str(tmp_path),
        kernels="f16_moe",
        cuda_version="13",
    )

    def stop_at_dependency_check(**_kwargs):
        raise RuntimeError("dependency check reached")

    # GPU probing was removed outright: an explicit target SM is required and
    # artifact generation never queries the build host's GPU.
    assert not hasattr(build_cutedsl, "detect_gpu_sm")
    monkeypatch.setattr(build_cutedsl, "check_dependencies",
                        stop_at_dependency_check)

    with pytest.raises(RuntimeError, match="dependency check reached"):
        build_cutedsl.build(args)


_BASE_METADATA = {
    "arch": "x86_64",
    "host_platform": build_cutedsl._host_platform(),
    "object_format": "coff" if build_cutedsl._is_windows() else "elf",
    "artifact_tag": "sm_100",
    "gpu_arch": "sm_100",
    "compile_gpu_arch": "sm_100a",
    "host_target": "",
    "cuda_package_variant": "cu12",
    "cutlass_dsl_version": build_cutedsl._CUTLASS_DSL_VERSION,
    "groups": ["gdn"],
    "variants": ["gdn_decode", "gdn_prefill"],
}


def test_merge_artifact_metadata_unions_groups_and_variants():
    existing = dict(_BASE_METADATA,
                    groups=["fmha", "gdn"],
                    variants=["fmha_d64", "gdn_decode"])
    new = dict(_BASE_METADATA,
               groups=["gdn"],
               variants=["gdn_decode", "gdn_prefill"])

    merged = build_cutedsl._merge_artifact_metadata(existing, new)

    assert merged["groups"] == ["fmha", "gdn"]
    assert merged["variants"] == ["fmha_d64", "gdn_decode", "gdn_prefill"]


def test_merge_artifact_metadata_without_existing_returns_new():
    new = dict(_BASE_METADATA)
    assert build_cutedsl._merge_artifact_metadata(None, new) is new


def test_merge_artifact_metadata_rejects_incompatible_artifact():
    existing = dict(_BASE_METADATA, cuda_package_variant="cu13")
    new = dict(_BASE_METADATA)

    with pytest.raises(RuntimeError, match="cuda_package_variant"):
        build_cutedsl._merge_artifact_metadata(existing, new)


def test_cmake_treats_aarch64_build_as_boolean():
    paths = [
        _REPO_ROOT / "CMakeLists.txt",
        _REPO_ROOT / "cmake" / "EdgeLLMBuild.cmake",
        _REPO_ROOT / "cmake" / "EdgeLLMDependencies.cmake",
        _REPO_ROOT / "cmake" / "EdgeLLMPlatform.cmake",
        _REPO_ROOT / "python" / "nanobind" / "CMakeLists.txt",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "DEFINED AARCH64_BUILD" not in text

    assert "if(AARCH64_BUILD)" in paths[-1].read_text(encoding="utf-8")
