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

import pathlib
import sys
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packaging"))

from wheellib import assemble, build, config, payload, verify  # noqa: E402


def _igx_row():
    _, rows = config.load_matrix(PROJECT_ROOT / "packaging" / "variants.toml")
    return config.require_variant(rows, "igx-thor-cu13-sm110-sm120")


def test_igx_payload_compiles_both_gpu_architectures():
    row = _igx_row()

    assert config.CONTRACT.matrix_variant_gpu_sms(row) == (110, 120)
    assert payload._cuda_architecture(row) == "110a;120"


def test_igx_local_build_selects_one_row_for_either_gpu(monkeypatch):
    _, rows = config.load_matrix(PROJECT_ROOT / "packaging" / "variants.toml")
    detected = SimpleNamespace(
        cpu_arch="aarch64",
        cuda_runtime_soname="libcudart.so.13",
        tensorrt_runtime_soname="libnvinfer.so.10",
        gpu_sm=120,
        platform_probe_source="device-model",
        platform_probe_value="NVIDIA IGX Thor Developer Kit",
    )
    detected.as_dict = lambda: vars(detected)
    monkeypatch.setattr(build, "_detect_platform", lambda: detected)

    assert build._local_variant(rows)["variant_id"] == (
        "igx-thor-cu13-sm110-sm120")


def test_igx_runtime_identities_share_one_payload():
    row = _igx_row()
    staged = {key: None for key in config.CONTRACT.RUNTIME_VARIANT_FIELDS}
    staged.update({
        "variant_id": row["variant_id"],
        "gpu_sm": row["gpu_sm"],
        "extension": "_native/payloads/igx/runtime.so",
        "plugin": "_native/payloads/igx/lib/plugin.so",
    })

    entries = assemble._runtime_entries([staged], [row])

    assert [entry["gpu_sm"] for entry in entries] == [110, 120]
    assert len({entry["extension"] for entry in entries}) == 1
    assert len({entry["plugin"] for entry in entries}) == 1


def test_multi_sm_cutedsl_evidence_uses_canonical_variant_names():
    metadata = {
        "variants": ["gdn_decode_mtp", "gdn_decode_mtp_cache"],
    }
    evidence = {
        "archive_members": [
            "gdn_decode_mtp__arch110.o",
            "gdn_decode_mtp__arch110_cache.o",
            "gdn_decode_mtp__arch120.o",
            "gdn_decode_mtp__arch120_cache.o",
        ],
        "generated_symbols": ["_mlir_gdn_decode_mtp"],
        "linked_symbols": ["_mlir_gdn_decode_mtp"],
        "defined_symbols": [],
        "unresolved_symbols": [],
        "header_sha256": {
            "cutedsl_all.h": "digest"
        },
        "link_map_sha256": {
            "extension.map": "digest"
        },
        "debug_sha256": {
            "plugin.debug": "digest"
        },
        "archive_sha256":
        "0" * 64,
    }

    verify._audit_cutedsl_evidence(metadata, evidence)


def test_multi_sm_device_image_audit_filters_each_architecture(monkeypatch):
    commands = []

    monkeypatch.setattr(verify, "_audit_tool", lambda name: name)

    def dump_architecture(command, **kwargs):
        commands.append(command)
        gpu_sm = command[3]
        return SimpleNamespace(
            returncode=1,
            stdout=f"arch = {gpu_sm}a\ncuobjdump fatal: later entry\n")

    monkeypatch.setattr(verify.subprocess, "run", dump_architecture)

    binary = pathlib.Path("runtime.so")
    verify._audit_device_images(binary, 110)
    verify._audit_device_images(binary, 120)

    assert commands == [
        [
            "cuobjdump", "--dump-elf", "--gpu-architecture", "sm_110",
            "runtime.so"
        ],
        [
            "cuobjdump", "--dump-elf", "--gpu-architecture", "sm_120",
            "runtime.so"
        ],
    ]


def test_cuobjdump_audit_uses_cuda_toolkit_before_path(monkeypatch, tmp_path):
    toolkit = tmp_path / "cuda"
    cuobjdump = toolkit / "bin" / "cuobjdump"
    cuobjdump.parent.mkdir(parents=True)
    cuobjdump.write_text("#!/bin/sh\n")
    cuobjdump.chmod(0o755)

    monkeypatch.setenv("CUDAToolkit_ROOT", str(toolkit))
    monkeypatch.setattr(verify.shutil, "which",
                        lambda name: f"/usr/bin/{name}")

    assert verify._audit_tool("cuobjdump") == str(cuobjdump)
