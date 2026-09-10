# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AOT-compile CuTe DSL kernels into a static library for CMake linking.

Kernel groups:
  gdn              — Gated Delta Net decode/prefill
  fmha             — FP16 Context/ViT FMHA, plus optimized Blackwell
                     FP16/FP8 variants on SM100/101/110
  ssd              — Mamba2 SSM chunk-scan prefill
  gemm             — Talker MLP GEMM (Ampere / Blackwell / BW GeForce)
  int4_fp16_gemm   — W4A16 INT4-weight FP16 GEMM (Ampere; 60-config sweep, bN=128) +
                     the decode GEMV (small-M, shares the GEMM's weight layout;
                     one exported function per M in 1..8) — built together
  f16_moe          — FP16 grouped FC1/FC2 MoE (Ampere / Blackwell / SM12x)
  nvfp4_moe        — split FC1/FC2 NVFP4 MoE (currently SM110/Thor only)
  nvfp4_a16_blackwell_gemm — SM110 dense NVFP4-weight FP16/BF16 GEMM
  nvfp4_fused_moe  — End-to-end NvFP4 fused MoE (Blackwell GeForce)
  rmsnorm          — FP16/BF16 RMSNorm for production hidden sizes

Usage (run from the repo root):
  python kernelSrcs/build_cutedsl.py                      # build all groups for this GPU
  python kernelSrcs/build_cutedsl.py --kernels gdn        # single group
  python kernelSrcs/build_cutedsl.py --kernels fmha,gdn   # multiple groups
  python kernelSrcs/build_cutedsl.py --gpu_arch sm_110    # override SM detection
  python kernelSrcs/build_cutedsl.py --gpu_arch sm_110 --arch aarch64  # cross-compile host objects
  python kernelSrcs/build_cutedsl.py --clean --verbose    # clean rebuild

The GPU SM is auto-detected via cupy / nvidia-smi and only matching variants
are built.  See KERNEL_VARIANTS below for the full variant list.

Output (under {output_dir}/{arch}/{artifact_tag}/):
  libcutedsl_{arch}.a   — merged static archive (kernel objects + CuTe DSL
                          runtime shim objects)
  include/cutedsl_all.h — umbrella header (#includes every variant header)
  metadata.json         — build provenance + group/variant list for CMake

Prebuilt tarballs:
  Prebuilt CuTe DSL artifacts are shipped in the release package. When building
  from the source tree, no tarball is committed — CI generates per-target
  artifacts for testing. The container/build flow still extracts a tarball if
  one is placed under
  kernelSrcs/cuteDSLPrebuilt/; otherwise build from source for your target:
    python kernelSrcs/build_cutedsl.py --gpu_arch sm_110 --arch aarch64 --clean
"""

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = (_SCRIPT_DIR / "../cpp/kernels/cuteDSLArtifact").resolve()
_CUTLASS_DSL_VERSION = os.environ.get("CUTE_DSL_VERSION", "4.7.0")
_CUPY_VERSIONS = {12: ("cupy-cuda12x", "12.3.0"), 13: ("cupy-cuda13x", "13.6.0")}
# Common flag sets for FMHA variants
_LLM = ["--is_causal", "--is_persistent", "--export_only", "--bottom_right_align"]
_LLM_FP8 = _LLM + ["--in_dtype", "Float8E4M3FN"]
_LLM_PAGED = _LLM + ["--paged_kv"]
_LLM_FP8_PAGED = _LLM_FP8 + ["--paged_kv"]
_LLM_DENSE_PAGED = ["--is_persistent", "--export_only", "--paged_kv"]
_LLM_DENSE_FP8_PAGED = _LLM_DENSE_PAGED + ["--in_dtype", "Float8E4M3FN"]
_VIT = ["--is_persistent", "--export_only", "--vit_mode"]
_VIT_FP8 = _VIT + ["--in_dtype", "Float8E4M3FN"]


@dataclass
class KernelVariant:
    """One compilable kernel variant in the CuTe DSL registry.

    Attributes:
        name:          Unique identifier — used as --file_name / --function_prefix.
        group:         Logical group ("gdn", "fmha", "f16_moe",
                       "nvfp4_fused_moe", "nvfp4_moe", "rmsnorm", "ssd",
                       "nvfp4_a16_blackwell_gemm", or "gemm").
                       cmake sets CUTE_DSL_<GROUP>_ENABLED for integrated groups.
        supported_sms: Explicit SM whitelist. With --kernels ALL, only variants whose
                       supported_sms contains the detected/requested SM are compiled.
        script:        Kernel script path relative to kernelSrcs/.
        script_args:   Args forwarded verbatim after --output_dir/--file_name/--function_prefix.
                       GDN variants MUST include "--export_only" here.

    """
    name: str
    group: str
    supported_sms: list[int]
    script: str
    script_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel registry — add new groups/variants here.
#
# Each KernelVariant has a supported_sms whitelist. Only variants matching
# the target SM are compiled. build_cutedsl.py derives the CuTe DSL compile
# architecture from the target SM and routes kernel scripts through
# cutedsl_utils/cutedsl_compile_wrapper.py. Cross-architecture targets also
# receive the corresponding CuTe DSL host-target option.
#
# Groups:
#   gdn              — Gated Delta Net decode/prefill
#   fmha             — FP16 Context/ViT FMHA, plus optimized Blackwell
#                      persistent variants on SM100/101/110.
#   ssd              — Mamba2 SSM chunk-scan prefill
#   gemm             — Talker MLP cuBLAS replacement (Ampere/Blackwell/BW GeForce)
#   f16_moe          — FP16 grouped FC1/FC2 MoE (Ampere/Blackwell/SM12x)
#   nvfp4_moe        — split FC1/FC2 NVFP4 MoE (currently SM110/Thor only)
#   nvfp4_a16_blackwell_gemm — SM110 dense W4A16 TCGen5 GEMM (FP16/BF16)
#   nvfp4_fused_moe  — End-to-end NvFP4 fused MoE (Blackwell GeForce)
#   rmsnorm          — FP16/BF16 RMSNorm for production hidden sizes
# ---------------------------------------------------------------------------
KERNEL_VARIANTS = [
    # --- GDN group ---
    KernelVariant(
        name="gdn_decode",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_decode.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_prefill.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill_blackwell",
        group="gdn",
        supported_sms=[100, 101, 110],
        script="gdn_cutedsl/gdn_prefill_blackwell.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill_blackwell_geforce",
        group="gdn",
        supported_sms=[120, 121],
        script="gdn_cutedsl/gdn_prefill_sm12x.py",
        script_args=["--export_only"],
    ),
    # MTP decode: multi-token speculative-decoding verification (Ampere SM80+).
    # Only the cache variant is exported (per-step intermediate state checkpointing for rollback).
    KernelVariant(
        name="gdn_decode_mtp",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_decode_mtp.py",
        script_args=["--export_only", "--cache_only"],
    ),
    # --- SSD group (Mamba2 SSM chunk scan) ---
    # --- SSD SM80 variants (D×N combinations) ---
    KernelVariant(
        name="ssd_prefill_d128_n128",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "128", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_d64_n128",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_d128_n64",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "128", "--dstate", "64"],
    ),
    KernelVariant(
        name="ssd_prefill_d64_n64",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64"],
    ),
    # --- SSD Blackwell variants ---
    # Two has_init_states modes per (D, N): the default variant assumes the SSM state
    # at chunk 0 is zero (fast path; covers all current Nemotron-H prefill calls). The
    # `_init_states` variant accepts an optional user-provided initial hidden state at
    # chunk 0, used when prefill carries SSM state across calls (continuous batching,
    # multi-call prefill) or by the SsdCuteDslBlackwellChunkedPrefill unit test.
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n128",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n128_init_states",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128",
                     "--has_init_states",
                     "--file_name", "ssd_prefill_blackwell_d64_n128_init_states",
                     "--function_prefix", "ssd_prefill_blackwell_d64_n128_init_states"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d80_n128",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "80", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d80_n128_init_states",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "80", "--dstate", "128",
                     "--has_init_states",
                     "--file_name", "ssd_prefill_blackwell_d80_n128_init_states",
                     "--function_prefix", "ssd_prefill_blackwell_d80_n128_init_states"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n64",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n64_init_states",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64",
                     "--has_init_states",
                     "--file_name", "ssd_prefill_blackwell_d64_n64_init_states",
                     "--function_prefix", "ssd_prefill_blackwell_d64_n64_init_states"],
    ),
    # --- FMHA optimized Blackwell overlay ---
    KernelVariant(
        name="fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM,
    ),
    KernelVariant(
        name="fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM,
    ),
    # Skip-softmax (BLASST) variants: causal-only. The threshold value here is
    # a SENTINEL whose only role is to compile the skip path in. Keep it tiny
    # (not 1.0): it doubles as the self-test lambda when fmha.py runs without
    # --export_only, and lambda = 1 fails the accuracy gate.
    KernelVariant(
        name="fmha_d64_skipsoftmax",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM + ["--skip_softmax_threshold", "1e-6"],
    ),
    KernelVariant(
        name="fmha_d128_skipsoftmax",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM + ["--skip_softmax_threshold", "1e-6"],
    ),
    KernelVariant(
        name="fmha_d256",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM,
    ),
    # D512 paged prefill. Keep every generated variant in the common Blackwell
    # FMHA artifact so D512 shares the existing LLM module lifecycle and ABI.
    KernelVariant(
        name="fmha_d512_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"] + _LLM_PAGED,
    ),
    KernelVariant(
        name="fmha_d512_paged_bidirectional",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"]
                    + _LLM_PAGED + ["--window_size", "4096,-1", "--bidirectional"],
    ),
    KernelVariant(
        name="fmha_d512_sw_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"]
                    + _LLM_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d512_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"] + _LLM_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d512_dense_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"] + _LLM_DENSE_PAGED,
    ),
    KernelVariant(
        name="fmha_d512_dense_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"] + _LLM_DENSE_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d512_sw_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,8,512", "--k_shape", "1,1024,1,512"]
                    + _LLM_FP8_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d64_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d256_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    # LLM FP8 input → FP16 output
    KernelVariant(
        name="fmha_d64_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d128_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d256_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d64_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d256_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    # LLM paged KV cache variants. Direct TMA path requires tokens_per_page == 128.
    KernelVariant(
        name="fmha_d64_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM_PAGED,
    ),
    KernelVariant(
        name="fmha_d128_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM_PAGED,
    ),
    # Skip-softmax (BLASST) paged variants: causal-only, tokens_per_page == 128.
    # The threshold is the same compile-in SENTINEL as the non-paged skip
    # variants; the runtime log2(lambda) is a trailing kernel argument.
    KernelVariant(
        name="fmha_d64_skipsoftmax_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_PAGED + ["--skip_softmax_threshold", "1e-6"],
    ),
    KernelVariant(
        name="fmha_d128_skipsoftmax_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_PAGED + ["--skip_softmax_threshold", "1e-6"],
    ),
    KernelVariant(
        name="fmha_d256_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM_PAGED,
    ),
    KernelVariant(
        name="fmha_d256_dense_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM_DENSE_PAGED,
    ),
    KernelVariant(
        name="fmha_d64_sw_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_PAGED + ["--window_size", "4096,-1"],
    ),
    # LLM paged KV cache (FP8 input, FP16 output)
    KernelVariant(
        name="fmha_d256_sw_paged",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"]
                    + _LLM_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d64_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d128_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d256_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d256_dense_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"] + _LLM_DENSE_FP8_PAGED,
    ),
    KernelVariant(
        name="fmha_d64_sw_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_FP8_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_FP8_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d256_sw_paged_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,16,256", "--k_shape", "1,1024,2,256"]
                    + _LLM_FP8_PAGED + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="vit_fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,14,64"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d72",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,72", "--k_shape", "1,1024,14,72"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d80",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,80", "--k_shape", "1,1024,14,80"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d96",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,96", "--k_shape", "1,1024,14,96"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,14,128"] + _VIT,
    ),
    # ViT FP8 input → FP16 output.
    # Supports d ∈ {64, 80, 96, 128}.  d=80 pads the MMA tiler K to 96.
    # d=72 has no direct variant: SM100 TMA requires the innermost GMEM
    # stride to be 16-byte aligned, d=72 models zero-pad Q/K/V to d=80 and pass the real 1/sqrt(72)
    # softmax scale.
    KernelVariant(
        name="vit_fmha_d64_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,14,64"] + _VIT_FP8 + ["--kv_stage", "2"],
    ),
    KernelVariant(
        name="vit_fmha_d80_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        # Pinned to the measured optimum: the padded K=96 smem tiles need a
        # deeper KV pipeline than the narrower dims (shallow staging costs
        # ~7% at long sequence lengths).
        script_args=["--q_shape", "1,1024,14,80", "--k_shape", "1,1024,14,80"] + _VIT_FP8 + ["--kv_stage", "4"],
    ),
    KernelVariant(
        name="vit_fmha_d96_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,96", "--k_shape", "1,1024,14,96"] + _VIT_FP8,
    ),
    KernelVariant(
        name="vit_fmha_d128_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,14,128"] + _VIT_FP8 + ["--kv_stage", "3"],
    ),
    # --- FMHA-v2 baseline (SM80/86/87/89/90/100/101/110/120/121) ---
    # These variants keep their fmha_v2 symbol names to preserve the generated
    # ABI, but belong to the canonical fmha artifact group. SM100/101/110 also
    # include the optimized variants above. Every target pack ships both the
    # dense and native-paged FP16 LLM families plus the existing special
    # padding, vision-block, and packed-ViT variants.
    KernelVariant(
        name="fmha_v2_d64",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "128", "--n_block_size", "128", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--skip_rescale", "--export_only",
        ],
    ),
    # Short-context specialization: smaller tiles increase the CTA count and
    # avoid the occupancy/tail penalty of the 128x128 long-context kernel.
    KernelVariant(
        name="fmha_v2_d64_small",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d128",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "128",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d64_sw",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d128_sw",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "128",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256_sw",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d64_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "128", "--n_block_size", "128", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d64_small_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d128_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "128",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d512_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "512",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d64_sw_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d128_sw_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "128",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256_sw_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d512_sw_paged",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "512",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--paged_kv", "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256_padding",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_vit_d64",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "64",
            "--m_block_size", "128", "--n_block_size", "128", "--num_threads", "128",
            "--dtype", "Float16", "--fmha_v2_vit", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_vit_d72",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "72",
            "--m_block_size", "128", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--fmha_v2_vit", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_vit_d80",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "80",
            "--m_block_size", "128", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--fmha_v2_vit", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_vit_d128",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "128",
            "--m_block_size", "64", "--n_block_size", "64", "--num_threads", "128",
            "--dtype", "Float16", "--fmha_v2_vit", "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d256_bidirectional",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "256",
            "--m_block_size", "64", "--n_block_size", "32", "--num_threads", "128",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--vision_block",
            "--window_size_left", "1024", "--num_head", "16", "--kv_group_size", "2",
            "--skip_rescale", "--export_only",
        ],
    ),
    # Dense D512 variants. The native-paged D512 kernels serve FP16 prefill;
    # these cover the two contracts that cannot read the paged pool directly:
    # the FP8-KV fallback (gather + dequantize to FP16) and the vision-block
    # overlay. D512 splits its output columns across two CTAs, so Br must stay
    # at 32 to keep Q/K/V within the sm_80 SMEM budget.
    KernelVariant(
        name="fmha_v2_d512",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "512",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context",
            "--skip_rescale", "--export_only",
        ],
    ),
    KernelVariant(
        name="fmha_v2_d512_sw",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "512",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context",
            "--window_size_left", "4096",
            "--skip_rescale", "--export_only",
        ],
    ),
    # Gemma4 Unified global d512 layers. The window is a runtime argument, so
    # this one variant serves both the sliding and the full-causal layers; the
    # runner passes an unbounded window for the latter. Traced with the
    # Gemma4-12B global-layer GQA shape (Hq=16, Hkv=1) — GQA stays
    # runtime-dynamic.
    KernelVariant(
        name="fmha_v2_d512_bidirectional",
        group="fmha",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="fmha_v2_cutedsl/fmha.py",
        script_args=[
            "--head_dim", "512",
            "--m_block_size", "32", "--n_block_size", "32", "--num_threads", "64",
            "--dtype", "Float16", "--is_causal", "--fmha_v2_context", "--vision_block",
            "--window_size_left", "4096", "--num_head", "16", "--kv_group_size", "16",
            "--skip_rescale", "--export_only",
        ],
    ),
    # --- NvFP4 MoE group (decomposed FC1/FC2; SM110/Thor today) ---
    # These variants build the decomposed grouped MoE pipeline:
    #   FC1 gather grouped GEMM + activation + FP4 requant
    #   FC2 grouped GEMM + router-scale finalize/scatter
    KernelVariant(
        name="nvfp4_moe_sm110_fc1_relu2_n128",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=[
            "--activation",
            "relu2",
            "--mma_tiler_n",
            "128",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc1_relu2_n256",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=[
            "--activation",
            "relu2",
            "--mma_tiler_n",
            "256",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc1_swiglu_n128",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=[
            "--activation",
            "swiglu",
            "--mma_tiler_n",
            "128",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc1_swiglu_n256",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=[
            "--activation",
            "swiglu",
            "--mma_tiler_n",
            "256",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc1_geglu_n128",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=[
            "--activation",
            "geglu",
            "--mma_tiler_n",
            "128",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc2_n128_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=[
            "--mma_tiler_n",
            "128",
            "--output_dtype",
            "fp16",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="nvfp4_moe_sm110_fc2_n256_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=[
            "--mma_tiler_n",
            "256",
            "--output_dtype",
            "fp16",
            "--dummy-experts",
            "128",
            "--dummy-top-k",
            "8",
            "--export_only",
        ],
    ),
    # --- F16 MoE group (one FP16 ABI, architecture-specific grouped GEMM) ---
    KernelVariant(
        name="f16_moe_ampere_grouped_fp16",
        group="f16_moe",
        supported_sms=[80, 86, 87, 89],
        script="f16_moe_cutedsl/export_grouped_gemm.py",
        script_args=["--family", "ampere"],
    ),
    KernelVariant(
        name="f16_moe_blackwell_grouped_fp16",
        group="f16_moe",
        supported_sms=[100, 101, 103, 110],
        script="f16_moe_cutedsl/export_grouped_gemm.py",
        script_args=["--family", "blackwell"],
    ),
    KernelVariant(
        name="f16_moe_blackwell_geforce_grouped_fp16",
        group="f16_moe",
        supported_sms=[120, 121],
        script="f16_moe_cutedsl/export_grouped_gemm.py",
        script_args=["--family", "blackwell_geforce"],
    ),
    # --- NvFP4 Fused MoE group (SM120/SM121 — Blackwell GeForce) ---
    # Fused route/pack + FC1 + activation + quant + FC2 + scatter kernels.
    # Decode backend: resident-grid barrier between route/pack and compute
    #   phases; best for small routed working sets (num_tokens*top_k <= 640;
    #   see CuteDslNvfp4MoeRunner::kDecodePrefillCutoverRoutedRows).
    # Prefill backend: global task-queue driven producer/consumer overlap;
    #   best for large routed working sets.
    # Nvfp4MoePlugin scope: FP16 io_dtype + {identity, silu, swiglu, gelu, relu2, geglu}
    # x {decode, prefill} x {n128 MMA N-tile} = 12 variants. Shape axes
    # N / E / top_k / hidden_size (K) are runtime (shape-polymorphic). The
    # MMA N-tile remains a compile-time variant axis. CuteDslNvfp4MoeRunner
    # currently dispatches n128 and accepts the bounded K set {1024, 2048}.
    # Decode backend, N-tile 128
    KernelVariant(
        name="nvfp4_fused_moe_decode_identity_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "identity", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_silu_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "silu", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_swiglu_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_gelu_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "gelu", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_relu2_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_geglu_m128_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "geglu", "--mma_tiler_m", "128",
                     "--mma_tiler_n", "128", "--export_only"],
    ),

    # Decode backend, M-tile 64. Decode routes only top_k rows per token, so a
    # 128-row tile runs mostly empty; m64 is the scheduler's minimum. Measured
    # on SM121 with E=256/I=512/top_k=8: MoE kernel -12.9/-13.4/-15.2/-14.0%
    # at routed_rows 8/24/40/56.
    KernelVariant(
        name="nvfp4_fused_moe_decode_identity_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "identity", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_silu_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "silu", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_swiglu_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_gelu_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "gelu", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_relu2_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_geglu_m64_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "geglu", "--mma_tiler_m", "64",
                     "--mma_tiler_n", "128", "--export_only"],
    ),

    # Prefill backend, N-tile 128
    KernelVariant(
        name="nvfp4_fused_moe_prefill_identity_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "identity", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_silu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "silu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_swiglu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_gelu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "gelu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_relu2_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_geglu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "geglu", "--mma_tiler_n", "128", "--export_only"],
    ),
    # Prefill backend, N-tile 256 — DISABLED (same bug as decode n256).

    # =====================================================================
    # GEMM group — Talker MLP cuBLAS replacement
    #
    # Dispatch strategy per architecture:
    #   Ampere (SM80-89):
    #     M==1           → decode (16×128×128) + Split-K=4 for SM utilization
    #     M=2-95         → small_prefill (16×128×128)
    #     M=96-192       → medium_prefill (64×128×64) + Split-K=2
    #     M=193-383      → medium_prefill (64×128×64)
    #     M=384-640      → large_prefill (128×128×64) + unpredicated epilogue
    #     M>640          → medium_prefill (64×128×64)
    #
    #   Blackwell DC (SM100-110):  tcgen05 + TMA store
    #     M<=4*SMs                 → small  (64×128)  cluster=(1,2)
    #     low-SM GPU AND M>=256    → 2-CTA  (256×256) cluster=(2,1)
    #     else                     → default (128×128) cluster=(1,1)
    #     (e.g. B100 144 SMs → small cap M<=576, no 2-CTA;
    #      Thor 20 SMs → small cap M<=80, 2-CTA above M>=256;
    #      see cuteDslGemmRunner sBlackwellSmallTileMaxM /
    #          sBlackwell2ctaMinM)
    #
    #   BW GeForce (SM120-121):
    #     M<=64          → small (64×128×64) — persistent+warp-spec+TMA
    #     M>=128         → default (128×128×64)
    # =====================================================================
    # --- Ampere (SM80-89) ---
    KernelVariant(
        name="gemm_ampere_decode_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "1,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_small_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "64,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_medium_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_large_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "512,2048,2048",
            "--cta_tiler_mnk", "128,128,64",
            "--atom_layout_mnk", "2,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    # Split-K variant for small M (decode): split_k=4 gives 4x more CTAs.
    KernelVariant(
        name="gemm_ampere_splitk4_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere_streamk.py",
        script_args=[
            "--mnk", "1,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",
            "--split_k", "4",
            "--export_only",
        ],
    ),
    # Split-K=2 for medium M (M=128): doubles CTA count from 32 to 64.
    KernelVariant(
        name="gemm_ampere_splitk2_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere_streamk.py",
        script_args=[
            "--mnk", "128,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",
            "--split_k", "2",
            "--export_only",
        ],
    ),
    # =====================================================================
    # Fused MLP epilogue variants (Plan C: 4→2 kernel launches)
    #
    # FC1 path: GEMM + bias + SiLU fused in epilogue
    # FC2 path: GEMM + bias fused in epilogue
    #
    # Only medium_prefill tile (64×128×64) is fused — it covers the most
    # common prefill M range. Decode (M=1) uses separate bias kernels
    # since the 2us kernel launch is negligible at that scale.
    # =====================================================================
    KernelVariant(
        name="gemm_ampere_medium_bias_silu_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_medium_bias_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # Blackwell DC GEMM tile variants:
    #   "default"  tile=(128,128) cluster=(1,1) — wins for M >= 768 (large MM
    #              has enough M-tiles to keep the GPU busy without sub-tiling)
    #   "small"    tile=(64,128)  cluster=(1,2) — wins for M <= 512  (more
    #              CTAs per wave -> better SM utilization on small/medium M;
    #              cluster (1,2) multicasts B across 2 CTAs in N)
    # See cuteDslGemmRunner::run() for the M threshold dispatch.
    KernelVariant(
        name="gemm_blackwell_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--export_only",
        ],
    ),
    # FP16-in / FP32-out twin of gemm_blackwell_small_fp16: same tile, cluster,
    # and MNK; differs only by --c_dtype float32, which makes the C epilogue
    # write FP32. A and B stay FP16 tensor-core operands.
    KernelVariant(
        name="gemm_blackwell_small_fp16in_fp32out",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--c_dtype", "float32",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # 2-CTA variants (tile=256x256, cluster=(2,1)): paired CTAs share a
    # 2x M tile per tcgen05.mma op. On low-SM-count GPUs (Thor 20 SMs)
    # this beats the single-CTA default for M >= 256 by ~15-25%; on
    # high-SM-count (B100 144 SMs) the single-CTA default already
    # saturates compute so 2-CTA isn't dispatched.
    KernelVariant(
        name="gemm_blackwell_2cta_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_2cta_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_2cta_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # BW GeForce (SM120/121): warp-specialized, TMA, persistent tile scheduling.
    # Small tile for M<=64 — more CTAs on N1Auto's 20 SMs.
    KernelVariant(
        name="gemm_bw_geforce_small_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "64,2048,2048", "--tile_shape_mnk", "64,128,64", "--export_only"],
    ),
    # Default tile for M>=128.
    KernelVariant(
        name="gemm_bw_geforce_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64", "--export_only"],
    ),
    KernelVariant(
        name="gemm_bw_geforce_bias_silu_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64",
                     "--fused_epilogue", "bias_silu", "--export_only"],
    ),
    KernelVariant(
        name="gemm_bw_geforce_bias_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64",
                     "--fused_epilogue", "bias", "--export_only"],
    ),
    # =====================================================================
    # GEMM_NVFP4 group — warp-specialised block-scaled NVFP4 GEMM on
    # Blackwell (SM100/101/103/110). Each variant covers a single
    # (mma_tiler_n) choice at sf_vec_size=16 (NVF4); MNK dims are dynamic
    # at runtime. SM110 support is provided by the CuTeDSL stub Python package
    # used for AOT export.
    #
    # The kernel body restructures load / MMA / store across separate
    # warp roles (epilog warps 0-3 + MMA warp 4 + TMA-load warp 5) to
    # hide TMA latency behind MMA issue.
    #
    # The variants are split into their own group (not reused under
    # "gemm") so that `build_cutedsl.py --kernels gemm_nvfp4` builds
    # them independently of the FP16 `gemm_blackwell_bias*` variants.
    # =====================================================================
    KernelVariant(
        name="gemm_blackwell_nvfp4_ws_fp16_tn64",
        group="gemm_nvfp4",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell_nvfp4_ws.py",
        script_args=[
            "--mnk", "128,256,128",
            "--mma_tiler_n", "64",
            "--sf_vec_size", "16",
            "--c_dtype", "fp16",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_nvfp4_ws_fp16_tn128",
        group="gemm_nvfp4",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell_nvfp4_ws.py",
        script_args=[
            "--mnk", "128,512,128",
            "--mma_tiler_n", "128",
            "--sf_vec_size", "16",
            "--c_dtype", "fp16",
            "--export_only",
        ],
    ),
    # tn256: larger N-tile for high-throughput shapes, combined with the
    # persistent tile scheduler.
    KernelVariant(
        name="gemm_blackwell_nvfp4_ws_fp16_tn256",
        group="gemm_nvfp4",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell_nvfp4_ws.py",
        script_args=[
            "--mnk", "128,512,128",
            "--mma_tiler_n", "256",
            "--sf_vec_size", "16",
            "--c_dtype", "fp16",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_nvfp4_ws_fp8_tn64",
        group="gemm_nvfp4",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell_nvfp4_ws.py",
        script_args=[
            "--mnk", "128,256,128",
            "--mma_tiler_n", "64",
            "--sf_vec_size", "16",
            "--c_dtype", "fp8_e4m3",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_nvfp4_ws_fp8_tn128",
        group="gemm_nvfp4",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell_nvfp4_ws.py",
        script_args=[
            "--mnk", "128,512,128",
            "--mma_tiler_n", "128",
            "--sf_vec_size", "16",
            "--c_dtype", "fp8_e4m3",
            "--export_only",
        ],
    ),
]


# RMSNorm is specialized by storage dtype, hidden size, and weight-before-cast
# mode. The row count and epsilon remain runtime arguments in each AOT ABI.
_RMSNORM_SUPPORTED_SMS = [80, 86, 87, 90, 100, 101, 110, 120, 121]
_RMSNORM_HIDDEN_SIZES = [4096, 5120, 7168, 8192]
_RMSNORM_WEIGHT_BEFORE_CAST_MODES = [0, 1]
for _rmsnorm_dtype in ("fp16", "bf16"):
    for _rmsnorm_hidden_size in _RMSNORM_HIDDEN_SIZES:
        for _rmsnorm_weight_before_cast in _RMSNORM_WEIGHT_BEFORE_CAST_MODES:
            KERNEL_VARIANTS.append(
                KernelVariant(
                    name=(f"rmsnorm_{_rmsnorm_dtype}_h{_rmsnorm_hidden_size}"
                          f"_wbc{_rmsnorm_weight_before_cast}"),
                    group="rmsnorm",
                    supported_sms=_RMSNORM_SUPPORTED_SMS,
                    script="rmsnorm_cutedsl/rmsnorm.py",
                    script_args=[
                        "--dtype", _rmsnorm_dtype,
                        "--hidden_size", str(_rmsnorm_hidden_size),
                        "--weight_before_cast", str(_rmsnorm_weight_before_cast),
                        "--export_only",
                    ],
                )
            )


# ---------------------------------------------------------------------------
# nvfp4_a16_blackwell_gemm group — dense W4A16 TCGen5 GEMM for SM110.
#
# The AOT entry point consumes the export-time opaque layouts directly:
#   qweight [N/128, K/64, 128, 32] packed E2M1 bytes
#   scale   [N/128, K/64, 128, 4] raw E4M3 bytes
# M/N/K and the persistent cluster count are runtime arguments. Each variant
# bakes only the activation dtype and post-transpose MMA (TM,TN,TK) tile.
# Logical M > 256 is grid-tiled by the tn256 variant.
# ---------------------------------------------------------------------------
_NVFP4_A16_BLACKWELL_TOKEN_TILES = (8, 16, 32, 64, 128, 256)
_NVFP4_A16_BLACKWELL_DTYPES = ("fp16", "bf16")

for _io_dtype in _NVFP4_A16_BLACKWELL_DTYPES:
    for _token_tile in _NVFP4_A16_BLACKWELL_TOKEN_TILES:
        KERNEL_VARIANTS.append(
            KernelVariant(
                name=(
                    f"nvfp4_a16_blackwell_gemm_{_io_dtype}_"
                    f"tm128_tn{_token_tile}_tk64"
                ),
                group="nvfp4_a16_blackwell_gemm",
                supported_sms=[110],
                script=(
                    "nvfp4_a16_blackwell_gemm/"
                    "nvfp4_a16_blackwell_gemm.py"
                ),
                script_args=[
                    "--io_dtype", _io_dtype,
                    "--token_tile", str(_token_tile),
                    "--export_only",
                ],
            )
        )


# ---------------------------------------------------------------------------
# int4_fp16_gemm group — W4A16 INT4-weight FP16 GEMM.  Ampere instruction floor
# (cp.async + mma.sync 16x8x16 + ldmatrix), forward-compatible to SM80 and newer
# (Ampere / Ada / Hopper / Blackwell).
#
# These are GENERATED rather than hand-listed: an AOT artifact has no runtime
# autotune, so the baked config set IS the plugin's autotune universe.  swizzle
# (grouped-M raster) stays a *runtime* Int32 kernel arg, so it is NOT a baked
# dimension.  split_k>1 uses an in-kernel reduction; a baked split_k=N is correct
# only when N divides ceil(K/64) (split_k=1 always works).
#
# bN is PINNED to 128 so the offline fragment weight repack is tile-independent
# (a single buffer serves every variant; Int4GroupwiseGemmPluginV2 / export-time
# repack). The bN=256 tile (16x256x64) is intentionally excluded.
#
# The full config space has size 60: 4 CTA tiles x {2,3,4} stages x
# {1,2,4,8,16} split-K = 60.  The 16-config subset below is selected for close
# to optimal perf on all three target SKUs — Orin (SM87), Thor (SM110), and DGX
# Spark (SM121).  Derived by greedy set-cover over clock-locked autotune traces;
# shrinks the baked universe — and thus the plugin's per-shape autotune candidate
# set — from 60 to 16 (~3.75x fewer tactics to time).  Split-K 8 and 16 are
# dropped entirely (never needed within 2% on any cell of any SKU).
#
# Each entry is (bM, stages, split_k); bN=128, bK=64 fixed.
# NOTE: keep this list in sync with INT4_FP16_GEMM_VARIANTS in
# cpp/plugins/int4GroupwiseGemmPluginV2/cuteDslInt4Gemm.cpp (same 16, same order).
_INT4_FP16_GEMM_CONFIGS = [
    (16, 3, 1), (16, 4, 1), (16, 4, 2), (16, 4, 4),
    (32, 2, 1), (32, 3, 2), (32, 3, 4), (32, 4, 1),
    (64, 2, 1), (64, 3, 1), (64, 3, 4), (64, 4, 2), (64, 4, 4),
    (128, 2, 1), (128, 2, 4), (128, 4, 1),
]
# Quant group size, baked per variant (it sizes the in-kernel scale smem, so it
# is compile-time, not a runtime arg). The kernel supports {16, 32, 64, 128, ...}
# (any multiple of 16 mutually divisible with bK=64); only G=128 is baked today.
# Single knob: set to 32 to build G=32 instead, or make it a list + add a
# `_g{gs}` suffix to the variant name below to ship both group sizes at once.
_INT4_FP16_GEMM_GROUP_SIZE = 128

for _bm, _stages, _sk in _INT4_FP16_GEMM_CONFIGS:
    _tile_tag = f"{_bm}x128x64"
    KERNEL_VARIANTS.append(
        KernelVariant(
            name=f"int4_fp16_gemm_{_tile_tag}_s{_stages}_sk{_sk}",
            group="int4_fp16_gemm",
            supported_sms=[80, 86, 87, 89, 100, 101, 110, 120, 121],
            script="int4_fp16_gemm_cutedsl/int4_fp16_gemm_ampere.py",
            script_args=[
                "--mnk", "256,512,1024",
                "--cta_tiler_mnk", f"{_bm},128,64",
                "--atom_layout_mnk", "1,4,1",
                "--num_stages", str(_stages),
                "--split_k", str(_sk),
                "--group_size", str(_INT4_FP16_GEMM_GROUP_SIZE),
                "--export_only",
            ],
        )
    )

# int4_fp16_gemm group also includes the W4A16 decode GEMV — a CUDA-core kernel
# for the decode regime (small M) that consumes the SAME offline fragment weight
# buffer as the GEMM.  Prefill (GEMM) and decode (GEMV) are always needed together
# (one weight copy serves both), so the GEMV is baked under the same group /
# CUTE_DSL_INT4_FP16_GEMM_ENABLED define rather than a separate selectable group.
# M is the baked dimension — one exported function per M in [1, 8] (8 total); N/K
# stay dynamic.  Single W=8 config; the kernel bakes SKU-independent per-M tuning
# (UNROLL2/MINB from _gemv_defaults) — no device-SM detection, so the AOT config
# is identical on every arch.  group_size 128 only for now (the kernel also
# supports 32).
# ---------------------------------------------------------------------------
_INT4_FP16_GEMV_MAX_M = 8
_INT4_FP16_GEMV_GROUP_SIZE = 128

for _m in range(1, _INT4_FP16_GEMV_MAX_M + 1):
    KERNEL_VARIANTS.append(
        KernelVariant(
            name=f"int4_fp16_gemv_m{_m}",
            group="int4_fp16_gemm",
            supported_sms=[80, 86, 87, 89, 100, 101, 110, 120, 121],
            script="int4_fp16_gemm_cutedsl/int4_fp16_gemv_ampere.py",
            script_args=[
                "--mnk", f"{_m},512,1024",
                "--group_size", str(_INT4_FP16_GEMV_GROUP_SIZE),
                "--export_only",
            ],
        )
    )

# All known group names (set for O(1) membership check — no manual maintenance
# needed).
_ALL_GROUPS: set[str] = {v.group for v in KERNEL_VARIANTS}


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

def _parse_sm(gpu_arch_str):
    """Parse SM number from "sm_87" → 87, or raise ValueError."""
    s = gpu_arch_str.strip().lower()
    if s.startswith("sm_"):
        s = s[3:]
    try:
        sm = int(s)
    except ValueError:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}. Expected format: sm_87, sm_100, etc."
        )
    if sm <= 0:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}: SM number must be positive (got {sm})."
        )
    return sm


def detect_gpu_sm() -> int:
    """Auto-detect the current GPU SM.

    Returns the SM as an integer, e.g. 87 for SM87, 100 for SM100, 110 for SM110.

    Detection order:
      1. cupy.cuda.Device — works on all platforms (Linux, QNX, etc.) since cupy is
         already a required dependency.  compute_capability returns e.g. "87", "100".
      2. nvidia-smi --query-gpu=compute_cap — fallback for environments where cupy
         is not yet importable at this point in the script (rare).

    Raises RuntimeError if both methods fail; caller should re-run with --gpu_arch.
    """
    # 1. Try cupy first — platform-agnostic, already a required dep.
    try:
        import cupy  # noqa: PLC0415
        cap = cupy.cuda.Device(0).compute_capability  # e.g. "87", "100", "110"
        sm = int(cap)
        if sm > 0:
            return sm
    except Exception:
        pass

    # 2. Fall back to nvidia-smi (Linux/x86; not available on QNX).
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not detect GPU SM: cupy unavailable and nvidia-smi not found. "
            "Pass --gpu_arch explicitly (e.g. --gpu_arch sm_87)."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}. "
            "Pass --gpu_arch explicitly to override."
        )
    # compute_cap format from nvidia-smi is "8.7" → 87, "10.0" → 100.
    line = result.stdout.strip().splitlines()[0].strip()
    parts = line.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise RuntimeError(
            f"Unexpected nvidia-smi compute_cap format: {line!r}. "
            "Pass --gpu_arch explicitly to override."
        )
    return int(parts[0]) * 10 + int(parts[1])


def select_variants(sm: int, kernels_arg: str):
    """Return the list of KernelVariants to compile for the given SM.

    sm:
      Integer SM number (e.g. 87, 100, 110) — used for supported_sms filtering.

    kernels_arg:
      "ALL"         — compile variants whose supported_sms contains the SM.
      "gdn"/"fmha"  — compile variants in that group whose supported_sms contains the SM.
      "gdn,fmha"    — same for the listed groups (unsupported variants are skipped).
    """
    groups_requested = kernels_arg.strip().upper()

    if groups_requested == "ALL":
        selected = [v for v in KERNEL_VARIANTS if sm in v.supported_sms]
        if not selected:
            print(f"WARNING: No CuTe DSL variants support SM{sm}. "
                  f"Check supported_sms in KERNEL_VARIANTS.")
        return selected

    # Parse explicit group list: "fmha,gdn" or "gdn".
    tokens = [t.strip().lower() for t in kernels_arg.split(",")]
    unknown = [t for t in tokens if t not in _ALL_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown kernel group(s): {unknown}. "
            f"Valid groups: {sorted(_ALL_GROUPS)}"
        )

    in_groups = [v for v in KERNEL_VARIANTS if v.group in tokens]
    skipped = [v for v in in_groups if sm not in v.supported_sms]
    selected = [v for v in in_groups if sm in v.supported_sms]

    if skipped:
        names = ", ".join(v.name for v in skipped)
        print(
            f"NOTE: Skipping {len(skipped)} variant(s) not supported on SM{sm}: {names}"
        )

    if selected:
        return selected

    # Explicit group list requested, but no variant in those groups supports the SM.
    requested_variants = [v for v in KERNEL_VARIANTS if v.group in tokens]
    names = ", ".join(v.name for v in requested_variants)
    raise ValueError(
        f"No variants in groups {tokens} support SM{sm}.\n"
        f"Requested variants: {names}\n"
        f"Use --kernels ALL to auto-filter across all groups, or check supported_sms in KERNEL_VARIANTS."
    )


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def detect_arch(override=None):
    if override:
        m = override.lower().replace("-", "_")
        if m in ("x86_64", "amd64"):
            return "x86_64"
        if m in ("aarch64", "arm64"):
            return "aarch64"
        raise ValueError(f"Unsupported --arch: {override!r}. Use 'x86_64' or 'aarch64'.")
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(
        f"Unsupported architecture: {platform.machine()!r}. Use --arch to override."
    )


def sm_to_artifact_tag(sm: int) -> str:
    return f"sm_{sm}"


def default_compile_gpu_arch(sm: int) -> str:
    """Return the CuTe DSL compile arch for an SM number.

    Blackwell tcgen05 kernels require the architecture suffix (for example,
    sm_110a) even though artifact tags and variant filtering use sm_110.
    """
    if sm in (100, 101, 103, 107, 109, 110, 120, 121):
        return f"sm_{sm}a"
    return f"sm_{sm}"


def default_host_target_for_arch(target_arch: str, host_arch: str) -> str:
    if target_arch == "aarch64" and host_arch != target_arch:
        return "linux-aarch64"
    if target_arch == "x86_64":
        # Pin a generic x86-64 host target so the exported host-side launch code uses
        # the portable baseline ISA; an empty target bakes in the build machine's
        # native ISA (e.g. AVX-512 on Zen4 runners), which SIGILLs on older deployment CPUs.
        return  "llvm -mtriple=x86_64-unknown-linux-gnu"
    return ""


def _nvcc_version():
    """Return CUDA version string (e.g. "12.6.0") or None.

    Detection order:
      1. nvcc on PATH
      2. /usr/local/cuda/bin/nvcc  (common on Jetson / embedded devices)
      3. cupy.cuda.runtime          (works on QNX and any platform with cupy)
    """
    # 1 & 2: try nvcc
    for nvcc in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            for token in out.split():
                if token.startswith("V") and token[1:2].isdigit():
                    return token[1:].split(",")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 3: cupy runtime API — runtimeGetVersion() returns e.g. 12060 for 12.6.0
    try:
        import cupy  # noqa: PLC0415
        v = cupy.cuda.runtime.runtimeGetVersion()   # e.g. 12060
        major, rest = divmod(v, 1000)
        minor, patch = divmod(rest, 10)
        return f"{major}.{minor}.{patch}"
    except Exception:
        pass

    return None


def _cutlass_dsl_install_hint(cuda_ver):
    package = "nvidia-cutlass-dsl"
    if cuda_ver is not None and cuda_ver.split(".")[0] in ("12", "13"):
        package = f"nvidia-cutlass-dsl[cu{cuda_ver.split('.')[0]}]"
    return f"pip install '{package}=={_CUTLASS_DSL_VERSION}'"


def _cutlass_dsl_lib_dir(pkg_dir, cuda_ver):
    candidates = []
    if cuda_ver:
        candidates.append(pkg_dir / f"cu{cuda_ver.split('.')[0]}" / "lib")
    candidates.append(pkg_dir / "lib")

    for candidate in candidates:
        if (candidate / "libcuda_dialect_runtime_static.a").exists():
            return candidate
    raise FileNotFoundError(
        f"libcuda_dialect_runtime_static.a not found in any of: "
        f"{[str(c) for c in candidates]}. "
        f"Ensure nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION} is installed correctly."
    )


def _cutlass_dsl_version_matches(version: str) -> bool:
    """Allow dev/local wheels built from the pinned release."""
    public_version = version.split("+", 1)[0]
    return public_version == _CUTLASS_DSL_VERSION or public_version.startswith(
        f"{_CUTLASS_DSL_VERSION}.dev"
    )



def check_dependencies(sm=None, selected_groups=None, cuda_ver=None):
    errors = []
    selected_groups = set(selected_groups or [])

    # Detect the system CUDA version. This drives the cupy version check, which
    # must always match the CUDA toolkit actually installed on the host.
    system_cuda_ver = _nvcc_version()

    # Effective CUDA version used to select the CuTeDSL runtime libs. An explicit
    # override (e.g. from --cuda-version) takes precedence over auto-detection,
    # which is useful when cross-compiling and the host nvcc reports a different
    # CUDA major than the target CuTeDSL runtime libs. It does NOT affect the
    # cupy check above.
    if cuda_ver is None:
        cuda_ver = system_cuda_ver
    cutlass_dsl_fix = _cutlass_dsl_install_hint(cuda_ver)

    # nvidia-cutlass-dsl
    try:
        ver = importlib.metadata.version("nvidia-cutlass-dsl")
        if not _cutlass_dsl_version_matches(ver):
            errors.append(
                f"nvidia-cutlass-dsl: found {ver}, need {_CUTLASS_DSL_VERSION}\n"
                f"  Fix: {cutlass_dsl_fix}"
            )
            lib_dir = None
        else:
            spec = importlib.util.find_spec("nvidia_cutlass_dsl")
            pkg_dir = (
                Path(next(iter(spec.submodule_search_locations)))
                if spec.submodule_search_locations
                else Path(spec.origin).parent
            )
            try:
                lib_dir = _cutlass_dsl_lib_dir(pkg_dir, cuda_ver)
            except FileNotFoundError as e:
                errors.append(str(e))
                lib_dir = None
    except importlib.metadata.PackageNotFoundError:
        errors.append(
            f"nvidia-cutlass-dsl not found.\n"
            f"  Fix: {cutlass_dsl_fix}"
        )
        lib_dir, ver = None, "unknown"

    # Verify that the package loader selected the requested compiler backend.
    # CuTe DSL can contain both cu12 and cu13 binaries, but it otherwise
    # chooses the newest flavor supported by the driver. Probe in a child
    # process so importing the MLIR runtime does not affect later process-pool
    # workers.
    dsl_cuda_version_string = None
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cutlass; print(f'{cutlass.CUDA_VERSION.major}."
            "{cutlass.CUDA_VERSION.minor}')",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        dsl_cuda_version_string = probe.stdout.strip().splitlines()[-1]
        try:
            dsl_cuda_major = int(dsl_cuda_version_string.split(".", 1)[0])
        except ValueError:
            errors.append(
                "Could not parse the loaded CuTe DSL CUDA flavor: "
                f"{dsl_cuda_version_string!r}"
            )
        else:
            requested_cuda_major = int(cuda_ver.split(".")[0]) if cuda_ver else None
            if requested_cuda_major is not None and dsl_cuda_major != requested_cuda_major:
                errors.append(
                    f"CuTe DSL loaded the cu{dsl_cuda_major} compiler backend, but "
                    f"artifact CUDA {requested_cuda_major} was requested.\n"
                    "  Use an environment containing only the requested CuTe DSL flavor."
                )
    else:
        probe_error = (probe.stderr or probe.stdout).strip()
        errors.append(
            "Could not determine the loaded CuTe DSL CUDA flavor: "
            f"{probe_error or 'probe produced no output'}"
        )

    # cupy — validated against the system CUDA version, not the --cuda-version
    # override, since cupy must match the CUDA toolkit installed on the host.
    cupy_pkg = None
    cupy_ver = None
    if system_cuda_ver:
        major = int(system_cuda_ver.split(".")[0])
        if major in _CUPY_VERSIONS:
            cupy_pkg, cupy_req = _CUPY_VERSIONS[major]
            try:
                found = importlib.metadata.version(cupy_pkg)
                cupy_ver = found
                if found != cupy_req:
                    errors.append(
                        f"cupy: found {cupy_pkg}=={found}, need {cupy_req}\n"
                        f"  Fix: pip install {cupy_pkg}=={cupy_req}"
                    )
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"cupy not found.\n  Fix: pip install {cupy_pkg}=={cupy_req}")
        else:
            errors.append(f"Unsupported CUDA major version {major} for cupy.")
    else:
        errors.append("Could not detect CUDA version (is nvcc on PATH?).")

    if not shutil.which("ar"):
        errors.append("'ar' not found on PATH. Install binutils.")

    # cuda-python (provides `cuda.bindings.driver`, used by all GEMM/FMHA scripts)
    try:
        importlib.metadata.version("cuda-python")
    except importlib.metadata.PackageNotFoundError:
        errors.append("cuda-python not found.\n  Fix: pip install cuda-python")

    if errors:
        print("Dependency check failed:\n" + "\n".join(f"  • {e}" for e in errors))
        sys.exit(1)

    assert (
        ver is not None
        and lib_dir is not None
        and cuda_ver is not None
        and system_cuda_ver is not None
        and cupy_pkg is not None
        and cupy_ver is not None
        and dsl_cuda_version_string is not None
    )
    print(
        f"  nvidia-cutlass-dsl=={ver} (CUDA {dsl_cuda_version_string}) ✓  "
        f"{cupy_pkg}=={cupy_ver} ✓  "
        f"host CUDA {system_cuda_ver} ✓  artifact CUDA {cuda_ver} ✓  ar ✓"
    )
    return (
        ver,
        lib_dir,
        cuda_ver,
        system_cuda_ver,
        cupy_pkg,
        cupy_ver,
        dsl_cuda_version_string,
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _compile_command(variant, staging_dir, compile_gpu_arch, host_target):
    script = _SCRIPT_DIR / variant.script
    if compile_gpu_arch or host_target:
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "cutedsl_utils/cutedsl_compile_wrapper.py"),
            "--script",
            str(script),
        ]
        if compile_gpu_arch:
            cmd += ["--gpu-arch", compile_gpu_arch]
        if host_target:
            cmd += ["--host-target", host_target]
        cmd += ["--"]
    else:
        cmd = [sys.executable, str(script)]

    cmd += ["--output_dir", str(staging_dir),
            "--file_name", variant.name,
            "--function_prefix", variant.name]
    cmd += variant.script_args
    return cmd


def _compile_one(variant, staging_dir, verbose, sm, compile_gpu_arch, host_target):
    """Invoke a kernel script to AOT-compile one variant into .o + .h.

    Returns (name, ok, elapsed_secs, error_msg).
    """
    cmd = _compile_command(variant, staging_dir, compile_gpu_arch, host_target)

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, cwd=str(_SCRIPT_DIR), capture_output=not verbose, text=True
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        # Show the head (traceback / first error) rather than the tail, which is typically more diagnostic.
        return variant.name, False, elapsed, (result.stderr or result.stdout or "")[:4000]
    obj = staging_dir / f"{variant.name}.o"
    hdr = staging_dir / f"{variant.name}.h"
    if not obj.exists() or not hdr.exists():
        # Some variants (e.g. gdn_decode_mtp --cache_only) produce artifacts with
        # a suffix (e.g. gdn_decode_mtp_cache.o/.h).  Accept any .o + .h pair.
        any_objs = list(staging_dir.glob("*.o"))
        any_hdrs = list(staging_dir.glob("*.h"))
        if not any_objs or not any_hdrs:
            return variant.name, False, elapsed, f"{obj.name} / {hdr.name} not found after successful exit"
    return variant.name, True, elapsed, ""


def compile_variants(variants, staging_dirs, jobs, verbose, sm, compile_gpu_arch, host_target):
    """Compile all selected variants in parallel via a process pool.

    staging_dirs: dict mapping variant.name → Path of its dedicated staging dir.
    """
    print(f"\nCompiling {len(variants)} kernel variant(s) (jobs={jobs})...")
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_compile_one, v, staging_dirs[v.name], verbose, sm, compile_gpu_arch, host_target): v
            for v in variants
        }
        for future in concurrent.futures.as_completed(futures):
            name, ok, elapsed, msg = future.result()
            print(f"  {'✓' if ok else '✗'} {name:<25} ({elapsed:.1f}s)")
            if not ok:
                failures.append((name, msg))

    if failures:
        for name, msg in failures:
            print(f"\n  [{name}]\n{msg}")
        sys.exit(1)

# ELF e_machine values for validating CuTe DSL runtime static objects.
_ELF_MACHINE = {"x86_64": 0x3E, "aarch64": 0xB7}
_RUNTIME_STATIC_ARCHIVE = "libcuda_dialect_runtime_static.a"


def _elf_machine(path: Path) -> int | None:
    """Return the ELF e_machine field of a shared object, or None if not ELF."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(20)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    byteorder = "little" if header[5] == 1 else "big"
    return int.from_bytes(header[18:20], byteorder)


def _cuda_package_variant(cuda_ver: str) -> str:
    """Return the nvidia-cutlass-dsl CUDA package variant for a CUDA version."""
    return f"cu{cuda_ver.split('.')[0]}"


def _runtime_libs_package(cuda_ver: str) -> str:
    """Return the distribution containing CuTe DSL runtime libraries."""
    return f"nvidia-cutlass-dsl-libs-{_cuda_package_variant(cuda_ver)}"


def _find_static_runtime_archive(pkg_dir: Path | None, cuda_ver: str) -> Path | None:
    """Locate libcuda_dialect_runtime_static.a inside an installed package tree."""
    if pkg_dir is None:
        return None

    cuda_variant_archive = (
        pkg_dir / _cuda_package_variant(cuda_ver) / "lib" / _RUNTIME_STATIC_ARCHIVE
    )
    if cuda_variant_archive.exists():
        return cuda_variant_archive

    generic_archive = pkg_dir / "lib" / _RUNTIME_STATIC_ARCHIVE
    if generic_archive.exists():
        return generic_archive

    return None


def _archive_first_member_machine(archive: Path) -> int | None:
    members = subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines()
    if not members:
        return None
    with tempfile.TemporaryDirectory(prefix="cutedsl_ar_check_") as tmp:
        subprocess.run(["ar", "x", str(archive), members[0]], cwd=tmp, check=True)
        return _elf_machine(Path(tmp) / members[0])


def _validate_static_runtime_archive(archive: Path, target_arch: str) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"{archive} does not exist")
    machine = _archive_first_member_machine(archive)
    want = _ELF_MACHINE.get(target_arch)
    if want is not None and machine != want:
        got = "not-ELF" if machine is None else f"0x{machine:x}"
        raise RuntimeError(
            f"Runtime static archive architecture mismatch: {archive}\n"
            f"  e_machine={got}, but target arch '{target_arch}' expects 0x{want:x}.\n"
            "Use a nvidia-cutlass-dsl-libs package whose wheel platform matches --arch."
        )
    return archive


def _target_wheel_platforms(target_arch: str) -> list[str]:
    if target_arch == "aarch64":
        return ["manylinux_2_28_aarch64", "manylinux2014_aarch64"]
    if target_arch == "x86_64":
        return ["manylinux_2_28_x86_64", "manylinux2014_x86_64"]
    return []


def _download_runtime_libs_wheel(
    target_arch: str, cuda_ver: str, dsl_ver: str, download_dir: Path
) -> Path:
    """Download the target-arch CuTe DSL runtime-libs wheel and return its path.

    Only the small runtime-libs distribution is downloaded with --no-deps.
    Downloading nvidia-cutlass-dsl[cuXX] for a foreign platform asks pip to
    resolve the full dependency tree and can fail on tool-only packages.
    """
    package = _runtime_libs_package(cuda_ver)
    wheelhouse = os.environ.get("CUTE_DSL_RUNTIME_WHEELHOUSE")
    if wheelhouse and not Path(wheelhouse).is_dir():
        raise FileNotFoundError(
            f"CUTE_DSL_RUNTIME_WHEELHOUSE does not exist or is not a directory: {wheelhouse}"
        )
    versions = []
    for version in (dsl_ver, dsl_ver.split("+", 1)[0]):
        if version and version not in versions:
            versions.append(version)

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platforms = _target_wheel_platforms(target_arch)
    if not platforms:
        raise RuntimeError(f"No Python wheel platform mapping for target arch '{target_arch}'.")

    errors = []
    download_dir.mkdir(parents=True, exist_ok=True)
    for version in versions:
        for wheel_platform in platforms:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                f"{package}=={version}",
                "--no-deps",
                "--only-binary=:all:",
                "--platform",
                wheel_platform,
                "--python-version",
                py_version,
                "--implementation",
                "cp",
                "--abi",
                py_abi,
                "-d",
                str(download_dir),
            ]
            if wheelhouse:
                cmd += ["--no-index", "--find-links", wheelhouse]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                wheels = sorted(download_dir.glob(f"{package.replace('-', '_')}*.whl"))
                if wheels:
                    return wheels[-1]
            errors.append(
                f"{package}=={version} for {wheel_platform} failed:\n"
                f"{(result.stderr or result.stdout).strip()}"
            )

    raise RuntimeError(
        "Could not download target CuTe DSL runtime-libs wheel.\n"
        + "\n\n".join(errors[-2:])
    )


def _extract_static_runtime_archive_from_wheel(
    wheel: Path, extract_dir: Path, cuda_ver: str
) -> Path:
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(extract_dir)
    archive = (
        extract_dir
        / "nvidia_cutlass_dsl"
        / _cuda_package_variant(cuda_ver)
        / "lib"
        / _RUNTIME_STATIC_ARCHIVE
    )
    if not archive.exists():
        raise FileNotFoundError(f"{_RUNTIME_STATIC_ARCHIVE} not found in downloaded wheel: {wheel}")
    return archive


def resolve_static_runtime_archive(
    target_arch: str,
    host_arch: str,
    pkg_dir: Path | None,
    cuda_ver: str,
    dsl_ver: str,
    staging_dir: Path,
) -> Path:
    """Return a target-arch libcuda_dialect_runtime_static.a.

    Native builds use the installed package directly. Cross builds first try the
    installed package, then download the target-arch runtime-libs wheel and use
    the static archive from it.
    """
    installed = _find_static_runtime_archive(pkg_dir, cuda_ver)
    if installed is not None:
        try:
            return _validate_static_runtime_archive(installed, target_arch)
        except RuntimeError:
            if target_arch == host_arch:
                raise

    wheel_dir = staging_dir / "runtime_wheel"
    wheel = _download_runtime_libs_wheel(target_arch, cuda_ver, dsl_ver, wheel_dir)
    extracted = staging_dir / "runtime_wheel_extracted"
    archive = _extract_static_runtime_archive_from_wheel(wheel, extracted, cuda_ver)
    return _validate_static_runtime_archive(archive, target_arch)


# Metadata keys that must match for a partial rebuild to merge into an
# existing artifact. A mismatch means the directory holds an artifact of a
# different flavor/target and merging would mislabel it.
_METADATA_COMPAT_KEYS = (
    "arch",
    "artifact_tag",
    "gpu_arch",
    "compile_gpu_arch",
    "host_target",
    "cuda_package_variant",
    "cutlass_dsl_version",
)


def _load_existing_metadata(metadata_path: Path):
    """Return the parsed existing metadata.json, or None if absent/unreadable."""
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return None


def _merge_artifact_metadata(existing, new):
    """Merge a partial (re)build's metadata into an existing artifact's.

    Without --clean, `ar rcs` replaces the rebuilt variants' objects and keeps
    the rest, so the artifact keeps containing every previously built group.
    Union groups/variants so metadata.json (which CMake uses to enable kernel
    call sites) keeps describing the whole artifact instead of only the last
    build's selection. Raises on incompatible provenance — that indicates the
    directory holds a different flavor/target; use --clean instead.
    """
    if existing is None:
        return new
    mismatched = [
        f"  {key}: existing={existing.get(key)!r}, new={new[key]!r}"
        for key in _METADATA_COMPAT_KEYS
        if existing.get(key) != new[key]
    ]
    if mismatched:
        raise RuntimeError(
            "Refusing to merge into an incompatible existing artifact:\n"
            + "\n".join(mismatched)
            + "\nRe-run with --clean to replace it."
        )
    merged = dict(new)
    merged["groups"] = sorted(set(existing.get("groups") or []) | set(new["groups"]))
    merged["variants"] = sorted(set(existing.get("variants") or []) | set(new["variants"]))
    return merged


def _check_obj_name_collision(kernel_objs, runtime_objs):
    kernel_names = {f.name for f in kernel_objs}
    runtime_names = {f.name for f in runtime_objs}
    collision = kernel_names & runtime_names
    if collision:
        raise RuntimeError(
            f"Object name collision between kernel and runtime archives: {collision}\n"
            "Rename the affected kernel variant(s) in KERNEL_VARIANTS to resolve."
        )


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build(args):
    # Resolve SM: explicit override or auto-detect from the running GPU.
    if args.gpu_arch:
        sm = _parse_sm(args.gpu_arch)
        sm_source = "--gpu_arch override"
    else:
        sm = detect_gpu_sm()
        sm_source = "auto-detected"

    host_arch = detect_arch()
    arch = detect_arch(args.arch)
    is_cross_compile = arch != host_arch
    artifact_tag = sm_to_artifact_tag(sm)
    output_dir = Path(args.output_dir) / arch / artifact_tag
    host_target = default_host_target_for_arch(arch, host_arch)
    compile_gpu_arch = default_compile_gpu_arch(sm)
    wrapper_gpu_arch = compile_gpu_arch

    print(f"Build host  : {host_arch}")
    print(f"Target arch : {arch}")
    print(f"Build mode  : {'cross' if is_cross_compile else 'native'}")
    print(f"GPU SM      : SM{sm} ({sm_source})")
    if compile_gpu_arch:
        print(f"Compile arch: {compile_gpu_arch}")
    if host_target:
        print(f"Host target : {host_target}")
    print(f"Artifact tag: {artifact_tag}")
    print(f"Output dir  : {output_dir}")

    variants = select_variants(sm, args.kernels)
    if not variants:
        print("No variants selected — nothing to build.")
        return

    groups_selected = sorted({v.group for v in variants})

    # Check dependencies before cleaning — so a failed dep check doesn't
    # silently destroy a previously good build.
    print("\nChecking dependencies...")
    (
        dsl_ver,
        lib_dir,
        cuda_ver,
        host_cuda_ver,
        cupy_pkg,
        cupy_ver,
        dsl_cuda_version,
    ) = check_dependencies(sm=sm, selected_groups=groups_selected, cuda_ver=args.cuda_version)
    runtime_libs_version = args.runtime_libs_version or dsl_ver

    print(f"Groups      : {groups_selected}")
    print(f"Variants    : {[v.name for v in variants]}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Per-variant staging dirs prevent .o / .h filename collisions when multiple
    # variants share the same underlying script (e.g. all fmha.py invocations).
    root_staging = Path(tempfile.mkdtemp(prefix="cutedsl_build_"))
    try:
        staging_dirs = {}
        for v in variants:
            d = root_staging / v.name
            d.mkdir()
            staging_dirs[v.name] = d

        compile_variants(variants, staging_dirs, args.jobs, args.verbose, sm, wrapper_gpu_arch, host_target)

        # Collect all .o files from per-variant staging dirs.
        # Some variants (e.g. gdn_decode_mtp) produce multiple .o files.
        kernel_obj_files = []
        for v in variants:
            kernel_obj_files.extend(sorted(staging_dirs[v.name].glob("*.o")))

        # Extract the target-arch CuTe DSL static runtime shim and pack it with
        # the kernel objects. Cross builds download the target-arch
        # nvidia-cutlass-dsl-libs-cuXX wheel when the installed package only
        # contains host-arch objects. This keeps artifacts self-contained without
        # depending on the large libcute_dsl_runtime.so dynamic-loading runtime.
        runtime_archive = resolve_static_runtime_archive(
            arch,
            host_arch,
            lib_dir.parent if lib_dir else None,
            cuda_ver,
            runtime_libs_version,
            root_staging,
        )
        runtime_obj_dir = root_staging / "runtime_objs"
        runtime_obj_dir.mkdir()
        subprocess.run(["ar", "x", str(runtime_archive)], cwd=str(runtime_obj_dir), check=True)
        runtime_objs = sorted(runtime_obj_dir.glob("*.o"))
        _check_obj_name_collision(kernel_obj_files, runtime_objs)

        # Pack kernel objects and runtime shim objects into one archive.
        output_dir.mkdir(parents=True, exist_ok=True)
        lib_path = output_dir / f"libcutedsl_{arch}.a"
        subprocess.run(
            ["ar", "rcs", str(lib_path)]
            + [str(o) for o in kernel_obj_files]
            + [str(o) for o in runtime_objs],
            check=True,
        )
        print(f"\n  Created {lib_path.name} ({lib_path.stat().st_size // 1024} KB)")
        print(
            f"  Embedded static runtime: {runtime_archive} "
            f"({runtime_archive.stat().st_size // 1024} KB)"
        )

        # Copy per-variant headers and write umbrella header.
        # Some variants produce multiple .h files (e.g. gdn_decode_mtp + gdn_decode_mtp_cache).
        inc_dir = output_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        for v in variants:
            for hdr in sorted(staging_dirs[v.name].glob("*.h")):
                shutil.copy2(hdr, inc_dir)

        # Per-group umbrella headers (e.g. cutedsl_nvfp4_moe_all.h).
        for group in groups_selected:
            group_headers = []
            for v in [vv for vv in variants if vv.group == group]:
                for hdr in sorted(staging_dirs[v.name].glob("*.h")):
                    group_headers.append(hdr.name)
            group_umbrella = inc_dir / f"cutedsl_{group}_all.h"
            group_umbrella.write_text(
                "#pragma once\n"
                "// Auto-generated by build_cutedsl.py -- do not edit\n"
                + "".join(f'#include "{h}"\n' for h in group_headers)
            )

        # Unified umbrella header. Enumerate the per-variant headers actually
        # present in include/ (not just this run's) so a partial rebuild
        # (--kernels <group> without --clean) into an existing artifact keeps
        # the untouched groups' headers included. `ar rcs` above has the same
        # merge semantics for the archive: named members are replaced, other
        # members are kept.
        umbrella = inc_dir / "cutedsl_all.h"
        umbrella_headers = sorted(
            h.name
            for h in inc_dir.glob("*.h")
            if h.name != "cutedsl_all.h" and not re.fullmatch(r"cutedsl_.*_all\.h", h.name)
        )
        umbrella.write_text(
            "#pragma once\n"
            "// Auto-generated by build_cutedsl.py -- do not edit\n"
            + "".join(f'#include "{h}"\n' for h in umbrella_headers)
        )

        # Write build provenance + metadata for cmake consumption. Partial
        # rebuilds merge groups/variants with compatible existing metadata so
        # the manifest keeps describing the whole artifact.
        metadata = {
            "arch": arch,
            "artifact_tag": artifact_tag,
            "gpu_arch": f"sm_{sm}",
            "compile_gpu_arch": compile_gpu_arch,
            "wrapper_gpu_arch": wrapper_gpu_arch,
            "host_target": host_target,
            "cuda_version": cuda_ver,
            "cuda_package_variant": _cuda_package_variant(cuda_ver),
            "host_cuda_version": host_cuda_ver,
            "cutlass_dsl_version": dsl_ver,
            "cutlass_dsl_cuda_version": dsl_cuda_version,
            "cupy_package": cupy_pkg,
            "cupy_version": cupy_ver,
            "runtime_libs_version": runtime_libs_version,
            "build_date": datetime.now(timezone.utc).isoformat(),
            "groups": groups_selected,
            "variants": [o.stem for o in kernel_obj_files],
            "runtime_static_archive": _RUNTIME_STATIC_ARCHIVE,
        }
        metadata_path = output_dir / "metadata.json"
        metadata = _merge_artifact_metadata(_load_existing_metadata(metadata_path), metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        shutil.rmtree(root_staging, ignore_errors=True)

    print(f"\nDone. Artifacts written to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_jobs() -> int:
    """Number of CPUs available to this process (cgroup/affinity-aware)."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpu_arch",
        default=None,
        help="Override target GPU SM (e.g. sm_87, sm_100). "
             "Default: auto-detect via cupy / nvidia-smi. "
             "Used only for variant filtering — never forwarded to kernel scripts.",
    )
    p.add_argument(
        "--kernels",
        default="ALL",
        help="Which kernels to build: ALL (default), a group name "
             "(fmha | gdn | f16_moe | nvfp4_moe | "
             "nvfp4_a16_blackwell_gemm | nvfp4_fused_moe | rmsnorm | ssd | gemm | "
             "int4_fp16_gemm), or a comma-separated list "
             "of group names. "
             "Variants whose supported_sms does not include the target SM are skipped.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Root output dir (artifacts go into {{output_dir}}/{{arch}}/sm_<NN>/). "
             f"Default: {_DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--arch",
        default=None,
        help="Target CPU architecture for generated artifacts: x86_64 or aarch64 "
             "(default: auto-detect current host). If target differs from the build host, "
             "CuTe DSL emits cross-target host objects.",
    )
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=_default_jobs(),
        help="Parallel compile jobs (use -j 1 if GPU memory is limited). "
             "Default: the number of CPUs available to this process.",
    )
    p.add_argument(
        "--cuda-version",
        default=None,
        help="Override the CUDA version used to select the CuTe DSL runtime libs "
             "(e.g. 12.6 or 13). Only the major version matters for locating the "
             "cuXX/lib directory. Default: auto-detect via nvcc / cupy. Useful when "
             "cross-compiling and the host nvcc reports a different CUDA major than "
             "the installed target runtime libs.",
    )
    p.add_argument(
        "--runtime-libs-version",
        default=None,
        help="Version of the target-architecture nvidia-cutlass-dsl-libs-cuXX wheel "
             "to download for cross builds. Default: the installed CuTe DSL version. "
             "Set this when the compiler package version differs from the "
             "target runtime-libs wheel version.",
    )
    p.add_argument("--verbose", action="store_true", help="Show per-variant kernel script output.")
    p.add_argument("--clean", action="store_true", help="Remove the target artifact dir before building.")
    build(p.parse_args())


if __name__ == "__main__":
    main()
