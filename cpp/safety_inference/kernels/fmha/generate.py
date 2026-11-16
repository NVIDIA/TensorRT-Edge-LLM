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
"""Export Standard-derived FMHA variants without deploying DSL host loaders."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import struct
import subprocess

SOURCE_COMMIT = "52dfc7e00295d8c3d6875aa4acd1962c6003dbb7"
SOURCE_PATH = "kernelSrcs/fmha_v2_cutedsl/fmha.py"
DSL_VERSION = "4.7.0"
POINTER = "!llvm.struct<(ptr<1>, struct<(struct<()>, struct<()>)>)>"
TENSOR32 = "!llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, struct<(i64, i64)>)>)>"
TENSOR24 = "!llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)>"
LENGTHS16 = "!llvm.struct<(ptr<1>, struct<(i32, struct<()>)>)>"


def verify_launch_abi(ir_text: str, visual: bool, tile: int, head_size: int):
    """Reject compiler drift instead of guessing the raw driver argument ABI."""
    entry = "__call_vit__" if visual else "__call__"
    begin = ir_text.index(f"  llvm.func @cutlass_{entry}")
    end = ir_text.index("\n  llvm.func ", begin + 1)
    host = ir_text[begin:end]
    launch = host.index("llvm.call @_cudaLaunchKernelEx")
    # Each argument is stored in a fresh alloca before its address is appended.
    stores = re.findall(
        r"llvm.store (%(?:arg)?\d+), %\d+ : (.+), !llvm.ptr\n"
        r"\s*%\d+ = llvm.getelementptr %\d+\[%\d+\] : "
        r"\(!llvm.ptr, i32\) -> !llvm.ptr, !llvm.ptr\n"
        r"\s*llvm.store %\d+, %\d+ : !llvm.ptr, !llvm.ptr", host[:launch])
    if visual:
        expected = [POINTER] + [TENSOR24] * 4 + [LENGTHS16] * 2 + [
            "f32", "i32", "i32"
        ]
        sources = [
            "%arg0", "%arg1", "%arg2", "%arg3", "%arg4", "%arg5", "%arg5",
            "%arg7", None, "%arg6"
        ]
        sizes = [8, 24, 24, 24, 24, 16, 16, 4, 4, 4]
    else:
        expected = [
            POINTER, TENSOR32, TENSOR32, TENSOR32, TENSOR32, POINTER, POINTER,
            "f32", "i32"
        ]
        sources = [
            "%arg0", "%arg1", "%arg2", "%arg3", "%arg4", "%arg5", "%arg6",
            None, "%arg8"
        ]
        sizes = [8, 32, 32, 32, 32, 8, 8, 4, 4]
    if [kind for _, kind in stores] != expected:
        raise ValueError(f"Unexpected raw FMHA parameter types: {stores}")
    definitions = dict(
        re.findall(r"^\s*(%(?:arg)?\d+) = (.+)$", host, re.MULTILINE))
    for (actual, _), wanted in zip(stores, sources):
        if wanted is not None and actual != wanted:
            raise ValueError(
                f"Unexpected FMHA source {actual}; expected {wanted}")
    if visual:
        head_operand = stores[-2][0]
        if not definitions[head_operand].startswith(
                "llvm.extractvalue %arg2[1, 0, 1]"):
            raise ValueError(
                "KV head count no longer uses the expected dimension")
    else:
        scale = definitions[stores[-2][0]]
        match = re.fullmatch(r"llvm.fmul %arg7, (%(?:arg)?\d+) : f32", scale)
        if not match or definitions[
                match[1]] != "llvm.mlir.constant(1.44269502 : f32) : f32":
            raise ValueError("Contiguous attention scale folding changed")
    shared_memory = 3 * tile * head_size * 2
    verify_launch_configuration(host, definitions, visual, tile, shared_memory)
    signature = {
        "types": expected,
        "sizes": sizes,
        "sources": sources,
        "shared_memory": shared_memory,
        "tile_m": tile,
        "threads": 128
    }
    signature["sha256"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return signature, host


def verify_launch_configuration(host,
                                definitions,
                                visual,
                                tile,
                                shared_memory,
                                blackwell=False):
    """Match launch dimensions and stream to the manually implemented adapter."""
    configuration = re.search(
        r"(%(?:arg)?\d+) = llvm.alloca %\d+ x !llvm.struct<\(array<3 x i32>, "
        r"array<3 x i32>, i64, ptr, ptr, i32\)>", host)[1]

    def expression(value):
        if value.startswith("%arg"):
            return value
        definition = definitions[value].strip()
        match = re.fullmatch(
            r"llvm.mlir.constant\(([^ ]+)(?: : i\d+)?\) : i\d+", definition)
        if match:
            scalar = match[1]
            return {
                "true": True,
                "false": False
            }.get(scalar,
                  int(scalar) if scalar.lstrip("-").isdigit() else scalar)
        match = re.match(r"llvm.extractvalue (%arg\d+)\[([^]]+)\]", definition)
        if match:
            return ("extract", match[1], match[2])
        match = re.match(
            r"llvm.(add|sub|mul|sdiv|and) (%(?:arg)?\d+), (%(?:arg)?\d+)",
            definition)
        if match:
            return (match[1], expression(match[2]), expression(match[3]))
        match = re.match(r'llvm.icmp "([^"]+)" (%(?:arg)?\d+), (%(?:arg)?\d+)',
                         definition)
        if match:
            return (match[1], expression(match[2]), expression(match[3]))
        match = re.match(
            r"llvm.select (%(?:arg)?\d+), (%(?:arg)?\d+), (%(?:arg)?\d+)",
            definition)
        if match:
            return ("select", *(expression(match[i]) for i in (1, 2, 3)))
        raise ValueError(
            f"Unadmitted launch configuration expression: {definition}")

    def field(indices):
        match = re.search(
            rf"(%(?:arg)?\d+) = llvm.getelementptr {configuration}\[0, {indices}\][^\n]+\n"
            r"\s*llvm.store (%(?:arg)?\d+), \1 :", host)
        if not match:
            raise ValueError(f"Missing launch configuration field {indices}")
        return expression(match[2])

    query = "%arg7" if blackwell else "%arg6" if visual else ("extract",
                                                              "%arg1",
                                                              "1, 0, 0")
    quotient = ("sdiv", query, tile)
    ceiling = ("select", ("and", ("ne", query, ("mul", quotient, tile)),
                          ("eq", ("slt", query, 0), False)), ("add", quotient,
                                                              1), quotient)
    expected = {
        "0, 0":
        ceiling,
        "0, 1": ("sub", ("extract", "%arg5", "1, 0"), 1) if visual else
        ("extract", "%arg1", "1, 0, 1"),
        "0, 2": ("extract", "%arg1", "1, 0, 1") if visual else 1,
        "1, 0":
        128,
        "1, 1":
        1,
        "1, 2":
        1,
        "2":
        shared_memory,
        "3":
        "%arg11" if visual else "%arg9"
    }
    if blackwell:
        heads = ("mul", ("sdiv", ("extract", "%arg1", "1, 0, 1"),
                         ("extract", "%arg2", "1, 0, 1")), ("extract", "%arg2",
                                                            "1, 0, 1"))
        batches = ("sub", ("extract", "%arg5", "1, 0"), 1)
        tiles = ("mul", ("mul", ceiling, heads), batches)
        expected = {
            "0, 0": ("select", ("slt", "%arg11", tiles), "%arg11", tiles),
            "0, 1": 1,
            "0, 2": 1,
            "1, 0": 512,
            "1, 1": 1,
            "1, 2": 1,
            "2": shared_memory,
            "3": "%arg12"
        }
    for indices, wanted in expected.items():
        actual = field(indices)
        if actual != wanted:
            raise ValueError(
                f"Launch field {indices} changed: {actual}; expected {wanted}")


def verify_blackwell_abi(ir_text: str, visual: bool, head_size: int):
    """Check the raw Blackwell parameter layout against the prepared C++ launcher."""
    begin = ir_text.index("  llvm.func @cutlass___call__")
    end = ir_text.index("\n  llvm.func ", begin + 1)
    host = ir_text[begin:end]
    launch = host.index("llvm.call @_cudaLaunchKernelEx")
    stores = re.findall(
        r"llvm.store (%(?:arg)?\d+), %\d+ : (.+), !llvm.ptr\n"
        r"\s*%\d+ = llvm.getelementptr %\d+\[%\d+\] : "
        r"\(!llvm.ptr, i32\) -> !llvm.ptr, !llvm.ptr\n"
        r"\s*llvm.store %\d+, %\d+ : !llvm.ptr, !llvm.ptr", host[:launch])
    mma = "!llvm.struct<(i1, i1, i1, vector<4xi32>)>"
    tma = "!llvm.struct<(struct<(array<16 x i64>)>)>"
    coordinates = "!llvm.struct<(struct<()>, struct<(struct<(i32, i32, i32)>, struct<()>)>)>"
    output = "!llvm.struct<(ptr<1>, struct<(struct<(i32, i32, i32)>, struct<(i32, i32)>)>)>"
    expected = [mma] * 2 + [tma, coordinates] * 4 + [
        output, LENGTHS16, LENGTHS16
    ] + ["f32"] * 3
    expected += ["i32"] * (3 if visual else 4) + [POINTER]
    if [kind for _, kind in stores] != expected:
        raise ValueError(f"Unexpected Blackwell raw parameters: {stores}")
    for index, source in ((11, "%arg5"), (12, "%arg6"), (13, "%arg8"),
                          (14, "%arg9"), (15, "%arg10"), (-1, "%arg0")):
        if stores[index][0] != source:
            raise ValueError(f"Blackwell argument {index} source changed")
    shared_memory = 115712 if head_size == 64 else 230400
    definitions = dict(
        re.findall(r"^\s*(%(?:arg)?\d+) = (.+)$", host, re.MULTILINE))
    verify_launch_configuration(host,
                                definitions,
                                visual,
                                256,
                                shared_memory,
                                blackwell=True)
    signature = {
        "types":
        expected,
        "sizes": [32, 32] + [128, 12] * 4 + [32, 16, 16] + [4] *
        (6 if visual else 7) + [8],
        "shared_memory":
        shared_memory,
        "tile_m":
        256,
        "threads":
        512
    }
    signature["sha256"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return signature, host


def extract_binary(compiled, ir_text: str):
    """Extract only the fatbinary, never the compiler host loader object."""
    from cutlass.base_dsl import jit_executor

    if len(compiled.kernel_info) != 1:
        raise ValueError("A Safety artifact must contain exactly one kernel")
    symbol = next(iter(compiled.kernel_info))
    match = re.search(r'@kernels_binary\("([^"\n]+)"\)', ir_text)
    if not match:
        raise ValueError("Pinned compiler fatbinary representation changed")
    binary = jit_executor.get_escaped_cubin_bytes(match[1].encode())
    validate_binary(symbol, binary)
    return symbol, binary


def validate_binary(symbol, binary):
    """Admit one native cubin with the expected entry point and no PTX fallback."""
    if binary[:4] != bytes.fromhex(
            "50ed55ba") or symbol.encode() not in binary:
        raise ValueError("Malformed fatbinary or missing declared entry point")
    # The generated fatbinary must contain one uncompressed ELF cubin, never PTX.
    magic, version, header_size, payload_size = struct.unpack_from(
        "<IHHQ", binary)
    kind, entry_version, entry_size, padded_size = struct.unpack_from(
        "<HHIQ", binary, header_size)
    if (magic != 0xBA55ED50 or version != 1 or header_size != 16 or kind != 2
            or entry_version != 0x101
            or header_size + payload_size != len(binary)
            or entry_size + padded_size != payload_size
            or binary[header_size + entry_size:header_size + entry_size + 4]
            != b"\x7fELF"):
        raise ValueError(
            "Only one native ELF cubin without PTX fallback is admitted")


def assemble_with_sdk(compiled, compiler, arch, destination, symbol):
    """Assemble unchanged device PTX with the target SDK, not CuTe's bundled assembler."""
    ptx = pathlib.Path(compiled.__ptx__)
    if not ptx.is_file():
        # CuTe 4.7 retains the pre-renumbering Thor name in its emitted PTX filename.
        ptx = pathlib.Path(str(ptx).replace("sm_110a", "sm_101a"))
    source = destination.with_suffix(".ptx")
    ptx.replace(source)
    subprocess.run([
        compiler, "--safety-compatibility",
        f"--generate-code=arch={arch.replace('sm_', 'compute_')},code={arch}",
        "--fatbin",
        str(source), "-o",
        str(destination)
    ],
                   check=True)
    binary = destination.read_bytes()
    validate_binary(symbol, binary)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--arch",
                        choices=("sm_80", "sm_110a", "80", "110a"),
                        required=True)
    parser.add_argument(
        "--cuda-compiler",
        default=os.environ.get("EDGELLM_SAFETY_CUDA_COMPILER"),
        help="Target SDK nvcc; required for QNX Safety artifact generation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sdk_compiler = None
    if args.cuda_compiler:
        sdk_compiler = subprocess.check_output(
            [args.cuda_compiler, "--version"], text=True).strip()
        os.environ["CUTE_DSL_KEEP"] = "ptx"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(args.output_dir)
    arch = args.arch if args.arch.startswith("sm_") else f"sm_{args.arch}"
    # Select the target before DSL imports to keep cross-generation independent of host GPUs.
    os.environ["CUTE_DSL_ARCH"] = arch
    import fmha_config
    import safetyFmha

    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != DSL_VERSION:
        raise RuntimeError(
            f"Raw ABI requires nvidia-cutlass-dsl=={DSL_VERSION}, got {version}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blackwell = arch == "sm_110a"
    manifest = {
        "source_commit":
        SOURCE_COMMIT,
        "source_path":
        "kernelSrcs/fmha_cutedsl_blackwell/fmha.py"
        if blackwell else SOURCE_PATH,
        "backend":
        "blackwell" if blackwell else "ampere",
        "artifact_contract_version":
        1,
        "cutlass_version":
        version,
        "architecture":
        arch,
        "sdk_compiler":
        sdk_compiler,
        "toolchain_versions": {
            name: importlib.metadata.version(name)
            for name in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
                         "nvidia-cutlass-dsl-libs-core",
                         "nvidia-cutlass-dsl-libs-cu12", "cuda-bindings")
        },
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in fmha_config.source_files(blackwell).items()
        },
        "numerical_options": {
            "dtype": "float16",
            "skip_correction": True,
            "skip_softmax": False,
            "sliding_window": False,
            "persistent": True,
        } if blackwell else fmha_config.NUMERICAL_OPTIONS,
        "variants": []
    }
    header = [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
        "// SPDX-License-Identifier: Apache-2.0\n#pragma once\n",
        '#include "safetyFmhaLaunch.h"\nnamespace trt_edgellm::safety::fmha\n{\n'
    ]
    records = []
    for visual in (False, True):
        for head_size in fmha_config.HEAD_SIZES:
            name = f"{'visual' if visual else 'contiguous'}D{head_size}"
            if blackwell:
                compiled, tile = fmha_config.compile_blackwell(
                    head_size, visual, arch)
            else:
                compiled, tile = fmha_config.compile_variant(
                    safetyFmha.FMHAV2Ampere, head_size, visual, arch)
            ir_text = str(compiled.ir_module)
            abi, host = (verify_blackwell_abi(ir_text, visual, head_size)
                         if blackwell else verify_launch_abi(
                             ir_text, visual, tile, head_size))
            symbol, binary = extract_binary(compiled, ir_text)
            if args.cuda_compiler:
                binary = assemble_with_sdk(compiled, args.cuda_compiler, arch,
                                           args.output_dir / f"{name}.fatbin",
                                           symbol)
            (args.output_dir / f"{name}.fatbin").write_bytes(binary)
            (args.output_dir / f"{name}.launch.mlir").write_text(host + "\n")
            # The ELF compiler note is optional; toolchain package versions are recorded independently.
            compiler_note = re.search(
                rb"Cuda compilation tools, release [^\x00]+", binary)
            manifest["variants"].append({
                "name":
                name,
                "symbol":
                symbol,
                "binary_sha256":
                hashlib.sha256(binary).hexdigest(),
                "head_size":
                head_size,
                "visual":
                visual,
                "abi":
                abi,
                "compiler":
                compiler_note[0].decode() if compiler_note else None
            })
            header.append(
                f"alignas(8) inline constexpr unsigned char k{name}[] = {{\n")
            header.extend("    " + ",".join(f"0x{byte:02x}"
                                            for byte in binary[i:i + 24]) +
                          ",\n" for i in range(0, len(binary), 24))
            header.append("};\n")
            records.append(
                f'    {{k{name}, sizeof(k{name}), "{symbol}", {abi["shared_memory"]}, '
                f'{head_size}, {str(visual).lower()}, {tile}, {abi["threads"]}}}'
            )
            print(f"Exported {name}: {len(binary)} bytes, ABI {abi['sha256']}",
                  flush=True)
    header.append("inline constexpr Artifact kARTIFACTS[] = {\n" +
                  ",\n".join(records) + "\n};\n")
    header.append("} // namespace trt_edgellm::safety::fmha\n")
    header_bytes = "".join(header).encode()
    (args.output_dir / "safetyFmhaArtifacts.h").write_bytes(header_bytes)
    manifest["header_sha256"] = hashlib.sha256(header_bytes).hexdigest()
    (args.output_dir /
     "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
