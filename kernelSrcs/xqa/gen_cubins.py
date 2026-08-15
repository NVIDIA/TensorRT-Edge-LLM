#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# NOTE: this file is for cubin generation, should not in final code release.

import argparse
import itertools
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from typing import List, Tuple, Union

CompileMacro = namedtuple('CompileMacro', 'macro_name short_name value')

CompileMacroOption = namedtuple('CompileMacroOption',
                                'macro_name short_name options')

CompileArchMacrosAndFile = namedtuple(
    'CompileArchMacrosAndFile', 'arch compile_arch macro_list input_file_name')

build_func_name_prefix = 'xqa_kernel'
TOKENS_PER_PAGE_OPTIONS = [0, 128]
arch_options = [80, 86, 90]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def check_tokens_per_page(tokens_per_page: int):
    if tokens_per_page == 0:
        return
    if not is_power_of_two(tokens_per_page):
        raise ValueError(
            'TOKENS_PER_PAGE must be 0 or a power-of-two page size.')


config_list = [
    # for llama v2 70b
    [
        CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
        CompileMacroOption('HEAD_ELEMS', 'd', [128, 256]),
        CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
        CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
        CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                           TOKENS_PER_PAGE_OPTIONS),  # 0 denotes contiguous kv cache.
        CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [8]),
        CompileMacroOption('M_TILESIZE', 'm', [8]),
    ],
    # for gptj beamWidth=4
    [
        CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
        CompileMacroOption('HEAD_ELEMS', 'd', [256]),
        CompileMacroOption('BEAM_WIDTH', 'beam', [4]),
        CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
        CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                           TOKENS_PER_PAGE_OPTIONS),  # 0 denotes contiguous kv cache.
        CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [1]),
        CompileMacroOption('M_TILESIZE', 'm', [4]),
    ]
]

clean_cubin = True

nvcc_bin = 'nvcc'
nvcc_flags = '-std=c++17 -O3 -cubin -DGENERATE_CUBIN=1 -DNDEBUG --use_fast_math -Xptxas=-v --allow-unsupported-compiler --expt-relaxed-constexpr -t 0'
# nvcc_flags = '-std=c++17 -G -cubin -DGENERATE_CUBIN=1 -Xptxas=-v --allow-unsupported-compiler --expt-relaxed-constexpr -t 0'
cuda_toolkit_version = None
cubin_dir = "../../cpp/kernels/decodeAttentionKernels/cubin/"
xqa_source_dir = os.path.dirname(os.path.abspath(__file__))
cpp_include_dir = os.path.abspath(os.path.join(xqa_source_dir, '..', '..',
                                               'cpp'))


def use_tiled_qkv_staging_head_dim512(arch: int, head_dim: int) -> bool:
    return head_dim == 512 and arch in (86, 89)


def get_default_m_tile_size(arch: int, head_dim: int, spec_decode: bool) -> int:
    if spec_decode and arch in (80, 87) and head_dim == 512:
        return 16
    return 32 if spec_decode else 8


def override_compile_macro(
        compile_macros: List[CompileMacro], macro_name: str,
        value) -> List[CompileMacro]:
    return [
        CompileMacro(macro.macro_name, macro.short_name,
                     value if macro.macro_name == macro_name else macro.value)
        for macro in compile_macros
    ]


cpp_file_prefix_text = R"""/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
namespace xqa
{
namespace kernels
{
// clang-format off
"""

cpp_file_suffex_text = R"""
// clang-format on
} // namespace kernels
} // namespace xqa
"""

cubin_meta_info_struct_prefix_text = R"""
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_87 = 87;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
constexpr int32_t kSM_101 = 101;
constexpr int32_t kSM_120 = 120;
constexpr int32_t kSM_121 = 121;

static const struct XQAKernelMetaInfo
{
    enum XQAKernelVariant
    {
        //! Default single-CTA path for non-head_dim512 kernels.
        KERNEL_VARIANT_STANDARD,
        //! Single-CTA head_dim512 path with full shared-memory tile and protected row-max update.
        KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512,
        //! Single-CTA head_dim512 path with full shared-memory tile and volatile row-max update.
        KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4,
        //! Single-CTA head_dim512 path with tiled Q/K/V shared-memory staging.
        KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512,
        //! 2CTA cluster head_dim512 path; requires cluster launch and distributed shared memory.
        KERNEL_VARIANT_2CTA_HEAD_DIM512,
    };

    Data_type mDataType;
    Data_type mKVDataType;
    unsigned int mHeadDim;
    unsigned int mBeamWidth;
    unsigned int mNumQHeadsOverKV;
    unsigned int mMTileSize;
    unsigned int mTokensPerPage;
    bool mSlidingWindow;
    bool mPagedKVCache;
    bool mMultiQueryTokens;
    unsigned int mSM;
    const unsigned long long* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    XQAKernelVariant mKernelVariant{KERNEL_VARIANT_STANDARD};
    bool mRequiresClusterLaunch{false};
    bool mRequiresDistributedSharedMemory{false};
} sXqaKernelMetaInfo[] = {
"""

cubin_meta_info_struct_suffix_text = R"""
};
"""

is_spec_dec = False


def generate_cubin_meta_info_line(arch: int,
                                  compile_macros: List[CompileMacro],
                                  function_name: str, cubin_size: int,
                                  is_last: bool, is_spec_dec: bool):
    data_type_str = None
    kv_data_type_str = None
    head_dim = None
    beam_width = None
    num_q_heads_per_kv = None
    m_tilesize = None
    paged_kv_cache = None
    tokens_per_page = None
    sliding_window = None
    use_tiled_qkv_staging_head_dim512 = False
    use_2cta_head_dim512 = False
    for compile_macro in compile_macros:
        if compile_macro.macro_name == 'DTYPE':
            data_type_upper_case = map_disp_value(compile_macro.value).upper()
            data_type_str = 'DATA_TYPE_' + data_type_upper_case
        if compile_macro.macro_name == 'CACHE_ELEM_ENUM':
            if compile_macro.value == 0:
                assert data_type_str is not None
                kv_data_type = '__half' if data_type_str == 'DATA_TYPE_FP16' else '__nv_bfloat16'
            elif compile_macro.value == 1:
                kv_data_type = 'int8_t'
            else:
                assert compile_macro.value == 2
                kv_data_type = '__nv_fp8_e4m3'
            kv_type_upper_case = map_disp_value(kv_data_type).upper()
            kv_data_type_str = 'DATA_TYPE_' + kv_type_upper_case
        if compile_macro.macro_name == 'BEAM_WIDTH':
            beam_width = compile_macro.value
        if compile_macro.macro_name == 'HEAD_ELEMS':
            head_dim = compile_macro.value
        if compile_macro.macro_name == 'HEAD_GRP_SIZE':
            num_q_heads_per_kv = compile_macro.value
        if compile_macro.macro_name == 'M_TILESIZE':
            m_tilesize = compile_macro.value
        if compile_macro.macro_name == 'TOKENS_PER_PAGE':
            if compile_macro.value not in TOKENS_PER_PAGE_OPTIONS:
                raise ValueError(
                    f'TOKENS_PER_PAGE must be one of {TOKENS_PER_PAGE_OPTIONS}.')
            check_tokens_per_page(compile_macro.value)
            tokens_per_page = compile_macro.value
            paged_kv_cache = 'true' if compile_macro.value > 0 else 'false'
        if compile_macro.macro_name == 'SLIDING_WINDOW':
            assert compile_macro.value in (0, 1)
            sliding_window = 'true' if compile_macro.value == 1 else 'false'
        if compile_macro.macro_name == 'TILED_QKV_STAGING_HEAD_DIM512':
            assert compile_macro.value == 1
            use_tiled_qkv_staging_head_dim512 = True
        if compile_macro.macro_name == 'XQA_2CTA_HEAD_DIM512':
            assert compile_macro.value == 1
            use_2cta_head_dim512 = True

    use_medusa = 'true' if is_spec_dec else 'false'
    assert data_type_str is not None
    assert kv_data_type_str is not None
    assert head_dim is not None
    assert beam_width is not None
    assert num_q_heads_per_kv is not None
    assert sliding_window is not None
    assert paged_kv_cache is not None
    assert tokens_per_page is not None
    unique_func_name = "kernel_mha"
    kernel_variant = 'XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD'
    requires_cluster_launch = 'false'
    requires_distributed_shared_memory = 'false'
    if head_dim == 512:
        kernel_variant = 'XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512'
        if arch in (80, 87):
            kernel_variant = 'XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4'
        if use_tiled_qkv_staging_head_dim512:
            kernel_variant = 'XQAKernelMetaInfo::KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512'
        if use_2cta_head_dim512:
            kernel_variant = 'XQAKernelMetaInfo::KERNEL_VARIANT_2CTA_HEAD_DIM512'
            requires_cluster_launch = 'true'
            requires_distributed_shared_memory = 'true'
    fields = [
        data_type_str, kv_data_type_str,
        str(head_dim),
        str(beam_width),
        str(num_q_heads_per_kv),
        str(m_tilesize),
        str(tokens_per_page), sliding_window, paged_kv_cache, use_medusa, f'kSM_{arch}',
        f'{function_name}_cubin', f'{function_name}_cubin_len',
        f'"{unique_func_name}"'
    ]
    if kernel_variant != 'XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD':
        fields.extend([
            kernel_variant, requires_cluster_launch,
            requires_distributed_shared_memory
        ])
    field_str = ', '.join(fields)
    line_segs = ["{ ", field_str, "}"]
    if not is_last:
        line_segs.append(',')
    return ''.join(line_segs)


def construct_name(
    func_name_prefix: str,
    arch: int,
    other_name_info: List[str],
    suffix: str = "",
) -> str:
    str_segments = [func_name_prefix, *other_name_info, f"sm_{arch}"]
    name_wo_suffix = '_'.join(str_segments)
    full_name = name_wo_suffix + suffix
    return full_name


name_mapping_dict = {
    '__half': 'fp16',
    '__nv_bfloat16': 'bf16',
    '__nv_fp8_e4m3': 'e4m3',
    'int8_t': 'int8',
    'float': 'fp32',
}


def map_disp_value(value):
    if isinstance(value, str):
        if value in name_mapping_dict.keys():
            return name_mapping_dict[value]
    return value


def build_name_info(compile_macros: List[CompileMacro]):
    compile_macros = [
        compile_macro for compile_macro in compile_macros
        if compile_macro.short_name not in ('tiled_qkv_staging', '2cta_head_dim512')
    ]
    short_names = [
        compile_macro.short_name for compile_macro in compile_macros
    ]
    values = []
    for compile_macro in compile_macros:
        if compile_macro.short_name == 'kvt':
            if compile_macro.value == 0:
                assert compile_macros[0].short_name == 'dt'
                value = compile_macros[0].value
            elif compile_macro.value == 1:
                value = "int8_t"
            elif compile_macro.value == 2:
                value = "__nv_fp8_e4m3"
        else:
            value = compile_macro.value
        values.append(value)
    disp_values = [map_disp_value(value) for value in values]
    name_info = [
        f"{short_name}_{disp_value}"
        for short_name, disp_value in list(zip(short_names, disp_values))
    ]
    if "pagedKV_0" in name_info:
        name_info.remove("pagedKV_0")
    if "sw_0" in name_info:
        name_info.remove("sw_0")
    return name_info


def build_commands(
    func_name_prefix: str,
    arch: int,
    compile_arch: Union[int, str],
    input_filename: str,
    compile_macros: List[CompileMacro],
) -> Tuple[str, str, str]:
    arch_str = str(compile_arch) + 'a' if compile_arch in (90, ) else str(compile_arch)
    arch_option = f"-arch=compute_{arch_str} -code=sm_{arch_str}"
    name_info = build_name_info(compile_macros)
    macro_options = [
        f"-D{compile_macro.macro_name}={compile_macro.value}"
        for compile_macro in compile_macros
    ]

    macro_options = []
    for compile_macro in compile_macros:
        if compile_macro.macro_name == "DTYPE":
            if compile_macro.value == "__half":
                macro_options.append(f"-DINPUT_FP16=1")
            elif compile_macro.value == "__nv_bfloat16":
                macro_options.append(f"-DINPUT_FP16=0")
        else:
            macro_options.append(
                f"-D{compile_macro.macro_name}={compile_macro.value}")

    function_name = construct_name(func_name_prefix, arch, name_info)
    macro_options.append(f"-DKERNEL_FUNC_NAME={function_name}")
    all_macro_option = ' '.join(macro_options)
    cubin_file_name = os.path.join(
        cubin_dir, construct_name(func_name_prefix, arch, name_info, ".cubin"))
    output_option = " ".join(["-o", cubin_file_name])
    include_option = f"-I{cpp_include_dir}"
    input_file_path = os.path.join(xqa_source_dir, input_filename)
    nvcc_command = " ".join([
        nvcc_bin, nvcc_flags, arch_option, output_option, include_option,
        all_macro_option, input_file_path
    ])
    xxd_command = " ".join(["xxd -i", cubin_file_name])
    return nvcc_command, xxd_command, cubin_file_name


def get_cubin_cpp_file_name(func_name_prefix: str, arch: int,
                            compile_macros: List[CompileMacro]) -> str:
    name_info = build_name_info(compile_macros)
    return construct_name(func_name_prefix, arch, name_info, ".cubin.cpp")


def save_cubin_cpp_file(xxd_output, func_name_prefix, arch, compile_macros):
    cubin_cpp_file_name = os.path.join(
        cubin_dir,
        get_cubin_cpp_file_name(func_name_prefix, arch, compile_macros))
    with open(cubin_cpp_file_name, "w") as f:
        f.write(''.join(
            [cpp_file_prefix_text, xxd_output, cpp_file_suffex_text]))


def convert_cubin_cpp_xxd(xxd_command: str, cubin_file_name: str):
    result = subprocess.run(xxd_command.split(' '),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True,
                            shell=False)
    cubin_cpp_str = result.stdout
    cubin_size = os.path.getsize(cubin_file_name)
    return cubin_cpp_str, cubin_size


def convert_cubin_cpp_np(cubin_file_name: str):
    cubin_size = os.path.getsize(cubin_file_name)
    with open(cubin_file_name, 'rb') as f:
        cubin_bin_data = f.read()
    remainder = len(cubin_bin_data) % 8
    if remainder != 0:
        padding = b'\x00' * (8 - remainder)
        cubin_bin_data += padding
    array = [
        int.from_bytes(cubin_bin_data[i:i + 8], byteorder='little')
        for i in range(0, len(cubin_bin_data), 8)
    ]
    array_name = os.path.basename(cubin_file_name).replace('.', '_')
    elements_per_line = 4
    cpp_array_content = ',\n'.join(
        ', '.join(
            '0x{:016x}ULL'.format(array[i])
            for i in range(start, min(start + elements_per_line, len(array))))
        for start in range(0, len(array), elements_per_line))
    cpp_array = 'unsigned long long ' + array_name + '[] = {\n' + cpp_array_content + '\n};\n' + 'unsigned int ' \
                + array_name + '_len = ' + str(cubin_size) + ';\n'
    return cpp_array, cubin_size


def init_cubin_gen_worker(worker_cubin_dir: str, worker_nvcc_bin: str,
                          worker_clean_cubin: bool):
    # Workers under a start method that re-imports this module (spawn/forkserver)
    # never execute the `if __name__ == "__main__":` block, so the module-scope
    # defaults for these globals would otherwise be used instead of the values
    # resolved from CLI args. Set them explicitly in each worker process.
    global cubin_dir, nvcc_bin, clean_cubin
    cubin_dir = worker_cubin_dir
    nvcc_bin = worker_nvcc_bin
    clean_cubin = worker_clean_cubin


def run_cubin_gen(arch_micro_file_list: CompileArchMacrosAndFile):
    nvcc_command, xxd_command, cubin_file_name = build_commands(
        build_func_name_prefix, arch_micro_file_list.arch,
        arch_micro_file_list.compile_arch,
        arch_micro_file_list.input_file_name, arch_micro_file_list.macro_list)
    function_name = construct_name(
        build_func_name_prefix, arch_micro_file_list.arch,
        build_name_info(arch_micro_file_list.macro_list))
    print(f'generating for {function_name}... command: {nvcc_command}')
    cubin_size = None
    try:
        result = subprocess.run(nvcc_command.split(' '),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True,
                                shell=False)
        # cubin_cpp_str, cubin_size = convert_cubin_cpp_xxd(xxd_command, cubin_file_name)
        cubin_cpp_str, cubin_size = convert_cubin_cpp_np(cubin_file_name)
        save_cubin_cpp_file(cubin_cpp_str, build_func_name_prefix,
                            arch_micro_file_list.arch,
                            arch_micro_file_list.macro_list)
        if clean_cubin:
            os.remove(cubin_file_name)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise
    print(f'generating for {function_name} done')
    return function_name, cubin_size


def get_cuda_toolkit_version() -> Tuple[int, int]:
    global cuda_toolkit_version
    if cuda_toolkit_version is not None:
        return cuda_toolkit_version
    ctk_version = os.environ.get('CUDA_CTK_VERSION')
    if ctk_version:
        version_parts = ctk_version.split('.')
        minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
        cuda_toolkit_version = (int(version_parts[0]), minor_version)
    else:
        nvcc_output = subprocess.check_output([nvcc_bin, '--version'], text=True)
        version_match = re.search(r'release (\d+)\.(\d+)', nvcc_output)
        cuda_toolkit_version = (
            int(version_match.group(1)),
            int(version_match.group(2))) if version_match else (13, 0)
    return cuda_toolkit_version


def get_cuda_toolkit_major_version() -> int:
    return get_cuda_toolkit_version()[0]


def supports_fp8() -> bool:
    major, minor = get_cuda_toolkit_version()
    return major > 11 or (major == 11 and minor >= 8)


def get_compile_arch(arch: int) -> Union[int, str]:
    # Runtime lookup and cubin names stay normalized to SM101. The actual Thor
    # code-generation target depends on CUDA toolkit numbering.
    if arch == 101:
        return '110a' if get_cuda_toolkit_major_version() >= 13 else '101a'
    return arch


def generate_compile_arch_macro_list(compile_macro_options: list):
    option_values = [
        compile_macro_option.options
        for compile_macro_option in compile_macro_options
    ]
    option_macro_names = [
        compile_macro_option.macro_name
        for compile_macro_option in compile_macro_options
    ]
    option_short_names = [
        compile_macro_option.short_name
        for compile_macro_option in compile_macro_options
    ]
    arch_and_macro_list = []
    for arch in arch_options:
        assert isinstance(arch, int)
        for option_combination in itertools.product(*option_values):
            if "__half" in option_combination and "__nv_bfloat16" in option_combination:
                continue
            assert option_macro_names[3] == "CACHE_ELEM_ENUM"
            if option_combination[3] == 2 and not supports_fp8():
                continue
            # fp8 kv cache is only supported on sm89 and next.
            if option_combination[3] == 2 and arch < 89:
                continue
            compile_macros = [
                CompileMacro(*x) for x in zip(
                    option_macro_names, option_short_names, option_combination)
            ]
            macro_values = {macro.macro_name: macro.value for macro in compile_macros}
            if macro_values.get('HEAD_ELEMS') == 512 and arch not in (80, 86, 87, 89, 100, 101, 120, 121):
                continue
            if macro_values.get('HEAD_ELEMS') == 512 and arch in (80, 87) and macro_values.get('SPEC_DEC') == 1:
                compile_macros = override_compile_macro(
                    compile_macros, 'M_TILESIZE',
                    get_default_m_tile_size(arch, macro_values['HEAD_ELEMS'],
                                            True))
            if use_tiled_qkv_staging_head_dim512(arch, macro_values.get('HEAD_ELEMS')):
                compile_macros.append(
                    CompileMacro('TILED_QKV_STAGING_HEAD_DIM512', 'tiled_qkv_staging', 1))
                input_file_name = "mha.cu"
            elif macro_values.get('HEAD_ELEMS') == 512 and arch in (120, 121):
                compile_macros.append(
                    CompileMacro('XQA_2CTA_HEAD_DIM512', '2cta_head_dim512', 1))
                input_file_name = "mha.cu"
            elif arch in (90, ) and option_combination[
                    3] == 2 and option_combination[2] == 1 and not is_spec_dec:
                input_file_name = "mha_sm90.cu"
            else:
                input_file_name = "mha.cu"
            compile_arch = get_compile_arch(arch)
            arch_and_macro_list.append(
                CompileArchMacrosAndFile(arch, compile_arch, compile_macros,
                                         input_file_name))
    return arch_and_macro_list


def generate_header_file_contents(
        all_arch_macros: List[CompileArchMacrosAndFile],
        name_size_list: List[Tuple[str, int]]):
    cubin_data_array = []
    cubin_length_array = []
    meta_line_array = []
    for i, (arch_macro,
            name_size) in enumerate(list(zip(all_arch_macros,
                                             name_size_list))):
        arch = arch_macro.arch
        macros = arch_macro.macro_list
        is_spec_dec = False
        for macro in macros:
            if macro.macro_name == 'SPEC_DEC':
                is_spec_dec = bool(macro.value)
                break

        #function_name = construct_name(build_func_name_prefix, arch, build_name_info(macros))
        function_name, cubin_size = name_size
        cubin_variable_name = f"{function_name}_cubin"
        cubin_data_array.append(
            f"extern unsigned long long {cubin_variable_name}[];\n")
        cubin_length_array.append(
            f"extern uint32_t {cubin_variable_name}_len;\n")
        meta_line_array.append(
            generate_cubin_meta_info_line(arch, macros, function_name,
                                          cubin_size,
                                          i == len(all_arch_macros) - 1,
                                          is_spec_dec))
    cubin_data = ''.join(cubin_data_array)
    cubin_length = ''.join(cubin_length_array)
    meta_struct = ''.join([
        cubin_meta_info_struct_prefix_text, '\n'.join(meta_line_array),
        cubin_meta_info_struct_suffix_text
    ])
    return '\n'.join([cubin_data, cubin_length, meta_struct])


def normalize_arch(arch: str) -> int:
    arch_num_match = re.match(r'(\d+)', arch)
    if not arch_num_match:
        raise ValueError(f'Unsupported CUDA architecture value: {arch}')
    arch_num = int(arch_num_match.group(1))
    # Thor builds use SM101 XQA runtime lookup. CUDA 13+ compiles that runtime
    # entry with the actual SM110a code-generation target in get_compile_arch().
    if arch_num == 110:
        return 101
    return arch_num


def parse_arches(arch_args: List[str]) -> List[int]:
    if not arch_args:
        return []
    parsed_arches = []
    for arch_arg in arch_args:
        for arch in re.split(r'[,;]', arch_arg):
            arch = arch.strip()
            if not arch:
                continue
            if arch in ('all', 'all-major', 'native'):
                parsed_arches.extend([80, 86, 87, 89, 100, 101, 120, 121])
            else:
                parsed_arches.append(normalize_arch(arch))
    return sorted(set(parsed_arches))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Internal CMake helper for generating XQA cubin C++ blobs.'
    )
    parser.add_argument(
        '--cmake-invoked',
        action='store_true',
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory where generated .cubin.cpp files and header are written.'
    )
    parser.add_argument(
        '--arches',
        nargs='+',
        default=None,
        help='CUDA SM architectures to generate, for example: 80 86 100 120.'
    )
    parser.add_argument(
        '--nvcc',
        default=None,
        help='nvcc executable used to compile cubins.')
    parser.add_argument(
        '--list-outputs',
        action='store_true',
        help='Print the generated output file list and exit without compiling.'
    )
    parser.add_argument(
        '--keep-cubin',
        action='store_true',
        help='Keep intermediate .cubin files in the output directory.')
    args = parser.parse_args()
    if not args.cmake_invoked:
        parser.error(
            'gen_cubins.py is a CMake-only helper. Configure/build the project to generate XQA cubins.'
        )
    if not args.output_dir:
        parser.error('--output-dir is required when invoked by CMake.')
    if not args.arches:
        parser.error('--arches is required when invoked by CMake.')
    return args


def get_generated_output_paths(
        output_dir: str,
        arch_macro_lists: List[CompileArchMacrosAndFile]) -> List[str]:
    output_paths = [
        os.path.join(output_dir, build_func_name_prefix + '_cubin.h')
    ]
    output_paths.extend(
        os.path.join(
            output_dir,
            get_cubin_cpp_file_name(build_func_name_prefix, arch_macro.arch,
                                    arch_macro.macro_list))
        for arch_macro in arch_macro_lists)
    return output_paths


if __name__ == "__main__":

    args = parse_args()
    arch_options = parse_arches(args.arches)
    cubin_dir = os.path.abspath(args.output_dir)
    nvcc_bin = args.nvcc if args.nvcc else nvcc_bin
    clean_cubin = not args.keep_cubin
    edgellm_config_list = [
        [
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [128, 64, 32]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv',
                               [1, 2, 3, 4, 5, 6, 7, 8]),
            CompileMacroOption('M_TILESIZE', 'm', [8]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
        ],
        [
            # nqpkv=16 is only needed for NemotronH (head_dim=128); other head
            # dims are not used by any supported model so we skip them to
            # reduce kernel binary size.
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [128]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [16]),
            CompileMacroOption('M_TILESIZE', 'm', [8]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
        ],
        [
            # nqpkv 4/6/8 cover Qwen3.5-MoE and Qwen3.5-Omni Thinker/Talker;
            # nqpkv=2 (16 Q heads / 8 KV heads) is needed by the Qwen3.5-Omni
            # Talker decode attention path.
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [256]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [2, 4, 6, 8]),
            CompileMacroOption('M_TILESIZE', 'm', [8]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
        ],
        [
            # Gemma 4 global attention uses 512-wide heads.
            # nqpkv=2: E4B assistant (4 Q heads / 2 KV heads)
            # nqpkv=4: E4B base (8 Q heads / 2 KV heads)
            # nqpkv=8: E2B (8 Q heads / 1 KV head)
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [512]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [2, 4, 8]),
            CompileMacroOption('M_TILESIZE', 'm', [8]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
        ],
        [
            # Gemma4 Unified 12B global attention: 16 Q heads / 1 KV head
            # (GQA ratio 16) with 512-wide heads. The decode kernel tiles Q
            # rows as roundUp(headGrpSize * beamWidth, 16), so ratio 16 fully
            # occupies the same 16-row M tile that the nqpkv 4/8 kernels pad;
            # shared-memory and register footprints are unchanged. Generated
            # for the same SM set as the other d512 kernels.
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [512]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [16]),
            CompileMacroOption('M_TILESIZE', 'm', [8]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
        ],
    ]

    edgellm_config_list_spec_dec = [
        [
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [256, 128, 64, 32]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [0]),
            CompileMacroOption('M_TILESIZE', 'm', [32]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [1]),
        ],
        [
            # MTP head_dim=512 supports both contiguous and paged separate K/V cache contracts.
            CompileMacroOption('DTYPE', 'dt', ['__half']),
            CompileMacroOption('HEAD_ELEMS', 'd', [512]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               TOKENS_PER_PAGE_OPTIONS),
            CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [0]),
            CompileMacroOption('M_TILESIZE', 'm', [32]),
            CompileMacroOption('SPEC_DEC', 'spec_dec', [1]),
        ],
    ]

    edgellm_config_list.extend(edgellm_config_list_spec_dec)
    arch_macro_lists = []
    for cfg in edgellm_config_list:
        arch_macro_lists.extend(generate_compile_arch_macro_list(cfg))

    if args.list_outputs:
        print(';'.join(get_generated_output_paths(cubin_dir,
                                                  arch_macro_lists)))
        sys.exit(0)

    if os.path.exists(cubin_dir):
        shutil.rmtree(cubin_dir)
    os.makedirs(cubin_dir)

    cpu_count = os.cpu_count()
    thread_count = cpu_count // 2 if cpu_count >= 2 else cpu_count
    with multiprocessing.Pool(
            processes=thread_count,
            initializer=init_cubin_gen_worker,
            initargs=(cubin_dir, nvcc_bin, clean_cubin)) as pool:
        name_size_list = pool.map(run_cubin_gen, arch_macro_lists)
    header_file_contents = generate_header_file_contents(
        arch_macro_lists, name_size_list)

    with open(os.path.join(cubin_dir, build_func_name_prefix + '_cubin.h'),
              "w") as f:
        f.write("".join(
            [cpp_file_prefix_text, header_file_contents,
             cpp_file_suffex_text]))
