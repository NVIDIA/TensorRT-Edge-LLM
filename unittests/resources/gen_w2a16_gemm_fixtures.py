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
"""Prepare dense IQ2_XXS checkpoint blocks and independent FP32 GEMM oracles."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kernelSrcs'))
import w2a16_reference
from w2a16_gemm_cutedsl import prepare


def tensor_bytes(tensor):
    """Serialize a contiguous CPU tensor without changing its representation."""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def write_fixture(path, n, k, m, tensors):
    """Write a versioned header and raw tensors; return the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open('xb') as output:
        header = struct.pack('<4I', 0x57324431, n, k, m)
        output.write(header)
        digest.update(header)
        for tensor in tensors:
            data = tensor_bytes(tensor)
            output.write(data)
            digest.update(data)
    return digest.hexdigest()


def checkpoint_projections(checkpoint, keys, slices):
    """Yield complete projections or exact row/block slices of one projection."""
    for key in dict.fromkeys(keys):
        raw = checkpoint.get_tensor(key)
        if not slices:
            yield key, raw, None
            continue
        for shape in slices:
            n, k = map(int, shape.split(':'))
            if (raw.ndim != 3 or raw.shape[-1] != 66 or n <= 0 or n % 8
                    or k <= 0 or k % 256 or n > raw.shape[0]
                    or k > raw.shape[1] * 256):
                raise ValueError(f'{key}: invalid IQ2_XXS slice {shape}')
            yield key, raw[:n, :k // 256].contiguous(), dict(
                source_shape=list(raw.shape),
                rows=[0, n],
                blocks=[0, k // 256])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    parser.add_argument(
        '--tensor',
        action='append',
        required=True,
        help='Exact dense projection key; repeat for each projection.')
    parser.add_argument('--revision',
                        required=True,
                        help='Immutable checkpoint revision for provenance.')
    parser.add_argument('--rows', type=int, nargs='+', required=True)
    parser.add_argument(
        '--slice',
        action='append',
        default=[],
        help=
        'Exact leading N:K row/block slice; requires one tensor. Repeat for boundary cases.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='New directory; existing output is never overwritten.')
    args = parser.parse_args()
    if args.slice and len(args.tensor) != 1:
        parser.error('--slice requires exactly one --tensor')
    if any(m < 1 or m > 32768 for m in args.rows):
        parser.error('Every M must be in [1,32768]')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    manifest = dict(
        schema_version=1,
        revision=args.revision,
        checkpoint_file=str(args.checkpoint_file.resolve()),
        reference=
        'independent GGML decode -> BF16 weights/inputs -> FP32 matmul -> BF16',
        cases=[])
    with safe_open(args.checkpoint_file, framework='pt',
                   device='cpu') as checkpoint:
        for index, (key, raw, source_slice) in enumerate(
                checkpoint_projections(checkpoint, args.tensor, args.slice)):
            values, bases = prepare.pack_weights(raw)
            n, kb, _ = raw.shape
            k = kb * 256
            weights = w2a16_reference.dequantize_blocks(raw).bfloat16().float()
            if not torch.isfinite(weights).all().item():
                raise ValueError(f'{key}: decoded BF16 weights must be finite')
            directory = args.output / f'projection{index}'
            directory.mkdir()
            entry = dict(tensor=key,
                         n=n,
                         k=k,
                         payload_sha256=hashlib.sha256(
                             tensor_bytes(raw)).hexdigest(),
                         directory=directory.name,
                         files={})
            entry['source_slice'] = source_slice
            entry['files']['weights.bin'] = write_fixture(
                directory / 'weights.bin', n, k, 0,
                (values, bases, prepare.make_lookup_table('cpu')))
            for m in dict.fromkeys(args.rows):
                generator = torch.Generator().manual_seed(1308 + m)
                # Two inputs allow replay to prove it uses changed device contents.
                tensors = []
                for _ in range(2):
                    x = (torch.randn(
                        (m, k), generator=generator) * 0.1).bfloat16()
                    y = (x.float() @ weights.T).bfloat16()
                    if not torch.isfinite(y).all().item():
                        raise ValueError(
                            f'{key}: nonfinite reference at M={m}')
                    tensors.extend((x, y))
                name = f'm{m}.bin'
                entry['files'][name] = write_fixture(directory / name, n, k, m,
                                                     tensors)
            manifest['cases'].append(entry)
    (args.output /
     'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
