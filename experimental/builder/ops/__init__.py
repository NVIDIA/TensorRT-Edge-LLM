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
"""Symbolic tensors, modules, and the unified functional operation API.

The package owns the PyTorch-like ``Tensor`` and ``Module`` abstractions,
checkpoint-backed layers, and one ``functional`` namespace for all operations.
Every operation discovers the active graph from module scope, so model
definitions never receive or pass a TensorRT network.
"""

import tensorrt as trt

from . import functional
from .audio import (AudioConvSubsampler, AudioPositionEmbedding,
                    AudioTransformer)
from .embedding import Embedding
from .gated_delta_net import GatedDeltaNet
from .linear import DynamicLinear, Linear
from .mlp import FP32GatedMLP, GatedMLP
from .module import BuildContext, BuildOptions, Module, NetworkModule
from .moe import (GatedExperts, GroupedSigmoidRouter, NonGatedNvfp4Experts,
                  TopKRouter, prepare_gated_int4_weights,
                  prepare_gated_nvfp4_weights)
from .normalization import LayerNorm, RMSNorm
from .ragged import RaggedDecoderInputs, add_ragged_decoder_inputs
from .tensor import Tensor
from .transformer import (DecoderAttention, DecoderLayer, DecoderModel,
                          GatedDecoderAttention, QKNormDecoderAttention,
                          TreeAttention, pack_qkv)
from .vision import (FastPositionEmbedding, PackedVisionAttention,
                     PatchEmbedding, VisionMLP, VisionPatchMerger,
                     VisionTransformerBlock)

bool = trt.bool
float16 = trt.float16
float32 = trt.float32
int8 = trt.int8
int32 = trt.int32
int64 = trt.int64

__all__ = [
    "BuildContext",
    "BuildOptions",
    "DecoderAttention",
    "DecoderLayer",
    "DecoderModel",
    "DynamicLinear",
    "Embedding",
    "FP32GatedMLP",
    "GatedMLP",
    "GatedExperts",
    "GatedDeltaNet",
    "GatedDecoderAttention",
    "GroupedSigmoidRouter",
    "LayerNorm",
    "Linear",
    "Module",
    "NetworkModule",
    "NonGatedNvfp4Experts",
    "QKNormDecoderAttention",
    "Tensor",
    "TreeAttention",
    "TopKRouter",
    "AudioConvSubsampler",
    "AudioPositionEmbedding",
    "AudioTransformer",
    "bool",
    "float16",
    "float32",
    "FastPositionEmbedding",
    "functional",
    "int8",
    "int32",
    "int64",
    "PackedVisionAttention",
    "PatchEmbedding",
    "pack_qkv",
    "prepare_gated_int4_weights",
    "prepare_gated_nvfp4_weights",
    "RMSNorm",
    "RaggedDecoderInputs",
    "VisionMLP",
    "VisionPatchMerger",
    "VisionTransformerBlock",
    "add_ragged_decoder_inputs",
]
