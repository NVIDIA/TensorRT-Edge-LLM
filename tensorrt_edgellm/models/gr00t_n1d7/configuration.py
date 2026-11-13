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
"""GR00T N1.7 checkpoint facts shared by the ONNX export and the ONNX-less builder.
The Cosmos-Reason2-2B backbone config lives in another repository, so its geometry is
declared here and checked against the checkpoint's own tensor shapes.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

MODEL_TYPE = "Gr00tN1d7"
TOKENIZER_DIRNAME = "tokenizer"
# Files the runtime loads from the staged tokenizer: the HF tokenizer itself and a
# chat template, standalone or embedded in one of these JSON files.
TOKENIZER_FILE = "tokenizer.json"
CHAT_TEMPLATE_SOURCES = ("chat_template.jinja", "chat_template.json",
                         "tokenizer_config.json")

# Qwen3-VL-2B-Instruct's published vision/text configs, which Cosmos-Reason2-2B
# keeps; validate_backbone() checks the shapes they imply against the checkpoint.
VISION_CONFIG = {
    "model_type": "qwen3_vl",
    "depth": 24,
    "hidden_size": 1024,
    "num_heads": 16,
    "intermediate_size": 4096,
    "out_hidden_size": 2048,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [5, 11, 17],
    "hidden_act": "gelu_pytorch_tanh",
    "initializer_range": 0.02,
}

TEXT_CONFIG = {
    "model_type": "qwen3_vl_text",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 28,  # replaced by the checkpoint's own count
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "attention_bias": False,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
    "max_position_embeddings": 262144,
}

UNKNOWN_EMBODIMENT_ID = 22
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

# Policy-head geometry the graphs need but that config.json states indirectly.
DIT_INNER_DIM_KEY = "action_head.model.transformer_blocks.0.attn1.to_q.weight"
LM_PREFIX = "backbone.model.model.language_model.layers."
_LM_LAYER_MARKER = ".self_attn.q_proj.weight"


def is_gr00t_checkpoint(model_dir: str) -> bool:
    """Whether ``model_dir`` holds a GR00T N1.7 checkpoint (by ``model_type``)."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return False
    with open(path) as stream:
        return json.load(stream).get("model_type") == MODEL_TYPE


def require_tokenizer(model_dir: str) -> str:
    """Return ``<model_dir>/tokenizer``, refusing a checkpoint the runtime could not load:
    both routes call this before any engine is written."""
    source = os.path.join(model_dir, TOKENIZER_DIRNAME)
    if not os.path.isdir(source):
        raise ValueError(
            f"GR00T checkpoint is missing {TOKENIZER_DIRNAME}/: {source}")
    if not os.path.isfile(os.path.join(source, TOKENIZER_FILE)):
        raise ValueError(f"GR00T checkpoint is missing "
                         f"{TOKENIZER_DIRNAME}/{TOKENIZER_FILE}: {source}")
    if not any(
            _carries_chat_template(os.path.join(source, name))
            for name in CHAT_TEMPLATE_SOURCES):
        raise ValueError(
            f"GR00T checkpoint has no chat template under {source}; expected "
            f"one of {', '.join(CHAT_TEMPLATE_SOURCES)}")
    return source


def _carries_chat_template(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if not path.endswith(".json"):
        return True
    with open(path) as stream:
        return bool(json.load(stream).get("chat_template"))


def load_policy_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as stream:
        return json.load(stream)


def checkpoint_shapes(model_dir: str) -> dict:
    """Read every tensor's shape from the checkpoint index, without loading."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return {}
    with open(index) as stream:
        weight_map = json.load(stream)["weight_map"]
    shapes = {}
    for shard in sorted(set(weight_map.values())):
        shapes.update(_shard_header(os.path.join(model_dir, shard)))
    return shapes


def _shard_header(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(length))
    return {
        name: tuple(entry["shape"])
        for name, entry in header.items() if name != "__metadata__"
    }


def validate_backbone(shapes: dict) -> None:
    """Reject a checkpoint whose backbone is not Cosmos-Reason2-2B."""
    if not shapes:
        return
    visual = "backbone.model.model.visual."
    expected = {
        visual + "patch_embed.proj.weight": (1024, 3, 2, 16, 16),
        visual + "pos_embed.weight": (2304, 1024),
        visual + "blocks.0.attn.qkv.weight": (3072, 1024),
        visual + "blocks.0.mlp.linear_fc1.weight": (4096, 1024),
        visual + "merger.linear_fc2.weight": (2048, 4096),
        LM_PREFIX + "0.self_attn.q_proj.weight": (2048, 2048),
        LM_PREFIX + "0.self_attn.k_proj.weight": (1024, 2048),
        LM_PREFIX + "0.self_attn.q_norm.weight": (128, ),
        LM_PREFIX + "0.mlp.gate_proj.weight": (6144, 2048),
        "backbone.model.model.language_model.embed_tokens.weight":
        (151936, 2048),
    }
    for name, shape in expected.items():
        actual = shapes.get(name)
        if actual is None:
            raise ValueError(f"GR00T checkpoint is missing {name!r}; this "
                             "build expects a Cosmos-Reason2-2B backbone")
        if tuple(actual) != shape:
            raise ValueError(f"GR00T checkpoint {name!r} has shape {actual}, "
                             f"expected {shape} for Cosmos-Reason2-2B")
    blocks = {
        key
        for key in shapes
        if key.startswith(visual + "blocks.") and key.endswith(".norm1.weight")
    }
    if len(blocks) != VISION_CONFIG["depth"]:
        raise ValueError(f"GR00T vision tower has {len(blocks)} blocks, "
                         f"expected {VISION_CONFIG['depth']}")
    mergers = {
        key.rsplit(".", 2)[0]
        for key in shapes if key.startswith(visual + "deepstack_merger_list.")
    }
    if len(mergers) != len(VISION_CONFIG["deepstack_visual_indexes"]):
        raise ValueError(
            f"GR00T vision tower has {len(mergers)} DeepStack mergers, "
            f"expected {len(VISION_CONFIG['deepstack_visual_indexes'])}")


def num_lm_layers(shapes: dict, policy: dict) -> int:
    """The released checkpoint already carries only ``select_layer`` layers."""
    layers = {
        key[len(LM_PREFIX):].split(".", 1)[0]
        for key in shapes
        if key.startswith(LM_PREFIX) and key.endswith(_LM_LAYER_MARKER)
    }
    if layers:
        return len(layers)
    return int(policy.get("select_layer", TEXT_CONFIG["num_hidden_layers"]))


def dit_inner_dim(shapes: dict) -> int:
    return int(shapes.get(DIT_INNER_DIM_KEY, (1536, ))[0])


def text_config(shapes: dict, policy: dict) -> dict:
    text = dict(TEXT_CONFIG)
    text["num_hidden_layers"] = num_lm_layers(shapes, policy)
    return text


def resolve_embodiment(model_dir: str) -> int:
    """Match the checkpoint directory name against ``embodiment_id.json``'s tags
    (``libero_10`` -> ``libero_sim``); no match falls back to ``unknown`` with a warning.
    ``--gr00t-embodiment-id`` overrides either way."""
    path = os.path.join(model_dir, "embodiment_id.json")
    if not os.path.isfile(path):
        return UNKNOWN_EMBODIMENT_ID
    with open(path) as stream:
        tags = json.load(stream)
    tokens = {
        token
        for token in os.path.basename(os.path.normpath(
            model_dir)).lower().split("_") if token.isalpha()
    }
    best, best_score = None, 0
    for tag, index in tags.items():
        score = len(tokens.intersection(tag.lower().split("_")))
        if score > best_score:
            best, best_score = index, score
    if best is None:
        logger.warning(
            "GR00T checkpoint %s matches no embodiment tag; falling back to "
            "'unknown'. Pass --gr00t-embodiment-id to choose one.", model_dir)
        return int(tags.get("unknown", UNKNOWN_EMBODIMENT_ID))
    return int(best)
