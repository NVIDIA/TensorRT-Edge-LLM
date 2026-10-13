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
"""Runtime sidecars that turn the pi0.5 engines into a policy.

Two artifacts, both staged next to the engines by the builder:

* ``text_tokenizer/`` -- the PaliGemma tokenizer in HuggingFace form. The
  checkpoint names ``google/paligemma-3b-pt-224``, whose HF repo is gated; the
  identical SentencePiece model is served anonymously from ``big_vision`` and is
  what openpi itself loads, so it is converted here instead.
* ``policy.json`` -- the single policy manifest: the observation/action contract
  (feature dimensions, camera list, image resolution) plus the export identity
  that ties the component directories to one checkpoint. The shapes come from the
  checkpoint pins; every robot-facing dimension comes from
  ``OPENPI_POLICY_CONTRACTS``, keyed by the configuration name.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import struct
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

# openpi's ``models/tokenizer.py`` downloads this anonymously; the HF mirror
# ``google/paligemma-3b-pt-224`` is gated and cannot be fetched unattended.
PALIGEMMA_TOKENIZER_URL = (
    "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model")

POLICY_CONTRACT_FILENAME = "policy.json"
# What the manifest says the policy *is*, as opposed to which parts of it have
# been exported. A partial re-export may grow ``components`` and may stage
# norm_stats, but must not silently redefine any of these.
POLICY_SEMANTIC_FIELDS = ("model_family", "policy_config", "adapter", "state",
                          "action", "cameras", "image_resolution", "tokenizer",
                          "discrete_state_input")
TEXT_TOKENIZER_DIRNAME = "text_tokenizer"
MODEL_FAMILY = "pi05"

# openpi discretizes the normalized state into this many bins before writing it
# into the prompt (``PaligemmaTokenizer.tokenize``).
STATE_NUM_BINS = 256

# ``Pi0Config.__post_init__`` for every pi0.5 configuration.
MAX_TOKEN_LEN = 200

# Image slots the architecture carries: one base view and two wrist views. A
# configuration that names fewer leaves the rest out of the prefix entirely.
MAX_CAMERA_SLOTS = 3

#: The policy contract each openpi configuration declares, transcribed from its
#: ``TrainConfig``. This is the authority, not the LeRobot processor sidecars: a
#: Hugging Face mirror distributes the weights, but its ``policy_preprocessor.json``
#: describes LeRobot's own policy (mean/std, a discretized state in the prompt), which
#: is a different contract from the one openpi and TSE serve.
#:
#: ``cameras`` is ordered as the prefix consumes the image slots. ``opt_views`` tunes the
#: visual profile and is not a limit; the canonical shape stays ``MAX_CAMERA_SLOTS``.
OPENPI_POLICY_CONTRACTS = {
    "pi05_libero": {
        # pi05_libero sets discrete_state_input False, and pi0.5 carries no state
        # projection, so the state reaches the model through neither path.
        "adapter": "libero",
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": 10,
        "discrete_state_input": False,
        "cameras":
        (("observation/image", True), ("observation/wrist_image", True)),
        "ignored_cameras": (),
        "opt_views": 2,
    },
    "pi05_droid": {
        "adapter":
        "droid",
        "state_dim":
        8,
        "action_dim":
        8,
        "action_horizon":
        15,
        "discrete_state_input":
        True,
        "cameras": (("observation/exterior_image_1_left", True),
                    ("observation/wrist_image_left", True)),
        "ignored_cameras": (),
        "opt_views":
        2,
    },
    "pi05_aloha": {
        "adapter":
        "aloha",
        "state_dim":
        14,
        "action_dim":
        14,
        "action_horizon":
        50,
        "discrete_state_input":
        True,
        "cameras": (("cam_high", True), ("cam_left_wrist", False),
                    ("cam_right_wrist", False)),
        # Accepted in the raw observation and dropped by AlohaInputs; it never reaches
        # the model.
        "ignored_cameras": ("cam_low", ),
        "opt_views":
        3,
    },
}

# The one embodiment nameable without being told: a LeRobot mirror whose feature
# contract is unambiguous. The openpi-converted checkpoints carry no field that
# separates the others, so they are named on the command line.
_AUTODETECTED_EMBODIMENTS = {("pi05", 8, 7, 2): "pi05_libero"}

# Bytes sampled per window by ``checkpoint_fingerprint``, three windows per tensor. The
# sampling is what keeps the fingerprint cheap on a multi-GB checkpoint, and it scopes
# what the fingerprint can prove.
FINGERPRINT_WINDOW = 4096


def load_json_if_present(path: str) -> "dict | None":
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _safetensors_header(path: str) -> "tuple[bytes, int, dict]":
    """Return ``(header_bytes, data_base_offset, tensor_entries)``."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = f.read(length)
    entries = json.loads(header)
    entries.pop("__metadata__", None)
    return header, 8 + length, entries


def checkpoint_fingerprint(checkpoint: str) -> str:
    """Fingerprint a checkpoint from its weight bytes, not its declared shapes.

    Hashes each shard's safetensors header plus three fixed ``FINGERPRINT_WINDOW``
    windows -- head, middle, tail -- of every tensor's data, walked in file-offset
    order. Two checkpoints that share an architecture but were trained differently
    disagree in every window; a difference confined to bytes outside the sampled
    windows is not detected.
    """
    shards = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"{checkpoint} holds no *.safetensors; a pi0.5 export cannot be "
            "identified without its weights")
    digest = hashlib.sha256()
    for shard in shards:
        header, base, entries = _safetensors_header(shard)
        digest.update(os.path.basename(shard).encode())
        digest.update(header)
        ordered = sorted(entries.items(),
                         key=lambda kv: kv[1]["data_offsets"][0])
        with open(shard, "rb") as f:
            for name, entry in ordered:
                start, end = entry["data_offsets"]
                size = end - start
                window = min(FINGERPRINT_WINDOW, size)
                digest.update(name.encode())
                for offset in sorted({0, (size - window) // 2, size - window}):
                    f.seek(base + start + offset)
                    digest.update(f.read(window))
    return digest.hexdigest()


def _feature_dim(features: dict, wanted_type: str) -> "int | None":
    for feature in features.values():
        if feature.get("type") == wanted_type:
            return int(feature["shape"][0])
    return None


def _visual_features(features: dict) -> "list[str]":
    return [k for k, v in features.items() if v.get("type") == "VISUAL"]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _sentencepiece_model():
    logger.info("Fetching %s", PALIGEMMA_TOKENIZER_URL)
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "paligemma_tokenizer.model")
        with urllib.request.urlopen(PALIGEMMA_TOKENIZER_URL) as response, open(
                path, "wb") as out:
            out.write(response.read())
        yield path


def _convert_sentencepiece(model_path: str) -> "tuple[object, int]":
    """Convert the PaliGemma SentencePiece BPE model to a fast tokenizer.

    Mirrors ``transformers`` ``GemmaConverter``, whose own SentencePiece
    extractor is broken in the pinned release. The model sets
    ``add_dummy_prefix = False``, so the normalizer is a bare space-to-marker
    replacement with no prepended word boundary.
    """
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import import_protobuf
    from transformers.tokenization_utils_base import generate_merges

    proto = import_protobuf().ModelProto()
    with open(model_path, "rb") as f:
        proto.ParseFromString(f.read())
    if proto.trainer_spec.model_type != 2:
        raise ValueError("Expected a SentencePiece BPE model for PaliGemma")

    vocab_scores = [(piece.piece, piece.score) for piece in proto.pieces]
    vocab = {piece: index for index, (piece, _) in enumerate(vocab_scores)}
    backend = Tokenizer(
        BPE(vocab,
            generate_merges(vocab, vocab_scores),
            unk_token=proto.trainer_spec.unk_piece,
            fuse_unk=True,
            byte_fallback=proto.trainer_spec.byte_fallback,
            dropout=None))
    backend.normalizer = normalizers.Replace(" ", "▁")
    backend.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    backend.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=proto.trainer_spec.bos_piece,
        eos_token=proto.trainer_spec.eos_piece,
        unk_token=proto.trainer_spec.unk_piece,
        pad_token=proto.trainer_spec.pad_piece,
        add_bos_token=True,
        add_eos_token=False,
    )
    return tokenizer, len(vocab_scores)


def stage_text_tokenizer(output_dir: str) -> dict:
    """Write ``text_tokenizer/`` and return its contract entry."""
    with _sentencepiece_model() as model_path:
        tokenizer, vocab_size = _convert_sentencepiece(model_path)
        digest = sha256_file(model_path)
    tokenizer_dir = os.path.join(output_dir, TEXT_TOKENIZER_DIRNAME)
    tokenizer.save_pretrained(tokenizer_dir)
    if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
        raise RuntimeError(
            f"{tokenizer_dir} has no tokenizer.json after conversion; the "
            "runtime cannot tokenize a prompt without it")
    logger.info("Staged %s (vocab %d)", tokenizer_dir, vocab_size)
    return {
        "hf_name": "google/paligemma-3b-pt-224",
        "sentencepiece_source": PALIGEMMA_TOKENIZER_URL,
        "sentencepiece_sha256": digest,
        "vocab_size": vocab_size,
        "add_bos": True,
        "add_eos": False,
    }


def _check_horizon(checkpoint: str, config: dict, name: str) -> None:
    """Refuse a checkpoint whose own horizon contradicts the named configuration.

    The configuration stays the authority for the horizon the graph is built at; this
    only rejects a pairing the weights cannot serve. All three converted checkpoints
    pad actions to 32 dims, so a mismatched pairing otherwise exports, builds and runs,
    and only the commands are wrong. openpi's converter writes ``action_horizon`` and
    the LeRobot releases ``n_action_steps``; ``chunk_size`` is a different number
    (50 on LIBERO's 10-step configuration) and is deliberately not read.
    """
    declared = config.get("n_action_steps", config.get("action_horizon"))
    if declared is None:
        return
    expected = OPENPI_POLICY_CONTRACTS[name]["action_horizon"]
    if int(declared) != expected:
        raise ValueError(
            f"{checkpoint} declares an action horizon of {int(declared)}, but "
            f"{name} is a {expected}-step configuration; the checkpoint and the "
            "policy configuration do not describe the same policy")


def policy_config_name(checkpoint: str,
                       config: dict,
                       requested: "str | None" = None) -> str:
    """Name the openpi configuration this export serves.

    The name gates the runtime's observation and action processing, so it comes from
    the caller or from an unambiguous feature contract -- never from a directory or a
    repository name.
    """
    if requested is not None:
        if requested not in OPENPI_POLICY_CONTRACTS:
            raise ValueError(
                f"Unsupported pi0.5 policy config {requested!r}; "
                f"expected one of {sorted(OPENPI_POLICY_CONTRACTS)}")
        _check_horizon(checkpoint, config, requested)
        return requested
    inputs = config.get("input_features", {})
    embodiment = (config.get("type"), _feature_dim(inputs, "STATE"),
                  _feature_dim(config.get("output_features", {}),
                               "ACTION"), len(_visual_features(inputs)))
    known = _AUTODETECTED_EMBODIMENTS.get(embodiment)
    if known is None:
        raise ValueError(
            f"{checkpoint}: its feature contract (type, state dim, action dim, "
            f"cameras) = {embodiment} names no openpi configuration on its own; "
            f"pass --pi05-policy-config with one of "
            f"{sorted(OPENPI_POLICY_CONTRACTS)}")
    _check_horizon(checkpoint, config, known)
    return known


def policy_semantics(contract: dict) -> dict:
    """The part of a manifest a partial re-export is not allowed to change."""
    return {k: contract[k] for k in POLICY_SEMANTIC_FIELDS if k in contract}


def build_policy_contract(checkpoint: str, tokenizer: dict, identity: dict,
                          config_name: str) -> dict:
    """Derive ``policy.json``: the model ABI from the checkpoint, the policy from openpi.

    The checkpoint pins only what the weights pin -- the padded action width, the image
    resolution. Every robot-facing dimension comes from ``OPENPI_POLICY_CONTRACTS``,
    because a Hugging Face mirror ships LeRobot's own configuration beside the weights
    and that is a different contract from the one openpi serves.
    """
    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)
    policy = OPENPI_POLICY_CONTRACTS[config_name]
    model_action_dim = int(config.get("max_action_dim", 32))
    if policy["action_dim"] > model_action_dim:
        raise ValueError(
            f"{checkpoint} pads actions to {model_action_dim} dims, too narrow for "
            f"{config_name}'s {policy['action_dim']}")
    return {
        "contract_version": identity["contract_version"],
        "model_family": MODEL_FAMILY,
        "policy_config": config_name,
        "adapter": policy["adapter"],
        "discrete_state_input": policy["discrete_state_input"],
        "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
        "export_id": identity["export_id"],
        "components": list(identity["components"]),
        "state": {
            "dim": policy["state_dim"],
            "num_bins": STATE_NUM_BINS,
        },
        "action": {
            "dim": policy["action_dim"],
            "max_dim": model_action_dim,
            "horizon": policy["action_horizon"],
        },
        "cameras": {
            "slots": [{
                "name": name,
                "required": required
            } for name, required in policy["cameras"]],
            "ignored":
            list(policy["ignored_cameras"]),
        },
        "image_resolution": list(config.get("image_resolution", [224, 224])),
        "tokenizer": {
            "max_length": MAX_TOKEN_LEN,
            **tokenizer
        },
    }


def write_policy_contract(output_dir: str, contract: dict) -> None:
    """Write the manifest through a temp file, so no reader sees a partial one."""
    path = os.path.join(output_dir, POLICY_CONTRACT_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(contract, f, indent=2)
    os.replace(tmp_path, path)
    logger.info("Staged %s (state %dd, action %dd)", path,
                contract["state"]["dim"], contract["action"]["dim"])
