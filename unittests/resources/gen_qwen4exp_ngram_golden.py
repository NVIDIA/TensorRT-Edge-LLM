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
"""Generate qwen4exp_ngram_row_ids_golden.json for unittests/cpp/runtime/embedding/ngramRowIdsTest.cpp.

Qwen3.8-Flash-Next (HF ``model_type = qwen4_exp``) injects a hashed n-gram memory table at
``layer_idx = 1``. The row IDs into that table depend only on ``input_ids``, so the runtime
computes them outside the engine. This script pins the row IDs against the HuggingFace
implementation.

It holds two independent implementations and asserts they agree bit-exactly before writing
anything:

- ``hf_row_ids`` drives the real ``Qwen4ExpTextNGramEmbedding`` with its embedding table on the
  ``meta`` device, capturing ``ngram_ids`` at the gather call. No table weights are needed.
- ``reference_row_ids`` is a self-contained reimplementation with the uint64 semantics the C++
  runtime has to reproduce. This is the model the C++ code mirrors.

Run from the repo root with a venv that has transformers main (>= the qwen4_exp addition):
    ./venv_qwen4exp_golden/bin/python unittests/resources/gen_qwen4exp_ngram_golden.py

The BF16 checkpoint supplies the config and the tokenizer, and is also cross-checked against
the constants recomputed from the config. It is read-only; nothing is written next to it.
"""

import argparse
import json
import pathlib

import torch

# Imported lazily: the reference implementation below is plain Python, and consumers that only need
# it run in environments without a transformers build carrying qwen4_exp.

DEFAULT_CHECKPOINT = pathlib.Path(
    "/code/edgellm-workspace/workspace/hub-llm-models/Qwen3.8-Flash-Next")
OUTPUT_NAME = "qwen4exp_ngram_row_ids_golden.json"

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63


# --------------------------------------------------------------------------------------
# Reference implementation. The C++ runtime mirrors this, so it deliberately spells out the
# integer semantics instead of leaning on Python's arbitrary-precision ints.
# --------------------------------------------------------------------------------------
def _shifted_token(history, index, shift, eos_token_id):
    """One entry of HF's ``_shift_right_ignore_eos``.

    HF expresses the segment boundary with a cummax over EOS positions; per position that
    reduces to "no EOS in the ``shift`` tokens preceding ``index``", because a token is only
    reachable when the whole span back to it stays inside the current segment.
    """
    if index - shift < 0:
        return eos_token_id
    for probe in range(index - shift, index):
        if history[probe] == eos_token_id:
            return eos_token_id
    return history[index - shift]


def _remainder(bits, divisor):
    """``torch.remainder`` on an int64 bit pattern: result carries the divisor's sign."""
    value = bits - (1 << 64) if bits & SIGN64 else bits
    return value % divisor  # Python % already floors, matching torch for a positive divisor.


def reference_row_ids(input_ids, previous_context, constants):
    """Row IDs for one sequence. Returns ``(rows[T][ngram_heads], next_previous_context)``."""
    ngram_size = constants["ngram_size"]
    heads_per_ngram = constants["heads_per_ngram"]
    context_len = constants["context_len"]
    eos_token_id = constants["eos_token_id"]
    multipliers = constants["layer_multipliers"]
    vocab_sizes = constants["ngram_heads_vocab_sizes"]
    offsets = constants["ngram_heads_offsets"]

    history = list(previous_context) + list(input_ids)
    rows = []
    for position in range(context_len, len(history)):
        shifted = [
            _shifted_token(history, position, shift, eos_token_id)
            for shift in range(ngram_size)
        ]
        row = []
        for ngram in range(2, ngram_size + 1):
            mixed = 0
            for order in range(ngram):
                mixed ^= (shifted[order] * multipliers[order]) & MASK64
            head_begin = (ngram - 2) * heads_per_ngram
            for head in range(head_begin, head_begin + heads_per_ngram):
                row.append(
                    _remainder(mixed, vocab_sizes[head]) + offsets[head])
        rows.append(row)

    next_context = (list(previous_context) + list(input_ids))[-context_len:]
    return rows, next_context


# --------------------------------------------------------------------------------------
# HuggingFace golden
# --------------------------------------------------------------------------------------
class _CaptureCache:
    """Minimal stand-in for the conv-state slot HF reuses to carry the token history.

    ``Qwen4ExpTextNGramEmbedding`` only ever asks the cache whether slot 2 exists, reads it, and
    writes back the trailing ``context_len`` tokens. Building a real ``DynamicCache`` would drag
    in all 48 layers' state for no added coverage.
    """

    def __init__(self, context_len, layer_idx=1):
        self.context_len = context_len
        self.state = None
        self.layers = [self] * (layer_idx + 1)
        self.conv_states = [None, None, None]

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return self.state is not None

    def update_conv_state(self,
                          conv_states,
                          layer_idx,
                          state_idx=0,
                          conv_kernel_size=None,
                          **kw):
        previous = self.state if self.state is not None else conv_states.new_empty(
            (conv_states.shape[0], 0))
        full = torch.cat([previous, conv_states], dim=-1)
        self.state = full[..., -self.context_len:].clone()
        self.conv_states[2] = self.state
        return full


class _CaptureEmbedding(torch.nn.Module):
    """Stands in for the 320M-row table: records the row IDs and returns a dummy of the shape
    ``flatten(-2)`` expects."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        # HF reads `weight.device` to decide where to run the gather; keeping it on meta selects
        # the "table not materialized" path.
        self.weight = torch.nn.Parameter(torch.empty(0, device="meta"),
                                         requires_grad=False)

    def forward(self, ngram_ids):
        self.sink["ngram_ids"] = ngram_ids.clone()
        return torch.zeros(*ngram_ids.shape, 1, dtype=torch.float32)


def build_module(config):
    """Construct the HF module with the 102 GB table left unmaterialized on ``meta``."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding, _build_layer_multipliers)

    with torch.device("meta"):
        module = Qwen4ExpTextNGramEmbedding(config,
                                            config.ple_embed_dim,
                                            layer_idx=1,
                                            ple_layer_index=0)
    # The three constant buffers land on meta along with the table; restore them exactly the way
    # `Qwen4ExpPreTrainedModel._init_weights` does, which is also what the checkpoint stores.
    module.layer_multipliers = _build_layer_multipliers(
        module.unigram_vocab_size, module.ngram_size, module.ple_layer_index,
        module.seed)
    module.ngram_heads_vocab_sizes = torch.tensor(module.head_vocab_sizes,
                                                  dtype=torch.long)
    module.ngram_heads_offsets = torch.tensor(module.head_offsets,
                                              dtype=torch.long)

    captured = {}
    module.ngram_embedding = _CaptureEmbedding(captured)
    return module, captured


def hf_row_ids(module, captured, input_ids, cache):
    module.forward(input_ids, cache)
    return captured["ngram_ids"]


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------
def build_cases(tokenizer, eos_token_id, vocab_size):
    """Each case is a list of steps sharing one cache, so state carried across steps is covered."""

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prose = encode(
        "TensorRT Edge-LLM deploys large language models on NVIDIA Jetson Thor."
    )
    code = encode("auto const rowIds = computeNgramRowIds(tokenIds);")
    assert len(prose) >= 12 and len(code) >= 8

    cases = []

    cases.append(("prose_single_shot", [[prose]]))
    cases.append(("code_single_shot", [[code]]))

    # A shift of 1 or 2 crossing an EOS must clamp back to EOS. Covers EOS at the very first
    # position, in the middle, and two in a row.
    with_eos = [eos_token_id] + prose[:4] + [eos_token_id] + prose[4:9] + \
        [eos_token_id, eos_token_id] + prose[9:]
    cases.append(("eos_boundaries", [[with_eos]]))

    # Whole-sequence prefill vs. the same sequence in chunks must agree; the first chunk is
    # shorter than context_len, which is the case HF pads with EOS rather than zeros.
    cases.append(("chunked_1_then_rest", [[prose[:1]], [prose[1:]]]))
    cases.append(("chunked_1_2_3_rest", [[prose[:1]], [prose[1:3]],
                                         [prose[3:6]], [prose[6:]]]))
    cases.append(("chunked_across_eos", [[with_eos[:3]], [with_eos[3:7]],
                                         [with_eos[7:]]]))

    # Prefill then single-token decode steps, including a decode step that emits EOS.
    decode_steps = [[prose[:6]]] + [[[token]] for token in prose[6:10]] + \
        [[[eos_token_id]], [[prose[10]]], [[prose[11]]]]
    cases.append(("prefill_then_decode", decode_steps))

    # Batch with per-sequence state: three sequences advanced together through chunks.
    batch_a = prose[:10]
    batch_b = [eos_token_id] + code[:7] + [eos_token_id, code[7]]
    batch_c = [
        0, vocab_size - 1, eos_token_id, 0, vocab_size - 1, 1, 2, eos_token_id,
        3, 4
    ]
    assert len(batch_a) == len(batch_b) == len(batch_c) == 10
    batch = [batch_a, batch_b, batch_c]
    cases.append(("batch_single_shot", [[row[:] for row in batch]]))
    cases.append(("batch_chunked", [[row[:1] for row in batch],
                                    [row[1:4] for row in batch],
                                    [row[4:] for row in batch]]))

    # Extremal token IDs: 0, the last valid ID, and EOS itself, where the multiply is largest.
    cases.append(
        ("extremal_ids",
         [[[vocab_size - 1] * 5 + [0, eos_token_id, vocab_size - 1]]]))
    cases.append(("single_token", [[[vocab_size - 1]]]))
    cases.append(("all_eos", [[[eos_token_id] * 6]]))

    return cases


def build_remainder_cases(vocab_sizes):
    """Pin ``torch.remainder`` on negative int64 bit patterns.

    In-range token IDs can never produce one: HF caps each multiplier at ``(2**63 - 1) // vocab_size``,
    so every product stays below 2**63 and their XOR leaves bit 63 clear. The C++ code implements the
    branch regardless, because C++ ``%`` would return a negative remainder there and silently address a
    different row, so it needs coverage the row-ID cases cannot give it.
    """
    patterns = [
        0,
        1,
        SIGN64,
        SIGN64 + 1,
        MASK64,
        MASK64 - 1,
        (1 << 63) + (1 << 40) + 12345,
        0xDEADBEEFCAFEBABE,
        0x8000000000000001,
        0xFFFFFFFF00000000,
    ]
    cases = []
    for bits in patterns:
        signed = bits - (1 << 64) if bits & SIGN64 else bits
        for divisor in (vocab_sizes[0], vocab_sizes[-1], 7):
            expected = torch.remainder(
                torch.tensor(signed, dtype=torch.int64),
                torch.tensor(divisor, dtype=torch.int64)).item()
            assert expected == _remainder(bits, divisor)
            cases.append({
                "bits": bits,
                "divisor": divisor,
                "expected": expected
            })
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        type=pathlib.Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / OUTPUT_NAME)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    with open(args.checkpoint / "config.json", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import \
        Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(**raw_config.get("text_config", raw_config))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    module, captured = build_module(config)
    constants = {
        "ngram_size": module.ngram_size,
        "heads_per_ngram": module.heads_per_ngram,
        "ngram_heads": module.ngram_heads,
        "context_len": module.context_len,
        "eos_token_id": module.eos_token_id,
        "unigram_vocab_size": module.unigram_vocab_size,
        "layer_multipliers": module.layer_multipliers.tolist(),
        "ngram_heads_vocab_sizes": module.ngram_heads_vocab_sizes.tolist(),
        "ngram_heads_offsets": module.ngram_heads_offsets.tolist(),
    }
    verify_constants_against_checkpoint(args.checkpoint, constants)

    cases = []
    for name, steps in build_cases(tokenizer, constants["eos_token_id"],
                                   constants["unigram_vocab_size"]):
        cache = _CaptureCache(constants["context_len"])
        batch_size = len(steps[0])
        contexts = [[constants["eos_token_id"]] * constants["context_len"]
                    for _ in range(batch_size)]

        emitted_steps = []
        for step_ids in steps:
            assert len(
                step_ids
            ) == batch_size, f"{name}: batch size must not change mid-case"
            golden = hf_row_ids(module, captured,
                                torch.tensor(step_ids, dtype=torch.long),
                                cache)
            assert golden.shape == (batch_size, len(step_ids[0]),
                                    constants["ngram_heads"])

            expected = []
            for sequence in range(batch_size):
                rows, contexts[sequence] = reference_row_ids(
                    step_ids[sequence], contexts[sequence], constants)
                expected.append(rows)
            assert expected == golden.tolist(), (
                f"{name}: reference row IDs disagree with HuggingFace")

            emitted_steps.append({
                "input_ids": step_ids,
                "row_ids": golden.tolist()
            })

        cases.append({
            "name": name,
            "steps": emitted_steps,
            "final_token_history": contexts,
        })

    # A chunked case must reproduce its single-shot counterpart; assert it here so a regression
    # shows up while generating rather than only in the C++ test.
    assert_chunking_consistency(cases)

    document = {
        "description":
        "Qwen3.8-Flash-Next n-gram row IDs, pinned against HuggingFace.",
        "generated_by":
        "unittests/resources/" + pathlib.Path(__file__).name,
        "reference":
        "transformers Qwen4ExpTextNGramEmbedding (model_type qwen4_exp)",
        "constants":
        constants,
        "cases":
        cases,
        "remainder_cases":
        build_remainder_cases(constants["ngram_heads_vocab_sizes"]),
    }
    # Compact: the file is machine-generated and regenerable, so readability buys nothing and the
    # indented form is twice the size.
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
        handle.write("\n")
    total = sum(
        len(step["input_ids"][0]) * len(step["input_ids"]) for case in cases
        for step in case["steps"])
    print(f"wrote {args.output} : {len(cases)} cases, {total} positions")


def verify_constants_against_checkpoint(checkpoint, constants):
    """The constants also ship as checkpoint buffers; a mismatch means the config is not the
    one the weights were built with."""
    from safetensors import safe_open

    with open(checkpoint / "model.safetensors.index.json",
              encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    for buffer_name in ("layer_multipliers", "ngram_heads_vocab_sizes",
                        "ngram_heads_offsets"):
        shard = weight_map.get(prefix + buffer_name)
        if shard is None:
            print(
                f"warning: {buffer_name} absent from checkpoint index, skipping cross-check"
            )
            continue
        with safe_open(checkpoint / shard, framework="pt") as handle:
            stored = handle.get_tensor(prefix + buffer_name).tolist()
        assert stored == constants[buffer_name], (
            f"{buffer_name} differs between checkpoint and config-derived value: "
            f"{stored} vs {constants[buffer_name]}")
    print("checkpoint constants cross-check: ok")


def assert_chunking_consistency(cases):
    by_name = {case["name"]: case for case in cases}

    def concatenated_rows(case):
        rows = [[] for _ in case["steps"][0]["row_ids"]]
        for step in case["steps"]:
            for sequence, sequence_rows in enumerate(step["row_ids"]):
                rows[sequence].extend(sequence_rows)
        return rows

    prose_rows = concatenated_rows(by_name["prose_single_shot"])
    for name in ("chunked_1_then_rest", "chunked_1_2_3_rest"):
        assert concatenated_rows(
            by_name[name]) == prose_rows, f"{name} != prose_single_shot"

    eos_rows = concatenated_rows(by_name["eos_boundaries"])
    assert concatenated_rows(by_name["chunked_across_eos"]) == eos_rows

    batch_rows = concatenated_rows(by_name["batch_single_shot"])
    assert concatenated_rows(by_name["batch_chunked"]) == batch_rows

    decode_rows = concatenated_rows(by_name["prefill_then_decode"])
    assert all(len(sequence) == 13 for sequence in decode_rows)
    print("chunk/decode consistency: ok")


if __name__ == "__main__":
    main()
