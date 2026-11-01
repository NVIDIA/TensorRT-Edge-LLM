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
"""Thin adapter to the shared checkpoint-native LLM API."""

import json
import pathlib

from experimental import server
from experimental.server.parsing import reasoning
from experimental.server.runtime import engine


def _weight_label(quant, dtype):
    """Describe checkpoint weights using explicit quantization metadata."""
    algo = quant.get("quant_algo")
    method = str(quant.get("quant_method") or "").upper()
    if not algo and method in ("AWQ", "GPTQ"):
        bits = quant.get("bits")
        if type(bits) is int and bits > 0:
            return f"INT{bits} {method}"
    value = str(algo or method or dtype or "unknown").upper()
    return {
        "BFLOAT16": "BF16",
        "FLOAT16": "FP16",
        "FLOAT32": "FP32",
        "W4A16_AWQ": "INT4 AWQ",
        "INT4_AWQ": "INT4 AWQ",
        "INT4_GPTQ": "INT4 GPTQ",
    }.get(value, value)


def _device_name():
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
        return "unknown"
    status, properties = runtime.cudaGetDeviceProperties(device)
    if status != runtime.cudaError_t.cudaSuccess:
        return "unknown"
    name = properties.name
    return name.decode("utf-8", errors="replace").rstrip("\0") if isinstance(
        name, bytes) else str(name)


class Backend:
    """Keep one LLM loaded for the lifetime of a text conversation."""

    def __init__(self, config, on_progress):
        self.sampling = config.sampling
        self.llm = engine.LLM(config.model,
                              on_progress=on_progress,
                              **config.runtime)

    def validate(self, messages):
        """Reject an oversized turn before changing conversation history."""
        return self.llm.validate_context(messages, self.sampling)

    @property
    def context_capacity(self):
        """Return the loaded engine's total token capacity."""
        return self.llm.max_kv_cache_capacity

    def details(self):
        """Describe the loaded model without equating checkpoint and compute precision."""
        with open(pathlib.Path(self.llm.model_dir) / "config.json",
                  encoding="utf-8") as file:
            checkpoint = json.load(file)
        text = checkpoint.get("text_config", checkpoint)
        quant = checkpoint.get("quantization_config") or text.get(
            "quantization_config") or {}
        quant_path = pathlib.Path(self.llm.model_dir) / "hf_quant_config.json"
        if quant_path.is_file():
            with quant_path.open(encoding="utf-8") as file:
                quant = json.load(file).get("quantization", quant)
        dtype = text.get("dtype") or text.get("torch_dtype") or checkpoint.get(
            "dtype") or checkpoint.get("torch_dtype")
        weights = _weight_label(quant, dtype)
        speculation = "off"
        if self.llm.has_draft_model and not self.sampling.disable_spec_decode:
            with open(pathlib.Path(self.llm.bundle_dir) / "base_config.json",
                      encoding="utf-8") as file:
                speculation = json.load(file).get("spec_decode_type", "on")
        return {
            "model": self.llm.model_id,
            "device": _device_name(),
            "weights": f"{weights} (checkpoint)",
            "speculative mode": speculation,
        }

    def stream(self, messages, on_metrics):
        """Adapt reasoning deltas using the public cancellable stream wrapper."""
        parser = reasoning.REASONING_PARSERS.resolve(
            "auto" if self.sampling.enable_thinking else "none",
            self.llm.model_dir)
        splitter = parser.stream() if parser else None
        stream = self.llm.generate_stream(messages,
                                          self.sampling,
                                          on_metrics=on_metrics)

        def iterate():
            try:
                for delta in stream:
                    if delta.finish_reason == "error":
                        raise RuntimeError(
                            "Generation failed in the backend; see the preceding diagnostics."
                        )
                    if splitter:
                        yield from splitter.feed(delta.text)
                    elif delta.text:
                        yield reasoning.ReasoningDelta("content", delta.text)
                if splitter:
                    yield from splitter.flush()
            finally:
                stream.close()

        return server.CancellableIterator(iterate(), stream.cancel)

    def close(self):
        """Release native engines after generation stops."""
        self.llm.close()


def rates(metrics):
    """Return native prompt and decode rates, or None when unavailable.

    Both rates use the native token count for the corresponding phase.
    """
    prompt = None
    generation = None
    if metrics.processed_prompt_tokens is not None and metrics.prompt_time > 0:
        prompt = metrics.processed_prompt_tokens / metrics.prompt_time
    if metrics.generation_time > 0:
        generation = metrics.decode_tokens_per_second
    return prompt, generation
