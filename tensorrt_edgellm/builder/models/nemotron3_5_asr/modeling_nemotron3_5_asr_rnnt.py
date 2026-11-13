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
"""Nemotron-3.5-ASR checkpoint-direct RNN-T decoder step."""

import tensorrt as trt

from ...ops import Embedding, Linear, Module, NetworkModule
from ...ops import functional as F


class NemotronLSTMStep(Module):
    """One provider-compatible step of the two-layer prediction LSTM."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_layers: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def _linear(self, hidden, name: str):
        return F.linear_with_weights(
            hidden,
            self.weights.f16(self.key(f"weight_{name}")),
            self.weights.f16(self.key(f"bias_{name}")),
            rank=2,
        )

    def forward(self, hidden, hidden_state, cell_state):
        present_hidden = []
        present_cell = []
        for layer in range(self.num_layers):
            previous_hidden = hidden_state[layer]
            previous_cell = cell_state[layer]
            gates = (self._linear(hidden, f"ih_l{layer}") +
                     self._linear(previous_hidden, f"hh_l{layer}"))
            input_gate = gates.slice_last_dim(0, self.hidden_size, 2)
            forget_gate = gates.slice_last_dim(self.hidden_size,
                                               self.hidden_size, 2)
            cell_gate = gates.slice_last_dim(2 * self.hidden_size,
                                             self.hidden_size, 2)
            output_gate = gates.slice_last_dim(3 * self.hidden_size,
                                               self.hidden_size, 2)
            cell = (forget_gate.sigmoid() * previous_cell +
                    input_gate.sigmoid() * cell_gate.tanh())
            hidden = output_gate.sigmoid() * cell.tanh()
            present_hidden.append(hidden.unsqueeze(0, hidden.rank))
            present_cell.append(cell.unsqueeze(0, cell.rank))
        return (hidden, F.concatenate(present_hidden,
                                      0), F.concatenate(present_cell, 0))


class NemotronPredictionNetwork(Module):
    """RNN-T embedding, prediction LSTM, and decoder projection."""

    def __init__(self, ctx, hidden_size: int, num_layers: int) -> None:
        super().__init__(ctx, "decoder")
        self.hidden_size = hidden_size
        self.embedding = Embedding(ctx, self.key("embedding"))
        self.lstm = NemotronLSTMStep(ctx, self.key("lstm"), hidden_size,
                                     num_layers)
        self.projector = Linear(ctx,
                                self.key("decoder_projector"),
                                rank=2,
                                tensor_parallel=False)

    def forward(self, input_ids, hidden_state, cell_state):
        hidden = self.embedding(input_ids).reshape((0, self.hidden_size))
        hidden, present_hidden, present_cell = self.lstm(
            hidden, hidden_state, cell_state)
        return self.projector(hidden), present_hidden, present_cell


class Nemotron3_5AsrRNNTStep(NetworkModule):
    """One static-shape greedy RNN-T decoder invocation."""

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        config = ctx.bundle.root
        self.hidden_size = int(config["decoder_hidden_size"])
        self.vocab_size = int(config["vocab_size"])
        self.num_layers = int(config.get("num_decoder_layers", 2))
        self.decoder = NemotronPredictionNetwork(ctx, self.hidden_size,
                                                 self.num_layers)
        self.head = Linear(ctx, "joint.head", rank=2, tensor_parallel=False)

    def input_tensors(self):
        state_shape = (self.num_layers, 1, self.hidden_size)
        return {
            "decoder_input_ids":
            self.add_input("decoder_input_ids", trt.int64, (1, 1)),
            "hidden_state":
            self.add_input("hidden_state", trt.float16, state_shape),
            "cell_state":
            self.add_input("cell_state", trt.float16, state_shape),
            "encoder_frame":
            self.add_input("encoder_frame", trt.float16,
                           (1, self.hidden_size)),
        }

    def forward(self, decoder_input_ids, hidden_state, cell_state,
                encoder_frame):
        decoder, present_hidden, present_cell = self.decoder(
            decoder_input_ids, hidden_state, cell_state)
        logits = self.head((encoder_frame + decoder).relu())
        return {
            "logits": logits,
            "present_hidden_state": present_hidden,
            "present_cell_state": present_cell,
        }
