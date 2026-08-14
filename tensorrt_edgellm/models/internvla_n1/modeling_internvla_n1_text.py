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
"""
InternVLA-N1-DualVLN System-2 backbone.

The text decoder is a stock Qwen2.5-VL transformer, so the only thing this adds
is the bridge to System 1.

InternVLA-N1 appends ``n_query`` learned trajectory queries to the prompt and
takes the hidden states at those positions as the entire conditioning signal for
the System-1 diffusion head -- token output is not used for navigation at all.
Those hidden states go through the final norm and a two-layer projector
(``cond_projector``) to become ``z_latents``.

Both steps are folded into the graph here, so the engine emits ``z_latents``
directly and the runtime does not have to carry projector weights or re-derive
the norm ordering. ``emit_hidden_states`` supplies the *pre-norm* residual (see
:class:`Transformer`), which is why the norm is applied explicitly below.
"""
from typing import Tuple

import torch.nn as nn

from ..default import modeling_default
from ..default.modeling_default import CausalLM, OnnxSpec
from ..linear import make_linear

#: ``LatentEmbSize`` in the reference implementation. InternVLA-N1 checkpoints do
#: not record it in ``config.json``, so it is a constant here and is validated
#: against the loaded ``cond_projector`` weights.
DEFAULT_LATENT_DIM = 768


class InternVLAN1LanguageModel(CausalLM):
    """Qwen2.5-VL decoder plus the System-2 -> System-1 bridge.

    ``cond_projector`` is attached to ``self.model`` rather than to this wrapper
    so that the checkpoint keys ``model.cond_projector.*`` resolve without a
    remap.
    """

    emit_hidden_states = True

    def __init__(self, config) -> None:
        super().__init__(config)
        n_query = int(getattr(config, "n_query", 0) or 0)
        if n_query <= 0:
            raise ValueError(
                "InternVLA-N1 requires n_query > 0 in config.json; got "
                f"{n_query!r}. Without it there is no way to know which "
                "positions carry the trajectory queries.")
        self.n_query = n_query
        latent_dim = int(getattr(config, "latent_dim", 0)
                         or DEFAULT_LATENT_DIM)
        self.latent_dim = latent_dim
        # Built with ``make_linear`` rather than ``nn.Linear`` so the projector
        # picks up the backbone's dtype and quantization policy; a raw
        # ``nn.Linear`` lands in fp32 and the forward dies on Half-vs-Float.
        # ``module_name`` routes it through ``config.module_quant_type``, which
        # keeps the bridge in fp16 unless a layer override says otherwise.
        self.model.cond_projector = nn.Sequential(
            make_linear(config,
                        config.hidden_size,
                        latent_dim,
                        bias=True,
                        module_name="cond_projector.0"),
            nn.GELU(approximate="tanh"),
            make_linear(config,
                        latent_dim,
                        latent_dim,
                        bias=True,
                        module_name="cond_projector.2"),
        )

    def onnx_export_spec(self) -> OnnxSpec:
        """Trace with a real sequence.

        The default dummy sequence is one token (``_SEQ_LEN``), which would make
        the ``[-n_query:]`` slice below a no-op at trace time and risks it being
        specialized away. Raising the constant around the parent call keeps every
        derived dummy (rope table, context lengths, last-token ids) consistent,
        which rebuilding ``spec.args`` by hand would not.

        The emitted tensor keeps the name ``hidden_states``. Renaming it to
        ``z_latents`` reads better but breaks the runtime: ``engineExecutor``
        requires every engine I/O tensor to be registry-bound, and the registry
        knows ``binding_names::kOutputHiddenStates``. The role is unchanged --
        this is still the tensor the next stage consumes -- so the name stays and
        the contents are what differ.
        """
        saved = modeling_default._SEQ_LEN
        modeling_default._SEQ_LEN = max(saved, self.n_query + 1)
        try:
            spec = super().onnx_export_spec()
        finally:
            modeling_default._SEQ_LEN = saved
        return spec

    def forward(self, *args, **kwargs) -> Tuple:
        logits, hidden_states, present_key_values = super().forward(
            *args, **kwargs)
        # ``hidden_states`` is the full-sequence pre-norm residual. The bridge
        # reads only the trailing trajectory-query positions, so slice before
        # projecting -- it keeps the output at [batch, n_query, latent_dim]
        # instead of the whole sequence.
        traj = hidden_states[:, -self.n_query:, :]
        z_latents = self.model.cond_projector(self.model.norm(traj))
        return logits, z_latents, present_key_values


__all__ = ["InternVLAN1LanguageModel", "DEFAULT_LATENT_DIM"]
