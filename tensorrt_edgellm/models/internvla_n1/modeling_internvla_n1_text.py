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

The engine emits the **full-sequence, model-width hidden states** and the norm and
projector run on the host. Folding them into the graph was tried and reverted:
the runtime sizes its hidden-states buffer ``{batch, maxInputLen, hiddenSize}``
and reshapes it per request, so a graph emitting ``[batch, n_query, 768]``
violates that contract. It does not fail loudly -- ``getBaseModelHiddenStates``
returns a plausible ``[1, 45, 3584]`` buffer either way -- which is exactly why
the contract has to be respected rather than worked around.

``emit_hidden_states`` supplies the *pre-norm* residual (see
:class:`Transformer`), so a consumer applies ``model.norm`` and then
``cond_projector`` itself. Those weights are exported alongside the engine.
"""

from ..default import modeling_default
from ..default.modeling_default import CausalLM, OnnxSpec

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
        # cond_projector is deliberately *not* a module here. It runs on the host,
        # so putting it in the graph would only change the engine's output shape
        # and break the runtime's hidden-states contract. Its weights ship as a
        # sidecar next to the engine.

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

    # No forward override: the parent already returns the full-sequence hidden states
    # when emit_hidden_states is set, and that is precisely what the runtime expects.


__all__ = ["InternVLAN1LanguageModel", "DEFAULT_LATENT_DIM"]
