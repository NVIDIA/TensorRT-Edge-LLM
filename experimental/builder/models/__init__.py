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
"""Model graph dispatch for the direct TensorRT builder."""


def build_model(net, bundle, cfg, weights, args) -> None:
    """Construct the registered HF-style component and emit its network."""
    from ..ops import BuildContext, BuildOptions, NetworkModule
    from . import registry

    options = BuildOptions(
        backend=getattr(args, "backend", "edgellm"),
        dense_quant=args.dense_quant,
        int4_gemm_plugin_version=args.int4_gemm_plugin_version,
        sm12x=args.sm12x,
        sm110=args.sm110,
        max_lora_rank=args.max_lora_rank,
    )
    context = BuildContext(net=net,
                           cfg=cfg,
                           weights=weights,
                           options=options,
                           bundle=bundle,
                           args=args)
    definition = registry.definition_for(bundle.root_model_type,
                                         args.resolved_component,
                                         args.spec_type,
                                         args.resolved_spec_role,
                                         args.dflash_version)
    model = definition.load().from_config(context)
    if not isinstance(model, NetworkModule):
        raise TypeError(f"{type(model).__name__} must inherit NetworkModule")
    model.build()
