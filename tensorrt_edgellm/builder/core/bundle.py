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
"""Checkpoint bundle loading with model-owned component selection."""

import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

from . import contracts

LLM_COMPONENTS = frozenset((
    contracts.Component.LLM,
    contracts.Component.DLLM,
    contracts.Component.TALKER,
    contracts.Component.CODE_PREDICTOR,
))


@dataclass(frozen=True)
class BundleConfig:
    """Raw checkpoint configuration and component selection helpers."""

    model_dir: str
    root: Dict[str, Any]
    root_model_type: str

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BundleConfig":
        """Load the checkpoint root configuration."""
        from os.path import isfile, join

        from ..models import registry as model_registry

        if isfile(join(model_dir, "config.json")):
            with open(join(model_dir, "config.json")) as config_file:
                root = json.load(config_file)
        else:
            # A Diffusers pipeline has a model index instead of a root config.
            with open(join(model_dir, "model_index.json")) as index_file:
                pipeline = json.load(index_file).get("_class_name")
            if pipeline not in model_registry.DIFFUSERS_PIPELINES:
                raise ValueError(
                    f"unsupported Diffusers pipeline {pipeline!r}")
            root = {"model_type": model_registry.DIFFUSERS_PIPELINES[pipeline]}
        architectures = root.get("architectures") or []
        if (any("dspark" in str(name).lower() for name in architectures)
                and not root.get("dspark_config")
                and isinstance(root.get("dflash_config"), dict)):
            root = dict(root)
            root["dspark_config"] = dict(root["dflash_config"])
        root_model_type = (root.get("model_type")
                           or root.get("speculators_model_type"))
        if root_model_type is None and root.get("type") == "pi05":
            root_model_type = "pi05"
        if not isinstance(root_model_type, str) or not root_model_type:
            raise ValueError(
                "config.json must define model_type or speculators_model_type")
        configuration = model_registry.configuration_module_for(
            root_model_type)
        prepare_root = getattr(configuration, "prepare_root", None)
        if prepare_root is not None:
            root = prepare_root(model_dir, root)
        return cls(model_dir=model_dir,
                   root=root,
                   root_model_type=root_model_type)

    @property
    def components(self) -> FrozenSet[contracts.Component]:
        """Return engine components present in the checkpoint."""
        registered = contracts.available_components(self.root_model_type)
        from ..models import registry as model_registry
        configuration = model_registry.configuration_module_for(
            self.root_model_type)
        resolve = getattr(configuration, "available_components", None)
        if resolve is None:
            return registered
        available = frozenset(resolve(self.root, registered))
        unexpected = available - registered
        if unexpected:
            names = ", ".join(
                sorted(component.value for component in unexpected))
            raise ValueError(
                f"{self.root_model_type} declared unregistered components: "
                f"{names}")
        return available

    def component_dict(self, component: contracts.Component) -> Dict[str, Any]:
        """Return the configuration dictionary for one component."""
        from ..models import registry as model_registry
        configuration = model_registry.configuration_module_for(
            self.root_model_type)
        selected = configuration.component_config(self.root, component)
        if not isinstance(selected, dict):
            raise TypeError(f"{self.root_model_type} returned a non-dict "
                            f"configuration for {component.value}")
        return selected
