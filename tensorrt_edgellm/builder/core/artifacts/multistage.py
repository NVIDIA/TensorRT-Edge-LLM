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
"""Emits the stage graph a built engine directory can be run with.

The graph is a property of the model, not of the caller, so it is derived here
rather than hand-written per deployment. Families whose components form more
than a single backbone provide a ``multistage_topology`` hook in their
configuration module; everything else falls back to the one-stage default.
A topology that returns None logs why the built components form no graph.
"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from .. import contracts

logger = logging.getLogger(__name__)

CONFIG_NAME = "multistage_config.json"
SCHEMA_VERSION = 1

#: Every runner declares its own kind, and the loader rejects a stage whose kind
#: disagrees. Keeping the mapping here rather than at each call site means a new
#: runner is one entry, not a value repeated across family modules.
RUNNER_KINDS = {
    "llm.backbone": "autoregressive",
    "qwen3_omni.talker": "autoregressive",
    "qwen3_omni.code2wav": "iterative",
    "minimax_h3.text_encoder": "engine",
    "minimax_h3.denoiser": "iterative",
    "minimax_h3.audio_decoder": "engine",
    "minimax_h3.video_decoder": "iterative",
}

#: Inputs a runner cannot proceed without: the backbone rejects a request whose
#: prompt is unbound, and the vocoder blocks until codes are bound.
REQUIRED_INPUTS = {
    "qwen3_omni.code2wav": ("codes", ),
    "llm.backbone": ("prompt", ),
    "minimax_h3.text_encoder": ("prompt", ),
    "minimax_h3.denoiser": ("text_hidden", ),
    "minimax_h3.audio_decoder": ("latents", ),
    "minimax_h3.video_decoder": ("latents", ),
}

#: Runners that assemble a token sequence from several optional inputs and
#: reject the result if it came out empty.
ONE_OF_TOKEN_INPUTS = {"qwen3_omni.talker": ("prompt_ids", "text_ids")}

#: Port kinds a caller can collect.
TERMINAL_KINDS = ("text", "audio", "image")

#: Runners whose `preallocate()` rejects a non-positive `max_seq`.
NEEDS_SEQUENCE_BUDGET = ("llm.backbone", "qwen3_omni.talker")

#: What the default graph's one `llm.backbone` stage runs: the decoder and the
#: encoders it loads through `multimodal_artifact`.
DEFAULT_GRAPH_COMPONENTS = frozenset({
    contracts.Component.LLM,
    contracts.Component.VISUAL,
    contracts.Component.AUDIO,
})


def artifact_of(component: contracts.Component) -> str:
    """Config-root-relative directory holding one component's engine.

    ComponentSpec is the single source of truth for the layout; an empty
    ``engine_subdir`` means the component lands in the engine directory itself,
    which the schema spells "." rather than "".
    """
    return contracts.component_spec(component).engine_subdir or "."


def parsed_request_input() -> Dict[str, Any]:
    """Backbone edge for a request a caller already parsed, media and options included.

    ``multistage_inference`` fills it from llm_inference's request parser, so a
    request means what it means there.
    """
    return {
        "name": "request",
        "from": "request:llm_request",
        "on_missing": "skip"
    }


def media_inputs(
        components: Iterable[contracts.Component]) -> List[Dict[str, Any]]:
    """Backbone request edges for the encoders this build produced.

    An edge the deployment cannot serve would accept the request and then drop
    it; no edge is what lets admission refuse it.
    """
    present = frozenset(components)
    inputs: List[Dict[str, Any]] = []
    if contracts.Component.AUDIO in present:
        inputs.append({
            "name": "audio",
            "from": "request:audio",
            "on_missing": "skip"
        })
    if contracts.Component.VISUAL in present:
        inputs.extend([{
            "name": "image",
            "from": "request:image",
            "on_missing": "skip"
        }, {
            "name": "vision",
            "from": "request:vision",
            "on_missing": "skip"
        }])
    return inputs


def _default_topology(
        components: Iterable[contracts.Component]) -> Optional[Dict[str, Any]]:
    """One backbone with a text terminal.

    Returns None when the build has no `llm` component. A diffusion backbone
    lands in the same directory under a different engine name and is not an
    `llm.backbone`, so describing it as one produces a graph that loads and
    then fails to find its engine. Also returns None when the build has a
    component other than an encoder, such as an action expert or an image
    generator: the backbone alone would run and drop what it produces.
    """
    present = frozenset(components)
    if contracts.Component.LLM not in present:
        logger.info(
            "No %s: the default graph is one llm.backbone stage and "
            "this build has no llm component", CONFIG_NAME)
        return None
    other = present - DEFAULT_GRAPH_COMPONENTS
    if other:
        logger.info(
            "No %s: the default graph is one llm.backbone stage and "
            "cannot run %s", CONFIG_NAME,
            ", ".join(sorted(component.value for component in other)))
        return None
    stage: Dict[str, Any] = {
        "id":
        "llm",
        "runner":
        "llm.backbone",
        "artifact":
        artifact_of(contracts.Component.LLM),
        "inputs": [{
            "name": "prompt",
            "from": "request:messages",
            "on_missing": "fail"
        }, {
            "name": "sampling",
            "from": "request:sampling",
            "on_missing": "skip"
        },
                   parsed_request_input()],
        "emits": [{
            "id": "text",
            "kind": "text",
            "transport": "chunk_stream",
            "terminal": True
        }],
    }
    media = media_inputs(present)
    if media:
        stage["inputs"].extend(media)
        # The runtime appends /audio and /visual itself, so this names their parent.
        stage["config"] = {"multimodal_artifact": "."}
    return {"terminals": ["llm:text"], "stages": [stage]}


def _topology_for(bundle, components, args) -> Optional[Dict[str, Any]]:
    from ...models import registry as model_registry

    configuration = model_registry.configuration_module_for(
        bundle.root_model_type)
    hook = getattr(configuration, "multistage_topology", None)
    if hook is None:
        return _default_topology(components)
    return hook(bundle, components, args)


def speculative_text_graph(
        topology: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The backbone alone, with its text terminal, for a speculative build.

    A speculative runtime allocates no hidden-state output, so nothing behind
    the backbone can be fed; llm_inference cannot pair --specDecode with audio
    output either. The empty ``spec_decode`` marks the deployment; the drafting
    parameters come from the engine at run time.
    """
    backbones = [
        stage for stage in topology.get("stages") or []
        if stage.get("runner") == "llm.backbone"
    ]
    if len(backbones) != 1:
        return None
    stage = backbones[0]
    stage["emits"] = [
        port for port in stage.get("emits", []) if port.get("terminal")
    ]
    if not stage["emits"]:
        return None
    config = stage.setdefault("config", {})
    config.pop("accept_hidden_layer", None)
    config["spec_decode"] = {}
    return {
        "terminals":
        [f"{stage['id']}:{port['id']}" for port in stage["emits"]],
        "stages": [stage],
    }


def _apply_runner_kinds(topology: Dict[str, Any]) -> None:
    for stage in topology.get("stages") or []:
        kind = RUNNER_KINDS.get(stage.get("runner"))
        if kind is not None:
            stage["kind"] = kind


def _apply_sequence_budget(topology: Dict[str, Any], args) -> None:
    """Give each NEEDS_SEQUENCE_BUDGET stage the KV budget its engine was built for.

    `max_seq` defaults to 0 in the schema, and those runners reject that at
    `preallocate()`, so leaving it out produces a config that parses and
    validates and then refuses to load.
    """
    budget = int(getattr(args, "max_kv_cache_capacity", 0) or 0)
    if budget <= 0:
        return
    for stage in topology.get("stages") or []:
        if stage.get("runner") in NEEDS_SEQUENCE_BUDGET:
            stage["max_seq"] = budget


def _apply_batch_budget(topology: Dict[str, Any], args) -> None:
    """Tell every stage how wide the engines it drives were built.

    `max_batch` defaults to 1 in the schema, so a bundle built with
    `--max-batch-size 4` otherwise describes a topology no request can batch on:
    the engines carry the dimension and the config hides it. A stage whose
    model hook already chose a width keeps it.
    """
    width = int(getattr(args, "max_batch_size", 0) or 0)
    if width <= 1:
        return
    for stage in topology.get("stages") or []:
        stage.setdefault("max_batch", width)


def _validate(topology: Dict[str, Any]) -> List[str]:
    """Catch, before anything is written, what the C++ loader or a runner would reject.

    Cheap structural checks only; multiStage::validate() is the authority.
    """
    problems: List[str] = []
    stages = topology.get("stages") or []
    ids = [stage.get("id") for stage in stages]
    if len(ids) != len(set(ids)):
        problems.append("duplicate stage ids")
    known = {stage_id for stage_id in ids if stage_id}
    for stage in stages:
        for name in REQUIRED_INPUTS.get(stage.get("runner"), ()):
            if not any(
                    edge.get("name") == name
                    for edge in stage.get("inputs", [])):
                problems.append(
                    f"stage '{stage.get('id')}' has no '{name}' input, which "
                    f"{stage.get('runner')} reads unconditionally")
        if stage.get("runner") in ONE_OF_TOKEN_INPUTS and not any(
                edge.get("name") in ONE_OF_TOKEN_INPUTS[stage["runner"]]
                for edge in stage.get("inputs", [])):
            problems.append(f"stage '{stage.get('id')}' has no token source; "
                            f"{stage['runner']} needs one of "
                            f"{sorted(ONE_OF_TOKEN_INPUTS[stage['runner']])}")
        if stage.get("runner") in NEEDS_SEQUENCE_BUDGET and int(
                stage.get("max_seq", 0)) <= 0:
            problems.append(
                f"stage '{stage.get('id')}' has no positive max_seq")
        for port in stage.get("emits", []):
            if port.get("terminal") and port.get("kind") not in TERMINAL_KINDS:
                problems.append(f"stage '{stage.get('id')}' marks a "
                                f"{port.get('kind')} port terminal")
        for edge in stage.get("inputs", []):
            source = str(edge.get("from", ""))
            if source.startswith("request:"):
                continue
            producer = source.split(":", 1)[0]
            if producer not in known:
                problems.append(
                    f"stage '{stage.get('id')}' reads unknown producer '{producer}'"
                )
    terminals = set(topology.get("terminals") or [])
    flagged = {
        f"{stage.get('id')}:{port.get('id')}"
        for stage in stages
        for port in stage.get("emits", []) if port.get("terminal")
    }
    if not terminals:
        problems.append("no terminals")
    elif terminals != flagged:
        problems.append(f"terminals {sorted(terminals)} differ from the "
                        f"ports flagged terminal {sorted(flagged)}")
    return problems


def _covers_default(bundle, components) -> bool:
    """Whether *components* include everything ``--components all`` builds.

    ``all`` leaves out the policy-generation engines, which no stage runs, so a
    build without them is still a full one.
    """
    return frozenset(components) >= (bundle.components -
                                     contracts.POLICY_COMPONENTS)


def drop_stale_config(bundle, components, args) -> None:
    """Remove the graph a full build writes no replacement for.

    It describes the engines this build just replaced. A build with fewer
    components than ``--components all`` keeps it, since the components it did
    not rebuild are still there.
    """
    if not _covers_default(bundle, components):
        return
    path = os.path.join(args.engine_dir, CONFIG_NAME)
    if os.path.exists(path):
        os.remove(path)
        logger.warning(
            "Removed %s: this build writes no graph, and the old one "
            "describes the engines it replaced", path)


def _fit_kept_graph(path: str, engine_dir: str) -> Optional[str]:
    """Lower a kept graph's budgets to the engines under *engine_dir*.

    Each stage's ``max_batch`` and ``max_seq`` are derived as ``--graph-only``
    derives them and only ever lowered: a narrower kept value may be a hook's
    cap, such as the chunked talker's, or a hand edit. Returns the path when
    the graph was rewritten.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning("Keeping %s as is: %s", path, error)
        return None
    stages = [
        stage for stage in document.get("stages") or []
        if stage.get("artifact")
    ]
    built = {
        "stages": [{
            key: value
            for key, value in stage.items()
            if key not in ("max_batch", "max_seq")
        } for stage in stages]
    }
    _apply_engine_budgets(built, engine_dir)
    lowered = []
    for stage, limit in zip(stages, built["stages"]):
        for key, unset in (("max_batch", 1), ("max_seq", 0)):
            ceiling = int(limit.get(key, unset))
            if 0 < ceiling < int(stage.get(key, unset)):
                stage[key] = ceiling
                lowered.append(f"{stage['id']}.{key}={ceiling}")
    if not lowered:
        logger.info(
            "Keeping %s: a partial --components build does not replace "
            "the graph", path)
        return None
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    logger.warning("Lowered %s to the engines it names: %s", path,
                   ", ".join(lowered))
    return path


def write_multistage_config(bundle,
                            components,
                            args,
                            speculative: bool = False) -> Optional[str]:
    """Write ``multistage_config.json`` next to the engines. Returns its path.

    Returns None when the build produced something the graph cannot describe;
    a missing config is recoverable (hand-write one), a wrong config is not.
    A build with fewer components than ``--components all`` keeps an existing
    config, which may describe components this run did not rebuild, and only
    lowers its budgets to the engines; see ``_fit_kept_graph``. A speculative
    build gets the text-only graph of ``speculative_text_graph``.
    """
    path = os.path.join(args.engine_dir, CONFIG_NAME)
    if not _covers_default(bundle, components) and os.path.exists(path):
        return _fit_kept_graph(path, args.engine_dir)
    topology = _topology_for(bundle, components, args)
    if topology is not None and speculative:
        topology = speculative_text_graph(topology)
        if topology is None:
            logger.info(
                "No %s: a speculative build serves text from one backbone "
                "and this topology has none", CONFIG_NAME)
    if topology is None:
        drop_stale_config(bundle, components, args)
        return None

    _apply_runner_kinds(topology)
    _apply_sequence_budget(topology, args)
    _apply_batch_budget(topology, args)
    problems = _validate(topology)
    if problems:
        logger.warning("Not writing %s: %s", CONFIG_NAME, "; ".join(problems))
        drop_stale_config(bundle, components, args)
        return None

    document = {
        "schema_version": SCHEMA_VERSION,
        "label": bundle.root_model_type,
        **topology,
    }
    if os.path.exists(path):
        logger.info("Replacing %s", path)
    os.makedirs(args.engine_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return path


#: Where the export-and-C++-builder recipes put the backbone, after the direct
#: builder's own place: the Qwen3-Omni guide builds it into thinker/, the
#: exporter's layout into llm/.
BACKBONE_DIRS = (".", "thinker", "llm")
#: Parents of the audio/ and visual/ encoder directories, the same way.
ENCODER_PARENTS = (".", "multimodal")


def locate_engines(bundle, engine_dir: str) -> Dict[contracts.Component, str]:
    """Each checkpoint component with an engine under *engine_dir*, and where.

    Directories are relative to *engine_dir*; the direct builder's layout is
    tried first. A component with no engine is left out.
    """
    located: Dict[contracts.Component, str] = {}
    for component in bundle.components:
        spec = contracts.component_spec(component)
        names = [spec.engine_filename]
        if spec.supports_spec_role:
            names.append(
                os.path.basename(
                    spec.output_path(engine_dir, contracts.SpecRole.BASE)))
        if component == contracts.Component.LLM:
            candidates = BACKBONE_DIRS
        elif component in (contracts.Component.AUDIO,
                           contracts.Component.VISUAL):
            candidates = tuple(
                os.path.normpath(os.path.join(parent, spec.engine_subdir))
                for parent in ENCODER_PARENTS)
        else:
            candidates = (spec.engine_subdir or ".", )
        for candidate in candidates:
            if any(
                    os.path.isfile(os.path.join(engine_dir, candidate, name))
                    for name in names):
                located[component] = candidate
                break
    return located


def _relocate(topology: Dict[str, Any], located: Dict[contracts.Component,
                                                      str]) -> None:
    """Point a topology laid out for the direct builder at *located*.

    Only the backbone and the encoders move between layouts; every other
    component keeps its subdirectory, so the paths between them hold.
    """
    direct = {artifact_of(component): component for component in located}
    backbone = located.get(contracts.Component.LLM)
    for stage in topology.get("stages") or []:
        component = direct.get(stage.get("artifact"))
        if component is None:
            continue
        artifact = located[component]
        stage["artifact"] = artifact
        config = stage.get("config") or {}
        if "multimodal_artifact" in config:
            encoders = [
                located[other] for other in (contracts.Component.AUDIO,
                                             contracts.Component.VISUAL)
                if other in located
            ]
            parent = os.path.dirname(encoders[0]) or "."
            config["multimodal_artifact"] = os.path.relpath(parent, artifact)
        if (stage.get("runner") == "qwen3_omni.talker" and backbone is not None
                and os.path.relpath(backbone, artifact) != os.pardir):
            # The talker reads the backbone's embedding table and tokenizer from
            # its own parent unless told where else to look.
            config["tokenizer_dir"] = os.path.relpath(backbone, artifact)


def _builder_config(engine_dir: str, artifact: str) -> Dict[str, Any]:
    """What the engine in *artifact* was built for; a speculative one reports its base."""
    for name in ("config.json", "base_config.json"):
        path = os.path.join(engine_dir, artifact, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle).get("builder_config") or {}
    return {}


def _engine_width(engine_dir: str, stage: Dict[str, Any],
                  own: Dict[str, Any]) -> int:
    """How many requests the stage's engines serve at once.

    The talker runs its code predictor over the whole cohort, so the narrower
    of the two engines bounds it.
    """
    width = int(own.get("max_batch_size", 1))
    predictor = (stage.get("config") or {}).get("code_predictor_artifact")
    if predictor:
        built = _builder_config(engine_dir,
                                os.path.join(stage["artifact"], predictor))
        width = min(width, int(built.get("max_batch_size", width)))
    return width


def _apply_engine_budgets(topology: Dict[str, Any], engine_dir: str) -> None:
    """Give each stage the sequence budget and batch width its engine was built with.

    The talker holds the backbone's whole sequence, so a backbone's KV budget
    bounds both. A stage that steps its members in turn takes any cohort the
    engines in front of it can.
    """
    stages = topology.get("stages") or []
    budgets = {
        stage["id"]: _builder_config(engine_dir, stage["artifact"])
        for stage in stages
    }
    backbone_kv = next(
        (int(budgets[stage["id"]].get("max_kv_cache_capacity", 0))
         for stage in stages if stage.get("runner") == "llm.backbone"), 0)
    widths = {
        stage["id"]: _engine_width(engine_dir, stage, budgets[stage["id"]])
        for stage in stages if stage.get("runner") in NEEDS_SEQUENCE_BUDGET
    }
    widest = max(widths.values(), default=1)
    for stage in stages:
        own = budgets[stage["id"]]
        if stage["id"] in widths:
            stage["max_seq"] = backbone_kv or int(
                own.get("max_kv_cache_capacity", 0))
            width = widths[stage["id"]]
        else:
            width = widest
        if width > 1:
            stage.setdefault("max_batch", width)


def write_graph_for_engines(bundle, engine_dir: str) -> Optional[str]:
    """Write ``multistage_config.json`` for engines another builder produced.

    ``tensorrt-edgellm-export`` and the C++ builders write no graph. The topology
    follows from the checkpoint as it does for a build here; each engine is
    looked for where the documented recipes put it, and its own builder config
    supplies the budgets. Returns None, after logging why, when the engines form
    no graph.
    """
    located = locate_engines(bundle, engine_dir)
    if not located:
        raise ValueError(f"no {bundle.root_model_type} engine under "
                         f"{engine_dir}")
    components = tuple(located)
    topology = _topology_for(bundle, components, None)
    backbone = located.get(contracts.Component.LLM)
    speculative = backbone is not None and os.path.isfile(
        os.path.join(
            engine_dir, backbone,
            os.path.basename(
                contracts.component_spec(contracts.Component.LLM).output_path(
                    engine_dir, contracts.SpecRole.BASE))))
    if topology is not None and speculative:
        topology = speculative_text_graph(topology)
    if topology is None:
        return None
    _relocate(topology, located)
    _apply_runner_kinds(topology)
    _apply_engine_budgets(topology, engine_dir)
    problems = _validate(topology)
    if problems:
        logger.warning("Not writing %s: %s", CONFIG_NAME, "; ".join(problems))
        return None
    path = os.path.join(engine_dir, CONFIG_NAME)
    if os.path.exists(path):
        logger.info("Replacing %s", path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "label": bundle.root_model_type,
                **topology
            },
            handle,
            indent=2)
        handle.write("\n")
    return path
