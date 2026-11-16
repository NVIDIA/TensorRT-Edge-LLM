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
"""Stage graphs the builder writes.

The tables pin what each family's graph holds and when a build writes, keeps,
lowers or removes one. ``test_emitted_graphs_load`` checks one graph per
topology branch with ``multistage_inference --validate-config``, which runs the
C++ loader against the runners linked into that binary without an engine.
"""

import json
import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from tensorrt_edgellm.builder import cli
from tensorrt_edgellm.builder.core import contracts
from tensorrt_edgellm.builder.core.artifacts import multistage

_OMNI_ROOT = {"talker_config": {"accept_hidden_layer": 18}}

_SPEECH_ONLY = frozenset({
    contracts.Component.TALKER,
    contracts.Component.CODE_PREDICTOR,
    contracts.Component.CODE2WAV,
})


class _Bundle:

    def __init__(self, root_model_type, root):
        self.root_model_type = root_model_type
        self.root = root

    @property
    def components(self):
        return contracts.available_components(self.root_model_type)


def _emit(tmp_path,
          root_model_type,
          root,
          components=None,
          max_batch_size=1,
          max_kv_cache_capacity=4096):
    """The graph a build of *components*, all of the checkpoint's by default, writes."""
    bundle = _Bundle(root_model_type, root)
    args = SimpleNamespace(engine_dir=str(tmp_path),
                           tp_size=1,
                           spec_type="none",
                           max_kv_cache_capacity=max_kv_cache_capacity,
                           max_batch_size=max_batch_size)
    path = multistage.write_multistage_config(
        bundle, bundle.components if components is None else components, args)
    return None if path is None else json.load(open(path, encoding="utf-8"))


def _graph(engine_dir):
    return json.load(
        open(engine_dir / multistage.CONFIG_NAME, encoding="utf-8"))


def _stage(document, stage_id):
    return next(stage for stage in document["stages"]
                if stage["id"] == stage_id)


def _request_edges(document, stage_id):
    return {edge["from"] for edge in _stage(document, stage_id)["inputs"]}


def _lay_out(root, engines):
    """Empty engine files and their builder configs, where a recipe put them."""
    for relative, builder in engines.items():
        directory = os.path.join(str(root), os.path.dirname(relative))
        os.makedirs(directory, exist_ok=True)
        open(os.path.join(str(root), relative), "wb").close()
        if builder is not None:
            name = ("base_config.json" if relative.endswith("spec_base.engine")
                    else "config.json")
            with open(os.path.join(directory, name), "w",
                      encoding="utf-8") as handle:
                json.dump({"builder_config": builder}, handle)


def _graph_for(root, root_model_type, checkpoint_root):
    path = multistage.write_graph_for_engines(
        _Bundle(root_model_type, checkpoint_root), str(root))
    return None if path is None else json.load(open(path, encoding="utf-8"))


@pytest.fixture
def multistage_inference(executable_files):
    """The built multistage_inference; without one, a local run skips and CI fails."""
    binary = executable_files["multistage_inference"]
    if not os.path.isfile(binary):
        reason = f"{binary} is not built"
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    return binary


@pytest.mark.parametrize("root_model_type,root,components", [
    pytest.param("qwen3", {}, None, id="qwen3"),
    pytest.param("qwen3_vl", {}, None, id="qwen3_vl"),
    pytest.param("qwen3_omni", _OMNI_ROOT, None, id="qwen3_omni"),
    pytest.param("qwen3_omni_next",
                 _OMNI_ROOT,
                 _SPEECH_ONLY,
                 id="qwen3_omni_next-speech_only"),
    pytest.param("qwen3_tts", {}, None, id="qwen3_tts-clone_encoders"),
    pytest.param("minimax_h3", {}, None, id="minimax_h3"),
])
def test_emitted_graphs_load(tmp_path, multistage_inference, root_model_type,
                             root, components):
    """Each topology branch writes a graph the loader accepts for the runners the binary links."""
    assert _emit(tmp_path, root_model_type, root, components,
                 max_batch_size=4) is not None
    result = subprocess.run(
        [multistage_inference, "--config",
         str(tmp_path), "--validate-config"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False)
    assert result.returncode == 0, result.stderr


_TEXT_EDGES = {"request:messages", "request:sampling", "request:llm_request"}


@pytest.mark.parametrize("root_model_type,root,components,backbone,media", [
    pytest.param("qwen3", {}, None, "llm", set(), id="qwen3"),
    pytest.param(
        "qwen3_asr", {}, None, "llm", {"request:audio"}, id="qwen3_asr"),
    pytest.param("qwen3_vl", {},
                 None,
                 "llm", {"request:image", "request:vision"},
                 id="qwen3_vl"),
    pytest.param("qwen3_omni",
                 _OMNI_ROOT, {contracts.Component.LLM},
                 "thinker",
                 set(),
                 id="qwen3_omni-llm"),
    pytest.param("qwen3_omni",
                 _OMNI_ROOT,
                 {contracts.Component.LLM, contracts.Component.AUDIO},
                 "thinker", {"request:audio"},
                 id="qwen3_omni-llm_audio"),
])
def test_backbone_request_edges(tmp_path, root_model_type, root, components,
                                backbone, media):
    """A backbone reads the prompt, sampling and parsed request, plus a media edge per encoder built."""
    # An edge for media no encoder was built for would admit the media and drop it;
    # without the edge, admission refuses it.
    document = _emit(tmp_path, root_model_type, root, components)
    assert _request_edges(document, backbone) == _TEXT_EDGES | media
    assert ("multimodal_artifact"
            in _stage(document, backbone).get("config", {})) == bool(media)


_TALKER_INPUTS = {"prompt_ids", "voice", "sampling", "speaker_codes", "style"}


@pytest.mark.parametrize(
    "root_model_type,root,components,max_batch_size,stages,prefill_builder,"
    "talker_inputs", [
        pytest.param("qwen3_omni",
                     _OMNI_ROOT,
                     None,
                     4, [("thinker", 4), ("talker", 4), ("code2wav", 4)],
                     "segments",
                     _TALKER_INPUTS | {"hidden", "text_ids"},
                     id="qwen3_omni"),
        pytest.param("qwen3_omni_next",
                     _OMNI_ROOT,
                     None,
                     4, [("thinker", 4), ("talker", 1), ("code2wav", 4)],
                     "chunked",
                     _TALKER_INPUTS | {"hidden", "text_ids"},
                     id="qwen3_omni_next"),
        pytest.param("qwen3_omni_next",
                     _OMNI_ROOT,
                     _SPEECH_ONLY,
                     4, [("talker", 1), ("code2wav", 4)],
                     "chunked",
                     _TALKER_INPUTS,
                     id="qwen3_omni_next-speech_only"),
        pytest.param("qwen3_tts", {},
                     None,
                     1, [("talker", None), ("code2wav", None)],
                     "tts",
                     _TALKER_INPUTS | {"ref_audio"},
                     id="qwen3_tts"),
    ])
def test_speech_stages(tmp_path, root_model_type, root, components,
                       max_batch_size, stages, prefill_builder, talker_inputs):
    """A speech graph has its family's stages, widths, talker prefill layout and talker inputs."""
    # The chunked talker re-prefills one member's own KV, so it stays one request wide; a
    # talker with no thinker in front reads no hidden states.
    document = _emit(tmp_path,
                     root_model_type,
                     root,
                     components,
                     max_batch_size=max_batch_size)
    assert [(stage["id"], stage.get("max_batch"))
            for stage in document["stages"]] == stages
    talker = _stage(document, "talker")
    assert talker["config"]["prefill_builder"] == prefill_builder
    assert {edge["name"] for edge in talker["inputs"]} == talker_inputs


def test_minimax_h3_stages(tmp_path):
    """MiniMax-H3 conditions its denoiser on the text encoder and decodes audio and video from its latents."""
    document = _emit(tmp_path, "minimax_h3", {})
    assert [(stage["id"], stage["artifact"], stage["kind"])
            for stage in document["stages"]] == [
                ("text_encoder", "text_encoder", "engine"),
                ("denoiser", "gen", "iterative"),
                ("audio_decoder", "audio_decoder", "engine"),
                ("video_decoder", "vae_decoder", "iterative"),
            ]
    assert document["terminals"] == [
        "audio_decoder:audio", "video_decoder:video"
    ]
    assert _request_edges(document, "denoiser") == {
        "text_encoder:text_hidden", "request:style", "request:sampling"
    }


def test_a_minimax_h3_build_without_the_video_decoder_serves_audio(tmp_path):
    """A MiniMax-H3 build without the video VAE still writes a graph, with only the audio terminal."""
    document = _emit(
        tmp_path, "minimax_h3", {},
        contracts.available_components("minimax_h3") -
        {contracts.Component.VAE_DECODER})
    assert document["terminals"] == ["audio_decoder:audio"]


@pytest.mark.parametrize("root_model_type,root,components", [
    pytest.param("diffusion_gemma", {}, None, id="diffusion_gemma"),
    pytest.param("minimax_h3", {},
                 {contracts.Component.TEXT_ENCODER, contracts.Component.GEN},
                 id="minimax_h3-no_decoder"),
    pytest.param("alpamayo_r1", {}, None, id="alpamayo_r1"),
    pytest.param("cosmos3", {}, None, id="cosmos3"),
    pytest.param(
        "qwen3_omni", _OMNI_ROOT, _SPEECH_ONLY, id="qwen3_omni-speech_only"),
    pytest.param("qwen3_omni_next",
                 _OMNI_ROOT, {contracts.Component.TALKER},
                 id="qwen3_omni_next-talker_only"),
])
def test_builds_that_form_no_graph(tmp_path, root_model_type, root,
                                   components):
    """A build gets no graph when a stage would miss what it loads or nothing would run a built component."""
    # dllm.engine is not an llm.backbone; one llm.backbone stage would drop what an action
    # expert or a policy engine produces; the segments talker reads the embedding file only
    # the llm build writes; and a talker needs code2wav behind it.
    assert _emit(tmp_path, root_model_type, root, components) is None


def test_a_build_without_a_kv_budget_writes_no_graph(tmp_path, caplog):
    """A build with no positive KV cache capacity writes no graph and warns about max_seq."""
    # Given a build whose --max-kv-cache-capacity is 0,
    # When the graph is emitted,
    # Then none is written and the warning names max_seq: llm.backbone rejects a
    # non-positive one at preallocate(), so the graph would validate and then refuse to load.
    with caplog.at_level(logging.WARNING):
        assert _emit(tmp_path, "qwen3", {}, max_kv_cache_capacity=0) is None
    assert "no positive max_seq" in caplog.text


def test_a_speculative_build_serves_text_from_the_backbone_alone(tmp_path):
    """A --spec-type build gets a graph of the backbone alone, marked speculative, with no hidden port."""
    # Given a full Qwen3-Omni build with --spec-type mtp,
    # When the graph is emitted,
    # Then it holds only the thinker with its text terminal and an empty spec_decode, and no
    # hidden or prompt-id port: a speculative runtime allocates no hidden-state output.
    bundle = _Bundle("qwen3_omni", _OMNI_ROOT)
    args = SimpleNamespace(engine_dir=str(tmp_path),
                           tp_size=1,
                           spec_type="mtp",
                           max_kv_cache_capacity=4096)
    path = cli._write_multistage_config(args, bundle, bundle.components)
    document = json.load(open(path, encoding="utf-8"))
    assert document["terminals"] == ["thinker:text"]
    assert [stage["id"] for stage in document["stages"]] == ["thinker"]
    thinker = document["stages"][0]
    assert [port["id"] for port in thinker["emits"]] == ["text"]
    assert thinker["config"]["spec_decode"] == {}
    assert "accept_hidden_layer" not in thinker["config"]


@pytest.mark.parametrize("root,tp_size,warnings", [
    pytest.param({"talker_config": {}},
                 1, ("accept_hidden_layer", "Removed"),
                 id="no_accept_layer"),
    pytest.param(_OMNI_ROOT, 2, ("Removed", ), id="tensor_parallel"),
])
def test_a_full_build_that_forms_no_graph_removes_the_stale_one(
        tmp_path, caplog, root, tp_size, warnings):
    """A full build that writes no graph removes the one describing the engines it replaced, and warns why."""
    assert _emit(tmp_path, "qwen3_omni", _OMNI_ROOT) is not None
    bundle = _Bundle("qwen3_omni", root)
    args = SimpleNamespace(engine_dir=str(tmp_path),
                           tp_size=tp_size,
                           spec_type="none",
                           max_kv_cache_capacity=4096)
    with caplog.at_level(logging.WARNING):
        assert cli._write_multistage_config(args, bundle,
                                            bundle.components) is None
    assert not (tmp_path / multistage.CONFIG_NAME).exists()
    for warning in warnings:
        assert warning in caplog.text


def test_a_default_build_without_the_policy_engines_is_a_full_build(tmp_path):
    """A build of what --components all selects replaces the graph, or removes it when it forms none."""
    # Given a cosmos3 checkpoint, whose default build leaves out its policy engines,
    default = contracts.resolve_components("cosmos3", ("all", ),
                                           _Bundle("cosmos3", {}).components)
    # When the default build runs 4 wide and then 8 wide in the same directory,
    for width in (4, 8):
        _emit(tmp_path, "cosmos3", {}, default, max_batch_size=width)
    # Then the graph is the second build's,
    assert _graph(tmp_path)["stages"][0]["max_batch"] == 8
    # and a default build that forms no graph removes it.
    assert _emit(tmp_path, "cosmos3", {}, default,
                 max_kv_cache_capacity=0) is None
    assert not (tmp_path / multistage.CONFIG_NAME).exists()


@pytest.mark.parametrize("components", [
    pytest.param(frozenset({contracts.Component.LLM}), id="llm"),
    pytest.param(frozenset({contracts.Component.TALKER}), id="talker"),
])
def test_a_partial_build_leaves_the_graph_alone(tmp_path, caplog, components):
    """A partial build with no engine narrower than the graph leaves the graph as it is, and says so."""
    full = _emit(tmp_path, "qwen3_omni", _OMNI_ROOT)
    with caplog.at_level(logging.INFO):
        assert _emit(tmp_path, "qwen3_omni", _OMNI_ROOT, components) is None
    assert _graph(tmp_path) == full
    assert "partial --components build" in caplog.text


_BUILT_4_WIDE = {"max_kv_cache_capacity": 4096, "max_batch_size": 4}


def _full_speech_build(tmp_path, root_model_type):
    """Engines built 4 wide with a 4096-token budget, and the graph their full build writes."""
    _lay_out(
        tmp_path, {
            "llm.engine": _BUILT_4_WIDE,
            "talker/llm.engine": _BUILT_4_WIDE,
            "code_predictor/llm.engine": _BUILT_4_WIDE,
            "code2wav/code2wav.engine": None,
        })
    return _emit(tmp_path, root_model_type, _OMNI_ROOT, max_batch_size=4)


def _budgets(document):
    return {
        stage["id"]: (stage.get("max_batch"), stage.get("max_seq"))
        for stage in document["stages"]
    }


def test_a_partial_llm_build_lowers_the_kept_graph(tmp_path):
    """A narrower llm rebuild lowers the kept backbone's width and every sequence budget it bounds."""
    # Given a full qwen3_omni build,
    _full_speech_build(tmp_path, "qwen3_omni")
    # When only the llm is rebuilt 1 wide with a 96-token budget,
    _lay_out(
        tmp_path,
        {"llm.engine": {
            "max_kv_cache_capacity": 96,
            "max_batch_size": 1
        }})
    _emit(tmp_path,
          "qwen3_omni",
          _OMNI_ROOT, {contracts.Component.LLM},
          max_kv_cache_capacity=96)
    # Then the thinker narrows, and the talker, which holds the thinker's sequence, shortens.
    assert _budgets(_graph(tmp_path)) == {
        "thinker": (1, 96),
        "talker": (4, 96),
        "code2wav": (4, None)
    }


def test_a_partial_speech_build_lowers_the_kept_talker(tmp_path):
    """A narrower talker rebuild without the llm lowers the kept talker's width."""
    # Given a full qwen3_omni build,
    _full_speech_build(tmp_path, "qwen3_omni")
    # When only the talker and its code predictor are rebuilt 1 wide,
    narrow = {"max_kv_cache_capacity": 4096, "max_batch_size": 1}
    _lay_out(tmp_path, {
        "talker/llm.engine": narrow,
        "code_predictor/llm.engine": narrow
    })
    _emit(tmp_path, "qwen3_omni", _OMNI_ROOT,
          {contracts.Component.TALKER, contracts.Component.CODE_PREDICTOR})
    # Then the kept graph's talker is 1 wide; the stages around it keep their width.
    assert _budgets(_graph(tmp_path)) == {
        "thinker": (4, 4096),
        "talker": (1, 4096),
        "code2wav": (4, None)
    }


def test_a_partial_build_never_widens_the_kept_graph(tmp_path):
    """A partial rebuild leaves a kept budget narrower than its engine, such as the chunked talker's width of 1."""
    # Given a full qwen3_omni_next build, whose chunked talker the graph keeps 1 wide,
    full = _full_speech_build(tmp_path, "qwen3_omni_next")
    # When only the llm is rebuilt, as wide as before,
    _lay_out(tmp_path, {"llm.engine": _BUILT_4_WIDE})
    # Then the graph is left as it was.
    assert _emit(tmp_path,
                 "qwen3_omni_next",
                 _OMNI_ROOT, {contracts.Component.LLM},
                 max_batch_size=4) is None
    assert _graph(tmp_path) == full
    assert _budgets(full)["talker"] == (1, 4096)


def test_a_graph_for_engines_follows_the_omni_guide_layout(tmp_path):
    """Engines laid out by the Qwen3-Omni guide get a graph naming each directory and budget."""
    # Given engines in thinker/, talker/, code_predictor/, code2wav/ and multimodal/{audio,visual}/,
    # the talker built with a smaller KV budget than the thinker,
    built = {"max_kv_cache_capacity": 2048, "max_batch_size": 2}
    _lay_out(
        tmp_path, {
            "thinker/llm.engine": built,
            "talker/llm.engine": {
                "max_kv_cache_capacity": 1024,
                "max_batch_size": 2
            },
            "code_predictor/llm.engine": {
                "max_kv_cache_capacity": 256,
                "max_batch_size": 2
            },
            "code2wav/code2wav.engine": None,
            "multimodal/audio/audio_encoder.engine": None,
            "multimodal/visual/visual.engine": None,
        })
    # When the graph is written for them,
    document = _graph_for(tmp_path, "qwen3_omni", _OMNI_ROOT)
    # Then each stage points at its own engine, takes its batch width, and holds the
    # thinker's whole sequence.
    thinker, talker, code2wav = document["stages"]
    assert [stage["artifact"] for stage in document["stages"]
            ] == ["thinker", "talker", "code2wav"]
    assert thinker["config"]["multimodal_artifact"] == "../multimodal"
    assert talker["config"]["tokenizer_dir"] == "../thinker"
    assert talker["config"]["code_predictor_artifact"] == "../code_predictor"
    assert [thinker["max_seq"], talker["max_seq"]] == [2048, 2048]
    assert [stage.get("max_batch")
            for stage in document["stages"]] == [2, 2, 2]


@pytest.mark.parametrize("thinker_width,talker_width,predictor_width,expected",
                         [
                             (4, 2, 4, [4, 2, 4]),
                             (2, 4, 1, [2, None, 2]),
                         ])
def test_a_graph_for_engines_sizes_the_talker_by_its_narrower_engine(
        tmp_path, thinker_width, talker_width, predictor_width, expected):
    """The talker stage is no wider than its code predictor engine."""
    # Given Omni-guide engines with the talker and its code predictor built at different
    # widths,
    _lay_out(
        tmp_path, {
            "thinker/llm.engine": {
                "max_kv_cache_capacity": 2048,
                "max_batch_size": thinker_width
            },
            "talker/llm.engine": {
                "max_kv_cache_capacity": 2048,
                "max_batch_size": talker_width
            },
            "code_predictor/llm.engine": {
                "max_kv_cache_capacity": 256,
                "max_batch_size": predictor_width
            },
            "code2wav/code2wav.engine": None,
        })
    # When the graph is written for them,
    document = _graph_for(tmp_path, "qwen3_omni", _OMNI_ROOT)
    # Then the talker takes the narrower of the two, since it runs the code predictor over its
    # whole cohort and a wider claim would load and then fail, and the vocoder takes the widest
    # stage in front of it as narrowed.
    assert [stage.get("max_batch") for stage in document["stages"]] == expected


def test_a_graph_for_engines_in_the_builders_layout_matches_the_build(
        tmp_path):
    """Engines where this builder puts them get the graph its own build writes."""
    # Given engines in the direct builder's layout, built 4 wide with a 4096-token budget,
    built = {"max_kv_cache_capacity": 4096, "max_batch_size": 4}
    _lay_out(
        tmp_path / "engines", {
            "llm.engine": built,
            "talker/llm.engine": built,
            "code_predictor/llm.engine": built,
            "code2wav/code2wav.engine": None,
            "audio/audio_encoder.engine": None,
            "visual/visual.engine": None,
        })
    # When the graph is written for them,
    document = _graph_for(tmp_path / "engines", "qwen3_omni", _OMNI_ROOT)
    # Then it is the graph the same build would have written.
    assert document == _emit(tmp_path / "build",
                             "qwen3_omni",
                             _OMNI_ROOT,
                             max_batch_size=4,
                             max_kv_cache_capacity=4096)


def test_a_graph_for_a_speculative_backbone_serves_its_text(tmp_path):
    """A speculative backbone under llm/ gets the text-only graph, marked speculative."""
    # Given base and draft engines in llm/ beside the speech engines,
    built = {"max_kv_cache_capacity": 4096, "max_batch_size": 4}
    _lay_out(
        tmp_path, {
            "llm/spec_base.engine": built,
            "llm/spec_draft.engine": None,
            "talker/llm.engine": built,
            "code_predictor/llm.engine": built,
            "code2wav/code2wav.engine": None,
        })
    # When the graph is written for them,
    document = _graph_for(tmp_path, "qwen3_omni", _OMNI_ROOT)
    # Then the backbone serves text alone: a speculative runtime emits no hidden states.
    assert document["terminals"] == ["thinker:text"]
    assert [stage["artifact"] for stage in document["stages"]] == ["llm"]
    assert document["stages"][0]["config"]["spec_decode"] == {}


_CLONE_ENCODERS = {
    "clone_encoders/speaker_encoder.engine": None,
    "clone_encoders/speech_tokenizer_encoder.engine": None,
}


@pytest.mark.parametrize("encoders,clone_encoder_artifact", [
    pytest.param(_CLONE_ENCODERS, "../clone_encoders", id="clone_encoders"),
    pytest.param({}, None, id="no_clone_encoders"),
])
def test_a_graph_for_speech_engines_reads_a_reference_voice_with_the_clone_encoders(
        tmp_path, encoders, clone_encoder_artifact):
    """A qwen3_tts layout offers the reference waveform exactly when its clone encoders are there."""
    # Given speech engines without a backbone, with or without clone_encoders/,
    built = {"max_kv_cache_capacity": 4096, "max_batch_size": 1}
    _lay_out(
        tmp_path, {
            "talker/llm.engine": built,
            "code_predictor/llm.engine": built,
            "code2wav/code2wav.engine": None,
            **encoders,
        })
    # When the graph is written for them,
    document = _graph_for(tmp_path, "qwen3_tts", {})
    # Then the talker reads ref_audio only beside the encoders, since without them it could
    # only refuse the request mid-prefill, and takes its own engine's sequence budget.
    talker = _stage(document, "talker")
    assert talker["config"].get(
        "clone_encoder_artifact") == clone_encoder_artifact
    assert ("request:ref_audio"
            in _request_edges(document, "talker")) == (clone_encoder_artifact
                                                       is not None)
    assert talker["max_seq"] == 4096


def test_a_graph_needs_engines_to_describe(tmp_path):
    """A directory holding none of the checkpoint's engines is refused rather than described."""
    # Given an empty directory, When a graph is asked for, Then it is refused by name.
    with pytest.raises(ValueError, match="no qwen3_omni engine"):
        _graph_for(tmp_path, "qwen3_omni", _OMNI_ROOT)
