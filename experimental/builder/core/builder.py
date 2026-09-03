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
"""TensorRT engine builder driver.

Loads the Edge-LLM plugin library, builds a strongly typed network from NumPy
weights, installs context and generation optimization profiles, and serializes
the engine.
"""

import ctypes
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorrt as trt

from ..ops.backend import Net
from ..ops.functional.attention import KV_PAGE_SIZE
from . import contracts, quantization, weight_policy
from .bundle import LLM_COMPONENTS, BundleConfig
from .config import DeviceConfig
from .weight_policy import WeightPolicy
from .weights import Weights

logger = logging.getLogger(__name__)


def _active_cuda_compute_capability() -> Tuple[int, int]:
    """Return the active CUDA device's compute capability."""
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDevice failed with CUDA error {status}")

    values = []
    for attribute, name in (
        (runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, "major"),
        (runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, "minor"),
    ):
        status, value = runtime.cudaDeviceGetAttribute(attribute, device)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"failed to query CUDA compute-capability {name} with error {status}"
            )
        values.append(value)
    logger.info("Detected CUDA device %d with compute capability %d.%d",
                device, values[0], values[1])
    return values[0], values[1]


@dataclass
class BuildArgs:
    model_dir: str
    engine_dir: str
    component: str = contracts.Component.LLM.value
    spec_role: str = contracts.SpecRole.NONE.value
    spec_type: str = "none"
    max_input_len: int = 1024
    max_kv_cache_capacity: int = 4096
    max_batch_size: int = 4
    max_lora_rank: int = 0
    max_verify_tree_size: int = 60
    max_draft_tree_size: int = 60
    tree_base: bool = False
    min_image_tokens: int = 4
    max_image_tokens: int = 1024
    max_image_tokens_per_image: int = 512
    min_time_steps: int = 100
    max_time_steps: int = 6000
    min_code_len: int = 1
    opt_code_len: int = 300
    max_code_len: int = 2000
    tp_size: int = 1
    tp_rank: int = 0
    reduced_vocab_dir: str = ""
    draft_reduced_vocab_dir: str = ""
    draft_model_dir: str = ""
    target_model_dir: str = ""
    fp8_embedding: bool = False
    plugin_path: Optional[str] = None
    profiling_detailed: bool = False
    dense_quant: str = "auto"  # auto | nvfp4-qdq | fp16
    int4_gemm_plugin_version: int = 2
    externalize_weights: Tuple[str, ...] = ()

    @functools.cached_property
    def weight_policy(self) -> WeightPolicy:
        """Externalization kinds this build can actually reproduce.

        Kinds are on by default, so an incompatible build setting narrows the
        policy and logs it. Explicitly requested kinds raise instead.
        """
        explicit = bool(self.externalize_weights)
        strict = explicit
        requested = WeightPolicy.from_request(self.externalize_weights)
        policy = requested
        spec = contracts.component_spec(self.resolved_component)
        unsupported = tuple(kind for kind in policy.kinds
                            if kind not in spec.external_weight_kinds)
        # Component capabilities are model workflow properties. Ignore an
        # unsupported kind here so one command can still build every component.
        policy = policy.without(unsupported)
        if self.tp_size > 1:
            tp_strict = (explicit and weight_policy.EXTERNAL_WEIGHT_ALL
                         not in self.externalize_weights)
            # Small FP16 normalization tensors save negligible plan space but
            # materially increase the per-rank TensorRT input set. Keep them
            # baked for the default TP policy; an explicit request still wins.
            if not explicit:
                policy = policy.without((weight_policy.EXTERNAL_WEIGHT_FP16, ))
            # TP load-time sharding covers identity FP16, embeddings, the LM
            # head, and NVFP4 weights consumed by the fused TP plugin. Dense
            # NVFP4 projections lowered through TensorRT-native GEMMs remain
            # rank-local engine constants.
            unsupported_tp = (
                weight_policy.EXTERNAL_WEIGHT_INT4_FFN,
                weight_policy.EXTERNAL_WEIGHT_INT4_MOE,
                weight_policy.EXTERNAL_WEIGHT_NVFP4_MOE,
            )
            policy = policy.without(unsupported_tp, strict=tp_strict)
        if self.fp8_embedding:
            policy = policy.without(
                (weight_policy.EXTERNAL_WEIGHT_EMBEDDING, ), strict=strict)
        vocab_dir = (self.draft_reduced_vocab_dir if self.resolved_spec_role
                     == contracts.SpecRole.DRAFT else self.reduced_vocab_dir)
        if vocab_dir:
            # A reduced vocabulary rewrites the head; the checkpoint has the
            # full one. Other FP16 projections are unaffected.
            policy = policy.baking_lm_head(strict=strict)
        return policy

    @functools.cached_property
    def sm12x(self) -> bool:
        """Return whether the active CUDA device belongs to SM12x."""
        major, _ = _active_cuda_compute_capability()
        return major == 12

    @property
    def resolved_component(self) -> contracts.Component:
        return contracts.Component(self.component)

    @property
    def resolved_spec_role(self) -> contracts.SpecRole:
        return contracts.SpecRole(self.spec_role)

    @property
    def profile_limits(self) -> contracts.ProfileLimits:
        return contracts.ProfileLimits(
            max_input_len=self.max_input_len,
            max_kv_cache_capacity=self.max_kv_cache_capacity,
            max_batch_size=self.max_batch_size,
            max_lora_rank=self.max_lora_rank,
            max_verify_tree_size=self.max_verify_tree_size,
            max_draft_tree_size=self.max_draft_tree_size,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
            max_image_tokens_per_image=self.max_image_tokens_per_image,
            min_time_steps=self.min_time_steps,
            max_time_steps=self.max_time_steps,
            min_code_len=self.min_code_len,
            opt_code_len=self.opt_code_len,
            max_code_len=self.max_code_len,
        )

    def validate(self) -> None:
        """Validate component, profile, TP, and speculative settings."""
        self.profile_limits.validate()
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if self.int4_gemm_plugin_version not in (1, 2):
            raise ValueError("int4_gemm_plugin_version must be 1 or 2")
        if self.resolved_spec_role != contracts.SpecRole.NONE:
            if self.resolved_component != contracts.Component.LLM:
                raise ValueError("speculative roles require --component llm")
            if self.spec_type == "none":
                raise ValueError("speculative roles require --spec-type")
        elif self.spec_type != "none":
            raise ValueError("--spec-type requires --spec-role base or draft")
        if self.tree_base and not (
                self.resolved_spec_role == contracts.SpecRole.BASE
                and self.spec_type in ("mtp", "dflash", "jetspec")):
            raise ValueError(
                "--tree-base is only valid for an MTP, DFlash, or JetSpec base engine"
            )
        if self.draft_reduced_vocab_dir and not (
                self.resolved_spec_role == contracts.SpecRole.DRAFT
                and self.spec_type in ("dflash", "jetspec")):
            raise ValueError(
                "--draft-reduced-vocab-dir is only valid for a DFlash or JetSpec draft"
            )
        paired_base = (self.resolved_spec_role == contracts.SpecRole.BASE
                       and self.spec_type
                       in ("eagle3", "dflash", "jetspec", "dspark"))
        if paired_base and not self.draft_model_dir:
            raise ValueError(
                f"{self.spec_type} base requires --draft-model-dir for pairing"
            )
        if self.draft_model_dir and not paired_base:
            raise ValueError(
                "--draft-model-dir is only valid for a paired base engine")
        paired_draft = self.resolved_spec_role == contracts.SpecRole.DRAFT
        if paired_draft and not self.target_model_dir:
            raise ValueError(f"{self.spec_type} draft requires "
                             "--target-model-dir")
        if self.target_model_dir and not paired_draft:
            raise ValueError(
                "--target-model-dir is only valid for a paired draft engine")
        if (self.reduced_vocab_dir
                and self.resolved_spec_role == contracts.SpecRole.BASE
                and self.spec_type
                in ("dflash", "jetspec", "dspark", "gemma4_mtp")):
            raise ValueError(
                f"{self.spec_type} base engines require the full vocabulary")
        if self.fp8_embedding and self.resolved_spec_role == contracts.SpecRole.DRAFT:
            raise ValueError("draft engines use the base embedding sidecar")


@dataclass(frozen=True)
class BuildResult:
    """Serialized engine path and its static runtime weight inputs."""

    engine_path: str
    checkpoint_weight_bindings: tuple
    # Set when the bindings resolve against a checkpoint other than the one
    # passed to the runtime, e.g. a component stored in its own subdirectory.
    checkpoint_dir: str
    checkpoint_identity: dict


class _TrtLogger(trt.ILogger):

    def __init__(self) -> None:
        trt.ILogger.__init__(self)

    def log(self, severity, msg) -> None:  # noqa: D401
        if severity <= trt.ILogger.Severity.ERROR:
            logger.error("[TRT] %s", msg)
        elif severity == trt.ILogger.Severity.WARNING:
            logger.warning("[TRT] %s", msg)
        elif severity == trt.ILogger.Severity.INFO:
            logger.info("[TRT] %s", msg)
        else:
            logger.debug("[TRT] %s", msg)


def load_plugin_library(plugin_path: Optional[str]) -> ctypes.CDLL:
    """dlopen ``libNvInfer_edgellm_plugin.so`` so its plugin creators register."""
    path = plugin_path or "build/libNvInfer_edgellm_plugin.so"
    if not os.path.exists(path):
        raise FileNotFoundError(f"plugin library not found: {path}")
    handle = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    logger.info("Loaded plugin library %s", path)
    return handle


def build_engine(args: BuildArgs,
                 cfg: Optional[DeviceConfig] = None,
                 bundle: Optional[BundleConfig] = None,
                 plugin_handle: Optional[ctypes.CDLL] = None) -> BuildResult:
    """Build an engine and return its runtime artifacts."""
    build_start = time.perf_counter()
    args.validate()
    plugin_handle = plugin_handle or load_plugin_library(args.plugin_path)

    bundle = bundle or BundleConfig.from_pretrained(args.model_dir)
    if args.resolved_component not in bundle.components:
        available = ", ".join(
            sorted(component.value for component in bundle.components))
        raise ValueError(
            f"{bundle.root_model_type!r} has no {args.component!r} component; "
            f"available components: {available}")
    if args.resolved_component in LLM_COMPONENTS:
        if cfg is None:
            cfg = load_device_config(args)
        logger.info("model_type=%s layers=%d hidden=%d experts=%d quant=%s",
                    cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size,
                    cfg.num_experts, cfg.quant_type)
    else:
        logger.info("root_model_type=%s component=%s", bundle.root_model_type,
                    args.component)

    trt_logger = _TrtLogger()
    builder = trt.Builder(trt_logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03,
                               True)
    if args.profiling_detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    from ..models import build_model
    from ..models import registry as model_registry

    weight_conversion = model_registry.weight_conversion_for(
        bundle.root_model_type, args.spec_type, args.resolved_spec_role)
    component_quant = (cfg.quant if cfg else quantization.parse_quantization(
        args.model_dir, bundle.root,
        bundle.component_dict(args.resolved_component), weight_conversion))
    vocab_dir = (args.draft_reduced_vocab_dir if args.resolved_spec_role
                 == contracts.SpecRole.DRAFT else args.reduced_vocab_dir)
    vocab_map = None
    if vocab_dir:
        if cfg is None:
            raise ValueError(
                "reduced vocabulary is only valid for LLM components")
        from .safetensors_np import load_safetensors_tensor
        vocab_path = os.path.join(vocab_dir, "vocab_map.safetensors")
        vocab_map = load_safetensors_tensor(vocab_path,
                                            "vocab_map").astype(np.int64)
        if vocab_map.ndim != 1 or vocab_map.size == 0:
            raise ValueError("vocab_map must be a non-empty 1D tensor")
        if np.any(vocab_map < 0) or np.any(vocab_map >= cfg.vocab_size):
            raise ValueError("vocab_map contains token IDs outside vocab_size")
        if np.unique(vocab_map).size != vocab_map.size:
            raise ValueError("vocab_map contains duplicate token IDs")
        cfg.reduced_vocab_size = int(vocab_map.size)
    policy = args.weight_policy
    configuration = model_registry.configuration_module_for(
        bundle.root_model_type)
    component_weight_policy = getattr(configuration, "component_weight_policy",
                                      None)
    if component_weight_policy is not None:
        policy = component_weight_policy(args, policy)
    requested_policy = WeightPolicy.from_request(args.externalize_weights)
    kept = tuple(kind for kind in requested_policy.kinds
                 if kind not in policy.kinds)
    logger.info("Externalizing %s%s", ", ".join(policy.kinds) or "nothing",
                ("; keeping " + ", ".join(kept) +
                 " in the engine") if kept else "")
    if policy.bake_lm_head and requested_policy.externalizes_fp16("lm_head"):
        logger.info("Keeping the LM head in the engine; a reduced vocabulary "
                    "rewrites it")
    weights = Weights(args.model_dir,
                      group_size=component_quant.group_size,
                      quant=component_quant,
                      component=args.component,
                      spec_type=args.spec_type,
                      spec_role=args.spec_role,
                      vocab_map=vocab_map,
                      conversion=weight_conversion,
                      int4_gemm_plugin_version=args.int4_gemm_plugin_version,
                      tie_word_embeddings=bool(cfg
                                               and cfg.tie_word_embeddings),
                      policy=policy)
    net = Net(builder,
              network,
              policy=policy,
              int4_gemm_plugin_version=args.int4_gemm_plugin_version)
    try:
        build_model(net, bundle, cfg, weights, args)
        from .artifacts import checkpoint_identity, checkpoint_weight_bindings
        bindings = checkpoint_weight_bindings(args, cfg,
                                              net.take_weight_bindings(),
                                              weights)
        weights_dir = os.path.abspath(weights.source_dir)
        identity_start = time.perf_counter()
        identity = checkpoint_identity(bindings, weights.source_dir,
                                       args.target_model_dir)
        identity_seconds = time.perf_counter() - identity_start
    finally:
        weights.close()
    runtime_checkpoint_dir = args.target_model_dir or args.model_dir
    binding_dir = ("" if weights_dir == os.path.abspath(runtime_checkpoint_dir)
                   else weights_dir)

    _setup_profiles(builder, config, network, cfg, args, bundle)

    logger.info(
        "Building serialized engine (this may take several minutes)...")
    trt_build_start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    trt_build_seconds = time.perf_counter() - trt_build_start
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    spec = contracts.component_spec(args.resolved_component)
    engine_path = spec.output_path(args.engine_dir, args.resolved_spec_role,
                                   args.tp_size, args.tp_rank)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))  # IHostMemory -> buffer
    logger.info("Engine written to %s (%d bytes)", engine_path,
                int(serialized.nbytes))
    total_seconds = time.perf_counter() - build_start
    logger.info(
        "Build completed in %.3f s (checkpoint identity %.3f s, TensorRT "
        "%.3f s, frontend and serialization %.3f s)", total_seconds,
        identity_seconds, trt_build_seconds,
        total_seconds - identity_seconds - trt_build_seconds)
    return BuildResult(
        engine_path=engine_path,
        checkpoint_weight_bindings=tuple(bindings),
        checkpoint_dir=binding_dir,
        checkpoint_identity=identity,
    )


def load_device_config(args: BuildArgs) -> DeviceConfig:
    """Load and configure the model metadata used for one engine build."""
    from ..models import registry as model_registry

    cfg = DeviceConfig.from_pretrained(args.model_dir, args.resolved_component,
                                       args.tp_size, args.tp_rank)
    paired_target = None
    if args.target_model_dir:
        paired_target = DeviceConfig.from_pretrained(args.target_model_dir,
                                                     contracts.Component.LLM,
                                                     args.tp_size,
                                                     args.tp_rank)
    return model_registry.configure_for_build(
        cfg,
        args.resolved_spec_role,
        args.spec_type,
        paired_target=paired_target,
        paired_draft_dir=args.draft_model_dir,
        build_args=args)


# ---------------------------------------------------------------------------
# Optimization profiles
# ---------------------------------------------------------------------------


def _setup_profiles(builder, config, network, cfg: Optional[DeviceConfig],
                    args: BuildArgs, bundle: BundleConfig) -> None:
    from ..models import registry as model_registry

    configuration = model_registry.configuration_module_for(
        bundle.root_model_type)
    model_profiles = getattr(configuration, "setup_profiles", None)
    if (model_profiles is not None
            and model_profiles(builder, config, network, args, bundle)):
        return
    if args.resolved_component == contracts.Component.DLLM:
        if cfg is None:
            raise ValueError("DLLM components require DeviceConfig")
        _setup_diffusion_profiles(builder, config, network, cfg, args)
        return
    if args.resolved_component in LLM_COMPONENTS:
        if cfg is None:
            raise ValueError("LLM components require DeviceConfig")
        _setup_llm_profiles(builder, config, network, cfg, args)
        return
    _setup_component_profile(builder, config, network, args)


def _setup_diffusion_profiles(builder, config, network, cfg: DeviceConfig,
                              args: BuildArgs) -> None:
    """Create prefill and denoise/commit profiles for one DLLM backbone."""
    prefill = builder.create_optimization_profile()
    diffusion = builder.create_optimization_profile()

    max_batch = args.max_batch_size
    max_input = args.max_input_len
    max_kv = args.max_kv_cache_capacity
    canvas = int(cfg.raw_root.get("canvas_length", 256))
    pages_per_sequence = (max_kv + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE
    pool_pages = max_batch * pages_per_sequence
    inputs = {
        network.get_input(index).name: network.get_input(index)
        for index in range(network.num_inputs)
    }

    def fixed_dim(name, axis, fallback):
        tensor = inputs.get(name)
        if tensor is None:
            return fallback
        value = int(tensor.shape[axis])
        return fallback if value < 0 else value

    def set_shapes(name, prefill_shapes, diffusion_shapes):
        if name not in inputs:
            return
        prefill.set_shape(name, *prefill_shapes)
        diffusion.set_shape(name, *diffusion_shapes)

    hidden = fixed_dim("inputs_embeds", -1, cfg.hidden_size)
    prefill_sequence = ((1, 1, hidden), (max_batch, max(1, max_input // 2),
                                         hidden), (max_batch, max_input,
                                                   hidden))
    diffusion_sequence = ((1, 1, hidden), (max_batch, canvas, hidden),
                          (max_batch, canvas, hidden))
    for name in ("inputs_embeds", "prev_self_conditioning_embeds"):
        set_shapes(name, prefill_sequence, diffusion_sequence)

    prefill_tokens = ((1, 1), (max_batch, max(1, max_input // 2)), (max_batch,
                                                                    max_input))
    diffusion_tokens = ((1, 1), (max_batch, canvas), (max_batch, canvas))
    set_shapes("canvas_ids", prefill_tokens, diffusion_tokens)

    batch_vector = ((1, ), (max_batch, ), (max_batch, ))
    for name in ("phase_is_encoder", "context_lengths", "kvcache_start_index"):
        set_shapes(name, batch_vector, batch_vector)

    set_shapes("select_token_indices",
               ((1, 1), (max_batch, 1), (max_batch, 1)),
               ((1, 1), (max_batch, canvas), (max_batch, canvas)))
    set_shapes("context_mask_selector", ((0, ), (0, ), (0, )),
               ((0, ), (max_batch, ), (max_batch, )))

    for rope_name in ("rope_rotary_cos_sin", "rope_rotary_cos_sin_sliding",
                      "rope_rotary_cos_sin_full"):
        rotary_dim = fixed_dim(rope_name, -1, cfg.rotary_dim)
        rope_shapes = ((1, max_kv, rotary_dim),
                       (max_batch, max_kv, rotary_dim), (max_batch, max_kv,
                                                         rotary_dim))
        set_shapes(rope_name, rope_shapes, rope_shapes)

    for index in range(cfg.num_hidden_layers):
        name = f"past_key_values_{index}"
        heads = fixed_dim(name, 3, cfg.layer_num_kv_heads(index))
        head_dim = fixed_dim(name, 4, cfg.layer_head_dim(index))
        pool_shape = (2, pool_pages, KV_PAGE_SIZE, heads, head_dim)
        pool_shapes = (pool_shape, pool_shape, pool_shape)
        set_shapes(name, pool_shapes, pool_shapes)

    page_shapes = ((1, 2, pages_per_sequence),
                   (max_batch, 2, pages_per_sequence), (max_batch, 2,
                                                        pages_per_sequence))
    set_shapes("kv_page_table", page_shapes, page_shapes)

    config.add_optimization_profile(prefill)
    config.add_optimization_profile(diffusion)


def _setup_llm_profiles(builder, config, network, cfg: DeviceConfig,
                        args: BuildArgs) -> None:
    ctx_prof = builder.create_optimization_profile()
    gen_prof = builder.create_optimization_profile()

    maxB = args.max_batch_size
    maxIn = args.max_input_len
    maxKV = args.max_kv_cache_capacity
    page_size = KV_PAGE_SIZE
    pages_per_sequence = (maxKV + page_size - 1) // page_size
    pool_pages = maxB * pages_per_sequence
    H = cfg.hidden_size
    Hkv = cfg.num_key_value_heads
    D = cfg.head_dim

    names = {network.get_input(i).name for i in range(network.num_inputs)}
    inputs = {
        network.get_input(i).name: network.get_input(i)
        for i in range(network.num_inputs)
    }

    def fixed_dim(name, axis, fallback):
        tensor = inputs.get(name)
        if tensor is None:
            return fallback
        value = int(tensor.shape[axis])
        return fallback if value < 0 else value

    def set_profile_shapes(name,
                           context_min,
                           context_opt,
                           context_max,
                           generation_min=None,
                           generation_opt=None,
                           generation_max=None):
        if name not in names:
            return
        ctx_prof.set_shape(name, context_min, context_opt, context_max)
        gen_prof.set_shape(name, generation_min or context_min, generation_opt
                           or context_opt, generation_max or context_max)

    # inputs_embeds [B, S, H]
    embed_width = fixed_dim("inputs_embeds", -1, H)
    generation_sequence_max = args.profile_limits.generation_sequence_max(
        args.resolved_spec_role)
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.spec_type in ("dflash", "jetspec", "dspark")):
        draft = args.max_draft_tree_size
        set_profile_shapes("inputs_embeds", (1, 1, embed_width),
                           (maxB, draft, embed_width),
                           (maxB, draft, embed_width),
                           generation_min=(1, 1, embed_width),
                           generation_opt=(maxB, draft, embed_width),
                           generation_max=(maxB, draft, embed_width))
    else:
        set_profile_shapes("inputs_embeds", (1, 1, embed_width),
                           (maxB, max(1, maxIn // 2), embed_width),
                           (maxB, maxIn, embed_width),
                           generation_min=(1, 1, embed_width),
                           generation_opt=(maxB, 1, embed_width),
                           generation_max=(maxB, generation_sequence_max,
                                           embed_width))

    # last_token_ids [B, T]
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.spec_type == "eagle3"):
        set_profile_shapes("last_token_ids", (1, 1), (maxB, 1), (maxB, 1),
                           generation_min=(1, 1),
                           generation_opt=(maxB, 1),
                           generation_max=(maxB, generation_sequence_max))
    else:
        selected_tokens = (args.max_verify_tree_size if args.resolved_spec_role
                           == contracts.SpecRole.BASE else 1)
        set_profile_shapes("last_token_ids", (1, 1), (maxB, 1),
                           (maxB, selected_tokens))

    # context_lengths [B]
    set_profile_shapes("context_lengths", (1, ), (maxB, ), (maxB, ))

    # Draft cache updates require one start position per K/V delta batch.
    block_draft = (args.resolved_spec_role == contracts.SpecRole.DRAFT
                   and args.spec_type in ("dflash", "jetspec", "dspark"))
    context_start_min = (1, ) if block_draft else (0, )
    set_profile_shapes("kvcache_start_index",
                       context_start_min, (maxB, ), (maxB, ),
                       generation_min=(1, ),
                       generation_opt=(maxB, ),
                       generation_max=(maxB, ))

    # rope cos/sin [ropeB, maxpos, rotary] -- both profiles identical
    for rope_name in ("rope_rotary_cos_sin", "rope_rotary_cos_sin_sliding",
                      "rope_rotary_cos_sin_full"):
        rotary_dim = fixed_dim(rope_name, -1, cfg.rotary_dim)
        set_profile_shapes(rope_name, (1, maxKV, rotary_dim),
                           (maxB, maxKV, rotary_dim),
                           (maxB, maxKV, rotary_dim))

    # Paged KV pool [2, numPages, 128, Hkv, D] -- both profiles identical.
    for i in range(
            cfg.num_attn_layers if cfg.is_hybrid else cfg.num_hidden_layers):
        name = f"past_key_values_{i}"
        layer_heads = fixed_dim(name, 3, Hkv)
        layer_dim = fixed_dim(name, 4, D)
        pool_shape = (2, pool_pages, page_size, layer_heads, layer_dim)
        set_profile_shapes(name, pool_shape, pool_shape, pool_shape)

    set_profile_shapes("kv_page_table", (1, 2, pages_per_sequence),
                       (maxB, 2, pages_per_sequence),
                       (maxB, 2, pages_per_sequence))

    # Mamba recurrent + conv states (hybrid only)
    if cfg.mamba_cfg is not None:
        mc = cfg.mamba_cfg
        for m in range(cfg.num_mamba_layers):
            set_profile_shapes(f"conv_state_{m}",
                               (1, mc.conv_dim, mc.conv_kernel),
                               (maxB, mc.conv_dim, mc.conv_kernel),
                               (maxB, mc.conv_dim, mc.conv_kernel))
            set_profile_shapes(
                f"recurrent_state_{m}",
                (1, mc.num_heads, mc.head_dim, mc.ssm_state_size),
                (maxB, mc.num_heads, mc.head_dim, mc.ssm_state_size),
                (maxB, mc.num_heads, mc.head_dim, mc.ssm_state_size))

    if cfg.gdn_cfg is not None:
        gc = cfg.gdn_cfg
        for index in range(cfg.num_gdn_layers):
            set_profile_shapes(f"conv_state_{index}",
                               (1, gc.conv_dim, gc.conv_kernel),
                               (maxB, gc.conv_dim, gc.conv_kernel),
                               (maxB, gc.conv_dim, gc.conv_kernel))
            set_profile_shapes(
                f"recurrent_state_{index}",
                (1, gc.num_value_heads, gc.key_head_dim, gc.value_head_dim),
                (maxB, gc.num_value_heads, gc.key_head_dim, gc.value_head_dim),
                (maxB, gc.num_value_heads, gc.key_head_dim, gc.value_head_dim))

    for index in range(cfg.num_deepstack_features):
        set_profile_shapes(f"deepstack_embeds_{index}", (1, 1, H),
                           (maxB, max(1, maxIn // 2), H), (maxB, maxIn, H),
                           generation_min=(1, 1, H),
                           generation_opt=(maxB, 1, H),
                           generation_max=(maxB, generation_sequence_max, H))

    verify = args.max_verify_tree_size
    tree_size = (args.max_draft_tree_size if args.resolved_spec_role
                 == contracts.SpecRole.DRAFT else verify)
    set_profile_shapes("attention_pos_id", (1, 1), (maxB, 1),
                       (maxB, tree_size))
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.spec_type in ("dflash", "jetspec", "dspark")):
        draft = args.max_draft_tree_size
        set_profile_shapes(
            "attention_mask", (1, 1, 1),
            (maxB, max(1, draft // 2), max(1, (draft // 2 + 31) // 32)),
            (maxB, draft, (draft + 31) // 32))
    else:
        set_profile_shapes("attention_mask", (1, 1, 1), (maxB, 1, 1),
                           (maxB, tree_size, maxKV + tree_size))
    set_profile_shapes("hidden_states", (1, 1, H), (maxB, 1, H),
                       (maxB, tree_size, H))
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.spec_type in ("eagle3", "mtp")):
        set_profile_shapes("hidden_states_from_draft", (1, 1, H),
                           (maxB, max(1, maxIn // 2), H), (maxB, maxIn, H),
                           generation_min=(1, 1, H),
                           generation_opt=(maxB, 1, H),
                           generation_max=(maxB, generation_sequence_max, H))
    else:
        set_profile_shapes("hidden_states_from_draft", (1, 1, H), (maxB, 1, H),
                           (maxB, verify, H))
    set_profile_shapes("lm_head_weight", (1, H), (cfg.vocab_size, H),
                       (cfg.vocab_size, H))
    set_profile_shapes("tree_parent_ids", (1, 1), (maxB, 1), (maxB, tree_size))
    set_profile_shapes("tree_depths", (1, 1), (maxB, 1), (maxB, tree_size))
    set_profile_shapes("spec_verify_phase_marker", (0, ), (0, ), (0, ),
                       generation_min=(0, ),
                       generation_opt=(1, ),
                       generation_max=(1, ))
    set_profile_shapes("dflash_delta_lengths", (1, ), (maxB, ), (maxB, ))
    dflash_width = fixed_dim("dflash_target_hidden_concat", -1, H)
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.spec_type in ("dflash", "jetspec", "dspark")):
        generation_hidden = args.max_draft_tree_size + 1
        if args.spec_type == "dspark":
            generation_hidden = max(generation_hidden,
                                    args.max_verify_tree_size)
        set_profile_shapes("dflash_target_hidden_concat", (1, 1, dflash_width),
                           (maxB, max(1, maxIn // 2), dflash_width),
                           (maxB, maxIn, dflash_width),
                           generation_min=(1, 1, dflash_width),
                           generation_opt=(maxB, generation_hidden,
                                           dflash_width),
                           generation_max=(maxB, generation_hidden,
                                           dflash_width))
    else:
        set_profile_shapes("dflash_target_hidden_concat", (1, 1, dflash_width),
                           (maxB, 1, dflash_width),
                           (maxB, verify, dflash_width))
    assistant_width = fixed_dim("hidden_states_input", -1, H)
    if (cfg.shares_target_kv
            or (args.resolved_spec_role == contracts.SpecRole.DRAFT
                and args.spec_type in ("eagle3", "mtp", "gemma4_mtp"))):
        set_profile_shapes("hidden_states_input", (1, 1, assistant_width),
                           (maxB, max(1, maxIn // 2), assistant_width),
                           (maxB, maxIn, assistant_width),
                           generation_min=(1, 1, assistant_width),
                           generation_opt=(maxB, 1, assistant_width),
                           generation_max=(maxB, generation_sequence_max,
                                           assistant_width))
    else:
        set_profile_shapes("hidden_states_input", (1, 1, assistant_width),
                           (maxB, 1, assistant_width),
                           (maxB, verify, assistant_width))

    for index in range(cfg.num_hidden_layers):
        name = f"ple_token_embeds_{index}"
        width = fixed_dim(name, -1, cfg.hidden_size_per_layer_input)
        set_profile_shapes(name, (1, 1, width),
                           (maxB, max(1, maxIn // 2), width),
                           (maxB, maxIn, width),
                           generation_min=(1, 1, width),
                           generation_opt=(maxB, 1, width),
                           generation_max=(maxB, generation_sequence_max,
                                           width))

    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)
        if ".lora_A.weight" not in tensor.name and ".lora_B.weight" not in tensor.name:
            continue
        shape = tuple(int(dim) for dim in tensor.shape)
        if ".lora_A.weight" in tensor.name:
            minimum = (shape[0], 0)
            maximum = (shape[0], args.max_lora_rank)
        else:
            minimum = (0, shape[1])
            maximum = (args.max_lora_rank, shape[1])
        optimum = maximum
        set_profile_shapes(tensor.name, minimum, optimum, maximum)

    config.add_optimization_profile(ctx_prof)
    config.add_optimization_profile(gen_prof)


def _setup_component_profile(builder, config, network,
                             args: BuildArgs) -> None:
    """Create the single profile used by non-autoregressive components."""
    profile = builder.create_optimization_profile()
    component = args.resolved_component
    input_shapes = {
        network.get_input(index).name:
        tuple(int(dim) for dim in network.get_input(index).shape)
        for index in range(network.num_inputs)
    }

    def dynamic_extent(name: str, axis: int) -> tuple:
        if component == contracts.Component.VISUAL:
            gemma_visual = "pooling_weights" in input_shapes
            gemma_unified = "pixel_position_ids" in input_shapes
            visual_input = input_shapes.get("input", ())
            if gemma_visual:
                patches_per_token = 9
                soft_opt = max(
                    args.min_image_tokens,
                    (args.min_image_tokens + args.max_image_tokens) // 2)
                if name in ("cu_seqlens", "kv_lengths"):
                    max_images = max(
                        1, args.max_image_tokens // args.min_image_tokens)
                    return 2, max_images + 1, max_images + 1
                if name == "max_seqlen_carrier":
                    maximum = (args.max_image_tokens_per_image *
                               patches_per_token)
                    return 1, max(1, maximum // 2), maximum
                if name == "pooling_weights":
                    if axis == 0:
                        return (args.min_image_tokens, soft_opt,
                                args.max_image_tokens)
                    return (args.min_image_tokens * patches_per_token,
                            soft_opt * patches_per_token,
                            args.max_image_tokens * patches_per_token)
                return (args.min_image_tokens * patches_per_token,
                        soft_opt * patches_per_token,
                        args.max_image_tokens * patches_per_token)
            if gemma_unified:
                return (args.min_image_tokens,
                        max(args.min_image_tokens,
                            (args.min_image_tokens + args.max_image_tokens) //
                            2), args.max_image_tokens)
            if len(visual_input) == 4:
                minimum = max(1, args.min_image_tokens // 256)
                maximum = max(1, args.max_image_tokens // 256)
                return minimum, (minimum + maximum) // 2, maximum
            if name in ("cu_seqlens", "kv_lengths"):
                max_images = max(
                    2, args.max_image_tokens // args.min_image_tokens + 1)
                return 2, max(2, max_images // 2), max_images
            if name in ("cu_window_seqlens", "kv_lengths_window"):
                return 2, args.max_image_tokens, args.max_image_tokens
            if name in ("window_index", "reverse_window_index"):
                return (args.min_image_tokens,
                        (args.min_image_tokens + args.max_image_tokens) // 2,
                        args.max_image_tokens)
            if name == "max_seqlen_carrier":
                maximum = args.max_image_tokens_per_image * 4
                return 1, max(1, maximum // 2), maximum
            patches_per_token = 4
            return (args.min_image_tokens * patches_per_token,
                    max(args.min_image_tokens,
                        (args.min_image_tokens + args.max_image_tokens) // 2) *
                    patches_per_token,
                    args.max_image_tokens * patches_per_token)
        if component == contracts.Component.AUDIO:
            if name == "padded_feature":
                window = int(input_shapes[name][-1])
                minimum = max(1, (args.min_time_steps + window - 1) // window)
                maximum = max(1, (args.max_time_steps + window - 1) // window)
                return minimum, (minimum + maximum) // 2, maximum
            if name in ("padded_mask_after_cnn_indices", "attention_mask"):
                feature_window = int(input_shapes["padded_feature"][-1])
                minimum_chunks = max(
                    1, (args.min_time_steps + feature_window - 1) //
                    feature_window)
                maximum_chunks = max(
                    1, (args.max_time_steps + feature_window - 1) //
                    feature_window)
                minimum = minimum_chunks
                # Qwen3-ASR and Qwen3-Omni use three stride-2 Conv2D stages.
                # Derive the full-chunk token bound from the engine's fixed
                # feature width instead of assuming the old n_window=50.
                tokens_per_chunk = (feature_window + 7) // 8
                maximum = maximum_chunks * tokens_per_chunk
                return minimum, (minimum + maximum) // 2, maximum
            if name in ("n_window", "cu_seqlens"):
                return 1, max(1, args.max_time_steps // 200), max(
                    1, args.max_time_steps // 100)
            if name == "valid":
                minimum = (args.min_time_steps + 3) // 4
                maximum = (args.max_time_steps + 3) // 4
                return minimum, (minimum + maximum) // 2, maximum
            if name == "input_features" and axis == 1:
                if input_shapes[name][-1] == 640:
                    return (args.min_time_steps,
                            (args.min_time_steps + args.max_time_steps) // 2,
                            args.max_time_steps)
                alignment = 4 if "valid" in input_shapes else 8

                def align(value):
                    return ((value + alignment - 1) // alignment * alignment)

                minimum = align(args.min_time_steps)
                maximum = align(args.max_time_steps)
                return minimum, align((minimum + maximum) // 2), maximum
            return (args.min_time_steps,
                    (args.min_time_steps + args.max_time_steps) // 2,
                    args.max_time_steps)
        if component == contracts.Component.CODE2WAV:
            if axis == 0:
                return 1, args.max_batch_size, args.max_batch_size
            return args.min_code_len, args.opt_code_len, args.max_code_len
        if component == contracts.Component.ACTION:
            if "cache" in name:
                if axis == 0:
                    return 1, args.max_batch_size, args.max_batch_size
                return (args.max_kv_cache_capacity, ) * 3
            if axis == 0:
                return 1, args.max_batch_size, args.max_batch_size
            return 1, args.max_input_len, args.max_input_len
        return 1, 1, 1

    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if -1 not in shape:
            continue
        minimum = list(shape)
        optimum = list(shape)
        maximum = list(shape)
        for axis, dim in enumerate(shape):
            if dim != -1:
                continue
            low, opt, high = dynamic_extent(tensor.name, axis)
            minimum[axis] = low
            optimum[axis] = opt
            maximum[axis] = high
        profile.set_shape(tensor.name, tuple(minimum), tuple(optimum),
                          tuple(maximum))
    config.add_optimization_profile(profile)
