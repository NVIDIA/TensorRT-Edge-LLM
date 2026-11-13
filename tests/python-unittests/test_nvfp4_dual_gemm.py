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
"""Device contract tests for the SM100/103/110 NVFP4 merged-N SwiGLU kernel."""

import ctypes
import gc
import json
import os
import pathlib
import re
import subprocess
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kernelSrcs"))
sys.path.insert(0, str(ROOT / "kernelSrcs" / "gemm_cutedsl"))

try:
    import cuda.bindings.driver as cuda
    import cupy as cp
    import cutlass
    import cutlass.cute as cute
    import gemm_blackwell_nvfp4_ws as nvfp4
    from cutlass.cute import runtime
except ImportError:
    cp = None

if cp is not None:

    @cute.kernel
    def block_scale_division_kernel(values: cute.Tensor, scales: cute.Tensor,
                                    output: cute.Tensor):
        index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if index < 65536 * 126:
            value = values[index % 65536]
            scale = scales[index // 65536]
            reciprocal = nvfp4._div_rn(1.0, scale)
            output[index] = nvfp4._half_div_fp8_rn(value, scale, reciprocal)

    @cute.jit
    def launch_block_scale_division(values: cute.Tensor, scales: cute.Tensor,
                                    output: cute.Tensor,
                                    stream: cuda.CUstream):
        block_scale_division_kernel(values, scales,
                                    output).launch(grid=[32256, 1, 1],
                                                   block=[256, 1, 1],
                                                   stream=stream)


def sf_atom(scales):
    """Convert row-major E4M3 bytes to the 128-row/four-block SF atom."""
    rows, blocks = scales.shape
    padded = cp.zeros(((rows + 127) // 128 * 128, blocks), dtype=cp.uint8)
    padded[:rows] = scales
    return padded.reshape(-1, 4, 32, blocks // 4,
                          4).transpose(0, 3, 2, 1, 4).copy().ravel()


def interleave(gate, up):
    """Keep every gate tile and matching up tile in one 256-row MMA."""
    n, width = gate.shape
    return cp.stack(
        (gate.reshape(n // 128, 128, width), up.reshape(n // 128, 128, width)),
        axis=1).reshape(2 * n, width)


def dequantize(packed, scales):
    """Decode E2M1 and positive E4M3 independently of the MMA kernel."""
    values = cp.asarray(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=cp.float32)
    bits = cp.stack((packed & 15, packed >> 4),
                    axis=-1).reshape(packed.shape[0], -1)
    exponent = (scales >> 3).astype(cp.int32)
    mantissa = (scales & 7).astype(cp.float32)
    scale_values = cp.where(exponent == 0, mantissa * (2.0**-9),
                            (1 + mantissa / 8) * cp.exp2(exponent - 7)).astype(
                                cp.float32)
    return values[bits] * cp.repeat(scale_values, 16, axis=1)


class Nvfp4DualGemmTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cp is None:
            raise unittest.SkipTest("CuPy and CuTe DSL are required")
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise unittest.SkipTest("A CUDA device is required") from error
        if (props["major"], props["minor"]) not in ((10, 0), (10, 3), (11, 0)):
            raise unittest.SkipTest(
                "NVFP4 dual GEMM requires SM100, SM103 or SM110")
        cls.stream = cp.cuda.Stream(non_blocking=True)
        cls.sm_count = props["multiProcessorCount"]
        cls.compiled = None

    def make_launcher(self, arrays, m, n, k):
        dtypes = [
            cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN, cutlass.Float16, cutlass.Float32
        ]
        pointers = [
            runtime.make_ptr(dtype,
                             array.data.ptr,
                             cute.AddressSpace.gmem,
                             assumed_align=16)
            for array, dtype in zip(arrays, dtypes)
        ]
        args = (*pointers, cutlass.Int64(m), cutlass.Int64(2 * n),
                cutlass.Int64(k), 16, cutlass.Int32(self.sm_count),
                cuda.CUstream(self.stream.ptr))
        if type(self).compiled is None:
            kernel = nvfp4.GemmBlackwellNvFp4WS(mma_tiler_mn=(128, 256),
                                                fused_swiglu=True)
            type(self).compiled = cute.compile(kernel.wrapper, *args)
        launch_args = args[:9] + args[10:]
        return lambda: self.compiled(*launch_args)

    def run_case(self, m, n=17408, k=5120, zero=False, graph=False):
        """Compare fused output with FP32 QDQ on independent gate/up scales."""
        with self.stream:
            rng = cp.random.RandomState(1265)
            a = rng.randint(0, 256, size=(m, k // 2), dtype=cp.uint8)
            gate = rng.randint(0, 256, size=(n, k // 2), dtype=cp.uint8)
            up = rng.randint(0, 256, size=(n, k // 2), dtype=cp.uint8)
            sx = rng.randint(40, 57, size=(m, k // 16), dtype=cp.uint8)
            sg = rng.randint(32, 49, size=(n, k // 16), dtype=cp.uint8)
            su = rng.randint(48, 65, size=(n, k // 16), dtype=cp.uint8)
            sx[:, ::7] = 0
            sg[:, ::11] = 0
            if zero:
                sx.fill(0)
            weights = interleave(gate, up)
            scales = sf_atom(interleave(sg, su))
            sfa = sf_atom(sx)
            alpha = cp.asarray([0.03125, 0.0078125], dtype=cp.float32)
            guard = cp.full(m * n + 256, 123.0, dtype=cp.float16)
            output = guard[128:-128].reshape(m, n)
            arrays = [a, weights, sfa, scales, output, alpha]
            launch = self.make_launcher(arrays, m, n, k)
            launch()
            eager = output.copy()
            if graph:
                self.stream.synchronize()
                self.stream.begin_capture()
                launch()
                captured = self.stream.end_capture()
                output.fill(cp.nan)
                captured.launch(self.stream)
            x32 = dequantize(a, sx)
            g32 = (x32 @ dequantize(gate, sg).T) * alpha[0]
            u32 = (x32 @ dequantize(up, su).T) * alpha[1]
            reference = g32 / (1 + cp.exp(-g32)) * u32
            actual = output.astype(cp.float32)
            error = actual - reference
            rel_l2 = float(
                cp.linalg.norm(error) /
                cp.maximum(cp.linalg.norm(reference), 1e-20))
            max_abs = float(cp.max(cp.abs(error)))
            finite = bool(cp.isfinite(actual).all())
            guards_ok = bool((guard[:128] == 123).all()
                             & (guard[-128:] == 123).all())
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record(self.stream)
            for _ in range(5):
                launch()
            event_end.record(self.stream)
            event_end.synchronize()
            elapsed_ms = cp.cuda.get_elapsed_time(event_start, event_end) / 5
            print(json.dumps(
                dict(m=m,
                     n=n,
                     k=k,
                     zero=zero,
                     graph=graph,
                     rel_l2=rel_l2,
                     max_abs=max_abs,
                     finite=finite,
                     guards_ok=guards_ok,
                     smoke_kernel_ms=elapsed_ms)),
                  flush=True)
            with self.subTest(check="finite"):
                self.assertTrue(finite)
            with self.subTest(check="guards"):
                self.assertTrue(guards_ok)
            with self.subTest(check="relative_l2"):
                self.assertLess(rel_l2, 0.002)
            with self.subTest(check="elementwise"):
                cp.testing.assert_allclose(actual,
                                           reference,
                                           rtol=0.003,
                                           atol=0.003)
            with self.subTest(check="eager"):
                cp.testing.assert_allclose(eager.astype(cp.float32),
                                           reference,
                                           rtol=0.003,
                                           atol=0.003)

    def test_tail_and_zero_blocks(self):
        """Unequal scales, zero blocks and partial M tiles preserve QDQ values and guards."""
        for m in (1, 8, 127, 128, 129, 257):
            self.run_case(m, n=256, k=256, graph=True)
        self.run_case(129, n=256, k=256, zero=True)

    def test_qwen35_dense_mlp_shape(self):
        """The exact Qwen3.5-27B gate/up shape passes across decode and prefill M."""
        for m in (1, 8, 64, 256, 1024, 4096, 8192, 16384, 32768):
            self.run_case(m, graph=True)


class Nvfp4DualGemmPluginTest(Nvfp4DualGemmTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = os.environ.get("NVFP4_DUAL_GEMM_PLUGIN_LIB")
        if not path:
            raise unittest.SkipTest(
                "Set NVFP4_DUAL_GEMM_PLUGIN_LIB to the built plugin")
        import tensorrt as trt
        cls.trt = trt
        cls.library = ctypes.CDLL(path)
        cls.logger = trt.Logger(trt.Logger.WARNING)
        cls.runtime = trt.Runtime(cls.logger)
        cls.engines = {}
        print("TensorRT version:", trt.__version__, flush=True)

    def make_plugin(self, n, k, quantized=False):
        trt = self.trt
        creator = trt.get_plugin_registry().get_creator(
            "Nvfp4DualGemmPlugin", "1", "")
        self.assertIsNotNone(creator)
        fields = [
            trt.PluginField(name, np.asarray([value], dtype=np.int32),
                            trt.PluginFieldType.INT32)
            for name, value in [("inter_size", n), ("hidden_size",
                                                    k), ("activation", 0)]
        ]
        if quantized:
            fields.append(
                trt.PluginField("quantized_output",
                                np.asarray([1], dtype=np.int32),
                                trt.PluginFieldType.INT32))
        return creator.create_plugin("nvfp4_dual",
                                     trt.PluginFieldCollection(fields),
                                     trt.TensorRTPhase.BUILD)

    def build_engine(self,
                     n,
                     k,
                     quantized=False,
                     opt_m=128,
                     max_m=32768,
                     retained_build_objects=None):
        trt = self.trt
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        shapes = [(-1, k // 2), (2 * n, k // 2), (-1, ), (2 * n * k // 16, ),
                  (2, )]
        names = ["x", "weights", "sx", "sw", "alpha"]
        if quantized:
            shapes.append((1, ))
            names.append("global_scale")
        tensors = [
            network.add_input(
                name,
                trt.float32 if name in ("alpha", "global_scale") else trt.int8,
                shape) for name, shape in zip(names, shapes)
        ]
        plugin = self.make_plugin(n, k, quantized)
        self.assertIsNotNone(plugin)
        layer = network.add_plugin_v3(tensors, [], plugin)
        output = layer.get_output(0)
        output.name = "output"
        network.mark_output(output)
        if quantized:
            scale_output = layer.get_output(1)
            scale_output.name = "output_scale"
            network.mark_output(scale_output)
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("x", (1, k // 2), (opt_m, k // 2), (max_m, k // 2))
        profile.set_shape("sx", (128 * k // 16, ),
                          ((opt_m + 127) // 128 * 128 * k // 16, ),
                          ((max_m + 127) // 128 * 128 * k // 16, ))
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        self.assertIsNotNone(serialized)
        if retained_build_objects is not None:
            retained_build_objects.extend((plugin, network, builder, config))
        engine = self.runtime.deserialize_cuda_engine(serialized)
        self.assertIsNotNone(engine)
        return engine

    def make_launcher(self, arrays, m, n, k):
        if (n, k) not in self.engines:
            engine = self.build_engine(n, k)
            context = engine.create_execution_context()
            self.assertIsNotNone(context)
            self.engines[n, k] = (engine, context)
        engine, context = self.engines[n, k]
        a, weights, sfa, scales, output, alpha = arrays
        self.assertTrue(context.set_input_shape("x", a.shape))
        self.assertTrue(context.set_input_shape("sx", sfa.shape))
        self.assertEqual(tuple(context.get_tensor_shape("output")), (m, n))
        for name, array in zip(["x", "weights", "sx", "sw", "output", "alpha"],
                               arrays):
            self.assertTrue(context.set_tensor_address(name, array.data.ptr))

        def launch():
            self.assertTrue(context.execute_async_v3(self.stream.ptr))

        return launch

    def test_invalid_attributes(self):
        """The plugin rejects N/K shapes that cannot satisfy its block-scale contract."""
        for n, k in [(127, 256), (128, 96), (0, 256)]:
            with self.subTest(n=n, k=k):
                self.assertIsNone(self.make_plugin(n, k))


def quantize_reference(values, global_scale):
    """CPU FP32 oracle with explicit nearest-even rounding for E2M1 and E4M3."""
    m, n = values.shape
    blocks = values.astype(np.float32).reshape(m, n // 16, 16)
    global_scale = np.float32(global_scale)
    raw = np.minimum(
        np.max(np.abs(blocks), axis=-1) / np.float32(6) / global_scale, 448)
    exponent = np.floor(np.log2(np.maximum(raw, 2.0**-6)))
    step = np.exp2(exponent - 3)
    sf = np.rint(raw / step) * step
    lut_bits = np.arange(127, dtype=np.uint8)
    lut_exp = (lut_bits >> 3).astype(np.int32)
    lut_mant = (lut_bits & 7).astype(np.float32)
    lut = np.where(lut_exp == 0, lut_mant * 2.0**-9, (1 + lut_mant / 8) *
                   np.exp2(lut_exp - 7)).astype(np.float32)
    sf_bits = np.searchsorted(lut, sf).astype(np.uint8)
    scaled = np.where(
        sf[:, :, None] > 0,
        blocks / np.maximum(sf[:, :, None], 2.0**-9) / global_scale, 0)
    magnitude = np.abs(scaled)
    midpoints = np.asarray([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5],
                           dtype=np.float32)
    bits = np.searchsorted(midpoints, magnitude).astype(np.uint8)
    bits += ((magnitude == 0.75) | (magnitude == 1.75) |
             (magnitude == 3.5)).astype(np.uint8)
    bits |= (np.signbit(scaled).astype(np.uint8) << 3)
    bits = bits.reshape(m, n)
    packed = bits[:, ::2] | (bits[:, 1::2] << 4)
    return packed, sf_bits


class Nvfp4QuantizedMlpTest(Nvfp4DualGemmPluginTest):

    def test_module_lifetime(self):
        """Build candidates unload; runtime owners share a winner until the last release."""
        # Given retained build objects and two live runtime engines, when execution
        # owners leave, then only unused modules unload and the survivor can replay its graph.
        if os.environ.get("NVFP4_MODULE_LIFETIME_CHILD") != "1":
            env = os.environ.copy()
            env.update(NVFP4_MODULE_LIFETIME_CHILD="1",
                       EDGELLM_LOG_LEVEL="DEBUG")
            result = subprocess.run([
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "Nvfp4QuantizedMlpTest.test_module_lifetime"
            ],
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout)
            resident = set()
            loads = 0
            phases = {}
            pattern = re.compile(
                r"CuTe DSL kernel module (loaded|unloaded): "
                r"(dual_gemm_blackwell_nvfp4_output(?:_mix_ab|_mix_b)?)\b")
            for line in result.stdout.splitlines():
                event = pattern.search(line)
                if event:
                    action, module = event.groups()
                    if action == "loaded":
                        self.assertNotIn(module, resident, result.stdout)
                        resident.add(module)
                        loads += 1
                    else:
                        self.assertIn(module, resident, result.stdout)
                        resident.remove(module)
                if line.startswith("MODULE_LIFETIME "):
                    phases[line.split()[1]] = (frozenset(resident), loads)
            self.assertEqual(phases["build_done"][0], frozenset(),
                             result.stdout)
            self.assertGreater(phases["build_done"][1], 0, result.stdout)
            self.assertEqual(len(phases["first_ready"][0]), 1, result.stdout)
            self.assertEqual(phases["first_ready"][1],
                             phases["build_done"][1] + 1, result.stdout)
            self.assertEqual(phases["second_ready"], phases["first_ready"],
                             result.stdout)
            self.assertEqual(phases["first_released"], phases["second_ready"],
                             result.stdout)
            self.assertEqual(phases["all_released"][0], frozenset(),
                             result.stdout)
            print(result.stdout, flush=True)
            return

        retained = []
        engine = self.build_engine(256,
                                   256,
                                   quantized=True,
                                   opt_m=128,
                                   max_m=257,
                                   retained_build_objects=retained)
        other_engine = self.runtime.deserialize_cuda_engine(engine.serialize())
        print("MODULE_LIFETIME build_done", flush=True)
        first = engine.create_execution_context()
        second = other_engine.create_execution_context()
        with self.stream:
            m, n, k = 129, 256, 256
            arrays = [
                cp.zeros((m, k // 2), dtype=cp.uint8),
                cp.zeros((2 * n, k // 2), dtype=cp.uint8),
                cp.full(256 * k // 16, 48, dtype=cp.uint8),
                cp.full(2 * n * k // 16, 48, dtype=cp.uint8),
                cp.ones(2, dtype=cp.float32),
                cp.ones(1, dtype=cp.float32),
                cp.full((m, n // 2), 0xAB, dtype=cp.uint8),
                cp.full(256 * n // 16, 0xAB, dtype=cp.uint8)
            ]
            self.quantized_engines[n, k] = (engine, first)
            first_launch = self.quantized_launcher(arrays, m, n, k)
            first_launch()
            self.stream.synchronize()
            print("MODULE_LIFETIME first_ready", flush=True)
            self.quantized_engines[n, k] = (other_engine, second)
            second_launch = self.quantized_launcher(arrays, m, n, k)
            second_launch()
            self.stream.synchronize()
            print("MODULE_LIFETIME second_ready", flush=True)
            self.stream.begin_capture()
            second_launch()
            graph = self.stream.end_capture()
            del first_launch, first, engine
            gc.collect()
            print("MODULE_LIFETIME first_released", flush=True)
            arrays[-2].fill(0xAB)
            arrays[-1].fill(0xAB)
            graph.launch(self.stream)
            self.stream.synchronize()
            cp.testing.assert_array_equal(arrays[-2], 0)
            cp.testing.assert_array_equal(arrays[-1], 0)
            del graph, second_launch
            self.quantized_engines.clear()
            del second, other_engine
            gc.collect()
            print("MODULE_LIFETIME all_released", flush=True)

    def test_exact_block_scale_division(self):
        """Match div.rn bits for every FP16 value and positive finite E4M3 scale."""
        reference = cp.RawKernel(r'''
#include <cuda_fp16.h>
#include <cuda_fp8.h>
extern "C" __global__ void compare_division(const float* output,
                                           unsigned int* mismatches) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 65536u * 126u) return;
    float value = __half2float(__ushort_as_half(index & 65535u));
    float scale = __half2float(
        __nv_cvt_fp8_to_halfraw(1u + (index >> 16), __NV_E4M3));
    if (__float_as_uint(output[index]) !=
        __float_as_uint(__fdiv_rn(value, scale))) {
        atomicAdd(mismatches, 1u);
    }
}
''',
                                 "compare_division",
                                 options=("--std=c++17", "--ftz=false",
                                          "--fmad=false"))
        with self.stream:
            bits = np.arange(65536, dtype=np.uint16)
            values = cp.asarray(bits.view(np.float16).astype(np.float32))
            codes = np.arange(1, 127, dtype=np.int32)
            exponent, mantissa = codes >> 3, codes & 7
            scales = cp.asarray(
                np.where(exponent == 0, mantissa * 2.0**-9,
                         (1 + mantissa / 8) * 2.0**(exponent - 7)).astype(
                             np.float32))
            output = cp.empty(65536 * 126, dtype=cp.float32)
            tensors = [
                runtime.from_dlpack(x) for x in (values, scales, output)
            ]
            stream = cuda.CUstream(self.stream.ptr)
            compiled = cute.compile(launch_block_scale_division, *tensors,
                                    stream)
            compiled(*tensors, stream)
            mismatches = cp.zeros(1, dtype=cp.uint32)
            reference((32256, ), (256, ), (output, mismatches))
            actual = int(mismatches.get()[0])
        self.assertEqual(actual, 0,
                         "Block-scale division changed FP32 quotient bits")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quantized_engines = {}
        cls.down_compiled = None

    def quantized_launcher(self, arrays, m, n, k):
        if getattr(self, "dual_candidate", None) is not None:
            return self.candidate_launcher(arrays, m, n, k)
        if (n, k) not in self.quantized_engines:
            engine = self.build_engine(n, k, quantized=True)
            self.quantized_engines[n, k] = (engine,
                                            engine.create_execution_context())
        engine, context = self.quantized_engines[n, k]
        a, weights, sfa, sw, alpha, global_scale, output, output_scale = arrays
        self.assertTrue(context.set_input_shape("x", a.shape))
        self.assertTrue(context.set_input_shape("sx", sfa.shape))
        self.assertEqual(tuple(context.get_tensor_shape("output")),
                         (m, n // 2))
        self.assertEqual(tuple(context.get_tensor_shape("output_scale")),
                         output_scale.shape)
        for name, array in zip([
                "x", "weights", "sx", "sw", "alpha", "global_scale", "output",
                "output_scale"
        ], arrays):
            self.assertTrue(context.set_tensor_address(name, array.data.ptr))

        def launch():
            self.assertTrue(context.execute_async_v3(self.stream.ptr))

        return launch

    def candidate_launcher(self, arrays, m, n, k):
        a, weights, sfa, sw, alpha, global_scale, output, output_scale = arrays
        cluster_m, cluster_n, group, coalesced, overlap = self.dual_candidate
        types = [
            cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN, cutlass.Float16, cutlass.Float32,
            cutlass.Float8E4M3FN, cutlass.Float32
        ]
        pointers = [
            runtime.make_ptr(dtype,
                             array.data.ptr,
                             cute.AddressSpace.gmem,
                             assumed_align=16)
            for array, dtype in zip([
                a, weights, sfa, sw, output, alpha, output_scale, global_scale
            ], types)
        ]
        args = (*pointers[:6], cutlass.Int64(m), cutlass.Int64(2 * n),
                cutlass.Int64(k), 16, cutlass.Int32(self.active_clusters),
                cuda.CUstream(self.stream.ptr), *pointers[6:])
        if self.candidate_compiled is None:
            kernel = nvfp4.GemmBlackwellNvFp4WS(
                mma_tiler_mn=(128, 256),
                cluster_shape_mn=(cluster_m, cluster_n),
                fused_swiglu=True,
                quantized_output=True,
                n_tile_group_size=group,
                coalesced_output=coalesced,
                overlap_output_store=overlap,
                fallback_cluster_shape_mn=getattr(self,
                                                  "fallback_cluster_shape",
                                                  None))
            self.candidate_compiled = cute.compile(kernel.wrapper, *args)
        return lambda: self.candidate_compiled(*(args[:9] + args[10:]))

    def down_launcher(self, arrays, m, n, k):
        cluster_shape = getattr(self, "down_cluster_shape", (1, 1))
        types = [
            cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN, cutlass.Float16, cutlass.Float32
        ]
        pointers = [
            runtime.make_ptr(dtype,
                             array.data.ptr,
                             cute.AddressSpace.gmem,
                             assumed_align=16)
            for array, dtype in zip(arrays, types)
        ]
        args = (*pointers, cutlass.Int64(m), cutlass.Int64(n),
                cutlass.Int64(k), 16,
                cutlass.Int32(self.sm_count //
                              (cluster_shape[0] * cluster_shape[1])),
                cuda.CUstream(self.stream.ptr))
        if type(self).down_compiled is None:
            kernel = nvfp4.GemmBlackwellNvFp4WS(
                mma_tiler_mn=(128, getattr(self, "down_tile_n", 128)),
                cluster_shape_mn=cluster_shape,
                n_tile_group_size=4)
            type(self).down_compiled = cute.compile(kernel.wrapper, *args)
        return lambda: self.down_compiled(*(args[:9] + args[10:]))

    def run_quantized_case(self,
                           m,
                           n=17408,
                           k=5120,
                           global_value=0.125,
                           zero=False):
        with self.stream:
            rng = cp.random.RandomState(1265)
            a = rng.randint(0, 256, (m, k // 2), dtype=cp.uint8)
            g = rng.randint(0, 256, (n, k // 2), dtype=cp.uint8)
            u = rng.randint(0, 256, (n, k // 2), dtype=cp.uint8)
            sx = rng.randint(40, 57, (m, k // 16), dtype=cp.uint8)
            sg = rng.randint(32, 49, (n, k // 16), dtype=cp.uint8)
            su = rng.randint(48, 65, (n, k // 16), dtype=cp.uint8)
            sx[:, ::7] = 0
            if zero:
                sx.fill(0)
            weights, sfa, sw = interleave(g, u), sf_atom(sx), sf_atom(
                interleave(sg, su))
            alpha = cp.asarray([0.03125, 0.0078125], dtype=cp.float32)
            h = cp.empty((m, n), dtype=cp.float16)
            fp16_launch = self.make_launcher([a, weights, sfa, sw, h, alpha],
                                             m, n, k)
            fp16_launch()
            global_scale = cp.asarray([global_value], dtype=cp.float32)
            guard = cp.full(m * n // 2 + 256, 0xAB, dtype=cp.uint8)
            q = guard[128:-128].reshape(m, n // 2)
            sf_size = (m + 127) // 128 * 128 * (n // 16)
            sf_guard = cp.full(sf_size + 256, 0xAB, dtype=cp.uint8)
            sh = sf_guard[128:-128]
            launch = self.quantized_launcher(
                [a, weights, sfa, sw, alpha, global_scale, q, sh], m, n, k)
            launch()
            q_eager, sh_eager = q.copy(), sh.copy()
            self.stream.synchronize()
            self.stream.begin_capture()
            launch()
            graph = self.stream.end_capture()
            q.fill(0xAB)
            sh.fill(0xAB)
            graph.launch(self.stream)
            ref_q, ref_sf = quantize_reference(cp.asnumpy(h), global_value)
            ref_q, ref_sf = cp.asarray(ref_q), cp.asarray(ref_sf)
            self.assertTrue(
                bool((guard[:128] == 0xAB).all()
                     & (guard[-128:] == 0xAB).all()))
            self.assertTrue(
                bool((sf_guard[:128] == 0xAB).all()
                     & (sf_guard[-128:] == 0xAB).all()))
            cp.testing.assert_array_equal(q, ref_q)
            cp.testing.assert_array_equal(sh, sf_atom(ref_sf))
            cp.testing.assert_array_equal(q_eager, q)
            cp.testing.assert_array_equal(sh_eager, sh)
            down_w = rng.randint(0, 256, (k, n // 2), dtype=cp.uint8)
            sd = rng.randint(32, 49, (k, n // 16), dtype=cp.uint8)
            down_alpha = cp.asarray([global_value * 0.03125], dtype=cp.float32)
            out_guard = cp.full(m * k + 256, 123, dtype=cp.float16)
            out = out_guard[128:-128].reshape(m, k)
            down_sf = sf_atom(sd)
            down = self.down_launcher(
                [q, down_w, sh, down_sf, out, down_alpha], m, k, n)
            down()
            eager = out.copy()
            self.stream.synchronize()
            self.stream.begin_capture()
            launch()
            down()
            full_graph = self.stream.end_capture()
            q.fill(0xAB)
            sh.fill(0xAB)
            out.fill(cp.nan)
            full_graph.launch(self.stream)
            down_ref = (dequantize(q, ref_sf) @ dequantize(
                down_w, sd).T) * down_alpha[0]
            rel = float(
                cp.linalg.norm(out.astype(cp.float32) - down_ref) /
                cp.maximum(cp.linalg.norm(down_ref), 1e-20))
            self.assertLess(rel, 0.002)
            self.assertTrue(bool(cp.isfinite(out).all()))
            cp.testing.assert_allclose(out.astype(cp.float32),
                                       down_ref,
                                       rtol=0.003,
                                       atol=0.05)
            cp.testing.assert_array_equal(eager, out)
            self.assertTrue(
                bool((out_guard[:128] == 123).all()
                     & (out_guard[-128:] == 123).all()))
            x32 = dequantize(a, sx)
            gate32 = (x32 @ dequantize(g, sg).T) * alpha[0]
            up32 = (x32 @ dequantize(u, su).T) * alpha[1]
            full_h = (gate32 / (1 + cp.exp(-gate32)) * up32).astype(cp.float16)
            fq, fs = quantize_reference(cp.asnumpy(full_h), global_value)
            fq, fs = cp.asarray(fq), cp.asarray(fs)
            full_ref = (
                dequantize(fq, fs) @ dequantize(down_w, sd).T) * down_alpha[0]
            full_rel = float(
                cp.linalg.norm(out.astype(cp.float32) - full_ref) /
                cp.maximum(cp.linalg.norm(full_ref), 1e-20))
            self.assertLess(full_rel, 0.02)
            print(json.dumps(
                dict(kind="quantized_mlp",
                     m=m,
                     n=n,
                     k=k,
                     global_scale=global_value,
                     zero=zero,
                     packed_exact=True,
                     sf_exact=True,
                     graph_exact=True,
                     down_rel_l2=rel,
                     complete_mlp_rel_l2=full_rel)),
                  flush=True)

    def test_tail_and_zero_blocks(self):
        """Quantized H and down GEMM preserve padding, guards and QDQ across partial M tiles."""
        for m in (1, 8, 127, 128, 129, 257):
            self.run_quantized_case(m, 256, 256)
        self.run_quantized_case(129, 256, 256, zero=True)

    def test_scale_range(self):
        """Output scales handle saturation and subnormal/zero rounding with nearest-even FP4 packing."""
        for scale in (2.0**-12, 2.0**-4, 2.0**4, 2.0**16):
            self.run_quantized_case(129, 256, 256, global_value=scale)

    def test_qwen35_dense_mlp_shape(self):
        """Serialized quantized plugin and down GEMM match complete QDQ for every issue M point."""
        for m in (1, 8, 64, 256, 1024, 4096, 8192, 16384, 32768):
            self.run_quantized_case(m)

    def test_prefill_group_boundaries(self):
        """Tile grouping preserves QDQ, graph replay and guards at M/N group boundaries."""
        for m, n in ((3968, 512), (3969, 512), (4096, 512), (4097, 512),
                     (4097, 384), (4097, 256)):
            self.run_quantized_case(m, n, 256)

    def test_multicast_coalesced_dual(self):
        """Coalesced FP4/SF stores and multicast preserve QDQ, guards and graph replay at cluster tails."""
        from cutlass import utils

        try:
            configs = ((1, 1, 4, True, False), (1, 2, 4, False, False),
                       (1, 2, 4, True, False), (1, 2, 2, True, False),
                       (1, 4, 1, True, False), (1, 1, 4, True, True),
                       (1, 2, 4, True, True), (2, 1, 4, False, False),
                       (2, 1, 4, True, False), (2, 1, 4, True,
                                                True), (2, 2, 4, True, True))
            for config in configs:
                self.dual_candidate = config
                cluster_m, cluster_n = config[:2]
                self.active_clusters = utils.HardwareInfo(
                ).get_max_active_clusters(cluster_m * cluster_n)
                self.candidate_compiled = None
                for m, n in ((1, 256), (129, 384), (4097, 640)):
                    self.run_quantized_case(m, n, 256)
                self.run_quantized_case(129, 384, 256, zero=True)
        finally:
            self.dual_candidate = None
            self.candidate_compiled = None

    def test_tactic_profile_endpoints(self):
        """Given a tuned engine, reloading preserves QDQ at profile MIN, OPT and MAX."""
        key = (384, 256)
        previous = self.quantized_engines.get(key)
        try:
            engine = self.build_engine(*key,
                                       quantized=True,
                                       opt_m=129,
                                       max_m=257)
            for _ in range(2):
                self.quantized_engines[key] = (
                    engine, engine.create_execution_context())
                for m in (1, 129, 257):
                    self.run_quantized_case(m, *key)
                engine = self.runtime.deserialize_cuda_engine(
                    engine.serialize())
                self.assertIsNotNone(engine)
        finally:
            self.quantized_engines.pop(key, None)
            if previous is not None:
                self.quantized_engines[key] = previous

    def test_mixed_cluster_dual(self):
        """Given mixed clusters, graph replay preserves QDQ and guards at M/N tails."""
        try:
            for cluster in ((2, 2), (4, 1)):
                self.dual_candidate = (*cluster, 4, True, True)
                self.fallback_cluster_shape = (2, 1)
                self.active_clusters = self.sm_count // 4
                self.candidate_compiled = None
                for m, n in ((1, 128), (127, 384), (129, 256), (4097, 640)):
                    self.run_quantized_case(m, n, 256)
                self.run_quantized_case(129, 384, 256, zero=True)
        finally:
            self.dual_candidate = None
            self.fallback_cluster_shape = None
            self.candidate_compiled = None

    def test_clustered_down_group_boundaries(self):
        """Cluster grouping preserves QDQ, graph replay and guards across partial M/N groups."""
        saved_compiled = type(self).down_compiled
        self.down_tile_n = 256
        self.down_cluster_shape = (2, 1)
        type(self).down_compiled = None
        try:
            for m, n, k in ((1, 256, 256), (129, 256, 256), (4097, 256, 768),
                            (4097, 384, 1280), (4225, 512, 2304)):
                self.run_quantized_case(m, n, k)
        finally:
            type(self).down_compiled = saved_compiled


if __name__ == "__main__":
    unittest.main()
