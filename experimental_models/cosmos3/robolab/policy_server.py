#!/usr/bin/env python3
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
"""Edge-LLM Cosmos3 policy server (HTTP + JSON).

Exposes the RoboLab observation -> action-chunk contract over HTTP so a RoboLab
``InferenceClient`` (Isaac Lab / Isaac Sim, x86-only) can drive the Edge-LLM
Cosmos3 policy running on a remote target.

Contract (all JSON):
  POST /infer
    request : {"image": <base64-PNG> | <path>, "instruction": <str>,
               "domain": <str, optional>, "steps": <int, optional>}
    response: {"action": [[...8 floats] x32], "shape": [1,32,8],
               "dtype": "float32", "domain": "...", "num_inference_steps": N,
               "finite": bool, "meta": {...}}
  GET  /metadata -> {"action_chunk_size":32, "raw_action_dim":8, "domain":..., ...}
  GET  /healthz  -> {"status":"ok"}
  POST /reset    -> {"status":"reset successful"}  (stateless; provided for parity)

The server is a thin wrapper around the ``cosmos3_policy_inference`` binary: it
writes the image to a temp PNG, runs the CLI with ``--output <tmp>.json`` (the
default JSON I/O interface), and returns the parsed JSON action. No heavy Python
deps beyond the standard library + Pillow (image decode) are required.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger("cosmos3.robolab.server")

ACTION_CHUNK_SIZE = 32
RAW_ACTION_DIM = 8


class Cosmos3PolicyBackend:
    """Runs the Edge-LLM Cosmos3 policy worker and returns one action chunk."""

    _READY_PREFIX = "@@EDGELLM_READY "
    _RESPONSE_PREFIX = "@@EDGELLM_RESPONSE "

    def __init__(
        self,
        binary: str,
        engine_dir: str,
        domain: str = "droid_lerobot",
        steps: int = 4,
        guidance: float = 3.0,
        viewpoint: str = "concat_view",
        action_chunk_size: int = ACTION_CHUNK_SIZE,
        seed: int = 0,
        deterministic_seed: bool = False,
        extra_env: dict | None = None,
        persistent: bool = True,
        worker_timeout_s: float = 300.0,
    ) -> None:
        self.binary = binary
        self.engine_dir = engine_dir
        self.domain = domain
        self.steps = steps
        self.guidance = guidance
        self.viewpoint = viewpoint
        self.action_chunk_size = action_chunk_size
        self.extra_env = extra_env or {}
        self.persistent = persistent
        if worker_timeout_s <= 0:
            raise ValueError("worker_timeout_s must be positive")
        self.worker_timeout_s = worker_timeout_s
        self.seed = seed
        self.deterministic_seed = deterministic_seed
        # Match cosmos-framework's RoboLab server: --seed initializes one
        # NumPy generator and each request draws a new diffusion seed.
        self._rng = np.random.default_rng(seed)
        self._rng_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._worker: subprocess.Popen[str] | None = None
        self._worker_messages: queue.Queue[str | None] | None = None
        self._worker_reader: threading.Thread | None = None
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"cosmos3_policy_inference binary not found: {binary}")
        if not os.path.isdir(engine_dir):
            raise FileNotFoundError(f"engine dir not found: {engine_dir}")
        self._validate_engine_contract()
        if self.persistent:
            self._start_worker()
            atexit.register(self.close)

    def _base_command(self) -> list[str]:
        return [
            self.binary,
            "--engineDir",
            self.engine_dir,
            "--domain",
            self.domain,
            "--steps",
            str(self.steps),
            "--seed",
            str(self.seed),
            "--guidance",
            str(self.guidance),
            "--viewPoint",
            self.viewpoint,
            "--action-chunk-size",
            str(self.action_chunk_size),
        ]

    def _read_worker_message(self, prefix: str) -> dict:
        if self._worker is None or self._worker_messages is None:
            raise RuntimeError("Cosmos3 policy worker is not running")
        deadline = time.monotonic() + self.worker_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Cosmos3 policy worker timed out waiting for {prefix.strip()} "
                    f"after {self.worker_timeout_s:g}s")
            try:
                line = self._worker_messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Cosmos3 policy worker timed out waiting for {prefix.strip()} "
                    f"after {self.worker_timeout_s:g}s") from exc
            if line is None:
                code = self._worker.poll()
                raise RuntimeError(
                    f"Cosmos3 policy worker exited before {prefix.strip()} (rc={code})"
                )
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])
            logger.debug("policy worker: %s", line.rstrip())

    @staticmethod
    def _collect_worker_output(worker: subprocess.Popen[str],
                               messages: queue.Queue[str | None]) -> None:
        assert worker.stdout is not None
        for line in worker.stdout:
            messages.put(line)
        messages.put(None)

    def _start_worker(self) -> None:
        env = dict(os.environ)
        env.update(self.extra_env)
        self._worker = subprocess.Popen(
            [*self._base_command(), "--request-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._worker_messages = queue.Queue()
        self._worker_reader = threading.Thread(
            target=self._collect_worker_output,
            args=(self._worker, self._worker_messages),
            daemon=True,
        )
        self._worker_reader.start()
        try:
            self._read_worker_message(self._READY_PREFIX)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        worker, self._worker = self._worker, None
        reader, self._worker_reader = self._worker_reader, None
        self._worker_messages = None
        if worker is None:
            return
        if worker.stdin is not None:
            try:
                worker.stdin.close()
            except BrokenPipeError:
                pass
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        if reader is not None:
            reader.join(timeout=1)

    def _restart_worker(self) -> None:
        self.close()
        self._start_worker()

    def _validate_engine_contract(self) -> None:
        """Fail before serving if the engine was built for another recipe."""
        config_path = os.path.join(self.engine_dir, "gen", "config.json")
        with open(config_path) as fh:
            config = json.load(fh)
        chunk = int(config["action_chunk_size"])
        fps = float(config["fps"])
        raw_action_dim = int(config["raw_action_dim"])
        state_rows = int(config.get("state_rows", 0))
        use_state = bool(config.get("use_state", False))
        action_offset = int(config["action_start_frame_offset"])
        if chunk != self.action_chunk_size:
            raise ValueError(
                f"engine action_chunk_size={chunk}, expected {self.action_chunk_size}"
            )
        if abs(fps - 15.0) > 1e-6:
            raise ValueError(f"engine fps={fps}, expected 15")
        if (raw_action_dim != RAW_ACTION_DIM or state_rows != 1
                or not use_state or action_offset != 0):
            raise ValueError(
                "Policy-DROID engine must have raw_action_dim=8, use_state=true, "
                "one clean state row, and action_start_frame_offset=0; got "
                f"raw_action_dim={raw_action_dim}, use_state={use_state}, "
                f"state_rows={state_rows}, action_start_frame_offset={action_offset}"
            )

    def _decode_image_to_png(self, image_field, tmpdir: str) -> str:
        """Materialize the request image to a PNG path.

        Accepts either an existing file path, a base64-encoded PNG/JPG, or a
        base64-encoded raw HxWx3 uint8 array (data-URI style not required).
        """
        # Case 1: an existing path on the server host.
        if isinstance(image_field, str) and os.path.isfile(image_field):
            return image_field
        # Case 2: base64 string (optionally a data URI).
        if isinstance(image_field, str):
            b64 = image_field.split(
                ",", 1)[1] if image_field.startswith("data:") else image_field
            raw = base64.b64decode(b64)
            try:
                from PIL import Image

                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                # Fall back to treating it as a raw PNG blob written to disk.
                path = os.path.join(tmpdir, "obs.png")
                with open(path, "wb") as fh:
                    fh.write(raw)
                return path
            path = os.path.join(tmpdir, "obs.png")
            img.save(path)
            return path
        # Case 3: a nested list / array (HxWx3 uint8).
        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(image_field, dtype=np.uint8)
            path = os.path.join(tmpdir, "obs.png")
            Image.fromarray(arr).save(path)
            return path
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Unsupported 'image' field: {exc!r}") from exc

    def infer(self, request: dict) -> dict:
        instruction = request.get("instruction") or request.get("prompt")
        if not instruction:
            raise ValueError("request missing 'instruction'")
        domain = request.get("domain", self.domain)
        steps = int(request.get("steps", self.steps))
        state = request.get("state")
        if not isinstance(state, list) or len(state) != RAW_ACTION_DIM:
            raise ValueError("request 'state' must contain 8 values")
        if request.get("seed") is not None:
            seed = int(request["seed"])
        elif self.deterministic_seed:
            seed = self.seed
        else:
            with self._rng_lock:
                seed = int(self._rng.integers(0, 2**31))
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._decode_image_to_png(request["image"], tmpdir)
            out_path = os.path.join(tmpdir, "action.json")
            cmd = [
                *self._base_command(),
                "--image",
                image_path,
                "--prompt",
                instruction,
                "--domain",
                domain,
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--warmup",
                "0",
                "--iters",
                "1",
                "--state",
                ",".join(str(float(value)) for value in state),
                "--output",
                out_path,
            ]
            t0 = time.perf_counter()
            if self.persistent:
                payload = {
                    "image": image_path,
                    "prompt": instruction,
                    "state": state,
                    "output": out_path,
                    "domain": domain,
                    "steps": steps,
                    "seed": seed,
                    "guidance": self.guidance,
                    "viewpoint": self.viewpoint,
                }
                with self._worker_lock:
                    try:
                        if self._worker is None or self._worker.stdin is None:
                            raise RuntimeError(
                                "Cosmos3 policy worker is not running")
                        self._worker.stdin.write(json.dumps(payload) + "\n")
                        self._worker.stdin.flush()
                        response = self._read_worker_message(
                            self._RESPONSE_PREFIX)
                    except Exception:
                        logger.exception(
                            "policy worker failed; restarting it for the next request"
                        )
                        try:
                            self._restart_worker()
                        except Exception:
                            logger.exception("policy worker restart failed")
                        raise
                if not response.get("ok"):
                    raise RuntimeError("cosmos3_policy_inference failed: "
                                       f"{response.get('error', response)}")
            else:
                env = dict(os.environ)
                env.update(self.extra_env)
                proc = subprocess.run(cmd,
                                      capture_output=True,
                                      text=True,
                                      check=False,
                                      timeout=self.worker_timeout_s,
                                      env=env)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"cosmos3_policy_inference failed (rc={proc.returncode}):\n"
                        f"{proc.stderr[-2000:]}")
            latency = time.perf_counter() - t0
            if not os.path.isfile(out_path):
                raise RuntimeError(
                    "cosmos3_policy_inference did not write its action output")
            with open(out_path) as fh:
                result = json.load(fh)
        result.setdefault("meta", {})["server_latency_s"] = latency
        result["meta"]["seed"] = seed
        return result


class _Handler(BaseHTTPRequestHandler):
    backend: Cosmos3PolicyBackend  # set on the server class

    def log_message(self, fmt, *args):  # keep the stdlib server quiet-ish
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.rstrip("/") == "/metadata":
            self._send_json(
                200,
                {
                    "policy": "edge-cosmos3",
                    "action_chunk_size": ACTION_CHUNK_SIZE,
                    "raw_action_dim": RAW_ACTION_DIM,
                    "action_shape": [ACTION_CHUNK_SIZE, RAW_ACTION_DIM],
                    "domain": self.server.backend.domain,
                    "num_inference_steps": self.server.backend.steps,
                    "guidance": self.server.backend.guidance,
                    "viewpoint": self.server.backend.viewpoint,
                },
            )
        else:
            self._send_json(404, {
                "type": "error",
                "message": f"no route {self.path}"
            })

    def do_POST(self):  # noqa: N802
        route = self.path.rstrip("/")
        try:
            req = self._read_json()
        except Exception as exc:
            self._send_json(400, {
                "type": "error",
                "message": f"bad JSON: {exc!r}"
            })
            return
        if route == "/reset":
            # Stateless server; reset is a no-op provided for client parity.
            self._send_json(200, {"status": "reset successful"})
            return
        if route == "/infer":
            try:
                result = self.server.backend.infer(req)
            except Exception as exc:
                logger.exception("inference failed")
                self._send_json(500, {"type": "error", "message": str(exc)})
                return
            self._send_json(200, result)
            return
        self._send_json(404, {
            "type": "error",
            "message": f"no route {self.path}"
        })


def serve(backend: Cosmos3PolicyBackend, host: str,
          port: int) -> ThreadingHTTPServer:
    handler = _Handler
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.backend = backend  # attach for the handler
    logger.info(
        "Cosmos3 policy server listening on http://%s:%d (engine_dir=%s)",
        host, port, backend.engine_dir)
    return httpd


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Edge-LLM Cosmos3 RoboLab policy server (HTTP+JSON)")
    ap.add_argument("--binary",
                    required=True,
                    help="path to cosmos3_policy_inference")
    ap.add_argument(
        "--engine-dir",
        required=True,
        help="Cosmos3 engine dir (und_prefill/, vae_encoder/, gen/, ...)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--domain", default="droid_lerobot")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--viewpoint", default="concat_view")
    ap.add_argument("--action-chunk-size", type=int, default=ACTION_CHUNK_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker-timeout", type=float, default=300.0)
    ap.add_argument("--deterministic-seed",
                    action="store_true",
                    help="reuse --seed for every request instead of advancing "
                    "the request RNG")
    args = ap.parse_args()

    backend = Cosmos3PolicyBackend(args.binary,
                                   args.engine_dir,
                                   args.domain,
                                   args.steps,
                                   args.guidance,
                                   args.viewpoint,
                                   args.action_chunk_size,
                                   args.seed,
                                   args.deterministic_seed,
                                   worker_timeout_s=args.worker_timeout)
    httpd = serve(backend, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
