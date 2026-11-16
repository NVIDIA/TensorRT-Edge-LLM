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
"""Generate a video with stereo audio from a text prompt with MiniMax-H3.

Builds the engines on the first run, then runs the stage graph and writes
``<output>.wav`` and, with ffmpeg on the PATH, ``<output>.mp4``; otherwise the
frames are left in ``<output>.rgb``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

from tensorrt_edgellm import cli, runtime
from tensorrt_edgellm._native.load import load_runtime_from_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="A red paper airplane flies through a sunlit classroom.")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("minimax_h3"))
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    return parser.parse_args()


def _load_runtime(build_dir: Path):
    """The native runtime of an installed wheel, else of a source build."""
    try:
        return runtime.load()
    except runtime.NativeRuntimeError:
        extension = next((build_dir / "lib").glob("_edgellm_runtime*.so"),
                         None)
        if extension is None:
            raise
        return load_runtime_from_path(extension)


def main() -> None:
    args = _arguments()
    if not (args.engine_dir / "multistage_config.json").is_file():
        build = ["build", "--model-dir", str(args.checkpoint)]
        cli.main(build + ["--engine-dir", str(args.engine_dir)])

    graph = _load_runtime(args.build_dir).MultiStageRuntime(
        str(args.engine_dir))
    style = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames
    }
    result = graph.generate(args.prompt,
                            style=json.dumps(style),
                            sampling=json.dumps({"seed": args.seed}))
    print(f"{result.video_frames} frames of {result.video_width}x"
          f"{result.video_height} at {result.video_fps} fps; stages "
          f"{result.stages}")

    audio = args.output.with_suffix(".wav")
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(result.audio_channels)
        handle.setsampwidth(2)
        handle.setframerate(result.audio_sample_rate)
        handle.writeframes(result.audio_pcm16)

    frames = args.output.with_suffix(".rgb")
    frames.write_bytes(result.video_rgb8)
    if shutil.which("ffmpeg") is None:
        print(f"Wrote {frames} (raw RGB) and {audio}")
        return
    video = args.output.with_suffix(".mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
        "rgb24", "-video_size", f"{result.video_width}x{result.video_height}",
        "-framerate",
        str(result.video_fps), "-i",
        str(frames), "-i",
        str(audio), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(video)
    ],
                   check=True)
    frames.unlink()
    print(f"Wrote {video} and {audio}")


if __name__ == "__main__":
    main()
