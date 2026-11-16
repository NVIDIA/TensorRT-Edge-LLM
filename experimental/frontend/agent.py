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
"""Tools and system prompt for ``edgellm chat --agentic``."""

import json
import pathlib
import re
import shutil
import subprocess
import threading

MAX_TOOL_ROUNDS = 3
TIMEOUT_SECONDS = 5
SYSTEM_PROMPT = (
    "You can call the device_status tool to read live GPU memory usage, "
    "utilization, and temperature of the device you are running on. Call it "
    "when the user asks about the device, then answer from its result. Null "
    "fields are unavailable on this device. When present, memory_reserved_mib "
    "is free memory in the huge-page pool reserved for GPU allocations.")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "device_status",
        "description":
        "Return GPU memory usage, utilization, and temperature as JSON.",
        "parameters": {
            "type": "object",
            "properties": {}
        },
    },
}]
FINAL_ROUND_MESSAGE = {
    "role":
    "user",
    "content":
    "No more tool calls are available. Answer now from the tool results above.",
}
NVIDIA_SMI_FIELDS = ("name", "memory.used", "memory.total", "utilization.gpu",
                     "temperature.gpu")


def with_system_prompt(messages):
    """Append the tool instructions to the first system message, or add one."""
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0],
                     content=messages[0]["content"] + "\n\n" + SYSTEM_PROMPT)
        return [first] + messages[1:]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def _cuda_pci_bus_id():
    """Return the PCI bus ID of the current CUDA device, or None."""
    from cuda.bindings import runtime

    status, device = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
        return None
    status, bus_id = runtime.cudaDeviceGetPCIBusId(32, device)
    if status != runtime.cudaError_t.cudaSuccess:
        return None
    return bus_id.split(b"\0")[0].decode("ascii")


def _from_nvidia_smi():
    bus_id = _cuda_pci_bus_id()
    # nvidia-smi and CUDA can enumerate GPUs in different orders.
    selector = ["--id=" + bus_id] if bus_id else []
    output = subprocess.run(
        [
            "nvidia-smi", *selector, "--query-gpu=" +
            ",".join(NVIDIA_SMI_FIELDS), "--format=csv,noheader,nounits"
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=True,
    ).stdout
    name, used, total, utilization, temperature = (
        field.strip() for field in output.splitlines()[0].split(","))
    return {
        "device": name,
        "memory_used_mib": _number(used),
        "memory_total_mib": _number(total),
        "utilization_percent": _number(utilization),
        "temperature_c": _number(temperature),
        "unified_memory": False,
    }


def _free_huge_pages_mib(meminfo):
    free = re.search(r"^HugePages_Free:\s+(\d+)", meminfo, re.MULTILINE)
    size = re.search(r"^Hugepagesize:\s+(\d+) kB", meminfo, re.MULTILINE)
    return int(free.group(1)) * int(
        size.group(1)) / 1024 if free and size else 0


def parse_tegrastats(line, meminfo=""):
    """Parse one tegrastats sample; GPU memory is the shared system RAM.

    tegrastats counts the whole huge-page pool as used, so unused huge pages
    from ``meminfo`` (``/proc/meminfo`` text) are reported as reserved instead.
    """
    memory = re.search(r"\bRAM (\d+)/(\d+)MB", line)
    utilization = re.search(r"\b(?:GR3D|IGPU)_FREQ (\d+)%", line)
    temperature = re.search(r"\bgpu@(-?[\d.]+)C", line, re.IGNORECASE)
    if memory is None:
        raise ValueError("Unrecognized tegrastats output")
    reserved = _free_huge_pages_mib(meminfo)
    return {
        "memory_used_mib": float(memory.group(1)) - reserved,
        "memory_reserved_mib": reserved,
        "memory_total_mib": float(memory.group(2)),
        "utilization_percent":
        float(utilization.group(1)) if utilization else None,
        "temperature_c": float(temperature.group(1)) if temperature else None,
        "unified_memory": True,
    }


def _from_tegrastats():
    process = subprocess.Popen(["tegrastats", "--interval", "100"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL,
                               text=True)
    timer = threading.Timer(TIMEOUT_SECONDS, process.kill)
    timer.start()
    try:
        # Without root, tegrastats prints a banner before the first sample.
        line = next((line for line in process.stdout if "RAM " in line), "")
    finally:
        timer.cancel()
        process.kill()
        process.wait()
    model = pathlib.Path("/proc/device-tree/model")
    name = (model.read_text(
        errors="replace").rstrip("\0") if model.is_file() else "NVIDIA Tegra")
    meminfo = pathlib.Path("/proc/meminfo").read_text()
    return {"device": name, **parse_tegrastats(line, meminfo)}


def device_status():
    """Sample the GPU with nvidia-smi, or tegrastats on Tegra without NVML."""
    if shutil.which("nvidia-smi"):
        return _from_nvidia_smi()
    if shutil.which("tegrastats"):
        return _from_tegrastats()
    raise RuntimeError("Neither nvidia-smi nor tegrastats is available")


def call(name):
    """Run one tool call and return its JSON result, reporting failures as data."""
    if name != "device_status":
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = device_status()
        except (OSError, ValueError, RuntimeError,
                subprocess.SubprocessError) as error:
            result = {"error": str(error)}
    return json.dumps(result)
