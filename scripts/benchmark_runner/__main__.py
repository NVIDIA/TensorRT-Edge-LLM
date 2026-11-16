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
"""Run the existing HTTP benchmark driver: python -m scripts.benchmark_runner."""

import argparse
import json
import os
from pathlib import Path

from . import online
from .adapters.openai import CompletionsAdapter
from .metrics import summarize_events


def main(argv=None):
    """Write request events and a summary, returning nonzero for failed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--ready-timeout', type=float, default=300)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    events_path = args.output / 'events.jsonl'
    if events_path.exists():
        parser.error(
            'Use a fresh output directory to preserve previous measurements')
    headers = {}
    if os.environ.get('SERVER_BENCH_API_KEY'):
        headers[
            'Authorization'] = 'Bearer ' + os.environ['SERVER_BENCH_API_KEY']
    adapter = CompletionsAdapter({
        'endpoint': args.url,
        'model': args.model,
        'headers': headers
    })
    error = None
    try:
        workload = json.loads(args.workload.read_text())
        if not isinstance(workload, dict):
            raise ValueError('workload must be a JSON object')
        online.run(adapter, workload, events_path, args.request_timeout,
                   args.ready_timeout)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    events = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ] if events_path.exists() else []
    summary = summarize_events(events)
    summary['error'] = error
    (args.output /
     'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 1 if error or not summary['run_complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
