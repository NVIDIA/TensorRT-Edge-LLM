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
"""Summaries shared by benchmark CLI and server pressure tests."""

import math
from collections import Counter


def distribution(values):
    """Return interpolated percentiles; an empty sample has null statistics."""
    values = sorted(values)
    if not values:
        return {key: None for key in ('p50', 'p95', 'p99', 'max')}

    def percentile(q):
        position = (len(values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        **{
            f'p{q}': percentile(q / 100)
            for q in (50, 95, 99)
        }, 'max': values[-1]
    }


def summarize(records, elapsed_s):
    """Measure success throughput over the full measurement and drain interval."""
    counts = Counter(row['outcome'] for row in records)
    success = [row for row in records if row['outcome'] == 'success']
    completed_tokens = [row['output_tokens'] for row in success]
    tokens = sum(completed_tokens) if all(t is not None
                                          for t in completed_tokens) else None
    return {
        'requests':
        len(records),
        'outcomes':
        dict(counts),
        'elapsed_s':
        elapsed_s,
        'success_rps':
        len(success) / elapsed_s if elapsed_s > 0 else None,
        'output_tokens':
        tokens,
        'output_tokens_per_s':
        tokens / elapsed_s if tokens is not None and elapsed_s > 0 else None,
        'error_rate':
        counts['error'] / (len(success) + counts['error']) if len(success) +
        counts['error'] else None,
        'latency_s':
        distribution([row['latency_s'] for row in success]),
        'ttft_s':
        distribution([
            row['ttft_s'] for row in success if row.get('ttft_s') is not None
        ]),
        'http_status_counts':
        dict(Counter(str(row.get('status')) for row in records)),
    }


def summarize_events(events):
    """Summarize only measured requests from the shared online driver."""
    start = next((e['timestamp_ns']
                  for e in events if e['event'] == 'measurement_started'),
                 None)
    end = next((e['timestamp_ns']
                for e in events if e['event'] == 'measurement_ended'), None)
    records = []
    for event in events:
        if event.get('phase') != 'measurement' or event['event'] not in (
                'request_completed', 'request_failed'):
            continue
        records.append({
            'status':
            event.get('status'),
            'outcome':
            'success' if event['event'] == 'request_completed' else 'error',
            'latency_s':
            (event['completion_ns'] - event['timestamp_ns']) / 1e9,
            'ttft_s':
            ((event['first_token_ns'] - event['timestamp_ns']) /
             1e9 if event.get('first_token_ns') is not None else None),
            'output_tokens':
            event.get('output_tokens'),
        })
    elapsed = (end -
               start) / 1e9 if start is not None and end is not None else 0
    result = summarize(records, elapsed)
    result['run_complete'] = bool(events
                                  and events[-1]['event'] == 'run_ended')
    return result
