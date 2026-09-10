# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Streaming tool-call parsing: hold-back, JSON scanner, and state machine.

Set TOOL_STREAM_TRIALS to raise the random-split trial count locally.
"""

import json
import os
import random

from experimental.server.parsing.tool_calling import (_GATE_CAP, _OPEN_TOKENS,
                                                      _JsonScanner,
                                                      _longest_partial_token,
                                                      _select_parser,
                                                      validate_tool_request)

_TRIALS = int(os.environ.get("TOOL_STREAM_TRIALS", "30"))
_MAX_HOLD = max(len(token) for token in _OPEN_TOKENS) - 1


def _config():
    return validate_tool_request(
        [{
            "role": "user",
            "content": "hi"
        }],
        [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "w",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string"
                        },
                        "base": {
                            "type": "integer"
                        },
                    },
                },
            },
        }],
        "auto",
    )


def test_hold_back_prefixes_and_bounds():
    for token in _OPEN_TOKENS:
        for length in range(1, len(token)):
            prefix = token[:length]
            assert _longest_partial_token(prefix, _OPEN_TOKENS) == length
            assert _longest_partial_token("text " + prefix,
                                          _OPEN_TOKENS) == length
        # A complete token is the marker search's job, not hold-back's.
        assert _longest_partial_token("abc" + token, _OPEN_TOKENS) == 0
    for text in ["", "hello world", ">", "]", "tool_call>", "[tool_call"]:
        assert _longest_partial_token(text, _OPEN_TOKENS) == 0
    # Ambiguity resolves to the longest live prefix; only the rightmost
    # restart counts.
    assert _longest_partial_token("x<tool_call", _OPEN_TOKENS) == 10
    assert _longest_partial_token("<t<tool_", _OPEN_TOKENS) == 6
    rng = random.Random(719)
    alphabet = "<>/[]_ TOOLCALSFUNCtoolcalsfunc=abc"
    for _ in range(300):
        text = "".join(
            rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        held = _longest_partial_token(text, _OPEN_TOKENS)
        assert 0 <= held <= _MAX_HOLD
        for length in range(held + 1, min(len(text), _MAX_HOLD) + 1):
            tail = text[len(text) - length:]
            assert not any(
                len(tail) < len(token) and token.startswith(tail)
                for token in _OPEN_TOKENS)


def _scan(text):
    scanner = _JsonScanner()
    consumed = 0
    result = "ok"
    for ch in text:
        result = scanner.push(ch)
        consumed += 1
        if result in ("done", "error"):
            break
    return result, consumed, scanner


def test_scanner_accepts_strict_json():
    for payload in [
            '{"city": "Shanghai"}',
            '{}',
            '[]',
            '{"a": 1, "b": [true, false, null], "c": {"d": "x"}}',
            '{"a": -1.5e-3}',
            '{"code": "line1\\nline2 \\"quoted\\" \\u00e9"}',
            '[1, "two", {"three": 3}]',
            '{ "spaced" :\t"out" }',
            '{"a": "brace } inside", "b": 1}',
    ]:
        result, consumed, scanner = _scan(payload)
        assert result == "done" and consumed == len(payload), payload
        assert scanner.done and scanner.confirmed


def test_scanner_rejects_dialect_violations():
    # At or before the first value byte: must reject while nothing has been
    # emitted yet (degrade to the buffered path), so confirmed stays False.
    for payload in [
            "{'city': 'x'}", '{city: 1}', '{"a": True}', '{"a": None}',
            '"scalar"', '{"a" 1}'
    ]:
        result, _, scanner = _scan(payload)
        assert result == "error" and not scanner.confirmed, payload
    # Past the first value byte the head may already be out: still
    # rejected, just no longer recallable.
    for payload in [
            '{"a": 1_000}', '{"a": 1,}', '{"a": "x" "b": 1}',
            '{"a": "bad \\x65"}', '{"a": "\\u00zk"}', '{"a": "raw\nnl"}',
            '{"a": 1]}', '{"a": 1.2.3}', '{"a": 1e+e2}', '{"a": 01}'
    ]:
        result, _, scanner = _scan(payload)
        assert result == "error" and scanner.confirmed, payload


def test_scanner_confirm_and_done_boundaries():
    scanner = _JsonScanner()
    for ch in '{"code": ':
        assert scanner.push(ch) == "ok"
        assert not scanner.confirmed  # a key alone must not confirm
    assert scanner.push('"') == "ok"  # first byte of a long string value...
    assert scanner.confirmed  # ...confirms without waiting for its close
    # After the arguments value closes, wrapper bytes must not leak.
    result, consumed, scanner = _scan('{"city": "X"}}</tool_call>')
    assert result == "done"
    assert consumed == len('{"city": "X"}')
    assert scanner.push("}") == "error"


def _stream_events(chunks, config):
    parser = _select_parser("/nonexistent", "generic").stream(config)
    out = []
    for chunk in chunks:
        out.extend(parser.feed(chunk))
    out.extend(parser.flush())
    return out


def _canon_args(text):
    """Compare arguments semantically (contract: json.loads-equal)."""
    try:
        return ("json", json.dumps(json.loads(text), sort_keys=True))
    except ValueError:
        return ("raw", text)


def _normalize_classic(events):
    merged = []
    for event in events:
        if event["type"] == "content":
            if event["text"]:
                if merged and merged[-1][0] == "content":
                    merged[-1] = ("content", merged[-1][1] + event["text"])
                else:
                    merged.append(("content", event["text"]))
        else:
            call = event["tool_call"]
            merged.append(("tool", call.name, _canon_args(call.arguments)))
    return merged


def _normalize_stream(events):
    merged = []
    names = {}
    args = {}
    done = {}
    for event in events:
        if event.kind == "tool_done":
            done[event.index] = done.get(event.index, 0) + 1
        if event.kind == "content":
            if event.text:
                if merged and merged[-1][0] == "content":
                    merged[-1] = ("content", merged[-1][1] + event.text)
                else:
                    merged.append(("content", event.text))
        elif event.kind == "tool_head":
            names[event.index] = event.name
            args[event.index] = ""
            merged.append(("slot", event.index))
        elif event.kind == "tool_args":
            args[event.index] += event.text
    assert all(done.get(index) == 1 for index in names), (names, done)
    return [(("tool", names[m[1]],
              _canon_args(args[m[1]])) if m[0] == "slot" else m)
            for m in merged]


_CORPUS = [
    "Just a plain answer with no tools involved at all.",
    'Let me check. <tool_call>{"name": "get_weather", "arguments": '
    '{"city": "Shanghai"}}</tool_call>',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "X"}}'
    "</tool_call> and after text",
    'a<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    'b<tool_call>{"name": "get_weather", "arguments": {"city": "Y"}}'
    "</tool_call>c",
    '<tool_calls>[{"name": "get_weather", "arguments": {"city": "A"}},'
    '{"name": "get_weather", "arguments": {"city": "B"}}]</tool_calls>',
    '<function=get_weather><parameter=city>Paris</parameter>'
    "<parameter=base>10</parameter></function>",
    '[TOOL_CALLS]{"name": "get_weather", "arguments": {"city": "M"}}',
    '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}] [/TOOL_CALLS] t',
    "<tool_call>not json at all</tool_call> mixed "
    '<tool_call>{"name": "unknown_fn", "arguments": {}}</tool_call>',
    'truncated <tool_call>{"name": "get_weather", "argu',
    'pre-confirm <tool_call>{"name": "get_weather", "arguments": {',
    '<tool_call>{"name": "get_weather", "arguments": {"cit',
    "half marker at end <tool_",
    "angle noise < tool < call > text",
    '{"name": "get_weather", "arguments": {"city": "bare"}}',
    'get_weather(city="Paris", base=10)',
    '```json\n{"name": "get_weather", "arguments": {}}\n```',
    "<function=>empty name</function> tail",
    '<function=<tool_call>x</tool_call>',
    "[TOOL_CAL not a real marker",
    "text ends exactly with marker <tool_call>",
    '<toolcall>{"name": "get_weather", "arguments": {}}</toolcall>',
    '<tool_call>{"name":"get_weather","arguments":{"city":"Z","base":3}}'
    "</tool_call>",
    '<tool_call>{"name": "get_weather", "arguments": []}</tool_call>',
    '<tool_call>{"arguments": {"city": "R"}, "name": "get_weather"}'
    "</tool_call>",
    '<tool_call>{"id": "x1", "name": "get_weather", "arguments": {}}'
    "</tool_call>",
    '<tool_call>{"name": "get_weather", "arguments": {"a": 1}, "note": 2}'
    "</tool_call> tail",
    '<toolcalls>{"name": "get_weather", "arguments": {}}</toolcalls>',
    '<function_call>{"name": "get_weather", "arguments": {}}</function_call>',
    '<function_calls>[{"name": "get_weather", "arguments": {}}]'
    "</function_calls>",
    '<tool_call>{"name": "get_weather", "arguments": '
    '{"city": "a\\"b\\u00e9\\n", "base": 12}}</tool_call>',
]


def test_stream_equals_classic_under_random_splits():
    config = _config()
    parser = _select_parser("/nonexistent", "generic")
    for text in _CORPUS:
        expected = _normalize_classic(parser.parse(text, config)[0])
        rng = random.Random(hash(text) & 0xffff)
        for trial in range(_TRIALS):
            if trial == 0:
                chunks = list(text)  # char-by-char
            elif trial == 1:
                chunks = [text]  # single feed
            else:
                chunks, pos = [], 0
                while pos < len(text):
                    step = rng.randrange(1, 9)
                    chunks.append(text[pos:pos + step])
                    pos += step
            got = _normalize_stream(_stream_events(chunks, config))
            assert got == expected, f"text={text!r} trial={trial}"


def test_plain_text_latency_bound():
    # Cumulative emitted content trails input by at most the hold-back
    # bound: the customer-facing "streams in real time" property.
    config = _config()
    parser = _select_parser("/nonexistent", "generic").stream(config)
    text = "This is a plain streamed answer with, punctuation and length."
    emitted = 0
    for position, ch in enumerate(text, start=1):
        for event in parser.feed(ch):
            emitted += len(event.text)
        if position > _GATE_CAP:  # the gate may hold the first few bytes
            assert emitted >= position - _MAX_HOLD
    total = emitted + sum(len(e.text) for e in parser.flush())
    assert total == len(text)


def test_content_streams_around_tool_block():
    config = _config()
    events = _stream_events([
        "before ",
        '<tool_call>{"name": "get_weather", "arguments": {}}',
        "</tool_call>",
        " after",
    ], config)
    kinds = [event.kind for event in events]
    assert kinds[0] == "content"  # "before " was not buffered
    assert "tool_head" in kinds and "tool_done" in kinds
    assert kinds[-1] == "content"  # " after" streams too


def test_raw_args_stream_incrementally():
    # The tier-2 marquee case: one long string argument must flow while the
    # model is still generating, byte-identical with the model's own text.
    config = _config()
    parser = _select_parser("/nonexistent", "generic").stream(config)
    payload = '{"city": "' + "x" * 600 + '"}'
    text = ('<tool_call>{"name": "get_weather", "arguments": ' + payload +
            "}</tool_call>")
    chunks = [text[i:i + 8] for i in range(0, len(text), 8)]
    fragments = []
    head_seen_before_args = None
    for chunk in chunks[:-1]:  # everything except the closing chunk
        for event in parser.feed(chunk):
            if event.kind == "tool_head" and head_seen_before_args is None:
                head_seen_before_args = not fragments
            if event.kind == "tool_args":
                fragments.append(event.text)
    assert head_seen_before_args is True
    assert len(fragments) > 5  # streamed, not one late flush
    tail = [
        event.text for chunk in chunks[-1:] for event in parser.feed(chunk)
        if event.kind == "tool_args"
    ]
    assert "".join(fragments + tail) == payload  # raw model bytes


def test_param_mode_streams_per_parameter():
    config = _config()
    parser = _select_parser("/nonexistent", "generic").stream(config)
    first = list(
        parser.feed("<function=get_weather>"
                    "<parameter=city>Paris</parameter>"))
    assert [e.kind for e in first] == ["tool_head", "tool_args"]
    second = list(parser.feed("<parameter=base>10</parameter></function>"))
    args = "".join(e.text for e in first + second if e.kind == "tool_args")
    assert args == '{"city": "Paris", "base": 10}'  # normalized + coerced
    assert second[-1].kind == "tool_done"


def test_raw_degrade_and_poison_paths():
    config = _config()
    # Pre-confirm violation (single-quoted key): lossless degrade to the
    # buffered path -- ast fallback still yields the call, same as classic.
    text = ('<tool_call>{"name": "get_weather", "arguments": '
            "{'city': 1}}</tool_call>")
    events = _stream_events([text[i:i + 5] for i in range(0, len(text), 5)],
                            config)
    kinds = [e.kind for e in events]
    assert kinds.count("tool_head") == 1 and "tool_done" in kinds
    args = "".join(e.text for e in events if e.kind == "tool_args")
    assert json.loads(args) == {"city": 1}
    # Post-confirm violation (R1): head already out; args stop, call closes.
    parser = _select_parser("/nonexistent", "generic").stream(config)
    events = list(
        parser.feed('<tool_call>{"name": "get_weather", "arguments": '
                    '{"a": 1_000}}</tool_call>'))
    kinds = [e.kind for e in events]
    assert kinds.count("tool_head") == 1
    assert kinds[-1] == "tool_done"


def test_raw_truncation_flush_emits_done():
    # R2: a call cut off by max_tokens finishes with tool_done; the partial
    # arguments already streamed stay as-is.
    config = _config()
    parser = _select_parser("/nonexistent", "generic").stream(config)
    events = list(
        parser.feed('<tool_call>{"name": "get_weather", "arguments": '
                    '{"city": "trunc'))
    events += list(parser.flush())
    kinds = [e.kind for e in events]
    assert kinds.count("tool_head") == 1
    assert kinds[-1] == "tool_done"
    args = "".join(e.text for e in events if e.kind == "tool_args")
    assert args == '{"city": "trunc'


def test_param_mode_repeated_key_stops_cleanly():
    config = _config()
    parser = _select_parser("/nonexistent", "generic").stream(config)
    events = list(
        parser.feed("<function=get_weather>"
                    "<parameter=city>long value</parameter>"
                    "<parameter=city>s</parameter></function>"))
    kinds = [event.kind for event in events]
    assert kinds.count("tool_head") == 1
    assert kinds[-1] == "tool_done"
    args = "".join(e.text for e in events if e.kind == "tool_args")
    assert args == '{"city": "long value"'  # sent prefix, nothing bogus after
    # A prefix-compatible rewrite (non-string scalar) keeps streaming and
    # matches the whole-text parser's last-wins result.
    parser = _select_parser("/nonexistent", "generic").stream(config)
    events = list(
        parser.feed("<function=get_weather><parameter=base>1</parameter>"
                    "<parameter=base>12</parameter></function>"))
    args = "".join(e.text for e in events if e.kind == "tool_args")
    assert json.loads(args) == {"base": 12}


def test_raw_tail_swallows_wrapper_junk():
    # Bytes between the closed arguments value and the close marker are
    # discarded: the streamed call stays clean even though the whole-text
    # parser would call this block malformed (accepted divergence).
    config = _config()
    events = _stream_events([
        '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}} '
        "junk</tool_call> t"
    ], config)
    kinds = [event.kind for event in events]
    assert kinds.count("tool_head") == 1 and kinds.count("tool_done") == 1
    args = "".join(e.text for e in events if e.kind == "tool_args")
    assert json.loads(args) == {"city": "A"}
    assert "".join(e.text for e in events if e.kind == "content") == " t"


def test_preconfirm_raw_truncation_flushes_as_content():
    # MR review (greptile P1): RAW mode entered but truncated before the
    # scanner confirmed -- nothing streamed yet, so flush must re-parse the
    # withheld bytes as content instead of dropping them or emitting a
    # phantom tool_done.
    config = _config()
    text = 'before <tool_call>{"name": "get_weather", "arguments": {'
    events = _stream_events(list(text), config)  # char-by-char
    assert all(event.kind == "content" for event in events)
    assert "".join(e.text for e in events) == text
