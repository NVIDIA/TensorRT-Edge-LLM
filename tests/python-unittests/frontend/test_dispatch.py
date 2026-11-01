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
"""Common command routing, lazy loading, and legacy entry-point contracts."""

import os
import pathlib
import subprocess
import sys
import types
from importlib import metadata

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from tensorrt_edgellm import cli

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUTES = [
    ("chat", "experimental.frontend.cli"),
    ("serve", "experimental.server.cli"),
    ("build", "experimental.builder.cli"),
    ("export", "tensorrt_edgellm.scripts.export"),
    ("quantize", "tensorrt_edgellm.scripts.quantize"),
    ("insert-lora", "tensorrt_edgellm.scripts.insert_lora"),
    ("process-lora", "tensorrt_edgellm.scripts.process_lora_weights"),
    ("merge-lora", "tensorrt_edgellm.scripts.merge_lora"),
    ("reduce-vocab", "tensorrt_edgellm.scripts.reduce_vocab"),
]


@pytest.mark.parametrize("command,module_name", ROUTES)
@pytest.mark.parametrize("status", [None, 0, 7])
def test_routes_arguments_and_return_status(monkeypatch, command, module_name,
                                            status):
    # Given a command with option-like arguments and a handler exit status.
    # When dispatched, then only its module is loaded and arguments stay intact.
    imported, calls = [], []
    arguments = [
        "llm", "path with spaces", "--dtype=fp16", "--help", "--", "-1"
    ]

    def main(argv, *, prog):
        calls.append((argv, prog))
        return status

    def load(name):
        imported.append(name)
        return types.SimpleNamespace(main=main)

    monkeypatch.setattr(cli.importlib, "import_module", load)
    assert cli.main([command, *arguments]) == status
    assert imported == [module_name]
    assert calls == [(arguments, "trtedge " + command)]


@pytest.mark.parametrize("status", [2, 130])
def test_handler_system_exit_is_preserved(monkeypatch, status):
    # Given a handler that exits with an error or interruption status.
    # When dispatched, then the shell receives that same status.
    def main(argv, *, prog):
        raise SystemExit(status)

    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda _: types.SimpleNamespace(main=main))
    with pytest.raises(SystemExit) as result:
        cli.main(["chat"])
    assert result.value.code == status


@pytest.mark.parametrize("arguments", [[], ["run"], ["unknown"], ["--bogus"]])
def test_invalid_top_level_command_does_not_import_handler(
        monkeypatch, capsys, arguments):
    # Given a missing or invalid command.
    # When invoked, then usage fails before any implementation is imported.
    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda _: pytest.fail("imported handler"))
    with pytest.raises(SystemExit) as result:
        cli.main(arguments)
    assert result.value.code == 2
    out, err = capsys.readouterr()
    assert not out
    assert "usage: trtedge" in err


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_top_level_information_needs_no_optional_dependencies(option):
    # Given a fresh interpreter that refuses command and optional imports.
    # When help or version is requested, then it succeeds without loading them.
    code = '''
import importlib.abc
import sys
class BlockCommands(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith(("experimental", "tensorrt_edgellm.scripts"))
            or fullname.split(".")[0] in {"torch", "tensorrt", "onnx", "fastapi", "prompt_toolkit", "cuda"}):
            raise AssertionError("unexpected import: " + fullname)
sys.meta_path.insert(0, BlockCommands())
from tensorrt_edgellm.cli import main
sys.exit(main())
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, "-c", code, option],
                            cwd=ROOT,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=10)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    if option == "--help":
        for command, _ in ROUTES:
            assert command in result.stdout
        assert "trtedge COMMAND --help" in result.stdout
    else:
        assert "edge-llm" in result.stdout


@pytest.mark.parametrize(
    "command,module_name,legacy",
    [(command, module_name, legacy) for command, module_name in ROUTES
     for legacy in (False, True) if command != "chat" or not legacy])
def test_real_handler_help_preserves_command_name(monkeypatch, capsys, command,
                                                  module_name, legacy):
    # Given either the combined command or a legacy entry point.
    # When help is requested, then the real parser shows that command's options.
    module = pytest.importorskip(module_name)
    prog = "tensorrt-edgellm-" + command if legacy else "trtedge " + command
    monkeypatch.setattr(sys, "argv", [prog, "--help"] if legacy else
                        ["unrelated-process", "--invalid-parent-argument"])
    with pytest.raises(SystemExit) as result:
        if legacy:
            module.main()
        else:
            cli.main([command, "--help"])
    assert result.value.code == 0
    out, err = capsys.readouterr()
    assert not err
    assert "usage: " + prog in out
    assert "--help" in out


def test_packaged_entry_points_keep_legacy_routes():
    # Given the package's script declarations.
    # When their callable targets are loaded, then trtedge dispatches and legacy routes remain.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    entry = metadata.EntryPoint(name="trtedge",
                                value=scripts["trtedge"],
                                group="console_scripts")
    assert entry.load() is cli.main
    for command, module_name in ROUTES:
        if command != "chat":
            assert scripts["tensorrt-edgellm-" +
                           command] == module_name + ":main"
