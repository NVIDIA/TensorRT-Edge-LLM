<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend tests

## Run tests

Run from the repository root with the Python package and pytest installed:

```bash
LLM_SDK_DIR="$PWD" python -m pytest --confcutdir=tests/python-unittests \
  -o addopts= -q tests/python-unittests/frontend
```

These tests use fake backends and run without a GPU or TensorRT. They are also
listed in `tests/test_lists/l0_python_ut.yml`.

`--confcutdir` skips the repository's integration fixtures. `-o addopts=` clears
the default pytest options.

| File | Coverage |
| --- | --- |
| `test_cli.py` | Help, version, exit codes, output routing, and an interactive smoke test |
| `test_dispatch.py` | Common command routing, lazy imports, exit codes, packaging, and legacy help |
| `test_config.py` | Input files, option precedence, defaults, and invalid settings |
| `test_backend.py` | API option forwarding, reasoning separation, model details, and rates |
| `test_session.py` | Conversation history, regeneration, cancellation, final metrics, and recovery |
| `test_terminal.py` | Completion, labels, color suppression, and loading progress |

`helpers.py` contains the shared settings factory and fake backend.
`conftest.py` clears CLI environment variables for these tests.

Real-handler help tests require the corresponding tool's optional dependencies
and skip when they are absent. Routing and top-level help tests do not require
those dependencies.

The interactive smoke test covers completion, cancellation, and a subsequent
turn. A subprocess test checks routing of buffered native output.

When changing stream lifecycle, run the frontend and shared runtime tests together:

```bash
LLM_SDK_DIR="$PWD" python -m pytest --confcutdir=tests/python-unittests \
  -o addopts= -q tests/python-unittests/frontend \
  tests/python-unittests/test_server_runtime.py
```

## Check a model

Before release, install the wheel on a supported GPU system. Use the commands
and launch options in the [interactive text guide](../../../docs/source/user_guide/examples/interactive-chat.md):

1. Run `trtedge --version` and `trtedge chat --help` to check the installed command.
2. Start a chat with an uncached checkpoint and confirm that TensorRT engines
   build. Restart with the same settings and confirm that cached engines load.
3. Send a prompt, stop the response with Ctrl+C, and send another prompt. Run
   `/regen` to replace the response, then `/clear` to reset the conversation.
4. Run one prompt with stdout redirected to a file. Check that the file contains
   response text and that status, timings, and diagnostics stay in the terminal.
5. Enable a speculative decoding method supported by the checkpoint. Check the
   startup mode, generated response, and timing summary.

Record the wheel version, GPU, checkpoint, launch command, and result with the
release validation.
