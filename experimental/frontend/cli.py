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
"""Command entry point for interactive and one-shot local text chat."""

import sys

from . import backend, config, logging, session, terminal


def _turn(conversation, display, prompt, regenerate=False):
    display.begin_response()
    try:
        if regenerate:
            conversation.regenerate(display.write, display.write_reasoning)
        else:
            conversation.generate(prompt, display.write,
                                  display.write_reasoning)
    finally:
        display.finish_response(conversation)


def run(settings, responses):
    """Load one model and run the requested conversation mode."""
    display = terminal.Terminal(responses, settings)
    display.startup()
    with display.loading():
        model = backend.Backend(settings, display.progress)
    conversation = session.Session(model, settings.messages)
    try:
        if display.interactive:
            display.runtime_details(model.details())
        if settings.prompt is not None:
            _turn(conversation, display, settings.prompt)
            return 130 if conversation.interrupted else 0
        while conversation.state != "closed":
            try:
                prompt = display.read_prompt()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                continue
            command = conversation.command(prompt.strip())
            if command is not None:
                display.status(command)
                continue
            try:
                if prompt.strip() == "/regen":
                    _turn(conversation, display, prompt, regenerate=True)
                    continue
                if prompt.startswith("/file "):
                    prompt = config.read_text(prompt[6:].strip())
                elif prompt.startswith("/"):
                    display.status("Unknown command. Use /help for commands.")
                    continue
                _turn(conversation, display, prompt)
            except BrokenPipeError:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                display.status(f"Error: {error}")
        return 0
    finally:
        conversation.close()


def main(argv=None, *, prog="trtedge chat"):
    """Run trtedge and return a shell exit status."""
    argument_parser = config.parser(prog=prog)
    args = argument_parser.parse_args(argv)
    try:
        settings = config.resolve(args)
    except (OSError, ValueError) as error:
        argument_parser.error(str(error))
    try:
        with logging.diagnostics(settings.verbose) as responses:
            return run(settings, responses)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
    except Exception as error:
        if settings.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            message = str(error).splitlines()
            print(f"Error: {message[0] if message else type(error).__name__}",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
