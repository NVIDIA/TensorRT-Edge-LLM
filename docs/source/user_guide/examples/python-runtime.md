<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Run inference with the Python runtime

Use the Edge-LLM C++ runtime through Python to generate text from local model
files. This example uses the wheel without optional dependencies. For chat or
HTTP serving, follow the [trtedge CLI](../getting_started/cli.md) guide.

## Prepare your environment and model

Complete [Install without optional dependencies](../getting_started/installation.md#install-without-optional-dependencies)
and keep that environment activated for the commands below.

Provide a complete local `Qwen3.5-0.8B` Safetensors checkpoint,
including `tokenizer.json`. The example does not download model files.
Replace `/path/to/Qwen3.5-0.8B` with your checkpoint directory.

## Build an engine and generate text

1. {download}`Download the example script <../../../../examples/python/installed_wheel_build_and_infer.py>`
   and save it outside a source checkout.
2. Choose an output directory for the generated engine. **The script deletes
   and replaces this directory**, so use a disposable directory.
3. Run the script, replacing its path and the checkpoint path below:

```bash
python -I /path/to/installed_wheel_build_and_infer.py \
  /path/to/Qwen3.5-0.8B /tmp/edgellm-base-engines \
  --workflow base
```

The script builds a text engine, runs a prompt through
`tensorrt_edgellm.runtime.LLMRuntime`, and checks for generated text and tokens.
On success, it prints the result and `INSTALLED_WHEEL_BUILD_AND_INFERENCE_PASSED`.

For more engine-build options, see the
[direct builder guide](../getting_started/direct-engine-builder.md).

## Use an existing engine

If you already have a compatible text engine and its local checkpoint, save
the following code as `run_engine.py`. Replace the checkpoint and engine paths
in the final line. This example runs inference without rebuilding the engine.

```{literalinclude} ../../../../examples/python/installed_wheel_build_and_infer.py
---
language: python
pyobject: _infer_base
prepend: |
  from pathlib import Path
  from typing import Tuple
append: |
  print(_infer_base(Path("/path/to/checkpoint"), Path("/path/to/engines"), "Hello", 32))
---
```

Run it in the same environment:

```bash
python /path/to/run_engine.py
```

Successful execution prints the generated text and token count.

C++ example executables require a
[source installation](../getting_started/installation.md#source-workflow-c-runtime).
