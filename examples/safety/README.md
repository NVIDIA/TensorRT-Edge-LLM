# Safety inference example

`llm_inference_safe` demonstrates the public `SafetyRuntime` API for InternVL
inference: loading a request with text and an image, initializing the runtime,
running inference, saving the response, and handling runtime cleanup. The runtime
selects its processing capabilities internally from the model inference definition.

## Build and run

Configure an Edge-LLM build with `ENABLE_SAFETY_RUNTIME=ON` and the appropriate
TensorRT Safety package and platform toolchain, then build:

Follow the [Safety build prerequisites](../../cpp/safety_inference/README.md),
including generation and selection of the matching FMHA artifacts.

```bash
cmake --build BUILD_DIR --target llm_inference_safe --parallel
```

The example targets D7L and QNX Standard development environments and uses
device 0. The build excludes this target on QNX Safety, where applications must
coordinate the VM lifecycle around CUDA initialization, Operational execution,
and teardown.

Create `request.json`. Image paths are absolute or relative to this JSON file:

```json
{
  "max_generate_length": 128,
  "requests": [{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image", "image": "image.jpg"},
        {"type": "text", "text": "Describe this image."}
      ]
    }]
  }]
}
```

```bash
BUILD_DIR/bin/llm_inference_safe \
    --engineDir /path/to/llm-engines \
    --multimodalEngineDir /path/to/visual-engines \
    --inputFile request.json --outputFile result.json
```

The engine directories contain the Safety builder's engines, companion
libraries, and configuration. The processing implementation also needs the
model's tokenizer, chat template, and embedding assets. Supply artifacts from
the same model/export. Add `--enableCudaGraphCapture` to request capture during
Initialization.

The example prepares the input and output containers before initialization. It
reserves `max_generate_length` output tokens and 64 KiB of output text; adjust
that application-owned text capacity for the workload. Generation uses the
recipe's greedy sampler. Results contain `output_ids` and `output_texts` and
are written only after successful inference and teardown.

After a fatal error, the example queries the retained recorder and skips SDK
cleanup. This one-shot process then returns failure, retaining the runtime until
process exit. A long-lived application must retain its Runtime owner for error
queries and let its supervisor decide final recovery or termination.
