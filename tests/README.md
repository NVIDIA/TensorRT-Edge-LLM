# TensorRT Edge LLM Testing

## Quick Start

### 1. Environment Setup
```bash
export LLM_SDK_DIR=$(pwd)                           # Required: Project root
export ONNX_DIR=/path/to/onnx/models                # Required: ONNX model directory
export ENGINE_DIR=/path/to/engine/outputs           # Required for pipeline tests
export LLM_MODELS_DIR=/path/to/pytorch/models       # Optional with an explicit model ID or path
export EDGELLM_DATA_DIR=/path/to/datasets           # Required only by tests using shared datasets
export QUANT_CHECKPOINT_DIR=/path/to/quantized/checkpoints
export HF_CHECKPOINT_DOWNLOAD_DIR=/managed/checkpoints/huggingface  # Optional
export EDGE_LLM_ALLOW_HF_DOWNLOAD=0                 # Optional: disable downloads
export TRT_PACKAGE_DIR=/path/to/tensorrt            # Required for C++ build/runtime tests
```

**Default Paths:**
- `LLM_MODELS_DIR` defaults to:
  - `/scratch.trt_llm_data/llm-models`
  - `/home/scratch.trt_llm_data/llm-models` (fallback)
- `EDGELLM_DATA_DIR` defaults to:
  - `/scratch.edge_llm_cache`
  - `/home/edge_llm_cache` (fallback)
  - `/home/scratch.edge_llm_cache` (fallback)

Checkpoint lookup first uses paths registered in the TRT-LLM model mirror,
then managed quantized checkpoints. If downloads are enabled, missing public
checkpoints are stored under `HF_CHECKPOINT_DOWNLOAD_DIR/<organization>/<model>`;
CI enables this fallback and uses
`/scratch.edge_llm_cache/checkpoints/huggingface`. Set `HF_HOME` beneath the same
managed directory to avoid implicit downloads in a user home directory. Local
tests also enable downloading by default and use
`EDGELLM_DATA_DIR/checkpoints/huggingface` when
`HF_CHECKPOINT_DOWNLOAD_DIR` is unset.
An explicit Hugging Face ID such as `Qwen/Qwen2.5-0.5B` may use this fallback
without an inventory entry. Unregistered short names can use local checkpoints;
downloads require the full `organization/model` ID. Set
`EDGE_LLM_ALLOW_HF_DOWNLOAD=0` to disable network downloads while retaining
access to checkpoints already in the managed cache.

For local checkpoints, place models under
`<LLM_MODELS_DIR>/<organization>/<model>` or `<LLM_MODELS_DIR>/<model>`.
Each model directory must contain `config.json` and Safetensors weights
(`*.safetensors`).

### 2. Install Dependencies
```bash
# Runtime/inference pipelines
pip install \
  -e ".[runtime]" \
  -r tests/requirements.txt \
  -r examples/accuracy/requirements-runtime.txt

# Checkpoint export pipelines
pip install -e ".[onnx]"

# Full Python unit-test suite
pip install -e ".[quantize,onnx,builder,runtime]"
pip install -r tests/requirements-ut.txt

# Python plugin tests on devices without a framework in the base image
pip install -r tests/requirements-plugin.txt
```

`tests/requirements.txt` contains only the shared pytest harness. Application
dependencies come from the matching project extra, so test jobs cannot silently
select different Torch, Transformers, ONNX, server, or ModelOpt versions.
When `--priority` selects a test-list YAML, pytest also skips unlisted modules
before importing them. Runtime jobs therefore do not need exporter frameworks
just to collect and later deselect the Python unit tests.

### 3. Build Project
```bash
# Install Python package
pip install build
python -m build --wheel --outdir dist .
pip install dist/*.whl

# Build C++ components
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DBUILD_UNIT_TESTS=ON
make -j$(nproc) && cd ..
```

### 4. Run Tests
```bash
# Run specific test suite
pytest --priority=l0_e2e_a30 -v
pytest --priority=l0_e2e_orin -v
```

## Test Structure

### Test Categories
- **Export Tests** (`test_model_export.py`) - PyTorch to ONNX conversion
- **Package Tests** (`test_package.py`) - Python package functionality
- **Pipeline Tests** (`test_llm_pipeline.py`, `test_vlm_pipeline.py`) - End-to-end inference
- **Common Tests** (`test_common.py`) - Build and unit tests

### L0 Model Cases

L0 deployment coverage is organized by base model in `model_configs/l0/`.
See [model_configs/README.md](model_configs/README.md) for the schema and the
custom merge-request pipeline format. Files under `test_lists/` retain only
suite prerequisites and non-model tests.

## Directory Structure

### Tests Organization
```
tests/
├── defs/                    # Test definitions
│   ├── config.py           # Unified configuration system
│   ├── test_common.py      # Build and unit tests
│   ├── test_llm_pipeline.py # LLM pipeline tests
│   ├── test_vlm_pipeline.py # VLM pipeline tests
│   ├── test_model_export.py # Export tests
│   ├── test_package.py     # Package tests
│   └── utils/              # Utility functions
│       ├── command_execution.py
│       ├── command_generation.py
│       └── accuracy.py
├── ifb/                    # In-flight batching trace driver and correctness checker
├── model_configs/l0/       # Model-owned L0 deployment cases
├── test_lists/             # Suite prerequisites and non-model tests
├── test_cases/             # Test input/reference data
└── conftest.py            # Pytest configuration
```

### Model Directory Structure
**ONNX Models (`ONNX_DIR/`):**

Pre-quantized GPTQ model identifiers keep their complete model name as the
top-level directory, for example
`Qwen3-8B-GPTQ-Int4/llm-int4_gptq-fp16/`.

```
{ModelName}/
├── llm-{precision}-{lm_head_precision}/
│   ├── model.onnx
│   ├── config.json
│   ├── tokenizer.json
│   └── lora_model.onnx         # (if LoRA enabled)
├── draft-{draft_id}-{precision}-{lm_head_precision}/  # EAGLE draft model
│   ├── model.onnx
│   ├── config.json
│   └── tokenizer.json
├── visual-{precision}/          # VLM visual models
│   ├── model.onnx
│   └── config.json
├── lora_weights/                # Processed LoRA weights (if LoRA enabled)
│   └── lora_0.safetensors
├── quantized/                   # Quantized base model checkpoints
│   └── quantized-{precision}-{lm_head_precision}/
└── quantized-draft/             # Quantized draft model checkpoints
    └── quantized-{draft_id}-{precision}-{lm_head_precision}/
```

**Engine Output (`ENGINE_DIR/`):**
```
{ModelName}/
├── llm-{precision}-{lm_head_precision}-mxsl{N}-mxil{N}-mxbs{N}-mxlr{N}/
│   └── llm.engine
└── visual-{visual_precision}-mnit{N}-mxit{N}-mxpiit{N}/  # VLM only
    └── visual.engine
```

## Remote Execution

Tests support remote execution on target devices (e.g., Jetson Orin):

```bash
pytest --priority=l0_e2e_orin \
       --execution-mode=remote \
       --remote-host=192.168.55.1 \
       --remote-user=nvidia \
       --remote-workspace=/home/nvidia/tensorrt-edge-llm \
       -v
```

Environment variables for remote execution:
- `BOARD_HOST`, `BOARD_USER`, `BOARD_PASSWORD_NVKS`
- `REMOTE_WORKSPACE`

## Troubleshooting

### Common Issues
**Model Files Not Found:**
```bash
FileNotFoundError: ONNX model not found
```
→ Verify `ONNX_DIR` path and model structure matches expected format.

**Build Executables Not Found:**
```bash
Unit test executable not found: build/bin/unitTests/unitTestRuntime
```
→ Ensure project is built with `cmake .. -DBUILD_UNIT_TESTS=ON`

**TensorRT Library Not Found:**
```bash
OSError: libnvinfer.so.x: cannot open shared object file
```
→ Set `TRT_PACKAGE_DIR` or `LD_LIBRARY_PATH=/path/to/tensorrt/lib`

### Debug Commands
```bash
# Check environment
echo $LLM_SDK_DIR $ONNX_DIR $ENGINE_DIR

# Verbose test output with logs
pytest --priority=l0_e2e_a30 -v -s --tb=long

# Check individual logs
ls logs/ && cat logs/test_build_project.log
```

## Adding New Tests

Add a readable deployment case to the base model's YAML under
`tests/model_configs/l0/`. Do not add an encoded model ID to a platform test
list. See [the model-case schema](model_configs/README.md) for ONNX,
checkpoint-direct, server, speculative-decoding, and inference fields.

Test the selected platform locally:

```bash
pytest --priority=l0_e2e_a30 -k "MyModel" -v
```

## Server stress testing

The opt-in server stress suite covers concurrent requests, overload recovery,
cancellation, sustained load, and process lifecycle on x86 and DGX Spark.
See [Server stress testing](server_stress/README.md) for setup, pytest commands,
configuration, result metrics, and the HTTP benchmark CLI.
