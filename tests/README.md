# TensorRT Edge LLM Testing

## Quick Start

### 1. Environment Setup
```bash
export LLM_SDK_DIR=$(pwd)                           # Required: Project root
export ONNX_DIR=/path/to/onnx/models                # Required: ONNX model directory
export ENGINE_DIR=/path/to/engine/outputs           # Required for pipeline tests
export LLM_MODELS_DIR=/path/to/pytorch/models       # Required for export tests (LLM torch models)
export EDGELLM_DATA_DIR=/path/to/datasets           # Required for datasets and draft models
export TRT_PACKAGE_DIR=/path/to/tensorrt            # Optional: TensorRT installation
```

**Default Paths:**
- `LLM_MODELS_DIR` defaults to:
  - `/scratch.trt_llm_data/llm-models`
  - `/home/scratch.trt_llm_data/llm-models` (fallback)
- `EDGELLM_DATA_DIR` defaults to:
  - `/scratch.edge_llm_cache`
  - `/home/edge_llm_cache` (fallback)
  - `/home/scratch.edge_llm_cache` (fallback)

### 2. Install Dependencies
```bash
pip install -r tests/requirements.txt
```

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
pytest --priority=l0_pipeline_a30 -v
pytest --priority=l0_export_ampere -v
```

## Test Structure

### Test Categories
- **Export Tests** (`test_model_export.py`) - PyTorch to ONNX conversion
- **Package Tests** (`test_package.py`) - Python package functionality
- **Pipeline Tests** (`test_llm_pipeline.py`, `test_vlm_pipeline.py`) - End-to-end inference
- **Common Tests** (`test_common.py`) - Build and unit tests

### Available Test Suites
- `l0_export_ampere.yml` - Model export tests (Ampere GPUs)
- `l0_export_blackwell.yml` - Model export tests (Blackwell GPUs)
- `l0_pipeline_a30.yml` - Pipeline tests (A30 GPU)
- `l0_pipeline_orin.yml` - Pipeline tests (Jetson Orin)
- `l0_pipeline_rtx5080.yml` - Pipeline tests (RTX 5080)
- `l0_pipeline_thor_ferrix.yml` - Pipeline tests (Thor/Ferrix)

## Parameter Format

### Model Configuration String
```
ModelName-Precision-[LmHeadPrecision-]MaxSeqLen-MaxBatchSize-MaxInputLen-[Additional-Params]
```

### Core Parameters
- **Model**: `Qwen2.5-0.5B-Instruct`, `InternVL3-1B-hf`
- **Precision**: `fp16`, `fp8`, `int8_sq`, `int4_awq`, `nvfp4`, `int4_gptq`
- **LM Head**: `lmfp16`, `lmfp8`, `lmint4_awq`, `lmnvfp4` (optional, defaults to fp16)
- **Engine Config**: `mxsl4096` (max seq len), `mxbs1` (max batch), `mxil2048` (max input len)

### Task-Specific Parameters
**Build/Inference:**
- `mxlr64` - Max LoRA rank (optional)
- `mnit128`, `mxit1024`, `mxpiit512` - Min/max image tokens (VLM only)
- `vitfp8` - Visual precision (VLM only)

**Benchmark:**
- `bs1` - Batch size, `isl2048` - Input seq len, `osl128` - Output seq len
- `ttl1024`, `itl1024` - Text/image token lengths (VLM only)

**Export:**
- `lora` - Enable LoRA support
- `eagle` - Enable EAGLE (speculative decoding) model export
- `draftfp16`, `draftfp8`, `draftnvfp4` - Draft model precision for EAGLE
- `draftlmfp16`, `draftlmfp8` - Draft model LM head precision (optional, defaults to fp16)

> **Note:** For detailed information on EAGLE model export with draft and base model support, 
> see [EAGLE_EXPORT_README.md](./EAGLE_EXPORT_README.md)

### Examples
```bash
# LLM with FP16 precision
Qwen2.5-0.5B-Instruct-fp16-mxsl4096-mxbs1-mxil2048

# VLM with INT4 AWQ and LoRA
Qwen2.5-VL-3B-Instruct-int4_awq-mxsl4096-mxbs1-mxil2048-mnit128-mxit2048-mxpiit512-mxlr32

# Benchmark test with FP8
Qwen2.5-0.5B-Instruct-fp8-mxsl4096-mxbs1-mxil2048-bs1-isl2048-osl128

# EAGLE export with FP16 base and draft
Qwen2.5-7B-Instruct-fp16-mxsl8192-eagle-draftfp16

# EAGLE export with mixed precision
Qwen2.5-7B-Instruct-fp8-lmfp8-mxsl8192-eagle-draftnvfp4-draftlmfp8
```

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
├── test_lists/             # Test configuration files
├── test_cases/             # Test input/reference data
└── conftest.py            # Pytest configuration
```

### Model Directory Structure
**ONNX Models (`ONNX_DIR/`):**
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
pytest --priority=l0_pipeline_orin \
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
Unit test executable not found: build/unitTest
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
pytest --priority=l0_pipeline_a30 -v -s --tb=long

# Check individual logs
ls logs/ && cat logs/test_build_project.log
```

## Adding New Tests

### 1. Add to Test Suite
Edit appropriate test list file (e.g., `tests/test_lists/l0_pipeline_a30.yml`):
```yaml
tests:
  - tests/defs/test_llm_pipeline.py::test_engine_build[MyModel-fp16-mxsl4096-mxbs1-mxil2048]
```

### 2. Ensure Model Files
Place model files in correct `ONNX_DIR` structure:
```
ONNX_DIR/MyModel/llm-fp16-fp16-4096/model.onnx
```

### 3. Test Locally
```bash
pytest --priority=l0_pipeline_a30 -k "MyModel" -v
```