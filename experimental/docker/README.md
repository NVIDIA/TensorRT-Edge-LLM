# Experimental Docker Build

This Dockerfile builds TensorRT Edge-LLM for Jetson Thor from the current
checkout using `nvcr.io/nvidia/pytorch:26.04-py3`.

This image workflow uses the matching prebuilt CuTe DSL tarball by default. If
you modify a CuTe DSL kernel or its registry entries, regenerate that artifact
manually first. Use either the local virtual environment or
`kernelSrcs/Dockerfile.cutedsl` as documented in the shared
[CuTe DSL kernel development workflow](../../kernelSrcs/README.md#cute-dsl-kernel-development-workflow).
The experimental image consumes that artifact while compiling Edge-LLM; it is
not the CuTe DSL development environment.

Use the wrapper script from the repository root:

```bash
experimental/docker/build_container.sh
```

The wrapper initializes the repository submodules before creating the Docker
build context.

To customize the image name, set `EXPERIMENTAL_DOCKER_IMAGE` before running the
wrapper.

Run the OpenAI-compatible experimental server:

```bash
docker run --runtime nvidia --rm -it --network host \
  -v /data:/data \
  tensorrt-edge-llm:experimental \
  tensorrt-edgellm-serve Qwen/Qwen3-1.7B --cache-dir /data/cache
```

Model downloads and complete runtime bundles are cached under `/data/cache`.
