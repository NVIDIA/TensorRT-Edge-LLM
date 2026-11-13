# Experimental TensorRT Edge-LLM container for Jetson Thor.
#
# Initialize submodules, then build from the repository root:
#   git submodule update --init --recursive
#   docker build --network=host --shm-size=8g \
#     -f experimental/docker/Dockerfile \
#     -t tensorrt-edge-llm:experimental .
#
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

ARG TRT_PACKAGE_DIR=/usr
ARG EMBEDDED_TARGET=jetson-thor
ARG CUTE_DSL_ARTIFACT_TAG=sm_110
ARG CUDA_CUBIN_ARCH=sm_110a
ARG MAX_JOBS

ENV EDGELLM_HOME=/opt/TensorRT-Edge-LLM \
    EDGELLM_WORKSPACE=/data/edgellm \
    EDGELLM_CACHE_DIR=/data/edgellm/artifacts \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/models/huggingface \
    HF_HUB_CACHE=/data/models/huggingface/hub \
    TRANSFORMERS_CACHE=/data/models/huggingface \
    TRT_PACKAGE_DIR=${TRT_PACKAGE_DIR} \
    EMBEDDED_TARGET=${EMBEDDED_TARGET} \
    CUTE_DSL_ARTIFACT_TAG=${CUTE_DSL_ARTIFACT_TAG} \
    CUDA_CUBIN_ARCH=${CUDA_CUBIN_ARCH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        g++-aarch64-linux-gnu \
        gcc-aarch64-linux-gnu \
        git \
        git-lfs \
        python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install --system

COPY . ${EDGELLM_HOME}
WORKDIR ${EDGELLM_HOME}

RUN experimental/docker/build.sh

ENV EDGELLM_PLUGIN_PATH=/opt/TensorRT-Edge-LLM/build/lib/libNvInfer_edgellm_plugin.so \
    PYTHONPATH=/opt/TensorRT-Edge-LLM:/opt/TensorRT-Edge-LLM/build/lib \
    LD_LIBRARY_PATH=/opt/TensorRT-Edge-LLM/build/lib:${TRT_PACKAGE_DIR}/lib:${TRT_PACKAGE_DIR}/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/TensorRT-Edge-LLM
EXPOSE 8000
