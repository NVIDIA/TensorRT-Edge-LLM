# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

USER root
ARG PYTHON_APT_REPOSITORY=ppa:deadsnakes/ppa
ARG PYTHON_VERSIONS="3.10 3.11 3.12"

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils build-essential ca-certificates git patchelf software-properties-common; \
    if [ -n "${PYTHON_APT_REPOSITORY}" ]; then \
        add-apt-repository -y "${PYTHON_APT_REPOSITORY}"; \
        apt-get update; \
    fi; \
    packages=""; \
    for version in ${PYTHON_VERSIONS}; do \
        packages="${packages} python${version} python${version}-dev python${version}-venv"; \
    done; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${packages}; \
    rm -rf /var/lib/apt/lists/*

COPY packaging/wheel-toolchain-requirements.txt /tmp/wheel-toolchain-requirements.txt
RUN set -eux; \
    mkdir -p /opt/edgellm-wheel-bin; \
    for version in ${PYTHON_VERSIONS}; do \
        abi="cp$(printf "%s" "${version}" | tr -d ".")"; \
        environment="/opt/edgellm-wheel-${abi}"; \
        "python${version}" -m venv "${environment}"; \
        "${environment}/bin/python" -m pip install --no-cache-dir \
            --requirement /tmp/wheel-toolchain-requirements.txt; \
        printf '#!/bin/sh\nexec %s/bin/python "$@"\n' "${environment}" \
            > "/opt/edgellm-wheel-bin/python${version}"; \
        chmod 0755 "/opt/edgellm-wheel-bin/python${version}"; \
    done

ENV PATH="/opt/edgellm-wheel-bin:/opt/edgellm-wheel-cp312/bin:${PATH}"
WORKDIR /workspace
