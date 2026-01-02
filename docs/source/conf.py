# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add necessary directories to Python path
sys.path.insert(0, str(Path(__file__).parent))  # For importing helper module
sys.path.insert(0, str(
    Path(__file__).parent.parent.parent))  # For importing tensorrt_edgellm

# Import helper functions for auto-generating API documentation
from helper import generate_module_rst_files, generate_python_api_rst

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TensorRT Edge-LLM'
copyright = '2025, Nvidia'
author = 'Nvidia'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',
    'myst_parser',  # For Markdown support
    'sphinx.ext.autodoc',  # For automatic Python documentation
    'sphinx.ext.autosummary',  # For generating summary tables
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',  # For adding links to source code
    'sphinxcontrib.mermaid',  # For mermaid diagram support
]

# MyST-Parser configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",  # ::: code fences
    "deflist",  # definition lists
]

# Enable mermaid code blocks in MyST markdown
myst_fence_as_directive = ["mermaid"]

# Configure MyST to generate anchors for headings (enables #heading-id links)
myst_heading_anchors = 3  # Generate anchors for h1, h2, h3

# Configure MyST to not treat markdown links as Sphinx cross-references
# myst_all_links_external = False
# myst_url_schemes = {
#     'http': None,
#     'https': None,
#     'mailto': None,
#     'ftp': None
# }
# Add Pygments lexers
pygments_style = 'sphinx'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for packages not needed during documentation builds
autodoc_mock_imports = [
    'modelopt',
    'torch',
    'transformers',
    'onnx',
    'onnx_graphsurgeon',
    'onnxruntime',
    'numpy',
    'PIL',
    'huggingface_hub',
    'datasets',
    'tqdm',
    'safetensors',
]

# Autosummary configuration
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nvidia_sphinx_theme'
html_static_path = ['_static']

# Breathe configuration
breathe_default_project = "TensorRT Edge-LLM"
breathe_projects = {"TensorRT Edge-LLM": "../cpp_docs/xml"}

# Breathe configuration for C++ API documentation
breathe_default_members = ('members', )
breathe_domain_by_extension = {
    "h": "cpp",
    "cuh": "cpp",
}
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
breathe_order_parameters_first = False

# -- Auto-generate API documentation at build time --------------------------

# Generate C++ and Python API RST files at configuration time
generated_modules = generate_module_rst_files()
generate_python_api_rst()
