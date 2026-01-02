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
# Helper functions for auto-generating API documentation.

import xml.etree.ElementTree as ET
from pathlib import Path

# Directory name for generated C++ API documentation
CPP_API_DIR = 'cpp_api'

# Heading mapping: Override titles for specific header files
# Key: header filename (without .h extension)
# Value: Custom title to use instead of auto-generated name
#
# By default, filenames are automatically converted from camelCase to Title Case:
#   - fileUtils.h -> "File Utils"
#   - contextFMHARunner.h -> "Context FMHA Runner"
#   - llmEngineRunner.h -> "LLM Engine Runner"
#
# Use this dictionary to override the automatic conversion for specific files:
HEADING_OVERRIDES = {
    # Example overrides:
    # 'fmhaParams_v2': 'FMHA Parameters V2',
    # 'int4GroupwiseGemm': 'INT4 Groupwise GEMM',
}


def camel_case_to_title(name):
    """
    Convert camelCase or PascalCase to Title Case with spaces.
    Preserves common acronyms in uppercase and handles underscores.
    
    Examples:
        fileUtils -> File Utils
        contextFMHARunner -> Context FMHA Runner
        int4GroupwiseGemm -> Int4 Groupwise Gemm
        llmEngineRunner -> LLM Engine Runner
        fmhaParams_v2 -> FMHA Params V2
        internViTRunner -> Intern ViT Runner
    """
    import re

    # List of known acronyms to keep uppercase
    acronyms = [
        'FMHA', 'XQA', 'KV', 'LLM', 'CUDA', 'TRT', 'API', 'GPU', 'CPU', 'GEMM',
        'VIT', 'MMAP', 'ViT', 'EAGLE'
    ]

    # Replace underscores with spaces first
    name = name.replace('_', ' ')

    # Insert space before uppercase letters (except at start)
    # Also handle numbers followed by uppercase
    result = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)

    # Split into words and process acronyms
    words = result.split()
    processed_words = []

    i = 0
    while i < len(words):
        word = words[i]
        # Check for "Vi T" pattern and merge to "ViT"
        if word == "Vi" and i + 1 < len(words) and words[i + 1] == "T":
            processed_words.append("ViT")
            i += 2  # Skip both Vi and T
            continue

        # Check if word is a known acronym
        if word.upper() in [a.upper() for a in acronyms]:
            # Find the original casing from acronyms list
            for acronym in acronyms:
                if word.upper() == acronym.upper():
                    processed_words.append(acronym)
                    break
        else:
            # Capitalize normally
            processed_words.append(word.capitalize())
        i += 1

    return ' '.join(processed_words)


def get_cpp_directory_structure():
    """
    Scan the cpp directory and return a dictionary mapping module names to header files.
    Dynamically discovers all subdirectories in cpp/ and excludes cubin directories
    (matching the Doxyfile EXCLUDE_PATTERNS = */cubin/*).
    """
    # Get path relative to this helper file (docs/source/sphinx_helpers.py)
    cpp_dir = Path(__file__).parent.parent.parent / 'cpp'
    module_headers = {}

    # Dynamically discover all subdirectories in cpp/
    if not cpp_dir.exists():
        return module_headers

    # Get all immediate subdirectories of cpp/
    modules = [
        d.name for d in cpp_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    for module in sorted(modules):
        module_path = cpp_dir / module
        # Find all .h and .cuh files in this module (recursively)
        headers = []
        for header_file in list(module_path.rglob('*.h')) + list(
                module_path.rglob('*.cuh')):
            # Get path relative to cpp directory
            rel_path = header_file.relative_to(cpp_dir)
            rel_path_str = str(rel_path).replace('\\', '/')

            # Exclude cubin directories (matching Doxyfile EXCLUDE_PATTERNS)
            if '/cubin/' not in rel_path_str:
                headers.append(rel_path_str)

        if headers:
            module_headers[module] = sorted(headers)

    return module_headers


def sanitize_filename(name):
    """
    Convert a header path to a safe RST filename.
    Example: kernels/contextAttentionKernels/contextFMHARunner.h -> contextFMHARunner.rst
    """
    return Path(name).stem + '.rst'


def generate_header_rst_file(header_path):
    """
    Generate RST content for a single header file.
    Uses HEADING_OVERRIDES for custom titles, otherwise converts camelCase to Title Case.
    Parses XML to find all documented entities (classes, structs, enums, functions, etc.)
    and generates appropriate Breathe directives for each type.
    Only includes public members (no private or protected members).
    
    Reference: https://breathe.readthedocs.io/en/latest/directives.html
    """
    header_name = Path(header_path).name
    base_name = Path(header_path).stem  # filename without .h extension

    # Check for manual override, otherwise use automatic conversion
    if base_name in HEADING_OVERRIDES:
        title = HEADING_OVERRIDES[base_name]
    else:
        title = camel_case_to_title(base_name)

    rst_content = f"{title}\n"
    rst_content += "=" * len(title) + "\n\n"

    # Find the corresponding XML file and extract all entities
    # The XML filename is based on just the filename, not the full path
    # e.g., builder/builder.h -> builder_8h.xml (not builder_builder_8h.xml)
    xml_filename = Path(header_path).name.replace('.h', '_8h.xml').replace(
        '.cuh', '_8cuh.xml')
    # Get path relative to this helper file (docs/source/sphinx_helpers.py)
    xml_path = Path(__file__).parent.parent / 'cpp_docs' / 'xml' / xml_filename

    # Collect different entity types
    entities = {
        'class': [],  # Classes (handled by doxygenclass)
        'struct': [],  # Structs (handled by doxygenstruct)
        'enum': [],  # Enums
        'typedef': [],  # Typedefs
        'function': [],  # Functions
        'define': [],  # Preprocessor defines
        'variable': [],  # Variables
    }

    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find all innerclass elements (classes and structs defined in this file)
            for innerclass in root.findall('.//innerclass'):
                prot = innerclass.get('prot', 'public')
                # Only include public classes/structs
                if prot == 'public' or prot is None:
                    class_name = innerclass.text
                    if class_name:
                        # Determine if it's a class or struct by checking the refid prefix
                        refid = innerclass.get('refid', '')
                        if refid.startswith('struct'):
                            entities['struct'].append(class_name)
                        else:
                            entities['class'].append(class_name)

            # Find all top-level members (functions, enums, typedefs, etc.)
            for member in root.findall('.//member'):
                prot = member.get('prot', 'public')
                # Only include public members
                if prot == 'public' or prot is None:
                    kind = member.get('kind')
                    name_elem = member.find('name')
                    if name_elem is not None and name_elem.text:
                        name = name_elem.text
                        if kind == 'enum':
                            entities['enum'].append(name)
                        elif kind == 'typedef':
                            entities['typedef'].append(name)
                        elif kind == 'function':
                            entities['function'].append(name)
                        elif kind == 'define':
                            entities['define'].append(name)
                        elif kind == 'variable':
                            entities['variable'].append(name)

            # Extract namespace-level functions from innernamespace elements
            # Functions defined in namespaces are stored in separate namespace XML files
            for innernamespace in root.findall('.//innernamespace'):
                namespace_refid = innernamespace.get('refid', '')
                if namespace_refid:
                    # Construct the namespace XML filename from the refid
                    namespace_xml_path = Path(
                        __file__
                    ).parent.parent / 'cpp_docs' / 'xml' / f'{namespace_refid}.xml'
                    if namespace_xml_path.exists():
                        try:
                            ns_tree = ET.parse(namespace_xml_path)
                            ns_root = ns_tree.getroot()

                            # Find all function members in this namespace
                            for ns_member in ns_root.findall(
                                    './/memberdef[@kind="function"]'):
                                ns_prot = ns_member.get('prot', 'public')
                                # Only include public functions
                                if ns_prot == 'public' or ns_prot is None:
                                    # Check if this function is declared in the current header file
                                    location = ns_member.find('location')
                                    if location is not None:
                                        declfile = location.get('declfile', '')
                                        # Only include if declared in this specific header file
                                        if declfile.endswith(
                                                header_path
                                        ) or declfile.endswith(header_name):
                                            # Get the qualified name to avoid duplicates
                                            qualified_name_elem = ns_member.find(
                                                'qualifiedname')
                                            if qualified_name_elem is not None and qualified_name_elem.text:
                                                func_name = qualified_name_elem.text
                                                if func_name not in entities[
                                                        'function']:
                                                    entities[
                                                        'function'].append(
                                                            func_name)
                        except Exception as e:
                            print(
                                f"Warning: Could not parse namespace XML {namespace_xml_path}: {e}"
                            )
        except Exception as e:
            print(f"Warning: Could not parse {xml_path}: {e}")

    # Generate directives for each entity type in logical order
    # Order: classes, structs, enums, typedefs, functions, defines, variables

    # Classes
    for class_name in entities['class']:
        rst_content += f".. doxygenclass:: {class_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n"
        rst_content += f"   :members:\n"
        rst_content += f"   :undoc-members:\n\n"

    # Structs
    for struct_name in entities['struct']:
        rst_content += f".. doxygenstruct:: {struct_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n"
        rst_content += f"   :members:\n"
        rst_content += f"   :undoc-members:\n\n"

    # Enums
    for enum_name in entities['enum']:
        rst_content += f".. doxygenenum:: {enum_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n\n"

    # Typedefs
    for typedef_name in entities['typedef']:
        rst_content += f".. doxygentypedef:: {typedef_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n\n"

    # Functions
    for function_name in entities['function']:
        rst_content += f".. doxygenfunction:: {function_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n\n"

    # Preprocessor Defines
    for define_name in entities['define']:
        rst_content += f".. doxygendefine:: {define_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n\n"

    # Variables
    for variable_name in entities['variable']:
        rst_content += f".. doxygenvariable:: {variable_name}\n"
        rst_content += f"   :project: TensorRT Edge-LLM\n\n"

    # If no entities found, fall back to doxygenfile for any other content
    total_entities = sum(len(v) for v in entities.values())
    if total_entities == 0:
        rst_content += f".. doxygenfile:: {header_path}\n"
        rst_content += "   :project: TensorRT Edge-LLM\n\n"

    return rst_content


def generate_module_index_rst(module_name, rst_files):
    """
    Generate index.rst for a module subdirectory.
    """
    title = f"{module_name.title()} Module"

    index_content = f"{title}\n"
    index_content += "=" * len(title) + "\n\n"
    index_content += f"API documentation for the {module_name} module.\n\n"
    index_content += ".. toctree::\n"
    index_content += "   :maxdepth: 1\n\n"

    for rst_file in sorted(rst_files):
        # Remove .rst extension for toctree
        index_content += f"   {rst_file[:-4]}\n"

    return index_content


def generate_module_rst_files():
    """
    Generate individual RST files for each header file, organized in module subdirectories.
    Each module gets its own directory with an index.rst and individual RST files for each header.
    """
    # Get path relative to this helper file (docs/source/sphinx_helpers.py)
    source_dir = Path(__file__).parent
    api_dir = source_dir / CPP_API_DIR

    # Create API directory if it doesn't exist
    api_dir.mkdir(exist_ok=True)

    module_headers = get_cpp_directory_structure()
    modules_data = {}

    # Generate RST files for each module
    for module_name, headers in module_headers.items():
        # Create module subdirectory
        module_dir = api_dir / module_name
        module_dir.mkdir(exist_ok=True)

        rst_files = []

        # Generate individual RST file for each header
        for header in headers:
            # Get the base filename for the RST file
            rst_filename = sanitize_filename(header)
            rst_files.append(rst_filename)

            # Generate RST content
            rst_content = generate_header_rst_file(header)

            # Write the RST file
            rst_file = module_dir / rst_filename
            with open(rst_file, 'w') as f:
                f.write(rst_content)

            print(f"Generated {rst_file}")

        # Generate index.rst for this module
        index_content = generate_module_index_rst(module_name, rst_files)
        index_file = module_dir / 'index.rst'
        with open(index_file, 'w') as f:
            f.write(index_content)

        print(f"Generated {index_file}")

        # Store module data for cpp_api.rst
        modules_data[module_name] = rst_files

    # Generate cpp_api.rst with all header files organized by module
    generate_cpp_api_rst(source_dir, modules_data)

    return list(modules_data.keys())


def generate_cpp_api_rst(source_dir, modules_data):
    """
    Generate cpp_api.rst to link to C++ module index pages, creating a hierarchical structure.
    The cpp_api.rst lists module index pages, which in turn list individual API files.
    
    Args:
        source_dir: Path to the source directory
        modules_data: Dictionary mapping module names to lists of RST filenames
    """
    cpp_api_file = source_dir / "cpp_api.rst"

    cpp_api_content = """C++ API Reference
==================

This section provides documentation for the TensorRT Edge-LLM C++ API.

.. toctree::
   :maxdepth: 2

"""

    # Add module index pages to the toctree
    for module_name in sorted(modules_data.keys()):
        cpp_api_content += f"   {CPP_API_DIR}/{module_name}/index\n"

    with open(cpp_api_file, 'w') as f:
        f.write(cpp_api_content)

    print(
        f"Generated {cpp_api_file} with {len(modules_data)} modules linking to their index pages"
    )


def generate_python_api_rst():
    """
    Generate python_api.rst file for the tensorrt_edgellm package.
    Since the top-level __init__.py exposes all necessary functions via __all__,
    we only need to document the main module.
    """
    # Get path relative to this helper file (docs/source/sphinx_helpers.py)
    source_dir = Path(__file__).parent
    python_api_file = source_dir / "python_api.rst"

    content = """Python API Reference
====================

This section provides documentation for the TensorRT Edge-LLM Python package.

The ``tensorrt_edgellm`` package provides utilities for quantizing large language models 
and exporting them to ONNX format for efficient inference on edge devices.

Main Module
-----------

.. automodule:: tensorrt_edgellm
   :members:
   :undoc-members:
   :show-inheritance:
"""

    with open(python_api_file, 'w') as f:
        f.write(content)

    print(f"Generated {python_api_file} with Python API documentation")
