# Documentation

This directory contains the documentation source for the TensorRT Edge-LLM project.

## Building the Documentation

The documentation is built using Doxygen, Sphinx, and Breathe. 

### Prerequisites

1. **Install Doxygen** (version 1.14.0 or later):
   ```bash
   wget https://www.doxygen.nl/files/doxygen-1.14.0.linux.bin.tar.gz
   tar -xzf doxygen-1.14.0.linux.bin.tar.gz
   cd doxygen-1.14.0
   make install
   ```

2. **Install Python dependencies**:
   ```bash
   cd docs
   pip install -r requirements.txt
   ```

### Build Commands

From the `docs` directory, run:

1. **Generate Doxygen documentation**:
   ```bash
   doxygen
   ```

2. **Build Sphinx HTML documentation**:
   ```bash
   make html
   ```

   Or: 
   ```bash
   sphinx-build -M html ./source/ ./build
   ```

### Output

The generated HTML documentation will be available in:
- `docs/build/html/`

Open `docs/build/html/index.html` in a browser to view the documentation.
