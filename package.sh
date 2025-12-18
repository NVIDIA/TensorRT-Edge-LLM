#!/bin/bash

# Define the output archive name
VERSION="0.4.0.1"
OUTPUT_FILE="tensorrt_edge_llm-${VERSION}.tar.gz"
TMP_DIR="tensorrt_edge_llm-${VERSION}"
mkdir -p $TMP_DIR

# List the directories to include (space-separated)
INCLUDE_DIRS="3rdParty cmake cpp docs examples tensorrt_edgellm CMakeLists.txt unittests README.md LICENSE CHANGELOG.md requirements.txt pyproject.toml"

# Use rsync to copy files while excluding the specified directory
for dir in $INCLUDE_DIRS; do
    if [ -d "$dir" ]; then
        rsync -av --exclude='multimodal/pics' --exclude='**/__pycache__' --exclude='3rdParty/googletest' --exclude='3rdParty/nlohmannJson' --prune-empty-dirs "$dir" "$TMP_DIR/"
    else
        cp "$dir" "$TMP_DIR/"
    fi
done

# Create the tar.gz archive
tar -czvf $OUTPUT_FILE $TMP_DIR

rm -rf $TMP_DIR
echo "Packaging complete: $OUTPUT_FILE"