# Embedding Kernels

CUDA kernels for efficient embedding lookup operations in TensorRT Edge-LLM.

## Overview

The embedding kernels provide two main functionalities:

1. **Standard Embedding Lookup**: Performs standard token-to-embedding lookup
2. **Embedding Lookup with Image Insertion**: Handles multimodal models with image embeddings following the PromptTuningEmbedding logic

**Note**: 
- These kernels only support FP16 (half precision) data type for optimal performance and memory efficiency
- For image embedding insertion, image tokens start from `vocab_size` and increase to `vocab_size + num_image_tokens`

## Functions

### `embeddingLookup`

Performs standard embedding lookup from input token IDs.

```cpp
void embeddingLookup(
    rt::Tensor const& inputIds,           // [batchSize, seqLen]
    rt::Tensor const& embeddingTable,     // [vocabSize, hiddenSize]
    rt::Tensor& output,                   // [batchSize, seqLen, hiddenSize]
    cudaStream_t stream = 0
);
```

**Parameters:**
- `inputIds`: Input token IDs with shape `[batchSize, seqLen]` (INT32)
- `embeddingTable`: Embedding table with shape `[vocabSize, hiddenSize]` (FP16)
- `output`: Output hidden states with shape `[batchSize, seqLen, hiddenSize]` (FP16)
- `stream`: CUDA stream for asynchronous execution (optional)

### `embeddingLookupWithImageInsertion`

Performs embedding lookup with image embedding insertion following the PromptTuningEmbedding logic.

```cpp
void embeddingLookupWithImageInsertion(
    rt::Tensor const& inputIds,           // [batchSize, seqLen]
    rt::Tensor const& embeddingTable,     // [vocabSize, hiddenSize]
    rt::Tensor const& imageEmbeds,        // [imageTokenLen, hiddenSize]
    int32_t vocabSize,                    // Vocabulary size for normal tokens
    rt::Tensor& output,                   // [batchSize, seqLen, hiddenSize]
    cudaStream_t stream = 0
);
```

**Parameters:**
- `inputIds`: Input token IDs with shape `[batchSize, seqLen]` (INT32)
- `embeddingTable`: Embedding table with shape `[vocabSize, hiddenSize]` (FP16)
- `imageEmbeds`: Image embeddings with shape `[imageTokenLen, hiddenSize]` (FP16)
- `vocabSize`: Vocabulary size for normal tokens (tokens > vocabSize are treated as image tokens)
- `output`: Output hidden states with shape `[batchSize, seqLen, hiddenSize]` (FP16)
- `stream`: CUDA stream for asynchronous execution (optional)

**Logic:**
- For tokens with `tokenId <= vocabSize - 1`: Use `embeddingTable[tokenId]`
- For tokens with `tokenId > vocabSize - 1`: Use `imageEmbeds[tokenId - vocabSize]`
- Error handling: If `tokenId - vocabSize >= imageTokenLen`, use zero embedding

## Usage Example

```cpp
#include "embeddingKernels.h"

// Create input tensors
rt::Coords inputShape{2, 5};  // [batchSize, seqLen]
rt::Tensor inputIds(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

rt::Coords embeddingShape{1000, 512};  // [vocabSize, hiddenSize]
rt::Tensor embeddingTable(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

rt::Coords outputShape{2, 5, 512};  // [batchSize, seqLen, hiddenSize]
rt::Tensor output(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

// Perform standard embedding lookup
kernels::embeddingLookup(inputIds, embeddingTable, output);

// For multimodal models with image embeddings
rt::Coords imageShape{3, 512};  // [imageTokenLen, hiddenSize]
rt::Tensor imageEmbeds(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

kernels::embeddingLookupWithImageInsertion(inputIds, embeddingTable, imageEmbeds, 1000, output);
```