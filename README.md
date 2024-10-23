
# Vision Transformer (ViT) from Scratch

This project demonstrates how to build a Vision Transformer (ViT) model from scratch using PyTorch. The code includes patch embedding, multi-head attention, feedforward networks, and training on the Oxford-IIIT Pet dataset.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Image Patching](#image-patching)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Vision Transformers Explained](#vision-transformers-explained)

## Overview
This repository contains the implementation of a Vision Transformer (ViT) model. The goal is to understand how transformers, originally designed for NLP tasks, can be adapted for vision tasks such as image classification.

## Setup

You can install the required dependencies by running:

```bash
pip install einops torch torchvision matplotlib
```

## Image Patching

Images are divided into smaller patches, and each patch is flattened and embedded into a vector space. This is a key part of how vision transformers process images, which avoids the need for traditional convolutional layers.

## Model Architecture

The model consists of:

- Patch embedding layer that breaks down images into patches.
- Transformer encoder blocks, which include attention mechanisms and feed-forward networks.
- A classification head to predict the output class.

Here's an overview of the architecture:

![Transformer Architecture](https://miro.medium.com/v2/resize:fit:828/format:webp/1*4A9zG9nUt2VsSIuNENAu4w.png)

### Patch Embedding
Patch embedding involves dividing the image into smaller patches and flattening each patch. These patches are then passed through a linear layer to project them into the embedding space.

### Multi-Head Attention
The core of the transformer is the multi-head attention mechanism, where the model learns to attend to different parts of the input simultaneously.

### Residual Connections
Each block uses residual connections to ensure stable gradients during backpropagation, allowing for deeper architectures.

### Output
After the transformer layers, a classification token is used for the final classification, followed by a simple feed-forward network to output the predicted class.

## Training

The model is trained using the Oxford-IIIT Pet dataset. It can be fine-tuned for other datasets as well by adjusting the input size, patch size, and the number of transformer layers.

Training for a 1000 epochs is required for the model to achieve reasonable performance. The following optimizer and loss function are used:

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

## Vision Transformers Explained

Vision Transformers (ViT) treat images as a sequence of patches, just like tokens (words) in NLP tasks. Unlike CNNs, which use convolutions to capture spatial information, transformers rely on the self-attention mechanism to capture relationships between different parts of the image.

The process consists of:

1. **Patch Embedding:** The input image is divided into fixed-size patches. Each patch is flattened and passed through a linear layer to obtain patch embeddings.
2. **Positional Encoding:** Since transformers don't have any inherent understanding of the order of patches, positional encodings are added to the patch embeddings.
3. **Transformer Layers:** The patch embeddings with positional encodings are passed through multiple transformer layers. Each layer applies multi-head self-attention and feed-forward networks.
4. **Classification Head:** A special classification token is concatenated with the patch embeddings, and its final state after the transformer layers is used to make the classification.

The following image shows a high-level overview of a Vision Transformer:

![ViT Overview]([https://i.imgur.com/1aN50Mr.png](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png))

## References

- [Original Transformer paper](https://arxiv.org/abs/1706.03762)
- [Vision Transformer paper](https://arxiv.org/abs/2010.11929)
