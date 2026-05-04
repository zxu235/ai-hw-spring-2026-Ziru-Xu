# MNIST Image Recognition — Assignment 4.1

Small image recognition neural networks trained and evaluated on the MNIST handwritten digit dataset.

## Overview

Three model architectures are implemented and compared on both clean and noisy test sets:

| Model | Architecture |
|---|---|
| **MLP** | Shallow fully-connected network |
| **CNN** | LeNet-style convolutional network |
| **Transformer** | Patch-based encoder |

---

## Dataset

- **Source:** `torchvision.datasets.MNIST`
- **Train split:** 60,000 images
- **Test split:** 10,000 images
- **Input format:** 28 × 28 grayscale images, normalised to mean=0.5, std=0.5

### Test Variants

| Variant | Description |
|---|---|
| **Clean** | Standard normalised test set |
| **Noisy** | Gaussian noise (σ=0.2) added to each image |

---

## Models

### Shallow MLP
Each image is flattened to a 784-dimensional vector and passed through two hidden layers:

```
Input (784) → Linear(256) → ReLU → Linear(128) → ReLU → Output(10)
```

### CNN
Preserves spatial structure with two convolutional blocks followed by a fully-connected head:

```
Conv2d(1→32, 3×3) → ReLU
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2×2)
Flatten → Linear(64×14×14 → 128) → ReLU → Output(10)
```

### Transformer Encoder
The image is split into **4 × 4 patches** (49 tokens total), each projected into a 64-dimensional embedding space with learnable positional encodings:

```
Patch embedding (4×4 patches → emb_dim=64)
+ Positional encoding
→ TransformerEncoder (2 layers, 4 heads)
→ Global average pool → Output(10)
```

---

## Training

- **Optimiser:** Adam (lr = 1e-3)
- **Loss:** CrossEntropyLoss
- **Epochs:** 5
- **Batch size:** 64 (train), 1000 (test)

---

## Results

Each model is evaluated on the **clean** test set and the **noisy** test set. Incorrect predictions are visualised (predicted label / true label) for both test variants.

A final bar chart compares all three models side-by-side on clean and noisy accuracy.

---
