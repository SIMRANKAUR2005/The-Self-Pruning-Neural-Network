# Self-Pruning Neural Network

This project implements a self-pruning neural network in PyTorch where the model learns to remove its own weights during training.

## Overview

Traditional pruning is applied after training. In this project, pruning is integrated into the training process using a learnable gating mechanism. Each weight is associated with a gate that determines whether it remains active.

## Key Features

- Custom PrunableLinear layer
- Learnable gate parameters for each weight
- L1 sparsity regularization
- Dynamic pruning during training
- Evaluation with sparsity metrics

## Method

Each weight is multiplied by a gate value:

w' = w × sigmoid(10 × gate_scores)

The model is trained with a combined loss:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = mean(|gate_scores|)

## Training Details

- Dataset: CIFAR-10
- Optimizer: Adam
- Learning rate:
  - Weights: 1e-3
  - Gate scores: 5e-3
- Warmup: First 2 epochs without sparsity loss
- Sparsity threshold: 0.1

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.01   | 49.44       | 17.90        |
| 0.1    | 50.37       | 11.24        |
| 0.5    | 48.52       | 0.85         |

## Observations

- The model achieves a balance between accuracy and sparsity.
- Best trade-off occurs at λ = 0.01.
- Increasing λ does not always increase sparsity due to training dynamics.

## How to Run

1. Install dependencies:
   pip install torch torchvision matplotlib

2. Run training:
   python train.py

## Future Improvements

- Structured pruning
- Hard Concrete gates
- CNN-based architecture
- Performance optimization

## Author

Simran Kaur
