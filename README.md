# mnist-neural-network-c-and-pytorch
A comparative implementation of MNIST digit classification using a custom Neural Network engine written in C from scratch and modern PyTorch models (MLP &amp; CNN).

# MNIST Digit Classification: C (From Scratch) vs. PyTorch

This repository contains two implementations of neural networks trained on the MNIST dataset:
1. **Low-Level C:** A fully connected network built entirely from scratch without external machine learning libraries.
2. **High-Level PyTorch:** A Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN) using modern Deep Learning frameworks.

## 1. C Implementation (From Scratch)
This implementation demonstrates the internal workings of a neural network, including dynamic memory allocation, matrix operations, and the backpropagation algorithm implemented manually.

### Features
* **Language:** C (Standard Libraries only: `math.h`, `stdlib.h`, etc.)
* **Architecture:** Configurable Multi-Layer Perceptron (MLP).
* **Optimization:** Stochastic Gradient Descent (SGD) with mini-batches.
* **Memory Management:** Dynamic allocation for layers, weights, biases, and gradients.

### Benchmarks (from `network.c`)
Various architectures were tested using a custom binary data loader:

| Architecture | Epochs | Learning Rate | Accuracy |
| :--- | :--- | :--- | :--- |
| 784 -> 10 | 200 | 1.0 | **92.14%** |
| 784 -> 16 -> 10 | 300 | 0.5 | **92.37%** |
| 784 -> 64 -> 10 | 200 | 1.0 | **94.65%** |
| 784 -> 128 -> 64 -> 10 | 35 | 0.05 | **93.12%** |
| **PyTorch (CNN)** | **Conv2d -> MaxPool -> Conv2d -> FC** | **99.25%** |
| PyTorch (MLP) | 784 -> 128 -> 64 -> 10 | **98.24% **|

### How to Run
1. Ensure `mnist_train_data.bin` and `mnist_test_data.bin` are in the current folder of c source file.
2. Compile the code:
   gcc network.c -o network
