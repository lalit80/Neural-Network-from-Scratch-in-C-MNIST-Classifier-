# 🧠 Neural Network from Scratch in C (MNIST Classifier)

This project implements a **feedforward neural network** (multilayer perceptron) completely **from scratch in C**, trained on the **MNIST handwritten digit dataset**.  
No external libraries such as TensorFlow, PyTorch, or OpenCV are used — only standard C libraries (`<math.h>`, `<stdlib.h>`, `<stdio.h>`).

The network learns to classify 28×28 grayscale digit images (0–9) through **stochastic gradient descent (SGD)** and **backpropagation**.

---

## 🚀 Key Features

- **Pure C Implementation** – built entirely without ML libraries.  
- **Configurable Architecture** – define any number of layers and neurons.  
- **Feedforward Computation** – using sigmoid activation.  
- **Backpropagation** – implemented manually to compute gradients.  
- **Mini-Batch Stochastic Gradient Descent (SGD)** – for efficient training.  
- **Binary MNIST Loader** – custom binary file format for fast loading.  
- **Performance Metrics** – prints test accuracy after training.  

---