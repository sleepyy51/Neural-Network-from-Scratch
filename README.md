# Neural Network from Scratch (NumPy)

A learning-focused project that implements neural networks from scratch using **NumPy**.  
It includes:
- a fully connected network for the XOR problem,
- a simple CNN pipeline for MNIST digits (**0 vs 1**).

The goal is to understand forward pass, backpropagation, and parameter updates without high-level deep learning frameworks.

## Project Overview

Core components are implemented manually:

- `layer.py`: base layer interface
- `dense_layer.py`: fully connected layer
- `convolutional_layer.py`: 2D convolutional layer (uses `scipy.signal`)
- `activation_layer.py` + `activations.py`: activation wrappers (`Tanh`, `Sigmoid`)
- `reshape_layer.py`: tensor reshape helper between conv and dense blocks
- `losses.py`: MSE and binary cross-entropy + derivatives

Example scripts:
- `solve_xor.py`: trains a small MLP on XOR
- `solve_mnist.py`: trains a small CNN-like model on MNIST classes 0 and 1

## Requirements

- Python 3.10+ (recommended)
- pip

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy tensorflow
```

> `solve_mnist.py` imports `keras.datasets` and `keras.utils` (provided by TensorFlow/Keras).

## Run the Examples

### 1. XOR (MLP)

```bash
python solve_xor.py
```

What you should see:
- training progress per epoch (`error = ...`)
- error decreasing over time

### 2. MNIST (CNN, classes 0 vs 1)

```bash
python solve_mnist.py
```

What you should see:
- training progress for 20 epochs
- test predictions printed as `pred: X, true: Y`

## Notes

- The learning setup is intentionally simple and educational.
- The MNIST script currently trains on a limited subset (`0` and `1`, with `limit=100`) for quick experimentation.
