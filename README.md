# MNIST Handwritten Digit Classification using CNN

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify handwritten digits from the MNIST dataset. The MNIST dataset is a classic dataset in the field of machine learning and computer vision, consisting of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
  
## Overview

In this project, we build and train a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is implemented using PyTorch and achieves high accuracy on both the training and test sets.

## Dataset

The MNIST dataset is automatically downloaded and preprocessed using torchvision. It includes 60,000 training images and 10,000 test images. Each image is normalized to have a mean of 0.1307 and a standard deviation of 0.3081.

## Model Architecture

The CNN model consists of the following layers:
- Two convolutional layers with ReLU activation and max pooling.
- Two fully connected layers with ReLU activation.
- Output layer with 10 units (one for each digit) and softmax activation.

Here is a summary of the architecture:

```python
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Training

The model is trained for 10 epochs using the Adam optimizer and CrossEntropyLoss. Training and test data loaders are created with a batch size of 64. The model is trained on GPU if available, otherwise on CPU.

## Results

The model achieves high accuracy on both the training and test sets. The following metrics are tracked during training:
- Training and test loss
- Training and test accuracy

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required packages using the following command:

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

To run the notebook and train the model, use the following command:

```bash
jupyter notebook MnistCNN.ipynb
```
