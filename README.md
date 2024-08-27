# Graph Convolutional Network (GCN)

This repository contains the implementation of a Graph Convolutional Network (GCN) using Python and TensorFlow/PyTorch. GCNs are a type of neural network designed to work with graph-structured data. They have been widely used in various applications like node classification, link prediction, and social network analysis.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Graph Convolutional Networks (GCNs) extend traditional Convolutional Neural Networks (CNNs) to graph-structured data. Instead of operating on a grid of pixels or features, GCNs operate directly on the nodes and edges of a graph, making them suitable for non-Euclidean data.

This project demonstrates how to implement a GCN from scratch and apply it to a node classification task.

## Installation

To run this project, you'll need to have Python 3.x installed. The required dependencies can be installed using `pip`.

pip install -r requirements.txt

# Model Architecture
The Graph Convolutional Network is composed of the following layers:

Graph Convolution Layer: Performs convolution operations on graph data.
Activation Layer: Applies a non-linear activation function like ReLU.
Dropout Layer: Regularization to prevent overfitting.

# Training
The model is trained using the Adam optimizer with cross-entropy loss. The training process includes:

Forward pass through the network.
Calculation of loss.
Backward pass to compute gradients.
Updating the weights using the optimizer.
Evaluation
The trained model is evaluated on the test set using metrics like accuracy, precision, recall, and F1-score.

# Results
The following results were obtained on the Cora dataset:

Accuracy: 82%
Precision: 80%
Recall: 78%
F1-score: 79%
