# Backpropagation Neural Network Project

This project implements a custom neural network for multi-class classification using NumPy and backpropagation. The goal is to train a model from scratch, understand the learning process, and compare different activation functions on a challenging dataset.

## Overview

The model is a fully connected multilayer perceptron (MLP) with:

- Input layer: 354 features
- Hidden layer 1: 256 neurons
- Hidden layer 2: 128 neurons
- Output layer: 8 classes

It uses:

- Backpropagation for gradient computation
- Momentum-based optimization
- Mini-batch training
- One-hot encoding for class labels
- SMOTE to balance the imbalanced dataset

## Project purpose

This project demonstrates how neural networks learn by adjusting weights and biases through the error signal propagated backward from the output layer to earlier layers. It is a practical exercise in:

- forward propagation
- loss computation
- backpropagation
- activation functions
- training dynamics and model evaluation

## Files in this repository

- `backprop.ipynb` — main training notebook with the network implementation and experiments
- `data.csv` — raw feature data
- `data_labels.csv` — class labels
- `test_set.csv` — test inputs for final evaluation
- `smote/` — processed training and validation datasets after SMOTE balancing
- `results/` — saved metrics, learning curves, and trained weights
- `report/` — the project report in LaTeX/PDF format

## Network design

The notebook compares two activation setups:

- Network A: Tanh → Tanh → Logistic
- Network B: Leaky ReLU → Leaky ReLU → Logistic

The output layer uses a logistic activation because this is a multi-class classification task with one-hot encoded labels.

## Training workflow

1. Load the dataset and labels
2. Apply one-hot encoding to class targets
3. Balance the dataset using SMOTE
4. Train the network with backpropagation and momentum
5. Evaluate validation loss and misclassification rate
6. Save weight matrices and performance summaries

## Example results

From the notebook, the model shows clear learning progress during training. For example:

- Initial validation error was high, with many misclassifications
- After several epochs, both training and validation error dropped significantly
- The network converged to a low error rate with a small number of validation misclassifications

This demonstrates the effectiveness of the backpropagation algorithm when combined with proper preprocessing and activation choices.

## Run the project

You can run the notebook in Jupyter or VS Code:

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook backprop.ipynb
```

Then execute the cells in order to:

- load the data
- train the network
- view learning curves
- evaluate performance
- export results

## Notes

This project is intended as a learning-focused implementation of neural networks and backpropagation, rather than a production-level deep learning library. It is useful for understanding the mechanics behind training neural networks from first principles.

## License

This project is for academic and learning purposes.
