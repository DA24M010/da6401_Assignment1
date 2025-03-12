# da6401_Assignment1
Repo for assignment1 submission in DA6401

## Project Overview
This project implements a **Feedforward Neural Network** from scratch using **NumPy**. The network is trained and tested on the **Fashion-MNIST dataset** for classifying images into **10 different categories**.

## Project Structure
```
├── nn.py                   # Feedforward neural network and training logic
├── optimizer.py            # Implementation of various optimizers
├── plot.py                 # Generates sample plots for the dataset in W&B
├── run_hyperparameter_tuning.py  # Runs W&B sweeps for hyperparameter tuning
├── confusion_matrix.py      # Computes test accuracy and confusion matrix
├── train.py                # Main script to train the model using CLI arguments
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

## Installation
### 1. Clone the repository:
```sh
git clone https://github.com/DA24M010/da6401_Assignment1.git
cd da6401_Assignment1
```

### 2. Install dependencies:
```sh
pip install -r requirements.txt
```

### 3. Setup Weights & Biases (W&B)
Create an account on [W&B](https://wandb.ai/) and log in:
```sh
wandb login
```
