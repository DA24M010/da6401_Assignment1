# da6401_Assignment1
Repo for assignment1 submission in DA6401

## Project Overview
This project implements a **Feedforward Neural Network** from scratch using **NumPy**. The network is trained and tested on the **Fashion-MNIST dataset** for classifying images into **10 different categories**.

The goal of this assignment is twofold:
1. Implement and use gradient descent (and its variants) with backpropagation for a classification task.
2. Get familiar with Weights & Biases (WandB), a tool for running and tracking multiple experiments efficiently.

## Project Structure
```
├── nn.py              
├── optimizer.py          
├── plot.py                
├── run_hyperparameter_tuning.py 
├── confusion_matrix.py    
├── train.py               
├── requirements.txt        
└── README.md               

```

- **`nn.py`**: Contains the FeedforwardNN class, implementing functions like initialize_weights, activation functions, train, forward, backward, and load_data.
- **`plot.py`**: Utility script for generating plots of one sample image per class from the Fashion-MNIST dataset in WandB, as required in Question 1.
- **`optimizer.py`**: Defines various optimizers (SGD, Momentum, Nesterov, RMSprop, Adam, NADAM) for use in `nn.py`.
- **`run_hyperparameter_tuning.py`**: Runs hyperparameter tuning using WandB sweeps, allowing automated searches for optimal values.
- **`run_model_w_mse_loss.py`**: Runs hyperparameter tuning sweeps using MSE loss to compare against cross-entropy loss.
- **`confusion_matrix.py`**: Plots the confusion matrix and reports test accuracy for the best model found using hyperparameter tuning.
- **`train.py`**: Implements training functionality and accepts command-line arguments to configure and run training experiments via WandB.

## Hyperparameter Tuning

WandB sweeps are used to explore various configurations for training the neural network. The following hyperparameters are considered:

- **Number of epochs**: {5, 10}
- **Number of hidden layers**: {3, 4, 5}
- **Hidden layer size**: {32, 64, 128}
- **Weight decay (L2 regularization)**: {0, 0.0005, 0.5}
- **Learning rate**: {1e-3, 1e-4}
- **Optimizer**: {SGD, Momentum, Nesterov, RMSprop, Adam, NADAM}
- **Batch size**: {16, 32, 64}
- **Weight initialization**: {Random, Xavier}
- **Activation functions**: {Sigmoid, Tanh, ReLU}

WandB will automatically generate plots for the sweeps. Each sweep is given a meaningful name (e.g., 
`LR_0.001_HL_3_HLS_64_OPT_nadam_ACTIVATION_tanh_NUM_EPOCHS_10_BATCH_SIZE_16_W_INIT_random_W_DECAY_0`) instead of using default WandB names.

## Running the Training Script

The `train.py` script supports various command-line arguments for customizing the training process.
The script generates training plots(loss, accuracy plots) for the following arguments and log them to the wandb project and entity specified in the command-line.

```bash
python train.py --wandb_entity myname --wandb_project myprojectname --dataset fashion_mnist --batch_size 128 --epochs 20
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
