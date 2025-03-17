# da6401_Assignment1
Repo for assignment1 submission in DA6401

## Wandb Report link : 
https://wandb.ai/da24m010-indian-institute-of-technology-madras/DA6401%20Assignments/reports/DA6401-Assignment-1--VmlldzoxMTU1NDkyNw

## Github repo line :
https://github.com/DA24M010/da6401_Assignment1.git

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


## Running the Training Script

The `train.py` script supports various command-line arguments for customizing the training process.
The script generates training plots(loss, accuracy plots) for the following arguments and log them to the wandb project and entity specified in the command-line.

```bash
python train.py --wandb_entity myname --wandb_project myprojectname --dataset fashion_mnist --batch_size 128 --epochs 20
```

### Evaluating the Model on the Test Set

If you need to evaluate the model on the test set after training, use the `--log_test` argument:
By default it is set to False.

```bash
python train.py --wandb_entity myname --wandb_project myprojectname --log_test True
```

The model training function supports logging using:

```python
model.train(epochs=args.epochs, batch_size=args.batch_size, dataset=args.dataset, wandb_logs=args.wandb_logs, log_test=args.log_test)
```
-  Running the command logs the testing accuracy, test set confusion matrix to wandb.
- `wandb_logs`: Enables or disables logging results to wandb (default value = True).
- `log_test`: Enables or disables test set evaluation (default value = False).

### Command-Line Arguments

| Argument | Default Value | Description |
|----------|--------------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | WandB project name for tracking experiments |
| `-we`, `--wandb_entity` | `myname` | WandB entity name |
| `-d`, `--dataset` | `fashion_mnist` | Dataset selection (`mnist` or `fashion_mnist`) |
| `-e`, `--epochs` | `10` | Number of training epochs |
| `-b`, `--batch_size` | `32` | Batch size |
| `-l`, `--loss` | `cross_entropy` | Loss function (`mean_squared_error` or `cross_entropy`) |
| `-o`, `--optimizer` | `nadam` | Optimizer selection |
| `-lr`, `--learning_rate` | `0.001` | Learning rate |
| `-m`, `--momentum` | `0.9` | Momentum for momentum-based optimizers |
| `-beta`, `--beta` | `0.9` | Beta for RMSprop |
| `-beta1`, `--beta1` | `0.9` | Beta1 for Adam and NADAM |
| `-beta2`, `--beta2` | `0.999` | Beta2 for Adam and NADAM |
| `-eps`, `--epsilon` | `0.00000001` | Epsilon for numerical stability |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay (L2 regularization) |
| `-w_i`, `--weight_init` | `random` | Weight initialization (`random` or `Xavier`) |
| `-nhl`, `--num_layers` | `3` | Number of hidden layers |
| `-sz`, `--hidden_size` | `64` | Number of hidden neurons per layer |
| `-a`, `--activation` | `tanh` | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |
| `--wandb_logs` | `True` | Enables or disables WandB logging |
| `--log_test` | `False` | Enables or disables evaluation on test set |
