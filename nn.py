import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb
from optimizer import *

class FeedforwardNN:
    # Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.
    # Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.
    def __init__(self, num_layers=1, hidden_size=64, learning_rate=0.01, momentum = 0.9, activation='relu', optimizer=None, weight_init="random"):
        self.layers = []
        self.biases = []
        # Output size for fashion mnist and mnist
        self.output_size = 10
        # Input size for fashion mnist and mnist, 28*28 images
        self.input_size = 28*28
        prev_size = self.input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.activation = activation
        self.momentum = momentum
        self.optimizer = optimizer if optimizer == None else self.set_optimizer(optimizer, learning_rate)

        for _ in range(num_layers):
            self.layers.append(self.initialize_weights(weight_init, prev_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size

        self.layers.append(self.initialize_weights(weight_init, prev_size, self.output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        
        # # Print layer sizes
        for i in range(len(self.layers)):
            print(f"Layer {i}: {self.layers[i].shape}, Bias: {self.biases[i].shape}")
    
    def initialize_weights(self, method, input_size, output_size):
        """
        Initializes weights based on the given method.
        
        Args:
            method (str): "random" or "Xavier".
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.

        """
        if method == "Xavier":
            return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        else:  # Default: Random small values
            return np.random.randn(input_size, output_size) * 0.01
    
    def set_optimizer(self, optimizer, learning_rate):
        opt_name = optimizer.lower()
        if(opt_name == 'sgd'):
            return SGDOptimizer(learning_rate)
        elif(opt_name == 'momentum'):
            return MomentumOptimizer(learning_rate, self.momentum)
        elif(opt_name == 'nesterov'):
            return NesterovOptimizer(learning_rate, self.momentum)
        else:
            return SGDOptimizer(learning_rate)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def activate(self, x):
        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
    
    def activate_derivative(self, x):
        if self.activation == 'relu':
            return self.relu_derivative(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return self.tanh_derivative(x)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        # X shape : m, input_size_l
        # w shape : layer_size_l*layer_size_l+1
        # b shape : 1*layer_size_l+1
        # outputs a probability distribution over the 10 classes
        self.a = [x]
        self.z = []  # Store pre-activation values

        for i in range(len(self.layers) - 1):
            z = np.dot(x, self.layers[i]) + self.biases[i]
            self.z.append(z)  # Store z before activation
            x = self.activate(z)
            self.a.append(x)

        z = np.dot(x, self.layers[-1]) + self.biases[-1]
        self.z.append(z)
        x = self.softmax(z)  # Softmax activation for output
        self.a.append(x)

        return x

    
    # def backward(self, y):
    #     m = y.shape[0]
    #     da = self.a[-1] - y
    #     grads = {'dw': [], 'db': []}
        
    #     for i in range(len(self.layers) - 1, -1, -1):
    #         dw = np.dot(self.a[i].T, da) / m
    #         db = np.sum(da, axis=0, keepdims=True) / m
    #         grads['dw'].insert(0, dw)
    #         grads['db'].insert(0, db)

    #         if i > 0:
    #             da = np.dot(da, self.layers[i].T) * self.activate_derivative(self.a[i])
        
    #     self.optimizer.update(self.layers, self.biases, grads)

    def backward(self, y, lookahead=False):
        """
        Compute gradients.
        If lookahead=True (for Nesterov), temporarily shift weights before computing gradients.
        """
        m = y.shape[0]
        grads = {'dw': [], 'db': []}

        # Ensure optimizer has initialized velocities before using them
        if isinstance(self.optimizer, NesterovOptimizer):
            if self.optimizer.velocity_w is None:
                self.optimizer.velocity_w = [np.zeros_like(w) for w in self.layers]
                self.optimizer.velocity_b = [np.zeros_like(b) for b in self.biases]

        # Use modified weights if lookahead is enabled for Nesterov
        if lookahead and isinstance(self.optimizer, NesterovOptimizer):
            layers_to_use = [w - self.optimizer.momentum * v for w, v in zip(self.layers, self.optimizer.velocity_w)]
            # biases_to_use = [b - self.optimizer.momentum * v for b, v in zip(self.biases, self.optimizer.velocity_b)]
        else:
            layers_to_use = self.layers
            # biases_to_use = self.biases

        # Compute activations at the selected position
        da = self.a[-1] - y

        for i in range(len(layers_to_use) - 1, -1, -1):
            dw = np.dot(self.a[i].T, da) / m
            db = np.sum(da, axis=0, keepdims=True) / m
            grads['dw'].insert(0, dw)
            grads['db'].insert(0, db)

            if i > 0:
                da = np.dot(da, layers_to_use[i].T) * (self.activate_derivative(self.z[i-1]))

        return grads

    def compute_accuracy(self, y_pred, y_true):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        if(isinstance(self.optimizer, SGDOptimizer)):
            batch_size = 1
        for epoch in range(epochs):
            num_samples = X_train.shape[0]
            
            # Shuffle data before each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                # Forward pass
                self.forward(X_batch)

                # Check if optimizer is Nesterov (lookahead required)
                lookahead = isinstance(self.optimizer, NesterovOptimizer)
                # Backward pass: computes grads (with or without lookahead)
                grads = self.backward(y_batch, lookahead=lookahead)
                self.optimizer.update(self.layers, self.biases, grads)

            # Compute Loss and Accuracy
            y_train_pred = self.forward(X_train)
            train_loss = -np.mean(y_train * np.log(y_train_pred + 1e-8))
            train_accuracy = self.compute_accuracy(y_train_pred, y_train)

            y_val_pred = self.forward(X_val)
            val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-8))
            val_accuracy = self.compute_accuracy(y_val_pred, y_val)

            # Log metrics to W&B
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })

            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    
    def summary(self):
        print("Model Summary:")
        print("---------------------------------")
        print(f"Input Layer: {self.layers[0].shape[0]} neurons")
        for i, hidden_size in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i+1}: {hidden_size} neurons, Activation: {self.activation}")
        print(f"Output Layer: {self.layers[-1].shape[1]} neurons (Softmax activation)")
        print("---------------------------------")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = pd.get_dummies(y_train).values

# Shuffle indices
num_samples = X_train.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split data into training and validation sets (90%-10%)
val_split = int(0.9 * num_samples)
train_indices, val_indices = indices[:val_split], indices[val_split:]
X_train, X_val = X_train[train_indices], X_train[val_indices]
y_train, y_val = y_train[train_indices], y_train[val_indices]

# Sweep Configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.001, 0.01]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov"]},
        "epochs": {"values": [5, 10]},
        "num_hidden": {"values": [3, 4]},
        "hidden_size": {"values": [32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "batch_size": {"values": [16, 32]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401 Assignments")

def train_sweep():
    run = wandb.init(project="DA6401 Assignments", entity="da24m010-indian-institute-of-technology-madras") 
    config = wandb.config 
    run.name = f"LR_{config.learning_rate}_HL_{config.num_hidden}_HLS_{config.hidden_size}_OPT_{config.optimizer}_ACTIVATION_{config.activation}_NUM_EPOCHS_{config.epochs}_BATCH_SIZE_{config.batch_size}_W_INIT_{config.weight_init}"

    model = FeedforwardNN(num_layers=config.num_hidden, hidden_size=config.hidden_size, learning_rate=config.learning_rate,
                          activation=config.activation, optimizer=config.optimizer, weight_init=config.weight_init)

    model.train(X_train, y_train, X_val, y_val, epochs=config.epochs, batch_size=config.batch_size)

wandb.agent(sweep_id, function=train_sweep)
