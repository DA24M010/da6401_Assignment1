import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layers, biases, grads):
        for i in range(len(layers)):
            layers[i] -= self.learning_rate * grads['dw'][i]
            biases[i] -= self.learning_rate * grads['db'][i]

class FeedforwardNN:
    # Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

    # Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.
    def __init__(self, input_size, hidden_layers, learning_rate, activation='relu', optimizer = None):
        self.layers = []
        self.biases = []
        self.output_size = 10
        prev_size = input_size
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer if optimizer == None else self.set_optimizer(optimizer, learning_rate)

        for hidden_size in hidden_layers:
            self.layers.append(np.random.randn(prev_size, hidden_size) * 0.01)
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size
        
        self.layers.append(np.random.randn(prev_size, self.output_size) * 0.01)
        self.biases.append(np.zeros((1, self.output_size)))
        
        for i in range(len(self.layers)):
            print(self.layers[i].shape, self.biases[i].shape)
    
    def set_optimizer(self, optimizer, learning_rate):
        if(optimizer == 'sgd'):
            return SGDOptimizer(learning_rate)
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
        for i in range(len(self.layers) - 1):
            x = self.activate(np.dot(x, self.layers[i]) + self.biases[i])
            self.a.append(x)
        x = self.softmax(np.dot(x, self.layers[-1]) + self.biases[-1])
        self.a.append(x)
        return x
    
    def backward(self, y):
        m = y.shape[0]
        da = self.a[-1] - y
        grads = {'dw': [], 'db': []}
        
        for i in range(len(self.layers) - 1, -1, -1):
            dw = np.dot(self.a[i].T, da) / m
            db = np.sum(da, axis=0, keepdims=True) / m
            grads['dw'].insert(0, dw)
            grads['db'].insert(0, db)

            if i > 0:
                da = np.dot(da, self.layers[i].T) * self.activate_derivative(self.a[i])
        
        self.optimizer.update(self.layers, self.biases, grads)

    def compute_accuracy(self, y_pred, y_true):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                self.forward(X_batch)
                self.backward(y_batch)
            
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

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
y_train = pd.get_dummies(y_train).values
X_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_test = y_test

# number of epochs: 5, 10
# number of hidden layers: 3, 4, 5
# size of every hidden layer: 32, 64, 128
# weight decay (L2 regularisation): 0, 0.0005, 0.5
# learning rate: 1e-3, 1 e-4
# optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
# batch size: 16, 32, 64
# weight initialisation: random, Xavier
# activation functions: sigmoid, tanh, ReLU

val_split = int(0.9 * X_train.shape[0])
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

# Sweep Configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.001, 0.01]},
        "hidden_layers": {"values": [[64], [128, 64], [256, 128, 64]]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]},
        "optimizer": {"values": ["sgd"]},
        "epochs": {"values": [5, 10]},
        "batch_size": {"values": [32, 64]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401 Assignments")

def train_sweep():
    run = wandb.init(project="DA6401 Assignments", entity="da24m010-indian-institute-of-technology-madras") 
    config = wandb.config 
    run.name = f"LR_{config.learning_rate}_HL_{config.hidden_layers}_OPT_{config.optimizer}"


    model = FeedforwardNN(input_size=28*28, hidden_layers=config.hidden_layers, learning_rate=config.learning_rate,
                          activation=config.activation, optimizer= config.optimizer)
    
    model.train(X_train, y_train, X_val, y_val, epochs=config.epochs, batch_size=64)

wandb.agent(sweep_id, function=train_sweep)
