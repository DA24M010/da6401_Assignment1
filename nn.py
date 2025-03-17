import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
from optimizer import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class FeedforwardNN:
    # Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.
    # Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.
    def __init__(self, num_layers=1, hidden_size=64, learning_rate=0.01, momentum = 0.9, activation='relu', optimizer=None, loss_function = "cross_entropy",
                 weight_init="random", beta = 0.9, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.9, weight_decay = 0.0):
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
        self.activation = activation.lower()
        self.loss_function = loss_function.lower()
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.optimizer = optimizer if optimizer == None else self.set_optimizer(optimizer, learning_rate)
        
        for _ in range(num_layers):
            self.layers.append(self.initialize_weights(weight_init, prev_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size

        self.layers.append(self.initialize_weights(weight_init, prev_size, self.output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        self.data_loaded = False
        # # Print layer sizes
        # for i in range(len(self.layers)):
        #     print(f"Layer {i}: {self.layers[i].shape}, Bias: {self.biases[i].shape}")
    
    def initialize_weights(self, method, input_size, output_size):
        if method == "Xavier":
            return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        else:  
            # Default: Random small values
            return np.random.randn(input_size, output_size) * 0.01
    
    def set_optimizer(self, optimizer, learning_rate):
        opt_name = optimizer.lower()
        optimizers = {
            'sgd': SGDOptimizer(learning_rate, self.weight_decay),
            'momentum': MomentumOptimizer(learning_rate, self.momentum, self.weight_decay),
            'nesterov': NesterovOptimizer(learning_rate, self.momentum, self.weight_decay),
            'rmsprop': RMSpropOptimizer(learning_rate, self.beta, self.epsilon, self.weight_decay),
            'adam': AdamOptimizer(learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay),
            'nadam': NAdamOptimizer(learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay)
        }

        # Return the corresponding optimizer, defaulting to SGD if not found
        return optimizers.get(opt_name, SGDOptimizer(learning_rate, self.weight_decay))

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
            return self.relu(x)
    
    def activate_derivative(self, x):
        if self.activation == 'relu':
            return self.relu_derivative(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return self.tanh_derivative(x)
        else:
            return self.relu_derivative(x)

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

        if lookahead and isinstance(self.optimizer, NesterovOptimizer):
            layers_to_use = [w - self.optimizer.momentum * v for w, v in zip(self.layers, self.optimizer.velocity_w)]
            # biases_to_use = [b - self.optimizer.momentum * v for b, v in zip(self.biases, self.optimizer.velocity_b)]
        else:
            layers_to_use = self.layers
            # biases_to_use = self.biases

        # Compute output layer gradient based on the loss function
        if self.loss_function == "cross_entropy":
            da = self.a[-1] - y
        elif self.loss_function == "mean_squared_error":
            da = 2 * (self.a[-1] - y) / m  

        for i in range(len(layers_to_use) - 1, -1, -1):
            dw = np.dot(self.a[i].T, da) / m
            db = np.sum(da, axis=0, keepdims=True) / m
            grads['dw'].insert(0, dw)
            grads['db'].insert(0, db)

            if i > 0:
                da = np.dot(da, layers_to_use[i].T) * (self.activate_derivative(self.z[i-1]))

        return grads

    def compute_accuracy(self, y_pred, y_true):
        """
        Computes the accuracy.
        """
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)
    
    def compute_loss(self, y_true, y_pred):
        """
        Computes the loss based on the specified loss function.
        """
        if self.loss_function == "mean_squared_error":
            return np.mean((y_true - y_pred) ** 2)
        else:
            return -np.mean(y_true * np.log(y_pred + 1e-8))
        
    def load_data(self, dataset, split = 0.9):
        self.data_loaded = True
        if(dataset.lower() == 'fashion_mnist'):
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
        else:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
        X_test = x_test.reshape(x_test.shape[0], -1) / 255.0
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values

        # Shuffle indices
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split data into training and validation sets (as per split percentage)
        # Keeping 10% of the training data aside as validation data for the hyperparameter search. 
        val_split = int(split * num_samples)
        train_indices, val_indices = indices[:val_split], indices[val_split:]
        X_train, X_val = X_train[train_indices], X_train[val_indices]
        y_train, y_val = y_train[train_indices], y_train[val_indices]

        return X_train, X_val, X_test, y_train, y_val, y_test, class_names
    
    def train(self, epochs=10, batch_size=64, dataset = 'fashion_mnist', wandb_logs = False, log_test = False):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.class_names = self.load_data(dataset, split = 0.9)
        if(isinstance(self.optimizer, SGDOptimizer)):
            batch_size = 1
        for epoch in range(epochs):
            num_samples = self.X_train.shape[0]
            
            # Shuffle data before each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            self.X_train, self.y_train = self.X_train[indices], self.y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train[i:i+batch_size]
                # Forward pass
                self.forward(X_batch)

                # Check if optimizer is Nesterov (lookahead required)
                lookahead = isinstance(self.optimizer, NesterovOptimizer)
                # Backward pass: computes grads (with or without lookahead)
                grads = self.backward(y_batch, lookahead=lookahead)
                self.optimizer.update(self.layers, self.biases, grads)

            # Compute Loss and Accuracy
            y_train_pred = self.forward(self.X_train)
            train_loss = self.compute_loss(self.y_train, y_train_pred)
            train_accuracy = self.compute_accuracy(y_train_pred, self.y_train)

            y_val_pred = self.forward(self.X_val)
            val_loss = self.compute_loss(self.y_val, y_val_pred)
            val_accuracy = self.compute_accuracy(y_val_pred, self.y_val)

            if(log_test == False):
                # Log train and val metrics to W&B
                if(wandb_logs):
                    wandb.log({
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "epoch": epoch
                    })

                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            else:
                y_test_pred = self.forward(self.X_test)
                test_loss = self.compute_loss(self.y_test, y_test_pred)
                test_accuracy = self.compute_accuracy(self.y_test, y_test_pred)

                y_test_labels = np.argmax(self.y_test, axis=1)
                y_test_preds = np.argmax(y_test_pred, axis=1)
                
                cm = confusion_matrix(y_test_labels, y_test_preds)
                plt.figure(figsize=(20, 16))
                sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", 
                            xticklabels=self.class_names, yticklabels=self.class_names)

                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title("Confusion Matrix (Fashion-MNIST)")
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)

                if(wandb_logs):
                    wandb.log({
                        "test_accuracy": test_accuracy,
                        "test_confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None, y_true=y_test_labels, preds=y_test_preds, class_names=self.class_names
                        ),
                        "Test Confusion Matrix": wandb.Image(plt),
                    }, step=epoch)

                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    
    def evaluate(self):
        if(self.data_loaded == False):
            print("No Test Data to evaluate!!!")
        else:
            y_test_pred = self.forward(self.X_test)
            test_accuracy = self.compute_accuracy(self.y_test, y_test_pred)
            print("Test Accuracy :", test_accuracy*100.0)

    def summary(self):
        print("Model Summary:")
        print("---------------------------------")
        print(f"Input Layer: {self.layers[0].shape[0]} neurons")
        for i, hidden_size in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i+1}: {hidden_size} neurons, Activation: {self.activation}")
        print(f"Output Layer: {self.layers[-1].shape[1]} neurons (Softmax activation)")
        print("---------------------------------")
