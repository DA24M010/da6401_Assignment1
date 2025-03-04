import numpy as np
import pandas as pd

class FeedforwardNN:
    # Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

    # Your code should be flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        self.layers = []
        self.biases = []
        prev_size = input_size
        self.activation = activation
        self.hidden_layers = hidden_layers

        for hidden_size in hidden_layers:
            self.layers.append(np.random.randn(prev_size, hidden_size) * 0.01)
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size
        
        self.layers.append(np.random.randn(prev_size, output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))
        
        for i in range(len(self.layers)):
            print(self.layers[i].shape, self.biases[i].shape)
    
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
    
    def summary(self):
        print("Model Summary:")
        print("---------------------------------")
        print(f"Input Layer: {self.layers[0].shape[0]} neurons")
        for i, hidden_size in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i+1}: {hidden_size} neurons, Activation: {self.activation}")
        print(f"Output Layer: {self.layers[-1].shape[1]} neurons (Softmax activation)")
        print("---------------------------------")

model = FeedforwardNN(784, [64, 128], 10)
model.summary()
x = np.random.randn(1, 784) * 0.01
print(model.forward(x))
print(model.forward(x).shape)