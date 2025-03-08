import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layers, biases, grads):
        for i in range(len(layers)):
            layers[i] -= self.learning_rate * grads['dw'][i]
            biases[i] -= self.learning_rate * grads['db'][i]

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, layers, biases, grads):
        # Initially velocity v(-1) is zero
        if self.velocity is None:
            self.velocity = {'dw': [np.zeros_like(w) for w in layers], 
                             'db': [np.zeros_like(b) for b in biases]}

        for i in range(len(layers)):
            self.velocity['dw'][i] = self.momentum * self.velocity['dw'][i] + (1 - self.momentum) * grads['dw'][i]
            self.velocity['db'][i] = self.momentum * self.velocity['db'][i] + (1 - self.momentum) * grads['db'][i]

            layers[i] -= self.learning_rate * self.velocity['dw'][i]
            biases[i] -= self.learning_rate * self.velocity['db'][i]

class NesterovOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layers, biases, grads):
        """
        Updates weights and biases using Nesterov Accelerated Gradient Descent.
        """
        if self.velocity_w is None:  # Initialize velocities to zero
            self.velocity_w = [np.zeros_like(w) for w in layers]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(layers)):
            # Compute lookahead velocity update
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grads['dw'][i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grads['db'][i]

            # Apply Nesterov lookahead update
            layers[i] += self.momentum * self.velocity_w[i] - self.learning_rate * grads['dw'][i]
            biases[i] += self.momentum * self.velocity_b[i] - self.learning_rate * grads['db'][i]


