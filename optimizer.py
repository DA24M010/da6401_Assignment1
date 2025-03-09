import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, weight_decay = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def update(self, layers, biases, grads):
        for i in range(len(layers)):
            layers[i] -= self.learning_rate * grads['dw'][i] + self.weight_decay * layers[i]
            biases[i] -= self.learning_rate * grads['db'][i]

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.weight_decay = weight_decay

    def update(self, layers, biases, grads):
        # Initially velocity v(-1) is zero
        if self.velocity is None:
            self.velocity = {'dw': [np.zeros_like(w) for w in layers], 
                             'db': [np.zeros_like(b) for b in biases]}

        for i in range(len(layers)):
            self.velocity['dw'][i] = self.momentum * self.velocity['dw'][i] + (1 - self.momentum) * grads['dw'][i]
            self.velocity['db'][i] = self.momentum * self.velocity['db'][i] + (1 - self.momentum) * grads['db'][i]

            layers[i] -= self.learning_rate * self.velocity['dw'][i] + self.weight_decay * layers[i]
            biases[i] -= self.learning_rate * self.velocity['db'][i]

class NesterovOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None
        self.weight_decay = weight_decay

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
            layers[i] += self.momentum * self.velocity_w[i] - self.learning_rate * grads['dw'][i] + self.weight_decay * layers[i]
            biases[i] += self.momentum * self.velocity_b[i] - self.learning_rate * grads['db'][i]

class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay = 0.0):
        self.learning_rate = learning_rate
        # beta for RMSprop
        self.beta = beta
        # epsilon for RMSprop
        self.epsilon = epsilon
        self.squared_gradients_w = None
        self.squared_gradients_b = None
        self.weight_decay = weight_decay

    def update(self, layers, biases, grads):
        if self.squared_gradients_w is None:
            self.squared_gradients_w = [np.zeros_like(w) for w in layers]
            self.squared_gradients_b = [np.zeros_like(b) for b in biases]

        for i in range(len(layers)):
            # Update moving average of squared gradients
            self.squared_gradients_w[i] = (
                self.beta * self.squared_gradients_w[i] + (1 - self.beta) * np.square(grads['dw'][i])
            )
            self.squared_gradients_b[i] = (
                self.beta * self.squared_gradients_b[i] + (1 - self.beta) * np.square(grads['db'][i])
            )

            # Weight and bias updates
            layers[i] -= (self.learning_rate / np.sqrt(self.squared_gradients_w[i] + self.epsilon)) * grads['dw'][i] + self.weight_decay * layers[i]
            biases[i] -= (self.learning_rate / np.sqrt(self.squared_gradients_b[i] + self.epsilon)) * grads['db'][i]

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay = 0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1  
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None 
        self.v_w = None 
        self.m_b = None  
        self.v_b = None
        self.t = 0  # Time step
        self.weight_decay = weight_decay

    def update(self, layers, biases, grads):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in layers]
            self.v_w = [np.zeros_like(w) for w in layers]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1 

        for i in range(len(layers)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads['dw'][i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads['db'][i]

            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(grads['dw'][i])
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grads['db'][i])

            # Bias correction 
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            layers[i] -= (self.learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * m_w_hat + self.weight_decay * layers[i]
            biases[i] -= (self.learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat

class NAdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay = 0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.m_w = None 
        self.v_w = None 
        self.m_b = None  
        self.v_b = None 
        self.t = 0 
        self.weight_decay = weight_decay

    def update(self, layers, biases, grads):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in layers]
            self.v_w = [np.zeros_like(w) for w in layers]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1 

        for i in range(len(layers)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads['dw'][i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads['db'][i]

            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(grads['dw'][i])
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grads['db'][i])

            # Bias correction 
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Nadam modification: Nesterov momentum applied to m_hat
            m_w_nesterov = self.beta1 * m_w_hat + ((1 - self.beta1)/(1 - self.beta1 ** (self.t+1))) * grads['dw'][i]
            m_b_nesterov = self.beta1 * m_b_hat + ((1 - self.beta1)/(1 - self.beta1 ** (self.t+1))) * grads['db'][i]

            layers[i] -= (self.learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * m_w_nesterov + self.weight_decay * layers[i]
            biases[i] -= (self.learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_nesterov
