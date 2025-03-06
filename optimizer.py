class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layers, biases, grads):
        for i in range(len(layers)):
            layers[i] -= self.learning_rate * grads['dw'][i]
            biases[i] -= self.learning_rate * grads['db'][i]

