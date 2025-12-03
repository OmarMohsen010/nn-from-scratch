import numpy as np
from .layers import Layer

class ReLU(Layer):
    """
    ReLU Activation Function
    Formula: f(x) = max(0, x)
    """
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        # Derivative is 1 where input > 0, else 0
        return output_gradient * (self.input > 0)

class Sigmoid(Layer):
    """
    Sigmoid Activation Function
    Formula: f(x) = 1 / (1 + e^(-x))
    """
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient):
        # Derivative: f(x) * (1 - f(x))
        s = self.output
        return output_gradient * s * (1 - s)

class Tanh(Layer):
    """
    Tanh Activation Function
    Formula: f(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient):
        # Derivative: 1 - tanh(x)^2
        return output_gradient * (1 - self.output ** 2)

class Softmax(Layer):
    """
    Softmax Activation Function
    """
    def forward(self, input):
        # Numerical stability shift (subtract max)
        e_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        # For a single sample, dL/dx_j = y_j * (dL/dy_j - sum(dL/dy_k * y_k))
        n = np.size(self.output)
        
        # Calculate the dot product (dL/dy . y) for each sample in batch
        # shape: (batch_size, 1)
        dot_product = np.sum(output_gradient * self.output, axis=1, keepdims=True)
        
        return self.output * (output_gradient - dot_product)