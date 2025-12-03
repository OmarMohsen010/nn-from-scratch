"""
layers.py
Base Layer class and Dense (Fully Connected) Layer implementation
"""

import numpy as np
from activations import get_activation

class Layer:
    """
    Base class for all layers in the neural network.
    All layers must implement forward() and backward() methods.
    """
    
    def forward(self, input):
        """
        Forward propagation
        
        Args:
            input: Input data
            
        Returns:
            Output of the layer
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def backward(self, output_gradient):
        """
        Backward propagation
        
        Args:
            output_gradient: Gradient of loss with respect to output (∂L/∂Y)
            
        Returns:
            Gradient of loss with respect to input (∂L/∂X)
        """
        raise NotImplementedError("Subclasses must implement backward()")


class Dense(Layer):
    """
    Fully connected (dense) layer
    Performs: Y = XW + b
    """
    
    def __init__(self, input_size, output_size, activation=None):
        """
        Initialize Dense layer
        
        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
            activation: Activation function name ('relu', 'sigmoid', 'tanh', 'softmax', or None(ely heya nkhleha linear))
        """
    
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        
        
        self.biases = np.zeros((1, output_size))
        
        # Store activation function name and get activation class
        self.activation_name = activation
        self.activation_fn = get_activation(activation)
        
        # Placeholders for gradients
        self.weights_gradient = None
        self.biases_gradient = None
        
        # Store intermediate values for backward pass
        self.input = None
        self.linear_output = None  # Output before activation
        self.output = None  # Output after activation
    
    def forward(self, input):
        """
        Forward pass: Y = XW + b
        
        Args:
            input: Input matrix of shape (batch_size, input_size)
            
        Returns:
            Output matrix of shape (batch_size, output_size)
        """
        # Store input for backward pass
        self.input = input
        
        # Compute output: Y = XW + b
        self.linear_output = np.dot(input, self.weights) + self.biases
        
        self.output = self.activation_fn.forward(self.linear_output)

        return self.output
    
    def backward(self, output_gradient):
        """
        Backward pass: compute gradients
        
        Given ∂L/∂Y, compute:
        - ∂L/∂X = ∂L/∂Y · W^T (to pass to previous layer)
        - ∂L/∂W = X^T · ∂L/∂Y (to update weights)
        - ∂L/∂b = sum(∂L/∂Y) (to update biases)
        
        where Z is the linear output (before activation)

        Args:
            output_gradient: Gradient of loss w.r.t. output (∂L/∂Y)
                           Shape: (batch_size, output_size)
            
        Returns:
            input_gradient: Gradient of loss w.r.t. input (∂L/∂X)
                          Shape: (batch_size, input_size)
        """
        
        # ∂L/∂Z = ∂L/∂Y · ∂Y/∂Z (element-wise multiplication)
        activation_derivative = self.activation_fn.backward(self.linear_output)
        grad_linear = output_gradient * activation_derivative
        
        # Step 2: Gradient with respect to weights: ∂L/∂W = X^T · ∂L/∂Z
        self.weights_gradient = np.dot(self.input.T, grad_linear)
        
        # Step 3: Gradient with respect to biases: ∂L/∂b = sum(∂L/∂Z, axis=0)
        self.biases_gradient = np.sum(grad_linear, axis=0, keepdims=True)
        
        # Step 4: Gradient with respect to input: ∂L/∂X = ∂L/∂Z · W^T
        input_gradient = np.dot(grad_linear, self.weights.T)
        
        return input_gradient
    
    def get_parameters(self):
        """
        Get layer parameters
        
        Returns:
            Dictionary containing weights and biases
        """
        return {
            'weights': self.weights,
            'biases': self.biases
        }
    
    def get_gradients(self):
        """
        Get layer gradients
        
        Returns:
            Dictionary containing weight and bias gradients
        """
        return {
            'weights': self.weights_gradient,
            'biases': self.biases_gradient
        }
    
    def update_parameters(self, learning_rate):
        """
        Update parameters using computed gradients (SGD update)
        
        Args:
            learning_rate: Learning rate (η)
        """
        # W_new = W_old - η · ∂L/∂W
        self.weights -= learning_rate * self.weights_gradient
        
        # b_new = b_old - η · ∂L/∂b
        self.biases -= learning_rate * self.biases_gradient