"""
layers.py
Base Layer class and Dense (Fully Connected) Layer implementation
"""

import numpy as np


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
    
    def __init__(self, input_size, output_size):
        """
        Initialize Dense layer
        
        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
        """
        # Initialize weights with small random values
        # Xavier/He initialization can be used for better performance
        self.weights = np.random.randn(input_size, output_size) * 0.1
        
        # Initialize biases to zeros
        self.biases = np.zeros((1, output_size))
        
        # Placeholders for gradients
        self.weights_gradient = None
        self.biases_gradient = None
        
        # Store input for backward pass
        self.input = None
    
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
        output = np.dot(input, self.weights) + self.biases
        
        return output
    
    def backward(self, output_gradient):
        """
        Backward pass: compute gradients
        
        Given ∂L/∂Y, compute:
        - ∂L/∂X = ∂L/∂Y · W^T (to pass to previous layer)
        - ∂L/∂W = X^T · ∂L/∂Y (to update weights)
        - ∂L/∂b = sum(∂L/∂Y) (to update biases)
        
        Args:
            output_gradient: Gradient of loss w.r.t. output (∂L/∂Y)
                           Shape: (batch_size, output_size)
            
        Returns:
            input_gradient: Gradient of loss w.r.t. input (∂L/∂X)
                          Shape: (batch_size, input_size)
        """
        # Compute gradient with respect to weights: ∂L/∂W = X^T · ∂L/∂Y
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Compute gradient with respect to biases: ∂L/∂b = sum(∂L/∂Y, axis=0)
        # Keep dims for broadcasting
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Compute gradient with respect to input: ∂L/∂X = ∂L/∂Y · W^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        
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