"""
Activation functions and their derivatives
"""

import numpy as np


class Activation:
    """
    Base class for activation functions
    """
    
    @staticmethod
    def forward(x):
        """Apply activation function"""
        raise NotImplementedError
    
    @staticmethod
    def backward(x):
        """Compute derivative of activation function"""
        raise NotImplementedError


class ReLU(Activation):
    """
    Rectified Linear Unit activation
    f(x) = max(0, x)
    f'(x) = 1 if x > 0, else 0
    """
    
    @staticmethod
    def forward(x):
        """
        Apply ReLU activation
        
        Args:
            x: Input array
            
        Returns:
            max(0, x)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        """
        Compute ReLU derivative
        
        Args:
            x: Input array (before activation)
            
        Returns:
            Derivative: 1 where x > 0, else 0
        """
        return (x > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid activation
    f(x) = 1 / (1 + e^(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    @staticmethod
    def forward(x):
        """
        Apply sigmoid activation
        
        Args:
            x: Input array
            
        Returns:
            1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x):
        """
        Compute sigmoid derivative
        
        Args:
            x: Input array (before activation)
            
        Returns:
            Derivative: sigmoid(x) * (1 - sigmoid(x))
        """
        sig = Sigmoid.forward(x)
        return sig * (1 - sig)


class Tanh(Activation):
    """
    Hyperbolic tangent activation
    f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    f'(x) = 1 - tanh(x)^2
    """
    
    @staticmethod
    def forward(x):
        """
        Apply tanh activation
        
        Args:
            x: Input array
            
        Returns:
            tanh(x)
        """
        return np.tanh(x)
    
    @staticmethod
    def backward(x):
        """
        Compute tanh derivative
        
        Args:
            x: Input array (before activation)
            
        Returns:
            Derivative: 1 - tanh(x)^2
        """
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    """
    Softmax activation (for multi-class classification)
    f(x_i) = e^(x_i) / sum(e^(x_j))
    """
    
    @staticmethod
    def forward(x):
        """
        Apply softmax activation
        
        Args:
            x: Input array of shape (batch_size, num_classes)
            
        Returns:
            Probability distribution over classes
        """
        # Subtract max for numerical stability
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x):
        """
        Compute softmax derivative (simplified)
        
        Args:
            x: Input array (before activation)
            
        Returns:
            Derivative (simplified for use with cross-entropy loss)
        """
        # Simplified derivative - 3shan full Jacobian is complex
        # This works well when combined with cross-entropy loss
        return np.ones_like(x)


class Linear(Activation):
    """
    Linear activation (no activation)
    f(x) = x
    f'(x) = 1
    """
    
    @staticmethod
    def forward(x):
        """
        Apply linear activation (identity function)
        
        Args:
            x: Input array
            
        Returns:
            x (unchanged)
        """
        return x
    
    @staticmethod
    def backward(x):
        """
        Compute linear derivative
        
        Args:
            x: Input array
            
        Returns:
            Derivative: 1 (ones with same shape)
        """
        return np.ones_like(x)


# Activation registry for easy lookup
ACTIVATION_FUNCTIONS = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'linear': Linear,
    None: Linear  # None defaults to linear (no activation)
}


def get_activation(name):
    """
    Get activation function class by name
    
    Args:
        name: Name of activation ('relu', 'sigmoid', 'tanh', 'softmax', None)
        
    Returns:
        Activation class
        
    Raises:
        ValueError: If activation name is not recognized
    """
    if name not in ACTIVATION_FUNCTIONS:
        valid_names = [k for k in ACTIVATION_FUNCTIONS.keys() if k is not None]
        raise ValueError(
            f"Unknown activation function: {name}. "
            f"Valid options are: {valid_names}"
        )
    
    return ACTIVATION_FUNCTIONS[name]