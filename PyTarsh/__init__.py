"""
Neural Network Library called PyTarsh aw PyZart
"""

from .layers import Layer, Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax, Linear, get_activation
from .losses import mse_loss, mse_gradient
from .optimizers import SGD
from .network import Sequential

__version__ = "1.0.0"
__author__ = "Omar Mohsen,Youssf Mostafa,Ahmed Abdallah"

__all__ = [
    # Layers
    'Layer',
    'Dense',
    
    # Activations
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'Linear',
    'get_activation',
    
    # Losses
    'mse_loss',
    'mse_gradient',
    
    # Optimizers
    'SGD',
    
    # Models
    'Sequential'
]