"""
Neural Network Library called PyTarsh aw PyZart
"""

from .layers import Layer, Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import mse_loss, mse_gradient, MSELoss
from .optimizer import SGD, SGDMomentum
from .network import Sequential

__version__ = "1.0.0"
__author__ = "Omar Mohsen,Youssf Mostafa,Ahmed Abdallah"

__all__ = [
    'Layer',
    'Dense',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'mse_loss',
    'mse_gradient',
    'MSELoss',
    'SGD',
    'SGDMomentum',
    'Sequential'
]