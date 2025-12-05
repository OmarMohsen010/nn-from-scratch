import numpy as np


"""
    Mean Squared Error Loss
    Formula: L = (1/N) * sum((Y_true - Y_pred)^2) 
    """
def mse_loss(y_pred, y_true):
        """
        Computes the loss value.
        """
        
        return np.mean(np.power(y_true - y_pred, 2))

def mse_gradient(y_pred, y_true):
        """
        Computes the derivative of MSE with respect to predictions.
        Derivative: (2/N) * (Y_pred - Y_true)
        """
        samples = len(y_pred)
        return 2 * (y_pred - y_true) / samples