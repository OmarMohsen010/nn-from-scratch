import numpy as np

class MSE:
    """
    Mean Squared Error Loss
    Formula: L = (1/N) * sum((Y_true - Y_pred)^2) 
    """
    def calculate(self, y_pred, y_true):
        """
        Computes the loss value.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.power(y_true - y_pred, 2))

    def calculate_gradient(self, y_pred, y_true):
        """
        Computes the derivative of MSE with respect to predictions.
        Derivative: (2/N) * (Y_pred - Y_true)
        """
        samples = len(y_pred)
        return 2 * (y_pred - y_true) / samples