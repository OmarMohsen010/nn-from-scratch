class SGD:
    """
    Stochastic Gradient Descent Optimizer
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []

    def register_layers(self, layers):
        """
        Store reference to layers so step() knows what to update
        """
        self.layers = layers

    def step(self):
        """
        Update parameters for all registered layers
        """
        for layer in self.layers:
            # Only update layers that have the update_parameters method (Dense)
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(self.learning_rate)