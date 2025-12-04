"""
network.py
Sequential neural network model
"""

import numpy as np
from losses import mse_loss, mse_gradient


class Sequential:
    """
    Sequential neural network model
    Manages layers and orchestrates forward/backward propagation
    """
    
    def __init__(self):

        self.layers = []
        self.loss_history = []
    
    def add(self, layer):
        """
        Add a layer to the network
        
        Args:
            layer: Layer object to add
        """
        self.layers.append(layer)
    
    def forward(self, X):
        """
        Forward propagation through all layers
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            
        Returns:
            Output of the network, shape (batch_size, output_dim)
        """
        output = X
        
        # Pass through each layer sequentially
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def backward(self, loss_gradient):
        """
        Backward propagation through all layers
        
        Args:
            loss_gradient: Gradient of loss w.r.t. network output (∂L/∂Y)
        """
        gradient = loss_gradient
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def predict(self, X):
        """
        Make predictions (same as forward, but clearer name for inference)
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.forward(X)
    
    def train_step(self, X, Y, loss_fn, loss_grad_fn, optimizer):
        """
        Perform one training step (forward + backward + optimize)
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            Y: Target data, shape (batch_size, output_dim)
            loss_fn: Loss function (e.g., mse_loss)
            loss_grad_fn: Loss gradient function (e.g., mse_gradient)
            optimizer: Optimizer object (e.g., SGD)
            
        Returns:
            Loss value for this step
        """
        # Forward pass
        predictions = self.forward(X)
        
        # Compute loss
        loss = loss_fn(Y, predictions)
        
        # Compute loss gradient
        loss_grad = loss_grad_fn(Y, predictions)
        
        # Backward pass
        self.backward(loss_grad)
        
        # Update parameters
        optimizer.step()
        
        return loss
    
    def fit(self, X, Y, epochs, learning_rate=0.01, 
            loss_fn=None, loss_grad_fn=None, 
            optimizer=None, verbose=True, print_every=100):
        """
        Train the network
        
        Args:
            X: Training input data, shape (batch_size, input_dim)
            Y: Training target data, shape (batch_size, output_dim)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            loss_fn: Loss function (default: MSE)
            loss_grad_fn: Loss gradient function (default: MSE gradient)
            optimizer: Optimizer object (default: creates new SGD)
            verbose: Whether to print training progress
            print_every: Print loss every N epochs
            
        Returns:
            List of loss values for each epoch
        """
        # Use MSE as default loss
        if loss_fn is None:
            loss_fn = mse_loss
        if loss_grad_fn is None:
            loss_grad_fn = mse_gradient
        
        # Create optimizer if not provided
        if optimizer is None:
            from optimizers import SGD
            optimizer = SGD(learning_rate=learning_rate)
            optimizer.register_layers(self.layers)
        
        # Training loop
        self.loss_history = []
        
        for epoch in range(epochs):
            # Perform one training step
            loss = self.train_step(X, Y, loss_fn, loss_grad_fn, optimizer)
            
            # Store loss
            self.loss_history.append(loss)
            
            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        if verbose:
            print(f"Training completed! Final Loss: {self.loss_history[-1]:.6f}")
        
        return self.loss_history
    
    def evaluate(self, X, Y, loss_fn=None):
        """
        Evaluate the network on test data
        
        Args:
            X: Test input data
            Y: Test target data
            loss_fn: Loss function (default: MSE)
            
        Returns:
            Loss value on test data
        """
        if loss_fn is None:
            loss_fn = mse_loss
        
        # Forward pass
        predictions = self.forward(X)
        
        # Compute loss
        loss = loss_fn(Y, predictions)
        
        return loss
    
    def get_trainable_layers(self):
        """
        Get list of layers with trainable parameters
        
        Returns:
            List of layers that have weights and biases
        """
        trainable = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                trainable.append(layer)
        return trainable
    
    def summary(self):
        """
        Print network architecture summary
        """
        print("=" * 70)
        print("Network Architecture")
        print("=" * 70)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'weights'):
                # Dense layer
                input_size = layer.weights.shape[0]
                output_size = layer.weights.shape[1]
                params = layer.weights.size + layer.biases.size
                total_params += params
                
                
                if hasattr(layer, 'activation_name') and layer.activation_name:
                    activation_str = f" + {layer.activation_name}"
                else:
                    activation_str = ""
                
                print(f"Layer {i+1}: {layer_name}{activation_str}")
                print(f"  Shape: ({input_size}, {output_size})")
                print(f"  Parameters: {params}")
            else:
                # Other layer types (if any)
                print(f"Layer {i+1}: {layer_name}")
                print(f"  Parameters: 0")
            
            print("-" * 70)
        
        print(f"Total Parameters: {total_params}")
        print("=" * 70)