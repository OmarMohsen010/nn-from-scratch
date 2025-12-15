"""
network.py
Sequential neural network model
"""

import numpy as np
from .losses import mse_loss, mse_gradient


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
        loss = loss_fn(predictions, Y)
        
        # Compute loss gradient
        loss_grad = loss_grad_fn(predictions, Y)
        
        # Backward pass
        self.backward(loss_grad)
        
        # Update parameters
        optimizer.step()
        
        return loss
    
    def fit(self, X, Y, epochs, batch_size=32, learning_rate=0.01, 
            loss_fn=None, loss_grad_fn=None, 
            optimizer=None, verbose=True, print_every=100):
        """
        Train the network using Mini-Batch Stochastic Gradient Descent.
        """
        # 1. Set default Loss functions if not provided
        if loss_fn is None: 
            # Local import to avoid circular dependency issues
            from .losses import mse_loss
            loss_fn = mse_loss
        if loss_grad_fn is None: 
            from .losses import mse_gradient
            loss_grad_fn = mse_gradient
        
        # 2. Setup Optimizer if not provided
        if optimizer is None:
            from .optimizers import SGD
            optimizer = SGD(learning_rate=learning_rate)
            optimizer.register_layers(self.layers)
        
        self.loss_history = []
        n_samples = X.shape[0]
        
        # 3. Training Loop
        for epoch in range(epochs):
            # Shuffle data at the start of every epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0
            
            # Mini-Batch Loop
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                # Slice the batch
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # Perform one training step (Forward -> Backward -> Update)
                batch_loss = self.train_step(X_batch, Y_batch, loss_fn, loss_grad_fn, optimizer)
                
                # Accumulate loss (weighted by batch size for accuracy)
                epoch_loss += batch_loss * (end_idx - start_idx)
            
            # Calculate average loss for the entire epoch
            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)
            
            # Print progress
            if verbose: 
                # Print strictly if it matches print_every OR if it's the very first/last epoch
                if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        if verbose:
            print(f"Training completed! Final Loss: {self.loss_history[-1]:.6f}")
        
        return self.loss_history
    
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
    
    def binary_classify(self, predictions, threshold=0.5):
        """
        Convert probabilities to binary class predictions
    
        Args:
        predictions: Array of probabilities
        threshold: Decision threshold (default: 0.5)
    
        Returns:
        Binary predictions (0 or 1)
        """
        return (predictions > threshold).astype(float)