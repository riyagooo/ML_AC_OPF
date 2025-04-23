import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    """
    Feed-Forward Neural Network for OPF prediction.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128], dropout_rate=0.1):
        """
        Initialize feed-forward neural network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output values
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(FeedForwardNN, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class ConstraintScreeningNN(nn.Module):
    """
    Neural network for constraint screening approach.
    Predicts which constraints will be binding.
    """
    def __init__(self, input_dim, num_constraints, hidden_dims=[64, 128, 64]):
        """
        Initialize constraint screening neural network.
        
        Args:
            input_dim: Dimension of input features
            num_constraints: Number of constraints to classify
            hidden_dims: List of hidden layer dimensions
        """
        super(ConstraintScreeningNN, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        # Add output layer with sigmoid activation for binary classification
        layers.append(nn.Linear(prev_dim, num_constraints))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class WarmStartNN(nn.Module):
    """
    Neural network for warm-starting approach.
    Predicts initial solution for optimization solver.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128], 
                 output_bounds=None, dropout_rate=0.1):
        """
        Initialize warm-start neural network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output values
            hidden_dims: List of hidden layer dimensions
            output_bounds: Tuple of (min_vals, max_vals) tensors for output bounds
            dropout_rate: Dropout rate for regularization
        """
        super(WarmStartNN, self).__init__()
        
        self.output_bounds = output_bounds
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        If output_bounds are provided, constrain outputs to those bounds.
        """
        outputs = self.network(x)
        
        if self.output_bounds is not None:
            min_vals, max_vals = self.output_bounds
            
            # Apply sigmoid and scale to bounds
            # This ensures outputs are within the specified bounds
            outputs = torch.sigmoid(outputs)
            outputs = min_vals + (max_vals - min_vals) * outputs
            
        return outputs 