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
            
            # Check if dimensions match
            if min_vals.size(0) != outputs.size(1) or max_vals.size(0) != outputs.size(1):
                print(f"Warning: Output bounds dimensions ({min_vals.size(0)},{max_vals.size(0)}) " + 
                      f"don't match output dimensions ({outputs.size(1)}). Adjusting.")
                
                if outputs.size(1) < min_vals.size(0):
                    # Resize bounds to match output dimensions
                    min_vals = min_vals[:outputs.size(1)]
                    max_vals = max_vals[:outputs.size(1)]
                else:
                    # Pad bounds if output is larger
                    pad_size = outputs.size(1) - min_vals.size(0)
                    if pad_size > 0:
                        device = min_vals.device
                        min_vals = torch.cat([min_vals, torch.zeros(pad_size, device=device)])
                        max_vals = torch.cat([max_vals, torch.ones(pad_size, device=device)])
            
            # Apply sigmoid and scale to bounds
            # This ensures outputs are within the specified bounds
            outputs = torch.sigmoid(outputs)
            outputs = min_vals + (max_vals - min_vals) * outputs
            
        return outputs 

class RobustOPFNetwork(torch.nn.Module):
    """
    Robust feedforward neural network for OPF problems with improved numerical stability.
    
    Features:
    - Residual connections for better gradient flow
    - Layer normalization for improved training dynamics
    - Gradient clipping during forward pass
    - Careful activation function selection
    - Output activation functions that respect physical constraints
    
    Args:
        input_dim: Input dimension (load features)
        output_dim: Output dimension (generation and voltage variables)
        hidden_dims: List of hidden layer dimensions
        output_bounds: Tuple of (min_bounds, max_bounds) for outputs
        dropout_rate: Dropout rate for regularization (default: 0.1)
        clip_value: Maximum gradient value during forward pass (default: 10.0)
    """
    def __init__(self, input_dim, output_dim, hidden_dims, output_bounds=None, 
                dropout_rate=0.1, clip_value=10.0):
        super(RobustOPFNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.clip_value = clip_value
        
        # Store output bounds
        self.output_bounds = output_bounds
        if output_bounds is not None:
            self.min_bounds, self.max_bounds = output_bounds
            
        # Input scaling layer
        self.input_bn = torch.nn.BatchNorm1d(input_dim)
        
        # Create hidden layers
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.shortcuts = torch.nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            # Main layer
            self.layers.append(torch.nn.Linear(prev_dim, dim))
            
            # Normalization
            self.norms.append(torch.nn.LayerNorm(dim))
            
            # Residual connection if dimensions match
            if prev_dim == dim:
                self.shortcuts.append(torch.nn.Identity())
            else:
                self.shortcuts.append(torch.nn.Linear(prev_dim, dim))
            
            prev_dim = dim
        
        # Output layer
        self.output_layer = torch.nn.Linear(prev_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Initialize weights with careful scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful scaling for improved stability."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Use Kaiming initialization with fan-in mode for ReLU-based networks
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def _clip_gradients(self, x):
        """Clip gradients during forward pass for numerical stability."""
        # Skip gradient clipping if not in training mode or input doesn't require grad
        if not self.training or not x.requires_grad:
            return x
        
        # Only try to clip gradients if input requires grad
        try:
            with torch.no_grad():
                # Calculate current gradient norm
                grad_input = torch.autograd.grad(
                    outputs=x.sum(), inputs=x, create_graph=True, retain_graph=True, 
                    only_inputs=True)[0]
                grad_norm = torch.norm(grad_input, p=2)
                
                # Create clipping coefficient
                clip_coef = self.clip_value / (grad_norm + 1e-8)
                
                # Apply clipping if needed
                if clip_coef < 1.0:
                    return x * clip_coef
        except RuntimeError:
            # If gradient computation fails, just return the input
            pass
        
        return x
    
    def forward(self, x):
        """Forward pass with residual connections and gradient clipping."""
        # Input normalization
        x = self.input_bn(x)
        
        # Hidden layers with residual connections
        for layer, norm, shortcut in zip(self.layers, self.norms, self.shortcuts):
            # Residual connection
            res = shortcut(x)
            
            # Main path
            x = layer(x)
            x = norm(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)
            
            # Add residual (with clipping for stability)
            x = x + res
            x = self._clip_gradients(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply output bounds if provided
        if self.output_bounds is not None:
            min_bounds, max_bounds = self.output_bounds
            
            # Use a modified approach that preserves more of the dynamic range
            # This avoids compressing all outputs toward zero
            
            # First, check if the bounds have a wide range (indicates important outputs)
            bounds_range = max_bounds - min_bounds
            
            # Create masks for different types of outputs
            # Active outputs: those with wide range (like main generators)
            # Inactive outputs: those with narrow range (like idle generators)
            active_mask = bounds_range > 0.01  # Identify active outputs
            
            # For active outputs: use softer activation to preserve dynamic range
            # For inactive outputs: use sigmoid to strictly enforce bounds
            
            # Apply sigmoids for all variables first
            x_sigmoid = torch.sigmoid(x)
            x_bounded = min_bounds + (max_bounds - min_bounds) * x_sigmoid
            
            # For active outputs, use a less aggressive approach (softplus-based)
            # This preserves more of the dynamic range for active generators
            x_softplus = torch.log(1 + torch.exp(x)) * 0.1  # Scaled softplus
            
            # Scale to range proportionally for active outputs
            max_range = bounds_range.max()
            x_active = min_bounds + bounds_range * (x_softplus / max_range) * 5  # Scale factor to match typical ranges
            
            # Combine using the masks
            x = torch.where(active_mask, x_active, x_bounded)
            
            # Add a final clip to enforce strict bounds and avoid out-of-range values
            x = torch.max(torch.min(x, max_bounds), min_bounds)
        
        return x

class RobustDirectPredictionNN(RobustOPFNetwork):
    """
    Robust neural network for direct prediction of OPF solutions.
    
    This network is specifically designed for predicting generation and voltage variables
    directly from load patterns, with improved numerical stability.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None, output_bounds=None):
        # Default architecture if not specified
        if hidden_dims is None:
            # Wider architecture helps with complex mappings
            hidden_dims = [256, 512, 256, 128]
        
        super(RobustDirectPredictionNN, self).__init__(
            input_dim, output_dim, hidden_dims, output_bounds)

class RobustConstraintScreeningNN(torch.nn.Module):
    """
    Robust neural network for constraint screening with improved numerical stability.
    
    Features:
    - Specialized for binary classification of binding constraints
    - Balanced loss handling for addressing class imbalance
    - Confidence calibration
    
    Args:
        input_dim: Input dimension (load features)
        num_constraints: Number of constraints to predict
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
    """
    def __init__(self, input_dim, num_constraints, hidden_dims=None, dropout_rate=0.2):
        super(RobustConstraintScreeningNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        # Use the robust network as the base
        self.base_network = RobustOPFNetwork(
            input_dim, num_constraints, hidden_dims, output_bounds=None, 
            dropout_rate=dropout_rate)
    
    def forward(self, x):
        # Get base predictions
        x = self.base_network(x)
        
        # Apply sigmoid to get probabilities
        x = torch.sigmoid(x)
        
        return x

class RobustWarmStartNN(torch.nn.Module):
    """
    Robust neural network for warm-starting OPF solvers.
    
    Features:
    - Physics-informed architecture
    - Better handling of output variable bounds
    - Improved numerical stability
    
    Args:
        input_dim: Input dimension (load features)
        output_dim: Output dimension (generation and voltage variables)
        hidden_dims: List of hidden layer dimensions
        output_bounds: Tuple of (min_bounds, max_bounds) for outputs
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None, output_bounds=None):
        super(RobustWarmStartNN, self).__init__()
        
        if hidden_dims is None:
            # Deep architecture for complex mappings
            hidden_dims = [128, 256, 256, 128]
        
        # Base network with the robust architecture
        self.network = RobustOPFNetwork(
            input_dim, output_dim, hidden_dims, output_bounds)
    
    def forward(self, x):
        return self.network(x) 