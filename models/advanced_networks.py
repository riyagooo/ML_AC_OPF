import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import math


class ActivationSelector(nn.Module):
    """
    Dynamically selectable activation function
    """
    def __init__(self, activation_type='leaky_relu', alpha=0.1):
        super(ActivationSelector, self).__init__()
        self.activation_type = activation_type
        self.alpha = alpha
    
    def forward(self, x):
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=self.alpha)
        elif self.activation_type == 'elu':
            return F.elu(x, alpha=self.alpha)
        elif self.activation_type == 'selu':
            return F.selu(x)
        elif self.activation_type == 'gelu':
            return F.gelu(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'swish':
            return x * torch.sigmoid(x)
        else:
            return F.relu(x)  # Default to ReLU


class ResidualBlock(nn.Module):
    """
    Residual block with configurable activation and normalization
    """
    def __init__(self, dim, dropout_rate=0.2, activation='leaky_relu', use_layer_norm=True):
        super(ResidualBlock, self).__init__()
        
        self.use_layer_norm = use_layer_norm
        
        # First transformation
        self.linear1 = nn.Linear(dim, dim)
        self.activation1 = ActivationSelector(activation)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second transformation
        self.linear2 = nn.Linear(dim, dim)
        self.activation2 = ActivationSelector(activation)
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Scaling factor for residual (helps with training stability)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Store input for residual connection
        identity = x
        
        # First layer
        x = self.linear1(x)
        if self.use_layer_norm:
            x = self.activation1(x)  # Activation before normalization for LayerNorm
            x = self.norm1(x)
        else:
            x = self.norm1(x)  # Normalization before activation for BatchNorm
            x = self.activation1(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.linear2(x)
        if self.use_layer_norm:
            x = self.activation2(x)
            x = self.norm2(x)
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        x = self.dropout2(x)
        
        # Residual connection with learned scaling
        x = identity + self.scale * x
        
        return x


class MixtralActivation(nn.Module):
    """
    A learnable mixture of multiple activation functions to better model
    complex nonlinearities in power systems
    """
    def __init__(self, hidden_dim):
        super(MixtralActivation, self).__init__()
        self.activations = [
            F.relu,
            torch.tanh,
            F.leaky_relu,
            F.gelu,
            lambda x: x * torch.sigmoid(x)  # swish
        ]
        # Learnable weights for each activation function
        self.weights = nn.Parameter(torch.ones(len(self.activations)))
    
    def forward(self, x):
        # Weighted sum of all activation functions
        result = 0
        weights = F.softmax(self.weights, dim=0)
        for i, activation in enumerate(self.activations):
            result += weights[i] * activation(x)
        return result


class PowerSystemEmbedding(nn.Module):
    """
    Specialized embedding layer for power system features
    that applies domain-specific transformations to better capture
    the relationships in AC-OPF problems
    """
    def __init__(self, input_dim, output_dim):
        super(PowerSystemEmbedding, self).__init__()
        
        # Main linear transformation
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Power domain-specific transformations - simplified
        self.power_transform = nn.Linear(input_dim, output_dim)
        
        # Final combination layer - simplified
        self.combine = nn.Linear(output_dim * 2, output_dim)
        
        # Initialize weights with small values to prevent explosion
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Standard linear transformation
        linear_out = self.linear(x)
        
        # Quadratic terms for nonlinear power flow relationships
        power_squared = x**2  # Element-wise squaring
        power_out = self.power_transform(power_squared)
        
        # Combine transformations - simplified approach
        combined = torch.cat([linear_out, power_out], dim=-1)
        
        # Final output
        x = self.combine(combined)
        
        return x


class AdvancedFeedforwardModel(nn.Module):
    """
    Advanced feedforward neural network featuring:
    - Deeper and wider architecture
    - Residual connections
    - Multiple activation functions
    - Power system specific feature transformations
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, 
                 num_layers=6, dropout_rate=0.2, 
                 activation='leaky_relu', use_residuals=True):
        super(AdvancedFeedforwardModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residuals = use_residuals
        
        # Input normalization for better numerical stability
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Input embedding with power system specific transformations
        self.input_embedding = PowerSystemEmbedding(input_dim, hidden_dim)
        
        # Initial activation and normalization
        self.activation = ActivationSelector(activation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create main layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_residuals and i % 2 == 1:  # Add residual blocks every other layer
                self.layers.append(ResidualBlock(
                    hidden_dim, dropout_rate, activation
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    ActivationSelector(activation),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_rate)
                ))
        
        # Additional layers for output refinement
        self.refine1 = nn.Linear(hidden_dim, hidden_dim)
        self.refine2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output projection
        self.output = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming/Xavier initialization based on activation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(getattr(self, 'activation', None), ActivationSelector) and self.activation.activation_type == 'relu':
                    # Kaiming initialization for ReLU-like activations
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Xavier initialization for other activations
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Apply input normalization
        x = self.input_norm(x)
        
        # Initial embedding
        x = self.input_embedding(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        # Process through main layers
        for layer in self.layers:
            x = layer(x)
        
        # Output refinement
        x = F.relu(self.refine1(x))
        x = F.relu(self.refine2(x))
        
        # Final output
        x = self.output(x)
        
        return x


class EnhancedPhysicsMessagePassing(MessagePassing):
    """
    Enhanced physics-informed message passing layer for power systems
    with improved feature interactions and multiple activation functions
    """
    def __init__(self, hidden_dim, activation='leaky_relu', edge_dim=4, aggr="add"):
        super(EnhancedPhysicsMessagePassing, self).__init__(aggr=aggr)
        
        # Message transformation with multiple paths
        self.message_path1 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            ActivationSelector(activation),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.message_path2 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            ActivationSelector('gelu'),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Path attention mechanism
        self.path_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Update function with residual structure
        self.update_layer1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ActivationSelector(activation),
            nn.LayerNorm(hidden_dim)
        )
        
        self.update_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ActivationSelector(activation)
        )
        
        # Gating mechanism for balancing previous state with updates
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Scale factor for adjusting self vs. neighbor importance
        self.scale_factor = nn.Parameter(torch.ones(1))
    
    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Process edge attributes - handle self-loops
        if edge_attr is None:
            # Create dummy edge features
            edge_attr = torch.zeros(edge_index.size(1), 4, device=x.device)
            # Set identity features for self-loops (1 for self-connection, 0 for others)
            edge_attr[edge_index[0] == edge_index[1], 0] = 1.0
        else:
            # Add dummy features for self-loops
            num_self_loops = x.size(0)
            dummy_attr = torch.zeros(num_self_loops, edge_attr.size(1), device=x.device)
            dummy_attr[:, 0] = 1.0  # Set identity feature
            edge_attr = torch.cat([edge_attr, dummy_attr], dim=0)
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Combine features from source node, target node, and edge
        message_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Process through two different paths 
        message1 = self.message_path1(message_input)
        message2 = self.message_path2(message_input)
        
        # Calculate attention weights for each path
        attention_input = torch.cat([message1, message2], dim=1)
        attention_weights = self.path_attention(attention_input)
        
        # Weight and combine the paths
        message = attention_weights[:, 0:1] * message1 + attention_weights[:, 1:2] * message2
        
        # Scale messages by distance between nodes
        # Self-loops (i==j) get a weight of 1.0, others get scale_factor
        is_self = (x_i == x_j).all(dim=1, keepdim=True)
        scale = torch.where(is_self, torch.ones_like(self.scale_factor), self.scale_factor)
        
        return message * scale
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node's previous state
        update_input = torch.cat([aggr_out, x], dim=1)
        
        # Calculate gate values to control information flow
        gate_value = self.gate(update_input)
        
        # Process through update layers with residual connection
        update1 = self.update_layer1(update_input)
        update2 = x + self.update_layer2(update1)  # Residual
        
        # Apply gating mechanism for final update
        return x * (1 - gate_value) + update2 * gate_value


class AdvancedGNN(nn.Module):
    """
    Advanced Graph Neural Network with:
    - Enhanced physics-informed message passing
    - Multiple paths of information flow
    - Residual connections and varied activation functions
    - Power system domain-specific feature engineering
    """
    def __init__(self, node_features, hidden_dim, output_dim, 
                 num_layers=4, dropout_rate=0.2, activation='leaky_relu',
                 edge_features=4, use_layer_attention=True):
        super(AdvancedGNN, self).__init__()
        
        # Input embedding for nodes with power-specific transformations
        self.node_embedding = PowerSystemEmbedding(node_features, hidden_dim)
        
        # Message passing layers with physics-informed functions
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate between different activation functions
            layer_activation = 'gelu' if i % 2 == 1 else activation
            self.mp_layers.append(EnhancedPhysicsMessagePassing(
                hidden_dim, activation=layer_activation, edge_dim=edge_features
            ))
        
        # Batch normalization after each layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Residual connections for every other layer
        self.has_residual = [i % 2 == 1 for i in range(num_layers)]
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer attention mechanism (optional)
        self.use_layer_attention = use_layer_attention
        if use_layer_attention:
            self.layer_attention = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Final output layers
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ActivationSelector(activation),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate/2),
            nn.Linear(hidden_dim, hidden_dim//2),
            ActivationSelector(activation)
        )
        
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming or Xavier based on activation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use orthogonal initialization for better gradient flow
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """Forward pass for advanced GNN"""
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Store per-layer outputs for layer attention
        layer_outputs = []
        
        # Store initial representation for residual connections
        x0 = x
        
        # Apply message passing layers
        for i, (mp, norm, has_res) in enumerate(zip(self.mp_layers, self.norms, self.has_residual)):
            # Message passing
            x_new = mp(x, edge_index, edge_attr)
            
            # Normalization
            x_new = norm(x_new)
            
            # Activation is handled inside the message passing layer
            
            # Dropout
            x_new = self.dropout(x_new)
            
            # Residual connection if applicable
            if has_res:
                x = x + x_new  # Simple residual
            else:
                x = x_new
            
            # Store layer output
            if self.use_layer_attention:
                layer_outputs.append(x)
        
        # Apply layer attention if enabled
        if self.use_layer_attention and len(layer_outputs) > 0:
            # Normalize attention weights
            attn = F.softmax(self.layer_attention, dim=0)
            
            # Weighted sum of layer outputs
            x = sum(w * out for w, out in zip(attn, layer_outputs))
        
        # Global pooling - aggregate node features to graph level
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, take mean of all nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction
        x = self.pre_output(x)
        x = self.output_layer(x)
        
        return x


class AdvancedConstraintScreeningGNN(AdvancedGNN):
    """
    Advanced GNN specialized for constraint screening with:
    - Output sigmoid activation for binary classification
    - Binary prediction specific enhancements
    """
    def __init__(self, node_features, hidden_dim, output_dim, 
                 num_layers=4, dropout_rate=0.3, activation='leaky_relu',
                 edge_features=4, use_layer_attention=True):
        super(AdvancedConstraintScreeningGNN, self).__init__(
            node_features, hidden_dim, output_dim, 
            num_layers, dropout_rate, activation,
            edge_features, use_layer_attention
        )
        
        # Replace output layer with one designed for binary classification
        self.binary_output = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            ActivationSelector(activation),
            nn.Dropout(dropout_rate/2),
            nn.Linear(hidden_dim//2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        # Get features up to the pre-output stage
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Store per-layer outputs for layer attention
        layer_outputs = []
        
        # Apply message passing layers
        for i, (mp, norm, has_res) in enumerate(zip(self.mp_layers, self.norms, self.has_residual)):
            # Message passing
            x_new = mp(x, edge_index, edge_attr)
            
            # Normalization
            x_new = norm(x_new)
            
            # Dropout
            x_new = self.dropout(x_new)
            
            # Residual connection if applicable
            if has_res:
                x = x + x_new
            else:
                x = x_new
            
            # Store layer output
            if self.use_layer_attention:
                layer_outputs.append(x)
        
        # Apply layer attention if enabled
        if self.use_layer_attention and len(layer_outputs) > 0:
            attn = F.softmax(self.layer_attention, dim=0)
            x = sum(w * out for w, out in zip(attn, layer_outputs))
        
        # Global pooling - aggregate node features to graph level
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Apply pre-output and binary classification output
        x = self.pre_output(x)
        x = self.binary_output(x)
        
        return x 