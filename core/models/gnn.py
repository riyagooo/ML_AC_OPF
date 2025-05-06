import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool
import numpy as np

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x
    
class TopologyAwareGNN(nn.Module):
    """
    Graph Neural Network for topology-aware OPF predictions.
    """
    def __init__(self, node_features, edge_features, hidden_channels, output_dim, 
                 num_layers=3, dropout_rate=0.1, output_bounds=None):
        """
        Initialize GNN model for topology-aware predictions.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_channels: Number of hidden channels
            output_dim: Dimension of output predictions
            num_layers: Number of graph convolutional layers
            dropout_rate: Dropout rate for regularization
            output_bounds: Tuple of (min_vals, max_vals) tensors for output bounds
        """
        super(TopologyAwareGNN, self).__init__()
        
        self.node_features = node_features
        self.output_dim = output_dim
        self.output_bounds = output_bounds
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNLayer(hidden_channels, hidden_channels))
                
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)
        
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_attr: Edge features
                - batch: Batch indices for multiple graphs
        
        Returns:
            Predictions for the requested output values
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial node embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)
        
        # Handle batched data - use global_mean_pool if batch is provided
        if hasattr(data, 'batch') and data.batch is not None:
            # Import global_mean_pool for batched graphs
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, just take mean of all nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply bounds if provided
        if self.output_bounds is not None:
            min_vals, max_vals = self.output_bounds
            
            # Check if dimensions match
            if min_vals.size(0) != self.output_dim or max_vals.size(0) != self.output_dim:
                print(f"Warning: Output bounds dimensions ({min_vals.size(0)},{max_vals.size(0)}) " + 
                      f"don't match output dimensions ({self.output_dim}). Adjusting.")
                
                if x.size(-1) < min_vals.size(0):
                    # Resize bounds to match output dimensions
                    min_vals = min_vals[:x.size(-1)]
                    max_vals = max_vals[:x.size(-1)]
                else:
                    # Pad bounds if output is larger
                    pad_size = x.size(-1) - min_vals.size(0)
                    if pad_size > 0:
                        device = min_vals.device
                        min_vals = torch.cat([min_vals, torch.zeros(pad_size, device=device)])
                        max_vals = torch.cat([max_vals, torch.ones(pad_size, device=device)])
            
            # Apply sigmoid and scale to bounds
            x = torch.sigmoid(x)
            x = min_vals + (max_vals - min_vals) * x
            
        return x

class HybridGNN(nn.Module):
    """
    Hybrid GNN model combining node-level and graph-level features.
    """
    def __init__(self, node_features, edge_features, global_features, 
                 hidden_channels, output_dim, num_layers=3, dropout_rate=0.1,
                 output_bounds=None):
        """
        Initialize hybrid GNN for OPF predictions.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            global_features: Number of global features (load patterns, etc.)
            hidden_channels: Number of hidden channels
            output_dim: Dimension of output predictions
            num_layers: Number of graph convolutional layers
            dropout_rate: Dropout rate for regularization
            output_bounds: Tuple of (min_vals, max_vals) tensors for output bounds
        """
        super(HybridGNN, self).__init__()
        
        self.node_features = node_features
        self.global_features = global_features
        self.output_dim = output_dim
        self.output_bounds = output_bounds
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_channels)
        
        # Global feature embedding
        self.global_embedding = nn.Linear(global_features, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNLayer(hidden_channels, hidden_channels))
                
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers combining graph and global features
        self.fc1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)
        
    def forward(self, data, global_features):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object
            global_features: Tensor of global features
            
        Returns:
            Predictions for the requested output values
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial node embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)
        
        # Global pooling (mean of all node embeddings)
        x_graph = torch.mean(x, dim=0)
        
        # Process global features
        x_global = F.relu(self.global_embedding(global_features))
        
        # Concatenate graph and global features
        x_combined = torch.cat([x_graph, x_global], dim=0)
        
        # Final prediction layers
        x = F.relu(self.fc1(x_combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply bounds if provided
        if self.output_bounds is not None:
            min_vals, max_vals = self.output_bounds
            
            # Check if dimensions match
            if min_vals.size(0) != self.output_dim or max_vals.size(0) != self.output_dim:
                print(f"Warning: Output bounds dimensions ({min_vals.size(0)},{max_vals.size(0)}) " + 
                      f"don't match output dimensions ({self.output_dim}). Adjusting.")
                
                if x.size(-1) < min_vals.size(0):
                    # Resize bounds to match output dimensions
                    min_vals = min_vals[:x.size(-1)]
                    max_vals = max_vals[:x.size(-1)]
                else:
                    # Pad bounds if output is larger
                    pad_size = x.size(-1) - min_vals.size(0)
                    if pad_size > 0:
                        device = min_vals.device
                        min_vals = torch.cat([min_vals, torch.zeros(pad_size, device=device)])
                        max_vals = torch.cat([max_vals, torch.ones(pad_size, device=device)])
            
            # Apply sigmoid and scale to bounds
            x = torch.sigmoid(x)
            x = min_vals + (max_vals - min_vals) * x
            
        return x

def prepare_pyg_data(G, node_features, edge_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric data.
    
    Args:
        G: NetworkX graph
        node_features: Dictionary of node features
        edge_features: Dictionary of edge features (optional)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Get node indices
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    
    # Prepare edge index
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_indices[u], node_indices[v]])
        edge_index.append([node_indices[v], node_indices[u]])  # Add reverse edge for undirected graph
    
    # Handle case with no edges
    if not edge_index:
        print("Warning: Graph has no edges!")
        # Create a self-loop on first node to prevent errors
        if G.nodes():
            first_node = list(G.nodes())[0]
            edge_index = [[0, 0], [0, 0]]  # Self-loop
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Prepare node features
    x = []
    for node in G.nodes():
        node_feat = []
        for feat in node_features:
            # Handle missing features gracefully
            if feat in G.nodes[node]:
                node_feat.append(G.nodes[node][feat])
            else:
                print(f"Warning: Feature '{feat}' not found for node {node}, using 0.0")
                node_feat.append(0.0)
        x.append(node_feat)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Prepare edge features if provided
    edge_attr = None
    if edge_features is not None and G.edges():
        edge_attr = []
        for u, v in G.edges():
            edge_feat = []
            for feat in edge_features:
                # Handle missing features gracefully
                if feat in G[u][v]:
                    edge_feat.append(G[u][v][feat])
                else:
                    print(f"Warning: Feature '{feat}' not found for edge {u}-{v}, using 0.0")
                    edge_feat.append(0.0)
            edge_attr.append(edge_feat)
            edge_attr.append(edge_feat)  # Add same features for reverse edge
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data 

class RobustTopologyAwareGNN(nn.Module):
    """
    Robust Graph Neural Network for topology-aware OPF predictions with improved numerical stability.
    
    Features:
    - Layer normalization for improved training dynamics
    - Adaptive activation functions
    - Better handling of output variable bounds
    - Improved numerical stability
    """
    def __init__(self, node_features, edge_features, hidden_channels, output_dim, 
                 num_layers=3, dropout_rate=0.1, output_bounds=None, clip_value=10.0):
        """
        Initialize robust GNN model for topology-aware predictions.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_channels: Number of hidden channels
            output_dim: Dimension of output predictions
            num_layers: Number of graph convolutional layers
            dropout_rate: Dropout rate for regularization
            output_bounds: Tuple of (min_vals, max_vals) tensors for output bounds
            clip_value: Maximum gradient value during forward pass
        """
        super(RobustTopologyAwareGNN, self).__init__()
        
        self.node_features = node_features
        self.output_dim = output_dim
        self.output_bounds = output_bounds
        self.clip_value = clip_value
        
        # Input normalization
        self.node_norm = nn.BatchNorm1d(node_features)
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_channels)
        
        # GNN layers with normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
            # Add layer normalization after each conv layer
            self.norms.append(nn.LayerNorm(hidden_channels))
                
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_norm = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)
        
        # Initialize weights with careful scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful scaling for improved stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization with fan-in mode for ReLU-based networks
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        
    def forward(self, data):
        """
        Forward pass through the network with improved numerical stability.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_attr: Edge features
                - batch: Batch indices for multiple graphs
        
        Returns:
            Predictions for the requested output values
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply input normalization if batch size > 1
        if x.size(0) > 1:
            x = self.node_norm(x)
        
        # Initial node embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers with residual connections and normalization
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Store original representation for residual connection
            identity = x
            
            # Apply GCN layer
            x = conv(x, edge_index)
            
            # Apply layer normalization
            x = norm(x)
            
            # Use leaky ReLU for better gradient flow
            x = F.leaky_relu(x, negative_slope=0.1)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Add residual connection if dimensions match
            if x.size(-1) == identity.size(-1):
                x = x + identity
                
            # Apply gradient clipping for stability
            x = self._clip_gradients(x)
        
        # Handle batched data - use global_mean_pool if batch is provided
        if hasattr(data, 'batch') and data.batch is not None:
            # Import global_mean_pool for batched graphs
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, just take mean of all nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction layers
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        
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

class RobustHybridGNN(nn.Module):
    """
    Robust Hybrid GNN model combining node-level and graph-level features
    with improved numerical stability.
    """
    def __init__(self, node_features, edge_features, global_features, 
                 hidden_channels, output_dim, num_layers=3, dropout_rate=0.1,
                 output_bounds=None, clip_value=10.0):
        """
        Initialize robust hybrid GNN for OPF predictions.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            global_features: Number of global features (load patterns, etc.)
            hidden_channels: Number of hidden channels
            output_dim: Dimension of output predictions
            num_layers: Number of graph convolutional layers
            dropout_rate: Dropout rate for regularization
            output_bounds: Tuple of (min_vals, max_vals) tensors for output bounds
            clip_value: Maximum gradient value during forward pass
        """
        super(RobustHybridGNN, self).__init__()
        
        self.node_features = node_features
        self.global_features = global_features
        self.output_dim = output_dim
        self.output_bounds = output_bounds
        self.clip_value = clip_value
        
        # Input normalization
        self.node_norm = nn.BatchNorm1d(node_features)
        self.global_norm = nn.BatchNorm1d(global_features)
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_channels)
        
        # Global feature embedding
        self.global_embedding = nn.Linear(global_features, hidden_channels)
        
        # GNN layers with normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
            # Add layer normalization after each conv layer
            self.norms.append(nn.LayerNorm(hidden_channels))
                
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers combining graph and global features
        self.fc1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc_norm = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)
        
        # Initialize weights with careful scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful scaling for improved stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization with fan-in mode for ReLU-based networks
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        
    def forward(self, data, global_features):
        """
        Forward pass through the network with improved numerical stability.
        
        Args:
            data: PyTorch Geometric Data object
            global_features: Tensor of global features
            
        Returns:
            Predictions for the requested output values
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply input normalization if batch size > 1
        if x.size(0) > 1:
            x = self.node_norm(x)
        
        # Normalize global features if batch dimension > 1
        if global_features.dim() > 1 and global_features.size(0) > 1:
            global_features = self.global_norm(global_features)
        
        # Initial node embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers with residual connections and normalization
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Store original representation for residual connection
            identity = x
            
            # Apply GCN layer
            x = conv(x, edge_index)
            
            # Apply layer normalization
            x = norm(x)
            
            # Use leaky ReLU for better gradient flow
            x = F.leaky_relu(x, negative_slope=0.1)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Add residual connection if dimensions match
            if x.size(-1) == identity.size(-1):
                x = x + identity
                
            # Apply gradient clipping for stability
            x = self._clip_gradients(x)
        
        # Global pooling (mean of all node embeddings)
        if hasattr(data, 'batch') and data.batch is not None:
            # Import global_mean_pool for batched graphs
            from torch_geometric.nn import global_mean_pool
            x_graph = global_mean_pool(x, data.batch)
        else:
            # For single graph, just take mean of all nodes
            x_graph = torch.mean(x, dim=0, keepdim=True)
        
        # Process global features
        x_global = F.leaky_relu(self.global_embedding(global_features), negative_slope=0.1)
        x_global = self._clip_gradients(x_global)
        
        # Concatenate graph and global features
        # Ensure dimensions match for concatenation
        if x_graph.dim() == 2 and x_global.dim() == 1:
            x_global = x_global.unsqueeze(0)
        
        x_combined = torch.cat([x_graph, x_global], dim=1)
        
        # Final prediction layers
        x = self.fc1(x_combined)
        x = self.fc_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        
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

# Add the DirectPredictionGNN class
class DirectPredictionGNN(nn.Module):
    """
    Graph Neural Network for direct prediction of voltage profiles.
    This model is specifically designed for the direct prediction task,
    where we predict voltage magnitudes and angles based on power injections.
    """
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=4, dropout_rate=0.2):
        super(DirectPredictionGNN, self).__init__()
        
        # Node embedding layer
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers//2)
        ])
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - batch: Batch indices for multiple graphs
        
        Returns:
            Predictions for voltage magnitudes and angles
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial node embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i % 2 == 0 and i > 0:  # Apply residual every other layer
                residual_idx = i // 2 - 1
                residual = self.residual_layers[residual_idx](x)
            else:
                residual = x
            
            # GCN layer
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Add residual connection
            if x.size() == residual.size():
                x = x + residual
        
        # Global pooling to get a graph-level representation
        if hasattr(data, 'batch') and data.batch is not None:
            # For batched graphs
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, data.batch)
        else:
            # For a single graph
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction layers
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 

# Add the ConstraintScreeningGNN class
class ConstraintScreeningGNN(nn.Module):
    """
    Graph Neural Network for constraint screening in power system optimization.
    This model is specifically designed to predict which constraints will be binding
    in the optimal power flow solution, using the network topology information.
    """
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(ConstraintScreeningGNN, self).__init__()
        
        # Node embedding layer
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # GNN layers with attention for better constraint sensitivity
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Use GAT for better attention to critical nodes/edges
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Dropout for regularization (higher dropout for this task)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - batch: Batch indices for multiple graphs
        
        Returns:
            Binary predictions for each constraint (whether it will be binding)
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial node embedding
        x = self.node_embedding(x)
        x = self.embedding_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        
        # Apply GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GAT layer
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)
        
        # Global pooling to get a graph-level representation
        if hasattr(data, 'batch') and data.batch is not None:
            # For batched graphs
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, data.batch)
        else:
            # For a single graph
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction layers
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply sigmoid for binary classification
        x = torch.sigmoid(x)
        
        return x 

class PhysicsInformedMessagePassing(MessagePassing):
    """
    Physics-informed message passing layer for power systems
    Incorporates physical properties like electrical admittance
    """
    def __init__(self, hidden_dim, aggr="add"):
        super(PhysicsInformedMessagePassing, self).__init__(aggr=aggr)
        
        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),  # node i + node j + edge attrs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # If no edge attributes are provided, create dummy ones
        if edge_attr is None:
            # For self-loops, create dummy edge features
            edge_attr = torch.zeros(edge_index.size(1), 4, device=x.device)
            
            # Set identity features for self-loops (1 for self-connection, 0 for others)
            edge_attr[edge_index[0] == edge_index[1], 0] = 1.0
        else:
            # For added self-loops, add dummy edge features
            dummy_attr = torch.zeros(x.size(0), 4, device=x.device)
            dummy_attr[:, 0] = 1.0  # Set identity feature
            edge_attr = torch.cat([edge_attr, dummy_attr], dim=0)
            
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Combine features from source node, target node, and edge
        message_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Transform using MLP
        return self.message_mlp(message_input)
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node's previous state
        update_input = torch.cat([aggr_out, x], dim=1)
        
        # Calculate gate values
        gate_value = self.gate(update_input)
        
        # Apply gated update
        updated = self.update_mlp(update_input)
        
        # Return gated combination of previous state and update
        return x * (1 - gate_value) + updated * gate_value

class NodeAttention(nn.Module):
    """
    Attention mechanism to weight node features based on their importance
    """
    def __init__(self, hidden_dim):
        super(NodeAttention, self).__init__()
        
        # Attention query vector
        self.query = nn.Parameter(torch.randn(hidden_dim))
        
        # Attention projection
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, x, batch=None):
        # Calculate attention scores
        scores = self.attention(x)
        
        # Apply softmax to get weights
        if batch is not None:
            # If in batch mode, apply softmax per graph
            weights = []
            for i in torch.unique(batch):
                batch_mask = (batch == i)
                batch_scores = scores[batch_mask]
                batch_weights = F.softmax(batch_scores, dim=0)
                weights.append(batch_weights)
            weights = torch.cat(weights, dim=0)
        else:
            # Single graph, apply softmax across all nodes
            weights = F.softmax(scores, dim=0)
        
        # Apply attention weights and sum
        return (x * weights).sum(dim=0, keepdim=True)

class EnhancedDirectPredictionGNN(nn.Module):
    """
    Enhanced GNN for direct prediction of power system variables
    Incorporates physics-informed layers and edge features
    """
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=4, dropout_rate=0.2):
        super(EnhancedDirectPredictionGNN, self).__init__()
        
        # Input embedding for nodes
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Physics-informed message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(PhysicsInformedMessagePassing(hidden_dim))
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Node attention for focusing on important buses
        self.attention = NodeAttention(hidden_dim)
        
        # Output prediction with residual connection
        self.pre_output = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass for enhanced GNN
        Args:
            data: PyTorch Geometric Data object with node features and graph structure
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Store initial representation for residual connection
        x0 = x
        
        # Apply message passing layers
        for i, (mp, bn) in enumerate(zip(self.mp_layers, self.bns)):
            # Message passing
            x = mp(x, edge_index, edge_attr)
            
            # Normalization, activation, and dropout
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Add residual connection every 2 layers
            if i % 2 == 1:
                x = x + x0
                x0 = x
        
        # Global pooling - aggregate node features to graph level
        # If batch information is available, use it
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, apply attention-based pooling
            x = self.attention(x)
        
        # Final prediction
        x = F.relu(self.pre_output(x))
        x = self.output_layer(x)
        
        return x

class EnhancedConstraintScreeningGNN(nn.Module):
    """
    Enhanced GNN for constraint screening in power systems
    Uses physics-informed message passing and edge features
    """
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(EnhancedConstraintScreeningGNN, self).__init__()
        
        # Input embedding for nodes
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Physics-informed message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(PhysicsInformedMessagePassing(hidden_dim))
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Node attention for focusing on important buses
        self.attention = NodeAttention(hidden_dim)
        
        # Output prediction
        self.pre_output = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass for enhanced constraint screening GNN
        Args:
            data: PyTorch Geometric Data object with node features and graph structure
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Apply message passing layers
        for i, (mp, bn) in enumerate(zip(self.mp_layers, self.bns)):
            # Message passing
            x = mp(x, edge_index, edge_attr)
            
            # Normalization, activation, and dropout
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling - aggregate node features to graph level
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, apply attention-based pooling
            x = self.attention(x)
        
        # Final prediction with sigmoid for binary classification
        x = F.relu(self.pre_output(x))
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x 

def train_model_gnn(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping=10):
    """
    Train a GNN model with early stopping
    
    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epochs: Maximum number of epochs
        early_stopping: Patience for early stopping
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            y_batch = batch.y
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            num_batches += 1
        
        # Average loss for the epoch
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                y_batch = batch.y
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                num_batches += 1
        
        # Average validation loss
        val_loss /= num_batches
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses


def evaluate_model_gnn(model, test_loader, criterion, device):
    """
    Evaluate a GNN model on test data
    
    Args:
        model: The trained GNN model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
    
    Returns:
        test_loss: Loss on test set
        mse: Mean squared error
        mae: Mean absolute error
        r2: R-squared value
        all_preds: Predictions
        all_targets: Ground truth targets
    """
    import numpy as np
    from sklearn.metrics import r2_score
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            y_batch = batch.y
            
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
            num_batches += 1
    
    # Average test loss
    test_loss /= num_batches
    
    # Concatenate all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Calculate R2 score for each dimension
    r2_values = []
    for i in range(all_targets.shape[1]):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        r2_values.append(r2)
    
    # Average R2 score
    avg_r2 = np.mean(r2_values)
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Avg R^2: {avg_r2:.6f}")
    
    # Log individual R2 scores
    for i, r2 in enumerate(r2_values):
        print(f"R^2 for output {i+1}: {r2:.6f}")
    
    return test_loss, mse, mae, avg_r2, all_preds, all_targets 