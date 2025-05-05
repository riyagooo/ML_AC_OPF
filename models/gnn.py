import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
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