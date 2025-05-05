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
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Prepare node features
    x = []
    for node in G.nodes():
        node_feat = [G.nodes[node][feat] for feat in node_features]
        x.append(node_feat)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Prepare edge features if provided
    edge_attr = None
    if edge_features is not None:
        edge_attr = []
        for u, v in G.edges():
            edge_feat = [G[u][v][feat] for feat in edge_features]
            edge_attr.append(edge_feat)
            edge_attr.append(edge_feat)  # Add same features for reverse edge
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data 