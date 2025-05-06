"""
Power system graph utilities for ML-AC-OPF.

This module provides functions for creating graph representations of power systems.
"""

import torch
import numpy as np
from torch_geometric.data import Data

def create_power_system_graph(data_batch):
    """
    Create a simple graph representation of power system data.
    
    This function creates a list of PyTorch Geometric Data objects from a batch of input data.
    Each sample in the batch will be converted to a separate graph.
    
    Args:
        data_batch: NumPy array or torch.Tensor with batch of input data
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    # Ensure data is a tensor
    if isinstance(data_batch, np.ndarray):
        data_batch = torch.tensor(data_batch, dtype=torch.float32)
    
    # Create a list of graphs, one for each sample in the batch
    graphs = []
    
    for i in range(data_batch.shape[0]):
        # Extract features for this sample
        features = data_batch[i]
        
        # Reshape features into nodes
        # For simplicity, we'll create a fully connected graph
        # with each pair of input features becoming a node
        num_nodes = features.shape[0] // 2
        
        # Create node features (pairs of values from the input)
        x = features.view(-1, 2)
        
        # Create edges (fully connected graph)
        edge_index = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst:  # Skip self-loops
                    edge_index.append([src, dst])
        
        # Convert edge_index to tensor
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            # If no edges, create a self-loop on node 0
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Create the graph
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
    
    return graphs 