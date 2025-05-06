import numpy as np
import torch
from torch_geometric.data import Data
import os
import pandas as pd

def create_power_network_graph(features, case_name="ieee39"):
    """
    Creates a graph representation of the power network for use with GNN models.
    
    Args:
        features (numpy.ndarray): Node features (one row per sample, columns are features)
        case_name (str): Name of the power system case (default: 'ieee39')
        
    Returns:
        torch_geometric.data.Data: Graph representation of the power network
    """
    if case_name.lower() == "ieee39":
        return _create_ieee39_graph(features)
    elif case_name.lower() == "case30":
        return _create_case30_graph(features)
    elif case_name.lower() == "case5":
        return _create_case5_graph(features)
    else:
        raise ValueError(f"Unsupported case name: {case_name}")

def _create_ieee39_graph(features):
    """
    Creates a graph representation of the IEEE 39-bus system.
    
    The IEEE 39-bus system has the following structure:
    - 39 buses (nodes)
    - 10 generators
    - 46 branches (edges)
    
    Args:
        features (numpy.ndarray): Node features for the IEEE 39-bus system
        
    Returns:
        torch_geometric.data.Data: Graph representation of the IEEE 39-bus system
    """
    # Number of buses in the IEEE 39-bus system
    n_buses = 39
    
    # Define the edge connections based on IEEE 39-bus topology
    # These are the branches in the IEEE 39-bus system
    # Format: (from_bus, to_bus)
    edges = [
        (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18),
        (4, 5), (4, 14), (5, 6), (5, 8), (6, 7), (6, 11),
        (7, 8), (8, 9), (9, 39), (10, 11), (10, 13), (13, 14),
        (14, 15), (15, 16), (16, 17), (16, 19), (16, 21), (16, 24),
        (17, 18), (17, 27), (19, 20), (19, 33), (20, 34), (21, 22),
        (22, 23), (22, 35), (23, 24), (23, 36), (25, 26), (25, 37),
        (26, 27), (26, 28), (26, 29), (28, 29), (29, 38), (30, 31),
        (31, 32), (32, 33), (33, 34), (34, 35), (36, 37), (36, 38),
        (38, 39)
    ]
    
    # Convert to 0-indexed for PyTorch Geometric
    edges = [(src-1, dst-1) for src, dst in edges]
    
    # Make edges bidirectional (undirected graph)
    edges += [(dst, src) for src, dst in edges]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # If features is a 2D array (multiple samples), reshape for the graph
    if len(features.shape) > 1 and features.shape[0] > 1:
        # If we have multiple samples, we need to reshape to get node features
        # Assume features are arranged by node
        n_samples = features.shape[0]
        feat_per_node = features.shape[1] // n_buses
        
        # Reshape to get separate node features: [n_samples, n_nodes, node_features]
        node_features = features.reshape(n_samples, n_buses, feat_per_node)
        
        # Create a batch of graph objects
        data_list = []
        for i in range(n_samples):
            data = Data(
                x=torch.tensor(node_features[i], dtype=torch.float),
                edge_index=edge_index
            )
            data_list.append(data)
        
        return data_list
    else:
        # If we have a single sample, create a single graph
        # Reshape to get node features: [n_nodes, node_features]
        if len(features.shape) == 1:
            # If features is a 1D array, reshape it
            features = features.reshape(1, -1)
        
        feat_per_node = features.shape[1] // n_buses
        node_features = features.reshape(n_buses, feat_per_node)
        
        # Create graph
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index
        )
        
        return data

def _create_case30_graph(features):
    """Creates a graph representation of the IEEE 30-bus system."""
    # Implementation similar to IEEE 39-bus, but with the IEEE 30-bus topology
    # This is a placeholder - would need actual IEEE 30-bus topology
    n_buses = 30
    
    # Define the edge connections (simplified example - not actual IEEE 30-bus)
    edges = [
        (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (2, 6), (4, 6),
        (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (9, 11), (9, 10),
        (4, 12), (12, 13), (12, 14), (12, 15), (12, 16), (14, 15),
        (16, 17), (15, 18), (18, 19), (19, 20), (10, 20), (10, 17),
        (10, 21), (10, 22), (21, 22), (15, 23), (22, 24), (23, 24),
        (24, 25), (25, 26), (25, 27), (28, 27), (27, 29), (29, 30),
        (8, 28), (6, 28)
    ]
    
    # Convert to 0-indexed for PyTorch Geometric
    edges = [(src-1, dst-1) for src, dst in edges]
    
    # Make edges bidirectional (undirected graph)
    edges += [(dst, src) for src, dst in edges]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Similar reshaping logic as in _create_ieee39_graph
    if len(features.shape) > 1 and features.shape[0] > 1:
        n_samples = features.shape[0]
        feat_per_node = features.shape[1] // n_buses
        node_features = features.reshape(n_samples, n_buses, feat_per_node)
        
        data_list = []
        for i in range(n_samples):
            data = Data(
                x=torch.tensor(node_features[i], dtype=torch.float),
                edge_index=edge_index
            )
            data_list.append(data)
        
        return data_list
    else:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        feat_per_node = features.shape[1] // n_buses
        node_features = features.reshape(n_buses, feat_per_node)
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index
        )
        
        return data

def _create_case5_graph(features):
    """Creates a graph representation of the 5-bus system."""
    # Simplified 5-bus system
    n_buses = 5
    
    # Define edges for a simple 5-bus system
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
    
    # Convert to 0-indexed for PyTorch Geometric
    edges = [(src-1, dst-1) for src, dst in edges]
    
    # Make edges bidirectional (undirected graph)
    edges += [(dst, src) for src, dst in edges]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Similar reshaping logic as above
    if len(features.shape) > 1 and features.shape[0] > 1:
        n_samples = features.shape[0]
        feat_per_node = features.shape[1] // n_buses
        node_features = features.reshape(n_samples, n_buses, feat_per_node)
        
        data_list = []
        for i in range(n_samples):
            data = Data(
                x=torch.tensor(node_features[i], dtype=torch.float),
                edge_index=edge_index
            )
            data_list.append(data)
        
        return data_list
    else:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        feat_per_node = features.shape[1] // n_buses
        node_features = features.reshape(n_buses, feat_per_node)
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index
        )
        
        return data

def batch_graph_data(graph_list):
    """
    Convert a list of graph data objects into a batched graph data object.
    
    Args:
        graph_list (list): List of torch_geometric.data.Data objects
        
    Returns:
        torch_geometric.data.Batch: Batched graph data
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(graph_list)

def create_graph_data_loader(features, targets, batch_size=32, shuffle=True, case_name="ieee39"):
    """
    Creates a DataLoader for graph data.
    
    Args:
        features (numpy.ndarray): Input features
        targets (numpy.ndarray): Target values
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        case_name (str): Name of the power system case
        
    Returns:
        torch_geometric.data.DataLoader: DataLoader for graph data
    """
    from torch_geometric.loader import DataLoader as GraphDataLoader
    
    # Create graph objects
    graph_list = create_power_network_graph(features, case_name)
    
    # Add targets to graph objects
    for i, graph in enumerate(graph_list):
        graph.y = torch.tensor(targets[i], dtype=torch.float)
    
    # Create DataLoader
    return GraphDataLoader(graph_list, batch_size=batch_size, shuffle=shuffle) 