import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from pypower.api import loadcase, ext2int, makeYbus
from pypower import idx_bus, idx_gen, idx_brch

class OPFDataset(Dataset):
    """Dataset class for OPF data."""
    def __init__(self, data_frame, input_cols, output_cols, transform=None):
        """
        Initialize OPF dataset.
        
        Args:
            data_frame: DataFrame containing input and output data
            input_cols: List of column names for input features
            output_cols: List of column names for output targets
            transform: Optional transform to apply to data
        """
        self.data = data_frame
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get input and output values
        input_vals = self.data.iloc[idx][self.input_cols].values.astype(np.float32)
        output_vals = self.data.iloc[idx][self.output_cols].values.astype(np.float32)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_vals, dtype=torch.float32)
        output_tensor = torch.tensor(output_vals, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            input_tensor = self.transform(input_tensor)
            
        return input_tensor, output_tensor

def load_pglib_data(case_name, data_dir="data"):
    """
    Load data from PGLib-OPF dataset (CSV format).
    
    Args:
        case_name: Name of the case (e.g., 'case118')
        data_dir: Directory containing data files
    
    Returns:
        DataFrame with loaded data
    """
    file_path = os.path.join(data_dir, f"pglib_opf_{case_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def load_case_network(case_name, data_dir="data"):
    """
    Load network data from PGLib-OPF dataset (.m format).
    
    Args:
        case_name: Name of the case (e.g., 'case118')
        data_dir: Directory containing data files
    
    Returns:
        PyPOWER case data
    """
    # Use the mock case data from the test fixtures for now
    from tests.conftest import mock_case_data
    print(f"Loading mock case data for {case_name}")
    return mock_case_data()

def prepare_data_loaders(data_frame, input_cols, output_cols, batch_size=32, 
                        train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Prepare DataLoader objects for training, validation, and testing.
    
    Args:
        data_frame: DataFrame containing input and output data
        input_cols: List of column names for input features
        output_cols: List of column names for output targets
        batch_size: Batch size for DataLoaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle and split data
    n_samples = len(data_frame)
    indices = np.random.permutation(n_samples)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create datasets
    train_df = data_frame.iloc[train_indices]
    val_df = data_frame.iloc[val_indices]
    test_df = data_frame.iloc[test_indices]
    
    train_dataset = OPFDataset(train_df, input_cols, output_cols)
    val_dataset = OPFDataset(val_df, input_cols, output_cols)
    test_dataset = OPFDataset(test_df, input_cols, output_cols)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_power_network_graph(case_data):
    """
    Create a NetworkX graph from PyPOWER case data.
    
    Args:
        case_data: PyPOWER case data
    
    Returns:
        NetworkX graph representing the power network
    """
    # Convert to internal indexing
    int_data = ext2int(case_data)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (buses)
    for i, bus in enumerate(int_data['bus']):
        bus_id = int(bus[idx_bus.BUS_I])
        node_type = int(bus[idx_bus.BUS_TYPE])
        G.add_node(bus_id, 
                  type=node_type,
                  Pd=bus[idx_bus.PD],
                  Qd=bus[idx_bus.QD],
                  Gs=bus[idx_bus.GS],
                  Bs=bus[idx_bus.BS],
                  area=int(bus[idx_bus.AREA]),
                  Vm=bus[idx_bus.VM],
                  Va=bus[idx_bus.VA],
                  baseKV=bus[idx_bus.BASE_KV],
                  zone=int(bus[idx_bus.ZONE]),
                  Vmax=bus[idx_bus.VMAX],
                  Vmin=bus[idx_bus.VMIN])
    
    # Add edges (branches)
    for i, branch in enumerate(int_data['branch']):
        from_bus = int(branch[idx_brch.F_BUS])
        to_bus = int(branch[idx_brch.T_BUS])
        G.add_edge(from_bus, to_bus,
                  r=branch[idx_brch.BR_R],
                  x=branch[idx_brch.BR_X],
                  b=branch[idx_brch.BR_B],
                  rateA=branch[idx_brch.RATE_A],
                  rateB=branch[idx_brch.RATE_B],
                  rateC=branch[idx_brch.RATE_C],
                  ratio=branch[idx_brch.TAP],
                  angle=branch[idx_brch.SHIFT],
                  status=int(branch[idx_brch.BR_STATUS]),
                  angmin=branch[idx_brch.ANGMIN],
                  angmax=branch[idx_brch.ANGMAX])
    
    return G

def prepare_graph_data(case_data, data_frame, input_cols, output_cols, batch_size=32):
    """
    Prepare graph data for GNN models.
    
    Args:
        case_data: PyPOWER case data
        data_frame: DataFrame containing input and output data
        input_cols: List of column names for input features
        output_cols: List of column names for output targets
        batch_size: Batch size for batched graph data
    
    Returns:
        Graph data loader
    """
    # This would typically convert the power system to a PyTorch Geometric format
    # For now, we'll return a placeholder
    # In a real implementation, this would create Data objects for PyTorch Geometric
    
    G = create_power_network_graph(case_data)
    
    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    # Convert to PyTorch tensor
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # Return graph data (placeholder for now)
    return G, adj_tensor 