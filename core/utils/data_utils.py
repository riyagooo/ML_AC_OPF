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
            
        # Get input and output values - filter columns that can be converted to float
        try:
            input_df = self.data.iloc[idx][self.input_cols]
            output_df = self.data.iloc[idx][self.output_cols]
            
            # Process any complex numbers in the data
            input_vals = []
            for col in self.input_cols:
                val = input_df[col]
                # Try to handle complex number strings
                if isinstance(val, str) and ('j' in val or 'i' in val):
                    # Extract real part only
                    val = float(val.split('+')[0].split('-')[0].strip())
                input_vals.append(float(val))
                
            output_vals = []
            for col in self.output_cols:
                val = output_df[col]
                # Try to handle complex number strings
                if isinstance(val, str) and ('j' in val or 'i' in val):
                    # Extract real part only
                    val = float(val.split('+')[0].split('-')[0].strip())
                output_vals.append(float(val))
                
            input_vals = np.array(input_vals, dtype=np.float32)
            output_vals = np.array(output_vals, dtype=np.float32)
        except Exception as e:
            if idx < 5:  # Only print for the first few problematic rows to avoid too much output
                print(f"Error processing row {idx}: {e}")
                if isinstance(val, str):
                    print(f"Problematic value: {val}")
            # Return zeros if there's an error processing the data
            input_vals = np.zeros(len(self.input_cols), dtype=np.float32)
            output_vals = np.zeros(len(self.output_cols), dtype=np.float32)
        
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
    # Check if custom_case_loader module is available
    try:
        from custom_case_loader import load_case
        file_path = os.path.join(data_dir, f"pglib_opf_{case_name}.m")
        if os.path.exists(file_path):
            print(f"Loading case data for {case_name} using custom loader")
            return load_case(file_path)
        else:
            raise FileNotFoundError(f"Case file not found: {file_path}")
    except ImportError:
        # Fallback to original mock data method
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
        
        # Build node attributes dictionary with only available attributes
        node_attrs = {
            'type': node_type,
            'Pd': bus[idx_bus.PD],
            'Qd': bus[idx_bus.QD],
            'Gs': bus[idx_bus.GS],
            'Bs': bus[idx_bus.BS],
            'Vm': bus[idx_bus.VM],
            'Va': bus[idx_bus.VA],
            'baseKV': bus[idx_bus.BASE_KV],
            'Vmax': bus[idx_bus.VMAX],
            'Vmin': bus[idx_bus.VMIN]
        }
        
        # Add optional attributes only if they exist in idx_bus
        if hasattr(idx_bus, 'AREA'):
            node_attrs['area'] = int(bus[idx_bus.AREA])
        if hasattr(idx_bus, 'ZONE'):
            node_attrs['zone'] = int(bus[idx_bus.ZONE])
        
        G.add_node(bus_id, **node_attrs)
    
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

def normalize_opf_data(data, case_data, input_cols=None, output_cols=None):
    """
    Apply robust normalization to OPF data following standard power system practices.
    
    Args:
        data: DataFrame with OPF data
        case_data: PyPOWER case data dictionary
        input_cols: List of input column names (if None, auto-detect)
        output_cols: List of output column names (if None, auto-detect)
        
    Returns:
        DataFrame with normalized data and normalization parameters
    """
    # Make a copy to avoid modifying the original
    normalized_data = data.copy()
    
    # Get system base MVA
    base_mva = case_data['baseMVA']
    
    # Auto-detect columns if not provided
    if input_cols is None:
        load_p_cols = [col for col in data.columns if col.startswith('load_p') or 
                      (col.startswith('load') and ':pl' in col)]
        load_q_cols = [col for col in data.columns if col.startswith('load_q') or 
                      (col.startswith('load') and ':ql' in col)]
        input_cols = load_p_cols + load_q_cols
    
    if output_cols is None:
        gen_p_cols = [col for col in data.columns if col.startswith('gen_p') or 
                     (col.startswith('gen') and ':pg' in col)]
        gen_q_cols = [col for col in data.columns if col.startswith('gen_q') or 
                     (col.startswith('gen') and ':qg' in col)]
        vm_cols = [col for col in data.columns if col.startswith('vm') or 
                  (col.startswith('bus') and ':v_m' in col)]
        va_cols = [col for col in data.columns if col.startswith('va') or 
                  (col.startswith('bus') and ':v_a' in col)]
        output_cols = gen_p_cols + gen_q_cols + vm_cols + va_cols
    
    # Store normalization parameters
    norm_params = {}
    
    # Process power-related inputs (normalize by base MVA)
    for col in input_cols:
        if any(x in col for x in ['load_p', 'load_q', ':pl', ':ql']):
            # Power quantities normalized by baseMVA
            normalized_data[col] = data[col] / base_mva
            norm_params[col] = {'type': 'power', 'scale': base_mva}
    
    # Process power-related outputs with special handling for generators
    for col in output_cols:
        if any(x in col for x in ['gen_p', 'gen_q', ':pg', ':qg']):
            # Check if this is an active generator (significant power output)
            mean_val = data[col].mean()
            std_val = data[col].std()
            is_active = (mean_val > 0.1) or (std_val > 0.1)
            
            if is_active:
                # Active generators: use gentler normalization to preserve dynamic range
                # Min-max scaling with padding to avoid compression
                min_val = data[col].min()
                max_val = data[col].max()
                range_val = max_val - min_val
                
                # Add 10% padding to range
                min_padded = min_val - 0.1 * range_val
                max_padded = max_val + 0.1 * range_val
                
                # Apply min-max scaling to [0, 1] range
                normalized_data[col] = (data[col] - min_padded) / (max_padded - min_padded)
                
                # Store normalization parameters
                norm_params[col] = {
                    'type': 'active_gen',
                    'min': min_padded,
                    'max': max_padded
                }
            else:
                # Inactive generators: standard normalization
                normalized_data[col] = data[col] / base_mva
                norm_params[col] = {'type': 'power', 'scale': base_mva}
        
        elif any(x in col for x in ['vm', 'v_m']):
            # Voltage magnitudes already in per-unit, but may need centering
            vm_mean = data[col].mean()
            normalized_data[col] = data[col] - vm_mean + 1.0  # Center around 1.0 p.u.
            norm_params[col] = {'type': 'voltage_mag', 'offset': vm_mean - 1.0}
            
        elif any(x in col for x in ['va', 'v_a']):
            # Convert voltage angles to radians if in degrees
            if data[col].abs().max() > 3.14159:  # Likely in degrees
                normalized_data[col] = np.radians(data[col])
                norm_params[col] = {'type': 'voltage_ang', 'convert': 'deg2rad'}
            
            # Normalize to range [-1, 1] by dividing by π
            normalized_data[col] = normalized_data[col] / np.pi
            norm_params[col]['scale'] = np.pi
    
    # Add robust scaling for any extreme values
    for col in input_cols + output_cols:
        # Skip active generator columns which have already been handled specially
        if col in norm_params and norm_params[col].get('type') == 'active_gen':
            continue
            
        # Check for extreme values using IQR
        q1 = normalized_data[col].quantile(0.25)
        q3 = normalized_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # If we have outliers, cap values to reduce their impact
        outliers = (normalized_data[col] < lower_bound) | (normalized_data[col] > upper_bound)
        if outliers.sum() > 0:
            print(f"Found {outliers.sum()} outliers in {col}, applying robust scaling")
            normalized_data.loc[normalized_data[col] < lower_bound, col] = lower_bound
            normalized_data.loc[normalized_data[col] > upper_bound, col] = upper_bound
            
            # Update normalization parameters
            if col in norm_params:
                norm_params[col]['robust_bounds'] = (lower_bound, upper_bound)
            else:
                norm_params[col] = {'robust_bounds': (lower_bound, upper_bound)}
    
    return normalized_data, norm_params

def denormalize_opf_data(normalized_data, norm_params):
    """
    Denormalize OPF data using stored normalization parameters.
    
    Args:
        normalized_data: DataFrame with normalized data
        norm_params: Dictionary with normalization parameters
        
    Returns:
        DataFrame with denormalized data
    """
    # Make a copy to avoid modifying the original
    denormalized_data = normalized_data.copy()
    
    # Apply denormalization for each column based on stored parameters
    for col, params in norm_params.items():
        if col in denormalized_data.columns:
            # First remove any robust scaling (no need to modify data)
            
            # Then reverse the main normalization
            if params.get('type') == 'power':
                denormalized_data[col] = normalized_data[col] * params['scale']
            
            elif params.get('type') == 'active_gen':
                # Reverse min-max scaling for active generators
                min_val = params['min']
                max_val = params['max']
                denormalized_data[col] = normalized_data[col] * (max_val - min_val) + min_val
                
            elif params.get('type') == 'voltage_mag':
                denormalized_data[col] = normalized_data[col] + params['offset']
                
            elif params.get('type') == 'voltage_ang':
                # First multiply by π to get back to radians
                denormalized_data[col] = normalized_data[col] * params['scale']
                
                # Convert back to degrees if originally in degrees
                if params.get('convert') == 'deg2rad':
                    denormalized_data[col] = np.degrees(denormalized_data[col])
    
    return denormalized_data

# Modified data loader function that applies normalization
def prepare_normalized_data_loaders(data, case_data, input_cols, output_cols, 
                                   batch_size=32, train_ratio=0.7, val_ratio=0.15,
                                   seed=42):
    """
    Prepare data loaders with normalized data following power system practices.
    
    Args:
        data: DataFrame with OPF data
        case_data: PyPOWER case data dictionary
        input_cols: List of input column names
        output_cols: List of output column names
        batch_size: Batch size for loaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, norm_params)
    """
    # Apply normalization
    normalized_data, norm_params = normalize_opf_data(
        data, case_data, input_cols, output_cols)
    
    # Get loaders using the standard function
    train_loader, val_loader, test_loader = prepare_data_loaders(
        normalized_data, input_cols, output_cols, batch_size, train_ratio, val_ratio, seed)
    
    # Return loaders and normalization parameters for later denormalization
    return train_loader, val_loader, test_loader, norm_params 