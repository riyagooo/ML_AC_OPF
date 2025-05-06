"""
Case39 (New England) power system utilities for ML-AC-OPF.

This module provides specialized functions for working with the IEEE 39-bus
(New England) test system, with improved numerical stability compared to case30.
"""

import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from pypower.api import case39, runopf, makeYbus, ext2int, runpf
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('case39_utils')

def load_case39_data(data_path=None):
    """
    Load the IEEE 39-bus (New England) test case data.
    
    Args:
        data_path: Path to case39 data file (.m or .raw format)
                  If None, uses PyPOWER's built-in case39
    
    Returns:
        PyPOWER case data for the 39-bus system formatted for evaluation
    """
    if data_path is not None and os.path.exists(data_path):
        try:
            from custom_case_loader import load_case
            logger.info(f"Loading case39 data from {data_path}")
            case_data = load_case(data_path)
        except ImportError:
            logger.warning("custom_case_loader not available, falling back to built-in case39")
            case_data = case39()
    else:
        # Fallback to built-in case39
        logger.info("Using PyPOWER's built-in case39 data")
        case_data = case39()
    
    # Add additional case parameters for PowerSystemValidator
    # These are placeholders for the required fields
    
    # Ensure attributes needed by PowerSystemValidator
    case_data['n_bus'] = len(case_data['bus'])
    case_data['n_gen'] = len(case_data['gen'])
    case_data['n_branch'] = len(case_data['branch'])
    
    return case_data

def load_case39(data_path=None):
    """
    Load the IEEE 39-bus (New England) test case.
    
    Args:
        data_path: Path to case39 data file (.m or .raw format)
                  If None, uses PyPOWER's built-in case39
    
    Returns:
        PyPOWER case data for the 39-bus system
    """
    if data_path is not None and os.path.exists(data_path):
        # Check if we have a custom loader module
        try:
            from custom_case_loader import load_case
            logger.info(f"Loading case39 data from {data_path}")
            return load_case(data_path)
        except ImportError:
            logger.warning("custom_case_loader not available, falling back to built-in case39")
    
    # Fallback to built-in case39
    logger.info("Using PyPOWER's built-in case39 data")
    return case39()

def create_case39_graph(case_data=None):
    """
    Create a NetworkX graph representation of the IEEE 39-bus system.
    
    Args:
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        NetworkX graph of the power system
    """
    if case_data is None:
        case_data = case39()
    
    # Convert to internal indexing
    int_case = ext2int(case_data)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (buses)
    for i, bus in enumerate(int_case['bus']):
        bus_id = int(bus[0])
        bus_type = int(bus[1])
        pd = bus[2] / case_data['baseMVA']  # Normalize active load
        qd = bus[3] / case_data['baseMVA']  # Normalize reactive load
        gs = bus[4] / case_data['baseMVA']  # Normalize shunt conductance
        bs = bus[5] / case_data['baseMVA']  # Normalize shunt susceptance
        vm = bus[7]  # Voltage magnitude (p.u.)
        va = bus[8] / 180.0 * np.pi  # Voltage angle (rad)
        base_kv = bus[9]  # Base voltage (kV)
        vmax = bus[11]  # Max voltage (p.u.)
        vmin = bus[12]  # Min voltage (p.u.)
        
        # Create a dictionary with node attributes
        node_attrs = {
            'type': bus_type,
            'pd': pd,
            'qd': qd,
            'gs': gs,
            'bs': bs,
            'vm': vm,
            'va': va,
            'base_kv': base_kv,
            'vmax': vmax,
            'vmin': vmin
        }
        
        G.add_node(bus_id, **node_attrs)
    
    # Add edges (branches)
    for branch in int_case['branch']:
        from_bus = int(branch[0])
        to_bus = int(branch[1])
        
        # Create a dictionary with edge attributes
        edge_attrs = {
            'r': branch[2],  # Resistance
            'x': branch[3],  # Reactance
            'b': branch[4],  # Line charging susceptance
            'rateA': branch[5],  # MVA rating A
            'rateB': branch[6],  # MVA rating B
            'rateC': branch[7],  # MVA rating C
            'ratio': branch[8],  # Transformer tap ratio
            'angle': branch[9],  # Transformer phase shift angle
            'status': int(branch[10]),  # Branch status
            'angmin': branch[11],  # Minimum angle difference
            'angmax': branch[12]   # Maximum angle difference
        }
        
        G.add_edge(from_bus, to_bus, **edge_attrs)
    
    return G

def pyg_graph_from_case39(case_data=None):
    """
    Create a PyTorch Geometric Data object from IEEE 39-bus system.
    
    Args:
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        PyTorch Geometric Data object
    """
    if case_data is None:
        case_data = case39()
    
    # Convert to internal indexing if not already done
    if not hasattr(case_data, 'internal'):
        int_case = ext2int(case_data)
    else:
        int_case = case_data
    
    # Extract bus, branch, and generator data
    buses = int_case['bus']
    branches = int_case['branch']
    
    num_buses = buses.shape[0]
    num_branches = branches.shape[0]
    
    # Create node features
    # Features: [PD, QD, GS, BS, BUS_TYPE, VM, VA, BASE_KV, VMAX, VMIN]
    node_features = []
    for i in range(num_buses):
        bus = buses[i]
        # Normalize by base MVA
        pd = bus[2] / case_data['baseMVA']
        qd = bus[3] / case_data['baseMVA']
        gs = bus[4] / case_data['baseMVA']
        bs = bus[5] / case_data['baseMVA']
        bus_type = bus[1] / 3.0  # Normalize bus type (1=PQ, 2=PV, 3=ref)
        vm = bus[7]  # Voltage magnitude
        va = bus[8] / 180.0  # Voltage angle (normalized)
        base_kv = bus[9] / 100.0  # Base KV (normalized)
        vmax = bus[11]  # Max voltage
        vmin = bus[12]  # Min voltage
        
        node_features.append([pd, qd, gs, bs, bus_type, vm, va, base_kv, vmax, vmin])
    
    # Create edge index and edge features
    edge_index = []
    edge_attr = []
    
    # Validate branches to ensure indices are within range
    valid_branches = []
    for i in range(num_branches):
        branch = branches[i]
        # Branches should already be 0-indexed in internal data format
        f_bus = int(branch[0])
        t_bus = int(branch[1])
        
        # Double-check that indices are valid
        if 0 <= f_bus < num_buses and 0 <= t_bus < num_buses:
            valid_branches.append(i)
        else:
            logger.warning(f"Skipping branch {i} with invalid indices: {f_bus} -> {t_bus} (num_buses={num_buses})")
    
    # Process only valid branches
    for i in valid_branches:
        branch = branches[i]
        
        # Get from and to buses (assumed to be 0-indexed after ext2int)
        f_bus = int(branch[0])
        t_bus = int(branch[1])
        
        # Add edge in both directions (undirected graph)
        edge_index.append([f_bus, t_bus])
        edge_index.append([t_bus, f_bus])
        
        # Edge features: [R, X, B, TAP, SHIFT, BR_STATUS]
        r = branch[2]  # Resistance
        x = branch[3]  # Reactance
        b = branch[4]  # Line charging susceptance
        tap = branch[8]  # Transformer tap ratio
        shift = branch[9] / 180.0  # Phase shift angle (normalized)
        status = branch[10]  # Branch status
        
        # Add the same features for both directions
        edge_attr.append([r, x, b, tap, shift, status])
        edge_attr.append([r, x, b, tap, shift, status])
    
    # If no valid branches, add a self-loop to the first node to avoid errors
    if not edge_index:
        logger.warning("No valid branches found. Adding a self-loop to the first node.")
        edge_index.append([0, 0])
        edge_attr.append([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def generate_case39_scenarios(num_samples=1000, load_scaling_range=(0.9, 1.1), case_data=None):
    """
    Generate synthetic scenarios for case39 for ML training.
    
    This function creates synthetic variations around the base case without
    running power flow calculations, since those are prone to convergence issues.
    
    Args:
        num_samples: Number of scenarios to generate
        load_scaling_range: Range for random load scaling factors
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        DataFrame with generated scenarios
    """
    if case_data is None:
        case_data = case39()
    
    # Get base data
    num_buses = len(case_data['bus'])
    num_gens = len(case_data['gen'])
    
    # Base values
    base_p_load = case_data['bus'][:, 2] / case_data['baseMVA']  # PD
    base_q_load = case_data['bus'][:, 3] / case_data['baseMVA']  # QD
    base_vm = case_data['bus'][:, 7]  # VM
    base_va = case_data['bus'][:, 8]  # VA
    gen_buses = case_data['gen'][:, 0].astype(int)  # Generator buses (1-indexed)
    base_gen_p = case_data['gen'][:, 1] / case_data['baseMVA']  # PG
    base_gen_q = case_data['gen'][:, 2] / case_data['baseMVA']  # QG
    
    # Create empty DataFrame for storing results
    columns = []
    
    # Add load columns
    for i in range(num_buses):
        columns.append(f'load_p_{i+1}')
        columns.append(f'load_q_{i+1}')
    
    # Add generator columns
    for i in range(num_gens):
        columns.append(f'gen_p_{i+1}')
        columns.append(f'gen_q_{i+1}')
    
    # Add voltage columns
    for i in range(num_buses):
        columns.append(f'vm_{i+1}')
        columns.append(f'va_{i+1}')
    
    # Create DataFrame with synthetic scenarios
    data = []
    
    logger.info(f"Generating {num_samples} synthetic scenarios for case39...")
    
    for i in range(num_samples):
        # Scale loads based on random factors
        load_scale_factors = np.random.uniform(load_scaling_range[0], load_scaling_range[1], num_buses)
        p_loads = base_p_load * load_scale_factors
        q_loads = base_q_load * load_scale_factors
        
        # Adjust total generation to match total load (with 5% losses)
        total_load = np.sum(p_loads)
        total_gen = np.sum(base_gen_p)
        gen_scale_factor = total_load * 1.05 / total_gen if total_gen > 0 else 1.0
        
        # Scale generator outputs
        gen_p = base_gen_p * gen_scale_factor
        
        # Perturb generator reactive power slightly (less significant for learning)
        gen_q = base_gen_q * np.random.uniform(0.95, 1.05, num_gens)
        
        # Perturb voltage magnitudes slightly from nominal values
        # (using much smaller perturbations for voltage)
        vm_scale_factors = np.random.uniform(0.98, 1.02, num_buses)
        vm = base_vm * vm_scale_factors
        
        # Voltage angles - perturb slightly based on load changes
        # (high load tends to result in larger angle differences)
        va_scale_factors = 1.0 + 0.1 * (load_scale_factors - 1.0)
        va = base_va * va_scale_factors
        # Fix reference bus angle to 0
        ref_bus_idx = np.where(case_data['bus'][:, 1] == 3)[0][0]
        va[ref_bus_idx] = 0.0
        
        # Create row dict
        row = {}
        
        # Add load values
        for j in range(num_buses):
            row[f'load_p_{j+1}'] = p_loads[j]
            row[f'load_q_{j+1}'] = q_loads[j]
        
        # Add generator values
        for j in range(num_gens):
            row[f'gen_p_{j+1}'] = gen_p[j]
            row[f'gen_q_{j+1}'] = gen_q[j]
        
        # Add voltage values
        for j in range(num_buses):
            row[f'vm_{j+1}'] = vm[j]
            row[f'va_{j+1}'] = va[j]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Successfully generated {num_samples} synthetic scenarios")
    
    return df

def get_case39_bounds(case_data=None):
    """
    Get bounds for generator outputs and voltage values for case39.
    
    Args:
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        Dictionary with bounds for different variables
    """
    if case_data is None:
        case_data = case39()
    
    # Extract generator bounds
    num_gens = len(case_data['gen'])
    gen_p_min = case_data['gen'][:, 9] / case_data['baseMVA']
    gen_p_max = case_data['gen'][:, 8] / case_data['baseMVA']
    gen_q_min = case_data['gen'][:, 4] / case_data['baseMVA']
    gen_q_max = case_data['gen'][:, 3] / case_data['baseMVA']
    
    # Extract voltage bounds
    num_buses = len(case_data['bus'])
    vm_min = case_data['bus'][:, 12]
    vm_max = case_data['bus'][:, 11]
    
    # Create bounds dictionary
    bounds = {
        'gen_p_min': gen_p_min,
        'gen_p_max': gen_p_max,
        'gen_q_min': gen_q_min,
        'gen_q_max': gen_q_max,
        'vm_min': vm_min,
        'vm_max': vm_max,
        'va_min': np.ones(num_buses) * (-np.pi),
        'va_max': np.ones(num_buses) * np.pi
    }
    
    return bounds

def case39_to_tensors(df, bounds=None, case_data=None):
    """
    Convert case39 DataFrame to input and output tensors for ML models.
    
    Args:
        df: DataFrame with scenarios
        bounds: Dictionary with bounds for outputs (from get_case39_bounds)
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        input_tensor, output_tensor, normalized_output_tensor, normalization_params
    """
    if case_data is None:
        case_data = case39()
    
    if bounds is None:
        bounds = get_case39_bounds(case_data)
    
    # Determine input and output columns
    num_buses = len(case_data['bus'])
    num_gens = len(case_data['gen'])
    
    input_cols = []
    output_cols = []
    
    # Input columns: loads
    for i in range(num_buses):
        input_cols.append(f'load_p_{i+1}')
        input_cols.append(f'load_q_{i+1}')
    
    # Output columns: generator outputs and voltages
    for i in range(num_gens):
        output_cols.append(f'gen_p_{i+1}')
        output_cols.append(f'gen_q_{i+1}')
    
    for i in range(num_buses):
        output_cols.append(f'vm_{i+1}')
        output_cols.append(f'va_{i+1}')
    
    # Handle empty dataframe
    if len(df) == 0:
        logger.warning("Empty dataframe provided to case39_to_tensors. Creating dummy tensors.")
        input_dim = len(input_cols)
        output_dim = len(output_cols)
        
        input_tensor = torch.zeros((1, input_dim), dtype=torch.float32)
        output_tensor = torch.zeros((1, output_dim), dtype=torch.float32)
        normalized_output_tensor = torch.zeros((1, output_dim), dtype=torch.float32)
        
        # Create normalization parameters
        normalization_params = {
            'gen_p': {'min': bounds['gen_p_min'], 'max': bounds['gen_p_max']},
            'gen_q': {'min': bounds['gen_q_min'], 'max': bounds['gen_q_max']},
            'vm': {'min': bounds['vm_min'], 'max': bounds['vm_max']},
            'va': {'min': bounds['va_min'], 'max': bounds['va_max']}
        }
        
        return input_tensor, output_tensor, normalized_output_tensor, normalization_params
    
    # Make sure all columns are present and are numeric
    for col in input_cols + output_cols:
        if col not in df.columns:
            df[col] = 0.0
        # Ensure numeric type
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Create input tensor
    input_data = df[input_cols].values
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Create output tensor
    output_data = df[output_cols].values
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    
    # Create normalization parameters
    normalization_params = {
        'gen_p': {'min': bounds['gen_p_min'], 'max': bounds['gen_p_max']},
        'gen_q': {'min': bounds['gen_q_min'], 'max': bounds['gen_q_max']},
        'vm': {'min': bounds['vm_min'], 'max': bounds['vm_max']},
        'va': {'min': bounds['va_min'], 'max': bounds['va_max']}
    }
    
    # Create normalized output tensor
    normalized_output = output_data.copy()
    
    # Normalize generator outputs
    for i in range(num_gens):
        p_min = bounds['gen_p_min'][i]
        p_max = bounds['gen_p_max'][i]
        q_min = bounds['gen_q_min'][i]
        q_max = bounds['gen_q_max'][i]
        
        # Normalize gen_p to [0, 1]
        normalized_output[:, i] = (output_data[:, i] - p_min) / (p_max - p_min)
        
        # Normalize gen_q to [0, 1]
        normalized_output[:, num_gens + i] = (output_data[:, num_gens + i] - q_min) / (q_max - q_min)
    
    # Normalize voltages
    offset = 2 * num_gens
    for i in range(num_buses):
        # Normalize vm to [0, 1]
        vm_min = bounds['vm_min'][i]
        vm_max = bounds['vm_max'][i]
        normalized_output[:, offset + i] = (output_data[:, offset + i] - vm_min) / (vm_max - vm_min)
        
        # Normalize va to [0, 1]
        normalized_output[:, offset + num_buses + i] = (output_data[:, offset + num_buses + i] + np.pi) / (2 * np.pi)
    
    normalized_output_tensor = torch.tensor(normalized_output, dtype=torch.float32)
    
    return input_tensor, output_tensor, normalized_output_tensor, normalization_params

def denormalize_case39_outputs(normalized_outputs, normalization_params, case_data=None):
    """
    Denormalize case39 outputs from ML predictions.
    
    Args:
        normalized_outputs: Tensor or numpy array with normalized outputs
        normalization_params: Dictionary with normalization parameters
        case_data: PyPOWER case data (uses case39 if None)
        
    Returns:
        Denormalized outputs
    """
    if case_data is None:
        case_data = case39()
    
    # Convert to numpy if tensor
    if torch.is_tensor(normalized_outputs):
        normalized_outputs = normalized_outputs.detach().cpu().numpy()
    
    num_buses = len(case_data['bus'])
    num_gens = len(case_data['gen'])
    
    # Create copy for denormalization
    denormalized = normalized_outputs.copy()
    
    # Denormalize generator outputs
    for i in range(num_gens):
        p_min = normalization_params['gen_p']['min'][i]
        p_max = normalization_params['gen_p']['max'][i]
        q_min = normalization_params['gen_q']['min'][i]
        q_max = normalization_params['gen_q']['max'][i]
        
        # Denormalize gen_p from [0, 1] to [p_min, p_max]
        denormalized[:, i] = normalized_outputs[:, i] * (p_max - p_min) + p_min
        
        # Denormalize gen_q from [0, 1] to [q_min, q_max]
        denormalized[:, num_gens + i] = normalized_outputs[:, num_gens + i] * (q_max - q_min) + q_min
    
    # Denormalize voltages
    offset = 2 * num_gens
    for i in range(num_buses):
        # Denormalize vm from [0, 1] to [vm_min, vm_max]
        vm_min = normalization_params['vm']['min'][i]
        vm_max = normalization_params['vm']['max'][i]
        denormalized[:, offset + i] = normalized_outputs[:, offset + i] * (vm_max - vm_min) + vm_min
        
        # Denormalize va from [0, 1] to [-π, π]
        denormalized[:, offset + num_buses + i] = normalized_outputs[:, offset + num_buses + i] * (2 * np.pi) - np.pi
    
    return denormalized 