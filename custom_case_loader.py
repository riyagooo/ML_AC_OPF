#!/usr/bin/env python
"""
Custom case loader for MATPOWER format files.
This is a workaround for PyPOWER's loadcase function that requires
the function name to match the file name.
"""

import os
import re
import numpy as np
from pypower.api import loadcase
from pypower import idx_bus, idx_brch, idx_gen

def load_matpower_case(file_path):
    """
    Load a MATPOWER case file regardless of the function name.
    
    Args:
        file_path: Path to the MATPOWER case file
        
    Returns:
        Dictionary with case data
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Case file not found: {file_path}")
    
    # Read the case file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the actual function name used in the file
    function_match = re.search(r'function\s+mpc\s*=\s*(\w+)', content)
    if not function_match:
        raise ValueError(f"Could not find function declaration in {file_path}")
    
    function_name = function_match.group(1)
    
    # Create a dictionary to hold the case data
    mpc = {}
    
    # Execute the file content to populate the mpc dictionary
    # This is a simplified approach, normally we would create temporary file
    # but we'll try a direct approach first by parsing key components
    
    # Extract the base MVA
    base_mva_match = re.search(r'mpc\.baseMVA\s*=\s*([0-9.]+)', content)
    if base_mva_match:
        mpc['baseMVA'] = float(base_mva_match.group(1))
    else:
        mpc['baseMVA'] = 100.0  # Default value
    
    # Extract bus data
    bus_match = re.search(r'mpc\.bus\s*=\s*\[\s*(.*?)\s*\];', content, re.DOTALL)
    if bus_match:
        bus_str = bus_match.group(1)
        # Parse bus data lines
        bus_lines = [line.strip() for line in bus_str.split('\n') if line.strip() and not line.strip().startswith('%')]
        bus_data = []
        for line in bus_lines:
            # Remove trailing semicolons and comments
            line = line.split(';')[0].split('%')[0].strip()
            if line:
                values = [float(val) for val in line.split()]
                bus_data.append(values)
        mpc['bus'] = np.array(bus_data)
    
    # Extract generator data
    gen_match = re.search(r'mpc\.gen\s*=\s*\[\s*(.*?)\s*\];', content, re.DOTALL)
    if gen_match:
        gen_str = gen_match.group(1)
        # Parse gen data lines
        gen_lines = [line.strip() for line in gen_str.split('\n') if line.strip() and not line.strip().startswith('%')]
        gen_data = []
        for line in gen_lines:
            # Remove trailing semicolons and comments
            line = line.split(';')[0].split('%')[0].strip()
            if line:
                values = [float(val) for val in line.split()]
                gen_data.append(values)
        mpc['gen'] = np.array(gen_data)
    
    # Extract branch data
    branch_match = re.search(r'mpc\.branch\s*=\s*\[\s*(.*?)\s*\];', content, re.DOTALL)
    if branch_match:
        branch_str = branch_match.group(1)
        # Parse branch data lines
        branch_lines = [line.strip() for line in branch_str.split('\n') if line.strip() and not line.strip().startswith('%')]
        branch_data = []
        for line in branch_lines:
            # Remove trailing semicolons and comments
            line = line.split(';')[0].split('%')[0].strip()
            if line:
                values = [float(val) for val in line.split()]
                branch_data.append(values)
        mpc['branch'] = np.array(branch_data)
    
    # Extract generator cost data
    gencost_match = re.search(r'mpc\.gencost\s*=\s*\[\s*(.*?)\s*\];', content, re.DOTALL)
    if gencost_match:
        gencost_str = gencost_match.group(1)
        # Parse gencost data lines
        gencost_lines = [line.strip() for line in gencost_str.split('\n') if line.strip() and not line.strip().startswith('%')]
        gencost_data = []
        for line in gencost_lines:
            # Remove trailing semicolons and comments
            line = line.split(';')[0].split('%')[0].strip()
            if line:
                values = [float(val) for val in line.split()]
                gencost_data.append(values)
        mpc['gencost'] = np.array(gencost_data)
    
    # Extract areas if present
    areas_match = re.search(r'mpc\.areas\s*=\s*\[\s*(.*?)\s*\];', content, re.DOTALL)
    if areas_match:
        areas_str = areas_match.group(1)
        areas_lines = [line.strip() for line in areas_str.split('\n') if line.strip() and not line.strip().startswith('%')]
        areas_data = []
        for line in areas_lines:
            line = line.split(';')[0].split('%')[0].strip()
            if line:
                values = [float(val) for val in line.split()]
                areas_data.append(values)
        mpc['areas'] = np.array(areas_data)
    
    # Fix the case data for PyPOWER compatibility
    mpc = prepare_case_for_pypower(mpc)
    
    return mpc

def prepare_case_for_pypower(mpc):
    """
    Prepare the case data to be compatible with PyPOWER.
    
    PyPOWER expects:
    1. Buses to be numbered consecutively from 1 to n
    2. Bus indices in branch.frombus and branch.tobus to be 0-indexed internally
    3. Bus indices in gen.bus to be 0-indexed
    
    Args:
        mpc: Case data dictionary
        
    Returns:
        Modified case data dictionary
    """
    if 'bus' not in mpc or 'branch' not in mpc or 'gen' not in mpc:
        return mpc
    
    # First, ensure buses are numbered consecutively
    old_bus_nums = mpc['bus'][:, idx_bus.BUS_I].astype(int)
    n_bus = len(old_bus_nums)
    
    # Sort buses by bus number
    sort_idx = np.argsort(old_bus_nums)
    mpc['bus'] = mpc['bus'][sort_idx]
    
    # Create a mapping from old bus numbers to new (from 1 to n_bus, 1-indexed)
    new_bus_nums = np.arange(1, n_bus + 1)
    bus_map = {old: new for old, new in zip(old_bus_nums[sort_idx], new_bus_nums)}
    
    # Update bus numbers in the bus data
    mpc['bus'][:, idx_bus.BUS_I] = new_bus_nums
    
    # Update branch connections (convert to 0-indexed for internal use)
    for i in range(len(mpc['branch'])):
        from_bus = int(mpc['branch'][i, idx_brch.F_BUS])
        to_bus = int(mpc['branch'][i, idx_brch.T_BUS])
        
        # Convert to new bus numbers
        if from_bus in bus_map:
            mpc['branch'][i, idx_brch.F_BUS] = bus_map[from_bus] - 1  # Convert to 0-indexed
        else:
            print(f"Warning: Branch {i} from-bus {from_bus} not found, using first bus")
            mpc['branch'][i, idx_brch.F_BUS] = 0  # First bus (0-indexed)
        
        if to_bus in bus_map:
            mpc['branch'][i, idx_brch.T_BUS] = bus_map[to_bus] - 1  # Convert to 0-indexed
        else:
            print(f"Warning: Branch {i} to-bus {to_bus} not found, using second bus")
            mpc['branch'][i, idx_brch.T_BUS] = min(1, n_bus - 1)  # Second bus or first if only one
    
    # Update generator bus indices (convert to 0-indexed for internal use)
    for i in range(len(mpc['gen'])):
        gen_bus = int(mpc['gen'][i, idx_gen.GEN_BUS])
        
        if gen_bus in bus_map:
            mpc['gen'][i, idx_gen.GEN_BUS] = bus_map[gen_bus] - 1  # Convert to 0-indexed
        else:
            print(f"Warning: Generator {i} bus {gen_bus} not found, using first bus")
            mpc['gen'][i, idx_gen.GEN_BUS] = 0  # First bus (0-indexed)
    
    print(f"Prepared case: {n_bus} buses normalized to 1-{n_bus} (external) and 0-{n_bus-1} (internal)")
    
    return mpc

def load_case(case_name_or_path):
    """
    Load a case from either a built-in PyPOWER case or a MATPOWER file.
    
    Args:
        case_name_or_path: Case name (for built-in cases) or file path
        
    Returns:
        Dictionary with case data
    """
    if os.path.isfile(case_name_or_path):
        # Load external file using our custom loader
        return load_matpower_case(case_name_or_path)
    else:
        # Try to load a built-in case
        try:
            return loadcase(case_name_or_path)
        except Exception as e:
            raise ValueError(f"Error loading case: {e}") 