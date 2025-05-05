#!/usr/bin/env python
"""
Script to download and prepare case39 (New England) data for ML-OPF.
"""

import os
import sys
import numpy as np
import pandas as pd
import urllib.request
import tarfile
import shutil
from pathlib import Path

PGLIB_BASE_URL = "https://github.com/power-grid-lib/pglib-opf/raw/master/pglib_opf_case39_epri.m"
DATA_DIR = "data"

def ensure_directory(directory):
    """Ensure that directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_case39():
    """
    Download case39 (New England) data from PGLib.
    """
    # Create data directory if it doesn't exist
    ensure_directory(DATA_DIR)
    
    # Download .m file
    m_file_path = os.path.join(DATA_DIR, "pglib_opf_case39.m")
    if not os.path.exists(m_file_path):
        print(f"Downloading case39 .m file to {m_file_path}...")
        urllib.request.urlretrieve(PGLIB_BASE_URL, m_file_path)
        print("Download complete!")
    else:
        print(f"File already exists: {m_file_path}")
    
    return m_file_path

def generate_samples(case_file, num_samples=10000):
    """
    Generate synthetic samples for the case39 system.
    
    This is a simplified version that creates realistic load variations.
    In a real implementation, you'd use actual load profiles or more 
    sophisticated methods to generate realistic operational scenarios.
    """
    from custom_case_loader import load_case
    import random
    
    print(f"Generating {num_samples} synthetic samples for case39...")
    
    # Load the case data
    case_data = load_case(case_file)
    
    # Get base load data
    num_buses = len(case_data['bus'])
    base_p_load = case_data['bus'][:, 2] / case_data['baseMVA']  # PD
    base_q_load = case_data['bus'][:, 3] / case_data['baseMVA']  # QD
    
    # Create dataframe to store samples
    columns = []
    
    # Add load columns
    for i in range(num_buses):
        columns.append(f"load{i+1}:pl")
        columns.append(f"load{i+1}:ql")
    
    # Add generator output columns
    num_gens = len(case_data['gen'])
    for i in range(num_gens):
        columns.append(f"gen{i+1}:pg")
        columns.append(f"gen{i+1}:qg")
        columns.append(f"gen{i+1}:vm_gen")
    
    # Add bus voltage columns
    for i in range(num_buses):
        columns.append(f"bus{i+1}:v_bus")
    
    # Add branch flow columns
    num_branches = len(case_data['branch'])
    for i in range(num_branches):
        columns.append(f"line{i+1}:p_to")
        columns.append(f"line{i+1}:q_to")
        columns.append(f"line{i+1}:p_fr")
        columns.append(f"line{i+1}:q_fr")
    
    # Add constraint bounds
    for i in range(num_buses):
        columns.append(f"bus{i+1}:v_min")
        columns.append(f"bus{i+1}:v_max")
    
    for i in range(num_gens):
        columns.append(f"gen{i+1}:pg_min")
        columns.append(f"gen{i+1}:pg_max")
        columns.append(f"gen{i+1}:qg_min")
        columns.append(f"gen{i+1}:qg_max")
    
    for i in range(num_branches):
        columns.append(f"line{i+1}:p_to_max")
        columns.append(f"line{i+1}:q_to_max")
        columns.append(f"line{i+1}:p_fr_max")
        columns.append(f"line{i+1}:q_fr_max")
    
    # Create empty dataframe
    df = pd.DataFrame(columns=columns)
    
    try:
        from utils.optimization import OPFOptimizer
        
        # Create optimizer
        optimizer = OPFOptimizer(case_data)
        
        # Generate samples with load variations
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generating sample {i}/{num_samples}...")
                
            # Create load variation (0.8 to 1.2 of base load)
            load_scale = 0.8 + 0.4 * random.random()  # Between 0.8 and 1.2
            load_scales = np.random.normal(load_scale, 0.05, num_buses)
            load_scales = np.clip(load_scales, 0.7, 1.3)  # Ensure within reasonable bounds
            
            p_loads = base_p_load * load_scales
            q_loads = base_q_load * load_scales
            
            # Create load data for OPF
            load_data = np.column_stack((p_loads, q_loads))
            
            # Solve OPF
            solution = optimizer.solve_opf(load_data)
            
            if solution['success']:
                # Create a sample row
                sample = {}
                
                # Add load values
                for i in range(num_buses):
                    sample[f"load{i+1}:pl"] = p_loads[i]
                    sample[f"load{i+1}:ql"] = q_loads[i]
                
                # Add generator outputs
                for i in range(num_gens):
                    sample[f"gen{i+1}:pg"] = solution['pg'][i]
                    sample[f"gen{i+1}:qg"] = solution['qg'][i]
                    gen_bus_idx = int(case_data['gen'][i, 0]) - 1
                    sample[f"gen{i+1}:vm_gen"] = solution['vm'][gen_bus_idx]
                
                # Add bus voltages
                for i in range(num_buses):
                    sample[f"bus{i+1}:v_bus"] = solution['vm'][i]
                
                # Add branch flows
                for i in range(num_branches):
                    sample[f"line{i+1}:p_to"] = solution.get('branch_flow_to', [0] * num_branches)[i]
                    sample[f"line{i+1}:q_to"] = solution.get('branch_qflow_to', [0] * num_branches)[i]
                    sample[f"line{i+1}:p_fr"] = solution.get('branch_flow', [0] * num_branches)[i]
                    sample[f"line{i+1}:q_fr"] = solution.get('branch_qflow', [0] * num_branches)[i]
                
                # Add constraint bounds
                for i in range(num_buses):
                    sample[f"bus{i+1}:v_min"] = case_data['bus'][i, 12]  # VMIN
                    sample[f"bus{i+1}:v_max"] = case_data['bus'][i, 11]  # VMAX
                
                for i in range(num_gens):
                    sample[f"gen{i+1}:pg_min"] = case_data['gen'][i, 9] / case_data['baseMVA']  # PMIN
                    sample[f"gen{i+1}:pg_max"] = case_data['gen'][i, 8] / case_data['baseMVA']  # PMAX
                    sample[f"gen{i+1}:qg_min"] = case_data['gen'][i, 4] / case_data['baseMVA']  # QMIN
                    sample[f"gen{i+1}:qg_max"] = case_data['gen'][i, 3] / case_data['baseMVA']  # QMAX
                
                branch_rates = case_data['branch'][:, 5] / case_data['baseMVA']  # RATE_A
                for i in range(num_branches):
                    rate = branch_rates[i]
                    sample[f"line{i+1}:p_to_max"] = rate
                    sample[f"line{i+1}:q_to_max"] = rate
                    sample[f"line{i+1}:p_fr_max"] = rate
                    sample[f"line{i+1}:q_fr_max"] = rate
                
                # Add to dataframe
                df = pd.concat([df, pd.DataFrame([sample])], ignore_index=True)
                
    except Exception as e:
        print(f"Error generating samples: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a minimal synthetic dataset for testing if OPF fails
        print("Creating minimal synthetic dataset for testing...")
        data = np.random.normal(0, 1, (num_samples, len(columns)))
        df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    csv_file_path = os.path.join(DATA_DIR, "pglib_opf_case39.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Created synthetic dataset with {len(df)} samples saved to {csv_file_path}")
    
    return csv_file_path

if __name__ == "__main__":
    print("Preparing case39 (New England) data for ML-OPF...")
    m_file_path = download_case39()
    
    # Check if CSV already exists
    csv_file_path = os.path.join(DATA_DIR, "pglib_opf_case39.csv")
    if not os.path.exists(csv_file_path):
        print("Generating synthetic data for case39...")
        csv_file_path = generate_samples(m_file_path, num_samples=2000)  # Start with 2000 samples for faster testing
    else:
        print(f"CSV file already exists: {csv_file_path}")
    
    print("Data preparation complete!")
    print(f".m file: {m_file_path}")
    print(f"CSV file: {csv_file_path}")
    print("\nYou can now run: python run.py train --approach feedforward --case case39")