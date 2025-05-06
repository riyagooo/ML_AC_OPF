"""
Utility functions for measuring execution times of different AC-OPF solution methods.
"""

import time
import numpy as np
import os
import json
import torch
import pandas as pd

# Constants
TIMING_DATA_FILE = "output/timing_data.json"

def time_ml_inference(model, input_data, device, num_runs=100, warmup_runs=10):
    """
    Measure inference time for ML model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The ML model to time
    input_data : tensor or DataLoader
        Input data for the model
    device : torch.device
        Device to run inference on
    num_runs : int
        Number of timing runs to perform (for statistical significance)
    warmup_runs : int
        Number of warmup runs to perform before timing
        
    Returns:
    --------
    float
        Average inference time in milliseconds
    """
    model.eval()  # Set to evaluation mode
    
    # Determine if input is a DataLoader or a single tensor
    is_dataloader = hasattr(input_data, "__iter__") and not isinstance(input_data, torch.Tensor)
    
    # Get a single batch if it's a DataLoader
    if is_dataloader:
        for batch in input_data:
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                # GNN data
                sample_input = batch.to(device)
            else:
                # Regular tensor data
                sample_input, _ = batch
                sample_input = sample_input.to(device)
            break
    else:
        # Single tensor
        sample_input = input_data.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(sample_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    print(f"Inference Time: {avg_time:.2f} ± {std_time:.2f} ms (min: {min_time:.2f} ms)")
    return avg_time

def time_traditional_solver(solver_func, problem_data, num_runs=10):
    """
    Measure execution time for traditional AC-OPF solver
    
    Parameters:
    -----------
    solver_func : callable
        Function that takes problem_data and solves the AC-OPF problem
    problem_data : any
        Data structure representing the AC-OPF problem
    num_runs : int
        Number of timing runs to perform
        
    Returns:
    --------
    float
        Average solving time in milliseconds
    """
    # Warmup run
    _ = solver_func(problem_data)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = solver_func(problem_data)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    print(f"Solver Time: {avg_time:.2f} ± {std_time:.2f} ms (min: {min_time:.2f} ms)")
    return avg_time

def save_timing_data(method_name, timing_value):
    """
    Save timing data to a JSON file
    
    Parameters:
    -----------
    method_name : str
        Name of the method being timed (e.g., 'Traditional AC-OPF Solver', 'Standard Feedforward')
    timing_value : float
        Timing value in milliseconds
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TIMING_DATA_FILE), exist_ok=True)
    
    # Load existing data if available
    if os.path.exists(TIMING_DATA_FILE):
        with open(TIMING_DATA_FILE, 'r') as f:
            timing_data = json.load(f)
    else:
        timing_data = {"methods": [], "inference_times": []}
    
    # Update data
    if method_name in timing_data["methods"]:
        idx = timing_data["methods"].index(method_name)
        timing_data["inference_times"][idx] = timing_value
    else:
        timing_data["methods"].append(method_name)
        timing_data["inference_times"].append(timing_value)
    
    # Save data
    with open(TIMING_DATA_FILE, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    print(f"Saved timing data for {method_name}: {timing_value:.2f} ms")
    return timing_data 