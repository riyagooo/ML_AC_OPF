#!/usr/bin/env python
"""
Optimization metrics module.
"""

import torch
import numpy as np

def optimality_gap_metric(pred_gen, target_gen, cost_coeffs):
    """
    Compute optimality gap between predicted and target generator outputs.
    
    Args:
        pred_gen: Predicted generator outputs
        target_gen: Target (optimal) generator outputs
        cost_coeffs: Linear cost coefficients for each generator
    
    Returns:
        Optimality gap as percentage
    """
    # Ensure inputs are on the same device
    device = pred_gen.device
    if cost_coeffs.device != device:
        cost_coeffs = cost_coeffs.to(device)
    
    try:
        # Compute generation costs
        pred_cost = torch.sum(pred_gen * cost_coeffs, dim=1)
        target_cost = torch.sum(target_gen * cost_coeffs, dim=1)
        
        # Compute optimality gap
        opt_gap = (pred_cost - target_cost) / target_cost * 100.0
        
        # Average over batch
        return torch.mean(opt_gap)
    except Exception as e:
        print(f"Error in optimality gap calculation: {e}")
        # Return a default value on error
        return torch.tensor(100.0, device=device)

def constraint_violation_metric(predictions, system_data):
    """
    Compute constraint violation metric for power system.
    
    Args:
        predictions: Predicted variables (generator outputs, voltages, etc.)
        system_data: Power system data (buses, generators, branches)
    
    Returns:
        Constraint violation as a scalar value
    """
    # This is a placeholder for a more comprehensive constraint violation check
    # In a real implementation, we would check power flow constraints, voltage limits, etc.
    return torch.tensor(0.0)

def accuracy_metric(pred, target, threshold=0.5):
    """
    Compute binary classification accuracy for constraint screening.
    
    Args:
        pred: Predicted binding constraints (probability)
        target: Actual binding constraints (binary)
        threshold: Decision threshold
    
    Returns:
        Classification accuracy
    """
    pred_binary = (pred > threshold).float()
    return (pred_binary == target).float().mean() 