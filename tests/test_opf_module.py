#!/usr/bin/env python
"""
Standalone test for OPFOptimizer with accurate power flow modeling.
This script tests the enhanced power flow equations in utils/optimization.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom case loader
from custom_case_loader import load_case

# Import optimization module
from utils.optimization import OPFOptimizer
from utils.error_handling import setup_error_handling

# Set up logging
setup_error_handling()

def test_opf_optimizer(case_name='case5', verbose=True):
    """
    Test the OPFOptimizer with a simple test case.
    
    Args:
        case_name: Name of the test case (default: case5)
        verbose: Whether to print detailed information
    """
    print(f"Testing OPFOptimizer with {case_name}...")
    
    # Load case data
    if case_name == 'case5':
        # Use direct path for case5
        data_file = os.path.join('data', 'pglib_opf_case5.m')
        if not os.path.exists(data_file):
            print(f"Error: File not found: {data_file}")
            return
        
        try:
            # Load case using our custom loader
            case_data = load_case(data_file)
            print("Successfully loaded case using custom loader")
        except Exception as e:
            print(f"Error loading case with custom loader: {e}")
            return
    else:
        # Try to load from PYPOWER built-in cases
        try:
            case_data = load_case(case_name)
        except Exception as e:
            print(f"Error loading case: {e}")
            return
    
    print(f"Case loaded: {len(case_data['bus'])} buses, {len(case_data['gen'])} generators, {len(case_data['branch'])} branches")
    
    # Create optimizer
    optimizer = OPFOptimizer(case_data)
    
    # Create some sample load data
    n_bus = len(case_data['bus'])
    base_load_p = case_data['bus'][:, 2] / case_data['baseMVA']  # Active load
    base_load_q = case_data['bus'][:, 3] / case_data['baseMVA']  # Reactive load
    
    # Create variations of the base load
    load_variations = []
    
    # Add base case
    load_variations.append(np.column_stack((base_load_p, base_load_q)))
    
    # Add cases with increased and decreased load
    factors = [0.8, 0.9, 1.1, 1.2]
    for factor in factors:
        scaled_p = base_load_p * factor
        scaled_q = base_load_q * factor
        load_variations.append(np.column_stack((scaled_p, scaled_q)))
    
    # Solve OPF for each load variation
    results = []
    
    for i, load_data in enumerate(load_variations):
        factor_str = "1.0" if i == 0 else f"{factors[i-1]:.1f}"
        print(f"\nSolving OPF for load factor {factor_str}...")
        
        # Solve with PyPOWER (for comparison)
        pypower_solution = optimizer.solve_opf(load_data)
        
        # Solve with Gurobi
        gurobi_solution = optimizer.solve_opf_gurobi(load_data)
        
        # Compare solutions
        if pypower_solution['success'] and gurobi_solution['success']:
            pp_obj = pypower_solution.get('f', float('inf'))
            gr_obj = gurobi_solution.get('f', float('inf'))
            
            print(f"  PyPOWER objective: {pp_obj:.4f}")
            print(f"  Gurobi objective:  {gr_obj:.4f}")
            print(f"  Difference: {abs(pp_obj - gr_obj):.4f} ({abs(pp_obj - gr_obj) / pp_obj * 100:.2f}%)")
            
            # Evaluate solution quality
            print("\nEvaluating solution quality...")
            violations = optimizer.evaluate_solution(gurobi_solution, load_data)
            
            print(f"  Power balance (P): {violations['power_balance_p']:.6f}")
            print(f"  Power balance (Q): {violations['power_balance_q']:.6f}")
            print(f"  Branch thermal violations: {violations['branch_thermal']:.6f}")
            print(f"  Generator violations: {violations['gen_p_limits'] + violations['gen_q_limits']:.6f}")
            print(f"  Voltage violations: {violations['voltage_limits']:.6f}")
            print(f"  Total violations: {violations['total']:.6f}")
            
            results.append({
                'load_factor': float(factor_str),
                'pypower_obj': pp_obj,
                'gurobi_obj': gr_obj,
                'violations': violations
            })
        elif not pypower_solution['success']:
            print("  PyPOWER solution failed")
            if gurobi_solution['success']:
                print("  Gurobi solution succeeded")
                # Evaluate Gurobi solution
                violations = optimizer.evaluate_solution(gurobi_solution, load_data)
                print(f"  Total violations: {violations['total']:.6f}")
            else:
                print("  Gurobi solution failed")
        elif not gurobi_solution['success']:
            print("  Gurobi solution failed")
            print(f"  Status: {gurobi_solution.get('status_message', 'Unknown')}")
    
    # Plot results if we have any
    if results:
        plot_results(results)
    
    return results

def plot_results(results):
    """
    Plot the results of the OPF solutions.
    
    Args:
        results: List of dictionaries with results
    """
    # Extract data for plotting
    load_factors = [r['load_factor'] for r in results]
    pypower_objs = [r['pypower_obj'] for r in results]
    gurobi_objs = [r['gurobi_obj'] for r in results]
    total_violations = [r['violations']['total'] for r in results]
    power_balance_violations = [r['violations']['power_balance_p'] + r['violations']['power_balance_q'] for r in results]
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot objective values
    ax1 = axs[0]
    ax1.plot(load_factors, pypower_objs, 'o-', label='PyPOWER')
    ax1.plot(load_factors, gurobi_objs, 's-', label='Gurobi')
    ax1.set_xlabel('Load Factor')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Objective Value Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot violations
    ax2 = axs[1]
    ax2.semilogy(load_factors, total_violations, 'o-', label='Total Violations')
    ax2.semilogy(load_factors, power_balance_violations, 's-', label='Power Balance Violations')
    ax2.set_xlabel('Load Factor')
    ax2.set_ylabel('Violation Magnitude (log scale)')
    ax2.set_title('Constraint Violations')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('opf_test_results.png')
    print("\nResults plot saved to 'opf_test_results.png'")
    
    # Close the plot
    plt.close(fig)

if __name__ == "__main__":
    # Get case name from command line if provided
    case_name = sys.argv[1] if len(sys.argv) > 1 else 'case5'
    
    # Run test
    test_opf_optimizer(case_name) 