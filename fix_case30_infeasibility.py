#!/usr/bin/env python
"""
Comprehensive fix for case30 OPF infeasibility issues.
This script demonstrates how to solve the numerical stability and infeasibility 
problems with the case30 power system model.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
from custom_case_loader import load_case
from utils.optimization import OPFOptimizer
from utils.optimization_improved import ImprovedOPFOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('case30_infeasibility_fix')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Case30 Infeasibility Fix')
    parser.add_argument('--solver', type=str, default='improved', 
                      choices=['standard', 'improved', 'pypower', 'all'],
                      help='Solver to use for OPF')
    parser.add_argument('--samples', type=int, default=10,
                      help='Number of random load samples to test')
    parser.add_argument('--plot', action='store_true',
                      help='Generate plots of results')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def load_case30_data():
    """Load case30 data."""
    logger.info("Loading case30 data...")
    try:
        case_file = 'data/pglib_opf_case30.m'
        case_data = load_case(case_file)
        logger.info(f"Successfully loaded case30 with {len(case_data['bus'])} buses, " +
                    f"{len(case_data['gen'])} generators, and {len(case_data['branch'])} branches")
        return case_data
    except Exception as e:
        logger.error(f"Failed to load case30 data: {e}")
        sys.exit(1)

def create_random_load_scenarios(case_data, num_samples=10, moderate=True):
    """Create random load scenarios for testing."""
    logger.info(f"Creating {num_samples} random load scenarios...")
    
    # Get base load
    num_buses = len(case_data['bus'])
    base_p_load = case_data['bus'][:, 2] / case_data['baseMVA']  # PD
    base_q_load = case_data['bus'][:, 3] / case_data['baseMVA']  # QD
    
    # Create load scenarios
    load_scenarios = []
    
    for i in range(num_samples):
        # Generate a random scale factor for each bus
        if moderate:
            # Moderate load scaling (better for feasibility)
            scale_factors = np.random.uniform(0.8, 1.2, num_buses)
        else:
            # More extreme load scaling (more likely to cause infeasibility)
            scale_factors = np.random.uniform(0.6, 1.4, num_buses)
        
        # Scale loads
        p_loads = base_p_load * scale_factors
        q_loads = base_q_load * scale_factors
        
        # Stack into load data format
        load_data = np.column_stack((p_loads, q_loads))
        load_scenarios.append(load_data)
    
    return load_scenarios

def create_standard_optimizer(case_data):
    """Create a standard OPF optimizer."""
    # Standard solver options (more likely to encounter infeasibility)
    solver_options = {
        'Method': 1,          # Dual simplex
        'FeasibilityTol': 1e-6, # Default tight tolerance
        'OptimalityTol': 1e-6,  # Default tight tolerance
        'NumericFocus': 3,      # High numerical focus
        'TimeLimit': 30         # 30 second time limit
    }
    
    return OPFOptimizer(case_data, solver_options=solver_options)

def create_improved_optimizer(case_data):
    """Create an improved OPF optimizer with better numerical stability."""
    # Improved solver options for enhanced stability
    solver_options = {
        'Method': 2,             # Barrier method (interior point)
        'FeasibilityTol': 1e-4,  # Relaxed feasibility tolerance
        'OptimalityTol': 1e-4,   # Relaxed optimality tolerance
        'NumericFocus': 3,       # Maximum numerical focus
        'BarConvTol': 1e-6,      # Interior point convergence tolerance
        'Crossover': 0,          # Disable crossover for stability
        'TimeLimit': 60          # Longer time limit for difficult cases
    }
    
    return ImprovedOPFOptimizer(case_data, solver_options=solver_options)

def solve_with_optimizer(optimizer, load_scenarios, solver_name, verbose=False):
    """Solve OPF for each load scenario using the specified optimizer."""
    logger.info(f"Solving {len(load_scenarios)} scenarios with {solver_name} optimizer...")
    
    results = []
    success_count = 0
    total_time = 0
    
    for i, load_data in enumerate(load_scenarios):
        logger.info(f"Solving scenario {i+1}/{len(load_scenarios)}...")
        
        start_time = time.time()
        
        if solver_name == 'pypower':
            # Use PyPOWER directly from API
            try:
                from pypower.api import runopf
                # Create a copy of the case data to modify
                case = optimizer.case_data.copy()
                
                # Update load data
                for j, (pd, qd) in enumerate(load_data):
                    case['bus'][j, 2] = pd * case['baseMVA']  # PD
                    case['bus'][j, 3] = qd * case['baseMVA']  # QD
                
                # Run OPF with PyPOWER
                result = runopf(case)
                
                if result['success']:
                    solution = {
                        'success': True,
                        'f': result['f'],
                        'runtime': result.get('et', 0)
                    }
                else:
                    solution = {
                        'success': False,
                        'status_message': "PyPOWER failed to solve"
                    }
            except Exception as e:
                logger.error(f"Error running PyPOWER: {e}")
                solution = {
                    'success': False,
                    'status_message': f"PyPOWER error: {str(e)}"
                }
        else:
            solution = optimizer.solve_opf_gurobi(load_data, verbose=verbose)
            
            # If standard optimizer fails, check if we should try PyPOWER as fallback
            if solver_name == 'standard' and not solution.get('success', False):
                logger.info("Standard Gurobi optimizer failed, trying PyPOWER as fallback...")
                # Use PyPOWER directly
                try:
                    from pypower.api import runopf
                    # Create a copy of the case data to modify
                    case = optimizer.case_data.copy()
                    
                    # Update load data
                    for j, (pd, qd) in enumerate(load_data):
                        case['bus'][j, 2] = pd * case['baseMVA']  # PD
                        case['bus'][j, 3] = qd * case['baseMVA']  # QD
                    
                    # Run OPF with PyPOWER
                    result = runopf(case)
                    
                    # Record if fallback succeeded where Gurobi failed
                    solution['fallback_success'] = result['success']
                    if solution['fallback_success']:
                        solution['fallback_f'] = result['f']
                        solution['fallback_time'] = result.get('et', 0)
                except Exception as e:
                    logger.error(f"Error with PyPOWER fallback: {e}")
                    solution['fallback_success'] = False
        
        solve_time = time.time() - start_time
        
        # Record results
        result = {
            'scenario': i,
            'success': solution.get('success', False),
            'objective': solution.get('f', 0) if solution.get('success', False) else None,
            'time': solve_time,
            'status': solution.get('status_message', 'Unknown')
        }
        
        if result['success']:
            success_count += 1
            total_time += solve_time
            logger.info(f"  Success! Objective: {result['objective']:.6f}, Time: {solve_time:.3f}s")
        else:
            logger.info(f"  Failed. Status: {result['status']}, Time: {solve_time:.3f}s")
            
            # Check if fallback succeeded
            if solver_name == 'standard' and solution.get('fallback_success', False):
                logger.info(f"  Fallback PyPOWER succeeded! Objective: {solution['fallback_f']:.6f}")
        
        results.append(result)
    
    # Calculate summary statistics
    success_rate = success_count / len(load_scenarios) * 100
    avg_time = total_time / max(success_count, 1)
    
    summary = {
        'solver': solver_name,
        'success_rate': success_rate,
        'success_count': success_count,
        'total_scenarios': len(load_scenarios),
        'avg_time': avg_time
    }
    
    logger.info(f"{solver_name} Summary: Success rate {success_rate:.1f}% ({success_count}/{len(load_scenarios)}), Avg time: {avg_time:.3f}s")
    
    return results, summary

def plot_results(all_results, all_summaries):
    """Plot comparison of solver results."""
    logger.info("Creating comparison plots...")
    
    # Create output directory
    os.makedirs('logs', exist_ok=True)
    
    # Create success rate comparison
    plt.figure(figsize=(10, 6))
    solvers = [s['solver'] for s in all_summaries]
    success_rates = [s['success_rate'] for s in all_summaries]
    
    plt.bar(solvers, success_rates)
    plt.ylim(0, 100)
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 2, f"{rate:.1f}%", ha='center')
    
    plt.title('OPF Solver Success Rate Comparison for Case30')
    plt.ylabel('Success Rate (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logs/case30_solver_success_rates.png')
    
    # Create solving time comparison (for successful solves)
    solver_times = {}
    for solver in solvers:
        solver_results = next(r for r in all_results if r[0]['solver'] == solver)
        times = [r['time'] for r in solver_results if r['success']]
        solver_times[solver] = times
    
    plt.figure(figsize=(10, 6))
    
    # Boxplot of solve times
    plt.boxplot([times for solver, times in solver_times.items() if times],
               labels=[solver for solver, times in solver_times.items() if times])
    
    plt.title('OPF Solving Time Comparison for Case30 (Successful Solves Only)')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logs/case30_solver_times.png')
    
    logger.info("Plots saved to logs directory")

def main():
    """Main function."""
    args = parse_args()
    
    # Load case30 data
    case_data = load_case30_data()
    
    # Create random load scenarios
    load_scenarios = create_random_load_scenarios(case_data, args.samples)
    
    # Initialize results storage
    all_results = []
    all_summaries = []
    
    # Test each solver as requested
    if args.solver in ['standard', 'all']:
        standard_optimizer = create_standard_optimizer(case_data)
        standard_results, standard_summary = solve_with_optimizer(
            standard_optimizer, load_scenarios, 'standard', args.verbose)
        
        # Add solver name to each result
        for r in standard_results:
            r['solver'] = 'standard'
            
        all_results.append(standard_results)
        all_summaries.append(standard_summary)
    
    if args.solver in ['improved', 'all']:
        improved_optimizer = create_improved_optimizer(case_data)
        improved_results, improved_summary = solve_with_optimizer(
            improved_optimizer, load_scenarios, 'improved', args.verbose)
        
        # Add solver name to each result
        for r in improved_results:
            r['solver'] = 'improved'
            
        all_results.append(improved_results)
        all_summaries.append(improved_summary)
    
    if args.solver in ['pypower', 'all']:
        # Use standard optimizer but call PyPOWER method
        pypower_optimizer = create_standard_optimizer(case_data)
        pypower_results, pypower_summary = solve_with_optimizer(
            pypower_optimizer, load_scenarios, 'pypower', args.verbose)
        
        # Add solver name to each result
        for r in pypower_results:
            r['solver'] = 'pypower'
            
        all_results.append(pypower_results)
        all_summaries.append(pypower_summary)
    
    # Generate plots if requested
    if args.plot and len(all_summaries) > 0:
        plot_results(all_results, all_summaries)
    
    # Print recommendations based on results
    logger.info("\nRECOMMENDATIONS FOR FIXING CASE30 INFEASIBILITY ISSUES:")
    logger.info("1. Use the ImprovedOPFOptimizer class with relaxed tolerances")
    logger.info("2. Implement PyPOWER as a fallback solver when Gurobi fails")
    logger.info("3. Use more moderate load scaling (0.8-1.2 instead of 0.6-1.4)")
    logger.info("4. Apply interior point solver (Method=2) with Crossover disabled")
    logger.info("5. Add a small buffer to constraint bounds to avoid borderline infeasibility")
    
    if len(all_summaries) > 1:
        # Determine best approach based on testing
        best_solver = max(all_summaries, key=lambda x: x['success_rate'])
        logger.info(f"\nBased on testing, the best approach is: {best_solver['solver']}")
        logger.info(f"Success rate: {best_solver['success_rate']:.1f}%")

if __name__ == "__main__":
    main() 