#!/usr/bin/env python
"""
Test script for case30 with local Gurobi solver.
This is a simple test to verify the setup works correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
import argparse

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_case30')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Case30 with Local Gurobi')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--solver', choices=['gurobi', 'pypower'], default='gurobi',
                        help='Solver to use (default: gurobi)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of test samples (default: 5)')
    return parser.parse_args()

def test_gurobi():
    """Test if Gurobi is working."""
    logger.info("Testing Gurobi installation...")
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # Create test model
        m = gp.Model("test")
        x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
        y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
        m.setObjective(x + y, GRB.MAXIMIZE)
        m.addConstr(x + 2 * y <= 10, "c0")
        m.addConstr(x <= 5, "c1")
        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            logger.info(f"Gurobi test successful: x={x.X:.6f}, y={y.X:.6f}, obj={m.ObjVal:.6f}")
            return True
        else:
            logger.error(f"Gurobi test failed with status {m.Status}")
            return False
    except ImportError:
        logger.error("Gurobi not installed. Please install gurobipy.")
        return False
    except Exception as e:
        logger.error(f"Gurobi test failed: {str(e)}")
        return False

def test_case30_load():
    """Test loading case30 data."""
    logger.info("Testing case30 data loading...")
    data_dir = 'data'
    m_file = os.path.join(data_dir, 'pglib_opf_case30.m')
    csv_file = os.path.join(data_dir, 'pglib_opf_case30.csv')
    
    if not os.path.exists(m_file):
        logger.error(f"Case30 .m file not found: {m_file}")
        logger.info("Please run prepare_case30.py first")
        return False
    
    if not os.path.exists(csv_file):
        logger.error(f"Case30 CSV file not found: {csv_file}")
        logger.info("Please run prepare_case30.py first")
        return False
    
    # Try to load case data
    try:
        from custom_case_loader import load_case
        case_data = load_case(m_file)
        num_buses = len(case_data['bus'])
        num_gens = len(case_data['gen'])
        num_branches = len(case_data['branch'])
        logger.info(f"Successfully loaded case30 data: {num_buses} buses, " 
                   f"{num_gens} generators, {num_branches} branches")
        
        # Try to load CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully loaded case30 CSV data: {len(df)} samples, {len(df.columns)} columns")
        
        return True
    except Exception as e:
        logger.error(f"Error loading case30 data: {str(e)}")
        return False

def test_case30_opf(solver='gurobi', num_samples=5, verbose=False):
    """Test running OPF for case30."""
    logger.info(f"Testing case30 OPF solution with {solver} solver...")
    
    try:
        # Load case data
        from custom_case_loader import load_case
        case_data = load_case('data/pglib_opf_case30.m')
        
        # Create optimizer
        from utils.optimization import OPFOptimizer
        optimizer = OPFOptimizer(case_data)
        
        # Get base load
        num_buses = len(case_data['bus'])
        base_p_load = case_data['bus'][:, 2] / case_data['baseMVA']  # PD
        base_q_load = case_data['bus'][:, 3] / case_data['baseMVA']  # QD
        
        # Run OPF for different load scenarios
        results = []
        
        for i in range(num_samples):
            logger.info(f"Running OPF for sample {i+1}/{num_samples}...")
            
            # Create random load variation
            load_scale = 0.8 + 0.4 * np.random.random()  # Between 0.8 and 1.2
            p_loads = base_p_load * load_scale
            q_loads = base_q_load * load_scale
            
            # Create load data
            load_data = np.column_stack((p_loads, q_loads))
            
            # Solve OPF
            start_time = time.time()
            if solver == 'gurobi':
                solution = optimizer.solve_opf_gurobi(load_data, verbose=verbose)
            else:
                solution = optimizer.solve_opf(load_data, verbose=verbose)
            
            solve_time = time.time() - start_time
            
            # Store results
            if solution['success']:
                cost = solution.get('f', 0)
                logger.info(f"OPF solved successfully in {solve_time:.4f} seconds, cost = {cost:.6f}")
                results.append((True, solve_time, cost))
            else:
                error = solution.get('error', 'Unknown error')
                logger.warning(f"OPF solution failed in {solve_time:.4f} seconds: {error}")
                results.append((False, solve_time, 0))
        
        # Report results
        success_count = sum(1 for r in results if r[0])
        avg_time = np.mean([r[1] for r in results])
        logger.info(f"OPF test results: {success_count}/{num_samples} successful, "
                   f"average time: {avg_time:.4f} seconds")
        
        # Create a simple bar chart of solve times
        plt.figure(figsize=(10, 6))
        times = [r[1] for r in results]
        colors = ['green' if r[0] else 'red' for r in results]
        plt.bar(range(1, num_samples+1), times, color=colors)
        plt.xlabel('Sample')
        plt.ylabel('Solve Time (s)')
        plt.title(f'Case30 OPF Solve Times ({solver.capitalize()} Solver)')
        plt.xticks(range(1, num_samples+1))
        plt.tight_layout()
        plt.savefig(f'case30_opf_times_{solver}.png')
        logger.info(f"Saved solve time chart to case30_opf_times_{solver}.png")
        
        return success_count == num_samples
        
    except Exception as e:
        logger.error(f"Error testing case30 OPF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    """Main function."""
    logger.info("Running case30 local tests...")
    
    # Test Gurobi
    gurobi_ok = test_gurobi()
    if not gurobi_ok and args.solver == 'gurobi':
        logger.error("Gurobi test failed, cannot continue with Gurobi solver")
        sys.exit(1)
    
    # Test data loading
    data_ok = test_case30_load()
    if not data_ok:
        logger.error("Data test failed, cannot continue")
        sys.exit(1)
    
    # Test OPF
    opf_ok = test_case30_opf(solver=args.solver, 
                             num_samples=args.num_samples, 
                             verbose=args.verbose)
    
    # Report results
    logger.info("Test Results:")
    logger.info(f"  Gurobi: {'OK' if gurobi_ok else 'FAIL'}")
    logger.info(f"  Data Loading: {'OK' if data_ok else 'FAIL'}")
    logger.info(f"  OPF Solution ({args.solver}): {'OK' if opf_ok else 'FAIL'}")
    
    # Overall result
    if args.solver == 'gurobi':
        ok = gurobi_ok and data_ok and opf_ok
    else:
        ok = data_ok and opf_ok
    
    logger.info(f"Overall: {'OK' if ok else 'FAIL'}")
    
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)