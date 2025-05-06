#!/usr/bin/env python
"""
Main entry point for ML-AC-OPF project.

This script provides a user-friendly interface to run models for different power system cases,
with a focus on case39 (New England) which has better numerical properties than case30.
"""

import os
import argparse
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml_opf')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-AC-OPF: Machine Learning for Optimal Power Flow')
    
    # Main command options
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Case30 command
    case30_parser = subparsers.add_parser('case30', help='Run case30 model')
    case30_parser.add_argument('--num-samples', type=int, default=100,
                              help='Number of samples to generate (default: 100)')
    case30_parser.add_argument('--load-range', type=float, nargs=2, default=[0.9, 1.1],
                              help='Load scaling range (default: 0.9 1.1)')
    case30_parser.add_argument('--hidden-dim', type=int, default=64,
                              help='Hidden layer dimension (default: 64)')
    case30_parser.add_argument('--num-layers', type=int, default=3,
                              help='Number of layers (default: 3)')
    case30_parser.add_argument('--dropout', type=float, default=0.1,
                              help='Dropout rate (default: 0.1)')
    case30_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size (default: 32)')
    case30_parser.add_argument('--epochs', type=int, default=100,
                              help='Number of epochs (default: 100)')
    case30_parser.add_argument('--lr', type=float, default=0.001,
                              help='Learning rate (default: 0.001)')
    case30_parser.add_argument('--use-gurobi', action='store_true',
                              help='Use Gurobi solver for validation')
    
    # Case39 command
    case39_parser = subparsers.add_parser('case39', help='Run case39 model')
    case39_parser.add_argument('--num-samples', type=int, default=100,
                              help='Number of samples to generate (default: 100)')
    case39_parser.add_argument('--load-range', type=float, nargs=2, default=[0.9, 1.1],
                              help='Load scaling range (default: 0.9 1.1)')
    case39_parser.add_argument('--hidden-dim', type=int, default=64,
                              help='Hidden layer dimension (default: 64)')
    case39_parser.add_argument('--num-layers', type=int, default=3,
                              help='Number of layers (default: 3)')
    case39_parser.add_argument('--dropout', type=float, default=0.1,
                              help='Dropout rate (default: 0.1)')
    case39_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size (default: 32)')
    case39_parser.add_argument('--epochs', type=int, default=100,
                              help='Number of epochs (default: 100)')
    case39_parser.add_argument('--lr', type=float, default=0.001,
                              help='Learning rate (default: 0.001)')
    case39_parser.add_argument('--use-gurobi', action='store_true',
                              help='Use Gurobi solver for validation')
    
    # Constraint screening command for realistic case39
    constraint_parser = subparsers.add_parser('constraint-screening', help='Run constraint screening on realistic case39 data')
    constraint_parser.add_argument('--data-dir', type=str, default='data/case39/processed/ml_data',
                              help='Data directory with processed IEEE data (default: data/case39/processed/ml_data)')
    constraint_parser.add_argument('--hidden-dims', type=str, default='64,128,64',
                              help='Hidden dimensions (comma-separated) (default: 64,128,64)')
    constraint_parser.add_argument('--batch-size', type=int, default=64,
                              help='Batch size (default: 64)')
    constraint_parser.add_argument('--epochs', type=int, default=50,
                              help='Number of epochs (default: 50)')
    constraint_parser.add_argument('--learning-rate', type=float, default=0.001,
                              help='Learning rate (default: 0.001)')
    constraint_parser.add_argument('--gpu', action='store_true',
                              help='Use GPU if available')
    constraint_parser.add_argument('--log-dir', type=str, default='logs/constraint_screening',
                              help='Log directory (default: logs/constraint_screening)')
    constraint_parser.add_argument('--early-stopping', type=int, default=10,
                              help='Early stopping patience (default: 10)')
    constraint_parser.add_argument('--save-model', action='store_true',
                              help='Save trained models and results')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup and test environment')
    setup_parser.add_argument('--test-gurobi', action='store_true',
                             help='Test Gurobi installation')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Run comparison between case30 and case39')
    compare_parser.add_argument('--num-samples', type=int, default=10,
                               help='Number of samples to compare (default: 10)')
    compare_parser.add_argument('--load-range', type=float, nargs=2, default=[0.9, 1.1],
                               help='Load scaling range (default: 0.9 1.1)')
    compare_parser.add_argument('--use-gurobi', action='store_true',
                               help='Use Gurobi solver for comparison')
    compare_parser.add_argument('--plot', action='store_true',
                               help='Generate plots of results')
    compare_parser.add_argument('--output-dir', type=str, default='output',
                               help='Output directory for results (default: output)')
    
    return parser.parse_args()

def run_setup(args):
    """Run setup tasks."""
    logger.info("Running setup tasks...")
    
    if args.test_gurobi:
        logger.info("Testing Gurobi solver...")
        try:
            logger.info("Creating test model...")
            import gurobipy as gp
            from gurobipy import GRB
            
            # Create a simple model
            model = gp.Model()
            
            # Add variables
            x = model.addVar(lb=0, name="x")
            y = model.addVar(lb=0, name="y")
            
            # Set objective
            model.setObjective(x + y, GRB.MINIMIZE)
            
            # Add constraints
            model.addConstr(2 * x + y >= 5, "c0")
            model.addConstr(x + 2 * y >= 5, "c1")
            
            # Optimize model
            model.optimize()
            
            # Check if optimal solution found
            if model.status == GRB.OPTIMAL:
                logger.info("Gurobi solver test successful!")
                logger.info(f"Optimal solution found: x={x.x}, y={y.x}, Objective={model.objVal}")
            else:
                logger.warning("Gurobi solver test failed. Status: %s", model.status)
        except Exception as e:
            logger.error(f"Error testing Gurobi: {e}")
            return False
        
        # Check Gurobi license details
        try:
            env = gp.Env()
            logger.info(f"Gurobi license type: {env.getParamInfo('LicenseType')}")
            logger.info(f"Gurobi license expiration: {env.getParamInfo('LicenseExpiration')}")
        except:
            logger.info("Could not retrieve license details")
    
    logger.info("Setup completed!")
    return True

def run_case30(args):
    """Run case30 model."""
    logger.info("Running case30 model...")
    
    # Prepare the command to run the case30 script
    cmd = [
        "python", "cases/case30/run.py",
        "--num-samples", str(args.num_samples),
        "--load-range", str(args.load_range[0]), str(args.load_range[1]),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr)
    ]
    
    if args.use_gurobi:
        cmd.append("--use-gurobi")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running case30 model: {e}")
        return e

def run_case39(args):
    """Run case39 model."""
    logger.info("Running case39 model...")
    
    # Prepare the command to run the case39 script
    cmd = [
        "python", "cases/case39/run.py",
        "--num-samples", str(args.num_samples),
        "--load-range", str(args.load_range[0]), str(args.load_range[1]),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr)
    ]
    
    if args.use_gurobi:
        cmd.append("--use-gurobi")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running case39 model: {e}")
        return e

def run_constraint_screening(args):
    """Run constraint screening on realistic case39 data."""
    logger.info("Running constraint screening on realistic case39 data...")
    
    # Check if the data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please run integrate_realistic_case39.py to process the IEEE data first.")
        return False
    
    # Prepare the command to run the constraint screening script
    cmd = [
        "python", "scripts/constraint_screening.py",
        "--case", "case39",
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-dims", args.hidden_dims,
        "--log-dir", args.log_dir
    ]
    
    if args.gpu:
        cmd.append("--gpu")
    
    if args.save_model:
        cmd.append("--save-model")
    
    if args.early_stopping > 0:
        cmd.append("--early-stopping")
        cmd.append(str(args.early_stopping))
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running constraint screening: {e}")
        return e

def run_compare(args):
    """Run comparison between case30 and case39."""
    logger.info("Running case comparison...")
    
    # Prepare the command to run the comparison script
    cmd = [
        "python", "run_case_comparison.py",
        "--num-samples", str(args.num_samples),
        "--load-range", str(args.load_range[0]), str(args.load_range[1]),
        "--output-dir", args.output_dir
    ]
    
    if args.use_gurobi:
        cmd.append("--use-gurobi")
    
    if args.plot:
        cmd.append("--plot")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running comparison: {e}")
        return e

def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'setup':
        return run_setup(args)
    elif args.command == 'case30':
        return run_case30(args)
    elif args.command == 'case39':
        return run_case39(args)
    elif args.command == 'constraint-screening':
        return run_constraint_screening(args)
    elif args.command == 'compare':
        return run_compare(args)
    else:
        logger.error("No command specified. Use --help for available commands.")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 