#!/usr/bin/env python
"""
Constraint Screening approach for ML-OPF project.
This script demonstrates how to use ML to predict binding constraints.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_pglib_data, load_case_network, prepare_data_loaders
from utils.optimization import OPFOptimizer
from utils.training import Trainer
from models.feedforward import ConstraintScreeningNN
from pypower.case39 import case39

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Constraint Screening')
    parser.add_argument('--case', type=str, default='case30', help='Case name (default: case30)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='64,128,64', 
                        help='Hidden dimensions (comma-separated)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--print-columns', action='store_true', help='Print column names for debugging')
    parser.add_argument('--early-stopping', type=int, default=5, help='Early stopping patience (default: 5)')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    
    return parser.parse_args()

def extract_binding_constraints(data, case_data):
    """
    Extract binding constraints from data.
    This is a simplified example that treats branch flows near their limits as binding.
    In a real implementation, this would analyze solver outputs for active constraints.
    
    Args:
        data: DataFrame with OPF solution data
        case_data: PyPOWER case data
    
    Returns:
        DataFrame with binary indicators for binding constraints
    """
    # Number of branches
    n_branch = len(case_data['branch'])
    
    # Extract branch flow columns with different naming patterns
    branch_cols = [col for col in data.columns if col.startswith('branch_flow')]
    
    # Try alternative naming pattern if needed
    if not branch_cols:
        branch_cols = [col for col in data.columns if col.startswith('branch') and ':pf' in col]
    
    # Create empty binding constraints DataFrame
    binding_data = pd.DataFrame()
    
    # Get branch flow limits
    branch_limits = case_data['branch'][:, 5] / case_data['baseMVA']  # RATE_A column
    
    # Determine binding constraints (simplified example)
    has_binding = False
    for i, col in enumerate(branch_cols):
        if i < n_branch:
            # Consider a constraint binding if flow is within 5% of the limit
            limit = branch_limits[i]
            if limit > 0:  # Only consider branches with limits
                is_binding = (data[col].abs() > 0.95 * limit).astype(int)
                if is_binding.sum() > 0:  # If we have at least one binding constraint
                    binding_data[f'binding_{i}'] = is_binding
                    has_binding = True
                else:
                    binding_data[f'binding_{i}'] = 0
            else:
                binding_data[f'binding_{i}'] = 0
    
    # If no binding constraints are found, create synthetic ones for demonstration
    if not has_binding or binding_data.empty:
        print("No real binding constraints detected. Creating synthetic constraints for demonstration.")
        # Use the first two branches as synthetic constraints (50% chance of being binding)
        for i in range(min(2, n_branch)):
            binding_data[f'synthetic_{i}'] = np.random.randint(0, 2, size=len(data))
    
    return binding_data

# Add a new function to load realistic data
def load_realistic_case39_data(data_dir):
    """
    Load realistic case39 data from CSV file.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        DataFrame with realistic case39 data
    """
    # Construct path to the realistic data file
    file_path = os.path.join('data/case39/realistic/realistic_case39_1000.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Realistic case39 data file not found: {file_path}")
    
    # Load the data
    print(f"Loading realistic case39 data from {file_path}")
    data = pd.read_csv(file_path)
    
    # Print data info
    print(f"Loaded {len(data)} samples with {len(data.columns)} features")
    
    return data

# Add function to load built-in case39 data
def load_builtin_case39():
    """
    Load built-in case39 data from PyPOWER.
    
    Returns:
        PyPOWER case data for case39
    """
    print("Loading built-in case39 data from PyPOWER")
    # Get the case data from PyPOWER
    case_data = case39()
    print(f"Loaded case39 with {len(case_data['bus'])} buses and {len(case_data['branch'])} branches")
    return case_data

def main(args):
    """Main function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"constraint_screening_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Load data based on case
        print(f"Loading data for {args.case}...")
        
        # Use realistic data for case39, pglib data for other cases
        if args.case == 'case39':
            data = load_realistic_case39_data(args.data_dir)
            # Use built-in case39 data instead of PGLib
            case_data = load_builtin_case39()
        else:
            data = load_pglib_data(args.case, args.data_dir)
            case_data = load_case_network(args.case, args.data_dir)
            
        print(f"Data loaded: {len(data)} samples")
        
        # Print all columns for debugging
        if args.print_columns:
            print("\nAll data columns:")
            for col in data.columns:
                print(f"  {col}")
            print("")
        
        # Extract input columns (load patterns) - handle both naming conventions
        input_cols = []
        
        # For realistic case39 data
        if args.case == 'case39':
            # Get load columns from realistic data (adjust based on actual column names)
            pg_cols = [col for col in data.columns if col.startswith('pg_')]
            vm_cols = [col for col in data.columns if col.startswith('vm_')]
            input_cols = pg_cols + vm_cols
        else:
            # Standard naming pattern for pglib data
            input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
            
            # Alternative naming pattern
            if not input_cols:
                input_cols = [col for col in data.columns if col.startswith('load') and (':pl' in col or ':ql' in col)]
            
        print(f"Input features: {len(input_cols)}")
        if args.print_columns and input_cols:
            print("Input columns:")
            for col in input_cols:
                print(f"  {col}")
        
        # Extract binding constraints - handle differently for realistic data
        if args.case == 'case39' and 'feasible' in data.columns:
            # For realistic data, we can use the feasibility flag
            print("Using feasibility information from realistic data")
            binding_data = pd.DataFrame()
            
            # Create synthetic binding constraints based on feasibility
            # We'll create 5 synthetic constraints for demonstration
            n_constraints = 5
            for i in range(n_constraints):
                # Initialize an array for binding constraints
                binding_constraints = np.zeros(len(data))
                
                # Loop through each row and set binding probability based on feasibility
                for j in range(len(data)):
                    # Higher probability of binding if solution is infeasible (feasible=0)
                    if data['feasible'].iloc[j] == 0:
                        binding_prob = 0.8  # 80% chance of binding if infeasible
                    else:
                        binding_prob = 0.2  # 20% chance of binding if feasible
                        
                    # Randomly determine if constraint is binding
                    binding_constraints[j] = np.random.binomial(1, binding_prob)
                
                # Add to binding data
                binding_data[f'binding_{i}'] = binding_constraints
            
            print(f"Created {n_constraints} synthetic binding constraints based on feasibility")
        else:
            # Extract binding constraints from solution data for other cases
            binding_data = extract_binding_constraints(data, case_data)
            
        output_cols = binding_data.columns.tolist()
        print(f"Output features (binding constraints): {len(output_cols)}")
        if args.print_columns and output_cols:
            print("Output columns:")
            for col in output_cols:
                print(f"  {col}")
        
        # Combine data
        combined_data = pd.concat([data[input_cols], binding_data], axis=1)
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            combined_data, input_cols, output_cols, batch_size=args.batch_size)
        
        # Parse hidden dimensions
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        
        # Initialize model
        model = ConstraintScreeningNN(
            input_dim=len(input_cols),
            num_constraints=len(output_cols),
            hidden_dims=hidden_dims
        )
        
        # Define binary cross-entropy loss
        criterion = torch.nn.BCELoss()
        
        # Define metrics for binary classification
        def precision_metric(pred, target):
            pred_bin = (pred > 0.5).float()
            return torch.tensor(precision_score(
                target.cpu().numpy(), pred_bin.cpu().numpy(), average='macro', zero_division=0))
        
        def recall_metric(pred, target):
            pred_bin = (pred > 0.5).float()
            return torch.tensor(recall_score(
                target.cpu().numpy(), pred_bin.cpu().numpy(), average='macro', zero_division=0))
        
        def f1_metric(pred, target):
            pred_bin = (pred > 0.5).float()
            return torch.tensor(f1_score(
                target.cpu().numpy(), pred_bin.cpu().numpy(), average='macro', zero_division=0))
        
        metrics = {
            'precision': precision_metric,
            'recall': recall_metric,
            'f1': f1_metric
        }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            criterion=criterion,
            device=device,
            log_dir=log_dir
        )
        
        # Train model
        print(f"Training constraint screening model for {args.epochs} epochs...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            metrics=metrics,
            save_best=True,
            early_stopping=args.early_stopping
        )
        
        # Plot training history
        trainer.plot_history(save_path=os.path.join(log_dir, 'training_history.png'))
        
        # Load best model and evaluate on test set
        trainer.load_best_model()
        test_loss, test_metrics = trainer.validate(test_loader, metrics)
        print(f"Test Loss: {test_loss:.6f}")
        for name, value in test_metrics.items():
            print(f"Test {name}: {value:.6f}")
        
        # Make predictions on test set
        predictions, targets = trainer.predict(test_loader)
        
        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics per constraint
        per_constraint_metrics = []
        for i in range(len(output_cols)):
            constraint_precision = precision_score(
                targets[:, i], binary_predictions[:, i], zero_division=0)
            constraint_recall = recall_score(
                targets[:, i], binary_predictions[:, i], zero_division=0)
            constraint_f1 = f1_score(
                targets[:, i], binary_predictions[:, i], zero_division=0)
            constraint_binding_rate = targets[:, i].mean()
            
            per_constraint_metrics.append({
                'constraint': output_cols[i],
                'precision': constraint_precision,
                'recall': constraint_recall,
                'f1': constraint_f1,
                'binding_rate': constraint_binding_rate
            })
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(per_constraint_metrics)
        metrics_df.to_csv(os.path.join(log_dir, 'per_constraint_metrics.csv'), index=False)
        
        # Create example of constraint screening for OPF
        print("\nDemonstrating constraint screening for OPF...")
        
        # Initialize optimizer
        optimizer = OPFOptimizer(case_data, device=device)
        
        # Get sample inputs from test set
        sample_inputs, sample_targets = next(iter(test_loader))
        sample_inputs = sample_inputs.to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(sample_inputs)
        
        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).cpu().numpy()
        
        # Prepare load data for the realistic case39 format
        if args.case == 'case39':
            # For realistic case39, we need to extract generator setpoints from the input features
            # and convert them to a format suitable for the OPF solver
            
            # Extract pg and vm values from sample inputs
            n_inputs = sample_inputs[0].shape[0]
            n_buses = 39  # Total buses in case39
            n_gens = 10   # Number of generators in case39
            
            # Create dummy load data with zeros
            load_data = [(0.0, 0.0) for _ in range(n_buses)]
            
            # Set a small default load at each bus to avoid numerical issues
            for i in range(n_buses):
                load_data[i] = (0.2, 0.1)  # Default active and reactive power
            
            print(f"Prepared load data for case39 OPF solver with {len(load_data)} buses")
        else:
            # For other cases, reshape to pairs of [p, q]
            load_data = sample_inputs[0].cpu().numpy().reshape(-1, 2)
        
        # Solve OPF with all constraints
        print("Solving OPF with all constraints...")
        solution_full = optimizer.solve_opf_gurobi(load_data, verbose=False)
        print(f"Solution found in {solution_full.get('runtime', 'N/A')} seconds.")
        
        # Solve OPF with screened constraints
        print("\nSolving OPF with screened constraints...")
        binding_constraints = binary_predictions[0]
        solution_screened = optimizer.solve_opf_gurobi(
            load_data, binding_constraints=binding_constraints, verbose=False)
        print(f"Solution found in {solution_screened.get('runtime', 'N/A')} seconds.")
        
        # Compute speedup
        if solution_full.get('runtime') and solution_screened.get('runtime'):
            speedup = solution_full['runtime'] / solution_screened['runtime']
            print(f"\nConstraint screening speedup: {speedup:.2f}x")
            
            # Compare objective values
            if solution_full.get('f') and solution_screened.get('f'):
                obj_diff = abs(solution_full['f'] - solution_screened['f']) / solution_full['f'] * 100
                print(f"Objective value difference: {obj_diff:.6f}%")
                
                with open(os.path.join(log_dir, 'speedup_results.txt'), 'w') as f:
                    f.write(f"Full OPF runtime: {solution_full['runtime']:.6f} seconds\n")
                    f.write(f"Screened OPF runtime: {solution_screened['runtime']:.6f} seconds\n")
                    f.write(f"Speedup: {speedup:.2f}x\n")
                    f.write(f"Objective value difference: {obj_diff:.6f}%\n")
        
        # Save model if requested
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(log_dir, 'constraint_screening_model.pt'))
            print(f"Model saved to {os.path.join(log_dir, 'constraint_screening_model.pt')}")
        
        print(f"Results saved to {log_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    main(args) 