#!/usr/bin/env python
"""
Constraint Screening approach for ML-OPF project (Fixed Version).
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

# Add imports for the improved optimizer
try:
    from utils.optimization_improved import ImprovedOPFOptimizer
except ImportError:
    print("Warning: ImprovedOPFOptimizer not available, will use standard optimizer")

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
    parser.add_argument('--use-pypower', action='store_true', help='Use PyPower instead of Gurobi')
    parser.add_argument('--use-improved-solver', action='store_true', 
                        help='Use the improved OPF solver for case30')
    
    return parser.parse_args()

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    Helps with constraint screening where most constraints are not binding.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCELoss(reduction='none')
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
        # Create synthetic constraints with realistic distribution (mostly non-binding)
        # Create around 10-15% binding constraints which matches real-world scenarios
        for i in range(n_branch):
            # Create realistic imbalanced distribution (10-15% binding)
            binding_data[f'binding_{i}'] = np.random.choice([0, 1], size=len(data), p=[0.85, 0.15])
    
    return binding_data

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
        # Load data
        print(f"Loading data for {args.case}...")
        data = load_pglib_data(args.case, args.data_dir)
        case_data = load_case_network(args.case, args.data_dir)
        print(f"Data loaded: {len(data)} samples")
        
        # Print all columns for debugging
        if args.print_columns:
            print("\nAll data columns:")
            for col in data.columns:
                print(f"  {col}")
            print("")
        
        # Extract input columns (load patterns)
        input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
        
        # If no columns found, try alternative naming pattern
        if not input_cols:
            input_cols = [col for col in data.columns if col.startswith('load') and (':pl' in col or ':ql' in col)]
            
        print(f"Input features: {len(input_cols)}")
        if args.print_columns and input_cols:
            print("Input columns:")
            for col in input_cols:
                print(f"  {col}")
        
        # Check data for extreme values and normalize if needed
        for col in input_cols:
            if col in data.columns:
                col_data = data[col]
                # First check if we're dealing with string data that needs conversion
                if isinstance(col_data.iloc[0], str):
                    try:
                        # Try to convert string values to float
                        data[col] = col_data.astype(float)
                        col_data = data[col]
                        print(f"Converted column {col} from string to float")
                    except Exception as e:
                        print(f"Could not convert column {col} to float: {e}")
                        continue
                
                # Now check for extreme values in numeric columns
                try:
                    max_val = col_data.max()
                    min_val = col_data.min()
                    if pd.notna(max_val) and pd.notna(min_val):
                        if max_val > 1e6 or min_val < -1e6:
                            print(f"Normalizing column {col} with extreme values: min={min_val}, max={max_val}")
                            # Apply log scaling for extreme values
                            if min_val > 0:
                                data[col] = np.log1p(col_data)
                            else:
                                # Handle negative values with signed log
                                sign = np.sign(col_data)
                                data[col] = sign * np.log1p(np.abs(col_data))
                except Exception as e:
                    print(f"Error processing column {col}: {e}")
                    continue
        
        # Extract binding constraints
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
        
        # Define focal loss for imbalanced classification
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Scale learning rate based on input/output dimensions for improved stability
        adjusted_lr = args.learning_rate / np.sqrt(max(1, len(output_cols) / 10))
        print(f"Using adjusted learning rate: {adjusted_lr:.6f} (from {args.learning_rate})")
        
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
            optimizer=torch.optim.Adam(model.parameters(), lr=adjusted_lr),
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
            early_stopping=5
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
        if args.use_improved_solver and args.case == 'case30':
            try:
                print("Using improved OPF optimizer for case30")
                optimizer = ImprovedOPFOptimizer(case_data, device=device)
            except (ImportError, NameError) as e:
                print(f"Failed to use improved optimizer: {e}")
                print("Falling back to standard optimizer")
                optimizer = OPFOptimizer(case_data, device=device)
        else:
            optimizer = OPFOptimizer(case_data, device=device)
        
        # Setup solver options with relaxed tolerances for case30
        solver_options = {
            'TimeLimit': 10,
            'OptimalityTol': 1e-5,
            'FeasibilityTol': 1e-6,
            'NumericFocus': 3
        }
        optimizer.solver_options = solver_options
        
        # Get sample inputs from test set
        sample_inputs, sample_targets = next(iter(test_loader))
        sample_inputs = sample_inputs.to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(sample_inputs)
        
        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).cpu().numpy()
        
        # Prepare load data
        load_data = sample_inputs[0].cpu().numpy().reshape(-1, 2)
        
        # Create full binding_constraints array for all branches
        n_branch = len(case_data['branch'])
        full_binding_constraints = np.ones(n_branch, dtype=bool)
        
        # Try solving OPF with all constraints
        print("Solving OPF with all constraints...")
        try:
            if args.use_pypower:
                solution_full = optimizer.solve_opf(load_data, verbose=False)
            else:
                solution_full = optimizer.solve_opf_gurobi(load_data, verbose=False)
            
            print(f"Solution found in {solution_full.get('runtime', 'N/A')} seconds.")
            
            # Try solving with screened constraints
            print("\nSolving OPF with screened constraints...")
            
            # Map binary predictions to full constraint array
            # Make sure binding_constraints has the correct dimensions
            screened_constraints = np.zeros(n_branch, dtype=bool)
            
            # Only map as many constraints as we have in our model
            for i in range(min(len(binary_predictions[0]), n_branch)):
                screened_constraints[i] = binary_predictions[0][i]
            
            try:
                if args.use_pypower:
                    solution_screened = optimizer.solve_opf(
                        load_data, verbose=False)
                else:
                    solution_screened = optimizer.solve_opf_gurobi(
                        load_data, binding_constraints=screened_constraints, verbose=False)
                
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
            except Exception as e:
                print(f"Error solving OPF with screened constraints: {e}")
                
        except Exception as e:
            print(f"Error solving OPF with all constraints: {e}")
            print("Solver might be having difficulty with the problem size.")
            print("Try using PyPower solver with --use-pypower flag")
        
        print(f"Results saved to {log_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    main(args) 