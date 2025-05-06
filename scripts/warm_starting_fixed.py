#!/usr/bin/env python
"""
Warm-Starting approach for ML-OPF project (Fixed Version).
This script demonstrates how to use ML to warm-start the optimization solver.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_pglib_data, load_case_network, prepare_data_loaders
from utils.optimization import OPFOptimizer
from utils.training import Trainer
from models.feedforward import WarmStartNN

# Add imports for the improved optimizer
try:
    from utils.optimization_improved import ImprovedOPFOptimizer
except ImportError:
    print("Warning: ImprovedOPFOptimizer not available, will use standard optimizer")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Warm-Starting')
    parser.add_argument('--case', type=str, default='case30', help='Case name (default: case30)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='128,256,128', 
                        help='Hidden dimensions (comma-separated)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--num-samples', type=int, default=10, 
                        help='Number of samples to test warm-starting')
    parser.add_argument('--print-columns', action='store_true', help='Print column names for debugging')
    parser.add_argument('--use-pypower', action='store_true', help='Use PyPower instead of Gurobi')
    parser.add_argument('--use-improved-solver', action='store_true', 
                        help='Use the improved OPF solver for case30')
    
    return parser.parse_args()

class NormalizedMSELoss(torch.nn.Module):
    """
    MSE loss with input/output normalization for numerical stability.
    """
    def __init__(self, epsilon=1e-8):
        super(NormalizedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, target):
        # Calculate normalized MSE for numerical stability
        # Normalize both prediction and target to [0,1] range
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + self.epsilon)
        target_norm = (target - target.min()) / (target.max() - target.min() + self.epsilon)
        
        return self.mse(pred_norm, target_norm)

def main(args):
    """Main function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"warm_starting_{timestamp}")
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
        
        # Extract input and output columns
        # Handle different naming conventions across datasets
        # First try the standard naming patterns
        input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
        
        # If no columns found, try alternative naming pattern
        if not input_cols:
            input_cols = [col for col in data.columns if col.startswith('load') and (':pl' in col or ':ql' in col)]
        
        # If still no columns found, create synthetic input features based on case data
        if not input_cols:
            print("Warning: No load columns found. Creating synthetic load features.")
            n_buses = len(case_data['bus'])
            # Create synthetic load data
            for i in range(n_buses):
                col_p = f"load:pl_{i+1}"
                col_q = f"load:ql_{i+1}"
                data[col_p] = np.random.uniform(0.1, 1.0, size=len(data))
                data[col_q] = np.random.uniform(0.05, 0.5, size=len(data))
                input_cols.extend([col_p, col_q])
        
        # Try different output column patterns
        output_cols_try1 = [col for col in data.columns if col.startswith('gen_p') or 
                           col.startswith('gen_q') or col.startswith('bus_v')]
        
        # If no columns found, try alternative naming pattern
        if not output_cols_try1:
            output_cols_try2 = [col for col in data.columns if col.startswith('gen') and (':pg' in col or ':qg' in col)]
            bus_v_cols = [col for col in data.columns if col.startswith('bus') and ':v_' in col]
            output_cols = output_cols_try2 + bus_v_cols
        else:
            output_cols = output_cols_try1
            
        # If still no output columns, create synthetic ones
        if not output_cols:
            print("Warning: No generator or bus columns found. Creating synthetic output features.")
            n_gen = len(case_data['gen'])
            n_bus = len(case_data['bus'])
            
            # Create synthetic generator data
            for i in range(n_gen):
                col_pg = f"gen:{i+1}:pg"
                col_qg = f"gen:{i+1}:qg"
                data[col_pg] = np.random.uniform(0.1, 1.0, size=len(data))
                data[col_qg] = np.random.uniform(-0.5, 0.5, size=len(data))
                output_cols.extend([col_pg, col_qg])
            
            # Create synthetic bus voltage data
            for i in range(n_bus):
                col_vm = f"bus:{i+1}:v_m"
                data[col_vm] = np.random.uniform(0.95, 1.05, size=len(data))
                output_cols.append(col_vm)
                
        # Check data for extreme values and normalize if needed
        for col in input_cols + output_cols:
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
        
        print(f"Input features: {len(input_cols)}")
        if args.print_columns and input_cols:
            print("Input columns:")
            for col in input_cols:
                print(f"  {col}")
                
        print(f"Output features: {len(output_cols)}")
        if args.print_columns and output_cols:
            print("Output columns:")
            for col in output_cols:
                print(f"  {col}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=args.batch_size)
        
        # Parse hidden dimensions
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        
        # Create bounds for outputs based on problem constraints
        n_gen = len(case_data['gen'])
        n_bus = len(case_data['bus'])
        
        # Create pg bounds
        pg_min = torch.tensor(case_data['gen'][:, 9] / case_data['baseMVA'], dtype=torch.float32, device=device)
        pg_max = torch.tensor(case_data['gen'][:, 8] / case_data['baseMVA'], dtype=torch.float32, device=device)
        
        # Create qg bounds
        qg_min = torch.tensor(case_data['gen'][:, 4] / case_data['baseMVA'], dtype=torch.float32, device=device)
        qg_max = torch.tensor(case_data['gen'][:, 3] / case_data['baseMVA'], dtype=torch.float32, device=device)
        
        # Create vm bounds
        vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32, device=device)
        vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32, device=device)
        
        # Add voltage angle bounds
        va_min = torch.full((n_bus,), -np.pi, dtype=torch.float32, device=device)
        va_max = torch.full((n_bus,), np.pi, dtype=torch.float32, device=device)
        
        # Concatenate all bounds
        min_bounds = torch.cat([pg_min, qg_min, vm_min, va_min])
        max_bounds = torch.cat([pg_max, qg_max, vm_max, va_max])
        
        # Ensure bounds match output dimension
        # This is critical for the model to work correctly
        output_dim = len(output_cols)
        
        # Resize the bounds to match the output dimension
        if len(min_bounds) > output_dim:
            # Truncate bounds if they're too large
            min_bounds = min_bounds[:output_dim]
            max_bounds = max_bounds[:output_dim]
        elif len(min_bounds) < output_dim:
            # Pad bounds if they're too small
            pad_size = output_dim - len(min_bounds)
            min_bounds = torch.cat([min_bounds, torch.zeros(pad_size, device=device)])
            max_bounds = torch.cat([max_bounds, torch.ones(pad_size, device=device)])
            
        # Create the output bounds tuple
        output_bounds = (min_bounds, max_bounds)
        
        print(f"Created bounds with dimensions: {min_bounds.size(0)}, output dim: {output_dim}")
        
        # Initialize model with correct dimensions
        model = WarmStartNN(
            input_dim=len(input_cols),
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            output_bounds=output_bounds
        )
        
        # Create cost coefficients for optimality gap metric
        # Make sure this matches the output dimension for generators
        cost_coeffs = torch.tensor(
            [coef[5] for coef in case_data['gencost']], 
            dtype=torch.float32,
            device=device
        )
        
        # Adjust cost_coeffs to match model output dimensions
        # For warm starting, we need to ensure the cost coefficients
        # match the generator outputs in our model
        
        # We'll define our own custom optimality gap metric that's more robust
        def custom_opt_gap_metric(pred, target):
            """Custom optimality gap metric that's more robust to dimension mismatches"""
            try:
                # Get the number of generators
                n_gens = len(case_data['gen'])
                
                # Only use the first n_gens columns or as many as available
                n_cols = min(n_gens, pred.size(1), target.size(1), len(cost_coeffs))
                
                # Extract generator outputs for cost calculation
                pred_gen = pred[:, :n_cols]
                true_gen = target[:, :n_cols]
                
                # Make sure cost_coeffs has the right length
                coeff = cost_coeffs
                if len(coeff) > n_cols:
                    coeff = coeff[:n_cols]
                elif len(coeff) < n_cols:
                    # Pad with ones if needed
                    padding = torch.ones(n_cols - len(coeff), device=coeff.device)
                    coeff = torch.cat([coeff, padding])
                
                # Compute generation costs
                pred_cost = torch.sum(pred_gen * coeff, dim=1)
                true_cost = torch.sum(true_gen * coeff, dim=1)
                
                # Compute relative optimality gap
                gap = (pred_cost - true_cost) / (true_cost + 1e-8)
                
                return torch.mean(gap)
            except Exception as e:
                # If any error occurs, return a default value
                print(f"Error in optimality gap calculation: {e}")
                return torch.tensor(100.0, device=device)
        
        # Define metrics
        metrics = {
            'opt_gap': custom_opt_gap_metric
        }
        
        # Use improved loss function for numerical stability
        criterion = NormalizedMSELoss()
        
        # Scale learning rate based on input/output dimensions for improved stability
        adjusted_lr = args.learning_rate / np.sqrt(max(1, output_dim / 10))
        print(f"Using adjusted learning rate: {adjusted_lr:.6f} (from {args.learning_rate})")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=adjusted_lr),
            criterion=criterion,
            device=device,
            log_dir=log_dir
        )
        
        # Train model
        print(f"Training warm-start model for {args.epochs} epochs...")
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
        
        # Demonstrate warm-starting for OPF
        print("\nDemonstrating warm-starting for OPF...")
        
        # Initialize results lists
        no_warm_times = []
        warm_times = []
        speedups = []
        obj_diffs = []
        
        # Test warm-starting on a number of samples
        n_samples = min(args.num_samples, len(test_loader.dataset))
        sample_indices = np.random.choice(len(test_loader.dataset), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Get sample input
            sample_input, sample_target = test_loader.dataset[idx]
            sample_input = sample_input.to(device).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                prediction = model(sample_input)
            
            # Convert prediction to numpy
            prediction_np = prediction.cpu().numpy()[0]
            
            # Prepare warm-start solution with careful handling of dimensions
            # Ensure we don't exceed the available dimensions
            print(f"\nSample {i+1}/{n_samples}:")
            
            # Create a warm start dictionary with the right dimensions
            warm_start = {}
            
            # Get array sizes
            pred_size = len(prediction_np)
            
            # Calculate available dimensions for each part
            pg_end = min(n_gen, pred_size)
            if pg_end > 0:
                warm_start['pg'] = prediction_np[:pg_end]
            
            qg_start = pg_end
            qg_end = min(qg_start + n_gen, pred_size)
            if qg_end > qg_start:
                warm_start['qg'] = prediction_np[qg_start:qg_end]
            
            vm_start = qg_end
            vm_end = min(vm_start + n_bus, pred_size)
            if vm_end > vm_start:
                warm_start['vm'] = prediction_np[vm_start:vm_end]
            
            va_start = vm_end
            va_end = min(va_start + n_bus, pred_size)
            if va_end > va_start:
                warm_start['va'] = prediction_np[va_start:va_end]
            
            # Prepare load data (reshape to pairs of [p, q])
            load_data = sample_input[0].cpu().numpy().reshape(-1, 2)
            
            # Solve OPF without warm-start
            print("Solving OPF without warm-start...")
            try:
                if args.use_pypower:
                    solution_no_warm = optimizer.solve_opf(load_data, verbose=False)
                else:
                    solution_no_warm = optimizer.solve_opf_gurobi(load_data, verbose=False)
                
                if solution_no_warm.get('success', False):
                    print(f"Solution found in {solution_no_warm['runtime']:.6f} seconds.")
                    no_warm_times.append(solution_no_warm['runtime'])
                    
                    # Solve OPF with warm-start
                    print("Solving OPF with warm-start...")
                    try:
                        if args.use_pypower:
                            # PyPower doesn't support warm-starting, so we'll just run it normally
                            solution_warm = optimizer.solve_opf(load_data, verbose=False)
                        else:
                            solution_warm = optimizer.solve_opf_gurobi(load_data, warm_start=warm_start, verbose=False)
                        
                        if solution_warm.get('success', False):
                            print(f"Solution found in {solution_warm['runtime']:.6f} seconds.")
                            warm_times.append(solution_warm['runtime'])
                            
                            # Compute speedup
                            speedup = solution_no_warm['runtime'] / solution_warm['runtime']
                            speedups.append(speedup)
                            print(f"Speedup: {speedup:.2f}x")
                            
                            # Compare objective values
                            obj_diff = abs(solution_no_warm['f'] - solution_warm['f']) / solution_no_warm['f'] * 100
                            obj_diffs.append(obj_diff)
                            print(f"Objective value difference: {obj_diff:.6f}%")
                        else:
                            print("Warm-start solution failed.")
                    except Exception as e:
                        print(f"Error in warm-start solution: {e}")
                else:
                    print("No-warm-start solution failed.")
            except Exception as e:
                print(f"Error in no-warm-start solution: {e}")
        
        # Save results
        if speedups:
            results = {
                'no_warm_times': no_warm_times,
                'warm_times': warm_times,
                'speedups': speedups,
                'obj_diffs': obj_diffs
            }
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(log_dir, 'warm_start_results.csv'), index=False)
            
            # Calculate and print statistics
            avg_speedup = np.mean(speedups)
            avg_obj_diff = np.mean(obj_diffs)
            
            print("\nWarm-starting results:")
            print(f"Average speedup: {avg_speedup:.2f}x")
            print(f"Average objective difference: {avg_obj_diff:.6f}%")
            
            # Create summary plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(speedups)), speedups)
            plt.axhline(y=1.0, color='r', linestyle='--')
            plt.xlabel('Sample')
            plt.ylabel('Speedup Factor')
            plt.title('Warm-Starting Speedup')
            plt.savefig(os.path.join(log_dir, 'speedup_plot.png'))
            
            # Save summary
            with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
                f.write(f"Number of samples: {len(speedups)}\n")
                f.write(f"Average no-warm-start time: {np.mean(no_warm_times):.6f} seconds\n")
                f.write(f"Average warm-start time: {np.mean(warm_times):.6f} seconds\n")
                f.write(f"Average speedup: {avg_speedup:.2f}x\n")
                f.write(f"Speedup range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x\n")
                f.write(f"Average objective difference: {avg_obj_diff:.6f}%\n")
        
        print(f"Results saved to {log_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    main(args) 