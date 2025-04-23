#!/usr/bin/env python
"""
Warm-Starting approach for ML-OPF project.
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

from utils import (
    load_pglib_data,
    load_case_network,
    prepare_data_loaders,
    OPFOptimizer,
    Trainer,
    optimality_gap_metric
)

from models import (
    WarmStartNN
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Warm-Starting')
    parser.add_argument('--case', type=str, default='case118', help='Case name (default: case118)')
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
    
    return parser.parse_args()

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
        
        # Initialize optimizer
        optimizer = OPFOptimizer(case_data, device=device)
        
        # Extract input and output columns
        input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
        output_cols = [col for col in data.columns if col.startswith('gen_p') or 
                       col.startswith('gen_q') or col.startswith('bus_v')]
        
        print(f"Input features: {len(input_cols)}")
        print(f"Output features: {len(output_cols)}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=args.batch_size)
        
        # Parse hidden dimensions
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        
        # Create bounds for outputs based on problem constraints
        pg_min = torch.tensor(case_data['gen'][:, 9] / case_data['baseMVA'], dtype=torch.float32, device=device)
        pg_max = torch.tensor(case_data['gen'][:, 8] / case_data['baseMVA'], dtype=torch.float32, device=device)
        qg_min = torch.tensor(case_data['gen'][:, 4] / case_data['baseMVA'], dtype=torch.float32, device=device)
        qg_max = torch.tensor(case_data['gen'][:, 3] / case_data['baseMVA'], dtype=torch.float32, device=device)
        vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32, device=device)
        vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32, device=device)
        
        # Assemble bounds
        n_gen = len(case_data['gen'])
        n_bus = len(case_data['bus'])
        
        min_vals = []
        max_vals = []
        
        # Add generator active power bounds
        min_vals.append(pg_min)
        max_vals.append(pg_max)
        
        # Add generator reactive power bounds
        min_vals.append(qg_min)
        max_vals.append(qg_max)
        
        # Add voltage magnitude bounds
        min_vals.append(vm_min)
        max_vals.append(vm_max)
        
        # Add voltage angle bounds
        min_vals.append(torch.full((n_bus,), -np.pi, device=device))
        max_vals.append(torch.full((n_bus,), np.pi, device=device))
        
        # Concatenate bounds
        min_bounds = torch.cat(min_vals)
        max_bounds = torch.cat(max_vals)
        
        # Ensure bounds match output dimension
        min_bounds = min_bounds[:len(output_cols)]
        max_bounds = max_bounds[:len(output_cols)]
        output_bounds = (min_bounds, max_bounds)
        
        # Initialize model
        model = WarmStartNN(
            input_dim=len(input_cols),
            output_dim=len(output_cols),
            hidden_dims=hidden_dims,
            output_bounds=output_bounds
        )
        
        # Create cost coefficients for optimality gap metric
        cost_coeffs = torch.tensor(
            [coef[5] for coef in case_data['gencost']], 
            dtype=torch.float32,
            device=device
        )
        
        # Define metrics
        metrics = {
            'opt_gap': lambda pred, target: optimality_gap_metric(pred, target, cost_coeffs)
        }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            criterion=torch.nn.MSELoss(),
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
            
            # Prepare warm-start solution
            warm_start = {
                'pg': prediction_np[:n_gen],
                'qg': prediction_np[n_gen:2*n_gen],
                'vm': prediction_np[2*n_gen:2*n_gen+n_bus],
                'va': prediction_np[2*n_gen+n_bus:2*n_gen+2*n_bus]
            }
            
            # Prepare load data (reshape to pairs of [p, q])
            load_data = sample_input[0].cpu().numpy().reshape(-1, 2)
            
            # Solve OPF without warm-start
            print(f"\nSample {i+1}/{n_samples}:")
            print("Solving OPF without warm-start...")
            solution_no_warm = optimizer.solve_opf_gurobi(load_data, verbose=False)
            
            if solution_no_warm['success']:
                print(f"Solution found in {solution_no_warm['runtime']:.6f} seconds.")
                no_warm_times.append(solution_no_warm['runtime'])
                
                # Solve OPF with warm-start
                print("Solving OPF with warm-start...")
                solution_warm = optimizer.solve_opf_gurobi(load_data, warm_start=warm_start, verbose=False)
                
                if solution_warm['success']:
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
            else:
                print("No-warm-start solution failed.")
        
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