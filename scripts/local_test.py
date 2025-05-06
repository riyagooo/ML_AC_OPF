#!/usr/bin/env python
"""
Local test script for ML-OPF project.
This script can be run locally to test the basic functionality.
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
    FeedForwardNN,
    WarmStartNN,
    ConstraintScreeningNN
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Local Test')
    parser.add_argument('--case', type=str, default='case30', help='Case name (default: case30)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model-type', type=str, default='feedforward', 
                        choices=['feedforward', 'warm_start', 'constraint_screening'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='128,256,128', 
                        help='Hidden dimensions (comma-separated)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    return parser.parse_args()

def main(args):
    """Main function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    try:
        print(f"Loading data for {args.case}...")
        data = load_pglib_data(args.case, args.data_dir)
        case_data = load_case_network(args.case, args.data_dir)
        print(f"Data loaded: {len(data)} samples")
        
        # Extract input and output columns based on CSV column format
        input_cols = [col for col in data.columns if col.startswith('load')]
        if not input_cols:
            # Try alternative naming pattern
            input_cols = [col for col in data.columns if ':pl' in col or ':ql' in col]
        
        output_cols = [col for col in data.columns if col.startswith('gen') or col.startswith('bus')]
        if not output_cols:
            # Try alternative naming pattern
            output_cols = [col for col in data.columns if ':pg' in col or ':qg' in col or ':v_' in col]
            
        if not input_cols:
            raise ValueError(f"No input columns found. Column names are: {list(data.columns)[:10]}...")
        if not output_cols:
            raise ValueError(f"No output columns found. Column names are: {list(data.columns)[:10]}...")
        
        print(f"Input features: {len(input_cols)}")
        print(f"Output features: {len(output_cols)}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=args.batch_size)
        
        # Initialize optimizer
        optimizer = OPFOptimizer(case_data, device=device)
        
        # Parse hidden dimensions
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        
        # Initialize model based on type
        if args.model_type == 'feedforward':
            model = FeedForwardNN(
                input_dim=len(input_cols),
                output_dim=len(output_cols),
                hidden_dims=hidden_dims
            )
        elif args.model_type == 'warm_start':
            # Create bounds for outputs
            bounds = (
                torch.tensor(np.zeros(len(output_cols)), dtype=torch.float32),
                torch.tensor(np.ones(len(output_cols)) * 2, dtype=torch.float32)
            )
            model = WarmStartNN(
                input_dim=len(input_cols),
                output_dim=len(output_cols),
                hidden_dims=hidden_dims,
                output_bounds=bounds
            )
        elif args.model_type == 'constraint_screening':
            # Assume number of constraints = number of branches
            num_branches = len(case_data['branch'])
            model = ConstraintScreeningNN(
                input_dim=len(input_cols),
                num_constraints=num_branches,
                hidden_dims=hidden_dims
            )
        
        # Create cost coefficients for optimality gap metric
        try:
            cost_coeffs = torch.tensor(
                [coef[5] for coef in case_data['gencost']], 
                dtype=torch.float32,
                device=device
            )
        except (IndexError, KeyError) as e:
            print(f"Warning: Error getting cost coefficients: {e}. Using default values.")
            # Default to ones if we can't get the coefficients
            n_gen = len([col for col in output_cols if ':pg' in col or 'gen' in col])
            if n_gen == 0:
                n_gen = 5  # Default for case5
            cost_coeffs = torch.ones(n_gen, dtype=torch.float32, device=device)
        
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
        print(f"Training {args.model_type} model for {args.epochs} epochs...")
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
            
        # Save test metrics
        with open(os.path.join(log_dir, 'test_metrics.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.6f}\n")
            for name, value in test_metrics.items():
                f.write(f"Test {name}: {value:.6f}\n")
        
        print(f"Results saved to {log_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    main(args) 