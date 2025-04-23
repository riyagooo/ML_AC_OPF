#!/usr/bin/env python
"""
Main run script for ML-OPF project.
This script provides a simple interface to run different experiments.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Project Runner')
    parser.add_argument('command', type=str, choices=['download', 'train', 'test', 'compare'],
                        help='Command to run: download, train, test, or compare')
    parser.add_argument('--approach', type=str, 
                        choices=['feedforward', 'constraint_screening', 'warm_starting', 'gnn'],
                        default='feedforward',
                        help='ML-OPF approach to use')
    parser.add_argument('--case', type=str, default='case118',
                        help='Case name (default: case118)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--colab', action='store_true',
                        help='Running in Google Colab environment')
    parser.add_argument('--model-type', type=str, default='gnn',
                        choices=['gnn', 'hybrid_gnn'],
                        help='GNN model type (only for GNN approach)')
    
    return parser.parse_args()

def run_command(command):
    """Run a command and print its output."""
    print(f"Running command: {command}")
    
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main(args):
    """Main function."""
    # Check if running in Colab
    in_colab = 'google.colab' in sys.modules or args.colab
    
    # Set GPU flag
    gpu_flag = '--gpu' if args.gpu else ''
    
    # List of small cases that can be run locally with GNN
    small_cases = ['case5', 'case14', 'case30']
    
    if args.command == 'download':
        # Download data
        cmd = f"python scripts/download_data.py --case {args.case}"
        run_command(cmd)
    
    elif args.command == 'train':
        # Train a model based on the approach
        if args.approach == 'feedforward':
            cmd = f"python scripts/local_test.py --case {args.case} --model-type feedforward --epochs {args.epochs} --batch-size {args.batch_size} {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'constraint_screening':
            cmd = f"python scripts/constraint_screening.py --case {args.case} --epochs {args.epochs} --batch-size {args.batch_size} {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'warm_starting':
            cmd = f"python scripts/warm_starting.py --case {args.case} --epochs {args.epochs} --batch-size {args.batch_size} {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'gnn':
            if args.case in small_cases and not in_colab:
                # For small cases, we can run GNN locally
                cmd = f"python scripts/local_gnn.py --case {args.case} --model-type {args.model_type} --epochs {args.epochs} --batch-size {args.batch_size} {gpu_flag}"
                print(f"Running GNN training locally on small case: {args.case}")
                run_command(cmd)
            elif in_colab:
                print("For GNN training in Colab, please use the notebook: notebooks/gnn_training.ipynb")
            else:
                print(f"Case {args.case} is too large for local GNN training. Use a smaller case (case5, case14, case30) or run in Colab.")
                print("For now, running simplified feedforward model as an alternative.")
                cmd = f"python scripts/local_test.py --case {args.case} --model-type feedforward --epochs {args.epochs} --batch-size {args.batch_size} {gpu_flag}"
                run_command(cmd)
    
    elif args.command == 'test':
        # Test a trained model
        if args.approach == 'feedforward':
            cmd = f"python scripts/local_test.py --case {args.case} --model-type feedforward --epochs 1 {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'constraint_screening':
            cmd = f"python scripts/constraint_screening.py --case {args.case} --epochs 1 {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'warm_starting':
            cmd = f"python scripts/warm_starting.py --case {args.case} --epochs 1 --num-samples 3 {gpu_flag}"
            run_command(cmd)
        
        elif args.approach == 'gnn':
            if args.case in small_cases and not in_colab:
                # For small cases, we can run GNN locally with fewer epochs for testing
                cmd = f"python scripts/local_gnn.py --case {args.case} --model-type {args.model_type} --epochs 1 --batch-size {args.batch_size} {gpu_flag}"
                print(f"Testing GNN locally on small case: {args.case}")
                run_command(cmd)
            elif in_colab:
                print("For GNN testing in Colab, please use the notebook: notebooks/gnn_training.ipynb")
            else:
                print(f"Case {args.case} is too large for local GNN testing. Use a smaller case (case5, case14, case30) or run in Colab.")
    
    elif args.command == 'compare':
        # Compare different approaches
        print("Comparing different ML-OPF approaches...")
        
        # Use a small case for comparison if GNN is included
        case = 'case5' if args.approach == 'gnn' else args.case
        
        # Run constraint screening
        print("\n\n== CONSTRAINT SCREENING APPROACH ==\n")
        cmd = f"python scripts/constraint_screening.py --case {case} --epochs 1 {gpu_flag}"
        run_command(cmd)
        
        # Run warm starting
        print("\n\n== WARM STARTING APPROACH ==\n")
        cmd = f"python scripts/warm_starting.py --case {case} --epochs 1 --num-samples 3 {gpu_flag}"
        run_command(cmd)
        
        # Run GNN if small case
        if case in small_cases:
            print("\n\n== GNN APPROACH ==\n")
            cmd = f"python scripts/local_gnn.py --case {case} --model-type {args.model_type} --epochs 1 {gpu_flag}"
            run_command(cmd)
        else:
            print("\n\n== GNN APPROACH ==\n")
            print(f"Case {case} is too large for local GNN training.")
            print("For GNN comparison, please use Google Colab with the notebook: notebooks/gnn_training.ipynb")
            print("Alternatively, run with a smaller case: python run.py compare --approach gnn --case case5")

if __name__ == '__main__':
    args = parse_args()
    main(args) 