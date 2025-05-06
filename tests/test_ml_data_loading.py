#!/usr/bin/env python
"""
Test script to validate ML data loading from MATPOWER and CSV files.
"""

import os
import sys
import torch
import pandas as pd

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_case_loader import load_case
from utils.data_utils import load_pglib_data, load_case_network, prepare_data_loaders, OPFDataset
from models.feedforward import FeedForwardNN

def test_matpower_loading():
    """Test MATPOWER file loading."""
    print("\n=== Testing MATPOWER File Loading ===")
    
    try:
        # Try loading case5
        case_data = load_case_network('case5')
        print(f"Successfully loaded case5 with {len(case_data['bus'])} buses, "
              f"{len(case_data['gen'])} generators, {len(case_data['branch'])} branches")
        
        # Check if we can load other cases
        cases = ['case14', 'case30']
        for case_name in cases:
            try:
                file_path = os.path.join('data', f'pglib_opf_{case_name}.m')
                if os.path.exists(file_path):
                    case_data = load_case_network(case_name)
                    print(f"Successfully loaded {case_name} with {len(case_data['bus'])} buses, "
                          f"{len(case_data['gen'])} generators, {len(case_data['branch'])} branches")
                else:
                    print(f"Skipping {case_name}: File not found")
            except Exception as e:
                print(f"Error loading {case_name}: {e}")
        
        return True
    except Exception as e:
        print(f"Error loading MATPOWER files: {e}")
        return False

def test_csv_loading():
    """Test CSV data loading."""
    print("\n=== Testing CSV Data Loading ===")
    
    try:
        # Get CSV files in data directory
        data_dir = "data"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files in data directory:")
        for csv_file in csv_files:
            print(f"- {csv_file}")
        
        # Try loading case5 CSV
        try:
            data = load_pglib_data('case5')
            print(f"Successfully loaded case5 CSV with {len(data)} rows and {len(data.columns)} columns")
            print(f"First few input columns: {[col for col in data.columns if col.startswith('load')][:5]}")
            print(f"First few output columns: {[col for col in data.columns if col.startswith('gen')][:5]}")
            return True
        except Exception as e:
            print(f"Error loading case5 CSV: {e}")
            return False
    except Exception as e:
        print(f"Error checking CSV files: {e}")
        return False

def test_dataloader_creation():
    """Test creating DataLoader objects."""
    print("\n=== Testing DataLoader Creation ===")
    
    try:
        # Load case5 data
        data = load_pglib_data('case5')
        
        # Define input and output columns
        input_cols = [col for col in data.columns if col.startswith('load')]
        output_cols = [col for col in data.columns if col.startswith('gen') and ':pg' in col]
        
        print(f"Input features: {len(input_cols)}")
        print(f"Output features: {len(output_cols)}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=32)
        
        print(f"Created data loaders with {len(train_loader.dataset)} training samples, "
              f"{len(val_loader.dataset)} validation samples, and {len(test_loader.dataset)} test samples")
        
        # Check batch dimensions
        inputs, targets = next(iter(train_loader))
        print(f"Batch shapes: inputs {inputs.shape}, targets {targets.shape}")
        
        return True
    except Exception as e:
        print(f"Error creating DataLoader objects: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass with loaded data."""
    print("\n=== Testing Model Forward Pass ===")
    
    try:
        # Load case5 data
        data = load_pglib_data('case5')
        
        # Define input and output columns
        input_cols = [col for col in data.columns if col.startswith('load')]
        output_cols = [col for col in data.columns if col.startswith('gen') and ':pg' in col]
        
        # Create data loaders
        train_loader, _, _ = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=32)
        
        # Initialize model
        model = FeedForwardNN(
            input_dim=len(input_cols),
            output_dim=len(output_cols),
            hidden_dims=[64, 128, 64]
        )
        
        # Get a batch of data
        inputs, targets = next(iter(train_loader))
        
        # Run forward pass
        outputs = model(inputs)
        
        print(f"Model forward pass successful with input shape {inputs.shape} and output shape {outputs.shape}")
        
        return True
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("Testing ML data loading and processing capabilities...")
    
    # Run tests
    matpower_success = test_matpower_loading()
    csv_success = test_csv_loading()
    dataloader_success = test_dataloader_creation()
    model_success = test_model_forward_pass()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"MATPOWER loading: {'✓' if matpower_success else '✗'}")
    print(f"CSV loading: {'✓' if csv_success else '✗'}")
    print(f"DataLoader creation: {'✓' if dataloader_success else '✗'}")
    print(f"Model forward pass: {'✓' if model_success else '✗'}")
    
    # Overall evaluation
    if all([matpower_success, csv_success, dataloader_success, model_success]):
        print("\nAll tests passed! ML models can correctly load and process data.")
    else:
        print("\nSome tests failed. Data loading or processing issues need to be addressed.")

if __name__ == "__main__":
    main() 