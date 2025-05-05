#!/usr/bin/env python
"""
Simple test script for ML-OPF project.
This is a minimal script to test the basic functionality.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import FeedForwardNN
from custom_case_loader import load_case

class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, n_samples=1000, input_dim=6, output_dim=15):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        
        # Generate random data
        self.inputs = torch.rand(n_samples, input_dim)
        self.outputs = torch.rand(n_samples, output_dim)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def main():
    """Main function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    
    # Load case5 data to check if it works
    try:
        case_file = os.path.join('data', 'pglib_opf_case5.m')
        case_data = load_case(case_file)
        print(f"Successfully loaded case5 data")
        print(f"Number of buses: {len(case_data['bus'])}")
        print(f"Number of generators: {len(case_data['gen'])}")
    except Exception as e:
        print(f"Error loading case data: {e}")
    
    # Create a simple dataset
    input_dim = 6  # For case5, 5 buses + 1 feature per bus
    output_dim = 15  # For case5, 5 generators + 5 buses + 5 branches
    train_dataset = SimpleDataset(n_samples=1000, input_dim=input_dim, output_dim=output_dim)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    hidden_dims = [64, 128, 64]
    model = FeedForwardNN(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize loss function
    criterion = torch.nn.MSELoss()
    
    # Train for a few epochs
    print(f"Training model for 2 epochs...")
    for epoch in range(2):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/2, Loss: {avg_loss:.6f}")
    
    print("Training completed successfully!")
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        test_input = torch.rand(1, input_dim).to(device)
        prediction = model(test_input)
        print(f"Test prediction shape: {prediction.shape}")
    
    print("Test completed successfully!")

if __name__ == '__main__':
    main()