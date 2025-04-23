#!/usr/bin/env python
"""
Simple test script to work with case5 data directly without using pypower's case loading.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feedforward import FeedForwardNN
from utils.training import Trainer
from utils.data_utils import OPFDataset

def main():
    # Configuration
    case_name = "case5"
    data_dir = "data"
    epochs = 5
    batch_size = 32
    learning_rate = 0.001
    hidden_dims = [64, 32]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    csv_file = os.path.join(data_dir, f"pglib_opf_{case_name}.csv")
    print(f"Loading data from {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return
    
    data = pd.read_csv(csv_file)
    print(f"Data loaded: {len(data)} samples")
    
    # Determine input and output columns
    print("Column names:", data.columns[:10].tolist(), "...")
    
    # Example column selection - adjust based on actual column names
    input_cols = [col for col in data.columns if col.startswith('load')]
    output_cols = [col for col in data.columns if col.startswith('gen') and ':pg' in col]
    
    print(f"Input features: {len(input_cols)}")
    print(f"Output features: {len(output_cols)}")
    print(f"First few input columns: {input_cols[:5]}")
    print(f"First few output columns: {output_cols[:5]}")
    
    # Split data
    n_samples = len(data)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets
    train_dataset = OPFDataset(train_data, input_cols, output_cols)
    val_dataset = OPFDataset(val_data, input_cols, output_cols)
    test_dataset = OPFDataset(test_data, input_cols, output_cols)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    # Get input and output dimensions
    input_dim = len(input_cols)
    output_dim = len(output_cols)
    
    # Initialize model
    model = FeedForwardNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim
    ).to(device)
    
    print(f"Model initialized with {input_dim} inputs, {output_dim} outputs, and hidden layers {hidden_dims}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        device=device
    )
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )
    
    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, f'case5_training_{timestamp}.png'))
    
    # Evaluate on test set
    test_loss, _ = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.6f}")
    
    # Plot predictions vs targets
    model.eval()
    with torch.no_grad():
        # Get first batch for visualization
        inputs, targets = next(iter(test_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Get predictions
        predictions = model(inputs)
        
        # Convert to numpy for plotting
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Plot first 5 samples for the first output
        plt.figure(figsize=(10, 5))
        x = np.arange(min(5, len(predictions)))
        width = 0.35
        
        plt.bar(x - width/2, targets[:5, 0], width, label='Target')
        plt.bar(x + width/2, predictions[:5, 0], width, label='Prediction')
        
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Predictions vs Targets for {output_cols[0]}')
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(log_dir, f'case5_predictions_{timestamp}.png'))
        
        print(f"Plots saved to {log_dir}")

if __name__ == "__main__":
    main()