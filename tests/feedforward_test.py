#!/usr/bin/env python
"""
Feedforward Neural Network test script for the ML-AC-OPF project.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import argparse
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('feedforward_test')

def parse_args():
    parser = argparse.ArgumentParser(description='Feedforward NN Test Script')
    parser.add_argument('--input-dir', type=str, default='output/ieee39_data', help='Input directory')
    parser.add_argument('--output-dir', type=str, default='output/ff_test', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    return parser.parse_args()

class FeedforwardModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(FeedforwardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    # Parse args
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    try:
        X = np.load(os.path.join(args.input_dir, 'X_direct.npy'))
        y = np.load(os.path.join(args.input_dir, 'y_direct.npy'))
        logger.info(f"Loaded data: {X.shape}, {y.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create standard datasets and loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    model = FeedforwardModel(X.shape[1], y.shape[1], args.hidden_dim).to(device)
    logger.info(f"Created feedforward model with {X.shape[1]} inputs, {y.shape[1]} outputs")
    
    # Train the model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_dataset)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Evaluate
    logger.info("Evaluating model")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    r2 = np.mean([r2_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])])
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test R²: {r2:.6f}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'ff_results.txt'), 'w') as f:
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test R²: {r2:.6f}\n")
    
    logger.info(f"Results saved to {os.path.join(args.output_dir, 'ff_results.txt')}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 