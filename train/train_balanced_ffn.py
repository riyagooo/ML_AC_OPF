#!/usr/bin/env python
"""
Balanced FFN model for AC-OPF prediction with medium complexity.
This script trains a feedforward neural network with a reasonable balance 
between speed and performance that matches the GNN configuration.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('balanced_ffn')

class BalancedFFN(nn.Module):
    """
    Balanced feedforward neural network for direct prediction of AC-OPF results.
    
    This model has a matched complexity to the GNN counterpart.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout_rate=0.2):
        super(BalancedFFN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

def parse_args():
    parser = argparse.ArgumentParser(description='Balanced FFN for AC-OPF Prediction')
    parser.add_argument('--input-dir', type=str, default='output/ieee39_data_small', 
                        help='Input directory with X and y data')
    parser.add_argument('--output-dir', type=str, default='output/balanced_ffn', 
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, 
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, 
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, 
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Early stopping patience')
    parser.add_argument('--use-scaled-data', action='store_true', 
                        help='Use pre-scaled data')
    parser.add_argument('--save-model', action='store_true', 
                        help='Save the trained model')
    return parser.parse_args()

def load_data(input_dir, use_scaled_data=True):
    """Load data from input_dir."""
    logger.info(f"Loading data from {input_dir}")
    
    try:
        if use_scaled_data:
            X = np.load(os.path.join(input_dir, 'X_direct_scaled.npy'))
            y = np.load(os.path.join(input_dir, 'y_direct_scaled.npy'))
            logger.info(f"Loaded scaled data from numpy files: {X.shape}, {y.shape}")
        else:
            X = np.load(os.path.join(input_dir, 'X_direct.npy'))
            y = np.load(os.path.join(input_dir, 'y_direct.npy'))
            logger.info(f"Loaded raw data from numpy files: {X.shape}, {y.shape}")
    except:
        logger.error("Failed to load numpy files. Please ensure the dataset is prepared correctly.")
        sys.exit(1)
    
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=10):
    """Train model with early stopping."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs}: Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test data."""
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # Move to CPU for evaluation
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    
    # Combine batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R² for the entire model (average across outputs)
    r2 = r2_score(targets, predictions, multioutput='variance_weighted')
    
    return test_loss, mse, mae, r2, predictions, targets

def plot_loss_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

def plot_predictions(predictions, targets, output_dir):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Plot for the first few outputs
    for i in range(min(4, predictions.shape[1])):
        plt.subplot(2, 2, i+1)
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([targets[:, i].min(), targets[:, i].max()], 
                [targets[:, i].min(), targets[:, i].max()], 'r--')
        plt.xlabel(f'Actual Value (Output {i+1})')
        plt.ylabel(f'Predicted Value (Output {i+1})')
        plt.title(f'Output {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X, y = load_data(args.input_dir, args.use_scaled_data)
    
    # Split data
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Start timer for training
    start_time = time.time()
    
    # Create standard datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = BalancedFFN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    ).to(device)
    
    logger.info(f"Created model with {input_dim} inputs, {output_dim} outputs, " 
               f"{args.hidden_dim} hidden dim, {args.num_layers} layers, {args.dropout} dropout")
    
    # Create optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train model
    logger.info(f"Starting training for up to {args.epochs} epochs with early stopping")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs, patience=args.patience
    )
    
    # Calculate total training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curves
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    test_loss, mse, mae, r2, predictions, targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Plot predictions
    plot_predictions(predictions, targets, args.output_dir)
    
    # Log R² for each output dimension
    for i in range(predictions.shape[1]):
        r2_i = r2_score(targets[:, i], predictions[:, i])
        logger.info(f"R^2 for output {i+1}: {r2_i:.6f}")
    
    # Save results
    results = {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'num_samples': n_samples,
        'model_complexity': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    }
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Add R² for each output
        for i in range(predictions.shape[1]):
            r2_i = r2_score(targets[:, i], predictions[:, i])
            f.write(f"R^2 for output {i+1}: {r2_i}\n")
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'balanced_ffn_model.pt'))
        logger.info(f"Model saved to {os.path.join(args.output_dir, 'balanced_ffn_model.pt')}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main() 