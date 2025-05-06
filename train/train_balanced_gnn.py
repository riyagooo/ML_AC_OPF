#!/usr/bin/env python
"""
Balanced GNN model for AC-OPF prediction with medium complexity.
This script trains a GNN with a reasonable balance between speed and performance.
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

# Import PyTorch Geometric modules for GNN
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from models.gnn import EnhancedDirectPredictionGNN, create_power_system_graph, train_model_gnn, evaluate_model_gnn
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("Error: torch_geometric not found. This script requires PyTorch Geometric.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('balanced_gnn')

def parse_args():
    parser = argparse.ArgumentParser(description='Balanced GNN for AC-OPF Prediction')
    parser.add_argument('--input-dir', type=str, default='output/ieee39_data_small', 
                        help='Input directory with X and y data')
    parser.add_argument('--output-dir', type=str, default='output/balanced_gnn', 
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
    
    # Create graph representation
    logger.info("Creating graph representation of power system data")
    train_graphs = create_power_system_graph(X_train)
    val_graphs = create_power_system_graph(X_val)
    test_graphs = create_power_system_graph(X_test)
    
    # Add target values to graphs
    for i, graph in enumerate(train_graphs):
        graph.y = torch.tensor(y_train[i:i+1], dtype=torch.float)
    for i, graph in enumerate(val_graphs):
        graph.y = torch.tensor(y_val[i:i+1], dtype=torch.float)
    for i, graph in enumerate(test_graphs):
        graph.y = torch.tensor(y_test[i:i+1], dtype=torch.float)
    
    # Create PyG DataLoaders
    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size)
    
    # Create GNN model
    node_features = train_graphs[0].x.size(1)
    output_dim = y.shape[1]
    
    # Use the enhanced GNN model with edge features
    model = EnhancedDirectPredictionGNN(
        node_features=node_features,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout
    ).to(device)
    
    logger.info(f"Created model with {node_features} node features, {output_dim} outputs, " 
               f"{args.hidden_dim} hidden dim, {args.num_layers} layers, {args.dropout} dropout")
    
    # Create optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train model
    logger.info(f"Starting training for up to {args.epochs} epochs with early stopping")
    model, train_losses, val_losses = train_model_gnn(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs, patience=args.patience
    )
    
    # Calculate total training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curves
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    test_loss, mse, mae, r2, predictions, targets = evaluate_model_gnn(
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
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'balanced_gnn_model.pt'))
        logger.info(f"Model saved to {os.path.join(args.output_dir, 'balanced_gnn_model.pt')}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main() 