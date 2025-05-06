#!/usr/bin/env python
"""
Simplified Direct Prediction model for ML-AC-OPF.
This script is a stripped-down version that focuses on GNN implementation.
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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('direct_prediction_simple')

def parse_args():
    parser = argparse.ArgumentParser(description='Direct Prediction ML-OPF (Simplified)')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with X and y data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use-gnn', action='store_true', help='Use Graph Neural Networks')
    return parser.parse_args()

def load_data(input_dir):
    """Load data from input_dir."""
    try:
        X = np.load(os.path.join(input_dir, 'X_direct.npy'))
        y = np.load(os.path.join(input_dir, 'y_direct.npy'))
        logger.info(f"Loaded data: {X.shape}, {y.shape}")
    except:
        logger.info("Numpy files not found, loading from CSV")
        X = pd.read_csv(os.path.join(input_dir, 'X_direct.csv')).values
        y = pd.read_csv(os.path.join(input_dir, 'y_direct.csv')).values
        logger.info(f"Loaded data from CSV: {X.shape}, {y.shape}")
    
    return X, y

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X, y = load_data(args.input_dir)
    
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
    
    logger.info(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]} samples")
    
    # Create standard datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Model parameters
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    # Choose model type
    if args.use_gnn:
        try:
            # Import required GNN modules
            import torch_geometric
            from models.gnn import DirectPredictionGNN
            from utils.network_utils import create_power_network_graph
            from torch_geometric.loader import DataLoader as GraphDataLoader
            
            logger.info(f"Using GNN architecture (PyTorch Geometric {torch_geometric.__version__})")
            
            # Create graph data
            logger.info("Converting data to graph format for GNN")
            
            try:
                # Instead of calling create_power_network_graph, let's manually construct the graphs
                # with proper debugging information
                
                # For IEEE 39-bus system
                n_buses = 39
                feat_per_node = X.shape[1] // n_buses
                
                if X.shape[1] % n_buses != 0:
                    logger.warning(f"Input feature dimension {X.shape[1]} not divisible by {n_buses} buses")
                    feat_per_node = 1  # Fallback to simpler features
                
                logger.info(f"Graph construction: {n_buses} buses with {feat_per_node} features per node")
                
                # Define the edge connections based on IEEE 39-bus topology
                edges = [
                    (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18),
                    (4, 5), (4, 14), (5, 6), (5, 8), (6, 7), (6, 11),
                    (7, 8), (8, 9), (9, 39), (10, 11), (10, 13), (13, 14),
                    (14, 15), (15, 16), (16, 17), (16, 19), (16, 21), (16, 24),
                    (17, 18), (17, 27), (19, 20), (19, 33), (20, 34), (21, 22),
                    (22, 23), (22, 35), (23, 24), (23, 36), (25, 26), (25, 37),
                    (26, 27), (26, 28), (26, 29), (28, 29), (29, 38), (30, 31),
                    (31, 32), (32, 33), (33, 34), (34, 35), (36, 37), (36, 38),
                    (38, 39)
                ]
                
                # Convert to 0-indexed for PyTorch Geometric
                edges = [(src-1, dst-1) for src, dst in edges]
                
                # Make edges bidirectional (undirected graph)
                edges += [(dst, src) for src, dst in edges]
                
                # Convert to PyTorch tensors
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                
                from torch_geometric.data import Data, Batch
                
                # Create a simpler approach using the full feature vector for each sample
                train_graphs = []
                val_graphs = []
                test_graphs = []
                
                # For training data
                for i in range(len(X_train)):
                    # Create a simple graph with all features as node features
                    # Distribute features evenly across nodes (simplification)
                    node_feats = torch.zeros((n_buses, max(1, feat_per_node)), dtype=torch.float)
                    
                    # For simplicity, just distribute the features across the nodes
                    # This is a placeholder and may not be physically meaningful
                    for j in range(min(X_train.shape[1], n_buses)):
                        node_feats[j, 0] = X_train[i, j]
                    
                    graph = Data(x=node_feats, edge_index=edge_index)
                    graph.y = torch.FloatTensor(y_train[i])
                    train_graphs.append(graph)
                
                # For validation data
                for i in range(len(X_val)):
                    node_feats = torch.zeros((n_buses, max(1, feat_per_node)), dtype=torch.float)
                    for j in range(min(X_val.shape[1], n_buses)):
                        node_feats[j, 0] = X_val[i, j]
                    
                    graph = Data(x=node_feats, edge_index=edge_index)
                    graph.y = torch.FloatTensor(y_val[i])
                    val_graphs.append(graph)
                
                # For test data
                for i in range(len(X_test)):
                    node_feats = torch.zeros((n_buses, max(1, feat_per_node)), dtype=torch.float)
                    for j in range(min(X_test.shape[1], n_buses)):
                        node_feats[j, 0] = X_test[i, j]
                    
                    graph = Data(x=node_feats, edge_index=edge_index)
                    graph.y = torch.FloatTensor(y_test[i])
                    test_graphs.append(graph)
                
                logger.info(f"Created {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs, {len(test_graphs)} test graphs")
                
                # Create graph data loaders
                train_loader = GraphDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
                val_loader = GraphDataLoader(val_graphs, batch_size=args.batch_size)
                test_loader = GraphDataLoader(test_graphs, batch_size=args.batch_size)
                
                logger.info(f"Created graph data loaders")
                
                # Create GNN model
                model = DirectPredictionGNN(
                    node_features=max(1, feat_per_node),  # Features per node
                    hidden_dim=args.hidden_dim,
                    output_dim=output_dim,
                    num_layers=4,
                    dropout_rate=args.dropout
                ).to(device)
                
                logger.info(f"Created GNN model with {feat_per_node} features per node, {output_dim} outputs")
                
            except Exception as e:
                logger.error(f"Failed to create graph data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("Falling back to standard model")
                args.use_gnn = False
    
    if not args.use_gnn:
        # Standard MLP
        logger.info("Using standard feedforward architecture")
        
        # Define straightforward model architecture
        class MLPModel(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(args.dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(args.dropout),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = MLPModel(input_dim, output_dim, args.hidden_dim).to(device)
        logger.info(f"Created MLP model with {input_dim} inputs, {output_dim} outputs")
    
    # Simple training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
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
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Evaluate
    logger.info("Evaluating model on test set")
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Calculate R^2 for each output dimension
    r2_values = []
    for i in range(output_dim):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        r2_values.append(r2)
    
    avg_r2 = np.mean(r2_values)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"Average R^2: {avg_r2:.6f}")
    
    for i, r2 in enumerate(r2_values):
        logger.info(f"R^2 for output {i+1}: {r2:.6f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    logger.info(f"Model saved to {os.path.join(args.output_dir, 'model.pt')}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 