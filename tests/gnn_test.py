#!/usr/bin/env python
"""
Simple GNN test script for the ML-AC-OPF project.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gnn_test')

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Test Script')
    parser.add_argument('--input-dir', type=str, default='output/ieee39_data', help='Input directory')
    parser.add_argument('--output-dir', type=str, default='output/gnn_test', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    return parser.parse_args()

def main():
    # Parse args
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
        have_gnn = True
    except ImportError:
        logger.warning("PyTorch Geometric not available")
        have_gnn = False
    
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
    
    if have_gnn:
        try:
            # Import GNN-specific modules
            from torch_geometric.data import Data, Batch
            from torch_geometric.loader import DataLoader as GraphDataLoader
            
            # Import GNN model from models directory
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models.gnn import DirectPredictionGNN
            
            # Create graph data
            logger.info("Creating graph data...")
            
            # IEEE 39-bus system topology
            n_buses = 39
            
            # Add edges (connections between buses)
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
            
            # Make bidirectional
            edges += [(dst, src) for src, dst in edges]
            
            # Convert to tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Create graph objects for each sample
            train_graphs = []
            val_graphs = []
            test_graphs = []
            
            # For simplicity, just use one feature per node
            for i in range(len(X_train)):
                # Create node features with simple distribution
                node_features = torch.zeros((n_buses, 1), dtype=torch.float)
                
                # Assign input features to nodes
                for j in range(min(X_train.shape[1], n_buses)):
                    node_features[j, 0] = X_train[i, j]
                
                graph = Data(x=node_features, edge_index=edge_index)
                # Reshape target to ensure dimensions match model output [batch_size, output_dim]
                graph.y = torch.tensor(y_train[i], dtype=torch.float).view(1, -1)
                train_graphs.append(graph)
            
            # Validation graphs
            for i in range(len(X_val)):
                node_features = torch.zeros((n_buses, 1), dtype=torch.float)
                for j in range(min(X_val.shape[1], n_buses)):
                    node_features[j, 0] = X_val[i, j]
                
                graph = Data(x=node_features, edge_index=edge_index)
                graph.y = torch.tensor(y_val[i], dtype=torch.float).view(1, -1)
                val_graphs.append(graph)
            
            # Test graphs
            for i in range(len(X_test)):
                node_features = torch.zeros((n_buses, 1), dtype=torch.float)
                for j in range(min(X_test.shape[1], n_buses)):
                    node_features[j, 0] = X_test[i, j]
                
                graph = Data(x=node_features, edge_index=edge_index)
                graph.y = torch.tensor(y_test[i], dtype=torch.float).view(1, -1)
                test_graphs.append(graph)
            
            # Create data loaders
            train_loader = GraphDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
            val_loader = GraphDataLoader(val_graphs, batch_size=args.batch_size)
            test_loader = GraphDataLoader(test_graphs, batch_size=args.batch_size)
            
            logger.info(f"Created graph data loaders with {len(train_graphs)} training samples")
            
            # Create GNN model
            model = DirectPredictionGNN(
                node_features=1,  # One feature per node
                hidden_dim=args.hidden_dim,
                output_dim=y.shape[1],
                num_layers=3,
                dropout_rate=0.2
            ).to(device)
            
            logger.info(f"Created GNN model")
            
            # Train the model
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            logger.info(f"Starting training for {args.epochs} epochs")
            
            for epoch in range(args.epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    batch = batch.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch)
                    
                    # Unbatch outputs to match target shape
                    loss = criterion(outputs, batch.y.view(-1, y.shape[1]))
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch.num_graphs
                
                train_loss /= len(train_graphs)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        loss = criterion(outputs, batch.y.view(-1, y.shape[1]))
                        val_loss += loss.item() * batch.num_graphs
                
                val_loss /= len(val_graphs)
                
                logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Evaluate
            logger.info("Evaluating model")
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(batch.y.cpu().numpy())
            
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Calculate metrics
            mse = np.mean((all_preds - all_targets) ** 2)
            r2 = np.mean([r2_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])])
            
            logger.info(f"Test MSE: {mse:.6f}")
            logger.info(f"Test R²: {r2:.6f}")
            
            # Save results
            with open(os.path.join(args.output_dir, 'gnn_results.txt'), 'w') as f:
                f.write(f"Test MSE: {mse:.6f}\n")
                f.write(f"Test R²: {r2:.6f}\n")
            
            logger.info(f"Results saved to {os.path.join(args.output_dir, 'gnn_results.txt')}")
            
        except Exception as e:
            logger.error(f"Failed to run GNN: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info("Skipping GNN test due to missing PyTorch Geometric")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 