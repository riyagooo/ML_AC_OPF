#!/usr/bin/env python
"""
Comparison test for case30 GNN model performance.
This script compares the standard and robust GNN approaches for case30.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# Import project modules
from utils.data_utils import load_pglib_data, load_case_network, normalize_opf_data, denormalize_opf_data
from utils.training import Trainer, RobustLoss
from models.gnn import TopologyAwareGNN, RobustTopologyAwareGNN, prepare_pyg_data

def create_graph_data(case_data, data=None):
    """Create graph-based dataset from case_data and load patterns."""
    # Create a NetworkX graph from case_data
    G = nx.Graph()
    
    # Create a mapping from bus ID to zero-based indices
    bus_ids = [int(bus[0]) for bus in case_data['bus']]
    bus_id_to_idx = {bus_id: i for i, bus_id in enumerate(bus_ids)}
    
    # Add nodes (buses)
    for i, bus in enumerate(case_data['bus']):
        bus_id = int(bus[0])  # Bus ID - original ID from the system
        idx = i  # Zero-based index
        
        # Create a dictionary of all available attributes
        node_attrs = {
            'type': int(bus[1]),  # Bus type
            'pd': float(bus[2]),  # Active power demand
            'qd': float(bus[3]),  # Reactive power demand
            'gs': float(bus[4]),  # Shunt conductance
            'bs': float(bus[5]),  # Shunt susceptance
            'area': int(bus[6]),  # Area number
            'vm': float(bus[7]),  # Voltage magnitude
            'va': float(bus[8]),  # Voltage angle
            'baseKV': float(bus[9]),  # Base voltage
            'zone': int(bus[10]),  # Zone
            'vmax': float(bus[11]),  # Max voltage
            'vmin': float(bus[12]),   # Min voltage
            'original_id': bus_id     # Store the original ID for reference
        }
        
        # Add node with index as the node ID (not the bus ID)
        G.add_node(idx, **node_attrs)
    
    # Add edges (branches)
    for i, branch in enumerate(case_data['branch']):
        from_bus = int(branch[0])
        to_bus = int(branch[1])
        
        # Convert to zero-based indices
        if from_bus in bus_id_to_idx and to_bus in bus_id_to_idx:
            from_idx = bus_id_to_idx[from_bus]
            to_idx = bus_id_to_idx[to_bus]
            
            # Create a dictionary of all available edge attributes
            edge_attrs = {
                'r': float(branch[2]),  # Resistance
                'x': float(branch[3]),  # Reactance
                'b': float(branch[4]),  # Line charging susceptance
                'rateA': float(branch[5]),  # MVA rating A
                'rateB': float(branch[6]),  # MVA rating B
                'rateC': float(branch[7]),  # MVA rating C
                'ratio': float(branch[8]),  # Transformer tap ratio
                'angle': float(branch[9]),  # Transformer phase shift
                'status': int(branch[10]),  # Initial branch status
                'angmin': float(branch[11]),  # Minimum angle difference
                'angmax': float(branch[12])   # Maximum angle difference
            }
            
            G.add_edge(from_idx, to_idx, **edge_attrs)
        else:
            print(f"Warning: Branch {i} connects non-existent buses {from_bus} and {to_bus}")
    
    # Define node features to use
    node_features = ['pd', 'qd', 'vm', 'va', 'vmax', 'vmin']
    
    # Define edge features to use
    edge_features = ['r', 'x', 'b', 'rateA']
    
    # Check if all node features exist
    for node in G.nodes():
        for feat in node_features:
            if feat not in G.nodes[node]:
                print(f"Warning: Feature '{feat}' missing for node {node}, setting to 0.0")
                G.nodes[node][feat] = 0.0
    
    # Check if all edge features exist
    for u, v in G.edges():
        for feat in edge_features:
            if feat not in G[u][v]:
                print(f"Warning: Feature '{feat}' missing for edge {u}-{v}, setting to 0.0")
                G[u][v][feat] = 0.0
    
    # Convert to PyTorch Geometric data
    data_obj = prepare_pyg_data(G, node_features, edge_features)
    
    return data_obj, G, node_features, edge_features

def create_node_features(G, node_features, load_data):
    """Create node features for the graph from load data."""
    # Create node feature matrix
    n_nodes = len(G.nodes())
    n_features = len(node_features)
    x = np.zeros((n_nodes, n_features))
    
    # Map node IDs to indices
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    
    # Fill in static features from graph
    for node in G.nodes():
        idx = node_to_idx[node]
        for i, feat in enumerate(node_features):
            if feat != 'pd' and feat != 'qd':  # Keep static features
                x[idx, i] = G.nodes[node][feat]
    
    # Update load features from load_data
    for i, (pd, qd) in enumerate(load_data):
        node = list(G.nodes())[i]  # Assume nodes are in same order as load_data
        idx = node_to_idx[node]
        pd_idx = node_features.index('pd')
        qd_idx = node_features.index('qd')
        x[idx, pd_idx] = pd
        x[idx, qd_idx] = qd
    
    return torch.tensor(x, dtype=torch.float32)

def train_standard_gnn(train_data, val_data, node_features, edge_features, output_dim):
    """Train a standard GNN model on case30 data."""
    print("\n=== Training Standard GNN Model ===")
    
    # Create model
    model = TopologyAwareGNN(
        node_features=len(node_features),
        edge_features=len(edge_features),
        hidden_channels=64,
        output_dim=output_dim,
        num_layers=3,
        dropout_rate=0.1
    )
    
    # Custom train and validate functions for PyG data
    def train_epoch(model, optimizer, criterion, train_data):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for data in train_data:
            # Forward pass
            pred = model(data)
            loss = criterion(pred, data.y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        return total_loss / batch_count
    
    def validate(model, criterion, val_data):
        model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for data in val_data:
                # Forward pass
                pred = model(data)
                loss = criterion(pred, data.y)
                
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Train model
    print("Training standard GNN model...")
    train_losses = []
    val_losses = []
    epochs = 10
    
    os.makedirs('logs/standard_gnn_model', exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, criterion, train_data)
        
        # Validate
        val_loss = validate(model, criterion, val_data)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'logs/standard_gnn_model/best_model.pt')
            print("  Saved best model")
    
    # Save history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

def train_robust_gnn(train_data, val_data, node_features, edge_features, output_dim, output_bounds):
    """Train a robust GNN model on case30 data."""
    print("\n=== Training Robust GNN Model ===")
    
    # Create model
    model = RobustTopologyAwareGNN(
        node_features=len(node_features),
        edge_features=len(edge_features),
        hidden_channels=64,
        output_dim=output_dim,
        num_layers=3,
        dropout_rate=0.1,
        output_bounds=output_bounds
    )
    
    # Custom train and validate functions for PyG data
    def train_epoch(model, optimizer, criterion, train_data):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for data in train_data:
            # Forward pass
            pred = model(data)
            loss = criterion(pred, data.y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        return total_loss / batch_count
    
    def validate(model, criterion, val_data):
        model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for data in val_data:
                # Forward pass
                pred = model(data)
                loss = criterion(pred, data.y)
                
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = RobustLoss()
    
    # Train model
    print("Training robust GNN model...")
    train_losses = []
    val_losses = []
    epochs = 10
    
    os.makedirs('logs/robust_gnn_model', exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, criterion, train_data)
        
        # Validate
        val_loss = validate(model, criterion, val_data)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'logs/robust_gnn_model/best_model.pt')
            print("  Saved best model")
    
    # Save history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

def create_data_loaders(data_list, batch_size=32):
    """Create PyTorch data loaders from list of data objects."""
    # Split into train/val/test
    n_samples = len(data_list)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]
    
    # Create data loaders (for PyTorch Geometric, we just use lists)
    train_loader = train_data
    val_loader = val_data
    test_loader = test_data
    
    return train_loader, val_loader, test_loader

def evaluate_models(standard_model, robust_model, test_data_standard, test_data_robust, output_cols, norm_params=None):
    """Evaluate both models on test data."""
    print("\n=== Evaluating Models ===")
    
    # Set models to evaluation mode
    standard_model.eval()
    robust_model.eval()
    
    # Lists to store predictions for visualization
    all_targets = []
    standard_preds = []
    robust_preds = []
    robust_preds_denorm = []
    
    # Create MSE loss function for evaluation
    mse_loss = torch.nn.MSELoss()
    
    # Calculate standard model MSE
    standard_losses = []
    for data in test_data_standard:
        with torch.no_grad():
            # Extract target
            target = data.y
            
            # Get prediction
            pred = standard_model(data)
            
            # Calculate loss
            loss = mse_loss(pred, target)
            standard_losses.append(loss.item())
            
            # Store predictions and targets
            all_targets.append(target.cpu().numpy())
            standard_preds.append(pred.cpu().numpy())
    
    # Calculate robust model MSE
    robust_losses = []
    for data in test_data_robust:
        with torch.no_grad():
            # Extract target
            target = data.y
            
            # Get prediction
            pred = robust_model(data)
            
            # Calculate loss
            loss = mse_loss(pred, target)
            robust_losses.append(loss.item())
            
            # Store predictions
            robust_preds.append(pred.cpu().numpy())
            
            # Denormalize if needed
            if norm_params is not None:
                # Create dataframes for denormalization
                pred_df = pd.DataFrame(pred.cpu().numpy(), columns=output_cols)
                
                # Denormalize predictions
                denorm_pred_df = denormalize_opf_data(pred_df, norm_params)
                
                # Store denormalized predictions
                robust_preds_denorm.append(denorm_pred_df.values)
    
    # Calculate average losses
    standard_mse = np.mean(standard_losses)
    robust_mse = np.mean(robust_losses)
    
    print(f"Standard GNN model MSE: {standard_mse:.6f}")
    print(f"Robust GNN model MSE: {robust_mse:.6f}")
    
    # Concatenate predictions if there are multiple samples
    if all_targets:
        all_targets = np.concatenate(all_targets)
        standard_preds = np.concatenate(standard_preds)
    
    if robust_preds:
        robust_preds = np.concatenate(robust_preds)
    
    if robust_preds_denorm:
        robust_preds_denorm = np.concatenate(robust_preds_denorm)
    
    # Print sample predictions for both models
    print("\n=== Sample Predictions ===")
    for i in range(min(3, len(all_targets))):  # Show up to 3 samples
        print(f"\nSample {i+1}:")
        print("Target values:")
        for j in range(min(len(output_cols), all_targets.shape[1])):
            print(f"  {output_cols[j]}: {all_targets[i, j]:.6f}")
        
        print("\nStandard model predictions:")
        for j in range(min(len(output_cols), standard_preds.shape[1])):
            print(f"  {output_cols[j]}: {standard_preds[i, j]:.6f}")
        
        if len(robust_preds_denorm) > 0:
            print("\nRobust model predictions (denormalized):")
            for j in range(min(len(output_cols), robust_preds_denorm.shape[1])):
                print(f"  {output_cols[j]}: {robust_preds_denorm[i, j]:.6f}")
    
    # Create comparison plots
    # Only create plot if we have compatible dimensions
    if len(all_targets) > 0 and len(standard_preds) > 0 and len(robust_preds_denorm) > 0:
        plot_prediction_comparison(all_targets, standard_preds, robust_preds_denorm, output_cols,
                                  title="GNN Predictions (Standard vs Robust)")
    
    return standard_mse, robust_mse

def plot_prediction_comparison(targets, standard_preds, robust_preds, output_cols, 
                             title="GNN Prediction Comparison", save_path="logs/gnn_prediction_comparison.png"):
    """Plot comparison of predictions from both models."""
    n_outputs = min(5, targets.shape[1])  # Plot at most 5 outputs
    
    plt.figure(figsize=(15, 10))
    
    for i in range(n_outputs):
        plt.subplot(2, 3, i + 1)
        
        # Extract predictions and targets for this output
        t = targets[:, i]
        s = standard_preds[:, i] 
        r = robust_preds[:, i]
        
        # Plot standard model predictions
        plt.scatter(t, s, alpha=0.5, color='blue', label='Standard GNN')
        
        # Plot robust model predictions 
        plt.scatter(t, r, alpha=0.5, color='red', label='Robust GNN')
        
        # Add diagonal line
        buffer = (np.max(t) - np.min(t)) * 0.1
        plt.plot([np.min(t) - buffer, np.max(t) + buffer], 
                [np.min(t) - buffer, np.max(t) + buffer], 'k--')
        
        plt.xlabel('Target')
        plt.ylabel('Prediction')
        plt.title(f'Output {i+1}: {output_cols[i]}')
        
        if i == 0:
            plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction comparison plot saved to {save_path}")

def plot_comparison(standard_history, robust_history, save_path="logs/gnn_model_comparison.png"):
    """Plot training history comparison."""
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(standard_history['train_loss'], label='Standard - Train')
    plt.plot(robust_history['train_loss'], label='Robust - Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.yscale('log')
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(standard_history['val_loss'], label='Standard - Val')
    plt.plot(robust_history['val_loss'], label='Robust - Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")

def prepare_data(case_data, data_df, input_cols, output_cols, G, node_features, edge_features):
    """Prepare data for GNN models."""
    # Create data objects list
    data_list = []
    
    # Check if we have data
    if len(data_df) == 0:
        print("Warning: No data samples found.")
        return []
    
    # Create dummy load data for the base graph (we'll update it for each sample)
    dummy_loads = np.zeros((len(G.nodes()), 2))
    
    # Get base graph data object
    base_data, _, _, _ = create_graph_data(case_data, dummy_loads)
    
    # Iterate through samples
    for i in range(len(data_df)):
        try:
            # Get load data
            if input_cols:
                # Try to reshape based on network size
                load_values = data_df[input_cols].iloc[i].values
                n_buses = len(G.nodes())
                
                # Check if we need to reshape or pad the load data
                if len(load_values) < 2 * n_buses:
                    # Pad with zeros if not enough values
                    padded = np.zeros((n_buses, 2))
                    for j in range(min(len(load_values) // 2, n_buses)):
                        padded[j, 0] = load_values[j*2]  # P load
                        if j*2+1 < len(load_values):
                            padded[j, 1] = load_values[j*2+1]  # Q load
                    load_data = padded
                else:
                    # Reshape to match format [bus, (p, q)]
                    load_data = load_values.reshape(-1, 2)
                    
                    # Trim if too large
                    if len(load_data) > n_buses:
                        load_data = load_data[:n_buses]
            else:
                # No input columns, use dummy loads
                load_data = dummy_loads
            
            # Create node features
            x = create_node_features(G, node_features, load_data)
            
            # Copy base data
            data = base_data.clone()
            
            # Update node features
            data.x = x
            
            # Add target values
            data.y = torch.tensor(data_df[output_cols].iloc[i].values, dtype=torch.float32).unsqueeze(0)
            
            # Add to list
            data_list.append(data)
            
        except Exception as e:
            print(f"Error preparing sample {i}: {e}")
    
    return data_list

def main():
    """Main function for comparing models."""
    print("Comparing standard and robust GNN approaches for case30")
    
    # Create log directory
    os.makedirs('logs/standard_gnn_model', exist_ok=True)
    os.makedirs('logs/robust_gnn_model', exist_ok=True)
    
    # Load case30 data
    case_data = load_case_network('case30', 'data')
    data = load_pglib_data('case30', 'data')
    
    # Use a small subset for quick testing
    data_sample = data.head(2000)
    
    # Auto-detect input/output columns
    load_cols = [col for col in data.columns if col.startswith('load') and (':pl' in col or ':ql' in col)]
    gen_cols = [col for col in data.columns if col.startswith('gen') and ':pg' in col]
    
    # Use a more comprehensive set of outputs
    input_cols = load_cols[:15]  # First 15 load columns
    output_cols = gen_cols[:10]  # First 10 generator columns
    
    print(f"Using {len(input_cols)} input features and {len(output_cols)} output features")
    
    # Create graph data
    base_data, G, node_features, edge_features = create_graph_data(case_data, None)
    
    # Prepare standard data
    standard_data_list = prepare_data(case_data, data_sample, input_cols, output_cols, G, node_features, edge_features)
    
    # Normalize data for robust model
    normalized_data, norm_params = normalize_opf_data(
        data_sample, case_data, input_cols, output_cols)
    
    # Prepare robust data
    robust_data_list = prepare_data(case_data, normalized_data, input_cols, output_cols, G, node_features, edge_features)
    
    # Create data loaders
    standard_train_loader, standard_val_loader, standard_test_loader = create_data_loaders(standard_data_list)
    robust_train_loader, robust_val_loader, robust_test_loader = create_data_loaders(robust_data_list)
    
    # Create output bounds for robust model
    output_dim = len(output_cols)
    min_bounds = torch.tensor([normalized_data[col].min() for col in output_cols])
    max_bounds = torch.tensor([normalized_data[col].max() for col in output_cols])
    output_bounds = (min_bounds, max_bounds)
    
    # Train standard model
    standard_model, standard_history = train_standard_gnn(
        standard_train_loader, standard_val_loader, 
        node_features, edge_features, len(output_cols))
    
    # Train robust model
    robust_model, robust_history = train_robust_gnn(
        robust_train_loader, robust_val_loader,
        node_features, edge_features, len(output_cols), output_bounds)
    
    # Evaluate models
    evaluate_models(standard_model, robust_model, standard_test_loader, 
                   robust_test_loader, output_cols, norm_params)
    
    # Plot comparison
    plot_comparison(standard_history, robust_history)
    
    print("\nComparison completed! Check logs directory for results.")

if __name__ == "__main__":
    main() 