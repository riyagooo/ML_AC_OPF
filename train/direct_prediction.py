import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import networkx as nx

# Import PyTorch Geometric modules for GNN
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from models.gnn import EnhancedDirectPredictionGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("Warning: torch_geometric not found. GNN models will not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('direct_prediction')

def parse_args():
    parser = argparse.ArgumentParser(description='Direct Prediction ML-OPF')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with X and y data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--use-scaled-data', action='store_true', help='Use pre-scaled data')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--use-gnn', action='store_true', help='Use Graph Neural Networks')
    return parser.parse_args()

class DirectPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout_rate=0.2):
        super(DirectPredictionModel, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DirectPredictionGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=4, dropout_rate=0.2):
        super(DirectPredictionGNN, self).__init__()
        
        # Input embedding layer
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output prediction layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """
        Forward pass for GNN
        Args:
            data: PyTorch Geometric Data object with node features and graph structure
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Graph convolution
            x = conv(x, edge_index)
            # Batch normalization
            x = bn(x)
            # Activation and dropout
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Global pooling - aggregate node features to graph level
        # If batch information is available, use it
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            # For single graph, take mean of all nodes
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction
        x = self.output_layer(x)
        
        return x

def load_data(input_dir, use_scaled_data=True):
    # Try to load scaled numpy files first if use_scaled_data is True
    if use_scaled_data:
        try:
            X = np.load(os.path.join(input_dir, 'X_direct_scaled.npy'))
            y = np.load(os.path.join(input_dir, 'y_direct_scaled.npy'))
            logger.info(f"Loaded scaled data from numpy files: {X.shape}, {y.shape}")
            return X, y
        except:
            logger.warning("Could not load scaled data, falling back to raw data")
    
    # Try to load raw numpy files
    try:
        X = np.load(os.path.join(input_dir, 'X_direct.npy'))
        y = np.load(os.path.join(input_dir, 'y_direct.npy'))
        logger.info(f"Loaded raw data from numpy files: {X.shape}, {y.shape}")
    except:
        logger.info("Numpy files not found, loading from CSV")
        X = pd.read_csv(os.path.join(input_dir, 'X_direct.csv')).values
        y = pd.read_csv(os.path.join(input_dir, 'y_direct.csv')).values
        logger.info(f"Loaded data from CSV files: {X.shape}, {y.shape}")
    
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=10):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
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
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Return losses for plotting
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Calculate R2 score for each dimension
    r2_values = []
    for i in range(all_targets.shape[1]):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        r2_values.append(r2)
    
    # Average R2 score
    avg_r2 = np.mean(r2_values)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"Avg R^2: {avg_r2:.6f}")
    
    # Log individual R2 scores
    for i, r2 in enumerate(r2_values):
        logger.info(f"R^2 for output {i+1}: {r2:.6f}")
    
    return test_loss, mse, mae, avg_r2, all_preds, all_targets

def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

def plot_predictions(predictions, targets, output_dir, max_samples=10):
    # Select a random set of samples and features to visualize
    n_samples = min(max_samples, predictions.shape[0])
    n_features = min(5, predictions.shape[1])  # Show first 5 features at most
    
    plt.figure(figsize=(12, 10))
    
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        
        # Select random samples
        indices = np.random.choice(predictions.shape[0], n_samples, replace=False)
        
        # Sort by target value for better visualization
        sorted_indices = np.argsort(targets[indices, i])
        sorted_targets = targets[indices[sorted_indices], i]
        sorted_preds = predictions[indices[sorted_indices], i]
        
        # Plot predictions vs targets
        plt.plot(range(n_samples), sorted_targets, 'b-', marker='o', label='Actual', markersize=5)
        plt.plot(range(n_samples), sorted_preds, 'r-', marker='x', label='Predicted', markersize=5)
        
        plt.title(f'Output Feature {i+1} - Predictions vs Actual')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'))
    plt.close()

def create_power_system_graph(X, num_buses=39, num_generators=10):
    """
    Create a graph representation of IEEE 39-bus power system
    
    Args:
        X: Input features (generator setpoints)
        num_buses: Number of buses in the system
        num_generators: Number of generators
        
    Returns:
        List of PyG Data objects representing the power system graphs
    """
    # Define the IEEE 39-bus system topology as an adjacency list
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7), (6, 7), (6, 8),
        (7, 9), (8, 9), (8, 10), (9, 11), (10, 11), (10, 12), (11, 13), (12, 13), (12, 14),
        (13, 15), (14, 15), (14, 16), (15, 17), (16, 17), (16, 18), (17, 19), (18, 19),
        (18, 20), (19, 21), (20, 21), (20, 22), (21, 23), (22, 23), (22, 24), (23, 25),
        (24, 25), (24, 26), (25, 27), (26, 27), (26, 28), (27, 29), (28, 29), (28, 30),
        (29, 31), (30, 31), (30, 32), (31, 33), (32, 33), (32, 34), (33, 35), (34, 35),
        (34, 36), (35, 37), (36, 37), (36, 38), (37, 38)
    ]
    
    # Define the electrical properties of the lines (simplified)
    # [resistance, reactance, charging susceptance, thermal limit]
    edge_attrs = {
        (0, 1): [0.01, 0.1, 0.02, 1.0],
        (0, 2): [0.02, 0.2, 0.04, 1.0],
        (1, 3): [0.03, 0.3, 0.06, 1.0],
        # ... Add more line parameters for all edges
    }
    
    # For edges without defined attributes, use default values
    default_attr = [0.02, 0.2, 0.04, 1.0]
    
    # Create a base graph
    G = nx.Graph()
    G.add_nodes_from(range(num_buses))
    G.add_edges_from(edges)
    
    # Create PyG Data objects for each sample
    data_list = []
    
    for i in range(len(X)):
        # Copy the base graph for this sample
        sample_graph = G.copy()
        
        # Add generator setpoints as node features
        # We'll set all buses to have zero features by default
        node_features = np.zeros((num_buses, num_generators))
        
        # Assign generator setpoints to corresponding buses
        # Generator locations in IEEE 39-bus: buses 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
        gen_buses = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]  # 0-indexed
        
        for j, bus in enumerate(gen_buses):
            if j < X.shape[1]:
                node_features[bus, j] = X[i, j]
        
        # Convert to PyG Data
        edge_index = torch.tensor(list(sample_graph.edges())).t().contiguous()
        
        # Create edge attributes tensor
        edge_features = []
        for src, dst in sample_graph.edges():
            edge_features.append(edge_attrs.get((src, dst), default_attr))
            edge_features.append(edge_attrs.get((dst, src), default_attr))  # For reverse direction
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create Data object with node and edge features
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    
    return data_list

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GNN is requested but not available
    if args.use_gnn and not GNN_AVAILABLE:
        logger.warning("GNN requested but torch_geometric not available. Falling back to standard model.")
        args.use_gnn = False
    
    # Log which model architecture is being used
    if args.use_gnn:
        logger.info("Using Graph Neural Network architecture")
    else:
        logger.info("Using standard Feedforward Neural Network architecture")
    
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
    
    # Create model and loaders based on architecture
    if args.use_gnn:
        # Create graph representation
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
    else:
        # Standard DataLoaders for feedforward model
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Create standard model
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        model = DirectPredictionModel(
            input_dim, 
            output_dim, 
            args.hidden_dim, 
            args.num_layers,
            args.dropout
        ).to(device)
        
        logger.info(f"Created model with {input_dim} inputs, {output_dim} outputs, " 
                   f"{args.hidden_dim} hidden dim, {args.num_layers} layers, {args.dropout} dropout")
    
    # Create optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Train model using the appropriate training function
    logger.info(f"Starting training for up to {args.epochs} epochs with early stopping")
    
    if args.use_gnn:
        from models.gnn import train_model_gnn
        model, train_losses, val_losses = train_model_gnn(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        )
    else:
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        )
    
    # Plot learning curves
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    
    if args.use_gnn:
        from models.gnn import evaluate_model_gnn
        test_loss, mse, mae, r2, predictions, targets = evaluate_model_gnn(
            model, test_loader, criterion, device
        )
    else:
        test_loss, mse, mae, r2, predictions, targets = evaluate_model(
            model, test_loader, criterion, device
        )
    
    # Plot predictions
    plot_predictions(predictions, targets, args.output_dir)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'direct_prediction_model.pt'))
        logger.info(f"Model saved to {os.path.join(args.output_dir, 'direct_prediction_model.pt')}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
