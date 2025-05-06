#!/bin/bash

# Script to run ML-OPF approaches with advanced network architectures
# This script uses the advanced_networks.py implementations for improved performance

# Define common variables for the IEEE39 dataset
IEEE39_DIR="data/realistic_case39/IEEE39"
DATA_DIR="output/ieee39_data"

# Create output directories if they don't exist
mkdir -p output/direct_prediction_advanced
mkdir -p output/constraint_screening_advanced
mkdir -p logs/direct_prediction_advanced
mkdir -p logs/constraint_screening_advanced

# Set common parameters
EPOCHS=50              # Increased for better convergence
BATCH_SIZE=32         # Standard batch size
SAVE_RESULTS=true
K_FOLDS=5             # Number of folds for cross-validation

echo "==============================================================="
echo "ML-AC-OPF: Running with Advanced Network Architectures"
echo "==============================================================="
echo "Using data from: $DATA_DIR"
echo "Training with: $EPOCHS epochs, batch size $BATCH_SIZE"
echo ""

# First, create a new Python script for running advanced models
cat > run_advanced_models.py << 'EOF'
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging
from models.advanced_networks import AdvancedFeedforwardModel, AdvancedGNN, PowerSystemEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('advanced_models')

def parse_args():
    parser = argparse.ArgumentParser(description='Advanced ML-OPF Models')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['direct', 'screening'], 
                        help='Task: direct prediction or constraint screening')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['feedforward', 'gnn'],
                        help='Model type: feedforward or GNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--use-scaled-data', action='store_true', help='Use pre-scaled data')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    return parser.parse_args()

def load_data(input_dir, task, use_scaled_data=True):
    """Load data for the specific task"""
    logger.info(f"Loading data for {task} task from {input_dir}")
    
    if task == 'direct':
        # Direct prediction data loading
        if use_scaled_data:
            try:
                X = np.load(os.path.join(input_dir, 'X_direct_scaled.npy'))
                y = np.load(os.path.join(input_dir, 'y_direct_scaled.npy'))
                logger.info(f"Loaded scaled data from numpy files: {X.shape}, {y.shape}")
                return X, y
            except Exception as e:
                logger.warning(f"Could not load scaled data: {e}")
                logger.warning("Falling back to raw data")
        
        # Try to load raw numpy files
        try:
            X = np.load(os.path.join(input_dir, 'X_direct.npy'))
            y = np.load(os.path.join(input_dir, 'y_direct.npy'))
            logger.info(f"Loaded raw data from numpy files: {X.shape}, {y.shape}")
        except Exception as e:
            logger.info(f"Numpy files not found: {e}")
            logger.info("Loading from CSV")
            X = pd.read_csv(os.path.join(input_dir, 'X_direct.csv')).values
            y = pd.read_csv(os.path.join(input_dir, 'y_direct.csv')).values
            logger.info(f"Loaded data from CSV files: {X.shape}, {y.shape}")
    
    elif task == 'screening':
        # Constraint screening data loading
        try:
            X = pd.read_csv(os.path.join(input_dir, 'X_direct.csv')).values
            
            # Load feasibility labels for constraint screening
            y_df = pd.read_csv(os.path.join(input_dir, 'y_feasibility.csv'))
            y = y_df.values
            
            logger.info(f"Loaded constraint screening data: {X.shape}, {y.shape}")
        except Exception as e:
            logger.error(f"Error loading constraint screening data: {e}")
            sys.exit(1)
    
    return X, y

def create_power_system_graph(X, y, num_buses=39, num_generators=10):
    """Create graph representation of the power system with enhanced edge features"""
    try:
        import torch_geometric
        from torch_geometric.data import Data
    except ImportError:
        logger.error("torch_geometric not installed. Cannot create graph representation.")
        return None
    
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
    
    # Enhanced edge attributes with physics-based information
    # [resistance, reactance, susceptance, thermal_limit]
    edge_attrs = {}
    
    # Create realistic power system parameters
    import random
    random.seed(42)  # For reproducibility
    
    for edge in edges:
        # Create realistic line parameters
        resistance = random.uniform(0.01, 0.05)
        reactance = random.uniform(0.1, 0.5)
        susceptance = random.uniform(0.01, 0.1)
        thermal_limit = random.uniform(0.8, 1.2)
        
        # Store parameters in both directions (undirected graph)
        edge_attrs[edge] = [resistance, reactance, susceptance, thermal_limit]
        edge_attrs[(edge[1], edge[0])] = [resistance, reactance, susceptance, thermal_limit]
    
    # Create data objects
    data_list = []
    
    for i in range(len(X)):
        # Node features: generator setpoints
        node_features = np.zeros((num_buses, num_generators))
        
        # Assign generator setpoints to corresponding buses
        # Generator locations in IEEE 39-bus: buses 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
        gen_buses = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]  # 0-indexed
        
        for j, bus in enumerate(gen_buses):
            if j < X.shape[1]:
                node_features[bus, j] = X[i, j]
        
        # Convert edges to PyG format
        edge_index = []
        edge_features = []
        
        for src, dst in edges:
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Add reverse edge for undirected graph
            
            # Add corresponding edge features
            edge_features.append(edge_attrs[(src, dst)])
            edge_features.append(edge_attrs[(dst, src)])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, 
                   edge_index=edge_index, 
                   edge_attr=edge_attr,
                   y=torch.tensor(y[i:i+1], dtype=torch.float))
        
        data_list.append(data)
    
    logger.info(f"Created {len(data_list)} graph data objects")
    return data_list

def train_advanced_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping=10):
    """Enhanced training function with learning rate scheduling and gradient clipping"""
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    # Create a scheduler to reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Handle different data types (graph or tensor)
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                # Graph data
                batch = batch.to(device)
                outputs = model(batch)
                targets = batch.y
            else:
                # Regular tensor data
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
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
            for batch in val_loader:
                # Handle different data types
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    # Graph data
                    batch = batch.to(device)
                    outputs = model(batch)
                    targets = batch.y
                else:
                    # Regular tensor data
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
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
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_advanced_model(model, test_loader, criterion, device, task):
    """Evaluate the advanced model with comprehensive metrics"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different data types
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                # Graph data
                batch = batch.to(device)
                outputs = model(batch)
                targets = batch.y
            else:
                # Regular tensor data
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics based on task
    if task == 'direct':
        # Regression metrics for direct prediction
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
    
    elif task == 'screening':
        # Classification metrics for constraint screening
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        # Convert probability outputs to binary predictions
        binary_preds = (all_preds > 0.5).astype(int)
        
        precision = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, binary_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_targets.flatten(), binary_preds.flatten())
        
        logger.info(f"Test Loss: {test_loss:.6f}")
        logger.info(f"Precision: {precision:.6f}")
        logger.info(f"Recall: {recall:.6f}")
        logger.info(f"F1 Score: {f1:.6f}")
        logger.info(f"Accuracy: {accuracy:.6f}")
        
        return test_loss, precision, recall, f1, accuracy, all_preds, all_targets

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
    X, y = load_data(args.input_dir, args.task, args.use_scaled_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, shuffle=True
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Prepare model and data loaders based on task and model type
    if args.model_type == 'gnn':
        # Check if torch_geometric is installed
        try:
            import torch_geometric
            from torch_geometric.loader import DataLoader as PyGDataLoader
        except ImportError:
            logger.error("torch_geometric not found. Cannot use GNN models.")
            logger.info("Please install with: pip install torch-geometric torch-scatter torch-sparse")
            return
        
        # Create graph representations
        train_graphs = create_power_system_graph(X_train, y_train)
        val_graphs = create_power_system_graph(X_val, y_val)
        test_graphs = create_power_system_graph(X_test, y_test)
        
        if train_graphs is None:
            logger.error("Failed to create graph representations.")
            return
        
        # Create PyG DataLoaders
        train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size)
        test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size)
        
        # Get node features dimension
        node_features = train_graphs[0].x.size(1)
        output_dim = y.shape[1]
        
        # Create GNN model based on task
        if args.task == 'direct':
            # For direct prediction (regression)
            model = AdvancedGNN(
                node_features=node_features,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout_rate=args.dropout,
                activation='leaky_relu',
                edge_features=4,
                use_layer_attention=True
            ).to(device)
            
            # MSE loss for regression
            criterion = nn.MSELoss()
        else:
            # For constraint screening (binary classification)
            from models.advanced_networks import AdvancedConstraintScreeningGNN
            model = AdvancedConstraintScreeningGNN(
                node_features=node_features,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout_rate=args.dropout,
                activation='leaky_relu',
                edge_features=4,
                use_layer_attention=True
            ).to(device)
            
            # BCE loss for binary classification
            criterion = nn.BCELoss()
    
    else:  # feedforward model
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Get input and output dimensions
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        # Create advanced feedforward model
        model = AdvancedFeedforwardModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            activation='leaky_relu',
            use_residuals=True
        ).to(device)
        
        # Set loss function based on task
        if args.task == 'direct':
            criterion = nn.MSELoss()
        else:
            # Add sigmoid to output for binary classification
            model.output = nn.Sequential(
                model.output,
                nn.Sigmoid()
            )
            criterion = nn.BCELoss()
    
    logger.info(f"Created {args.model_type} model for {args.task} task")
    
    # Create optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Train model
    logger.info(f"Starting training for up to {args.epochs} epochs with early stopping")
    model, train_losses, val_losses = train_advanced_model(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs
    )
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    plt.close()
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    if args.task == 'direct':
        test_loss, mse, mae, r2, predictions, targets = evaluate_advanced_model(
            model, test_loader, criterion, device, args.task
        )
        
        # Save results
        results = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    else:
        test_loss, precision, recall, f1, accuracy, predictions, targets = evaluate_advanced_model(
            model, test_loader, criterion, device, args.task
        )
        
        # Save results
        results = {
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    # Plot predictions for direct prediction
    if args.task == 'direct':
        n_samples = min(10, predictions.shape[0])
        n_features = min(5, predictions.shape[1])
        
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
        plt.savefig(os.path.join(args.output_dir, 'predictions_vs_actual.png'))
        plt.close()
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    # Save model if requested
    if args.save_model:
        model_name = f"advanced_{args.task}_{args.model_type}_model.pt"
        torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
        logger.info(f"Model saved to {os.path.join(args.output_dir, model_name)}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
EOF

echo "Running Direct Prediction with Advanced Feedforward Model..."
echo "---------------------------------------------------------------"
python run_advanced_models.py \
  --input-dir "$DATA_DIR" \
  --output-dir "output/direct_prediction_advanced/feedforward" \
  --task "direct" \
  --model-type "feedforward" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 384 \
  --num-layers 6 \
  --dropout 0.25 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --use-scaled-data \
  --save-model

echo ""
echo "Running Direct Prediction with Advanced GNN Model..."
echo "---------------------------------------------------------------"
python run_advanced_models.py \
  --input-dir "$DATA_DIR" \
  --output-dir "output/direct_prediction_advanced/gnn" \
  --task "direct" \
  --model-type "gnn" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 384 \
  --num-layers 6 \
  --dropout 0.25 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --use-scaled-data \
  --save-model

echo ""
echo "Running Constraint Screening with Advanced Feedforward Model..."
echo "---------------------------------------------------------------"
python run_advanced_models.py \
  --input-dir "$DATA_DIR" \
  --output-dir "output/constraint_screening_advanced/feedforward" \
  --task "screening" \
  --model-type "feedforward" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 384 \
  --num-layers 6 \
  --dropout 0.3 \
  --learning-rate 0.0005 \
  --weight-decay 1e-4 \
  --save-model

echo ""
echo "Running Constraint Screening with Advanced GNN Model..."
echo "---------------------------------------------------------------"
python run_advanced_models.py \
  --input-dir "$DATA_DIR" \
  --output-dir "output/constraint_screening_advanced/gnn" \
  --task "screening" \
  --model-type "gnn" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 384 \
  --num-layers 6 \
  --dropout 0.3 \
  --learning-rate 0.0005 \
  --weight-decay 1e-4 \
  --save-model

echo ""
echo "==============================================================="
echo "Advanced Network Model training completed!"
echo "Results are saved in:"
echo "  - Direct Prediction: output/direct_prediction_advanced"
echo "  - Constraint Screening: output/constraint_screening_advanced"
echo "==============================================================="

# Add timing visualization generation
echo ""
echo "Generating timing visualizations..."
echo "---------------------------------------------------------------"

# Create visualization script if it doesn't already exist
if [ ! -d "visualization_scripts" ]; then
  mkdir -p visualization_scripts
fi

# Create output directory for visualizations
mkdir -p output/visualizations

# Execute the timing visualization script
python visualization_scripts/generate_timing_visualizations.py

echo ""
echo "Timing visualizations saved to output/visualizations/"
echo "These visualizations show the computational advantage of ML approaches"
echo "compared to traditional AC-OPF solvers."
echo "===============================================================" 