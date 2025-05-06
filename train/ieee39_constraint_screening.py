import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('constraint_screening')

def parse_args():
    parser = argparse.ArgumentParser(description='IEEE39 Constraint Screening')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with processed data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=str, default='128,256,128', help='Hidden dimensions (comma-separated)')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--use-gnn', action='store_true', help='Use Graph Neural Networks')
    return parser.parse_args()

class ConstraintScreeningModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.3):
        super(ConstraintScreeningModel, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # For binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ConstraintScreeningGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(ConstraintScreeningGNN, self).__init__()
        # GNN architecture with sigmoid output for binary classification
        # ...

def load_data(input_dir):
    # Load the preprocessed data
    # For constraint screening, we use generator setpoints as inputs and feasibility as target
    X = pd.read_csv(os.path.join(input_dir, 'X_direct.csv'))
    
    # Load feasibility labels
    try:
        y = pd.read_csv(os.path.join(input_dir, 'y_feasibility.csv'))
        logger.info(f"Loaded feasibility data: {len(y)} samples")
    except:
        logger.error(f"Could not load feasibility data from {input_dir}")
        sys.exit(1)
    
    # Ensure X and y have the same length
    if len(X) != len(y):
        logger.warning(f"X and y lengths don't match: {len(X)} vs {len(y)}")
        # Use the smaller length
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        logger.info(f"Adjusted data to {min_len} samples")
    
    return X.values, y.values

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping=10):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []
        
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
            
            # Store predictions and targets for metrics
            all_preds.append((outputs > 0.5).float().cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        train_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append((outputs > 0.5).float().cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        val_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Train precision: {train_precision:.6f}")
            logger.info(f"  Train recall: {train_recall:.6f}")
            logger.info(f"  Train f1: {train_f1:.6f}")
            logger.info(f"  Val precision: {val_precision:.6f}")
            logger.info(f"  Val recall: {val_recall:.6f}")
            logger.info(f"  Val f1: {val_f1:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info("  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_outputs = []  # To store raw outputs for ROC curve
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # Store predictions, raw outputs and targets
            all_preds.append((outputs > 0.5).float().cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.vstack(all_preds)
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for all constraints together
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())
    
    logger.info(f"Test Results:")
    logger.info(f"  Precision: {precision:.6f}")
    logger.info(f"  Recall: {recall:.6f}")
    logger.info(f"  F1 Score: {f1:.6f}")
    logger.info(f"  Accuracy: {accuracy:.6f}")
    
    # Calculate metrics for each constraint
    n_constraints = all_targets.shape[1]
    constraint_metrics = []
    
    for i in range(n_constraints):
        constraint_precision = precision_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        constraint_recall = recall_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        constraint_f1 = f1_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        constraint_accuracy = accuracy_score(all_targets[:, i], all_preds[:, i])
        constraint_binding_rate = all_targets[:, i].mean()
        
        logger.info(f"  Constraint {i}: Prec={constraint_precision:.4f}, Rec={constraint_recall:.4f}, F1={constraint_f1:.4f}")
        
        constraint_metrics.append({
            'constraint': f'binding_{i}',
            'precision': constraint_precision,
            'recall': constraint_recall,
            'f1': constraint_f1,
            'accuracy': constraint_accuracy,
            'binding_rate': constraint_binding_rate
        })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(constraint_metrics)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'per_constraint': metrics_df,
        'predictions': all_preds,
        'targets': all_targets,
        'raw_outputs': all_outputs
    }

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

def plot_confusion_matrices(predictions, targets, output_dir):
    # Create a confusion matrix for each constraint
    n_constraints = targets.shape[1]
    
    plt.figure(figsize=(15, n_constraints * 4))
    
    for i in range(n_constraints):
        cm = confusion_matrix(targets[:, i], predictions[:, i])
        
        plt.subplot(n_constraints, 2, i*2+1)
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Constraint {i+1} Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Not Binding', 'Binding'])
        plt.yticks([0, 1], ['Not Binding', 'Binding'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Display the values in the confusion matrix
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                plt.text(k, j, str(cm[j, k]), 
                        horizontalalignment='center', 
                        verticalalignment='center')
        
        # Plot the distribution of predictions
        plt.subplot(n_constraints, 2, i*2+2)
        labels = ['Not Binding', 'Binding']
        actual_counts = [np.sum(targets[:, i] == 0), np.sum(targets[:, i] == 1)]
        predicted_counts = [np.sum(predictions[:, i] == 0), np.sum(predictions[:, i] == 1)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, actual_counts, width, label='Actual')
        plt.bar(x + width/2, predicted_counts, width, label='Predicted')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'Constraint {i+1} Distribution')
        plt.xticks(x, labels)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

def plot_roc_curves(targets, raw_outputs, output_dir):
    from sklearn.metrics import roc_curve, auc
    
    n_constraints = targets.shape[1]
    plt.figure(figsize=(12, 8))
    
    for i in range(n_constraints):
        fpr, tpr, _ = roc_curve(targets[:, i], raw_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'Constraint {i+1} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Check if GNN is requested but not available
    if args.use_gnn and not GNN_AVAILABLE:
        logger.warning("GNN requested but torch_geometric not available. Falling back to standard model.")
        args.use_gnn = False
    
    # Log which model architecture is being used
    if args.use_gnn:
        logger.info("Using Graph Neural Network architecture")
    else:
        logger.info("Using standard Feedforward Neural Network architecture")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X, y = load_data(args.input_dir)
    
    # Ensure X and y are properly formatted
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y[:, 0])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train[:, 0])
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Create DataLoaders - different approach based on model type
    if args.use_gnn:
        # For GNN: Create graph representation of power system
        logger.info("Creating graph representation of power system")
        
        try:
            # Import the create_power_system_graph function
            from direct_prediction import create_power_system_graph
            
            # Get number of generators from input data
            num_generators = X.shape[1]
            
            # Get power system graph representation
            train_graphs = create_power_system_graph(X_train, num_generators=num_generators)
            val_graphs = create_power_system_graph(X_val, num_generators=num_generators)
            test_graphs = create_power_system_graph(X_test, num_generators=num_generators)
            
            # Add target values to graphs
            for i, graph in enumerate(train_graphs):
                graph.y = torch.tensor(y_train[i:i+1], dtype=torch.float)
            for i, graph in enumerate(val_graphs):
                graph.y = torch.tensor(y_val[i:i+1], dtype=torch.float)
            for i, graph in enumerate(test_graphs):
                graph.y = torch.tensor(y_test[i:i+1], dtype=torch.float)
            
            # Create PyG DataLoaders
            try:
                from torch_geometric.loader import DataLoader as PyGDataLoader
                train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
                val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size)
                test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size)
            except ImportError:
                from torch_geometric.data import DataLoader as PyGDataLoader
                train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
                val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size)
                test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size)
            
            # Create enhanced GNN model
            from models.gnn import EnhancedConstraintScreeningGNN
            node_features = train_graphs[0].x.size(1)
            output_dim = y.shape[1]
            model = EnhancedConstraintScreeningGNN(
                node_features, 
                int(args.hidden_dims.split(',')[0]), 
                output_dim, 
                num_layers=len(args.hidden_dims.split(',')),
                dropout_rate=args.dropout
            ).to(device)
            
            logger.info(f"Created GNN model with {node_features} node features, {output_dim} outputs")
            logger.info(f"Hidden dimension: {args.hidden_dims.split(',')[0]}, Layers: {len(args.hidden_dims.split(','))}, Dropout: {args.dropout}")
        except Exception as e:
            logger.error(f"Error creating GNN model: {str(e)}")
            logger.warning("Falling back to standard model")
            args.use_gnn = False
    
    if not args.use_gnn:
        # For standard model: Use regular PyTorch DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Parse hidden dimensions
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        
        # Create standard model
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        model = ConstraintScreeningModel(input_dim, output_dim, hidden_dims, args.dropout).to(device)
        logger.info(f"Created standard model with {input_dim} inputs, {output_dim} outputs")
        logger.info(f"Hidden dimensions: {hidden_dims}, Dropout: {args.dropout}")
    
    # Create optimizer and loss function
    # For binary classification with imbalanced classes, use weighted BCE loss
    if torch.cuda.is_available():
        class_weights = [weight.cuda() for weight in class_weights]
    
    # Use Binary Cross Entropy since both models have sigmoid outputs for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create a scheduler to reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs with early stopping patience {args.early_stopping}")
    
    # Use the appropriate training function based on model type
    if args.use_gnn:
        from models.gnn import train_model_gnn
        model, train_losses, val_losses = train_model_gnn(
            model, train_loader, val_loader, criterion, optimizer, device, 
            args.epochs, args.early_stopping
        )
    else:
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, 
            args.epochs, args.early_stopping
        )
    
    # Plot learning curves
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    
    # Use the appropriate evaluation function based on model type
    if args.use_gnn:
        from models.gnn import evaluate_model_gnn
        results = evaluate_model_gnn(model, test_loader, device)
    else:
        results = evaluate_model(model, test_loader, device)
    
    # Plot confusion matrices
    plot_confusion_matrices(results['predictions'], results['targets'], args.output_dir)
    
    # Plot ROC curves
    plot_roc_curves(results['targets'], results['raw_outputs'], args.output_dir)
    
    # Save per-constraint metrics
    results['per_constraint'].to_csv(os.path.join(args.output_dir, 'per_constraint_metrics.csv'), index=False)
    
    # Save overall metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Model type: {'GNN' if args.use_gnn else 'Feedforward'}\n")
        f.write(f"Precision: {results['precision']:.6f}\n")
        f.write(f"Recall: {results['recall']:.6f}\n")
        f.write(f"F1 Score: {results['f1']:.6f}\n")
        f.write(f"Accuracy: {results['accuracy']:.6f}\n")
    
    # Save model if requested
    if args.save_model:
        model_type = "gnn" if args.use_gnn else "standard"
        model_path = os.path.join(args.output_dir, f'constraint_screening_{model_type}_model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
