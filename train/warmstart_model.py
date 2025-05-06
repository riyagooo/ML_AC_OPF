import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('warm_starting')

def parse_args():
    parser = argparse.ArgumentParser(description='Warm Starting for OPF')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--use-scaled-data', action='store_true', help='Use pre-scaled data')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for L2 regularization')
    parser.add_argument('--model-type', type=str, default="gnn", help='Model type')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')
    return parser.parse_args()

class WarmStartingModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout_rate=0.2):
        super(WarmStartingModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers
        hidden_layers = []
        for _ in range(num_layers - 1):
            hidden_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers//2)
        ])
    
    def forward(self, x):
        x = self.input_layer(x)
        
        # Apply hidden layers with residual connections
        for i in range(len(self.residual_layers)):
            residual = x
            idx = i * 4  # Each block has 4 operations
            x = self.hidden_layers[idx:idx+4](x)
            x = x + self.residual_layers[i](residual)  # Residual connection
        
        # Apply remaining layers if any
        remaining_idx = len(self.residual_layers) * 4
        if remaining_idx < len(self.hidden_layers):
            x = self.hidden_layers[remaining_idx:](x)
        
        return self.output_layer(x)

def load_data(input_dir, use_scaled_data=True):
    # Try to load scaled numpy files first if use_scaled_data is True
    if use_scaled_data:
        try:
            X = np.load(os.path.join(input_dir, 'X_warmstart_scaled.npy'))
            y = np.load(os.path.join(input_dir, 'y_warmstart_scaled.npy'))
            logger.info(f"Loaded scaled data from numpy files: {X.shape}, {y.shape}")
            return X, y
        except Exception as e:
            logger.warning(f"Could not load scaled data: {e}. Falling back to raw data")
    
    # Try to load raw numpy files
    try:
        X = np.load(os.path.join(input_dir, 'X_warmstart.npy'))
        y = np.load(os.path.join(input_dir, 'y_warmstart.npy'))
        logger.info(f"Loaded raw data from numpy files: {X.shape}, {y.shape}")
    except:
        try:
            X = pd.read_csv(os.path.join(input_dir, 'X_warmstart.csv')).values
            y = pd.read_csv(os.path.join(input_dir, 'y_warmstart.csv')).values
            logger.info(f"Loaded data from CSV files: {X.shape}, {y.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise e
    
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
            
            # Gradient clipping
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
    
    # Calculate R2 score for each output dimension
    r2_values = []
    for i in range(all_targets.shape[1]):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        r2_values.append(r2)
    
    avg_r2 = np.mean(r2_values)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"Average R²: {avg_r2:.6f}")
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'r2': avg_r2,
        'r2_values': r2_values,
        'predictions': all_preds,
        'targets': all_targets
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

def plot_predictions(predictions, targets, output_dir, max_features=5, max_samples=10):
    # Select a random subset of features to visualize
    n_features = min(max_features, predictions.shape[1])
    feature_indices = np.random.choice(predictions.shape[1], n_features, replace=False)
    
    # Select a random subset of samples
    n_samples = min(max_samples, predictions.shape[0])
    sample_indices = np.random.choice(predictions.shape[0], n_samples, replace=False)
    
    plt.figure(figsize=(15, 4 * n_features))
    
    for i, feature_idx in enumerate(feature_indices):
        plt.subplot(n_features, 1, i+1)
        
        # Extract predictions and targets for this feature
        pred_values = predictions[sample_indices, feature_idx]
        target_values = targets[sample_indices, feature_idx]
        
        # Sort by target values for better visualization
        sort_idx = np.argsort(target_values)
        pred_values = pred_values[sort_idx]
        target_values = target_values[sort_idx]
        
        # Plot
        plt.plot(range(n_samples), target_values, 'bo-', label='Actual')
        plt.plot(range(n_samples), pred_values, 'ro-', label='Predicted')
        
        plt.title(f'Feature {feature_idx}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'))
    plt.close()

def plot_r2_distribution(r2_values, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(r2_values, bins=20, alpha=0.7, color='skyblue')
    plt.axvline(np.mean(r2_values), color='red', linestyle='--', label=f'Mean R²: {np.mean(r2_values):.4f}')
    plt.xlabel('R² Value')
    plt.ylabel('Count')
    plt.title('Distribution of R² Values Across Output Features')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'r2_distribution.png'))
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
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = WarmStartingModel(input_dim, output_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    logger.info(f"Created model with {input_dim} inputs, {output_dim} outputs, " 
               f"{args.hidden_dim} hidden dim, {args.num_layers} layers, {args.dropout} dropout")
    
    # Create optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs with early stopping patience {args.early_stopping}")
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.early_stopping)
    
    # Plot learning curves
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    results = evaluate_model(model, test_loader, criterion, device)
    
    # Plot predictions
    plot_predictions(results['predictions'], results['targets'], args.output_dir)
    
    # Plot R2 distribution
    plot_r2_distribution(results['r2_values'], args.output_dir)
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {results['test_loss']:.6f}\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n")
        f.write(f"Average R²: {results['r2']:.6f}\n")
        
        # Write individual R2 scores
        f.write("\nR² for each output feature:\n")
        for i, r2 in enumerate(results['r2_values']):
            f.write(f"Feature {i+1}: {r2:.6f}\n")
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'warmstarting_model.pt'))
        logger.info(f"Model saved to {os.path.join(args.output_dir, 'warmstarting_model.pt')}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
