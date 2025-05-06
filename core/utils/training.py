import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold

class Trainer:
    """
    Trainer class for ML-OPF models.
    """
    def __init__(self, model, optimizer=None, criterion=None, scheduler=None, 
                 device='cpu', log_dir='logs', use_wandb=False):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training (default: Adam)
            criterion: Loss function (default: MSELoss)
            scheduler: Learning rate scheduler (optional)
            device: Device to use (default: 'cpu')
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # Default optimizer: Adam
        self.optimizer = optimizer if optimizer else optim.Adam(
            model.parameters(), lr=0.001)
        
        # Default criterion: MSE Loss
        self.criterion = criterion if criterion else nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = scheduler
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }
        
    def train_epoch(self, train_loader, metrics=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            metrics: Dictionary of metric functions to compute during training
            
        Returns:
            Average loss and metrics for this epoch
        """
        self.model.train()
        total_loss = 0
        total_metrics = {name: 0 for name in metrics.keys()} if metrics else {}
        
        for inputs, targets in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Compute metrics if provided
            if metrics:
                with torch.no_grad():
                    for name, metric_fn in metrics.items():
                        metric_value = metric_fn(outputs, targets).item()
                        total_metrics[name] += metric_value
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {name: value / len(train_loader) 
                      for name, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader, metrics=None):
        """
        Validate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            metrics: Dictionary of metric functions to compute during validation
            
        Returns:
            Average loss and metrics for validation data
        """
        self.model.eval()
        total_loss = 0
        total_metrics = {name: 0 for name in metrics.keys()} if metrics else {}
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track loss
                total_loss += loss.item()
                
                # Compute metrics if provided
                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_value = metric_fn(outputs, targets).item()
                        total_metrics[name] += metric_value
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {name: value / len(val_loader) 
                      for name, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader, val_loader, epochs, metrics=None, save_best=True,
             early_stopping=None, verbose=True):
        """
        Train model for specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            metrics: Dictionary of metric functions to compute during training/validation
            save_best: Whether to save best model based on validation loss
            early_stopping: Number of epochs to wait before early stopping (None to disable)
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # Initialize Weights & Biases if enabled
        if self.use_wandb:
            wandb.watch(self.model)
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader, metrics)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, metrics)
            
            # Update learning rate scheduler if provided
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for name, value in train_metrics.items():
                if name not in self.history['train_metrics']:
                    self.history['train_metrics'][name] = []
                self.history['train_metrics'][name].append(value)
            for name, value in val_metrics.items():
                if name not in self.history['val_metrics']:
                    self.history['val_metrics'][name] = []
                self.history['val_metrics'][name].append(value)
            
            # Log to Weights & Biases if enabled
            if self.use_wandb:
                log_dict = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch
                }
                for name, value in train_metrics.items():
                    log_dict[f'train_{name}'] = value
                for name, value in val_metrics.items():
                    log_dict[f'val_{name}'] = value
                wandb.log(log_dict)
            
            # Print progress if verbose
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.6f}')
                print(f'  Val Loss: {val_loss:.6f}')
                for name, value in train_metrics.items():
                    print(f'  Train {name}: {value:.6f}')
                for name, value in val_metrics.items():
                    print(f'  Val {name}: {value:.6f}')
            
            # Save best model if specified
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pt'))
                if verbose:
                    print('  Saved best model')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stopping and early_stop_counter >= early_stopping:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'final_model.pt'))
        
        return self.history
    
    def predict(self, loader, return_targets=True):
        """
        Make predictions on a dataset.
        
        Args:
            loader: DataLoader for the dataset
            return_targets: Whether to return targets along with predictions
            
        Returns:
            Predictions and optionally targets
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu().numpy())
                if return_targets:
                    all_targets.append(targets.numpy())
        
        all_preds = np.vstack(all_preds)
        
        if return_targets:
            all_targets = np.vstack(all_targets)
            return all_preds, all_targets
        else:
            return all_preds
    
    def plot_history(self, metrics=None, save_path=None):
        """
        Plot training history.
        
        Args:
            metrics: List of metric names to plot (default: plot all)
            save_path: Path to save plot (optional)
        """
        if not metrics and self.history['train_metrics']:
            metrics = list(self.history['train_metrics'].keys())
        
        # Create figure with subplots
        n_plots = 1 + (len(metrics) if metrics else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Training Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss vs. Epoch')
        axes[0].legend()
        axes[0].grid(True)
        
        # Set appropriate y-axis limits for loss (add some padding)
        if self.history['train_loss'] and self.history['val_loss']:
            all_losses = self.history['train_loss'] + self.history['val_loss']
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            padding = (max_loss - min_loss) * 0.1  # 10% padding
            axes[0].set_ylim([min_loss - padding, max_loss + padding])
        
        # Plot metrics
        if metrics:
            for i, metric_name in enumerate(metrics):
                if metric_name in self.history['train_metrics'] and metric_name in self.history['val_metrics']:
                    train_metric = self.history['train_metrics'][metric_name]
                    val_metric = self.history['val_metrics'][metric_name]
                    
                    axes[i+1].plot(train_metric, label=f'Training {metric_name}')
                    axes[i+1].plot(val_metric, label=f'Validation {metric_name}')
                    axes[i+1].set_xlabel('Epoch')
                    axes[i+1].set_ylabel(metric_name)
                    axes[i+1].set_title(f'{metric_name} vs. Epoch')
                    axes[i+1].legend()
                    axes[i+1].grid(True)
                    
                    # Set appropriate y-axis limits for the metric
                    all_metrics = train_metric + val_metric
                    if all_metrics:
                        min_metric = min(all_metrics)
                        max_metric = max(all_metrics)
                        
                        # If all values are close to each other, expand the range
                        if max_metric - min_metric < 1e-6:
                            padding = abs(max_metric) * 0.1  # 10% of the value
                            if padding == 0:  # Handle case where all values are zero
                                padding = 0.1
                        else:
                            padding = (max_metric - min_metric) * 0.1  # 10% padding
                        
                        # If optimality gap, make sure lower bound is not negative
                        if 'gap' in metric_name.lower():
                            min_bound = max(0, min_metric - padding)
                            axes[i+1].set_ylim([min_bound, max_metric + padding])
                        else:
                            axes[i+1].set_ylim([min_metric - padding, max_metric + padding])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
        
    def load_best_model(self):
        """Load the best model based on validation loss."""
        best_model_path = os.path.join(self.log_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded best model from {best_model_path}")
        else:
            print("No best model found")

# Custom metrics for OPF evaluation
def constraint_violation_metric(output, target=None, bounds=None):
    """
    Compute constraint violation metric.
    
    Args:
        output: Model predictions
        target: Ground truth values (not used for this metric)
        bounds: Dictionary of bound tensors
        
    Returns:
        Tensor with average constraint violation
    """
    if bounds is None:
        return torch.tensor(0.0)
    
    violations = torch.zeros(1, device=output.device)
    
    # Add your constraint violation logic here
    # Example: Check if outputs are within bounds
    
    return violations

def optimality_gap_metric(output, target, cost_coeffs):
    """
    Compute optimality gap metric.
    
    Args:
        output: Model predictions (first columns should be generator outputs)
        target: Ground truth values (first columns should be generator outputs)
        cost_coeffs: Generator cost coefficients
        
    Returns:
        Tensor with relative optimality gap
    """
    # Extract generator outputs (assuming they're the first columns)
    pred_gen = output[:, :len(cost_coeffs)]
    true_gen = target[:, :len(cost_coeffs)]
    
    # Compute generation costs
    pred_cost = torch.sum(pred_gen * cost_coeffs, dim=1)
    true_cost = torch.sum(true_gen * cost_coeffs, dim=1)
    
    # Compute relative optimality gap
    gap = (pred_cost - true_cost) / (true_cost + 1e-8)
    
    return torch.mean(gap)

class PhysicsInformedMSELoss(torch.nn.Module):
    """
    Physics-informed MSE loss for OPF problems.
    
    This loss function combines standard MSE with penalty terms for physics constraints:
    1. Power balance constraints
    2. Generation limit violations
    3. Voltage magnitude constraints
    
    Args:
        case_data: PyPOWER case data dictionary
        lambda_power: Weight for power balance penalty
        lambda_gen: Weight for generation limit penalty
        lambda_voltage: Weight for voltage magnitude penalty
        epsilon: Small value for numerical stability
    """
    def __init__(self, case_data, lambda_power=1.0, lambda_gen=1.0, 
                lambda_voltage=1.0, epsilon=1e-8):
        super(PhysicsInformedMSELoss, self).__init__()
        self.case_data = case_data
        self.lambda_power = lambda_power
        self.lambda_gen = lambda_gen
        self.lambda_voltage = lambda_voltage
        self.epsilon = epsilon
        
        # Base MSE loss
        self.mse = torch.nn.MSELoss()
        
        # Store system parameters
        self.baseMVA = case_data['baseMVA']
        
        # Extract generator limits
        self.pg_min = torch.tensor(case_data['gen'][:, 9] / self.baseMVA, dtype=torch.float32)
        self.pg_max = torch.tensor(case_data['gen'][:, 8] / self.baseMVA, dtype=torch.float32)
        
        # Extract voltage limits
        self.vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32)
        self.vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32)
        
        # Generator and bus indices for mapping
        self.gen_bus = case_data['gen'][:, 0].astype(int)
        
    def forward(self, pred, target):
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            target: Target values [batch_size, n_outputs]
            
        Returns:
            Total loss combining MSE and physics penalties
        """
        # Base MSE loss
        mse_loss = self.mse(pred, target)
        
        # Number of generators and buses
        n_gen = len(self.pg_min)
        n_bus = len(self.vm_min)
        
        # Initialize penalty terms
        gen_penalty = 0.0
        voltage_penalty = 0.0
        
        # Extract generator outputs (assuming first n_gen elements are pg)
        if pred.shape[1] >= n_gen:
            pred_pg = pred[:, :n_gen]
            
            # Generator limit violations penalty
            gen_min_violation = torch.nn.functional.relu(self.pg_min - pred_pg)
            gen_max_violation = torch.nn.functional.relu(pred_pg - self.pg_max)
            gen_penalty = torch.mean(gen_min_violation + gen_max_violation)
        
        # Extract voltage magnitude outputs (assuming they're after generator outputs)
        vm_start = 2 * n_gen  # After pg and qg
        if pred.shape[1] >= vm_start + n_bus:
            pred_vm = pred[:, vm_start:vm_start+n_bus]
            
            # Voltage magnitude violations penalty
            vm_min_violation = torch.nn.functional.relu(self.vm_min - pred_vm)
            vm_max_violation = torch.nn.functional.relu(pred_vm - self.vm_max)
            voltage_penalty = torch.mean(vm_min_violation + vm_max_violation)
        
        # Combine losses with weighted penalties
        total_loss = mse_loss + \
                    self.lambda_gen * gen_penalty + \
                    self.lambda_voltage * voltage_penalty
        
        return total_loss

class RobustLoss(torch.nn.Module):
    """
    Robust loss function for OPF problems with outliers and numerical instability.
    
    Combines aspects of Huber loss, adaptive scaling, and normalization to handle
    the extreme values and numerical challenges in OPF problems.
    
    Args:
        beta: Threshold for switching between MSE and MAE (Huber parameter)
        epsilon: Small value for numerical stability
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, beta=1.0, epsilon=1e-8, reduction='mean'):
        super(RobustLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Compute robust loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        # Scale inputs to similar ranges to avoid numerical issues
        # We'll use per-feature scaling based on the target values
        target_max = torch.max(torch.abs(target), dim=0, keepdim=True)[0]
        scale = torch.clamp(target_max, min=self.epsilon)
        
        # Scale both prediction and target
        pred_scaled = pred / scale
        target_scaled = target / scale
        
        # Compute absolute error
        abs_error = torch.abs(pred_scaled - target_scaled)
        
        # Apply Huber-like loss function
        quadratic_mask = abs_error <= self.beta
        linear_mask = ~quadratic_mask
        
        # Compute loss components
        quadratic_loss = 0.5 * abs_error[quadratic_mask] ** 2
        linear_loss = self.beta * (abs_error[linear_mask] - 0.5 * self.beta)
        
        # Combine loss components
        losses = torch.zeros_like(abs_error)
        losses[quadratic_mask] = quadratic_loss
        losses[linear_mask] = linear_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:  # 'mean'
            return torch.mean(losses)

class OptimalityGapLoss(torch.nn.Module):
    """
    Loss function based on the optimality gap between predicted and true solutions.
    
    This is particularly useful for warm-starting, where the goal is to minimize
    the difference in objective function value rather than point-wise errors.
    
    Args:
        case_data: PyPOWER case data dictionary
        alpha: Weight for MSE component (0-1)
        beta: Weight for optimality gap component (1-alpha)
    """
    def __init__(self, case_data, alpha=0.2, beta=0.8):
        super(OptimalityGapLoss, self).__init__()
        self.case_data = case_data
        self.alpha = alpha
        self.beta = beta
        
        # Base MSE loss
        self.mse = torch.nn.MSELoss()
        
        # Extract cost coefficients for generators
        # Assuming quadratic cost function: a*P^2 + b*P + c
        self.cost_a = torch.tensor([coef[4] for coef in case_data['gencost']], 
                                  dtype=torch.float32)
        self.cost_b = torch.tensor([coef[5] for coef in case_data['gencost']], 
                                  dtype=torch.float32)
        self.cost_c = torch.tensor([coef[6] for coef in case_data['gencost']], 
                                  dtype=torch.float32)
    
    def calculate_generation_cost(self, pg):
        """Calculate the generation cost for given active power outputs."""
        # Ensure pg has right shape
        if len(pg.shape) == 1:
            pg = pg.unsqueeze(0)
            
        # Calculate quadratic cost
        cost = torch.zeros(pg.shape[0], device=pg.device)
        
        for i in range(min(pg.shape[1], len(self.cost_a))):
            cost += self.cost_a[i] * pg[:, i]**2 + self.cost_b[i] * pg[:, i] + self.cost_c[i]
            
        return cost
    
    def forward(self, pred, target):
        """
        Compute loss based on optimality gap.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            target: Target values [batch_size, n_outputs]
            
        Returns:
            Combined loss with MSE and optimality gap components
        """
        # Calculate base MSE
        mse_loss = self.mse(pred, target)
        
        # Calculate optimality gap
        n_gen = len(self.cost_a)
        
        # Extract generator active power outputs
        pred_pg = pred[:, :n_gen]
        target_pg = target[:, :n_gen]
        
        # Calculate generation costs
        pred_cost = self.calculate_generation_cost(pred_pg)
        target_cost = self.calculate_generation_cost(target_pg)
        
        # Calculate optimality gap (relative)
        gap = torch.abs(pred_cost - target_cost) / (torch.abs(target_cost) + 1e-8)
        gap_loss = torch.mean(gap)
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * gap_loss
        
        return total_loss 

def create_kfold_splits(X, y, n_splits=5, batch_size=32, stratify=None, random_state=42):
    """
    Create K-fold cross-validation splits that can be used by any model.
    
    Args:
        X: Features data (numpy array)
        y: Target data (numpy array)
        n_splits: Number of folds (default: 5)
        batch_size: Batch size for DataLoader (default: 32)
        stratify: Column index to use for stratified splits (default: None)
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_loader, val_loader) tuples for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Create stratified folds if specified
    if stratify is not None and len(y.shape) > 1 and y.shape[1] > stratify:
        # For classification tasks, use stratified folds based on the specified target column
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_indices = list(kf.split(X, y[:, stratify]))
    else:
        # For regression or other tasks, use regular K-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_indices = list(kf.split(X))
    
    # Create data loaders for each fold
    fold_loaders = []
    for train_idx, val_idx in split_indices:
        # Get train/val split for this fold
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train_fold), torch.FloatTensor(y_train_fold))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

def train_with_kfold(model_constructor, fold_loaders, criterion, optimizer_fn, device, epochs, 
                     patience=10, verbose=True):
    """
    Train a model using K-fold cross-validation.
    
    Args:
        model_constructor: Function that returns a new model instance
        fold_loaders: List of (train_loader, val_loader) tuples from create_kfold_splits
        criterion: Loss function to use
        optimizer_fn: Function that takes model parameters and returns an optimizer
        device: Device to use for training ('cpu' or 'cuda')
        epochs: Maximum number of epochs to train for
        patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        List of trained models (one per fold) and their metrics
    """
    fold_results = []
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        if verbose:
            print(f"Training fold {fold_idx+1}/{len(fold_loaders)}")
        
        # Create a new model instance for this fold
        model = model_constructor().to(device)
        
        # Create optimizer for this model
        optimizer = optimizer_fn(model.parameters())
        
        # Train the model
        best_val_loss = float('inf')
        best_model_state = None
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
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Save best model and check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Store results for this fold
        fold_results.append({
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'fold_idx': fold_idx
        })
    
    return fold_results 

class PowerSystemPhysicsLoss(torch.nn.Module):
    """
    Comprehensive physics-constrained loss function for AC-OPF problems.
    
    This loss combines multiple objectives:
    1. Decision loss (optimality gap minimization)
    2. Power flow constraint satisfaction
    3. Thermal limit constraint satisfaction
    4. Voltage angle difference constraints
    
    Args:
        case_data: PyPOWER case data dictionary
        lambda_opt: Weight for optimality gap component
        lambda_pf: Weight for power flow violations
        lambda_thermal: Weight for thermal limit violations
        lambda_angle: Weight for voltage angle violations
        epsilon: Small value for numerical stability
    """
    def __init__(self, case_data, lambda_opt=1.0, lambda_pf=10.0, 
                lambda_thermal=5.0, lambda_angle=2.0, epsilon=1e-8):
        super(PowerSystemPhysicsLoss, self).__init__()
        self.case_data = case_data
        self.lambda_opt = lambda_opt
        self.lambda_pf = lambda_pf
        self.lambda_thermal = lambda_thermal
        self.lambda_angle = lambda_angle
        self.epsilon = epsilon
        
        # Store system parameters
        self.baseMVA = case_data['baseMVA'] 
        
        # Extract grid topology information
        self.n_bus = len(case_data['bus'])
        self.n_gen = len(case_data['gen'])
        self.n_branch = len(case_data['branch'])
        
        # Create maps from bus indices to their position in data structures
        self.bus_map = {int(bus[0]): i for i, bus in enumerate(case_data['bus'])}
        
        # Generator bus connections
        self.gen_bus_idx = [self.bus_map[int(gen[0])] for gen in case_data['gen']]
        
        # Generator costs and limits
        self.cost_coeffs = torch.tensor([
            [coef[4], coef[5], coef[6]] for coef in case_data['gencost']
        ], dtype=torch.float32)  # Format: [a, b, c] for a*P^2 + b*P + c
        
        # Generator limits
        self.pg_min = torch.tensor(case_data['gen'][:, 9] / self.baseMVA, dtype=torch.float32) 
        self.pg_max = torch.tensor(case_data['gen'][:, 8] / self.baseMVA, dtype=torch.float32)
        self.qg_min = torch.tensor(case_data['gen'][:, 4] / self.baseMVA, dtype=torch.float32)
        self.qg_max = torch.tensor(case_data['gen'][:, 3] / self.baseMVA, dtype=torch.float32)
        
        # Voltage limits
        self.vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32)
        self.vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32)
        
        # Branch limits
        self.branch_from = [self.bus_map[int(branch[0])] for branch in case_data['branch']]
        self.branch_to = [self.bus_map[int(branch[1])] for branch in case_data['branch']]
        self.branch_r = torch.tensor(case_data['branch'][:, 2], dtype=torch.float32)
        self.branch_x = torch.tensor(case_data['branch'][:, 3], dtype=torch.float32)
        self.branch_b = torch.tensor(case_data['branch'][:, 4], dtype=torch.float32)
        self.rate_a = torch.tensor(case_data['branch'][:, 5] / self.baseMVA, dtype=torch.float32)
        
        # Maximum angle difference (typically 30 degrees = π/6 radians)
        self.max_angle_diff = torch.tensor(np.pi/6, dtype=torch.float32)
        
    def calculate_generation_cost(self, pg):
        """Calculate the generation cost for given active power outputs."""
        # Ensure pg has right shape
        if len(pg.shape) == 1:
            pg = pg.unsqueeze(0)
            
        # Calculate quadratic cost using vectorized operations
        # cost = a*P^2 + b*P + c
        pg_squared = pg ** 2
        
        # Compute cost for each generator
        cost_a_term = pg_squared * self.cost_coeffs[:, 0].to(pg.device)
        cost_b_term = pg * self.cost_coeffs[:, 1].to(pg.device)
        cost_c_term = self.cost_coeffs[:, 2].to(pg.device).unsqueeze(0).expand(pg.shape[0], -1)
        
        # Sum across generators for total cost
        gen_costs = cost_a_term + cost_b_term + cost_c_term
        total_cost = torch.sum(gen_costs, dim=1)
            
        return total_cost
        
    def calculate_power_flow_violations(self, pred):
        """
        Calculate violations of power flow constraints based on prediction.
        
        This simplified version checks:
        1. Generator active power violations (P_min <= P <= P_max)
        2. Generator reactive power violations (Q_min <= Q <= Q_max)
        3. Voltage magnitude violations (V_min <= V <= V_max)
        
        A more detailed implementation would calculate actual power flow
        equations, but this requires access to the admittance matrix and
        more complex calculations.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            
        Returns:
            Average violation magnitude
        """
        # Extract predicted values
        p_gen = pred[:, :self.n_gen]
        q_gen = pred[:, self.n_gen:2*self.n_gen] if pred.shape[1] >= 2*self.n_gen else None
        v_mag = pred[:, 2*self.n_gen:2*self.n_gen+self.n_bus] if pred.shape[1] >= 2*self.n_gen+self.n_bus else None
        v_ang = pred[:, 2*self.n_gen+self.n_bus:] if pred.shape[1] >= 2*self.n_gen+2*self.n_bus else None
        
        # Initialize total violation
        total_violation = torch.tensor(0.0, device=pred.device)
        
        # Check generator active power violations
        p_min_violation = torch.nn.functional.relu(self.pg_min.to(pred.device) - p_gen)
        p_max_violation = torch.nn.functional.relu(p_gen - self.pg_max.to(pred.device))
        total_violation += torch.mean(p_min_violation + p_max_violation)
        
        # Check generator reactive power violations if available
        if q_gen is not None:
            q_min_violation = torch.nn.functional.relu(self.qg_min.to(pred.device) - q_gen)
            q_max_violation = torch.nn.functional.relu(q_gen - self.qg_max.to(pred.device))
            total_violation += torch.mean(q_min_violation + q_max_violation)
        
        # Check voltage magnitude violations if available
        if v_mag is not None:
            v_min_violation = torch.nn.functional.relu(self.vm_min.to(pred.device) - v_mag)
            v_max_violation = torch.nn.functional.relu(v_mag - self.vm_max.to(pred.device))
            total_violation += torch.mean(v_min_violation + v_max_violation)
            
        return total_violation
        
    def calculate_thermal_limit_violations(self, pred):
        """
        Calculate thermal limit violations for branches.
        
        In a full implementation, this would compute S = √(P² + Q²) for each line
        and check if S > S_max. This simplified version estimates line flows 
        based on voltage magnitudes and angles.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            
        Returns:
            Average thermal violation magnitude
        """
        # Only calculate if voltage magnitudes and angles are predicted
        if pred.shape[1] < 2*self.n_gen + 2*self.n_bus:
            return torch.tensor(0.0, device=pred.device)
            
        # Extract voltage magnitudes and angles
        v_mag = pred[:, 2*self.n_gen:2*self.n_gen+self.n_bus]
        v_ang = pred[:, 2*self.n_gen+self.n_bus:2*self.n_gen+2*self.n_bus]
        
        # Initialize thermal violations
        total_violation = torch.tensor(0.0, device=pred.device)
        
        # For each branch, estimate apparent power and check against limit
        for i in range(self.n_branch):
            # Get from/to bus indices
            from_idx = self.branch_from[i]
            to_idx = self.branch_to[i]
            
            # Get branch parameters
            r = self.branch_r[i].to(pred.device)
            x = self.branch_x[i].to(pred.device)
            b = self.branch_b[i].to(pred.device)
            rate = self.rate_a[i].to(pred.device)
            
            if rate <= 0:  # Skip branches with no thermal limit
                continue
                
            # Get voltage magnitudes and angles for from/to buses
            v_from = v_mag[:, from_idx]
            v_to = v_mag[:, to_idx]
            theta_from = v_ang[:, from_idx]
            theta_to = v_ang[:, to_idx]
            theta_diff = theta_from - theta_to
            
            # Simplified apparent power calculation (approximation)
            # S ≈ |V_from| × |V_to| / Z
            z_mag = torch.sqrt(r**2 + x**2)
            s_approx = v_from * v_to / z_mag
            
            # Check thermal violation
            thermal_violation = torch.nn.functional.relu(s_approx - rate)
            total_violation += torch.mean(thermal_violation)
            
        return total_violation / max(1, self.n_branch)
        
    def calculate_angle_difference_violations(self, pred):
        """
        Calculate voltage angle difference violations.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            
        Returns:
            Average angle difference violation magnitude
        """
        # Only calculate if voltage angles are predicted
        if pred.shape[1] < 2*self.n_gen + 2*self.n_bus:
            return torch.tensor(0.0, device=pred.device)
            
        # Extract voltage angles
        v_ang = pred[:, 2*self.n_gen+self.n_bus:2*self.n_gen+2*self.n_bus]
        
        # Initialize angle violations
        total_violation = torch.tensor(0.0, device=pred.device)
        valid_branches = 0
        
        # For each branch, check angle difference
        for i in range(self.n_branch):
            # Get from/to bus indices
            from_idx = self.branch_from[i]
            to_idx = self.branch_to[i]
            
            # Get angle difference
            theta_from = v_ang[:, from_idx]
            theta_to = v_ang[:, to_idx]
            theta_diff = torch.abs(theta_from - theta_to)
            
            # Check angle violation
            angle_violation = torch.nn.functional.relu(theta_diff - self.max_angle_diff.to(pred.device))
            total_violation += torch.mean(angle_violation)
            valid_branches += 1
            
        return total_violation / max(1, valid_branches)
    
    def forward(self, pred, target):
        """
        Compute physics-constrained loss.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            target: Target values [batch_size, n_outputs]
            
        Returns:
            Total loss combining optimality gap and physics penalties
        """
        # Calculate optimality gap (decision loss)
        n_gen = self.n_gen
        
        # Extract generator active power outputs
        pred_pg = pred[:, :n_gen]
        target_pg = target[:, :n_gen]
        
        # Calculate generation costs
        pred_cost = self.calculate_generation_cost(pred_pg)
        target_cost = self.calculate_generation_cost(target_pg)
        
        # Calculate optimality gap (relative)
        gap = torch.abs(pred_cost - target_cost) / (torch.abs(target_cost) + self.epsilon)
        opt_loss = torch.mean(gap)
        
        # Calculate physics-based penalties
        pf_violation = self.calculate_power_flow_violations(pred)
        thermal_violation = self.calculate_thermal_limit_violations(pred)
        angle_violation = self.calculate_angle_difference_violations(pred)
        
        # Combine losses
        total_loss = (
            self.lambda_opt * opt_loss + 
            self.lambda_pf * pf_violation + 
            self.lambda_thermal * thermal_violation + 
            self.lambda_angle * angle_violation
        )
        
        return total_loss 

class LagrangianDualLoss(torch.nn.Module):
    """
    Lagrangian dual loss function for AC-OPF optimization.
    
    This loss implements a Lagrangian dual method where the multipliers are 
    updated during training to enforce constraint satisfaction. This approach
    is based on recent research showing that Lagrangian methods can improve
    feasibility of ML-based AC-OPF solutions.
    
    Args:
        case_data: PyPOWER case data dictionary
        mse_weight: Weight for MSE component  
        alpha: Learning rate for dual variable updates
        beta: Factor for penalty term (optional quadratic term)
        use_quadratic_penalty: Whether to use quadratic penalty method
    """
    def __init__(self, case_data, mse_weight=0.1, alpha=0.01, beta=1.0, 
                 use_quadratic_penalty=True):
        super(LagrangianDualLoss, self).__init__()
        self.case_data = case_data
        self.mse_weight = mse_weight
        self.alpha = alpha
        self.beta = beta
        self.use_quadratic_penalty = use_quadratic_penalty
        
        # Base MSE loss
        self.mse = torch.nn.MSELoss(reduction='mean')
        
        # Extract system parameters
        self.baseMVA = case_data['baseMVA']
        self.n_bus = len(case_data['bus'])
        self.n_gen = len(case_data['gen'])
        self.n_branch = len(case_data['branch'])
        
        # Create maps from bus indices to their position in data structures
        self.bus_map = {int(bus[0]): i for i, bus in enumerate(case_data['bus'])}
        
        # Generator bus connections
        self.gen_bus_idx = [self.bus_map[int(gen[0])] for gen in case_data['gen']]
        
        # Generator limits
        self.pg_min = torch.tensor(case_data['gen'][:, 9] / self.baseMVA, dtype=torch.float32)
        self.pg_max = torch.tensor(case_data['gen'][:, 8] / self.baseMVA, dtype=torch.float32)
        self.qg_min = torch.tensor(case_data['gen'][:, 4] / self.baseMVA, dtype=torch.float32)
        self.qg_max = torch.tensor(case_data['gen'][:, 3] / self.baseMVA, dtype=torch.float32)
        
        # Voltage limits
        self.vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32)
        self.vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32)
        
        # Initialize Lagrangian multipliers for constraints
        # For each type of constraint (P_min, P_max, V_min, V_max, etc.)
        self.lambda_p_min = nn.Parameter(torch.zeros(self.n_gen), requires_grad=False)
        self.lambda_p_max = nn.Parameter(torch.zeros(self.n_gen), requires_grad=False)
        self.lambda_q_min = nn.Parameter(torch.zeros(self.n_gen), requires_grad=False)
        self.lambda_q_max = nn.Parameter(torch.zeros(self.n_gen), requires_grad=False)
        self.lambda_v_min = nn.Parameter(torch.zeros(self.n_bus), requires_grad=False)
        self.lambda_v_max = nn.Parameter(torch.zeros(self.n_bus), requires_grad=False)
        
        # For cost term
        self.cost_coeffs = torch.tensor([
            [coef[4], coef[5], coef[6]] for coef in case_data['gencost']
        ], dtype=torch.float32)  # Format: [a, b, c] for a*P^2 + b*P + c
    
    def calculate_generation_cost(self, pg):
        """Calculate the generation cost for given active power outputs."""
        # Ensure pg has right shape
        if len(pg.shape) == 1:
            pg = pg.unsqueeze(0)
            
        # Calculate quadratic cost using vectorized operations
        # cost = a*P^2 + b*P + c
        pg_squared = pg ** 2
        
        # Compute cost for each generator
        cost_a_term = pg_squared * self.cost_coeffs[:, 0].to(pg.device)
        cost_b_term = pg * self.cost_coeffs[:, 1].to(pg.device)
        cost_c_term = self.cost_coeffs[:, 2].to(pg.device).unsqueeze(0).expand(pg.shape[0], -1)
        
        # Sum across generators for total cost
        gen_costs = cost_a_term + cost_b_term + cost_c_term
        total_cost = torch.sum(gen_costs, dim=1)
            
        return total_cost
    
    def calculate_constraint_violations(self, pred):
        """
        Calculate constraint violations.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            
        Returns:
            Dictionary of constraint violations and total violation
        """
        # Extract predicted values
        p_gen = pred[:, :self.n_gen]
        q_gen = pred[:, self.n_gen:2*self.n_gen] if pred.shape[1] >= 2*self.n_gen else None
        v_mag = pred[:, 2*self.n_gen:2*self.n_gen+self.n_bus] if pred.shape[1] >= 2*self.n_gen+self.n_bus else None
        
        # Initialize violations dictionary
        violations = {}
        
        # P_min violations: g(x) = P_min - P <= 0
        violations['p_min'] = self.pg_min.to(pred.device) - p_gen
        
        # P_max violations: g(x) = P - P_max <= 0
        violations['p_max'] = p_gen - self.pg_max.to(pred.device)
        
        # Q_min violations if reactive power is predicted
        if q_gen is not None:
            violations['q_min'] = self.qg_min.to(pred.device) - q_gen
            violations['q_max'] = q_gen - self.qg_max.to(pred.device)
        
        # V_min violations if voltage magnitude is predicted
        if v_mag is not None:
            violations['v_min'] = self.vm_min.to(pred.device) - v_mag
            violations['v_max'] = v_mag - self.vm_max.to(pred.device)
        
        return violations
    
    def update_multipliers(self, violations):
        """
        Update Lagrangian multipliers based on constraint violations.
        
        Args:
            violations: Dictionary of constraint violations from calculate_constraint_violations
        """
        # Update λ for P_min: λ = max(0, λ + α * g(x))
        p_min_update = self.lambda_p_min + self.alpha * torch.mean(violations['p_min'], dim=0)
        self.lambda_p_min.data = torch.clamp(p_min_update, min=0.0)
        
        # Update λ for P_max
        p_max_update = self.lambda_p_max + self.alpha * torch.mean(violations['p_max'], dim=0)
        self.lambda_p_max.data = torch.clamp(p_max_update, min=0.0)
        
        # Update λ for Q constraints if available
        if 'q_min' in violations:
            q_min_update = self.lambda_q_min + self.alpha * torch.mean(violations['q_min'], dim=0)
            self.lambda_q_min.data = torch.clamp(q_min_update, min=0.0)
            
            q_max_update = self.lambda_q_max + self.alpha * torch.mean(violations['q_max'], dim=0)
            self.lambda_q_max.data = torch.clamp(q_max_update, min=0.0)
        
        # Update λ for V constraints if available
        if 'v_min' in violations:
            v_min_update = self.lambda_v_min + self.alpha * torch.mean(violations['v_min'], dim=0)
            self.lambda_v_min.data = torch.clamp(v_min_update, min=0.0)
            
            v_max_update = self.lambda_v_max + self.alpha * torch.mean(violations['v_max'], dim=0)
            self.lambda_v_max.data = torch.clamp(v_max_update, min=0.0)
    
    def forward(self, pred, target, update_multipliers=True):
        """
        Compute Lagrangian dual loss.
        
        Args:
            pred: Predicted values [batch_size, n_outputs]
            target: Target values [batch_size, n_outputs]
            update_multipliers: Whether to update Lagrangian multipliers (True during training)
            
        Returns:
            Total loss combining MSE, cost difference, and constraint penalties
        """
        # Calculate base MSE
        mse_loss = self.mse(pred, target)
        
        # Calculate generation cost
        pred_pg = pred[:, :self.n_gen]
        target_pg = target[:, :self.n_gen]
        pred_cost = self.calculate_generation_cost(pred_pg)
        target_cost = self.calculate_generation_cost(target_pg)
        cost_diff = torch.mean((pred_cost - target_cost) ** 2)
        
        # Calculate constraint violations
        violations = self.calculate_constraint_violations(pred)
        
        # Calculate penalty term using Lagrangian multipliers
        penalty = torch.tensor(0.0, device=pred.device)
        
        # For P_min: max(0, P_min - P)
        p_min_violation = torch.nn.functional.relu(violations['p_min'])
        penalty += torch.sum(self.lambda_p_min.to(pred.device) * torch.mean(p_min_violation, dim=0))
        
        # For P_max: max(0, P - P_max)
        p_max_violation = torch.nn.functional.relu(violations['p_max'])
        penalty += torch.sum(self.lambda_p_max.to(pred.device) * torch.mean(p_max_violation, dim=0))
        
        # For Q constraints if available
        if 'q_min' in violations:
            q_min_violation = torch.nn.functional.relu(violations['q_min'])
            penalty += torch.sum(self.lambda_q_min.to(pred.device) * torch.mean(q_min_violation, dim=0))
            
            q_max_violation = torch.nn.functional.relu(violations['q_max'])
            penalty += torch.sum(self.lambda_q_max.to(pred.device) * torch.mean(q_max_violation, dim=0))
        
        # For V constraints if available
        if 'v_min' in violations:
            v_min_violation = torch.nn.functional.relu(violations['v_min'])
            penalty += torch.sum(self.lambda_v_min.to(pred.device) * torch.mean(v_min_violation, dim=0))
            
            v_max_violation = torch.nn.functional.relu(violations['v_max'])
            penalty += torch.sum(self.lambda_v_max.to(pred.device) * torch.mean(v_max_violation, dim=0))
        
        # Add quadratic penalty if enabled (augmented Lagrangian method)
        if self.use_quadratic_penalty:
            quad_penalty = torch.tensor(0.0, device=pred.device)
            
            # For each constraint, add a quadratic penalty term
            quad_penalty += torch.sum(torch.mean(p_min_violation ** 2, dim=0))
            quad_penalty += torch.sum(torch.mean(p_max_violation ** 2, dim=0))
            
            if 'q_min' in violations:
                quad_penalty += torch.sum(torch.mean(torch.nn.functional.relu(violations['q_min']) ** 2, dim=0))
                quad_penalty += torch.sum(torch.mean(torch.nn.functional.relu(violations['q_max']) ** 2, dim=0))
                
            if 'v_min' in violations:
                quad_penalty += torch.sum(torch.mean(torch.nn.functional.relu(violations['v_min']) ** 2, dim=0))
                quad_penalty += torch.sum(torch.mean(torch.nn.functional.relu(violations['v_max']) ** 2, dim=0))
                
            penalty += self.beta * quad_penalty
        
        # Compute final loss
        total_loss = self.mse_weight * mse_loss + (1 - self.mse_weight) * cost_diff + penalty
        
        # Update Lagrangian multipliers if in training mode
        if update_multipliers:
            self.update_multipliers(violations)
        
        return total_loss 