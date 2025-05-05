import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb

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