import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats

from .robust_data import DataNormalizer

class OPFMetrics:
    """
    Comprehensive metrics for evaluating OPF solutions.
    """
    def __init__(self, case_data, normalizer=None, device='cpu'):
        """
        Initialize OPF metrics calculator.
        
        Args:
            case_data: PyPOWER case data
            normalizer: Optional data normalizer for de-normalizing predictions
            device: Device for torch tensors
        """
        self.case_data = case_data
        self.normalizer = normalizer
        self.device = device
        
        # Extract cost coefficients for calculating generation cost
        self.baseMVA = case_data['baseMVA']
        self.cost_coeffs = torch.tensor(
            [coef[5] for coef in case_data['gencost']], 
            dtype=torch.float32,
            device=device
        )
        
        # Get the number of generators and buses
        self.n_gen = len(case_data['gen'])
        self.n_bus = len(case_data['bus'])
        
    def compute_metrics(self, predictions, targets, denormalize=True):
        """
        Compute comprehensive metrics for OPF predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            denormalize: Whether to denormalize predictions and targets
            
        Returns:
            Dictionary of metrics
        """
        # Convert to torch tensors if needed
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, device=self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, device=self.device)
            
        # Denormalize if needed
        if denormalize and self.normalizer is not None:
            predictions = self.normalizer.inverse_transform_outputs(predictions)
            targets = self.normalizer.inverse_transform_outputs(targets)
        
        # Calculate basic regression metrics
        metrics = {}
        metrics['mse'] = self._compute_mse(predictions, targets)
        metrics['rmse'] = torch.sqrt(metrics['mse'])
        metrics['mae'] = self._compute_mae(predictions, targets)
        metrics['r2'] = self._compute_r2(predictions, targets)
        
        # Calculate OPF-specific metrics
        metrics['optimality_gap'] = self._compute_optimality_gap(predictions, targets)
        metrics['constraint_violation'] = self._compute_constraint_violation(predictions)
        
        return metrics
    
    def _compute_mse(self, predictions, targets):
        """Compute mean squared error."""
        return torch.mean((predictions - targets) ** 2)
    
    def _compute_mae(self, predictions, targets):
        """Compute mean absolute error."""
        return torch.mean(torch.abs(predictions - targets))
    
    def _compute_r2(self, predictions, targets):
        """Compute R^2 score."""
        ss_tot = torch.sum((targets - torch.mean(targets, dim=0)) ** 2)
        ss_res = torch.sum((targets - predictions) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2
    
    def _compute_optimality_gap(self, predictions, targets):
        """Compute optimality gap for generation cost."""
        # Extract generator outputs (assuming first n_gen columns are Pg)
        pred_pg = predictions[:, :self.n_gen]
        true_pg = targets[:, :self.n_gen]
        
        # Compute generation costs
        pred_cost = torch.sum(pred_pg * self.cost_coeffs, dim=1)
        true_cost = torch.sum(true_pg * self.cost_coeffs, dim=1)
        
        # Compute relative optimality gap
        gap = (pred_cost - true_cost) / (true_cost + 1e-8)
        
        return torch.mean(gap) * 100  # Return as percentage
    
    def _compute_constraint_violation(self, predictions):
        """Compute constraint violations."""
        # In a real implementation, this would check various constraints
        # For now, we'll just return a placeholder
        return torch.tensor(0.0, device=self.device)

def calculate_mse(predictions, targets):
    """
    Calculate Mean Squared Error between predictions and targets.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        
    Returns:
        float: Mean Squared Error value
    """
    squared_diff = (predictions - targets) ** 2
    return torch.mean(squared_diff).item()

def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error between predictions and targets.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        
    Returns:
        float: Mean Absolute Error value
    """
    abs_diff = torch.abs(predictions - targets)
    return torch.mean(abs_diff).item()

def calculate_constraints_violation(constraint_values):
    """
    Calculate the total constraint violation by summing positive values of constraint violations.
    
    Args:
        constraint_values (torch.Tensor): Tensor containing constraint violation values
            Positive values indicate a constraint violation
        
    Returns:
        float: Total constraint violation
    """
    # Only sum positive values (violations)
    violations = torch.clamp(constraint_values, min=0)
    return torch.sum(violations).item()

def evaluate_model(model, data_loader):
    """
    Evaluate a model on a test dataset.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: PyTorch DataLoader containing test data
        
    Returns:
        dict: Dictionary with evaluation metrics (mse, mae, constraint_violation)
    """
    model.eval()
    
    predictions_list = []
    targets_list = []
    constraints_list = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for data in data_loader:
            # Forward pass
            pred = model(data)
            
            # Store predictions and actual values
            predictions_list.append(pred)
            targets_list.append(data.y)
            
            # If constraints are available, store them
            if hasattr(data, 'constraints'):
                constraints_list.append(data.constraints)
    
    # Concatenate data from all batches
    all_predictions = torch.cat(predictions_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate metrics
    results = {
        'mse': calculate_mse(all_predictions, all_targets),
        'mae': calculate_mae(all_predictions, all_targets)
    }
    
    # Add constraint violation if constraints are available
    if constraints_list:
        all_constraints = torch.cat(constraints_list, dim=0)
        results['constraint_violation'] = calculate_constraints_violation(all_constraints)
    
    return results

def compare_models(models, test_loader, metrics_calculator, device='cpu'):
    """
    Compare multiple models on test data.
    
    Args:
        models: Dictionary of {model_name: model}
        test_loader: DataLoader for test data
        metrics_calculator: OPFMetrics instance
        device: Device for torch tensors
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, test_loader)
        
        # Add model name to metrics
        metrics['model'] = name
        results.append(metrics)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def visualize_comparison(comparison_df, metric_names=None, figsize=(12, 8)):
    """
    Visualize model comparison.
    
    Args:
        comparison_df: DataFrame with comparison results
        metric_names: List of metrics to visualize (default: all numeric columns)
        figsize: Figure size
        
    Returns:
        Figure
    """
    # If metric_names not provided, use all numeric columns except 'model'
    if metric_names is None:
        metric_names = [col for col in comparison_df.columns 
                       if col != 'model' and pd.api.types.is_numeric_dtype(comparison_df[col])]
    
    # Create figure
    fig, axes = plt.subplots(len(metric_names), 1, figsize=figsize)
    
    # Make axes iterable if only one metric
    if len(metric_names) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        sns.barplot(x='model', y=metric, data=comparison_df, ax=axes[i])
        axes[i].set_title(f'{metric} by Model')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('')
        
        # Add value labels
        for j, p in enumerate(axes[i].patches):
            axes[i].annotate(f'{p.get_height():.4f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    return fig

def visualize_predictions(model, test_loader, metrics_calculator, 
                         normalizer=None, n_samples=5, figsize=(15, 10)):
    """
    Visualize model predictions against ground truth.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        metrics_calculator: OPFMetrics instance
        normalizer: Optional data normalizer
        n_samples: Number of samples to visualize
        figsize: Figure size
        
    Returns:
        Figure
    """
    model.eval()
    
    # Get a batch of data
    inputs, targets = next(iter(test_loader))
    
    # Make predictions
    with torch.no_grad():
        predictions = model(inputs.to(metrics_calculator.device))
    
    # Denormalize if needed
    if normalizer is not None:
        predictions = normalizer.inverse_transform_outputs(predictions.cpu())
        targets = normalizer.inverse_transform_outputs(targets)
    
    # Convert to numpy for plotting
    predictions = predictions.cpu().numpy()
    targets = targets.numpy()
    
    # Limit to n_samples
    n_samples = min(n_samples, len(predictions))
    predictions = predictions[:n_samples]
    targets = targets[:n_samples]
    
    # Create figure
    n_gen = metrics_calculator.n_gen
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
    
    # Make axes iterable if only one sample
    if n_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(n_samples):
        # Plot generator outputs
        pred_pg = predictions[i, :n_gen]
        true_pg = targets[i, :n_gen]
        
        x = np.arange(n_gen)
        width = 0.35
        
        axes[i].bar(x - width/2, true_pg, width, label='Ground Truth')
        axes[i].bar(x + width/2, pred_pg, width, label='Prediction')
        
        axes[i].set_xlabel('Generator')
        axes[i].set_ylabel('Active Power (p.u.)')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xticks(x)
        axes[i].legend()
    
    plt.tight_layout()
    return fig

def evaluate_gnn_model(model, data_loader, device='cpu'):
    """
    Evaluate a GNN model on a test dataset.
    
    Args:
        model: The PyTorch GNN model to evaluate
        data_loader: PyTorch DataLoader containing graph data
        device: Device for torch tensors
        
    Returns:
        dict: Dictionary with evaluation metrics for both node-level and graph-level predictions
    """
    model.eval()
    model = model.to(device)
    
    node_predictions_list = []
    node_targets_list = []
    graph_predictions_list = []
    graph_targets_list = []
    constraints_list = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass (assuming model returns both node and graph level predictions)
            node_pred, graph_pred = model(batch)
            
            # Store predictions and actual values
            if hasattr(batch, 'x'):
                node_predictions_list.append(node_pred)
                node_targets_list.append(batch.x)
                
            if hasattr(batch, 'y'):
                graph_predictions_list.append(graph_pred)
                graph_targets_list.append(batch.y)
            
            # If constraints are available, store them
            if hasattr(batch, 'constraints'):
                constraints_list.append(batch.constraints)
    
    results = {}
    
    # Calculate node-level metrics if available
    if node_predictions_list:
        all_node_predictions = torch.cat(node_predictions_list, dim=0)
        all_node_targets = torch.cat(node_targets_list, dim=0)
        
        results['node_mse'] = calculate_mse(all_node_predictions, all_node_targets)
        results['node_mae'] = calculate_mae(all_node_predictions, all_node_targets)
    
    # Calculate graph-level metrics if available
    if graph_predictions_list:
        all_graph_predictions = torch.cat(graph_predictions_list, dim=0)
        all_graph_targets = torch.cat(graph_targets_list, dim=0)
        
        results['graph_mse'] = calculate_mse(all_graph_predictions, all_graph_targets)
        results['graph_mae'] = calculate_mae(all_graph_predictions, all_graph_targets)
    
    # Add constraint violation if constraints are available
    if constraints_list:
        all_constraints = torch.cat(constraints_list, dim=0)
        results['constraint_violation'] = calculate_constraints_violation(all_constraints)
    
    return results 

def evaluate_constraint_screening_model(model, data_loader, threshold=0.5, device='cpu'):
    """
    Evaluate a constraint screening model that predicts which constraints will be active.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: PyTorch DataLoader containing test data
        threshold: Threshold for converting probabilities to binary predictions
        device: Device for torch tensors
        
    Returns:
        dict: Dictionary with classification metrics (accuracy, precision, recall, F1, AUC)
    """
    model.eval()
    model = model.to(device)
    
    all_probabilities = []
    all_targets = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                # If batch is a tuple/list, assume it's (features, targets)
                features, targets = batch
                features = features.to(device)
                
                # Forward pass
                probabilities = model(features)
                
                all_probabilities.append(probabilities.cpu())
                all_targets.append(targets.cpu())
            else:
                # For graph data objects
                batch = batch.to(device)
                
                # Forward pass
                probabilities = model(batch)
                
                all_probabilities.append(probabilities.cpu())
                all_targets.append(batch.y.cpu())
    
    # Concatenate all batches
    all_probabilities = torch.cat(all_probabilities, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Convert probabilities to binary predictions using threshold
    all_predictions = (all_probabilities >= threshold).astype(int)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    }
    
    # Calculate AUC if possible (requires probability outputs)
    try:
        if all_probabilities.shape[1] == 1:  # Binary classification
            results['auc'] = roc_auc_score(all_targets, all_probabilities)
        else:  # Multi-class
            results['auc'] = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
    except ValueError:
        # If AUC calculation fails (e.g., due to only one class being present)
        results['auc'] = float('nan')
    
    return results 

def evaluate_warm_starting_model(model, data_loader, optimizer_fn=None, max_iterations=100, device='cpu'):
    """
    Evaluate a warm starting model by measuring the optimization performance
    when starting from the model's predictions.
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: PyTorch DataLoader containing test data
        optimizer_fn: Function that takes initial points and returns iterations and final value
                    If None, only evaluates prediction accuracy
        max_iterations: Maximum number of iterations for the optimizer
        device: Device for torch tensors
        
    Returns:
        dict: Dictionary with warm starting performance metrics
    """
    model.eval()
    model = model.to(device)
    
    predictions_list = []
    targets_list = []
    opt_iterations_list = []
    opt_values_list = []
    cold_start_iterations_list = []
    cold_start_values_list = []
    
    # Collect predictions and optimization results
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                # If batch is a tuple/list, assume it's (features, targets)
                features, targets = batch
                features = features.to(device)
                
                # Forward pass
                predictions = model(features)
                
                # Store predictions and targets
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
                
                # Run optimizer if provided
                if optimizer_fn:
                    for i in range(len(predictions)):
                        # Optimization with warm start
                        warm_iterations, warm_value = optimizer_fn(
                            predictions[i].cpu().numpy(), 
                            features[i].cpu().numpy(),
                            max_iterations
                        )
                        opt_iterations_list.append(warm_iterations)
                        opt_values_list.append(warm_value)
                        
                        # Optimization with cold start (zeros or random)
                        cold_iterations, cold_value = optimizer_fn(
                            np.zeros_like(predictions[i].cpu().numpy()),
                            features[i].cpu().numpy(),
                            max_iterations
                        )
                        cold_start_iterations_list.append(cold_iterations)
                        cold_start_values_list.append(cold_value)
            else:
                # For graph data objects
                batch = batch.to(device)
                
                # Forward pass
                predictions = model(batch)
                
                # Store predictions and targets
                predictions_list.append(predictions.cpu())
                targets_list.append(batch.y.cpu())
                
                # Run optimizer if provided
                # Implementation would depend on the specific problem structure
    
    # Concatenate predictions and targets
    all_predictions = torch.cat(predictions_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate prediction accuracy metrics
    results = {
        'mse': calculate_mse(all_predictions, all_targets),
        'mae': calculate_mae(all_predictions, all_targets)
    }
    
    # Calculate optimization performance metrics if optimizer was used
    if optimizer_fn:
        results['avg_warm_iterations'] = np.mean(opt_iterations_list)
        results['avg_cold_iterations'] = np.mean(cold_start_iterations_list)
        results['iteration_reduction_pct'] = (1 - np.mean(opt_iterations_list) / np.mean(cold_start_iterations_list)) * 100
        
        # Optimality gap between warm start solution and optimal solution
        results['avg_optimality_gap'] = np.mean([(w - t) / (t + 1e-8) for w, t in zip(opt_values_list, all_targets.numpy().flatten())])
        
        # Improvement over cold start
        results['improvement_over_cold_pct'] = (1 - np.mean(opt_values_list) / np.mean(cold_start_values_list)) * 100
    
    return results 

# ---- Cross-Validation Framework ----

def cross_validate_opf_model(
    model_factory: Callable,
    data_config: Any,
    cv_loaders: List[Tuple],
    device: str = 'cpu',
    model_type: str = 'feedforward',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive cross-validation framework for OPF models.
    
    Args:
        model_factory: Function that creates a new model instance
        data_config: Configuration for the data
        cv_loaders: List of (train_loader, val_loader) tuples for cross-validation
        device: Device for torch tensors
        model_type: Type of model ('feedforward', 'gnn', 'constraint_screening', 'warm_starting')
        verbose: Whether to print progress
        
    Returns:
        Dictionary with cross-validation results
    """
    fold_results = []
    all_metrics = []
    training_times = []
    
    # Perform cross-validation
    for fold_idx, (train_loader, val_loader) in enumerate(cv_loaders):
        if verbose:
            print(f"Training fold {fold_idx+1}/{len(cv_loaders)}...")
        
        # Create a new model for each fold
        model = model_factory().to(device)
        
        # Train the model (simplified - in practice, use actual training loop)
        start_time = time.time()
        # ... training code would go here ...
        # For the purpose of this function, we'll assume model is already trained
        train_time = time.time() - start_time
        training_times.append(train_time)
        
        # Evaluate the model based on its type
        if model_type == 'feedforward':
            metrics = evaluate_model(model, val_loader)
        elif model_type == 'gnn':
            metrics = evaluate_gnn_model(model, val_loader, device)
        elif model_type == 'constraint_screening':
            metrics = evaluate_constraint_screening_model(model, val_loader, device=device)
        elif model_type == 'warm_starting':
            metrics = evaluate_warm_starting_model(model, val_loader, device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        metrics['fold'] = fold_idx
        metrics['train_time'] = train_time
        
        fold_results.append((model, metrics))
        all_metrics.append(metrics)
    
    # Aggregate results
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate mean and std for each metric
    mean_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)
    
    # Combine results
    cv_results = {
        'fold_results': fold_results,
        'mean_metrics': mean_metrics.to_dict(),
        'std_metrics': std_metrics.to_dict(),
        'metrics_df': metrics_df,
        'avg_train_time': np.mean(training_times)
    }
    
    return cv_results

# ---- Error Analysis Tools ----

def analyze_prediction_errors(
    model: torch.nn.Module,
    data_loader: Any,
    normalizer: Optional[DataNormalizer] = None,
    case_data: Optional[Dict] = None,
    categorize_by: Optional[List[str]] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Analyze prediction errors across different categories and components.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        normalizer: Optional normalizer for denormalizing predictions
        case_data: Power system case data
        categorize_by: List of categories to analyze errors by
            (e.g., ['load_level', 'congestion'])
        device: Device for torch tensors
        
    Returns:
        Dictionary with error analysis results
    """
    model.eval()
    model = model.to(device)
    
    # Collect predictions, targets and features
    all_predictions = []
    all_targets = []
    all_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                # If batch is a tuple/list, assume it's (features, targets)
                features, targets = batch
                features = features.to(device)
                
                # Forward pass
                predictions = model(features)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_features.append(features.cpu())
            else:
                # For graph data objects
                batch = batch.to(device)
                features = batch.x if hasattr(batch, 'x') else None
                
                # Forward pass
                predictions = model(batch)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu())
                if features is not None:
                    all_features.append(features.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    if all_features:
        features = torch.cat(all_features, dim=0)
    else:
        features = None
    
    # Denormalize if needed
    if normalizer is not None:
        predictions = normalizer.inverse_transform_outputs(predictions)
        targets = normalizer.inverse_transform_outputs(targets)
    
    # Calculate errors
    absolute_errors = torch.abs(predictions - targets)
    squared_errors = (predictions - targets) ** 2
    relative_errors = absolute_errors / (torch.abs(targets) + 1e-8)
    
    # Convert to numpy for analysis
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    absolute_errors_np = absolute_errors.numpy()
    squared_errors_np = squared_errors.numpy()
    relative_errors_np = relative_errors.numpy()
    
    # Overall error statistics
    error_stats = {
        'mae': np.mean(absolute_errors_np),
        'rmse': np.sqrt(np.mean(squared_errors_np)),
        'mean_relative_error': np.mean(relative_errors_np),
        'max_absolute_error': np.max(absolute_errors_np),
        'median_absolute_error': np.median(absolute_errors_np),
        '90th_percentile_error': np.percentile(absolute_errors_np, 90)
    }
    
    # Component-wise error analysis (assuming multi-output predictions)
    n_components = predictions.shape[1] if len(predictions.shape) > 1 else 1
    component_errors = {}
    
    if n_components > 1:
        for i in range(n_components):
            component_errors[f'component_{i}'] = {
                'mae': np.mean(absolute_errors_np[:, i]),
                'rmse': np.sqrt(np.mean(squared_errors_np[:, i])),
                'mean_relative_error': np.mean(relative_errors_np[:, i]),
                'max_absolute_error': np.max(absolute_errors_np[:, i])
            }
    
    # Categorize errors if requested
    categorized_errors = {}
    if categorize_by is not None and features is not None:
        features_np = features.numpy()
        
        for category in categorize_by:
            if category == 'load_level':
                # Example: categorize by total load level
                if case_data:
                    # Assuming first n_load columns are load values
                    n_load = len([b for b in case_data['bus'] if b[2] > 0])
                    load_values = features_np[:, :n_load]
                    total_loads = np.sum(load_values, axis=1)
                    
                    # Create load level categories
                    low_load_idx = total_loads < np.percentile(total_loads, 33)
                    high_load_idx = total_loads > np.percentile(total_loads, 66)
                    medium_load_idx = ~(low_load_idx | high_load_idx)
                    
                    categorized_errors['load_level'] = {
                        'low_load': {
                            'mae': np.mean(absolute_errors_np[low_load_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[low_load_idx]))
                        },
                        'medium_load': {
                            'mae': np.mean(absolute_errors_np[medium_load_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[medium_load_idx]))
                        },
                        'high_load': {
                            'mae': np.mean(absolute_errors_np[high_load_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[high_load_idx]))
                        }
                    }
            
            elif category == 'congestion':
                # Example: categorize by congestion level (if constraint information is available)
                if case_data and 'constraint_values' in case_data:
                    constraint_values = case_data['constraint_values']
                    congestion = np.sum(constraint_values > 0.8, axis=1)  # Count near-binding constraints
                    
                    low_congestion_idx = congestion < 1
                    high_congestion_idx = congestion > 2
                    medium_congestion_idx = ~(low_congestion_idx | high_congestion_idx)
                    
                    categorized_errors['congestion'] = {
                        'low_congestion': {
                            'mae': np.mean(absolute_errors_np[low_congestion_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[low_congestion_idx]))
                        },
                        'medium_congestion': {
                            'mae': np.mean(absolute_errors_np[medium_congestion_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[medium_congestion_idx]))
                        },
                        'high_congestion': {
                            'mae': np.mean(absolute_errors_np[high_congestion_idx]),
                            'rmse': np.sqrt(np.mean(squared_errors_np[high_congestion_idx]))
                        }
                    }
    
    # Compile results
    results = {
        'error_stats': error_stats,
        'component_errors': component_errors,
        'categorized_errors': categorized_errors,
        'predictions': predictions_np,
        'targets': targets_np,
        'absolute_errors': absolute_errors_np,
        'relative_errors': relative_errors_np
    }
    
    return results

def visualize_error_distribution(
    analysis_results: Dict[str, Any], 
    component_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize error distributions from analysis results.
    
    Args:
        analysis_results: Results from analyze_prediction_errors
        component_idx: Index of the component to visualize (if multi-output)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    absolute_errors = analysis_results['absolute_errors']
    relative_errors = analysis_results['relative_errors']
    predictions = analysis_results['predictions']
    targets = analysis_results['targets']
    
    # Extract specific component if provided and if multi-output
    if component_idx is not None and len(absolute_errors.shape) > 1:
        absolute_errors = absolute_errors[:, component_idx]
        relative_errors = relative_errors[:, component_idx]
        predictions = predictions[:, component_idx]
        targets = targets[:, component_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot error histograms
    sns.histplot(absolute_errors.flatten(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Absolute Error Distribution')
    axes[0, 0].set_xlabel('Absolute Error')
    
    sns.histplot(relative_errors.flatten(), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Relative Error Distribution')
    axes[0, 1].set_xlabel('Relative Error')
    
    # Plot prediction vs. targets
    axes[1, 0].scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
    axes[1, 0].plot([min(targets.flatten()), max(targets.flatten())],
                   [min(targets.flatten()), max(targets.flatten())],
                   'r--')
    axes[1, 0].set_title('Predictions vs. Targets')
    axes[1, 0].set_xlabel('Targets')
    axes[1, 0].set_ylabel('Predictions')
    
    # Plot error vs. target
    axes[1, 1].scatter(targets.flatten(), absolute_errors.flatten(), alpha=0.5)
    axes[1, 1].set_title('Error vs. Target Value')
    axes[1, 1].set_xlabel('Target Value')
    axes[1, 1].set_ylabel('Absolute Error')
    
    plt.tight_layout()
    return fig

def visualize_categorical_errors(
    analysis_results: Dict[str, Any],
    category: str,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize errors by category.
    
    Args:
        analysis_results: Results from analyze_prediction_errors
        category: Category to visualize (e.g., 'load_level', 'congestion')
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    if category not in analysis_results['categorized_errors']:
        raise ValueError(f"Category '{category}' not found in analysis results")
    
    category_data = analysis_results['categorized_errors'][category]
    subcategories = list(category_data.keys())
    
    # Extract MAE and RMSE for each subcategory
    mae_values = [category_data[subcat]['mae'] for subcat in subcategories]
    rmse_values = [category_data[subcat]['rmse'] for subcat in subcategories]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot MAE by subcategory
    sns.barplot(x=subcategories, y=mae_values, ax=ax1)
    ax1.set_title(f'MAE by {category}')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel(category.replace('_', ' ').title())
    
    # Plot RMSE by subcategory
    sns.barplot(x=subcategories, y=rmse_values, ax=ax2)
    ax2.set_title(f'RMSE by {category}')
    ax2.set_ylabel('RMSE')
    ax2.set_xlabel(category.replace('_', ' ').title())
    
    plt.tight_layout()
    return fig

# ---- Comparative Model Evaluator ----

def compare_models_statistically(
    models: Dict[str, torch.nn.Module],
    test_loader: Any,
    metrics: List[str] = ['mse', 'mae'],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compare multiple models with statistical significance testing.
    
    Args:
        models: Dictionary mapping model names to PyTorch models
        test_loader: DataLoader with test data
        metrics: List of metrics to compare
        n_bootstrap: Number of bootstrap samples for significance testing
        alpha: Significance level for confidence intervals
        device: Device for torch tensors
        
    Returns:
        Dictionary with comparison results and statistical tests
    """
    model_names = list(models.keys())
    n_models = len(model_names)
    
    # Store all predictions and targets
    all_predictions = {}
    all_targets = []
    all_sample_metrics = {metric: {name: [] for name in model_names} for metric in metrics}
    
    # First, evaluate each model on the test set
    for name, model in models.items():
        model.eval()
        model = model.to(device)
        
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    features, targets = batch
                    features = features.to(device)
                    
                    # Forward pass
                    predictions = model(features)
                    
                    predictions_list.append(predictions.cpu())
                    if not all_targets:  # Only collect targets once
                        targets_list.append(targets.cpu())
                else:
                    batch = batch.to(device)
                    
                    # Forward pass
                    predictions = model(batch)
                    
                    predictions_list.append(predictions.cpu())
                    if not all_targets:  # Only collect targets once
                        targets_list.append(batch.y.cpu())
        
        # Concatenate predictions
        all_predictions[name] = torch.cat(predictions_list, dim=0)
        
        # Collect targets only once
        if not all_targets:
            all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate overall metrics for each model
    overall_metrics = {metric: {name: 0.0 for name in model_names} for metric in metrics}
    
    for metric in metrics:
        for name in model_names:
            if metric == 'mse':
                overall_metrics[metric][name] = calculate_mse(all_predictions[name], all_targets)
            elif metric == 'mae':
                overall_metrics[metric][name] = calculate_mae(all_predictions[name], all_targets)
            # Add other metrics as needed
    
    # Calculate sample-wise metrics for bootstrap analysis
    for i in range(all_targets.shape[0]):
        for name in model_names:
            pred_i = all_predictions[name][i:i+1]
            target_i = all_targets[i:i+1]
            
            for metric in metrics:
                if metric == 'mse':
                    sample_metric = ((pred_i - target_i) ** 2).mean().item()
                elif metric == 'mae':
                    sample_metric = torch.abs(pred_i - target_i).mean().item()
                # Add other metrics as needed
                
                all_sample_metrics[metric][name].append(sample_metric)
    
    # Perform bootstrap sampling for confidence intervals
    bootstrap_metrics = {metric: {name: [] for name in model_names} for metric in metrics}
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        indices = np.random.choice(all_targets.shape[0], all_targets.shape[0], replace=True)
        
        for metric in metrics:
            for name in model_names:
                # Extract metrics for this bootstrap sample
                bootstrap_values = [all_sample_metrics[metric][name][i] for i in indices]
                bootstrap_metrics[metric][name].append(np.mean(bootstrap_values))
    
    # Calculate confidence intervals
    confidence_intervals = {metric: {name: (0.0, 0.0) for name in model_names} for metric in metrics}
    
    for metric in metrics:
        for name in model_names:
            lower = np.percentile(bootstrap_metrics[metric][name], alpha/2 * 100)
            upper = np.percentile(bootstrap_metrics[metric][name], (1 - alpha/2) * 100)
            confidence_intervals[metric][name] = (lower, upper)
    
    # Perform paired statistical tests between models
    statistical_tests = {metric: {} for metric in metrics}
    
    for metric in metrics:
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(
                    all_sample_metrics[metric][name1],
                    all_sample_metrics[metric][name2]
                )
                
                # Store test results
                test_key = f"{name1}_vs_{name2}"
                statistical_tests[metric][test_key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'better_model': name1 if t_stat < 0 else name2 if t_stat > 0 else None
                }
    
    # Create comparison DataFrame
    comparison_data = []
    
    for name in model_names:
        model_data = {'model': name}
        
        for metric in metrics:
            model_data[metric] = overall_metrics[metric][name]
            model_data[f"{metric}_lower"] = confidence_intervals[metric][name][0]
            model_data[f"{metric}_upper"] = confidence_intervals[metric][name][1]
            
        comparison_data.append(model_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Determine best model for each metric
    best_models = {}
    
    for metric in metrics:
        if metric in ['mse', 'mae']:  # Lower is better
            best_idx = comparison_df[metric].idxmin()
        else:  # Higher is better (e.g., r2)
            best_idx = comparison_df[metric].idxmax()
            
        best_models[metric] = comparison_df.loc[best_idx, 'model']
    
    # Return all results
    return {
        'overall_metrics': overall_metrics,
        'confidence_intervals': confidence_intervals,
        'statistical_tests': statistical_tests,
        'comparison_df': comparison_df,
        'best_models': best_models,
        'all_predictions': all_predictions,
        'all_targets': all_targets
    }

def visualize_model_comparison(
    comparison_results: Dict[str, Any],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize model comparison results with confidence intervals.
    
    Args:
        comparison_results: Results from compare_models_statistically
        metrics: List of metrics to visualize (default: all metrics in the comparison)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    comparison_df = comparison_results['comparison_df']
    
    # If metrics not specified, use all metrics from the comparison
    if metrics is None:
        metrics = [col for col in comparison_df.columns 
                 if col not in ['model'] and not col.endswith('_lower') and not col.endswith('_upper')]
    
    # Create figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    # Make axes iterable if only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Extract model names, metric values, and confidence intervals
        models = comparison_df['model']
        values = comparison_df[metric]
        lower = comparison_df[f"{metric}_lower"]
        upper = comparison_df[f"{metric}_upper"]
        
        # Compute error bars
        error_bars = [(v - l, u - v) for v, l, u in zip(values, lower, upper)]
        error_bars_array = np.array(error_bars).T
        
        # Plot bar chart with error bars
        bars = axes[i].bar(models, values, yerr=error_bars_array, capsize=10)
        
        # Highlight best model
        best_model = comparison_results['best_models'][metric]
        best_idx = list(models).index(best_model)
        bars[best_idx].set_color('green')
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{values.iloc[j]:.4f}', ha='center', va='bottom')
        
        # Add title and labels
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        
        # Add significance markers
        if 'statistical_tests' in comparison_results and metric in comparison_results['statistical_tests']:
            tests = comparison_results['statistical_tests'][metric]
            y_max = max(values) * 1.1
            
            for test_key, test_result in tests.items():
                if test_result['significant']:
                    model1, model2 = test_key.split('_vs_')
                    idx1 = list(models).index(model1)
                    idx2 = list(models).index(model2)
                    
                    # Add significance star
                    x = (idx1 + idx2) / 2
                    axes[i].annotate('*', xy=(x, y_max), xytext=(0, 5), 
                                  textcoords='offset points', ha='center', va='bottom',
                                  fontsize=16)
    
    plt.tight_layout()
    return fig

# ---- Solution Quality Assessment ----

def calculate_power_flow_violations(
    predictions: torch.Tensor,
    case_data: Dict[str, Any],
    normalizer: Optional[DataNormalizer] = None,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Calculate violations of power flow constraints based on model predictions.
    
    Args:
        predictions: Model predictions (generator outputs and voltages)
        case_data: PyPOWER case data
        normalizer: Optional data normalizer for denormalizing predictions
        device: Device for torch tensors
        
    Returns:
        Dictionary with constraint violation metrics
    """
    # Denormalize predictions if needed
    if normalizer is not None:
        predictions = normalizer.inverse_transform_outputs(predictions)
    
    # Extract system parameters from case data
    n_gen = len(case_data['gen'])
    n_bus = len(case_data['bus'])
    n_branch = len(case_data['branch'])
    baseMVA = case_data['baseMVA']
    
    # Convert predictions to numpy for easier handling
    predictions_np = predictions.cpu().numpy()
    
    # Extract predicted generator outputs and voltages
    pred_pg = predictions_np[:, :n_gen]
    pred_qg = predictions_np[:, n_gen:2*n_gen] if predictions_np.shape[1] >= 2*n_gen else None
    pred_vm = predictions_np[:, 2*n_gen:2*n_gen+n_bus] if predictions_np.shape[1] >= 2*n_gen+n_bus else None
    pred_va = predictions_np[:, 2*n_gen+n_bus:] if predictions_np.shape[1] >= 2*n_gen+2*n_bus else None
    
    # Initialize violation counters
    gen_p_violations = 0
    gen_q_violations = 0
    voltage_violations = 0
    branch_violations = 0
    power_balance_violations = 0
    
    # Check generator active power limits
    for g in range(n_gen):
        p_min = case_data['gen'][g][9] / baseMVA  # PMIN
        p_max = case_data['gen'][g][8] / baseMVA  # PMAX
        
        # Count violations
        if pred_pg is not None:
            below_min = pred_pg[:, g] < p_min
            above_max = pred_pg[:, g] > p_max
            gen_p_violations += np.sum(below_min) + np.sum(above_max)
    
    # Check generator reactive power limits (if available)
    if pred_qg is not None:
        for g in range(n_gen):
            q_min = case_data['gen'][g][4] / baseMVA  # QMIN
            q_max = case_data['gen'][g][3] / baseMVA  # QMAX
            
            # Count violations
            below_min = pred_qg[:, g] < q_min
            above_max = pred_qg[:, g] > q_max
            gen_q_violations += np.sum(below_min) + np.sum(above_max)
    
    # Check voltage magnitude limits (if available)
    if pred_vm is not None:
        for b in range(n_bus):
            v_min = case_data['bus'][b][12]  # VMIN
            v_max = case_data['bus'][b][11]  # VMAX
            
            # Count violations
            below_min = pred_vm[:, b] < v_min
            above_max = pred_vm[:, b] > v_max
            voltage_violations += np.sum(below_min) + np.sum(above_max)
    
    # In a complete implementation, we would also check:
    # 1. Branch flow limits using power flow calculations
    # 2. Power balance constraints
    # These would require more complex calculations using network topology
    
    # Calculate total violations and violation rate
    total_samples = predictions_np.shape[0]
    total_constraints = n_gen  # P limits
    if pred_qg is not None:
        total_constraints += n_gen  # Q limits
    if pred_vm is not None:
        total_constraints += n_bus  # V limits
    
    total_violations = gen_p_violations + gen_q_violations + voltage_violations
    violation_rate = total_violations / (total_samples * total_constraints)
    
    return {
        'gen_p_violations': gen_p_violations,
        'gen_q_violations': gen_q_violations,
        'voltage_violations': voltage_violations,
        'branch_violations': branch_violations,
        'power_balance_violations': power_balance_violations,
        'total_violations': total_violations,
        'violation_rate': violation_rate
    }

def calculate_distance_to_feasibility(
    predictions: torch.Tensor,
    case_data: Dict[str, Any],
    normalizer: Optional[DataNormalizer] = None,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Calculate the distance from ML predictions to the nearest feasible solution.
    
    Args:
        predictions: Model predictions
        case_data: PyPOWER case data
        normalizer: Optional data normalizer
        device: Device for torch tensors
        
    Returns:
        Dictionary with distance to feasibility metrics
    """
    # Denormalize predictions if needed
    if normalizer is not None:
        predictions = normalizer.inverse_transform_outputs(predictions)
    
    # Extract system parameters from case data
    n_gen = len(case_data['gen'])
    n_bus = len(case_data['bus'])
    baseMVA = case_data['baseMVA']
    
    # Convert predictions to numpy for easier handling
    predictions_np = predictions.cpu().numpy()
    
    # Extract predicted generator outputs and voltages
    pred_pg = predictions_np[:, :n_gen]
    pred_qg = predictions_np[:, n_gen:2*n_gen] if predictions_np.shape[1] >= 2*n_gen else None
    pred_vm = predictions_np[:, 2*n_gen:2*n_gen+n_bus] if predictions_np.shape[1] >= 2*n_gen+n_bus else None
    
    # Calculate distance to feasible generator active power limits
    p_distance = 0
    for g in range(n_gen):
        p_min = case_data['gen'][g][9] / baseMVA  # PMIN
        p_max = case_data['gen'][g][8] / baseMVA  # PMAX
        
        # Calculate distance to feasible region
        below_min_dist = np.maximum(0, p_min - pred_pg[:, g])
        above_max_dist = np.maximum(0, pred_pg[:, g] - p_max)
        p_distance += np.sum(below_min_dist) + np.sum(above_max_dist)
    
    # Calculate distance to feasible generator reactive power limits
    q_distance = 0
    if pred_qg is not None:
        for g in range(n_gen):
            q_min = case_data['gen'][g][4] / baseMVA  # QMIN
            q_max = case_data['gen'][g][3] / baseMVA  # QMAX
            
            # Calculate distance to feasible region
            below_min_dist = np.maximum(0, q_min - pred_qg[:, g])
            above_max_dist = np.maximum(0, pred_qg[:, g] - q_max)
            q_distance += np.sum(below_min_dist) + np.sum(above_max_dist)
    
    # Calculate distance to feasible voltage magnitude limits
    v_distance = 0
    if pred_vm is not None:
        for b in range(n_bus):
            v_min = case_data['bus'][b][12]  # VMIN
            v_max = case_data['bus'][b][11]  # VMAX
            
            # Calculate distance to feasible region
            below_min_dist = np.maximum(0, v_min - pred_vm[:, b])
            above_max_dist = np.maximum(0, pred_vm[:, b] - v_max)
            v_distance += np.sum(below_min_dist) + np.sum(above_max_dist)
    
    # Calculate total distance and average distance per sample
    total_samples = predictions_np.shape[0]
    total_distance = p_distance + q_distance + v_distance
    avg_distance_per_sample = total_distance / total_samples
    
    return {
        'p_distance': p_distance,
        'q_distance': q_distance,
        'v_distance': v_distance,
        'total_distance': total_distance,
        'avg_distance_per_sample': avg_distance_per_sample
    }

def evaluate_solution_feasibility(
    model: torch.nn.Module,
    data_loader: Any,
    case_data: Dict[str, Any],
    normalizer: Optional[DataNormalizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate the feasibility of solutions produced by ML models.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        case_data: PyPOWER case data
        normalizer: Optional data normalizer
        device: Device for torch tensors
        
    Returns:
        Dictionary with solution feasibility metrics
    """
    model.eval()
    model = model.to(device)
    
    predictions_list = []
    targets_list = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                features, targets = batch
                features = features.to(device)
                
                # Forward pass
                predictions = model(features)
                
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
            else:
                batch = batch.to(device)
                
                # Forward pass
                predictions = model(batch)
                
                predictions_list.append(predictions.cpu())
                targets_list.append(batch.y.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(predictions_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate power flow violations
    violations = calculate_power_flow_violations(
        all_predictions, case_data, normalizer, device
    )
    
    # Calculate distance to feasibility
    feasibility_distance = calculate_distance_to_feasibility(
        all_predictions, case_data, normalizer, device
    )
    
    # Calculate prediction accuracy
    if normalizer is not None:
        norm_predictions = all_predictions
        norm_targets = all_targets
        predictions = normalizer.inverse_transform_outputs(all_predictions)
        targets = normalizer.inverse_transform_outputs(all_targets)
    else:
        norm_predictions = all_predictions
        norm_targets = all_targets
        predictions = all_predictions
        targets = all_targets
    
    accuracy_metrics = {
        'mse': calculate_mse(norm_predictions, norm_targets),
        'mae': calculate_mae(norm_predictions, norm_targets),
        'denorm_mse': calculate_mse(predictions, targets),
        'denorm_mae': calculate_mae(predictions, targets)
    }
    
    # Combine all results
    results = {
        'violations': violations,
        'feasibility_distance': feasibility_distance,
        'accuracy_metrics': accuracy_metrics
    }
    
    return results

# ---- Robustness Testing ----

def perform_sensitivity_analysis(
    model: torch.nn.Module,
    base_input: torch.Tensor,
    perturbation_range: float = 0.1,
    n_points: int = 10,
    component_idx: Optional[int] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis by perturbing inputs.
    
    Args:
        model: Trained PyTorch model
        base_input: Base input tensor to perturb
        perturbation_range: Range of perturbation as fraction of original value
        n_points: Number of perturbation points
        component_idx: Index of specific input component to perturb (None = all)
        device: Device for torch tensors
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    model.eval()
    model = model.to(device)
    base_input = base_input.to(device)
    
    # Get base prediction
    with torch.no_grad():
        base_prediction = model(base_input).cpu()
    
    # Determine perturbation points
    perturbation_factors = torch.linspace(
        1 - perturbation_range, 
        1 + perturbation_range, 
        n_points
    )
    
    # Initialize results containers
    perturbed_inputs = []
    perturbed_predictions = []
    sensitivities = []
    
    # Perform perturbations
    for factor in perturbation_factors:
        if component_idx is not None:
            # Perturb only one component
            perturbed_input = base_input.clone()
            perturbed_input[0, component_idx] *= factor
        else:
            # Perturb all components
            perturbed_input = base_input * factor
        
        # Get prediction for perturbed input
        with torch.no_grad():
            perturbed_prediction = model(perturbed_input).cpu()
        
        # Calculate sensitivity
        # Sensitivity = % change in output / % change in input
        output_change = (perturbed_prediction - base_prediction) / base_prediction
        input_change = (factor - 1)
        sensitivity = output_change / input_change if input_change != 0 else torch.zeros_like(output_change)
        
        # Store results
        perturbed_inputs.append(perturbed_input.cpu())
        perturbed_predictions.append(perturbed_prediction)
        sensitivities.append(sensitivity)
    
    # Concatenate results
    perturbed_inputs = torch.cat(perturbed_inputs, dim=0)
    perturbed_predictions = torch.cat(perturbed_predictions, dim=0)
    sensitivities = torch.cat(sensitivities, dim=0)
    
    # Calculate average sensitivities
    avg_sensitivity = torch.mean(torch.abs(sensitivities), dim=0)
    
    # Create sensitivity ranking
    if avg_sensitivity.dim() > 0:
        ranked_indices = torch.argsort(avg_sensitivity, descending=True)
        sensitivity_ranking = [(int(idx.item()), float(avg_sensitivity[idx].item())) 
                              for idx in ranked_indices]
    else:
        sensitivity_ranking = [(0, float(avg_sensitivity.item()))]
    
    return {
        'perturbation_factors': perturbation_factors.numpy(),
        'perturbed_inputs': perturbed_inputs.numpy(),
        'perturbed_predictions': perturbed_predictions.numpy(),
        'sensitivities': sensitivities.numpy(),
        'avg_sensitivity': avg_sensitivity.numpy(),
        'sensitivity_ranking': sensitivity_ranking
    }

def evaluate_robustness(
    model: torch.nn.Module,
    data_loader: Any,
    noise_level: float = 0.05,
    n_repetitions: int = 10,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate model robustness by adding noise to inputs.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        noise_level: Level of Gaussian noise to add (fraction of std)
        n_repetitions: Number of times to repeat with different noise
        device: Device for torch tensors
        
    Returns:
        Dictionary with robustness evaluation results
    """
    model.eval()
    model = model.to(device)
    
    # Store clean predictions and targets
    clean_predictions = []
    all_targets = []
    
    # Get clean predictions
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                features, targets = batch
                features = features.to(device)
                
                # Forward pass
                predictions = model(features)
                
                clean_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
            else:
                batch = batch.to(device)
                
                # Forward pass
                predictions = model(batch)
                
                clean_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu())
    
    # Concatenate batches
    clean_predictions = torch.cat(clean_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate baseline metrics
    clean_mse = calculate_mse(clean_predictions, all_targets)
    clean_mae = calculate_mae(clean_predictions, all_targets)
    
    # Store noisy results
    noisy_mse_values = []
    noisy_mae_values = []
    prediction_stds = []
    
    # Repeat with different noise realizations
    for _ in range(n_repetitions):
        noisy_predictions = []
        
        # Get predictions with noisy inputs
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    features, _ = batch
                    
                    # Add noise
                    noise = torch.randn_like(features) * features.std() * noise_level
                    noisy_features = features + noise
                    noisy_features = noisy_features.to(device)
                    
                    # Forward pass with noisy features
                    predictions = model(noisy_features)
                    
                    noisy_predictions.append(predictions.cpu())
                else:
                    batch = batch.to(device)
                    
                    if hasattr(batch, 'x'):
                        # Add noise to node features
                        noise = torch.randn_like(batch.x) * batch.x.std() * noise_level
                        batch.x = batch.x + noise
                    
                    # Forward pass with noisy features
                    predictions = model(batch)
                    
                    noisy_predictions.append(predictions.cpu())
        
        # Concatenate batches
        noisy_predictions = torch.cat(noisy_predictions, dim=0)
        
        # Calculate metrics with noise
        noisy_mse = calculate_mse(noisy_predictions, all_targets)
        noisy_mae = calculate_mae(noisy_predictions, all_targets)
        
        noisy_mse_values.append(noisy_mse)
        noisy_mae_values.append(noisy_mae)
        
        # Calculate prediction stability
        if len(prediction_stds) == 0:
            # Initialize array for prediction stds
            prediction_stds = torch.zeros_like(noisy_predictions)
            prediction_means = noisy_predictions
        else:
            # Update using Welford's online algorithm
            old_mean = prediction_means
            prediction_means = old_mean + (noisy_predictions - old_mean) / (len(prediction_stds) + 1)
            prediction_stds = prediction_stds + (noisy_predictions - old_mean) * (noisy_predictions - prediction_means)
    
    # Finalize prediction standard deviations
    prediction_stds = torch.sqrt(prediction_stds / n_repetitions)
    avg_prediction_std = torch.mean(prediction_stds).item()
    
    # Calculate robustness metrics
    avg_noisy_mse = np.mean(noisy_mse_values)
    avg_noisy_mae = np.mean(noisy_mae_values)
    std_noisy_mse = np.std(noisy_mse_values)
    std_noisy_mae = np.std(noisy_mae_values)
    
    mse_degradation = (avg_noisy_mse - clean_mse) / clean_mse * 100
    mae_degradation = (avg_noisy_mae - clean_mae) / clean_mae * 100
    
    # Compile results
    results = {
        'clean_mse': clean_mse,
        'clean_mae': clean_mae,
        'noisy_mse_mean': avg_noisy_mse,
        'noisy_mse_std': std_noisy_mse,
        'noisy_mae_mean': avg_noisy_mae,
        'noisy_mae_std': std_noisy_mae,
        'mse_degradation_percent': mse_degradation,
        'mae_degradation_percent': mae_degradation,
        'prediction_stability': avg_prediction_std,
        'noise_level': noise_level,
        'n_repetitions': n_repetitions
    }
    
    return results

# ---- Execution Time Benchmarking ----

def benchmark_execution_time(
    model: torch.nn.Module,
    data_loader: Any,
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark model execution time.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        n_runs: Number of runs to average over
        warmup_runs: Number of warmup runs
        device: Device for torch tensors
        
    Returns:
        Dictionary with execution time metrics
    """
    model.eval()
    model = model.to(device)
    
    # Get a sample batch
    sample_batch = next(iter(data_loader))
    
    if isinstance(sample_batch, tuple) or isinstance(sample_batch, list):
        features, _ = sample_batch
        sample_input = features[0:1].to(device)  # Use first sample
    else:
        sample_input = sample_batch[0:1].to(device)  # Use first sample
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Benchmarking runs
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_sample = total_time / n_runs
    samples_per_second = n_runs / total_time
    
    return {
        'total_time': total_time,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': samples_per_second,
        'n_runs': n_runs
    }

def compare_execution_with_solver(
    model: torch.nn.Module,
    test_loader: Any,
    solver_fn: Callable,
    n_samples: int = 10,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compare execution time between ML model and traditional solver.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        solver_fn: Function that solves OPF using traditional methods
        n_samples: Number of samples to compare
        device: Device for torch tensors
        
    Returns:
        Dictionary with comparison results
    """
    model.eval()
    model = model.to(device)
    
    # Collect samples
    samples = []
    targets = []
    
    for batch in test_loader:
        if isinstance(batch, tuple) or isinstance(batch, list):
            features, target = batch
            samples.append(features.cpu())
            targets.append(target.cpu())
        else:
            samples.append(batch.cpu())
            if hasattr(batch, 'y'):
                targets.append(batch.y.cpu())
        
        if len(samples) >= n_samples:
            break
    
    # Measure ML model execution time
    ml_times = []
    ml_results = []
    
    for sample in samples:
        sample = sample.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            prediction = model(sample)
        end_time = time.time()
        
        ml_time = end_time - start_time
        ml_times.append(ml_time)
        ml_results.append(prediction.cpu())
    
    # Measure traditional solver execution time
    solver_times = []
    solver_results = []
    
    for sample in samples:
        sample_np = sample.numpy()
        
        start_time = time.time()
        solver_result = solver_fn(sample_np)
        end_time = time.time()
        
        solver_time = end_time - start_time
        solver_times.append(solver_time)
        solver_results.append(solver_result)
    
    # Calculate speedup
    avg_ml_time = np.mean(ml_times)
    avg_solver_time = np.mean(solver_times)
    speedup = avg_solver_time / avg_ml_time
    
    # Calculate solution quality comparison
    solution_gaps = []
    
    for i in range(len(ml_results)):
        ml_sol = ml_results[i].numpy()
        solver_sol = solver_results[i]
        
        # Calculate relative difference
        rel_diff = np.mean(np.abs(ml_sol - solver_sol) / (np.abs(solver_sol) + 1e-8))
        solution_gaps.append(rel_diff)
    
    avg_solution_gap = np.mean(solution_gaps)
    
    # Compile results
    results = {
        'avg_ml_time': avg_ml_time,
        'avg_solver_time': avg_solver_time,
        'speedup_factor': speedup,
        'avg_solution_gap': avg_solution_gap,
        'ml_times': ml_times,
        'solver_times': solver_times,
        'solution_gaps': solution_gaps
    }
    
    return results

def visualize_execution_comparison(
    comparison_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize execution time comparison.
    
    Args:
        comparison_results: Results from compare_execution_with_solver
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    ml_times = comparison_results['ml_times']
    solver_times = comparison_results['solver_times']
    speedup = comparison_results['speedup_factor']
    solution_gap = comparison_results['avg_solution_gap']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot execution times
    ax1.bar(['ML Model', 'Traditional Solver'], 
            [np.mean(ml_times), np.mean(solver_times)])
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title(f'Average Execution Time\n(Speedup: {speedup:.2f}x)')
    
    # Plot speedup vs. solution gap
    ax2.scatter(ml_times, solver_times)
    ax2.plot([0, max(solver_times)], [0, max(solver_times)], 'r--')  # y=x line
    ax2.set_xlabel('ML Model Time (s)')
    ax2.set_ylabel('Solver Time (s)')
    ax2.set_title(f'ML vs. Solver Time\n(Solution Gap: {solution_gap*100:.2f}%)')
    
    plt.tight_layout()
    return fig

# ---- Interactive Visualization Dashboard ----

def create_model_evaluation_dashboard(
    evaluation_results: Dict[str, Any],
    case_data: Optional[Dict] = None,
    output_file: str = 'evaluation_dashboard.html'
) -> None:
    """
    Create an interactive dashboard for model evaluation results.
    
    Note: This function requires plotly to be installed.
    
    Args:
        evaluation_results: Dictionary with evaluation results
        case_data: Optional power system case data
        output_file: Path to save the HTML dashboard
        
    Returns:
        None (saves dashboard to file)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly is required for interactive dashboards. Install with: pip install plotly")
        return
    
    # Extract data from evaluation results
    predictions = evaluation_results.get('predictions', None)
    targets = evaluation_results.get('targets', None)
    metrics = evaluation_results.get('accuracy_metrics', {})
    
    if predictions is None or targets is None:
        print("Predictions and targets are required for dashboard creation")
        return
    
    # Create dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Predictions vs Targets', 'Error Distribution',
            'Component-wise Metrics', 'Error Heatmap',
            'Metric Summary', 'System Diagram'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'heatmap'}],
            [{'type': 'table'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Plot 1: Predictions vs Targets scatter
    if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
        for i in range(min(predictions.shape[1], 5)):  # Show up to 5 components
            fig.add_trace(
                go.Scatter(
                    x=targets[:, i],
                    y=predictions[:, i],
                    mode='markers',
                    name=f'Component {i+1}',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Add diagonal line (perfect predictions)
        min_val = min(np.min(predictions), np.min(targets))
        max_val = max(np.max(predictions), np.max(targets))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='y=x',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
    
    # Plot 2: Error distribution histogram
    if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
        errors = predictions - targets
        
        fig.add_trace(
            go.Histogram(
                x=errors.flatten(),
                name='Error Distribution',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=2
        )
    
    # Plot 3: Component-wise metrics bar chart
    component_errors = evaluation_results.get('component_errors', {})
    
    if component_errors:
        components = list(component_errors.keys())
        mae_values = [component_errors[comp]['mae'] for comp in components]
        rmse_values = [component_errors[comp]['rmse'] for comp in components]
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=mae_values,
                name='MAE',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=rmse_values,
                name='RMSE',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Plot 4: Error heatmap
    if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
        if predictions.shape[1] <= 20:  # Only show heatmap for reasonable dimensions
            abs_errors = np.abs(predictions - targets)
            
            fig.add_trace(
                go.Heatmap(
                    z=abs_errors.T,
                    colorscale='Viridis',
                    name='Absolute Error'
                ),
                row=2, col=2
            )
    
    # Plot 5: Metrics summary table
    if metrics:
        metric_names = list(metrics.keys())
        metric_values = [f"{metrics[name]:.6f}" for name in metric_names]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[metric_names, metric_values])
            ),
            row=3, col=1
        )
    
    # Plot 6: System diagram (if case_data available)
    if case_data is not None:
        try:
            # Create simplified power system diagram
            bus_pos = case_data.get('bus_pos', {})
            
            if not bus_pos and 'bus' in case_data:
                # Create positions if not available
                n_bus = len(case_data['bus'])
                angles = np.linspace(0, 2*np.pi, n_bus, endpoint=False)
                radius = 10
                bus_pos = {
                    i: (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
                    for i in range(n_bus)
                }
            
            # Add buses
            bus_x = [pos[0] for pos in bus_pos.values()]
            bus_y = [pos[1] for pos in bus_pos.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=bus_x,
                    y=bus_y,
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    name='Buses',
                    text=[f"Bus {i}" for i in bus_pos.keys()]
                ),
                row=3, col=2
            )
            
            # Add branches if available
            if 'branch' in case_data:
                branch_x = []
                branch_y = []
                
                for branch in case_data['branch']:
                    f_bus = int(branch[0]) - 1  # PyPOWER is 1-indexed
                    t_bus = int(branch[1]) - 1
                    
                    if f_bus in bus_pos and t_bus in bus_pos:
                        # Add from bus
                        branch_x.append(bus_pos[f_bus][0])
                        branch_y.append(bus_pos[f_bus][1])
                        
                        # Add to bus
                        branch_x.append(bus_pos[t_bus][0])
                        branch_y.append(bus_pos[t_bus][1])
                        
                        # Add None to break the line
                        branch_x.append(None)
                        branch_y.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=branch_x,
                        y=branch_y,
                        mode='lines',
                        line=dict(color='gray', width=1),
                        name='Branches'
                    ),
                    row=3, col=2
                )
            
            # Add generators if available
            if 'gen' in case_data:
                gen_bus = [int(gen[0]) - 1 for gen in case_data['gen']]  # PyPOWER is 1-indexed
                gen_x = [bus_pos[bus][0] for bus in gen_bus if bus in bus_pos]
                gen_y = [bus_pos[bus][1] for bus in gen_bus if bus in bus_pos]
                
                fig.add_trace(
                    go.Scatter(
                        x=gen_x,
                        y=gen_y,
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name='Generators'
                    ),
                    row=3, col=2
                )
        except Exception as e:
            print(f"Error creating system diagram: {e}")
    
    # Update layout
    fig.update_layout(
        title_text="Model Evaluation Dashboard",
        height=900,
        width=1200,
        showlegend=True
    )
    
    # Save to HTML file
    fig.write_html(output_file)
    print(f"Dashboard saved to {output_file}")

def visualize_power_system_predictions(
    predictions: np.ndarray,
    case_data: Dict[str, Any],
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize predictions on a power system diagram.
    
    Args:
        predictions: Model predictions
        case_data: PyPOWER case data
        sample_idx: Index of the sample to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    # Extract power system parameters
    n_gen = len(case_data['gen'])
    n_bus = len(case_data['bus'])
    n_branch = len(case_data['branch'])
    
    # Extract positions (or create positions if not available)
    bus_pos = case_data.get('bus_pos', {})
    
    if not bus_pos:
        # Create positions in a circle
        angles = np.linspace(0, 2*np.pi, n_bus, endpoint=False)
        radius = 10
        bus_pos = {
            i: (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
            for i in range(n_bus)
        }
    
    # Extract predictions for the specified sample
    sample_pred = predictions[sample_idx]
    
    # Extract generator outputs
    gen_p = sample_pred[:n_gen] if sample_pred.shape[0] >= n_gen else None
    gen_q = sample_pred[n_gen:2*n_gen] if sample_pred.shape[0] >= 2*n_gen else None
    
    # Extract voltage magnitudes and angles
    vm = sample_pred[2*n_gen:2*n_gen+n_bus] if sample_pred.shape[0] >= 2*n_gen+n_bus else None
    va = sample_pred[2*n_gen+n_bus:] if sample_pred.shape[0] >= 2*n_gen+2*n_bus else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot branches
    for branch in case_data['branch']:
        f_bus = int(branch[0]) - 1  # PyPOWER is 1-indexed
        t_bus = int(branch[1]) - 1
        
        if f_bus in bus_pos and t_bus in bus_pos:
            ax.plot(
                [bus_pos[f_bus][0], bus_pos[t_bus][0]],
                [bus_pos[f_bus][1], bus_pos[t_bus][1]],
                'k-', alpha=0.5, linewidth=1
            )
    
    # Plot buses
    bus_x = [pos[0] for pos in bus_pos.values()]
    bus_y = [pos[1] for pos in bus_pos.values()]
    
    if vm is not None:
        # Use voltage magnitude to determine bus color
        sc = ax.scatter(
            bus_x, bus_y, 
            c=vm, cmap='viridis', 
            s=100, edgecolor='k', zorder=10
        )
        plt.colorbar(sc, ax=ax, label='Voltage Magnitude (p.u.)')
    else:
        # Use default color
        ax.scatter(
            bus_x, bus_y, 
            c='skyblue', 
            s=100, edgecolor='k', zorder=10
        )
    
    # Plot generators
    gen_bus = [int(gen[0]) - 1 for gen in case_data['gen']]  # PyPOWER is 1-indexed
    gen_x = [bus_pos[bus][0] for bus in gen_bus if bus in bus_pos]
    gen_y = [bus_pos[bus][1] for bus in gen_bus if bus in bus_pos]
    
    if gen_p is not None:
        # Use generator output to determine size
        max_size = 300
        sizes = [max_size * (p / max(gen_p) if max(gen_p) > 0 else 1) for p in gen_p]
        ax.scatter(
            gen_x, gen_y, 
            c='red', s=sizes, 
            marker='*', edgecolor='k', 
            zorder=20, label='Generators'
        )
        
        # Add generator output labels
        for i, bus in enumerate(gen_bus):
            if bus in bus_pos:
                ax.annotate(
                    f"P={gen_p[i]:.2f}",
                    xy=(bus_pos[bus][0], bus_pos[bus][1]),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
    else:
        ax.scatter(
            gen_x, gen_y, 
            c='red', s=200, 
            marker='*', edgecolor='k', 
            zorder=20, label='Generators'
        )
    
    # Add bus labels
    for bus_idx, pos in bus_pos.items():
        ax.annotate(
            f"{bus_idx+1}",
            xy=pos,
            xytext=(0, -15),
            textcoords='offset points',
            ha='center'
        )
    
    # Set titles and labels
    ax.set_title(f"Power System Prediction Visualization (Sample {sample_idx})")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_model_comparison_dashboard(
    comparison_results: Dict[str, Any],
    output_file: str = 'model_comparison_dashboard.html'
) -> None:
    """
    Create an interactive dashboard for model comparison.
    
    Note: This function requires plotly to be installed.
    
    Args:
        comparison_results: Results from compare_models_statistically
        output_file: Path to save the HTML dashboard
        
    Returns:
        None (saves dashboard to file)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly is required for interactive dashboards. Install with: pip install plotly")
        return
    
    # Extract data from comparison results
    comparison_df = comparison_results.get('comparison_df', None)
    
    if comparison_df is None:
        print("Comparison DataFrame is required for dashboard creation")
        return
    
    # Get metrics and models
    metrics = [col for col in comparison_df.columns 
              if col not in ['model'] and not col.endswith('_lower') and not col.endswith('_upper')]
    models = comparison_df['model'].tolist()
    
    # Create dashboard
    fig = make_subplots(
        rows=len(metrics) + 1, cols=2,
        subplot_titles=(
            ['Model Comparison: ' + metric.upper() for metric in metrics] +
            ['Statistical Significance Matrix', 'Performance Overview']
        ),
        specs=(
            # One row per metric, with bar chart and box plot
            [{'type': 'bar'}, {'type': 'box'}] * len(metrics) +
            # Last row with heatmap and radar chart
            [{'type': 'heatmap'}, {'type': 'polar'}]
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.1
    )
    
    # Add bar charts and box plots for each metric
    for i, metric in enumerate(metrics):
        # Bar chart with confidence intervals
        values = comparison_df[metric]
        lower = comparison_df[f"{metric}_lower"]
        upper = comparison_df[f"{metric}_upper"]
        
        # Compute error bars
        error_y = dict(
            type='data',
            array=upper - values,
            arrayminus=values - lower,
            visible=True
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                error_y=error_y,
                name=metric.upper(),
                marker_color='blue',
                opacity=0.7
            ),
            row=i+1, col=1
        )
        
        # Box plot from bootstrap samples (if available)
        if 'bootstrap_metrics' in comparison_results and metric in comparison_results['bootstrap_metrics']:
            bootstrap_data = comparison_results['bootstrap_metrics'][metric]
            
            for j, model in enumerate(models):
                if model in bootstrap_data:
                    fig.add_trace(
                        go.Box(
                            y=bootstrap_data[model],
                            name=model,
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        row=i+1, col=2
                    )
    
    # Add statistical significance heatmap
    if 'statistical_tests' in comparison_results and metrics[0] in comparison_results['statistical_tests']:
        # Create significance matrix
        sig_matrix = np.zeros((len(models), len(models)))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    test_key = f"{model1}_vs_{model2}"
                    rev_test_key = f"{model2}_vs_{model1}"
                    
                    if test_key in comparison_results['statistical_tests'][metrics[0]]:
                        test = comparison_results['statistical_tests'][metrics[0]][test_key]
                        sig_matrix[i, j] = -np.log10(test['p_value']) if test['significant'] else 0
                    elif rev_test_key in comparison_results['statistical_tests'][metrics[0]]:
                        test = comparison_results['statistical_tests'][metrics[0]][rev_test_key]
                        sig_matrix[i, j] = -np.log10(test['p_value']) if test['significant'] else 0
        
        fig.add_trace(
            go.Heatmap(
                z=sig_matrix,
                x=models,
                y=models,
                colorscale='Viridis',
                colorbar=dict(title='-log10(p-value)'),
                name='Significance'
            ),
            row=len(metrics)+1, col=1
        )
    
    # Add radar chart for performance overview
    # Normalize metrics for radar chart
    radar_data = {}
    
    for metric in metrics:
        values = comparison_df[metric].values
        
        if metric in ['mse', 'mae']:  # Lower is better
            normalized = 1 - (values - min(values)) / (max(values) - min(values) + 1e-8)
        else:  # Higher is better
            normalized = (values - min(values)) / (max(values) - min(values) + 1e-8)
        
        radar_data[metric] = normalized
    
    # Create radar traces for each model
    for i, model in enumerate(models):
        radar_values = [radar_data[metric][i] for metric in metrics]
        
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],  # Close the loop
                theta=metrics + [metrics[0]],  # Close the loop
                name=model,
                fill='toself',
                opacity=0.7
            ),
            row=len(metrics)+1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Model Comparison Dashboard",
        height=300 * (len(metrics) + 1),
        width=1200,
        showlegend=True
    )
    
    # Save to HTML file
    fig.write_html(output_file)
    print(f"Dashboard saved to {output_file}")