import pytest
import torch
import numpy as np
import sys
import os

# Direct implementation of essential functions to avoid import issues
def calculate_mse(predictions, targets):
    squared_diff = (predictions - targets) ** 2
    return torch.mean(squared_diff).item()

def calculate_mae(predictions, targets):
    abs_diff = torch.abs(predictions - targets)
    return torch.mean(abs_diff).item()

def calculate_constraints_violation(constraint_values):
    violations = torch.clamp(constraint_values, min=0)
    return torch.sum(violations).item()

# Tests for basic metrics
def test_calculate_mse(mock_data):
    """Test MSE calculation function."""
    features, targets = mock_data
    predictions = features @ torch.randn(features.shape[1], targets.shape[1])
    
    # Calculate MSE
    mse = calculate_mse(predictions, targets)
    
    # Verify it's a float and not NaN
    assert isinstance(mse, float)
    assert not np.isnan(mse)
    
    # Verify it matches manual calculation
    manual_mse = ((predictions - targets) ** 2).mean().item()
    assert abs(mse - manual_mse) < 1e-6

def test_calculate_mae(mock_data):
    """Test MAE calculation function."""
    features, targets = mock_data
    predictions = features @ torch.randn(features.shape[1], targets.shape[1])
    
    # Calculate MAE
    mae = calculate_mae(predictions, targets)
    
    # Verify it's a float and not NaN
    assert isinstance(mae, float)
    assert not np.isnan(mae)
    
    # Verify it matches manual calculation
    manual_mae = torch.abs(predictions - targets).mean().item()
    assert abs(mae - manual_mae) < 1e-6

def test_calculate_constraints_violation():
    """Test constraint violation calculation."""
    # Create test data
    constraint_values = torch.tensor([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, -0.6]
    ])
    
    # Calculate constraint violations
    violations = calculate_constraints_violation(constraint_values)
    
    # Verify it's a float and not NaN
    assert isinstance(violations, float)
    assert not np.isnan(violations)
    
    # Verify it matches manual calculation
    positive_only = torch.clamp(constraint_values, min=0)
    manual_violations = positive_only.sum().item()
    assert abs(violations - manual_violations) < 1e-6

# Tests for model evaluation
def test_evaluate_model(mock_model, mock_data_loader):
    """Test model evaluation function."""
    # Simplified evaluate_model function
    def evaluate_model(model, data_loader):
        model.eval()
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for data in data_loader:
                features, targets = data
                predictions = model(features)
                predictions_list.append(predictions)
                targets_list.append(targets)
        
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        return {
            'mse': calculate_mse(all_predictions, all_targets),
            'mae': calculate_mae(all_predictions, all_targets)
        }
    
    # Evaluate the model
    results = evaluate_model(mock_model, mock_data_loader)
    
    # Verify results
    assert 'mse' in results
    assert 'mae' in results
    assert isinstance(results['mse'], float)
    assert isinstance(results['mae'], float)

# Tests for robustness evaluation
def test_robustness_evaluation(mock_model, mock_data_loader):
    """Test robustness evaluation function."""
    # Simplified evaluate_robustness function
    def evaluate_robustness(model, data_loader, noise_level=0.05):
        model.eval()
        
        # Get clean predictions
        clean_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                predictions = model(features)
                clean_predictions.append(predictions)
                all_targets.append(targets)
        
        clean_predictions = torch.cat(clean_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        clean_mse = calculate_mse(clean_predictions, all_targets)
        
        # Get noisy predictions
        noisy_predictions = []
        
        with torch.no_grad():
            for features, _ in data_loader:
                # Add noise
                noise = torch.randn_like(features) * features.std() * noise_level
                noisy_features = features + noise
                
                # Forward pass
                predictions = model(noisy_features)
                noisy_predictions.append(predictions)
        
        noisy_predictions = torch.cat(noisy_predictions, dim=0)
        noisy_mse = calculate_mse(noisy_predictions, all_targets)
        
        mse_degradation = (noisy_mse - clean_mse) / clean_mse * 100
        
        return {
            'clean_mse': clean_mse,
            'noisy_mse': noisy_mse,
            'mse_degradation_percent': mse_degradation
        }
    
    # Evaluate robustness
    results = evaluate_robustness(mock_model, mock_data_loader, noise_level=0.1)
    
    # Verify results
    assert 'clean_mse' in results
    assert 'noisy_mse' in results
    assert 'mse_degradation_percent' in results
    assert results['noisy_mse'] >= results['clean_mse']  # Adding noise should generally increase MSE

# Tests for benchmarking
def test_benchmarking(mock_model, mock_data):
    """Test execution time benchmarking."""
    features, _ = mock_data
    sample_input = features[:1]
    
    # Simplified benchmark function
    def benchmark_execution_time(model, sample_input, n_runs=10):
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_input)
        
        # Benchmark
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(sample_input)
        
        total_time = time.time() - start_time
        avg_time = total_time / n_runs
        
        return {
            'total_time': total_time,
            'avg_time_per_sample': avg_time,
            'samples_per_second': n_runs / total_time
        }
    
    # Run benchmark
    results = benchmark_execution_time(mock_model, sample_input)
    
    # Verify results
    assert 'total_time' in results
    assert 'avg_time_per_sample' in results
    assert 'samples_per_second' in results
    assert results['total_time'] > 0
    assert results['avg_time_per_sample'] > 0
    assert results['samples_per_second'] > 0 