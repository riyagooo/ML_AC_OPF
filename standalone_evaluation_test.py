import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Direct implementation of evaluation functions to avoid import issues
def calculate_mse(predictions, targets):
    """Calculate Mean Squared Error between predictions and targets."""
    squared_diff = (predictions - targets) ** 2
    return torch.mean(squared_diff).item()

def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error between predictions and targets."""
    abs_diff = torch.abs(predictions - targets)
    return torch.mean(abs_diff).item()

def calculate_constraints_violation(constraint_values):
    """Calculate the total constraint violation."""
    violations = torch.clamp(constraint_values, min=0)
    return torch.sum(violations).item()

def evaluate_model(model, data_loader):
    """Evaluate a model on a test dataset."""
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
    
    results = {
        'mse': calculate_mse(all_predictions, all_targets),
        'mae': calculate_mae(all_predictions, all_targets)
    }
    
    return results

def benchmark_execution_time(model, sample_input, n_runs=100):
    """Benchmark model execution time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Benchmark
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

def evaluate_robustness(model, data_loader, noise_level=0.05, n_repetitions=3):
    """Evaluate model robustness by adding noise to inputs."""
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
    noisy_mse_values = []
    
    for _ in range(n_repetitions):
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
        noisy_mse_values.append(noisy_mse)
    
    # Calculate metrics
    avg_noisy_mse = np.mean(noisy_mse_values)
    mse_degradation = (avg_noisy_mse - clean_mse) / clean_mse * 100
    
    return {
        'clean_mse': clean_mse,
        'noisy_mse_mean': avg_noisy_mse,
        'mse_degradation_percent': mse_degradation
    }

def visualize_predictions(predictions, targets, figsize=(12, 6)):
    """Visualize model predictions vs targets."""
    plt.figure(figsize=figsize)
    
    # Flatten data if multi-dimensional
    if len(predictions.shape) > 1:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
    
    # Create scatter plot
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(predictions.min(), targets.min())
    max_val = max(predictions.max(), targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.title('Model Predictions vs Targets')
    
    # Show plot
    plt.tight_layout()
    plt.savefig('predictions_vs_targets.png')
    plt.close()
    
    return 'predictions_vs_targets.png'

# ---- Test Functions ----

def create_mock_data(n_samples=100, n_features=10, n_outputs=5):
    """Create mock data for testing."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data
    features = torch.randn(n_samples, n_features)
    targets = torch.randn(n_samples, n_outputs)
    
    return features, targets

def create_mock_model(n_features=10, n_outputs=5, bias=0.0):
    """Create a mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self, n_features, n_outputs, bias):
            super().__init__()
            self.linear = torch.nn.Linear(n_features, n_outputs)
            self.bias = bias
        
        def forward(self, x):
            return self.linear(x) + self.bias
        
        def eval(self):
            return self
    
    return MockModel(n_features, n_outputs, bias)

def create_mock_data_loader(features, targets, batch_size=32):
    """Create a mock data loader for testing."""
    n_samples = features.shape[0]
    
    class MockDataLoader:
        def __init__(self, features, targets, batch_size):
            self.features = features
            self.targets = targets
            self.batch_size = batch_size
            self.n_samples = features.shape[0]
            self.n_batches = (n_samples + batch_size - 1) // batch_size
        
        def __iter__(self):
            self.batch_idx = 0
            return self
        
        def __next__(self):
            if self.batch_idx < self.n_batches:
                start_idx = self.batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.n_samples)
                
                features_batch = self.features[start_idx:end_idx]
                targets_batch = self.targets[start_idx:end_idx]
                
                self.batch_idx += 1
                return features_batch, targets_batch
            else:
                raise StopIteration
        
        def __len__(self):
            return self.n_batches
    
    return MockDataLoader(features, targets, batch_size)

# ---- Main Test Script ----

def run_tests():
    """Run all tests."""
    print("Running evaluation utility tests...")
    
    # Create mock data and model
    features, targets = create_mock_data()
    model = create_mock_model()
    data_loader = create_mock_data_loader(features, targets)
    
    # Test MSE calculation
    mse = calculate_mse(model(features), targets)
    print(f"MSE: {mse:.6f}")
    
    # Test MAE calculation
    mae = calculate_mae(model(features), targets)
    print(f"MAE: {mae:.6f}")
    
    # Test model evaluation
    eval_results = evaluate_model(model, data_loader)
    print(f"Model Evaluation: {eval_results}")
    
    # Test benchmarking
    benchmark_results = benchmark_execution_time(model, features[:1], n_runs=100)
    print(f"Benchmark Results: {benchmark_results}")
    
    # Test robustness
    robustness_results = evaluate_robustness(model, data_loader, noise_level=0.1, n_repetitions=3)
    print(f"Robustness Results: {robustness_results}")
    
    # Test visualization
    with torch.no_grad():
        predictions = model(features)
    vis_path = visualize_predictions(predictions, targets)
    print(f"Visualization saved to: {vis_path}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    run_tests() 