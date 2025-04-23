import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def mock_data():
    """Create mock data for testing."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 100
    n_features = 10
    n_outputs = 5
    
    # Create data
    features = torch.randn(n_samples, n_features)
    targets = torch.randn(n_samples, n_outputs)
    
    return features, targets

@pytest.fixture(scope="session")
def mock_model():
    """Create a mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self, n_features=10, n_outputs=5, bias=0.0):
            super().__init__()
            self.linear = torch.nn.Linear(n_features, n_outputs)
            self.bias = bias
        
        def forward(self, x):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]  # Extract features
            
            if hasattr(x, 'x'):
                x = x.x  # Extract node features for GNN
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # If x doesn't have the right shape, create a random tensor
            if x.shape[1] != 10:
                x = torch.randn(x.shape[0], 10)
            
            return self.linear(x) + self.bias
        
        def eval(self):
            return self
    
    return MockModel()

@pytest.fixture(scope="session")
def mock_case_data():
    """Create mock power system case data for testing."""
    return {
        'baseMVA': 100.0,
        'bus': [
            [1, 3, 0, 0, 0, 0, 1, 1.0, 0.0, 132.0, 1.1, 0.9],  # Bus 1
            [2, 2, 0, 0, 0, 0, 1, 1.0, 0.0, 132.0, 1.1, 0.9],  # Bus 2
            [3, 1, 100.0, 35.0, 0, 0, 1, 1.0, 0.0, 132.0, 1.1, 0.9],  # Bus 3
            [4, 1, 50.0, 20.0, 0, 0, 1, 1.0, 0.0, 132.0, 1.1, 0.9],  # Bus 4
            [5, 1, 80.0, 30.0, 0, 0, 1, 1.0, 0.0, 132.0, 1.1, 0.9]   # Bus 5
        ],
        'gen': [
            [1, 0, 100.0, 50.0, -50.0, 1.1, 0.9, 0, 300.0, 0.0],  # Gen 1
            [2, 0, 100.0, 50.0, -50.0, 1.1, 0.9, 0, 300.0, 0.0]   # Gen 2
        ],
        'branch': [
            [1, 2, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [1, 3, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [2, 3, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [2, 4, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [2, 5, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [3, 4, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360],
            [4, 5, 0.01, 0.1, 0.0, 100.0, 0, 0, 0, 0, 1, -360, 360]
        ],
        'gencost': [
            [2, 0, 0, 3, 0.01, 10.0, 0],
            [2, 0, 0, 3, 0.01, 10.0, 0]
        ]
    }

@pytest.fixture(scope="session")
def mock_data_loader(mock_data):
    """Create a mock data loader for testing."""
    features, targets = mock_data
    n_samples = features.shape[0]
    batch_size = 32
    
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