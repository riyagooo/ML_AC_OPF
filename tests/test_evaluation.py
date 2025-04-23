import unittest
import torch
import numpy as np
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions directly from utils.evaluation
from utils.evaluation import (
    calculate_mse, 
    calculate_mae, 
    calculate_constraints_violation, 
    evaluate_model
)

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create mock predictions and targets
        self.predictions = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        self.targets = torch.tensor([
            [1.2, 1.8, 3.1],
            [3.8, 5.2, 6.3],
            [7.1, 7.9, 8.7]
        ])
        
        # Create mock constraint values
        self.constraint_values = torch.tensor([
            [0.01, 0.05],
            [0.02, -0.01],
            [-0.03, 0.02]
        ])
        
    def test_calculate_mse(self):
        mse = calculate_mse(self.predictions, self.targets)
        
        # Calculate expected MSE manually
        diff = self.predictions - self.targets
        expected_mse = torch.mean(diff * diff).item()
        
        self.assertAlmostEqual(mse, expected_mse, places=5)
        
    def test_calculate_mae(self):
        mae = calculate_mae(self.predictions, self.targets)
        
        # Calculate expected MAE manually
        expected_mae = torch.mean(torch.abs(self.predictions - self.targets)).item()
        
        self.assertAlmostEqual(mae, expected_mae, places=5)
        
    def test_calculate_constraints_violation(self):
        violations = calculate_constraints_violation(self.constraint_values)
        
        # Expected violations: sum of positive values in constraint_values
        expected_violations = sum([
            max(0, self.constraint_values[0, 0]),
            max(0, self.constraint_values[0, 1]),
            max(0, self.constraint_values[1, 0]),
            max(0, self.constraint_values[1, 1]),
            max(0, self.constraint_values[2, 0]),
            max(0, self.constraint_values[2, 1])
        ])
        
        self.assertAlmostEqual(violations, expected_violations, places=5)
        
    def test_evaluate_model(self):
        # Create a mock model that returns fixed predictions
        class MockModel:
            def __init__(self, predictions):
                self.predictions = predictions
                
            def eval(self):
                return self
                
            def __call__(self, data):
                return self.predictions
        
        mock_model = MockModel(self.predictions)
        
        # Create mock data loader that returns a single batch
        class MockDataLoader:
            def __init__(self, features, targets, constraints):
                self.features = features
                self.targets = targets
                self.constraints = constraints
                
            def __iter__(self):
                self.index = 0
                return self
                
            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    mock_data = type('', (), {})()
                    mock_data.y = self.targets
                    mock_data.constraints = self.constraints
                    return mock_data
                else:
                    raise StopIteration
                    
            def __len__(self):
                return 1
                
        mock_data_loader = MockDataLoader(None, self.targets, self.constraint_values)
        
        # Evaluate the model
        results = evaluate_model(mock_model, mock_data_loader)
        
        # Verify results
        self.assertIn('mse', results)
        self.assertIn('mae', results)
        self.assertIn('constraint_violation', results)
        
        # Verify the values
        expected_mse = calculate_mse(self.predictions, self.targets)
        expected_mae = calculate_mae(self.predictions, self.targets)
        expected_violations = calculate_constraints_violation(self.constraint_values)
        
        self.assertAlmostEqual(results['mse'], expected_mse, places=5)
        self.assertAlmostEqual(results['mae'], expected_mae, places=5)
        self.assertAlmostEqual(results['constraint_violation'], expected_violations, places=5)

if __name__ == '__main__':
    unittest.main() 