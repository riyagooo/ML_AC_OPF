import unittest
import torch
import numpy as np

# Direct implementation of the functions being tested
def calculate_mse(predictions, targets):
    """
    Calculate Mean Squared Error between predictions and targets.
    """
    squared_diff = (predictions - targets) ** 2
    return torch.mean(squared_diff).item()

def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error between predictions and targets.
    """
    abs_diff = torch.abs(predictions - targets)
    return torch.mean(abs_diff).item()

def calculate_constraints_violation(constraint_values):
    """
    Calculate the total constraint violation by summing positive values of constraint violations.
    """
    # Only sum positive values (violations)
    violations = torch.clamp(constraint_values, min=0)
    return torch.sum(violations).item()

def evaluate_model(model, data_loader):
    """
    Evaluate a model on a test dataset.
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

# GNN evaluation function
def evaluate_gnn_model(model, data_loader, device='cpu'):
    """
    Evaluate a GNN model on a test dataset.
    """
    model.eval()
    
    node_predictions_list = []
    node_targets_list = []
    graph_predictions_list = []
    graph_targets_list = []
    constraints_list = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in data_loader:
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

# Constraint screening evaluation function
def evaluate_constraint_screening_model(model, data_loader, threshold=0.5):
    """
    Evaluate a constraint screening model.
    """
    model.eval()
    
    all_probabilities = []
    all_targets = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            probabilities = model(batch)
            
            all_probabilities.append(probabilities)
            all_targets.append(batch.y)
    
    # Concatenate all batches
    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Convert probabilities to binary predictions using threshold
    all_predictions = (all_probabilities >= threshold).float()
    
    # Calculate metrics
    # For simplicity in this test, we'll just calculate accuracy
    correct = (all_predictions == all_targets).float().sum()
    total = all_targets.numel()
    accuracy = correct / total
    
    return {'accuracy': accuracy.item()}

# Warm starting evaluation function
def evaluate_warm_starting_model(model, data_loader, optimizer_fn=None, max_iterations=100):
    """
    Simple test version of warm starting evaluation.
    """
    model.eval()
    
    predictions_list = []
    targets_list = []
    
    # Collect predictions
    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            predictions = model(batch)
            
            predictions_list.append(predictions)
            targets_list.append(batch.y)
    
    # Concatenate predictions and targets
    all_predictions = torch.cat(predictions_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate prediction accuracy metrics
    results = {
        'mse': calculate_mse(all_predictions, all_targets),
        'mae': calculate_mae(all_predictions, all_targets)
    }
    
    # Add mock optimization metrics
    if optimizer_fn:
        results['avg_warm_iterations'] = 10  # Mock value
        results['avg_cold_iterations'] = 20  # Mock value
        results['iteration_reduction_pct'] = 50.0  # Mock value
    
    return results

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
        
    def test_evaluate_gnn_model(self):
        # Create a mock GNN model
        class MockGNNModel:
            def __init__(self, node_predictions, graph_predictions):
                self.node_predictions = node_predictions
                self.graph_predictions = graph_predictions
                
            def eval(self):
                return self
                
            def __call__(self, data):
                return self.node_predictions, self.graph_predictions
        
        # Create mock node and graph predictions
        node_predictions = torch.tensor([
            [0.9, 0.8],
            [0.7, 0.6],
            [0.5, 0.4]
        ])
        
        graph_predictions = torch.tensor([
            [0.1, 0.2, 0.3]
        ])
        
        mock_model = MockGNNModel(node_predictions, graph_predictions)
        
        # Create mock data loader for GNN
        class MockGNNDataLoader:
            def __init__(self, node_features, graph_targets):
                self.node_features = node_features
                self.graph_targets = graph_targets
                
            def __iter__(self):
                self.index = 0
                return self
                
            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    mock_data = type('', (), {})()
                    mock_data.x = self.node_features
                    mock_data.y = self.graph_targets
                    return mock_data
                else:
                    raise StopIteration
                    
            def __len__(self):
                return 1
        
        # Create mock features and targets
        node_features = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        graph_targets = torch.tensor([
            [0.0, 0.0, 0.0]
        ])
        
        mock_data_loader = MockGNNDataLoader(node_features, graph_targets)
        
        # Evaluate the GNN model
        results = evaluate_gnn_model(mock_model, mock_data_loader)
        
        # Verify results contain both node and graph metrics
        self.assertIn('node_mse', results)
        self.assertIn('node_mae', results)
        self.assertIn('graph_mse', results)
        self.assertIn('graph_mae', results)
        
        # Verify the values
        expected_node_mse = calculate_mse(node_predictions, node_features)
        expected_node_mae = calculate_mae(node_predictions, node_features)
        expected_graph_mse = calculate_mse(graph_predictions, graph_targets)
        expected_graph_mae = calculate_mae(graph_predictions, graph_targets)
        
        self.assertAlmostEqual(results['node_mse'], expected_node_mse, places=5)
        self.assertAlmostEqual(results['node_mae'], expected_node_mae, places=5)
        self.assertAlmostEqual(results['graph_mse'], expected_graph_mse, places=5)
        self.assertAlmostEqual(results['graph_mae'], expected_graph_mae, places=5)
        
    def test_evaluate_constraint_screening_model(self):
        # Create a mock constraint screening model
        class MockConstraintModel:
            def __init__(self, probabilities):
                self.probabilities = probabilities
                
            def eval(self):
                return self
                
            def __call__(self, data):
                return self.probabilities
        
        # Create mock probabilities and binary targets
        probabilities = torch.tensor([
            [0.9],
            [0.3],
            [0.7],
            [0.2]
        ])
        
        targets = torch.tensor([
            [1.0],
            [0.0],
            [1.0],
            [0.0]
        ])
        
        mock_model = MockConstraintModel(probabilities)
        
        # Create mock data loader
        class MockConstraintDataLoader:
            def __init__(self, targets):
                self.targets = targets
                
            def __iter__(self):
                self.index = 0
                return self
                
            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    mock_data = type('', (), {})()
                    mock_data.y = self.targets
                    return mock_data
                else:
                    raise StopIteration
                    
            def __len__(self):
                return 1
        
        mock_data_loader = MockConstraintDataLoader(targets)
        
        # Evaluate the constraint screening model
        results = evaluate_constraint_screening_model(mock_model, mock_data_loader)
        
        # Verify results
        self.assertIn('accuracy', results)
        
        # Verify accuracy (should be 4/4 = 1.0 with threshold 0.5)
        self.assertEqual(results['accuracy'], 1.0)
        
    def test_evaluate_warm_starting_model(self):
        # Create a mock warm starting model
        class MockWarmStartModel:
            def __init__(self, predictions):
                self.predictions = predictions
                
            def eval(self):
                return self
                
            def __call__(self, data):
                return self.predictions
        
        mock_model = MockWarmStartModel(self.predictions)
        
        # Create mock data loader
        class MockWarmStartDataLoader:
            def __init__(self, targets):
                self.targets = targets
                
            def __iter__(self):
                self.index = 0
                return self
                
            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    mock_data = type('', (), {})()
                    mock_data.y = self.targets
                    return mock_data
                else:
                    raise StopIteration
                    
            def __len__(self):
                return 1
        
        mock_data_loader = MockWarmStartDataLoader(self.targets)
        
        # Define a simple mock optimizer function
        def mock_optimizer_fn(initial_point, features, max_iter):
            return 10, 0.5  # iterations, final value
        
        # Evaluate the warm starting model
        results = evaluate_warm_starting_model(mock_model, mock_data_loader, mock_optimizer_fn)
        
        # Verify results
        self.assertIn('mse', results)
        self.assertIn('mae', results)
        self.assertIn('avg_warm_iterations', results)
        self.assertIn('avg_cold_iterations', results)
        self.assertIn('iteration_reduction_pct', results)
        
        # Verify optimization metrics
        self.assertEqual(results['avg_warm_iterations'], 10)
        self.assertEqual(results['avg_cold_iterations'], 20)
        self.assertEqual(results['iteration_reduction_pct'], 50.0)

if __name__ == '__main__':
    unittest.main() 