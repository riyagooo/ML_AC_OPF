import unittest
import torch
from models.gnn import GNNModel
from utils.data import OPFDataset

class TestGNNModel(unittest.TestCase):
    def setUp(self):
        # Define a simple model for testing
        self.hidden_dim = 64
        self.output_size = 10
        self.model = GNNModel(
            node_in_dim=5,
            edge_in_dim=3,
            hidden_dim=self.hidden_dim,
            output_size=self.output_size,
            num_layers=2
        )
        
        # Mock input data
        num_nodes = 8
        num_edges = 10
        self.node_features = torch.rand(1, num_nodes, 5)  # [batch, nodes, features]
        self.edge_index = torch.randint(0, num_nodes, (2, num_edges))  # [2, edges]
        self.edge_features = torch.rand(1, num_edges, 3)  # [batch, edges, features]
        
    def test_model_initialization(self):
        # Test model creation
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        
    def test_forward_pass(self):
        # Test model forward pass
        output = self.model(self.node_features, self.edge_index, self.edge_features)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.output_size))
        
    def test_model_training(self):
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create a mock target
        target = torch.rand(1, self.output_size)
        
        # Forward pass
        output = self.model(self.node_features, self.edge_index, self.edge_features)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # If we get here without errors, the test passes
        self.assertTrue(True)

class TestOPFDataset(unittest.TestCase):
    def test_dataset_creation(self):
        # This test only checks if the dataset class is implemented properly
        # A more comprehensive test would require actual data files
        try:
            # Attempt to initialize the dataset with a non-existent directory
            # We're just testing the interface, not the actual data loading
            dataset = OPFDataset(
                data_dir="dummy_path",
                split="train",
                transform=None
            )
            # If the class is implemented properly, this should not raise an error
            # Even though the actual loading might fail
            self.assertTrue(True)
        except TypeError:
            # TypeError would indicate problems with the function signature/interface
            self.fail("OPFDataset class has incorrect interface")
        except FileNotFoundError:
            # FileNotFoundError is expected since we're using a dummy path
            self.assertTrue(True)

if __name__ == '__main__':
    unittest.main() 