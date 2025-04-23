import unittest
import torch
import numpy as np
import tempfile
import os
from utils.training import EarlyStopping, save_checkpoint, load_checkpoint

class TestTrainingUtils(unittest.TestCase):
    def setUp(self):
        # Create a mock model for testing
        self.model = torch.nn.Linear(10, 1)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_early_stopping(self):
        # Initialize early stopping
        patience = 5
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        
        # Mock validation losses
        val_losses = [0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        
        # Minimum should be at index 3 (value 0.6)
        # With patience 5, early stopping should trigger at index 8
        expected_best_score = 0.6
        expected_early_stop = True
        
        for i, loss in enumerate(val_losses):
            early_stopping(loss, self.model)
            if early_stopping.early_stop:
                break
                
        self.assertEqual(early_stopping.best_score, expected_best_score)
        self.assertEqual(early_stopping.early_stop, expected_early_stop)
        self.assertEqual(i, 8)  # Should stop at index 8 (9th iteration)
        
    def test_checkpoint_save_load(self):
        # Create a temp directory for saving checkpoints
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save a checkpoint
            checkpoint_path = os.path.join(tmpdirname, "checkpoint.pt")
            
            # Initialize model with known weights
            self.model.weight.data = torch.ones_like(self.model.weight.data)
            self.model.bias.data = torch.zeros_like(self.model.bias.data)
            
            # Create optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            
            # Save checkpoint
            epoch = 10
            valid_loss = 0.5
            train_loss = 0.7
            save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                epoch=epoch,
                valid_loss=valid_loss,
                train_loss=train_loss,
                checkpoint_path=checkpoint_path
            )
            
            # Check that checkpoint file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Reset model weights
            self.model.weight.data = torch.zeros_like(self.model.weight.data)
            
            # Load checkpoint
            loaded_epoch, loaded_valid_loss, loaded_train_loss = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=optimizer
            )
            
            # Check loaded values
            self.assertEqual(loaded_epoch, epoch)
            self.assertEqual(loaded_valid_loss, valid_loss)
            self.assertEqual(loaded_train_loss, train_loss)
            
            # Check that model weights were restored
            self.assertTrue(torch.all(self.model.weight.data == 1.0))
            self.assertTrue(torch.all(self.model.bias.data == 0.0))

if __name__ == '__main__':
    unittest.main() 