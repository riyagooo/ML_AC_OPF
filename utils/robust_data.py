import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from typing import List, Tuple, Dict, Any, Optional

from .data_utils import OPFDataset, load_pglib_data, load_case_network
from .config import DataConfig

class DataNormalizer:
    """
    Utility for normalizing input and output data.
    """
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
    def fit(self, data_frame, input_cols, output_cols):
        """
        Compute normalization parameters from data.
        
        Args:
            data_frame: DataFrame containing data
            input_cols: List of input column names
            output_cols: List of output column names
        """
        self.input_mean = data_frame[input_cols].mean().values
        self.input_std = data_frame[input_cols].std().values
        self.input_std[self.input_std == 0] = 1.0  # Avoid division by zero
        
        self.output_mean = data_frame[output_cols].mean().values
        self.output_std = data_frame[output_cols].std().values
        self.output_std[self.output_std == 0] = 1.0  # Avoid division by zero
        
    def transform_inputs(self, inputs):
        """
        Normalize input data.
        
        Args:
            inputs: Input data (numpy array or torch tensor)
            
        Returns:
            Normalized inputs
        """
        if self.input_mean is None or self.input_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(inputs, torch.Tensor):
            input_mean = torch.tensor(self.input_mean, dtype=inputs.dtype, device=inputs.device)
            input_std = torch.tensor(self.input_std, dtype=inputs.dtype, device=inputs.device)
            return (inputs - input_mean) / input_std
        else:
            return (inputs - self.input_mean) / self.input_std
    
    def transform_outputs(self, outputs):
        """
        Normalize output data.
        
        Args:
            outputs: Output data (numpy array or torch tensor)
            
        Returns:
            Normalized outputs
        """
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(outputs, torch.Tensor):
            output_mean = torch.tensor(self.output_mean, dtype=outputs.dtype, device=outputs.device)
            output_std = torch.tensor(self.output_std, dtype=outputs.dtype, device=outputs.device)
            return (outputs - output_mean) / output_std
        else:
            return (outputs - self.output_mean) / self.output_std
    
    def inverse_transform_outputs(self, normalized_outputs):
        """
        Convert normalized outputs back to original scale.
        
        Args:
            normalized_outputs: Normalized output data
            
        Returns:
            Outputs in original scale
        """
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(normalized_outputs, torch.Tensor):
            output_mean = torch.tensor(self.output_mean, dtype=normalized_outputs.dtype, 
                                      device=normalized_outputs.device)
            output_std = torch.tensor(self.output_std, dtype=normalized_outputs.dtype, 
                                     device=normalized_outputs.device)
            return normalized_outputs * output_std + output_mean
        else:
            return normalized_outputs * self.output_std + self.output_mean

class NormalizedOPFDataset(OPFDataset):
    """
    OPF dataset with built-in normalization.
    """
    def __init__(self, data_frame, input_cols, output_cols, normalizer=None):
        """
        Initialize normalized OPF dataset.
        
        Args:
            data_frame: DataFrame containing input and output data
            input_cols: List of column names for input features
            output_cols: List of column names for output targets
            normalizer: Optional pre-fitted normalizer
        """
        super().__init__(data_frame, input_cols, output_cols)
        self.normalizer = normalizer or DataNormalizer()
        
        # Fit normalizer if not provided
        if normalizer is None:
            self.normalizer.fit(data_frame, input_cols, output_cols)
    
    def __getitem__(self, idx):
        """
        Get normalized item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (normalized_input, normalized_output)
        """
        input_tensor, output_tensor = super().__getitem__(idx)
        
        # Normalize data
        normalized_input = self.normalizer.transform_inputs(input_tensor)
        normalized_output = self.normalizer.transform_outputs(output_tensor)
        
        return normalized_input, normalized_output

def create_cross_validation_splits(
    data_frame: pd.DataFrame,
    input_cols: List[str],
    output_cols: List[str],
    config: DataConfig,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[List[Tuple[DataLoader, DataLoader]], DataLoader, DataNormalizer]:
    """
    Create cross-validation splits for robust model evaluation.
    
    Args:
        data_frame: DataFrame containing data
        input_cols: List of input column names
        output_cols: List of output column names
        config: Data configuration
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (cv_loaders, test_loader, normalizer)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Split data into train+val and test
    train_val_data, test_data = train_test_split(
        data_frame, test_size=config.test_size, random_state=seed
    )
    
    # Create normalizer and fit on training data only
    normalizer = DataNormalizer()
    normalizer.fit(train_val_data, input_cols, output_cols)
    
    # Create normalized datasets
    train_val_dataset = NormalizedOPFDataset(
        train_val_data, input_cols, output_cols, normalizer
    )
    test_dataset = NormalizedOPFDataset(
        test_data, input_cols, output_cols, normalizer
    )
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create cross-validation splits
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    
    cv_loaders = []
    for train_idx, val_idx in kf.split(range(len(train_val_dataset))):
        # Create subset datasets
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        
        # Create dataloaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        cv_loaders.append((train_loader, val_loader))
    
    return cv_loaders, test_loader, normalizer

def load_and_prepare_data(config: DataConfig, batch_size: int = 32) -> Dict[str, Any]:
    """
    Load and prepare data based on configuration.
    
    Args:
        config: Data configuration
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary containing data loaders and related information
    """
    # Load data
    data = load_pglib_data(config.case, config.data_dir)
    case_data = load_case_network(config.case, config.data_dir)
    
    # Extract input and output columns
    input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
    output_cols = [col for col in data.columns if col.startswith('gen_p') or 
                  col.startswith('gen_q') or col.startswith('bus_v')]
    
    # Create cross-validation splits
    cv_loaders, test_loader, normalizer = create_cross_validation_splits(
        data, input_cols, output_cols, config, batch_size, config.seed
    )
    
    return {
        'data': data,
        'case_data': case_data,
        'input_cols': input_cols,
        'output_cols': output_cols,
        'cv_loaders': cv_loaders,
        'test_loader': test_loader,
        'normalizer': normalizer
    }

def noise_injection(inputs, noise_level=0.05, noise_type='gaussian'):
    """
    Add noise to input data for testing model robustness.
    
    Args:
        inputs: Input data (numpy array or torch tensor)
        noise_level: Level of noise to inject (as a fraction of data standard deviation)
        noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            
    Returns:
        Inputs with added noise
    """
    if isinstance(inputs, torch.Tensor):
        if noise_type == 'gaussian':
            noise = torch.randn_like(inputs) * torch.std(inputs) * noise_level
        elif noise_type == 'uniform':
            noise = (torch.rand_like(inputs) * 2 - 1) * torch.std(inputs) * noise_level
        elif noise_type == 'salt_pepper':
            mask = torch.rand_like(inputs) < noise_level
            noise = torch.zeros_like(inputs)
            noise[mask] = inputs.max() - inputs.min()
            # Randomly choose whether to add or subtract the noise
            sign_mask = torch.rand_like(inputs) < 0.5
            noise[sign_mask] = -noise[sign_mask]
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return inputs + noise
    else:
        # Numpy implementation
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 1, inputs.shape) * np.std(inputs) * noise_level
        elif noise_type == 'uniform':
            noise = (np.random.rand(*inputs.shape) * 2 - 1) * np.std(inputs) * noise_level
        elif noise_type == 'salt_pepper':
            mask = np.random.rand(*inputs.shape) < noise_level
            noise = np.zeros_like(inputs)
            noise[mask] = inputs.max() - inputs.min()
            # Randomly choose whether to add or subtract the noise
            sign_mask = np.random.rand(*inputs.shape) < 0.5
            noise[sign_mask] = -noise[sign_mask]
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return inputs + noise

def adversarial_perturbation(model, inputs, targets, epsilon=0.01, steps=3):
    """
    Generate adversarial perturbations for testing model robustness.
    Uses Fast Gradient Sign Method (FGSM) or its iterative variant.
    
    Args:
        model: PyTorch model
        inputs: Input data
        targets: Target outputs
        epsilon: Maximum perturbation size
        steps: Number of iterations (1 for FGSM, >1 for iterative FGSM)
            
    Returns:
        Perturbed inputs
    """
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    
    # Create a copy of inputs that requires gradient
    x_adv = inputs.clone().detach().requires_grad_(True)
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Iterative FGSM
    for _ in range(steps):
        # Forward pass
        outputs = model(x_adv)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial example
        with torch.no_grad():
            # Get sign of gradient
            grad_sign = x_adv.grad.sign()
            
            # Add perturbation
            x_adv = x_adv + (epsilon / steps) * grad_sign
            
            # Ensure the perturbation is within bounds
            x_adv = torch.clamp(x_adv, inputs - epsilon, inputs + epsilon)
            
            # Prepare for next iteration
            x_adv.requires_grad_(True)
    
    return x_adv.detach() 