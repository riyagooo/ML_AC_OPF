import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    type: str  # 'feedforward', 'gnn', 'hybrid_gnn'
    hidden_dims: List[int]  # Hidden layer dimensions for FNN or hidden channels for GNN
    dropout_rate: float
    num_layers: Optional[int] = None  # Number of GNN layers (if applicable)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping: int
    optimizer: str = "adam"
    criterion: str = "mse"
    validation_split: float = 0.1
    use_wandb: bool = False
    seed: int = 42

@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading."""
    case: str  # 'case5', 'case118', etc.
    data_dir: str = "data"
    test_size: float = 0.1
    cv_folds: int = 5
    use_augmentation: bool = False
    normalize_inputs: bool = True
    normalize_outputs: bool = True

@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str
    log_dir: str = "logs"

def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ExperimentConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects
    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    data_config = DataConfig(**config_dict['data'])
    
    # Extract experiment name and log directory
    experiment_name = config_dict.get('experiment_name', 'experiment')
    log_dir = config_dict.get('log_dir', 'logs')
    
    # Create and return full config
    return ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment_name=experiment_name,
        log_dir=log_dir
    )

def save_config(config: ExperimentConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: ExperimentConfig object
        output_path: Path to save the YAML configuration file
    """
    # Convert dataclasses to dictionaries
    config_dict = {
        'model': {k: v for k, v in config.model.__dict__.items() if v is not None},
        'training': {k: v for k, v in config.training.__dict__.items()},
        'data': {k: v for k, v in config.data.__dict__.items()},
        'experiment_name': config.experiment_name,
        'log_dir': config.log_dir
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def create_default_config(model_type: str, case: str) -> ExperimentConfig:
    """
    Create default configuration for a specific model type and case.
    
    Args:
        model_type: One of 'feedforward', 'constraint_screening', 'warm_starting', 'gnn', 'hybrid_gnn'
        case: Case name (e.g., 'case5', 'case118')
        
    Returns:
        ExperimentConfig with default settings
    """
    # Default settings for model
    if model_type == 'feedforward':
        model_config = ModelConfig(
            type='feedforward',
            hidden_dims=[128, 256, 128],
            dropout_rate=0.1
        )
    elif model_type == 'constraint_screening':
        model_config = ModelConfig(
            type='constraint_screening',
            hidden_dims=[64, 128, 64],
            dropout_rate=0.1
        )
    elif model_type == 'warm_starting':
        model_config = ModelConfig(
            type='warm_starting',
            hidden_dims=[128, 256, 128],
            dropout_rate=0.1
        )
    elif model_type == 'gnn':
        model_config = ModelConfig(
            type='gnn',
            hidden_dims=[64],
            num_layers=3,
            dropout_rate=0.1
        )
    elif model_type == 'hybrid_gnn':
        model_config = ModelConfig(
            type='hybrid_gnn',
            hidden_dims=[64],
            num_layers=3,
            dropout_rate=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Default settings for training
    training_config = TrainingConfig(
        batch_size=32 if 'case5' in case or 'case14' in case or 'case30' in case else 64,
        learning_rate=0.001,
        epochs=10 if 'case5' in case or 'case14' in case or 'case30' in case else 20,
        early_stopping=5,
        optimizer="adam",
        criterion="mse",
        validation_split=0.1,
        use_wandb=False,
        seed=42
    )
    
    # Default settings for data
    data_config = DataConfig(
        case=case,
        data_dir="data",
        test_size=0.1,
        cv_folds=5,
        use_augmentation=False,
        normalize_inputs=True,
        normalize_outputs=True
    )
    
    # Create and return full config
    return ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment_name=f"{model_type}_{case}",
        log_dir="logs"
    ) 