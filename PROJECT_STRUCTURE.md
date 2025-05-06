# ML-AC-OPF Project Structure

This document explains the structure of the ML-AC-OPF project, designed for readability, modularity, and maintainability.

## Directory Structure

```
ML_AC_OPF/
│
├── core/                   # Core project modules
│   ├── models/             # Neural network model definitions
│   ├── utils/              # General utility functions
│   └── validation/         # Power system validation components
│
├── train/                  # Training scripts for ML models
│
├── evaluation/             # Evaluation scripts for domain metrics
│
├── analysis/               # Data analysis scripts
│
├── data/                   # Data preprocessing and case file management
│
├── docs/                   # Project documentation
│   ├── DOMAIN_METRICS_EVALUATION.md
│   ├── BALANCED_MODEL_COMPARISON.md
│   ├── METHODOLOGY.md
│   ├── GNN_IMPLEMENTATION.md
│   ├── MODEL_DEVELOPMENT_JOURNEY.md
│   ├── RESULTS.md
│   ├── ML_AC_OPF_MATHEMATICAL_FORMULATIONS.md
│   ├── FAST_TRAINING_README.md
│   ├── VISUALIZATION.md
│   └── TROUBLESHOOTING.md
│
├── scripts/                # Utility scripts
│
├── configs/                # Configuration files
│
├── output/                 # Model outputs and results
│   ├── domain_metrics/     # Domain-specific evaluation metrics
│   ├── balanced_ffn/       # FFN model outputs
│   ├── balanced_gnn/       # GNN model outputs
│   ├── balanced_comparison/# Comparison between FFN and GNN
│   ├── model_comparison/   # General model comparison
│   ├── data_exploration/   # Dataset visualizations
│   └── ieee39_data_small/  # Small test dataset
│
├── README.md               # Project overview
├── PROJECT_STRUCTURE.md    # This file
├── requirements.txt        # Python dependencies
└── mldl_environment.yml    # Conda environment file
```

## Key Components

### Core Module

- **models/**: Contains neural network model definitions
  - `feedforward.py` - Feedforward neural network models
  - `gnn.py` - Graph neural network models
  - `advanced_networks.py` - Advanced network architectures
  - `power_system_graph.py` - Power system graph representation

- **utils/**: Utility functions used across the project
  - `case39_utils.py` - IEEE 39-bus system utilities
  - `data_utils.py` - Data loading and preprocessing
  - `metrics.py` - Performance metrics calculation
  - `network_utils.py` - Neural network utility functions
  - `training.py` - Training utility functions
  - `timing_utils.py` - Timing and benchmarking utilities

- **validation/**: Power system validation components
  - `power_system_validation.py` - Validate ML solutions against power system constraints
  - `optimization.py` - Optimization algorithms for power system validation

### Training Scripts

- `train_balanced_ffn.py` - Train balanced FFN model
- `train_balanced_gnn.py` - Train balanced GNN model
- `compare_balanced_models.py` - Compare FFN and GNN model performance

### Evaluation Scripts

- `evaluate_domain_metrics.py` - Evaluate domain-specific metrics for power system validity
- `evaluate_ffn_complete.py` - Comprehensive FFN evaluation with state reconstruction
- `evaluate_ffn_metrics.py` - Basic FFN metrics evaluation
- `run_complete_ffn_evaluation.sh` - Shell script to run evaluation

### Data Processing

- `create_small_dataset.py` - Create a reduced dataset for experiments
- `preprocess_ieee39.py` - Preprocess IEEE 39-bus case data
- `custom_case_loader.py` - Custom power system case loading utilities

### Output Management

- `cleanup_output.sh` - Script to clean up output directories while preserving essential ones

## Files Removed or Consolidated

The following files were removed or consolidated as they were redundant or no longer relevant:

1. **Backup Files**: All files in the `backup/` directory 
2. **Duplicate Script Files**: Multiple script files performing similar functions
3. **Test Files**: Test files not essential for the core functionality
4. **Visualization Scripts**: Consolidated into the analysis module
5. **Unused Case Files**: Removed case files not actively used in the current workflow
6. **Redundant Shell Scripts**: Consolidated multiple shell scripts with similar functionality

## Domain-Specific Metrics Workflow

The key workflow for evaluating domain-specific metrics follows these steps:

1. **Train Models**: Use scripts in the `train/` directory to train FFN and GNN models
2. **Evaluate Models**: Use scripts in the `evaluation/` directory to evaluate domain-specific metrics
3. **View Results**: Results are stored in the `output/domain_metrics/` directory

## How to Navigate the Codebase

1. **New to the Project?** Start with the main `README.md` file and documentation in the `docs/` directory
2. **Understanding the Methodology?** Read `docs/METHODOLOGY.md` and `docs/MODEL_DEVELOPMENT_JOURNEY.md`
3. **Looking for Results?** Check `docs/BALANCED_MODEL_COMPARISON.md` and `docs/DOMAIN_METRICS_EVALUATION.md`
4. **Want to Run Experiments?** Use scripts in the `train/` and `evaluation/` directories
5. **Need to Modify Models?** Check the model definitions in `core/models/`
6. **Need Mathematical Details?** Review `docs/ML_AC_OPF_MATHEMATICAL_FORMULATIONS.md` for comprehensive mathematical formulations

## Dependencies

Dependencies are managed through:
- `requirements.txt` - For pip installation
- `mldl_environment.yml` - For conda environment creation

## Documentation Files

The project includes the following documentation files:

- `README.md` - Main project overview
- `docs/DOMAIN_METRICS_EVALUATION.md` - Details on domain-specific metrics evaluation
- `docs/BALANCED_MODEL_COMPARISON.md` - Comparison between FFN and GNN models
- `docs/METHODOLOGY.md` - Overview of the methodological approach
- `docs/GNN_IMPLEMENTATION.md` - Details on the GNN implementation
- `docs/MODEL_DEVELOPMENT_JOURNEY.md` - The development process and insights gained
- `docs/RESULTS.md` - Comprehensive results and analysis
- `docs/ML_AC_OPF_MATHEMATICAL_FORMULATIONS.md` - Mathematical formulations for the entire project
- `docs/FAST_TRAINING_README.md` - Approach for fast training with balanced models
- `docs/VISUALIZATION.md` - Details of visualizations created for the project
- `docs/TROUBLESHOOTING.md` - Guide for resolving common issues
- `PROJECT_STRUCTURE.md` - This file, explaining the project organization

## Issues and Known Limitations

1. **GNN Evaluation**: There are issues with PyTorch Geometric imports that prevent complete GNN evaluation
2. **Voltage Constraint Satisfaction**: Current models show poor voltage constraint satisfaction (VCSR of 0.05%), which likely indicates implementation errors in voltage reconstruction rather than fundamental model limitations
3. **Output Dimension Mismatch**: Current models predict only voltage magnitudes (10 outputs) instead of the full power system state (98 variables)

For more details on these issues and ongoing work to address them, see `docs/DOMAIN_METRICS_EVALUATION.md`. 