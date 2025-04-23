# Hybrid ML-OPF Project

[![Tests](https://github.com/yourusername/ML-OPF-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/ML-OPF-Project/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/yourusername/ML-OPF-Project/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/ML-OPF-Project)

A machine learning approach to solve Optimal Power Flow (OPF) problems using hybrid methods combining neural networks and traditional optimization solvers.

## Project Overview

This project aims to accelerate Optimal Power Flow solutions by using machine learning techniques in conjunction with traditional solvers. The implementation explores three key approaches:

1. **Constraint Screening**: Using ML to predict binding constraints to reduce problem size
2. **Warm-Starting**: Using ML predictions as initial points for optimization solvers
3. **Topology-Aware Predictions**: Leveraging Graph Neural Networks (GNNs) to incorporate grid topology

## Key Components

- **Solver**: Gurobi for optimization
- **ML Framework**: PyTorch for neural network models
- **Dataset**: PGLib-OPF (118-bus case)
- **Deployment**: Hybrid approach supporting both local environments and Google Colab

## Project Structure

```
ML_OPF_Project/
├── data/                # Data storage and processing
├── models/              # Neural network model definitions
├── utils/               # Utility functions and helpers
├── notebooks/           # Jupyter notebooks for exploration and visualization
├── scripts/             # Training and evaluation scripts
└── requirements.txt     # Project dependencies
```

## Getting Started

### Local Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run local tests: `python scripts/local_test.py`

### Google Colab Setup

1. Upload the notebook files from `notebooks/` to Google Colab
2. Run the setup cell in the notebook to install dependencies
3. Connect to the dataset and run training

## References

- PGLib-OPF dataset: https://github.com/power-grid-lib/pglib-opf
- DeepOPF: https://www.mhchen.com/papers/OPF.DeepOPF.SGC.23.pdf

## Installation

```bash
# Install from PyPI
pip install ml-opf

# Install from source with development dependencies
git clone https://github.com/yourusername/ML-OPF-Project.git
cd ML-OPF-Project
pip install -e ".[dev]"
```

## Usage

```python
from utils.evaluation import evaluate_model, visualize_predictions
from models import create_model

# Load your data
...

# Create and train a model
model = create_model()
...

# Evaluate the model
results = evaluate_model(model, test_loader)
print(f"MSE: {results['mse']}, MAE: {results['mae']}")

# Visualize predictions
visualize_predictions(predictions, targets)
```

## Testing

We use pytest for our test suite. To run the tests:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=./

# Run tests for a specific module
pytest tests/test_evaluation_pytest.py
```

## Development

This project follows these development practices:

1. **Branch Structure**:
   - `main`: Production-ready code
   - `develop`: Integration branch for features
   - Feature branches: `feature/name`

2. **Pull Requests**:
   - All changes are made through PRs
   - PRs require passing tests
   - Code review required before merging

3. **CI/CD Pipeline**:
   - Automated tests run on all PRs
   - Code coverage tracked via Codecov
   - Automated deployment on version tags

## Evaluation Module Features

Our comprehensive model evaluation module includes:

1. **Basic Evaluation Functions**
   - MSE, MAE, constraint violation metrics
   - General model evaluation

2. **Model-Specific Evaluation**
   - GNN model evaluation
   - Constraint screening evaluation
   - Warm starting evaluation

3. **Cross-Validation Framework**
   - Comprehensive cross-validation with stratification

4. **Error Analysis**
   - Error distribution visualization
   - Component-wise error analysis
   - Error categorization by operating conditions

5. **Comparative Evaluation**
   - Statistical comparison of multiple models
   - Bootstrap confidence intervals
   - Significance testing

6. **Solution Quality Assessment**
   - Constraint violation metrics
   - Distance to feasibility

7. **Robustness Testing**
   - Sensitivity analysis
   - Input noise robustness

8. **Execution Time Benchmarking**
   - Model speed measurement
   - Comparison with traditional solvers

9. **Interactive Visualization Dashboard**
   - Model prediction visualization
   - Model comparison visualization
   - Power system diagram visualization
