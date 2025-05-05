# Running Machine Learning Models for OPF

This guide provides instructions for running the machine learning models for Optimal Power Flow (OPF) solutions in this project.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Gurobi solver with a valid license (for optimization)

## Setup Environment

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have a valid Gurobi license. The `gurobi.lic` file should be properly placed in your system.

## Running Local Tests (Case5)

For small cases like case5, you can run all models locally:

### Feedforward Neural Network

```bash
python run.py train --approach feedforward --case case5 --epochs 10
```

### Constraint Screening

```bash
python run.py train --approach constraint_screening --case case5 --epochs 10
```

### Warm Starting

```bash
python run.py train --approach warm_starting --case case5 --epochs 10
```

### Graph Neural Network

```bash
python run.py train --approach gnn --case case5 --epochs 10
```

## Running Larger Cases (Case118) on Google Colab

For larger cases like case118, it's recommended to use Google Colab for training:

1. Upload the project to Google Colab
2. Open one of the provided notebook templates:
   - `notebooks/simple_ml_opf_colab.ipynb` - Simplified notebook for getting started
   - `notebooks/gnn_training.ipynb` - Full GNN training notebook

3. Run the cells in the notebook to train the model

The notebooks include code to download the necessary data, set up the environment, and train the models.

## Simple Test Script

If you're experiencing issues with the full training scripts, you can try the simplified test script:

```bash
python simple_test.py
```

This script creates a synthetic dataset and trains a small feedforward neural network to verify that the basic functionality is working correctly.

## Files and Directories

- `models/` - Contains model implementations
  - `feedforward.py` - Feedforward neural networks
  - `gnn.py` - Graph neural networks
- `utils/` - Utility functions
  - `data_utils.py` - Data loading and processing
  - `optimization.py` - OPF optimization
  - `training.py` - Model training
- `scripts/` - Training scripts
  - `local_test.py` - Local testing script
  - `constraint_screening.py` - Constraint screening implementation
  - `warm_starting.py` - Warm starting implementation
  - `local_gnn.py` - Local GNN implementation
- `notebooks/` - Jupyter notebooks for Google Colab
  - `simple_ml_opf_colab.ipynb` - Simplified Colab notebook
  - `gnn_training.ipynb` - Full GNN training notebook

## Troubleshooting

If you encounter issues:

1. **Complex Number Error**: If you see errors related to complex numbers in the data, make sure you're using the updated version of `utils/data_utils.py` that handles complex numbers correctly.

2. **Missing Cost Coefficients**: If there are errors related to cost coefficients, the code should now include fallbacks for this case.

3. **Memory Issues**: For large cases, you might need to reduce the batch size (e.g., `--batch-size 16`).

4. **CUDA Out of Memory**: If running on GPU, try reducing model size with `--hidden-dims 64,128,64` or switch to CPU.

5. **Gurobi License Issues**: Make sure your Gurobi license is valid and properly installed.

For any other issues, please refer to the error messages for specific guidance.