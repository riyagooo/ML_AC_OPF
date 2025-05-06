# Troubleshooting Guide

This document provides solutions to common issues you might encounter when working with the ML-AC-OPF project.

## Import Errors

### ModuleNotFoundError: No module named 'cases'

If you see this error, the Python interpreter can't find the module path. There are two solutions:

1. **Run from project root**: Make sure you're running the script from the project root directory
   ```bash
   cd /path/to/ML_AC_OPF
   python run_ml_opf.py case39
   ```

2. **Check sys.path**: If you're creating custom scripts, make sure to add the project root to the Python path:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
   ```

## Data Issues

### Case39 data file not found

The system will automatically download case39 data when needed, but if this fails:

1. Run the setup script explicitly:
   ```bash
   python run_ml_opf.py setup --download-case39
   ```

2. Check if `data/case39/case39.m` exists. If not, you may need to manually download it from the MATPOWER repository.

### CSV data parsing errors

The MATPOWER data may have formatting issues that prevent automatic conversion to CSV. However, the system is designed to fall back to PyPOWER's built-in case39 data when needed, so this warning can be safely ignored.

## Model Training Issues

### CUDA out of memory

If you're using a GPU and encounter memory issues:

1. Reduce batch size: `--batch-size 16` or lower
2. Reduce model size: `--hidden-dim 32 --num-layers 2`
3. Try CPU training by forcing PyTorch to use CPU: 
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```

### Training divergence or NaN values

1. Reduce learning rate: `--lr 0.0001`
2. Increase regularization: `--weight-decay 1e-4`
3. Try gradient clipping (this is already implemented in the training loop)

## PyPOWER Issues

### Unable to create admittance matrix

PyPOWER sometimes has issues with specific data formats. Ensure your Python environment has a compatible version:

```bash
pip install pypower==5.1.15
```

### Power flow convergence issues

Power flow convergence can be challenging for some system configurations:

1. Try using the DC approximation by modifying the relevant utilities file
2. Use our synthetic data generation approach which bypasses direct power flow calculations
3. Increase the solver tolerance

## Performance Issues

### Slow training 

1. Use case39 instead of case30 (better numerical properties)
2. Reduce number of samples: `--num-samples 500`
3. Profile the code to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.out run_ml_opf.py case39 --num-samples 10 --epochs 2
   ```

## Path Issues

### File not found errors

If the system can't find data or output files:

1. Make sure all paths are relative to the project root
2. Check that all required directories exist:
   ```bash
   python run_ml_opf.py setup
   ```
3. Use absolute paths when necessary for external data sources

## Code Structure Issues

### "Namespace has no attribute" errors

If you've added new utilities or models, make sure they're properly imported and exposed in the relevant `__init__.py` files.

## Getting Help

If you encounter issues not covered here:

1. Check the code comments for specific functions
2. Read the function docstrings for parameter descriptions
3. Look at the test scripts for usage examples
4. File an issue on the GitHub repository